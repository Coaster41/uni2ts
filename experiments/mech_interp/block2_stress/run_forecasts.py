"""
Forecast runner for block2_stress.

Runs any registered foundation-model adapter on each stress-test family+level,
saving only forecast_quantiles, context, and target (no activations).

Models are described in a YAML spec (default ``configs/models.yaml``) mapping a
model NAME to an adapter spec. Adapters import their backend lazily, so this
module stays import-light and a subset of models can be run in whatever venv has
the right dependencies — all writing to the same ``--output-dir`` (per-model
env / shared npz). Select a subset with ``--only NAME ...``.

Usage (smoke test with tiny in-memory custom models — ckpt: null in the spec):
    python -m experiments.mech_interp.block2_stress.run_forecasts \
        --only moiraic moiraie \
        --data-dir experiments/mech_interp/block2_stress/data/stress \
        --output-dir experiments/mech_interp/block2_stress/data/forecasts \
        --device cpu

Full run (custom + moirai2 in uni2ts venv):
    python -m experiments.mech_interp.block2_stress.run_forecasts \
        --models-config experiments/mech_interp/block2_stress/configs/models.yaml \
        --only moiraic moiraie moirai2 \
        --data-dir experiments/mech_interp/block2_stress/data/stress \
        --output-dir experiments/mech_interp/block2_stress/data/forecasts \
        --device cuda:7

External model (run in an env that has the package, repo on PYTHONPATH):
    python -m experiments.mech_interp.block2_stress.run_forecasts \
        --only chronos2 --device cuda \
        --data-dir .../data/stress --output-dir .../data/forecasts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block2_stress import load_stress_dataset
from experiments.mech_interp.block2_stress.models import load_adapter

DEFAULT_MODELS_CONFIG = str(Path(__file__).resolve().parent / "configs" / "models.yaml")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_family(
    adapter,
    family: str,
    level_key: str,
    data_dir: str,
    cfg: dict,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Run one adapter on one stress-test file. Returns dict without saving.
    Keys: "forecast_quantiles" [n,Q,H], "context" [n,ctx_len], "target" [n,H]
    """
    series, _meta, _cfg = load_stress_dataset(data_dir, family, level_key)

    patch_size = cfg["patch_len"]
    context_patches = cfg["context_patches"]
    pred_patches = cfg["horizon_patches"]
    ctx_len = context_patches * patch_size
    horizon = pred_patches * patch_size

    context = series[:, :ctx_len].astype(np.float32)
    target = series[:, ctx_len:].astype(np.float32)

    forecast_quantiles = adapter.predict_quantiles(
        context, horizon, batch_size=batch_size
    ).astype(np.float32)

    return {
        "forecast_quantiles": forecast_quantiles,
        "context": context,
        "target": target,
    }


def load_models_config(models_config: str | None) -> dict[str, dict]:
    """Load the ``models:`` mapping from the YAML spec."""
    path = models_config or DEFAULT_MODELS_CONFIG
    with open(path) as f:
        spec = yaml.safe_load(f)
    return spec["models"]


def run_and_save_all(
    data_dir: str,
    output_dir: str,
    models_config: str | None = None,
    only: list[str] | None = None,
    families: list[str] | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """
    Run the selected models over all (or specified) families, saving to:
        {output_dir}/{model_name}/{family}/{level_key}.npz
    Logs monotonicity violations (does not crash on violations).
    """
    models = load_models_config(models_config)
    if only is not None:
        missing = [m for m in only if m not in models]
        if missing:
            raise KeyError(f"--only names not in models config: {missing}")
        models = {m: models[m] for m in only}

    index = np.load(os.path.join(data_dir, "index.npz"), allow_pickle=True)
    all_families = [str(f) for f in index["families"]]
    all_level_keys = [str(k) for k in index["level_keys"]]
    cfg = json.loads(str(index["config_json"][0]))

    if families is not None:
        pairs = [(f, k) for f, k in zip(all_families, all_level_keys) if f in families]
    else:
        pairs = list(zip(all_families, all_level_keys))

    for model_name, spec in models.items():
        spec = dict(spec)
        adapter_name = spec.pop("adapter")
        spec.setdefault("device", device)
        print(f"\n=== {model_name} (adapter={adapter_name}) ===")
        adapter = load_adapter(adapter_name, **spec)

        for family, level_key in pairs:
            print(f"  {family}/{level_key} ...", end=" ", flush=True)
            result = run_family(adapter, family, level_key, data_dir, cfg, batch_size)
            fq = result["forecast_quantiles"]

            violations = int((fq[:, :-1, :] > fq[:, 1:, :]).sum())
            if violations > 0:
                print(f"WARN: {violations} monotonicity violations", end=" ")

            out_path = os.path.join(output_dir, model_name, family, f"{level_key}.npz")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez(out_path, **result)
            print(f"-> {out_path}")

    print("\nDone.")


def load_forecasts(
    output_dir: str,
    model_name: str,
    family: str,
    level_key: str,
) -> dict[str, np.ndarray]:
    """Load a saved forecast file. Returns same keys as run_family."""
    path = os.path.join(output_dir, model_name, family, f"{level_key}.npz")
    npz = np.load(path)
    return {k: npz[k].astype(np.float32) for k in npz.files}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test forecast runner.")
    parser.add_argument(
        "--models-config",
        default=None,
        help=f"YAML model spec (default: {DEFAULT_MODELS_CONFIG})",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of model names from the spec to run (default: all)",
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to block2_stress/data/stress/"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root, e.g. block2_stress/data/forecasts/",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Subset of families to run (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Default device passed to adapters lacking an explicit one",
    )
    args = parser.parse_args()

    run_and_save_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_config=args.models_config,
        only=args.only,
        families=args.families,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
