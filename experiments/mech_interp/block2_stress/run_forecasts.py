"""
PR-2: Forecast Runner for block2_stress.

Runs moiraie and moiraic on each stress-test family+level, saving only
forecast_quantiles, context, and target (no activations).

Usage (smoke test with tiny models):
    python -m experiments.mech_interp.block2_stress.run_forecasts \
        --data-dir experiments/mech_interp/block2_stress/data/stress \
        --output-dir experiments/mech_interp/block2_stress/data/forecasts \
        --device cpu

Full run:
    python -m experiments.mech_interp.block2_stress.run_forecasts \
        --moiraie-ckpt /srv/.../moiraie_training_7/HF_checkpoints/last \
        --moiraic-ckpt /srv/.../moiraic_training_11/HF_checkpoints/last \
        --data-dir experiments/mech_interp/block2_stress/data/stress \
        --output-dir experiments/mech_interp/block2_stress/data/forecasts \
        --device cuda:7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.lib import make_batch
from experiments.mech_interp.lib.utils import _load_module
from experiments.mech_interp.block2_stress import load_stress_dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_fq(
    result: np.ndarray,
    is_moiraic: bool,
    npt: int,
    Q: int,
    P: int,
    context_patches: int,
    pred_patches: int,
) -> np.ndarray:
    """result: [B, n_patches, npt*Q*P] -> [B, Q, pred_patches*P]."""
    B = result.shape[0]
    if is_moiraic:
        pred = result[:, context_patches - 1, :]        # [B, npt*Q*P]
        pred = pred.reshape(B, npt, Q, P)               # [B, npt, Q, P]
        pred = pred.transpose(0, 2, 1, 3)               # [B, Q, npt, P]
        pred = pred.reshape(B, Q, -1)[:, :, :pred_patches * P]
    else:
        pred = result[:, context_patches:, :]            # [B, pred_patches, npt*Q*P]
        pred = pred.reshape(B, pred_patches, npt, Q, P)
        pred = pred[:, :, 0, :, :]                       # [B, pred_patches, Q, P]
        pred = pred.transpose(0, 2, 1, 3)                # [B, Q, pred_patches, P]
        pred = pred.reshape(B, Q, pred_patches * P)      # [B, Q, H]
    return pred


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_family(
    module,
    family: str,
    level_key: str,
    data_dir: str,
    cfg: dict,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Run model on one stress-test file. Returns dict without saving.
    Keys: "forecast_quantiles" [n,9,H], "context" [n,ctx_len], "target" [n,H]
    """
    series, _meta, _cfg = load_stress_dataset(data_dir, family, level_key)

    patch_size = cfg["patch_len"]
    context_patches = cfg["context_patches"]
    pred_patches = cfg["horizon_patches"]
    ctx_len = context_patches * patch_size

    n = len(series)
    is_moiraic = type(module).__name__.startswith("Moiraic")
    npt = module.num_predict_token
    Q = module.num_quantiles
    P = module.patch_size

    module.eval()
    module.to(device)

    fq_buffer: list[np.ndarray] = []

    for i in range(0, n, batch_size):
        chunk = series[i: i + batch_size]
        batch = make_batch(chunk, patch_size, context_patches, pred_patches, device)

        with torch.no_grad():
            result = module(**batch, training_mode=False)

        result_np = result.detach().cpu().float().numpy()
        fq = _extract_fq(result_np, is_moiraic, npt, Q, P, context_patches, pred_patches)
        fq_buffer.append(fq)

    forecast_quantiles = np.concatenate(fq_buffer, axis=0).astype(np.float32)
    context = series[:, :ctx_len].astype(np.float32)
    target = series[:, ctx_len:].astype(np.float32)

    return {
        "forecast_quantiles": forecast_quantiles,
        "context": context,
        "target": target,
    }


def run_and_save_all(
    moiraie_ckpt: str | None,
    moiraic_ckpt: str | None,
    data_dir: str,
    output_dir: str,
    families: list[str] | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """
    Run both models over all (or specified) families, save to:
        {output_dir}/{model_name}/{family}/{level_key}.npz
    Also logs monotonicity violations (does not crash on violations).
    """
    index = np.load(os.path.join(data_dir, "index.npz"), allow_pickle=True)
    all_families = [str(f) for f in index["families"]]
    all_level_keys = [str(k) for k in index["level_keys"]]
    cfg_json = str(index["config_json"][0])
    cfg = json.loads(cfg_json)

    if families is not None:
        pairs = [
            (f, k) for f, k in zip(all_families, all_level_keys)
            if f in families
        ]
    else:
        pairs = list(zip(all_families, all_level_keys))

    ckpt_map = {"moiraie": moiraie_ckpt, "moiraic": moiraic_ckpt}

    for model_name in ["moiraie", "moiraic"]:
        ckpt = ckpt_map[model_name]
        print(f"\n=== {model_name} ===")
        module = _load_module(ckpt, model_name, device)

        for family, level_key in pairs:
            print(f"  {family}/{level_key} ...", end=" ", flush=True)
            result = run_family(module, family, level_key, data_dir, cfg, batch_size, device)
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
    parser.add_argument("--moiraie-ckpt", default=None)
    parser.add_argument("--moiraic-ckpt", default=None)
    parser.add_argument("--data-dir", required=True,
                        help="Path to block2_stress/data/stress/")
    parser.add_argument("--output-dir", required=True,
                        help="Output root, e.g. block2_stress/data/forecasts/")
    parser.add_argument("--families", nargs="*", default=None,
                        help="Subset of families to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_and_save_all(
        moiraie_ckpt=args.moiraie_ckpt,
        moiraic_ckpt=args.moiraic_ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        families=args.families,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
