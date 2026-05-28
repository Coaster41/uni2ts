"""
PR-13: Forecast Runner — combined residual + forecast capture.

Runs a model on a dataset in a single forward-pass loop, capturing both
per-layer residual activations and quantile forecasts, and saves them to .npz.

Usage
-----
# Smoke-test (tiny in-memory models, n=50):
python -m experiments.mech_interp.block1_probing.forecast_runner \
    --dataset synth --n-synth 50 \
    --output-dir /tmp/forecast_runner_smoke/

# Full run:
python -m experiments.mech_interp.block1_probing.forecast_runner \
    --dataset real \
    --moiraie-ckpt /srv/.../moiraie_training_7/HF_checkpoints/last \
    --moiraic-ckpt /srv/.../moiraic_training_11/HF_checkpoints/last \
    --device cuda:7 \
    --output-dir experiments/mech_interp/block1_probing/results/forecast/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.lib import ResidualExtractor, make_batch
from experiments.mech_interp.lib.utils import _load_module
from experiments.mech_interp.block1_probing.probe_utils import (
    PATCH_SIZE,
    CONTEXT_PATCHES,
    PRED_PATCHES,
    HORIZON,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pool_mean_ctx(acts: np.ndarray) -> np.ndarray:
    """acts: [B, n_patches, d_model] -> [B, d_model] (mean over context patches)."""
    return acts[:, :CONTEXT_PATCHES, :].mean(axis=1)


def _pool_last_ctx(acts: np.ndarray) -> np.ndarray:
    """acts: [B, n_patches, d_model] -> [B, d_model] (last context patch, position 31)."""
    return acts[:, CONTEXT_PATCHES - 1, :]


def _extract_fq(
    result: np.ndarray,
    is_moiraic: bool,
    npt: int,
    Q: int,
    P: int,
) -> np.ndarray:
    """result: [B, n_patches, npt*Q*P] -> [B, Q, pred_patches*P]."""
    B = result.shape[0]
    if is_moiraic:
        pred = result[:, CONTEXT_PATCHES - 1, :]      # [B, npt*Q*P]
        pred = pred.reshape(B, npt, Q, P)              # [B, npt, Q, P]
        pred = pred.transpose(0, 2, 1, 3)              # [B, Q, npt, P]
        pred = pred.reshape(B, Q, -1)[:, :, :PRED_PATCHES * P]
    else:
        pred = result[:, CONTEXT_PATCHES:, :]          # [B, pred_patches, npt*Q*P]
        pred = pred.reshape(B, PRED_PATCHES, npt, Q, P)
        pred = pred[:, :, 0, :, :]                     # [B, pred_patches, Q, P]
        pred = pred.transpose(0, 2, 1, 3)              # [B, Q, pred_patches, P]
        pred = pred.reshape(B, Q, PRED_PATCHES * P)    # [B, Q, 64]
    return pred


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_and_save(
    module,
    dataset: dict[str, np.ndarray],
    out_path: str,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    ctx_period: int = 24,
) -> None:
    """Run model on dataset["series"], capture activations + forecasts, write .npz."""
    series = dataset["series"]  # [n, 576]
    n = len(series)
    is_moiraic = type(module).__name__.startswith("Moiraic")
    npt = module.num_predict_token
    Q = module.num_quantiles
    P = module.patch_size

    module.eval()
    module.to(device)

    mean_ctx_buffers: dict[int, list[np.ndarray]] = {}
    last_ctx_buffers: dict[int, list[np.ndarray]] = {}
    fq_buffer: list[np.ndarray] = []

    for i in range(0, n, batch_size):
        chunk = series[i : i + batch_size]
        batch = make_batch(chunk, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES, device)

        with ResidualExtractor(module) as extractor:
            raw_acts_tensors, result = extractor.run(batch)
        raw_acts = {k: v.float().numpy() for k, v in raw_acts_tensors.items()}

        result_np = result.detach().cpu().float().numpy()
        fq = _extract_fq(result_np, is_moiraic, npt, Q, P)
        fq_buffer.append(fq)

        for layer_idx, acts in raw_acts.items():
            mean_ctx_buffers.setdefault(layer_idx, []).append(_pool_mean_ctx(acts))
            last_ctx_buffers.setdefault(layer_idx, []).append(_pool_last_ctx(acts))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for layer_idx in mean_ctx_buffers:
        arrays[f"activations_mean_ctx_layer_{layer_idx}"] = np.concatenate(
            mean_ctx_buffers[layer_idx], axis=0
        ).astype(np.float32)
        arrays[f"activations_last_ctx_layer_{layer_idx}"] = np.concatenate(
            last_ctx_buffers[layer_idx], axis=0
        ).astype(np.float32)

    arrays["forecast_quantiles"] = np.concatenate(fq_buffer, axis=0).astype(np.float32)
    arrays["context"] = series[:, :512].astype(np.float32)
    arrays["target"] = series[:, 512:].astype(np.float32)

    np.savez(out_path, **arrays)


def load_runner_output(path: str) -> dict[str, np.ndarray]:
    """Load .npz written by run_and_save. Returns all arrays as float32 numpy."""
    npz = np.load(path)
    return {k: npz[k].astype(np.float32) for k in npz.files}


# ---------------------------------------------------------------------------
# Forecast target computation (used by variance_partition and train_probes)
# ---------------------------------------------------------------------------

_FORECAST_REGRESSION_FEATURES = [
    "fc_std", "fc_range", "fc_ctx_corr", "fc_ctx_corr_seasonal",
    "fc_iqr_mean", "fc_iqr_slope", "mase", "swql", "quantile_calibration_err",
]


def _parse_layer_indices(runner_output: dict[str, np.ndarray]) -> list[int]:
    """Extract sorted integer layer indices from runner_output keys."""
    prefix = "activations_mean_ctx_layer_"
    return sorted(
        int(k[len(prefix):])
        for k in runner_output
        if k.startswith(prefix)
    )


def compute_forecast_targets(
    runner_output: dict[str, np.ndarray],
    ctx_period: int | np.ndarray = 24,
) -> dict[str, np.ndarray]:
    """Compute per-series forecast-output property targets.

    Parameters
    ----------
    runner_output : dict from load_runner_output; must contain
        "forecast_quantiles" [n, 9, 64], "target" [n, 64], "context" [n, 512].
    ctx_period : int (broadcast) or int32[n] per-series dominant period.

    Returns
    -------
    dict with 11 keys: 9 regression floats + "is_flat"/"is_poor" int32, each [n].
    """
    from experiments.mech_interp.block1_probing.forecast_properties import (
        compute_all,
        derive_binary_labels,
    )

    fq_all = runner_output["forecast_quantiles"]   # [n, 9, 64]
    tgt_all = runner_output["target"]               # [n, 64]
    ctx_all = runner_output["context"]              # [n, 512]
    n = len(fq_all)
    per_series = isinstance(ctx_period, np.ndarray)

    accum: dict[str, list[float]] = {k: [] for k in _FORECAST_REGRESSION_FEATURES}
    for i in range(n):
        p = int(ctx_period[i]) if per_series else int(ctx_period)
        props = compute_all(fq_all[i], tgt_all[i], ctx_all[i], p)
        for k in _FORECAST_REGRESSION_FEATURES:
            accum[k].append(props[k])

    result: dict[str, np.ndarray] = {k: np.array(v, dtype=np.float32) for k, v in accum.items()}
    binary = derive_binary_labels(
        fc_stds=result["fc_std"].astype(np.float64),
        mases=result["mase"].astype(np.float64),
    )
    result["is_flat"] = binary["is_flat"]
    result["is_poor"] = binary["is_poor"]
    return result


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _build_synth_dataset(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    series = rng.standard_normal((n, 576)).astype(np.float32)
    return {"series": series}


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast runner: captures activations + forecasts.")
    parser.add_argument("--dataset", choices=["synth", "real", "both"], default="synth")
    parser.add_argument("--model", choices=["moiraie", "moiraic", "both"], default="both")
    parser.add_argument("--moiraie-ckpt", default=None)
    parser.add_argument("--moiraic-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-synth", type=int, default=5000)
    parser.add_argument("--n-per-dataset", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_names = ["moiraie", "moiraic"] if args.model == "both" else [args.model]
    ckpt_map = {"moiraie": args.moiraie_ckpt, "moiraic": args.moiraic_ckpt}

    datasets_to_run: list[tuple[str, dict]] = []

    if args.dataset in ("synth", "both"):
        rng = np.random.default_rng(42)
        synth_ds = _build_synth_dataset(args.n_synth, rng)
        datasets_to_run.append(("synth", synth_ds))

    if args.dataset in ("real", "both"):
        from experiments.mech_interp.lib.real_data import load_gift_subset
        print("Loading real dataset (may take a few minutes if cache is cold)...")
        real_ds = load_gift_subset(n_per_dataset=args.n_per_dataset)
        datasets_to_run.append(("real", real_ds))

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        module = _load_module(ckpt_map[model_name], model_name, args.device)

        for ds_name, ds in datasets_to_run:
            out_path = os.path.join(args.output_dir, f"{model_name}_{ds_name}.npz")
            print(f"  Running on {ds_name} (n={len(ds['series'])}) -> {out_path}")

            ctx_period: int | np.ndarray = 24
            if ds_name == "real" and "fft_dominant_period" in ds:
                ctx_period = np.array([
                    int(round(float(np.exp(ds["fft_dominant_period"][i]))))
                    for i in range(len(ds["series"]))
                ], dtype=np.int32)

            run_and_save(
                module, ds, out_path,
                batch_size=args.batch_size,
                device=args.device,
                ctx_period=24 if isinstance(ctx_period, int) else 24,
            )
            print(f"  Saved: {out_path}")

            # Write JSON sidecar with metadata
            meta = {
                "model": model_name,
                "dataset": ds_name,
                "n": int(len(ds["series"])),
                "checkpoint": ckpt_map[model_name],
            }
            meta_path = out_path.replace(".npz", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
