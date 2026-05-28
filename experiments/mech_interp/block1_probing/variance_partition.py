"""
PR-15: Variance partitioning for Experiment C (C.3).

For the headline forecast-output targets (mase, fc_iqr_mean, fc_std), compares:
  - Surface probe:   hand-crafted context stats → R²_surface
  - Neural probe:    mean_ctx(H^(ℓ)) → R²_neural
  - Combined probe:  [surface || neural] → R²_combined

The novel-information delta is R²_combined − R²_surface.

Usage
-----
# Requires .npz files from forecast_runner.py:
python -m experiments.mech_interp.block1_probing.variance_partition \
    --npz-dir /tmp/forecast_runner_smoke/ \
    --dataset synth --model moiraie \
    --output-dir /tmp/variance_partition_results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.block1_probing.forecast_runner import (
    load_runner_output,
    compute_forecast_targets,
    _parse_layer_indices,
)
from experiments.mech_interp.block1_probing.probe_utils import fit_probe
from experiments.mech_interp.lib.pseudo_labels import (
    context_std as _ctx_std,
    context_acf_lag1 as _ctx_acf,
    fft_top1_power_frac as _fft_power,
    n_changepoints as _n_changepoints,
)

HEADLINE_TARGETS = ("mase", "fc_iqr_mean", "fc_std")
N_BASE_SURFACE_FEATURES = 5


def _log_noise_var_est(ctx: np.ndarray) -> float:
    """log(std(first-diff(ctx)) + 1e-6) — simple noise variance proxy."""
    return float(np.log(np.diff(ctx.astype(np.float64)).std() + 1e-6))


def compute_surface_features(
    context: np.ndarray,
    dataset_ids: np.ndarray | None = None,
    n_datasets: int = 9,
) -> np.ndarray:
    """Compute shallow hand-crafted surface features from context windows.

    Parameters
    ----------
    context : float32 [n, 512]
    dataset_ids : optional int32 [n] — if provided, one-hot appended
    n_datasets : number of one-hot classes (only used when dataset_ids provided)

    Returns
    -------
    float32 [n, 5] or [n, 5+n_datasets]
    NaN/inf values are replaced with 0.0.
    """
    n = len(context)
    base = np.zeros((n, N_BASE_SURFACE_FEATURES), dtype=np.float32)
    for i in range(n):
        ctx = context[i].astype(np.float64)
        base[i, 0] = _ctx_std(ctx)
        base[i, 1] = _ctx_acf(ctx)
        base[i, 2] = _log_noise_var_est(ctx)
        base[i, 3] = _fft_power(ctx)
        base[i, 4] = _n_changepoints(ctx)

    np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    if dataset_ids is None:
        return base

    one_hot = np.zeros((n, n_datasets), dtype=np.float32)
    ids = np.asarray(dataset_ids, dtype=int)
    one_hot[np.arange(n), np.clip(ids, 0, n_datasets - 1)] = 1.0
    return np.hstack([base, one_hot])


def run_variance_partition(
    runner_output: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    headline_targets: tuple[str, ...] = HEADLINE_TARGETS,
    ctx_period: int | np.ndarray = 24,
    dataset_ids: np.ndarray | None = None,
    n_datasets: int = 9,
) -> dict:
    """Variance partitioning: surface vs neural vs combined probes.

    Returns
    -------
    Nested dict: {target: {pooling: {str(layer_idx): {
        "surface": R², "neural": R², "combined": R², "delta": R²
    }}}}
    """
    context = runner_output["context"]
    S = compute_surface_features(context, dataset_ids, n_datasets)
    targets = compute_forecast_targets(runner_output, ctx_period)
    layer_indices = _parse_layer_indices(runner_output)

    results: dict = {}

    for target_name in headline_targets:
        y = targets[target_name].astype(np.float64)
        nan_mask = np.isfinite(y)
        tr_nan = nan_mask[train_idx]
        va_nan = nan_mask[val_idx]

        if tr_nan.sum() < 10 or va_nan.sum() < 10:
            continue

        y_tr = y[train_idx][tr_nan]
        y_va = y[val_idx][va_nan]

        S_tr = S[train_idx][tr_nan]
        S_va = S[val_idx][va_nan]

        r2_surface = fit_probe(S_tr, S_va, y_tr, y_va, "regression")

        target_results: dict = {}

        for pooling in ("mean_ctx", "last_ctx"):
            pooling_results: dict = {}
            for layer_idx in layer_indices:
                X = runner_output[f"activations_{pooling}_layer_{layer_idx}"]
                X_tr = X[train_idx][tr_nan]
                X_va = X[val_idx][va_nan]

                r2_neural = fit_probe(X_tr, X_va, y_tr, y_va, "regression")

                X_comb_tr = np.hstack([S_tr, X_tr])
                X_comb_va = np.hstack([S_va, X_va])
                r2_combined = fit_probe(X_comb_tr, X_comb_va, y_tr, y_va, "regression")

                delta = r2_combined - r2_surface
                pooling_results[layer_idx] = {
                    "surface": r2_surface,
                    "neural": r2_neural,
                    "combined": r2_combined,
                    "delta": delta,
                }
                print(
                    f"    [{target_name}][{pooling}][layer {layer_idx}] "
                    f"surface={r2_surface:.3f}  neural={r2_neural:.3f}  "
                    f"combined={r2_combined:.3f}  delta={delta:+.3f}"
                )

            target_results[pooling] = pooling_results

        results[target_name] = target_results

    return results


def _serialize(results: dict) -> dict:
    """Convert integer layer keys to strings for JSON serialization."""
    out = {}
    for target, pooling_dict in results.items():
        out[target] = {}
        for pooling, layer_dict in pooling_dict.items():
            out[target][pooling] = {
                str(layer_idx): scores
                for layer_idx, scores in layer_dict.items()
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Variance partitioning: surface vs neural vs combined probes (C.3)."
    )
    parser.add_argument("--npz-dir", required=True,
                        help="Directory with {model}_{dataset}.npz from forecast_runner.")
    parser.add_argument("--dataset", choices=["synth", "real", "both"], default="both")
    parser.add_argument("--model", choices=["moiraie", "moiraic", "both"], default="both")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_names = ["moiraie", "moiraic"] if args.model == "both" else [args.model]
    dataset_names = ["synth", "real"] if args.dataset == "both" else [args.dataset]

    for model_name in model_names:
        for ds_name in dataset_names:
            npz_path = os.path.join(args.npz_dir, f"{model_name}_{ds_name}.npz")
            if not os.path.exists(npz_path):
                print(f"  Skipping {npz_path} (not found)")
                continue

            print(f"\n=== {model_name} / {ds_name} ===")
            runner_output = load_runner_output(npz_path)
            n = len(runner_output["forecast_quantiles"])
            n_train = int(n * 0.8)

            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(n)
            train_idx, val_idx = idx[:n_train], idx[n_train:]
            print(f"  n={n}, train={n_train}, val={n - n_train}")

            results = run_variance_partition(runner_output, train_idx, val_idx)

            out_path = os.path.join(args.output_dir, f"{model_name}_{ds_name}_variance_partition.json")
            with open(out_path, "w") as f:
                json.dump(_serialize(results), f, indent=2)
            print(f"  Saved: {out_path}")

    metadata = {
        "surface_features": [
            "context_std", "context_acf_lag1", "log_noise_var_est",
            "fft_top1_power_frac", "n_changepoints",
        ],
        "headline_targets": list(HEADLINE_TARGETS),
        "metrics": {k: "R²" for k in HEADLINE_TARGETS},
    }
    meta_path = os.path.join(args.output_dir, "variance_partition_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
