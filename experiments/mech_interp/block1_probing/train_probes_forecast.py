"""
PR-14: Forecast-output probe training (Experiment C).

Reads .npz files written by forecast_runner.py, computes forecast-property targets
via forecast_properties.py, trains linear probes on pre-pooled activations, and
writes results/forecast/{model}_{dataset}.json.

Usage
-----
# Smoke-test (requires PR-13 npz output):
python -m experiments.mech_interp.block1_probing.forecast_runner \
    --dataset synth --n-synth 200 \
    --output-dir /tmp/forecast_runner_smoke/

python -m experiments.mech_interp.block1_probing.train_probes_forecast \
    --npz-dir /tmp/forecast_runner_smoke/ \
    --dataset synth --model moiraie \
    --output-dir /tmp/forecast_probe_results/

# Full run:
python -m experiments.mech_interp.block1_probing.train_probes_forecast \
    --npz-dir experiments/mech_interp/block1_probing/results/forecast/ \
    --output-dir experiments/mech_interp/block1_probing/results/forecast/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.block1_probing.forecast_runner import load_runner_output
from experiments.mech_interp.block1_probing.forecast_properties import (
    compute_all,
    derive_binary_labels,
)
from experiments.mech_interp.block1_probing.train_probes import fit_probe

FORECAST_REGRESSION_FEATURES = [
    "fc_std",
    "fc_range",
    "fc_ctx_corr",
    "fc_ctx_corr_seasonal",
    "fc_iqr_mean",
    "fc_iqr_slope",
    "mase",
    "swql",
    "quantile_calibration_err",
]
FORECAST_BINARY_FEATURES = ["is_flat", "is_poor"]
ALL_FORECAST_FEATURES = FORECAST_REGRESSION_FEATURES + FORECAST_BINARY_FEATURES


def _parse_layer_indices(runner_output: dict[str, np.ndarray]) -> list[int]:
    """Extract sorted integer layer indices from runner_output keys."""
    prefix = "activations_mean_ctx_layer_"
    indices = []
    for k in runner_output:
        if k.startswith(prefix):
            suffix = k[len(prefix):]
            indices.append(int(suffix))
    return sorted(indices)


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
    fq_all = runner_output["forecast_quantiles"]   # [n, 9, 64]
    tgt_all = runner_output["target"]               # [n, 64]
    ctx_all = runner_output["context"]              # [n, 512]
    n = len(fq_all)

    per_series_ctx_period = isinstance(ctx_period, np.ndarray)

    accum: dict[str, list[float]] = {k: [] for k in FORECAST_REGRESSION_FEATURES}
    for i in range(n):
        p = int(ctx_period[i]) if per_series_ctx_period else int(ctx_period)
        props = compute_all(fq_all[i], tgt_all[i], ctx_all[i], p)
        for k in FORECAST_REGRESSION_FEATURES:
            accum[k].append(props[k])

    result: dict[str, np.ndarray] = {
        k: np.array(v, dtype=np.float32) for k, v in accum.items()
    }

    binary = derive_binary_labels(
        fc_stds=result["fc_std"].astype(np.float64),
        mases=result["mase"].astype(np.float64),
    )
    result["is_flat"] = binary["is_flat"]
    result["is_poor"] = binary["is_poor"]
    return result


def run_forecast_probes(
    runner_output: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ctx_period: int | np.ndarray = 24,
) -> dict[str, dict[str, dict[int, float]]]:
    """Train linear probes predicting forecast-output targets from pre-pooled activations.

    Returns
    -------
    {pooling_mode: {target_name: {layer_idx: score}}}
    pooling_mode ∈ {"mean_ctx", "last_ctx"}
    Regression targets scored by val R²; binary targets scored by val AUROC.
    """
    layer_indices = _parse_layer_indices(runner_output)
    targets = compute_forecast_targets(runner_output, ctx_period)
    results: dict[str, dict[str, dict[int, float]]] = {}

    for pooling in ("mean_ctx", "last_ctx"):
        X_by_layer: dict[int, np.ndarray] = {
            idx: runner_output[f"activations_{pooling}_layer_{idx}"]
            for idx in layer_indices
        }
        pooling_results: dict[str, dict[int, float]] = {}

        for feature in FORECAST_REGRESSION_FEATURES:
            y = targets[feature]
            mask = np.isfinite(y)
            tr_mask = mask[train_idx]
            va_mask = mask[val_idx]
            if tr_mask.sum() < 10 or va_mask.sum() < 10:
                continue
            layer_scores: dict[int, float] = {}
            for idx in layer_indices:
                X = X_by_layer[idx]
                score = fit_probe(
                    X[train_idx][tr_mask],
                    X[val_idx][va_mask],
                    y[train_idx][tr_mask],
                    y[val_idx][va_mask],
                    "regression",
                )
                layer_scores[idx] = score
            pooling_results[feature] = layer_scores
            best = max(layer_scores.values())
            print(f"    [{pooling}] {feature}: best layer R² = {best:.4f}")

        for feature in FORECAST_BINARY_FEATURES:
            y = targets[feature]
            layer_scores = {}
            for idx in layer_indices:
                X = X_by_layer[idx]
                score = fit_probe(
                    X[train_idx], X[val_idx],
                    y[train_idx], y[val_idx],
                    "binary",
                )
                layer_scores[idx] = score
            pooling_results[feature] = layer_scores
            best = max(layer_scores.values())
            print(f"    [{pooling}] {feature}: best layer AUROC = {best:.4f}")

        results[pooling] = pooling_results

    return results


def _serialize(results: dict) -> dict:
    """Convert layer-index keys (int) to strings for JSON serialization."""
    return {
        pooling: {
            feat: {str(k): v for k, v in layers.items()}
            for feat, layers in feat_dict.items()
        }
        for pooling, feat_dict in results.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear probes on forecast-output properties (Experiment C)."
    )
    parser.add_argument(
        "--npz-dir", required=True,
        help="Directory containing {model}_{dataset}.npz files from forecast_runner.",
    )
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

            results = run_forecast_probes(runner_output, train_idx, val_idx, ctx_period=24)

            out_path = os.path.join(args.output_dir, f"{model_name}_{ds_name}.json")
            with open(out_path, "w") as f:
                json.dump(_serialize(results), f, indent=2)
            print(f"  Saved: {out_path}")

    metadata = {
        "features": {
            **{f: {"type": "regression", "baseline": 0.0, "metric": "R²"}
               for f in FORECAST_REGRESSION_FEATURES},
            "is_flat": {"type": "binary", "baseline": 0.5, "metric": "AUROC"},
            "is_poor": {"type": "binary", "baseline": 0.5, "metric": "AUROC"},
        }
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
