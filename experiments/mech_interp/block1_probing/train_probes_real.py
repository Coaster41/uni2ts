"""
PR-11: Linear probe training on residual stream activations using real GIFT-Eval data.

Mirrors the structure of train_probes.py but operates on pseudo-labels from real
time series (structural properties) rather than synthetic ground-truth labels.

Usage
-----
# Smoke-test with tiny in-memory models (no checkpoint needed):
python -m experiments.mech_interp.block1_probing.train_probes_real \
    --output-dir /tmp/real_probe_results/

# Full run with checkpoints:
python -m experiments.mech_interp.block1_probing.train_probes_real \
    --moiraie-ckpt /srv/disk00/ctadler/uni2ts/outputs/pretrain/moiraie/gift_eval_pretrain_weighted/moiraie_training_7/HF_checkpoints/last \
    --moiraic-ckpt /srv/disk00/ctadler/uni2ts/outputs/pretrain/moiraic/gift_eval_pretrain_weighted/moiraic_training_11/HF_checkpoints/last \
    --device cuda:7 \
    --output-dir experiments/mech_interp/block1_probing/results/real/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.lib import load_gift_subset, split_dataset, _load_module
from experiments.mech_interp.block1_probing.train_probes import (
    extract_activations,
    fit_probe,
    PATCH_SIZE,
    CONTEXT_PATCHES,
    PRED_PATCHES,
)

REAL_REGRESSION_FEATURES = [
    "stl_trend_strength",
    "stl_seasonal_strength",
    "fft_dominant_period",
    "fft_top1_power_frac",
    "spectral_flatness",
    "adf_pvalue",
    "hurst_exponent",
    "sample_entropy",
    "n_changepoints",
    "context_std",
    "context_acf_lag1",
]
REAL_CLASSIFICATION_FEATURES = [
    "dataset_id",
]
N_DATASET_CLASSES = 9


def run_real_probes_for_model(
    module,
    dataset: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, dict[str, dict[int, float]]]:
    """
    Returns {pooling_mode: {feature_name: {layer_idx: score}}}
    pooling_mode ∈ {"mean_ctx", "last_ctx"}
    layer_idx includes -1 (post-projection baseline)
    """
    series = dataset["series"]
    results: dict = {}

    for pooling in ("mean_ctx", "last_ctx"):
        print(f"  Extracting train activations [{pooling}] ({len(train_idx)} examples)...")
        X_train_by_layer = extract_activations(
            module, series[train_idx], batch_size=batch_size,
            device=device, pooling=pooling,
        )

        print(f"  Extracting val activations [{pooling}] ({len(val_idx)} examples)...")
        X_val_by_layer = extract_activations(
            module, series[val_idx], batch_size=batch_size,
            device=device, pooling=pooling,
        )

        layer_keys = sorted(X_train_by_layer.keys())
        pooling_results: dict[str, dict[int, float]] = {}

        # Regression features (NaN-safe)
        for feature in REAL_REGRESSION_FEATURES:
            if feature not in dataset:
                continue
            y_train = dataset[feature][train_idx].ravel()
            y_val = dataset[feature][val_idx].ravel()
            tr_mask = np.isfinite(y_train)
            va_mask = np.isfinite(y_val)
            if tr_mask.sum() < 10 or va_mask.sum() < 10:
                continue
            layer_scores: dict[int, float] = {}
            for layer_idx in layer_keys:
                score = fit_probe(
                    X_train_by_layer[layer_idx][tr_mask],
                    X_val_by_layer[layer_idx][va_mask],
                    y_train[tr_mask],
                    y_val[va_mask],
                    "regression",
                )
                layer_scores[layer_idx] = score
            pooling_results[feature] = layer_scores
            best = max(layer_scores.values())
            print(f"    [{pooling}] {feature}: best layer score = {best:.4f}")

        # Classification: dataset_id (9-class, never NaN)
        if "dataset_id" in dataset:
            y_tr = dataset["dataset_id"][train_idx].ravel().astype(int)
            y_va = dataset["dataset_id"][val_idx].ravel().astype(int)
            layer_scores = {}
            for layer_idx in layer_keys:
                score = fit_probe(
                    X_train_by_layer[layer_idx],
                    X_val_by_layer[layer_idx],
                    y_tr,
                    y_va,
                    "classification",
                )
                layer_scores[layer_idx] = score
            pooling_results["dataset_id"] = layer_scores
            best = max(layer_scores.values())
            print(f"    [{pooling}] dataset_id: best layer score = {best:.4f}")

        results[pooling] = pooling_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on real GIFT-Eval data.")
    parser.add_argument("--moiraie-ckpt", default=None, help="Path to moiraie checkpoint")
    parser.add_argument("--moiraic-ckpt", default=None, help="Path to moiraic checkpoint")
    parser.add_argument("--output-dir", default="experiments/mech_interp/block1_probing/results/real",
                        help="Directory for results JSON files")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for train/val split")
    parser.add_argument("--n-per-dataset", type=int, default=600,
                        help="Number of windows per GIFT dataset (default 600; use 50 for smoke test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for model forward passes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading GIFT subset (n_per_dataset={args.n_per_dataset})...")
    dataset = load_gift_subset(n_per_dataset=args.n_per_dataset)
    N = len(dataset["series"])
    n_train = int(N * 0.8)
    train_idx, val_idx = split_dataset(dataset, n_train=n_train, seed=args.seed)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")
    print(f"Device: {args.device}")

    for model_name, ckpt_path in [("moiraie", args.moiraie_ckpt), ("moiraic", args.moiraic_ckpt)]:
        print(f"\n=== {model_name} ===")
        module = _load_module(ckpt_path, model_name, device=args.device)

        results = run_real_probes_for_model(
            module, dataset, train_idx, val_idx,
            batch_size=args.batch_size,
            device=args.device,
        )

        results_serializable = {
            pooling: {
                feat: {str(k): v for k, v in layers.items()}
                for feat, layers in feat_dict.items()
            }
            for pooling, feat_dict in results.items()
        }
        out_path = os.path.join(args.output_dir, f"{model_name}.json")
        with open(out_path, "w") as f:
            json.dump(results_serializable, f, indent=2)
        print(f"  Saved to {out_path}")

    metadata = {
        "features": {
            **{f: {"type": "regression", "baseline": 0.0, "metric": "R²"}
               for f in REAL_REGRESSION_FEATURES},
            "dataset_id": {
                "type": "classification",
                "n_classes": N_DATASET_CLASSES,
                "baseline": round(1 / N_DATASET_CLASSES, 4),
                "metric": "accuracy",
            },
        }
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
