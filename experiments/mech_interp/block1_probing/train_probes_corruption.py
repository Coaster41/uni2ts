"""
PR-17: Corruption probe training (Experiment D).

Generates 8 corrupted variants of each series on-the-fly, extracts activations
from a frozen model, trains linear probes (8-way corruption-ID classifier and
binary is-corrupted classifier), and writes results/corruption/{model}.json.

Usage
-----
python -m experiments.mech_interp.block1_probing.train_probes_corruption \
    --moiraie-ckpt /path/to/moiraie_ckpt \
    --dataset-path /path/to/composite_dataset.npz \
    --output-dir experiments/mech_interp/block1_probing/results/corruption/ \
    --n-synth 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.lib import (
    corrupt_mean_center,
    corrupt_noise,
    corrupt_reverse,
    corrupt_seasonal,
    corrupt_shuffle_patches,
    corrupt_trend,
    corrupt_zero_segment,
    load_dataset,
)
from experiments.mech_interp.lib.utils import _load_module
from experiments.mech_interp.block1_probing.train_probes import extract_activations, fit_probe

CORRUPTION_NAMES = [
    "clean",
    "no_trend",
    "no_seasonal",
    "noise",
    "mean_center",
    "reverse",
    "shuffle",
    "zero_segment",
]
N_CORRUPTIONS = len(CORRUPTION_NAMES)

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4


def _apply_corruptions(
    dataset: dict,
    series_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return {corr_name: float32 [n_series, T]} for all 8 corruption types."""
    n = len(series_indices)
    T = dataset["series"].shape[1]
    result: dict[str, np.ndarray] = {c: np.empty((n, T), dtype=np.float32) for c in CORRUPTION_NAMES}

    for out_i, orig_i in enumerate(series_indices):
        series = dataset["series"][orig_i]
        slope = float(dataset["slope"][orig_i])
        period_idx = int(dataset["period_idx"][orig_i])
        phase = float(np.arctan2(dataset["phase_sin"][orig_i], dataset["phase_cos"][orig_i]))
        seed = int(orig_i) * 7919

        result["clean"][out_i] = series
        result["no_trend"][out_i] = corrupt_trend(series, slope)
        result["no_seasonal"][out_i] = corrupt_seasonal(series, period_idx, phase)
        result["noise"][out_i] = corrupt_noise(series, seed)
        result["mean_center"][out_i] = corrupt_mean_center(series)
        result["reverse"][out_i] = corrupt_reverse(series)
        result["shuffle"][out_i] = corrupt_shuffle_patches(series, seed)
        result["zero_segment"][out_i] = corrupt_zero_segment(series, seed)

    return result


def build_corruption_activations(
    module,
    dataset: dict,
    series_indices: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Extract activations for all 8 corruption variants of each series.

    Returns flat dict with keys "{corr_name}_{pooling}_layer_{k}".
    Shapes: [n_series, d_model] for mean_ctx/last_ctx, [n_series, 32, d_model] for per_patch.
    """
    corrupted = _apply_corruptions(dataset, series_indices)

    out: dict[str, np.ndarray] = {}
    for corr_name, series_arr in corrupted.items():
        print(f"  Extracting activations: {corr_name} ...")
        for pooling in ("mean_ctx", "last_ctx", "per_patch"):
            acts = extract_activations(
                module, series_arr,
                batch_size=batch_size,
                device=device,
                pooling=pooling,
            )
            for layer_idx, arr in acts.items():
                out[f"{corr_name}_{pooling}_layer_{layer_idx}"] = arr

    return out


def _parse_layer_indices(corruption_acts: dict[str, np.ndarray]) -> list[int]:
    """Extract sorted integer layer indices from corruption_acts keys."""
    prefix = "clean_mean_ctx_layer_"
    indices = []
    for k in corruption_acts:
        if k.startswith(prefix):
            indices.append(int(k[len(prefix):]))
    return sorted(indices)


def run_corruption_probes(
    corruption_acts: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, dict[str, dict[int, float | list[float]]]]:
    """Train corruption-ID (8-class) and is-corrupted (binary) probes.

    train_idx / val_idx index into [n_series] — the same split applies to all corruption types.

    Returns
    -------
    {pooling: {feature: {layer_idx: score}}}
    pooling ∈ {"mean_ctx", "last_ctx", "per_patch"}
    feature ∈ {"corruption_id", "is_corrupted"}
    score: float for mean_ctx/last_ctx; list[float] of length 32 for per_patch.
    """
    layer_indices = _parse_layer_indices(corruption_acts)
    n_series = len(train_idx) + len(val_idx)

    results: dict = {}

    for pooling in ("mean_ctx", "last_ctx"):
        pooling_results: dict[str, dict[int, float]] = {"corruption_id": {}, "is_corrupted": {}}
        for layer_idx in layer_indices:
            # Stack X and y across all 8 corruption types
            X_tr_parts, X_va_parts = [], []
            y_id_tr, y_id_va = [], []
            for c_idx, c_name in enumerate(CORRUPTION_NAMES):
                X = corruption_acts[f"{c_name}_{pooling}_layer_{layer_idx}"]
                X_tr_parts.append(X[train_idx])
                X_va_parts.append(X[val_idx])
                y_id_tr.append(np.full(len(train_idx), c_idx, dtype=np.int32))
                y_id_va.append(np.full(len(val_idx), c_idx, dtype=np.int32))

            X_tr = np.concatenate(X_tr_parts, axis=0)
            X_va = np.concatenate(X_va_parts, axis=0)
            y_id_tr_arr = np.concatenate(y_id_tr)
            y_id_va_arr = np.concatenate(y_id_va)

            # Binary: 0=clean (c_idx==0), 1=any corruption
            y_bin_tr = (y_id_tr_arr > 0).astype(np.int32)
            y_bin_va = (y_id_va_arr > 0).astype(np.int32)

            acc = fit_probe(X_tr, X_va, y_id_tr_arr, y_id_va_arr, "classification")
            pooling_results["corruption_id"][layer_idx] = acc

            auroc = fit_probe(X_tr, X_va, y_bin_tr, y_bin_va, "binary")
            pooling_results["is_corrupted"][layer_idx] = auroc

        for feat, layer_scores in pooling_results.items():
            best = max(layer_scores.values())
            metric = "accuracy" if feat == "corruption_id" else "AUROC"
            print(f"    [{pooling}] {feat}: best layer {metric} = {best:.4f}")
        results[pooling] = pooling_results

    # per_patch: train one probe per patch position
    per_patch_results: dict[str, dict[int, list[float]]] = {
        "corruption_id": {}, "is_corrupted": {}
    }
    for layer_idx in layer_indices:
        pos_acc = []
        pos_auroc = []
        for p in range(CONTEXT_PATCHES):
            X_tr_parts, X_va_parts = [], []
            y_id_tr, y_id_va = [], []
            for c_idx, c_name in enumerate(CORRUPTION_NAMES):
                X = corruption_acts[f"{c_name}_per_patch_layer_{layer_idx}"]  # [n, 32, d]
                X_tr_parts.append(X[train_idx, p, :])
                X_va_parts.append(X[val_idx, p, :])
                y_id_tr.append(np.full(len(train_idx), c_idx, dtype=np.int32))
                y_id_va.append(np.full(len(val_idx), c_idx, dtype=np.int32))

            X_tr = np.concatenate(X_tr_parts, axis=0)
            X_va = np.concatenate(X_va_parts, axis=0)
            y_id_tr_arr = np.concatenate(y_id_tr)
            y_id_va_arr = np.concatenate(y_id_va)
            y_bin_tr = (y_id_tr_arr > 0).astype(np.int32)
            y_bin_va = (y_id_va_arr > 0).astype(np.int32)

            pos_acc.append(fit_probe(X_tr, X_va, y_id_tr_arr, y_id_va_arr, "classification"))
            pos_auroc.append(fit_probe(X_tr, X_va, y_bin_tr, y_bin_va, "binary"))

        per_patch_results["corruption_id"][layer_idx] = pos_acc
        per_patch_results["is_corrupted"][layer_idx] = pos_auroc

    results["per_patch"] = per_patch_results
    return results


def _serialize(results: dict) -> dict:
    """Convert int layer-index keys to strings for JSON serialization."""
    out = {}
    for pooling, feat_dict in results.items():
        out[pooling] = {}
        for feat, layer_scores in feat_dict.items():
            out[pooling][feat] = {str(k): v for k, v in layer_scores.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train corruption-identity probes on residual stream (Experiment D)."
    )
    parser.add_argument("--moiraie-ckpt", default=None)
    parser.add_argument("--moiraic-ckpt", default=None)
    parser.add_argument("--dataset-path", required=True,
                        help="Path to composite synthetic dataset .npz.")
    parser.add_argument("--n-synth", type=int, default=2000,
                        help="Number of series to use (first N from dataset).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["moiraie", "moiraic", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_path)
    n_total = len(dataset["series"])
    n = min(args.n_synth, n_total)
    series_indices = np.arange(n)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    print(f"Dataset: n={n}, train={n_train}, val={n - n_train}")

    ckpt_map = {"moiraie": args.moiraie_ckpt, "moiraic": args.moiraic_ckpt}
    model_names = ["moiraie", "moiraic"] if args.model == "both" else [args.model]

    for model_name in model_names:
        ckpt = ckpt_map[model_name]
        if ckpt is None:
            print(f"\nSkipping {model_name} (no checkpoint provided)")
            continue

        print(f"\n=== {model_name} ===")
        module = _load_module(ckpt, model_name, device=args.device)

        corruption_acts = build_corruption_activations(
            module, dataset, series_indices,
            batch_size=args.batch_size, device=args.device,
        )

        results = run_corruption_probes(corruption_acts, train_idx, val_idx)

        out_path = os.path.join(args.output_dir, f"{model_name}.json")
        with open(out_path, "w") as f:
            json.dump(_serialize(results), f, indent=2)
        print(f"  Saved: {out_path}")

    metadata = {
        "features": {
            "corruption_id": {
                "type": "classification",
                "n_classes": N_CORRUPTIONS,
                "baseline": 1.0 / N_CORRUPTIONS,
                "metric": "accuracy",
                "classes": CORRUPTION_NAMES,
            },
            "is_corrupted": {
                "type": "binary",
                "baseline": 0.5,
                "metric": "AUROC",
            },
        }
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
