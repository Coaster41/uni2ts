"""
PR-4: Linear probe training on residual stream activations.

Fits per-layer linear probes that predict ground-truth time series properties
from mean-pooled context patch representations.

Usage
-----
# With real checkpoints:
python -m experiments.mech_interp.block1_probing.train_probes \
    --moiraie-ckpt /path/to/moiraie.ckpt \
    --moiraic-ckpt /path/to/moiraic.ckpt \
    --output-dir experiments/mech_interp/block1_probing/results/

# Smoke-test with tiny in-memory models (no checkpoint needed):
python -m experiments.mech_interp.block1_probing.train_probes \
    --output-dir /tmp/probe_results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow running as __main__ from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.lib import ResidualExtractor, generate_dataset, load_dataset, make_batch

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4

REGRESSION_FEATURES = ["slope", "log_noise_var", "phase_cos", "phase_sin"]
CLASSIFICATION_FEATURES = ["period_idx"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def extract_activations(
    module,
    series: np.ndarray,
    batch_size: int = 32,
    patch_size: int = PATCH_SIZE,
    context_patches: int = CONTEXT_PATCHES,
    pred_patches: int = PRED_PATCHES,
    device: str | torch.device = "cpu",
) -> dict[int, np.ndarray]:
    """
    Extract mean-pooled context activations for all examples.

    Parameters
    ----------
    module : MoiraieModule or MoiraicModule
    series : float32 [n, time]
    batch_size : int

    Returns
    -------
    dict layer_idx -> float32 [n, d_model]
        Mean-pooled over context patches (positions 0..context_patches-1).
    """
    module.eval()
    accumulated: dict[int, list[np.ndarray]] = {}

    with ResidualExtractor(module) as extractor:
        for start in range(0, len(series), batch_size):
            chunk = series[start : start + batch_size]
            batch = make_batch(chunk, patch_size=patch_size, context_patches=context_patches, pred_patches=pred_patches, device=device)
            acts = extractor.run(batch)  # dict[int, Tensor[chunk, n_patches, d_model]]
            for layer_idx, tensor in acts.items():
                # Mean-pool context patches: positions 0..context_patches-1
                ctx = tensor[:, :context_patches, :].mean(dim=1).numpy()  # [chunk, d_model]
                accumulated.setdefault(layer_idx, []).append(ctx)

    return {layer_idx: np.concatenate(chunks, axis=0) for layer_idx, chunks in accumulated.items()}


def fit_probe(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_type: str,
) -> float:
    """
    Fit a linear probe and return the validation score.

    Parameters
    ----------
    feature_type : "regression" or "classification"

    Returns
    -------
    float
        R² (regression) or accuracy (classification) on the validation set.
    """
    if feature_type == "regression":
        # StandardScaler + RidgeCV: normalizing helps CV stability and is fair across layers
        probe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5))])
        probe.fit(X_train, y_train)
        return float(probe.score(X_val, y_val))
    elif feature_type == "classification":
        from sklearn.linear_model import LogisticRegression

        y_train_int = y_train.astype(int)
        y_val_int = y_val.astype(int)
        classes, counts = np.unique(y_train_int, return_counts=True)
        if len(classes) < 2:
            # Degenerate: only one class in train — predict majority, return accuracy
            return float(np.mean(y_val_int == classes[0]))
        min_count = int(counts.min())
        if min_count >= 2:
            # Enough samples per class for stratified CV
            n_cv = min(5, min_count)
            clf = LogisticRegressionCV(cv=n_cv, max_iter=5000, n_jobs=-1)
        else:
            # Class imbalance too severe for CV — use fixed regularization
            clf = LogisticRegression(C=1.0, max_iter=5000)
        probe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        probe.fit(X_train, y_train_int)
        return float(probe.score(X_val, y_val_int))
    else:
        raise ValueError(f"Unknown feature_type {feature_type!r}; expected 'regression' or 'classification'")


def run_probes_for_model(
    module,
    dataset: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
    patch_size: int = PATCH_SIZE,
    context_patches: int = CONTEXT_PATCHES,
    pred_patches: int = PRED_PATCHES,
    device: str | torch.device = "cpu",
) -> dict[str, dict[int, float]]:
    """
    Full probe training pipeline for one model.

    Parameters
    ----------
    module : MoiraieModule or MoiraicModule (eval mode)
    dataset : dict with keys "series", plus label arrays
    train_idx, val_idx : integer index arrays for the 80/20 split

    Returns
    -------
    {feature_name: {layer_idx: score}}
    """
    series = dataset["series"]

    print(f"  Extracting train activations ({len(train_idx)} examples)...")
    X_train_by_layer = extract_activations(
        module, series[train_idx], batch_size=batch_size,
        patch_size=patch_size, context_patches=context_patches, pred_patches=pred_patches,
        device=device,
    )

    print(f"  Extracting val activations ({len(val_idx)} examples)...")
    X_val_by_layer = extract_activations(
        module, series[val_idx], batch_size=batch_size,
        patch_size=patch_size, context_patches=context_patches, pred_patches=pred_patches,
        device=device,
    )

    num_layers = len(X_train_by_layer)
    results: dict[str, dict[int, float]] = {}

    all_features = (
        [(f, "regression") for f in REGRESSION_FEATURES if f in dataset]
        + [(f, "classification") for f in CLASSIFICATION_FEATURES if f in dataset]
    )

    for feature, ftype in all_features:
        y_train = dataset[feature][train_idx].ravel()
        y_val = dataset[feature][val_idx].ravel()
        layer_scores: dict[int, float] = {}
        for layer_idx in range(num_layers):
            score = fit_probe(X_train_by_layer[layer_idx], X_val_by_layer[layer_idx], y_train, y_val, ftype)
            layer_scores[layer_idx] = score
        results[feature] = layer_scores
        best = max(layer_scores.values())
        print(f"    {feature}: best layer score = {best:.4f}")

    return results


def _load_module(ckpt_path: str | None, model_name: str, device: str | torch.device = "cpu"):
    """Load a model module from a checkpoint path, or build a tiny model if None."""
    from uni2ts.model.moiraic.module import MoiraicModule
    from uni2ts.model.moiraie.module import MoiraieModule

    if ckpt_path is None:
        print(f"  No checkpoint for {model_name} — using tiny in-memory model.")
        tiny = dict(d_model=64, d_ff=128, num_layers=2, patch_size=PATCH_SIZE,
                    max_seq_len=64, attn_dropout_p=0.0, dropout_p=0.0)
        if model_name == "moiraie":
            module = MoiraieModule(**tiny, num_predict_token=1)
        else:
            module = MoiraicModule(**tiny, num_predict_token=PRED_PATCHES)
        return module.eval().to(device)

    print(f"  Loading {model_name} from {ckpt_path} (device={device})")
    if model_name == "moiraie":
        module = MoiraieModule.from_pretrained(ckpt_path)
    else:
        module = MoiraicModule.from_pretrained(ckpt_path)

    return module.eval().to(device)


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on residual stream activations.")
    parser.add_argument("--moiraie-ckpt", default=None, help="Path to moiraie checkpoint (.ckpt or hub id)")
    parser.add_argument("--moiraic-ckpt", default=None, help="Path to moiraic checkpoint (.ckpt or hub id)")
    parser.add_argument("--dataset-path", default=None, help="Path to .npz dataset (generated in memory if omitted)")
    parser.add_argument("--output-dir", default="experiments/mech_interp/block1_probing/results",
                        help="Directory for results JSON files")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for train/val split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for model forward passes (default: cuda if available, else cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate dataset
    if args.dataset_path:
        print(f"Loading dataset from {args.dataset_path}")
        dataset = load_dataset(args.dataset_path)
    else:
        print("Generating synthetic dataset (n=1000, seed=42) in memory...")
        dataset = generate_dataset(n=1000, seed=42)

    n = len(dataset["series"])
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")

    print(f"Device: {args.device}")

    for model_name, ckpt_path in [("moiraie", args.moiraie_ckpt), ("moiraic", args.moiraic_ckpt)]:
        print(f"\n=== {model_name} ===")
        module = _load_module(ckpt_path, model_name, device=args.device)

        results = run_probes_for_model(
            module, dataset, train_idx, val_idx,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Convert int keys to strings for JSON serialization
        results_serializable = {feat: {str(k): v for k, v in layers.items()} for feat, layers in results.items()}
        out_path = os.path.join(args.output_dir, f"{model_name}.json")
        with open(out_path, "w") as f:
            json.dump(results_serializable, f, indent=2)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
