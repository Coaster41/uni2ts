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

REGRESSION_FEATURES = [
    "slope", "log_noise_var", "phase_cos", "phase_sin",
    "level_magnitude", "level_time_norm", "ar_phi",
    "seasonal_amplitude", "log_sigma_ratio", "var_shift_time_norm",
]
CLASSIFICATION_FEATURES = [
    "period_idx",
    "spike_present", "rw_present",
    # spike_patch_idx is 32-class; add when per-patch probing is wired in PR-8
]
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


def extract_activations_per_patch(
    module,
    series: np.ndarray,
    batch_size: int = 32,
    patch_size: int = PATCH_SIZE,
    context_patches: int = CONTEXT_PATCHES,
    pred_patches: int = PRED_PATCHES,
    device: str | torch.device = "cpu",
) -> dict[int, np.ndarray]:
    """
    Like extract_activations but preserves the patch axis — no mean-pooling.

    Returns
    -------
    dict layer_idx -> float32 [n, context_patches, d_model]
    """
    module.eval()
    accumulated: dict[int, list[np.ndarray]] = {}

    with ResidualExtractor(module) as extractor:
        for start in range(0, len(series), batch_size):
            chunk = series[start : start + batch_size]
            batch = make_batch(
                chunk,
                patch_size=patch_size,
                context_patches=context_patches,
                pred_patches=pred_patches,
                device=device,
            )
            acts = extractor.run(batch)
            for layer_idx, tensor in acts.items():
                ctx = tensor[:, :context_patches, :].numpy()  # [chunk, context_patches, d_model]
                accumulated.setdefault(layer_idx, []).append(ctx)

    return {li: np.concatenate(chunks, axis=0) for li, chunks in accumulated.items()}


_DEFAULT_ALPHAS = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3])


def batched_ridge_per_patch(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_train: torch.Tensor,
    Y_val: torch.Tensor,
    alphas: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Batched ridge regression with LOO-CV alpha selection via SVD.

    Fits B independent ridge regressions simultaneously (one per patch position),
    selecting the best regularization strength per (B, feature) cell via leave-one-out CV.

    Parameters
    ----------
    X_train : [B, n_train, d]
    X_val   : [B, n_val,   d]
    Y_train : [n_train, k]   — same labels for all B patch positions
    Y_val   : [n_val,   k]
    alphas  : [A] alpha candidates (default: 7 log-spaced values)

    Returns
    -------
    r2 : [B, k]   — validation R² per (patch, feature); clamped to [-1, 1]
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS.to(X_train.device)

    B, n, d = X_train.shape
    k = Y_train.shape[1]
    A = len(alphas)

    # Standardize X per (B, d) using train statistics
    X_mean = X_train.mean(dim=1, keepdim=True)                        # [B, 1, d]
    X_std = X_train.std(dim=1, keepdim=True, correction=0).clamp(min=1e-8)          # [B, 1, d]
    X_train_n = (X_train - X_mean) / X_std                            # [B, n, d]
    X_val_n = (X_val - X_mean) / X_std                                # [B, n_val, d]

    # Center Y (matches sklearn fit_intercept=True; preserves intercept-free SVD)
    Y_mean = Y_train.mean(dim=0, keepdim=True)                        # [1, k]
    Y_train_c = Y_train - Y_mean                                      # [n, k]

    # Batched thin SVD
    U, S, Vh = torch.linalg.svd(X_train_n, full_matrices=False)       # U:[B,n,d], S:[B,d], Vh:[B,d,d]
    S2 = S.pow(2)                                                      # [B, d]
    UtY = torch.einsum("bni,nk->bik", U, Y_train_c)                   # [B, d, k]

    # LOO-CV on centered y: select best alpha per (B, k)
    # ŷ_c = U diag(s²/(s²+α)) Uᵀ y_c  ;  hat_diag = diag(U diag(s²/(s²+α)) Uᵀ)
    loo_mse_per_alpha = []
    for alpha in alphas:
        hat_filt = S2 / (S2 + alpha)                                   # [B, d]
        y_hat_c = torch.einsum("bni,bi,bik->bnk", U, hat_filt, UtY)  # [B, n, k]
        hat_diag = torch.einsum("bni,bi->bn", U.pow(2), hat_filt)     # [B, n]
        resid = (Y_train_c[None] - y_hat_c) / (1 - hat_diag[:, :, None]).clamp(min=1e-6)
        loo_mse_per_alpha.append(resid.pow(2).mean(dim=1))             # [B, k]

    loo_mse_all = torch.stack(loo_mse_per_alpha, dim=0)                # [A, B, k]
    best_alpha_idx = loo_mse_all.argmin(dim=0)                         # [B, k]

    # Build β for all alphas then gather the best per (B, k)
    beta_all = torch.stack(
        [torch.einsum("bdi,bi,bik->bdk", Vh.mT, S / (S2 + alpha), UtY) for alpha in alphas],
        dim=0,
    )                                                                   # [A, B, d, k]
    idx_exp = best_alpha_idx[None, :, None, :].expand(1, B, d, k)
    beta_best = beta_all.gather(0, idx_exp).squeeze(0)                 # [B, d, k]

    # Validation R² (restore Y_mean as intercept)
    y_val_hat = torch.einsum("bnd,bdk->bnk", X_val_n, beta_best) + Y_mean[None]  # [B, n_val, k]
    ss_res = (Y_val[None] - y_val_hat).pow(2).sum(dim=1)              # [B, k]
    ss_tot = (Y_val - Y_val.mean(0)).pow(2).sum(dim=0).clamp(min=1e-8)  # [k]
    return (1 - ss_res / ss_tot).clamp(min=-1.0)                       # [B, k]


def run_probes_per_patch(
    module,
    dataset: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
    patch_size: int = PATCH_SIZE,
    context_patches: int = CONTEXT_PATCHES,
    pred_patches: int = PRED_PATCHES,
    device: str | torch.device = "cpu",
) -> dict[str, dict[int, dict[int, float]]]:
    """
    Fit a linear probe at each (layer, patch_idx) independently using batched SVD ridge.

    Regression features use R² directly. Classification (period_idx) uses one-vs-rest
    binary ridge regression with argmax prediction → accuracy.

    Returns
    -------
    {feature_name: {layer_idx: {patch_idx: score}}}
    """
    series = dataset["series"]

    print(f"  Extracting per-patch train activations ({len(train_idx)} examples)...")
    X_train_by_layer = extract_activations_per_patch(
        module, series[train_idx], batch_size=batch_size,
        patch_size=patch_size, context_patches=context_patches,
        pred_patches=pred_patches, device=device,
    )
    print(f"  Extracting per-patch val activations ({len(val_idx)} examples)...")
    X_val_by_layer = extract_activations_per_patch(
        module, series[val_idx], batch_size=batch_size,
        patch_size=patch_size, context_patches=context_patches,
        pred_patches=pred_patches, device=device,
    )
    num_layers = len(X_train_by_layer)

    reg_features = [f for f in REGRESSION_FEATURES if f in dataset]
    clf_features = [f for f in CLASSIFICATION_FEATURES if f in dataset]

    # Regression: stack all regression labels into Y [n, k] and solve jointly
    results: dict[str, dict[int, dict[int, float]]] = {f: {} for f in reg_features + clf_features}

    if reg_features:
        Y_train_reg = torch.from_numpy(
            np.stack([dataset[f][train_idx].ravel() for f in reg_features], axis=1).astype(np.float32)
        )  # [n_train, k_reg]
        Y_val_reg = torch.from_numpy(
            np.stack([dataset[f][val_idx].ravel() for f in reg_features], axis=1).astype(np.float32)
        )  # [n_val, k_reg]

    # One-vs-rest encoding for classification
    if clf_features:
        # We handle each classification feature independently (usually just period_idx)
        clf_label_sets = {}
        for feat in clf_features:
            labels_train = dataset[feat][train_idx].ravel().astype(int)
            labels_val = dataset[feat][val_idx].ravel().astype(int)
            n_classes = int(labels_train.max()) + 1
            Y_onehot_train = torch.zeros(len(train_idx), n_classes)
            Y_onehot_train[torch.arange(len(train_idx)), labels_train] = 1.0
            Y_onehot_val = torch.zeros(len(val_idx), n_classes)
            Y_onehot_val[torch.arange(len(val_idx)), labels_val] = 1.0
            clf_label_sets[feat] = (labels_val, Y_onehot_train, Y_onehot_val)

    for layer_idx in range(num_layers):
        # Stack patch positions along batch dim: [B=context_patches, n, d]
        X_tr = torch.from_numpy(
            X_train_by_layer[layer_idx].transpose(1, 0, 2)  # [context_patches, n, d]
        )
        X_va = torch.from_numpy(
            X_val_by_layer[layer_idx].transpose(1, 0, 2)
        )
        B = X_tr.shape[0]

        # Regression probe
        if reg_features:
            with torch.no_grad():
                r2 = batched_ridge_per_patch(X_tr, X_va, Y_train_reg, Y_val_reg)  # [B, k_reg]
            for ki, feat in enumerate(reg_features):
                results[feat][layer_idx] = {p: float(r2[p, ki]) for p in range(B)}

        # Classification probe (one-vs-rest ridge)
        for feat in clf_features:
            labels_val, Y_oh_train, Y_oh_val = clf_label_sets[feat]
            with torch.no_grad():
                # r2 shape [B, n_classes] — use argmax on raw logits instead
                y_val_scores = _batched_ridge_predict(X_tr, X_va, Y_oh_train)  # [B, n_val, n_classes]
            preds = y_val_scores.argmax(dim=2).numpy()  # [B, n_val]
            acc_per_patch = (preds == labels_val[None, :]).mean(axis=1)  # [B]
            results[feat][layer_idx] = {p: float(acc_per_patch[p]) for p in range(B)}

        best_reg = {f: max(results[f][layer_idx].values()) for f in reg_features} if reg_features else {}
        best_clf = {f: max(results[f][layer_idx].values()) for f in clf_features} if clf_features else {}
        print(f"    Layer {layer_idx}: best patch scores — " +
              ", ".join(f"{f}={v:.3f}" for f, v in {**best_reg, **best_clf}.items()))

    return results


def _batched_ridge_predict(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_train: torch.Tensor,
    alphas: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Batched ridge regression returning raw val predictions (not R²).

    Used for classification via argmax on one-vs-rest targets.

    Returns
    -------
    y_val_hat : [B, n_val, k]
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS.to(X_train.device)

    B, n, d = X_train.shape
    k = Y_train.shape[1]
    A = len(alphas)

    X_mean = X_train.mean(dim=1, keepdim=True)
    X_std = X_train.std(dim=1, keepdim=True, correction=0).clamp(min=1e-8)
    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std

    Y_mean = Y_train.mean(dim=0, keepdim=True)
    Y_train_c = Y_train - Y_mean

    U, S, Vh = torch.linalg.svd(X_train_n, full_matrices=False)
    S2 = S.pow(2)
    UtY = torch.einsum("bni,nk->bik", U, Y_train_c)

    loo_mse_per_alpha = []
    for alpha in alphas:
        hat_filt = S2 / (S2 + alpha)
        y_hat_c = torch.einsum("bni,bi,bik->bnk", U, hat_filt, UtY)
        hat_diag = torch.einsum("bni,bi->bn", U.pow(2), hat_filt)
        resid = (Y_train_c[None] - y_hat_c) / (1 - hat_diag[:, :, None]).clamp(min=1e-6)
        loo_mse_per_alpha.append(resid.pow(2).mean(dim=1))

    loo_mse_all = torch.stack(loo_mse_per_alpha, dim=0)
    best_alpha_idx = loo_mse_all.argmin(dim=0)

    beta_all = torch.stack(
        [torch.einsum("bdi,bi,bik->bdk", Vh.mT, S / (S2 + alpha), UtY) for alpha in alphas],
        dim=0,
    )
    idx_exp = best_alpha_idx[None, :, None, :].expand(1, B, d, k)
    beta_best = beta_all.gather(0, idx_exp).squeeze(0)
    return torch.einsum("bnd,bdk->bnk", X_val_n, beta_best) + Y_mean[None]


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
    parser.add_argument("--per-patch", action="store_true",
                        help="Also run per-patch probing and write {model}_per_patch.json")
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

        if args.per_patch:
            print(f"\n=== {model_name} (per-patch) ===")
            per_patch_results = run_probes_per_patch(
                module, dataset, train_idx, val_idx,
                batch_size=args.batch_size,
                device=args.device,
            )
            # Serialize: int keys → str at both levels
            per_patch_serializable = {
                feat: {str(l): {str(p): v for p, v in patch_dict.items()} for l, patch_dict in layer_dict.items()}
                for feat, layer_dict in per_patch_results.items()
            }
            pp_path = os.path.join(args.output_dir, f"{model_name}_per_patch.json")
            with open(pp_path, "w") as f:
                json.dump(per_patch_serializable, f, indent=2)
            print(f"  Saved to {pp_path}")

    # Write feature metadata for downstream plotting (baselines, metric labels)
    metadata = {
        "features": {
            **{f: {"type": "regression", "baseline": 0.0, "metric": "R²"} for f in REGRESSION_FEATURES},
            **{
                f: {"type": "classification", "baseline": 1.0 / 8, "metric": "accuracy", "n_classes": 8}
                for f in CLASSIFICATION_FEATURES
            },
        }
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
