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

from experiments.mech_interp.lib import ResidualExtractor, generate_composite_dataset, generate_dataset, load_dataset, make_batch

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4

REGRESSION_FEATURES = [
    "slope", "log_noise_var", "phase_cos", "phase_sin",
    "level_magnitude", "level_time_norm", "ar_phi",
    "seasonal_amplitude", "log_sigma_ratio", "var_shift_time_norm",
]
BINARY_FEATURES = ["spike_present", "rw_present"]
CLASSIFICATION_FEATURES = [
    "period_idx",
    "spike_patch_idx",   # 32-class per-patch; handled in run_probes_per_patch only
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
    pooling: str = "mean_ctx",
) -> dict[int, np.ndarray]:
    """
    Extract pooled context activations for all examples.

    Parameters
    ----------
    module : MoiraieModule or MoiraicModule
    series : float32 [n, time]
    batch_size : int
    pooling : "mean_ctx" | "last_ctx" | "per_patch"
        "mean_ctx": mean over all context patches → [n, d_model]
        "last_ctx": last context patch (position context_patches-1) → [n, d_model]
        "per_patch": no pooling, preserves patch axis → [n, context_patches, d_model]

    Returns
    -------
    dict layer_idx -> float32 array
        Shape [n, d_model] for "mean_ctx"/"last_ctx", [n, context_patches, d_model] for "per_patch".
        Includes key -1 for the post-projection, pre-attention (in_proj) activation.
    """
    module.eval()
    accumulated: dict[int, list[np.ndarray]] = {}

    with ResidualExtractor(module) as extractor:
        for start in range(0, len(series), batch_size):
            chunk = series[start : start + batch_size]
            batch = make_batch(chunk, patch_size=patch_size, context_patches=context_patches, pred_patches=pred_patches, device=device)
            acts = extractor.run(batch)  # dict[int, Tensor[chunk, n_patches, d_model]]
            for layer_idx, tensor in acts.items():
                ctx_acts = tensor[:, :context_patches, :]   # [chunk, context_patches, d_model]
                if pooling == "mean_ctx":
                    pooled = ctx_acts.mean(dim=1).numpy()   # [chunk, d_model]
                elif pooling == "last_ctx":
                    pooled = ctx_acts[:, -1, :].numpy()     # [chunk, d_model] — position context_patches-1
                elif pooling == "per_patch":
                    pooled = ctx_acts.numpy()               # [chunk, context_patches, d_model]
                else:
                    raise ValueError(f"Unknown pooling {pooling!r}; expected 'mean_ctx', 'last_ctx', or 'per_patch'")
                accumulated.setdefault(layer_idx, []).append(pooled)

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
    elif feature_type == "binary":
        from sklearn.metrics import roc_auc_score

        clf = Pipeline([("scaler", StandardScaler()),
                        ("clf", LogisticRegressionCV(cv=5, max_iter=5000, n_jobs=-1))])
        clf.fit(X_train, y_train.astype(int))
        proba = clf.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val.astype(int), proba))
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
        raise ValueError(f"Unknown feature_type {feature_type!r}; expected 'regression', 'binary', or 'classification'")


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
) -> dict[str, dict[str, dict[int, float]]]:
    """
    Full probe training pipeline for one model, run under both pooling modes.

    Parameters
    ----------
    module : MoiraieModule or MoiraicModule (eval mode)
    dataset : dict with keys "series", plus label arrays
    train_idx, val_idx : integer index arrays for the 80/20 split

    Returns
    -------
    {pooling_mode: {feature_name: {layer_idx: score}}}
        pooling_mode ∈ {"mean_ctx", "last_ctx"}
        layer_idx includes -1 (post-projection, pre-attention baseline)
    """
    series = dataset["series"]
    all_features = (
        [(f, "regression") for f in REGRESSION_FEATURES if f in dataset]
        + [(f, "binary")       for f in BINARY_FEATURES       if f in dataset]
        + [(f, "classification") for f in CLASSIFICATION_FEATURES if f in dataset]
    )

    results: dict[str, dict[str, dict[int, float]]] = {}

    for pooling in ("mean_ctx", "last_ctx"):
        print(f"  Extracting train activations [{pooling}] ({len(train_idx)} examples)...")
        X_train_by_layer = extract_activations(
            module, series[train_idx], batch_size=batch_size,
            patch_size=patch_size, context_patches=context_patches, pred_patches=pred_patches,
            device=device, pooling=pooling,
        )

        print(f"  Extracting val activations [{pooling}] ({len(val_idx)} examples)...")
        X_val_by_layer = extract_activations(
            module, series[val_idx], batch_size=batch_size,
            patch_size=patch_size, context_patches=context_patches, pred_patches=pred_patches,
            device=device, pooling=pooling,
        )

        layer_keys = sorted(X_train_by_layer.keys())
        pooling_results: dict[str, dict[int, float]] = {}

        for feature, ftype in all_features:
            y_train = dataset[feature][train_idx].ravel()
            y_val = dataset[feature][val_idx].ravel()

            # Compute valid-row masks once per feature (same across all layers)
            if np.issubdtype(y_train.dtype, np.floating):
                tr_mask = np.isfinite(y_train)
                va_mask = np.isfinite(y_val)
            elif feature == "period_idx":
                tr_mask = y_train >= 0
                va_mask = y_val >= 0
            else:
                tr_mask = np.ones(len(y_train), dtype=bool)
                va_mask = np.ones(len(y_val), dtype=bool)

            if tr_mask.sum() < 10 or va_mask.sum() < 10:
                continue

            layer_scores: dict[int, float] = {}
            for layer_idx in layer_keys:
                X_tr = X_train_by_layer[layer_idx][tr_mask]
                X_va = X_val_by_layer[layer_idx][va_mask]
                score = fit_probe(X_tr, X_va, y_train[tr_mask], y_val[va_mask], ftype)
                layer_scores[layer_idx] = score
            pooling_results[feature] = layer_scores
            best = max(layer_scores.values())
            print(f"    [{pooling}] {feature}: best layer score = {best:.4f}")

        results[pooling] = pooling_results

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
    """Deprecated: use extract_activations(..., pooling='per_patch') instead."""
    return extract_activations(
        module, series,
        batch_size=batch_size, patch_size=patch_size,
        context_patches=context_patches, pred_patches=pred_patches,
        device=device, pooling="per_patch",
    )


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
    y_val_hat = _batched_ridge_predict(X_train, X_val, Y_train, alphas)  # [B, n_val, k]
    ss_res = (Y_val[None] - y_val_hat).pow(2).sum(dim=1)                 # [B, k]
    ss_tot = (Y_val - Y_val.mean(0)).pow(2).sum(dim=0).clamp(min=1e-8)  # [k]
    return (1 - ss_res / ss_tot).clamp(min=-1.0)                          # [B, k]


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
    X_train_by_layer = extract_activations(
        module, series[train_idx], batch_size=batch_size,
        patch_size=patch_size, context_patches=context_patches,
        pred_patches=pred_patches, device=device, pooling="per_patch",
    )
    print(f"  Extracting per-patch val activations ({len(val_idx)} examples)...")
    X_val_by_layer = extract_activations(
        module, series[val_idx], batch_size=batch_size,
        patch_size=patch_size, context_patches=context_patches,
        pred_patches=pred_patches, device=device, pooling="per_patch",
    )
    layer_keys = sorted(X_train_by_layer.keys())

    reg_features = [f for f in REGRESSION_FEATURES if f in dataset]
    clf_features = [f for f in CLASSIFICATION_FEATURES if f in dataset]

    # Regression: stack all regression labels into Y [n, k] and solve jointly
    # (clf_label_sets populated below; results initialized after to exclude skipped clf features)
    clf_label_sets: dict = {}
    results: dict[str, dict[int, dict[int, float]]] = {f: {} for f in reg_features}

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
        for feat in clf_features:
            if feat == "spike_patch_idx":
                # Binary per-patch probe: is this the spike patch?
                # Mask out rows with no spike (spike_patch_idx == -1)
                spike_arr = dataset[feat]
                mask_tr = spike_arr[train_idx] >= 0  # [n_train] bool
                mask_va = spike_arr[val_idx] >= 0    # [n_val] bool
                n_sp_tr = int(mask_tr.sum())
                n_sp_va = int(mask_va.sum())
                if n_sp_tr < 2 or n_sp_va < 2:
                    continue
                labels_tr_sp = spike_arr[train_idx][mask_tr].astype(int)
                labels_va_sp = spike_arr[val_idx][mask_va].astype(int)
                Y_oh_tr_sp = torch.zeros(n_sp_tr, context_patches)
                Y_oh_tr_sp[torch.arange(n_sp_tr), labels_tr_sp] = 1.0
                clf_label_sets[feat] = ("spike", mask_tr, mask_va, labels_va_sp, Y_oh_tr_sp)
            else:
                labels_train = dataset[feat][train_idx].ravel().astype(int)
                labels_val = dataset[feat][val_idx].ravel().astype(int)
                n_classes = int(labels_train.max()) + 1
                Y_onehot_train = torch.zeros(len(train_idx), n_classes)
                Y_onehot_train[torch.arange(len(train_idx)), labels_train] = 1.0
                Y_onehot_val = torch.zeros(len(val_idx), n_classes)
                Y_onehot_val[torch.arange(len(val_idx)), labels_val] = 1.0
                clf_label_sets[feat] = ("standard", labels_val, Y_onehot_train, Y_onehot_val)

    # Initialize results for clf features that were not skipped
    for feat in clf_label_sets:
        results[feat] = {}

    for layer_idx in layer_keys:
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
        for feat, entry in clf_label_sets.items():
            kind = entry[0]
            if kind == "spike":
                _, mask_tr_sp, mask_va_sp, labels_va_sp, Y_oh_tr_sp = entry
                X_tr_sp = X_tr[:, mask_tr_sp, :]   # [B, n_sp_tr, d]
                X_va_sp = X_va[:, mask_va_sp, :]   # [B, n_sp_va, d]
                with torch.no_grad():
                    y_scores = _batched_ridge_predict(X_tr_sp, X_va_sp, Y_oh_tr_sp).numpy()  # [B, n_sp_va, 32]
                from sklearn.metrics import roc_auc_score
                auroc = np.zeros(B)
                for p in range(B):
                    binary_tgt = (labels_va_sp == p).astype(int)
                    pos = int(binary_tgt.sum())
                    if pos < 1 or pos == len(binary_tgt):
                        auroc[p] = 0.5
                    else:
                        try:
                            auroc[p] = float(roc_auc_score(binary_tgt, y_scores[p, :, p]))
                        except Exception:
                            auroc[p] = 0.5
                results[feat][layer_idx] = {p: float(auroc[p]) for p in range(B)}
            else:
                _, labels_val, Y_oh_train, Y_oh_val = entry
                with torch.no_grad():
                    y_val_scores = _batched_ridge_predict(X_tr, X_va, Y_oh_train)  # [B, n_val, n_classes]
                preds = y_val_scores.argmax(dim=2).numpy()  # [B, n_val]
                acc_per_patch = (preds == labels_val[None, :]).mean(axis=1)  # [B]
                results[feat][layer_idx] = {p: float(acc_per_patch[p]) for p in range(B)}

        active_clf = list(clf_label_sets.keys())
        best_reg = {f: max(results[f][layer_idx].values()) for f in reg_features} if reg_features else {}
        best_clf = {f: max(results[f][layer_idx].values()) for f in active_clf} if active_clf else {}
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
    parser.add_argument("--output-dir", default="experiments/mech_interp/block1_probing/results/synthetic",
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
        print("Generating composite synthetic dataset (n=5000, seed=42) in memory...")
        dataset = generate_composite_dataset(n=5000, seed=42)

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

        # Convert int layer keys to strings for JSON serialization
        # Format: {pooling_mode: {feature: {layer_str: score}}}
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
    clf_meta = {
        "period_idx":      {"n_classes": 8,  "baseline": 1 / 8,  "metric": "accuracy"},
        "spike_patch_idx": {"n_classes": 32, "baseline": 1 / 32, "metric": "accuracy"},
    }
    bin_meta = {
        "spike_present": {"metric": "AUROC", "baseline": 0.5},
        "rw_present":    {"metric": "AUROC", "baseline": 0.5},
    }
    metadata = {
        "features": {
            **{f: {"type": "regression", "baseline": 0.0, "metric": "R²"} for f in REGRESSION_FEATURES},
            **{f: {"type": "binary",         **bin_meta[f]}  for f in BINARY_FEATURES},
            **{f: {"type": "classification", **clf_meta.get(f, {"n_classes": 2, "baseline": 0.5, "metric": "accuracy"})}
               for f in CLASSIFICATION_FEATURES},
        }
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
