"""Shared probe utilities: constants, fit_probe, extract_activations, batched ridge."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.mech_interp.lib import ResidualExtractor, make_batch

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4
HORIZON = PRED_PATCHES * PATCH_SIZE  # 64
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
_DEFAULT_ALPHAS = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3])


def fit_probe(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_type: str,
) -> float:
    """
    Fit a linear probe and return the validation score.

    feature_type : "regression" → val R²
                   "binary"     → val AUROC
                   "classification" → val accuracy
    """
    if feature_type == "regression":
        probe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5))])
        probe.fit(X_train, y_train)
        return float(probe.score(X_val, y_val))
    elif feature_type == "binary":
        from sklearn.metrics import roc_auc_score

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(cv=5, max_iter=5000, n_jobs=-1)),
        ])
        clf.fit(X_train, y_train.astype(int))
        proba = clf.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val.astype(int), proba))
    elif feature_type == "classification":
        y_train_int = y_train.astype(int)
        y_val_int = y_val.astype(int)
        classes, counts = np.unique(y_train_int, return_counts=True)
        if len(classes) < 2:
            return float(np.mean(y_val_int == classes[0]))
        min_count = int(counts.min())
        if min_count >= 2:
            n_cv = min(5, min_count)
            clf = LogisticRegressionCV(cv=n_cv, max_iter=5000, n_jobs=-1)
        else:
            clf = LogisticRegression(C=1.0, max_iter=5000)
        probe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        probe.fit(X_train, y_train_int)
        return float(probe.score(X_val, y_val_int))
    else:
        raise ValueError(f"Unknown feature_type {feature_type!r}; expected 'regression', 'binary', or 'classification'")


def fit_probes(
    X_tr,
    X_va,
    y_specs: dict,
    backend: str = "auto",
    min_samples: int = 10,
) -> dict:
    """
    Fit probes for multiple features.

    y_specs : {feat: (y_tr, y_va, mask_tr, mask_va, ftype)}
              mask_tr/mask_va may be None for the per_patch backend.

    backend="pooled"   : calls fit_probe per feature, skips sparse masks.
                         Returns {feat: float_score}.
    backend="per_patch": calls batched_ridge_per_patch for all regression features at once.
                         Returns {feat: {patch_str: float_score}}.
    backend="auto"     : dispatches on X_tr type and ndim.
    """
    if backend == "auto":
        backend = "per_patch" if (isinstance(X_tr, torch.Tensor) and X_tr.ndim == 3) else "pooled"

    if backend == "pooled":
        results: dict = {}
        for feat, (y_tr, y_va, mask_tr, mask_va, ftype) in y_specs.items():
            if mask_tr.sum() < min_samples or mask_va.sum() < min_samples:
                continue
            results[feat] = fit_probe(
                X_tr[mask_tr], X_va[mask_va],
                y_tr[mask_tr], y_va[mask_va],
                ftype,
            )
        return results

    elif backend == "per_patch":
        reg_items = [(f, s) for f, s in y_specs.items() if s[4] == "regression"]
        if not reg_items:
            return {}

        # Separate NaN-containing features from clean ones; NaN must be masked
        # per-feature because each concept has a different missingness pattern.
        clean_items = [(f, s) for f, s in reg_items
                       if not (np.isnan(s[0]).any() or np.isnan(s[1]).any())]
        nan_items   = [(f, s) for f, s in reg_items
                       if np.isnan(s[0]).any() or np.isnan(s[1]).any()]

        results: dict = {}

        if clean_items:
            Y_tr = torch.from_numpy(
                np.stack([s[0] for _, s in clean_items], axis=1).astype(np.float32)
            )
            Y_va = torch.from_numpy(
                np.stack([s[1] for _, s in clean_items], axis=1).astype(np.float32)
            )
            with torch.no_grad():
                r2 = batched_ridge_per_patch(X_tr, X_va, Y_tr, Y_va)
            B = r2.shape[0]
            for ki, (feat, _) in enumerate(clean_items):
                results[feat] = {str(p): float(r2[p, ki]) for p in range(B)}

        B = X_tr.shape[0]
        for feat, s in nan_items:
            y_tr_f, y_va_f = s[0], s[1]
            mask_tr = np.isfinite(y_tr_f)
            mask_va = np.isfinite(y_va_f)
            if mask_tr.sum() < min_samples or mask_va.sum() < min_samples:
                # Too few finite labels to fit: emit NaN per patch rather than dropping
                # the feature, so every regression feature is always returned.
                results[feat] = {str(p): float("nan") for p in range(B)}
                continue
            Y_tr_f = torch.from_numpy(y_tr_f[mask_tr, None].astype(np.float32))
            Y_va_f = torch.from_numpy(y_va_f[mask_va, None].astype(np.float32))
            X_tr_f = X_tr[:, mask_tr, :]
            X_va_f = X_va[:, mask_va, :]
            with torch.no_grad():
                r2_f = batched_ridge_per_patch(X_tr_f, X_va_f, Y_tr_f, Y_va_f)
            B = r2_f.shape[0]
            results[feat] = {str(p): float(r2_f[p, 0]) for p in range(B)}

        return results

    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'pooled', or 'per_patch'")


def extract_activations(
    module,
    series: np.ndarray,
    batch_size: int = 32,
    patch_size: int = PATCH_SIZE,
    context_patches: int = CONTEXT_PATCHES,
    pred_patches: int = PRED_PATCHES,
    device: str | torch.device = "cpu",
    pooling: str | int = "mean_ctx",
) -> dict[int, np.ndarray]:
    """
    Extract pooled context activations for all examples.

    pooling:
        "mean_ctx" : mean over context patches → [n, d_model]
        "per_patch": no pooling → [n, context_patches, d_model]
        int k      : patch at position k → [n, d_model]

    Returns dict layer_idx → float32 array. Key -1 is post-projection (pre-attention) baseline.
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
            acts, _ = extractor.run(batch)
            for layer_idx, tensor in acts.items():
                ctx_acts = tensor[:, :context_patches, :]
                if pooling == "mean_ctx":
                    pooled = ctx_acts.mean(dim=1).numpy()
                elif pooling == "per_patch":
                    pooled = ctx_acts.numpy()
                elif isinstance(pooling, int):
                    pooled = ctx_acts[:, pooling, :].numpy()
                else:
                    raise ValueError(
                        f"Unknown pooling {pooling!r}; expected 'mean_ctx', 'per_patch', or int"
                    )
                accumulated.setdefault(layer_idx, []).append(pooled)

    return {layer_idx: np.concatenate(chunks, axis=0) for layer_idx, chunks in accumulated.items()}


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


def batched_ridge_per_patch(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_train: torch.Tensor,
    Y_val: torch.Tensor,
    alphas: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Batched ridge regression with LOO-CV alpha selection via SVD.

    Parameters
    ----------
    X_train : [B, n_train, d]
    X_val   : [B, n_val,   d]
    Y_train : [n_train, k]
    Y_val   : [n_val,   k]
    alphas  : [A] alpha candidates (default: 7 log-spaced values)

    Returns
    -------
    r2 : [B, k]  validation R² per (patch, feature); clamped to [-1, 1]
    """
    y_val_hat = _batched_ridge_predict(X_train, X_val, Y_train, alphas)
    ss_res = (Y_val[None] - y_val_hat).pow(2).sum(dim=1)
    ss_tot = (Y_val - Y_val.mean(0)).pow(2).sum(dim=0).clamp(min=1e-8)
    return (1 - ss_res / ss_tot).clamp(min=-1.0)


def _batched_ridge_predict(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_train: torch.Tensor,
    alphas: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Batched ridge regression returning raw val predictions (not R²).

    Returns
    -------
    y_val_hat : [B, n_val, k]
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS.to(X_train.device)

    B, n, d = X_train.shape
    k = Y_train.shape[1]

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


def _batched_ridge_predict_local(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_train: torch.Tensor,
    alphas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched ridge with *per-patch* labels (Y varies along the batch/patch dim).

    Mirrors ``_batched_ridge_predict`` but ``Y_train`` is ``[B, n_train, k]`` instead of the
    shared ``[n_train, k]``; returns ``y_val_hat : [B, n_val, k]``.
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS.to(X_train.device)

    B, n, d = X_train.shape
    k = Y_train.shape[2]

    X_mean = X_train.mean(dim=1, keepdim=True)
    X_std = X_train.std(dim=1, keepdim=True, correction=0).clamp(min=1e-8)
    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std

    Y_mean = Y_train.mean(dim=1, keepdim=True)        # [B, 1, k]
    Y_train_c = Y_train - Y_mean

    U, S, Vh = torch.linalg.svd(X_train_n, full_matrices=False)
    S2 = S.pow(2)
    UtY = torch.einsum("bni,bnk->bik", U, Y_train_c)

    loo_mse_per_alpha = []
    for alpha in alphas:
        hat_filt = S2 / (S2 + alpha)
        y_hat_c = torch.einsum("bni,bi,bik->bnk", U, hat_filt, UtY)
        hat_diag = torch.einsum("bni,bi->bn", U.pow(2), hat_filt)
        resid = (Y_train_c - y_hat_c) / (1 - hat_diag[:, :, None]).clamp(min=1e-6)
        loo_mse_per_alpha.append(resid.pow(2).mean(dim=1))

    loo_mse_all = torch.stack(loo_mse_per_alpha, dim=0)
    best_alpha_idx = loo_mse_all.argmin(dim=0)

    beta_all = torch.stack(
        [torch.einsum("bdi,bi,bik->bdk", Vh.mT, S / (S2 + alpha), UtY) for alpha in alphas],
        dim=0,
    )
    idx_exp = best_alpha_idx[None, :, None, :].expand(1, B, d, k)
    beta_best = beta_all.gather(0, idx_exp).squeeze(0)
    return torch.einsum("bnd,bdk->bnk", X_val_n, beta_best) + Y_mean


def batched_ridge_per_patch_local(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_train: torch.Tensor,
    Y_val: torch.Tensor,
    alphas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-patch ridge R² where labels are local to each patch.

    X_train/X_val : [B, n, d] / [B, m, d]   (B = patch positions)
    Y_train/Y_val : [B, n, k] / [B, m, k]
    Returns r2 : [B, k] clamped to [-1, ∞).

    Positions whose target column is all-NaN (e.g. boundary patches for the
    cross-patch features) are skipped and reported as NaN — mirroring the
    per-position guard on the binary path.
    """
    # All-NaN (position, feature) slices: skip the ridge fit and emit NaN.
    nan_mask = torch.isnan(Y_train).all(dim=1)            # [B, k]
    Y_train = torch.nan_to_num(Y_train)
    Y_val = torch.nan_to_num(Y_val)

    y_val_hat = _batched_ridge_predict_local(X_train, X_val, Y_train, alphas)
    ss_res = (Y_val - y_val_hat).pow(2).sum(dim=1)
    ss_tot = (Y_val - Y_val.mean(1, keepdim=True)).pow(2).sum(dim=1).clamp(min=1e-8)
    r2 = (1 - ss_res / ss_tot).clamp(min=-1.0)
    r2[nan_mask] = float("nan")
    return r2
