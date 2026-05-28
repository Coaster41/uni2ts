"""
Universal probe runner (block1_probing).

Usage
-----
python -m experiments.mech_interp.block1_probing.train_probes \\
    --dataset [synth | real | corruption-synth | corruption-real] \\
    --pooling mean [<int> ...] [all] \\
    --forecast / --no-forecast \\
    --model [moiraie | moiraic | both] \\
    --moiraie-ckpt PATH --moiraic-ckpt PATH \\
    --output-dir PATH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.lib import ResidualExtractor, compute_patch_features, make_batch
from experiments.mech_interp.lib.utils import _load_module
from experiments.mech_interp.block1_probing.probe_utils import (
    PATCH_SIZE,
    CONTEXT_PATCHES,
    PRED_PATCHES,
    HORIZON,
    RIDGE_ALPHAS,
    fit_probe,
    fit_probes,
    extract_activations,
    extract_activations_per_patch,
    batched_ridge_per_patch,
    batched_ridge_per_patch_local,
    _batched_ridge_predict,
    _batched_ridge_predict_local,
)

# ---------------------------------------------------------------------------
# Feature registries
# ---------------------------------------------------------------------------

SYNTH_REGRESSION_FEATURES = [
    "slope", "log_noise_var", "phase_cos", "phase_sin",
    "level_magnitude", "level_time_norm", "ar_phi",
    "seasonal_amplitude", "log_sigma_ratio", "var_shift_time_norm",
]
SYNTH_BINARY_FEATURES = ["spike_present", "rw_present"]
SYNTH_CLASSIFICATION_FEATURES = [
    "period_idx",
    "spike_patch_idx",  # per-patch only
]

# Backward-compat aliases used by tests and downstream scripts
REGRESSION_FEATURES = SYNTH_REGRESSION_FEATURES
BINARY_FEATURES = SYNTH_BINARY_FEATURES
CLASSIFICATION_FEATURES = SYNTH_CLASSIFICATION_FEATURES

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
REAL_CLASSIFICATION_FEATURES = ["dataset_id"]
N_DATASET_CLASSES = 9

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

FORECAST_REGRESSION_FEATURES = [
    "fc_std",
    "fc_range",
    "fc_ctx_corr",
    "fc_ctx_corr_seasonal",
    "fc_iqr_mean",
    "fc_iqr_mean_scaled",
    "fc_iqr_slope",
    "mase",
    "swql",
    "quantile_calibration_err",
]
FORECAST_BINARY_FEATURES = ["is_flat", "is_poor"]

# Per-patch *local* features (computed in lib/patch_features.py). Group C (the
# ground-truth anomaly flags) is only produced for synthetic data.
PATCH_REGRESSION_FEATURES = [
    "patch_mean", "patch_std", "patch_slope", "patch_range",
    "patch_acf1", "patch_net_change",
    "patch_mean_dev", "patch_logstd_ratio", "patch_mean_rank",
]
PATCH_BINARY_FEATURES = [
    "patch_is_level_outlier", "patch_has_pointspike",          # data-driven (any data)
    "patch_has_spike", "patch_has_levelshift",                 # ground-truth (synth only)
    "patch_has_varshift", "patch_is_post_varshift",
]


# ---------------------------------------------------------------------------
# Shared probe helpers
# ---------------------------------------------------------------------------

def _make_mask(
    y_tr: np.ndarray, y_va: np.ndarray, feat: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean validity masks for train/val labels."""
    if np.issubdtype(y_tr.dtype, np.floating):
        return np.isfinite(y_tr), np.isfinite(y_va)
    elif feat == "period_idx":
        return y_tr >= 0, y_va >= 0
    else:
        return np.ones(len(y_tr), dtype=bool), np.ones(len(y_va), dtype=bool)


# ---------------------------------------------------------------------------
# Corruption helpers
# ---------------------------------------------------------------------------

def _apply_corruptions(
    base_dataset: dict,
    series_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return {corr_name: float32 [n, T]} for all 8 corruption variants."""
    from experiments.mech_interp.lib import (
        corrupt_mean_center,
        corrupt_noise,
        corrupt_reverse,
        corrupt_seasonal,
        corrupt_shuffle_patches,
        corrupt_trend,
        corrupt_zero_segment,
    )
    from experiments.mech_interp.lib.synthetic import PERIOD_BINS

    n = len(series_indices)
    T = base_dataset["series"].shape[1]
    result: dict[str, np.ndarray] = {c: np.empty((n, T), dtype=np.float32) for c in CORRUPTION_NAMES}

    has_synth_labels = "slope" in base_dataset and "period_idx" in base_dataset

    for out_i, orig_i in enumerate(series_indices):
        series = base_dataset["series"][orig_i]
        seed = int(orig_i) * 7919

        result["clean"][out_i] = series
        result["no_trend"][out_i] = corrupt_trend(series, slope=0.0)

        if has_synth_labels:
            period_idx = int(base_dataset["period_idx"][orig_i])
            phase = float(np.arctan2(base_dataset["phase_sin"][orig_i], base_dataset["phase_cos"][orig_i]))
        else:
            # For real data: estimate period from fft_dominant_period (log-period), phase=0
            if "fft_dominant_period" in base_dataset:
                raw_period = float(np.exp(base_dataset["fft_dominant_period"][orig_i]))
            else:
                raw_period = 24.0
            period_idx = int(np.argmin(np.abs(np.array(PERIOD_BINS, dtype=float) - raw_period)))
            phase = 0.0

        if np.isnan(phase) or period_idx < 0:
            result["no_seasonal"][out_i] = series
        else:
            result["no_seasonal"][out_i] = corrupt_seasonal(series, period_idx, phase)
        result["noise"][out_i] = corrupt_noise(series, seed)
        result["mean_center"][out_i] = corrupt_mean_center(series)
        result["reverse"][out_i] = corrupt_reverse(series)
        result["shuffle"][out_i] = corrupt_shuffle_patches(series, seed)
        result["zero_segment"][out_i] = corrupt_zero_segment(series, seed)

    return result


# ---------------------------------------------------------------------------
# Extraction with optional forecast capture
# ---------------------------------------------------------------------------

def _extract_multi_pooling(
    module,
    series: np.ndarray,
    batch_size: int,
    poolings: list,          # list of "mean_ctx" | int k | "per_patch"
    device: str | torch.device,
    capture_forecast: bool = False,
) -> tuple[dict, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Single forward-pass extraction for multiple pooling modes.

    Returns
    -------
    acts : dict[pooling, dict[layer_idx, np.ndarray]]
    fq   : [n, 9, HORIZON] or None
    ctx  : [n, CONTEXT_PATCHES * PATCH_SIZE] or None
    tgt  : [n, PRED_PATCHES * PATCH_SIZE] or None
    """
    from experiments.mech_interp.block1_probing.forecast_runner import _extract_fq

    n = len(series)
    is_moiraic = type(module).__name__.startswith("Moiraic")
    if capture_forecast:
        npt = module.num_predict_token
        Q = module.num_quantiles
        P = module.patch_size

    module.eval()
    if hasattr(module, "to"):
        module.to(device)

    accum: dict = {p: {} for p in poolings}
    fq_buf: list[np.ndarray] = []

    with ResidualExtractor(module) as extractor:
        for start in range(0, n, batch_size):
            chunk = series[start : start + batch_size]
            batch = make_batch(chunk, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES, device)
            acts_tensors, output = extractor.run(batch)

            if capture_forecast:
                output_np = output.detach().cpu().float().numpy()
                fq_buf.append(_extract_fq(output_np, is_moiraic, npt, Q, P))

            for layer_idx, tensor in acts_tensors.items():
                ctx_acts = tensor[:, :CONTEXT_PATCHES, :]
                for pooling in poolings:
                    if pooling == "mean_ctx":
                        pooled = ctx_acts.mean(dim=1).numpy()
                    elif pooling == "per_patch":
                        pooled = ctx_acts.numpy()
                    elif isinstance(pooling, int):
                        pooled = ctx_acts[:, pooling, :].numpy()
                    else:
                        raise ValueError(f"Unknown pooling {pooling!r}")
                    accum[pooling].setdefault(layer_idx, []).append(pooled)

    result_acts = {
        p: {l: np.concatenate(chunks, axis=0) for l, chunks in layer_dict.items()}
        for p, layer_dict in accum.items()
    }
    if capture_forecast:
        fq_all = np.concatenate(fq_buf, axis=0)
        ctx_all = series[:, : CONTEXT_PATCHES * PATCH_SIZE].astype(np.float32)
        tgt_all = series[:, CONTEXT_PATCHES * PATCH_SIZE :].astype(np.float32)
        return result_acts, fq_all, ctx_all, tgt_all
    return result_acts, None, None, None


# ---------------------------------------------------------------------------
# Forecast target computation (inline)
# ---------------------------------------------------------------------------

def _compute_forecast_targets(
    fq_all: np.ndarray,
    ctx_all: np.ndarray,
    tgt_all: np.ndarray,
    ctx_period: int | np.ndarray = 24,
) -> dict[str, np.ndarray]:
    """Compute per-series forecast property targets from quantile forecast tensors."""
    from experiments.mech_interp.block1_probing.forecast_properties import (
        compute_all,
        derive_binary_labels,
    )

    n = len(fq_all)
    per_series_period = isinstance(ctx_period, np.ndarray)

    accum: dict[str, list[float]] = {k: [] for k in FORECAST_REGRESSION_FEATURES}
    for i in range(n):
        p = int(ctx_period[i]) if per_series_period else int(ctx_period)
        props = compute_all(fq_all[i], tgt_all[i], ctx_all[i], p)
        for k in FORECAST_REGRESSION_FEATURES:
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
# Core probe training loop (per model, per mode)
# ---------------------------------------------------------------------------

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
) -> dict:
    """
    Full probe training pipeline for one model; runs mean_ctx and position-(context_patches-1) pooling.

    Returns {pooling_mode: {feature_name: {layer_idx: score}}}
        pooling_mode ∈ {"mean_ctx", context_patches - 1}
    """
    series = dataset["series"]
    all_features = (
        [(f, "regression")     for f in SYNTH_REGRESSION_FEATURES     if f in dataset]
        + [(f, "binary")       for f in SYNTH_BINARY_FEATURES         if f in dataset]
        + [(f, "classification") for f in SYNTH_CLASSIFICATION_FEATURES if f in dataset]
    )

    results: dict = {}
    last_ctx_idx = context_patches - 1

    for pooling in ("mean_ctx", last_ctx_idx):
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

        # Build y_specs once per pooling (masks computed from train/val labels)
        y_specs: dict = {}
        for feature, ftype in all_features:
            y_tr = dataset[feature][train_idx].ravel()
            y_va = dataset[feature][val_idx].ravel()
            mask_tr, mask_va = _make_mask(y_tr, y_va, feature)
            y_specs[feature] = (y_tr, y_va, mask_tr, mask_va, ftype)

        pooling_results: dict[str, dict[int, float]] = {}
        for layer_idx in layer_keys:
            layer_scores = fit_probes(X_train_by_layer[layer_idx], X_val_by_layer[layer_idx], y_specs)
            for feat, score in layer_scores.items():
                pooling_results.setdefault(feat, {})[layer_idx] = score  # integer layer key

        for feat, layer_dict in pooling_results.items():
            best = max(layer_dict.values())
            print(f"    [{pooling}] {feat}: best layer score = {best:.4f}")

        results[pooling] = pooling_results

    return results


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

    Returns {feature_name: {layer_idx: {patch_idx: score}}}
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

    reg_features = [f for f in SYNTH_REGRESSION_FEATURES if f in dataset]
    clf_features = [f for f in SYNTH_CLASSIFICATION_FEATURES if f in dataset]

    clf_label_sets: dict = {}
    results: dict[str, dict[int, dict[int, float]]] = {f: {} for f in reg_features}

    if reg_features:
        Y_train_reg = torch.from_numpy(
            np.stack([dataset[f][train_idx].ravel() for f in reg_features], axis=1).astype(np.float32)
        )
        Y_val_reg = torch.from_numpy(
            np.stack([dataset[f][val_idx].ravel() for f in reg_features], axis=1).astype(np.float32)
        )

    if clf_features:
        for feat in clf_features:
            if feat == "spike_patch_idx":
                spike_arr = dataset[feat]
                mask_tr = spike_arr[train_idx] >= 0
                mask_va = spike_arr[val_idx] >= 0
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

    for feat in clf_label_sets:
        results[feat] = {}

    for layer_idx in layer_keys:
        X_tr = torch.from_numpy(X_train_by_layer[layer_idx].transpose(1, 0, 2))
        X_va = torch.from_numpy(X_val_by_layer[layer_idx].transpose(1, 0, 2))
        B = X_tr.shape[0]

        if reg_features:
            with torch.no_grad():
                r2 = batched_ridge_per_patch(X_tr, X_va, Y_train_reg, Y_val_reg)
            for ki, feat in enumerate(reg_features):
                results[feat][layer_idx] = {p: float(r2[p, ki]) for p in range(B)}

        for feat, entry in clf_label_sets.items():
            kind = entry[0]
            if kind == "spike":
                _, mask_tr_sp, mask_va_sp, labels_va_sp, Y_oh_tr_sp = entry
                X_tr_sp = X_tr[:, mask_tr_sp, :]
                X_va_sp = X_va[:, mask_va_sp, :]
                with torch.no_grad():
                    y_scores = _batched_ridge_predict(X_tr_sp, X_va_sp, Y_oh_tr_sp).numpy()
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
                    y_val_scores = _batched_ridge_predict(X_tr, X_va, Y_oh_train)
                preds = y_val_scores.argmax(dim=2).numpy()
                acc_per_patch = (preds == labels_val[None, :]).mean(axis=1)
                results[feat][layer_idx] = {p: float(acc_per_patch[p]) for p in range(B)}

        active_clf = list(clf_label_sets.keys())
        best_reg = {f: max(results[f][layer_idx].values()) for f in reg_features} if reg_features else {}
        best_clf = {f: max(results[f][layer_idx].values()) for f in active_clf} if active_clf else {}
        print(
            f"    Layer {layer_idx}: best patch scores — "
            + ", ".join(f"{f}={v:.3f}" for f, v in {**best_reg, **best_clf}.items())
        )

    return results


# ---------------------------------------------------------------------------
# Universal runner helpers
# ---------------------------------------------------------------------------

def _pooling_key_str(pooling: str | int) -> str:
    """Convert pooling spec to a JSON-safe string key."""
    if isinstance(pooling, int):
        return f"pos_{pooling}"
    return pooling  # "mean_ctx", "per_patch"


def _parse_pooling_args(raw: list[str]) -> tuple[list[str | int], bool]:
    """
    Parse --pooling values.

    Returns (poolings, run_per_patch) where:
        poolings     : list of "mean_ctx" | int k  (for pooled results)
        run_per_patch: True if "all" was present
    """
    poolings: list[str | int] = []
    run_per_patch = False
    has_all = "all" in raw
    if has_all:
        run_per_patch = True

    for token in raw:
        if token == "all":
            continue
        elif token == "mean":
            if "mean_ctx" not in poolings:
                poolings.append("mean_ctx")
        else:
            try:
                k = int(token)
            except ValueError:
                raise ValueError(f"Unknown --pooling token {token!r}; expected 'mean', 'all', or an integer")
            if has_all:
                warnings.warn(
                    f"--pooling {k} is redundant when 'all' is also specified; skipping.",
                    stacklevel=2,
                )
            elif k not in poolings:
                poolings.append(k)

    if not poolings and not run_per_patch:
        poolings = ["mean_ctx"]

    return poolings, run_per_patch


def _run_pooled_probes(
    module,
    series: np.ndarray,
    features: list[tuple[str, str]],    # [(name, feature_type), ...]
    feature_data: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    poolings: list[str | int],
    batch_size: int,
    device: str | torch.device,
    with_forecast: bool,
    ctx_period: int | np.ndarray = 24,
) -> dict:
    """
    Run pooled probes for a list of pooling modes.

    Extracts activations for the full dataset in one pass, then slices by
    train_idx / val_idx when fitting probes.

    Returns {pooling_key_str: {feature: {layer_str: score}}}
    """
    acts, fq_all, ctx_all, tgt_all = _extract_multi_pooling(
        module, series, batch_size, poolings, device,
        capture_forecast=with_forecast,
    )

    if with_forecast:
        fc_targets = _compute_forecast_targets(fq_all, ctx_all, tgt_all, ctx_period)
        all_features = (
            features
            + [(f, "regression") for f in FORECAST_REGRESSION_FEATURES]
            + [(f, "binary")     for f in FORECAST_BINARY_FEATURES]
        )
        merged_data = {**feature_data, **fc_targets}
    else:
        all_features = features
        merged_data = feature_data

    # Build y_specs once — shared across all poolings and layers
    y_specs: dict = {}
    for feat, ftype in all_features:
        if feat not in merged_data:
            continue
        y = merged_data[feat]
        y_tr = y[train_idx].ravel()
        y_va = y[val_idx].ravel()
        mask_tr, mask_va = _make_mask(y_tr, y_va, feat)
        y_specs[feat] = (y_tr, y_va, mask_tr, mask_va, ftype)

    results: dict = {}

    for pooling in poolings:
        X_by_layer = acts[pooling]
        pkey = _pooling_key_str(pooling)
        pooling_results: dict[str, dict[str, float]] = {}

        for layer_idx in sorted(X_by_layer.keys()):
            X_full = X_by_layer[layer_idx]    # [n, d]
            layer_scores = fit_probes(X_full[train_idx], X_full[val_idx], y_specs)
            for feat, score in layer_scores.items():
                pooling_results.setdefault(feat, {})[str(layer_idx)] = score

        for feat, layer_dict in pooling_results.items():
            best = max(float(v) for v in layer_dict.values())
            print(f"    [{pkey}] {feat}: best layer score = {best:.4f}")

        results[pkey] = pooling_results

    return results


# ---------------------------------------------------------------------------
# Corruption-mode probe runner
# ---------------------------------------------------------------------------

def _stack_corruption_xy(
    corr_acts: dict,
    pooling,
    layer_idx,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_id_tr: np.ndarray,    # [N_CORRUPTIONS * n_tr], pre-built
    y_id_va: np.ndarray,    # [N_CORRUPTIONS * n_va], pre-built
    fc_tr: dict[str, np.ndarray] | None,  # {feat: stacked [N_CORRUPTIONS * n_tr]} or None
    fc_va: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build (X_tr, X_va, y_specs) for a single corruption fit_probes call."""
    X_tr = np.concatenate([corr_acts[c][pooling][layer_idx][train_idx] for c in CORRUPTION_NAMES])
    X_va = np.concatenate([corr_acts[c][pooling][layer_idx][val_idx]   for c in CORRUPTION_NAMES])

    n_tr, n_va = len(y_id_tr), len(y_id_va)
    ones_tr = np.ones(n_tr, dtype=bool)
    ones_va = np.ones(n_va, dtype=bool)

    y_specs: dict = {
        "corruption_id": (y_id_tr, y_id_va, ones_tr, ones_va, "classification"),
        "is_corrupted": (
            (y_id_tr > 0).astype(np.int32),
            (y_id_va > 0).astype(np.int32),
            ones_tr, ones_va, "binary",
        ),
    }

    if fc_tr is not None:
        for feat in FORECAST_REGRESSION_FEATURES:
            y_tr_fc = fc_tr[feat]
            y_va_fc = fc_va[feat]
            mask_tr = np.isfinite(y_tr_fc)
            mask_va = np.isfinite(y_va_fc)
            y_specs[feat] = (y_tr_fc, y_va_fc, mask_tr, mask_va, "regression")
        for feat in FORECAST_BINARY_FEATURES:
            y_specs[feat] = (fc_tr[feat], fc_va[feat], ones_tr, ones_va, "binary")

    return X_tr, X_va, y_specs


def _run_corruption_probes(
    module,
    base_dataset: dict,
    n: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    poolings: list[str | int],
    batch_size: int,
    device: str | torch.device,
    with_forecast: bool,
    ctx_period: int | np.ndarray = 24,
) -> dict:
    """
    Extract per-corruption activations, stack, train corruption-identity probes.

    Returns {pooling_key_str: {feature: {layer_str: score}}}
    """
    series_indices = np.arange(n)
    corrupted = _apply_corruptions(base_dataset, series_indices)

    corr_acts: dict[str, dict] = {}
    corr_fqs: dict[str, tuple] = {}

    for corr_name, series_arr in corrupted.items():
        print(f"  Extracting activations for corruption '{corr_name}'...")
        acts, fq, ctx, tgt = _extract_multi_pooling(
            module, series_arr, batch_size, poolings, device,
            capture_forecast=with_forecast,
        )
        corr_acts[corr_name] = acts
        if with_forecast:
            corr_fqs[corr_name] = (fq, ctx, tgt)

    # Pre-compute and stack forecast targets outside the layer loop
    fc_tr_stacked: dict[str, np.ndarray] | None = None
    fc_va_stacked: dict[str, np.ndarray] | None = None
    if with_forecast:
        fc_tr_parts: dict[str, list] = {f: [] for f in FORECAST_REGRESSION_FEATURES + FORECAST_BINARY_FEATURES}
        fc_va_parts: dict[str, list] = {f: [] for f in FORECAST_REGRESSION_FEATURES + FORECAST_BINARY_FEATURES}
        for c_name in CORRUPTION_NAMES:
            fq_c, ctx_c, tgt_c = corr_fqs[c_name]
            p = ctx_period if isinstance(ctx_period, int) else 24
            fc_c = _compute_forecast_targets(fq_c, ctx_c, tgt_c, ctx_period=p)
            for feat in fc_tr_parts:
                fc_tr_parts[feat].append(fc_c[feat][train_idx])
                fc_va_parts[feat].append(fc_c[feat][val_idx])
        fc_tr_stacked = {f: np.concatenate(parts) for f, parts in fc_tr_parts.items()}
        fc_va_stacked = {f: np.concatenate(parts) for f, parts in fc_va_parts.items()}

    # Pre-build corruption labels (same across all layers)
    n_tr, n_va = len(train_idx), len(val_idx)
    y_id_tr = np.concatenate([np.full(n_tr, c, dtype=np.int32) for c in range(N_CORRUPTIONS)])
    y_id_va = np.concatenate([np.full(n_va, c, dtype=np.int32) for c in range(N_CORRUPTIONS)])

    sample_acts = corr_acts["clean"][poolings[0] if poolings else "mean_ctx"]
    layer_keys = sorted(sample_acts.keys())

    results: dict = {}

    for pooling in poolings:
        pkey = _pooling_key_str(pooling)
        pooling_results: dict[str, dict[str, float]] = {}

        for layer_idx in layer_keys:
            X_tr, X_va, y_specs = _stack_corruption_xy(
                corr_acts, pooling, layer_idx,
                train_idx, val_idx,
                y_id_tr, y_id_va,
                fc_tr_stacked, fc_va_stacked,
            )
            layer_scores = fit_probes(X_tr, X_va, y_specs)
            for feat, score in layer_scores.items():
                pooling_results.setdefault(feat, {})[str(layer_idx)] = score

        for feat in list(pooling_results.keys()):
            if not pooling_results[feat]:
                del pooling_results[feat]
            else:
                best = max(float(v) for v in pooling_results[feat].values())
                print(f"    [{pkey}] {feat}: best layer score = {best:.4f}")

        results[pkey] = pooling_results

    return results


# ---------------------------------------------------------------------------
# Per-patch probe runner for universal CLI
# ---------------------------------------------------------------------------

def _run_per_patch_probes_universal(
    module,
    series: np.ndarray,
    features: list[tuple[str, str]],
    feature_data: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    device: str | torch.device,
    with_forecast: bool,
    ctx_period: int | np.ndarray = 24,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Run per-patch (batched SVD ridge) probes for all regression features.

    Returns {feature: {layer_str: {patch_str: score}}}
    """
    acts_map, fq_all, ctx_all, tgt_all = _extract_multi_pooling(
        module, series, batch_size, ["per_patch"], device,
        capture_forecast=with_forecast,
    )
    X_by_layer_pp = acts_map["per_patch"]  # {layer: [n, ctx, d]}
    layer_keys = sorted(X_by_layer_pp.keys())

    if with_forecast:
        fc_targets = _compute_forecast_targets(fq_all, ctx_all, tgt_all, ctx_period)
        all_features = features + [(f, "regression") for f in FORECAST_REGRESSION_FEATURES]
        merged_data = {**feature_data, **fc_targets}
    else:
        all_features = features
        merged_data = feature_data

    y_specs: dict = {}
    for feat, ftype in all_features:
        if ftype != "regression" or feat not in merged_data:
            continue
        y_tr = merged_data[feat][train_idx].ravel().astype(np.float32)
        y_va = merged_data[feat][val_idx].ravel().astype(np.float32)
        y_specs[feat] = (y_tr, y_va, None, None, "regression")

    if not y_specs:
        return {}

    results: dict[str, dict[str, dict[str, float]]] = {f: {} for f in y_specs}

    for layer_idx in layer_keys:
        X_tr = torch.from_numpy(X_by_layer_pp[layer_idx][train_idx].transpose(1, 0, 2))
        X_va = torch.from_numpy(X_by_layer_pp[layer_idx][val_idx].transpose(1, 0, 2))
        layer_scores = fit_probes(X_tr, X_va, y_specs, backend="per_patch")
        for feat, patch_scores in layer_scores.items():
            results[feat][str(layer_idx)] = patch_scores

        best_by_feat = {f: max(results[f][str(layer_idx)].values()) for f in y_specs}
        print(f"    Layer {layer_idx}: best patch — " + ", ".join(f"{f}={v:.3f}" for f, v in best_by_feat.items()))

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _run_patch_feature_probes(
    module,
    dataset: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    device: str | torch.device,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Probe per-patch *local* labels: at each (layer, patch_idx) predict the local feature
    of that patch. Regression scored by R², binary anomaly flags by per-patch AUROC.

    Returns {feature: {layer_str: {patch_str: score}}}
    """
    from sklearn.metrics import roc_auc_score

    reg, binf = compute_patch_features(dataset, CONTEXT_PATCHES, PATCH_SIZE)
    series = dataset["series"]

    print(f"  Extracting per-patch activations ({len(train_idx)}+{len(val_idx)} examples)...")
    acts = extract_activations(
        module, series, batch_size=batch_size, device=device, pooling="per_patch",
    )  # {layer: [n, C, d]}
    layer_keys = sorted(acts.keys())

    # Labels -> [C, n_split] (patch dim first to align with [C, n, d] activations).
    def _split_labels(d):
        return ({f: v[train_idx].T for f, v in d.items()},
                {f: v[val_idx].T for f, v in d.items()})
    reg_tr, reg_va = _split_labels(reg)
    bin_tr, bin_va = _split_labels(binf)

    Y_reg_tr = torch.from_numpy(np.stack([reg_tr[f] for f in reg], axis=2).astype(np.float32))  # [C,n,k]
    Y_reg_va = torch.from_numpy(np.stack([reg_va[f] for f in reg], axis=2).astype(np.float32))

    results: dict[str, dict[str, dict[str, float]]] = {f: {} for f in list(reg) + list(binf)}

    for layer_idx in layer_keys:
        X_tr = torch.from_numpy(acts[layer_idx][train_idx].transpose(1, 0, 2))  # [C, n_tr, d]
        X_va = torch.from_numpy(acts[layer_idx][val_idx].transpose(1, 0, 2))
        B = X_tr.shape[0]

        with torch.no_grad():
            r2 = batched_ridge_per_patch_local(X_tr, X_va, Y_reg_tr, Y_reg_va)
        for ki, f in enumerate(reg):
            results[f][str(layer_idx)] = {str(p): float(r2[p, ki]) for p in range(B)}

        # Binary flags: ridge-as-scorer, per-patch AUROC with min-positive guard.
        for f in binf:
            Y_tr = torch.from_numpy(bin_tr[f][:, :, None].astype(np.float32))  # [C, n_tr, 1]
            with torch.no_grad():
                scores = _batched_ridge_predict_local(X_tr, X_va, Y_tr).squeeze(-1).numpy()  # [C, n_va]
            y_va = bin_va[f]  # [C, n_va]
            auroc: dict[str, float] = {}
            for p in range(B):
                pos = int(y_va[p].sum())
                if pos < 1 or pos == y_va.shape[1]:
                    auroc[str(p)] = float("nan")
                else:
                    try:
                        auroc[str(p)] = float(roc_auc_score(y_va[p].astype(int), scores[p]))
                    except Exception:
                        auroc[str(p)] = float("nan")
            results[f][str(layer_idx)] = auroc

        best = {f: max(results[f][str(layer_idx)].values()) for f in reg}
        print(f"    Layer {layer_idx}: best patch R² — "
              + ", ".join(f"{f}={v:.3f}" for f, v in best.items()))

    return results


def _build_metadata(
    mode: str,
    features: list[tuple[str, str]],
    with_forecast: bool,
    with_patch_features: bool = False,
) -> dict:
    clf_meta = {
        "period_idx":      {"n_classes": 8,           "baseline": 1 / 8,           "metric": "accuracy"},
        "spike_patch_idx": {"n_classes": CONTEXT_PATCHES, "baseline": 1 / CONTEXT_PATCHES, "metric": "accuracy"},
        "dataset_id":      {"n_classes": N_DATASET_CLASSES, "baseline": 1 / N_DATASET_CLASSES, "metric": "accuracy"},
        "corruption_id":   {"n_classes": N_CORRUPTIONS,  "baseline": 1 / N_CORRUPTIONS,  "metric": "accuracy",
                            "classes": CORRUPTION_NAMES},
    }
    bin_meta = {
        "spike_present": {"metric": "AUROC", "baseline": 0.5},
        "rw_present":    {"metric": "AUROC", "baseline": 0.5},
        "is_corrupted":  {"metric": "AUROC", "baseline": 0.5},
        "is_flat":       {"metric": "AUROC", "baseline": 0.5},
        "is_poor":       {"metric": "AUROC", "baseline": 0.5},
    }
    all_feats = list(features)
    if with_forecast:
        all_feats += [(f, "regression") for f in FORECAST_REGRESSION_FEATURES]
        all_feats += [(f, "binary") for f in FORECAST_BINARY_FEATURES]
    if with_patch_features:
        all_feats += [(f, "regression") for f in PATCH_REGRESSION_FEATURES]
        all_feats += [(f, "binary") for f in PATCH_BINARY_FEATURES]

    feat_meta: dict = {}
    for f, ftype in all_feats:
        if ftype == "regression":
            feat_meta[f] = {"type": "regression", "baseline": 0.0, "metric": "R²"}
        elif ftype == "binary":
            feat_meta[f] = {"type": "binary", **bin_meta.get(f, {"metric": "AUROC", "baseline": 0.5})}
        elif ftype == "classification":
            feat_meta[f] = {"type": "classification", **clf_meta.get(f, {"n_classes": 2, "baseline": 0.5, "metric": "accuracy"})}
    return {"features": feat_meta}


def main():
    parser = argparse.ArgumentParser(description="Universal linear probe runner.")
    parser.add_argument("--dataset", choices=["synth", "real", "corruption-synth", "corruption-real"],
                        default="synth")
    parser.add_argument("--pooling", nargs="+", default=["mean"],
                        help="Pooling modes: 'mean', int patch index, 'all' (per-patch)")
    parser.add_argument("--forecast", action=argparse.BooleanOptionalAction, default=True,
                        help="Include forecast-output features (default: True)")
    parser.add_argument("--patch-features", action="store_true",
                        help="Probe per-patch local labels (writes {model}_patch_features.json)")
    parser.add_argument("--model", choices=["moiraie", "moiraic", "both"], default="both")
    parser.add_argument("--moiraie-ckpt", default=None)
    parser.add_argument("--moiraic-ckpt", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-synth", type=int, default=5000)
    parser.add_argument("--n-per-dataset", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    poolings, run_per_patch = _parse_pooling_args(args.pooling)
    is_corruption = args.dataset.startswith("corruption-")
    base_mode = args.dataset.replace("corruption-", "")  # "synth" or "real"

    # ---- Load dataset ----
    if base_mode == "synth":
        from experiments.mech_interp.lib import generate_composite_dataset
        print(f"Generating composite synthetic dataset (n={args.n_synth}, seed={args.seed})...")
        dataset = generate_composite_dataset(n=args.n_synth, seed=args.seed)
        features = (
            [(f, "regression")     for f in SYNTH_REGRESSION_FEATURES]
            + [(f, "binary")       for f in SYNTH_BINARY_FEATURES]
            + [(f, "classification") for f in SYNTH_CLASSIFICATION_FEATURES]
        )
        ctx_period = 24  # use default; synth period variety averages out
    else:
        from experiments.mech_interp.lib.real_data import load_gift_subset
        print(f"Loading GIFT-Eval subset (n_per_dataset={args.n_per_dataset})...")
        dataset = load_gift_subset(n_per_dataset=args.n_per_dataset)
        features = (
            [(f, "regression")     for f in REAL_REGRESSION_FEATURES]
            + [(f, "classification") for f in REAL_CLASSIFICATION_FEATURES]
        )
        ctx_period = 24  # open issue — heterogeneous; keep hardcoded 24

    if is_corruption:
        features = (
            [("corruption_id", "classification"), ("is_corrupted", "binary")]
        )

    n = len(dataset["series"])
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")
    print(f"Device: {args.device}")

    model_names = ["moiraie", "moiraic"] if args.model == "both" else [args.model]
    ckpt_map = {"moiraie": args.moiraie_ckpt, "moiraic": args.moiraic_ckpt}

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        module = _load_module(ckpt_map[model_name], model_name, device=args.device)

        # ---- Pooled probes ----
        if poolings:
            if is_corruption:
                results = _run_corruption_probes(
                    module, dataset, n=n,
                    train_idx=train_idx, val_idx=val_idx,
                    poolings=poolings, batch_size=args.batch_size,
                    device=args.device, with_forecast=args.forecast,
                    ctx_period=ctx_period,
                )
            else:
                feature_data = {f: dataset[f] for f, _ in features if f in dataset}
                results = _run_pooled_probes(
                    module=module,
                    series=dataset["series"],
                    features=features,
                    feature_data=feature_data,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    poolings=poolings,
                    batch_size=args.batch_size,
                    device=args.device,
                    with_forecast=args.forecast,
                    ctx_period=ctx_period,
                )

            out_path = os.path.join(args.output_dir, f"{model_name}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {out_path}")

        # ---- Per-patch probes ----
        if run_per_patch:
            if is_corruption:
                print("  NOTE: --pooling all for corruption modes uses pooled extraction; per-patch skipped.")
            else:
                print(f"\n=== {model_name} (per-patch) ===")
                feature_data = {f: dataset[f] for f, _ in features if f in dataset}
                pp_results = _run_per_patch_probes_universal(
                    module=module,
                    series=dataset["series"],
                    features=features,
                    feature_data=feature_data,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    batch_size=args.batch_size,
                    device=args.device,
                    with_forecast=args.forecast,
                    ctx_period=ctx_period,
                )
                pp_path = os.path.join(args.output_dir, f"{model_name}_per_patch.json")
                with open(pp_path, "w") as f:
                    json.dump(pp_results, f, indent=2)
                print(f"  Saved: {pp_path}")

        # ---- Per-patch local feature probes ----
        if args.patch_features:
            print(f"\n=== {model_name} (patch-features) ===")
            pf_results = _run_patch_feature_probes(
                module, dataset, train_idx, val_idx,
                batch_size=args.batch_size, device=args.device,
            )
            pf_path = os.path.join(args.output_dir, f"{model_name}_patch_features.json")
            with open(pf_path, "w") as f:
                json.dump(pf_results, f, indent=2)
            print(f"  Saved: {pf_path}")

    # ---- Metadata ----
    metadata = _build_metadata(args.dataset, features, args.forecast, args.patch_features)
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
