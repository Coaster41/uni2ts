"""Per-patch *local* labels for patch-level probing.

Unlike the series-level labels in ``synthetic.py``/``pseudo_labels.py`` (one scalar per
series), every feature here is computed from each individual patch's content and returned
as ``[n, context_patches]``. The probe at patch position ``k`` is then trained to predict
the local feature of patch ``k`` (see ``_run_patch_feature_probes`` in train_probes.py).

Groups:
    A  local stats        — within-patch mean/std/slope/range/acf1/net-change   (any data)
    B  relative           — patch-vs-series level/volatility/rank               (any data)
    C  anomaly flags (gt) — spike/level/var-shift location from synth metadata  (synth only)
    D  anomaly flags (dd) — data-driven level-outlier / point-spike             (any data)
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _patchify(series: np.ndarray, context_patches: int, patch_size: int) -> np.ndarray:
    """[n, T] -> [n, context_patches, patch_size] over the context window."""
    ctx = series[:, : context_patches * patch_size].astype(np.float64)
    return ctx.reshape(len(series), context_patches, patch_size)


def _group_local_stats(patches: np.ndarray) -> dict[str, np.ndarray]:
    """Group A — within-patch statistics. patches: [n, C, P]."""
    P = patches.shape[2]
    t = np.arange(P, dtype=np.float64)
    t_c = t - t.mean()
    # least-squares slope of each patch against time
    slope = (patches * t_c).sum(axis=2) / (t_c**2).sum()

    pm = patches.mean(axis=2)
    xm = patches - pm[:, :, None]
    denom = (xm**2).sum(axis=2)
    acf1 = np.where(denom > _EPS, (xm[:, :, :-1] * xm[:, :, 1:]).sum(axis=2) / denom, 0.0)

    return {
        "patch_mean": pm,
        "patch_std": patches.std(axis=2),
        "patch_slope": slope,
        "patch_range": patches.max(axis=2) - patches.min(axis=2),
        "patch_acf1": acf1,
        "patch_net_change": patches[:, :, -1] - patches[:, :, 0],
    }


def _group_relative(patches: np.ndarray) -> dict[str, np.ndarray]:
    """Group B — patch relative to the whole context window. patches: [n, C, P]."""
    n, C, _ = patches.shape
    pm = patches.mean(axis=2)
    pstd = patches.std(axis=2)
    g_mean = patches.reshape(n, -1).mean(axis=1, keepdims=True)
    g_std = patches.reshape(n, -1).std(axis=1, keepdims=True)
    # rank of each patch mean among the C patches, normalized to [0, 1]
    rank = pm.argsort(axis=1).argsort(axis=1) / max(C - 1, 1)
    return {
        "patch_mean_dev": (pm - g_mean) / (g_std + _EPS),
        "patch_logstd_ratio": np.log((pstd + _EPS) / (g_std + _EPS)),
        "patch_mean_rank": rank,
    }


def _group_anomaly_gt(
    dataset: dict, n: int, C: int, T: int, patch_size: int
) -> dict[str, np.ndarray]:
    """Group C — ground-truth anomaly locations from synth metadata. All [n, C]."""
    out: dict[str, np.ndarray] = {}

    spike = np.zeros((n, C), dtype=np.float64)
    sp_idx = dataset["spike_patch_idx"].astype(int)
    has_sp = sp_idx >= 0
    spike[np.where(has_sp)[0], sp_idx[has_sp]] = 1.0
    out["patch_has_spike"] = spike

    def _flag_from_time_norm(time_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(time_norm)
        patch = np.full(n, -1, dtype=int)
        patch[valid] = np.clip((time_norm[valid] * T / patch_size).astype(int), 0, C - 1)
        flag = np.zeros((n, C), dtype=np.float64)
        flag[np.where(valid)[0], patch[valid]] = 1.0
        return flag, patch

    lvl_flag, _ = _flag_from_time_norm(dataset["level_time_norm"])
    out["patch_has_levelshift"] = lvl_flag

    var_flag, var_patch = _flag_from_time_norm(dataset["var_shift_time_norm"])
    out["patch_has_varshift"] = var_flag

    post = np.zeros((n, C), dtype=np.float64)
    has_var = var_patch >= 0
    patch_grid = np.arange(C)[None, :]
    post[has_var] = (patch_grid >= var_patch[has_var, None]).astype(np.float64)
    out["patch_is_post_varshift"] = post

    return out


def _group_anomaly_dd(
    patches: np.ndarray, z_level: float, z_spike: float
) -> dict[str, np.ndarray]:
    """Group D — data-driven anomaly flags (no metadata). patches: [n, C, P]."""
    n, C, _ = patches.shape
    flat = patches.reshape(n, -1)
    g_mean = flat.mean(axis=1, keepdims=True)
    g_std = flat.std(axis=1, keepdims=True) + _EPS
    pm = patches.mean(axis=2)
    z = np.abs(patches - g_mean[:, :, None]) / g_std[:, :, None]
    return {
        "patch_is_level_outlier": (np.abs(pm - g_mean) / g_std > z_level).astype(np.float64),
        "patch_has_pointspike": (z.max(axis=2) > z_spike).astype(np.float64),
    }


def compute_patch_features(
    dataset: dict,
    context_patches: int = 32,
    patch_size: int = 16,
    z_level: float = 2.5,
    z_spike: float = 4.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute per-patch local labels.

    Returns ``(reg_labels, bin_labels)`` where every value is ``float64[n, context_patches]``.
    Group C (ground-truth anomaly flags) is included only when the synth metadata keys
    (``spike_patch_idx``, ``level_time_norm``, ``var_shift_time_norm``) are present.
    """
    series = dataset["series"]
    n, T = series.shape
    patches = _patchify(series, context_patches, patch_size)

    reg = {**_group_local_stats(patches), **_group_relative(patches)}

    bin_labels = _group_anomaly_dd(patches, z_level, z_spike)
    if all(k in dataset for k in ("spike_patch_idx", "level_time_norm", "var_shift_time_norm")):
        bin_labels.update(_group_anomaly_gt(dataset, n, context_patches, T, patch_size))

    return reg, bin_labels
