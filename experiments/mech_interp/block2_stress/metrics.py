"""
PR-2: Metric functions for block2_stress response curves.

Pure numpy — no model calls, no IO. Each function takes pre-computed
median forecasts and ground-truth metadata arrays and returns per-series scores.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Family A — Structure-commitment
# ---------------------------------------------------------------------------

def retention_periodic(
    median_fc: ndarray,   # [n, H]
    meta: dict,
    cfg: dict,
) -> ndarray:
    """
    Regress median forecast onto {sin, cos} at ground-truth frequency 2π/P.
    R = sqrt(a² + b²)  (recovered amplitude; true ≈ 1 for high SNR).
    Returns R [n], clipped to [0, 2].
    """
    H = median_fc.shape[1]
    h = np.arange(H, dtype=np.float64)
    periods = meta["period_ts"].astype(np.float64)  # [n]

    results = np.empty(len(median_fc), dtype=np.float32)
    for i in range(len(median_fc)):
        P = periods[i]
        sin_col = np.sin(2 * np.pi * h / P)
        cos_col = np.cos(2 * np.pi * h / P)
        A = np.column_stack([sin_col, cos_col, np.ones(H)])  # [H, 3]
        coeffs, _, _, _ = np.linalg.lstsq(A, median_fc[i].astype(np.float64), rcond=None)
        a, b = coeffs[0], coeffs[1]
        results[i] = float(np.sqrt(a**2 + b**2))

    return np.clip(results, 0.0, 2.0)


def retention_trend(
    median_fc: ndarray,   # [n, H]
    meta: dict,
    cfg: dict,
) -> ndarray:
    """
    Fit line to median forecast → recovered slope ĝ.
    R = ĝ / g_true  (clipped ≥ 0).
    g_true = slope_sign * 1/T_half where T_half = (T-1)/2 and T = total series length.
    Returns R [n].
    """
    H = median_fc.shape[1]
    h = np.arange(H, dtype=np.float64)
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    t_half = (T - 1) / 2.0
    g_true_mag = 1.0 / t_half  # magnitude of the unit-amplitude trend slope

    slope_signs = meta["slope_sign"].astype(np.float64)  # [n]
    results = np.empty(len(median_fc), dtype=np.float32)

    for i in range(len(median_fc)):
        coeffs = np.polyfit(h, median_fc[i].astype(np.float64), deg=1)
        g_hat = coeffs[0]  # recovered slope
        g_true = slope_signs[i] * g_true_mag
        R = g_hat / (g_true + 1e-12)  # avoid divide by zero
        results[i] = float(np.clip(R, 0.0, None))

    return results


def spectral_flatness(median_fc: ndarray) -> ndarray:
    """
    Geometric mean / arithmetic mean of power spectrum of each horizon forecast.
    High (~1) = white/flat (correct for white noise). Drop toward 0 = model confabulated structure.
    Returns flatness [n].
    """
    H = median_fc.shape[1]
    # rfft gives H//2 + 1 frequency bins
    fft = np.fft.rfft(median_fc.astype(np.float64), axis=1)
    power = np.abs(fft) ** 2  # [n, H//2+1]
    power = np.maximum(power, 1e-12)  # avoid log(0)

    log_geo_mean = np.mean(np.log(power), axis=1)
    arith_mean = np.mean(power, axis=1)

    flatness = np.exp(log_geo_mean) / (arith_mean + 1e-12)
    return flatness.astype(np.float32)


# ---------------------------------------------------------------------------
# Family B — Parroting
# ---------------------------------------------------------------------------

def persistence_alignment(
    median_fc: ndarray,   # [n, H]
    context: ndarray,     # [n, ctx_len]
    cfg: dict,
) -> ndarray:
    """
    PA = cosine(median_fc, flat_hold)
    where flat_hold = last context value repeated H times.
    Measures how aligned the forecast is with a flat (persistence) prediction.
    Returns PA [n] ∈ [-1, 1].

    Note: flat_hold is a constant vector per series; cosine is computed without
    centering so that a perfectly flat forecast achieves PA = 1.
    """
    H = median_fc.shape[1]
    last_val = context[:, -1:].astype(np.float64)  # [n, 1]
    flat_hold = np.repeat(last_val, H, axis=1)      # [n, H]

    fc = median_fc.astype(np.float64)

    dot = (fc * flat_hold).sum(axis=1)
    norm_fc = np.linalg.norm(fc, axis=1) + 1e-12
    norm_fh = np.linalg.norm(flat_hold, axis=1) + 1e-12

    return (dot / (norm_fc * norm_fh)).astype(np.float32)


def optimal_pa(meta: dict, cfg: dict) -> ndarray:
    """
    PA of the analytic optimal AR(1) continuation vs flat_hold.
    Analytic continuation: x_h = phi^h * x_origin for h=1..H.
    Reference curve; should be ~1 for phi near 1 and decrease / go negative for phi < 0.
    Returns PA_opt [n].
    """
    H = cfg["horizon_patches"] * cfg["patch_len"]
    phis = meta["phi"].astype(np.float64)           # [n]
    x_origins = meta["x_origin"].astype(np.float64) # [n]

    h = np.arange(1, H + 1, dtype=np.float64)
    optimal_cont = x_origins[:, None] * (phis[:, None] ** h)  # [n, H]
    flat_hold    = np.repeat(x_origins[:, None], H, axis=1)    # [n, H]

    dot = (optimal_cont * flat_hold).sum(axis=1)
    norm_oc = np.linalg.norm(optimal_cont, axis=1) + 1e-12
    norm_fh = np.linalg.norm(flat_hold, axis=1) + 1e-12

    return (dot / (norm_oc * norm_fh)).astype(np.float32)


def spike_follow_ratio(
    median_fc: ndarray,   # [n, H]
    meta: dict,
) -> ndarray:
    """
    (m_1 - base_level) / injected_delta  where m_1 = median_fc[:, 0].
    Response: 1 = full follow, 0 = spike ignored.
    Returns ratio [n].
    """
    m1 = median_fc[:, 0].astype(np.float64)
    base = meta["base_level"].astype(np.float64)
    delta = meta["injected_delta"].astype(np.float64)

    ratio = (m1 - base) / (delta + 1e-12)
    return ratio.astype(np.float32)


def overshoot(
    median_fc: ndarray,   # [n, H]
    meta: dict,
    cfg: dict,
) -> ndarray:
    """
    mean_{h >= apex_h}(m_h - true_h) where true_h is the triangle continuation.
    Positive = model kept ramping past the apex.
    Returns overshoot [n] (NaN for series where apex_h >= H).
    """
    H = median_fc.shape[1]
    apex_hs = meta["apex_h"]  # [n], 0-indexed horizon step of apex
    period_ts = meta["period_ts"].astype(np.float64)   # [n]

    ctx_len = cfg["context_patches"] * cfg["patch_len"]
    results = np.full(len(median_fc), np.nan, dtype=np.float32)

    for i in range(len(median_fc)):
        apex_h = int(apex_hs[i])
        if apex_h >= H:
            continue

        P = period_ts[i]
        # Reconstruct triangle continuation for horizon steps
        # Series was built with a phase offset so apex_h is the triangle peak in horizon.
        # true_h decreases from apex at rate 4/P per step (triangle slope).
        slope = 4.0 / P  # triangle descends at 4/P per step after apex
        h_range = np.arange(H, dtype=np.float64)
        true_horizon = 1.0 - slope * np.abs(h_range - apex_h)

        post_apex = np.arange(apex_h, H)
        if len(post_apex) == 0:
            continue

        mean_overshoot = (median_fc[i, post_apex].astype(np.float64) - true_horizon[post_apex]).mean()
        results[i] = float(mean_overshoot)

    return results


# ---------------------------------------------------------------------------
# Family C — Style
# ---------------------------------------------------------------------------

def forecast_spike_rate(
    median_fc: ndarray,   # [n, H]
    meta: dict,
    threshold: float = 0.3,
) -> ndarray:
    """
    Fraction of period-length windows in the forecast that contain a spike.

    Divides the H-step forecast into floor(H / P) non-overlapping windows of
    length P (the ground-truth period). A window is "active" (contains a spike)
    if its maximum exceeds `threshold`.

    Bumps in family_c_intermittent peak at 1.0; threshold=0.3 cleanly separates
    active windows from noise-only windows (base_noise_sigma=0.05).  For a
    model that captures the true Bernoulli(p) process the expected return value
    is p.

    Returns rate [n] in [0, 1].
    """
    H = median_fc.shape[1]
    periods = meta["period_ts"].astype(np.float64)
    results = np.empty(len(median_fc), dtype=np.float32)

    for i in range(len(median_fc)):
        P = max(1, int(round(periods[i])))
        n_windows = max(1, H // P)
        spike_count = sum(
            1 for w in range(n_windows)
            if median_fc[i, w * P : (w + 1) * P].max() > threshold
        )
        results[i] = spike_count / n_windows

    return results


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_level(scores: ndarray) -> tuple[float, float, float]:
    """Returns (median, q25, q75) across n series for error-bar plotting."""
    med = float(np.nanmedian(scores))
    q25 = float(np.nanpercentile(scores, 25))
    q75 = float(np.nanpercentile(scores, 75))
    return med, q25, q75
