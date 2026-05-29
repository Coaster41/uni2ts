from __future__ import annotations

import os

import numpy as np
from joblib import Memory

_cache = Memory(
    location=os.path.join(os.path.dirname(__file__), "../../.cache/pseudo_labels"),
    verbose=0,
)

_PERIOD_CANDIDATES = [7, 12, 24, 48, 168, 365]
_MIN_PERIOD = 4


def _estimate_period_int(ctx: np.ndarray) -> int:
    """Round FFT-dominant period to nearest calendar value, clamped for STL."""
    T = len(ctx)
    fft_vals = np.abs(np.fft.rfft(ctx - ctx.mean()))[1 : T // 2 + 1]
    dominant_bin = int(np.argmax(fft_vals)) + 1  # +1 because index 0 = freq bin 1
    raw_period = T / dominant_bin

    # Round to nearest candidate
    period = min(_PERIOD_CANDIDATES, key=lambda p: abs(p - raw_period))

    # STL requires period >= 4 and at least 2 full cycles
    period = max(period, _MIN_PERIOD)
    period = min(period, T // 3)
    if period < _MIN_PERIOD:
        period = 24
    return int(period)


@_cache.cache
def stl_trend_strength(ctx: np.ndarray, period: int | None = None) -> float:
    """F_T = max(0, 1 - Var(resid) / Var(trend + resid))."""
    from statsmodels.tsa.seasonal import STL

    if period is None:
        period = _estimate_period_int(ctx)
    try:
        res = STL(ctx, period=period).fit()
        var_resid = np.var(res.resid)
        var_trend_resid = np.var(res.trend + res.resid)
        if var_trend_resid < 1e-10:
            return np.nan
        return float(max(0.0, 1.0 - var_resid / var_trend_resid))
    except Exception:
        return np.nan


@_cache.cache
def stl_seasonal_strength(ctx: np.ndarray, period: int | None = None) -> float:
    """F_S = max(0, 1 - Var(resid) / Var(seasonal + resid))."""
    from statsmodels.tsa.seasonal import STL

    if period is None:
        period = _estimate_period_int(ctx)
    try:
        res = STL(ctx, period=period).fit()
        var_resid = np.var(res.resid)
        var_seas_resid = np.var(res.seasonal + res.resid)
        if var_seas_resid < 1e-10:
            return np.nan
        return float(max(0.0, 1.0 - var_resid / var_seas_resid))
    except Exception:
        return np.nan


def fft_dominant_period(ctx: np.ndarray) -> float:
    """log(T / argmax(|FFT(ctx - mean)|_{1..T/2})). Returns log-period."""
    T = len(ctx)
    fft_vals = np.abs(np.fft.rfft(ctx - ctx.mean()))[1 : T // 2 + 1]
    dominant_bin = int(np.argmax(fft_vals)) + 1
    period = T / dominant_bin
    return float(np.log(period))


def fft_top1_power_frac(ctx: np.ndarray) -> float:
    """Top-1 frequency power / total spectral power."""
    T = len(ctx)
    power = np.abs(np.fft.rfft(ctx - ctx.mean()))[1 : T // 2 + 1] ** 2
    total = power.sum()
    if total < 1e-10:
        return 0.0
    return float(power.max() / total)


def spectral_flatness(ctx: np.ndarray) -> float:
    """Wiener entropy: geometric_mean(power) / arithmetic_mean(power)."""
    T = len(ctx)
    power = np.abs(np.fft.rfft(ctx - ctx.mean()))[1 : T // 2 + 1] ** 2
    power = power + 1e-10
    arith = power.mean()
    geom = np.exp(np.log(power).mean())
    return float(geom / arith)


def adf_pvalue(ctx: np.ndarray) -> float:
    """statsmodels adfuller p-value."""
    from statsmodels.tsa.stattools import adfuller

    try:
        return float(adfuller(ctx)[1])
    except Exception:
        return np.nan


def hurst_exponent(ctx: np.ndarray) -> float:
    """R/S estimator, clipped to [0.2, 1.0]."""
    x = np.asarray(ctx, dtype=np.float64)
    n = len(x)
    if n < 20:
        return 0.5

    lags = []
    rs_vals = []
    min_chunk = 8
    for chunk_size in range(min_chunk, n // 2 + 1, max(1, (n // 2 - min_chunk) // 8)):
        n_chunks = n // chunk_size
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            seg = x[i * chunk_size : (i + 1) * chunk_size]
            seg = seg - seg.mean()
            cumsum = np.cumsum(seg)
            R = cumsum.max() - cumsum.min()
            S = seg.std()
            if S > 1e-10:
                rs_list.append(R / S)
        if rs_list:
            lags.append(np.log(chunk_size))
            rs_vals.append(np.log(np.mean(rs_list)))

    if len(lags) < 2:
        return 0.5

    slope, _ = np.polyfit(lags, rs_vals, 1)
    return float(np.clip(slope, 0.2, 1.0))


@_cache.cache
def sample_entropy(ctx: np.ndarray) -> float:
    """antropy.sample_entropy(ctx, order=2, metric='chebyshev')."""
    import antropy

    try:
        val = float(antropy.sample_entropy(ctx, order=2, metric="chebyshev"))
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def n_changepoints(ctx: np.ndarray) -> float:
    """Number of changepoints via squared first-differences + find_peaks."""
    from scipy.signal import find_peaks

    diff = np.diff(ctx.astype(np.float64))
    sq_diff = diff ** 2
    threshold = 0.5 * sq_diff.std()
    peaks, _ = find_peaks(sq_diff, prominence=threshold)
    return float(len(peaks))


def context_std(ctx: np.ndarray) -> float:
    """ctx.std()."""
    return float(ctx.std())


def context_acf_lag1(ctx: np.ndarray) -> float:
    """First-lag autocorrelation."""
    x = ctx.astype(np.float64)
    xm = x - x.mean()
    denom = (xm ** 2).sum()
    if denom < 1e-10:
        return np.nan
    return float((xm[:-1] * xm[1:]).sum() / denom)


PSEUDO_LABEL_FUNCTIONS: dict[str, callable] = {
    "stl_trend_strength": stl_trend_strength,
    "stl_seasonal_strength": stl_seasonal_strength,
    "fft_dominant_period": fft_dominant_period,
    "fft_top1_power_frac": fft_top1_power_frac,
    "spectral_flatness": spectral_flatness,
    "adf_pvalue": adf_pvalue,
    "hurst_exponent": hurst_exponent,
    "sample_entropy": sample_entropy,
    "n_changepoints": n_changepoints,
    # context_std omitted: PackedStdScaler normalizes the context to std≈1 before
    # the transformer forward pass, so this label is always ~1 in the model's view.
    "context_acf_lag1": context_acf_lag1,
}
