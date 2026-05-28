from __future__ import annotations

import numpy as np

from experiments.mech_interp.lib.metrics import mase as _mase
from experiments.mech_interp.lib.metrics import scaled_weighted_quantile_loss as _swql

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_MEDIAN_IDX = 4  # q=0.5 is at index 4 in QUANTILE_LEVELS
HORIZON = 64     # 4 pred patches * 16


def fc_std(forecast_quantiles: np.ndarray) -> float:
    """log(forecast_median.std() + 1e-6). Low value = flat / regressed-to-mean forecast."""
    median = forecast_quantiles[_MEDIAN_IDX]
    return float(np.log(median.std() + 1e-6))


def fc_range(forecast_quantiles: np.ndarray) -> float:
    """log(forecast_median.max() - forecast_median.min() + 1e-6)."""
    median = forecast_quantiles[_MEDIAN_IDX]
    return float(np.log(median.max() - median.min() + 1e-6))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-10 or b_std < 1e-10:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def fc_ctx_corr(forecast_quantiles: np.ndarray, context: np.ndarray) -> float:
    """Pearson corr between forecast median and last HORIZON context steps."""
    median = forecast_quantiles[_MEDIAN_IDX]
    last_ctx = context[-HORIZON:]
    return _pearson(median, last_ctx)


def fc_ctx_corr_seasonal(
    forecast_quantiles: np.ndarray, context: np.ndarray, ctx_period: int
) -> float:
    """Pearson corr between forecast median and ctx[-(HORIZON+P):-P].

    Returns nan if context is too short to extract the seasonal reference.
    """
    median = forecast_quantiles[_MEDIAN_IDX]
    P = int(ctx_period)
    if P <= 0 or len(context) < HORIZON + P:
        return np.nan
    seasonal_ref = context[-(HORIZON + P):-P]
    if len(seasonal_ref) != HORIZON:
        return np.nan
    return _pearson(median, seasonal_ref)


# TODO: Add a scaled version of iqr
def fc_iqr_mean(forecast_quantiles: np.ndarray) -> float:
    """Mean of (q0.9 - q0.1) over the forecast horizon."""
    iqr = forecast_quantiles[-1] - forecast_quantiles[0]  # q0.9 - q0.1
    return float(iqr.mean())


def fc_iqr_slope(forecast_quantiles: np.ndarray) -> float:
    """Linear slope of IQR over the forecast horizon. Positive = growing uncertainty."""
    iqr = forecast_quantiles[-1] - forecast_quantiles[0]
    t = np.arange(len(iqr), dtype=np.float64)
    slope = float(np.polyfit(t, iqr.astype(np.float64), 1)[0])
    return slope


def compute_mase(
    forecast_quantiles: np.ndarray, target: np.ndarray, context: np.ndarray
) -> float:
    """MASE of the forecast median against target, scaled by context naive-walk."""
    median = forecast_quantiles[_MEDIAN_IDX]
    return _mase(median, target, context)


def compute_swql(
    forecast_quantiles: np.ndarray, target: np.ndarray, context: np.ndarray
) -> float:
    """Scaled weighted quantile loss across all 9 quantile levels."""
    return _swql(forecast_quantiles, QUANTILE_LEVELS, target, context)


def quantile_calibration_err(
    forecast_quantiles: np.ndarray, target: np.ndarray
) -> float:
    """Mean over quantile levels of |empirical_coverage_q - q|.

    empirical_coverage_q = fraction of horizon steps where target <= forecast_q.
    """
    target = np.asarray(target, dtype=np.float64)
    errs = []
    for q_idx, q in enumerate(QUANTILE_LEVELS):
        coverage = float(np.mean(target <= forecast_quantiles[q_idx]))
        errs.append(abs(coverage - q))
    return float(np.mean(errs))


def compute_all(
    forecast_quantiles: np.ndarray,
    target: np.ndarray,
    context: np.ndarray,
    ctx_period: int,
) -> dict[str, float]:
    """Compute all 9 per-series forecast-output properties.

    Args:
        forecast_quantiles: [9, HORIZON] — rows correspond to QUANTILE_LEVELS.
        target:             [HORIZON]    — ground-truth horizon values.
        context:            [512]        — context time series (standardized).
        ctx_period:         dominant period of the context (integer timesteps).

    Returns dict with keys matching C.1 target names.
    """
    fq = np.asarray(forecast_quantiles, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    ctx = np.asarray(context, dtype=np.float64)
    return {
        "fc_std": fc_std(fq),
        "fc_range": fc_range(fq),
        "fc_ctx_corr": fc_ctx_corr(fq, ctx),
        "fc_ctx_corr_seasonal": fc_ctx_corr_seasonal(fq, ctx, ctx_period),
        "fc_iqr_mean": fc_iqr_mean(fq),
        "fc_iqr_slope": fc_iqr_slope(fq),
        "mase": compute_mase(fq, tgt, ctx),
        "swql": compute_swql(fq, tgt, ctx),
        "quantile_calibration_err": quantile_calibration_err(fq, tgt),
    }


def derive_binary_labels(
    fc_stds: np.ndarray,
    mases: np.ndarray,
) -> dict[str, np.ndarray]:
    """Median-split binary labels over the dataset.

    is_flat: fc_std < median(fc_std)  — forecast is unusually flat.
    is_poor: mase  > median(mase)     — forecast error is above average.
    """
    fc_stds = np.asarray(fc_stds, dtype=np.float64)
    mases = np.asarray(mases, dtype=np.float64)
    return {
        "is_flat": (fc_stds < np.median(fc_stds)).astype(np.int32),
        "is_poor": (mases > np.median(mases)).astype(np.int32),
    }
