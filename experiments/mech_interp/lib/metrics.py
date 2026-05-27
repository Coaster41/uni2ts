import numpy as np


def mase(forecast: np.ndarray, target: np.ndarray, context: np.ndarray) -> float:
    """Mean Absolute Scaled Error.

    forecast, target, context: 1D arrays.
    Denominator: mean absolute first difference of context (naive walk baseline).
    """
    mae = np.mean(np.abs(forecast - target))
    scale = np.mean(np.abs(np.diff(context))) + 1e-8
    return float(mae / scale)


def weighted_quantile_loss(
    forecast_quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    target: np.ndarray,
) -> float:
    """Mean pinball loss across quantile levels (unscaled).

    forecast_quantiles: [Q, T] — one row per quantile level.
    quantile_levels:    [Q]    — levels in (0, 1).
    target:             [T].
    """
    q = np.asarray(quantile_levels, dtype=np.float64)[:, None]  # [Q, 1]
    f = np.asarray(forecast_quantiles, dtype=np.float64)        # [Q, T]
    y = np.asarray(target, dtype=np.float64)[None, :]           # [1, T]
    errors = y - f
    pinball = np.where(errors >= 0, q * errors, (q - 1) * errors)  # [Q, T]
    return float(pinball.mean())


def scaled_weighted_quantile_loss(
    forecast_quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    target: np.ndarray,
    context: np.ndarray,
) -> float:
    """Mean pinball loss scaled by mean absolute first difference of context.

    Scale-free variant of weighted_quantile_loss; comparable across series.
    """
    scale = np.mean(np.abs(np.diff(context))) + 1e-8
    return weighted_quantile_loss(forecast_quantiles, quantile_levels, target) / float(scale)
