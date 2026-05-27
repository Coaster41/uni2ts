import numpy as np

from .synthetic import PERIOD_BINS


def corrupt_trend(series: np.ndarray, slope: float) -> np.ndarray:
    """Subtract fitted linear trend from series, leaving seasonal + noise."""
    t = np.arange(len(series), dtype=np.float64)
    coeffs = np.polyfit(t, series, 1)
    trend = np.polyval(coeffs, t)
    return (series - trend).astype(series.dtype)


def corrupt_seasonal(series: np.ndarray, period_idx: int, phase: float) -> np.ndarray:
    """Subtract estimated sinusoidal component using ground-truth period and phase.

    Amplitude is estimated via least squares — do not pass ground-truth amplitude.
    phase is in radians; callers should reconstruct from (phase_cos, phase_sin) via
    np.arctan2(phase_sin, phase_cos) if using the dataset labels.
    """
    period = PERIOD_BINS[period_idx]
    t = np.arange(len(series), dtype=np.float64)
    omega = 2 * np.pi / period
    basis = np.stack([np.cos(omega * t + phase), np.sin(omega * t + phase)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(basis, series.astype(np.float64), rcond=None)
    seasonal = basis @ coeffs
    return (series - seasonal).astype(series.dtype)


def corrupt_noise(series: np.ndarray, seed: int) -> np.ndarray:
    """Replace series with white noise matching series std. seed = example_idx * 7919."""
    return np.random.default_rng(seed).normal(0, series.std(), series.shape).astype(series.dtype)


def corrupt_add_noise(series: np.ndarray, seed: int, std: float | None = None) -> np.ndarray:
    """Add white noise to series. std defaults to series.std(). seed = example_idx * 7919."""
    sigma = series.std() if std is None else std
    noise = np.random.default_rng(seed).normal(0, sigma, series.shape).astype(series.dtype)
    return series + noise
