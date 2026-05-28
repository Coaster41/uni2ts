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


def corrupt_mean_center(series: np.ndarray, context_len: int = 512) -> np.ndarray:
    """Subtract context-window mean from the entire series."""
    mean = series[:context_len].mean()
    return (series - mean).astype(series.dtype)


def corrupt_reverse(series: np.ndarray, context_len: int = 512) -> np.ndarray:
    """Reverse the context portion; keep the horizon unchanged."""
    out = series.copy()
    out[:context_len] = series[:context_len][::-1]
    return out


def corrupt_shuffle_patches(
    series: np.ndarray, seed: int, patch_size: int = 16, context_len: int = 512
) -> np.ndarray:
    """Randomly permute the context patches; keep the horizon unchanged."""
    out = series.copy()
    n_patches = context_len // patch_size
    patches = out[:context_len].reshape(n_patches, patch_size).copy()
    np.random.default_rng(seed).shuffle(patches)
    out[:context_len] = patches.reshape(context_len)
    return out


def corrupt_zero_segment(
    series: np.ndarray,
    seed: int,
    patch_size: int = 16,
    context_len: int = 512,
    n_zero_patches: int = 4,
) -> np.ndarray:
    """Zero out a random contiguous n_zero_patches-patch segment in the context."""
    out = series.copy()
    n_patches = context_len // patch_size
    max_start = n_patches - n_zero_patches
    start_patch = int(np.random.default_rng(seed).integers(0, max_start + 1))
    out[start_patch * patch_size : (start_patch + n_zero_patches) * patch_size] = 0
    return out.astype(series.dtype)
