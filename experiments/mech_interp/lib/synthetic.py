import numpy as np

PERIOD_BINS = [7, 24, 30, 12, 8, 16, 32, 64]  # dominant period choices, in time steps


def generate_dataset(
    n: int = 1000,
    seed: int = 42,
    patch_size: int = 16,
    context_patches: int = 32,
    pred_patches: int = 4,
) -> dict[str, np.ndarray]:
    """
    Generate a frozen synthetic dataset with ground-truth labels.

    Series composition per example (univariate):
        y[t] = slope * t + amp * sin(2π / period_ts * t + phase) + σ * ε[t]

    where period_ts = period_patches * patch_size (period in time steps),
    σ = exp(log_noise_var / 2). `amp` is a nuisance parameter and is NOT
    included in the returned dict; it cannot be recovered from the labels.
    Slope is in units of value-per-time-step (t ranges 0..series_length-1).

    Returns
    -------
    dict with keys:
        series       float32[n, series_length]
        slope        float32[n]   regression label (value / time step)
        period_idx   int32[n]     index into PERIOD_BINS, classification label
        phase_cos    float32[n]   circular regression label
        phase_sin    float32[n]   circular regression label
        log_noise_var float32[n]  regression label
    """
    rng = np.random.default_rng(seed)
    series_length = (context_patches + pred_patches) * patch_size
    t = np.arange(series_length, dtype=np.float32)  # shape [T]

    slopes = rng.uniform(-0.05, 0.05, size=n).astype(np.float32)
    period_idxs = rng.integers(0, len(PERIOD_BINS), size=n).astype(np.int32)
    phases = rng.uniform(0.0, 2 * np.pi, size=n).astype(np.float32)
    log_noise_vars = rng.uniform(-4.0, 0.0, size=n).astype(np.float32)
    amps = rng.uniform(0.5, 2.0, size=n).astype(np.float32)

    period_ts = np.array(PERIOD_BINS, dtype=np.float32)[period_idxs]  # [n]
    sigmas = np.exp(log_noise_vars / 2.0)  # [n], float32

    # Batch computation: [n, T]
    angles = 2 * np.pi / period_ts[:, None] * t[None, :] + phases[:, None]
    noise = rng.standard_normal((n, series_length)).astype(np.float32) * sigmas[:, None]
    series = (slopes[:, None] * t[None, :] + amps[:, None] * np.sin(angles) + noise).astype(np.float32)

    return {
        "series": series,
        "slope": slopes,
        "period_idx": period_idxs,
        "phase_cos": np.cos(phases).astype(np.float32),
        "phase_sin": np.sin(phases).astype(np.float32),
        "log_noise_var": log_noise_vars,
    }


def save_dataset(data: dict[str, np.ndarray], path: str) -> None:
    """Save dataset to a compressed .npz file (numpy appends .npz if missing)."""
    np.savez_compressed(path, **data)


def load_dataset(path: str) -> dict[str, np.ndarray]:
    """Load dataset from a .npz file returned by save_dataset."""
    f = np.load(path)
    return {k: f[k] for k in f.files}
