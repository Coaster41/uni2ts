from itertools import combinations as _combinations

import numpy as np

PERIOD_BINS = [7, 24, 30, 12, 8, 16, 32, 64]  # dominant period choices, in time steps
MODES_DIST = (0.2, 0.35, 0.45)
CONCEPT_WEIGHTS = [1, 0.8, 2, 5, 0.7, 1, 1.5] 


# ── Per-concept component generators ─────────────────────────────────────────
# Each function takes (rng, n, T, **kwargs) and returns the component array
# [n, T] plus raw parameter arrays needed for label assembly.


def component_trend(
    rng: np.random.Generator,
    n: int,
    T: int,
    slope_min: float = -3.0,
    slope_max: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (c_trend [n, T], slopes [n]).
    c_trend[i, t] = slope_i * t / T
    """
    slopes = rng.uniform(slope_min, slope_max, size=n).astype(np.float32)
    t = np.arange(T, dtype=np.float32)
    return slopes[:, None] * t[None, :] / T, slopes


def component_level_shift(
    rng: np.random.Generator,
    n: int,
    T: int,
    mag_min: float = -2.0,
    mag_max: float = 2.0,
    t_frac_min: float = 0.3,
    t_frac_max: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (c_level [n, T], magnitudes [n], t_shifts [n]).
    c_level[i, t] = magnitude_i  if t >= t_shift_i  else 0
    """
    magnitudes = rng.uniform(mag_min, mag_max, size=n).astype(np.float32)
    t_shifts = rng.integers(int(t_frac_min * T), int(t_frac_max * T), size=n)
    t = np.arange(T, dtype=np.float32)
    return magnitudes[:, None] * (t[None, :] >= t_shifts[:, None]), magnitudes, t_shifts


def component_ar1(
    rng: np.random.Generator,
    n: int,
    T: int,
    phi_min: float = -0.95,
    phi_max: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (c_ar [n, T], phi_all [n]).
    Iterative AR(1): c_ar[t] = phi * c_ar[t-1] + eta[t],  c_ar[0] = 0.
    """
    phi_all = rng.uniform(phi_min, phi_max, size=n).astype(np.float32)
    eta = rng.standard_normal((n, T)).astype(np.float32)
    c_ar = np.zeros((n, T), dtype=np.float32)
    for t_step in range(1, T):
        c_ar[:, t_step] = phi_all * c_ar[:, t_step - 1] + eta[:, t_step]
    return c_ar, phi_all


def component_seasonal(
    rng: np.random.Generator,
    n: int,
    T: int,
    amp_min: float = 0.5,
    amp_max: float = 2.0,
    period_bins: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (c_seasonal [n, T], period_idxs [n], amps [n], phases [n]).
    c_seasonal[i, t] = amp_i * sin(2π / period_ts_i * t + phase_i)
    """
    if period_bins is None:
        period_bins = PERIOD_BINS
    period_idxs = rng.integers(0, len(period_bins), size=n).astype(np.int32)
    amps = rng.uniform(amp_min, amp_max, size=n).astype(np.float32)
    phases = rng.uniform(0.0, 2 * np.pi, size=n).astype(np.float32)
    period_ts = np.array(period_bins, dtype=np.float32)[period_idxs]
    t = np.arange(T, dtype=np.float32)
    angles = 2 * np.pi / period_ts[:, None] * t[None, :] + phases[:, None]
    return amps[:, None] * np.sin(angles), period_idxs, amps, phases


def component_var_shift(
    rng: np.random.Generator,
    n: int,
    T: int,
    sigma_before_min: float = 0.2,
    sigma_before_max: float = 0.5,
    sigma_after_min: float = 0.5,
    sigma_after_max: float = 2.0,
    t_frac_min: float = 0.3,
    t_frac_max: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (c_var [n, T], log_sigma_before [n], log_sigma_after [n], t_shifts [n]).
    Noise with different std before/after the shift point.
    """
    log_sigma_before = rng.uniform(np.log(sigma_before_min), np.log(sigma_before_max), size=n).astype(np.float32)
    log_sigma_after = rng.uniform(np.log(sigma_after_min), np.log(sigma_after_max), size=n).astype(np.float32)
    sigma_before = np.exp(log_sigma_before)
    sigma_after = np.exp(log_sigma_after)
    t_shifts = rng.integers(int(t_frac_min * T), int(t_frac_max * T), size=n)
    eps = rng.standard_normal((n, T)).astype(np.float32)
    t = np.arange(T, dtype=np.float32)
    before_mask = t[None, :] < t_shifts[:, None]
    c_var = np.where(before_mask, sigma_before[:, None] * eps, sigma_after[:, None] * eps)
    return c_var.astype(np.float32), log_sigma_before, log_sigma_after, t_shifts


def component_spike(
    rng: np.random.Generator,
    n: int,
    T: int,
    patch_size: int = 16,
    context_patches: int = 32,
    mag_min: float = 1.5,
    mag_max: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (c_spike [n, T], spike_ts [n]).
    Single-timestep impulse restricted to the context window.
    """
    mag_abs = rng.uniform(mag_min, mag_max, size=n).astype(np.float32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    spike_ts = rng.integers(0, context_patches * patch_size, size=n)
    c_spike = np.zeros((n, T), dtype=np.float32)
    c_spike[np.arange(n), spike_ts] = mag_abs * signs
    return c_spike, spike_ts


def component_rw(
    rng: np.random.Generator,
    n: int,
    T: int,
    step_std: float = 0.3,
) -> np.ndarray:
    """
    Returns c_rw [n, T].
    Random walk: c_rw[0]=0, c_rw[t] = c_rw[t-1] + xi_t,  xi ~ N(0, step_std).
    """
    xi = rng.normal(0, step_std, size=(n, T - 1)).astype(np.float32)
    return np.concatenate([np.zeros((n, 1), dtype=np.float32), np.cumsum(xi, axis=1)], axis=1)


def component_noise_floor(
    rng: np.random.Generator,
    n: int,
    T: int,
    log_sigma_min: float = 0.05,
    log_sigma_max: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (noise [n, T], log_noise_var [n]).
    Always-on Gaussian noise; sigma ~ LogUniform(log_sigma_min, log_sigma_max).
    """
    log_sigma = rng.uniform(np.log(log_sigma_min), np.log(log_sigma_max), size=n).astype(np.float32)
    sigma = np.exp(log_sigma)
    noise = rng.standard_normal((n, T)).astype(np.float32) * sigma[:, None]
    return noise, (2.0 * log_sigma).astype(np.float32)


# ── Dataset generators ────────────────────────────────────────────────────────


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


def generate_composite_dataset(
    n: int = 5000,
    modes_dist: tuple[float, float, float] = MODES_DIST,
    concept_weights: tuple[float, ...] | None = CONCEPT_WEIGHTS,
    seed: int = 42,
    patch_size: int = 16,
    context_patches: int = 32,
    pred_patches: int = 4,
) -> dict[str, np.ndarray]:
    """
    Generate a synthetic dataset with 7 additive concepts and compositional mixing.

    Each example has 1 (atomic), 2 (pair), or 3 (triple) concepts active, sampled
    according to modes_dist. Series = sum of active components + always-on noise floor.

    Concept index order in concept_mask columns:
        [trend=0, level_shift=1, ar1=2, seasonal=3, var_shift=4, spike=5, rw=6]

    Parameters
    ----------
    concept_weights : length-7 sequence of non-negative floats, optional
        Relative sampling weight for each concept. Controls how often each
        concept appears across all modes. Pair/triple weights are the product
        of constituent concept weights. None defaults to uniform (equal weights).
        Example: (1,1,1,3,1,1,1) makes seasonal 3x more likely than others.

    Returns dict with 16 keys; see HANDOFF_PR6.md for full schema.
    """
    rng = np.random.default_rng(seed)
    T = (context_patches + pred_patches) * patch_size  # 576

    # ── Concept sampling probabilities ────────────────────────────────────────
    w = np.ones(7, dtype=np.float64) if concept_weights is None else np.asarray(concept_weights, dtype=np.float64)
    w = w / w.sum()

    pair_options = np.array(list(_combinations(range(7), 2)))    # [21, 2]
    triple_options = np.array(list(_combinations(range(7), 3)))  # [35, 3]

    pair_w = np.array([w[i] * w[j] for i, j in pair_options])
    pair_w = pair_w / pair_w.sum()
    triple_w = np.array([w[i] * w[j] * w[k] for i, j, k in triple_options])
    triple_w = triple_w / triple_w.sum()

    # ── concept mask ─────────────────────────────────────────────────────────
    n_atomic = int(round(n * modes_dist[0]))
    n_pair   = int(round(n * modes_dist[1]))
    n_triple = n - n_atomic - n_pair

    concept_mask = np.zeros((n, 7), dtype=bool)

    atomic_choices = rng.choice(7, size=n_atomic, p=w)
    concept_mask[np.arange(n_atomic), atomic_choices] = True

    pair_idx = rng.choice(len(pair_options), size=n_pair, p=pair_w)
    pair_chosen = pair_options[pair_idx]
    for k in range(2):
        concept_mask[n_atomic + np.arange(n_pair), pair_chosen[:, k]] = True

    triple_idx = rng.choice(len(triple_options), size=n_triple, p=triple_w)
    triple_chosen = triple_options[triple_idx]
    for k in range(3):
        concept_mask[n_atomic + n_pair + np.arange(n_triple), triple_chosen[:, k]] = True

    has_trend    = concept_mask[:, 0]
    has_level    = concept_mask[:, 1]
    has_ar1      = concept_mask[:, 2]
    has_seasonal = concept_mask[:, 3]
    has_var      = concept_mask[:, 4]
    has_spike    = concept_mask[:, 5]
    has_rw       = concept_mask[:, 6]

    # ── Generate all concept components ──────────────────────────────────────
    c_trend,    slopes                                       = component_trend(rng, n, T)
    c_level,    level_magnitudes, level_t_shifts             = component_level_shift(rng, n, T)
    c_ar,       phi_all                                      = component_ar1(rng, n, T)
    c_seasonal, period_idxs, seasonal_amps, phases           = component_seasonal(rng, n, T)
    c_var,      log_sigma_before, log_sigma_after, var_t_shifts = component_var_shift(rng, n, T)
    c_spike,    spike_ts                                     = component_spike(rng, n, T, patch_size, context_patches)
    c_rw                                                     = component_rw(rng, n, T)
    noise_floor, log_noise_var                               = component_noise_floor(rng, n, T)

    # ── Assemble series ───────────────────────────────────────────────────────
    series = noise_floor.copy()
    series += np.where(has_trend[:, None],    c_trend,    0.0)
    series += np.where(has_level[:, None],    c_level,    0.0)
    series += np.where(has_ar1[:, None],      c_ar,       0.0)
    series += np.where(has_seasonal[:, None], c_seasonal, 0.0)
    series += np.where(has_var[:, None],      c_var,      0.0)
    series += np.where(has_spike[:, None],    c_spike,    0.0)
    series += np.where(has_rw[:, None],       c_rw,       0.0)
    series = series.astype(np.float32)

    # ── Labels ────────────────────────────────────────────────────────────────
    _nan = np.float32(np.nan)

    return {
        "series":               series,
        "concept_mask":         concept_mask,
        "log_noise_var":        log_noise_var,
        # Trend (0.0 when absent — slope of 0 means no trend)
        "slope":                np.where(has_trend, slopes, np.float32(0.0)).astype(np.float32),
        # Level shift (NaN when absent)
        "level_magnitude":      np.where(has_level, level_magnitudes, _nan).astype(np.float32),
        "level_time_norm":      np.where(has_level, level_t_shifts.astype(np.float32) / T, _nan).astype(np.float32),
        # AR(1) (NaN when absent)
        "ar_phi":               np.where(has_ar1, phi_all, _nan).astype(np.float32),
        # Seasonal (NaN/-1 when absent)
        "period_idx":           np.where(has_seasonal, period_idxs, np.int32(-1)).astype(np.int32),
        "seasonal_amplitude":   np.where(has_seasonal, seasonal_amps, _nan).astype(np.float32),
        "phase_cos":            np.where(has_seasonal, np.cos(phases), _nan).astype(np.float32),
        "phase_sin":            np.where(has_seasonal, np.sin(phases), _nan).astype(np.float32),
        # Variance shift (NaN when absent)
        "log_sigma_ratio":      np.where(has_var, log_sigma_after - log_sigma_before, _nan).astype(np.float32),
        "var_shift_time_norm":  np.where(has_var, var_t_shifts.astype(np.float32) / T, _nan).astype(np.float32),
        # Spike (always defined)
        "spike_present":        has_spike.astype(np.int32),
        "spike_patch_idx":      np.where(has_spike, (spike_ts // patch_size).astype(np.int32), np.int32(-1)).astype(np.int32),
        # Random walk (always defined)
        "rw_present":           has_rw.astype(np.int32),
    }


def split_dataset(
    data: dict[str, np.ndarray],
    n_train: int = 4000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_idx, val_idx) integer index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data["series"]))
    return idx[:n_train], idx[n_train:]


def save_dataset(data: dict[str, np.ndarray], path: str) -> None:
    """Save dataset to a compressed .npz file (numpy appends .npz if missing)."""
    np.savez_compressed(path, **data)


def load_dataset(path: str) -> dict[str, np.ndarray]:
    """Load dataset from a .npz file returned by save_dataset."""
    f = np.load(path)
    return {k: f[k] for k in f.files}
