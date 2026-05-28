from __future__ import annotations

import numpy as np
import pytest

from experiments.mech_interp.block1_probing.forecast_properties import (
    HORIZON,
    QUANTILE_LEVELS,
    compute_all,
    compute_mase,
    compute_swql,
    derive_binary_labels,
    fc_ctx_corr,
    fc_ctx_corr_seasonal,
    fc_iqr_mean,
    fc_iqr_slope,
    fc_range,
    fc_std,
    quantile_calibration_err,
)

_Q = len(QUANTILE_LEVELS)
_CTX_LEN = 512


def _flat_fq(value: float = 1.0) -> np.ndarray:
    """All quantiles equal to value everywhere — zero spread."""
    return np.full((_Q, HORIZON), value, dtype=np.float64)


def _rng_fq(seed: int = 0) -> np.ndarray:
    """Random forecast quantiles with proper quantile ordering."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(HORIZON)
    # add offset per quantile level so q ordering holds
    offsets = np.linspace(-2.0, 2.0, _Q)[:, None]
    return base[None, :] + offsets


def _ctx(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_CTX_LEN)


# ---------------------------------------------------------------------------
# fc_std
# ---------------------------------------------------------------------------

def test_fc_std_flat():
    val = fc_std(_flat_fq())
    assert val < -10, f"flat forecast should give fc_std ≈ log(1e-6) ≈ -13.8, got {val}"


def test_fc_std_varying():
    fq_flat = _flat_fq()
    fq_var = _rng_fq()
    assert fc_std(fq_var) > fc_std(fq_flat)


# ---------------------------------------------------------------------------
# fc_range
# ---------------------------------------------------------------------------

def test_fc_range_flat():
    val = fc_range(_flat_fq())
    assert val < -10


def test_fc_range_varying():
    fq = _flat_fq()
    fq[4] = np.linspace(0, 10, HORIZON)  # median spans [0, 10]
    val = fc_range(fq)
    assert val > np.log(9.9), f"range of ~10 → log(10)≈2.3, got {val}"


# ---------------------------------------------------------------------------
# fc_ctx_corr
# ---------------------------------------------------------------------------

def test_fc_ctx_corr_perfect():
    ctx = _ctx()
    fq = _flat_fq()
    fq[4] = ctx[-HORIZON:]  # median = last HORIZON ctx steps
    corr = fc_ctx_corr(fq, ctx)
    assert abs(corr - 1.0) < 1e-6, f"expected corr≈1.0, got {corr}"


def test_fc_ctx_corr_anti():
    ctx = _ctx()
    fq = _flat_fq()
    last = ctx[-HORIZON:]
    fq[4] = -last + last.mean() * 2  # anti-correlated, same spread
    corr = fc_ctx_corr(fq, ctx)
    assert corr < -0.99, f"expected corr≈-1.0, got {corr}"


def test_fc_ctx_corr_flat_returns_nan():
    ctx = _ctx()
    fq = _flat_fq()  # constant median → nan
    corr = fc_ctx_corr(fq, ctx)
    assert np.isnan(corr)


# ---------------------------------------------------------------------------
# fc_ctx_corr_seasonal
# ---------------------------------------------------------------------------

def test_fc_ctx_corr_seasonal_known():
    ctx = _ctx()
    P = 24
    fq = _flat_fq()
    fq[4] = ctx[-(HORIZON + P):-P]  # forecast = seasonal lag
    corr = fc_ctx_corr_seasonal(fq, ctx, ctx_period=P)
    assert abs(corr - 1.0) < 1e-6, f"expected corr≈1.0, got {corr}"


def test_fc_ctx_corr_seasonal_short_context_returns_nan():
    short_ctx = np.ones(50)
    fq = _rng_fq()
    val = fc_ctx_corr_seasonal(fq, short_ctx, ctx_period=24)
    assert np.isnan(val)


def test_fc_ctx_corr_seasonal_zero_period_returns_nan():
    fq = _rng_fq()
    val = fc_ctx_corr_seasonal(fq, _ctx(), ctx_period=0)
    assert np.isnan(val)


# ---------------------------------------------------------------------------
# fc_iqr_mean
# ---------------------------------------------------------------------------

def test_fc_iqr_mean_tight():
    fq = _flat_fq()  # q0.1 == q0.9 == 1.0
    val = fc_iqr_mean(fq)
    assert abs(val) < 1e-10


def test_fc_iqr_mean_wide():
    fq = _flat_fq()
    fq[0] = np.zeros(HORIZON)   # q0.1
    fq[-1] = np.full(HORIZON, 2.0)  # q0.9
    val = fc_iqr_mean(fq)
    assert abs(val - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# fc_iqr_slope
# ---------------------------------------------------------------------------

def test_fc_iqr_slope_flat():
    fq = _flat_fq()
    fq[0] = np.zeros(HORIZON)
    fq[-1] = np.ones(HORIZON) * 2.0  # constant IQR = 2
    slope = fc_iqr_slope(fq)
    assert abs(slope) < 1e-8


def test_fc_iqr_slope_growing():
    fq = _flat_fq()
    fq[0] = np.zeros(HORIZON)
    fq[-1] = np.linspace(0, 4, HORIZON)  # IQR linearly increases
    slope = fc_iqr_slope(fq)
    assert slope > 0


def test_fc_iqr_slope_shrinking():
    fq = _flat_fq()
    fq[0] = np.zeros(HORIZON)
    fq[-1] = np.linspace(4, 0, HORIZON)  # IQR linearly decreases
    slope = fc_iqr_slope(fq)
    assert slope < 0


# ---------------------------------------------------------------------------
# compute_mase
# ---------------------------------------------------------------------------

def test_mase_perfect():
    ctx = _ctx()
    target = np.random.default_rng(2).standard_normal(HORIZON)
    fq = _flat_fq()
    fq[4] = target  # median = target
    val = compute_mase(fq, target, ctx)
    assert abs(val) < 1e-8


def test_mase_nonzero_on_bad_forecast():
    ctx = _ctx()
    target = np.zeros(HORIZON)
    fq = _flat_fq(value=10.0)
    val = compute_mase(fq, target, ctx)
    assert val > 0


# ---------------------------------------------------------------------------
# compute_swql
# ---------------------------------------------------------------------------

def test_swql_perfect_median():
    ctx = _ctx()
    target = np.random.default_rng(3).standard_normal(HORIZON)
    # All quantiles equal target → pinball loss = 0
    fq = np.tile(target[None, :], (_Q, 1))
    val = compute_swql(fq, target, ctx)
    assert abs(val) < 1e-8


def test_swql_nonnegative():
    ctx = _ctx()
    target = np.random.default_rng(4).standard_normal(HORIZON)
    fq = _rng_fq()
    assert compute_swql(fq, target, ctx) >= 0


# ---------------------------------------------------------------------------
# quantile_calibration_err
# ---------------------------------------------------------------------------

def test_calibration_err_perfectly_calibrated():
    # Use a uniform target; set forecast_q = empirical q-th percentile.
    # With large T this converges to 0.
    rng = np.random.default_rng(5)
    target = rng.uniform(0, 1, HORIZON)
    fq = np.array([np.percentile(target, q * 100) * np.ones(HORIZON) for q in QUANTILE_LEVELS])
    err = quantile_calibration_err(fq, target)
    # Not exactly 0 due to discrete coverage, but should be small
    assert err < 0.15, f"expected near-zero calibration error, got {err}"


def test_calibration_err_worst_case():
    # Forecast always below target → coverage = 0 for all quantiles
    target = np.ones(HORIZON) * 10.0
    fq = np.zeros((_Q, HORIZON))  # all forecasts = 0 < target
    err = quantile_calibration_err(fq, target)
    # |0 - q| averaged over QUANTILE_LEVELS = mean([0.1..0.9]) = 0.5
    assert abs(err - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# derive_binary_labels
# ---------------------------------------------------------------------------

def test_derive_binary_labels_shape_and_dtype():
    n = 100
    fc_stds = np.random.default_rng(6).standard_normal(n)
    mases = np.abs(np.random.default_rng(7).standard_normal(n))
    labels = derive_binary_labels(fc_stds, mases)
    assert set(labels.keys()) == {"is_flat", "is_poor"}
    assert labels["is_flat"].shape == (n,)
    assert labels["is_poor"].shape == (n,)


def test_derive_binary_labels_median_split():
    # Exactly half should be positive
    n = 100
    fc_stds = np.arange(n, dtype=float)  # strictly ordered, even n
    mases = np.arange(n, dtype=float)
    labels = derive_binary_labels(fc_stds, mases)
    # is_flat: values < median → lower half
    assert labels["is_flat"].sum() == n // 2
    # is_poor: values > median → upper half
    assert labels["is_poor"].sum() == n // 2


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all_keys():
    expected = {
        "fc_std", "fc_range", "fc_ctx_corr", "fc_ctx_corr_seasonal",
        "fc_iqr_mean", "fc_iqr_slope", "mase", "swql", "quantile_calibration_err",
    }
    fq = _rng_fq()
    ctx = _ctx()
    target = np.random.default_rng(8).standard_normal(HORIZON)
    result = compute_all(fq, target, ctx, ctx_period=24)
    assert set(result.keys()) == expected


def test_compute_all_values_are_finite_or_nan():
    fq = _rng_fq()
    ctx = _ctx()
    target = np.random.default_rng(9).standard_normal(HORIZON)
    result = compute_all(fq, target, ctx, ctx_period=24)
    for k, v in result.items():
        assert np.isfinite(v) or np.isnan(v), f"{k} = {v} is neither finite nor nan"
