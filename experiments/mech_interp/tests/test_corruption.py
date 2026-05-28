import numpy as np
import pytest

from experiments.mech_interp.lib.corruption import (
    corrupt_add_noise,
    corrupt_mean_center,
    corrupt_noise,
    corrupt_reverse,
    corrupt_seasonal,
    corrupt_shuffle_patches,
    corrupt_trend,
    corrupt_zero_segment,
)
from experiments.mech_interp.lib.metrics import (
    mase,
    scaled_weighted_quantile_loss,
    weighted_quantile_loss,
)
from experiments.mech_interp.lib.synthetic import PERIOD_BINS

T = 576  # default series length


# ---------------------------------------------------------------------------
# corrupt_trend
# ---------------------------------------------------------------------------

def test_corrupt_trend_shape():
    series = np.random.default_rng(0).standard_normal(T).astype(np.float32)
    out = corrupt_trend(series, slope=0.01)
    assert out.shape == series.shape
    assert out.dtype == series.dtype


def test_corrupt_trend_removes_trend():
    t = np.arange(T, dtype=np.float32)
    slope, intercept = 0.03, 5.0
    series = slope * t + intercept  # pure trend, no noise
    out = corrupt_trend(series, slope=slope)
    # residual should be near zero (just floating-point noise)
    assert np.abs(out).max() < 1e-3


def test_corrupt_trend_zero_slope():
    series = np.ones(T, dtype=np.float32) * 3.0
    out = corrupt_trend(series, slope=0.0)
    assert np.abs(out).max() < 1e-3


# ---------------------------------------------------------------------------
# corrupt_seasonal
# ---------------------------------------------------------------------------

def test_corrupt_seasonal_shape():
    series = np.random.default_rng(1).standard_normal(T).astype(np.float32)
    out = corrupt_seasonal(series, period_idx=4, phase=0.5)
    assert out.shape == series.shape
    assert out.dtype == series.dtype


def test_corrupt_seasonal_removes_sinusoid():
    period_idx = 4  # PERIOD_BINS[4] = 8
    period = PERIOD_BINS[period_idx]
    phase = 1.2
    amp = 1.5
    t = np.arange(T, dtype=np.float64)
    series = (amp * np.sin(2 * np.pi / period * t + phase)).astype(np.float32)
    out = corrupt_seasonal(series, period_idx=period_idx, phase=phase)
    assert np.abs(out).max() < 1e-4


def test_corrupt_seasonal_all_period_bins():
    """All period bins should produce near-zero residuals on pure sinusoids."""
    T_local = 576
    t = np.arange(T_local, dtype=np.float64)
    for idx, period in enumerate(PERIOD_BINS):
        phase = 0.7
        amp = 2.0
        series = (amp * np.sin(2 * np.pi / period * t + phase)).astype(np.float32)
        out = corrupt_seasonal(series, period_idx=idx, phase=phase)
        assert np.abs(out).max() < 1e-3, f"period_idx={idx} (period={period}) failed"


# ---------------------------------------------------------------------------
# corrupt_noise
# ---------------------------------------------------------------------------

def test_corrupt_noise_shape():
    series = np.random.default_rng(2).standard_normal(T).astype(np.float32)
    out = corrupt_noise(series, seed=0)
    assert out.shape == series.shape
    assert out.dtype == series.dtype


def test_corrupt_noise_different_from_input():
    series = np.ones(T, dtype=np.float32)
    out = corrupt_noise(series, seed=7919)
    assert not np.allclose(out, series)


def test_corrupt_noise_similar_std():
    rng = np.random.default_rng(3)
    series = rng.standard_normal(T).astype(np.float32)
    out = corrupt_noise(series, seed=42)
    # std should be close (same scale), not necessarily identical
    assert abs(float(out.std()) - float(series.std())) < 0.3 * float(series.std())


def test_corrupt_noise_reproducible():
    series = np.random.default_rng(4).standard_normal(T).astype(np.float32)
    out1 = corrupt_noise(series, seed=100)
    out2 = corrupt_noise(series, seed=100)
    np.testing.assert_array_equal(out1, out2)


def test_corrupt_noise_different_seeds():
    series = np.random.default_rng(5).standard_normal(T).astype(np.float32)
    out1 = corrupt_noise(series, seed=0)
    out2 = corrupt_noise(series, seed=7919)
    assert not np.allclose(out1, out2)


# ---------------------------------------------------------------------------
# corrupt_add_noise
# ---------------------------------------------------------------------------

def test_corrupt_add_noise_shape():
    series = np.random.default_rng(6).standard_normal(T).astype(np.float32)
    out = corrupt_add_noise(series, seed=0)
    assert out.shape == series.shape
    assert out.dtype == series.dtype


def test_corrupt_add_noise_preserves_signal():
    series = np.random.default_rng(7).standard_normal(T).astype(np.float32)
    out = corrupt_add_noise(series, seed=0)
    assert not np.allclose(out, series)
    # output should correlate with input (signal still present)
    assert float(np.corrcoef(series, out)[0, 1]) > 0.5


def test_corrupt_add_noise_custom_std():
    series = np.ones(T, dtype=np.float32) * 5.0
    out = corrupt_add_noise(series, seed=0, std=0.01)
    # tiny std — output should still be close to original
    assert np.abs(out - series).max() < 0.1


def test_corrupt_add_noise_reproducible():
    series = np.random.default_rng(8).standard_normal(T).astype(np.float32)
    assert np.array_equal(corrupt_add_noise(series, seed=42), corrupt_add_noise(series, seed=42))


def test_corrupt_add_noise_different_from_replace():
    series = np.random.default_rng(9).standard_normal(T).astype(np.float32)
    assert not np.allclose(corrupt_add_noise(series, seed=0), corrupt_noise(series, seed=0))


# ---------------------------------------------------------------------------
# mase
# ---------------------------------------------------------------------------

def test_mase_perfect_forecast():
    target = np.array([1.0, 2.0, 3.0])
    context = np.array([0.0, 1.0, 2.0, 3.0])
    assert mase(target, target, context) == pytest.approx(0.0, abs=1e-8)


def test_mase_positive():
    target = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.5, 2.5, 3.5])
    context = np.array([0.0, 1.0, 2.0, 3.0])
    assert mase(forecast, target, context) > 0.0


def test_mase_known_value():
    # mae = 0.5, scale = mean(|diff([0,1,2,3])|) + 1e-8 = 1.0 + 1e-8
    target = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.5, 2.5, 3.5])
    context = np.array([0.0, 1.0, 2.0, 3.0])
    expected = 0.5 / (1.0 + 1e-8)
    assert mase(forecast, target, context) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# weighted_quantile_loss
# ---------------------------------------------------------------------------

def test_wql_shape_single_quantile():
    q = np.array([0.5])
    f = np.zeros((1, 10))
    y = np.zeros(10)
    assert weighted_quantile_loss(f, q, y) == pytest.approx(0.0, abs=1e-8)


def test_wql_median_exact():
    # median forecast equals target → loss = 0
    y = np.array([1.0, 2.0, 3.0])
    f = y[None, :]
    q = np.array([0.5])
    assert weighted_quantile_loss(f, q, y) == pytest.approx(0.0, abs=1e-8)


def test_wql_positive_for_nonzero_error():
    y = np.array([1.0, 2.0, 3.0])
    f = np.array([[0.0, 0.0, 0.0]])  # underforecast for q=0.5
    q = np.array([0.5])
    assert weighted_quantile_loss(f, q, y) > 0.0


def test_wql_known_value():
    # q=0.9, forecast=0, target=1 → error=1, pinball = 0.9*1 = 0.9
    q = np.array([0.9])
    f = np.zeros((1, 1))
    y = np.ones(1)
    assert weighted_quantile_loss(f, q, y) == pytest.approx(0.9, rel=1e-6)


def test_wql_overforecast():
    # q=0.1, forecast=2, target=1 → error=-1, pinball = (0.1-1)*(-1) = 0.9
    q = np.array([0.1])
    f = np.array([[2.0]])
    y = np.array([1.0])
    assert weighted_quantile_loss(f, q, y) == pytest.approx(0.9, rel=1e-6)


def test_wql_multi_quantile():
    q = np.array([0.1, 0.5, 0.9])
    y = np.zeros(5)
    f = np.zeros((3, 5))  # perfect for all quantiles
    assert weighted_quantile_loss(f, q, y) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# scaled_weighted_quantile_loss
# ---------------------------------------------------------------------------

def test_swql_scales_by_context():
    q = np.array([0.9])
    f = np.zeros((1, 1))
    y = np.ones(1)
    # context with mean |diff| = 2.0
    context = np.array([0.0, 2.0, 4.0])
    unscaled = weighted_quantile_loss(f, q, y)
    scale = np.mean(np.abs(np.diff(context))) + 1e-8
    expected = unscaled / scale
    assert scaled_weighted_quantile_loss(f, q, y, context) == pytest.approx(expected, rel=1e-6)


def test_swql_perfect_forecast_zero():
    y = np.array([1.0, 2.0])
    f = y[None, :]
    q = np.array([0.5])
    context = np.array([0.0, 1.0, 2.0, 3.0])
    assert scaled_weighted_quantile_loss(f, q, y, context) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# corrupt_mean_center
# ---------------------------------------------------------------------------

def test_corrupt_mean_center_shape():
    series = np.random.default_rng(10).standard_normal(T).astype(np.float32)
    out = corrupt_mean_center(series)
    assert out.shape == series.shape
    assert out.dtype == series.dtype
    assert abs(float(out[:512].mean())) < 1e-5


def test_corrupt_mean_center_horizon_shifted():
    series = np.random.default_rng(11).standard_normal(T).astype(np.float32)
    ctx_mean = float(series[:512].mean())
    out = corrupt_mean_center(series)
    np.testing.assert_allclose(out[512:], series[512:] - ctx_mean, rtol=1e-5)


# ---------------------------------------------------------------------------
# corrupt_reverse
# ---------------------------------------------------------------------------

def test_corrupt_reverse_shape():
    series = np.random.default_rng(12).standard_normal(T).astype(np.float32)
    out = corrupt_reverse(series)
    assert out.shape == series.shape
    assert out.dtype == series.dtype
    np.testing.assert_array_equal(out[:512], series[:512][::-1])
    np.testing.assert_array_equal(out[512:], series[512:])


# ---------------------------------------------------------------------------
# corrupt_shuffle_patches
# ---------------------------------------------------------------------------

def test_corrupt_shuffle_patches_shape():
    series = np.random.default_rng(13).standard_normal(T).astype(np.float32)
    out = corrupt_shuffle_patches(series, seed=0)
    assert out.shape == series.shape
    assert out.dtype == series.dtype
    # horizon unchanged
    np.testing.assert_array_equal(out[512:], series[512:])
    # same values present, just reordered
    np.testing.assert_array_equal(np.sort(out[:512]), np.sort(series[:512]))


def test_corrupt_shuffle_patches_reproducible():
    series = np.random.default_rng(14).standard_normal(T).astype(np.float32)
    out1 = corrupt_shuffle_patches(series, seed=42)
    out2 = corrupt_shuffle_patches(series, seed=42)
    np.testing.assert_array_equal(out1, out2)
    out3 = corrupt_shuffle_patches(series, seed=99)
    assert not np.array_equal(out1, out3)


# ---------------------------------------------------------------------------
# corrupt_zero_segment
# ---------------------------------------------------------------------------

def test_corrupt_zero_segment_shape():
    series = np.random.default_rng(15).standard_normal(T).astype(np.float32)
    out = corrupt_zero_segment(series, seed=0)
    assert out.shape == series.shape
    assert out.dtype == series.dtype
    # horizon unchanged
    np.testing.assert_array_equal(out[512:], series[512:])
    # exactly 4*16=64 zeros exist somewhere in context
    assert (out[:512] == 0).sum() >= 64


def test_corrupt_zero_segment_reproducible():
    series = np.random.default_rng(16).standard_normal(T).astype(np.float32)
    out1 = corrupt_zero_segment(series, seed=7)
    out2 = corrupt_zero_segment(series, seed=7)
    np.testing.assert_array_equal(out1, out2)
    out3 = corrupt_zero_segment(series, seed=999)
    # different seeds likely zero different segments (not guaranteed but extremely probable)
    assert not np.array_equal(out1[:512], out3[:512])
