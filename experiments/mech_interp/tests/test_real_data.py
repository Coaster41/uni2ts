from __future__ import annotations

import numpy as np
import pytest

from lib.pseudo_labels import (
    adf_pvalue,
    context_acf_lag1,
    context_std,
    fft_dominant_period,
    hurst_exponent,
    n_changepoints,
    sample_entropy,
    spectral_flatness,
    stl_trend_strength,
)


def test_fft_dominant_period_known_sine():
    T = 512
    period = 24
    t = np.arange(T, dtype=np.float32)
    ctx = np.sin(2 * np.pi * t / period).astype(np.float32)
    result = fft_dominant_period(ctx)
    assert abs(np.exp(result) - period) < 2.0, f"Expected period≈{period}, got {np.exp(result):.1f}"


def test_adf_pvalue_stationary_vs_rw():
    rng = np.random.default_rng(0)
    stationary = rng.standard_normal(512).astype(np.float32)
    rw = np.cumsum(rng.standard_normal(512)).astype(np.float32)
    p_stat = adf_pvalue(stationary)
    p_rw = adf_pvalue(rw)
    assert p_stat < 0.05, f"White noise should reject unit root, got p={p_stat:.4f}"
    assert p_rw > 0.1, f"Random walk should fail to reject, got p={p_rw:.4f}"


def test_hurst_exponent_bounds():
    rng = np.random.default_rng(1)
    for _ in range(5):
        ctx = rng.standard_normal(512).astype(np.float32)
        h = hurst_exponent(ctx)
        assert 0.2 <= h <= 1.0, f"Hurst out of clip range: {h}"


def test_stl_trend_strength_pure_trend():
    ctx = np.linspace(0, 10, 512).astype(np.float32)
    f_t = stl_trend_strength(ctx)
    assert f_t > 0.8, f"Pure trend should have high F_T, got {f_t:.3f}"


def test_context_std_and_acf_known():
    ctx = np.ones(512, dtype=np.float32) * 3.0
    assert context_std(ctx) == pytest.approx(0.0, abs=1e-5)
    acf = context_acf_lag1(ctx)
    assert np.isnan(acf) or abs(acf - 1.0) < 0.01


def test_load_gift_subset_schema(monkeypatch):
    def _stub(dataset_name, **kwargs):
        rng = np.random.default_rng(0)
        return [rng.standard_normal(1000).astype(np.float32) for _ in range(10)]

    import lib.real_data as real_data_mod

    monkeypatch.setattr(real_data_mod, "load_gift_eval_series", _stub)

    ds = real_data_mod.load_gift_subset(n_per_dataset=20)
    N = 9 * 20
    assert ds["series"].shape == (N, 576)
    assert ds["dataset_id"].shape == (N,)
    expected_labels = {
        "stl_trend_strength",
        "stl_seasonal_strength",
        "fft_dominant_period",
        "fft_top1_power_frac",
        "spectral_flatness",
        "adf_pvalue",
        "hurst_exponent",
        "sample_entropy",
        "n_changepoints",
        "context_std",
        "context_acf_lag1",
    }
    assert expected_labels.issubset(ds.keys())
    assert ds["context_std"].dtype == np.float32
    assert not np.any(np.isnan(ds["context_std"]))


def test_n_changepoints_flat_vs_step():
    flat = np.zeros(512, dtype=np.float32)  # truly constant — zero first-differences
    step = np.concatenate([np.zeros(256), np.ones(256) * 10]).astype(np.float32)
    n_flat = n_changepoints(flat)
    n_step = n_changepoints(step)
    assert n_step >= 1, f"Step signal should detect at least 1 changepoint, got {n_step}"
    assert n_step >= n_flat, f"Step should have >= changepoints than flat: {n_step} vs {n_flat}"


def test_sample_entropy_constant():
    ctx = np.ones(512, dtype=np.float32)
    val = sample_entropy(ctx)
    assert np.isfinite(val), "sample_entropy on constant should not raise"


def test_spectral_flatness_bounds():
    rng = np.random.default_rng(3)
    ctx = rng.standard_normal(512).astype(np.float32)
    sf = spectral_flatness(ctx)
    assert 0.0 <= sf <= 1.0, f"Spectral flatness should be in [0,1], got {sf}"
