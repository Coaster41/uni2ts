import numpy as np
import pytest

from lib import (
    DEFAULT_GENERATORS,
    NoiseVarLabelGenerator,
    SeasonalLabelGenerator,
    TrendLabelGenerator,
    generate_dataset,
    load_dataset,
    save_dataset,
    wrap_existing_dataset,
)

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4
SERIES_LENGTH = (CONTEXT_PATCHES + PRED_PATCHES) * PATCH_SIZE  # 576

SYNTH_KEYS = {"series", "slope", "period_idx", "phase_cos", "phase_sin", "log_noise_var"}


@pytest.fixture(scope="module")
def synth_series_list() -> list[np.ndarray]:
    """Synthetic series used as a stand-in source for wrapping tests."""
    data = generate_dataset(n=50, seed=42)
    return list(data["series"])


# ── generate_dataset ─────────────────────────────────────────────────────────


def test_generate_dataset_shapes():
    data = generate_dataset(n=10, seed=42)
    assert data["series"].shape == (10, SERIES_LENGTH)
    for k in ("slope", "period_idx", "phase_cos", "phase_sin", "log_noise_var"):
        assert data[k].shape == (10,), f"wrong shape for {k}"


def test_generate_dataset_dtypes():
    data = generate_dataset(n=10, seed=0)
    assert data["series"].dtype == np.float32
    assert data["slope"].dtype == np.float32
    assert data["period_idx"].dtype == np.int32
    assert data["phase_cos"].dtype == np.float32
    assert data["phase_sin"].dtype == np.float32
    assert data["log_noise_var"].dtype == np.float32


def test_generate_dataset_label_ranges():
    data = generate_dataset(n=200, seed=0)
    assert np.all(data["slope"] >= -0.05) and np.all(data["slope"] <= 0.05)
    assert np.all(np.isin(data["period_idx"], list(range(8))))
    assert np.all(np.isfinite(data["phase_cos"])) and np.all(np.isfinite(data["phase_sin"]))
    assert np.all(data["log_noise_var"] >= -4.0) and np.all(data["log_noise_var"] <= 0.0)


def test_generate_dataset_reproducible():
    a = generate_dataset(n=50, seed=7)
    b = generate_dataset(n=50, seed=7)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"mismatch in {k}")


def test_generate_dataset_different_seeds():
    a = generate_dataset(n=50, seed=1)
    b = generate_dataset(n=50, seed=2)
    assert not np.array_equal(a["series"], b["series"])


def test_save_load_roundtrip(tmp_path):
    data = generate_dataset(n=10, seed=1)
    path = str(tmp_path / "dataset")
    save_dataset(data, path)
    loaded = load_dataset(path + ".npz")
    assert set(loaded.keys()) == set(data.keys())
    for k in data:
        np.testing.assert_array_equal(data[k], loaded[k], err_msg=f"mismatch in {k}")


# ── wrap_existing_dataset ────────────────────────────────────────────────────


def test_wrap_existing_dataset_label_keys(synth_series_list):
    wrapped = wrap_existing_dataset(
        synth_series_list, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=0
    )
    assert set(wrapped.keys()) == SYNTH_KEYS


def test_wrap_existing_dataset_shapes(synth_series_list):
    wrapped = wrap_existing_dataset(
        synth_series_list, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=0
    )
    assert wrapped["series"].shape == (20, SERIES_LENGTH)
    assert wrapped["slope"].shape == (20,)
    assert wrapped["period_idx"].shape == (20,)
    assert wrapped["phase_cos"].shape == (20,)
    assert wrapped["phase_sin"].shape == (20,)
    assert wrapped["log_noise_var"].shape == (20,)


def test_wrap_existing_dataset_finite(synth_series_list):
    wrapped = wrap_existing_dataset(
        synth_series_list, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=0
    )
    for k, v in wrapped.items():
        assert np.all(np.isfinite(v)), f"Non-finite values in '{k}'"


def test_wrap_existing_dataset_reproducible(synth_series_list):
    a = wrap_existing_dataset(
        synth_series_list, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=99
    )
    b = wrap_existing_dataset(
        synth_series_list, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=99
    )
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"mismatch in {k}")


def test_wrap_existing_dataset_different_seeds(synth_series_list):
    a = wrap_existing_dataset(
        synth_series_list, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=1
    )
    b = wrap_existing_dataset(
        synth_series_list, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=2
    )
    assert not np.array_equal(a["series"], b["series"])


def test_wrap_existing_dataset_insufficient_data():
    with pytest.raises(ValueError, match="valid windows"):
        wrap_existing_dataset(
            [np.zeros(100, dtype=np.float32)],
            DEFAULT_GENERATORS,
            series_length=SERIES_LENGTH,
            n=10,
            seed=0,
        )


def test_wrap_existing_dataset_nan_excluded():
    good = np.ones(SERIES_LENGTH, dtype=np.float32)
    bad = np.ones(SERIES_LENGTH, dtype=np.float32)
    bad[10] = float("nan")
    wrapped = wrap_existing_dataset(
        [good, bad] * 10, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=5, seed=0
    )
    assert wrapped["series"].shape[0] == 5
    assert np.all(np.isfinite(wrapped["series"]))


# ── individual label generators ──────────────────────────────────────────────


def _random_series(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=SERIES_LENGTH).astype(np.float32)


def test_trend_label_generator():
    gen = TrendLabelGenerator()
    out = gen(_random_series())
    assert set(out.keys()) == {"slope"}
    assert np.isfinite(out["slope"])
    assert out["slope"].dtype == np.float32


def test_seasonal_label_generator():
    gen = SeasonalLabelGenerator()
    out = gen(_random_series())
    assert set(out.keys()) == {"period_idx", "phase_cos", "phase_sin"}
    assert int(out["period_idx"]) in range(8)
    assert np.isfinite(out["phase_cos"]) and np.isfinite(out["phase_sin"])


def test_noise_var_label_generator():
    gen = NoiseVarLabelGenerator()
    out = gen(_random_series())
    assert set(out.keys()) == {"log_noise_var"}
    assert np.isfinite(out["log_noise_var"])
    assert out["log_noise_var"].dtype == np.float32


def test_trend_generator_recovers_slope():
    """Generator should recover a known slope from a pure-trend series (normalized axis)."""
    t = np.arange(SERIES_LENGTH, dtype=np.float32) / SERIES_LENGTH
    true_slope = 3.5
    series = (true_slope * t).astype(np.float32)
    out = TrendLabelGenerator()(series)
    assert abs(float(out["slope"]) - true_slope) < 1e-3


def test_seasonal_generator_recovers_period():
    """Generator should identify the correct period bin for a pure sinusoid."""
    # period_ts=24 is index 1 in PERIOD_BINS=[7, 24, 30, 12, 8, 16, 32, 64]
    period_ts = 24
    t = np.arange(SERIES_LENGTH, dtype=np.float32)
    series = np.sin(2 * np.pi / period_ts * t).astype(np.float32)
    out = SeasonalLabelGenerator()(series)
    assert int(out["period_idx"]) == 1
