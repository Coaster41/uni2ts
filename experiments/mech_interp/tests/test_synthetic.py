import numpy as np
import pytest

from lib import (
    AR1LabelGenerator,
    DEFAULT_GENERATORS,
    LevelShiftLabelGenerator,
    NoiseVarLabelGenerator,
    SeasonalLabelGenerator,
    TrendLabelGenerator,
    generate_composite_dataset,
    generate_dataset,
    load_dataset,
    save_dataset,
    split_dataset,
    wrap_existing_dataset,
)

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4
SERIES_LENGTH = (CONTEXT_PATCHES + PRED_PATCHES) * PATCH_SIZE  # 576

SYNTH_KEYS = {
    "series", "slope", "period_idx", "phase_cos", "phase_sin", "log_noise_var",
    # keys added by AR1LabelGenerator and LevelShiftLabelGenerator in DEFAULT_GENERATORS
    "ar_phi", "level_magnitude", "level_time_norm",
}


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


# ── generate_composite_dataset ───────────────────────────────────────────────

_COMPOSITE_KEYS = {
    "series", "concept_mask", "log_noise_var",
    "slope",
    "level_magnitude", "level_time_norm",
    "ar_phi",
    "period_idx", "seasonal_amplitude", "phase_cos", "phase_sin",
    "log_sigma_ratio", "var_shift_time_norm",
    "spike_present", "spike_patch_idx",
    "rw_present",
}


@pytest.fixture(scope="module")
def composite_data():
    return generate_composite_dataset(n=500, seed=42)


def test_composite_dataset_shapes(composite_data):
    data = composite_data
    assert set(data.keys()) == _COMPOSITE_KEYS
    assert data["series"].shape == (500, SERIES_LENGTH)
    assert data["concept_mask"].shape == (500, 7)
    for k in _COMPOSITE_KEYS - {"series", "concept_mask"}:
        assert data[k].shape == (500,), f"wrong shape for {k}"


def test_composite_dataset_dtypes(composite_data):
    data = composite_data
    float_keys = {
        "series", "log_noise_var", "slope",
        "level_magnitude", "level_time_norm", "ar_phi",
        "seasonal_amplitude", "phase_cos", "phase_sin",
        "log_sigma_ratio", "var_shift_time_norm",
    }
    int_keys = {"period_idx", "spike_present", "spike_patch_idx", "rw_present"}
    for k in float_keys:
        assert data[k].dtype == np.float32, f"expected float32 for {k}, got {data[k].dtype}"
    for k in int_keys:
        assert data[k].dtype == np.int32, f"expected int32 for {k}, got {data[k].dtype}"
    assert data["concept_mask"].dtype == bool


def test_composite_dataset_reproducible():
    a = generate_composite_dataset(n=100, seed=7)
    b = generate_composite_dataset(n=100, seed=7)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"mismatch in {k}")


def test_composite_dataset_different_seeds():
    a = generate_composite_dataset(n=100, seed=1)
    b = generate_composite_dataset(n=100, seed=2)
    assert not np.array_equal(a["series"], b["series"])


def test_composite_presence_distribution():
    data = generate_composite_dataset(n=5000, seed=42)
    counts = data["concept_mask"].sum(axis=1)
    n = len(counts)
    assert abs((counts == 1).mean() - 0.50) < 0.05
    assert abs((counts == 2).mean() - 0.35) < 0.05
    assert abs((counts == 3).mean() - 0.15) < 0.05


def test_composite_label_ranges(composite_data):
    data = composite_data
    assert np.all(data["slope"] >= -3.0) and np.all(data["slope"] <= 3.0)
    phi_present = data["ar_phi"][data["concept_mask"][:, 2]]
    assert np.all(phi_present >= -0.95) and np.all(phi_present <= 0.95)
    assert np.all(np.isin(data["period_idx"], [-1] + list(range(8))))
    assert np.all(np.isin(data["spike_patch_idx"], [-1] + list(range(32))))


def test_composite_absent_labels_nan(composite_data):
    data = composite_data
    # NaN-when-absent checks (concept column → label keys)
    nan_absent = [
        (1, ["level_magnitude", "level_time_norm"]),
        (2, ["ar_phi"]),
        (3, ["seasonal_amplitude", "phase_cos", "phase_sin"]),
        (4, ["log_sigma_ratio", "var_shift_time_norm"]),
    ]
    for col, keys in nan_absent:
        absent = ~data["concept_mask"][:, col]
        present = data["concept_mask"][:, col]
        for k in keys:
            assert np.all(np.isnan(data[k][absent])), f"{k} should be NaN when concept {col} absent"
            assert np.all(np.isfinite(data[k][present])), f"{k} should be finite when concept {col} present"


def test_composite_always_defined_labels(composite_data):
    data = composite_data
    for k in ("log_noise_var", "spike_present", "rw_present"):
        assert not np.any(np.isnan(data[k].astype(np.float32))), f"{k} must have no NaN"


def test_composite_backward_compat():
    data = generate_dataset(n=10, seed=42)
    assert set(data.keys()) == {"series", "slope", "period_idx", "phase_cos", "phase_sin", "log_noise_var"}
    assert data["series"].shape == (10, SERIES_LENGTH)
    assert data["series"].dtype == np.float32


def test_split_dataset_sizes():
    data = generate_composite_dataset(n=5000, seed=42)
    train_idx, val_idx = split_dataset(data, n_train=4000, seed=42)
    assert len(train_idx) == 4000
    assert len(val_idx) == 1000
    assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_ar1_label_generator_recovers_phi():
    rng = np.random.default_rng(0)
    phi_true = 0.7
    T = SERIES_LENGTH
    s = np.zeros(T, dtype=np.float64)
    eta = rng.normal(0, 1, size=T)
    for i in range(1, T):
        s[i] = phi_true * s[i - 1] + eta[i]
    out = AR1LabelGenerator()(s.astype(np.float32))
    assert abs(float(out["ar_phi"]) - phi_true) < 0.1


def test_composite_concept_weights_bias():
    # Make seasonal (col 3) weight 6x others; it should dominate in atomic mode
    weights = [1, 1, 1, 6, 1, 1, 1]
    data = generate_composite_dataset(n=5000, seed=0, concept_weights=weights)
    atomic_mask = data["concept_mask"][data["concept_mask"].sum(axis=1) == 1]
    seasonal_frac = atomic_mask[:, 3].mean()
    # Expect ~6/12 = 50% of atomic examples to be seasonal; at least 40%
    assert seasonal_frac > 0.40, f"expected seasonal dominance, got {seasonal_frac:.2f}"


def test_composite_concept_weights_uniform_distribution():
    # With equal weights all 7 concepts should appear at similar rates in atomic mode
    data = generate_composite_dataset(n=7000, seed=42)
    atomic_mask = data["concept_mask"][data["concept_mask"].sum(axis=1) == 1]
    for j in range(7):
        frac = atomic_mask[:, j].mean()
        assert abs(frac - 1 / 7) < 0.05, f"concept {j} atomic freq={frac:.3f}, expected ~{1/7:.3f}"


def test_level_shift_label_generator_keys():
    rng = np.random.default_rng(1)
    s = rng.normal(size=SERIES_LENGTH).astype(np.float32)
    s[SERIES_LENGTH // 2 :] += 5.0  # insert clear shift
    out = LevelShiftLabelGenerator()(s)
    assert set(out.keys()) == {"level_magnitude", "level_time_norm"}
    assert np.isfinite(out["level_magnitude"]) and out["level_magnitude"].dtype == np.float32
    assert np.isfinite(out["level_time_norm"]) and out["level_time_norm"].dtype == np.float32
