import math

import numpy as np
import pytest
import torch

from uni2ts.model.moiraic.module import MoiraicModule
from uni2ts.model.moiraie.module import MoiraieModule

from experiments.mech_interp.lib import (
    DEFAULT_GENERATORS,
    ResidualExtractor,
    generate_dataset,
    make_batch,
    wrap_existing_dataset,
)
from experiments.mech_interp.block1_probing.train_probes import (
    CLASSIFICATION_FEATURES,
    CONTEXT_PATCHES,
    PATCH_SIZE,
    PRED_PATCHES,
    REGRESSION_FEATURES,
    extract_activations,
    fit_probe,
    run_probes_for_model,
)

_TINY = dict(
    d_model=64,
    d_ff=128,
    num_layers=2,
    patch_size=16,
    max_seq_len=64,
    attn_dropout_p=0.0,
    dropout_p=0.0,
)

N = 10
SERIES_LENGTH = (CONTEXT_PATCHES + PRED_PATCHES) * PATCH_SIZE  # 576


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def series(rng):
    return rng.standard_normal((N, SERIES_LENGTH)).astype(np.float32)


@pytest.fixture
def module_e():
    return MoiraieModule(**_TINY, num_predict_token=1).eval()


@pytest.fixture
def module_c():
    return MoiraicModule(**_TINY, num_predict_token=PRED_PATCHES).eval()


@pytest.fixture
def tiny_dataset():
    # n=100 ensures all 8 period_idx classes appear in 80-example train splits
    return generate_dataset(n=100, seed=0)


# ---------------------------------------------------------------------------
# extract_activations
# ---------------------------------------------------------------------------

def test_extract_activations_shapes_moiraie(module_e, series):
    acts = extract_activations(module_e, series, batch_size=4)
    assert set(acts.keys()) == set(range(_TINY["num_layers"]))
    for layer_idx, arr in acts.items():
        assert arr.shape == (N, _TINY["d_model"]), (
            f"Layer {layer_idx}: expected ({N}, {_TINY['d_model']}), got {arr.shape}"
        )


def test_extract_activations_shapes_moiraic(module_c, series):
    acts = extract_activations(module_c, series, batch_size=4)
    assert set(acts.keys()) == set(range(_TINY["num_layers"]))
    for layer_idx, arr in acts.items():
        assert arr.shape == (N, _TINY["d_model"])


def test_extract_activations_returns_numpy(module_e, series):
    acts = extract_activations(module_e, series)
    for arr in acts.values():
        assert isinstance(arr, np.ndarray)
        assert np.isfinite(arr).all(), "Activations contain NaN or Inf"


def test_extract_activations_batch_boundary(module_e, series):
    # batch_size larger than dataset — should still work
    acts = extract_activations(module_e, series, batch_size=100)
    for arr in acts.values():
        assert arr.shape[0] == N


# ---------------------------------------------------------------------------
# fit_probe
# ---------------------------------------------------------------------------

def test_fit_probe_regression_returns_float(rng):
    X_train = rng.standard_normal((80, 64)).astype(np.float32)
    X_val = rng.standard_normal((20, 64)).astype(np.float32)
    y_train = rng.standard_normal(80).astype(np.float32)
    y_val = rng.standard_normal(20).astype(np.float32)
    score = fit_probe(X_train, X_val, y_train, y_val, "regression")
    assert isinstance(score, float)
    assert math.isfinite(score)


def test_fit_probe_classification_returns_float(rng):
    X_train = rng.standard_normal((80, 64)).astype(np.float32)
    X_val = rng.standard_normal((20, 64)).astype(np.float32)
    y_train = rng.integers(0, 4, size=80).astype(np.int32)
    y_val = rng.integers(0, 4, size=20).astype(np.int32)
    score = fit_probe(X_train, X_val, y_train, y_val, "classification")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_fit_probe_invalid_type_raises():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    y = rng.standard_normal(10).astype(np.float32)
    with pytest.raises(ValueError):
        fit_probe(X, X, y, y, "invalid_type")


# ---------------------------------------------------------------------------
# run_probes_for_model
# ---------------------------------------------------------------------------

def test_run_probes_for_model_structure(module_e, tiny_dataset):
    n = len(tiny_dataset["series"])
    idx = np.arange(n)
    train_idx = idx[:80]
    val_idx = idx[80:]

    results = run_probes_for_model(module_e, tiny_dataset, train_idx, val_idx, batch_size=8)

    expected_features = set(REGRESSION_FEATURES) | set(CLASSIFICATION_FEATURES)
    assert set(results.keys()) == expected_features, (
        f"Missing or extra features: got {set(results.keys())}"
    )
    for feature, layer_scores in results.items():
        assert set(layer_scores.keys()) == set(range(_TINY["num_layers"])), (
            f"{feature}: layer keys mismatch"
        )
        for layer_idx, score in layer_scores.items():
            assert isinstance(score, float), f"{feature} layer {layer_idx}: score is not float"
            assert math.isfinite(score), f"{feature} layer {layer_idx}: score is not finite"


def test_run_probes_for_model_moiraic(module_c, tiny_dataset):
    n = len(tiny_dataset["series"])
    idx = np.arange(n)
    results = run_probes_for_model(module_c, tiny_dataset, idx[:80], idx[80:], batch_size=8)
    assert set(results.keys()) == set(REGRESSION_FEATURES) | set(CLASSIFICATION_FEATURES)


# ---------------------------------------------------------------------------
# PR-0 compatibility: wrap_existing_dataset + DEFAULT_GENERATORS
# ---------------------------------------------------------------------------

def test_pr0_compat_wrap_existing_dataset_label_keys():
    """wrap_existing_dataset + DEFAULT_GENERATORS must produce the same label keys as generate_dataset."""
    synth = generate_dataset(n=20, seed=0)
    expected_keys = set(synth.keys())

    windows = [synth["series"][i] for i in range(20)]
    wrapped = wrap_existing_dataset(windows, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=0)

    assert set(wrapped.keys()) == expected_keys, (
        f"Key mismatch: wrapped={set(wrapped.keys())}, expected={expected_keys}"
    )


def test_pr0_compat_run_probes_accepts_wrapped_dataset(module_e):
    """run_probes_for_model must accept output of wrap_existing_dataset without error."""
    synth = generate_dataset(n=200, seed=0)
    windows = [synth["series"][i] for i in range(200)]
    wrapped = wrap_existing_dataset(windows, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=100, seed=1)

    n = len(wrapped["series"])
    idx = np.arange(n)
    results = run_probes_for_model(module_e, wrapped, idx[:80], idx[80:], batch_size=8)

    assert set(results.keys()) == set(REGRESSION_FEATURES) | set(CLASSIFICATION_FEATURES)
    for feature, layer_scores in results.items():
        assert len(layer_scores) == _TINY["num_layers"]
