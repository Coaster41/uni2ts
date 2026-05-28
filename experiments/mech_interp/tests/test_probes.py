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
from experiments.mech_interp.lib.synthetic import generate_composite_dataset
from experiments.mech_interp.block1_probing.probe_utils import (
    CONTEXT_PATCHES,
    PATCH_SIZE,
    PRED_PATCHES,
    extract_activations,
    extract_activations_per_patch,
    batched_ridge_per_patch,
    fit_probe,
)
from experiments.mech_interp.block1_probing.train_probes import (
    BINARY_FEATURES,
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
    run_probes_for_model,
    run_probes_per_patch,
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
    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(acts.keys()) == expected_keys
    for layer_idx, arr in acts.items():
        assert arr.shape == (N, _TINY["d_model"]), (
            f"Layer {layer_idx}: expected ({N}, {_TINY['d_model']}), got {arr.shape}"
        )


def test_extract_activations_shapes_moiraic(module_c, series):
    acts = extract_activations(module_c, series, batch_size=4)
    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(acts.keys()) == expected_keys
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

    expected_poolings = {"mean_ctx", CONTEXT_PATCHES - 1}
    assert set(results.keys()) == expected_poolings, (
        f"Expected pooling modes {expected_poolings}, got {set(results.keys())}"
    )
    all_known_features = set(REGRESSION_FEATURES) | set(BINARY_FEATURES) | set(CLASSIFICATION_FEATURES)
    expected_layers = {-1} | set(range(_TINY["num_layers"]))
    for pooling, feat_dict in results.items():
        assert set(feat_dict.keys()) <= all_known_features, (
            f"[{pooling}] Unknown features returned: {set(feat_dict.keys()) - all_known_features}"
        )
        assert len(feat_dict) > 0, f"[{pooling}] No features in result"
        for feature, layer_scores in feat_dict.items():
            assert set(layer_scores.keys()) == expected_layers, (
                f"[{pooling}] {feature}: layer keys mismatch, got {set(layer_scores.keys())}"
            )
            for layer_idx, score in layer_scores.items():
                assert isinstance(score, float), f"[{pooling}] {feature} layer {layer_idx}: score is not float"
                assert math.isfinite(score), f"[{pooling}] {feature} layer {layer_idx}: score is not finite"


def test_run_probes_for_model_moiraic(module_c, tiny_dataset):
    n = len(tiny_dataset["series"])
    idx = np.arange(n)
    results = run_probes_for_model(module_c, tiny_dataset, idx[:80], idx[80:], batch_size=8)
    expected_poolings = {"mean_ctx", CONTEXT_PATCHES - 1}
    assert set(results.keys()) == expected_poolings
    all_known_features = set(REGRESSION_FEATURES) | set(BINARY_FEATURES) | set(CLASSIFICATION_FEATURES)
    for pooling_dict in results.values():
        assert set(pooling_dict.keys()) <= all_known_features
        assert len(pooling_dict) > 0


# ---------------------------------------------------------------------------
# PR-0 compatibility: wrap_existing_dataset + DEFAULT_GENERATORS
# ---------------------------------------------------------------------------

def test_pr0_compat_wrap_existing_dataset_label_keys():
    """wrap_existing_dataset + DEFAULT_GENERATORS must cover all generate_dataset label keys."""
    synth = generate_dataset(n=20, seed=0)
    original_keys = set(synth.keys())

    windows = [synth["series"][i] for i in range(20)]
    wrapped = wrap_existing_dataset(windows, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=20, seed=0)

    # wrapped may have MORE keys than generate_dataset (PR-6 expanded DEFAULT_GENERATORS),
    # but must include all original keys
    assert original_keys <= set(wrapped.keys()), (
        f"Missing keys: {original_keys - set(wrapped.keys())}"
    )


def test_pr0_compat_run_probes_accepts_wrapped_dataset(module_e):
    """run_probes_for_model must accept output of wrap_existing_dataset without error."""
    synth = generate_dataset(n=200, seed=0)
    windows = [synth["series"][i] for i in range(200)]
    wrapped = wrap_existing_dataset(windows, DEFAULT_GENERATORS, series_length=SERIES_LENGTH, n=100, seed=1)

    n = len(wrapped["series"])
    idx = np.arange(n)
    results = run_probes_for_model(module_e, wrapped, idx[:80], idx[80:], batch_size=8)

    # results is now {pooling_mode: {feature: {layer: score}}}
    expected_poolings = {"mean_ctx", CONTEXT_PATCHES - 1}
    assert set(results.keys()) == expected_poolings
    expected_layers = {-1} | set(range(_TINY["num_layers"]))
    for pooling, feat_dict in results.items():
        for feature, layer_scores in feat_dict.items():
            assert set(layer_scores.keys()) == expected_layers


# ---------------------------------------------------------------------------
# PR-5a: extract_activations_per_patch
# ---------------------------------------------------------------------------

def test_extract_activations_per_patch_shape_moiraie(module_e, series):
    """Shape must be [n, context_patches, d_model] per layer — patch axis preserved."""
    acts = extract_activations_per_patch(module_e, series, batch_size=4)
    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(acts.keys()) == expected_keys
    for layer_idx, arr in acts.items():
        assert arr.shape == (N, CONTEXT_PATCHES, _TINY["d_model"]), (
            f"Layer {layer_idx}: expected ({N}, {CONTEXT_PATCHES}, {_TINY['d_model']}), got {arr.shape}"
        )


def test_extract_activations_per_patch_shape_moiraic(module_c, series):
    acts = extract_activations_per_patch(module_c, series, batch_size=4)
    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(acts.keys()) == expected_keys
    for arr in acts.values():
        assert arr.shape == (N, CONTEXT_PATCHES, _TINY["d_model"])


def test_extract_activations_per_patch_returns_numpy(module_e, series):
    acts = extract_activations_per_patch(module_e, series)
    for arr in acts.values():
        assert isinstance(arr, np.ndarray)
        assert np.isfinite(arr).all()


# ---------------------------------------------------------------------------
# PR-5a: batched_ridge_per_patch
# ---------------------------------------------------------------------------

def test_batched_ridge_per_patch_shape():
    """Output shape must be [B, k]."""
    B, n_train, n_val, d, k = 8, 40, 10, 32, 3
    rng_t = torch.manual_seed(0)
    X_tr = torch.randn(B, n_train, d)
    X_va = torch.randn(B, n_val, d)
    Y_tr = torch.randn(n_train, k)
    Y_va = torch.randn(n_val, k)
    r2 = batched_ridge_per_patch(X_tr, X_va, Y_tr, Y_va)
    assert r2.shape == (B, k), f"Expected ({B}, {k}), got {r2.shape}"


def test_batched_ridge_per_patch_range():
    """R² values should be clamped to [-1, 1]."""
    B, n, d, k = 4, 30, 16, 2
    X_tr = torch.randn(B, n, d)
    X_va = torch.randn(B, 10, d)
    Y_tr = torch.randn(n, k)
    Y_va = torch.randn(10, k)
    r2 = batched_ridge_per_patch(X_tr, X_va, Y_tr, Y_va)
    assert (r2 >= -1.0).all() and (r2 <= 1.0).all()


def test_batched_ridge_agrees_with_ridge_cv():
    """batched_ridge_per_patch R² should be close to sklearn RidgeCV for the same data."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    rng_np = np.random.default_rng(7)
    n_train, n_val, d = 80, 20, 16

    X_tr_np = rng_np.standard_normal((n_train, d)).astype(np.float32)
    X_va_np = rng_np.standard_normal((n_val, d)).astype(np.float32)
    y_tr = (rng_np.standard_normal(n_train) * 2 + X_tr_np[:, 0]).astype(np.float32)
    y_va = (rng_np.standard_normal(n_val) * 2 + X_va_np[:, 0]).astype(np.float32)

    # sklearn reference
    probe = Pipeline([("scaler", StandardScaler()),
                      ("ridge", RidgeCV(alphas=[1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3], cv=5))])
    probe.fit(X_tr_np, y_tr)
    r2_sklearn = float(probe.score(X_va_np, y_va))

    # batched ridge (B=1, k=1)
    X_tr_t = torch.from_numpy(X_tr_np[None])          # [1, n, d]
    X_va_t = torch.from_numpy(X_va_np[None])
    Y_tr_t = torch.from_numpy(y_tr[:, None])           # [n, 1]
    Y_va_t = torch.from_numpy(y_va[:, None])
    r2_batched = float(batched_ridge_per_patch(X_tr_t, X_va_t, Y_tr_t, Y_va_t)[0, 0])

    assert abs(r2_sklearn - r2_batched) < 0.05, (
        f"R² gap too large: sklearn={r2_sklearn:.4f}, batched={r2_batched:.4f}"
    )


# ---------------------------------------------------------------------------
# PR-5a: run_probes_per_patch
# ---------------------------------------------------------------------------

def test_run_probes_per_patch_structure(module_e, tiny_dataset):
    """Output must be {feature: {layer: {patch_idx: float}}} with all values finite."""
    n = len(tiny_dataset["series"])
    idx = np.arange(n)
    results = run_probes_per_patch(module_e, tiny_dataset, idx[:80], idx[80:], batch_size=8)

    all_known_features = set(REGRESSION_FEATURES) | set(BINARY_FEATURES) | set(CLASSIFICATION_FEATURES)
    assert set(results.keys()) <= all_known_features
    assert len(results) > 0

    expected_layers = {-1} | set(range(_TINY["num_layers"]))
    for feat, layer_dict in results.items():
        assert set(layer_dict.keys()) == expected_layers
        for layer_idx, patch_dict in layer_dict.items():
            assert len(patch_dict) == CONTEXT_PATCHES, (
                f"{feat} layer {layer_idx}: expected {CONTEXT_PATCHES} patches, got {len(patch_dict)}"
            )
            for patch_idx, score in patch_dict.items():
                assert isinstance(score, float)
                assert math.isfinite(score), f"{feat} L{layer_idx} P{patch_idx}: score={score}"


def test_run_probes_per_patch_moiraic(module_c, tiny_dataset):
    n = len(tiny_dataset["series"])
    idx = np.arange(n)
    results = run_probes_per_patch(module_c, tiny_dataset, idx[:80], idx[80:], batch_size=8)
    all_known_features = set(REGRESSION_FEATURES) | set(BINARY_FEATURES) | set(CLASSIFICATION_FEATURES)
    assert set(results.keys()) <= all_known_features
    assert len(results) > 0
    expected_layers = {-1} | set(range(_TINY["num_layers"]))
    for feat, layer_dict in results.items():
        assert set(layer_dict.keys()) == expected_layers


# ---------------------------------------------------------------------------
# PR-7: Layer-0 baseline + last_ctx pooling
# ---------------------------------------------------------------------------

def test_layer0_extraction_shape(module_e, series):
    """ResidualExtractor returns key -1 with shape [n, context_patches, d_model] after slice."""
    from experiments.mech_interp.lib import make_batch
    batch = make_batch(series, patch_size=PATCH_SIZE, context_patches=CONTEXT_PATCHES, pred_patches=PRED_PATCHES)
    with ResidualExtractor(module_e) as ext:
        acts, _ = ext.run(batch)
    assert -1 in acts, "ResidualExtractor must return key -1 for in_proj activation"
    ctx_slice = acts[-1][:, :CONTEXT_PATCHES, :]
    assert ctx_slice.shape == (N, CONTEXT_PATCHES, _TINY["d_model"]), (
        f"Expected ({N}, {CONTEXT_PATCHES}, {_TINY['d_model']}), got {ctx_slice.shape}"
    )


def test_layer0_included_in_activations(module_e, series):
    """extract_activations with default pooling='mean_ctx' returns key -1 alongside 0..num_layers-1."""
    acts = extract_activations(module_e, series, batch_size=4)
    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(acts.keys()) == expected_keys, (
        f"Expected layer keys {expected_keys}, got {set(acts.keys())}"
    )
    assert acts[-1].shape == (N, _TINY["d_model"]), (
        f"Layer -1 shape: expected ({N}, {_TINY['d_model']}), got {acts[-1].shape}"
    )


def test_int_pooling_matches_per_patch_position(module_e, series):
    """extract_activations(..., pooling=k)[layer] equals per_patch[layer][:, k, :] for all layers."""
    k = CONTEXT_PATCHES - 1
    pos_acts = extract_activations(module_e, series, batch_size=4, pooling=k)
    per_patch_acts = extract_activations_per_patch(module_e, series, batch_size=4)

    for layer_idx in sorted(pos_acts.keys()):
        expected = per_patch_acts[layer_idx][:, k, :]
        actual = pos_acts[layer_idx]
        assert np.array_equal(actual, expected), (
            f"Layer {layer_idx}: pooling={k} not bit-exact with per_patch[:, {k}, :]"
        )


def test_mean_ctx_and_int_pooling_shapes(module_e, series):
    """Both mean_ctx and integer-index pooling return [n, d_model] per layer including layer -1."""
    mean_acts = extract_activations(module_e, series, batch_size=4, pooling="mean_ctx")
    pos_acts = extract_activations(module_e, series, batch_size=4, pooling=CONTEXT_PATCHES - 1)

    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(mean_acts.keys()) == expected_keys
    assert set(pos_acts.keys()) == expected_keys

    for layer_idx in expected_keys:
        assert mean_acts[layer_idx].shape == (N, _TINY["d_model"]), (
            f"mean_ctx layer {layer_idx}: expected ({N}, {_TINY['d_model']}), got {mean_acts[layer_idx].shape}"
        )
        assert pos_acts[layer_idx].shape == (N, _TINY["d_model"]), (
            f"pos_{CONTEXT_PATCHES-1} layer {layer_idx}: expected ({N}, {_TINY['d_model']}), got {pos_acts[layer_idx].shape}"
        )


# ---------------------------------------------------------------------------
# PR-8: composite dataset, NaN masking, binary AUROC, spike_patch_idx
# ---------------------------------------------------------------------------

@pytest.fixture
def composite_dataset():
    return generate_composite_dataset(n=200, seed=0)


def test_nan_masking_no_crash(module_e, composite_dataset):
    """run_probes_for_model completes without error on composite dataset; regression features present."""
    n = len(composite_dataset["series"])
    idx = np.arange(n)
    results = run_probes_for_model(module_e, composite_dataset, idx[:160], idx[160:], batch_size=8)

    expected_poolings = {"mean_ctx", CONTEXT_PATCHES - 1}
    assert set(results.keys()) == expected_poolings
    present_reg = [f for f in REGRESSION_FEATURES if f in composite_dataset]
    for pooling, feat_dict in results.items():
        for feat in present_reg:
            assert feat in feat_dict, f"[{pooling}] regression feature {feat!r} missing from results"


def test_binary_features_use_auroc(rng):
    """fit_probe with feature_type='binary' returns a value in [0, 1]."""
    X_train = rng.standard_normal((80, 64)).astype(np.float32)
    X_val = rng.standard_normal((20, 64)).astype(np.float32)
    y_train = rng.integers(0, 2, size=80).astype(np.float32)
    y_val = rng.integers(0, 2, size=20).astype(np.float32)
    score = fit_probe(X_train, X_val, y_train, y_val, "binary")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_conditional_feature_scores_not_nan(module_e, composite_dataset):
    """Conditional features (ar_phi, level_magnitude, log_sigma_ratio) return finite scores."""
    conditional = ["ar_phi", "level_magnitude", "log_sigma_ratio"]
    n = len(composite_dataset["series"])
    idx = np.arange(n)
    results = run_probes_for_model(module_e, composite_dataset, idx[:160], idx[160:], batch_size=8)

    for pooling, feat_dict in results.items():
        for feat in conditional:
            if feat not in composite_dataset:
                continue
            assert feat in feat_dict, f"[{pooling}] {feat} missing despite being in dataset"
            for layer_idx, score in feat_dict[feat].items():
                assert math.isfinite(score), f"[{pooling}] {feat} layer {layer_idx}: score={score} is not finite"


def test_spike_patch_idx_per_patch(module_e, composite_dataset):
    """run_probes_per_patch returns spike_patch_idx key with finite patch scores."""
    if "spike_patch_idx" not in composite_dataset:
        pytest.skip("composite_dataset missing spike_patch_idx")
    n = len(composite_dataset["series"])
    idx = np.arange(n)
    results = run_probes_per_patch(module_e, composite_dataset, idx[:160], idx[160:], batch_size=8)

    assert "spike_patch_idx" in results, "spike_patch_idx missing from per-patch results"
    layer_dict = results["spike_patch_idx"]
    expected_layers = {-1} | set(range(_TINY["num_layers"]))
    assert set(layer_dict.keys()) == expected_layers
    for layer_idx, patch_dict in layer_dict.items():
        assert len(patch_dict) == CONTEXT_PATCHES
        for patch_idx, score in patch_dict.items():
            assert isinstance(score, float)
            assert math.isfinite(score), f"spike_patch_idx L{layer_idx} P{patch_idx}: score={score}"
