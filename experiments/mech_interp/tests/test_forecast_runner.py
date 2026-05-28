"""Tests for forecast_runner.py — PR-13.

All tests use tiny in-memory models and synthetic data; no real checkpoints required.
"""
from __future__ import annotations

import numpy as np
import pytest


_TINY = dict(
    d_model=64,
    d_ff=128,
    num_layers=2,
    patch_size=16,
    max_seq_len=64,
    attn_dropout_p=0.0,
    dropout_p=0.0,
)


@pytest.fixture
def module_e():
    from uni2ts.model.moiraie.module import MoiraieModule
    return MoiraieModule(**_TINY, num_predict_token=1).eval()


@pytest.fixture
def module_c():
    from uni2ts.model.moiraic.module import MoiraicModule
    return MoiraicModule(**_TINY, num_predict_token=4).eval()


@pytest.fixture
def tiny_series():
    return np.random.default_rng(0).standard_normal((10, 576)).astype(np.float32)


def test_output_shapes_moiraie(module_e, tiny_series, tmp_path):
    import experiments.mech_interp.block1_probing.forecast_runner as fr

    out = str(tmp_path / "test.npz")
    fr.run_and_save(module_e, {"series": tiny_series}, out)
    data = fr.load_runner_output(out)

    n = len(tiny_series)
    assert data["forecast_quantiles"].shape == (n, 9, 64)
    assert data["target"].shape == (n, 64)
    assert data["context"].shape == (n, 512)
    assert "activations_mean_ctx_layer_-1" in data
    assert "activations_mean_ctx_layer_0" in data
    assert "activations_last_ctx_layer_-1" in data
    for k in data:
        if k.startswith("activations_mean_ctx"):
            assert data[k].shape == (n, 64), f"{k}: {data[k].shape}"


def test_output_shapes_moiraic(module_c, tiny_series, tmp_path):
    import experiments.mech_interp.block1_probing.forecast_runner as fr

    out = str(tmp_path / "test.npz")
    fr.run_and_save(module_c, {"series": tiny_series}, out)
    data = fr.load_runner_output(out)

    n = len(tiny_series)
    assert data["forecast_quantiles"].shape == (n, 9, 64)
    assert data["target"].shape == (n, 64)
    assert data["context"].shape == (n, 512)
    assert "activations_mean_ctx_layer_-1" in data
    assert "activations_mean_ctx_layer_0" in data
    assert "activations_last_ctx_layer_-1" in data
    for k in data:
        if k.startswith("activations_mean_ctx"):
            assert data[k].shape == (n, 64), f"{k}: {data[k].shape}"


def test_single_pass_consistency(module_e, tiny_series, tmp_path):
    import experiments.mech_interp.block1_probing.forecast_runner as fr

    out = str(tmp_path / "test.npz")
    fr.run_and_save(module_e, {"series": tiny_series}, out)
    data = fr.load_runner_output(out)

    np.testing.assert_allclose(data["context"], tiny_series[:, :512], rtol=1e-5)
    np.testing.assert_allclose(data["target"], tiny_series[:, 512:], rtol=1e-5)


def test_all_layers_present_moiraie(module_e, tiny_series, tmp_path):
    """Both pooling modes and all expected layers (-1, 0, 1) must appear."""
    import experiments.mech_interp.block1_probing.forecast_runner as fr

    out = str(tmp_path / "test.npz")
    fr.run_and_save(module_e, {"series": tiny_series}, out)
    data = fr.load_runner_output(out)

    for layer_idx in (-1, 0, 1):
        assert f"activations_mean_ctx_layer_{layer_idx}" in data
        assert f"activations_last_ctx_layer_{layer_idx}" in data


def test_dtypes_are_float32(module_e, tiny_series, tmp_path):
    import experiments.mech_interp.block1_probing.forecast_runner as fr

    out = str(tmp_path / "test.npz")
    fr.run_and_save(module_e, {"series": tiny_series}, out)
    data = fr.load_runner_output(out)

    for k, arr in data.items():
        assert arr.dtype == np.float32, f"{k} has dtype {arr.dtype}"


def test_batching_matches_full_pass(module_e, tmp_path):
    """Results should be identical regardless of batch size."""
    import experiments.mech_interp.block1_probing.forecast_runner as fr

    rng = np.random.default_rng(1)
    series = rng.standard_normal((8, 576)).astype(np.float32)
    ds = {"series": series}

    out1 = str(tmp_path / "bs8.npz")
    out2 = str(tmp_path / "bs3.npz")
    fr.run_and_save(module_e, ds, out1, batch_size=8)
    fr.run_and_save(module_e, ds, out2, batch_size=3)

    d1 = fr.load_runner_output(out1)
    d2 = fr.load_runner_output(out2)

    np.testing.assert_allclose(d1["forecast_quantiles"], d2["forecast_quantiles"], rtol=1e-4)
    np.testing.assert_allclose(
        d1["activations_mean_ctx_layer_0"],
        d2["activations_mean_ctx_layer_0"],
        rtol=1e-4,
    )
