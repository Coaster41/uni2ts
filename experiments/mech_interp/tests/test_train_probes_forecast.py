"""Tests for train_probes_forecast.py — PR-14.

All tests use tiny in-memory fixture data (n=40, d_model=64, layers={-1,0,1}).
No model forward passes or real checkpoints required.
"""
from __future__ import annotations

import json

import numpy as np
import pytest


def _make_runner_output(n: int = 40, d_model: int = 64, seed: int = 0) -> dict[str, np.ndarray]:
    """Build a dict matching the structure of a forecast_runner .npz file."""
    rng = np.random.default_rng(seed)

    # Quantile forecasts: enforce monotonicity across the Q axis so iqr >= 0.
    raw = rng.standard_normal((n, 9, 64)).astype(np.float32)
    fq = np.sort(raw, axis=1)

    out = {
        "forecast_quantiles": fq,
        "target": rng.standard_normal((n, 64)).astype(np.float32),
        "context": rng.standard_normal((n, 512)).astype(np.float32),
    }
    for k in (-1, 0, 1):
        out[f"activations_mean_ctx_layer_{k}"] = rng.standard_normal((n, d_model)).astype(np.float32)
        out[f"activations_last_ctx_layer_{k}"] = rng.standard_normal((n, d_model)).astype(np.float32)
    return out


@pytest.fixture
def tiny_runner_output() -> dict[str, np.ndarray]:
    return _make_runner_output(n=60, d_model=64)


@pytest.fixture
def tiny_split(tiny_runner_output):
    n = len(tiny_runner_output["forecast_quantiles"])
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    return idx[:n_train], idx[n_train:]   # 48 train / 12 val


def test_compute_forecast_targets_shapes(tiny_runner_output):
    from experiments.mech_interp.block1_probing.train_probes_forecast import (
        compute_forecast_targets,
        ALL_FORECAST_FEATURES,
    )

    targets = compute_forecast_targets(tiny_runner_output, ctx_period=24)

    n = len(tiny_runner_output["forecast_quantiles"])
    assert set(targets.keys()) == set(ALL_FORECAST_FEATURES), (
        f"Missing keys: {set(ALL_FORECAST_FEATURES) - set(targets.keys())}"
    )
    for k, arr in targets.items():
        assert arr.shape == (n,), f"{k}: expected ({n},), got {arr.shape}"

    # Binary labels must be int32 and exactly half ones (strict median split)
    n = len(tiny_runner_output["forecast_quantiles"])
    for bl in ("is_flat", "is_poor"):
        assert targets[bl].dtype == np.int32, f"{bl} dtype: {targets[bl].dtype}"
        assert targets[bl].sum() == n // 2, (
            f"{bl}: expected {n // 2} ones, got {targets[bl].sum()}"
        )


def test_run_forecast_probes_output_structure(tiny_runner_output, tiny_split):
    from experiments.mech_interp.block1_probing.train_probes_forecast import (
        run_forecast_probes,
        ALL_FORECAST_FEATURES,
    )

    train_idx, val_idx = tiny_split
    results = run_forecast_probes(tiny_runner_output, train_idx, val_idx, ctx_period=24)

    assert set(results.keys()) == {"mean_ctx", "last_ctx"}
    for pooling, feat_dict in results.items():
        for feature in ALL_FORECAST_FEATURES:
            assert feature in feat_dict, f"[{pooling}] missing feature: {feature}"
            layer_scores = feat_dict[feature]
            assert set(layer_scores.keys()) == {-1, 0, 1}, (
                f"[{pooling}][{feature}] unexpected layer keys: {set(layer_scores.keys())}"
            )
            for layer_idx, score in layer_scores.items():
                assert np.isfinite(score), (
                    f"[{pooling}][{feature}][{layer_idx}] non-finite score: {score}"
                )


def test_end_to_end_file_output(tiny_runner_output, tmp_path):
    """main() creates moiraie_synth.json with expected structure."""
    import sys
    from unittest.mock import patch

    # Write the tiny runner output as a .npz file.
    npz_path = tmp_path / "moiraie_synth.npz"
    np.savez(str(npz_path), **tiny_runner_output)

    out_dir = str(tmp_path / "out")

    argv = [
        "train_probes_forecast",
        "--npz-dir", str(tmp_path),
        "--dataset", "synth",
        "--model", "moiraie",
        "--output-dir", out_dir,
    ]
    with patch.object(sys, "argv", argv):
        from experiments.mech_interp.block1_probing import train_probes_forecast as tpf
        tpf.main()

    result_path = tmp_path / "out" / "moiraie_synth.json"
    assert result_path.exists(), f"Expected {result_path} to exist"

    with open(result_path) as f:
        data = json.load(f)

    assert "mean_ctx" in data, "Top-level key 'mean_ctx' missing"
    assert "last_ctx" in data, "Top-level key 'last_ctx' missing"

    # Spot-check a feature is present and has string layer keys.
    mean_ctx = data["mean_ctx"]
    assert "fc_std" in mean_ctx, "'fc_std' missing from mean_ctx"
    layer_keys = list(mean_ctx["fc_std"].keys())
    assert all(isinstance(k, str) for k in layer_keys), "Layer keys should be strings"
    assert "-1" in layer_keys, "Layer -1 missing from fc_std results"

    # metadata.json should also exist
    meta_path = tmp_path / "out" / "metadata.json"
    assert meta_path.exists(), "metadata.json not written"
