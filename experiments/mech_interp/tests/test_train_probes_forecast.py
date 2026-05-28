"""Tests for forecast probe targets and training via the consolidated API.

Replaces the old train_probes_forecast tests. compute_forecast_targets now
lives in forecast_runner; probe training uses fit_probe from probe_utils.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from experiments.mech_interp.block1_probing.train_probes import (
    FORECAST_REGRESSION_FEATURES,
    FORECAST_BINARY_FEATURES,
)

ALL_FORECAST_FEATURES = FORECAST_REGRESSION_FEATURES + FORECAST_BINARY_FEATURES


def _make_runner_output(n: int = 40, d_model: int = 64, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
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
def tiny_runner_output():
    return _make_runner_output(n=60, d_model=64)


@pytest.fixture
def tiny_split(tiny_runner_output):
    n = len(tiny_runner_output["forecast_quantiles"])
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    return idx[:n_train], idx[n_train:]


def test_compute_forecast_targets_shapes(tiny_runner_output):
    from experiments.mech_interp.block1_probing.forecast_runner import compute_forecast_targets

    targets = compute_forecast_targets(tiny_runner_output, ctx_period=24)
    n = len(tiny_runner_output["forecast_quantiles"])

    assert set(targets.keys()) == set(ALL_FORECAST_FEATURES), (
        f"Missing keys: {set(ALL_FORECAST_FEATURES) - set(targets.keys())}"
    )
    for k, arr in targets.items():
        assert arr.shape == (n,), f"{k}: expected ({n},), got {arr.shape}"

    for bl in ("is_flat", "is_poor"):
        assert targets[bl].dtype == np.int32, f"{bl} dtype: {targets[bl].dtype}"
        assert targets[bl].sum() == n // 2, (
            f"{bl}: expected {n // 2} ones, got {targets[bl].sum()}"
        )


def test_forecast_probe_training_finite(tiny_runner_output, tiny_split):
    """Probe training on forecast targets returns finite scores."""
    import math
    from experiments.mech_interp.block1_probing.forecast_runner import (
        compute_forecast_targets,
        _parse_layer_indices,
    )
    from experiments.mech_interp.block1_probing.probe_utils import fit_probe

    train_idx, val_idx = tiny_split
    targets = compute_forecast_targets(tiny_runner_output, ctx_period=24)
    layer_indices = _parse_layer_indices(tiny_runner_output)

    for pooling_suffix in ("mean_ctx", "last_ctx"):
        for feature in FORECAST_REGRESSION_FEATURES[:2]:  # spot-check two features
            y = targets[feature]
            tr_mask = np.isfinite(y[train_idx])
            va_mask = np.isfinite(y[val_idx])
            if tr_mask.sum() < 5 or va_mask.sum() < 5:
                continue
            for layer_idx in layer_indices[:2]:  # spot-check two layers
                X = tiny_runner_output[f"activations_{pooling_suffix}_layer_{layer_idx}"]
                score = fit_probe(
                    X[train_idx][tr_mask],
                    X[val_idx][va_mask],
                    y[train_idx][tr_mask],
                    y[val_idx][va_mask],
                    "regression",
                )
                assert math.isfinite(score), (
                    f"[{pooling_suffix}][{feature}][{layer_idx}] non-finite score: {score}"
                )


def test_end_to_end_synth_forecast(tmp_path):
    """main() with --dataset synth --forecast creates expected output files."""
    rng = np.random.default_rng(0)
    n = 9 * 20
    fake_ds = {
        "series": rng.standard_normal((n, 576)).astype(np.float32),
        "slope": rng.standard_normal(n).astype(np.float32),
        "log_noise_var": rng.standard_normal(n).astype(np.float32),
        "phase_cos": rng.standard_normal(n).astype(np.float32),
        "phase_sin": rng.standard_normal(n).astype(np.float32),
        "level_magnitude": rng.standard_normal(n).astype(np.float32),
        "level_time_norm": rng.standard_normal(n).astype(np.float32),
        "ar_phi": rng.standard_normal(n).astype(np.float32),
        "seasonal_amplitude": rng.standard_normal(n).astype(np.float32),
        "log_sigma_ratio": rng.standard_normal(n).astype(np.float32),
        "var_shift_time_norm": rng.standard_normal(n).astype(np.float32),
        "spike_present": rng.integers(0, 2, n).astype(np.float32),
        "rw_present": rng.integers(0, 2, n).astype(np.float32),
        "period_idx": rng.integers(0, 8, n).astype(np.int32),
        "spike_patch_idx": rng.integers(-1, 32, n).astype(np.int32),
    }

    argv = [
        "train_probes",
        "--dataset", "synth",
        "--pooling", "mean",
        "--forecast",
        "--model", "moiraie",
        "--output-dir", str(tmp_path),
    ]

    import experiments.mech_interp.block1_probing.train_probes as tp
    with (
        patch("experiments.mech_interp.lib.generate_composite_dataset", return_value=fake_ds),
        patch.object(sys, "argv", argv),
    ):
        tp.main()

    assert (tmp_path / "moiraie.json").exists()
    assert (tmp_path / "metadata.json").exists()
    data = json.loads((tmp_path / "moiraie.json").read_text())
    assert "mean_ctx" in data
    # Forecast features should be present in the results
    assert any(f in data["mean_ctx"] for f in FORECAST_REGRESSION_FEATURES + FORECAST_BINARY_FEATURES)
