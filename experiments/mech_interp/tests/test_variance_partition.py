"""Tests for variance_partition.py — PR-15.

All tests use tiny in-memory fixture data; no model checkpoints required.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_runner_output(
    n: int = 80,
    d_model: int = 32,
    seed: int = 0,
) -> dict[str, np.ndarray]:
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
def tiny_ro():
    return _make_runner_output(n=80, d_model=32)


@pytest.fixture
def tiny_split(tiny_ro):
    n = len(tiny_ro["forecast_quantiles"])
    rng = np.random.default_rng(7)
    idx = rng.permutation(n)
    return idx[:64], idx[64:]   # 64 train / 16 val


# ---------------------------------------------------------------------------
# Test 1: surface feature shapes and finiteness
# ---------------------------------------------------------------------------

def test_compute_surface_features_shapes():
    from experiments.mech_interp.block1_probing.variance_partition import (
        compute_surface_features,
        N_BASE_SURFACE_FEATURES,
    )

    rng = np.random.default_rng(0)
    context = rng.standard_normal((20, 512)).astype(np.float32)

    feats = compute_surface_features(context)
    assert feats.shape == (20, N_BASE_SURFACE_FEATURES), f"Base shape mismatch: {feats.shape}"
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats)), "Surface features contain non-finite values"

    # With dataset_ids one-hot
    ids = np.zeros(20, dtype=np.int32)
    feats_ext = compute_surface_features(context, dataset_ids=ids, n_datasets=3)
    assert feats_ext.shape == (20, N_BASE_SURFACE_FEATURES + 3)
    assert np.all(np.isfinite(feats_ext))


# ---------------------------------------------------------------------------
# Test 2: output structure
# ---------------------------------------------------------------------------

def test_run_variance_partition_structure(tiny_ro, tiny_split):
    from experiments.mech_interp.block1_probing.variance_partition import (
        run_variance_partition,
        HEADLINE_TARGETS,
    )

    train_idx, val_idx = tiny_split
    results = run_variance_partition(tiny_ro, train_idx, val_idx)

    assert set(results.keys()) == set(HEADLINE_TARGETS), (
        f"Expected {set(HEADLINE_TARGETS)}, got {set(results.keys())}"
    )
    for target in HEADLINE_TARGETS:
        assert set(results[target].keys()) == {"mean_ctx", "last_ctx"}, (
            f"[{target}] pooling keys wrong: {set(results[target].keys())}"
        )
        for pooling in ("mean_ctx", "last_ctx"):
            layer_dict = results[target][pooling]
            assert set(layer_dict.keys()) == {-1, 0, 1}, (
                f"[{target}][{pooling}] layer keys: {set(layer_dict.keys())}"
            )
            for layer_idx, scores in layer_dict.items():
                assert set(scores.keys()) == {"surface", "neural", "combined", "delta"}, (
                    f"[{target}][{pooling}][{layer_idx}] score keys: {set(scores.keys())}"
                )
                for k, v in scores.items():
                    assert np.isfinite(v), (
                        f"[{target}][{pooling}][{layer_idx}][{k}] non-finite: {v}"
                    )


# ---------------------------------------------------------------------------
# Test 3: delta ≈ 0 when neural features are random
# ---------------------------------------------------------------------------

def test_delta_near_zero_when_neural_random():
    """When neural activations are pure noise, combined probe ≈ surface probe."""
    from experiments.mech_interp.block1_probing.variance_partition import run_variance_partition

    rng = np.random.default_rng(42)
    n = 200
    # Context that has real structure so surface probe captures something
    t = np.linspace(0, 4 * np.pi, 512)
    context = (
        np.sin(t)[None, :] + 0.3 * rng.standard_normal((n, 512))
    ).astype(np.float32)

    fq = np.sort(rng.standard_normal((n, 9, 64)).astype(np.float32), axis=1)
    tgt = rng.standard_normal((n, 64)).astype(np.float32)

    ro = {
        "forecast_quantiles": fq,
        "target": tgt,
        "context": context,
    }
    # Pure random neural features — should add no signal
    for k in (-1, 0, 1):
        ro[f"activations_mean_ctx_layer_{k}"] = rng.standard_normal((n, 32)).astype(np.float32)
        ro[f"activations_last_ctx_layer_{k}"] = rng.standard_normal((n, 32)).astype(np.float32)

    idx = rng.permutation(n)
    train_idx, val_idx = idx[:160], idx[160:]
    results = run_variance_partition(ro, train_idx, val_idx)

    for target in results:
        for pooling in results[target]:
            for layer_idx, scores in results[target][pooling].items():
                assert abs(scores["delta"]) < 0.35, (
                    f"[{target}][{pooling}][{layer_idx}] delta too large with random neural: "
                    f"{scores['delta']:.3f}"
                )


# ---------------------------------------------------------------------------
# Test 4: combined R² beats surface when neural features are informative
# ---------------------------------------------------------------------------

def test_combined_r2_better_when_neural_informative():
    """When neural activations correlate with the target, combined > surface."""
    from experiments.mech_interp.block1_probing.variance_partition import (
        run_variance_partition,
        compute_surface_features,
    )

    rng = np.random.default_rng(99)
    n = 200
    # Random (near-uninformative) context → low surface R²
    context = rng.standard_normal((n, 512)).astype(np.float32)
    fq = np.sort(rng.standard_normal((n, 9, 64)).astype(np.float32), axis=1)
    tgt = rng.standard_normal((n, 64)).astype(np.float32)

    ro = {
        "forecast_quantiles": fq,
        "target": tgt,
        "context": context,
    }

    # Compute targets to know what "fc_std" will be so we can build informative neural acts.
    from experiments.mech_interp.block1_probing.forecast_runner import compute_forecast_targets
    targets = compute_forecast_targets(ro, ctx_period=24)
    # Use fc_std as the informative signal: neural layer_0 ≈ fc_std broadcast to d_model
    fc_std_signal = targets["fc_std"].astype(np.float32)  # [n]
    d = 32
    # Neural acts = signal + small noise, so they predict fc_std well
    noise_scale = 0.05 * float(fc_std_signal.std() + 1e-6)
    informative_acts = (
        fc_std_signal[:, None] * np.ones((1, d), dtype=np.float32)
        + rng.standard_normal((n, d)).astype(np.float32) * noise_scale
    )

    for k in (-1, 0, 1):
        ro[f"activations_mean_ctx_layer_{k}"] = informative_acts.copy()
        ro[f"activations_last_ctx_layer_{k}"] = informative_acts.copy()

    idx = rng.permutation(n)
    train_idx, val_idx = idx[:160], idx[160:]
    results = run_variance_partition(
        ro, train_idx, val_idx, headline_targets=("fc_std",)
    )

    # For mean_ctx layer_0, combined should significantly outperform surface
    scores = results["fc_std"]["mean_ctx"][0]
    assert scores["combined"] > scores["surface"] + 0.2, (
        f"Expected combined >> surface, got combined={scores['combined']:.3f}, "
        f"surface={scores['surface']:.3f}"
    )
