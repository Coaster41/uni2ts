"""Tests for train_probes_corruption.py — PR-17.

All tests use tiny in-memory fixture data (n_series=40, d_model=64, layers={-1,0,1}).
No model forward passes or real checkpoints required.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mech_interp.block1_probing.train_probes_corruption import (
    CORRUPTION_NAMES,
    N_CORRUPTIONS,
    run_corruption_probes,
)

CONTEXT_PATCHES = 32
_LAYERS = (-1, 0, 1)


def _make_corruption_acts(
    n_series: int = 40,
    d_model: int = 64,
    layers: tuple[int, ...] = _LAYERS,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build fake activations where each corruption class has a well-separated centroid."""
    rng = np.random.default_rng(seed)
    # Large separation (4σ), small noise (0.3σ) → probe should achieve near-perfect accuracy
    centroids = rng.standard_normal((N_CORRUPTIONS, d_model)) * 4.0
    out = {}
    for layer in layers:
        for pool in ("mean_ctx", "last_ctx"):
            for c_idx, c_name in enumerate(CORRUPTION_NAMES):
                acts = centroids[c_idx] + rng.standard_normal((n_series, d_model)) * 0.3
                out[f"{c_name}_{pool}_layer_{layer}"] = acts.astype(np.float32)
        for c_idx, c_name in enumerate(CORRUPTION_NAMES):
            # per_patch: centroid broadcast over patch axis
            acts = (
                centroids[c_idx][None, None, :]
                + rng.standard_normal((n_series, CONTEXT_PATCHES, d_model)) * 0.3
            )
            out[f"{c_name}_per_patch_layer_{layer}"] = acts.astype(np.float32)
    return out


@pytest.fixture
def corruption_acts() -> dict[str, np.ndarray]:
    return _make_corruption_acts(n_series=40, d_model=64)


@pytest.fixture
def split(corruption_acts) -> tuple[np.ndarray, np.ndarray]:
    n = 40
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    return perm[:n_train], perm[n_train:]


def test_output_structure(corruption_acts, split):
    train_idx, val_idx = split
    results = run_corruption_probes(corruption_acts, train_idx, val_idx)

    assert set(results.keys()) == {"mean_ctx", "last_ctx", "per_patch"}

    for pooling in ("mean_ctx", "last_ctx"):
        for feat in ("corruption_id", "is_corrupted"):
            assert feat in results[pooling], f"[{pooling}] missing feature '{feat}'"
            layer_scores = results[pooling][feat]
            assert set(layer_scores.keys()) == set(_LAYERS), (
                f"[{pooling}][{feat}] unexpected layer keys: {set(layer_scores.keys())}"
            )
            for layer_idx, score in layer_scores.items():
                assert isinstance(score, float), f"[{pooling}][{feat}][{layer_idx}] not float"
                assert np.isfinite(score), f"[{pooling}][{feat}][{layer_idx}] non-finite: {score}"

    for feat in ("corruption_id", "is_corrupted"):
        assert feat in results["per_patch"]
        layer_scores = results["per_patch"][feat]
        assert set(layer_scores.keys()) == set(_LAYERS)
        for layer_idx, scores in layer_scores.items():
            assert isinstance(scores, list), (
                f"[per_patch][{feat}][{layer_idx}] expected list, got {type(scores)}"
            )
            assert len(scores) == CONTEXT_PATCHES, (
                f"[per_patch][{feat}][{layer_idx}] expected length {CONTEXT_PATCHES}, got {len(scores)}"
            )


def test_8way_accuracy_above_chance(corruption_acts, split):
    """8-way corruption-ID accuracy must exceed chance (1/8 = 0.125) for all layers."""
    train_idx, val_idx = split
    results = run_corruption_probes(corruption_acts, train_idx, val_idx)
    chance = 1.0 / N_CORRUPTIONS  # 0.125

    for layer_idx in _LAYERS:
        acc = results["mean_ctx"]["corruption_id"][layer_idx]
        assert acc > chance, (
            f"[mean_ctx][corruption_id][layer {layer_idx}] accuracy {acc:.4f} <= chance {chance:.4f}"
        )


def test_binary_auroc_above_chance(corruption_acts, split):
    """Binary is-corrupted AUROC must exceed 0.5 for all layers."""
    train_idx, val_idx = split
    results = run_corruption_probes(corruption_acts, train_idx, val_idx)

    for layer_idx in _LAYERS:
        auroc = results["mean_ctx"]["is_corrupted"][layer_idx]
        assert auroc > 0.5, (
            f"[mean_ctx][is_corrupted][layer {layer_idx}] AUROC {auroc:.4f} <= chance 0.5"
        )
