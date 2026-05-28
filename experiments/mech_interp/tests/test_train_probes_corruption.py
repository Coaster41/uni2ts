"""Tests for corruption probe training via probe_utils and train_probes.

Equivalent coverage to the old train_probes_corruption tests, but using
the consolidated API: fit_probe from probe_utils, CORRUPTION_NAMES/N_CORRUPTIONS
from train_probes.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mech_interp.block1_probing.train_probes import (
    CORRUPTION_NAMES,
    N_CORRUPTIONS,
)
from experiments.mech_interp.block1_probing.probe_utils import fit_probe

CONTEXT_PATCHES = 32
_LAYERS = (-1, 0, 1)


def _make_stacked_acts(
    n_series: int = 40,
    d_model: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stacked train/val activations for all corruption variants.

    Returns X_tr [N_CORRUPTIONS*n_train, d], X_va [N_CORRUPTIONS*n_val, d],
            y_id_tr, y_id_va.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((N_CORRUPTIONS, d_model)) * 4.0
    perm = rng.permutation(n_series)
    n_train = int(n_series * 0.8)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_tr_parts, X_va_parts = [], []
    y_id_tr, y_id_va = [], []

    for c_idx in range(N_CORRUPTIONS):
        acts = centroids[c_idx] + rng.standard_normal((n_series, d_model)) * 0.3
        X_tr_parts.append(acts[train_idx].astype(np.float32))
        X_va_parts.append(acts[val_idx].astype(np.float32))
        y_id_tr.append(np.full(n_train, c_idx, dtype=np.int32))
        y_id_va.append(np.full(n_series - n_train, c_idx, dtype=np.int32))

    return (
        np.concatenate(X_tr_parts),
        np.concatenate(X_va_parts),
        np.concatenate(y_id_tr),
        np.concatenate(y_id_va),
    )


def test_corruption_id_above_chance():
    """8-way corruption-ID accuracy must exceed chance (1/8) for well-separated activations."""
    X_tr, X_va, y_tr, y_va = _make_stacked_acts(n_series=40, d_model=64)
    acc = fit_probe(X_tr, X_va, y_tr, y_va, "classification")
    chance = 1.0 / N_CORRUPTIONS
    assert acc > chance, f"accuracy {acc:.4f} <= chance {chance:.4f}"
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_is_corrupted_auroc_above_chance():
    """Binary is-corrupted AUROC must exceed 0.5 for well-separated activations."""
    X_tr, X_va, y_tr, y_va = _make_stacked_acts(n_series=40, d_model=64)
    y_bin_tr = (y_tr > 0).astype(np.int32)  # 0 = clean, 1 = any corruption
    y_bin_va = (y_va > 0).astype(np.int32)
    auroc = fit_probe(X_tr, X_va, y_bin_tr, y_bin_va, "binary")
    assert auroc > 0.5, f"AUROC {auroc:.4f} <= 0.5"


def test_corruption_names_count():
    assert len(CORRUPTION_NAMES) == N_CORRUPTIONS
    assert N_CORRUPTIONS == 8
    assert "clean" in CORRUPTION_NAMES
    assert "no_trend" in CORRUPTION_NAMES
    assert "noise" in CORRUPTION_NAMES


def test_fit_probe_classification_finite():
    """fit_probe with classification type returns finite float."""
    import math
    X_tr, X_va, y_tr, y_va = _make_stacked_acts(n_series=20, d_model=16, seed=99)
    score = fit_probe(X_tr, X_va, y_tr, y_va, "classification")
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
