import numpy as np
import torch

from experiments.mech_interp.lib import compute_patch_features
from experiments.mech_interp.lib.synthetic import generate_composite_dataset
from experiments.mech_interp.block1_probing.probe_utils import (
    CONTEXT_PATCHES,
    PATCH_SIZE,
    batched_ridge_per_patch_local,
)


def test_compute_patch_features_shapes_and_groups():
    ds = generate_composite_dataset(n=64, seed=0)
    reg, binf = compute_patch_features(ds, CONTEXT_PATCHES, PATCH_SIZE)
    n = len(ds["series"])

    # Cross-patch features have all-NaN boundary columns by design (see dedicated test).
    cross_patch = {
        "patch_boundary_jump_left", "patch_boundary_jump_right", "patch_local_curvature_3",
    }
    for d in (reg, binf):
        for name, arr in d.items():
            assert arr.shape == (n, CONTEXT_PATCHES), name
            if name not in cross_patch:
                assert np.isfinite(arr).all(), name

    # local stat sanity: patch_mean matches a manual reshape mean
    ctx = ds["series"][:, : CONTEXT_PATCHES * PATCH_SIZE].reshape(n, CONTEXT_PATCHES, PATCH_SIZE)
    np.testing.assert_allclose(reg["patch_mean"], ctx.mean(axis=2), rtol=1e-5, atol=1e-5)

    # Group C present for synth + one-hot matches spike_patch_idx
    assert "patch_has_spike" in binf
    sp = ds["spike_patch_idx"].astype(int)
    for i in np.where(sp >= 0)[0][:20]:
        row = binf["patch_has_spike"][i]
        assert row.sum() == 1.0 and row[sp[i]] == 1.0


def test_group_c_absent_without_metadata():
    ds = generate_composite_dataset(n=16, seed=1)
    bare = {"series": ds["series"]}
    reg, binf = compute_patch_features(bare, CONTEXT_PATCHES, PATCH_SIZE)
    assert "patch_has_spike" not in binf          # ground-truth flags skipped
    assert "patch_is_level_outlier" in binf       # data-driven flags still present
    assert len(reg) == 16


def test_batched_ridge_per_patch_local_recovers_planted_signal():
    rng = np.random.default_rng(0)
    B, n_tr, n_va, d = 4, 200, 80, 8
    w = rng.standard_normal((B, d))
    X_tr = rng.standard_normal((B, n_tr, d))
    X_va = rng.standard_normal((B, n_va, d))
    # per-patch label = X @ w_patch  (each patch has its own linear map)
    Y_tr = np.einsum("bnd,bd->bn", X_tr, w)[:, :, None]
    Y_va = np.einsum("bnd,bd->bn", X_va, w)[:, :, None]

    r2 = batched_ridge_per_patch_local(
        torch.from_numpy(X_tr).float(), torch.from_numpy(X_va).float(),
        torch.from_numpy(Y_tr).float(), torch.from_numpy(Y_va).float(),
    )
    assert r2.shape == (B, 1)
    assert (r2[:, 0] > 0.99).all()


def _ds_from_patches(patches: np.ndarray) -> dict:
    """Wrap a [n, C, P] patch array as a dataset dict (raw-series path)."""
    n, C, P = patches.shape
    return {"series": patches.reshape(n, C * P).astype(np.float64)}


def test_cross_patch_features_shapes_and_boundary_nan():
    ds = generate_composite_dataset(n=32, seed=2)
    reg, _ = compute_patch_features(ds, CONTEXT_PATCHES, PATCH_SIZE)
    n = len(ds["series"])

    new_feats = (
        "patch_curvature_coef", "patch_spearman_trend", "patch_argmax_pos",
        "patch_n_sign_changes", "patch_boundary_jump_left",
        "patch_boundary_jump_right", "patch_local_curvature_3",
    )
    for name in new_feats:
        arr = reg[name]
        assert arr.shape == (n, CONTEXT_PATCHES), name
        assert arr.dtype == np.float64, name

    jl, jr, c3 = (reg["patch_boundary_jump_left"], reg["patch_boundary_jump_right"],
                  reg["patch_local_curvature_3"])
    assert np.isnan(jl[:, 0]).all() and np.isfinite(jl[:, 1:]).all()
    assert np.isnan(jr[:, -1]).all() and np.isfinite(jr[:, :-1]).all()
    assert np.isnan(c3[:, [0, -1]]).all() and np.isfinite(c3[:, 1:-1]).all()


def test_within_patch_features_correctness():
    n, C, P = 4, CONTEXT_PATCHES, PATCH_SIZE

    # (a) per-patch positive linear ramp -> spearman_trend ~ +1, sign matches slope
    ramp = np.broadcast_to(np.arange(P, dtype=np.float64), (n, C, P)).copy()
    reg, _ = compute_patch_features(_ds_from_patches(ramp), C, P)
    assert np.allclose(reg["patch_spearman_trend"], 1.0, atol=1e-6)
    assert np.all(np.sign(reg["patch_spearman_trend"]) == np.sign(reg["patch_slope"]))
    reg_neg, _ = compute_patch_features(_ds_from_patches(-ramp), C, P)
    assert np.allclose(reg_neg["patch_spearman_trend"], -1.0, atol=1e-6)

    # (b) delta at a known index -> argmax_pos = idx / (P-1)
    idx = 5
    delta = np.zeros((n, C, P))
    delta[:, :, idx] = 1.0
    reg_d, _ = compute_patch_features(_ds_from_patches(delta), C, P)
    assert np.allclose(reg_d["patch_argmax_pos"], idx / (P - 1))

    # (c) alternating +/-1 -> exactly one sign change per step
    alt = np.broadcast_to(
        np.where(np.arange(P) % 2 == 0, 1.0, -1.0), (n, C, P)
    ).copy()
    reg_a, _ = compute_patch_features(_ds_from_patches(alt), C, P)
    assert np.allclose(reg_a["patch_n_sign_changes"], 1.0, atol=1e-6)

    # (d) pure quadratic x = a*u^2 (u = linspace(-1, 1, P)) -> curvature_coef = a
    a_true = 0.7
    u = np.linspace(-1.0, 1.0, P)
    quad = np.broadcast_to(a_true * u**2, (n, C, P)).copy()
    reg_q, _ = compute_patch_features(_ds_from_patches(quad), C, P)
    assert np.allclose(reg_q["patch_curvature_coef"], a_true, atol=1e-6)


def test_cross_patch_features_match_exact_differences():
    n, C, P = 3, CONTEXT_PATCHES, PATCH_SIZE
    rng = np.random.default_rng(7)
    m = rng.standard_normal((n, C))                              # planted per-patch means
    patches = np.broadcast_to(m[:, :, None], (n, C, P)).copy()   # piecewise-constant
    reg, _ = compute_patch_features(_ds_from_patches(patches), C, P)

    np.testing.assert_allclose(reg["patch_mean"], m, atol=1e-10)
    np.testing.assert_allclose(
        reg["patch_boundary_jump_left"][:, 1:], m[:, 1:] - m[:, :-1], atol=1e-10)
    np.testing.assert_allclose(
        reg["patch_boundary_jump_right"][:, :-1], m[:, 1:] - m[:, :-1], atol=1e-10)
    np.testing.assert_allclose(
        reg["patch_local_curvature_3"][:, 1:-1],
        m[:, :-2] - 2 * m[:, 1:-1] + m[:, 2:], atol=1e-10)


def test_batched_ridge_per_patch_local_skips_boundary_nan():
    rng = np.random.default_rng(1)
    B, n_tr, n_va, d = CONTEXT_PATCHES, 200, 80, 8
    w = rng.standard_normal((B, d))
    X_tr = rng.standard_normal((B, n_tr, d))
    X_va = rng.standard_normal((B, n_va, d))
    # interior positions linearly encode the (jump_right-like) target
    Y_tr = np.einsum("bnd,bd->bn", X_tr, w)[:, :, None]
    Y_va = np.einsum("bnd,bd->bn", X_va, w)[:, :, None]
    # boundary position K-1 is all-NaN (every sample), mirroring patch_boundary_jump_right
    Y_tr[-1] = np.nan
    Y_va[-1] = np.nan

    r2 = batched_ridge_per_patch_local(
        torch.from_numpy(X_tr).float(), torch.from_numpy(X_va).float(),
        torch.from_numpy(Y_tr).float(), torch.from_numpy(Y_va).float(),
    )
    assert r2.shape == (B, 1)
    assert (r2[:-1, 0] > 0.99).all()       # interior positions recovered
    assert torch.isnan(r2[-1, 0])          # boundary position emits NaN
