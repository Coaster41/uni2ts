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

    for d in (reg, binf):
        for name, arr in d.items():
            assert arr.shape == (n, CONTEXT_PATCHES), name
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
    assert len(reg) == 9


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
