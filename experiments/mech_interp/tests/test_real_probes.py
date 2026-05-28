from __future__ import annotations

import numpy as np
import pytest

from experiments.mech_interp.block1_probing.train_probes_real import REAL_REGRESSION_FEATURES

_TINY = dict(
    d_model=64, d_ff=128, num_layers=2, patch_size=16,
    max_seq_len=64, attn_dropout_p=0.0, dropout_p=0.0,
)


@pytest.fixture
def module_e():
    from uni2ts.model.moiraie.module import MoiraieModule
    return MoiraieModule(**_TINY, num_predict_token=1).eval()


@pytest.fixture
def fake_dataset():
    rng = np.random.default_rng(0)
    N = 9 * 20
    ds = {
        "series": rng.standard_normal((N, 576)).astype(np.float32),
        "dataset_id": np.repeat(np.arange(9), 20).astype(np.int32),
    }
    for label in REAL_REGRESSION_FEATURES:
        vals = rng.standard_normal(N).astype(np.float32)
        if label in ("stl_trend_strength", "stl_seasonal_strength", "context_acf_lag1"):
            vals[::20] = np.nan
        ds[label] = vals
    return ds


def test_run_real_probes_schema(module_e, fake_dataset):
    from experiments.mech_interp.lib.synthetic import split_dataset
    import experiments.mech_interp.block1_probing.train_probes_real as tpr

    train_idx, val_idx = split_dataset(fake_dataset, n_train=144, seed=0)
    results = tpr.run_real_probes_for_model(
        module_e, fake_dataset, train_idx, val_idx, batch_size=32,
    )
    assert set(results.keys()) == {"mean_ctx", "last_ctx"}
    for pooling, pd in results.items():
        assert "dataset_id" in pd
        assert "stl_trend_strength" in pd
        layer_keys = set(pd["stl_trend_strength"].keys())
        assert -1 in layer_keys
        assert 0 in layer_keys
        assert 1 in layer_keys


def test_nan_labels_do_not_crash(module_e):
    import experiments.mech_interp.block1_probing.train_probes_real as tpr
    from experiments.mech_interp.lib.synthetic import split_dataset

    rng = np.random.default_rng(1)
    N = 9 * 20
    ds = {
        "series": rng.standard_normal((N, 576)).astype(np.float32),
        "dataset_id": np.repeat(np.arange(9), 20).astype(np.int32),
        "stl_trend_strength": np.full(N, np.nan, dtype=np.float32),
        "context_std": rng.standard_normal(N).astype(np.float32),
    }
    train_idx, val_idx = split_dataset(ds, n_train=144, seed=0)
    results = tpr.run_real_probes_for_model(module_e, ds, train_idx, val_idx)
    assert "stl_trend_strength" not in results["mean_ctx"]
    assert "context_std" in results["mean_ctx"]


def test_end_to_end_output_files(tmp_path, monkeypatch):
    import experiments.mech_interp.block1_probing.train_probes_real as tpr

    def _fake_load(**kwargs):
        rng = np.random.default_rng(0)
        N = 9 * 20
        ds = {
            "series": rng.standard_normal((N, 576)).astype(np.float32),
            "dataset_id": np.repeat(np.arange(9), 20).astype(np.int32),
        }
        for label in tpr.REAL_REGRESSION_FEATURES:
            ds[label] = rng.standard_normal(N).astype(np.float32)
        return ds

    monkeypatch.setattr(tpr, "load_gift_subset", _fake_load)
    import sys
    argv_backup = sys.argv[:]
    sys.argv = ["train_probes_real", "--output-dir", str(tmp_path)]
    try:
        tpr.main()
    finally:
        sys.argv = argv_backup

    assert (tmp_path / "moiraie.json").exists()
    assert (tmp_path / "moiraic.json").exists()
    assert (tmp_path / "metadata.json").exists()
    import json
    data = json.loads((tmp_path / "moiraie.json").read_text())
    assert "mean_ctx" in data
    assert "last_ctx" in data
