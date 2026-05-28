"""Tests for real-data probe training via the universal train_probes runner."""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from experiments.mech_interp.block1_probing.train_probes import (
    REAL_REGRESSION_FEATURES,
    REAL_CLASSIFICATION_FEATURES,
    _run_pooled_probes,
)
from experiments.mech_interp.block1_probing.probe_utils import CONTEXT_PATCHES

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

    train_idx, val_idx = split_dataset(fake_dataset, n_train=144, seed=0)
    features = (
        [(f, "regression") for f in REAL_REGRESSION_FEATURES]
        + [(f, "classification") for f in REAL_CLASSIFICATION_FEATURES]
    )
    feature_data = {f: fake_dataset[f] for f, _ in features if f in fake_dataset}
    results = _run_pooled_probes(
        module=module_e,
        series=fake_dataset["series"],
        features=features,
        feature_data=feature_data,
        train_idx=train_idx,
        val_idx=val_idx,
        poolings=["mean_ctx", CONTEXT_PATCHES - 1],
        batch_size=32,
        device="cpu",
        with_forecast=False,
    )

    expected_poolings = {"mean_ctx", f"pos_{CONTEXT_PATCHES - 1}"}
    assert set(results.keys()) == expected_poolings
    for pkey, pd in results.items():
        assert "dataset_id" in pd
        assert "stl_trend_strength" in pd
        layer_keys = set(pd["stl_trend_strength"].keys())
        assert "-1" in layer_keys
        assert "0" in layer_keys
        assert "1" in layer_keys


def test_nan_labels_do_not_crash(module_e):
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
    features = [
        ("stl_trend_strength", "regression"),
        ("context_std", "regression"),
        ("dataset_id", "classification"),
    ]
    feature_data = {f: ds[f] for f, _ in features if f in ds}
    results = _run_pooled_probes(
        module=module_e,
        series=ds["series"],
        features=features,
        feature_data=feature_data,
        train_idx=train_idx,
        val_idx=val_idx,
        poolings=["mean_ctx"],
        batch_size=32,
        device="cpu",
        with_forecast=False,
    )
    assert "stl_trend_strength" not in results["mean_ctx"]
    assert "context_std" in results["mean_ctx"]


def test_end_to_end_output_files(tmp_path):
    """main() with --dataset real --no-forecast creates expected output files."""
    rng = np.random.default_rng(0)
    N = 9 * 20
    fake_ds = {
        "series": rng.standard_normal((N, 576)).astype(np.float32),
        "dataset_id": np.repeat(np.arange(9), 20).astype(np.int32),
    }
    for label in REAL_REGRESSION_FEATURES:
        fake_ds[label] = rng.standard_normal(N).astype(np.float32)

    argv = [
        "train_probes",
        "--dataset", "real",
        "--pooling", "mean",
        "--no-forecast",
        "--model", "moiraie",
        "--output-dir", str(tmp_path),
    ]

    import experiments.mech_interp.block1_probing.train_probes as tp
    with (
        patch("experiments.mech_interp.lib.real_data.load_gift_subset", return_value=fake_ds),
        patch.object(sys, "argv", argv),
    ):
        tp.main()

    assert (tmp_path / "moiraie.json").exists()
    assert (tmp_path / "metadata.json").exists()
    data = json.loads((tmp_path / "moiraie.json").read_text())
    assert "mean_ctx" in data
