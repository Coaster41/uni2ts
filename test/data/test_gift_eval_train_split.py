"""Leakage guard for the FIX G dataset (gift_eval_train_split).

Re-derives each config's safe boundary from gift_eval.data.Dataset (the same
class the eval harness uses) and checks the stored series against it:

  cut(name) = max over evaluated terms of PL_term * (windows_term + 1 + extra)

Every stored series must be exactly `cut` shorter than its source series, so no
training value can fall inside any term's validation or test region (with an
extra prediction-length margin). Skipped when the dataset is not built.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gift_eval")

from uni2ts.data.builder.lotsa_v1.gift_eval_train import (
    EXTRA_WINDOWS,
    conservative_cut,
    evaluated_terms,
)


def _built_path() -> Path | None:
    try:
        from uni2ts.common.env import env

        p = Path(env.LOTSA_V1_PATH) / "gift_eval_train_split"
        return p if (p / "split_manifest.json").exists() else None
    except Exception:
        return None


@pytest.fixture(scope="module")
def built():
    path = _built_path()
    if path is None:
        pytest.skip("gift_eval_train_split not built")
    os.environ.setdefault(
        "GIFT_EVAL", "/srv/disk00/ctadler/uni2ts/datasets/giftevaltest"
    )
    import json

    from datasets import load_from_disk

    return load_from_disk(str(path)), json.load(open(path / "split_manifest.json"))


# Small configs spanning the shapes: univariate short-only, multivariate
# med-long (term-max boundary), m4 (windows=1), with-missing (NaNs).
SAMPLE_CONFIGS = ["covid_deaths", "ett2/H", "m4_hourly", "kdd_cup_2018_with_missing/D"]


@pytest.mark.slow
@pytest.mark.parametrize("name", SAMPLE_CONFIGS)
def test_stored_series_end_at_safe_boundary(built, name):
    from gift_eval.data import Dataset as GEDataset

    ds, manifest = built
    cut = conservative_cut(name, manifest["extra_windows"])
    assert cut == manifest["configs"][name]["cut"]

    probe = GEDataset(name=name, term="short", to_univariate=False)
    src = GEDataset(name=name, term="short", to_univariate=probe.target_dim != 1)
    src_len = {e["item_id"]: len(e["target"]) for e in src.gluonts_dataset}

    prefix = f"gift_{name.replace('/', '_')}_"
    stored = ds.filter(
        lambda ex: ex["item_id"].startswith(prefix), load_from_cache_file=False
    )
    assert len(stored) > 0, f"no stored rows for {name}"
    for ex in stored:
        sid = ex["item_id"][len(prefix):]
        assert len(ex["target"]) == src_len[sid] - cut, (name, sid)

    # values must match the source prefix (no shifting/reindexing); one series
    # per config suffices since the length check above covers every series
    first = stored[0]
    sid = first["item_id"][len(prefix):]
    src_entry = next(e for e in src.gluonts_dataset if str(e["item_id"]) == sid)
    a = np.asarray(first["target"][-8:], dtype=np.float32)
    b = np.asarray(
        src_entry["target"][len(src_entry["target"]) - cut - 8 :
                            len(src_entry["target"]) - cut],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_cut_exceeds_official_train_boundary(built):
    """Our boundary must be strictly deeper than the benchmark's own
    training_dataset split (windows+1) for every config, for every term."""
    from gift_eval.data import Dataset as GEDataset

    _, manifest = built
    assert manifest["extra_windows"] >= 1
    for name, info in manifest["configs"].items():
        for term in evaluated_terms(name):
            d = GEDataset(name=name, term=term, to_univariate=False)
            official = d.prediction_length * (d.windows + 1)
            assert info["cut"] >= official + d.prediction_length, (name, term)
