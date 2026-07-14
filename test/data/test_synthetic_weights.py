#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Validate the synthetic-data sampling weights used by the Moirai-2.0-style
pretraining mixture (``cli/conf/pretrain/data/gift_eval_pretrain_moirai2.yaml``).

How the weights turn into a sampling fraction
---------------------------------------------
Each dataset is wrapped in a ``TimeSeriesDataset`` whose ``dataset_weight`` (taken
from the config ``weight_map``) multiplies its reported length::

    len(ds) == ceil(num_ts * dataset_weight)            # dataset.py::__len__

All datasets are concatenated into one ``torch.utils.data.ConcatDataset`` and the
pretraining loader draws indices with a plain uniform ``RandomSampler``
(``shuffle: true`` in ``cli/conf/pretrain/default.yaml`` with no custom sampler).
A uniform draw over ``[0, len(concat))`` therefore lands in sub-dataset ``d`` with
probability ``len(ds_d) / sum_i len(ds_i)``. Hence the *effective sampling share*
of a dataset is::

    share_d = (num_ts_d * weight_d) / sum_i (num_ts_i * weight_i)

These tests pin that relationship down so a future weight edit can't silently
change the real-vs-synthetic mix.

Target mixture (committed weights): ~20% synthetic / ~80% real at a 9:1
TSMixup:KernelSynth window ratio. The rationale, from the literature search:

- Moirai 2.0 (arXiv:2511.11698) reports a ~86%-synthetic *corpus* (Chronos-Mixup
  30M + KernelSynth 1M of 36M series) but does NOT disclose its training SAMPLING
  weights -- corpus counts != how often a source is drawn.
- That ~86% is mostly TSMixup, which is convex combinations of REAL series
  (augmented-real), not pure synthetic. Only KernelSynth (~3%) is pure GP synthetic.
- Chronos (arXiv:2403.07815 sec 5.6): pure synthetic (KernelSynth) is best at ~10%
  and degrades beyond; Chronos trained TSMix:KernelSynth at 9:1.
- Treating BOTH Chronos streams as synthetic, ~20% total keeps real data clearly
  dominant while keeping KernelSynth (~2%) under the 10% pure-synthetic ceiling.

For reference the band assertions also cover the historical extremes the project
passed through: ~8% synthetic (old gift_eval_pretrain_weighted) and ~88%
synthetic (an earlier rev of moirai2.yaml).
"""

import math
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import ConcatDataset

from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset

# ---------------------------------------------------------------------------
# Reference counts and weights (kept here so the test is the single source of
# truth for "weight X => share Y"). num_ts are the real on-disk row counts of the
# Chronos builders; REAL_EFFECTIVE_POOL is sum_i num_ts_i*weight_i over every
# *non*-Chronos dataset in gift_eval_pretrain_moirai2.yaml (each real dataset is
# pre-weighted to contribute an equal ~126.6k windows -> ~2.77M total).
# ---------------------------------------------------------------------------
NUM_TS_MIXUP = 10_000_000  # chronos_tsmixup_10m
NUM_TS_KSYNTH = 1_000_000  # chronos_kernel_synth_1m
REAL_EFFECTIVE_POOL = 2_770_242

# Weights currently committed in gift_eval_pretrain_moirai2.yaml. Target is
# ~20% synthetic / ~80% real at a 9:1 TSMixup:KernelSynth window ratio.
CURRENT_WEIGHT_MIXUP = 0.06233
CURRENT_WEIGHT_KSYNTH = 0.06926

# Historical settings, asserted below so the documented comparison stays honest:
#   old gift_eval_pretrain_weighted.yaml -> ~8% synthetic
#   an earlier rev of moirai2.yaml       -> ~88% synthetic
OLD_WEIGHTED_MIXUP = 0.023
OLD_WEIGHTED_KSYNTH = 0.0069


def effective_sampling_shares(
    num_ts: dict[str, float], weights: dict[str, float]
) -> dict[str, float]:
    """Replicate ``TimeSeriesDataset.__len__`` proportions analytically.

    Returns ``{name: fraction}`` where ``fraction`` is the probability a uniform
    draw over the concatenated dataset lands in ``name``.
    """
    eff = {k: math.ceil(num_ts[k] * weights.get(k, 1.0)) for k in num_ts}
    total = sum(eff.values())
    return {k: v / total for k, v in eff.items()}


# ===========================================================================
# 1. The length formula every share calculation relies on.
# ===========================================================================
@pytest.mark.parametrize(
    "hf_dataset_path",
    [(3, 2, 1, (10, 10))],  # num_examples, target_dim, past_feat_real_dim, length
    indirect=True,
)
@pytest.mark.parametrize("weight", [0.6, 1.0, 2.0, 33.99])
def test_len_is_ceil_num_ts_times_weight(hf_dataset_path, weight):
    """``len(ds) == ceil(num_ts * weight)`` -- the contract the share math uses."""
    from datasets import load_from_disk
    from uni2ts.data.indexer import HuggingFaceDatasetIndexer
    from uni2ts.transform import Identity

    dataset_path, num_examples, *_ = hf_dataset_path
    indexer = HuggingFaceDatasetIndexer(load_from_disk(dataset_path))
    ds = TimeSeriesDataset(
        indexer,
        transform=Identity(),
        sample_time_series=SampleTimeSeriesType.PROPORTIONAL,
        dataset_weight=weight,
    )
    assert ds.num_ts == num_examples
    assert len(ds) == int(np.ceil(num_examples * weight))


# ===========================================================================
# 2. A uniform sampler over a ConcatDataset visits each sub-dataset in
#    proportion to its (weighted) length. This is the actual sampling mechanism
#    used in pretraining -- we verify the empirical frequencies match the
#    closed-form share to within Monte-Carlo tolerance.
# ===========================================================================
class _SizedStub:
    """Stands in for a weighted TimeSeriesDataset.

    Cross-dataset sampling only depends on ``__len__`` (RandomSampler indexes the
    ConcatDataset), so a stub with the right length is faithful to the mechanism
    while avoiding building multi-million-row HF datasets in a unit test.
    """

    def __init__(self, num_ts: int, weight: float):
        self._len = int(np.ceil(num_ts * weight))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):  # pragma: no cover - never sampled for content
        return idx


def test_uniform_sampler_visits_datasets_proportional_to_weight():
    num_ts = {"mixup": 1_000_000, "ksynth": 1_000_000, "real": 1_000_000}
    weights = {"mixup": 2.0, "ksynth": 0.6, "real": 1.0}

    names = list(num_ts)
    concat = ConcatDataset([_SizedStub(num_ts[n], weights[n]) for n in names])
    boundaries = np.asarray(concat.cumulative_sizes)  # right edges per sub-dataset

    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(concat), size=400_000)
    sub_ids = np.searchsorted(boundaries, draws, side="right")
    empirical = {n: float(np.mean(sub_ids == i)) for i, n in enumerate(names)}

    expected = effective_sampling_shares(num_ts, weights)
    for n in names:
        assert empirical[n] == pytest.approx(
            expected[n], abs=0.005
        ), f"{n}: empirical {empirical[n]:.4f} vs expected {expected[n]:.4f}"


# ===========================================================================
# 3. Pin the concrete "weight -> share" answers for the real corpus scale.
#    This is the "0.06233 means ~20% synthetic" check the mixture is calibrated to,
#    plus the historical reference points (~8% old, ~88%/~30% earlier revs).
# ===========================================================================
def _corpus_shares(weight_mixup: float, weight_ksynth: float) -> dict[str, float]:
    num_ts = {
        "mixup": NUM_TS_MIXUP,
        "ksynth": NUM_TS_KSYNTH,
        "real": REAL_EFFECTIVE_POOL,  # already-weighted real pool, weight 1.0
    }
    weights = {"mixup": weight_mixup, "ksynth": weight_ksynth, "real": 1.0}
    s = effective_sampling_shares(num_ts, weights)
    return {
        "mixup": s["mixup"],
        "ksynth": s["ksynth"],
        "real": s["real"],
        "synthetic": s["mixup"] + s["ksynth"],
        "mixup_to_ksynth": (
            s["mixup"] / s["ksynth"] if s["ksynth"] > 0 else float("inf")
        ),
    }


def test_current_weights_match_documented_shares():
    """Committed tsmixup=0.06233, ksynth=0.06926 -> ~18% mixup, ~20% synthetic,
    ~80% real, 9:1 mixup:ksynth."""
    s = _corpus_shares(CURRENT_WEIGHT_MIXUP, CURRENT_WEIGHT_KSYNTH)
    assert s["mixup"] == pytest.approx(0.180, abs=0.01)
    assert s["ksynth"] == pytest.approx(0.020, abs=0.005)
    assert s["synthetic"] == pytest.approx(0.200, abs=0.01)
    assert s["real"] == pytest.approx(0.800, abs=0.01)
    assert s["mixup_to_ksynth"] == pytest.approx(9.0, rel=0.05)


def test_old_weighted_config_was_low_synthetic():
    """The prior gift_eval_pretrain_weighted weights (0.023 / 0.0069) gave only
    ~8% synthetic -- already within Chronos's evidence-supported range, and the
    point of comparison showing the 88% earlier rev was the outlier."""
    s = _corpus_shares(OLD_WEIGHTED_MIXUP, OLD_WEIGHTED_KSYNTH)
    assert s["synthetic"] == pytest.approx(0.079, abs=0.01)
    assert s["real"] == pytest.approx(0.921, abs=0.01)


def test_moirai2_corpus_replication_weights():
    """For reference: tsmixup=1.5, ksynth=0.5 reproduce Moirai 2.0's *corpus*
    proportions (~85% synthetic, ~15% real) -- not the committed target, kept to
    document what 'matching the paper's composition' would require."""
    s = _corpus_shares(1.5, 0.5)
    assert s["synthetic"] == pytest.approx(0.85, abs=0.02)
    assert s["real"] == pytest.approx(0.152, abs=0.02)


@pytest.mark.parametrize(
    "weight_mixup, weight_ksynth, lo, hi",
    [
        (0.0, 0.0, 0.0, 0.0),  # no synthetic
        (0.023, 0.0069, 0.05, 0.10),  # old gift_eval_pretrain_weighted (~8%)
        (0.06233, 0.06926, 0.17, 0.23),  # committed target (~20%)
        (0.107, 0.119, 0.27, 0.33),  # prior rev (~30%)
        (1.5, 0.5, 0.82, 0.88),  # Moirai 2.0 corpus replication (~85%)
        (10.0, 1.0, 0.95, 1.0),  # heavy synthetic outlier
    ],
)
def test_synthetic_share_is_monotonic_and_banded(weight_mixup, weight_ksynth, lo, hi):
    s = _corpus_shares(weight_mixup, weight_ksynth)
    assert lo <= s["synthetic"] <= hi


# ===========================================================================
# 4. Integration: the committed config file actually produces the ~20%-synthetic
#    target mixture given the real on-disk dataset sizes.
#    Marked slow + skipped when the LOTSA datasets are not present.
# ===========================================================================
CONFIG_DIR = Path("cli/conf/pretrain/data")
CONFIG_NAME = "gift_eval_pretrain_moirai2"


def _lotsa_path() -> Path | None:
    try:
        from uni2ts.common.env import env

        p = Path(env.LOTSA_V1_PATH)
        return p if p.exists() else None
    except Exception:
        return None


def _num_examples_on_disk(root: Path, name: str) -> int | None:
    import json

    info = root / name / "dataset_info.json"
    if not info.exists():
        return None
    with open(info) as f:
        return json.load(f)["splits"]["train"]["num_examples"]


@pytest.mark.slow
def test_committed_config_yields_moirai2_like_mixture():
    root = _lotsa_path()
    if root is None or _num_examples_on_disk(root, "chronos_tsmixup_10m") is None:
        pytest.skip("LOTSA pretrain datasets not available on disk")

    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import uni2ts.common.hydra_util  # noqa: F401  registers cls_getattr resolver

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name=CONFIG_NAME)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    eff_total = 0.0
    eff_synth = 0.0
    eff_mixup = eff_ksynth = 0.0
    for builder in cfg["_args_"]:
        weight_map = builder.get("weight_map", {}) or {}
        for ds in builder["datasets"]:
            n = _num_examples_on_disk(root, ds)
            if n is None:
                pytest.skip(f"dataset {ds} missing on disk")
            eff = math.ceil(n * weight_map.get(ds, 1.0))
            eff_total += eff
            if ds.startswith("chronos_"):
                eff_synth += eff
                if "tsmixup" in ds:
                    eff_mixup += eff
                elif "kernel_synth" in ds:
                    eff_ksynth += eff

    synth_share = eff_synth / eff_total
    real_share = 1.0 - synth_share
    mixup_to_ksynth = eff_mixup / eff_ksynth

    # Committed target: ~20% synthetic, ~80% real, 9:1 mixup:ksynth.
    assert 0.15 <= synth_share <= 0.25, f"synthetic share {synth_share:.3f} off-band"
    assert 0.75 <= real_share <= 0.85, f"real share {real_share:.3f} off-band"
    assert 8.0 <= mixup_to_ksynth <= 10.0, f"mixup:ksynth {mixup_to_ksynth:.1f} off"


# ===========================================================================
# 5. Track A synth_stress mixture (gift_eval_pretrain_moirai2_synthstress.yaml):
#    the committed w_stress=0.9444 over 500k series must give synth_stress a
#    ~12% sampling share on top of the unchanged 20%-Chronos mix
#    (HANDOFF_SYNTH_DATA.md §A.2 targets 10-15%).
# ===========================================================================
NUM_TS_STRESS = 500_000  # synth_stress_v0 --num-series default
CURRENT_WEIGHT_STRESS = 0.9444


def test_synthstress_weight_matches_documented_share():
    chronos_eff = (
        NUM_TS_MIXUP * CURRENT_WEIGHT_MIXUP + NUM_TS_KSYNTH * CURRENT_WEIGHT_KSYNTH
    )
    stress_eff = NUM_TS_STRESS * CURRENT_WEIGHT_STRESS
    total = REAL_EFFECTIVE_POOL + chronos_eff + stress_eff
    stress_share = stress_eff / total
    assert stress_share == pytest.approx(0.12, abs=0.005)
    # real data must stay dominant (handoff guardrail)
    assert REAL_EFFECTIVE_POOL / total > 0.65


@pytest.mark.slow
def test_synthstress_config_yields_documented_mixture():
    root = _lotsa_path()
    if root is None or _num_examples_on_disk(root, "synth_stress_v0") is None:
        pytest.skip("synth_stress_v0 not built on disk")

    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import uni2ts.common.hydra_util  # noqa: F401

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name="gift_eval_pretrain_moirai2_synthstress")
    cfg = OmegaConf.to_container(cfg, resolve=True)

    eff_total = 0.0
    eff_stress = 0.0
    for builder in cfg["_args_"]:
        weight_map = builder.get("weight_map", {}) or {}
        for ds in builder["datasets"]:
            n = _num_examples_on_disk(root, ds)
            if n is None:
                pytest.skip(f"dataset {ds} missing on disk")
            eff = math.ceil(n * weight_map.get(ds, 1.0))
            eff_total += eff
            if ds == "synth_stress_v0":
                eff_stress += eff

    stress_share = eff_stress / eff_total
    assert 0.10 <= stress_share <= 0.15, f"synth_stress share {stress_share:.3f} off"


# ===========================================================================
# 6. Track B retrain mixtures (gift_eval_pretrain_trackb_p{20,30}_{sde,nosde}):
#    pure-synth share must hit p with the documented family split; real stays
#    dominant; TSMixup weight unchanged.
# ===========================================================================
TRACKB_NUM = dict(ks=1_000_000, stress=500_000, backbone=1_000_000,
                  sarima2=300_000, sde=500_000)
TRACKB_W = {  # committed weights (yaml values)
    ("p20", "sde"): dict(backbone=0.29693, sde=0.42419, sarima2=0.2828,
                         stress=0.33935, ks=0.08484),
    ("p20", "nosde"): dict(backbone=0.42419, sarima2=0.42419,
                           stress=0.42419, ks=0.08484),
    ("p30", "sde"): dict(backbone=0.50903, sde=0.72719, sarima2=0.48479,
                         stress=0.58175, ks=0.14544),
    ("p30", "nosde"): dict(backbone=0.72719, sarima2=0.72719,
                           stress=0.72719, ks=0.14544),
}


@pytest.mark.parametrize("p_tag, arm", list(TRACKB_W))
def test_trackb_weights_hit_target_share(p_tag, arm):
    p = int(p_tag[1:]) / 100
    w = TRACKB_W[(p_tag, arm)]
    base = REAL_EFFECTIVE_POOL + NUM_TS_MIXUP * CURRENT_WEIGHT_MIXUP
    synth_eff = sum(TRACKB_NUM[f] * w[f] for f in w)
    total = base + synth_eff
    assert synth_eff / total == pytest.approx(p, abs=0.003)
    assert REAL_EFFECTIVE_POOL / total > 0.55  # real stays dominant
    if arm == "sde":  # SDE gets the handoff's rank-2 share (25% of synth)
        assert TRACKB_NUM["sde"] * w["sde"] / synth_eff == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize(
    "p_tag, gift_weight",
    [("p30", 5.5431), ("p20", 4.8502)],
)
def test_gift_train_split_weight_hits_10pct_share(p_tag, gift_weight):
    """Gift-split mixes (gift_eval_pretrain_trackb_p{30,20}_nosde_gift.yaml):
    the committed weight must give the GIFT train split ~10% sampling share on
    top of the corresponding nosde pool."""
    pool = (
        REAL_EFFECTIVE_POOL
        + NUM_TS_MIXUP * CURRENT_WEIGHT_MIXUP
        + sum(TRACKB_NUM[f] * w for f, w in TRACKB_W[(p_tag, "nosde")].items())
    )
    gift_eff = 97_177 * gift_weight
    assert gift_eff / (pool + gift_eff) == pytest.approx(0.10, abs=0.003)
