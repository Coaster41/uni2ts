#  FIX G (.claude/HANDOFF_MOIRAI2_PARITY.md): GIFT-Eval TrainTest TRAIN split
#  as a pretraining dataset. Moirai2's mixture includes these 144K in-domain
#  series; the benchmark's own gift_eval.data.Dataset class defines the splits
#  (per series: last windows*PL steps = test, one PL window before that = val,
#  the rest = train). Using this forfeits the leaderboard "zero-shot" tag but
#  is NOT test leakage (moirai2 is tagged non-leaking with it).
#
#  Safety posture (deliberately more conservative than the official boundary):
#    cut(series) = max over ALL evaluated terms of
#                    PL_term * (windows_term + 1 + EXTRA_WINDOWS)
#  i.e. the official train boundary (windows+1) plus EXTRA_WINDOWS=1 additional
#  prediction-length margin, taken across short/medium/long so no term's test
#  or validation window can overlap what we train on. A JSON manifest with the
#  per-config cut and counts is written next to the Arrow dataset; the leakage
#  test (test/data/test_gift_eval_train_split.py) re-derives the boundary from
#  gift_eval.data.Dataset and checks stored lengths against it.
#
#  Requires the gift-eval repo importable (shared venv) and $GIFT_EVAL pointing
#  at the benchmark data (defaults to $GIFT_EVAL_TEST_PATH / uni2ts .env value).
#
#  Build:
#    python -m uni2ts.data.builder.lotsa_v1.gift_eval_train

import argparse
import json
import os
from collections import defaultdict
from functools import partial
from typing import Optional

import datasets
import numpy as np

from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder
from .synth_stress import LOTSA_FEATURES

# Same evaluated-config lists as gift-eval/moiraic_eval/evaluate_worker.py:
# every config is evaluated at term=short; MED_LONG configs also at medium+long.
SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T "
    "M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()

EXTRA_WINDOWS = 1  # safety margin beyond the official (windows+1) boundary
MIN_KEEP_LEN = 64  # >= 2 patches at patch_size 32; drop shorter remnants


def _ensure_gift_eval_env():
    if "GIFT_EVAL" not in os.environ:
        from uni2ts.common.env import env

        path = getattr(env, "GIFT_EVAL_TEST_PATH", None)
        assert path is not None, "set GIFT_EVAL or GIFT_EVAL_TEST_PATH"
        os.environ["GIFT_EVAL"] = str(path)


def evaluated_terms(name: str) -> list[str]:
    return ["short", "medium", "long"] if name in MED_LONG_DATASETS else ["short"]


def conservative_cut(name: str, extra_windows: int = EXTRA_WINDOWS) -> int:
    """Steps to drop from the END of every series of `name`, maxed over terms."""
    from gift_eval.data import Dataset as GEDataset

    cut = 0
    for term in evaluated_terms(name):
        d = GEDataset(name=name, term=term, to_univariate=False)
        cut = max(cut, d.prediction_length * (d.windows + 1 + extra_windows))
    return cut


def _config_gen(names: list[str], extra_windows: int, min_keep_len: int):
    """Yield LOTSA rows for each evaluated config, truncated at the safe cut."""
    _ensure_gift_eval_env()
    from gift_eval.data import Dataset as GEDataset

    for name in names:
        cut = conservative_cut(name, extra_windows)
        probe = GEDataset(name=name, term="short", to_univariate=False)
        ds = GEDataset(
            name=name, term="short", to_univariate=probe.target_dim != 1
        )
        freq = ds.freq
        safe_name = name.replace("/", "_")
        for entry in ds.gluonts_dataset:
            target = np.asarray(entry["target"], dtype=np.float32)
            assert target.ndim == 1
            kept = target[: len(target) - cut]
            if len(kept) < min_keep_len:
                continue
            yield dict(
                item_id=f"gift_{safe_name}_{entry['item_id']}",
                start=entry["start"].to_timestamp(),
                freq=freq,
                target=kept,
            )


class GiftEvalTrainSplitDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = ["gift_eval_train_split"]
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(
        self,
        dataset: str = "gift_eval_train_split",
        extra_windows: int = EXTRA_WINDOWS,
        min_keep_len: int = MIN_KEEP_LEN,
        num_proc: Optional[int] = None,
    ):
        _ensure_gift_eval_env()
        num_proc = num_proc or 8
        names = sorted(set(SHORT_DATASETS) | set(MED_LONG_DATASETS))

        hf_dataset = datasets.Dataset.from_generator(
            _config_gen,
            features=LOTSA_FEATURES,
            gen_kwargs=dict(
                names=names, extra_windows=extra_windows, min_keep_len=min_keep_len
            ),
            num_proc=num_proc,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset)

        # Build manifest: per-config cut + counts, for the leakage test and docs.
        from gift_eval.data import Dataset as GEDataset

        manifest = dict(
            extra_windows=extra_windows,
            min_keep_len=min_keep_len,
            total_series=len(hf_dataset),
            configs={},
        )
        for name in names:
            probe = GEDataset(name=name, term="short", to_univariate=False)
            manifest["configs"][name] = dict(
                cut=conservative_cut(name, extra_windows),
                terms=evaluated_terms(name),
                target_dim=int(probe.target_dim),
                num_source_rows=len(probe.hf_dataset),
            )
        with open(self.storage_path / dataset / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
        print(
            f"[{dataset}] Saved {len(hf_dataset)} series "
            f"({len(names)} configs) to {self.storage_path / dataset}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the GIFT-Eval train-split pretraining dataset (FIX G)."
    )
    parser.add_argument("--extra-windows", type=int, default=EXTRA_WINDOWS)
    parser.add_argument("--min-keep-len", type=int, default=MIN_KEEP_LEN)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--storage-path", default=None)
    args = parser.parse_args()

    from pathlib import Path

    kwargs = {}
    if args.storage_path is not None:
        kwargs["storage_path"] = Path(args.storage_path)
    builder = GiftEvalTrainSplitDatasetBuilder(
        datasets=["gift_eval_train_split"], **kwargs
    )
    builder.build_dataset(
        extra_windows=args.extra_windows,
        min_keep_len=args.min_keep_len,
        num_proc=args.num_proc,
    )
