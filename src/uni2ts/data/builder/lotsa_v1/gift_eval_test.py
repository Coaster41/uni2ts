import os
import shutil
from collections import defaultdict
from itertools import chain
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from datasets import Features, Sequence, Value

from uni2ts.common.env import env
from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder, SampleTimeSeriesType


GIFT_EVAL_TEST_DATASETS = [
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]

GIFT_EVAL_TO_UNIVARIATE = {
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
}

GIFT_EVAL_DEFAULT_WEIGHT_MAP = dict.fromkeys(GIFT_EVAL_TEST_DATASETS, 1.0)


def maybe_reconvert_freq(freq: str) -> str:
    deprecated_map = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    return deprecated_map.get(freq, freq)


def normalize_array(x):
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr.squeeze(0)
        elif arr.shape[1] == 1:
            arr = arr.squeeze(1)

    return arr


def make_sequence_feature(x):
    arr = np.asarray(x)

    if arr.ndim <= 1:
        return Sequence(Value("float32"))

    return Sequence(Sequence(Value("float32")), length=arr.shape[0])


class GiftEvalTestDatasetBuilder(LOTSADatasetBuilder):
    """
    Imports already-built GIFT-Eval datasets from $GIFT_EVAL and rewrites them into
    the on-disk HF format that uni2ts LOTSA builders expect.

    Important:
    - Datasets listed as to_univariate=true in gift_eval are split at import time.
    - All datasets are loaded as plain TimeSeriesDataset instances.
    """

    source_env_var = "GIFT_EVAL_TEST_PATH"

    dataset_list = GIFT_EVAL_TEST_DATASETS
    to_univariate_datasets = GIFT_EVAL_TO_UNIVARIATE
    default_weight_map = GIFT_EVAL_DEFAULT_WEIGHT_MAP

    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: TimeSeriesDataset)

    def __init__(
        self,
        datasets: list[str],
        weight_map: dict[str, float] | None = None,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        storage_path: Path | str | None = None,
    ):
        if storage_path is None:
            storage_path = env.GIFT_EVAL_TEST_PATH
        storage_path = Path(storage_path)

        super().__init__(
            datasets=datasets,
            weight_map=weight_map,
            sample_time_series=sample_time_series,
            storage_path=storage_path,
        )

    def _iter_entries(self, dataset_name: str, hf_dataset):
        make_uni = dataset_name in self.to_univariate_datasets

        for row_idx, row in enumerate(hf_dataset):
            item_id = row.get("item_id", None)
            if item_id is None:
                item_id = f"{dataset_name.replace('/', '__')}_{row_idx}"
            item_id = str(item_id)

            start = pd.Timestamp(row["start"]).to_pydatetime()
            freq = maybe_reconvert_freq(str(row["freq"]))
            target = normalize_array(row["target"])

            extra = {}

            cov_key = None
            if "past_feat_dynamic_real" in row:
                cov_key = "past_feat_dynamic_real"
            elif "feat_dynamic_real" in row:
                cov_key = "feat_dynamic_real"

            if cov_key is not None and row[cov_key] is not None:
                extra["past_feat_dynamic_real"] = normalize_array(row[cov_key])

            if make_uni and np.asarray(target).ndim > 1:
                for dim, target_dim in enumerate(target):
                    yield {
                        "item_id": f"{item_id}_dim{dim}",
                        "start": start,
                        "freq": freq,
                        "target": np.asarray(target_dim, dtype=np.float32),
                        **extra,
                    }
            else:
                yield {
                    "item_id": item_id,
                    "start": start,
                    "freq": freq,
                    "target": target,
                    **extra,
                }

    def _make_features(self, sample: dict) -> Features:
        features = {
            "item_id": Value("string"),
            "start": Value("timestamp[s]"),
            "freq": Value("string"),
            "target": make_sequence_feature(sample["target"]),
        }

        if "past_feat_dynamic_real" in sample:
            features["past_feat_dynamic_real"] = make_sequence_feature(
                sample["past_feat_dynamic_real"]
            )

        return Features(features)

    def build_dataset(self, dataset: str, num_proc: int = os.cpu_count()):
        source_root = os.getenv(self.source_env_var)
        if not source_root:
            raise EnvironmentError(
                f"{self.source_env_var} is not set. "
                f"Set it to the root directory containing the GIFT-Eval HF datasets."
            )

        source_path = Path(source_root) / dataset
        if not source_path.exists():
            raise FileNotFoundError(
                f"Could not find source GIFT-Eval dataset at {source_path}"
            )

        src = datasets.load_from_disk(str(source_path)).with_format("numpy")

        iterator = iter(self._iter_entries(dataset, src))
        first = next(iterator, None)
        if first is None:
            raise ValueError(f"GIFT-Eval dataset {dataset} is empty")

        dst_path = self.storage_path / dataset
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            shutil.rmtree(dst_path)

        out = datasets.Dataset.from_generator(
            lambda: chain([first], iterator),
            features=self._make_features(first),
            cache_dir=env.HF_CACHE_PATH,
        )
        out.info.dataset_name = dataset
        out.save_to_disk(dst_path)