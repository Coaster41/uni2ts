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

from collections import defaultdict
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator, Optional
from zipfile import ZipFile

import datasets
import gluonts
import numpy as np
from datasets import Features, Sequence, Value
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._tsf_datasets import Dataset as MonashDataset
from gluonts.dataset.repository._tsf_datasets import TSFReader, convert_data
from gluonts.dataset.repository._tsf_reader import frequency_converter
from gluonts.dataset.repository._util import metadata
from gluonts.dataset.repository.datasets import get_dataset
from pandas.tseries.frequencies import to_offset

from uni2ts.common.env import env
from uni2ts.data.dataset import MultiSampleTimeSeriesDataset, TimeSeriesDataset

from ._base import LOTSADatasetBuilder


def default_prediction_length_from_frequency(freq: str) -> int:
    prediction_length_map = {
        "T": 60 * 24 * 7,
        "H": 24 * 7,
        "D": 30,
        "W-SUN": 8,
        "M": 12,
        "Y": 4,
        "S": 60 * 60 * 24 * 7,
    }
    try:
        freq = to_offset(freq).name
        return prediction_length_map[freq]
    except KeyError as err:
        raise ValueError(
            f"Cannot obtain default prediction length from frequency `{freq}`."
        ) from err


gluonts.dataset.repository._tsf_datasets.default_prediction_length_from_frequency = (
    default_prediction_length_from_frequency
)


def generate_forecasting_dataset(
    dataset_path: Path,
    dataset_name: str,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
):
    dataset = gluonts.dataset.repository._tsf_datasets.datasets[dataset_name]
    dataset_path.mkdir(exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        with ZipFile(dataset.download(temp_path)) as archive:
            archive.extractall(path=temp_path)

        # only one file is exptected
        reader = TSFReader(temp_path / archive.namelist()[0])
        meta, data = reader.read()

    if dataset_name.startswith("cif_2016") and len(dataset_name) > len("cif_2016"):
        horizon = int(dataset_name[len("cif_2016_") :])
        data = list(filter(lambda x: x if x["horizon"] == horizon else False, data))
        meta.forecast_horizon = horizon

    if dataset_name.startswith("monash_m3_other"):
        meta.frequency = "quarterly"

    freq = frequency_converter(meta.frequency)
    if prediction_length is None:
        if hasattr(meta, "forecast_horizon"):
            prediction_length = int(meta.forecast_horizon)
        else:
            prediction_length = default_prediction_length_from_frequency(freq)

    # Impute missing start dates with unix epoch and remove time series whose
    # length is less than or equal to the prediction length
    data = [
        {**d, "start_timestamp": d.get("start_timestamp", "1970-01-01")}
        for d in data
        if len(d[FieldName.TARGET]) > prediction_length
    ]
    train_data, test_data = convert_data(data, prediction_length)

    meta = MetaData(
        **metadata(
            cardinality=len(data),
            freq=freq,
            prediction_length=prediction_length,
        )
    )

    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(path_str=str(dataset_path), writer=dataset_writer, overwrite=True)


gluonts.dataset.repository._tsf_datasets.generate_forecasting_dataset = (
    generate_forecasting_dataset
)


additional_datasets = {
    "wind_power": MonashDataset(
        file_name="wind_4_seconds_dataset.zip",
        record="4656032",
        ROOT="https://zenodo.org/record",
    ),
}

gluonts.dataset.repository._tsf_datasets.datasets |= additional_datasets
gluonts.dataset.repository.datasets.dataset_recipes |= {
    k: partial(
        generate_forecasting_dataset,
        dataset_name=k,
    )
    for k in additional_datasets.keys()
}

PRETRAIN_GROUP = [
    "wind_power"
]

TRAIN_TEST_GROUP = {
}


MULTI_SAMPLE_DATASETS = [
]


class GluonTSSubDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = PRETRAIN_GROUP + list(TRAIN_TEST_GROUP)
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset) | {
        dataset: MultiSampleTimeSeriesDataset for dataset in MULTI_SAMPLE_DATASETS
    }
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset)) | {
        dataset: partial(
            MultiSampleTimeSeriesDataset,
            max_ts=128,
            combine_fields=("target", "past_feat_dynamic_real"),
        )
        for dataset in MULTI_SAMPLE_DATASETS
    }

    def build_dataset(self, dataset):
        if dataset in TRAIN_TEST_GROUP:
            gluonts_dataset = get_dataset(
                dataset,
                prediction_length=TRAIN_TEST_GROUP[dataset],
                regenerate=True,
            ).train
        elif dataset in PRETRAIN_GROUP:
            gluonts_dataset = get_dataset(dataset, regenerate=True).test
        else:
            raise ValueError(f"Unrecognized dataset: {dataset}")

        def gen_func() -> Generator[dict[str, Any], None, None]:
            for item in gluonts_dataset:
                if dataset == "covid_mobility":
                    if (
                        len(item["target"]) < 100
                        or np.isnan(item["target"]).sum() / len(item["target"]) > 0.25
                    ):
                        continue
                if len(item["target"]) < 20:
                    continue

                freq = item["start"].freqstr
                if freq is None or freq == "":
                    raise ValueError
                item["freq"] = freq
                item["start"] = item["start"].to_timestamp()
                del item["feat_static_cat"]
                yield item

        hf_dataset = datasets.Dataset.from_generator(
            generator=gen_func,
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                )
            ),
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(dataset_path=self.storage_path / dataset)
