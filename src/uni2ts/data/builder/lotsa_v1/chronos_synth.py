#  New builder for Chronos synthetic datasets (tsmixup, kernel_synth)

import os
from collections import defaultdict
from functools import partial
from typing import Any, Generator, Optional

import datasets
import numpy as np
from datasets import Features, Sequence, Value

from uni2ts.common.env import env
from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder

# Internal name -> HuggingFace config name
HF_CONFIG_MAP = {
    "chronos_tsmixup_10m": "training_corpus_tsmixup_10m",
    "chronos_kernel_synth_1m": "training_corpus_kernel_synth_1m",
}

LOTSA_FEATURES = Features(
    dict(
        item_id=Value("string"),
        start=Value("timestamp[s]"),
        freq=Value("string"),
        target=Sequence(Value("float32")),
    )
)


class ChronosSyntheticDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = list(HF_CONFIG_MAP.keys())
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(
        self,
        dataset: str,
        source_path: Optional[str] = None,   # <-- NEW: local download dir
        num_proc: int = os.cpu_count(),
    ):
        """
        Downloads the Chronos parquet dataset from HF Hub or local,
        maps it to the LOTSA Arrow schema, and saves to disk.
        """
        hf_config = HF_CONFIG_MAP[dataset]

        # --- Step 1: Load from HF Hub (parquet) ----
        if source_path is not None:
            # ── Load from local parquet files ──
            # Expects: <source_path>/<config_name>/*.parquet
            # e.g.  /data/chronos/training_corpus_tsmixup_10m/train-00000-of-00042.parquet
            import glob
            parquet_pattern = os.path.join(source_path, hf_config, "*.parquet")
            parquet_files = sorted(glob.glob(parquet_pattern))

            if not parquet_files:
                # Also check one level deeper (e.g. <source_path>/<config>/train/*.parquet)
                parquet_pattern = os.path.join(source_path, hf_config, "**", "*.parquet")
                parquet_files = sorted(glob.glob(parquet_pattern, recursive=True))

            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files found at {parquet_pattern}. "
                    f"Check that your --local-dir structure matches: "
                    f"<source_path>/{hf_config}/*.parquet"
                )

            print(f"[{dataset}] Loading {len(parquet_files)} local parquet files")
            src = datasets.load_dataset(
                "parquet",
                data_files=parquet_files,
                split="train",
            )
        else:
            src = datasets.load_dataset(
                "autogluon/chronos_datasets",
                hf_config,
                split="train",
            )

        # --- Step 2: Inspect and log the source schema ---
        print(f"[{dataset}] Source columns : {src.column_names}")
        print(f"[{dataset}] Source features: {src.features}")
        print(f"[{dataset}] Num rows      : {len(src)}")

        # --- Step 3: Determine the source column that holds the values ---
        # Chronos datasets may use "values" or "target"
        if "target" in src.column_names:
            values_col = "target"
        elif "values" in src.column_names:
            values_col = "values"
        else:
            raise ValueError(
                f"Cannot find a values column in {src.column_names}. "
                "Expected 'target' or 'values'."
            )

        has_start = "start" in src.column_names
        has_freq = "freq" in src.column_names
        has_item_id = "item_id" in src.column_names

        # --- Step 4: Map to LOTSA schema ---
        def _map_to_lotsa(example, idx):
            return {
                "item_id": (
                    str(example["item_id"]) if has_item_id else str(idx)
                ),
                "start": (
                    example["start"] if has_start else "1970-01-01"
                ),
                "freq": (
                    example["freq"] if has_freq else "H"
                ),
                "target": example[values_col],
            }

        mapped = src.map(
            _map_to_lotsa,
            with_indices=True,
            remove_columns=src.column_names,   # drop all original columns
            features=LOTSA_FEATURES,
            num_proc=num_proc,
            desc=f"Mapping {dataset} to LOTSA schema",
        )

        # --- Step 5: Save as Arrow ---
        mapped.info.dataset_name = dataset
        mapped.save_to_disk(
            dataset_path=self.storage_path / dataset,
            num_proc=num_proc,
        )
        print(f"[{dataset}] Saved to {self.storage_path / dataset}")