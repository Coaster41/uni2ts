from __future__ import annotations

import numpy as np

from .dataset import load_gift_eval_series, wrap_existing_dataset
from .pseudo_labels import PSEUDO_LABEL_FUNCTIONS

CONTEXT_LEN = 512  # 32 patches * 16

GIFT_DATASETS = [
    "electricity/H",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/H",
    "m4_daily",
    "m4_hourly",
    "ett1/H",
    "temperature_rain_with_missing",
    "jena_weather",
]


def load_gift_subset(
    seed: int = 42,
    n_per_dataset: int = 600,
    series_length: int = 576,
    context_len: int = CONTEXT_LEN,
    storage_path: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Load and window 9 GIFT-Eval datasets, compute pseudo-labels on context portion.

    Returns dict matching the synthetic-data format so that run_probes_for_model
    works unchanged:
        "series":                   float32 [N, series_length]
        "dataset_id":               int32   [N]
        <pseudo-label keys>:        float32 [N]  (11 keys)
    """
    all_windows: list[np.ndarray] = []
    all_dataset_ids: list[int] = []
    all_labels: dict[str, list[float]] = {f: [] for f in PSEUDO_LABEL_FUNCTIONS}

    for dataset_id, dataset_name in enumerate(GIFT_DATASETS):
        series_list = load_gift_eval_series(dataset_name, storage_path=storage_path)
        ds = wrap_existing_dataset(
            series_list,
            label_generators=[],
            n=n_per_dataset,
            series_length=series_length,
            seed=seed + dataset_id,
        )
        windows = ds["series"]  # [n_per_dataset, series_length]

        all_windows.append(windows)
        all_dataset_ids.extend([dataset_id] * n_per_dataset)

        for window in windows:
            ctx = window[:context_len]
            for label_name, fn in PSEUDO_LABEL_FUNCTIONS.items():
                all_labels[label_name].append(fn(ctx))

    result: dict[str, np.ndarray] = {
        "series": np.concatenate(all_windows, axis=0),
        "dataset_id": np.array(all_dataset_ids, dtype=np.int32),
    }
    for label_name, vals in all_labels.items():
        result[label_name] = np.array(vals, dtype=np.float32)
    return result


def save_gift_subset(ds: dict[str, np.ndarray], path: str) -> None:
    """Save the output of load_gift_subset to a .npz file."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **ds)
