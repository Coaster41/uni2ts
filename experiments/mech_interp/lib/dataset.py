from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable

import numpy as np

from .label_generators import LabelGenerator

# Fallback path used when GIFT_EVAL_TEST_PATH env var is not set.
_GIFT_EVAL_DEFAULT_PATH = os.getenv(
    "GIFT_EVAL_TEST_PATH",
    "/srv/disk00/ctadler/uni2ts/datasets/giftevaltest",
)


def wrap_existing_dataset(
    series_source: Iterable[np.ndarray],
    label_generators: list[LabelGenerator],
    series_length: int = 576,
    n: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Wrap any iterable of 1D time series into the mech interp dataset format.

    Samples n random windows of series_length from series_source.
    Series shorter than series_length or containing NaN are excluded.
    Windows are sampled with probability proportional to the number of valid
    start positions in each series, then start positions are uniform within.

    Labels are computed on raw (un-normalized) windows via label_generators,
    consistent with how synthetic ground-truth labels relate to raw series.

    Parameters
    ----------
    series_source:
        Iterable of 1D numpy arrays (variable length, any float dtype).
    label_generators:
        List of callables: series -> dict[str, np.ndarray].
    series_length:
        Window size in time steps. Default 576 = (32+4) patches * 16.
    n:
        Number of windows to sample.
    seed:
        RNG seed; calling twice with the same seed yields identical output.

    Returns
    -------
    dict with key "series" (float32[n, series_length]) plus one key per label
    generator output. Raises ValueError if not enough valid windows exist.
    """
    all_series = [np.asarray(s, dtype=np.float32) for s in series_source]
    eligible = [s for s in all_series if len(s) >= series_length and not np.any(np.isnan(s))]

    n_windows = np.array([len(s) - series_length + 1 for s in eligible], dtype=np.float64)
    total_windows = int(n_windows.sum())
    if total_windows < n:
        raise ValueError(
            f"Only {total_windows} valid windows across {len(eligible)} eligible "
            f"series (need {n}). Reduce n or use a larger dataset."
        )

    rng = np.random.default_rng(seed)
    weights = n_windows / n_windows.sum()
    series_idxs = rng.choice(len(eligible), size=n, replace=True, p=weights)

    windows: list[np.ndarray] = []
    labels: dict[str, list] = defaultdict(list)
    for si in series_idxs:
        s = eligible[si]
        max_start = len(s) - series_length
        start = int(rng.integers(0, max_start + 1))
        window = s[start : start + series_length]
        windows.append(window)
        for gen in label_generators:
            for k, v in gen(window).items():
                labels[k].append(v)

    result: dict[str, np.ndarray] = {"series": np.stack(windows, axis=0)}
    for k, vals in labels.items():
        result[k] = np.array(vals)
    return result


def load_gift_eval_series(
    dataset_name: str,
    term: str = "short",
    split: str = "train",
    storage_path: str | None = None,
) -> list[np.ndarray]:
    """
    Load univariate time series from a GiftEval dataset.

    Loads directly from the HuggingFace Arrow files to avoid GluonTS split
    machinery issues. Multivariate entries (target shape [V, T]) are
    decomposed into V individual 1D arrays.

    Only split='train' is accepted to prevent test contamination; the last
    10% of each series (the GiftEval test split) is withheld automatically.

    Parameters
    ----------
    dataset_name:
        GiftEval dataset identifier, e.g. "electricity/H" or "m4_monthly".
    term:
        Unused (kept for API compatibility with gift_eval.data.Dataset).
    split:
        Must be "train".
    storage_path:
        Path to the giftevaltest root directory. Defaults to
        GIFT_EVAL_TEST_PATH env var, or the hardcoded fallback path.

    Returns
    -------
    List of 1D float32 numpy arrays, one per univariate series.
    """
    import datasets as hf_datasets  # type: ignore[import]

    if split != "train":
        raise ValueError(f"Only split='train' is supported; got {split!r}.")

    root = storage_path if storage_path is not None else _GIFT_EVAL_DEFAULT_PATH
    hf_dataset = hf_datasets.load_from_disk(os.path.join(root, dataset_name))

    series_list: list[np.ndarray] = []
    for entry in hf_dataset:
        target = np.asarray(entry["target"], dtype=np.float32)
        if target.ndim == 1:
            cutoff = int(len(target) * 0.9)
            series_list.append(target[:cutoff])
        else:
            cutoff = int(target.shape[1] * 0.9)
            for i in range(target.shape[0]):
                series_list.append(target[i, :cutoff])

    return series_list
