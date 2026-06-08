"""
Base adapter interface for the block2_stress forecast harness.

Every model the stress battery runs is wrapped in a :class:`ForecastAdapter`
exposing a single method::

    predict_quantiles(context: np.ndarray[n, ctx_len], horizon: int) -> np.ndarray[n, Q, H]

This is the common denominator the rest of block2 needs: the stress battery
operates on fixed-length raw numpy series, splitting each into a context window
and a horizon, and the metrics consume the median quantile. All adapters
standardize on the 9 deciles (0.1 .. 0.9, median at index 4), matching both the
custom moirai models and the gift-eval default quantile levels.

This module is intentionally dependency-light (numpy only) so it imports cleanly
in any environment. Heavy / model-specific imports live inside the concrete
adapter modules and are loaded lazily.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Sequence

import numpy as np

# The canonical quantile grid for the whole stress battery. Median is index 4.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


def batcher(items: Sequence, batch_size: int) -> Iterator[list]:
    """Yield successive ``batch_size``-sized chunks from ``items``."""
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


class ForecastAdapter(ABC):
    """Uniform forecast interface over heterogeneous foundation models.

    Subclasses implement :meth:`predict_quantiles`. The constructor of each
    concrete adapter takes whatever model-specific kwargs it needs (checkpoint
    path, HF repo id, device, ...) and performs lazy imports of its backend.
    """

    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS

    @abstractmethod
    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Forecast quantiles for a batch of context windows.

        Parameters
        ----------
        context : float ndarray [n, ctx_len]
            Past values; one row per series.
        horizon : int
            Number of future steps to forecast.
        batch_size : int
            Internal batching hint for the backend.

        Returns
        -------
        float32 ndarray [n, Q, horizon]
            Quantile forecasts, ``Q == len(self.quantile_levels)``, ordered to
            match ``self.quantile_levels`` (median at index 4 for the default
            decile grid).
        """
        raise NotImplementedError

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_levels)


def gluonts_forecasts_to_qarray(
    forecasts: Iterable,
    quantile_levels: Sequence[float],
    horizon: int,
) -> np.ndarray:
    """Convert an iterable of gluonts ``Forecast`` objects to ``[n, Q, H]``.

    Works for both ``QuantileForecast`` and ``SampleForecast`` because every
    gluonts forecast implements ``.quantile(q)`` (samples are reduced via the
    empirical quantile). Univariate forecasts are assumed; any trailing target
    dimension is squeezed.
    """
    rows: list[np.ndarray] = []
    for fc in forecasts:
        per_q = []
        for q in quantile_levels:
            arr = np.asarray(fc.quantile(q), dtype=np.float32)  # [H] or [H, dim]
            if arr.ndim > 1:
                arr = arr.reshape(arr.shape[0], -1)[:, 0]
            per_q.append(arr[:horizon])
        rows.append(np.stack(per_q, axis=0))  # [Q, H]
    return np.stack(rows, axis=0).astype(np.float32)  # [n, Q, H]


def stack_quantile_levels(
    per_series: Sequence[np.ndarray],
) -> np.ndarray:
    """Stack a list of ``[Q, H]`` arrays into ``[n, Q, H]`` float32."""
    return np.stack(per_series, axis=0).astype(np.float32)
