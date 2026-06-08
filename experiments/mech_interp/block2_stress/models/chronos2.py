"""
Adapter for Chronos-2 (``amazon/chronos-2``).

Mirrors the gift-eval ``chronos-2`` notebook: build the pipeline with
``BaseChronosPipeline.from_pretrained`` and call ``predict_quantiles`` with our
canonical decile grid. Requires ``chronos-forecasting>=2.x`` which is *not*
installed in the uni2ts venv — run this adapter in an env that has it (e.g.
gift-eval's), all writing to the shared forecasts dir.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastAdapter, batcher


def _to_qhb(arr: np.ndarray, n_q: int, horizon: int) -> np.ndarray:
    """Normalize a chronos quantile output chunk to ``[b, Q, H]``."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:  # [Q, H] or [H, Q] for a single series
        arr = arr[None]
    # arr is now 3-D; figure out which axis is the quantile axis.
    if arr.shape[-1] == n_q and arr.shape[1] == horizon:  # [b, H, Q]
        arr = arr.transpose(0, 2, 1)
    elif arr.shape[1] == n_q:  # [b, Q, H]
        pass
    elif arr.shape[-1] == n_q:  # [b, *, Q]
        arr = np.moveaxis(arr, -1, 1)
    return arr[:, :, :horizon]


class Chronos2Adapter(ForecastAdapter):
    def __init__(
        self,
        model_name: str = "amazon/chronos-2",
        device: str = "cuda",
        **kwargs,
    ):
        from chronos import BaseChronosPipeline

        self.model_name = model_name
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, **kwargs
        )

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 256,
    ) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        q_levels = list(self.quantile_levels)
        out: list[np.ndarray] = []
        for chunk in batcher(list(context), batch_size):
            inputs = [{"target": row.astype(np.float32)} for row in chunk]
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=horizon,
                quantile_levels=q_levels,
            )
            out.append(_to_qhb(quantiles, len(q_levels), horizon))
        return np.concatenate(out, axis=0).astype(np.float32)
