"""
Adapter for TimesFMX (the trainable TimesFM fork in /srv/disk00/ctadler/timesfm).

Uses the same inference path as the gift-eval worker: ``TimesFMXModule
.from_pretrained(ckpt)`` wrapped in ``TimesFMXForecast`` with ``ar_method``
(default "naive", matching the short-protocol eval). ``TimesFMXForecast
.predict(list[np.ndarray])`` already returns ``[n, Q, H]`` over the module's
9 decile quantile_levels, so no reshaping is needed beyond a horizon slice.

``prediction_length``/``context_length`` are constructor hparams of the
Forecast wrapper, so the wrapper is (re)built lazily per (ctx_len, horizon)
workload; the loaded module is reused across rebuilds.

Requires the ``timesfm`` fork installed in the venv (it is, in uni2ts/venv).
"""
from __future__ import annotations

import numpy as np

from .base import ForecastAdapter, batcher


class TimesFMXAdapter(ForecastAdapter):
    def __init__(
        self,
        ckpt: str,
        device: str = "cuda:0",
        ar_method: str = "naive",
        single_pass_horizon: int | None = None,
        **_ignored,
    ):
        import torch
        from timesfm.pretrain import TimesFMXModule

        self.device = torch.device(device if device not in (None, "auto") else "cuda:0")
        self.module = TimesFMXModule.from_pretrained(ckpt)
        self.ar_method = ar_method
        self.single_pass_horizon = single_pass_horizon
        self._forecast = None
        self._forecast_key = None

    def _ensure_forecast(self, ctx_len: int, horizon: int):
        from timesfm.pretrain import TimesFMXForecast

        key = (ctx_len, horizon)
        if self._forecast_key == key:
            return
        kwargs = {}
        if self.single_pass_horizon is not None:
            kwargs["single_pass_horizon"] = self.single_pass_horizon
        self._forecast = TimesFMXForecast(
            module=self.module,
            prediction_length=horizon,
            context_length=ctx_len,
            target_dim=1,
            ar_method=self.ar_method,
            **kwargs,
        ).to(self.device)
        self._forecast_key = key

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        self._ensure_forecast(context.shape[1], horizon)
        out: list[np.ndarray] = []
        for chunk in batcher(list(context), batch_size):
            preds = self._forecast.predict([row for row in chunk])  # [b, Q, >=H]
            out.append(np.asarray(preds, dtype=np.float32)[:, :, :horizon])
        return np.concatenate(out, axis=0).astype(np.float32)
