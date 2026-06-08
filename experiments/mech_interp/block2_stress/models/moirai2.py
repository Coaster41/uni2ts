"""
Adapter for Moirai 2.0 (``uni2ts.model.moirai2``).

Moirai2 ships with the canonical decile grid by default and a convenient
``Moirai2Forecast.predict(List[np.ndarray]) -> [n, num_quantiles, H, *tgt]``
method, so no gluonts conversion is needed — we just stack and squeeze the
trailing univariate dimension.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastAdapter, batcher


class Moirai2Adapter(ForecastAdapter):
    def __init__(
        self,
        model_path: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 256,
        device: str = "cpu",
        **_ignored,
    ):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        self.model_path = model_path
        self.context_length = context_length
        self.device = device
        self._module = Moirai2Module.from_pretrained(model_path)
        self.quantile_levels = tuple(self._module.quantile_levels)
        self._Moirai2Forecast = Moirai2Forecast

    def _make_model(self, horizon: int):
        model = self._Moirai2Forecast(
            module=self._module,
            prediction_length=horizon,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        return model.to(self.device)

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        self.context_length = context.shape[1]
        model = self._make_model(horizon)

        out: list[np.ndarray] = []
        for chunk in batcher(list(context), batch_size):
            past_target = [row.astype(np.float32) for row in chunk]
            preds = np.asarray(model.predict(past_target))  # [b, Q, H, *tgt]
            if preds.ndim == 4:
                preds = preds[..., 0]  # squeeze univariate target dim -> [b, Q, H]
            out.append(preds[..., :horizon])
        return np.concatenate(out, axis=0).astype(np.float32)
