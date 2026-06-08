"""
Adapter for PatchTST-FM (IBM Granite TSFM).

Reuses the canonical ``PatchTSTFMEvalPredictor`` wrapper that ships with the
granite-tsfm demo (``granite-tsfm/notebooks/hfdemo/patchtst_fm/``), exactly as
gift-eval's ``patchtst_fm`` notebook does. We build gluonts-style ``DataEntry``
dicts from each context window, run the predictor, and convert the returned
gluonts ``Forecast`` objects to ``[n, Q, H]`` via
:func:`~.base.gluonts_forecasts_to_qarray`.

Requires ``tsfm_public`` plus the demo predictor module on ``sys.path`` — pass
``predictor_path`` (the dir containing ``patchtst_fm_predictor.py``). Not
installed in the uni2ts venv; run in an env that has it.
"""
from __future__ import annotations

import sys

import numpy as np

from .base import ForecastAdapter, batcher, gluonts_forecasts_to_qarray

# Synthetic stress series have no real timestamp; gluonts only needs a valid,
# consistent (start, freq) pair, so a constant placeholder is fine.
_PLACEHOLDER_FREQ = "H"
_PLACEHOLDER_START = "2020-01-01"


class PatchTSTFMAdapter(ForecastAdapter):
    def __init__(
        self,
        ckpt_path: str = "ibm-granite/granite-timeseries-patchtst-fm-r1",
        predictor_path: str | None = None,
        device: str = "cuda",
        dataset_name: str = "stress",
        **_ignored,
    ):
        from tsfm_public import PatchTSTFMForPrediction

        if predictor_path and predictor_path not in sys.path:
            sys.path.append(predictor_path)
        from patchtst_fm_predictor import PatchTSTFMEvalPredictor

        self._model = PatchTSTFMForPrediction.from_pretrained(
            ckpt_path, device_map=device
        )
        self._PatchTSTFMEvalPredictor = PatchTSTFMEvalPredictor
        self.dataset_name = dataset_name

    def _data_entries(self, context: np.ndarray):
        import pandas as pd

        start = pd.Period(_PLACEHOLDER_START, freq=_PLACEHOLDER_FREQ)
        return [
            {"target": row.astype(np.float32), "start": start, "item_id": str(i)}
            for i, row in enumerate(context)
        ]

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 256,
    ) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        predictor = self._PatchTSTFMEvalPredictor(
            model=self._model,
            prediction_length=horizon,
            dataset_name=self.dataset_name,
            quantile_levels=list(self.quantile_levels),
        )
        out: list[np.ndarray] = []
        for chunk in batcher(list(context), batch_size):
            forecasts = predictor.predict(
                self._data_entries(chunk), batch_size=batch_size
            )
            out.append(
                gluonts_forecasts_to_qarray(forecasts, self.quantile_levels, horizon)
            )
        return np.concatenate(out, axis=0).astype(np.float32)
