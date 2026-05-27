from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .synthetic import PERIOD_BINS as _DEFAULT_PERIOD_BINS


@runtime_checkable
class LabelGenerator(Protocol):
    """
    A label generator maps a 1D time series window to a dict of scalar labels.

    Hyperparameters (e.g. period_bins) are constructor arguments so that
    wrap_existing_dataset can call all generators uniformly via __call__.
    """

    def __call__(self, series: np.ndarray) -> dict[str, np.ndarray]:
        ...


class TrendLabelGenerator:
    """
    Estimate linear trend slope via least-squares on a normalized time axis.

    The time axis is normalized to [0, 1] so the slope magnitude is comparable
    across series of different lengths. Note: the returned slope is in units of
    value-per-normalized-step, which differs from the raw-time slope stored in
    generate_dataset() by a factor of series_length.
    """

    def __call__(self, series: np.ndarray) -> dict[str, np.ndarray]:
        s = series.astype(np.float64)
        t = np.arange(len(s), dtype=np.float64) / len(s)
        slope = np.polyfit(t, s, 1)[0]
        return {"slope": np.float32(slope)}


class SeasonalLabelGenerator:
    """
    Estimate dominant period and phase via the real FFT.

    The dominant non-DC frequency determines the period in time steps, which is
    mapped to the nearest entry in period_bins (also in time steps). Phase is
    extracted from the angle of the dominant FFT coefficient.

    Output keys match synthetic ground-truth: period_idx, phase_cos, phase_sin.
    """

    def __init__(
        self,
        period_bins: list[int] | None = None,
    ) -> None:
        self.period_bins = period_bins if period_bins is not None else _DEFAULT_PERIOD_BINS
        self._bins = np.array(self.period_bins, dtype=np.float64)

    def __call__(self, series: np.ndarray) -> dict[str, np.ndarray]:
        s = series.astype(np.float64)
        fft = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(len(s))
        dominant_i = int(np.argmax(np.abs(fft[1:])))
        dominant_freq = freqs[1 + dominant_i]
        period_ts = 1.0 / dominant_freq  # period in time steps
        period_idx = int(np.argmin(np.abs(self._bins - period_ts)))
        phase = float(np.angle(fft[1 + dominant_i]))
        return {
            "period_idx": np.int32(period_idx),
            "phase_cos": np.float32(np.cos(phase)),
            "phase_sin": np.float32(np.sin(phase)),
        }


class NoiseVarLabelGenerator:
    """
    Estimate log noise variance after removing a linear trend.

    Fits a degree-1 polynomial to the series and computes the log variance of
    the residuals, matching the log_noise_var label from generate_dataset().
    """

    def __call__(self, series: np.ndarray) -> dict[str, np.ndarray]:
        s = series.astype(np.float64)
        t = np.arange(len(s), dtype=np.float64)
        coeffs = np.polyfit(t, s, 1)
        residuals = s - np.polyval(coeffs, t)
        return {"log_noise_var": np.float32(np.log(np.var(residuals) + 1e-8))}


# To add custom label generators, implement the LabelGenerator protocol:
#   class MyGenerator:
#       def __call__(self, series: np.ndarray) -> dict[str, np.ndarray]: ...
# Constructor kwargs should capture any hyperparameters.


DEFAULT_GENERATORS: list[LabelGenerator] = [
    TrendLabelGenerator(),
    SeasonalLabelGenerator(),
    NoiseVarLabelGenerator(),
]
