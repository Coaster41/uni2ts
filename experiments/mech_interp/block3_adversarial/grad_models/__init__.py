"""Registry of differentiable forecasters for the white-box attack path.

Mirrors ``block2_stress.models.load_adapter`` but returns a
:class:`GradForecaster` (torch in, torch out, graph intact) instead of a numpy
``ForecastAdapter``. Backends are imported lazily.
"""
from __future__ import annotations

import importlib

from .base import (
    DEFAULT_QUANTILE_LEVELS,
    MEDIAN_IDX,
    GradForecaster,
    make_batch_torch,
)

_PKG = __name__

# kind -> (module suffix, class name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "moiraix": ("moirai_grad", "MoiraiGradForecaster"),
    "moirai2": ("moirai_grad", "MoiraiGradForecaster"),
    "moiraic": ("moirai_grad", "MoiraiGradForecaster"),
    "moiraie": ("moirai_grad", "MoiraiGradForecaster"),
    "timesfmx": ("timesfmx_grad", "TimesFMXGradForecaster"),
    # Official TimesFM-2.5 loads into TimesFMXModule with strict=True (checkpoint
    # parity), so the frontier 200M model is white-box too -- it takes no `ckpt`,
    # it pulls its own weights from HF.
    "timesfm25": ("timesfmx_grad", "TimesFM25GradForecaster"),
}
_MOIRAI_KINDS = ("moiraix", "moirai2", "moiraic", "moiraie")
# Kinds that fetch their own weights and take no `ckpt` argument.
_NO_CKPT_KINDS = ("timesfm25",)


def available_kinds() -> list[str]:
    return sorted(_REGISTRY)


def load_grad_forecaster(kind: str, **kwargs) -> GradForecaster:
    """Instantiate the differentiable forecaster registered under ``kind``.

    ``kwargs`` are forwarded to the constructor (``ckpt``, ``device``). The
    moirai backends share one class and disambiguate via a ``kind`` kwarg.
    """
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown kind {kind!r}. Available: {available_kinds()}")
    module_suffix, class_name = _REGISTRY[kind]
    cls = getattr(importlib.import_module(f"{_PKG}.{module_suffix}"), class_name)
    if kind in _MOIRAI_KINDS:
        kwargs.setdefault("kind", kind)
    if kind in _NO_CKPT_KINDS:
        kwargs.pop("ckpt", None)
    return cls(**kwargs)


__all__ = [
    "GradForecaster",
    "make_batch_torch",
    "DEFAULT_QUANTILE_LEVELS",
    "MEDIAN_IDX",
    "available_kinds",
    "load_grad_forecaster",
]
