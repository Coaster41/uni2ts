"""
Pluggable forecast-adapter registry for the block2_stress stress battery.

Each entry maps a short adapter name to the dotted module path and class of a
:class:`~.base.ForecastAdapter` subclass. Adapters are imported **lazily** via
:func:`load_adapter` so that selecting one backend never drags in another's
heavy dependencies — e.g. running ``chronos2`` in the gift-eval venv does not
import torch/uni2ts, and running ``moiraic`` in the uni2ts venv does not import
chronos.

Add a new model by writing ``models/<name>.py`` with a ``ForecastAdapter``
subclass and registering it below.
"""
from __future__ import annotations

import importlib

from .base import DEFAULT_QUANTILE_LEVELS, ForecastAdapter

_PKG = __name__  # "experiments.mech_interp.block2_stress.models"

# adapter name -> (module suffix, class name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "moiraic": ("custom_moirai", "MoiraicAdapter"),
    "moiraie": ("custom_moirai", "MoiraieAdapter"),
    "moirai2": ("moirai2", "Moirai2Adapter"),
    "chronos2": ("chronos2", "Chronos2Adapter"),
    "timesfm25": ("timesfm25", "TimesFM25Adapter"),
    "patchtst_fm": ("patchtst_fm", "PatchTSTFMAdapter"),
    "toto": ("toto", "TotoAdapter"),
}


def available_adapters() -> list[str]:
    """Names of all registered adapters."""
    return sorted(_REGISTRY)


def load_adapter(adapter: str, **kwargs) -> ForecastAdapter:
    """Instantiate the adapter registered under ``adapter``.

    ``kwargs`` are forwarded to the adapter constructor. The backend module is
    imported only here, so a missing optional dependency surfaces as a clear
    ImportError naming the package, and only when that adapter is actually used.
    """
    if adapter not in _REGISTRY:
        raise KeyError(
            f"Unknown adapter {adapter!r}. Available: {available_adapters()}"
        )
    module_suffix, class_name = _REGISTRY[adapter]
    module = importlib.import_module(f"{_PKG}.{module_suffix}")
    cls = getattr(module, class_name)
    return cls(**kwargs)


__all__ = [
    "ForecastAdapter",
    "DEFAULT_QUANTILE_LEVELS",
    "available_adapters",
    "load_adapter",
]
