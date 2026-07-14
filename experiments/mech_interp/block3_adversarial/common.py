"""Shared config plumbing for block3.

Checkpoint paths are *not* duplicated in ``configs/adv.yaml`` — they are read
from ``block2_stress/configs/models.yaml``, which is the single source of truth
for every model in this project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[2]  # .../uni2ts
ADV_CONFIG = _HERE / "configs" / "adv.yaml"
MODELS_CONFIG = _HERE.parent / "block2_stress" / "configs" / "models.yaml"
STRESS_DIR = _HERE.parent / "block2_stress" / "data" / "stress"
DATA_DIR = _HERE / "data" / "adv"
RESULTS_DIR = _HERE / "results"


def load_adv_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or ADV_CONFIG) as f:
        return yaml.safe_load(f)


def load_model_specs(path: str | Path | None = None) -> dict[str, dict]:
    with open(path or MODELS_CONFIG) as f:
        return yaml.safe_load(f)["models"]


def model_spec(name: str, specs: dict[str, dict] | None = None) -> dict:
    specs = specs if specs is not None else load_model_specs()
    if name not in specs:
        raise KeyError(f"model {name!r} not in models.yaml (have: {sorted(specs)})")
    return specs[name]


def grad_forecaster_for(name: str, cfg: dict, device: str = "cuda:0"):
    """Build the differentiable forecaster for a white-box model name."""
    from .grad_models import _NO_CKPT_KINDS, load_grad_forecaster

    spec = model_spec(name)
    kind = cfg["grad_kind"][name]
    if kind in _NO_CKPT_KINDS:  # fetches its own weights from HF
        return load_grad_forecaster(kind, device=device)
    ckpt = spec.get("ckpt") or spec.get("model_path")
    if ckpt is None:
        raise ValueError(f"no ckpt/model_path for {name!r} in models.yaml")
    return load_grad_forecaster(kind, ckpt=ckpt, device=device)


def adapter_for(name: str, device: str = "cuda:0", **overrides):
    """Build the block2 numpy adapter for any model name (white- or black-box).

    ``overrides`` are forwarded to the adapter constructor — used to expose the raw
    point head on adapters that polish it at inference (see parity_check).
    """
    from experiments.mech_interp.block2_stress.models import load_adapter

    spec = dict(model_spec(name))
    adapter = spec.pop("adapter")
    spec.setdefault("device", device)
    spec.update(overrides)
    return load_adapter(adapter, **spec)
