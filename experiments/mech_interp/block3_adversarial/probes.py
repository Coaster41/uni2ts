"""Positional sensitivity probes — the headline measurement.

Two independent estimates of "how much does context position i matter":

1. :func:`saliency` — white-box ``|dL/dx_i|``. Cheap (one backward pass) but can
   be misleading: gradients flow through the instance-norm scaler, and gradient
   masking is a known failure mode.
2. :func:`bump_profile` — black-box finite difference. Perturb position i by
   ``±kappa*sigma`` and measure how far the median forecast moves. Model-agnostic,
   so it also covers chronos2 / timesfm25, which we do not backprop through.

The two must **agree in shape** where both are computable. If they disagree,
trust the bump probe: finite differences cannot be fooled by gradient masking.
"""
from __future__ import annotations

import numpy as np
import torch

from .attacks import attack_loss


def saliency(
    model, x: torch.Tensor, y: torch.Tensor, horizon: int, batch_size: int = 64
) -> np.ndarray:
    """``|dL/dx|`` per context position. ``[n, ctx]``."""
    outs = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].clone().requires_grad_(True)
        yb = y[i : i + batch_size]
        loss = attack_loss(model, xb, yb, horizon).sum()
        (g,) = torch.autograd.grad(loss, xb)
        outs.append(g.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _drop_dead(a: np.ndarray) -> np.ndarray:
    """Drop rows whose gradient is identically zero.

    Such a row carries no positional information (the head is locally flat on
    that window), and max-normalizing it would divide by zero and poison every
    downstream mean with NaN. TimesFMX hits this on ~4% of windows.
    """
    return a[a.max(axis=1) > 0]


def normalize_curve(g: np.ndarray) -> np.ndarray:
    """Per-series max-normalized |g|, averaged over series. ``[n, ctx] -> [ctx]``.

    Normalizing each series by its own max stops one high-variance series from
    dominating the mean curve.
    """
    a = _drop_dead(np.abs(g))
    if not len(a):
        return np.full(g.shape[1], np.nan, dtype=np.float64)
    a = a / a.max(axis=1, keepdims=True)
    return a.mean(axis=0)


def centered_curve(g: np.ndarray) -> np.ndarray:
    """Saliency curve after removing each series' uniform (mean-shift) component.

    CONFOUND CONTROL (HANDOFF §6.1): every model normalizes by context mean/std,
    so a perturbation *anywhere* moves loc/scale and hence the whole forecast. If
    the boundary peak survives mean-centering, it is a genuine positional effect;
    if it vanishes, the "vulnerability" was normalization leverage.
    """
    return normalize_curve(g - g.mean(axis=1, keepdims=True))


def topk_frequency(g: np.ndarray, k: int = 25) -> np.ndarray:
    """How often each position lands in a series' top-k |g|. ``[n, ctx] -> [ctx]``."""
    ctx = g.shape[1]
    idx = np.argsort(-np.abs(g), axis=1)[:, :k]
    counts = np.zeros(ctx, dtype=np.float64)
    np.add.at(counts, idx.ravel(), 1.0)
    return counts / len(g)


def patch_offset_profile(g: np.ndarray, patch_size: int) -> np.ndarray:
    """Mean |g| by within-patch offset ``i mod P``. ``[ctx] -> [P]``.

    CONFOUND CONTROL (HANDOFF §6.3): separates "sensitivity rises smoothly toward
    the boundary" from "sensitivity is a step function over the final patch"
    (which is what a causal next-token readout mechanically predicts).
    """
    a = _drop_dead(np.abs(g))
    a = a / a.max(axis=1, keepdims=True)
    ctx = a.shape[1]
    offs = np.arange(ctx) % patch_size
    return np.array([a[:, offs == o].mean() for o in range(patch_size)])


def patch_index_profile(g: np.ndarray, patch_size: int) -> np.ndarray:
    """Mean |g| by patch index. ``[ctx] -> [ctx // P]``."""
    a = _drop_dead(np.abs(g))
    a = a / a.max(axis=1, keepdims=True)
    n, ctx = a.shape
    return a.reshape(n, ctx // patch_size, patch_size).mean(axis=(0, 2))


def _median_forecast(
    adapter, ctx_np: np.ndarray, horizon: int, batch_size: int, median_idx: int
) -> np.ndarray:
    """Median forecast ``[n, H]`` from any adapter.

    Adapters are supposed to return ``[n, Q, H]``, but chronos-2 emits a
    univariate target axis (``[n, Q, 1, H]``). Flatten whatever is left after
    indexing the quantile axis rather than trusting the rank.
    """
    q = np.asarray(
        adapter.predict_quantiles(ctx_np, horizon, batch_size), dtype=np.float32
    )
    med = q[:, median_idx]
    return med.reshape(med.shape[0], -1)[:, :horizon]


def bump_profile(
    adapter,
    ctx_np: np.ndarray,
    horizon: int,
    kappa: float = 0.5,
    stride: int = 1,
    batch_size: int = 64,
    median_idx: int = 4,
) -> np.ndarray:
    """Finite-difference positional sensitivity. ``[ctx]``.

    For each context position i, perturb it by ``±kappa*sigma_ctx`` and measure
    the mean absolute displacement of the median forecast, in sigma units.

    Costs ``2 * ctx/stride`` batched forwards. The batch size is held constant on
    purpose: timesfm25 is ``torch.compile``d with fixed buffers and recompiles per
    (ctx, horizon, batch) shape, so a varying final batch would trigger a second
    compile per position.
    """
    n, ctx = ctx_np.shape
    sigma = ctx_np.std(axis=1, keepdims=True)  # [n, 1]
    base = _median_forecast(adapter, ctx_np, horizon, batch_size, median_idx)

    prof = np.zeros(ctx, dtype=np.float32)
    for i in range(0, ctx, stride):
        d = np.zeros_like(ctx_np)
        d[:, i] = 1.0
        acc = np.zeros(n, dtype=np.float64)
        for s in (+1.0, -1.0):
            pert = _median_forecast(
                adapter, ctx_np + s * kappa * sigma * d, horizon, batch_size, median_idx
            )
            acc += np.abs(pert - base).mean(axis=1) / sigma[:, 0]
        prof[i : i + stride] = float((acc / 2.0).mean())
    return prof
