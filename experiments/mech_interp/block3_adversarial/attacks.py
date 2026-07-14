"""Support-restricted adversarial attacks on the forecast context.

Only the **context** is ever perturbed — the horizon is the label and is never
touched. The whole point of this module is the *support mask*: the attacker gets
the same L_inf budget but may only spend it on a chosen slice of the context
(the last k points, the first k, a random k, ...). Comparing damage across
supports at matched budget is the actual test of "points near the forecast
boundary are most vulnerable".

Budget: ``eps_abs = eps * sigma_ctx`` per series, where ``sigma_ctx`` is the std
of the clean context. The paper writes ``eps* = eps * var(x)``; we deliberately
use **std, not var**, because var is dimensionally wrong for an L_inf bound on
the signal (it scales quadratically with the series' units) and makes eps
non-comparable across datasets of different scale. See notes.md.
"""
from __future__ import annotations

import numpy as np
import torch

SUPPORTS = ("last", "first", "random", "mid", "all")


def support_mask(
    n: int,
    ctx: int,
    kind: str,
    ratio: float,
    rng: np.random.Generator,
    device: str | torch.device,
) -> torch.Tensor:
    """``[n, ctx]`` float mask; 1 where the attacker may perturb."""
    k = max(1, int(round(ratio * ctx)))
    m = torch.zeros(n, ctx, device=device)
    if kind == "all":
        m[:] = 1.0
    elif kind == "last":  # nearest the forecast boundary
        m[:, ctx - k :] = 1.0
    elif kind == "first":
        m[:, :k] = 1.0
    elif kind == "mid":
        s = (ctx - k) // 2
        m[:, s : s + k] = 1.0
    elif kind == "random":
        idx = torch.stack(
            [
                torch.from_numpy(rng.choice(ctx, size=k, replace=False))
                for _ in range(n)
            ]
        ).to(device)
        m.scatter_(1, idx, 1.0)
    else:
        raise ValueError(f"unknown support {kind!r} (expected one of {SUPPORTS})")
    return m


def attack_loss(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    horizon: int,
    targeted: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-series loss ``[n]``.

    Untargeted: MAE(median, ground truth) — the attacker *maximizes* this.
    Targeted: MAE(median, target) — the attacker *minimizes* this.
    """
    med = model.median(x, horizon)  # [n, H]
    ref = y if targeted is None else targeted
    return (med - ref).abs().mean(dim=1)


def pgd(
    model,
    x0: torch.Tensor,
    y: torch.Tensor,
    horizon: int,
    eps_abs: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 10,
    targeted: torch.Tensor | None = None,
) -> torch.Tensor:
    """Projected gradient descent inside (L_inf ball ∩ support). Returns ``x_adv``.

    ``steps=1`` with the full budget as the step size is exactly FGSM.
    ``eps_abs``: ``[n, 1]`` per-series bound. ``mask``: ``[n, ctx]`` in {0, 1}.
    """
    alpha = eps_abs * (2.5 / steps) if steps > 1 else eps_abs
    sign = 1.0 if targeted is None else -1.0  # ascend the loss / descend to target
    delta = torch.zeros_like(x0)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = attack_loss(model, x0 + delta, y, horizon, targeted).sum()
        (g,) = torch.autograd.grad(loss, delta)
        delta = delta.detach() + sign * alpha * g.sign()
        # Project: clamp into the L_inf ball, THEN zero outside the support.
        delta = torch.clamp(delta, -eps_abs, eps_abs) * mask
    return (x0 + delta).detach()


def fgsm(model, x0, y, horizon, eps_abs, mask, targeted=None) -> torch.Tensor:
    """Single full-budget signed-gradient step."""
    return pgd(
        model, x0, y, horizon, eps_abs, mask, steps=1, targeted=targeted
    )


def random_perturb(
    x0: torch.Tensor, eps_abs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """CONTROL: same budget, same support, no gradient (random signs).

    If this is nearly as damaging as PGD, the attack is not exploiting anything
    and the whole positional result is noise. Never omit it.
    """
    s = torch.randint(0, 2, x0.shape, device=x0.device, dtype=x0.dtype) * 2 - 1
    return (x0 + (s * eps_abs) * mask).detach()


def make_target(
    med_clean: torch.Tensor, sigma: torch.Tensor, kind: str
) -> torch.Tensor:
    """Targeted-attack goals, built from the model's own clean forecast."""
    H = med_clean.shape[1]
    tau = torch.arange(H, device=med_clean.device, dtype=med_clean.dtype)
    if kind == "flip":  # reflect the forecast about its own mean
        return -med_clean + 2 * med_clean.mean(1, keepdim=True)
    if kind == "drift":  # add a +1 sigma ramp across the horizon
        return med_clean + sigma[:, None] * tau / H
    if kind == "amp":  # double the deviation from the forecast mean
        m = med_clean.mean(1, keepdim=True)
        return m + 2.0 * (med_clean - m)
    raise ValueError(f"unknown target kind {kind!r}")


def run_attack(
    model,
    x0: torch.Tensor,
    y: torch.Tensor,
    horizon: int,
    attack: str,
    eps_abs: torch.Tensor,
    mask: torch.Tensor,
    targeted: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch by attack name (``fgsm`` | ``pgd<N>`` | ``random``)."""
    if attack == "random":
        return random_perturb(x0, eps_abs, mask)
    if attack == "fgsm":
        return fgsm(model, x0, y, horizon, eps_abs, mask, targeted)
    if attack.startswith("pgd"):
        steps = int(attack[3:]) if attack[3:] else 10
        return pgd(model, x0, y, horizon, eps_abs, mask, steps=steps, targeted=targeted)
    raise ValueError(f"unknown attack {attack!r}")
