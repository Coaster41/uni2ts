"""Scale-free adversarial metrics.

Everything is normalized by the clean context std (``sigma``) so numbers are
comparable across datasets whose raw scales differ by orders of magnitude.

Aggregation warning: **aggregate RED_E with the median, not the mean.** E_clean
sits in the denominator and is near zero on easy series, so the mean is dominated
by a handful of huge ratios and the resulting table is meaningless. Report median
+ IQR.
"""
from __future__ import annotations

import numpy as np


def smae(med: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Scale-free MAE. ``[n,H],[n,H],[n] -> [n]``."""
    return np.abs(med - y).mean(axis=1) / (sigma + 1e-8)


def wql(q: np.ndarray, y: np.ndarray, levels) -> np.ndarray:
    """Weighted quantile loss. ``[n,Q,H],[n,H] -> [n]``."""
    e = y[:, None, :] - q
    lv = np.asarray(levels, dtype=np.float64)[None, :, None]
    num = 2.0 * np.maximum(lv * e, (lv - 1.0) * e).sum(axis=(1, 2))
    return num / (np.abs(y).sum(axis=1) + 1e-8)


def red_e(e_adv: np.ndarray, e_clean: np.ndarray) -> np.ndarray:
    """Relative error degradation, per series.

    The scale normalizer cancels in the ratio, so this is well defined for any
    positive per-series error. Values > 0 mean the attack hurt.
    """
    return (e_adv - e_clean) / (e_clean + 1e-8)


def displacement(
    med_adv: np.ndarray, med_clean: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Ground-truth-free 'how far did the forecast move', in sigma units."""
    return np.abs(med_adv - med_clean).mean(axis=1) / (sigma + 1e-8)


def targeted_red(
    e_adv_to_target: np.ndarray, e_clean_to_target: np.ndarray
) -> np.ndarray:
    """Progress toward the attacker's target. 1.0 = target hit, <0 = moved away."""
    return (e_clean_to_target - e_adv_to_target) / (e_clean_to_target + 1e-8)


def gradient_mass_last(g_abs: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Share of total |grad| mass in the last ``frac`` of the context. ``[n,ctx] -> [n]``.

    Uniform sensitivity gives exactly ``frac`` (0.10 by default), so this is a
    boundary-concentration index that needs no attack at all.
    """
    ctx = g_abs.shape[1]
    k = max(1, int(round(frac * ctx)))
    total = g_abs.sum(axis=1) + 1e-12
    return g_abs[:, ctx - k :].sum(axis=1) / total


def bvi(red_last: np.ndarray, red_random: np.ndarray) -> float:
    """Boundary Vulnerability Index = median RED_E(last) / median RED_E(random).

    > 1 => boundary points carry disproportionate leverage (the paper's claim).
    ~ 1 => position does not matter.
    """
    a = float(np.median(red_last))
    b = float(np.median(red_random))
    return a / (b + 1e-8)


def iqr(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))
