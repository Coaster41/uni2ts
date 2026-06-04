"""
PR-2: Sigmoid fitting, bootstrap CIs, overlay plots, and summary CSV.

All functions are pure (no IO except make_summary_csv and plot_response_curve
with file output). Compatible with notebook use: pass ax= to avoid creating
new figures.
"""
from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
from numpy import ndarray
from scipy.optimize import curve_fit

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
except ImportError:
    plt = None  # type: ignore

from experiments.mech_interp.block2_stress.metrics import aggregate_level


# ---------------------------------------------------------------------------
# Sigmoid model
# ---------------------------------------------------------------------------

def _sigmoid(x: ndarray, x_star: float, beta: float) -> ndarray:
    """R(x) = 1 / (1 + (x_star / x)^beta).

    Increasing in x: R→0 as x→0, R=0.5 at x=x_star, R→1 as x→∞.
    Correct for axes where higher x → better score (SNR, delta, etc.).
    """
    return 1.0 / (1.0 + (x_star / x) ** beta)


def fit_sigmoid(x: ndarray, y: ndarray) -> dict[str, float]:
    """
    Fit R(x) = 1 / (1 + (x_star/x)^beta).

    x_star is the sweep value where score crosses 0.5.
    beta controls sharpness of the transition.

    Parameters
    ----------
    x : [n_levels] — sweep axis values (e.g. SNR values), must be > 0
    y : [n_levels] — aggregated response scores (e.g. median retention)

    Returns
    -------
    dict with keys "x_star", "beta", "r2".
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]

    if len(x) < 2:
        return {"x_star": float("nan"), "beta": float("nan"), "r2": float("nan")}

    try:
        popt, _ = curve_fit(
            _sigmoid, x, y,
            p0=[float(np.median(x)), 1.0],
            bounds=([1e-6, 0.01], [1e6, 100.0]),
            maxfev=5000,
        )
        x_star, beta = float(popt[0]), float(popt[1])
        y_pred = _sigmoid(x, x_star, beta)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    except Exception:
        x_star, beta, r2 = float("nan"), float("nan"), float("nan")

    return {"x_star": x_star, "beta": beta, "r2": r2}


def bootstrap_sigmoid(
    x: ndarray,
    y: ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Bootstrap CIs on x_star and beta.

    Parameters
    ----------
    x : [n_levels] — sweep axis values
    y : [n_levels] — aggregated response scores

    Returns
    -------
    dict with keys "x_star", "x_star_ci", "beta", "beta_ci", "r2".
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base = fit_sigmoid(x, y)

    rng = np.random.default_rng(seed)
    n = len(x)
    x_stars, betas = [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_sigmoid(x[idx], y[idx])
        if np.isfinite(fit["x_star"]) and np.isfinite(fit["beta"]):
            x_stars.append(fit["x_star"])
            betas.append(fit["beta"])

    if len(x_stars) < 10:
        x_star_ci = (float("nan"), float("nan"))
        beta_ci = (float("nan"), float("nan"))
    else:
        x_star_ci = (float(np.percentile(x_stars, 5)), float(np.percentile(x_stars, 95)))
        beta_ci = (float(np.percentile(betas, 5)), float(np.percentile(betas, 95)))

    return {
        "x_star": base["x_star"],
        "x_star_ci": x_star_ci,
        "beta": base["beta"],
        "beta_ci": beta_ci,
        "r2": base["r2"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_response_curve(
    x_vals: list[float],
    encoder_scores: ndarray,    # [n_levels, n_series]
    decoder_scores: ndarray,
    encoder_fit: dict,
    decoder_fit: dict,
    xlabel: str,
    ylabel: str,
    title: str,
    ax=None,
    log_x: bool = False,
) -> "matplotlib.figure.Figure":
    """
    Overlay encoder vs decoder response curves with bootstrap CI bands.

    Parameters
    ----------
    x_vals : sweep axis values, one per level
    encoder_scores : [n_levels, n_series] — moiraie scores
    decoder_scores : [n_levels, n_series] — moiraic scores
    encoder_fit / decoder_fit : output of bootstrap_sigmoid
    ax : optional existing Axes; if None, a new figure is created
    log_x : if True, use a log-scale x axis (recommended for SNR sweeps)

    Returns
    -------
    The Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    x_arr = np.asarray(x_vals, dtype=np.float64)

    def _plot_model(scores, fit, color, label):
        meds, q25s, q75s = [], [], []
        for row in scores:
            med, q25, q75 = aggregate_level(row)
            meds.append(med)
            q25s.append(q25)
            q75s.append(q75)
        meds, q25s, q75s = np.array(meds), np.array(q25s), np.array(q75s)

        ax.errorbar(
            x_arr, meds,
            yerr=[meds - q25s, q75s - meds],
            fmt="o", color=color, label=label, capsize=3, zorder=3,
        )

        if np.isfinite(fit.get("x_star", float("nan"))):
            x_smooth = np.geomspace(x_arr[x_arr > 0].min(), x_arr.max(), 200)
            y_smooth = _sigmoid(x_smooth, fit["x_star"], fit["beta"])
            ax.plot(x_smooth, y_smooth, color=color, linewidth=1.5, zorder=2)

            # CI band
            x_star_lo, x_star_hi = fit.get("x_star_ci", (float("nan"), float("nan")))
            beta_lo, beta_hi = fit.get("beta_ci", (float("nan"), float("nan")))
            if np.isfinite(x_star_lo) and np.isfinite(beta_lo):
                y_lo = _sigmoid(x_smooth, x_star_hi, beta_lo)   # conservative extremes
                y_hi = _sigmoid(x_smooth, x_star_lo, beta_hi)
                ax.fill_between(x_smooth, y_lo, y_hi, alpha=0.15, color=color, zorder=1)

    _plot_model(encoder_scores, encoder_fit, color="#ff7f0e", label="Encoder (moiraie)")
    _plot_model(decoder_scores, decoder_fit, color="#1f77b4", label="Decoder (moiraic)")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def make_summary_csv(results: list[dict], out_path: str) -> None:
    """
    Write results/curves/summary.csv with columns:
    model, family, carrier, x_star, x_star_lo, x_star_hi, beta, beta_lo, beta_hi
    """
    fieldnames = [
        "model", "family", "carrier",
        "x_star", "x_star_lo", "x_star_hi",
        "beta", "beta_lo", "beta_hi",
        "r2",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
