"""
PR-1: Stress-test synthetic generators.

Produces three families of controlled synthetic time series:
  Family A — Structure-commitment: SNR sweep with periodic and trend carriers + white-noise anchor
  Family B — Parroting: phi-sweep AR(1), outlier-delta sweep, triangle diagnostic
  Family C — Style: intermittent events, random-amplitude periodic

All series are stored as float32[n, T] where T = (context_patches + horizon_patches) * patch_len.
Ground-truth metadata is stored alongside each file.

Usage (CLI):
    python -m experiments.mech_interp.block2_stress.generators \
        --config experiments/mech_interp/block2_stress/configs/stress.yaml \
        --output-dir experiments/mech_interp/block2_stress/data/stress

Programmatic:
    from experiments.mech_interp.block2_stress import generate_all, load_stress_dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.lib.synthetic import (
    component_ar1,
    load_dataset,
    save_dataset,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(cfg_path: str) -> dict[str, Any]:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Seed discipline
# ---------------------------------------------------------------------------

def _series_seed(global_seed: int, family: str, level_idx: int, series_id: int) -> int:
    """Deterministic per-series seed derived from (global_seed, family, level_idx, series_id)."""
    key = f"{global_seed}:{family}:{level_idx}:{series_id}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31)


def _make_rngs(n: int, global_seed: int, family: str, level_idx: int) -> list[np.random.Generator]:
    return [np.random.default_rng(_series_seed(global_seed, family, level_idx, i)) for i in range(n)]


# ---------------------------------------------------------------------------
# Family A — Structure-commitment (SNR sweep)
# ---------------------------------------------------------------------------

def gen_a_periodic(
    cfg: dict,
    snr: float,
    level_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Periodic carrier: x_t = sin(2π t/P + φ) + σ·ε_t.

    Signal power = 0.5 (unit-amplitude sine); σ = sqrt(0.5 / SNR).
    Each period in cfg.family_a.period_bins gets exactly n_per_level_per_period series.
    φ ~ Uniform[0, 2π) per series.
    """
    n_per_period = cfg["n_per_level_per_period"]
    period_bins = cfg["family_a"]["period_bins"]
    n = n_per_period * len(period_bins)
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    sigma = float(np.sqrt(0.5 / snr))
    t = np.arange(T, dtype=np.float32)

    period_ts = np.repeat(period_bins, n_per_period).astype(np.float32)  # [n], balanced
    rngs = _make_rngs(n, cfg["seed"], "a_periodic", level_idx)
    phases = np.array([r.uniform(0.0, 2 * np.pi) for r in rngs], dtype=np.float32)
    noises = np.array([
        r.standard_normal(T).astype(np.float32) for r in rngs
    ])  # [n, T]

    angles = 2 * np.pi / period_ts[:, None] * t[None, :] + phases[:, None]
    series = (np.sin(angles) + sigma * noises).astype(np.float32)

    meta = {
        "period_ts": period_ts,
        "phase": phases,
        "sigma": np.full(n, sigma, dtype=np.float32),
        "snr": np.full(n, snr, dtype=np.float32),
    }
    return series, meta


def gen_a_trend(
    cfg: dict,
    snr: float,
    level_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Aperiodic trend carrier: x_t = sign * (t - t_mid) / T_half + intercept + σ·ε_t.

    The trend has unit signal power over the window; σ = 1 / sqrt(SNR).
    sign and intercept are randomized per series.
    """
    n = cfg["n_per_level_per_period"]
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    sigma = float(1.0 / np.sqrt(snr))
    t = np.arange(T, dtype=np.float32)
    t_mid = (T - 1) / 2.0
    t_half = (T - 1) / 2.0

    rngs = _make_rngs(n, cfg["seed"], "a_trend", level_idx)
    signs = np.array([r.choice([-1.0, 1.0]) for r in rngs], dtype=np.float32)
    intercepts = np.array([r.uniform(-0.5, 0.5) for r in rngs], dtype=np.float32)
    noises = np.array([r.standard_normal(T).astype(np.float32) for r in rngs])

    carrier = signs[:, None] * (t[None, :] - t_mid) / t_half  # [n, T], unit amplitude range
    series = (carrier + intercepts[:, None] + sigma * noises).astype(np.float32)

    meta = {
        "slope_sign": signs,
        "intercept": intercepts,
        "sigma": np.full(n, sigma, dtype=np.float32),
        "snr": np.full(n, snr, dtype=np.float32),
    }
    return series, meta


def gen_a_white_noise(cfg: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    White-noise anchor: x_t = σ·ε_t, σ ~ LogUniform(0.5, 2.0).

    Used as the SNR→0 no-structure endpoint. The varied sigma ensures
    instance normalization sees diverse scaling but structure is absent.
    """
    n = cfg["n_per_level_per_period"]
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]

    rngs = _make_rngs(n, cfg["seed"], "a_white_noise", 0)
    log_sigmas = np.array([r.uniform(np.log(0.5), np.log(2.0)) for r in rngs], dtype=np.float32)
    sigmas = np.exp(log_sigmas)
    series = np.array([
        (sigmas[i] * rngs[i].standard_normal(T)).astype(np.float32)
        for i in range(n)
    ])

    meta = {"sigma": sigmas}
    return series, meta


# ---------------------------------------------------------------------------
# Family B — Parroting
# ---------------------------------------------------------------------------

def gen_b_phi(
    cfg: dict,
    phi: float,
    level_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    AR(1) phi-sweep: x_t = phi * x_{t-1} + sigma * ε_t, fixed phi per level.

    Records analytic optimal continuation: phi^k * x_{L-1} for k=1..H,
    where L = context_patches * patch_len and H = horizon_patches * patch_len.
    """
    n = cfg["n_per_level_per_period"]
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    ctx_len = cfg["context_patches"] * cfg["patch_len"]
    H = cfg["horizon_patches"] * cfg["patch_len"]
    sigma = cfg["family_b"]["phi_fixed_sigma"]

    start_value      = float(cfg["family_b"].get("phi_start_value", 0.0))
    start_noise      = float(cfg["family_b"].get("phi_start_noise_sigma", 0.0))
    restart_prob     = float(cfg["family_b"].get("phi_restart_prob", 0.0))
    restart_noise    = float(cfg["family_b"].get("phi_restart_noise_sigma", 0.0))

    rngs = _make_rngs(n, cfg["seed"], "b_phi", level_idx)
    series = np.zeros((n, T), dtype=np.float32)
    for i, rng in enumerate(rngs):
        eps = rng.standard_normal(T).astype(np.float32)
        series[i, 0] = start_value + start_noise * float(rng.standard_normal())
        for t in range(1, T):
            if restart_prob > 0.0 and rng.random() < restart_prob:
                series[i, t] = start_value + restart_noise * float(rng.standard_normal())
            else:
                series[i, t] = phi * series[i, t - 1] + sigma * eps[t]

    x_origin = series[:, ctx_len - 1]  # last context value
    ks = np.arange(1, H + 1, dtype=np.float32)
    optimal_continuation = x_origin[:, None] * (float(phi) ** ks)[None, :]  # [n, H]

    meta = {
        "phi": np.full(n, phi, dtype=np.float32),
        "sigma": np.full(n, sigma, dtype=np.float32),
        "x_origin": x_origin,
        "optimal_continuation": optimal_continuation,
    }
    return series, meta


def gen_b_outlier(
    cfg: dict,
    delta: float,
    level_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Outlier delta-sweep: smooth AR(1) base (phi=0.3) + spike of size delta*context_std
    injected at the final context timestep (index ctx_len - 1).
    """
    n = cfg["n_per_level_per_period"]
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    ctx_len = cfg["context_patches"] * cfg["patch_len"]
    base_sigma = 0.1   # low-noise smooth base
    base_phi = 0.3

    rngs = _make_rngs(n, cfg["seed"], "b_outlier", level_idx)
    base = np.zeros((n, T), dtype=np.float32)
    for i, rng in enumerate(rngs):
        eps = rng.standard_normal(T).astype(np.float32)
        for t in range(1, T):
            base[i, t] = base_phi * base[i, t - 1] + base_sigma * eps[t]

    context_std = base[:, :ctx_len].std(axis=1).clip(min=1e-6)  # [n]
    injected_delta = delta * context_std  # absolute spike magnitude per series

    series = base.copy()
    series[:, ctx_len - 1] += injected_delta

    meta = {
        "base_level": base[:, ctx_len - 1] - injected_delta,  # pre-spike value
        "injected_delta": injected_delta,
        "context_std": context_std,
        "delta_units": np.full(n, delta, dtype=np.float32),
    }
    return series, meta


def gen_b_triangle(
    cfg: dict,
    period_idx: int,
    level_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Triangle wave (fixed diagnostic): piecewise-linear with period P sampled from
    cfg.family_b.triangle_period_bins[period_idx]. Apex placed inside the horizon.

    The wave has unit amplitude. Low noise (sigma=0.05) added.
    meta includes apex_h: step within horizon where downward turn begins (0-indexed).
    """
    n = cfg["n_per_level_per_period"]
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    ctx_len = cfg["context_patches"] * cfg["patch_len"]
    H = cfg["horizon_patches"] * cfg["patch_len"]
    noise_sigma = 0.05
    period_bins = cfg["family_b"]["triangle_period_bins"]
    P = period_bins[period_idx % len(period_bins)]

    rngs = _make_rngs(n, cfg["seed"], "b_triangle", level_idx)

    t = np.arange(T, dtype=np.float32)
    # Piecewise-linear triangle: sawtooth with absolute value = triangle
    # triangle(t, P) = 2/P * |((t mod P) - P/2)| - ... → simpler: (2/P)*|(t mod P) - P/2| * 2 - 1
    # We use: tri[t] = 1 - 2*abs((t mod P)/P - 0.5) * 2  (ranges -1 to 1)
    t_mod = (t[None, :] % P) / P  # [1, T] broadcast
    carrier = (1.0 - 4.0 * np.abs(t_mod - 0.5)).astype(np.float32)  # [1, T], ±1

    # Per-series: randomize phase offset (shift in time) so apex lands in horizon
    # apex_h: horizon step (0-indexed) where apex occurs
    series = np.zeros((n, T), dtype=np.float32)
    apex_hs = np.zeros(n, dtype=np.int32)
    period_ts = np.full(n, P, dtype=np.float32)

    for i, rng in enumerate(rngs):
        # Find a phase so the apex (maximum of the triangle) lands in horizon
        # Maximum of carrier occurs at t where t mod P == 0 (i.e., 4*(0.5 - 0) = 2, wait)
        # Actually carrier = 1 - 4*|t_mod - 0.5|; max=1 at t_mod=0.5 i.e. t = P/2 + k*P
        # We want t_apex ∈ [ctx_len, ctx_len + H), i.e. in [0, H) of the horizon
        apex_frac = rng.uniform(0.25, 0.75)
        apex_in_horizon = int(apex_frac * H)  # 0-indexed horizon step
        t_apex_global = ctx_len + apex_in_horizon
        # phase offset: t_apex_global should satisfy (t_apex_global - offset) mod P == P/2
        offset = (t_apex_global - P / 2) % P
        t_shifted = (t - offset) % P / P
        c = (1.0 - 4.0 * np.abs(t_shifted - 0.5)).astype(np.float32)
        noise = rng.standard_normal(T).astype(np.float32) * noise_sigma
        series[i] = c + noise
        apex_hs[i] = apex_in_horizon

    meta = {
        "period_ts": period_ts,
        "apex_h": apex_hs,
    }
    return series, meta


# ---------------------------------------------------------------------------
# Family C — Style
# ---------------------------------------------------------------------------

def gen_c_intermittent(
    cfg: dict,
    p: float,
    level_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Intermittent events: x_t = Σ_k b_k * bump(t - k*P) + ε_t.
    b_k ~ Bernoulli(p) per cycle; bump = half-sine over bump_half_width timesteps.
    Each period in cfg.family_a.period_bins gets exactly n_per_level_per_period series.
    """
    n_per_period = cfg["n_per_level_per_period"]
    period_bins = cfg["family_a"]["period_bins"]
    n = n_per_period * len(period_bins)
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    half_width_frac = cfg["family_c"]["bump_half_width_frac"]
    noise_sigma = cfg["family_c"]["base_noise_sigma"]

    all_period_ts = np.repeat(period_bins, n_per_period).astype(np.float32)  # [n], balanced
    rngs = _make_rngs(n, cfg["seed"], "c_intermittent", level_idx)
    series = np.zeros((n, T), dtype=np.float32)

    for i, rng in enumerate(rngs):
        P = int(all_period_ts[i])
        half_width = max(1, int(half_width_frac * P))
        bump_t = np.arange(half_width, dtype=np.float32)
        bump = np.sin(np.pi * bump_t / half_width).astype(np.float32)  # half-sine, peak=1

        n_cycles = T // P + 1
        x = np.zeros(T, dtype=np.float32)
        for k in range(n_cycles):
            if rng.random() < p:
                start = k * P
                end = min(start + half_width, T)
                blen = end - start
                x[start:end] += bump[:blen]

        noise = rng.standard_normal(T).astype(np.float32) * noise_sigma
        series[i] = x + noise
        all_period_ts[i] = float(P)

    meta = {
        "period_ts": all_period_ts,
        "p": np.full(n, p, dtype=np.float32),
    }
    return series, meta


def gen_c_rand_amp(
    cfg: dict,
    var_level: float,
    level_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Random-amplitude periodic: x_t = A_{k(t)} * sin(2π t/P) + ε_t.
    A_k ~ LogNormal(0, sqrt(var_level)) i.i.d. per cycle.
    Each period in cfg.family_a.period_bins gets exactly n_per_level_per_period series.
    """
    n_per_period = cfg["n_per_level_per_period"]
    period_bins = cfg["family_a"]["period_bins"]
    n = n_per_period * len(period_bins)
    T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
    noise_sigma = cfg["family_c"]["base_noise_sigma"]

    all_period_ts = np.repeat(period_bins, n_per_period).astype(np.float32)  # [n], balanced
    rngs = _make_rngs(n, cfg["seed"], "c_rand_amp", level_idx)
    series = np.zeros((n, T), dtype=np.float32)

    for i, rng in enumerate(rngs):
        P = int(all_period_ts[i])
        n_cycles = T // P + 1
        # Draw one amplitude per cycle
        log_amps = rng.normal(0.0, float(np.sqrt(var_level)), size=n_cycles).astype(np.float32)
        amps = np.exp(log_amps)

        t = np.arange(T, dtype=np.float32)
        cycle_idx = (t / P).astype(np.int32).clip(0, n_cycles - 1)  # [T]
        amp_per_t = amps[cycle_idx]  # [T]
        phase = rng.uniform(0.0, 2 * np.pi)
        x = amp_per_t * np.sin(2 * np.pi * t / P + phase).astype(np.float32)
        noise = rng.standard_normal(T).astype(np.float32) * noise_sigma
        series[i] = x + noise
        all_period_ts[i] = float(P)

    meta = {
        "period_ts": all_period_ts,
        "amp_var": np.full(n, var_level, dtype=np.float32),
    }
    return series, meta


# ---------------------------------------------------------------------------
# generate_all / load_stress_dataset
# ---------------------------------------------------------------------------

def _save(data: dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dataset(data, path)


def generate_all(cfg_path: str, output_dir: str | None = None) -> None:
    """
    Generate all stress-test families and persist to disk as .npz files.

    Layout:
        {output_dir}/family_a_periodic/snr_{i:02d}.npz
        {output_dir}/family_a_trend/snr_{i:02d}.npz
        {output_dir}/family_a_white_noise/anchor.npz
        {output_dir}/family_b_phi/phi_{i:02d}.npz
        {output_dir}/family_b_outlier/delta_{i:02d}.npz
        {output_dir}/family_b_triangle/period_{i:02d}.npz
        {output_dir}/family_c_intermittent/p_{i:02d}.npz
        {output_dir}/family_c_rand_amp/var_{i:02d}.npz
        {output_dir}/index.npz
    """
    cfg = load_config(cfg_path)
    cfg_json = json.dumps(cfg, default=str)

    if output_dir is None:
        base = Path(cfg_path).parent.parent  # block2_stress/
        output_dir = str(base / cfg["output_dir"])

    index_entries: list[dict] = []

    def _save_level(family: str, level_key: str, series: np.ndarray, meta: dict) -> None:
        data = {"series": series, "config_json": np.array([cfg_json])}
        data.update({f"meta_{k}": v for k, v in meta.items()})
        path = os.path.join(output_dir, family, f"{level_key}.npz")
        _save(data, path)
        index_entries.append({"family": family, "level_key": level_key, "n": len(series)})
        print(f"  saved {path}  shape={series.shape}")

    print("=== Family A — periodic ===")
    for i, snr in enumerate(cfg["family_a"]["snr_levels"]):
        s, m = gen_a_periodic(cfg, snr, i)
        _save_level("family_a_periodic", f"snr_{i:02d}", s, m)

    print("=== Family A — trend ===")
    for i, snr in enumerate(cfg["family_a"]["snr_levels"]):
        s, m = gen_a_trend(cfg, snr, i)
        _save_level("family_a_trend", f"snr_{i:02d}", s, m)

    if cfg["family_a"].get("ar1_carrier", False):
        print("=== Family A — AR(1) carrier ===")
        for i, snr in enumerate(cfg["family_a"]["snr_levels"]):
            # AR(1) carrier with fixed phi=0.8, SNR-defined sigma
            n = cfg["n_per_level"]
            T = (cfg["context_patches"] + cfg["horizon_patches"]) * cfg["patch_len"]
            phi = 0.8
            # AR(1) stationary variance = sigma^2 / (1 - phi^2)
            # To get SNR = signal_var / noise_var, we control noise_var added on top
            # Here we treat the AR(1) process as the carrier and add extra noise for SNR
            sigma_extra = float(np.sqrt(1.0 / snr))  # extra noise
            rngs = _make_rngs(n, cfg["seed"], "a_ar1", i)
            series = np.zeros((n, T), dtype=np.float32)
            for j, rng in enumerate(rngs):
                eps = rng.standard_normal(T).astype(np.float32) * 0.1  # low-noise AR drive
                noise = rng.standard_normal(T).astype(np.float32) * sigma_extra
                for t in range(1, T):
                    series[j, t] = phi * series[j, t - 1] + eps[t]
                # normalize carrier to unit std, then add extra noise
                std = series[j].std()
                if std > 1e-6:
                    series[j] = series[j] / std
                series[j] += noise
            meta = {"phi": np.full(n, phi, dtype=np.float32), "sigma_extra": np.full(n, sigma_extra, dtype=np.float32), "snr": np.full(n, snr, dtype=np.float32)}
            _save_level("family_a_ar1", f"snr_{i:02d}", series, meta)

    print("=== Family A — white noise anchor ===")
    s, m = gen_a_white_noise(cfg)
    _save_level("family_a_white_noise", "anchor", s, m)

    print("=== Family B — phi sweep ===")
    for i, phi in enumerate(cfg["family_b"]["phi_levels"]):
        s, m = gen_b_phi(cfg, phi, i)
        _save_level("family_b_phi", f"phi_{i:02d}", s, m)

    print("=== Family B — outlier delta sweep ===")
    for i, delta in enumerate(cfg["family_b"]["outlier_deltas"]):
        s, m = gen_b_outlier(cfg, delta, i)
        _save_level("family_b_outlier", f"delta_{i:02d}", s, m)

    print("=== Family B — triangle ===")
    period_bins = cfg["family_b"]["triangle_period_bins"]
    for i in range(len(period_bins)):
        s, m = gen_b_triangle(cfg, i, i)
        _save_level("family_b_triangle", f"period_{i:02d}", s, m)

    print("=== Family C — intermittent ===")
    for i, p in enumerate(cfg["family_c"]["intermittent_p"]):
        s, m = gen_c_intermittent(cfg, p, i)
        _save_level("family_c_intermittent", f"p_{i:02d}", s, m)

    print("=== Family C — random amplitude ===")
    for i, var in enumerate(cfg["family_c"]["rand_amp_var_levels"]):
        s, m = gen_c_rand_amp(cfg, var, i)
        _save_level("family_c_rand_amp", f"var_{i:02d}", s, m)

    # Write index
    families = [e["family"] for e in index_entries]
    level_keys = [e["level_key"] for e in index_entries]
    ns = [e["n"] for e in index_entries]
    index_path = os.path.join(output_dir, "index.npz")
    np.savez_compressed(
        index_path,
        families=np.array(families),
        level_keys=np.array(level_keys),
        n_per_level=np.array(ns),
        config_json=np.array([cfg_json]),
    )
    print(f"\nIndex saved to {index_path}  ({len(index_entries)} files)")


def load_stress_dataset(
    output_dir: str,
    family: str,
    level_key: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """
    Load a single stress-test file.

    Returns
    -------
    series : float32[n, T]
    meta   : dict of meta_* arrays (keys without the 'meta_' prefix)
    cfg    : parsed config dict (from the stored config_json)
    """
    path = os.path.join(output_dir, family, f"{level_key}.npz")
    data = load_dataset(path)
    cfg = json.loads(str(data.pop("config_json")[0]))
    series = data.pop("series")
    meta = {k[len("meta_"):]: v for k, v in data.items() if k.startswith("meta_")}
    return series, meta, cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stress-test synthetic datasets (PR-1).")
    p.add_argument("--config", required=True, help="Path to stress.yaml")
    p.add_argument("--output-dir", default=None, help="Override output_dir from config")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_all(args.config, output_dir=args.output_dir)
