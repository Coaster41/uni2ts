#  Track B gap-covering synthetic families (.claude/HANDOFF_SYNTH_DATA.md §4).
#  Three independent Arrow datasets (one per generator family, so mixture
#  weights can be tuned and leave-one-out ablations run per family):
#
#    synth_backbone_v0  — parametric backbone: {poly/piecewise/exp/logistic
#                         trend} x multi-period harmonic seasonality (stochastic
#                         amp/phase, harmonic decay) x pole-space ARMA, plus
#                         event/impulse and observation layers. Subsumes
#                         ForecastPFN/TimesFM-synth/RealTS; superset of Track
#                         A's synth_stress_v0 (kept separate for ablation).
#    synth_sde_v0       — regime-switching time-inhomogeneous OU (TempoPFN's
#                         single most valuable component).
#    synth_sarima2_v0   — SARIMA-2 superposition: seasonal ARMA base x slow
#                         positive envelope (cross-frequency amplitude
#                         coupling; directly attacks F2).
#
#  Deliberately NOT included: smooth-Gaussian GP draws (KernelSynth already
#  covers that region), chaotic ODEs, symbolic, causal-DAG generators.
#
#  Build:
#    python -m uni2ts.data.builder.lotsa_v1.synth_trackb --dataset synth_backbone_v0 --num-series 1000000
#    python -m uni2ts.data.builder.lotsa_v1.synth_trackb --dataset synth_sde_v0 --num-series 500000
#    python -m uni2ts.data.builder.lotsa_v1.synth_trackb --dataset synth_sarima2_v0 --num-series 300000

import argparse
import os
import shutil
from collections import defaultdict
from functools import partial
from typing import Optional

import datasets
import numpy as np
from scipy.signal import lfilter

from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder
from .synth_stress import LOTSA_FEATURES, MAX_LEN, MIN_LEN, _log_uniform

# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------


def _series_seed(global_seed: int, dataset: str, series_id: int) -> int:
    import hashlib

    key = f"{global_seed}:{dataset}:{series_id}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _pole_arma(rng: np.random.Generator, T: int, min_r=0.7, max_r=0.995) -> np.ndarray:
    """Stationary ARMA draw via pole-space parameterization (unit output std)."""
    denom = np.array([1.0])
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.3:  # real pole
            r = rng.uniform(min_r, max_r) * rng.choice([-1.0, 1.0])
            denom = np.convolve(denom, [1.0, -r])
        else:  # complex pair
            r = rng.uniform(min_r, max_r)
            w = rng.uniform(0.0, np.pi)
            denom = np.convolve(denom, [1.0, -2 * r * np.cos(w), r * r])
    numer = np.array([1.0])
    for _ in range(int(rng.integers(0, 3))):  # MA zeros
        numer = np.convolve(numer, [1.0, rng.uniform(-0.9, 0.9)])
    burn = 256
    x = lfilter(numer, denom, rng.standard_normal(T + burn))[burn:]
    std = x.std()
    return (x / std if std > 1e-8 else x).astype(np.float32)


def _trend(rng: np.random.Generator, T: int, force: bool = False) -> np.ndarray:
    """Trend component (unit-scale). kinds: none/linear/piecewise/exp/logistic."""
    t01 = np.linspace(0.0, 1.0, T, dtype=np.float32)
    kind = int(rng.integers(0, 4)) if force else int(rng.integers(-1, 4))
    sign = float(rng.choice([-1.0, 1.0]))
    if kind < 0:
        return np.zeros(T, dtype=np.float32)
    if kind == 0:  # linear
        return sign * rng.uniform(0.3, 2.0) * t01
    if kind == 1:  # piecewise-linear changepoints
        n_knots = int(rng.integers(2, 6))
        kt = np.concatenate([[0.0], np.sort(rng.uniform(0, 1, n_knots)), [1.0]])
        kv = rng.uniform(-1.5, 1.5, len(kt))
        return np.interp(t01, kt, kv).astype(np.float32)
    if kind == 2:  # exp growth/decay
        factor = _log_uniform(rng, 2.0, 15.0)
        curve = np.expm1(t01 * np.log(factor)) / (factor - 1.0)  # 0->1
        return sign * rng.uniform(0.5, 2.0) * curve.astype(np.float32)
    from scipy.special import expit  # logistic

    t0, width = rng.uniform(0.2, 0.8), rng.uniform(0.03, 0.4)
    return (sign * rng.uniform(0.5, 2.0) * expit((t01 - t0) / (width / 8.0))).astype(
        np.float32
    )


def _seasonality(rng: np.random.Generator, T: int) -> np.ndarray:
    """Multi-period harmonic seasonality with decaying harmonics; may be zero."""
    n_periods = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.28, 0.12]))
    t = np.arange(T, dtype=np.float32)
    x = np.zeros(T, dtype=np.float32)
    for _ in range(n_periods):
        P = _log_uniform(rng, 4.0, 512.0)
        amp = _log_uniform(rng, 0.2, 2.0)
        decay = rng.uniform(0.3, 0.9)
        for h in range(1, int(rng.integers(1, 5)) + 1):
            a_h = amp * decay ** (h - 1) * rng.uniform(0.7, 1.3)
            x += a_h * np.sin(2 * np.pi * h * t / P + rng.uniform(0, 2 * np.pi))
    return x


def _event_layer(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """Spikes / steps / impulse-response events (handoff §4.4)."""
    T = len(y)
    scale = max(float(np.nanstd(y)), 1e-3)
    kind = rng.integers(0, 3)
    if kind == 0:  # spikes
        times = rng.choice(T, size=min(1 + rng.poisson(T / 700), T // 8), replace=False)
        y[times] += rng.choice([-1, 1], len(times)) * rng.uniform(2, 8, len(times)) * scale
    elif kind == 1:  # steps
        for _ in range(int(rng.integers(1, 4))):
            y[int(rng.integers(int(0.05 * T), int(0.95 * T))):] += (
                float(rng.choice([-1.0, 1.0])) * rng.uniform(0.5, 3.0) * scale
            )
    else:  # impulse response: sparse impulses * exponential-decay kernel
        tau = rng.uniform(2.0, 50.0)
        kernel = np.exp(-np.arange(int(5 * tau)) / tau).astype(np.float32)
        impulses = np.zeros(T, dtype=np.float32)
        times = rng.choice(T, size=1 + rng.poisson(T / 500), replace=False)
        impulses[times] = rng.choice([-1, 1], len(times)) * rng.uniform(1, 6, len(times)) * scale
        y += np.convolve(impulses, kernel)[:T]
    return y


def _observation_layer(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """Noise / quantization / censoring / MNAR gaps (handoff §4.5)."""
    T = len(y)
    scale = max(float(np.nanstd(y)), 1e-3)
    snr = _log_uniform(rng, 2.0, 100.0)
    sigma = scale / np.sqrt(snr)
    if rng.random() < 0.15:  # heavy-tailed
        y += (sigma * rng.standard_t(df=rng.uniform(2.0, 8.0), size=T)).astype(np.float32)
    else:
        y += (sigma * rng.standard_normal(T)).astype(np.float32)
    if rng.random() < 0.10:  # quantization
        n_levels = int(rng.integers(20, 201))
        lo, hi = np.nanmin(y), np.nanmax(y)
        step = max((hi - lo) / n_levels, 1e-8)
        y = np.round((y - lo) / step) * step + lo
    if rng.random() < 0.07:  # censoring
        q = rng.uniform(0.85, 0.98)
        if rng.random() < 0.5:
            y = np.minimum(y, np.nanquantile(y, q))
        else:
            y = np.maximum(y, np.nanquantile(y, 1 - q))
    if rng.random() < 0.08:  # MNAR gaps (<=20% of series)
        budget = int(0.2 * T)
        for _ in range(int(rng.integers(1, 5))):
            glen = int(rng.integers(2, max(3, budget // 2)))
            if glen > budget:
                break
            start = int(rng.integers(0, T - glen))
            y[start : start + glen] = np.nan
            budget -= glen
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# family 1 — parametric backbone
# ---------------------------------------------------------------------------


def gen_backbone(rng: np.random.Generator, T: int) -> np.ndarray:
    # 20% of series are trend-dominant (no seasonality) to press on F3
    trend_dominant = rng.random() < 0.20
    trend = _trend(rng, T, force=trend_dominant)
    seas = np.zeros(T, dtype=np.float32) if trend_dominant else _seasonality(rng, T)
    arma_ratio = _log_uniform(rng, 0.05, 1.5)  # stochastic power rel. to structure
    struct_scale = max(float(np.std(trend + seas)), 0.3)
    arma = _pole_arma(rng, T) * struct_scale * arma_ratio
    if rng.random() < 0.3 and seas.any():  # multiplicative seasonality
        y = (1.0 + trend) * (1.0 + 0.5 * seas) + arma
    else:
        y = trend + seas + arma
    if rng.random() < 0.35:
        y = _event_layer(rng, y)
    return _observation_layer(rng, y)


# ---------------------------------------------------------------------------
# family 2 — regime-switching time-inhomogeneous OU
# ---------------------------------------------------------------------------


def gen_sde(rng: np.random.Generator, T: int) -> np.ndarray:
    n_regimes = int(rng.integers(1, 5))
    thetas = np.exp(rng.uniform(np.log(0.002), np.log(0.2), n_regimes))
    mus = rng.normal(0.0, 1.0, n_regimes)
    sigmas = np.exp(rng.uniform(np.log(0.05), np.log(1.0), n_regimes))
    mean_dwell = rng.uniform(T / 20, T / 4)

    # time-inhomogeneous mean: slow sinusoid + optional linear drift
    t01 = np.linspace(0.0, 1.0, T, dtype=np.float64)
    mu_t = np.zeros(T)
    if rng.random() < 0.7:
        P = _log_uniform(rng, T / 4, 2 * T)
        mu_t += rng.uniform(0.0, 1.5) * np.sin(
            2 * np.pi * np.arange(T) / P + rng.uniform(0, 2 * np.pi)
        )
    if rng.random() < 0.5:
        mu_t += rng.uniform(-1.5, 1.5) * t01

    heavy = rng.random() < 0.2
    df = rng.uniform(3.0, 8.0)
    x = np.empty(T)
    pos, regime, x_last = 0, int(rng.integers(n_regimes)), 0.0
    while pos < T:
        seg = min(T - pos, max(2, int(rng.exponential(mean_dwell))))
        th, mu_r, sg = thetas[regime], mus[regime], sigmas[regime]
        a = np.exp(-th)
        s = sg * np.sqrt(1 - a * a)
        eps = rng.standard_t(df, seg) if heavy else rng.standard_normal(seg)
        u = (1 - a) * (mu_r + mu_t[pos : pos + seg]) + s * eps
        # exact OU discretization is AR(1): x_t = a x_{t-1} + u_t
        x[pos : pos + seg], zf = lfilter([1.0], [1.0, -a], u, zi=[a * x_last])
        x_last = x[pos + seg - 1]
        pos += seg
        if n_regimes > 1:  # Markov switch to a different regime
            regime = int((regime + rng.integers(1, n_regimes)) % n_regimes)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# family 3 — SARIMA-2 superposition (base x envelope coupling)
# ---------------------------------------------------------------------------


def _seasonal_arma(rng: np.random.Generator, T: int, P: float, r_lo: float, r_hi: float) -> np.ndarray:
    r = rng.uniform(r_lo, r_hi)
    w = 2 * np.pi / P
    denom = np.array([1.0, -2 * r * np.cos(w), r * r])
    if rng.random() < 0.5:  # extra non-seasonal pole
        denom = np.convolve(denom, [1.0, -rng.uniform(0.3, 0.9)])
    burn = int(min(4 * P, 2048))
    x = lfilter([1.0], denom, rng.standard_normal(T + burn))[burn:]
    std = x.std()
    return (x / std if std > 1e-8 else x).astype(np.float32)


def gen_sarima2(rng: np.random.Generator, T: int) -> np.ndarray:
    def unit(P_base):
        base = _seasonal_arma(rng, T, P_base, 0.9, 0.99)
        P_env = _log_uniform(rng, 8 * P_base, max(16 * P_base, T))
        env = _seasonal_arma(rng, T, P_env, 0.95, 0.999)
        return base * (0.3 + np.abs(env) * rng.uniform(0.5, 2.0))

    y = unit(_log_uniform(rng, 4.0, 64.0))
    if rng.random() < 0.4:  # superpose a second coupled unit
        y = y + rng.uniform(0.3, 1.0) * unit(_log_uniform(rng, 4.0, 64.0))
    y += 0.05 * np.abs(y).mean() * rng.standard_normal(T).astype(np.float32)
    return y.astype(np.float32)


GENERATORS = dict(
    synth_backbone_v0=gen_backbone,
    synth_sde_v0=gen_sde,
    synth_sarima2_v0=gen_sarima2,
)


def _example_gen(shard_bounds):
    """HF from_generator worker; items are (dataset, global_seed, lo, hi)."""
    for dataset, global_seed, lo, hi in shard_bounds:
        gen = GENERATORS[dataset]
        tag = dataset.replace("synth_", "").replace("_v0", "")
        for i in range(lo, hi):
            rng = np.random.default_rng(_series_seed(global_seed, dataset, i))
            T = int(rng.integers(MIN_LEN, MAX_LEN + 1))
            values = gen(rng, T)
            yield dict(
                item_id=f"synth_{tag}_{i:07d}",
                start="2000-01-01",
                freq="H",
                target=values,
            )


class _SynthTrackBMixin:
    """Shared build machinery; mixed into concrete LOTSADatasetBuilder subclasses
    (kept out of the LOTSADatasetBuilder hierarchy so the abstract
    dataset_list check only fires on the concrete classes)."""

    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(
        self,
        dataset: str,
        num_series: int,
        seed: int = 2026,
        num_proc: Optional[int] = None,
    ):
        assert dataset in self.dataset_list
        num_proc = num_proc or min(32, os.cpu_count())
        bounds = np.linspace(0, num_series, num_proc * 4 + 1, dtype=int)
        shard_bounds = [
            (dataset, seed, int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:])
        ]
        cache_dir = str(self.storage_path / f"{dataset}_build_cache")
        hf_dataset = datasets.Dataset.from_generator(
            _example_gen,
            features=LOTSA_FEATURES,
            gen_kwargs=dict(shard_bounds=shard_bounds),
            num_proc=num_proc,
            cache_dir=cache_dir,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset, num_proc=num_proc)
        del hf_dataset
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[{dataset}] Saved {num_series} series to {self.storage_path / dataset}")


class SynthBackboneDatasetBuilder(_SynthTrackBMixin, LOTSADatasetBuilder):
    dataset_list = ["synth_backbone_v0"]


class SynthSDEDatasetBuilder(_SynthTrackBMixin, LOTSADatasetBuilder):
    dataset_list = ["synth_sde_v0"]


class SynthSarima2DatasetBuilder(_SynthTrackBMixin, LOTSADatasetBuilder):
    dataset_list = ["synth_sarima2_v0"]


BUILDER_OF = dict(
    synth_backbone_v0=SynthBackboneDatasetBuilder,
    synth_sde_v0=SynthSDEDatasetBuilder,
    synth_sarima2_v0=SynthSarima2DatasetBuilder,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Track B synthetic corpus.")
    parser.add_argument("--dataset", required=True, choices=list(GENERATORS))
    parser.add_argument("--num-series", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--storage-path", default=None)
    args = parser.parse_args()

    from pathlib import Path

    kwargs = {}
    if args.storage_path is not None:
        kwargs["storage_path"] = Path(args.storage_path)
    builder = BUILDER_OF[args.dataset](datasets=[args.dataset], **kwargs)
    builder.build_dataset(
        dataset=args.dataset,
        num_series=args.num_series,
        seed=args.seed,
        num_proc=args.num_proc,
    )
