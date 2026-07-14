#  synth_stress_v0: targeted synthetic corpus for the Track A causal probe
#  (.claude/HANDOFF_SYNTH_DATA.md §3). Attacks the three structural failures
#  measured in exp-004:
#    F1 patch-aligned period collapse -> bare carriers sweeping period U{4..64}
#    F2 compositional brittleness     -> carrier + {trend, level shift, amp mod, spikes}
#    F3 trend under-commitment        -> pure linear/exp/logistic trends, both signs
#  plus negative controls (AR(1), white noise, intermittent) so the model doesn't
#  learn "everything is periodic".
#
#  Anti-teaching-to-the-test: parameters are FRESH continuous draws spanning the
#  frozen stress-battery eval grid (block2_stress/data/stress stays eval-only) —
#  period ~ U{4..64}, phase ~ U[0,P), amplitude ~ LogU(0.3,3), SNR ~ LogU(0.5,50),
#  length ~ U{2048..4096} so eval-length 320-step windows are croppable.
#
#  Build:
#    python -m uni2ts.data.builder.lotsa_v1.synth_stress \
#        --num-series 500000 --seed 1337 --num-proc 32

import argparse
import hashlib
import os
import shutil
from collections import defaultdict
from functools import partial
from typing import Optional

import datasets
import numpy as np
from datasets import Features, Sequence, Value

from uni2ts.common.env import env
from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder

LOTSA_FEATURES = Features(
    dict(
        item_id=Value("string"),
        start=Value("timestamp[s]"),
        freq=Value("string"),
        target=Sequence(Value("float32")),
    )
)

# Composition mix (handoff §A.1). Categories are drawn per series.
CATEGORIES = ("carrier", "composition", "trend", "control")
CATEGORY_PROBS = (0.30, 0.40, 0.15, 0.15)

MIN_LEN, MAX_LEN = 2048, 4096
PERIOD_LO, PERIOD_HI = 4, 64  # integer periods; includes patch-aligned 8/16/32


def _series_seed(global_seed: int, series_id: int) -> int:
    """Deterministic per-series seed, independent of sharding layout."""
    key = f"{global_seed}:synth_stress_v0:{series_id}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _draw_carrier(
    rng: np.random.Generator, T: int
) -> tuple[np.ndarray, float, float]:
    """Unit-phase periodic carrier. Returns (x, amplitude, signal_power)."""
    period = int(rng.integers(PERIOD_LO, PERIOD_HI + 1))
    amp = _log_uniform(rng, 0.3, 3.0)
    offset = rng.uniform(0.0, period)
    t = np.arange(T, dtype=np.float32)
    if rng.random() < 0.5:  # sine
        x = amp * np.sin(2 * np.pi * (t - offset) / period)
        power = amp**2 / 2.0
    else:  # triangle
        t_mod = ((t - offset) % period) / period
        x = amp * (1.0 - 4.0 * np.abs(t_mod - 0.5))
        power = amp**2 / 3.0
    return x.astype(np.float32), amp, power


def _noise_for_power(
    rng: np.random.Generator, T: int, signal_power: float, snr_lo: float, snr_hi: float
) -> np.ndarray:
    snr = _log_uniform(rng, snr_lo, snr_hi)
    sigma = np.sqrt(signal_power / snr)
    return (sigma * rng.standard_normal(T)).astype(np.float32)


# ---------------------------------------------------------------------------
# F1 — bare carriers
# ---------------------------------------------------------------------------


def gen_carrier(rng: np.random.Generator, T: int) -> np.ndarray:
    x, _, power = _draw_carrier(rng, T)
    return x + _noise_for_power(rng, T, power, 0.5, 50.0)


# ---------------------------------------------------------------------------
# F2 — two-element compositions: carrier + one structural element
# ---------------------------------------------------------------------------


def _elem_linear_trend(rng, T, amp, power):
    # trend_snr = carrier power / trend power; trend power over [-1,1] is 1/3
    trend_snr = _log_uniform(rng, 0.1, 10.0)
    trend_amp = np.sqrt(power / trend_snr * 3.0)
    sign = rng.choice([-1.0, 1.0])
    t_norm = np.linspace(-1.0, 1.0, T, dtype=np.float32)
    return sign * trend_amp * t_norm


def _elem_piecewise_trend(rng, T, amp, power):
    n_knots = int(rng.integers(2, 6))  # 1..4 changepoints
    trend_amp = _log_uniform(rng, 0.5, 3.0) * amp
    knot_t = np.sort(rng.uniform(0.0, 1.0, size=n_knots))
    knot_t = np.concatenate([[0.0], knot_t, [1.0]])
    knot_v = rng.uniform(-trend_amp, trend_amp, size=len(knot_t))
    t01 = np.linspace(0.0, 1.0, T, dtype=np.float32)
    return np.interp(t01, knot_t, knot_v).astype(np.float32)


def _elem_level_shift(rng, T, amp, power):
    shift_t = int(rng.integers(int(0.1 * T), int(0.9 * T)))
    mag = float(rng.choice([-1.0, 1.0])) * rng.uniform(0.5, 3.0) * amp
    x = np.zeros(T, dtype=np.float32)
    x[shift_t:] = mag
    return x


def _elem_spikes(rng, T, amp, power):
    n_spikes = 1 + rng.poisson(T / 512)
    times = rng.choice(T, size=min(n_spikes, T // 8), replace=False)
    mags = rng.choice([-1.0, 1.0], size=len(times)) * rng.uniform(
        2.0, 8.0, size=len(times)
    )
    x = np.zeros(T, dtype=np.float32)
    x[times] = (mags * amp).astype(np.float32)
    return x


def gen_composition(rng: np.random.Generator, T: int) -> np.ndarray:
    carrier, amp, power = _draw_carrier(rng, T)
    element = rng.integers(0, 5)
    if element == 4:  # multiplicative amplitude envelope (monotone, stays positive)
        sign = float(rng.choice([-1.0, 1.0]))
        rate = rng.uniform(0.3, 2.0) if sign > 0 else rng.uniform(0.3, 0.9)
        t01 = np.linspace(0.0, 1.0, T, dtype=np.float32)
        x = carrier * (1.0 + sign * rate * t01)
    else:
        elem_fn = (
            _elem_linear_trend,
            _elem_piecewise_trend,
            _elem_level_shift,
            _elem_spikes,
        )[element]
        x = carrier + elem_fn(rng, T, amp, power).astype(np.float32)
    return x + _noise_for_power(rng, T, power, 2.0, 50.0)


# ---------------------------------------------------------------------------
# F3 — pure trends
# ---------------------------------------------------------------------------


def gen_trend(rng: np.random.Generator, T: int) -> np.ndarray:
    kind = rng.integers(0, 3)
    sign = float(rng.choice([-1.0, 1.0]))
    amp = _log_uniform(rng, 0.3, 3.0)
    t01 = np.linspace(0.0, 1.0, T, dtype=np.float32)
    if kind == 0:  # linear
        intercept = rng.uniform(-amp, amp)
        x = sign * amp * (2.0 * t01 - 1.0) + intercept
    elif kind == 1:  # exponential growth/decay over a LogU(2,20) factor
        factor = _log_uniform(rng, 2.0, 20.0)
        curve = np.exp(t01 * np.log(factor), dtype=np.float32)
        x = sign * amp * (curve if rng.random() < 0.5 else curve[::-1].copy())
    else:  # logistic
        from scipy.special import expit

        t0 = rng.uniform(0.25, 0.75)
        width = rng.uniform(0.05, 0.5)
        x = sign * amp * expit((t01.astype(np.float64) - t0) / (width / 8.0))
    x = x.astype(np.float32)
    signal_power = max(float(x.var()), 1e-8)
    return x + _noise_for_power(rng, T, signal_power, 1.0, 50.0)


# ---------------------------------------------------------------------------
# Negative controls — AR(1) / white noise / intermittent
# ---------------------------------------------------------------------------


def gen_control(rng: np.random.Generator, T: int) -> np.ndarray:
    kind = rng.integers(0, 3)
    if kind == 0:  # AR(1)
        from scipy.signal import lfilter

        phi = rng.uniform(0.5, 0.99)
        sigma = _log_uniform(rng, 0.1, 1.0)
        eps = sigma * rng.standard_normal(T)
        return lfilter([1.0], [1.0, -phi], eps).astype(np.float32)
    if kind == 1:  # white noise
        sigma = _log_uniform(rng, 0.5, 2.0)
        return (sigma * rng.standard_normal(T)).astype(np.float32)
    # intermittent half-sine bumps
    period = int(rng.integers(8, 129))
    p_event = rng.uniform(0.1, 0.9)
    amp = _log_uniform(rng, 0.3, 3.0)
    half_width = max(1, int(rng.uniform(0.1, 0.5) * period))
    bump = amp * np.sin(np.pi * np.arange(half_width) / half_width)
    x = np.zeros(T, dtype=np.float32)
    for k in range(T // period + 1):
        if rng.random() < p_event:
            start = k * period
            end = min(start + half_width, T)
            x[start:end] += bump[: end - start]
    noise_sigma = 0.05 * amp
    return (x + noise_sigma * rng.standard_normal(T)).astype(np.float32)


GENERATORS = dict(
    carrier=gen_carrier,
    composition=gen_composition,
    trend=gen_trend,
    control=gen_control,
)


def generate_series(global_seed: int, series_id: int) -> tuple[str, np.ndarray]:
    """One series: category + values, fully determined by (global_seed, series_id)."""
    rng = np.random.default_rng(_series_seed(global_seed, series_id))
    category = str(rng.choice(CATEGORIES, p=CATEGORY_PROBS))
    T = int(rng.integers(MIN_LEN, MAX_LEN + 1))
    values = GENERATORS[category](rng, T)
    return category, values


def _example_gen(shard_bounds: list[tuple[int, int, int]]):
    """HF from_generator worker; shard_bounds items are (global_seed, lo, hi)."""
    for global_seed, lo, hi in shard_bounds:
        for i in range(lo, hi):
            category, values = generate_series(global_seed, i)
            yield dict(
                item_id=f"synth_stress_{category}_{i:07d}",
                start="2000-01-01",
                freq="H",
                target=values,
            )


class SynthStressDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = ["synth_stress_v0"]
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(
        self,
        dataset: str = "synth_stress_v0",
        num_series: int = 500_000,
        seed: int = 1337,
        num_proc: Optional[int] = None,
    ):
        num_proc = num_proc or min(32, os.cpu_count())
        num_shards = num_proc * 4
        bounds = np.linspace(0, num_series, num_shards + 1, dtype=int)
        shard_bounds = [
            (seed, int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:])
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
        hf_dataset.save_to_disk(
            dataset_path=self.storage_path / dataset, num_proc=num_proc
        )
        del hf_dataset
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[{dataset}] Saved {num_series} series to {self.storage_path / dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the synth_stress_v0 LOTSA-format corpus."
    )
    parser.add_argument("--num-series", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument(
        "--storage-path",
        default=None,
        help="Override env.LOTSA_V1_PATH (e.g. for smoke tests)",
    )
    args = parser.parse_args()

    from pathlib import Path

    kwargs = {}
    if args.storage_path is not None:
        kwargs["storage_path"] = Path(args.storage_path)
    builder = SynthStressDatasetBuilder(datasets=["synth_stress_v0"], **kwargs)
    builder.build_dataset(
        num_series=args.num_series, seed=args.seed, num_proc=args.num_proc
    )
