"""Build the two attack corpora, cached so every model attacks identical windows.

    python -m experiments.mech_interp.block3_adversarial.data

Writes ``data/adv/gift_windows.npz`` and ``data/adv/stress_windows.npz``, each
holding ``[n, T]`` float32 windows (T = ctx + horizon) plus the per-window
context std, which every scale-relative budget and metric depends on.

GIFT windows come from ``split="train"`` (``load_gift_eval_series`` truncates each
series at 90%), so this is *not* an attack on the benchmark test set — that is
deliberate: this is a robustness study, not a leaderboard run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block2_stress import load_stress_dataset  # noqa: E402
from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    DATA_DIR,
    STRESS_DIR,
    load_adv_config,
)
from experiments.mech_interp.lib.dataset import (  # noqa: E402
    load_gift_eval_series,
    wrap_existing_dataset,
)

# Constant-context windows make every scale-relative budget meaningless and blow
# up RED_E (E_clean ~ 0), so they are dropped and counted.
MIN_CTX_STD = 1e-6


def _filter(series: np.ndarray, ctx: int, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN / constant-context windows. Returns (kept, ctx_std)."""
    finite = np.isfinite(series).all(axis=1)
    ctx_std = series[:, :ctx].std(axis=1)
    keep = finite & (ctx_std > MIN_CTX_STD)
    n_drop = int((~keep).sum())
    if n_drop:
        print(
            f"    {tag}: dropped {n_drop}/{len(series)} windows "
            f"({int((~finite).sum())} non-finite, "
            f"{int((finite & (ctx_std <= MIN_CTX_STD)).sum())} constant-context)"
        )
    return series[keep], ctx_std[keep]


def build_gift_windows(cfg: dict, seed: int = 0) -> dict[str, np.ndarray]:
    ctx = cfg["geometry"]["ctx"]
    T = ctx + cfg["geometry"]["horizon"]
    n = cfg["geometry"]["n_per_source"]

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(cfg["gift_sets"]):
        print(f"  [gift] {name}")
        series = load_gift_eval_series(name, split="train")
        # Oversample: some windows will be dropped as constant/NaN below.
        d = wrap_existing_dataset(
            series,
            label_generators=[],
            series_length=T,
            n=int(n * 1.5),
            seed=seed + i,
        )
        kept, ctx_std = _filter(d["series"].astype(np.float32), ctx, name)
        if len(kept) < n:
            raise ValueError(f"{name}: only {len(kept)} usable windows (need {n})")
        key = name.replace("/", "_")
        out[key] = kept[:n]
        out[f"{key}__ctx_std"] = ctx_std[:n].astype(np.float32)
    return out


def build_stress_windows(cfg: dict, seed: int = 0) -> dict[str, np.ndarray]:
    ctx = cfg["geometry"]["ctx"]
    T = ctx + cfg["geometry"]["horizon"]
    n = cfg["geometry"]["n_per_source"]

    out: dict[str, np.ndarray] = {}
    for i, (family, level) in enumerate(cfg["stress_sets"]):
        print(f"  [stress] {family}/{level}")
        series, meta, scfg = load_stress_dataset(str(STRESS_DIR), family, level)
        got_T = (scfg["context_patches"] + scfg["horizon_patches"]) * scfg["patch_len"]
        if got_T != T or series.shape[1] != T:
            raise ValueError(
                f"{family}/{level}: stress geometry T={got_T}/{series.shape[1]} "
                f"!= block3 T={T}. Regenerate the stress corpus or fix adv.yaml."
            )
        rng = np.random.default_rng(seed + 100 + i)
        idx = rng.choice(len(series), size=min(int(n * 1.5), len(series)), replace=False)
        kept, ctx_std = _filter(
            series[idx].astype(np.float32), ctx, f"{family}/{level}"
        )
        if len(kept) < n:
            raise ValueError(f"{family}/{level}: only {len(kept)} usable windows")
        key = family
        out[key] = kept[:n]
        out[f"{key}__ctx_std"] = ctx_std[:n].astype(np.float32)
        # Keep the ground-truth structure metadata: retention_periodic /
        # retention_trend need period_ts / slope_sign to score the adversarial
        # forecasts (HANDOFF §4b).
        for mk, mv in meta.items():
            arr = np.asarray(mv)
            if arr.shape[:1] == series.shape[:1]:
                out[f"{key}__meta_{mk}"] = arr[idx][: len(kept)][:n]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(DATA_DIR))
    args = p.parse_args()

    cfg = load_adv_config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building GIFT-Eval attack windows (train region only)...")
    gift = build_gift_windows(cfg, seed=args.seed)
    np.savez_compressed(out_dir / "gift_windows.npz", **gift)

    print("Building stress attack windows...")
    stress = build_stress_windows(cfg, seed=args.seed)
    np.savez_compressed(out_dir / "stress_windows.npz", **stress)

    n_gift = sum(1 for k in gift if "__" not in k)
    n_stress = sum(1 for k in stress if "__" not in k)
    print(
        f"\nWrote {out_dir}/gift_windows.npz ({n_gift} datasets) and "
        f"stress_windows.npz ({n_stress} families)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
