"""Positional sensitivity probe runner -> results/probe_<model>.npz

    python -m experiments.mech_interp.block3_adversarial.run_probe \
        --model moiraix_dec_cpm_4 --device cuda:7 [--limit 16]

White-box models get gradient saliency (+ the mean-centered control, top-k
frequency, and the patch-structure profiles). Black-box models (chronos2,
timesfm25) get the finite-difference bump probe. White-box models also get the
bump probe unless --no-bump, because agreement between the two is the check that
the saliency curve isn't a gradient-masking artifact.

This step alone answers the headline question.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block3_adversarial import probes  # noqa: E402
from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    adapter_for,
    grad_forecaster_for,
    load_adv_config,
)


def load_sources(cfg: dict, limit: int | None) -> dict[str, np.ndarray]:
    """All attack windows, keyed ``<corpus>:<name>`` -> ``[n, T]``."""
    out: dict[str, np.ndarray] = {}
    for corpus, fname in (("gift", "gift_windows.npz"), ("stress", "stress_windows.npz")):
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `python -m ...block3_adversarial.data`")
        with np.load(path) as z:
            for key in z.files:
                if "__" in key:  # ctx_std / meta sidecars
                    continue
                arr = z[key].astype(np.float32)
                out[f"{corpus}:{key}"] = arr[:limit] if limit else arr
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--limit", type=int, default=None, help="windows per source")
    p.add_argument("--no-bump", action="store_true", help="skip the finite-difference probe")
    p.add_argument("--bump-stride", type=int, default=None)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    cfg = load_adv_config()
    ctx_len = cfg["geometry"]["ctx"]
    horizon = cfg["geometry"]["horizon"]
    is_whitebox = args.model in cfg["whitebox"]

    sources = load_sources(cfg, args.limit)
    out: dict[str, np.ndarray] = {}
    t0 = time.time()

    # ---- white-box: gradient saliency -------------------------------------
    if is_whitebox:
        gf = grad_forecaster_for(args.model, cfg, device=args.device)
        out["patch_size"] = np.array(gf.patch_size)
        out["predict_next"] = np.array(gf.predict_next)
        for key, series in sources.items():
            x = torch.from_numpy(series[:, :ctx_len]).to(args.device)
            y = torch.from_numpy(series[:, ctx_len : ctx_len + horizon]).to(args.device)
            g = probes.saliency(gf, x, y, horizon)
            out[f"{key}|grad"] = g.astype(np.float32)
            out[f"{key}|sal"] = probes.normalize_curve(g).astype(np.float32)
            out[f"{key}|sal_centered"] = probes.centered_curve(g).astype(np.float32)
            out[f"{key}|topk"] = probes.topk_frequency(g, k=25).astype(np.float32)
            out[f"{key}|patch_off"] = probes.patch_offset_profile(g, gf.patch_size).astype(np.float32)
            out[f"{key}|patch_idx"] = probes.patch_index_profile(g, gf.patch_size).astype(np.float32)
            print(f"  [saliency] {key}: {g.shape}  ({time.time() - t0:.0f}s)")
        del gf
        torch.cuda.empty_cache()

    # ---- bump probe (everyone, unless disabled) ---------------------------
    if not args.no_bump:
        bump = cfg["bump"]
        stride = args.bump_stride if args.bump_stride is not None else bump["stride"]
        ad = adapter_for(args.model, device=args.device)
        med_idx = min(
            range(len(ad.quantile_levels)),
            key=lambda i: abs(ad.quantile_levels[i] - 0.5),
        )
        for key, series in sources.items():
            prof = probes.bump_profile(
                ad,
                series[:, :ctx_len],
                horizon,
                kappa=bump["kappa"],
                stride=stride,
                batch_size=bump["batch_size"],
                median_idx=med_idx,
            )
            out[f"{key}|bump"] = prof.astype(np.float32)
            print(f"  [bump] {key}: peak@{int(prof.argmax())}  ({time.time() - t0:.0f}s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"probe_{args.model}.npz"
    np.savez_compressed(dest, **out)
    print(f"\nWrote {dest}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
