"""Transfer attack -> results/transfer.csv

    python -m experiments.mech_interp.block3_adversarial.run_transfer \
        --source moiraix_dec_cpm_4 --device cuda:7

Crafts support-restricted perturbations on ONE white-box source model, then
evaluates them on every other model — including `chronos2` and `timesfm25`, which
we cannot backprop through. This is the only way those two get a **damage**
measurement (RED_E / BVI) rather than just a sensitivity profile: the bump probe
says where they are sensitive, the transfer attack says whether that sensitivity
can actually be exploited by an attacker who never sees their weights.

It is also the realistic threat model: an attacker holds an open model and aims at
a closed one.

Targets are scored through the block2 numpy adapter (forward passes only), so
white-box and black-box targets go through the identical evaluation path.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block3_adversarial import metrics_adv as M  # noqa: E402
from experiments.mech_interp.block3_adversarial.attacks import (  # noqa: E402
    run_attack,
    support_mask,
)
from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    adapter_for,
    grad_forecaster_for,
    load_adv_config,
)
from experiments.mech_interp.block3_adversarial.probes import (  # noqa: E402
    _median_forecast,
)

SUPPORTS = ("last", "first", "random", "all")


def load_sources(cfg, limit):
    out = {}
    for corpus, fname in (("gift", "gift_windows.npz"), ("stress", "stress_windows.npz")):
        with np.load(DATA_DIR / fname) as z:
            for key in z.files:
                if "__" in key:
                    continue
                a = z[key].astype(np.float32)
                out[(corpus, key)] = a[:limit] if limit else a
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="moiraix_dec_cpm_4")
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--ratio", type=float, default=0.10)
    p.add_argument("--attack", default="pgd10")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    cfg = load_adv_config()
    ctx_len, horizon = cfg["geometry"]["ctx"], cfg["geometry"]["horizon"]
    targets = args.targets or (cfg["whitebox"] + cfg["blackbox"])
    sources = load_sources(cfg, args.limit)
    rng = np.random.default_rng(0)

    # ---- 1. craft perturbations once, on the source model ------------------
    gf = grad_forecaster_for(args.source, cfg, device=args.device)
    adv: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, series in sources.items():
        x0 = torch.from_numpy(series[:, :ctx_len]).to(args.device)
        y = torch.from_numpy(series[:, ctx_len : ctx_len + horizon]).to(args.device)
        sigma = x0.std(dim=1, keepdim=True)
        per_support = {}
        for sup in SUPPORTS:
            r = 1.0 if sup == "all" else args.ratio
            mask = support_mask(len(x0), ctx_len, sup, r, rng, args.device)
            xa = run_attack(gf, x0, y, horizon, args.attack, args.eps * sigma, mask)
            per_support[sup] = xa.cpu().numpy().astype(np.float32)
        adv[key] = per_support
        print(f"  crafted {key[0]}:{key[1]}")
    del gf
    torch.cuda.empty_cache()

    # ---- 2. evaluate on every target, forward passes only -------------------
    dest = RESULTS_DIR / "transfer.csv"
    fields = [
        "source", "target", "corpus", "dataset", "support", "eps", "ratio",
        "series_idx", "smae_clean", "smae_adv", "red_e", "displacement",
    ]
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for tgt in targets:
            ad = adapter_for(tgt, device=args.device)
            mi = min(
                range(len(ad.quantile_levels)),
                key=lambda i: abs(ad.quantile_levels[i] - 0.5),
            )
            for key, series in sources.items():
                corpus, name = key
                ctx = series[:, :ctx_len]
                y = series[:, ctx_len : ctx_len + horizon]
                sigma = ctx.std(axis=1)
                med_c = _median_forecast(ad, ctx, horizon, args.batch_size, mi)
                e_clean = M.smae(med_c, y, sigma)
                for sup in SUPPORTS:
                    med_a = _median_forecast(
                        ad, adv[key][sup], horizon, args.batch_size, mi
                    )
                    e_adv = M.smae(med_a, y, sigma)
                    rede = M.red_e(e_adv, e_clean)
                    disp = M.displacement(med_a, med_c, sigma)
                    for i in range(len(ctx)):
                        w.writerow({
                            "source": args.source, "target": tgt,
                            "corpus": corpus, "dataset": name, "support": sup,
                            "eps": args.eps,
                            "ratio": 1.0 if sup == "all" else args.ratio,
                            "series_idx": i,
                            "smae_clean": float(e_clean[i]),
                            "smae_adv": float(e_adv[i]),
                            "red_e": float(rede[i]),
                            "displacement": float(disp[i]),
                        })
            print(f"  evaluated target {tgt}")
            del ad
            torch.cuda.empty_cache()

    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
