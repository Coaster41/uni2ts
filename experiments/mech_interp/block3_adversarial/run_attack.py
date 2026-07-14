"""Support-restricted attack runner -> results/attack_<model>.csv

    python -m experiments.mech_interp.block3_adversarial.run_attack \
        --model moiraix_dec_cpm_4 --device cuda:7 [--limit 16] [--targeted]

Writes one tidy row per (model, corpus, dataset, attack, support, ratio, eps,
series_idx). Long format on purpose — do not pre-aggregate here; analyze.py
pivots, and you will want to re-slice.

The grid: ``support="all"`` is only run at ratio 1.0 (it *is* the dense attack),
and the restricted supports are only run at ratio < 1.0 (at ratio 1.0 they all
degenerate to the dense attack).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block3_adversarial import metrics_adv as M  # noqa: E402
from experiments.mech_interp.block3_adversarial.attacks import (  # noqa: E402
    make_target,
    run_attack,
    support_mask,
)
from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    grad_forecaster_for,
    load_adv_config,
)

FIELDS = [
    "model", "corpus", "dataset", "attack", "support", "ratio", "eps",
    "series_idx", "ctx_std", "smae_clean", "smae_adv", "wql_clean", "wql_adv",
    "red_e", "red_e_wql", "displacement", "target_kind", "targeted_red",
]


def load_sources(cfg, limit):
    out = {}
    for corpus, fname in (("gift", "gift_windows.npz"), ("stress", "stress_windows.npz")):
        with np.load(DATA_DIR / fname) as z:
            for key in z.files:
                if "__" in key:
                    continue
                arr = z[key].astype(np.float32)
                out[(corpus, key)] = arr[:limit] if limit else arr
    return out


def merge_shards(model: str, out_dir: Path) -> int:
    """Concatenate attack_<model>.shard*.csv -> attack_<model>.csv."""
    parts = sorted(out_dir.glob(f"attack_{model}.shard*.csv"))
    if not parts:
        raise SystemExit(f"no shard CSVs for {model} in {out_dir}")
    dest = out_dir / f"attack_{model}.csv"
    n = 0
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for p in parts:
            with open(p) as src:
                for row in csv.DictReader(src):
                    w.writerow(row)
                    n += 1
    print(f"Merged {len(parts)} shards -> {dest} ({n} rows)")
    for p in parts:
        p.unlink()
    return 0


def grid(cfg):
    """(support, ratio) cells: dense once, restricted supports at ratio < 1."""
    cells = []
    for support in cfg["supports"]:
        for ratio in cfg["ratios"]:
            if support == "all" and ratio != 1.0:
                continue
            if support != "all" and ratio == 1.0:
                continue
            cells.append((support, ratio))
    return cells


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targeted", action="store_true", help="also run flip/drift/amp targets")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    # The outer loop is over independent data sources, so sharding over them is
    # exact: shard i takes sources[i::num_shards] and writes its own CSV part.
    # Merge with `python -m ...run_attack --merge --model <name>`.
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="concatenate shard CSVs and exit")
    args = p.parse_args()

    if args.merge:
        return merge_shards(args.model, Path(args.out_dir))

    cfg = load_adv_config()
    ctx_len, horizon = cfg["geometry"]["ctx"], cfg["geometry"]["horizon"]
    gf = grad_forecaster_for(args.model, cfg, device=args.device)
    levels = gf.quantile_levels

    sources = load_sources(cfg, args.limit)
    # Seed per source, keyed on its position in the FULL source list, so the
    # random supports a source gets do not depend on how the run is sharded (and
    # so two shards don't replay the same mask sequence on different data).
    seed_of = {k: args.seed * 1000 + i for i, k in enumerate(sorted(sources))}
    if args.num_shards > 1:
        keys = sorted(sources)[args.shard :: args.num_shards]
        sources = {k: sources[k] for k in keys}
        print(
            f"shard {args.shard}/{args.num_shards}: "
            f"{len(sources)} sources -> {[k[1] for k in sources]}"
        )
    cells = grid(cfg)
    target_kinds = [None] + (["flip", "drift", "amp"] if args.targeted else [])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{args.shard}" if args.num_shards > 1 else ""
    dest = out_dir / f"attack_{args.model}{suffix}.csv"
    t0 = time.time()
    n_rows = 0

    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()

        for (corpus, name), series in sources.items():
            rng = np.random.default_rng(seed_of[(corpus, name)])
            torch.manual_seed(seed_of[(corpus, name)])  # random_perturb control
            x0_full = torch.from_numpy(series[:, :ctx_len]).to(args.device)
            y_full = torch.from_numpy(series[:, ctx_len : ctx_len + horizon]).to(args.device)

            for b0 in range(0, len(series), args.batch_size):
                x0 = x0_full[b0 : b0 + args.batch_size]
                y = y_full[b0 : b0 + args.batch_size]
                n = len(x0)
                sigma_t = x0.std(dim=1, keepdim=True)  # [n,1] -- the eps unit
                sigma = sigma_t[:, 0].cpu().numpy()

                with torch.no_grad():
                    q_clean = gf.quantiles(x0, horizon)
                med_clean = q_clean[:, gf.median_idx, :]
                q_clean_np = q_clean.cpu().numpy()
                med_clean_np = med_clean.cpu().numpy()
                y_np = y.cpu().numpy()

                smae_clean = M.smae(med_clean_np, y_np, sigma)
                wql_clean = M.wql(q_clean_np, y_np, levels)

                for tk in target_kinds:
                    tgt = make_target(med_clean, sigma_t[:, 0], tk) if tk else None
                    # Targeted progress is measured against the target, so its
                    # "clean" reference is the clean forecast's distance to it.
                    e_clean_tgt = (
                        M.smae(med_clean_np, tgt.cpu().numpy(), sigma)
                        if tk
                        else None
                    )

                    for support, ratio in cells:
                        mask = support_mask(n, ctx_len, support, ratio, rng, args.device)
                        for eps in cfg["eps_grid"]:
                            eps_abs = eps * sigma_t
                            for attack in cfg["attacks"]:
                                if tk and attack == "random":
                                    continue  # a random control has no target
                                x_adv = run_attack(
                                    gf, x0, y, horizon, attack, eps_abs, mask, targeted=tgt
                                )
                                with torch.no_grad():
                                    q_adv = gf.quantiles(x_adv, horizon)
                                q_adv_np = q_adv.cpu().numpy()
                                med_adv_np = q_adv_np[:, gf.median_idx, :]

                                smae_adv = M.smae(med_adv_np, y_np, sigma)
                                wql_adv = M.wql(q_adv_np, y_np, levels)
                                disp = M.displacement(med_adv_np, med_clean_np, sigma)
                                rede = M.red_e(smae_adv, smae_clean)
                                rede_wql = M.red_e(wql_adv, wql_clean)
                                tred = (
                                    M.targeted_red(
                                        M.smae(med_adv_np, tgt.cpu().numpy(), sigma),
                                        e_clean_tgt,
                                    )
                                    if tk
                                    else None
                                )

                                for i in range(n):
                                    w.writerow({
                                        "model": args.model,
                                        "corpus": corpus,
                                        "dataset": name,
                                        "attack": attack,
                                        "support": support,
                                        "ratio": ratio,
                                        "eps": eps,
                                        "series_idx": b0 + i,
                                        "ctx_std": float(sigma[i]),
                                        "smae_clean": float(smae_clean[i]),
                                        "smae_adv": float(smae_adv[i]),
                                        "wql_clean": float(wql_clean[i]),
                                        "wql_adv": float(wql_adv[i]),
                                        "red_e": float(rede[i]),
                                        "red_e_wql": float(rede_wql[i]),
                                        "displacement": float(disp[i]),
                                        "target_kind": tk or "",
                                        "targeted_red": float(tred[i]) if tk else "",
                                    })
                                    n_rows += 1
            print(f"  {corpus}:{name} done ({n_rows} rows, {time.time() - t0:.0f}s)")

    print(f"\nWrote {dest}: {n_rows} rows ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
