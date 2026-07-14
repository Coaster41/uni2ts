"""Example-attack figure + stress-retention drop.

    python -m experiments.mech_interp.block3_adversarial.make_examples --device cuda:7

Shows what a boundary-restricted attack actually *does* to a forecast, and scores
it with the block2 stress metrics: on `family_a_periodic` the attack should damage
the recovered sine amplitude/phase (`retention_periodic`), and on `family_a_trend`
it should bend the recovered slope (`retention_trend`). Those retention drops are
the most interpretable statement of the damage — "NMAE went up on electricity" is
not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block2_stress.metrics import (  # noqa: E402
    retention_periodic,
    retention_trend,
)
from experiments.mech_interp.block3_adversarial.attacks import (  # noqa: E402
    run_attack,
    support_mask,
)
from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    DATA_DIR,
    grad_forecaster_for,
    load_adv_config,
)

OUT = Path("/srv/disk00/ctadler/tsfm-mi-experiments/adversarial/exp-000-boundary/figures")

# Stress cfg for the T=320 files (ctx 256 + H 64) -- retention_trend needs these.
STRESS_CFG = {"context_patches": 16, "horizon_patches": 4, "patch_len": 16}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--model", default="moiraix_enc_cpm_gift")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--ratio", type=float, default=0.10)
    p.add_argument("--support", default="last")
    args = p.parse_args()

    cfg = load_adv_config()
    ctx_len, horizon = cfg["geometry"]["ctx"], cfg["geometry"]["horizon"]
    gf = grad_forecaster_for(args.model, cfg, device=args.device)
    rng = np.random.default_rng(0)

    z = np.load(DATA_DIR / "stress_windows.npz")
    print(f"{args.model}: {args.support}-{args.ratio:.0%} support, eps={args.eps}·σ\n")

    fams = ["family_a_periodic", "family_a_trend"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    for r, fam in enumerate(fams):
        series = z[fam].astype(np.float32)
        meta = {
            k.split("__meta_")[1]: z[k]
            for k in z.files
            if k.startswith(f"{fam}__meta_")
        }
        x0 = torch.from_numpy(series[:, :ctx_len]).to(args.device)
        y = torch.from_numpy(series[:, ctx_len : ctx_len + horizon]).to(args.device)
        sigma = x0.std(dim=1, keepdim=True)

        mask = support_mask(len(x0), ctx_len, args.support, args.ratio, rng, args.device)
        x_adv = run_attack(gf, x0, y, horizon, "pgd10", args.eps * sigma, mask)

        with torch.no_grad():
            med_clean = gf.median(x0, horizon).cpu().numpy()
            med_adv = gf.median(x_adv, horizon).cpu().numpy()

        fn = retention_periodic if fam == "family_a_periodic" else retention_trend
        r_clean = fn(med_clean, meta, STRESS_CFG)
        r_adv = fn(med_adv, meta, STRESS_CFG)
        name = "amplitude" if fam == "family_a_periodic" else "slope"
        print(
            f"  {fam}: retention ({name}) {np.median(r_clean):.3f} -> "
            f"{np.median(r_adv):.3f}  "
            f"({100 * (np.median(r_adv) / np.median(r_clean) - 1):+.1f}%)"
        )

        x_adv_np = x_adv.cpu().numpy()
        for c, i in enumerate([0, 1]):
            ax = axes[r][c]
            t_ctx = np.arange(ctx_len)
            t_h = np.arange(ctx_len, ctx_len + horizon)
            ax.plot(t_ctx, series[i, :ctx_len], color="0.6", lw=0.9, label="context")
            ax.plot(t_ctx, x_adv_np[i], color="#d7301f", lw=0.7, alpha=0.8,
                    label="adversarial context")
            ax.plot(t_h, series[i, ctx_len:], color="k", lw=1.4, label="ground truth")
            ax.plot(t_h, med_clean[i], color="#2c7fb8", lw=1.6, label="clean forecast")
            ax.plot(t_h, med_adv[i], color="#d7301f", lw=1.6, ls="--",
                    label="adversarial forecast")
            ax.axvline(ctx_len, color="k", lw=0.6, ls=":")
            ax.axvspan(ctx_len - int(args.ratio * ctx_len), ctx_len,
                       color="#d7301f", alpha=0.08)
            ax.set_xlim(ctx_len - 128, ctx_len + horizon)
            ax.set_title(f"{fam} #{i}", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=6)

    fig.suptitle(
        f"Boundary-restricted attack ({args.model}, PGD-10, support={args.support} "
        f"{args.ratio:.0%} of context, ε={args.eps}·σ_ctx). "
        "Shaded = the only region the attacker may touch."
    )
    fig.tight_layout()
    fig.savefig(OUT / "example_attacks.png", dpi=150)
    print(f"\nWrote {OUT / 'example_attacks.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
