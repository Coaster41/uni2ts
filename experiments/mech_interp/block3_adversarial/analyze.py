"""Aggregate block3 results -> summary CSVs + figures in tsfm-mi-experiments.

    python -m experiments.mech_interp.block3_adversarial.analyze

Reads ``results/probe_*.npz`` and ``results/attack_*.csv``; writes the BVI table,
the support ablation, and every figure into
``/srv/disk00/ctadler/tsfm-mi-experiments/adversarial/exp-000-boundary/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    RESULTS_DIR,
    load_adv_config,
)

OUT_ROOT = Path("/srv/disk00/ctadler/tsfm-mi-experiments/adversarial/exp-000-boundary")

# Reference cell for the headline table.
REF_EPS = 0.05
REF_RATIO = 0.10
REF_ATTACK = "pgd10"

# Colour by readout, linestyle by attention -- the two axes we are separating.
READOUT_COLOR = {"mask_slot": "#2c7fb8", "next_token": "#d95f0e", "unknown": "#777777"}
ATTN_STYLE = {"bidirectional": "-", "causal": "--", "unknown": ":"}


def _style(cfg, model):
    a = cfg["arch"].get(model, {})
    return {
        "color": READOUT_COLOR.get(a.get("readout"), "#777777"),
        "linestyle": ATTN_STYLE.get(a.get("attention"), ":"),
    }


def _label(cfg, model):
    a = cfg["arch"].get(model, {})
    attn = {"bidirectional": "bidir", "causal": "causal"}.get(a.get("attention"), "?")
    rd = {"mask_slot": "mask-slot", "next_token": "next-tok"}.get(a.get("readout"), "?")
    return f"{model}\n({attn}, {rd}, P={a.get('patch', '?')})"


def load_probes(models):
    out = {}
    for m in models:
        p = RESULTS_DIR / f"probe_{m}.npz"
        if p.exists():
            out[m] = dict(np.load(p))
    return out


def load_attacks(models):
    frames = []
    for m in models:
        p = RESULTS_DIR / f"attack_{m}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_curve(pz, suffix, corpus=None):
    """Average a per-source curve across sources."""
    keys = [k for k in pz if k.endswith(f"|{suffix}")]
    if corpus:
        keys = [k for k in keys if k.startswith(f"{corpus}:")]
    if not keys:
        return None
    return np.stack([pz[k] for k in keys]).mean(axis=0)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def bvi_table(cfg, probes, atk) -> pd.DataFrame:
    rows = []
    for m in cfg["whitebox"] + cfg["blackbox"]:
        a = cfg["arch"].get(m, {})
        row = {
            "model": m,
            "attention": a.get("attention"),
            "readout": a.get("readout"),
            "patch": a.get("patch"),
            "mix": a.get("mix"),
        }

        pz = probes.get(m)
        if pz is not None:
            g = np.concatenate(
                [pz[k] for k in pz if k.endswith("|grad")], axis=0
            ) if any(k.endswith("|grad") for k in pz) else None
            if g is not None:
                ag = np.abs(g)
                ctx = ag.shape[1]
                k = max(1, int(round(0.10 * ctx)))
                total = ag.sum(1)
                # Some windows get an exactly-zero input gradient (the head is
                # locally flat there). 0/0 is not a boundary concentration of
                # 0 -- it is no measurement, so drop and count those.
                ok = total > 0
                gm = ag[ok, ctx - k :].sum(1) / total[ok]
                row["GM_last10"] = float(np.median(gm)) if ok.any() else np.nan
                row["n_zero_grad"] = int((~ok).sum())
            # BM_last10: the bump-profile twin of GM_last10, and the ONLY
            # boundary-concentration index that exists for the black-box models
            # (no gradient => no GM_last10 for them). BM runs ~0.02 below GM on the
            # same model, so the two columns are NOT interchangeable: comparing a
            # black-box BM against a white-box GM makes the former look spuriously
            # flat. Compare BM to BM.
            bump = mean_curve(pz, "bump")
            if bump is not None:
                ctx = len(bump)
                k = max(1, int(round(0.10 * ctx)))
                row["bump_last10"] = float(bump[ctx - k :].sum() / (bump.sum() + 1e-12))

        if len(atk):
            d = atk[(atk.model == m) & (atk.target_kind.isna())]
            ref = d[(d.eps == REF_EPS) & (d.attack == REF_ATTACK) & (d.ratio == REF_RATIO)]
            last = ref[ref.support == "last"].red_e
            rnd = ref[ref.support == "random"].red_e
            first = ref[ref.support == "first"].red_e
            if len(last) and len(rnd):
                row["RED_last"] = float(last.median())
                row["RED_random"] = float(rnd.median())
                row["RED_first"] = float(first.median())
                row["BVI"] = row["RED_last"] / (row["RED_random"] + 1e-8)
            dense = d[(d.eps == REF_EPS) & (d.attack == REF_ATTACK) & (d.support == "all")]
            if len(dense):
                row["RED_dense@0.05"] = float(dense.red_e.median())
            ctrl = d[(d.eps == REF_EPS) & (d.attack == "random") & (d.support == "all")]
            if len(ctrl):
                row["RED_randnoise@0.05"] = float(ctrl.red_e.median())

        if "BVI" in row:
            row["verdict"] = "confirms" if row["BVI"] > 1.25 else (
                "contradicts" if row["BVI"] < 0.8 else "flat"
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_saliency(cfg, probes, fname, suffix="sal", title=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in cfg["whitebox"]:
        pz = probes.get(m)
        if pz is None:
            continue
        c = mean_curve(pz, suffix)
        if c is None:
            continue
        ctx = len(c)
        # x = distance from the forecast boundary, so all models align at 0.
        x = np.arange(ctx) - ctx
        ax.plot(x, c / (c.sum() + 1e-12), label=_label(cfg, m), lw=1.6, **_style(cfg, m))

    if suffix == "sal":
        for m in cfg["blackbox"]:
            pz = probes.get(m)
            if pz is None:
                continue
            c = mean_curve(pz, "bump")
            if c is None:
                continue
            x = np.arange(len(c)) - len(c)
            ax.plot(x, c / (c.sum() + 1e-12), label=f"{m} (bump)", lw=1.4,
                    color="#555555", linestyle=":", alpha=0.9)

    for pb in range(0, 257, 16):
        ax.axvline(-pb, color="0.9", lw=0.5, zorder=0)
    ax.axhline(1 / 256, color="k", lw=0.8, ls=(0, (2, 2)), alpha=0.5,
               label="uniform (no positional preference)")
    ax.set_xlabel("distance from forecast boundary (0 = last context point)")
    ax.set_ylabel("normalized sensitivity (sums to 1)")
    ax.set_title(title or "Positional sensitivity: |∂L/∂x| by context position")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(-256, 0)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / fname, dpi=150)
    plt.close(fig)


def fig_topk(cfg, probes):
    models = [m for m in cfg["whitebox"] if m in probes]
    mat, labels = [], []
    for m in models:
        c = mean_curve(probes[m], "topk")
        if c is None:
            continue
        mat.append(c)
        labels.append(m)
    if not mat:
        return
    mat = np.stack(mat)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(mat) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="magma",
                   extent=[-mat.shape[1], 0, len(mat) - 0.5, -0.5])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("distance from forecast boundary")
    ax.set_title("Top-25 saliency frequency by context position")
    fig.colorbar(im, ax=ax, label="P(position in series' top-25 |g|)")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "topk_heatmap.png", dpi=150)
    plt.close(fig)


def fig_support_ablation(cfg, atk):
    models = [m for m in cfg["whitebox"] if m in set(atk.model)]
    if not models:
        return
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow), squeeze=False)
    d0 = atk[atk.target_kind.isna()]
    for i, m in enumerate(models):
        ax = axes[i // ncol][i % ncol]
        d = d0[(d0.model == m) & (d0.attack == REF_ATTACK)]
        for support, col in [("last", "#d7301f"), ("random", "#3690c0"),
                             ("first", "#8c96c6"), ("mid", "#a6bddb")]:
            s = d[(d.support == support) & (d.ratio == REF_RATIO)]
            if not len(s):
                continue
            g = s.groupby("eps").red_e.median()
            ax.plot(g.index, g.values, "o-", ms=3, color=col, label=support, lw=1.5)
        dense = d[d.support == "all"].groupby("eps").red_e.median()
        if len(dense):
            ax.plot(dense.index, dense.values, "s-", ms=3, color="0.25",
                    label="all (dense)", lw=1.2)
        ctrl = d0[(d0.model == m) & (d0.attack == "random") & (d0.support == "last")
                  & (d0.ratio == REF_RATIO)].groupby("eps").red_e.median()
        if len(ctrl):
            ax.plot(ctrl.index, np.maximum(ctrl.values, 1e-4), ":", color="k",
                    label="random noise (control)", lw=1.2)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(_label(cfg, m).replace("\n", " "), fontsize=8)
        ax.set_xlabel("ε (× σ_ctx)")
        ax.set_ylabel("median RED_E")
        ax.legend(fontsize=6)
    for j in range(len(models), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Support ablation — {REF_ATTACK}, ratio={REF_RATIO} of context perturbed")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "support_ablation.png", dpi=150)
    plt.close(fig)


def fig_patch_structure(cfg, probes):
    """Is sensitivity a smooth ramp, or a step over the final patch?"""
    models = [m for m in cfg["whitebox"] if m in probes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in models:
        c = mean_curve(probes[m], "patch_idx")
        if c is None:
            continue
        P = int(probes[m]["patch_size"])
        x = np.arange(len(c)) - len(c)  # patch index from the boundary
        axes[0].plot(x, c / (c.sum() + 1e-12), "o-", ms=3,
                     label=f"{m} (P={P})", **_style(cfg, m))
        off = mean_curve(probes[m], "patch_off")
        axes[1].plot(np.arange(len(off)) / (len(off) - 1), off / (off.mean() + 1e-12),
                     "o-", ms=3, label=f"{m} (P={P})", **_style(cfg, m))
    axes[0].set_xlabel("patch index from boundary (-1 = last context patch)")
    axes[0].set_ylabel("normalized |g| mass per patch")
    axes[0].set_title("Sensitivity by patch index")
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("within-patch offset (normalized: 0 = patch start, 1 = patch end)")
    axes[1].set_ylabel("relative |g| (1.0 = flat)")
    axes[1].set_title("Sensitivity by within-patch offset")
    axes[1].legend(fontsize=6)
    fig.suptitle("Patch structure of the vulnerable region "
                 "(does it track patch size? P=16 vs P=32)")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "bvi_by_patchsize.png", dpi=150)
    plt.close(fig)


def fig_stress_vs_gift(cfg, probes):
    """Boundary peak on AR(1) phi=0.95 (where it is CORRECT) vs elsewhere."""
    models = [m for m in cfg["whitebox"] if m in probes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in models:
        pz = probes[m]
        for ax, corpus, title in (
            (axes[0], "gift", "GIFT-Eval (real data)"),
            (axes[1], "stress", "Synthetic stress"),
        ):
            c = mean_curve(pz, "sal", corpus=corpus)
            if c is None:
                continue
            x = np.arange(len(c)) - len(c)
            ax.plot(x, c / (c.sum() + 1e-12), lw=1.4, label=m, **_style(cfg, m))
            ax.set_title(title)
            ax.set_xlabel("distance from forecast boundary")
            ax.set_ylabel("normalized sensitivity")
    # The AR(1) control on its own: optimal forecast depends ONLY on x_T.
    ax = axes[1]
    for m in models:
        k = "stress:family_b_phi|sal"
        if k in probes[m]:
            c = probes[m][k]
            ax.plot(np.arange(len(c)) - len(c), c / (c.sum() + 1e-12),
                    alpha=0.25, lw=0.8, color="k")
    axes[0].legend(fontsize=6)
    fig.suptitle("Saliency by corpus (thin black = AR(1) φ=0.95, where boundary "
                 "reliance is statistically optimal)")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "saliency_by_corpus.png", dpi=150)
    plt.close(fig)


def main() -> int:
    global OUT_ROOT

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(OUT_ROOT))
    args = p.parse_args()

    OUT_ROOT = Path(args.out_root)
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "results").mkdir(parents=True, exist_ok=True)

    cfg = load_adv_config()
    all_models = cfg["whitebox"] + cfg["blackbox"]
    probes = load_probes(all_models)
    atk = load_attacks(cfg["whitebox"])
    print(f"Loaded probes for {sorted(probes)}")
    print(f"Loaded {len(atk)} attack rows for {sorted(set(atk.model)) if len(atk) else []}")

    tbl = bvi_table(cfg, probes, atk)
    tbl.to_csv(OUT_ROOT / "results" / "bvi_table.csv", index=False)
    print("\n=== BVI table ===")
    print(tbl.to_string(index=False))

    if len(atk):
        d0 = atk[atk.target_kind.isna()]
        sup = (d0.groupby(["model", "attack", "support", "ratio", "eps"])
               .red_e.median().reset_index())
        sup.to_csv(OUT_ROOT / "results" / "support_ablation.csv", index=False)

        by_ds = (d0[(d0.eps == REF_EPS) & (d0.attack == REF_ATTACK)]
                 .groupby(["model", "corpus", "dataset", "support"])
                 .red_e.median().reset_index())
        by_ds.to_csv(OUT_ROOT / "results" / "red_e_by_dataset.csv", index=False)

        tg = atk[atk.target_kind.notna()]
        if len(tg):
            t = (tg.groupby(["model", "target_kind", "support", "eps", "attack"])
                 .targeted_red.median().reset_index())
            t.to_csv(OUT_ROOT / "results" / "targeted.csv", index=False)

    if probes:
        fig_saliency(cfg, probes, "saliency_curves.png", "sal")
        fig_saliency(cfg, probes, "saliency_centered.png", "sal_centered",
                     title="Mean-centered saliency (instance-norm leverage removed)")
        fig_topk(cfg, probes)
        fig_patch_structure(cfg, probes)
        fig_stress_vs_gift(cfg, probes)
    if len(atk):
        fig_support_ablation(cfg, atk)

    print(f"\nWrote tables + figures to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
