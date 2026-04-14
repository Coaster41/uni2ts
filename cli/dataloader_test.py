#!/usr/bin/env python
"""
test_dataloader.py — Verify MOIRAI dataloader pipeline and visualize patch masking.

Usage:
    python -m cli.dataloader_test -cp conf/pretrain model=moiraic data=gift_eval_pretrain_weighted run_name=dataloader_test
"""

from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from uni2ts.common import hydra_util  # noqa: registers resolvers
from uni2ts.data.loader import DataLoader, PackCollate, PadCollate
from torch.utils.data import ConcatDataset



# ── Config ──────────────────────────────────────────────────────────
NUM_BATCHES = 3
SAMPLES_PER_BATCH = 4
OUTPUT_DIR = Path("./diagnostics")

COLORS = {
    "context":  "#4ecdc4",
    "predict":  "#ff6b6b",
    "padding":  "#d9d9d9",
    "line":     "#2c3e50",
    "missing":  "#e74c3c",
    "obs_good": "#2ecc71",
    "obs_bad":  "#e74c3c",
}
SAMPLE_PALETTE = [
    "#d9d9d9", "#3498db", "#e67e22", "#9b59b6",
    "#1abc9c", "#e74c3c", "#f39c12", "#2ecc71",
]

def find_bad_samples(dataset, n_attempts=5000):
    """Brute-force find samples that crash."""
    import traceback
    bad = []
    for i in range(min(n_attempts, len(dataset))):
        try:
            dataset[i]
        except Exception as e:
            # Identify which sub-dataset
            if isinstance(dataset, ConcatDataset):
                cum = 0
                for j, ds in enumerate(dataset.datasets):
                    if i < cum + len(ds):
                        ds_name = "?"
                        if hasattr(ds, "indexer"):
                            inner = getattr(ds.indexer, "_dataset", None)
                            if inner and hasattr(inner, "info"):
                                ds_name = getattr(inner.info, "dataset_name", "?")
                        print(f"  ✗ idx={i} → sub-dataset {j} ({ds_name}), "
                              f"local_idx={i - cum}: {e}")
                        bad.append((i, j, ds_name))
                        break
                    cum += len(ds)
            else:
                print(f"  ✗ idx={i}: {e}")
                bad.append((i, None, None))
    print(f"\n  Found {len(bad)} bad samples out of {n_attempts} tested")
    return bad

def find_bad_leaf_datasets(train_dataset, attempts_per_leaf=50):
    """Recursively find and test every leaf TimeSeriesDataset."""
    from torch.utils.data import ConcatDataset

    def _get_leaves(ds, path=""):
        if isinstance(ds, ConcatDataset):
            for i, sub in enumerate(ds.datasets):
                yield from _get_leaves(sub, f"{path}/{i}")
        else:
            yield path, ds

    for path, ds in _get_leaves(train_dataset):
        name = "?"
        if hasattr(ds, "indexer"):
            inner = getattr(ds.indexer, "_dataset", None)
            if inner and hasattr(inner, "info"):
                name = getattr(inner.info, "dataset_name", "?")

        n_test = min(attempts_per_leaf, len(ds))
        errors = 0
        first_err = None
        for j in range(n_test):
            try:
                _ = ds[j]
            except Exception as e:
                errors += 1
                if first_err is None:
                    first_err = str(e)
        if errors:
            print(f"  ✗ {path} [{name}] ({type(ds).__name__}) "
                  f"— {errors}/{n_test} FAILED: {first_err}")
        else:
            print(f"  ✓ {path} [{name}] ({type(ds).__name__}) — {n_test} OK")


# ── Console diagnostics ────────────────────────────────────────────
def print_batch_summary(batch: dict, batch_idx: int):
    print(f"\n{'=' * 80}")
    print(f"  BATCH {batch_idx}")
    print(f"{'=' * 80}")
    for key in sorted(batch.keys()):
        val = batch[key]
        if isinstance(val, torch.Tensor):
            print(
                f"  {key:25s}  shape={str(list(val.shape)):20s}"
                f"  dtype={str(val.dtype):15s}"
                f"  min={val.float().min().item():12.4f}"
                f"  max={val.float().max().item():12.4f}"
            )
        else:
            print(f"  {key:25s}  type={type(val).__name__}")


def print_masking_stats(batch: dict, batch_idx: int):
    pred_mask = batch["prediction_mask"]
    sample_id = batch["sample_id"]
    obs_mask = batch["observed_mask"]
    target = batch["target"]

    B, seq_len = pred_mask.shape[0], pred_mask.shape[1]
    patch_size = target.shape[2] if target.dim() == 3 else 1

    print(f"\n  Masking Stats (Batch {batch_idx}) "
          f"[seq_len={seq_len}, patch_size={patch_size}]")
    print(f"  {'─' * 90}")
    print(f"  {'Row':>4}  {'Packed':>6}  {'Patches':>8}  {'Context':>8}"
          f"  {'Predict':>8}  {'Padding':>8}  {'Pred%':>7}  {'ObsMiss%':>9}")
    print(f"  {'─' * 90}")

    for i in range(B):
        sid = sample_id[i]
        pm = pred_mask[i]
        om = obs_mask[i]

        n_padding = (sid == 0).sum().item()
        n_active = (sid > 0).sum().item()
        n_predict = ((pm == 1) & (sid > 0)).sum().item()
        n_context = n_active - n_predict

        pred_pct = n_predict / max(n_active, 1) * 100
        active_om = om[sid > 0]
        miss_pct = (active_om == 0).float().mean().item() * 100 if active_om.numel() > 0 else 0

        n_packed = (sid.unique() > 0).sum().item()

        print(
            f"  {i:>4}  {n_packed:>6}  {n_active:>8}  {n_context:>8}"
            f"  {n_predict:>8}  {n_padding:>8}  {pred_pct:>6.1f}%  {miss_pct:>8.1f}%"
        )


def print_corpus_summary(dataset):
    """Print sub-dataset breakdown for ConcatDataset."""
    if not hasattr(dataset, "datasets"):
        print(f"  Single dataset: num_ts={dataset.num_ts:,}  "
              f"weight={dataset.dataset_weight:.4f}  eff_len={len(dataset):,}")
        return

    total_eff = 0
    rows = []
    for i, ds in enumerate(dataset.datasets):
        if isinstance(ds, torch.utils.data.ConcatDataset):
            # print(f"{i} dataset is a Concat Dataset taking first instance. Total length: {len(ds.datasets)}")
            for j, dss in enumerate(ds.datasets):
                name = "?"
                if hasattr(dss, "indexer"):
                    inner = getattr(dss.indexer, "_dataset", None)
                    if inner and hasattr(inner, "info"):
                        name = getattr(inner.info, "dataset_name", None) or "?"
                eff = len(dss)
                total_eff += eff
                rows.append((f"{i}:{j}", name, dss.num_ts, dss.dataset_weight, eff))

    print(f"\n  {'#':>7}  {'dataset':40s}  {'num_ts':>12}  "
          f"{'weight':>12}  {'eff_len':>12}  {'%':>8}")
    print(f"  {'─' * 96}")
    for i, name, nts, w, eff in rows:
        pct = eff / total_eff * 100
        print(f"  {i:>7}  {name:40s}  {nts:>12,}  "
              f"{w:>12.4f}  {eff:>12,}  {pct:>7.2f}%")
    print(f"  {'─' * 96}")
    print(f"  {'':>7}  {'TOTAL':40s}  {'':>12}  {'':>12}  {total_eff:>12,}  {'100.00%':>8}")


# ── Visualization ──────────────────────────────────────────────────
def visualize_sample(
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    sample_id: torch.Tensor,
    batch_idx: int,
    sample_idx: int,
    output_dir: Path,
):
    """
    4-panel figure for one row in the batch:
      1) Time series with patch-colored backgrounds
      2) Prediction mask bar
      3) Observed mask bar
      4) Sample-ID bar (packing boundaries)
    """
    seq_len = target.shape[0]
    patch_size = target.shape[1] if target.dim() == 2 else 1
    total_len = seq_len * patch_size

    flat_tgt = target.reshape(-1).float().numpy()
    flat_obs = observed_mask.reshape(-1).float().numpy()
    flat_pm = (
        prediction_mask.unsqueeze(-1).expand(-1, patch_size).reshape(-1).numpy()
    )
    flat_sid = (
        sample_id.unsqueeze(-1).expand(-1, patch_size).reshape(-1).numpy()
    )
    t = np.arange(total_len)

    fig, axes = plt.subplots(
        4, 1,
        figsize=(min(28, max(14, total_len * 0.02)), 12),
        gridspec_kw={"height_ratios": [5, 1, 1, 1]},
        sharex=True,
    )
    fig.suptitle(
        f"Batch {batch_idx} | Row {sample_idx} | "
        f"{seq_len} patches × {patch_size} = {total_len} pts | "
        f"packed={int(sample_id.max().item())} samples",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: time series with colored patch backgrounds ──
    ax = axes[0]
    for p in range(seq_len):
        lo, hi = p * patch_size, (p + 1) * patch_size
        sid = int(sample_id[p].item())
        pm = int(prediction_mask[p].item())
        if sid == 0:
            c, a = COLORS["padding"], 0.45
        elif pm == 1:
            c, a = COLORS["predict"], 0.28
        else:
            c, a = COLORS["context"], 0.18
        ax.axvspan(lo, hi, color=c, alpha=a)
        ax.axvline(lo, color="gray", lw=0.2, alpha=0.3)

    obs_vals = np.where(flat_obs > 0, flat_tgt, np.nan)
    ax.plot(t, obs_vals, color=COLORS["line"], lw=0.7, zorder=3)

    miss_idx = np.where((flat_obs == 0) & (flat_sid > 0))[0]
    if len(miss_idx) > 0:
        ax.scatter(miss_idx, flat_tgt[miss_idx], color=COLORS["missing"],
                   s=4, alpha=0.5, zorder=4)

    handles = [
        mpatches.Patch(fc=COLORS["context"], alpha=0.35, label="Context"),
        mpatches.Patch(fc=COLORS["predict"], alpha=0.45, label="Prediction"),
        mpatches.Patch(fc=COLORS["padding"], alpha=0.55, label="Padding"),
        plt.Line2D([0], [0], color=COLORS["line"], lw=1, label="Observed"),
        plt.Line2D([0], [0], marker="o", color=COLORS["missing"],
                   lw=0, ms=4, label="Missing"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.set_ylabel("Value")
    ax.set_title("Time Series with Patch Role Overlay", fontsize=10)

    # ── Panel 2: prediction mask ──
    ax = axes[1]
    for p in range(seq_len):
        lo = p * patch_size
        sid = int(sample_id[p].item())
        pm = int(prediction_mask[p].item())
        c = COLORS["padding"] if sid == 0 else (COLORS["predict"] if pm else COLORS["context"])
        ax.barh(0, patch_size, left=lo, height=0.8, color=c, edgecolor="white", lw=0.3)
        if seq_len <= 64:
            ax.text(lo + patch_size / 2, 0, str(pm), ha="center", va="center", fontsize=5)
    ax.set_yticks([])
    ax.set_ylabel("PredMask", fontsize=8)
    ax.set_title("Prediction Mask  (teal=context, red=predict)", fontsize=9)

    # ── Panel 3: observed mask ──
    ax = axes[2]
    for i_t in range(total_len):
        c = COLORS["obs_good"] if flat_obs[i_t] > 0 else COLORS["obs_bad"]
        ax.barh(0, 1, left=i_t, height=0.8, color=c, edgecolor="none")
    ax.set_yticks([])
    ax.set_ylabel("ObsMask", fontsize=8)
    ax.set_title("Observed Mask  (green=observed, red=missing)", fontsize=9)

    # ── Panel 4: sample ID (packing) ──
    ax = axes[3]
    for p in range(seq_len):
        lo = p * patch_size
        sid = int(sample_id[p].item())
        c = SAMPLE_PALETTE[sid % len(SAMPLE_PALETTE)]
        ax.barh(0, patch_size, left=lo, height=0.8, color=c, edgecolor="white", lw=0.3)
        if seq_len <= 64:
            ax.text(lo + patch_size / 2, 0, str(sid), ha="center", va="center", fontsize=5)
    ax.set_yticks([])
    ax.set_ylabel("SampleID", fontsize=8)
    ax.set_xlabel("Timepoint index")
    ax.set_title("Sample ID  (packing boundaries, 0=padding)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = output_dir / f"batch{batch_idx}_row{sample_idx}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      → saved {fname}")


# ── Structural sanity checks ───────────────────────────────────────
def run_assertions(batch: dict, batch_idx: int):
    """Programmatic checks that catch common masking bugs."""
    pm = batch["prediction_mask"]
    sid = batch["sample_id"]
    om = batch["observed_mask"]
    target = batch["target"]

    B, seq_len = pm.shape
    errors = []

    for i in range(B):
        # 1. Prediction mask should be 0 where sample_id is 0 (padding)
        pad_positions = sid[i] == 0
        if pm[i][pad_positions].any():
            errors.append(f"  Row {i}: prediction_mask is 1 in padding region!")

        # 2. Each packed sample should have contiguous patches
        unique_sids = sid[i].unique()
        for s in unique_sids:
            if s == 0:
                continue
            positions = (sid[i] == s).nonzero(as_tuple=True)[0]
            if len(positions) > 1:
                diffs = positions[1:] - positions[:-1]
                if (diffs != 1).any():
                    errors.append(f"  Row {i}: sample_id {s.item()} is non-contiguous!")

        # 3. Within each packed sample, prediction patches should come AFTER context
        for s in unique_sids:
            if s == 0:
                continue
            mask = sid[i] == s
            sample_pm = pm[i][mask]
            # Find first prediction patch
            pred_positions = (sample_pm == 1).nonzero(as_tuple=True)[0]
            ctx_positions = (sample_pm == 0).nonzero(as_tuple=True)[0]
            if len(pred_positions) > 0 and len(ctx_positions) > 0:
                if ctx_positions[-1] > pred_positions[0]:
                    errors.append(
                        f"  Row {i}, sample {s.item()}: context patches appear "
                        f"AFTER prediction patches (ctx ends at {ctx_positions[-1].item()}, "
                        f"pred starts at {pred_positions[0].item()})!"
                    )

        # 4. Each packed sample should have at least 1 context patch
        for s in unique_sids:
            if s == 0:
                continue
            mask = sid[i] == s
            if (pm[i][mask] == 0).sum() == 0:
                errors.append(f"  Row {i}, sample {s.item()}: NO context patches!")

        # 5. Each packed sample should have at least 1 prediction patch
        for s in unique_sids:
            if s == 0:
                continue
            mask = sid[i] == s
            if (pm[i][mask] == 1).sum() == 0:
                errors.append(f"  Row {i}, sample {s.item()}: NO prediction patches!")

    if errors:
        print(f"\n  ⚠ ASSERTION FAILURES (Batch {batch_idx}):")
        for e in errors:
            print(f"    {e}")
    else:
        print(f"\n  ✓ All structural checks passed (Batch {batch_idx})")

    return len(errors) == 0


# ── Main ───────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_name="default.yaml")
def main(cfg: DictConfig):
    print("\n" + "=" * 80)
    print("  MOIRAI Dataloader Diagnostic Tool")
    print("=" * 80)

    # ── Debug: print resolved config keys ──
    print("\n[DEBUG] Top-level config keys:", list(cfg.keys()))
    if "model" not in cfg:
        from omegaconf import OmegaConf
        print("[DEBUG] Full resolved config:\n")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        raise RuntimeError(
            "Key 'model' not found in config. "
            "Check the config structure above and adjust access paths."
        )
    
    # ── Overrideable params ──
    num_batches = cfg.get("diag_num_batches", NUM_BATCHES)
    samples_vis = cfg.get("diag_samples_per_batch", SAMPLES_PER_BATCH)
    output_dir = Path(cfg.get("diag_output_dir", str(OUTPUT_DIR)))
    diag_batch_size = cfg.get("diag_batch_size", 8)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model (needed for transform_map) ──
    print("\n[1/4] Instantiating model (for transform maps)...")
    model = instantiate(cfg.model, _convert_="all")

    # ── Dataset ──
    print("[2/4] Loading dataset...")
    train_dataset = instantiate(cfg.data).load_dataset(model.train_transform_map)
    print(f"  Total effective length: {len(train_dataset):,}")
    print_corpus_summary(train_dataset)

    # print("\n[2.5/4] Scanning for bad samples (num_workers=0)...")
    # # find_bad_samples(train_dataset, n_attempts=10000)
    find_bad_leaf_datasets(train_dataset, attempts_per_leaf=1000)

    # # ── Dataloader ──
    # print(f"\n[3/4] Creating dataloader (batch_size={diag_batch_size})...")

    # collate_fn = (
    #     instantiate(cfg.train_dataloader.collate_fn)
    #     if "collate_fn" in cfg.train_dataloader
    #     else None
    # )
    # # num_workers = min(cfg.train_dataloader.get("num_workers", 0), 2)
    # num_workers = 0 # temporary use 0 workers

    # dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=diag_batch_size,
    #     batch_size_factor=cfg.train_dataloader.get("batch_size_factor", 1.0),
    #     cycle=False,
    #     num_batches_per_epoch=None,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     collate_fn=collate_fn,
    #     drop_last=True,
    #     fill_last=False,
    # )

    # # ── Pull batches ──
    # print(f"\n[4/4] Pulling {num_batches} batches, visualizing {samples_vis} rows each...")
    # print(f"  Output directory: {output_dir.resolve()}")

    # all_passed = True
    # dl_iter = iter(dataloader)
    # for b in range(num_batches):
    #     try:
    #         batch = next(dl_iter)
    #     except StopIteration:
    #         print(f"\n  ⚠ Dataloader exhausted after {b} batches")
    #         break

    #     print_batch_summary(batch, b)
    #     print_masking_stats(batch, b)
    #     passed = run_assertions(batch, b)
    #     all_passed = all_passed and passed

    #     B = batch["target"].shape[0]
    #     n_vis = min(samples_vis, B)
    #     for s in range(n_vis):
    #         visualize_sample(
    #             target=batch["target"][s],
    #             observed_mask=batch["observed_mask"][s],
    #             prediction_mask=batch["prediction_mask"][s],
    #             sample_id=batch["sample_id"][s],
    #             batch_idx=b,
    #             sample_idx=s,
    #             output_dir=output_dir,
    #         )

    # # ── Summary ──
    # print(f"\n{'=' * 80}")
    # if all_passed:
    #     print("  ✓ ALL CHECKS PASSED")
    # else:
    #     print("  ⚠ SOME CHECKS FAILED — review output above")
    # print(f"  Visualizations saved to: {output_dir.resolve()}")
    # print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()