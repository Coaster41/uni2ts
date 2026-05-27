"""
PR-5a: Per-patch × layer probe score heatmaps (headless CLI).

Loads *_per_patch.json results and saves two figures per feature:
  - probe_patch_{feature}.pdf    : 2D heatmap (patch × layer) per model, side-by-side
  - probe_patch_slice_{feature}.pdf : score vs. patch_idx at the last layer, all models

Models are auto-discovered by globbing results/*_per_patch.json.
Feature metadata (baselines) is loaded from metadata.json when present.

Usage
-----
python -m experiments.mech_interp.block1_probing.plot_probes_patch \
    [--results-dir experiments/mech_interp/block1_probing/results] \
    [--figures-dir experiments/mech_interp/block1_probing/figures]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.block1_probing.train_probes import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
)

_DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), "results")
_DEFAULT_FIGURES = os.path.join(os.path.dirname(__file__), "figures")


def load_per_patch_results(path: str) -> dict[str, dict[int, dict[int, float]]]:
    """Load per-patch JSON, converting string layer and patch keys to ints."""
    with open(path) as f:
        raw = json.load(f)
    return {
        feat: {int(l): {int(p): v for p, v in patch_dict.items()} for l, patch_dict in layer_dict.items()}
        for feat, layer_dict in raw.items()
    }


def load_metadata(results_dir: str) -> dict[str, dict]:
    """Load feature metadata from metadata.json, or return defaults."""
    meta_path = os.path.join(results_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)["features"]
    return {
        **{f: {"type": "regression", "baseline": 0.0, "metric": "R²"} for f in REGRESSION_FEATURES},
        **{
            f: {"type": "classification", "baseline": 1.0 / 8, "metric": "accuracy", "n_classes": 8}
            for f in CLASSIFICATION_FEATURES
        },
    }


def _scores_to_grid(layer_dict: dict[int, dict[int, float]]) -> np.ndarray:
    """Convert {layer: {patch: score}} to 2D array [num_layers, num_patches]."""
    layers = sorted(layer_dict)
    patches = sorted(layer_dict[layers[0]])
    return np.array([[layer_dict[l].get(p, float("nan")) for p in patches] for l in layers])


def plot_probes_patch(results_dir: str, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    # Auto-discover per-patch model files
    per_patch_jsons = sorted(glob.glob(os.path.join(results_dir, "*_per_patch.json")))
    if not per_patch_jsons:
        print(f"No *_per_patch.json files found in {results_dir}. Run train_probes.py --per-patch first.")
        return

    model_names = [os.path.basename(p).replace("_per_patch.json", "") for p in per_patch_jsons]
    results = {name: load_per_patch_results(path) for name, path in zip(model_names, per_patch_jsons)}
    feature_meta = load_metadata(results_dir)

    all_features = list(dict.fromkeys(f for r in results.values() for f in r))
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {name: prop_cycle[i % len(prop_cycle)] for i, name in enumerate(model_names)}

    print(f"Models   : {model_names}")
    print(f"Features : {all_features}")

    for feature in all_features:
        meta = feature_meta.get(feature, {"type": "regression", "baseline": 0.0, "metric": "R²"})
        baseline = meta["baseline"]
        metric_label = meta["metric"]

        # Determine shared color range across models for fair comparison
        all_grids = {
            name: _scores_to_grid(results[name][feature])
            for name in model_names
            if feature in results[name]
        }
        if not all_grids:
            continue

        vmin = baseline
        vmax = max(np.nanmax(g) for g in all_grids.values())
        vmax = max(vmax, baseline + 0.01)  # ensure non-zero range

        # ── Heatmap: patch × layer ────────────────────────────────────────────
        n_models = len(all_grids)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models + 1, 4), squeeze=False)
        im = None
        for ax, model_name in zip(axes[0], model_names):
            if model_name not in all_grids:
                ax.set_visible(False)
                continue
            grid = all_grids[model_name]           # [num_layers, num_patches]
            num_patches = grid.shape[1]
            im = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xlabel("Patch index")
            ax.set_ylabel("Layer")
            ax.set_title(model_name)
            ax.set_xticks(range(0, num_patches, max(1, num_patches // 8)))
            ax.set_yticks(range(grid.shape[0]))

        if im is not None:
            cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
            cb.set_label(f"{metric_label}  (baseline={baseline:.3f})")

        fig.suptitle(f"{feature} — probe score by patch × layer", fontsize=12, fontweight="bold")
        plt.tight_layout()
        heatmap_path = os.path.join(figures_dir, f"probe_patch_{feature}.pdf")
        fig.savefig(heatmap_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {heatmap_path}")

        # ── Slice: score vs. patch_idx at the last layer ──────────────────────
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
        for model_name in model_names:
            if model_name not in all_grids:
                continue
            grid = all_grids[model_name]
            last_layer_scores = grid[-1, :]          # last layer row
            patch_indices = list(range(grid.shape[1]))
            ax.plot(patch_indices, last_layer_scores, marker="o", markersize=3,
                    color=model_colors[model_name], label=model_name)
        ax.set_xlabel("Patch index (context position)")
        ax.set_ylabel(f"{metric_label} ({feature})")
        ax.set_title(f"{feature} — last layer, score vs. context position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        slice_path = os.path.join(figures_dir, f"probe_patch_slice_{feature}.pdf")
        fig.savefig(slice_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {slice_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-patch × layer probe score heatmaps.")
    parser.add_argument("--results-dir", default=_DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", default=_DEFAULT_FIGURES)
    args = parser.parse_args()
    plot_probes_patch(args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
