"""
PR-5: Feature emergence plots (headless CLI).

Loads probe results from results/*.json and saves one PDF per feature to figures/.
Models and features are auto-discovered — adding new models or features requires
only re-running train_probes.py, not editing this file.

Usage
-----
# Default paths:
python -m experiments.mech_interp.block1_probing.plot_probes

# Custom paths:
python -m experiments.mech_interp.block1_probing.plot_probes \
    --results-dir experiments/mech_interp/block1_probing/results \
    --figures-dir experiments/mech_interp/block1_probing/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless — must come before pyplot import
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.block1_probing.train_probes import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
)

_DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), "results")
_DEFAULT_FIGURES = os.path.join(os.path.dirname(__file__), "figures")


def load_results(path: str) -> dict[str, dict[int, float]]:
    """Load a model's probe JSON, converting string layer keys to ints."""
    with open(path) as f:
        raw = json.load(f)
    return {feat: {int(k): v for k, v in layers.items()} for feat, layers in raw.items()}


def load_metadata(results_dir: str) -> dict[str, dict]:
    """Load feature metadata from results_dir/metadata.json, or return defaults."""
    meta_path = os.path.join(results_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)["features"]
    # Fallback: build from train_probes constants
    return {
        **{f: {"type": "regression", "baseline": 0.0, "metric": "R²"} for f in REGRESSION_FEATURES},
        **{
            f: {"type": "classification", "baseline": 1.0 / 8, "metric": "accuracy", "n_classes": 8}
            for f in CLASSIFICATION_FEATURES
        },
    }


def plot_probes(results_dir: str, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    # Auto-discover models
    model_jsons = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    model_jsons = [p for p in model_jsons if os.path.basename(p) != "metadata.json"]
    if not model_jsons:
        print(f"No model JSON files found in {results_dir}. Run train_probes.py first.")
        return

    model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_jsons]
    results = {name: load_results(path) for name, path in zip(model_names, model_jsons)}
    feature_meta = load_metadata(results_dir)

    # Feature + layer discovery
    all_features = list(dict.fromkeys(f for r in results.values() for f in r))
    num_layers = max(len(v) for r in results.values() for v in r.values())
    layers = list(range(num_layers))

    # Deterministic color assignment
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {name: prop_cycle[i % len(prop_cycle)] for i, name in enumerate(model_names)}

    print(f"Models   : {model_names}")
    print(f"Features : {all_features}")
    print(f"Layers   : {num_layers}")

    for feature in all_features:
        meta = feature_meta.get(feature, {"type": "regression", "baseline": 0.0, "metric": "R²"})
        baseline = meta["baseline"]
        metric_label = meta["metric"]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")

        for model_name in model_names:
            if feature not in results[model_name]:
                continue
            scores = [results[model_name][feature].get(l, float("nan")) for l in layers]
            ax.plot(layers, scores, marker="o", markersize=4,
                    color=model_colors[model_name], label=model_name)

        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{metric_label} ({feature})")
        ax.set_title(feature)
        ax.set_xticks(layers)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(figures_dir, f"probe_{feature}.pdf")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot linear probe feature emergence curves.")
    parser.add_argument("--results-dir", default=_DEFAULT_RESULTS,
                        help="Directory containing model *.json result files")
    parser.add_argument("--figures-dir", default=_DEFAULT_FIGURES,
                        help="Directory to save PDF figures")
    args = parser.parse_args()
    plot_probes(args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
