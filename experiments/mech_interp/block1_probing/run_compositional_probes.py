"""
PR-9: Pandey-style compositional probe transfer (Experiment A.3).

For each ordered concept pair (C1, C2):
  1. Train a probe for C1's regression label on the *atomic* subset (only C1 present).
  2. Evaluate the same frozen probe on the *pair* subset (C1 + C2 co-present).
  3. Report delta = atomic_score - pair_score as compositional interference.

Run for all 5 regression-capable concepts → 20 directed transfers (C(5,2)×2).

Usage
-----
# Smoke-test with tiny in-memory models:
python -m experiments.mech_interp.block1_probing.run_compositional_probes \
    --output-dir /tmp/pr9_smoke/

# Full run with real checkpoints:
python -m experiments.mech_interp.block1_probing.run_compositional_probes \
    --moiraie-ckpt /srv/disk00/ctadler/uni2ts/outputs/pretrain/moiraie/gift_eval_pretrain_weighted/moiraie_training_7/HF_checkpoints/last \
    --moiraic-ckpt /srv/disk00/ctadler/uni2ts/outputs/pretrain/moiraic/gift_eval_pretrain_weighted/moiraic_training_11/HF_checkpoints/last \
    --output-dir experiments/mech_interp/block1_probing/results/synthetic/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.mech_interp.lib.synthetic import generate_composite_dataset, split_dataset
from experiments.mech_interp.block1_probing.train_probes import (
    PATCH_SIZE,
    CONTEXT_PATCHES,
    PRED_PATCHES,
    extract_activations,
    fit_probe,
)
from experiments.mech_interp.lib.utils import _load_module

# 5 regression-capable concepts: (concept_name, concept_col_idx, label_key, feature_type)
COMPOSITIONAL_CONCEPTS = [
    ("trend",        0, "slope",              "regression"),
    ("level_shift",  1, "level_magnitude",    "regression"),
    ("ar1",          2, "ar_phi",             "regression"),
    ("seasonal",     3, "seasonal_amplitude", "regression"),
    ("var_shift",    4, "log_sigma_ratio",    "regression"),
]


def run_compositional_probes(
    module,
    dataset: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 128,
    patch_size: int = PATCH_SIZE,
    context_patches: int = CONTEXT_PATCHES,
    pred_patches: int = PRED_PATCHES,
    device: str | torch.device = "cpu",
    pooling: str = "mean_ctx",
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    """
    Pandey-style probe transfer for all concept pairs.

    Returns
    -------
    {c1_name: {c2_name: {layer_idx: {"atomic": float, "pair": float|None, "delta": float|None}}}}
        delta = atomic_score - pair_score  (positive = C2 interferes with C1)
        "pair"/"delta" are None if pair_va has fewer than 5 rows.
    """
    concept_mask = dataset["concept_mask"]          # [n, 7] bool
    n_concepts = concept_mask.sum(axis=1)           # [n] int

    series = dataset["series"]                       # [n, T]
    n = len(series)

    print(f"  [{pooling}] Extracting activations for all {n} examples...")
    X_all = extract_activations(
        module, series,
        batch_size=batch_size,
        patch_size=patch_size,
        context_patches=context_patches,
        pred_patches=pred_patches,
        device=device,
        pooling=pooling,
    )  # {layer_idx: np.ndarray[n, d_model]}

    layer_keys = sorted(X_all.keys())
    results: dict[str, dict[str, dict[int, dict[str, float]]]] = {}

    for c1_name, c1_idx, c1_label, c1_ftype in COMPOSITIONAL_CONCEPTS:
        # Atomic subset: exactly one concept present and it is c1
        atomic_c1 = (n_concepts == 1) & concept_mask[:, c1_idx]
        atomic_all = np.where(atomic_c1)[0]
        atomic_tr = atomic_all[np.isin(atomic_all, train_idx)]
        atomic_va = atomic_all[np.isin(atomic_all, val_idx)]

        if len(atomic_tr) < 10 or len(atomic_va) < 5:
            print(f"  Skipping {c1_name}: only {len(atomic_tr)} atomic train / {len(atomic_va)} val rows")
            continue

        y_atomic_tr = dataset[c1_label][atomic_tr]
        y_atomic_va = dataset[c1_label][atomic_va]

        results[c1_name] = {}

        for c2_name, c2_idx, _, _ in COMPOSITIONAL_CONCEPTS:
            if c2_idx == c1_idx:
                continue

            # Pair subset: exactly two concepts, both c1 and c2 present
            pair_c1_c2 = (n_concepts == 2) & concept_mask[:, c1_idx] & concept_mask[:, c2_idx]
            pair_all = np.where(pair_c1_c2)[0]
            pair_va = pair_all[np.isin(pair_all, val_idx)]
            y_pair_va = dataset[c1_label][pair_va] if len(pair_va) >= 5 else None

            layer_dict: dict[int, dict[str, float]] = {}

            for layer_idx in layer_keys:
                X_tr = X_all[layer_idx][atomic_tr]
                X_atomic_va = X_all[layer_idx][atomic_va]

                # Train once on atomic(C1) and score on atomic val
                score_atomic = fit_probe(X_tr, X_atomic_va, y_atomic_tr, y_atomic_va, c1_ftype)

                if y_pair_va is not None:
                    X_pair_va = X_all[layer_idx][pair_va]
                    score_pair = fit_probe(X_tr, X_pair_va, y_atomic_tr, y_pair_va, c1_ftype)
                    delta = score_atomic - score_pair
                else:
                    score_pair = None
                    delta = None

                layer_dict[layer_idx] = {
                    "atomic": score_atomic,
                    "pair":   score_pair,
                    "delta":  delta,
                }

            results[c1_name][c2_name] = layer_dict
            n_pair_str = str(len(pair_va)) if y_pair_va is not None else "skipped"
            best_atomic = max(v["atomic"] for v in layer_dict.values())
            print(f"    {c1_name}→{c2_name}: best atomic={best_atomic:.3f}, pair_va_n={n_pair_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compositional probe transfer (PR-9).")
    parser.add_argument("--moiraie-ckpt", default=None)
    parser.add_argument("--moiraic-ckpt", default=None)
    parser.add_argument("--dataset-path", default=None,
                        help="Path to .npz dataset (generated in memory if omitted)")
    parser.add_argument("--output-dir",
                        default="experiments/mech_interp/block1_probing/results/synthetic")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_path:
        from experiments.mech_interp.lib import load_dataset
        print(f"Loading dataset from {args.dataset_path}")
        dataset = load_dataset(args.dataset_path)
    else:
        print("Generating composite synthetic dataset (n=5000, seed=42)...")
        dataset = generate_composite_dataset(n=5000, seed=42)

    train_idx, val_idx = split_dataset(dataset, n_train=4000, seed=42)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")

    for model_name, ckpt_path in [("moiraie", args.moiraie_ckpt), ("moiraic", args.moiraic_ckpt)]:
        print(f"\n=== {model_name} ===")
        module = _load_module(ckpt_path, model_name, device=args.device)

        combined: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

        for pooling in ("mean_ctx", "last_ctx"):
            print(f"\n--- {model_name} / {pooling} ---")
            raw = run_compositional_probes(
                module, dataset, train_idx, val_idx,
                batch_size=args.batch_size,
                device=args.device,
                pooling=pooling,
            )
            # Flatten nested c1/c2 keys into "c1→c2" and convert int layer keys to str
            flat: dict[str, dict[str, dict[str, float]]] = {}
            for c1_name, inner in raw.items():
                for c2_name, layer_dict in inner.items():
                    pair_key = f"{c1_name}→{c2_name}"
                    flat[pair_key] = {
                        str(layer_idx): entry
                        for layer_idx, entry in layer_dict.items()
                    }
            combined[pooling] = flat

        out_path = os.path.join(args.output_dir, f"{model_name}_compositional.json")
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
