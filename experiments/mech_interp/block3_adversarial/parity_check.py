"""GATE: torch GradForecaster must agree with the numpy block2 adapter.

The readout indexing (which patch slot the forecast is read from, and how the
``npt * Q * P`` block is unflattened) is the single most likely place to get a
plausible-looking but wrong answer. Every downstream number is meaningless until
this passes.

    python -m experiments.mech_interp.block3_adversarial.parity_check \
        --device cuda:7 [--models moiraix_dec_cpm_4 ...]

Post-hoc heads: the TimesFM adapters apply quantile-crossing fixes, flip-invariance
and (for 2.5) a continuous-quantile head *after* the raw point head. A gradient path
necessarily reads the **raw** head, so for those models we rebuild the adapter with
the post-hoc options OFF and compare against that — otherwise we would be comparing
two genuinely different forecasts and calling it a bug. See HANDOFF §3a / trap #5.
Moirai2/MoiraiX have no post-hoc step and match to ~0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    adapter_for,
    grad_forecaster_for,
    load_adv_config,
)

# Median must always match; outer quantiles are allowed to drift where the
# adapter applies post-hoc smoothing (TimesFMX).
MEDIAN_TOL = 1e-3
QUANTILE_TOL = 1e-3

# Adapters that polish the head at inference. To compare like with like, rebuild
# them exposing the raw point head that the gradient path reads.
RAW_HEAD_KWARGS: dict[str, dict] = {
    "timesfm25": {
        "use_continuous_quantile_head": False,
        "fix_quantile_crossing": False,
        "force_flip_invariance": False,
        "infer_is_positive": False,
    },
}


def check(name: str, cfg: dict, device: str, n: int = 8, seed: int = 0) -> bool:
    ctx_len = cfg["geometry"]["ctx"]
    horizon = cfg["geometry"]["horizon"]

    rng = np.random.default_rng(seed)
    # A mix of structure and noise, at a non-unit scale/offset so any missing
    # denormalization shows up loudly.
    t = np.arange(ctx_len, dtype=np.float32)
    context = (
        3.0 * np.sin(2 * np.pi * t / 24.0)[None, :]
        + rng.normal(0, 1.0, size=(n, ctx_len))
        + 10.0
    ).astype(np.float32)

    gf = grad_forecaster_for(name, cfg, device=device)
    x = torch.from_numpy(context).to(device)
    with torch.no_grad():
        q_torch = gf.quantiles(x, horizon).cpu().numpy()

    # The grad path reads the raw point head; timesfm25's adapter defaults to the
    # polished head (continuous-quantile blend + crossing fix + flip invariance),
    # which is a *different forecast*, not a bug. Compare raw-to-raw.
    ad_kwargs = RAW_HEAD_KWARGS.get(name, {})
    ad = adapter_for(name, device=device, **ad_kwargs)
    if ad_kwargs:
        print(f"         (adapter rebuilt with raw head: {sorted(ad_kwargs)})")
    q_numpy = ad.predict_quantiles(context, horizon, batch_size=n)

    if q_torch.shape != q_numpy.shape:
        print(f"  [{name}] SHAPE MISMATCH torch={q_torch.shape} numpy={q_numpy.shape}")
        return False

    med_i = gf.median_idx
    med_err = float(np.abs(q_torch[:, med_i] - q_numpy[:, med_i]).max())
    all_err = float(np.abs(q_torch - q_numpy).max())
    # Scale-relative, since the contexts are O(10).
    denom = float(np.abs(q_numpy).mean()) + 1e-8
    med_rel = med_err / denom
    all_rel = all_err / denom

    ok = med_rel < MEDIAN_TOL
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{name}] {status}  shape={q_torch.shape}  "
        f"median rel-err={med_rel:.2e}  all-quantile rel-err={all_rel:.2e}  "
        f"(predict_next={gf.predict_next}, P={gf.patch_size}, npt={gf.npt})"
    )
    if ok and all_rel >= QUANTILE_TOL:
        print(
            "         note: outer quantiles differ — expected iff this adapter "
            "applies post-hoc quantile fixes (TimesFMX). Attack reads the raw head."
        )

    # A gradient must actually reach the input, or the white-box path is dead.
    x = torch.from_numpy(context).to(device).requires_grad_(True)
    loss = gf.median(x, horizon).abs().sum()
    (g,) = torch.autograd.grad(loss, x)
    gnorm = float(g.abs().sum())
    if gnorm == 0.0 or not np.isfinite(gnorm):
        print(f"  [{name}] FAIL: input gradient is {gnorm} (graph is broken)")
        return False
    print(f"         grad OK: sum|dL/dx| = {gnorm:.3e}")

    del gf, ad
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    cfg = load_adv_config()
    models = args.models or cfg["whitebox"]

    print(f"Parity check (torch GradForecaster vs numpy adapter) on {args.device}")
    results = {}
    for name in models:
        try:
            results[name] = check(name, cfg, args.device, n=args.n)
        except Exception as e:  # noqa: BLE001 — report and keep going
            print(f"  [{name}] ERROR: {type(e).__name__}: {e}")
            results[name] = False

    failed = [k for k, v in results.items() if not v]
    print()
    if failed:
        print(f"GATE FAILED for: {failed}")
        return 1
    print(f"GATE PASSED for all {len(results)} models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
