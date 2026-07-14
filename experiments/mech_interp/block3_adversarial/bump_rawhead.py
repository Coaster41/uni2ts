"""Raw-head bump probe -> results/probe_<model>_rawhead.npz

    python -m experiments.mech_interp.block3_adversarial.bump_rawhead \
        --model timesfm25 --device cuda:0

CONTROL for a specific confound. TimesFM-2.5's shipped inference recipe polishes
the head: it blends a continuous-quantile head, fixes quantile crossing, and
applies ``force_flip_invariance`` (which averages f(x) with -f(-x)). The default
bump probe goes through that polished head, but a gradient path necessarily reads
the **raw** point head.

That matters for a claim we actually make: `timesfm25` has the lowest boundary
concentration of any model (BM_last10 = 0.132), and flip-invariance averaging is
exactly the kind of operation that could smooth a sensitivity profile. So: measure
the bump profile through the *raw* head and see whether the flatness survives. If it
does not, the "timesfm25 is flatter" finding is an artifact of its inference polish,
not a property of the model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mech_interp.block3_adversarial import probes  # noqa: E402
from experiments.mech_interp.block3_adversarial.common import (  # noqa: E402
    RESULTS_DIR,
    adapter_for,
    load_adv_config,
)
from experiments.mech_interp.block3_adversarial.parity_check import (  # noqa: E402
    RAW_HEAD_KWARGS,
)
from experiments.mech_interp.block3_adversarial.run_probe import load_sources  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="timesfm25")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cfg = load_adv_config()
    ctx_len, horizon = cfg["geometry"]["ctx"], cfg["geometry"]["horizon"]
    bump = cfg["bump"]

    raw_kwargs = RAW_HEAD_KWARGS.get(args.model)
    if not raw_kwargs:
        raise SystemExit(f"{args.model} has no polished head to strip; nothing to do.")

    ad = adapter_for(args.model, device=args.device, **raw_kwargs)
    med_idx = min(
        range(len(ad.quantile_levels)),
        key=lambda i: abs(ad.quantile_levels[i] - 0.5),
    )

    out: dict[str, np.ndarray] = {}
    for key, series in load_sources(cfg, args.limit).items():
        prof = probes.bump_profile(
            ad,
            series[:, :ctx_len],
            horizon,
            kappa=bump["kappa"],
            stride=bump["stride"],
            batch_size=bump["batch_size"],
            median_idx=med_idx,
        )
        out[f"{key}|bump"] = prof.astype(np.float32)
        print(f"  [rawhead bump] {key}: peak@{int(prof.argmax())}")

    dest = RESULTS_DIR / f"probe_{args.model}_rawhead.npz"
    np.savez_compressed(dest, **out)
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
