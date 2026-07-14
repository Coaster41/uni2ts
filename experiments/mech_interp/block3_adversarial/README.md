# block3_adversarial — boundary vulnerability probe

Tests the claim of arXiv 2505.19397 — *"points near the forecast boundary are
most vulnerable"* — on our models, and asks whether that vulnerability is a
consequence of **causal attention**, of **next-token readout**, or of neither.

Geometry is fixed at **ctx=256, H=64, 9 deciles**. H=64 is load-bearing: it is
the single-forward-pass cap for a next-token model with `npt*patch = 64`, so
every attack gradient is one clean backward pass with no AR unroll.

## Run

```shell
# 0. GATE: torch grad path must match the numpy block2 adapter. Nothing
#    downstream is meaningful until this passes.
python -m experiments.mech_interp.block3_adversarial.parity_check --device cuda:7

# 1. Build the shared attack corpora (8 GIFT datasets + 4 stress families).
python -m experiments.mech_interp.block3_adversarial.data

# 2. Probes (saliency + bump) and attacks (support ablation + eps sweep).
python -m experiments.mech_interp.block3_adversarial.run_probe  --model <name> --device cuda:7
python -m experiments.mech_interp.block3_adversarial.run_attack --model <name> --device cuda:7 --targeted
#    or: MODEL=<name> MODE=both sbatch slurm/scripts/adv_block3.sh

# 3. Tables + figures -> tsfm-mi-experiments/adversarial/exp-000-boundary/
python -m experiments.mech_interp.block3_adversarial.analyze
```

Models come from `block2_stress/configs/models.yaml` (single source of truth for
checkpoints); `configs/adv.yaml` selects which and records their architecture.

## Two things the handoff got wrong (both fixed here)

- **`moiraix_dec_cpm_4` is not a next-token decoder.** It is
  `causal=true, mask_inputs=true, predict_next=false, npt=1` — causal attention,
  but it still reads the forecast out of *mask slots*. So `enc_cpm_2` vs
  `dec_cpm_4` isolates the **attention mask alone**. `moirai2_repro_0` (same data
  mix, `predict_next=true, npt=4`) was added as the true next-token arm, which
  lets us vary one axis at a time. `moiraic`/`moiraie` are the true decoder /
  encoder pair on the older mix, used as an independent confirmation.
- **The stress corpus is mixed-geometry.** Most level files are T=400 (ctx 320 +
  H 80); only some are T=320. No `family_d` file exists at T=320, so the
  handoff's `family_d_sine_level_shift` pick is unavailable. `family_b_phi`
  (AR(1), φ=0.95) replaces it and is a better control anyway: for a
  high-persistence AR(1) the Bayes-optimal forecast depends *only* on the last
  context point, so a boundary peak there is statistically **correct**, not a
  vulnerability.

`moiraic`/`moiraie` checkpoints predate the moiraix flags and carry none of them
in `config.json`; they **must** be loaded through their preset module classes
(`grad_kind: moiraic` / `moiraie`) or `predict_next` silently defaults to False
and the forecast is read from the wrong slot.

## A note on `PackedStdScaler`

`src/uni2ts/module/packed_scaler.py` used to do `loc[sample_id == 0] = 0` in
place, which mutates the sqrt/div outputs and makes `backward()` through the
input impossible. It is now out-of-place (`torch.where`); values are identical
and all 996 existing tests pass. Without this, no gradient attack works at all.
