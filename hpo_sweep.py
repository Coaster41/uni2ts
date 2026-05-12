#!/usr/bin/env python
"""
Automated HPO sweep using Optuna.
Launches DDP training via torchrun for each trial.
"""

import optuna
import subprocess
import os
import time
import json
import glob
from pathlib import Path

# === CONFIGURATION ===
N_TRIALS = 25
SCREENING_EPOCHS = 10
N_GPUS = 8
BASE_OUTPUT_DIR = "/srv/disk00/ctadler/uni2ts/outputs/pretrain/moiraie/gift_eval_pretrain_weighted"
STUDY_NAME = "moiraie_encoder_hpo"
STUDY_DB = f"sqlite:///{STUDY_NAME}.db"  # Persistent storage for resume


def objective(trial: optuna.Trial) -> float:
    # --- Sample hyperparameters ---
    lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
    num_training_steps = SCREENING_EPOCHS * 1000  # max_epochs * num_batches_per_epoch
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.02, 0.15)
    warmup_steps = int(warmup_ratio * num_training_steps)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.15, log=True)
    beta2 = trial.suggest_categorical("beta2", [0.95, 0.98, 0.99, 0.999])
    grad_clip = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0])
    
    # Optional: mask ratios
    min_mask_ratio = trial.suggest_float("min_mask_ratio", 0.1, 0.25)
    max_mask_ratio = trial.suggest_float("max_mask_ratio", 0.3, 0.6)

    run_name = f"hpo_trial_{trial.number:03d}"
    expected_output_dir = os.path.join(BASE_OUTPUT_DIR, run_name)

    # --- Build command ---

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: lr={lr:.6f}, warmup={warmup_steps}, "
          f"wd={weight_decay:.4f}, beta2={beta2}, clip={grad_clip}")
    print(f"{'='*60}\n")

    # --- Run training ---
    log_dir = f"/srv/disk00/ctadler/uni2ts/slurm/logs/hpo_trials"
    os.makedirs(log_dir, exist_ok=True)
    trial_log_dir = os.path.join(log_dir, f"trial_{trial.number:03d}")
    stdout_log = os.path.join(log_dir, f"trial_{trial.number:03d}_stdout.txt")
    stderr_log = os.path.join(log_dir, f"trial_{trial.number:03d}_stderr.txt")
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"

    
    cmd = [
        "torchrun",
        f"--nproc_per_node={N_GPUS}",
        "--master_port", str(29500 + trial.number % 100),
        "--log-dir", trial_log_dir,  # This saves per-rank output!
        "-m", "cli.train",
        "-cp", "conf/pretrain",
        f"run_name={run_name}",
        "model=moiraie",
        "data=gift_eval_pretrain_weighted",
        "val_data=gift_eval_test",
        f"model.lr={lr}",
        f"model.num_warmup_steps={warmup_steps}",
        f"model.weight_decay={weight_decay}",
        f"model.beta2={beta2}",
        f"model.min_mask_ratio={min_mask_ratio}",
        f"model.max_mask_ratio={max_mask_ratio}",
        f"trainer.gradient_clip_val={grad_clip}",
        f"trainer.max_epochs={SCREENING_EPOCHS}",
        "train_dataloader.batch_size=256",
        "val_dataloader.batch_size=256",
    ]

    start_time = time.time()
    try:
        with open(stdout_log, "w") as out_f, open(stderr_log, "w") as err_f:
            result = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=err_f,
                timeout=3600,
                cwd="/srv/disk00/ctadler/uni2ts",
                env=env,  # Pass the environment with HYDRA_FULL_ERROR
            )
        elapsed = time.time() - start_time
        print(f"Trial {trial.number} completed in {elapsed/60:.1f} min (returncode={result.returncode})")

        if result.returncode != 0:
            # Print last part of stderr to main log too
            with open(stderr_log, "r") as f:
                err_content = f.read()
            print(f"Trial {trial.number} FAILED. Last 1000 chars of stderr:")
            print(err_content[-1000:])
            return float("inf")

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} TIMED OUT")
        return float("inf")

    # --- Read validation metric ---
    metric_path = os.path.join(expected_output_dir, "best_val_metric.txt")
    
    # Sometimes output dir has slightly different path; search for it
    if not os.path.exists(metric_path):
        # Try glob pattern
        candidates = glob.glob(os.path.join(BASE_OUTPUT_DIR, run_name + "*", "best_val_metric.txt"))
        if candidates:
            metric_path = candidates[0]

    if os.path.exists(metric_path):
        with open(metric_path, "r") as f:
            val_metric = float(f.read().strip())
        print(f"Trial {trial.number}: val_metric = {val_metric:.4f}")
        return val_metric
    else:
        print(f"Trial {trial.number}: metric file not found at {metric_path}")
        # Fallback: try to parse from stdout
        return float("inf")


def main():
    # Create or load study (can resume if interrupted)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_DB,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=8,  # Random trials before Bayesian optimization
            seed=42,
        ),
        pruner=None,  # No pruning for fixed-budget trials
    )

    print(f"Starting HPO sweep: {N_TRIALS} trials, {SCREENING_EPOCHS} epochs each")
    print(f"Estimated time: {N_TRIALS * 30 / 60:.1f} hours (sequential)")
    print(f"Study stored at: {STUDY_DB}")
    print()

    study.optimize(objective, n_trials=N_TRIALS)

    # --- Print results ---
    print("\n" + "=" * 60)
    print("HPO RESULTS")
    print("=" * 60)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Print top 5
    print(f"\nTop 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else float("inf"))
    for t in trials_sorted[:5]:
        print(f"  Trial {t.number}: {t.value:.4f} | {t.params}")

    # Save results
    results_path = f"{STUDY_NAME}_results.json"
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ]
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()