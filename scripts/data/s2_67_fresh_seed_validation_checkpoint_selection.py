from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)
from s2_24_full_train_loader_sanity import CachedRealTrainWindowDataset
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_39_b0_epsilon_to_v_conversion_sanity import make_v_target, q_sample
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import build_reverse_schedule
from s2_47_b1_architecture_warmstart_sanity import (
    SubjectIndexedDataset,
    state_dict_sha256,
)
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import (
    compute_candidate_metrics,
    sample_b1_candidate_state,
)
from s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep import (
    build_model_from_state,
    evaluate_eligibility,
    full_log_psd_mae,
)
from s2_57_b2_matched_lambda0_full_continuation_control import (
    build_b2_model,
    finite_gradient_summary,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

FROZEN_B1_GENERATOR = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b1"
    / "b1_subject_state_raw_epoch010_v1"
    / "checkpoints"
    / "b1_subject_state_raw_epoch010_generator_v1.pt"
)

B1_SELECTION_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b1"
    / "b1_subject_state_raw_epoch010_v1"
    / "metrics"
    / "b1_frozen_selection_v1.json"
)

S2_66_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "validation_checkpoint_selector_v1"
)

S2_66_PROTOCOL = (
    S2_66_ROOT
    / "metrics"
    / "s2_66_validation_checkpoint_selector_protocol_v1.json"
)

S2_66_SUMMARY = (
    S2_66_ROOT
    / "metrics"
    / "s2_66_validation_checkpoint_selector_summary_v1.json"
)

S2_66_BASELINES = (
    S2_66_ROOT
    / "metrics"
    / "s2_66_validation_b1_baselines_v1.json"
)

VAL_DEV_REAL_NPZ = (
    S2_66_ROOT
    / "references"
    / "s2_66_val_dev_real_reference_v1.npz"
)

VAL_CONF_REAL_NPZ = (
    S2_66_ROOT
    / "references"
    / "s2_66_val_conf_real_reference_v1.npz"
)

VAL_DEV_B1_SYNTHETIC_NPZ = (
    S2_66_ROOT
    / "generated"
    / "s2_66_val_dev_b1_synthetic_v1.npz"
)

VAL_CONF_B1_SYNTHETIC_NPZ = (
    S2_66_ROOT
    / "generated"
    / "s2_66_val_conf_b1_synthetic_v1.npz"
)


# ============================================================
# s2_67 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "fresh_seed_validation_checkpoint_selection_v1"
)

CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
GENERATED_DEV_DIR = RUN_ROOT / "generated" / "val_dev"
GENERATED_CONF_DIR = RUN_ROOT / "generated" / "val_conf"
METRIC_DIR = RUN_ROOT / "metrics"
LOG_DIR = RUN_ROOT / "logs"

PROTOCOL_SNAPSHOT = (
    METRIC_DIR
    / "s2_67_selector_protocol_snapshot_v1.json"
)

TRAINING_HISTORY_CSV = (
    LOG_DIR
    / "s2_67_fresh_training_history_v1.csv"
)

VAL_DEV_METRICS_CSV = (
    METRIC_DIR
    / "s2_67_val_dev_checkpoint_metrics_v1.csv"
)

VAL_CONF_METRICS_CSV = (
    METRIC_DIR
    / "s2_67_val_conf_selected_checkpoint_metrics_v1.csv"
)

SELECTION_JSON = (
    METRIC_DIR
    / "s2_67_checkpoint_selection_decisions_v1.json"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_67_fresh_seed_selector_summary_v1.json"
)


# ============================================================
# Frozen training recipe
# ============================================================

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

TIMESTEPS = 1000
COSINE_S = 0.008

EPOCHS = [1, 2, 3, 4, 5]

BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

MIXED_PRECISION = True
EMA_ENABLED = False
LAMBDA_SPECTRAL = 0.0

MAX_AMP_SKIPS_PER_EPOCH = 5
MAX_CONSECUTIVE_AMP_SKIPS = 3

N_CHANNELS = 19
N_TIMES = 512

REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

EXPECTED_COHORT_SUBJECTS = 64

EXPECTED_REAL_SHAPE = (
    EXPECTED_COHORT_SUBJECTS,
    2,
    REAL_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)

EXPECTED_SYNTH_SHAPE = (
    EXPECTED_COHORT_SUBJECTS,
    2,
    SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def atomic_npz_save(path: Path, **arrays) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def save_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Cannot save empty CSV: {path}")

    temp = Path(str(path) + ".tmp")

    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp, path)


def checkpoint_path(run_name: str, epoch: int) -> Path:
    return (
        CHECKPOINT_DIR
        / f"s2_67_{run_name.lower()}_epoch_{epoch:03d}_raw.pt"
    )


def dev_synthetic_path(run_name: str, epoch: int) -> Path:
    return (
        GENERATED_DEV_DIR
        / f"s2_67_{run_name.lower()}_epoch_{epoch:03d}_val_dev_synthetic_v1.npz"
    )


def conf_synthetic_path(run_name: str, epoch: int) -> Path:
    return (
        GENERATED_CONF_DIR
        / f"s2_67_{run_name.lower()}_epoch_{epoch:03d}_val_conf_synthetic_v1.npz"
    )


def load_reference_pair(
    *,
    real_path: Path,
    b1_synth_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[str],
]:
    real_package = np.load(real_path, allow_pickle=False)
    b1_package = np.load(b1_synth_path, allow_pickle=False)

    real_z = real_package["real_z"].astype(np.float32, copy=False)
    b1_synth_z = b1_package["synth_z"].astype(np.float32, copy=False)

    subject_indices = (
        real_package["selected_subject_indices"]
        .astype(np.int64, copy=False)
        .tolist()
    )

    subject_ids = (
        real_package["selected_subject_ids"]
        .astype(str)
        .tolist()
    )

    b1_indices = (
        b1_package["selected_subject_indices"]
        .astype(np.int64, copy=False)
        .tolist()
    )

    b1_ids = (
        b1_package["selected_subject_ids"]
        .astype(str)
        .tolist()
    )

    if subject_indices != b1_indices:
        raise RuntimeError("Real/B1 subject-index mismatch.")

    if subject_ids != b1_ids:
        raise RuntimeError("Real/B1 subject-ID mismatch.")

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(f"Unexpected Real shape: {real_z.shape}")

    if b1_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(f"Unexpected B1 synthetic shape: {b1_synth_z.shape}")

    return (
        real_z,
        b1_synth_z,
        subject_indices,
        subject_ids,
    )


def evaluate_bundle(
    *,
    real_z: np.ndarray,
    synth_z: np.ndarray,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    generic, _, _ = compute_candidate_metrics(
        real_z=real_z,
        synth_z=synth_z,
        scaler=scaler,
    )

    full_psd = full_log_psd_mae(
        real_z=real_z,
        synth_z=synth_z,
        scaler=scaler,
    )

    return {
        "generic": generic,
        "full_log_psd": full_psd,
    }


def flatten_candidate_metrics(
    *,
    run_name: str,
    epoch: int,
    stage: str,
    bundle: dict[str, Any],
    b1_bundle: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    eligibility = evaluate_eligibility(
        candidate=bundle,
        b1_baseline=b1_bundle,
    )

    generic = bundle["generic"]
    subject = generic["subject_identity"]

    failed_criteria = [
        criterion
        for criterion, passed
        in eligibility["criteria"].items()
        if not bool(passed)
    ]

    row = {
        "run": run_name,
        "stage": stage,
        "epoch": epoch,
        "eligible_vs_B1": bool(
            eligibility["eligible"]
        ),
        "failed_criteria": ";".join(failed_criteria),
        "full_psd_mae_db": float(
            bundle["full_log_psd"]["mean_mae_db"]
        ),
        "mean_bandpower_mae_db": float(
            generic["mean_bandpower_mae_db"]
        ),
        "alpha_effect_mae_db": float(
            generic["alpha_effect"]["channel_mae_db"]
        ),
        "scale_EO": float(
            generic["scale"]["EO"]["normalized_std_ratio"]
        ),
        "scale_EC": float(
            generic["scale"]["EC"]["normalized_std_ratio"]
        ),
        "correlation_r_EO": float(
            generic["spatial"]["EO"]["correlation_r"]
        ),
        "correlation_r_EC": float(
            generic["spatial"]["EC"]["correlation_r"]
        ),
        "alpha_dwpli2_r_EO": float(
            generic["spatial"]["EO"]["alpha_dwpli2_r"]
        ),
        "alpha_dwpli2_r_EC": float(
            generic["spatial"]["EC"]["alpha_dwpli2_r"]
        ),
        "subject_top1": float(
            subject["subject_retrieval_top1_accuracy"]
        ),
        "subject_top5": float(
            subject["subject_retrieval_top5_accuracy"]
        ),
        "subject_fingerprint_r_median": float(
            subject["subject_fingerprint_pearson_r_median"]
        ),
        "subject_fingerprint_rmse_mean": float(
            subject["subject_fingerprint_standardized_rmse_mean"]
        ),
        "subject_margin": float(
            subject["subject_correct_vs_median_wrong_margin_mean"]
        ),
        "fixed_wrong_win_fraction": float(
            subject["subject_correct_better_than_fixed_wrong_fraction"]
        ),
    }

    return row, eligibility


# ============================================================
# Fresh training
# ============================================================

def train_fresh_run(
    *,
    run_name: str,
    seeds: dict[str, int],
    frozen_b1: dict[str, Any],
    train_dataset,
    device: torch.device,
) -> tuple[
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    print("\n" + "=" * 80)
    print(f"TRAIN FRESH RUN {run_name}")
    print("=" * 80)

    print(f"master seed: {seeds['master_seed']}")
    print(f"loader seed: {seeds['loader_seed']}")
    print(f"noise seed: {seeds['noise_seed']}")

    set_seed(int(seeds["master_seed"]))

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seeds["loader_seed"]))

    noise_generator = torch.Generator(device="cuda")
    noise_generator.manual_seed(int(seeds["noise_seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    if len(train_loader) != 748:
        raise RuntimeError("Expected 748 Train batches.")

    model = build_b2_model(
        frozen_b1,
        device,
    )

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=MIXED_PRECISION,
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    alpha_bar = schedule_cpu["alpha_cumprod"].to(
        device=device,
        dtype=torch.float32,
    )

    checkpoints = {}
    history_rows = []

    for epoch in EPOCHS:
        epoch_start = time.perf_counter()

        model.train()

        train_v_sum = 0.0
        presented = 0

        successful_steps = 0
        amp_skips = 0
        consecutive_amp_skips = 0

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            x0 = batch["x"].to(
                device,
                non_blocking=True,
            )

            state = batch["y"].to(
                device,
                non_blocking=True,
            )

            subject = batch["subject_index"].to(
                device,
                non_blocking=True,
            )

            current_batch_size = int(
                x0.shape[0]
            )

            timesteps = torch.randint(
                0,
                TIMESTEPS,
                (current_batch_size,),
                device=device,
                dtype=torch.long,
                generator=noise_generator,
            )

            epsilon = torch.randn(
                x0.shape,
                device=device,
                dtype=x0.dtype,
                generator=noise_generator,
            )

            xt = q_sample(
                x0,
                epsilon,
                timesteps,
                alpha_bar,
            )

            target_v = make_v_target(
                x0,
                epsilon,
                timesteps,
                alpha_bar,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=MIXED_PRECISION,
            ):
                pred_v = model(
                    xt,
                    timesteps,
                    state,
                    subject,
                )

            v_loss = torch.mean(
                (
                    pred_v.float()
                    - target_v.float()
                )
                ** 2
            )

            if not torch.isfinite(v_loss):
                raise RuntimeError(
                    f"{run_name}: nonfinite v-loss."
                )

            v_value = float(
                v_loss.item()
            )

            grad_scaler.scale(
                v_loss
            ).backward()

            grad_scaler.unscale_(
                optimizer
            )

            finite, _, _, bad_names = finite_gradient_summary(
                model
            )

            if finite:
                clip_return = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP_NORM,
                )

                if not torch.isfinite(
                    clip_return
                ):
                    raise RuntimeError(
                        f"{run_name}: nonfinite clip norm."
                    )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()

                successful_steps += 1
                consecutive_amp_skips = 0
            else:
                scale_before = float(
                    grad_scaler.get_scale()
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()

                amp_skips += 1
                consecutive_amp_skips += 1

                print(
                    f"AMP SKIP | run={run_name} "
                    f"| epoch={epoch} "
                    f"| batch={batch_index} "
                    f"| scale={scale_before:g}->{grad_scaler.get_scale():g} "
                    f"| bad={bad_names[:3]}"
                )

                if amp_skips > MAX_AMP_SKIPS_PER_EPOCH:
                    raise RuntimeError(
                        f"{run_name}: too many AMP skips."
                    )

                if (
                    consecutive_amp_skips
                    >= MAX_CONSECUTIVE_AMP_SKIPS
                ):
                    raise RuntimeError(
                        f"{run_name}: too many consecutive AMP skips."
                    )

            train_v_sum += (
                v_value
                * current_batch_size
            )

            presented += (
                current_batch_size
            )

            if (
                batch_index == 1
                or batch_index % 100 == 0
                or batch_index == len(train_loader)
            ):
                print(
                    f"{run_name} "
                    f"| epoch={epoch}/5 "
                    f"| batch={batch_index:>3}/748 "
                    f"| v={train_v_sum/presented:.6f} "
                    f"| skips={amp_skips}"
                )

        if presented != EXPECTED_TRAIN_WINDOWS:
            raise RuntimeError(
                f"{run_name}: Train accounting mismatch."
            )

        if (
            successful_steps
            + amp_skips
            != len(train_loader)
        ):
            raise RuntimeError(
                f"{run_name}: step accounting mismatch."
            )

        model_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        model_hash = state_dict_sha256(
            model_state
        )

        checkpoint = {
            "status": (
                "S2_67_FRESH_VALIDATION_SELECTOR_CHECKPOINT"
            ),
            "run": run_name,
            "epoch": epoch,
            "model_state_dict": model_state,
            "model_state_sha256": model_hash,
            "training_recipe": {
                "lambda_spectral": 0.0,
                "prediction_target": "v",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "mixed_precision": MIXED_PRECISION,
                "EMA_enabled": EMA_ENABLED,
            },
            "seeds": seeds,
            "train_v_mse": float(
                train_v_sum
                / presented
            ),
            "successful_optimizer_steps": successful_steps,
            "amp_skipped_steps": amp_skips,
            "Validation_EEG_accessed_during_training": False,
            "Final_Test_accessed": False,
        }

        atomic_torch_save(
            checkpoint,
            checkpoint_path(
                run_name,
                epoch,
            ),
        )

        checkpoints[
            epoch
        ] = checkpoint

        history_rows.append(
            {
                "run": run_name,
                "epoch": epoch,
                "train_v_mse": checkpoint["train_v_mse"],
                "successful_optimizer_steps": successful_steps,
                "amp_skipped_steps": amp_skips,
                "model_state_sha256": model_hash,
                "epoch_seconds": float(
                    time.perf_counter()
                    - epoch_start
                ),
            }
        )

        print(
            f"{run_name} E{epoch} COMPLETE "
            f"| Train-v={checkpoint['train_v_mse']:.6f} "
            f"| skips={amp_skips} "
            f"| hash={model_hash[:16]}..."
        )

    del model
    torch.cuda.empty_cache()

    return checkpoints, history_rows


# ============================================================
# Free generation
# ============================================================

def generate_candidate(
    *,
    run_name: str,
    epoch: int,
    checkpoint: dict[str, Any],
    output_path: Path,
    frozen_b1: dict[str, Any],
    selected_subjects: list[int],
    selected_subject_ids: list[str],
    sampling_seed: int,
    reverse_schedule,
    device: torch.device,
    cohort_name: str,
) -> np.ndarray:
    if output_path.is_file():
        package = np.load(
            output_path,
            allow_pickle=False,
        )

        if (
            package["selected_subject_indices"]
            .astype(np.int64)
            .tolist()
            != selected_subjects
        ):
            raise RuntimeError(
                f"{run_name} E{epoch} {cohort_name}: cached cohort mismatch."
            )

        if (
            package["selected_subject_ids"]
            .astype(str)
            .tolist()
            != selected_subject_ids
        ):
            raise RuntimeError(
                f"{run_name} E{epoch} {cohort_name}: cached ID mismatch."
            )

        if (
            str(
                package["model_state_sha256"][0]
            )
            != checkpoint["model_state_sha256"]
        ):
            raise RuntimeError(
                f"{run_name} E{epoch} {cohort_name}: cached model hash mismatch."
            )

        if int(
            package["sampling_seed"][0]
        ) != sampling_seed:
            raise RuntimeError(
                f"{run_name} E{epoch} {cohort_name}: cached seed mismatch."
            )

        synth_z = package["synth_z"].astype(
            np.float32,
            copy=False,
        )

        if synth_z.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"{run_name} E{epoch} {cohort_name}: cached shape mismatch."
            )

        print(
            f"{run_name} E{epoch} {cohort_name}: validated cache -> REUSE"
        )

        return synth_z

    model = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=(
            checkpoint["model_state_dict"]
        ),
        device=device,
    )

    print("\n" + "-" * 80)
    print(
        f"FREE GENERATION — {run_name} E{epoch} — {cohort_name}"
    )
    print("-" * 80)

    start = time.perf_counter()

    eo = sample_b1_candidate_state(
        model=model,
        selected_subjects=selected_subjects,
        eye_state_value=0,
        reverse_schedule=reverse_schedule,
        device=device,
        base_seed=sampling_seed,
    )

    ec = sample_b1_candidate_state(
        model=model,
        selected_subjects=selected_subjects,
        eye_state_value=1,
        reverse_schedule=reverse_schedule,
        device=device,
        base_seed=sampling_seed,
    )

    synth_z = np.stack(
        [eo, ec],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    if synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"{run_name} E{epoch} {cohort_name}: unexpected shape "
            f"{synth_z.shape}"
        )

    atomic_npz_save(
        output_path,
        selected_subject_indices=np.asarray(
            selected_subjects,
            dtype=np.int64,
        ),
        selected_subject_ids=np.asarray(
            selected_subject_ids,
            dtype="U32",
        ),
        synth_z=synth_z,
        sampling_seed=np.asarray(
            [sampling_seed],
            dtype=np.int64,
        ),
        model_state_sha256=np.asarray(
            [checkpoint["model_state_sha256"]],
            dtype="U64",
        ),
        generation_seconds=np.asarray(
            [time.perf_counter() - start],
            dtype=np.float64,
        ),
    )

    print(
        f"{run_name} E{epoch} {cohort_name} generation: "
        f"{(time.perf_counter()-start)/60:.2f} min"
    )

    del model
    torch.cuda.empty_cache()

    return synth_z


# ============================================================
# Selection
# ============================================================

def select_epoch_from_val_dev(
    rows: list[dict[str, Any]],
) -> int | None:
    eligible = [
        row
        for row in rows
        if bool(
            row["eligible_vs_B1"]
        )
    ]

    if not eligible:
        return None

    selected = min(
        eligible,
        key=lambda row: (
            float(
                row["full_psd_mae_db"]
            ),
            float(
                row["mean_bandpower_mae_db"]
            ),
            float(
                row["alpha_effect_mae_db"]
            ),
            int(
                row["epoch"]
            ),
        ),
    )

    return int(
        selected["epoch"]
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — s2_67 FRESH-SEED VALIDATION CHECKPOINT SELECTION")
    print("=" * 80)

    print("\nScientific question:")
    print(
        "  Does the pre-frozen s2_66 VAL-DEV selector + VAL-CONF "
        "confirmation mechanism replicate on two completely fresh training seeds?"
    )

    print("\nGovernance:")
    print("  Train EEG: YES")
    print("  Validation EEG: YES, exactly per frozen s2_66 protocol")
    print("  Final Test EEG: NO")
    print("  Threshold changes: NO")
    print("  Winner-picking between training seeds: NO")

    for directory in (
        CHECKPOINT_DIR,
        GENERATED_DEV_DIR,
        GENERATED_CONF_DIR,
        METRIC_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    required_files = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        S2_66_PROTOCOL,
        S2_66_SUMMARY,
        S2_66_BASELINES,
        VAL_DEV_REAL_NPZ,
        VAL_CONF_REAL_NPZ,
        VAL_DEV_B1_SYNTHETIC_NPZ,
        VAL_CONF_B1_SYNTHETIC_NPZ,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    with B1_SELECTION_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        b1_selection = json.load(f)

    with S2_66_PROTOCOL.open(
        "r",
        encoding="utf-8",
    ) as f:
        selector_protocol = json.load(f)

    with S2_66_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_66_summary = json.load(f)

    if (
        b1_selection["status"]
        != "B1_SELECTION_FROZEN"
    ):
        raise RuntimeError("B1 is not frozen.")

    if (
        s2_66_summary["status"]
        != "S2_66_VALIDATION_CHECKPOINT_SELECTOR_READY"
    ):
        raise RuntimeError("s2_66 selector is not ready.")

    if (
        selector_protocol["status"]
        != "S2_66_VALIDATION_CHECKPOINT_SELECTOR_PROTOCOL_FROZEN"
    ):
        raise RuntimeError("Unexpected s2_66 protocol status.")

    primary_seeds = selector_protocol[
        "future_training_runs"
    ][
        "PRIMARY"
    ]

    replication_seeds = selector_protocol[
        "future_training_runs"
    ][
        "REPLICATION"
    ]

    dev_sampling_seed = int(
        selector_protocol[
            "B1_baseline_sampling"
        ][
            "VAL_DEV_seed"
        ]
    )

    conf_sampling_seed = int(
        selector_protocol[
            "B1_baseline_sampling"
        ][
            "VAL_CONF_seed"
        ]
    )

    run_seeds = {
        "PRIMARY": {
            key: int(value)
            for key, value
            in primary_seeds.items()
        },
        "REPLICATION": {
            key: int(value)
            for key, value
            in replication_seeds.items()
        },
    }

    protocol_snapshot = {
        "status": (
            "S2_67_SELECTOR_EXECUTION_PROTOCOL_SNAPSHOT"
        ),
        "source_protocol_sha256": (
            sha256_file(
                S2_66_PROTOCOL
            )
        ),
        "source_protocol_path": str(
            S2_66_PROTOCOL
        ),
        "run_seeds": run_seeds,
        "VAL_DEV_sampling_seed": dev_sampling_seed,
        "VAL_CONF_sampling_seed": conf_sampling_seed,
        "selection_rule": (
            selector_protocol[
                "VAL_DEV_selection_rule"
            ]
        ),
        "confirmation_rule": (
            selector_protocol[
                "VAL_CONF_confirmation_rule"
            ]
        ),
        "replication_success_rule": (
            selector_protocol[
                "replication_success_rule"
            ]
        ),
        "final_candidate_policy": (
            selector_protocol[
                "final_candidate_if_successful"
            ]
        ),
        "Final_Test_accessed": False,
    }

    if PROTOCOL_SNAPSHOT.is_file():
        with PROTOCOL_SNAPSHOT.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(f)

        if existing != protocol_snapshot:
            raise RuntimeError(
                "Conflicting s2_67 protocol snapshot."
            )

        print("identical s2_67 protocol snapshot already exists.")
    else:
        atomic_json_save(
            protocol_snapshot,
            PROTOCOL_SNAPSHOT,
        )

    b1_hash_before = sha256_file(
        FROZEN_B1_GENERATOR
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    if (
        state_dict_sha256(
            frozen_b1["model_state_dict"]
        )
        != frozen_b1["selected_model_state_sha256"]
    ):
        raise RuntimeError(
            "Frozen B1 model-state hash mismatch."
        )

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        scaler = json.load(f)

    if (
        scaler["status"]
        != "TRAIN_NORMALIZATION_LOCKED"
    ):
        raise RuntimeError(
            "Normalization is not locked."
        )

    # ========================================================
    # Step 1 — Train data
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — BUILD FROZEN REAL TRAIN DATA")
    print("=" * 80)

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    train_records, train_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )

    if len(train_records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(train_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Unexpected subject count."
        )

    train_base_dataset = CachedRealTrainWindowDataset(
        records=train_records,
        scaler=scaler,
        max_open_files=8,
    )

    train_dataset = SubjectIndexedDataset(
        base_dataset=train_base_dataset,
        records=train_records,
    )

    print(f"Train windows: {len(train_dataset)}")
    print(f"Train subjects: {len(train_mapping)}")

    # ========================================================
    # Step 2 — Load validation references/baselines
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — LOAD LOCKED VAL-DEV / VAL-CONF REFERENCES")
    print("=" * 80)

    (
        dev_real_z,
        dev_b1_z,
        dev_subject_indices,
        dev_subject_ids,
    ) = load_reference_pair(
        real_path=VAL_DEV_REAL_NPZ,
        b1_synth_path=VAL_DEV_B1_SYNTHETIC_NPZ,
    )

    (
        conf_real_z,
        conf_b1_z,
        conf_subject_indices,
        conf_subject_ids,
    ) = load_reference_pair(
        real_path=VAL_CONF_REAL_NPZ,
        b1_synth_path=VAL_CONF_B1_SYNTHETIC_NPZ,
    )

    dev_b1_bundle = evaluate_bundle(
        real_z=dev_real_z,
        synth_z=dev_b1_z,
        scaler=scaler,
    )

    conf_b1_bundle = evaluate_bundle(
        real_z=conf_real_z,
        synth_z=conf_b1_z,
        scaler=scaler,
    )

    print(
        f"VAL-DEV B1 PSD: "
        f"{dev_b1_bundle['full_log_psd']['mean_mae_db']:.6f} dB"
    )

    print(
        f"VAL-CONF B1 PSD: "
        f"{conf_b1_bundle['full_log_psd']['mean_mae_db']:.6f} dB"
    )

    # ========================================================
    # Step 3 — Fresh training runs
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — TRAIN TWO PREDECLARED FRESH RUNS")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for s2_67."
        )

    device = torch.device("cuda")

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    all_checkpoints = {}
    all_history_rows = []

    for run_name in (
        "PRIMARY",
        "REPLICATION",
    ):
        checkpoints, history_rows = train_fresh_run(
            run_name=run_name,
            seeds=run_seeds[run_name],
            frozen_b1=frozen_b1,
            train_dataset=train_dataset,
            device=device,
        )

        all_checkpoints[
            run_name
        ] = checkpoints

        all_history_rows.extend(
            history_rows
        )

    save_csv_rows(
        all_history_rows,
        TRAINING_HISTORY_CSV,
    )

    # ========================================================
    # Step 4 — VAL-DEV selection
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — VAL-DEV CHECKPOINT SELECTION")
    print("=" * 80)

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    dev_metric_rows = []

    run_selection = {}

    for run_name in (
        "PRIMARY",
        "REPLICATION",
    ):
        run_rows = []

        print("\n" + "#" * 80)
        print(f"VAL-DEV — {run_name}")
        print("#" * 80)

        for epoch in EPOCHS:
            synth_z = generate_candidate(
                run_name=run_name,
                epoch=epoch,
                checkpoint=(
                    all_checkpoints[
                        run_name
                    ][
                        epoch
                    ]
                ),
                output_path=(
                    dev_synthetic_path(
                        run_name,
                        epoch,
                    )
                ),
                frozen_b1=frozen_b1,
                selected_subjects=dev_subject_indices,
                selected_subject_ids=dev_subject_ids,
                sampling_seed=dev_sampling_seed,
                reverse_schedule=reverse_schedule,
                device=device,
                cohort_name="VAL_DEV",
            )

            bundle = evaluate_bundle(
                real_z=dev_real_z,
                synth_z=synth_z,
                scaler=scaler,
            )

            row, eligibility = flatten_candidate_metrics(
                run_name=run_name,
                epoch=epoch,
                stage="VAL_DEV",
                bundle=bundle,
                b1_bundle=dev_b1_bundle,
            )

            dev_metric_rows.append(
                row
            )

            run_rows.append(
                row
            )

            print(
                f"{run_name} E{epoch}: "
                f"eligible={row['eligible_vs_B1']} "
                f"| PSD={row['full_psd_mae_db']:.6f} "
                f"| band={row['mean_bandpower_mae_db']:.6f} "
                f"| alpha={row['alpha_effect_mae_db']:.6f} "
                f"| failed={row['failed_criteria'] or 'NONE'}"
            )

        selected_epoch = select_epoch_from_val_dev(
            run_rows
        )

        run_selection[
            run_name
        ] = {
            "selected_epoch": selected_epoch,
            "VAL_DEV_has_eligible_checkpoint": (
                selected_epoch
                is not None
            ),
        }

        print(
            f"\n{run_name} VAL-DEV selected epoch: "
            f"{selected_epoch}"
        )

    save_csv_rows(
        dev_metric_rows,
        VAL_DEV_METRICS_CSV,
    )

    # ========================================================
    # Step 5 — single VAL-CONF confirmation
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — SINGLE VAL-CONF CONFIRMATION")
    print("=" * 80)

    conf_metric_rows = []

    for run_name in (
        "PRIMARY",
        "REPLICATION",
    ):
        selected_epoch = run_selection[
            run_name
        ][
            "selected_epoch"
        ]

        if selected_epoch is None:
            print(
                f"{run_name}: no VAL-DEV eligible checkpoint -> "
                f"VAL-CONF NOT ACCESSED for this run."
            )

            run_selection[
                run_name
            ][
                "VAL_CONF_evaluated"
            ] = False

            run_selection[
                run_name
            ][
                "VAL_CONF_pass"
            ] = False

            continue

        checkpoint = all_checkpoints[
            run_name
        ][
            selected_epoch
        ]

        synth_z = generate_candidate(
            run_name=run_name,
            epoch=selected_epoch,
            checkpoint=checkpoint,
            output_path=(
                conf_synthetic_path(
                    run_name,
                    selected_epoch,
                )
            ),
            frozen_b1=frozen_b1,
            selected_subjects=conf_subject_indices,
            selected_subject_ids=conf_subject_ids,
            sampling_seed=conf_sampling_seed,
            reverse_schedule=reverse_schedule,
            device=device,
            cohort_name="VAL_CONF",
        )

        bundle = evaluate_bundle(
            real_z=conf_real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

        row, eligibility = flatten_candidate_metrics(
            run_name=run_name,
            epoch=selected_epoch,
            stage="VAL_CONF",
            bundle=bundle,
            b1_bundle=conf_b1_bundle,
        )

        conf_metric_rows.append(
            row
        )

        run_selection[
            run_name
        ][
            "VAL_CONF_evaluated"
        ] = True

        run_selection[
            run_name
        ][
            "VAL_CONF_pass"
        ] = bool(
            row[
                "eligible_vs_B1"
            ]
        )

        run_selection[
            run_name
        ][
            "VAL_CONF_failed_criteria"
        ] = (
            row[
                "failed_criteria"
            ]
        )

        run_selection[
            run_name
        ][
            "selected_checkpoint_path"
        ] = str(
            checkpoint_path(
                run_name,
                selected_epoch,
            )
        )

        run_selection[
            run_name
        ][
            "selected_model_state_sha256"
        ] = checkpoint[
            "model_state_sha256"
        ]

        print(
            f"{run_name} selected E{selected_epoch} on VAL-CONF: "
            f"PASS={row['eligible_vs_B1']} "
            f"| PSD={row['full_psd_mae_db']:.6f} "
            f"| band={row['mean_bandpower_mae_db']:.6f} "
            f"| alpha={row['alpha_effect_mae_db']:.6f} "
            f"| failed={row['failed_criteria'] or 'NONE'}"
        )

    if conf_metric_rows:
        save_csv_rows(
            conf_metric_rows,
            VAL_CONF_METRICS_CSV,
        )

    # ========================================================
    # Step 6 — replication decision
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — APPLY FROZEN REPLICATION SUCCESS RULE")
    print("=" * 80)

    selector_replicated = bool(
        all(
            run_selection[
                run_name
            ].get(
                "VAL_CONF_pass",
                False,
            )
            for run_name in (
                "PRIMARY",
                "REPLICATION",
            )
        )
    )

    print(
        f"PRIMARY pass: "
        f"{run_selection['PRIMARY'].get('VAL_CONF_pass', False)}"
    )

    print(
        f"REPLICATION pass: "
        f"{run_selection['REPLICATION'].get('VAL_CONF_pass', False)}"
    )

    print(
        f"SELECTOR MECHANISM REPLICATED: "
        f"{selector_replicated}"
    )

    formal_freeze_candidate = None

    if selector_replicated:
        primary_epoch = int(
            run_selection[
                "PRIMARY"
            ][
                "selected_epoch"
            ]
        )

        formal_freeze_candidate = {
            "run": "PRIMARY",
            "epoch": primary_epoch,
            "checkpoint_path": str(
                checkpoint_path(
                    "PRIMARY",
                    primary_epoch,
                )
            ),
            "model_state_sha256": (
                run_selection[
                    "PRIMARY"
                ][
                    "selected_model_state_sha256"
                ]
            ),
            "reason": (
                "Predeclared PRIMARY fresh run selected on VAL-DEV, "
                "passed VAL-CONF, and selector mechanism independently "
                "replicated on predeclared REPLICATION fresh run."
            ),
        }

    atomic_json_save(
        {
            "status": (
                "S2_67_CHECKPOINT_SELECTION_DECISIONS_COMPLETE"
            ),
            "run_selection": run_selection,
            "selector_mechanism_replicated": (
                selector_replicated
            ),
            "formal_freeze_candidate": (
                formal_freeze_candidate
            ),
            "candidate_is_frozen": False,
            "Final_Test_accessed": False,
        },
        SELECTION_JSON,
    )

    # ========================================================
    # Step 7 — immutability + summary
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — VERIFY IMMUTABILITY + SAVE SUMMARY")
    print("=" * 80)

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != b1_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during s2_67."
        )

    summary = {
        "status": (
            "S2_67_FRESH_SEED_VALIDATION_SELECTOR_COMPLETE"
        ),
        "selector_mechanism_replicated": (
            selector_replicated
        ),
        "run_selection": (
            run_selection
        ),
        "formal_freeze_candidate": (
            formal_freeze_candidate
        ),
        "candidate_is_frozen": False,
        "artifacts": {
            "protocol_snapshot": str(
                PROTOCOL_SNAPSHOT
            ),
            "training_history_csv": str(
                TRAINING_HISTORY_CSV
            ),
            "VAL_DEV_metrics_csv": str(
                VAL_DEV_METRICS_CSV
            ),
            "VAL_CONF_metrics_csv": (
                str(
                    VAL_CONF_METRICS_CSV
                )
                if conf_metric_rows
                else None
            ),
            "selection_json": str(
                SELECTION_JSON
            ),
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": True,
            "Final_Test_accessed": False,
            "thresholds_changed": False,
            "winner_picking_between_training_seeds": False,
            "formal_model_freeze_decision_made": False,
            "frozen_B1_unchanged": True,
        },
        "next": (
            "If selector_mechanism_replicated is True: perform a separate "
            "formal B2 freeze stage for the predeclared PRIMARY selected "
            "checkpoint. If False: do not freeze B2."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
    )

    train_base_dataset.close()

    print("Frozen B1 unchanged: PASS")
    print("Final Test accessed: NO")
    print("Formal model freeze decision made: NO")
    print(f"summary JSON: {SUMMARY_JSON}")

    print("\n" + "=" * 80)
    print("s2_67 FRESH-SEED VALIDATION SELECTOR COMPLETE")
    print("=" * 80)

    print(
        f"selector mechanism replicated: "
        f"{selector_replicated}"
    )

    if selector_replicated:
        print(
            f"PRIMARY selected epoch: "
            f"{formal_freeze_candidate['epoch']}"
        )

        print(
            "\nNEXT:"
        )

        print(
            "FORMALLY FREEZE THE PREDECLARED PRIMARY SELECTED "
            "CHECKPOINT AS B2 IN A SEPARATE FREEZE STAGE."
        )
    else:
        print(
            "\nNEXT:"
        )

        print(
            "DO NOT FREEZE B2. INTERPRET WHICH FRESH RUN / "
            "WHICH CONFIRMATION RULE FAILED."
        )


if __name__ == "__main__":
    main()
