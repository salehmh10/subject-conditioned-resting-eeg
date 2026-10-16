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
from s2_24_full_train_loader_sanity import (
    CachedRealTrainWindowDataset,
)
from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)
from s2_39_b0_epsilon_to_v_conversion_sanity import (
    make_v_target,
    q_sample,
)
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    build_reverse_schedule,
)
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

# Frozen B1 source.
FROZEN_B1_ROOT = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b1"
    / "b1_subject_state_raw_epoch010_v1"
)

FROZEN_B1_GENERATOR = (
    FROZEN_B1_ROOT
    / "checkpoints"
    / "b1_subject_state_raw_epoch010_generator_v1.pt"
)

B1_SELECTION_MANIFEST = (
    FROZEN_B1_ROOT
    / "metrics"
    / "b1_frozen_selection_v1.json"
)

# Exact Train-only B1 audit reference and historical B1 samples.
B1_SWEEP_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "raw_candidate_subject_aware_sweep_v2"
)

B1_REAL_REFERENCE_NPZ = (
    B1_SWEEP_ROOT
    / "generated"
    / "b1_subject_aware_real_train_reference_v2.npz"
)

B1_SYNTHETIC_REFERENCE_NPZ = (
    B1_SWEEP_ROOT
    / "generated"
    / "b1_raw_candidate_subject_aware_synthetic_v2.npz"
)

# Canonical Control E5 already discovered before this replication.
CONTROL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "matched_lambda0_control_v1"
)

CONTROL_E5_CHECKPOINT = (
    CONTROL_ROOT
    / "checkpoints"
    / "candidate_epoch_005_raw.pt"
)

# Historical Control E5 synthetic cache, sampling seed 20261001.
S2_58_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "joint_b1_lambda0_lambda001_freegen_audit_v1"
)

CONTROL_E5_SYNTHETIC_NPZ = (
    S2_58_ROOT
    / "generated"
    / "lambda0_control_epoch_005_synthetic_v1.npz"
)

# Required completed stages.
S2_61_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "control_e5_sampling_robustness_v1"
    / "metrics"
    / "s2_61_control_e5_sampling_robustness_summary_v1.json"
)

S2_62_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "control_e5_preserving_distributional_adapter_pilot_v1"
    / "metrics"
    / "s2_62_distributional_adapter_summary_v1.json"
)


# ============================================================
# s2_63 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "control_e5_training_seed_replication_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
GENERATED_DIR = RUN_ROOT / "generated"
LOG_DIR = RUN_ROOT / "logs"

PROTOCOL_FILE = (
    METRIC_DIR
    / "s2_63_control_e5_training_seed_replication_protocol_v1.json"
)

TRAIN_HISTORY_CSV = (
    LOG_DIR
    / "s2_63_training_seed_replication_history_v1.csv"
)

FREEGEN_CSV = (
    METRIC_DIR
    / "s2_63_training_seed_replication_freegen_metrics_v1.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_63_control_e5_training_seed_replication_summary_v1.json"
)


# ============================================================
# Frozen replication design
# ============================================================

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

TIMESTEPS = 1000
COSINE_S = 0.008

MAX_EPOCHS = 5
BATCH_SIZE = 64

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

MIXED_PRECISION = True
EMA_ENABLED = False
LAMBDA_SPECTRAL = 0.0

MAX_AMP_SKIPS_PER_EPOCH = 5
MAX_CONSECUTIVE_AMP_SKIPS = 3

# Two genuinely new training streams.
REPLICA_CONFIG = {
    "REPLICA_A": {
        "master_seed": 20261201,
        "loader_seed": 20261202,
        "noise_seed": 20261203,
    },
    "REPLICA_B": {
        "master_seed": 20261301,
        "loader_seed": 20261302,
        "noise_seed": 20261303,
    },
}

# Fixed sampling seed across ALL training-seed variants.
# This isolates training-seed variability from sampling-seed variability.
SAMPLING_SEED_BASE = 20261001

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

N_CHANNELS = 19
N_TIMES = 512

EXPECTED_REAL_SHAPE = (
    SELECTED_SUBJECT_COUNT,
    2,
    REAL_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)

EXPECTED_SYNTH_SHAPE = (
    SELECTED_SUBJECT_COUNT,
    2,
    SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)


# ============================================================
# IO / reproducibility
# ============================================================

def set_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda: f.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def atomic_json_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(
        str(path)
        + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    os.replace(
        temp,
        path,
    )


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(
        str(path)
        + ".tmp"
    )

    torch.save(
        payload,
        temp,
    )

    os.replace(
        temp,
        path,
    )


def atomic_npz_save(
    path: Path,
    **arrays,
) -> None:
    temp = Path(
        str(path)
        + ".tmp"
    )

    with temp.open(
        "wb"
    ) as f:
        np.savez_compressed(
            f,
            **arrays,
        )

    os.replace(
        temp,
        path,
    )


def save_csv_rows(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            f"Cannot save empty CSV: {path}"
        )

    temp = Path(
        str(path)
        + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    os.replace(
        temp,
        path,
    )


def replica_checkpoint_path(
    replica_name: str,
) -> Path:
    return (
        CHECKPOINT_DIR
        / f"s2_63_{replica_name.lower()}_epoch005_raw.pt"
    )


def replica_synthetic_path(
    replica_name: str,
) -> Path:
    return (
        GENERATED_DIR
        / f"s2_63_{replica_name.lower()}_seed_{SAMPLING_SEED_BASE}_synthetic_v1.npz"
    )


# ============================================================
# Exact audit references
# ============================================================

def load_exact_audit_reference() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[str],
]:
    real_package = np.load(
        B1_REAL_REFERENCE_NPZ,
        allow_pickle=False,
    )

    b1_package = np.load(
        B1_SYNTHETIC_REFERENCE_NPZ,
        allow_pickle=False,
    )

    canonical_package = np.load(
        CONTROL_E5_SYNTHETIC_NPZ,
        allow_pickle=False,
    )

    real_z = real_package[
        "real_z"
    ].astype(
        np.float32,
        copy=False,
    )

    subject_indices = (
        real_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    subject_ids = (
        real_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    b1_indices = (
        b1_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    b1_ids = (
        b1_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    canonical_indices = (
        canonical_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    canonical_ids = (
        canonical_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if not (
        subject_indices
        == b1_indices
        == canonical_indices
    ):
        raise RuntimeError(
            "Training-seed replication audit cohort mismatch."
        )

    if not (
        subject_ids
        == b1_ids
        == canonical_ids
    ):
        raise RuntimeError(
            "Training-seed replication subject-ID mismatch."
        )

    b1_synth_z = b1_package[
        "epoch_010_z"
    ].astype(
        np.float32,
        copy=False,
    )

    canonical_synth_z = canonical_package[
        "synth_z"
    ].astype(
        np.float32,
        copy=False,
    )

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(
            f"Unexpected Real shape: "
            f"{real_z.shape}"
        )

    if b1_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected B1 synthetic shape: "
            f"{b1_synth_z.shape}"
        )

    if canonical_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected canonical Control-E5 synthetic shape: "
            f"{canonical_synth_z.shape}"
        )

    return (
        real_z,
        b1_synth_z,
        canonical_synth_z,
        subject_indices,
        subject_ids,
    )


# ============================================================
# Training
# ============================================================

def run_replica_training(
    *,
    replica_name: str,
    replica_config: dict[str, int],
    frozen_b1: dict[str, Any],
    train_dataset: SubjectIndexedDataset,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TRAIN {replica_name}"
    )

    print(
        "=" * 80
    )

    print(
        f"master seed: "
        f"{replica_config['master_seed']}"
    )

    print(
        f"loader seed: "
        f"{replica_config['loader_seed']}"
    )

    print(
        f"noise seed: "
        f"{replica_config['noise_seed']}"
    )

    set_seed(
        int(
            replica_config[
                "master_seed"
            ]
        )
    )

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        int(
            replica_config[
                "loader_seed"
            ]
        )
    )

    noise_generator = torch.Generator(
        device="cuda"
    )

    noise_generator.manual_seed(
        int(
            replica_config[
                "noise_seed"
            ]
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    if len(
        train_loader
    ) != 748:
        raise RuntimeError(
            "Expected 748 Train batches."
        )

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

    alpha_bar = (
        schedule_cpu[
            "alpha_cumprod"
        ]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    history = []

    total_amp_skips = 0
    cumulative_training_seconds = 0.0

    for epoch in range(
        1,
        MAX_EPOCHS
        + 1,
    ):
        epoch_start = time.perf_counter()

        model.train()

        train_v_sum = 0.0
        presented = 0

        successful_optimizer_steps = 0
        amp_skips = 0
        consecutive_amp_skips = 0

        gradient_norms = []

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            x0 = batch[
                "x"
            ].to(
                device,
                non_blocking=True,
            )

            state = batch[
                "y"
            ].to(
                device,
                non_blocking=True,
            )

            subject = batch[
                "subject_index"
            ].to(
                device,
                non_blocking=True,
            )

            current_batch_size = int(
                x0.shape[
                    0
                ]
            )

            timesteps = torch.randint(
                0,
                TIMESTEPS,
                (
                    current_batch_size,
                ),
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

            if not torch.isfinite(
                v_loss
            ):
                raise RuntimeError(
                    f"Nonfinite v-loss in {replica_name}."
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

            (
                finite,
                grad_count,
                grad_norm,
                bad_names,
            ) = finite_gradient_summary(
                model
            )

            del grad_count

            scale_before = float(
                grad_scaler.get_scale()
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
                        f"Nonfinite clip norm in {replica_name}."
                    )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()

                successful_optimizer_steps += 1

                consecutive_amp_skips = 0

                gradient_norms.append(
                    float(
                        grad_norm
                    )
                )
            else:
                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()

                scale_after = float(
                    grad_scaler.get_scale()
                )

                amp_skips += 1
                total_amp_skips += 1
                consecutive_amp_skips += 1

                print(
                    f"AMP SKIP "
                    f"| replica={replica_name} "
                    f"| epoch={epoch} "
                    f"| batch={batch_index} "
                    f"| scale={scale_before:g}->{scale_after:g} "
                    f"| bad={bad_names[:3]}"
                )

                if amp_skips > MAX_AMP_SKIPS_PER_EPOCH:
                    raise RuntimeError(
                        f"Too many AMP skips in "
                        f"{replica_name}, epoch {epoch}."
                    )

                if (
                    consecutive_amp_skips
                    >= MAX_CONSECUTIVE_AMP_SKIPS
                ):
                    raise RuntimeError(
                        f"Too many consecutive AMP skips "
                        f"in {replica_name}."
                    )

            presented += current_batch_size

            train_v_sum += (
                v_value
                * current_batch_size
            )

            if (
                batch_index == 1
                or batch_index
                % 100
                == 0
                or batch_index
                == len(
                    train_loader
                )
            ):
                print(
                    f"{replica_name} "
                    f"| epoch={epoch}/5 "
                    f"| batch={batch_index:>3}/748 "
                    f"| v={train_v_sum/presented:.6f} "
                    f"| amp_skips={amp_skips} "
                    f"| scale={grad_scaler.get_scale():g}"
                )

        if presented != EXPECTED_TRAIN_WINDOWS:
            raise RuntimeError(
                f"Train accounting mismatch in {replica_name}."
            )

        if (
            successful_optimizer_steps
            + amp_skips
            != len(
                train_loader
            )
        ):
            raise RuntimeError(
                f"Optimizer-step accounting mismatch "
                f"in {replica_name}."
            )

        if not gradient_norms:
            raise RuntimeError(
                f"No successful optimizer steps in "
                f"{replica_name}, epoch {epoch}."
            )

        torch.cuda.synchronize()

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        cumulative_training_seconds += (
            epoch_seconds
        )

        row = {
            "replica": (
                replica_name
            ),
            "epoch": (
                epoch
            ),
            "train_v_mse": float(
                train_v_sum
                / presented
            ),
            "successful_optimizer_steps": (
                successful_optimizer_steps
            ),
            "amp_skipped_steps": (
                amp_skips
            ),
            "amp_scale_end": float(
                grad_scaler.get_scale()
            ),
            "mean_gradient_norm": float(
                np.mean(
                    gradient_norms
                )
            ),
            "max_gradient_norm": float(
                np.max(
                    gradient_norms
                )
            ),
            "epoch_seconds": (
                epoch_seconds
            ),
        }

        history.append(
            row
        )

        print(
            "\n"
            f"{replica_name} EPOCH {epoch} COMPLETE "
            f"| Train-v={row['train_v_mse']:.6f} "
            f"| successful={successful_optimizer_steps} "
            f"| skips={amp_skips} "
            f"| time={epoch_seconds/60:.2f} min"
        )

    model_state = model.state_dict()

    checkpoint = {
        "status": (
            "S2_63_CONTROL_E5_TRAINING_SEED_REPLICA_CHECKPOINT"
        ),
        "replica": (
            replica_name
        ),
        "initialization": (
            "exact frozen B1 RAW epoch 10"
        ),
        "epochs": (
            MAX_EPOCHS
        ),
        "lambda_spectral": (
            LAMBDA_SPECTRAL
        ),
        "prediction_target": (
            "v"
        ),
        "EMA_enabled": (
            EMA_ENABLED
        ),
        "learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "gradient_clip_norm": (
            GRADIENT_CLIP_NORM
        ),
        "mixed_precision": (
            MIXED_PRECISION
        ),
        "seeds": (
            replica_config
        ),
        "model_state_dict": (
            model_state
        ),
        "model_state_sha256": (
            state_dict_sha256(
                model_state
            )
        ),
        "history": (
            history
        ),
        "total_amp_skips": (
            total_amp_skips
        ),
        "training_seconds": (
            cumulative_training_seconds
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "formal_model_selected": False,
    }

    print(
        f"{replica_name} TRAINING COMPLETE "
        f"| total time="
        f"{cumulative_training_seconds/60:.2f} min "
        f"| total skips={total_amp_skips}"
    )

    del model

    torch.cuda.empty_cache()

    return (
        checkpoint,
        history,
    )


# ============================================================
# Free generation
# ============================================================

def generate_replica(
    *,
    replica_name: str,
    checkpoint: dict[str, Any],
    frozen_b1: dict[str, Any],
    selected_subjects: list[int],
    selected_subject_ids: list[str],
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    checkpoint_path = replica_checkpoint_path(
        replica_name
    )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    output_path = replica_synthetic_path(
        replica_name
    )

    if output_path.is_file():
        package = np.load(
            output_path,
            allow_pickle=False,
        )

        if (
            package[
                "selected_subject_indices"
            ]
            .astype(
                np.int64
            )
            .tolist()
            != selected_subjects
        ):
            raise RuntimeError(
                f"Cached cohort mismatch for {replica_name}."
            )

        if (
            package[
                "selected_subject_ids"
            ]
            .astype(
                str
            )
            .tolist()
            != selected_subject_ids
        ):
            raise RuntimeError(
                f"Cached subject-ID mismatch for {replica_name}."
            )

        if str(
            package[
                "checkpoint_sha256"
            ][0]
        ) != checkpoint_hash:
            raise RuntimeError(
                f"Cached checkpoint hash mismatch for {replica_name}."
            )

        if int(
            package[
                "sampling_seed"
            ][0]
        ) != SAMPLING_SEED_BASE:
            raise RuntimeError(
                f"Cached sampling seed mismatch for {replica_name}."
            )

        synth_z = package[
            "synth_z"
        ].astype(
            np.float32,
            copy=False,
        )

        if synth_z.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"Cached synthetic shape mismatch for {replica_name}."
            )

        print(
            f"{replica_name}: validated synthetic cache -> REUSE"
        )

        return synth_z

    model = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=(
            checkpoint[
                "model_state_dict"
            ]
        ),
        device=device,
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"FREE GENERATION — {replica_name}"
    )

    print(
        "-" * 80
    )

    start = time.perf_counter()

    eo = sample_b1_candidate_state(
        model=model,
        selected_subjects=(
            selected_subjects
        ),
        eye_state_value=0,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        base_seed=(
            SAMPLING_SEED_BASE
        ),
    )

    ec = sample_b1_candidate_state(
        model=model,
        selected_subjects=(
            selected_subjects
        ),
        eye_state_value=1,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        base_seed=(
            SAMPLING_SEED_BASE
        ),
    )

    synth_z = np.stack(
        [
            eo,
            ec,
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    if synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected synthetic shape for "
            f"{replica_name}: {synth_z.shape}"
        )

    generation_seconds = (
        time.perf_counter()
        - start
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
        checkpoint_sha256=np.asarray(
            [
                checkpoint_hash
            ],
            dtype="U64",
        ),
        sampling_seed=np.asarray(
            [
                SAMPLING_SEED_BASE
            ],
            dtype=np.int64,
        ),
        generation_seconds=np.asarray(
            [
                generation_seconds
            ],
            dtype=np.float64,
        ),
    )

    print(
        f"{replica_name} generation: "
        f"{generation_seconds/60:.2f} min"
    )

    del model

    torch.cuda.empty_cache()

    return synth_z


# ============================================================
# Evaluation
# ============================================================

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
        "generic": (
            generic
        ),
        "full_log_psd": (
            full_psd
        ),
    }


def flatten_metrics(
    *,
    variant_name: str,
    bundle: dict[str, Any],
    b1_bundle: dict[str, Any],
) -> dict[str, Any]:
    generic = bundle[
        "generic"
    ]

    subject = generic[
        "subject_identity"
    ]

    if variant_name == "B1":
        eligibility = None
    else:
        eligibility = evaluate_eligibility(
            candidate=bundle,
            b1_baseline=b1_bundle,
        )

    b1_psd = float(
        b1_bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    current_psd = float(
        bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    return {
        "variant": (
            variant_name
        ),
        "sampling_seed": (
            SAMPLING_SEED_BASE
        ),
        "full_psd_mae_db": (
            current_psd
        ),
        "full_psd_improvement_vs_B1_fraction": float(
            (
                b1_psd
                - current_psd
            )
            / b1_psd
        ),
        "eo_scale": float(
            generic[
                "scale"
            ][
                "EO"
            ][
                "normalized_std_ratio"
            ]
        ),
        "ec_scale": float(
            generic[
                "scale"
            ][
                "EC"
            ][
                "normalized_std_ratio"
            ]
        ),
        "mean_bandpower_mae_db": float(
            generic[
                "mean_bandpower_mae_db"
            ]
        ),
        "alpha_effect_mae_db": float(
            generic[
                "alpha_effect"
            ][
                "channel_mae_db"
            ]
        ),
        "subject_top1": float(
            subject[
                "subject_retrieval_top1_accuracy"
            ]
        ),
        "subject_top5": float(
            subject[
                "subject_retrieval_top5_accuracy"
            ]
        ),
        "subject_fingerprint_r_median": float(
            subject[
                "subject_fingerprint_pearson_r_median"
            ]
        ),
        "subject_margin": float(
            subject[
                "subject_correct_vs_median_wrong_margin_mean"
            ]
        ),
        "passes_all_existing_B2_guards_vs_B1": (
            ""
            if eligibility is None
            else bool(
                eligibility[
                    "eligible"
                ]
            )
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — s2_63 CONTROL-E5 TRAINING-SEED REPLICATION"
    )

    print(
        "=" * 80
    )

    print(
        "\nScientific question:"
    )

    print(
        "  Does the simple 5-epoch lambda=0 continuation recipe that"
    )

    print(
        "  produced Control E5 replicate under independent TRAINING seeds?"
    )

    print(
        "\nImportant design:"
    )

    print(
        "  Sampling seed is FIXED across all training variants."
    )

    print(
        "  Therefore this stage isolates training-seed variability."
    )

    print(
        "  Replicas are NOT competing to beat the canonical Control E5."
    )

    print(
        "  They only need to independently reproduce B1-relative eligibility."
    )

    print(
        "\nNo Validation EEG is accessed."
    )

    print(
        "No Final Test EEG is accessed."
    )

    print(
        "No final model is frozen by this stage."
    )

    for directory in (
        METRIC_DIR,
        CHECKPOINT_DIR,
        GENERATED_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — prerequisite audit
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B1 / CANONICAL CONTROL-E5 / s2_61 / s2_62"
    )

    print(
        "=" * 80
    )

    required_files = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_REAL_REFERENCE_NPZ,
        B1_SYNTHETIC_REFERENCE_NPZ,
        CONTROL_E5_CHECKPOINT,
        CONTROL_E5_SYNTHETIC_NPZ,
        S2_61_SUMMARY,
        S2_62_SUMMARY,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B1_SELECTION_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        b1_selection = json.load(
            f
        )

    with S2_61_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_61_summary = json.load(
            f
        )

    with S2_62_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_62_summary = json.load(
            f
        )

    if b1_selection[
        "status"
    ] != "B1_SELECTION_FROZEN":
        raise RuntimeError(
            "B1 is not frozen."
        )

    if s2_61_summary[
        "status"
    ] != "S2_61_CONTROL_E5_SAMPLING_ROBUSTNESS_COMPLETE":
        raise RuntimeError(
            "s2_61 is not complete."
        )

    if s2_61_summary[
        "development_anchor_robustly_supported"
    ] is not True:
        raise RuntimeError(
            "Control E5 did not pass sampling-seed robustness."
        )

    if s2_62_summary[
        "status"
    ] != "S2_62_CONTROL_E5_PRESERVING_DISTRIBUTIONAL_ADAPTER_PILOT_COMPLETE":
        raise RuntimeError(
            "s2_62 is not complete."
        )

    if s2_62_summary[
        "adapter_dist_advances"
    ] is not False:
        raise RuntimeError(
            "s2_62 unexpectedly advanced the adapter family."
        )

    frozen_b1_hash_before = sha256_file(
        FROZEN_B1_GENERATOR
    )

    canonical_control_hash_before = sha256_file(
        CONTROL_E5_CHECKPOINT
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    canonical_control = torch.load(
        CONTROL_E5_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    frozen_b1_state_hash = state_dict_sha256(
        frozen_b1[
            "model_state_dict"
        ]
    )

    canonical_control_state_hash = state_dict_sha256(
        canonical_control[
            "model_state_dict"
        ]
    )

    if (
        frozen_b1_state_hash
        != frozen_b1[
            "selected_model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B1 model-state hash mismatch."
        )

    if (
        canonical_control_state_hash
        != canonical_control[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Canonical Control-E5 model-state hash mismatch."
        )

    print(
        f"Frozen B1 SHA256: "
        f"{frozen_b1_hash_before}"
    )

    print(
        f"Canonical Control E5 SHA256: "
        f"{canonical_control_hash_before}"
    )

    print(
        "prerequisite audit: PASS"
    )

    # ========================================================
    # Step 2 — freeze replication protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE s2_63 TRAINING-SEED REPLICATION PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_63_CONTROL_E5_TRAINING_SEED_REPLICATION_PROTOCOL_FROZEN"
        ),
        "scientific_role": (
            "Independent training-seed replication of the already-discovered "
            "5-epoch lambda=0 continuation recipe. Not a new hyperparameter search."
        ),
        "canonical_checkpoint": {
            "path": str(
                CONTROL_E5_CHECKPOINT
            ),
            "file_sha256": (
                canonical_control_hash_before
            ),
            "model_state_sha256": (
                canonical_control_state_hash
            ),
            "selection_timing": (
                "Canonical Control E5 was identified before these replication runs; "
                "replicas will not compete to replace it."
            ),
        },
        "training_recipe": {
            "initialization": (
                "exact frozen B1 RAW epoch 10"
            ),
            "epochs": (
                MAX_EPOCHS
            ),
            "lambda_spectral": (
                LAMBDA_SPECTRAL
            ),
            "prediction_target": (
                "v"
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),
            "mixed_precision": (
                MIXED_PRECISION
            ),
            "EMA_enabled": (
                EMA_ENABLED
            ),
        },
        "independent_training_replicas": (
            REPLICA_CONFIG
        ),
        "free_generation": {
            "sampling_seed_fixed": (
                SAMPLING_SEED_BASE
            ),
            "reason": (
                "Hold sampling randomness fixed so differences primarily reflect "
                "training-seed variability."
            ),
            "sampler": (
                "1000-step ancestral DDPM / v-prediction / FP32"
            ),
            "subjects": (
                SELECTED_SUBJECT_COUNT
            ),
            "real_windows_per_subject_state": (
                REAL_WINDOWS_PER_SUBJECT_STATE
            ),
            "synthetic_windows_per_subject_state": (
                SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
            ),
        },
        "replication_success_rule": (
            "Training recipe is considered independently replicated only if "
            "BOTH new replicas pass ALL existing frozen B2 eligibility/guard "
            "criteria versus the same frozen B1 baseline at the fixed sampling seed. "
            "Replicas are NOT required to beat the canonical Control E5."
        ),
        "if_successful": (
            "Proceed to a separate freeze stage for the pre-existing canonical "
            "Control E5 checkpoint, transparently labeled as a post-B2 diagnostic "
            "selection with independent training-seed replication."
        ),
        "if_unsuccessful": (
            "Do not freeze Control E5; investigate training-seed instability."
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "thresholds_changed": False,
        "formal_model_freeze_decision_made": False,
    }

    if PROTOCOL_FILE.is_file():
        with PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_protocol = json.load(
                f
            )

        if existing_protocol != protocol:
            raise RuntimeError(
                "Conflicting s2_63 protocol already exists."
            )

        print(
            "identical s2_63 protocol already exists."
        )
    else:
        atomic_json_save(
            protocol,
            PROTOCOL_FILE,
        )

        print(
            f"protocol: "
            f"{PROTOCOL_FILE}"
        )

    # ========================================================
    # Step 3 — Train data + exact audit reference
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD REAL TRAIN DATA + EXACT TRAIN-ONLY AUDIT REFERENCE"
    )

    print(
        "=" * 80
    )

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        scaler = json.load(
            f
        )

    if scaler[
        "status"
    ] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError(
            "Normalization is not locked."
        )

    train_records, train_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )

    if len(
        train_records
    ) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(
        train_mapping
    ) != EXPECTED_SUBJECTS:
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

    (
        real_z,
        b1_synth_z,
        canonical_synth_z,
        selected_subject_indices,
        selected_subject_ids,
    ) = load_exact_audit_reference()

    print(
        f"Train windows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Audit subjects: "
        f"{len(selected_subject_indices)}"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test EEG accessed: NO"
    )

    # ========================================================
    # Step 4 — Run two independent 5-epoch training replicas
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — RUN TWO INDEPENDENT 5-EPOCH TRAINING REPLICAS"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for s2_63."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    replica_checkpoints = {}
    all_history_rows = []

    for replica_name, replica_config in REPLICA_CONFIG.items():
        (
            checkpoint,
            history,
        ) = run_replica_training(
            replica_name=replica_name,
            replica_config=replica_config,
            frozen_b1=frozen_b1,
            train_dataset=train_dataset,
            device=device,
        )

        checkpoint_path = replica_checkpoint_path(
            replica_name
        )

        atomic_torch_save(
            checkpoint,
            checkpoint_path,
        )

        replica_checkpoints[
            replica_name
        ] = checkpoint

        all_history_rows.extend(
            history
        )

        print(
            f"checkpoint: "
            f"{checkpoint_path}"
        )

    save_csv_rows(
        all_history_rows,
        TRAIN_HISTORY_CSV,
    )

    # ========================================================
    # Step 5 — Fixed-seed free generation
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — FIXED-SEED FREE GENERATION FOR BOTH TRAINING REPLICAS"
    )

    print(
        "=" * 80
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    replica_synth = {}

    for replica_name in REPLICA_CONFIG:
        replica_synth[
            replica_name
        ] = generate_replica(
            replica_name=replica_name,
            checkpoint=(
                replica_checkpoints[
                    replica_name
                ]
            ),
            frozen_b1=frozen_b1,
            selected_subjects=(
                selected_subject_indices
            ),
            selected_subject_ids=(
                selected_subject_ids
            ),
            reverse_schedule=(
                reverse_schedule
            ),
            device=device,
        )

    # ========================================================
    # Step 6 — Exact existing free-generation audit
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — TRAINING-SEED REPLICATION FREE-GENERATION AUDIT"
    )

    print(
        "=" * 80
    )

    bundles = {
        "B1": evaluate_bundle(
            real_z=real_z,
            synth_z=b1_synth_z,
            scaler=scaler,
        ),
        "CANONICAL_CONTROL_E5": evaluate_bundle(
            real_z=real_z,
            synth_z=canonical_synth_z,
            scaler=scaler,
        ),
    }

    for replica_name, synth_z in replica_synth.items():
        bundles[
            replica_name
        ] = evaluate_bundle(
            real_z=real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

    b1_bundle = bundles[
        "B1"
    ]

    metric_rows = []

    for variant_name in (
        "B1",
        "CANONICAL_CONTROL_E5",
        "REPLICA_A",
        "REPLICA_B",
    ):
        row = flatten_metrics(
            variant_name=variant_name,
            bundle=(
                bundles[
                    variant_name
                ]
            ),
            b1_bundle=b1_bundle,
        )

        metric_rows.append(
            row
        )

        print(
            "\n"
            f"{variant_name}:"
        )

        print(
            f"  full PSD MAE: "
            f"{row['full_psd_mae_db']:.6f} dB"
        )

        print(
            f"  improvement vs B1: "
            f"{100.0 * row['full_psd_improvement_vs_B1_fraction']:+.2f}%"
        )

        print(
            f"  mean bandpower MAE: "
            f"{row['mean_bandpower_mae_db']:.6f} dB"
        )

        print(
            f"  alpha-effect MAE: "
            f"{row['alpha_effect_mae_db']:.6f} dB"
        )

        print(
            f"  subject Top1: "
            f"{row['subject_top1']:.6f}"
        )

        print(
            f"  all existing B2 guards pass: "
            f"{row['passes_all_existing_B2_guards_vs_B1']}"
        )

    save_csv_rows(
        metric_rows,
        FREEGEN_CSV,
    )

    # ========================================================
    # Step 7 — Predeclared replication decision
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — APPLY PREDECLARED TRAINING-SEED REPLICATION RULE"
    )

    print(
        "=" * 80
    )

    rows_by_variant = {
        row[
            "variant"
        ]: row
        for row in metric_rows
    }

    replica_support = {}

    for replica_name in (
        "REPLICA_A",
        "REPLICA_B",
    ):
        passed = bool(
            rows_by_variant[
                replica_name
            ][
                "passes_all_existing_B2_guards_vs_B1"
            ]
        )

        replica_support[
            replica_name
        ] = {
            "passes_all_existing_B2_guards_vs_B1": (
                passed
            ),
            "full_psd_mae_db": float(
                rows_by_variant[
                    replica_name
                ][
                    "full_psd_mae_db"
                ]
            ),
            "full_psd_improvement_vs_B1_fraction": float(
                rows_by_variant[
                    replica_name
                ][
                    "full_psd_improvement_vs_B1_fraction"
                ]
            ),
            "replication_supported": (
                passed
            ),
        }

        print(
            f"{replica_name}: "
            f"replication supported = "
            f"{passed}"
        )

    training_recipe_independently_replicated = bool(
        all(
            item[
                "replication_supported"
            ]
            for item in replica_support.values()
        )
    )

    print(
        "\nTRAINING RECIPE INDEPENDENTLY REPLICATED: "
        f"{training_recipe_independently_replicated}"
    )

    # Aggregate 3 training-seed variants:
    # canonical original + two new independent replicas.
    training_seed_psd = np.asarray(
        [
            rows_by_variant[
                "CANONICAL_CONTROL_E5"
            ][
                "full_psd_mae_db"
            ],
            rows_by_variant[
                "REPLICA_A"
            ][
                "full_psd_mae_db"
            ],
            rows_by_variant[
                "REPLICA_B"
            ][
                "full_psd_mae_db"
            ],
        ],
        dtype=np.float64,
    )

    aggregate = {
        "three_training_seed_full_psd_mae_db_mean": float(
            np.mean(
                training_seed_psd
            )
        ),
        "three_training_seed_full_psd_mae_db_std_ddof0": float(
            np.std(
                training_seed_psd,
                ddof=0,
            )
        ),
        "three_training_seed_full_psd_mae_db_min": float(
            np.min(
                training_seed_psd
            )
        ),
        "three_training_seed_full_psd_mae_db_max": float(
            np.max(
                training_seed_psd
            )
        ),
    }

    print(
        "3-training-seed Full PSD MAE "
        f"mean±std: "
        f"{aggregate['three_training_seed_full_psd_mae_db_mean']:.6f} ± "
        f"{aggregate['three_training_seed_full_psd_mae_db_std_ddof0']:.6f} dB"
    )

    # ========================================================
    # Step 8 — Immutability + final summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — VERIFY IMMUTABILITY + SAVE FINAL SUMMARY"
    )

    print(
        "=" * 80
    )

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != frozen_b1_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during s2_63."
        )

    if sha256_file(
        CONTROL_E5_CHECKPOINT
    ) != canonical_control_hash_before:
        raise RuntimeError(
            "Canonical Control E5 changed during s2_63."
        )

    summary = {
        "status": (
            "S2_63_CONTROL_E5_TRAINING_SEED_REPLICATION_COMPLETE"
        ),
        "scientific_role": (
            "Independent training-seed replication of the pre-existing "
            "Control-E5 5-epoch lambda=0 continuation recipe."
        ),
        "canonical_control_E5": {
            "checkpoint": str(
                CONTROL_E5_CHECKPOINT
            ),
            "file_sha256": (
                canonical_control_hash_before
            ),
            "model_state_sha256": (
                canonical_control_state_hash
            ),
            "selected_before_replication": True,
        },
        "replica_support": (
            replica_support
        ),
        "training_recipe_independently_replicated": (
            training_recipe_independently_replicated
        ),
        "aggregate": (
            aggregate
        ),
        "sampling_seed_fixed": (
            SAMPLING_SEED_BASE
        ),
        "artifacts": {
            "protocol": str(
                PROTOCOL_FILE
            ),
            "training_history_csv": str(
                TRAIN_HISTORY_CSV
            ),
            "freegen_metrics_csv": str(
                FREEGEN_CSV
            ),
            "replica_checkpoints": {
                replica_name: str(
                    replica_checkpoint_path(
                        replica_name
                    )
                )
                for replica_name in REPLICA_CONFIG
            },
            "replica_synthetic": {
                replica_name: str(
                    replica_synthetic_path(
                        replica_name
                    )
                )
                for replica_name in REPLICA_CONFIG
            },
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "thresholds_changed": False,
            "formal_model_freeze_decision_made": False,
            "Frozen_B1_unchanged": True,
            "Canonical_Control_E5_unchanged": True,
        },
        "next": (
            "If training_recipe_independently_replicated is True, "
            "freeze the pre-existing canonical Control E5 checkpoint in a "
            "separate transparent selection/freeze stage, then proceed to "
            "synthetic-data utility experiments. If False, do not freeze it."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
    )

    print(
        "Frozen B1 unchanged: PASS"
    )

    print(
        "Canonical Control E5 unchanged: PASS"
    )

    print(
        f"summary JSON: "
        f"{SUMMARY_JSON}"
    )

    train_base_dataset.close()

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "s2_63 CONTROL-E5 TRAINING-SEED REPLICATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "training recipe independently replicated: "
        f"{training_recipe_independently_replicated}"
    )

    print(
        "\nValidation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Formal model freeze decision made: NO"
    )

    print(
        "\nNEXT:"
    )

    if training_recipe_independently_replicated:
        print(
            "FREEZE THE PRE-EXISTING CANONICAL CONTROL E5 IN A SEPARATE "
            "TRANSPARENT SELECTION STAGE, THEN MOVE TO UTILITY."
        )
    else:
        print(
            "DO NOT FREEZE CONTROL E5 — INVESTIGATE TRAINING-SEED INSTABILITY."
        )


if __name__ == "__main__":
    main()
