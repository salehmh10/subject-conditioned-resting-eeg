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
from s2_53_b2_lambda_calibration_pilot import (
    PhysicalLogPSDLoss,
    reconstruct_x0_from_v,
)
from s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep import (
    build_model_from_state,
    evaluate_eligibility,
    full_log_psd_mae,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen B1 context/architecture source.
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

# Exact B1 V2 Train-only audit reference/baseline.
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

# Strongest current Train-only development baseline:
# matched lambda=0 continuation at epoch 5.
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

CONTROL_RESULT = (
    CONTROL_ROOT
    / "metrics"
    / "b2_matched_lambda0_control_result_v1.json"
)

# s2_58 generated cache for the exact Control-E5 baseline.
S2_58_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "joint_b1_lambda0_lambda001_freegen_audit_v1"
)

S2_58_SUMMARY = (
    S2_58_ROOT
    / "metrics"
    / "s2_58_joint_freegen_summary_v1.json"
)

CONTROL_E5_SYNTHETIC_NPZ = (
    S2_58_ROOT
    / "generated"
    / "lambda0_control_epoch_005_synthetic_v1.npz"
)

# s2_59 must be complete before this pilot is allowed.
S2_59_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "denoiser_mechanism_diagnostics_v1"
)

S2_59_SUMMARY = (
    S2_59_ROOT
    / "metrics"
    / "s2_59_denoiser_diagnostics_summary_v1.json"
)


# ============================================================
# s2_60 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "corrective_low_t_spectral_pilot_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
GENERATED_DIR = RUN_ROOT / "generated"
LOG_DIR = RUN_ROOT / "logs"

PROTOCOL_FILE = (
    METRIC_DIR
    / "s2_60_corrective_low_t_spectral_protocol_v1.json"
)

HISTORY_CSV = (
    LOG_DIR
    / "s2_60_corrective_low_t_spectral_history_v1.csv"
)

METRICS_CSV = (
    METRIC_DIR
    / "s2_60_corrective_low_t_spectral_freegen_metrics_v1.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_60_corrective_low_t_spectral_summary_v1.json"
)


# ============================================================
# Frozen pilot design
# ============================================================

MASTER_SEED = 20261160
TRAIN_LOADER_SEED = 20261161
TRAIN_NOISE_SEED = 20261162

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

TIMESTEPS = 1000
COSINE_S = 0.008

BATCH_SIZE = 64
PILOT_SUCCESSFUL_STEPS = 500

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

# Deliberately FP32 for this small matched mechanism pilot.
# This prevents AMP-skip location from breaking exact stream matching
# across the three arms. If a candidate later advances, its full follow-up
# protocol must explicitly state whether FP32 is retained or AMP is restored.
MIXED_PRECISION = False

SPECTRAL_LAMBDA = 0.01

SPECTRAL_FMIN_HZ = 1.0
SPECTRAL_FMAX_HZ = 40.0
SPECTRAL_LOG_EPS = 1e-12
SPECTRAL_HUBER_BETA_DB = 1.0

ARM_CONFIG = {
    "L0": {
        "lambda_spectral": 0.0,
        "max_spectral_timestep": None,
        "role": (
            "Matched 500-step continuation control from Control E5."
        ),
    },
    "GATE100": {
        "lambda_spectral": SPECTRAL_LAMBDA,
        "max_spectral_timestep": 100,
        "role": (
            "Current paired spectral loss, but gradient only from samples "
            "with timestep <= 100."
        ),
    },
    "GATE200": {
        "lambda_spectral": SPECTRAL_LAMBDA,
        "max_spectral_timestep": 200,
        "role": (
            "Current paired spectral loss, but gradient only from samples "
            "with timestep <= 200."
        ),
    },
}

# Exact same free-generation convention as B1 V2 / s2_56 / s2_58.
SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4
SAMPLING_SEED_BASE = 20261001

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
# Reproducibility / IO
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
        str(
            path
        )
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

        f.write(
            "\n"
        )

    os.replace(
        temp,
        path,
    )


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(
        str(
            path
        )
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
        str(
            path
        )
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
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            f"Cannot save empty CSV: "
            f"{path}"
        )

    temp = Path(
        str(
            path
        )
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
                rows[
                    0
                ].keys()
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


def arm_checkpoint_path(
    arm_name: str,
) -> Path:
    return (
        CHECKPOINT_DIR
        / f"s2_60_{arm_name.lower()}_step500.pt"
    )


def arm_synthetic_path(
    arm_name: str,
) -> Path:
    return (
        GENERATED_DIR
        / f"s2_60_{arm_name.lower()}_step500_synthetic_v1.npz"
    )


# ============================================================
# Exact references
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

    control_package = np.load(
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

    control_indices = (
        control_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    control_ids = (
        control_package[
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
        == control_indices
    ):
        raise RuntimeError(
            "Audit subject-index cohort mismatch."
        )

    if not (
        subject_ids
        == b1_ids
        == control_ids
    ):
        raise RuntimeError(
            "Audit subject-ID cohort mismatch."
        )

    b1_synth_z = b1_package[
        "epoch_010_z"
    ].astype(
        np.float32,
        copy=False,
    )

    control_synth_z = control_package[
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

    if control_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected Control-E5 synthetic shape: "
            f"{control_synth_z.shape}"
        )

    return (
        real_z,
        b1_synth_z,
        control_synth_z,
        subject_indices,
        subject_ids,
    )


# ============================================================
# Matched training stream
# ============================================================

def update_stream_digest(
    digest: hashlib._hashlib.HASH,
    *,
    x0: torch.Tensor,
    state: torch.Tensor,
    subject: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon: torch.Tensor,
) -> None:
    """
    Cheap but strong matched-stream audit.

    Hashes:
      state
      subject
      timestep
      first 8 temporal samples from all channels for x0 and epsilon

    This is not a cryptographic hash of every sample byte, but any arm-level
    difference in loader order / timestep stream / noise stream should change it.
    """

    digest.update(
        state.detach().cpu().numpy().tobytes()
    )

    digest.update(
        subject.detach().cpu().numpy().tobytes()
    )

    digest.update(
        timesteps.detach().cpu().numpy().tobytes()
    )

    digest.update(
        x0[
            ...,
            :8
        ]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    )

    digest.update(
        epsilon[
            ...,
            :8
        ]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    )


def finite_gradient_norm(
    model: torch.nn.Module,
) -> float:
    squared_sum = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        if not torch.isfinite(
            grad
        ).all():
            raise RuntimeError(
                "Nonfinite gradient in s2_60."
            )

        squared_sum += float(
            torch.sum(
                grad.float()
                * grad.float()
            ).item()
        )

    return float(
        math.sqrt(
            squared_sum
        )
    )


def run_arm_training(
    *,
    arm_name: str,
    arm_config: dict[str, Any],
    frozen_b1: dict[str, Any],
    parent_control_checkpoint: dict[str, Any],
    train_dataset: SubjectIndexedDataset,
    scaler: dict[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TRAIN ARM {arm_name}"
    )

    print(
        "=" * 80
    )

    print(
        f"role: "
        f"{arm_config['role']}"
    )

    print(
        f"lambda_spectral: "
        f"{arm_config['lambda_spectral']}"
    )

    print(
        f"max_spectral_timestep: "
        f"{arm_config['max_spectral_timestep']}"
    )

    # Reinitialize ALL random state so every arm sees the exact same
    # loader / t / epsilon streams.
    set_seed(
        MASTER_SEED
    )

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        TRAIN_LOADER_SEED
    )

    noise_generator = torch.Generator(
        device="cuda"
    )

    noise_generator.manual_seed(
        TRAIN_NOISE_SEED
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

    # Exact same parent weights for every arm.
    model = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=(
            parent_control_checkpoint[
                "model_state_dict"
            ]
        ),
        device=device,
    )

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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

    spectral_loss_fn = (
        PhysicalLogPSDLoss(
            means_uv=np.asarray(
                scaler[
                    "means_uv"
                ],
                dtype=np.float32,
            ),
            stds_uv=np.asarray(
                scaler[
                    "stds_uv"
                ],
                dtype=np.float32,
            ),
            sampling_rate_hz=SFREQ,
            fmin_hz=SPECTRAL_FMIN_HZ,
            fmax_hz=SPECTRAL_FMAX_HZ,
            log_eps=SPECTRAL_LOG_EPS,
            huber_beta_db=(
                SPECTRAL_HUBER_BETA_DB
            ),
            n_times=N_TIMES,
        )
        .to(
            device
        )
    )

    lambda_spectral = float(
        arm_config[
            "lambda_spectral"
        ]
    )

    max_spectral_timestep = arm_config[
        "max_spectral_timestep"
    ]

    stream_digest = hashlib.sha256()

    successful_steps = 0
    presented_examples = 0

    v_sum = 0.0
    active_spec_sum = 0.0
    full_spec_monitor_sum = 0.0
    total_sum = 0.0

    active_spec_examples = 0

    gradient_norms = []

    arm_start = time.perf_counter()

    loader_iterator = iter(
        train_loader
    )

    while successful_steps < PILOT_SUCCESSFUL_STEPS:
        try:
            batch = next(
                loader_iterator
            )
        except StopIteration:
            raise RuntimeError(
                "s2_60 unexpectedly exhausted one Train epoch "
                "before 500 successful steps."
            )

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

        update_stream_digest(
            stream_digest,
            x0=x0,
            state=state,
            subject=subject,
            timesteps=timesteps,
            epsilon=epsilon,
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

        # FP32 model forward by protocol.
        pred_v = model(
            xt,
            timesteps,
            state,
            subject,
        )

        pred_v_f = pred_v.float()

        target_v_f = target_v.float()

        v_loss = torch.mean(
            (
                pred_v_f
                - target_v_f
            )
            ** 2
        )

        x0_hat = reconstruct_x0_from_v(
            xt=xt,
            v_pred=pred_v_f,
            timesteps=timesteps,
            alpha_bar=alpha_bar,
        )

        # Full spectral monitor is detached and never affects gradients.
        with torch.no_grad():
            full_spec_monitor = spectral_loss_fn(
                x0_hat.detach(),
                x0,
            )

        if (
            lambda_spectral
            > 0.0
            and max_spectral_timestep
            is not None
        ):
            active_mask = (
                timesteps
                <= int(
                    max_spectral_timestep
                )
            )

            active_count = int(
                torch.sum(
                    active_mask
                ).item()
            )

            if active_count > 0:
                active_spec_loss = spectral_loss_fn(
                    x0_hat[
                        active_mask
                    ],
                    x0[
                        active_mask
                    ],
                )

                total_loss = (
                    v_loss
                    + lambda_spectral
                    * active_spec_loss
                )
            else:
                # Zero contribution, same dtype/device, no spectral graph.
                active_spec_loss = torch.zeros(
                    (),
                    device=device,
                    dtype=torch.float32,
                )

                total_loss = v_loss
        else:
            active_count = 0

            active_spec_loss = torch.zeros(
                (),
                device=device,
                dtype=torch.float32,
            )

            total_loss = v_loss

        if not (
            torch.isfinite(
                v_loss
            )
            and torch.isfinite(
                full_spec_monitor
            )
            and torch.isfinite(
                active_spec_loss
            )
            and torch.isfinite(
                total_loss
            )
        ):
            raise RuntimeError(
                f"Nonfinite s2_60 loss in arm {arm_name}."
            )

        total_loss.backward()

        preclip_grad_norm = finite_gradient_norm(
            model
        )

        clip_return = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        if not torch.isfinite(
            clip_return
        ):
            raise RuntimeError(
                f"Nonfinite clip norm in arm {arm_name}."
            )

        optimizer.step()

        successful_steps += 1

        presented_examples += current_batch_size

        v_sum += (
            float(
                v_loss.item()
            )
            * current_batch_size
        )

        full_spec_monitor_sum += (
            float(
                full_spec_monitor.item()
            )
            * current_batch_size
        )

        total_sum += (
            float(
                total_loss.item()
            )
            * current_batch_size
        )

        if active_count > 0:
            active_spec_sum += (
                float(
                    active_spec_loss.item()
                )
                * active_count
            )

            active_spec_examples += active_count

        gradient_norms.append(
            preclip_grad_norm
        )

        if (
            successful_steps == 1
            or successful_steps
            % 50
            == 0
            or successful_steps
            == PILOT_SUCCESSFUL_STEPS
        ):
            active_fraction = (
                active_spec_examples
                / presented_examples
            )

            active_spec_mean = (
                active_spec_sum
                / active_spec_examples
                if active_spec_examples
                > 0
                else float(
                    "nan"
                )
            )

            print(
                f"{arm_name} "
                f"| step={successful_steps:>3}/"
                f"{PILOT_SUCCESSFUL_STEPS} "
                f"| v={v_sum/presented_examples:.6f} "
                f"| full-spec-mon="
                f"{full_spec_monitor_sum/presented_examples:.4f} "
                f"| active-spec="
                f"{active_spec_mean:.4f} "
                f"| active-frac="
                f"{active_fraction:.3f} "
                f"| total="
                f"{total_sum/presented_examples:.6f}"
            )

    arm_seconds = (
        time.perf_counter()
        - arm_start
    )

    stream_sha256 = stream_digest.hexdigest()

    final_model_state = model.state_dict()

    checkpoint = {
        "status": (
            "S2_60_CORRECTIVE_LOW_T_SPECTRAL_PILOT_CHECKPOINT"
        ),
        "arm": (
            arm_name
        ),
        "parent": str(
            CONTROL_E5_CHECKPOINT
        ),
        "parent_model_state_sha256": (
            parent_control_checkpoint[
                "model_state_sha256"
            ]
        ),
        "successful_optimizer_steps": (
            successful_steps
        ),
        "presented_examples": (
            presented_examples
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
        "lambda_spectral": (
            lambda_spectral
        ),
        "max_spectral_timestep": (
            max_spectral_timestep
        ),
        "model_state_dict": (
            final_model_state
        ),
        "model_state_sha256": (
            state_dict_sha256(
                final_model_state
            )
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "matched_stream_sha256": (
            stream_sha256
        ),
        "train_summary": {
            "mean_v_mse": float(
                v_sum
                / presented_examples
            ),
            "mean_full_spectral_monitor": float(
                full_spec_monitor_sum
                / presented_examples
            ),
            "mean_active_spectral_loss": (
                float(
                    active_spec_sum
                    / active_spec_examples
                )
                if active_spec_examples
                > 0
                else None
            ),
            "active_spectral_examples": (
                active_spec_examples
            ),
            "active_spectral_fraction": float(
                active_spec_examples
                / presented_examples
            ),
            "mean_total_objective": float(
                total_sum
                / presented_examples
            ),
            "mean_preclip_gradient_norm": float(
                np.mean(
                    gradient_norms
                )
            ),
            "max_preclip_gradient_norm": float(
                np.max(
                    gradient_norms
                )
            ),
            "runtime_seconds": (
                arm_seconds
            ),
        },
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "formal_candidate_selected": False,
    }

    history_row = {
        "arm": (
            arm_name
        ),
        "lambda_spectral": (
            lambda_spectral
        ),
        "max_spectral_timestep": (
            ""
            if max_spectral_timestep
            is None
            else max_spectral_timestep
        ),
        "successful_steps": (
            successful_steps
        ),
        "presented_examples": (
            presented_examples
        ),
        "mean_v_mse": (
            checkpoint[
                "train_summary"
            ][
                "mean_v_mse"
            ]
        ),
        "mean_full_spectral_monitor": (
            checkpoint[
                "train_summary"
            ][
                "mean_full_spectral_monitor"
            ]
        ),
        "mean_active_spectral_loss": (
            ""
            if checkpoint[
                "train_summary"
            ][
                "mean_active_spectral_loss"
            ]
            is None
            else checkpoint[
                "train_summary"
            ][
                "mean_active_spectral_loss"
            ]
        ),
        "active_spectral_fraction": (
            checkpoint[
                "train_summary"
            ][
                "active_spectral_fraction"
            ]
        ),
        "mean_total_objective": (
            checkpoint[
                "train_summary"
            ][
                "mean_total_objective"
            ]
        ),
        "mean_preclip_gradient_norm": (
            checkpoint[
                "train_summary"
            ][
                "mean_preclip_gradient_norm"
            ]
        ),
        "max_preclip_gradient_norm": (
            checkpoint[
                "train_summary"
            ][
                "max_preclip_gradient_norm"
            ]
        ),
        "runtime_seconds": (
            arm_seconds
        ),
        "stream_sha256": (
            stream_sha256
        ),
        "model_state_sha256": (
            checkpoint[
                "model_state_sha256"
            ]
        ),
    }

    print(
        f"{arm_name} complete "
        f"| stream={stream_sha256[:16]}... "
        f"| runtime={arm_seconds/60:.2f} min"
    )

    return (
        checkpoint,
        history_row,
    )


# ============================================================
# Free-generation
# ============================================================

def generate_arm_samples(
    *,
    arm_name: str,
    checkpoint: dict[str, Any],
    frozen_b1: dict[str, Any],
    selected_subjects: list[int],
    selected_subject_ids: list[str],
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    output_path = arm_synthetic_path(
        arm_name
    )

    checkpoint_path = arm_checkpoint_path(
        arm_name
    )

    checkpoint_hash = sha256_file(
        checkpoint_path
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
                f"Cached cohort mismatch for arm "
                f"{arm_name}."
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
                f"Cached subject-ID mismatch for arm "
                f"{arm_name}."
            )

        if str(
            package[
                "checkpoint_sha256"
            ][
                0
            ]
        ) != checkpoint_hash:
            raise RuntimeError(
                f"Cached checkpoint hash mismatch "
                f"for arm {arm_name}."
            )

        if int(
            package[
                "sampling_seed_base"
            ][
                0
            ]
        ) != SAMPLING_SEED_BASE:
            raise RuntimeError(
                f"Cached sampling seed mismatch "
                f"for arm {arm_name}."
            )

        synth_z = package[
            "synth_z"
        ].astype(
            np.float32,
            copy=False,
        )

        if synth_z.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"Cached synthetic shape mismatch "
                f"for arm {arm_name}."
            )

        print(
            f"{arm_name}: validated synthetic cache -> REUSE"
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

    model.eval()

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"FREE GENERATION — ARM {arm_name}"
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
            f"{arm_name}: {synth_z.shape}"
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
        sampling_seed_base=np.asarray(
            [
                SAMPLING_SEED_BASE
            ],
            dtype=np.int64,
        ),
        generation_seconds=np.asarray(
            [
                time.perf_counter()
                - start
            ],
            dtype=np.float64,
        ),
    )

    print(
        f"{arm_name} generation: "
        f"{(time.perf_counter() - start)/60:.2f} min"
    )

    del model

    torch.cuda.empty_cache()

    return synth_z


# ============================================================
# Metric package
# ============================================================

def evaluate_variant(
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


def metric_row(
    *,
    variant: str,
    bundle: dict[str, Any],
    b1_baseline: dict[str, Any],
    control_parent_bundle: dict[str, Any],
) -> dict[str, Any]:
    generic = bundle[
        "generic"
    ]

    subject = generic[
        "subject_identity"
    ]

    full_psd = float(
        bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    b1_psd = float(
        b1_baseline[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    parent_psd = float(
        control_parent_bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    if variant == "B1":
        guard = None
    else:
        guard = evaluate_eligibility(
            candidate=bundle,
            b1_baseline=b1_baseline,
        )

    return {
        "variant": (
            variant
        ),
        "full_psd_mae_db": (
            full_psd
        ),
        "full_psd_improvement_vs_B1_fraction": float(
            (
                b1_psd
                - full_psd
            )
            / b1_psd
        ),
        "full_psd_improvement_vs_ControlE5_fraction": float(
            (
                parent_psd
                - full_psd
            )
            / parent_psd
        ),
        "eo_scale": (
            generic[
                "scale"
            ][
                "EO"
            ][
                "normalized_std_ratio"
            ]
        ),
        "ec_scale": (
            generic[
                "scale"
            ][
                "EC"
            ][
                "normalized_std_ratio"
            ]
        ),
        "mean_bandpower_mae_db": (
            generic[
                "mean_bandpower_mae_db"
            ]
        ),
        "alpha_effect_mae_db": (
            generic[
                "alpha_effect"
            ][
                "channel_mae_db"
            ]
        ),
        "correlation_r_eo": (
            generic[
                "spatial"
            ][
                "EO"
            ][
                "correlation_r"
            ]
        ),
        "correlation_r_ec": (
            generic[
                "spatial"
            ][
                "EC"
            ][
                "correlation_r"
            ]
        ),
        "alpha_dwpli2_r_eo": (
            generic[
                "spatial"
            ][
                "EO"
            ][
                "alpha_dwpli2_r"
            ]
        ),
        "alpha_dwpli2_r_ec": (
            generic[
                "spatial"
            ][
                "EC"
            ][
                "alpha_dwpli2_r"
            ]
        ),
        "subject_top1": (
            subject[
                "subject_retrieval_top1_accuracy"
            ]
        ),
        "subject_top5": (
            subject[
                "subject_retrieval_top5_accuracy"
            ]
        ),
        "subject_fingerprint_r_median": (
            subject[
                "subject_fingerprint_pearson_r_median"
            ]
        ),
        "subject_fingerprint_rmse_mean": (
            subject[
                "subject_fingerprint_standardized_rmse_mean"
            ]
        ),
        "subject_margin": (
            subject[
                "subject_correct_vs_median_wrong_margin_mean"
            ]
        ),
        "fixed_wrong_win_fraction": (
            subject[
                "subject_correct_better_than_fixed_wrong_fraction"
            ]
        ),
        "passes_existing_B2_guards_vs_B1": (
            ""
            if guard is None
            else bool(
                guard[
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
        "SCENARIO 2 — s2_60 B2 CORRECTIVE LOW-T SPECTRAL 500-STEP PILOT"
    )

    print(
        "=" * 80
    )

    print(
        "\nScientific question:"
    )

    print(
        "  Does restricting the CURRENT paired spectral gradient to low"
    )

    print(
        "  diffusion timesteps preserve the strong Control-E5 generator"
    )

    print(
        "  while providing additional free-generation spectral benefit?"
    )

    print(
        "\nArms, all initialized from exact matched lambda=0 Control E5:"
    )

    for arm_name, arm_config in ARM_CONFIG.items():
        print(
            f"  {arm_name}: "
            f"lambda={arm_config['lambda_spectral']} "
            f"| max spectral t="
            f"{arm_config['max_spectral_timestep']}"
        )

    print(
        "\nPilot design:"
    )

    print(
        f"  successful optimizer steps/arm: "
        f"{PILOT_SUCCESSFUL_STEPS}"
    )

    print(
        f"  lr: "
        f"{LEARNING_RATE}"
    )

    print(
        f"  batch: "
        f"{BATCH_SIZE}"
    )

    print(
        "  precision: FP32 (intentional matched-pilot choice)"
    )

    print(
        "  same loader/timestep/noise streams across all arms: YES"
    )

    print(
        "  Validation EEG accessed: NO"
    )

    print(
        "  Final Test accessed: NO"
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
    # Step 1 — Source audit
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B1 / CONTROL-E5 / s2_58 / s2_59 SOURCES"
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
        CONTROL_RESULT,
        CONTROL_E5_SYNTHETIC_NPZ,
        S2_58_SUMMARY,
        S2_59_SUMMARY,
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

    with CONTROL_RESULT.open(
        "r",
        encoding="utf-8",
    ) as f:
        control_result = json.load(
            f
        )

    with S2_58_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_58_summary = json.load(
            f
        )

    with S2_59_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_59_summary = json.load(
            f
        )

    if b1_selection[
        "status"
    ] != "B1_SELECTION_FROZEN":
        raise RuntimeError(
            "B1 is not frozen."
        )

    if int(
        b1_selection[
            "selected_epoch"
        ]
    ) != 10:
        raise RuntimeError(
            "Unexpected frozen B1 epoch."
        )

    if control_result[
        "status"
    ] != "B2_MATCHED_LAMBDA0_CONTROL_20_EPOCHS_COMPLETE":
        raise RuntimeError(
            "Matched lambda=0 control is not complete."
        )

    if s2_58_summary[
        "status"
    ] != "S2_58_JOINT_B1_LAMBDA0_LAMBDA001_FREEGEN_AUDIT_COMPLETE":
        raise RuntimeError(
            "s2_58 is not complete."
        )

    if s2_59_summary[
        "status"
    ] != "S2_59_CONTROL_E5_GUARDRAILS_AND_DENOISER_DIAGNOSTICS_COMPLETE":
        raise RuntimeError(
            "s2_59 is not complete."
        )

    if s2_59_summary[
        "control_E5_guardrail_diagnostic"
    ][
        "all_existing_B2_guards_pass_diagnostic"
    ] is not True:
        raise RuntimeError(
            "Control E5 did not pass the existing B2 guardrails."
        )

    if control_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Control result reports Final Test access."
        )

    frozen_b1_hash_before = sha256_file(
        FROZEN_B1_GENERATOR
    )

    parent_control_hash_before = sha256_file(
        CONTROL_E5_CHECKPOINT
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    parent_control = torch.load(
        CONTROL_E5_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if parent_control[
        "status"
    ] != "B2_MATCHED_LAMBDA0_CONTROL_CHECKPOINT":
        raise RuntimeError(
            "Unexpected Control E5 checkpoint status."
        )

    if int(
        parent_control[
            "epoch"
        ]
    ) != 5:
        raise RuntimeError(
            "Unexpected Control E5 checkpoint epoch."
        )

    if abs(
        float(
            parent_control[
                "lambda_spectral"
            ]
        )
        - 0.0
    ) > 1e-12:
        raise RuntimeError(
            "Control E5 parent lambda mismatch."
        )

    parent_model_state_hash = state_dict_sha256(
        parent_control[
            "model_state_dict"
        ]
    )

    if (
        parent_model_state_hash
        != parent_control[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Control E5 parent model-state hash mismatch."
        )

    print(
        f"frozen B1 SHA256: "
        f"{frozen_b1_hash_before}"
    )

    print(
        f"Control E5 parent SHA256: "
        f"{parent_control_hash_before}"
    )

    print(
        f"Control E5 model-state SHA256: "
        f"{parent_model_state_hash}"
    )

    print(
        "source audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze pilot protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE s2_60 PILOT PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_60_CORRECTIVE_LOW_T_SPECTRAL_500STEP_PROTOCOL_FROZEN"
        ),
        "scientific_role": (
            "Matched Train-only mechanism pilot. Not a final B2 selection."
        ),
        "parent": {
            "model": (
                "matched lambda=0 Control E5"
            ),
            "checkpoint": str(
                CONTROL_E5_CHECKPOINT
            ),
            "file_sha256": (
                parent_control_hash_before
            ),
            "model_state_sha256": (
                parent_model_state_hash
            ),
        },
        "arms": (
            ARM_CONFIG
        ),
        "training": {
            "successful_steps_per_arm": (
                PILOT_SUCCESSFUL_STEPS
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
            "fresh_optimizer_per_arm": True,
            "same_initial_weights_across_arms": True,
            "same_loader_timestep_noise_streams_across_arms": True,
            "master_seed": (
                MASTER_SEED
            ),
            "loader_seed": (
                TRAIN_LOADER_SEED
            ),
            "noise_seed": (
                TRAIN_NOISE_SEED
            ),
        },
        "spectral_loss": {
            "class": (
                "PhysicalLogPSDLoss from existing s2_53"
            ),
            "lambda_when_active": (
                SPECTRAL_LAMBDA
            ),
            "fmin_hz": (
                SPECTRAL_FMIN_HZ
            ),
            "fmax_hz": (
                SPECTRAL_FMAX_HZ
            ),
            "huber_beta_db": (
                SPECTRAL_HUBER_BETA_DB
            ),
            "only_change_in_gated_arms": (
                "spectral gradient is computed only for batch samples "
                "with timestep <= predeclared gate"
            ),
            "estimator_changed": False,
            "distributional_target_added": False,
            "B1_distillation_added": False,
            "adapter_added": False,
        },
        "free_generation_audit": {
            "real_reference": str(
                B1_REAL_REFERENCE_NPZ
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
            "sampler": (
                "1000-step ancestral DDPM / v-prediction / FP32"
            ),
            "sampling_seed_base": (
                SAMPLING_SEED_BASE
            ),
            "existing_metric_helpers_reused": True,
        },
        "pilot_advancement_rule": (
            "A gated arm may advance to a later follow-up only if it "
            "passes ALL existing frozen B2 guards versus frozen B1 AND "
            "its full 1-40 Hz PSD MAE is lower than BOTH the Control-E5 "
            "parent and the matched new L0 500-step continuation arm. "
            "Among multiple advancing gated arms: lowest full PSD MAE, "
            "then lower mean bandpower MAE, then lower alpha-effect MAE."
        ),
        "important_limit": (
            "Passing this pilot does NOT freeze a new B2. It only identifies "
            "a candidate mechanism for a separately frozen follow-up protocol."
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "bulk_generation_started": False,
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
                "Conflicting s2_60 protocol already exists."
            )

        print(
            "identical s2_60 protocol already exists."
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
    # Step 3 — Build Train dataset + exact audit reference
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD REAL TRAIN DATA + EXACT SUBJECT-AWARE AUDIT REFERENCE"
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
        control_parent_synth_z,
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
        f"Real audit reference shape: "
        f"{real_z.shape}"
    )

    # ========================================================
    # Step 4 — Train all matched arms
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — RUN THREE MATCHED 500-STEP PILOT ARMS"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for s2_60."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    arm_checkpoints: dict[
        str,
        dict[str, Any],
    ] = {}

    history_rows = []

    for arm_name, arm_config in ARM_CONFIG.items():
        (
            checkpoint,
            history_row,
        ) = run_arm_training(
            arm_name=arm_name,
            arm_config=arm_config,
            frozen_b1=frozen_b1,
            parent_control_checkpoint=(
                parent_control
            ),
            train_dataset=train_dataset,
            scaler=scaler,
            device=device,
        )

        checkpoint_path = arm_checkpoint_path(
            arm_name
        )

        atomic_torch_save(
            checkpoint,
            checkpoint_path,
        )

        arm_checkpoints[
            arm_name
        ] = checkpoint

        history_rows.append(
            history_row
        )

        print(
            f"checkpoint: "
            f"{checkpoint_path}"
        )

        torch.cuda.empty_cache()

    save_csv_rows(
        history_rows,
        HISTORY_CSV,
    )

    # Exact stream matching is REQUIRED.
    stream_hashes = {
        row[
            "arm"
        ]: row[
            "stream_sha256"
        ]
        for row in history_rows
    }

    if len(
        set(
            stream_hashes.values()
        )
    ) != 1:
        raise RuntimeError(
            "Matched training streams differ across s2_60 arms."
        )

    matched_stream_hash = next(
        iter(
            stream_hashes.values()
        )
    )

    print(
        "\nMatched training-stream audit: PASS"
    )

    print(
        f"stream SHA256: "
        f"{matched_stream_hash}"
    )

    # ========================================================
    # Step 5 — Build fixed sampler + generate each trained arm
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — MATCHED 1000-STEP FREE GENERATION FOR ALL PILOT ARMS"
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

    arm_synth: dict[
        str,
        np.ndarray,
    ] = {}

    for arm_name in ARM_CONFIG:
        arm_synth[
            arm_name
        ] = generate_arm_samples(
            arm_name=arm_name,
            checkpoint=(
                arm_checkpoints[
                    arm_name
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
    # Step 6 — Evaluate B1 / Control-E5 parent / all new arms
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — FREE-GENERATION FIDELITY AUDIT"
    )

    print(
        "=" * 80
    )

    bundles = {
        "B1": evaluate_variant(
            real_z=real_z,
            synth_z=b1_synth_z,
            scaler=scaler,
        ),
        "CONTROL_E5_PARENT": evaluate_variant(
            real_z=real_z,
            synth_z=control_parent_synth_z,
            scaler=scaler,
        ),
    }

    for arm_name, synth_z in arm_synth.items():
        bundles[
            arm_name
        ] = evaluate_variant(
            real_z=real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

    b1_baseline = bundles[
        "B1"
    ]

    parent_bundle = bundles[
        "CONTROL_E5_PARENT"
    ]

    metric_rows = []

    for variant_name in (
        "B1",
        "CONTROL_E5_PARENT",
        "L0",
        "GATE100",
        "GATE200",
    ):
        row = metric_row(
            variant=variant_name,
            bundle=(
                bundles[
                    variant_name
                ]
            ),
            b1_baseline=(
                b1_baseline
            ),
            control_parent_bundle=(
                parent_bundle
            ),
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
            f"  vs B1 improvement: "
            f"{100.0 * row['full_psd_improvement_vs_B1_fraction']:+.2f}%"
        )

        print(
            f"  vs Control-E5 improvement: "
            f"{100.0 * row['full_psd_improvement_vs_ControlE5_fraction']:+.2f}%"
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
            f"  existing B2 guards vs B1: "
            f"{row['passes_existing_B2_guards_vs_B1']}"
        )

    save_csv_rows(
        metric_rows,
        METRICS_CSV,
    )

    # ========================================================
    # Step 7 — Apply predeclared pilot advancement rule
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — APPLY PREDECLARED PILOT ADVANCEMENT RULE"
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

    parent_psd = float(
        rows_by_variant[
            "CONTROL_E5_PARENT"
        ][
            "full_psd_mae_db"
        ]
    )

    l0_psd = float(
        rows_by_variant[
            "L0"
        ][
            "full_psd_mae_db"
        ]
    )

    advancing_arms = []

    advancement_detail = {}

    for arm_name in (
        "GATE100",
        "GATE200",
    ):
        row = rows_by_variant[
            arm_name
        ]

        conditions = {
            "passes_existing_B2_guards_vs_B1": bool(
                row[
                    "passes_existing_B2_guards_vs_B1"
                ]
            ),
            "beats_Control_E5_parent_full_PSD": bool(
                float(
                    row[
                        "full_psd_mae_db"
                    ]
                )
                < parent_psd
            ),
            "beats_matched_L0_500step_full_PSD": bool(
                float(
                    row[
                        "full_psd_mae_db"
                    ]
                )
                < l0_psd
            ),
        }

        advance = bool(
            all(
                conditions.values()
            )
        )

        advancement_detail[
            arm_name
        ] = {
            "conditions": (
                conditions
            ),
            "advance": (
                advance
            ),
        }

        if advance:
            advancing_arms.append(
                arm_name
            )

        print(
            f"{arm_name}:"
        )

        for condition_name, passed in conditions.items():
            print(
                f"  {condition_name}: "
                f"{passed}"
            )

        print(
            f"  ADVANCE: "
            f"{advance}"
        )

    recommended_arm = None

    if advancing_arms:
        recommended_arm = min(
            advancing_arms,
            key=lambda arm_name: (
                float(
                    rows_by_variant[
                        arm_name
                    ][
                        "full_psd_mae_db"
                    ]
                ),
                float(
                    rows_by_variant[
                        arm_name
                    ][
                        "mean_bandpower_mae_db"
                    ]
                ),
                float(
                    rows_by_variant[
                        arm_name
                    ][
                        "alpha_effect_mae_db"
                    ]
                ),
            ),
        )

    print(
        f"\nadvancing gated arms: "
        f"{advancing_arms}"
    )

    print(
        f"recommended gated mechanism: "
        f"{recommended_arm}"
    )

    # ========================================================
    # Step 8 — Verify source immutability and final summary
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
            "Frozen B1 changed during s2_60."
        )

    if sha256_file(
        CONTROL_E5_CHECKPOINT
    ) != parent_control_hash_before:
        raise RuntimeError(
            "Control E5 parent changed during s2_60."
        )

    summary = {
        "status": (
            "S2_60_CORRECTIVE_LOW_T_SPECTRAL_500STEP_PILOT_COMPLETE"
        ),
        "scientific_role": (
            "Matched Train-only low-t spectral gating pilot. "
            "No final B2 was frozen."
        ),
        "parent": {
            "checkpoint": str(
                CONTROL_E5_CHECKPOINT
            ),
            "file_sha256": (
                parent_control_hash_before
            ),
            "model_state_sha256": (
                parent_model_state_hash
            ),
        },
        "training_stream_sha256": (
            matched_stream_hash
        ),
        "training_history": {
            row[
                "arm"
            ]: row
            for row in history_rows
        },
        "free_generation_metrics": {
            row[
                "variant"
            ]: {
                key: value
                for key, value in row.items()
                if key
                != "variant"
            }
            for row in metric_rows
        },
        "advancement_detail": (
            advancement_detail
        ),
        "advancing_gated_arms": (
            advancing_arms
        ),
        "recommended_gated_mechanism": (
            recommended_arm
        ),
        "artifacts": {
            "protocol": str(
                PROTOCOL_FILE
            ),
            "history_csv": str(
                HISTORY_CSV
            ),
            "metrics_csv": str(
                METRICS_CSV
            ),
            "checkpoints": {
                arm_name: str(
                    arm_checkpoint_path(
                        arm_name
                    )
                )
                for arm_name in ARM_CONFIG
            },
            "synthetic": {
                arm_name: str(
                    arm_synthetic_path(
                        arm_name
                    )
                )
                for arm_name in ARM_CONFIG
            },
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "bulk_synthetic_generation_started": False,
            "B1_unchanged": True,
            "Control_E5_parent_unchanged": True,
            "thresholds_changed": False,
            "formal_B2_freeze_decision_made": False,
        },
        "next": (
            "If a gated arm advances, define a separate frozen follow-up "
            "protocol before any longer training. If no gated arm advances, "
            "do not sweep more timestep cutoffs post-hoc; move to a new "
            "objective family (e.g. distributional or B1-anchored correction)."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
    )

    print(
        "frozen B1 unchanged: PASS"
    )

    print(
        "Control E5 parent unchanged: PASS"
    )

    print(
        f"summary JSON: "
        f"{SUMMARY_JSON}"
    )

    # Close cached HDF5 handles.
    train_base_dataset.close()

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "s2_60 CORRECTIVE LOW-T SPECTRAL 500-STEP PILOT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"matched stream SHA256: "
        f"{matched_stream_hash}"
    )

    print(
        f"advancing gated arms: "
        f"{advancing_arms}"
    )

    print(
        f"recommended gated mechanism: "
        f"{recommended_arm}"
    )

    print(
        "\nValidation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Formal B2 freeze decision made: NO"
    )

    print(
        "\nNEXT:"
    )

    if recommended_arm is None:
        print(
            "NO GATED ARM ADVANCED — INTERPRET AND MOVE TO A NEW "
            "CORRECTIVE OBJECTIVE FAMILY; DO NOT POST-HOC SWEEP MORE GATES."
        )
    else:
        print(
            "FREEZE A NEW FOLLOW-UP PROTOCOL AROUND THE RECOMMENDED "
            "GATED MECHANISM BEFORE ANY LONGER TRAINING."
        )


if __name__ == "__main__":
    main()
