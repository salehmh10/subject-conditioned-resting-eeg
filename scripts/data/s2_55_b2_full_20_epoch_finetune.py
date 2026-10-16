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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)
from s2_24_full_train_loader_sanity import (
    CachedRealTrainWindowDataset,
)
from s2_27_logbandpower_logreg_validation import (
    build_validation_window_index,
)
from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)
from s2_39_b0_epsilon_to_v_conversion_sanity import (
    make_v_target,
    q_sample,
)
from s2_47_b1_architecture_warmstart_sanity import (
    B1SubjectStateUNet1D,
    SubjectIndexedDataset,
    state_dict_sha256,
)
from s2_53_b2_lambda_calibration_pilot import (
    PhysicalLogPSDLoss,
    reconstruct_x0_from_v,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

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

B2_PROTOCOL_V1_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "spectral_regularized_v1"
)

B2_PROTOCOL_V1 = (
    B2_PROTOCOL_V1_ROOT
    / "metrics"
    / "b2_spectral_regularized_protocol_v1.json"
)

B2_PROTOCOL_V2_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "spectral_regularized_v2"
)

B2_SELECTED_LAMBDA_FILE = (
    B2_PROTOCOL_V2_ROOT
    / "metrics"
    / "b2_selected_lambda_v2.json"
)

B2_FULL_RUN_HANDOFF = (
    B2_PROTOCOL_V2_ROOT
    / "handoff"
    / "b2_full_run_handoff_v2.json"
)


# ============================================================
# Full B2 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "full_finetune_v1"
)

CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
LOG_DIR = RUN_ROOT / "logs"
METRIC_DIR = RUN_ROOT / "metrics"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_SNAPSHOT = (
    METRIC_DIR
    / "b2_full_finetune_protocol_snapshot_v1.json"
)

STATE_FILE = (
    METRIC_DIR
    / "b2_full_finetune_state_v1.json"
)

RESULT_FILE = (
    METRIC_DIR
    / "b2_full_finetune_result_v1.json"
)

HISTORY_FILE = (
    LOG_DIR
    / "b2_full_finetune_history_v1.csv"
)

EVENT_FILE = (
    LOG_DIR
    / "b2_full_finetune_events_v1.jsonl"
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "latest.pt"
)

BEST_VALIDATION_MONITOR_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best_validation_monitor.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "final.pt"
)

LOSS_FIGURE = (
    FIGURE_DIR
    / "b2_full_finetune_loss_v1.png"
)

VALIDATION_SPECTRAL_FIGURE = (
    FIGURE_DIR
    / "b2_full_finetune_validation_spectral_v1.png"
)

SUBJECT_GAP_FIGURE = (
    FIGURE_DIR
    / "b2_full_finetune_validation_subject_gap_v1.png"
)

GRADIENT_FIGURE = (
    FIGURE_DIR
    / "b2_full_finetune_gradient_norm_v1.png"
)

AMP_SCALE_FIGURE = (
    FIGURE_DIR
    / "b2_full_finetune_amp_scale_v1.png"
)

AMP_SKIP_FIGURE = (
    FIGURE_DIR
    / "b2_full_finetune_amp_skips_v1.png"
)

MEMORY_FIGURE = (
    FIGURE_DIR
    / "b2_full_finetune_gpu_memory_v1.png"
)


# ============================================================
# Frozen full-run configuration
# ============================================================

SEED = 20260908
TRAIN_LOADER_SEED = 20261021
TRAIN_NOISE_SEED = 20261022
VALIDATION_SEED = 20261023

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_VALIDATION_WINDOWS = 9_635
EXPECTED_SUBJECTS = 198

TIMESTEPS = 1000
COSINE_S = 0.008

MAX_EPOCHS = 20
BATCH_SIZE = 64

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

LAMBDA_SPECTRAL = 0.01

MIXED_PRECISION = True
EMA_ENABLED = False

RAW_CANDIDATE_EPOCHS = [
    5,
    10,
    20,
]

CHECKPOINT_EVERY_EPOCHS = 5
PRINT_EVERY_BATCHES = 100

MAX_AMP_SKIPS_PER_EPOCH = 5
MAX_CONSECUTIVE_AMP_SKIPS = 3

WRONG_SUBJECT_OFFSET = 97

SPECTRAL_FMIN_HZ = 1.0
SPECTRAL_FMAX_HZ = 40.0
SPECTRAL_LOG_EPS = 1e-12
SPECTRAL_HUBER_BETA_DB = 1.0


# ============================================================
# Reproducibility / IO
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(str(path) + ".tmp")

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

    os.replace(temp, path)


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(str(path) + ".tmp")

    torch.save(payload, temp)
    os.replace(temp, path)


def append_event(
    kind: str,
    **fields,
) -> None:
    payload = {
        "time_unix": time.time(),
        "kind": kind,
        **fields,
    }

    with EVENT_FILE.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(
            json.dumps(
                payload,
                sort_keys=True,
            )
            + "\n"
        )


def capture_rng(
    loader_generator: torch.Generator,
    noise_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "loader_generator": loader_generator.get_state(),
        "noise_generator": noise_generator.get_state(),
    }


def restore_rng(
    state: dict[str, Any],
    loader_generator: torch.Generator,
    noise_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(
        state["loader_generator"]
    )
    noise_generator.set_state(
        state["noise_generator"]
    )


def finite_gradient_summary(
    model: nn.Module,
) -> tuple[
    bool,
    int,
    float,
    list[str],
]:
    grad_count = 0
    squared_sum = 0.0
    bad_names = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        grad_count += 1
        grad = parameter.grad.detach()

        if not torch.isfinite(
            grad
        ).all():
            bad_names.append(name)
            continue

        squared_sum += float(
            torch.sum(
                grad.float()
                * grad.float()
            ).item()
        )

    finite = len(bad_names) == 0

    norm = (
        math.sqrt(squared_sum)
        if finite
        else float("nan")
    )

    return (
        finite,
        grad_count,
        norm,
        bad_names,
    )


def assert_finite_state_dict(
    model: nn.Module,
    label: str,
) -> None:
    for name, tensor in model.state_dict().items():
        if (
            torch.is_tensor(tensor)
            and torch.is_floating_point(tensor)
            and not torch.isfinite(
                tensor
            ).all()
        ):
            raise RuntimeError(
                f"Nonfinite tensor in {label}: "
                f"{name}"
            )


# ============================================================
# Model builder
# ============================================================

def build_b2_model(
    frozen_b1: dict[str, Any],
    device: torch.device,
) -> B1SubjectStateUNet1D:
    contract = frozen_b1[
        "effective_contract"
    ]

    model = B1SubjectStateUNet1D(
        b0_contract=(
            contract[
                "parent_B0_effective_contract"
            ]
        ),
        injection_module_name=str(
            contract[
                "condition_injection_module"
            ]
        ),
        num_subjects=int(
            contract[
                "subject_count"
            ]
        ),
        subject_embedding_dim=int(
            contract[
                "subject_embedding_dim"
            ]
        ),
        condition_dim=int(
            contract[
                "subject_projection_output_dim"
            ]
        ),
    )

    model.load_state_dict(
        frozen_b1[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(device)

    return model


# ============================================================
# Deterministic Validation monitor
# ============================================================

@torch.inference_mode()
def evaluate_validation(
    *,
    model: B1SubjectStateUNet1D,
    loader: DataLoader,
    alpha_bar: torch.Tensor,
    spectral_loss_fn: PhysicalLogPSDLoss,
    device: torch.device,
) -> dict[str, float]:
    """
    Deterministic RAW B2 Validation monitor.

    Same Validation windows, t/noise stream, and wrong-subject mapping
    every epoch. This monitor is NOT the final scientific B2 selector.
    """

    model.eval()

    generator = torch.Generator(
        device="cuda"
    )
    generator.manual_seed(
        VALIDATION_SEED
    )

    correct_v_sum = 0.0
    wrong_v_sum = 0.0
    spectral_sum = 0.0
    total_objective_sum = 0.0
    output_difference_sum = 0.0
    total_examples = 0

    for batch in loader:
        x0 = batch["x"].to(
            device,
            non_blocking=True,
        )

        state = batch["y"].to(
            device,
            non_blocking=True,
        )

        subject = batch[
            "subject_index"
        ].to(
            device,
            non_blocking=True,
        )

        wrong_subject = (
            subject
            + WRONG_SUBJECT_OFFSET
        ) % EXPECTED_SUBJECTS

        if torch.any(
            wrong_subject
            == subject
        ):
            raise RuntimeError(
                "Wrong-subject mapping collision."
            )

        batch_size = int(
            x0.shape[0]
        )

        timesteps = torch.randint(
            0,
            TIMESTEPS,
            (batch_size,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

        epsilon = torch.randn(
            x0.shape,
            device=device,
            dtype=x0.dtype,
            generator=generator,
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

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=MIXED_PRECISION,
        ):
            correct_pred = model(
                xt,
                timesteps,
                state,
                subject,
            )

            wrong_pred = model(
                xt,
                timesteps,
                state,
                wrong_subject,
            )

        correct_pred_f = (
            correct_pred.float()
        )
        wrong_pred_f = (
            wrong_pred.float()
        )
        target_v_f = target_v.float()

        correct_v = torch.mean(
            (
                correct_pred_f
                - target_v_f
            )
            ** 2
        )

        wrong_v = torch.mean(
            (
                wrong_pred_f
                - target_v_f
            )
            ** 2
        )

        x0_hat = reconstruct_x0_from_v(
            xt=xt,
            v_pred=correct_pred_f,
            timesteps=timesteps,
            alpha_bar=alpha_bar,
        )

        spectral = spectral_loss_fn(
            x0_hat,
            x0,
        )

        total_objective = (
            correct_v
            + LAMBDA_SPECTRAL
            * spectral
        )

        output_difference = torch.mean(
            torch.abs(
                correct_pred_f
                - wrong_pred_f
            )
        )

        if not (
            torch.isfinite(correct_v)
            and torch.isfinite(wrong_v)
            and torch.isfinite(spectral)
            and torch.isfinite(
                total_objective
            )
            and torch.isfinite(
                output_difference
            )
        ):
            raise RuntimeError(
                "Nonfinite deterministic B2 Validation metric."
            )

        correct_v_sum += (
            float(
                correct_v.item()
            )
            * batch_size
        )

        wrong_v_sum += (
            float(
                wrong_v.item()
            )
            * batch_size
        )

        spectral_sum += (
            float(
                spectral.item()
            )
            * batch_size
        )

        total_objective_sum += (
            float(
                total_objective.item()
            )
            * batch_size
        )

        output_difference_sum += (
            float(
                output_difference.item()
            )
            * batch_size
        )

        total_examples += batch_size

    if (
        total_examples
        != EXPECTED_VALIDATION_WINDOWS
    ):
        raise RuntimeError(
            "Validation accounting mismatch."
        )

    correct_v_mse = (
        correct_v_sum
        / total_examples
    )

    wrong_v_mse = (
        wrong_v_sum
        / total_examples
    )

    return {
        "correct_subject_v_mse": float(
            correct_v_mse
        ),
        "wrong_subject_v_mse": float(
            wrong_v_mse
        ),
        "wrong_minus_correct_v_mse": float(
            wrong_v_mse
            - correct_v_mse
        ),
        "spectral_loss_db_huber": float(
            spectral_sum
            / total_examples
        ),
        "total_objective": float(
            total_objective_sum
            / total_examples
        ),
        "correct_vs_wrong_output_mean_abs": float(
            output_difference_sum
            / total_examples
        ),
    }


# ============================================================
# History / figures
# ============================================================

HISTORY_FIELDS = [
    "epoch",
    "train_v_mse",
    "train_spectral_loss_db_huber",
    "train_total_objective",
    "validation_correct_subject_v_mse",
    "validation_wrong_subject_v_mse",
    "validation_wrong_minus_correct_v_mse",
    "validation_spectral_loss_db_huber",
    "validation_total_objective",
    "validation_correct_vs_wrong_output_mean_abs",
    "mean_gradient_norm",
    "max_gradient_norm",
    "amp_skipped_steps",
    "amp_scale_end",
    "successful_optimizer_steps",
    "epoch_seconds",
    "train_windows_per_second",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "best_validation_monitor_epoch",
    "best_validation_total_objective",
]


def save_history(
    history: list[dict[str, Any]],
) -> None:
    if not history:
        return

    temp = Path(
        str(HISTORY_FILE)
        + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                HISTORY_FIELDS
            ),
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    os.replace(
        temp,
        HISTORY_FILE,
    )


def save_figures(
    history: list[dict[str, Any]],
) -> None:
    if not history:
        return

    epochs = np.asarray(
        [
            row["epoch"]
            for row in history
        ],
        dtype=np.int64,
    )

    train_v = np.asarray(
        [
            row["train_v_mse"]
            for row in history
        ],
        dtype=np.float64,
    )

    train_spec = np.asarray(
        [
            row[
                "train_spectral_loss_db_huber"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    train_total = np.asarray(
        [
            row["train_total_objective"]
            for row in history
        ],
        dtype=np.float64,
    )

    val_v = np.asarray(
        [
            row[
                "validation_correct_subject_v_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    val_spec = np.asarray(
        [
            row[
                "validation_spectral_loss_db_huber"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    val_total = np.asarray(
        [
            row[
                "validation_total_objective"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    val_gap = np.asarray(
        [
            row[
                "validation_wrong_minus_correct_v_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    grad = np.asarray(
        [
            row["mean_gradient_norm"]
            for row in history
        ],
        dtype=np.float64,
    )

    amp_scale = np.asarray(
        [
            row["amp_scale_end"]
            for row in history
        ],
        dtype=np.float64,
    )

    amp_skips = np.asarray(
        [
            row["amp_skipped_steps"]
            for row in history
        ],
        dtype=np.float64,
    )

    memory = np.asarray(
        [
            row["peak_allocated_mib"]
            for row in history
        ],
        dtype=np.float64,
    )

    # Total objective / v loss.
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        epochs,
        train_total,
        label="Train total objective",
    )

    ax.plot(
        epochs,
        val_total,
        label="Validation total objective",
    )

    ax.plot(
        epochs,
        train_v,
        linestyle="--",
        alpha=0.7,
        label="Train v-MSE",
    )

    ax.plot(
        epochs,
        val_v,
        linestyle="--",
        alpha=0.7,
        label="Validation v-MSE",
    )

    ax.set_xlabel(
        "B2 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "B2 Full Fine-Tuning — v + Spectral Objective"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        LOSS_FIGURE,
        dpi=170,
    )

    plt.close(fig)

    # Spectral loss.
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        epochs,
        train_spec,
        label="Train spectral loss",
    )

    ax.plot(
        epochs,
        val_spec,
        label="Validation spectral loss",
    )

    ax.set_xlabel(
        "B2 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Spectral Huber loss (dB)"
    )

    ax.set_title(
        "B2 Full Fine-Tuning — Spectral Loss"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        VALIDATION_SPECTRAL_FIGURE,
        dpi=170,
    )

    plt.close(fig)

    # Subject gap.
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        epochs,
        val_gap,
        marker="o",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "B2 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Wrong-subject MSE - Correct-subject MSE"
    )

    ax.set_title(
        "B2 Validation — Subject-Specific Denoising Advantage"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        SUBJECT_GAP_FIGURE,
        dpi=170,
    )

    plt.close(fig)

    # Gradient.
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        epochs,
        grad,
    )

    ax.set_xlabel(
        "B2 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Mean global gradient norm"
    )

    ax.set_title(
        "B2 Full Fine-Tuning — Gradient Norm"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        GRADIENT_FIGURE,
        dpi=170,
    )

    plt.close(fig)

    # AMP scale.
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        epochs,
        amp_scale,
    )

    ax.set_xlabel(
        "B2 fine-tuning epoch"
    )

    ax.set_ylabel(
        "GradScaler scale"
    )

    ax.set_title(
        "B2 Full Fine-Tuning — AMP Scale"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        AMP_SCALE_FIGURE,
        dpi=170,
    )

    plt.close(fig)

    # AMP skips.
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        epochs,
        amp_skips,
    )

    ax.set_xlabel(
        "B2 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Skipped optimizer steps"
    )

    ax.set_title(
        "B2 Full Fine-Tuning — AMP Skips"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        AMP_SKIP_FIGURE,
        dpi=170,
    )

    plt.close(fig)

    # GPU memory.
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        epochs,
        memory,
    )

    ax.set_xlabel(
        "B2 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Peak allocated GPU memory (MiB)"
    )

    ax.set_title(
        "B2 Full Fine-Tuning — GPU Memory"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        MEMORY_FIGURE,
        dpi=170,
    )

    plt.close(fig)


# ============================================================
# Checkpoint payload
# ============================================================

def make_checkpoint(
    *,
    epoch: int,
    protocol_snapshot: dict[str, Any],
    model: B1SubjectStateUNet1D,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    noise_generator: torch.Generator,
    history: list[dict[str, Any]],
    best_validation_monitor_epoch: int,
    best_validation_total_objective: float,
    cumulative_training_seconds: float,
    total_amp_skips: int,
    frozen_b1_file_sha256: str,
    frozen_b1_model_state_sha256: str,
) -> dict[str, Any]:
    model_state = model.state_dict()

    return {
        "status": (
            "B2_FULL_FINETUNE_CHECKPOINT"
        ),
        "epoch": int(epoch),
        "prediction_target": "v",
        "lambda_spectral": (
            LAMBDA_SPECTRAL
        ),
        "selected_weight_kind": "RAW",
        "EMA_enabled": False,
        "B2_protocol_snapshot": (
            protocol_snapshot
        ),
        "frozen_B1_parent": str(
            FROZEN_B1_GENERATOR
        ),
        "frozen_B1_file_sha256": (
            frozen_b1_file_sha256
        ),
        "frozen_B1_model_state_sha256": (
            frozen_b1_model_state_sha256
        ),
        "lambda_pilot_checkpoint_used": False,
        "model_state_dict": (
            model_state
        ),
        "model_state_sha256": (
            state_dict_sha256(
                model_state
            )
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "grad_scaler_state_dict": (
            grad_scaler.state_dict()
        ),
        "rng_state": capture_rng(
            loader_generator,
            noise_generator,
        ),
        "history": history,
        "best_validation_monitor_epoch": int(
            best_validation_monitor_epoch
        ),
        "best_validation_total_objective": float(
            best_validation_total_objective
        ),
        "cumulative_training_seconds": float(
            cumulative_training_seconds
        ),
        "total_amp_skips": int(
            total_amp_skips
        ),
        "Validation_accessed": True,
        "Final_Test_accessed": False,
        "bulk_synthetic_generation_started": False,
        "final_scientific_B2_selected": False,
        "note": (
            "Scientific B2 full-run checkpoint. Final B2 generator "
            "selection is deferred to predeclared RAW candidate "
            "generative-fidelity audit at epochs 5, 10, and 20."
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
        "SCENARIO 2 — B2 20-EPOCH SPECTRAL-REGULARIZED FULL FINE-TUNING"
    )

    print(
        "=" * 80
    )

    print(
        "\nCurrent project stage:"
    )

    print(
        "  B0: FROZEN"
    )

    print(
        "  B1: FROZEN"
    )

    print(
        "  B2 lambda selection V2: FROZEN"
    )

    print(
        "  B2 scientific 20-epoch full fine-tuning: NOW"
    )

    print(
        "  B2 final generative-fidelity selection: NOT YET"
    )

    print(
        "  Bulk synthetic generation: NOT STARTED"
    )

    print(
        "  Final Test: SEALED"
    )

    print(
        "\nFrozen full-run definition:"
    )

    print(
        "  initialization: exact frozen B1 RAW epoch 10"
    )

    print(
        "  lambda-pilot checkpoint used: NO"
    )

    print(
        "  prediction target: v"
    )

    print(
        f"  lambda_spectral: "
        f"{LAMBDA_SPECTRAL}"
    )

    print(
        f"  epochs: "
        f"{MAX_EPOCHS}"
    )

    print(
        f"  batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"  LR: "
        f"{LEARNING_RATE}"
    )

    print(
        "  EMA: OFF"
    )

    print(
        f"  predeclared RAW candidate epochs: "
        f"{RAW_CANDIDATE_EPOCHS}"
    )

    for directory in (
        CHECKPOINT_DIR,
        LOG_DIR,
        METRIC_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for B2 full fine-tuning."
        )

    device = torch.device(
        "cuda"
    )

    # ========================================================
    # Step 1 — Audit frozen sources / selected lambda
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B1 + B2 SELECTED LAMBDA"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B2_PROTOCOL_V1,
        B2_SELECTED_LAMBDA_FILE,
        B2_FULL_RUN_HANDOFF,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B2_PROTOCOL_V1.open(
        "r",
        encoding="utf-8",
    ) as f:
        b2_protocol_v1 = json.load(
            f
        )

    with B2_SELECTED_LAMBDA_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        selected_lambda_info = json.load(
            f
        )

    with B2_FULL_RUN_HANDOFF.open(
        "r",
        encoding="utf-8",
    ) as f:
        full_handoff = json.load(
            f
        )

    if b2_protocol_v1[
        "status"
    ] != "B2_SPECTRAL_REGULARIZED_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B2 protocol V1 is not frozen."
        )

    if selected_lambda_info[
        "status"
    ] != "B2_SELECTED_LAMBDA_FROZEN_V2":
        raise RuntimeError(
            "B2 selected lambda V2 is not frozen."
        )

    if abs(
        float(
            selected_lambda_info[
                "selected_lambda_spectral"
            ]
        )
        - LAMBDA_SPECTRAL
    ) > 1e-12:
        raise RuntimeError(
            "Selected lambda differs from expected 0.01."
        )

    if full_handoff[
        "status"
    ] != "B2_FULL_RUN_READY_V2":
        raise RuntimeError(
            "B2 full-run handoff is not ready."
        )

    if int(
        full_handoff[
            "full_run"
        ][
            "epochs"
        ]
    ) != MAX_EPOCHS:
        raise RuntimeError(
            "B2 epoch count mismatch."
        )

    if float(
        full_handoff[
            "full_run"
        ][
            "learning_rate"
        ]
    ) != LEARNING_RATE:
        raise RuntimeError(
            "B2 learning rate mismatch."
        )

    if [
        int(value)
        for value in full_handoff[
            "full_run"
        ][
            "RAW_candidate_epochs"
        ]
    ] != RAW_CANDIDATE_EPOCHS:
        raise RuntimeError(
            "B2 candidate epoch set mismatch."
        )

    frozen_b1_file_sha256_before = (
        sha256_file(
            FROZEN_B1_GENERATOR
        )
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    frozen_b1_model_state_sha256 = (
        state_dict_sha256(
            frozen_b1[
                "model_state_dict"
            ]
        )
    )

    if (
        frozen_b1_model_state_sha256
        != frozen_b1[
            "selected_model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B1 model-state hash mismatch."
        )

    protocol_snapshot = {
        "status": (
            "B2_FULL_FINETUNE_PROTOCOL_SNAPSHOT"
        ),
        "scientific_run": True,
        "frozen_B1_generator": str(
            FROZEN_B1_GENERATOR
        ),
        "frozen_B1_file_sha256": (
            frozen_b1_file_sha256_before
        ),
        "frozen_B1_model_state_sha256": (
            frozen_b1_model_state_sha256
        ),
        "B2_protocol_V1": str(
            B2_PROTOCOL_V1
        ),
        "B2_protocol_V1_sha256": (
            sha256_file(
                B2_PROTOCOL_V1
            )
        ),
        "selected_lambda_file": str(
            B2_SELECTED_LAMBDA_FILE
        ),
        "selected_lambda_file_sha256": (
            sha256_file(
                B2_SELECTED_LAMBDA_FILE
            )
        ),
        "lambda_spectral": (
            LAMBDA_SPECTRAL
        ),
        "prediction_target": "v",
        "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": (
            GRADIENT_CLIP_NORM
        ),
        "mixed_precision": (
            MIXED_PRECISION
        ),
        "EMA_enabled": False,
        "scientific_weights": "RAW",
        "predeclared_RAW_candidate_epochs": (
            RAW_CANDIDATE_EPOCHS
        ),
        "lambda_pilot_checkpoint_used": False,
        "automatic_resume": True,
        "Validation_monitoring": True,
        "Final_Test_accessed": False,
        "bulk_synthetic_generation_started": False,
    }

    if PROTOCOL_SNAPSHOT.exists():
        with PROTOCOL_SNAPSHOT.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != protocol_snapshot:
            raise RuntimeError(
                "Conflicting B2 full-run protocol snapshot exists."
            )

        print(
            "identical B2 full-run protocol snapshot already exists."
        )
    else:
        atomic_json_save(
            protocol_snapshot,
            PROTOCOL_SNAPSHOT,
        )

    print(
        f"frozen B1 SHA256: "
        f"{frozen_b1_file_sha256_before}"
    )

    print(
        f"frozen B1 model-state SHA256: "
        f"{frozen_b1_model_state_sha256}"
    )

    print(
        f"selected lambda: "
        f"{LAMBDA_SPECTRAL}"
    )

    print(
        "lambda-pilot checkpoint used: NO"
    )

    print(
        "source/protocol audit: PASS"
    )

    # ========================================================
    # Step 2 — Build Train / Validation data
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — BUILD REAL TRAIN / VALIDATION SUBJECT-INDEXED DATA"
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

    train_records, train_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    validation_records, validation_mapping = (
        build_validation_window_index(
            split
        )
    )

    if len(
        train_records
    ) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(
        validation_records
    ) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            "Unexpected Validation window count."
        )

    if len(
        train_mapping
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Unexpected subject count."
        )

    if (
        train_mapping
        != validation_mapping
    ):
        raise RuntimeError(
            "Train/Validation subject mapping mismatch."
        )

    train_base_dataset = (
        CachedRealTrainWindowDataset(
            records=train_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    validation_base_dataset = (
        CachedRealTrainWindowDataset(
            records=validation_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    train_dataset = SubjectIndexedDataset(
        base_dataset=train_base_dataset,
        records=train_records,
    )

    validation_dataset = SubjectIndexedDataset(
        base_dataset=(
            validation_base_dataset
        ),
        records=validation_records,
    )

    print(
        f"Train windows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation windows: "
        f"{len(validation_dataset)}"
    )

    print(
        f"subjects: "
        f"{len(train_mapping)}"
    )

    print(
        "Final Test EEG accessed: NO"
    )

    # ========================================================
    # Step 3 — Build/resume model and optimizer
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD OR RESUME B2 FULL RUN"
    )

    print(
        "=" * 80
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
            sampling_rate_hz=128,
            fmin_hz=SPECTRAL_FMIN_HZ,
            fmax_hz=SPECTRAL_FMAX_HZ,
            log_eps=SPECTRAL_LOG_EPS,
            huber_beta_db=(
                SPECTRAL_HUBER_BETA_DB
            ),
            n_times=512,
        )
        .to(device)
    )

    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        TRAIN_LOADER_SEED
    )

    noise_generator = torch.Generator(
        device="cuda"
    )

    noise_generator.manual_seed(
        TRAIN_NOISE_SEED
    )

    model = build_b2_model(
        frozen_b1,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=MIXED_PRECISION,
    )

    history: list[
        dict[str, Any]
    ] = []

    best_validation_monitor_epoch = 0
    best_validation_total_objective = float(
        "inf"
    )

    cumulative_training_seconds = 0.0
    total_amp_skips = 0
    start_epoch = 1
    resumed = False

    if LATEST_CHECKPOINT.exists():
        checkpoint = torch.load(
            LATEST_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint[
            "status"
        ] != "B2_FULL_FINETUNE_CHECKPOINT":
            raise RuntimeError(
                "Unexpected B2 resume checkpoint status."
            )

        if (
            checkpoint[
                "B2_protocol_snapshot"
            ]
            != protocol_snapshot
        ):
            raise RuntimeError(
                "B2 resume protocol mismatch."
            )

        if abs(
            float(
                checkpoint[
                    "lambda_spectral"
                ]
            )
            - LAMBDA_SPECTRAL
        ) > 1e-12:
            raise RuntimeError(
                "B2 resume lambda mismatch."
            )

        if checkpoint[
            "lambda_pilot_checkpoint_used"
        ] is not False:
            raise RuntimeError(
                "Scientific B2 run reports pilot checkpoint reuse."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        grad_scaler.load_state_dict(
            checkpoint[
                "grad_scaler_state_dict"
            ]
        )

        history = list(
            checkpoint["history"]
        )

        best_validation_monitor_epoch = int(
            checkpoint[
                "best_validation_monitor_epoch"
            ]
        )

        best_validation_total_objective = float(
            checkpoint[
                "best_validation_total_objective"
            ]
        )

        cumulative_training_seconds = float(
            checkpoint[
                "cumulative_training_seconds"
            ]
        )

        total_amp_skips = int(
            checkpoint.get(
                "total_amp_skips",
                0,
            )
        )

        start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        restore_rng(
            checkpoint[
                "rng_state"
            ],
            loader_generator,
            noise_generator,
        )

        resumed = True

        print(
            "RESUME PASS"
        )

        print(
            f"completed epoch: "
            f"{start_epoch - 1}"
        )

        print(
            f"next epoch: "
            f"{start_epoch}"
        )
    else:
        print(
            "Starting scientific B2 full run from frozen B1 RAW epoch 10."
        )

        print(
            "Fresh AdamW optimizer: YES"
        )

        print(
            "Lambda pilot checkpoint used: NO"
        )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
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

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    if len(train_loader) != 748:
        raise RuntimeError(
            "Expected 748 Train batches."
        )

    if len(validation_loader) != 151:
        raise RuntimeError(
            "Expected 151 Validation batches."
        )

    # ========================================================
    # Step 4 — Deterministic Validation baseline/replay
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — DETERMINISTIC RAW VALIDATION CHECK"
    )

    print(
        "=" * 80
    )

    validation_now = evaluate_validation(
        model=model,
        loader=validation_loader,
        alpha_bar=alpha_bar,
        spectral_loss_fn=(
            spectral_loss_fn
        ),
        device=device,
    )

    if resumed:
        expected = history[-1]

        replay_keys = [
            (
                "validation_correct_subject_v_mse",
                "correct_subject_v_mse",
            ),
            (
                "validation_wrong_subject_v_mse",
                "wrong_subject_v_mse",
            ),
            (
                "validation_spectral_loss_db_huber",
                "spectral_loss_db_huber",
            ),
            (
                "validation_total_objective",
                "total_objective",
            ),
        ]

        for history_key, metric_key in replay_keys:
            difference = abs(
                float(
                    expected[
                        history_key
                    ]
                )
                - float(
                    validation_now[
                        metric_key
                    ]
                )
            )

            print(
                f"{history_key} replay diff: "
                f"{difference:.3e}"
            )

            if difference > 1e-6:
                raise RuntimeError(
                    "Deterministic B2 Validation replay mismatch."
                )

        print(
            "Validation replay: PASS"
        )
    else:
        print(
            "initial frozen-B1 correct-subject Validation v-MSE: "
            f"{validation_now['correct_subject_v_mse']:.6f}"
        )

        print(
            "initial frozen-B1 Validation spectral loss: "
            f"{validation_now['spectral_loss_db_huber']:.6f}"
        )

        print(
            "initial frozen-B1 Validation total objective: "
            f"{validation_now['total_objective']:.6f}"
        )

        print(
            "initial frozen-B1 subject gap: "
            f"{validation_now['wrong_minus_correct_v_mse']:.6e}"
        )

        print(
            "This is a monitor baseline, not a selected B2 checkpoint."
        )

    # ========================================================
    # Step 5 — Run 20 epochs
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — RUN B2 20-EPOCH SPECTRAL-REGULARIZED FINE-TUNING"
    )

    print(
        "=" * 80
    )

    append_event(
        "run_started_or_resumed",
        start_epoch=start_epoch,
        max_epochs=MAX_EPOCHS,
        resumed=resumed,
        lambda_spectral=(
            LAMBDA_SPECTRAL
        ),
        frozen_B1_sha256=(
            frozen_b1_file_sha256_before
        ),
    )

    try:
        for epoch in range(
            start_epoch,
            MAX_EPOCHS + 1,
        ):
            epoch_start = (
                time.perf_counter()
            )

            torch.cuda.reset_peak_memory_stats()

            model.train()

            train_v_sum = 0.0
            train_spec_sum = 0.0
            train_total_sum = 0.0
            presented = 0

            gradient_norms = []

            amp_skips = 0
            consecutive_amp_skips = 0
            successful_optimizer_steps = 0

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

                pred_v_f = (
                    pred_v.float()
                )

                target_v_f = (
                    target_v.float()
                )

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

                spectral_loss = (
                    spectral_loss_fn(
                        x0_hat,
                        x0,
                    )
                )

                total_loss = (
                    v_loss
                    + LAMBDA_SPECTRAL
                    * spectral_loss
                )

                if not (
                    torch.isfinite(v_loss)
                    and torch.isfinite(
                        spectral_loss
                    )
                    and torch.isfinite(
                        total_loss
                    )
                ):
                    raise RuntimeError(
                        "Nonfinite B2 training loss."
                    )

                v_value = float(
                    v_loss.item()
                )

                spec_value = float(
                    spectral_loss.item()
                )

                total_value = float(
                    total_loss.item()
                )

                grad_scaler.scale(
                    total_loss
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

                scale_before = float(
                    grad_scaler.get_scale()
                )

                if finite:
                    clip_return = (
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            GRADIENT_CLIP_NORM,
                        )
                    )

                    if not torch.isfinite(
                        clip_return
                    ):
                        raise RuntimeError(
                            "Nonfinite B2 clip norm."
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
                        "AMP SKIP "
                        f"| epoch={epoch} "
                        f"batch={batch_index} "
                        f"| scale "
                        f"{scale_before:g}->"
                        f"{scale_after:g} "
                        f"| optimizer=SKIP "
                        f"| bad={bad_names[:3]}"
                    )

                    append_event(
                        "amp_step_skipped",
                        epoch=epoch,
                        batch=batch_index,
                        v_loss=v_value,
                        spectral_loss=(
                            spec_value
                        ),
                        total_loss=(
                            total_value
                        ),
                        scale_before=(
                            scale_before
                        ),
                        scale_after=(
                            scale_after
                        ),
                        bad_parameter_names=(
                            bad_names[:20]
                        ),
                    )

                    if (
                        amp_skips
                        > MAX_AMP_SKIPS_PER_EPOCH
                    ):
                        raise RuntimeError(
                            "Too many AMP skips "
                            f"in epoch {epoch}."
                        )

                    if (
                        consecutive_amp_skips
                        >= MAX_CONSECUTIVE_AMP_SKIPS
                    ):
                        raise RuntimeError(
                            "Too many consecutive "
                            "AMP skips."
                        )

                presented += (
                    current_batch_size
                )

                train_v_sum += (
                    v_value
                    * current_batch_size
                )

                train_spec_sum += (
                    spec_value
                    * current_batch_size
                )

                train_total_sum += (
                    total_value
                    * current_batch_size
                )

                if (
                    batch_index == 1
                    or batch_index
                    % PRINT_EVERY_BATCHES
                    == 0
                    or batch_index
                    == len(train_loader)
                ):
                    print(
                        f"epoch "
                        f"{epoch:>2}/"
                        f"{MAX_EPOCHS} "
                        f"| batch "
                        f"{batch_index:>3}/"
                        f"{len(train_loader)} "
                        f"| v="
                        f"{train_v_sum/presented:.6f} "
                        f"| spec="
                        f"{train_spec_sum/presented:.6f} "
                        f"| total="
                        f"{train_total_sum/presented:.6f} "
                        f"| amp_skips="
                        f"{amp_skips} "
                        f"| scale="
                        f"{grad_scaler.get_scale():g}"
                    )

            if (
                presented
                != EXPECTED_TRAIN_WINDOWS
            ):
                raise RuntimeError(
                    "B2 Train accounting mismatch."
                )

            if (
                successful_optimizer_steps
                + amp_skips
                != len(train_loader)
            ):
                raise RuntimeError(
                    "B2 optimizer-step accounting mismatch."
                )

            if not gradient_norms:
                raise RuntimeError(
                    "No successful B2 optimizer steps."
                )

            assert_finite_state_dict(
                model,
                "live B2 model",
            )

            validation_metrics = (
                evaluate_validation(
                    model=model,
                    loader=validation_loader,
                    alpha_bar=alpha_bar,
                    spectral_loss_fn=(
                        spectral_loss_fn
                    ),
                    device=device,
                )
            )

            torch.cuda.synchronize()

            epoch_seconds = (
                time.perf_counter()
                - epoch_start
            )

            cumulative_training_seconds += (
                epoch_seconds
            )

            train_v_mse = (
                train_v_sum
                / presented
            )

            train_spectral = (
                train_spec_sum
                / presented
            )

            train_total = (
                train_total_sum
                / presented
            )

            mean_gradient_norm = float(
                np.mean(
                    gradient_norms
                )
            )

            max_gradient_norm = float(
                np.max(
                    gradient_norms
                )
            )

            peak_allocated_mib = float(
                torch.cuda.max_memory_allocated()
                / 1024
                ** 2
            )

            peak_reserved_mib = float(
                torch.cuda.max_memory_reserved()
                / 1024
                ** 2
            )

            throughput = (
                presented
                / epoch_seconds
            )

            validation_total = float(
                validation_metrics[
                    "total_objective"
                ]
            )

            validation_improved = (
                validation_total
                < best_validation_total_objective
            )

            if validation_improved:
                best_validation_total_objective = (
                    validation_total
                )

                best_validation_monitor_epoch = (
                    epoch
                )

            row = {
                "epoch": int(epoch),
                "train_v_mse": float(
                    train_v_mse
                ),
                "train_spectral_loss_db_huber": float(
                    train_spectral
                ),
                "train_total_objective": float(
                    train_total
                ),
                "validation_correct_subject_v_mse": float(
                    validation_metrics[
                        "correct_subject_v_mse"
                    ]
                ),
                "validation_wrong_subject_v_mse": float(
                    validation_metrics[
                        "wrong_subject_v_mse"
                    ]
                ),
                "validation_wrong_minus_correct_v_mse": float(
                    validation_metrics[
                        "wrong_minus_correct_v_mse"
                    ]
                ),
                "validation_spectral_loss_db_huber": float(
                    validation_metrics[
                        "spectral_loss_db_huber"
                    ]
                ),
                "validation_total_objective": float(
                    validation_total
                ),
                "validation_correct_vs_wrong_output_mean_abs": float(
                    validation_metrics[
                        "correct_vs_wrong_output_mean_abs"
                    ]
                ),
                "mean_gradient_norm": (
                    mean_gradient_norm
                ),
                "max_gradient_norm": (
                    max_gradient_norm
                ),
                "amp_skipped_steps": int(
                    amp_skips
                ),
                "amp_scale_end": float(
                    grad_scaler.get_scale()
                ),
                "successful_optimizer_steps": int(
                    successful_optimizer_steps
                ),
                "epoch_seconds": float(
                    epoch_seconds
                ),
                "train_windows_per_second": float(
                    throughput
                ),
                "peak_allocated_mib": (
                    peak_allocated_mib
                ),
                "peak_reserved_mib": (
                    peak_reserved_mib
                ),
                "best_validation_monitor_epoch": int(
                    best_validation_monitor_epoch
                ),
                "best_validation_total_objective": float(
                    best_validation_total_objective
                ),
            }

            history.append(row)

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"B2 EPOCH "
                f"{epoch} COMPLETE"
            )

            print(
                f"Train v-MSE: "
                f"{train_v_mse:.6f}"
            )

            print(
                "Train spectral loss: "
                f"{train_spectral:.6f}"
            )

            print(
                "Train total objective: "
                f"{train_total:.6f}"
            )

            print(
                "Validation correct-subject v-MSE: "
                f"{validation_metrics['correct_subject_v_mse']:.6f}"
            )

            print(
                "Validation spectral loss: "
                f"{validation_metrics['spectral_loss_db_huber']:.6f}"
            )

            print(
                "Validation total objective: "
                f"{validation_total:.6f}"
            )

            print(
                "Validation wrong-correct gap: "
                f"{validation_metrics['wrong_minus_correct_v_mse']:.6e}"
            )

            print(
                "Best Validation monitor total objective: "
                f"{best_validation_total_objective:.6f} "
                f"@ epoch "
                f"{best_validation_monitor_epoch}"
            )

            print(
                f"Mean gradient norm: "
                f"{mean_gradient_norm:.4f}"
            )

            print(
                f"AMP skipped steps: "
                f"{amp_skips}"
            )

            print(
                f"GradScaler scale end: "
                f"{grad_scaler.get_scale():g}"
            )

            print(
                f"Epoch time: "
                f"{epoch_seconds/60:.2f} min"
            )

            print(
                f"Peak GPU allocated: "
                f"{peak_allocated_mib:.0f} MiB"
            )

            print(
                f"Validation monitor improved: "
                f"{validation_improved}"
            )

            print(
                "Predeclared final-candidate epoch: "
                f"{epoch in RAW_CANDIDATE_EPOCHS}"
            )

            print(
                "-" * 80
                + "\n"
            )

            checkpoint = make_checkpoint(
                epoch=epoch,
                protocol_snapshot=(
                    protocol_snapshot
                ),
                model=model,
                optimizer=optimizer,
                grad_scaler=(
                    grad_scaler
                ),
                loader_generator=(
                    loader_generator
                ),
                noise_generator=(
                    noise_generator
                ),
                history=history,
                best_validation_monitor_epoch=(
                    best_validation_monitor_epoch
                ),
                best_validation_total_objective=(
                    best_validation_total_objective
                ),
                cumulative_training_seconds=(
                    cumulative_training_seconds
                ),
                total_amp_skips=(
                    total_amp_skips
                ),
                frozen_b1_file_sha256=(
                    frozen_b1_file_sha256_before
                ),
                frozen_b1_model_state_sha256=(
                    frozen_b1_model_state_sha256
                ),
            )

            atomic_torch_save(
                checkpoint,
                LATEST_CHECKPOINT,
            )

            if (
                validation_improved
                or not BEST_VALIDATION_MONITOR_CHECKPOINT.exists()
            ):
                atomic_torch_save(
                    checkpoint,
                    BEST_VALIDATION_MONITOR_CHECKPOINT,
                )

                print(
                    "BEST VALIDATION-MONITOR CHECKPOINT UPDATED: "
                    f"{BEST_VALIDATION_MONITOR_CHECKPOINT}"
                )

                print(
                    "NOTE: this is NOT automatically the "
                    "final scientific B2 generator."
                )

            if (
                epoch
                % CHECKPOINT_EVERY_EPOCHS
                == 0
            ):
                milestone_path = (
                    CHECKPOINT_DIR
                    / f"epoch_{epoch:03d}.pt"
                )

                atomic_torch_save(
                    checkpoint,
                    milestone_path,
                )

                print(
                    f"milestone checkpoint: "
                    f"{milestone_path}"
                )

            if (
                epoch
                in RAW_CANDIDATE_EPOCHS
            ):
                candidate_path = (
                    CHECKPOINT_DIR
                    / f"candidate_epoch_{epoch:03d}_raw.pt"
                )

                atomic_torch_save(
                    checkpoint,
                    candidate_path,
                )

                print(
                    "PREDECLARED RAW SCIENTIFIC CANDIDATE SAVED: "
                    f"{candidate_path}"
                )

            save_history(history)
            save_figures(history)

            state_payload = {
                "status": (
                    "B2_FULL_FINETUNE_IN_PROGRESS"
                    if epoch < MAX_EPOCHS
                    else "B2_FULL_FINETUNE_20_EPOCHS_COMPLETE"
                ),
                "completed_epoch": (
                    epoch
                ),
                "next_epoch": (
                    epoch + 1
                    if epoch < MAX_EPOCHS
                    else None
                ),
                "lambda_spectral": (
                    LAMBDA_SPECTRAL
                ),
                "best_validation_monitor_epoch": (
                    best_validation_monitor_epoch
                ),
                "best_validation_total_objective": (
                    best_validation_total_objective
                ),
                "total_amp_skips": (
                    total_amp_skips
                ),
                "cumulative_training_minutes": (
                    cumulative_training_seconds
                    / 60.0
                ),
                "final_scientific_B2_selected": False,
                "predeclared_RAW_candidate_epochs": (
                    RAW_CANDIDATE_EPOCHS
                ),
                "Validation_accessed": True,
                "Final_Test_accessed": False,
                "bulk_synthetic_generation_started": False,
            }

            atomic_json_save(
                state_payload,
                STATE_FILE,
            )

            append_event(
                "epoch_complete",
                epoch=epoch,
                train_v_mse=(
                    train_v_mse
                ),
                train_spectral_loss=(
                    train_spectral
                ),
                validation_v_mse=(
                    validation_metrics[
                        "correct_subject_v_mse"
                    ]
                ),
                validation_spectral_loss=(
                    validation_metrics[
                        "spectral_loss_db_huber"
                    ]
                ),
                validation_total_objective=(
                    validation_total
                ),
                validation_subject_gap=(
                    validation_metrics[
                        "wrong_minus_correct_v_mse"
                    ]
                ),
                amp_skips=(
                    amp_skips
                ),
                candidate_epoch=(
                    epoch
                    in RAW_CANDIDATE_EPOCHS
                ),
            )

    except KeyboardInterrupt:
        print(
            "\nKeyboardInterrupt received."
        )

        print(
            "The last COMPLETED epoch is safely stored "
            "in latest.pt."
        )

        print(
            "Run the same command again to resume "
            "from the next epoch."
        )

        append_event(
            "keyboard_interrupt",
            completed_epoch=(
                history[-1]["epoch"]
                if history
                else 0
            ),
        )

        raise

    # ========================================================
    # Step 6 — Finalize 20-epoch run
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — FINALIZE B2 20-EPOCH FULL FINE-TUNING"
    )

    print(
        "=" * 80
    )

    if len(history) != MAX_EPOCHS:
        raise RuntimeError(
            "B2 full-run history does not contain "
            f"{MAX_EPOCHS} completed epochs."
        )

    latest = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if int(
        latest["epoch"]
    ) != MAX_EPOCHS:
        raise RuntimeError(
            "Latest B2 checkpoint is not epoch 20."
        )

    atomic_torch_save(
        latest,
        FINAL_CHECKPOINT,
    )

    frozen_b1_file_sha256_after = (
        sha256_file(
            FROZEN_B1_GENERATOR
        )
    )

    if (
        frozen_b1_file_sha256_after
        != frozen_b1_file_sha256_before
    ):
        raise RuntimeError(
            "Frozen B1 changed during B2 full run."
        )

    candidate_records = []

    for epoch in RAW_CANDIDATE_EPOCHS:
        path = (
            CHECKPOINT_DIR
            / f"candidate_epoch_{epoch:03d}_raw.pt"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        row = checkpoint[
            "history"
        ][
            -1
        ]

        candidate_records.append(
            {
                "epoch": epoch,
                "path": str(path),
                "file_sha256": (
                    sha256_file(path)
                ),
                "model_state_sha256": (
                    checkpoint[
                        "model_state_sha256"
                    ]
                ),
                "train_v_mse": float(
                    row["train_v_mse"]
                ),
                "train_spectral_loss_db_huber": float(
                    row[
                        "train_spectral_loss_db_huber"
                    ]
                ),
                "validation_correct_subject_v_mse": float(
                    row[
                        "validation_correct_subject_v_mse"
                    ]
                ),
                "validation_spectral_loss_db_huber": float(
                    row[
                        "validation_spectral_loss_db_huber"
                    ]
                ),
                "validation_total_objective": float(
                    row[
                        "validation_total_objective"
                    ]
                ),
                "validation_wrong_minus_correct_v_mse": float(
                    row[
                        "validation_wrong_minus_correct_v_mse"
                    ]
                ),
            }
        )

    final_row = history[-1]

    result = {
        "status": (
            "B2_FULL_FINETUNE_20_EPOCHS_COMPLETE"
        ),
        "completed_epochs": (
            MAX_EPOCHS
        ),
        "initialization": (
            "frozen B1 RAW epoch 10"
        ),
        "lambda_pilot_checkpoint_used": False,
        "lambda_spectral": (
            LAMBDA_SPECTRAL
        ),
        "prediction_target": "v",
        "EMA_enabled": False,
        "scientific_checkpoint_weights": "RAW",
        "final_train_v_mse": float(
            final_row[
                "train_v_mse"
            ]
        ),
        "final_train_spectral_loss_db_huber": float(
            final_row[
                "train_spectral_loss_db_huber"
            ]
        ),
        "final_validation_correct_subject_v_mse": float(
            final_row[
                "validation_correct_subject_v_mse"
            ]
        ),
        "final_validation_spectral_loss_db_huber": float(
            final_row[
                "validation_spectral_loss_db_huber"
            ]
        ),
        "final_validation_total_objective": float(
            final_row[
                "validation_total_objective"
            ]
        ),
        "final_validation_wrong_minus_correct_v_mse": float(
            final_row[
                "validation_wrong_minus_correct_v_mse"
            ]
        ),
        "best_validation_monitor_epoch": (
            best_validation_monitor_epoch
        ),
        "best_validation_total_objective": (
            best_validation_total_objective
        ),
        "total_amp_skips": (
            total_amp_skips
        ),
        "training_seconds": (
            cumulative_training_seconds
        ),
        "training_minutes": (
            cumulative_training_seconds
            / 60.0
        ),
        "predeclared_RAW_candidates": (
            candidate_records
        ),
        "final_scientific_B2_selected": False,
        "next_selection_rule": (
            "Compare RAW epochs 5,10,20 against frozen B1 using "
            "Train-only generic + subject-aware + spectral "
            "generative-fidelity metrics before freezing B2."
        ),
        "matched_lambda0_control_required_before_causal_claim": True,
        "source_immutability": {
            "frozen_B1_unchanged": True,
        },
        "Validation_accessed": True,
        "Final_Test_accessed": False,
        "bulk_synthetic_generation_started": False,
        "final_checkpoint": str(
            FINAL_CHECKPOINT
        ),
        "best_validation_monitor_checkpoint": str(
            BEST_VALIDATION_MONITOR_CHECKPOINT
        ),
        "history_csv": str(
            HISTORY_FILE
        ),
        "next": (
            "B2 RAW epoch 5/10/20 subject-aware + spectral "
            "generative fidelity sweep against frozen B1."
        ),
    }

    atomic_json_save(
        result,
        RESULT_FILE,
    )

    append_event(
        "run_complete",
        completed_epochs=MAX_EPOCHS,
        lambda_spectral=(
            LAMBDA_SPECTRAL
        ),
        best_validation_monitor_epoch=(
            best_validation_monitor_epoch
        ),
        best_validation_total_objective=(
            best_validation_total_objective
        ),
        total_amp_skips=(
            total_amp_skips
        ),
        training_seconds=(
            cumulative_training_seconds
        ),
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B2 20-EPOCH FULL FINE-TUNING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"lambda_spectral: "
        f"{LAMBDA_SPECTRAL}"
    )

    print(
        f"completed epochs: "
        f"{MAX_EPOCHS}"
    )

    print(
        f"final Train v-MSE: "
        f"{final_row['train_v_mse']:.6f}"
    )

    print(
        f"final Train spectral loss: "
        f"{final_row['train_spectral_loss_db_huber']:.6f}"
    )

    print(
        "final Validation correct-subject v-MSE: "
        f"{final_row['validation_correct_subject_v_mse']:.6f}"
    )

    print(
        "final Validation spectral loss: "
        f"{final_row['validation_spectral_loss_db_huber']:.6f}"
    )

    print(
        "final Validation total objective: "
        f"{final_row['validation_total_objective']:.6f}"
    )

    print(
        "final Validation wrong-correct gap: "
        f"{final_row['validation_wrong_minus_correct_v_mse']:.6e}"
    )

    print(
        "best Validation monitor epoch: "
        f"{best_validation_monitor_epoch}"
    )

    print(
        "best Validation total objective: "
        f"{best_validation_total_objective:.6f}"
    )

    print(
        f"total AMP skipped steps: "
        f"{total_amp_skips}"
    )

    print(
        f"training time: "
        f"{cumulative_training_seconds/60:.2f} min"
    )

    print(
        "frozen B1 unchanged: YES"
    )

    print(
        "lambda-pilot checkpoint used: NO"
    )

    print(
        "Validation accessed: YES"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "bulk synthetic generation started: NO"
    )

    print(
        "\nPredeclared RAW scientific candidates:"
    )

    for candidate in candidate_records:
        print(
            f"  epoch "
            f"{candidate['epoch']:>2} "
            f"| Train-v="
            f"{candidate['train_v_mse']:.6f} "
            f"| Train-spec="
            f"{candidate['train_spectral_loss_db_huber']:.4f} "
            f"| Val-v="
            f"{candidate['validation_correct_subject_v_mse']:.6f} "
            f"| Val-spec="
            f"{candidate['validation_spectral_loss_db_huber']:.4f} "
            f"| gap="
            f"{candidate['validation_wrong_minus_correct_v_mse']:.3e}"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  best_validation_monitor.pt is only a monitor."
    )

    print(
        "  It is NOT automatically the final B2 generator."
    )

    print(
        "  Final B2 selection must compare RAW epochs 5/10/20 "
        "on Train-only generative fidelity and subject fidelity."
    )

    print(
        "\nMATCHED CONTROL:"
    )

    print(
        "  A full lambda=0 continuation remains required before "
        "making a causal publication claim that spectral loss "
        "itself caused an improvement."
    )

    print(
        f"\nresult JSON:"
        f"\n{RESULT_FILE}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "B2 RAW EPOCH 5/10/20 SUBJECT-AWARE + "
        "SPECTRAL GENERATIVE FIDELITY SWEEP"
    )


if __name__ == "__main__":
    main()
