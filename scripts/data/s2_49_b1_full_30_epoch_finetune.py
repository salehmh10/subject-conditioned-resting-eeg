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


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

FROZEN_B0_ROOT = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b0"
    / "b0_v_raw_epoch020_v1"
)

FROZEN_B0_GENERATOR = (
    FROZEN_B0_ROOT
    / "checkpoints"
    / "b0_v_raw_epoch020_generator_v1.pt"
)

B1_PROTOCOL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b1"
    / "subject_state_v1"
)

B1_PROTOCOL_FILE = (
    B1_PROTOCOL_ROOT
    / "metrics"
    / "b1_subject_state_finetune_protocol_v1.json"
)

ARCH_SANITY_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "architecture_sanity_v1"
)

WARM_START_TEMPLATE = (
    ARCH_SANITY_ROOT
    / "checkpoints"
    / "b1_warm_start_template_v1.pt"
)

ARCH_SANITY_SUMMARY = (
    ARCH_SANITY_ROOT
    / "metrics"
    / "b1_architecture_warmstart_sanity_v1.json"
)

PILOT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "short_pilot_v1"
)

PILOT_CHECKPOINT = (
    PILOT_ROOT
    / "checkpoints"
    / "b1_short_pilot_final_v1.pt"
)

PILOT_SUMMARY = (
    PILOT_ROOT
    / "metrics"
    / "b1_short_pilot_summary_v1.json"
)


# ============================================================
# Full B1 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "full_finetune_v1"
)

CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
LOG_DIR = RUN_ROOT / "logs"
METRIC_DIR = RUN_ROOT / "metrics"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_SNAPSHOT = (
    METRIC_DIR
    / "b1_full_finetune_protocol_snapshot_v1.json"
)

STATE_FILE = (
    METRIC_DIR
    / "b1_full_finetune_state_v1.json"
)

RESULT_FILE = (
    METRIC_DIR
    / "b1_full_finetune_result_v1.json"
)

HISTORY_FILE = (
    LOG_DIR
    / "b1_full_finetune_history_v1.csv"
)

EVENT_FILE = (
    LOG_DIR
    / "b1_full_finetune_events_v1.jsonl"
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
    / "b1_full_finetune_loss_v1.png"
)

SUBJECT_GAP_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_validation_subject_gap_v1.png"
)

SUBJECT_OUTPUT_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_validation_subject_output_difference_v1.png"
)

GRADIENT_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_gradient_norm_v1.png"
)

SUBJECT_GRADIENT_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_subject_gradient_norm_v1.png"
)

AMP_SCALE_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_amp_scale_v1.png"
)

AMP_SKIP_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_amp_skips_v1.png"
)

THROUGHPUT_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_throughput_v1.png"
)

MEMORY_FIGURE = (
    FIGURE_DIR
    / "b1_full_finetune_gpu_memory_v1.png"
)


# ============================================================
# Frozen B1 full-run protocol expectations
# ============================================================

SEED = 20260908
TRAIN_LOADER_SEED = 20260928
TRAIN_NOISE_SEED = 20260929
VALIDATION_SEED = 20260930

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_VALIDATION_WINDOWS = 9_635
EXPECTED_SUBJECTS = 198

N_CHANNELS = 19
N_TIMES = 512

TIMESTEPS = 1000
COSINE_S = 0.008

MAX_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

MIXED_PRECISION = True
EMA_ENABLED = False

CHECKPOINT_EVERY_EPOCHS = 5
PREDECLARED_RAW_CANDIDATE_EPOCHS = [
    10,
    20,
    30,
]

PRINT_EVERY_BATCHES = 100

MAX_AMP_SKIPS_PER_EPOCH = 5
MAX_CONSECUTIVE_AMP_SKIPS = 3

WRONG_SUBJECT_OFFSET = 97


# ============================================================
# Reproducibility / IO
# ============================================================

def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    data: dict[str, Any],
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
            data,
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
    data: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(
        str(path)
        + ".tmp"
    )

    torch.save(
        data,
        temp,
    )

    os.replace(
        temp,
        path,
    )


def append_event(
    kind: str,
    **fields,
) -> None:
    payload = {
        "time_unix": (
            time.time()
        ),
        "kind": (
            kind
        ),
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


def assert_finite_state_dict(
    model: nn.Module,
    label: str,
) -> None:
    for name, tensor in model.state_dict().items():
        if (
            torch.is_tensor(
                tensor
            )
            and torch.is_floating_point(
                tensor
            )
            and not torch.isfinite(
                tensor
            ).all()
        ):
            raise RuntimeError(
                f"Nonfinite tensor in {label}: "
                f"{name}"
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
    bad_names: list[str] = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        grad_count += 1
        grad = parameter.grad.detach()

        if not torch.isfinite(
            grad
        ).all():
            bad_names.append(
                name
            )
            continue

        squared_sum += float(
            torch.sum(
                grad.float()
                * grad.float()
            ).item()
        )

    finite = (
        len(
            bad_names
        )
        == 0
    )

    norm = (
        math.sqrt(
            squared_sum
        )
        if finite
        else float(
            "nan"
        )
    )

    return (
        finite,
        grad_count,
        norm,
        bad_names,
    )


def group_gradient_norm(
    parameters,
) -> float:
    squared_sum = 0.0
    any_grad = False

    for parameter in parameters:
        if parameter.grad is None:
            continue

        any_grad = True
        grad = parameter.grad.detach()

        if not torch.isfinite(
            grad
        ).all():
            return float(
                "nan"
            )

        squared_sum += float(
            torch.sum(
                grad.float()
                * grad.float()
            ).item()
        )

    if not any_grad:
        return 0.0

    return math.sqrt(
        squared_sum
    )


def capture_rng(
    loader_generator: torch.Generator,
    noise_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": (
            random.getstate()
        ),
        "numpy": (
            np.random.get_state()
        ),
        "torch_cpu": (
            torch.get_rng_state()
        ),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
        ),
        "loader_generator": (
            loader_generator.get_state()
        ),
        "noise_generator": (
            noise_generator.get_state()
        ),
    }


def restore_rng(
    state: dict[str, Any],
    loader_generator: torch.Generator,
    noise_generator: torch.Generator,
) -> None:
    random.setstate(
        state[
            "python"
        ]
    )

    np.random.set_state(
        state[
            "numpy"
        ]
    )

    torch.set_rng_state(
        state[
            "torch_cpu"
        ]
    )

    torch.cuda.set_rng_state_all(
        state[
            "torch_cuda"
        ]
    )

    loader_generator.set_state(
        state[
            "loader_generator"
        ]
    )

    noise_generator.set_state(
        state[
            "noise_generator"
        ]
    )


# ============================================================
# Deterministic Validation
# ============================================================

@torch.inference_mode()
def evaluate_validation(
    *,
    model: B1SubjectStateUNet1D,
    loader: DataLoader,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """
    Deterministic correct-vs-wrong subject Validation diagnostic.

    Every epoch reuses the exact same:
      - Validation windows
      - timestep draws
      - epsilon draws
      - wrong-subject mapping

    The primary monitor is correct_subject_v_mse.

    IMPORTANT:
    This Validation monitor does NOT define the final scientific
    B1 generator checkpoint by itself. Final B1 generator selection
    is deferred to the predeclared RAW milestone fidelity comparison
    at epochs 10, 20, and 30.
    """

    model.eval()

    generator = torch.Generator(
        device="cuda"
    )

    generator.manual_seed(
        VALIDATION_SEED
    )

    correct_loss_sum = 0.0
    wrong_loss_sum = 0.0
    output_difference_sum = 0.0
    total_examples = 0

    for batch in loader:
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
            x0.shape[
                0
            ]
        )

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                batch_size,
            ),
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
            t,
            alpha_bar,
        )

        target_v = make_v_target(
            x0,
            epsilon,
            t,
            alpha_bar,
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=MIXED_PRECISION,
        ):
            correct_pred = model(
                xt,
                t,
                state,
                subject,
            )

            wrong_pred = model(
                xt,
                t,
                state,
                wrong_subject,
            )

        correct_loss = torch.mean(
            (
                correct_pred.float()
                - target_v.float()
            )
            ** 2
        )

        wrong_loss = torch.mean(
            (
                wrong_pred.float()
                - target_v.float()
            )
            ** 2
        )

        output_difference = torch.mean(
            torch.abs(
                correct_pred.float()
                - wrong_pred.float()
            )
        )

        if not (
            torch.isfinite(
                correct_loss
            )
            and torch.isfinite(
                wrong_loss
            )
            and torch.isfinite(
                output_difference
            )
        ):
            raise RuntimeError(
                "Nonfinite deterministic B1 Validation metric."
            )

        correct_loss_sum += (
            float(
                correct_loss.item()
            )
            * batch_size
        )

        wrong_loss_sum += (
            float(
                wrong_loss.item()
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
            "Validation accounting mismatch: "
            f"{total_examples} != "
            f"{EXPECTED_VALIDATION_WINDOWS}"
        )

    correct_v_mse = (
        correct_loss_sum
        / total_examples
    )

    wrong_v_mse = (
        wrong_loss_sum
        / total_examples
    )

    output_difference_mean_abs = (
        output_difference_sum
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
        "correct_vs_wrong_output_mean_abs": float(
            output_difference_mean_abs
        ),
    }


# ============================================================
# History / figures
# ============================================================

HISTORY_FIELDS = [
    "epoch",
    "train_v_mse",
    "validation_correct_subject_v_mse",
    "validation_wrong_subject_v_mse",
    "validation_wrong_minus_correct_v_mse",
    "validation_correct_vs_wrong_output_mean_abs",
    "mean_gradient_norm",
    "max_gradient_norm",
    "mean_subject_embedding_gradient_norm",
    "mean_subject_projection_gradient_norm",
    "amp_skipped_steps",
    "amp_scale_end",
    "successful_optimizer_steps",
    "epoch_seconds",
    "train_windows_per_second",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "best_validation_monitor_epoch",
    "best_validation_correct_subject_v_mse",
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
            row[
                "epoch"
            ]
            for row in history
        ],
        dtype=np.int64,
    )

    train_loss = np.asarray(
        [
            row[
                "train_v_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    validation_correct = np.asarray(
        [
            row[
                "validation_correct_subject_v_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    validation_wrong = np.asarray(
        [
            row[
                "validation_wrong_subject_v_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    validation_gap = np.asarray(
        [
            row[
                "validation_wrong_minus_correct_v_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    output_difference = np.asarray(
        [
            row[
                "validation_correct_vs_wrong_output_mean_abs"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    gradient_norm = np.asarray(
        [
            row[
                "mean_gradient_norm"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    subject_embedding_gradient = np.asarray(
        [
            row[
                "mean_subject_embedding_gradient_norm"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    subject_projection_gradient = np.asarray(
        [
            row[
                "mean_subject_projection_gradient_norm"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    amp_scale = np.asarray(
        [
            row[
                "amp_scale_end"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    amp_skips = np.asarray(
        [
            row[
                "amp_skipped_steps"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    throughput = np.asarray(
        [
            row[
                "train_windows_per_second"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    memory = np.asarray(
        [
            row[
                "peak_allocated_mib"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.plot(
        epochs,
        train_loss,
        label="Train v-MSE",
    )

    ax.plot(
        epochs,
        validation_correct,
        label="Validation correct-subject v-MSE",
    )

    ax.plot(
        epochs,
        validation_wrong,
        label="Validation wrong-subject v-MSE",
        alpha=0.8,
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "v-MSE"
    )

    ax.set_title(
        "B1 Full Fine-Tuning — Train / Validation v-MSE"
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

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        epochs,
        validation_gap,
        marker="o",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Wrong-subject MSE - Correct-subject MSE"
    )

    ax.set_title(
        "B1 Validation — Subject-Specific Denoising Advantage"
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

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        epochs,
        output_difference,
        marker="o",
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Mean |correct-subject output - wrong-subject output|"
    )

    ax.set_title(
        "B1 Validation — Subject Conditioning Output Contrast"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        SUBJECT_OUTPUT_FIGURE,
        dpi=170,
    )

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        epochs,
        gradient_norm,
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Mean global gradient norm"
    )

    ax.set_title(
        "B1 Full Fine-Tuning — Gradient Norm"
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

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        epochs,
        subject_embedding_gradient,
        label="Subject embedding",
    )

    ax.plot(
        epochs,
        subject_projection_gradient,
        label="Subject projection",
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Mean gradient norm"
    )

    ax.set_title(
        "B1 Full Fine-Tuning — Subject Branch Gradients"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SUBJECT_GRADIENT_FIGURE,
        dpi=170,
    )

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        epochs,
        amp_scale,
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "GradScaler scale"
    )

    ax.set_title(
        "B1 Full Fine-Tuning — AMP Scale"
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

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.bar(
        epochs,
        amp_skips,
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Skipped optimizer steps"
    )

    ax.set_title(
        "B1 Full Fine-Tuning — AMP Skips"
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

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        epochs,
        throughput,
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Train windows / second"
    )

    ax.set_title(
        "B1 Full Fine-Tuning — Throughput"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        THROUGHPUT_FIGURE,
        dpi=170,
    )

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        epochs,
        memory,
    )

    ax.set_xlabel(
        "B1 fine-tuning epoch"
    )

    ax.set_ylabel(
        "Peak allocated GPU memory (MiB)"
    )

    ax.set_title(
        "B1 Full Fine-Tuning — GPU Memory"
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

    plt.close(
        fig
    )


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
    best_validation_correct_subject_v_mse: float,
    cumulative_training_seconds: float,
    total_amp_skips: int,
    template_file_sha256: str,
    template_model_state_sha256: str,
    frozen_B0_file_sha256: str,
) -> dict[str, Any]:
    return {
        "status": (
            "B1_FULL_FINETUNE_CHECKPOINT"
        ),
        "epoch": int(
            epoch
        ),
        "prediction_target": (
            "v"
        ),
        "selected_weight_kind": (
            "RAW"
        ),
        "EMA_enabled": False,
        "B1_protocol_snapshot": (
            protocol_snapshot
        ),
        "zero_update_template": str(
            WARM_START_TEMPLATE
        ),
        "zero_update_template_file_sha256": (
            template_file_sha256
        ),
        "zero_update_template_model_state_sha256": (
            template_model_state_sha256
        ),
        "pilot_checkpoint_used": False,
        "pilot_checkpoint_path_record_only": str(
            PILOT_CHECKPOINT
        ),
        "frozen_B0": str(
            FROZEN_B0_GENERATOR
        ),
        "frozen_B0_file_sha256": (
            frozen_B0_file_sha256
        ),
        "condition_injection_module": (
            protocol_snapshot[
                "condition_injection_module"
            ]
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "model_state_sha256": (
            state_dict_sha256(
                model.state_dict()
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
        "history": (
            history
        ),
        "best_validation_monitor_epoch": int(
            best_validation_monitor_epoch
        ),
        "best_validation_correct_subject_v_mse": float(
            best_validation_correct_subject_v_mse
        ),
        "cumulative_training_seconds": float(
            cumulative_training_seconds
        ),
        "total_amp_skips": int(
            total_amp_skips
        ),
        "validation_accessed": True,
        "final_test_accessed": False,
        "bulk_synthetic_generation_started": False,
        "final_scientific_B1_selected": False,
        "note": (
            "This checkpoint belongs to the 30-epoch RAW B1 full run. "
            "Final scientific B1 generator selection is deferred to "
            "the predeclared RAW milestone fidelity comparison at "
            "epochs 10, 20, and 30."
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
        "SCENARIO 2 — B1 30-EPOCH SUBJECT-CONDITIONED FULL FINE-TUNING"
    )

    print(
        "=" * 80
    )

    print(
        "\nCurrent project stage:"
    )

    print(
        "  B0 final generator: FROZEN"
    )

    print(
        "  B1 protocol: FROZEN"
    )

    print(
        "  B1 architecture sanity: PASSED"
    )

    print(
        "  B1 1000-step engineering pilot: PASSED"
    )

    print(
        "  B1 scientific 30-epoch full fine-tuning: NOW"
    )

    print(
        "  B1 final generative fidelity selection: NOT YET"
    )

    print(
        "  B2: NOT STARTED"
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
        "  initialization: ZERO-UPDATE B1 warm-start template"
    )

    print(
        "  pilot checkpoint used: NO"
    )

    print(
        "  prediction target: v"
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
        f"  AdamW LR: "
        f"{LEARNING_RATE}"
    )

    print(
        "  EMA: OFF"
    )

    print(
        "  scientific checkpoint weights: RAW"
    )

    print(
        "  predeclared final-candidate epochs: "
        f"{PREDECLARED_RAW_CANDIDATE_EPOCHS}"
    )

    print(
        "  deterministic Validation monitor: correct-vs-wrong subject v-MSE"
    )

    print(
        "  Final Test: SEALED"
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

    set_seed(
        SEED
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for B1 full fine-tuning."
        )

    device = torch.device(
        "cuda"
    )

    # ========================================================
    # Step 1 — Audit frozen protocol / sources
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B1 PROTOCOL + ZERO-UPDATE SOURCE"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B0_GENERATOR,
        B1_PROTOCOL_FILE,
        WARM_START_TEMPLATE,
        ARCH_SANITY_SUMMARY,
        PILOT_SUMMARY,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B1_PROTOCOL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        frozen_protocol = json.load(
            f
        )

    with ARCH_SANITY_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        architecture_summary = json.load(
            f
        )

    with PILOT_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        pilot_summary = json.load(
            f
        )

    if frozen_protocol[
        "status"
    ] != "B1_SUBJECT_STATE_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B1 protocol is not frozen."
        )

    if int(
        frozen_protocol[
            "full_fine_tuning"
        ][
            "epochs"
        ]
    ) != MAX_EPOCHS:
        raise RuntimeError(
            "Full B1 epoch count conflicts with frozen protocol."
        )

    if frozen_protocol[
        "full_fine_tuning"
    ][
        "ema_enabled"
    ] is not False:
        raise RuntimeError(
            "Frozen B1 protocol unexpectedly enables EMA."
        )

    if frozen_protocol[
        "full_fine_tuning"
    ][
        "selection_weights"
    ] != "RAW":
        raise RuntimeError(
            "Frozen B1 protocol does not select RAW weights."
        )

    frozen_candidate_epochs = [
        int(
            value
        )
        for value
        in frozen_protocol[
            "checkpoint_selection"
        ][
            "predeclared_RAW_candidate_epochs"
        ]
    ]

    if (
        frozen_candidate_epochs
        != PREDECLARED_RAW_CANDIDATE_EPOCHS
    ):
        raise RuntimeError(
            "Predeclared RAW candidate epochs mismatch."
        )

    if architecture_summary[
        "status"
    ] != "B1_ARCHITECTURE_WARMSTART_SANITY_PASS":
        raise RuntimeError(
            "B1 architecture sanity did not pass."
        )

    if pilot_summary[
        "status"
    ] != "B1_SHORT_PILOT_COMPLETE":
        raise RuntimeError(
            "B1 engineering pilot did not complete."
        )

    if pilot_summary[
        "governance"
    ][
        "pilot_checkpoint_reusable_for_scientific_full_run"
    ] is not False:
        raise RuntimeError(
            "Pilot governance unexpectedly allows pilot checkpoint reuse."
        )

    template_file_sha256_before = (
        sha256_file(
            WARM_START_TEMPLATE
        )
    )

    frozen_B0_file_sha256_before = (
        sha256_file(
            FROZEN_B0_GENERATOR
        )
    )

    template = torch.load(
        WARM_START_TEMPLATE,
        map_location="cpu",
        weights_only=False,
    )

    if template[
        "status"
    ] != "B1_ZERO_UPDATE_WARM_START_TEMPLATE":
        raise RuntimeError(
            "Unexpected B1 warm-start template status."
        )

    if int(
        template[
            "training_updates_executed"
        ]
    ) != 0:
        raise RuntimeError(
            "Scientific B1 full run must start from zero updates."
        )

    template_model_state_sha256 = (
        state_dict_sha256(
            template[
                "model_state_dict"
            ]
        )
    )

    if (
        template_model_state_sha256
        != template[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "B1 warm-start template tensor hash mismatch."
        )

    condition_injection_module = str(
        template[
            "condition_injection_module"
        ]
    )

    protocol_snapshot = {
        "status": (
            "B1_FULL_FINETUNE_PROTOCOL_SNAPSHOT"
        ),
        "scientific_run": True,
        "frozen_protocol": str(
            B1_PROTOCOL_FILE
        ),
        "frozen_protocol_sha256": (
            sha256_file(
                B1_PROTOCOL_FILE
            )
        ),
        "zero_update_template": str(
            WARM_START_TEMPLATE
        ),
        "zero_update_template_file_sha256": (
            template_file_sha256_before
        ),
        "zero_update_template_model_state_sha256": (
            template_model_state_sha256
        ),
        "frozen_B0": str(
            FROZEN_B0_GENERATOR
        ),
        "frozen_B0_file_sha256": (
            frozen_B0_file_sha256_before
        ),
        "engineering_pilot_checkpoint_used": False,
        "engineering_pilot_checkpoint_record_only": str(
            PILOT_CHECKPOINT
        ),
        "condition_injection_module": (
            condition_injection_module
        ),
        "prediction_target": (
            "v"
        ),
        "epochs": (
            MAX_EPOCHS
        ),
        "early_stopping": False,
        "batch_size": (
            BATCH_SIZE
        ),
        "optimizer": (
            "AdamW"
        ),
        "optimizer_state_reused": False,
        "learning_rate": (
            LEARNING_RATE
        ),
        "learning_rate_schedule": (
            "constant"
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
        "EMA_enabled": False,
        "scientific_checkpoint_weights": (
            "RAW"
        ),
        "checkpoint_every_epochs": (
            CHECKPOINT_EVERY_EPOCHS
        ),
        "predeclared_RAW_candidate_epochs": (
            PREDECLARED_RAW_CANDIDATE_EPOCHS
        ),
        "validation_monitor": (
            "deterministic correct-subject and wrong-subject RAW v-MSE"
        ),
        "validation_seed": (
            VALIDATION_SEED
        ),
        "wrong_subject_offset": (
            WRONG_SUBJECT_OFFSET
        ),
        "automatic_resume": True,
        "seed": (
            SEED
        ),
        "final_scientific_selection": (
            "deferred to RAW milestone generative-fidelity audit"
        ),
        "Final_Test_accessed": False,
        "bulk_synthetic_generation_started": False,
    }

    if PROTOCOL_SNAPSHOT.exists():
        with PROTOCOL_SNAPSHOT.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_snapshot = json.load(
                f
            )

        if (
            existing_snapshot
            != protocol_snapshot
        ):
            raise RuntimeError(
                "Conflicting B1 full-run protocol snapshot already exists."
            )

        print(
            "identical B1 full-run protocol snapshot already exists."
        )
    else:
        atomic_json_save(
            protocol_snapshot,
            PROTOCOL_SNAPSHOT,
        )

        print(
            f"protocol snapshot: "
            f"{PROTOCOL_SNAPSHOT}"
        )

    print(
        f"zero-update template SHA256: "
        f"{template_file_sha256_before}"
    )

    print(
        f"zero-update model-state SHA256: "
        f"{template_model_state_sha256}"
    )

    print(
        f"frozen B0 SHA256: "
        f"{frozen_B0_file_sha256_before}"
    )

    print(
        "pilot checkpoint loaded: NO"
    )

    print(
        "source/protocol audit: PASS"
    )

    # ========================================================
    # Step 2 — Build Train / Validation subject-indexed data
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
            f"Unexpected Train windows: "
            f"{len(train_records)}"
        )

    if len(
        validation_records
    ) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            f"Unexpected Validation windows: "
            f"{len(validation_records)}"
        )

    if len(
        train_mapping
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Unexpected subject count: "
            f"{len(train_mapping)}"
        )

    if (
        train_mapping
        != validation_mapping
    ):
        raise RuntimeError(
            "Train / Validation subject mapping mismatch."
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
        base_dataset=(
            train_base_dataset
        ),
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
    # Step 3 — Build or resume B1
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD OR RESUME B1 FULL RUN"
    )

    print(
        "=" * 80
    )

    frozen_B0 = torch.load(
        FROZEN_B0_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    b0_contract = frozen_B0[
        "effective_contract"
    ]

    schedule_cpu = build_cosine_schedule(
        TIMESTEPS,
        COSINE_S,
    )

    alpha_bar = schedule_cpu[
        "alpha_cumprod"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        TRAIN_LOADER_SEED
    )

    noise_generator = (
        torch.Generator(
            device="cuda"
        )
    )

    noise_generator.manual_seed(
        TRAIN_NOISE_SEED
    )

    model = B1SubjectStateUNet1D(
        b0_contract=b0_contract,
        injection_module_name=(
            condition_injection_module
        ),
        num_subjects=(
            EXPECTED_SUBJECTS
        ),
        subject_embedding_dim=int(
            frozen_protocol[
                "conditioning"
            ][
                "subject"
            ][
                "embedding_dim"
            ]
        ),
        condition_dim=int(
            frozen_protocol[
                "conditioning"
            ][
                "subject"
            ][
                "projection_output_dim"
            ]
        ),
    ).to(
        device
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

    loss_fn = nn.MSELoss()

    history: list[
        dict[str, Any]
    ] = []

    best_validation_monitor_epoch = 0

    best_validation_correct_subject_v_mse = float(
        "inf"
    )

    cumulative_training_seconds = 0.0

    total_amp_skips = 0

    start_epoch = 1

    resumed = False

    if LATEST_CHECKPOINT.exists():
        print(
            f"resume checkpoint found: "
            f"{LATEST_CHECKPOINT}"
        )

        checkpoint = torch.load(
            LATEST_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint[
            "status"
        ] != "B1_FULL_FINETUNE_CHECKPOINT":
            raise RuntimeError(
                "Unexpected B1 resume checkpoint status."
            )

        if (
            checkpoint[
                "B1_protocol_snapshot"
            ]
            != protocol_snapshot
        ):
            raise RuntimeError(
                "B1 resume checkpoint protocol mismatch."
            )

        if (
            checkpoint[
                "zero_update_template_file_sha256"
            ]
            != template_file_sha256_before
        ):
            raise RuntimeError(
                "B1 resume template hash mismatch."
            )

        if (
            checkpoint[
                "frozen_B0_file_sha256"
            ]
            != frozen_B0_file_sha256_before
        ):
            raise RuntimeError(
                "B1 resume frozen-B0 hash mismatch."
            )

        if checkpoint[
            "pilot_checkpoint_used"
        ] is not False:
            raise RuntimeError(
                "B1 scientific resume reports pilot checkpoint use."
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
            checkpoint[
                "history"
            ]
        )

        best_validation_monitor_epoch = int(
            checkpoint[
                "best_validation_monitor_epoch"
            ]
        )

        best_validation_correct_subject_v_mse = float(
            checkpoint[
                "best_validation_correct_subject_v_mse"
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
            checkpoint[
                "epoch"
            ]
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

        print(
            f"best Validation monitor epoch so far: "
            f"{best_validation_monitor_epoch}"
        )

        print(
            "best correct-subject Validation v-MSE: "
            f"{best_validation_correct_subject_v_mse:.6f}"
        )

    else:
        model.load_state_dict(
            template[
                "model_state_dict"
            ],
            strict=True,
        )

        print(
            "Starting B1 scientific full run from ZERO-UPDATE template."
        )

        print(
            "Fresh AdamW optimizer: YES"
        )

        print(
            "Pilot checkpoint used: NO"
        )

        print(
            "EMA: OFF"
        )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    # Build loaders only after possible RNG restoration.
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

    if len(
        train_loader
    ) != 748:
        raise RuntimeError(
            f"Expected 748 Train batches, got "
            f"{len(train_loader)}."
        )

    if len(
        validation_loader
    ) != 151:
        raise RuntimeError(
            f"Expected 151 Validation batches, got "
            f"{len(validation_loader)}."
        )

    # ========================================================
    # Step 4 — Deterministic Validation baseline / replay
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
        device=device,
    )

    if resumed:
        expected_correct = float(
            history[
                -1
            ][
                "validation_correct_subject_v_mse"
            ]
        )

        expected_wrong = float(
            history[
                -1
            ][
                "validation_wrong_subject_v_mse"
            ]
        )

        correct_diff = abs(
            validation_now[
                "correct_subject_v_mse"
            ]
            - expected_correct
        )

        wrong_diff = abs(
            validation_now[
                "wrong_subject_v_mse"
            ]
            - expected_wrong
        )

        print(
            f"saved correct-subject Validation v-MSE: "
            f"{expected_correct:.8f}"
        )

        print(
            f"replayed correct-subject Validation v-MSE: "
            f"{validation_now['correct_subject_v_mse']:.8f}"
        )

        print(
            f"correct difference: "
            f"{correct_diff:.3e}"
        )

        print(
            f"saved wrong-subject Validation v-MSE: "
            f"{expected_wrong:.8f}"
        )

        print(
            f"replayed wrong-subject Validation v-MSE: "
            f"{validation_now['wrong_subject_v_mse']:.8f}"
        )

        print(
            f"wrong difference: "
            f"{wrong_diff:.3e}"
        )

        if (
            correct_diff
            > 1e-6
            or wrong_diff
            > 1e-6
        ):
            raise RuntimeError(
                "Deterministic B1 Validation replay mismatch."
            )

        print(
            "Validation replay: PASS"
        )

    else:
        print(
            "initial zero-update correct-subject Validation v-MSE: "
            f"{validation_now['correct_subject_v_mse']:.6f}"
        )

        print(
            "initial zero-update wrong-subject Validation v-MSE: "
            f"{validation_now['wrong_subject_v_mse']:.6f}"
        )

        print(
            "initial wrong-correct gap: "
            f"{validation_now['wrong_minus_correct_v_mse']:.6e}"
        )

        print(
            "initial correct-vs-wrong output mean abs: "
            f"{validation_now['correct_vs_wrong_output_mean_abs']:.6e}"
        )

        print(
            "This is a baseline monitor, not a selected B1 checkpoint."
        )

    # ========================================================
    # Step 5 — Run 30 epochs
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — RUN B1 30-EPOCH FULL FINE-TUNING"
    )

    print(
        "=" * 80
    )

    append_event(
        "run_started_or_resumed",
        start_epoch=(
            start_epoch
        ),
        max_epochs=(
            MAX_EPOCHS
        ),
        resumed=(
            resumed
        ),
        zero_update_template_sha256=(
            template_file_sha256_before
        ),
        frozen_B0_sha256=(
            frozen_B0_file_sha256_before
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

            train_loss_sum = 0.0
            presented = 0

            gradient_norms: list[
                float
            ] = []

            subject_embedding_gradient_norms: list[
                float
            ] = []

            subject_projection_gradient_norms: list[
                float
            ] = []

            amp_skips = 0
            consecutive_amp_skips = 0
            successful_optimizer_steps = 0

            for (
                batch_index,
                batch,
            ) in enumerate(
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

                if not torch.isfinite(
                    x0
                ).all():
                    raise RuntimeError(
                        "Nonfinite Train input at "
                        f"epoch={epoch}, "
                        f"batch={batch_index}."
                    )

                t = torch.randint(
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
                    t,
                    alpha_bar,
                )

                target_v = make_v_target(
                    x0,
                    epsilon,
                    t,
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
                        t,
                        state,
                        subject,
                    )

                    loss = loss_fn(
                        pred_v.float(),
                        target_v.float(),
                    )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Nonfinite B1 v-loss at "
                        f"epoch={epoch}, "
                        f"batch={batch_index}."
                    )

                if not torch.isfinite(
                    pred_v
                ).all():
                    raise RuntimeError(
                        "Nonfinite B1 prediction at "
                        f"epoch={epoch}, "
                        f"batch={batch_index}."
                    )

                loss_value = float(
                    loss.item()
                )

                grad_scaler.scale(
                    loss
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

                subject_embedding_grad_norm = (
                    group_gradient_norm(
                        model.subject_embedding.parameters()
                    )
                )

                subject_projection_grad_norm = (
                    group_gradient_norm(
                        model.subject_projection.parameters()
                    )
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
                            "Nonfinite clip norm after finite "
                            "B1 gradient inspection."
                        )

                    grad_scaler.step(
                        optimizer
                    )

                    grad_scaler.update()

                    successful_optimizer_steps += 1

                    gradient_norms.append(
                        float(
                            grad_norm
                        )
                    )

                    subject_embedding_gradient_norms.append(
                        float(
                            subject_embedding_grad_norm
                        )
                    )

                    subject_projection_gradient_norms.append(
                        float(
                            subject_projection_grad_norm
                        )
                    )

                    consecutive_amp_skips = 0

                else:
                    # GradScaler already observed found_inf in unscale_.
                    # step() safely skips the optimizer update.
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
                        epoch=(
                            epoch
                        ),
                        batch=(
                            batch_index
                        ),
                        loss=(
                            loss_value
                        ),
                        scale_before=(
                            scale_before
                        ),
                        scale_after=(
                            scale_after
                        ),
                        bad_parameter_names=(
                            bad_names[
                                :20
                            ]
                        ),
                    )

                    if (
                        amp_skips
                        > MAX_AMP_SKIPS_PER_EPOCH
                    ):
                        raise RuntimeError(
                            f"Too many AMP skips in "
                            f"epoch {epoch}: "
                            f"{amp_skips}."
                        )

                    if (
                        consecutive_amp_skips
                        >= MAX_CONSECUTIVE_AMP_SKIPS
                    ):
                        raise RuntimeError(
                            "Too many consecutive AMP skips "
                            f"in epoch {epoch}: "
                            f"{consecutive_amp_skips}."
                        )

                presented += (
                    current_batch_size
                )

                train_loss_sum += (
                    loss_value
                    * current_batch_size
                )

                if (
                    batch_index == 1
                    or batch_index
                    % PRINT_EVERY_BATCHES
                    == 0
                    or batch_index
                    == len(
                        train_loader
                    )
                ):
                    print(
                        f"epoch "
                        f"{epoch:>2}/"
                        f"{MAX_EPOCHS} "
                        f"| batch "
                        f"{batch_index:>3}/"
                        f"{len(train_loader)} "
                        f"| train_v_MSE="
                        f"{train_loss_sum/presented:.6f} "
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
                    "Train presentation accounting failed: "
                    f"{presented}"
                )

            if (
                successful_optimizer_steps
                + amp_skips
                != len(
                    train_loader
                )
            ):
                raise RuntimeError(
                    "B1 optimizer-step accounting failed."
                )

            if not gradient_norms:
                raise RuntimeError(
                    "No successful B1 optimizer steps in epoch."
                )

            assert_finite_state_dict(
                model,
                "live B1 model",
            )

            validation_metrics = (
                evaluate_validation(
                    model=model,
                    loader=(
                        validation_loader
                    ),
                    alpha_bar=(
                        alpha_bar
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
                train_loss_sum
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

            mean_subject_embedding_gradient_norm = float(
                np.mean(
                    subject_embedding_gradient_norms
                )
            )

            mean_subject_projection_gradient_norm = float(
                np.mean(
                    subject_projection_gradient_norms
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

            validation_correct = float(
                validation_metrics[
                    "correct_subject_v_mse"
                ]
            )

            validation_improved = (
                validation_correct
                < best_validation_correct_subject_v_mse
            )

            if validation_improved:
                best_validation_correct_subject_v_mse = (
                    validation_correct
                )

                best_validation_monitor_epoch = int(
                    epoch
                )

            row = {
                "epoch": int(
                    epoch
                ),
                "train_v_mse": float(
                    train_v_mse
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
                "mean_subject_embedding_gradient_norm": (
                    mean_subject_embedding_gradient_norm
                ),
                "mean_subject_projection_gradient_norm": (
                    mean_subject_projection_gradient_norm
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
                "best_validation_correct_subject_v_mse": float(
                    best_validation_correct_subject_v_mse
                ),
            }

            history.append(
                row
            )

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"B1 EPOCH "
                f"{epoch} COMPLETE"
            )

            print(
                f"Train v-MSE: "
                f"{train_v_mse:.6f}"
            )

            print(
                "Validation correct-subject v-MSE: "
                f"{validation_metrics['correct_subject_v_mse']:.6f}"
            )

            print(
                "Validation wrong-subject v-MSE: "
                f"{validation_metrics['wrong_subject_v_mse']:.6f}"
            )

            print(
                "Validation wrong-correct gap: "
                f"{validation_metrics['wrong_minus_correct_v_mse']:.6e}"
            )

            print(
                "Validation correct-vs-wrong output mean abs: "
                f"{validation_metrics['correct_vs_wrong_output_mean_abs']:.6e}"
            )

            print(
                "Best Validation monitor correct-subject v-MSE: "
                f"{best_validation_correct_subject_v_mse:.6f} "
                f"@ epoch "
                f"{best_validation_monitor_epoch}"
            )

            print(
                f"Mean gradient norm: "
                f"{mean_gradient_norm:.4f}"
            )

            print(
                "Mean subject-embedding grad norm: "
                f"{mean_subject_embedding_gradient_norm:.6e}"
            )

            print(
                "Mean subject-projection grad norm: "
                f"{mean_subject_projection_gradient_norm:.6e}"
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
                f"Predeclared final-candidate epoch: "
                f"{epoch in PREDECLARED_RAW_CANDIDATE_EPOCHS}"
            )

            print(
                "-" * 80
                + "\n"
            )

            checkpoint = make_checkpoint(
                epoch=(
                    epoch
                ),
                protocol_snapshot=(
                    protocol_snapshot
                ),
                model=(
                    model
                ),
                optimizer=(
                    optimizer
                ),
                grad_scaler=(
                    grad_scaler
                ),
                loader_generator=(
                    loader_generator
                ),
                noise_generator=(
                    noise_generator
                ),
                history=(
                    history
                ),
                best_validation_monitor_epoch=(
                    best_validation_monitor_epoch
                ),
                best_validation_correct_subject_v_mse=(
                    best_validation_correct_subject_v_mse
                ),
                cumulative_training_seconds=(
                    cumulative_training_seconds
                ),
                total_amp_skips=(
                    total_amp_skips
                ),
                template_file_sha256=(
                    template_file_sha256_before
                ),
                template_model_state_sha256=(
                    template_model_state_sha256
                ),
                frozen_B0_file_sha256=(
                    frozen_B0_file_sha256_before
                ),
            )

            # Numerical recovery state first.
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
                    "NOTE: this is NOT automatically the final "
                    "scientific B1 generator."
                )

            if (
                epoch
                % CHECKPOINT_EVERY_EPOCHS
                == 0
            ):
                milestone = (
                    CHECKPOINT_DIR
                    / f"epoch_{epoch:03d}.pt"
                )

                atomic_torch_save(
                    checkpoint,
                    milestone,
                )

                print(
                    f"milestone checkpoint: "
                    f"{milestone}"
                )

            if (
                epoch
                in PREDECLARED_RAW_CANDIDATE_EPOCHS
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

            save_history(
                history
            )

            save_figures(
                history
            )

            state_payload = {
                "status": (
                    "B1_FULL_FINETUNE_IN_PROGRESS"
                    if epoch
                    < MAX_EPOCHS
                    else "B1_FULL_FINETUNE_30_EPOCHS_COMPLETE"
                ),
                "completed_epoch": (
                    epoch
                ),
                "next_epoch": (
                    epoch
                    + 1
                    if epoch
                    < MAX_EPOCHS
                    else None
                ),
                "max_epochs": (
                    MAX_EPOCHS
                ),
                "best_validation_monitor_epoch": (
                    best_validation_monitor_epoch
                ),
                "best_validation_correct_subject_v_mse": (
                    best_validation_correct_subject_v_mse
                ),
                "total_amp_skips": (
                    total_amp_skips
                ),
                "cumulative_training_minutes": (
                    cumulative_training_seconds
                    / 60.0
                ),
                "final_scientific_B1_selected": False,
                "predeclared_RAW_candidate_epochs": (
                    PREDECLARED_RAW_CANDIDATE_EPOCHS
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
                epoch=(
                    epoch
                ),
                train_v_mse=(
                    train_v_mse
                ),
                validation_correct_subject_v_mse=(
                    validation_metrics[
                        "correct_subject_v_mse"
                    ]
                ),
                validation_wrong_subject_v_mse=(
                    validation_metrics[
                        "wrong_subject_v_mse"
                    ]
                ),
                validation_wrong_minus_correct_v_mse=(
                    validation_metrics[
                        "wrong_minus_correct_v_mse"
                    ]
                ),
                amp_skips=(
                    amp_skips
                ),
                epoch_seconds=(
                    epoch_seconds
                ),
                candidate_epoch=(
                    epoch
                    in PREDECLARED_RAW_CANDIDATE_EPOCHS
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
                history[
                    -1
                ][
                    "epoch"
                ]
                if history
                else 0
            ),
        )

        raise

    # ========================================================
    # Step 6 — Finalize 30-epoch run
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — FINALIZE B1 30-EPOCH FULL FINE-TUNING"
    )

    print(
        "=" * 80
    )

    if len(
        history
    ) != MAX_EPOCHS:
        raise RuntimeError(
            "B1 full run history does not contain "
            f"{MAX_EPOCHS} completed epochs."
        )

    latest = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if int(
        latest[
            "epoch"
        ]
    ) != MAX_EPOCHS:
        raise RuntimeError(
            "Latest B1 checkpoint is not epoch 30."
        )

    atomic_torch_save(
        latest,
        FINAL_CHECKPOINT,
    )

    template_file_sha256_after = (
        sha256_file(
            WARM_START_TEMPLATE
        )
    )

    frozen_B0_file_sha256_after = (
        sha256_file(
            FROZEN_B0_GENERATOR
        )
    )

    if (
        template_file_sha256_after
        != template_file_sha256_before
    ):
        raise RuntimeError(
            "Zero-update B1 template changed "
            "during full run."
        )

    if (
        frozen_B0_file_sha256_after
        != frozen_B0_file_sha256_before
    ):
        raise RuntimeError(
            "Frozen B0 changed during B1 full run."
        )

    candidate_records = []

    for epoch in (
        PREDECLARED_RAW_CANDIDATE_EPOCHS
    ):
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

        candidate_records.append(
            {
                "epoch": (
                    epoch
                ),
                "path": str(
                    path
                ),
                "file_sha256": (
                    sha256_file(
                        path
                    )
                ),
                "model_state_sha256": (
                    checkpoint[
                        "model_state_sha256"
                    ]
                ),
                "train_v_mse": float(
                    checkpoint[
                        "history"
                    ][
                        -1
                    ][
                        "train_v_mse"
                    ]
                ),
                "validation_correct_subject_v_mse": float(
                    checkpoint[
                        "history"
                    ][
                        -1
                    ][
                        "validation_correct_subject_v_mse"
                    ]
                ),
                "validation_wrong_subject_v_mse": float(
                    checkpoint[
                        "history"
                    ][
                        -1
                    ][
                        "validation_wrong_subject_v_mse"
                    ]
                ),
                "validation_wrong_minus_correct_v_mse": float(
                    checkpoint[
                        "history"
                    ][
                        -1
                    ][
                        "validation_wrong_minus_correct_v_mse"
                    ]
                ),
                "validation_correct_vs_wrong_output_mean_abs": float(
                    checkpoint[
                        "history"
                    ][
                        -1
                    ][
                        "validation_correct_vs_wrong_output_mean_abs"
                    ]
                ),
            }
        )

    final_row = history[
        -1
    ]

    result = {
        "status": (
            "B1_FULL_FINETUNE_30_EPOCHS_COMPLETE"
        ),
        "completed_epochs": (
            MAX_EPOCHS
        ),
        "initialization": (
            "zero-update B1 warm-start template"
        ),
        "pilot_checkpoint_used": False,
        "prediction_target": (
            "v"
        ),
        "EMA_enabled": False,
        "scientific_checkpoint_weights": (
            "RAW"
        ),
        "final_train_v_mse": float(
            final_row[
                "train_v_mse"
            ]
        ),
        "final_validation_correct_subject_v_mse": float(
            final_row[
                "validation_correct_subject_v_mse"
            ]
        ),
        "final_validation_wrong_subject_v_mse": float(
            final_row[
                "validation_wrong_subject_v_mse"
            ]
        ),
        "final_validation_wrong_minus_correct_v_mse": float(
            final_row[
                "validation_wrong_minus_correct_v_mse"
            ]
        ),
        "final_validation_correct_vs_wrong_output_mean_abs": float(
            final_row[
                "validation_correct_vs_wrong_output_mean_abs"
            ]
        ),
        "best_validation_monitor_epoch": (
            best_validation_monitor_epoch
        ),
        "best_validation_correct_subject_v_mse": (
            best_validation_correct_subject_v_mse
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
        "final_scientific_B1_selected": False,
        "next_selection_rule": (
            "Compare RAW epochs 10, 20, and 30 with "
            "subject-aware generative fidelity metrics; "
            "do not select solely from Validation v-MSE."
        ),
        "source_immutability": {
            "zero_update_template_unchanged": True,
            "frozen_B0_unchanged": True,
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
            "Run B1 RAW epoch 10/20/30 subject-aware "
            "generative fidelity sweep before freezing B1."
        ),
    }

    atomic_json_save(
        result,
        RESULT_FILE,
    )

    append_event(
        "run_complete",
        completed_epochs=(
            MAX_EPOCHS
        ),
        best_validation_monitor_epoch=(
            best_validation_monitor_epoch
        ),
        best_validation_correct_subject_v_mse=(
            best_validation_correct_subject_v_mse
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
        "B1 30-EPOCH FULL FINE-TUNING COMPLETE"
    )

    print(
        "=" * 80
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
        "final Validation correct-subject v-MSE: "
        f"{final_row['validation_correct_subject_v_mse']:.6f}"
    )

    print(
        "final Validation wrong-subject v-MSE: "
        f"{final_row['validation_wrong_subject_v_mse']:.6f}"
    )

    print(
        "final Validation wrong-correct gap: "
        f"{final_row['validation_wrong_minus_correct_v_mse']:.6e}"
    )

    print(
        "final Validation correct-vs-wrong output mean abs: "
        f"{final_row['validation_correct_vs_wrong_output_mean_abs']:.6e}"
    )

    print(
        "best Validation monitor epoch: "
        f"{best_validation_monitor_epoch}"
    )

    print(
        "best correct-subject Validation v-MSE: "
        f"{best_validation_correct_subject_v_mse:.6f}"
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
        "zero-update template unchanged: YES"
    )

    print(
        "frozen B0 unchanged: YES"
    )

    print(
        "pilot checkpoint used: NO"
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
            f"  epoch {candidate['epoch']:>2} "
            f"| Train={candidate['train_v_mse']:.6f} "
            f"| Val-correct={candidate['validation_correct_subject_v_mse']:.6f} "
            f"| Val-wrong={candidate['validation_wrong_subject_v_mse']:.6f} "
            f"| gap={candidate['validation_wrong_minus_correct_v_mse']:.3e}"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  best_validation_monitor.pt is only a Validation monitor."
    )

    print(
        "  It is NOT automatically the final scientific B1 generator."
    )

    print(
        "  Final B1 selection must compare RAW epochs 10/20/30 "
        "on predeclared subject-aware generative fidelity."
    )

    print(
        f"\nresult JSON:"
        f"\n{RESULT_FILE}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "B1 RAW EPOCH 10/20/30 SUBJECT-AWARE "
        "GENERATIVE FIDELITY SWEEP"
    )


if __name__ == "__main__":
    main()
