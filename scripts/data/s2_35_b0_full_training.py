from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from pathlib import Path

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

from s2_32_b0_unet_shape_test import (
    B0StateOnlyUNet1D,
    CONTRACT_FILE,
)

from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
    q_sample,
    assert_finite_gradients,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

B0_METRIC_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "b0"
)

B0_FIGURE_DIR = (
    REPO_ROOT
    / "outputs"
    / "figures"
    / "scenario2"
    / "b0"
)

B0_LOG_DIR = (
    REPO_ROOT
    / "outputs"
    / "logs"
    / "scenario2"
    / "b0"
)

B0_MODEL_DIR = (
    REPO_ROOT
    / "outputs"
    / "models"
    / "scenario2"
    / "b0"
)


# ============================================================
# Full-training output files
# ============================================================

PROTOCOL_FILE = (
    B0_METRIC_DIR
    / "b0_full_training_protocol_v1.json"
)

STATE_FILE = (
    B0_METRIC_DIR
    / "b0_full_training_state_v1.json"
)

FINAL_RESULT_FILE = (
    B0_METRIC_DIR
    / "b0_full_training_result_v1.json"
)

HISTORY_FILE = (
    B0_LOG_DIR
    / "b0_full_training_history_v1.csv"
)

LATEST_CHECKPOINT_FILE = (
    B0_MODEL_DIR
    / "b0_full_latest_v1.pt"
)

BEST_CHECKPOINT_FILE = (
    B0_MODEL_DIR
    / "b0_full_best_v1.pt"
)

FINAL_CHECKPOINT_FILE = (
    B0_MODEL_DIR
    / "b0_full_final_v1.pt"
)

LOSS_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_full_training_loss_v1.png"
)

GRADIENT_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_full_training_gradient_norm_v1.png"
)

THROUGHPUT_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_full_training_throughput_v1.png"
)

MEMORY_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_full_training_gpu_memory_v1.png"
)


# ============================================================
# Frozen full-training protocol
# ============================================================

SEED = 20260907

VALIDATION_SEED = (
    SEED
    + 2000
)

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_VALIDATION_WINDOWS = 9_635
EXPECTED_SUBJECTS = 198

BATCH_SIZE = 64

MAX_EPOCHS = 200

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

GRADIENT_CLIP_NORM = 1.0

EMA_DECAY = 0.9999

TIMESTEPS = 1000
COSINE_S = 0.008

MIXED_PRECISION = True

CHECKPOINT_EVERY_EPOCHS = 5

PRINT_EVERY_BATCHES = 100

RESUME_IF_AVAILABLE = True


# ============================================================
# Frozen training protocol object
# ============================================================

FULL_PROTOCOL = {
    "schema_version": 1,

    "run_name": (
        "B0_STATE_ONLY_DDPM_FULL_V1"
    ),

    "training_start": (
        "FROM_SCRATCH"
    ),

    "pilot_checkpoint_used": False,

    "train": {
        "windows": (
            EXPECTED_TRAIN_WINDOWS
        ),

        "subjects": (
            EXPECTED_SUBJECTS
        ),

        "shuffle": True,

        "drop_last": False,
    },

    "validation": {
        "windows": (
            EXPECTED_VALIDATION_WINDOWS
        ),

        "subjects": (
            EXPECTED_SUBJECTS
        ),

        "shuffle": False,

        "fixed_noise_and_timesteps": True,

        "fixed_validation_seed": (
            VALIDATION_SEED
        ),

        "gradient_updates": False,

        "checkpoint_selection_metric": (
            "EMA_validation_epsilon_MSE"
        ),
    },

    "final_test": {
        "accessed": False,

        "status": "SEALED",
    },

    "diffusion": {
        "timesteps": (
            TIMESTEPS
        ),

        "beta_schedule": (
            "cosine"
        ),

        "cosine_s": (
            COSINE_S
        ),

        "prediction_target": (
            "epsilon"
        ),
    },

    "optimization": {
        "epochs": (
            MAX_EPOCHS
        ),

        "batch_size": (
            BATCH_SIZE
        ),

        "optimizer": (
            "AdamW"
        ),

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

        "ema_decay": (
            EMA_DECAY
        ),
    },

    "checkpointing": {
        "best_metric": (
            "lowest_EMA_validation_epsilon_MSE"
        ),

        "latest_every_epoch": True,

        "milestone_every_epochs": (
            CHECKPOINT_EVERY_EPOCHS
        ),

        "automatic_resume": (
            RESUME_IF_AVAILABLE
        ),
    },

    "early_stopping": False,

    "scientific_hyperparameter_tuning": False,

    "synthetic_scientific_metrics_during_training": False,

    "seed": (
        SEED
    ),
}


# ============================================================
# Determinism
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


# ============================================================
# Atomic file helpers
# ============================================================

def atomic_torch_save(
    obj: dict,
    path: Path,
) -> None:

    temp_path = Path(
        str(path)
        + ".tmp"
    )

    torch.save(
        obj,
        temp_path,
    )

    os.replace(
        temp_path,
        path,
    )


def atomic_json_save(
    obj: dict,
    path: Path,
) -> None:

    temp_path = Path(
        str(path)
        + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# Freeze full protocol
# ============================================================

def freeze_full_protocol() -> None:

    if PROTOCOL_FILE.exists():

        with PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            existing = json.load(f)

        if existing != FULL_PROTOCOL:

            raise RuntimeError(
                "A conflicting B0 full-training "
                "protocol already exists."
            )

        print(
            "identical full-training protocol "
            "already frozen."
        )

        return

    atomic_json_save(
        FULL_PROTOCOL,
        PROTOCOL_FILE,
    )

    print(
        f"full-training protocol frozen: "
        f"{PROTOCOL_FILE}"
    )


# ============================================================
# EMA update
# ============================================================

@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:

    model_parameters = dict(
        model.named_parameters()
    )

    for (
        name,
        ema_parameter,
    ) in ema_model.named_parameters():

        model_parameter = (
            model_parameters[
                name
            ]
        )

        ema_parameter.mul_(
            decay
        )

        ema_parameter.add_(
            model_parameter,
            alpha=(
                1.0
                - decay
            ),
        )

    model_buffers = dict(
        model.named_buffers()
    )

    for (
        name,
        ema_buffer,
    ) in ema_model.named_buffers():

        if name in model_buffers:

            ema_buffer.copy_(
                model_buffers[
                    name
                ]
            )


# ============================================================
# Deterministic Validation diffusion MSE
# ============================================================

@torch.no_grad()
def evaluate_validation_mse(
    model: nn.Module,
    loader: DataLoader,
    schedule_device: dict[str, torch.Tensor],
    device: torch.device,
) -> float:

    model.eval()

    validation_generator = (
        torch.Generator(
            device="cuda"
        )
    )

    validation_generator.manual_seed(
        VALIDATION_SEED
    )

    total_loss = 0.0
    total_examples = 0

    for batch in loader:

        x0 = (
            batch[
                "x"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        eye_state = (
            batch[
                "y"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        batch_size = int(
            x0.shape[0]
        )

        timesteps = torch.randint(
            low=0,
            high=TIMESTEPS,

            size=(
                batch_size,
            ),

            device=device,

            dtype=torch.long,

            generator=(
                validation_generator
            ),
        )

        epsilon = torch.randn(
            x0.shape,

            device=device,

            dtype=x0.dtype,

            generator=(
                validation_generator
            ),
        )

        xt = q_sample(
            x0=x0,

            timesteps=timesteps,

            noise=epsilon,

            sqrt_alpha_cumprod=(
                schedule_device[
                    "sqrt_alpha_cumprod"
                ]
            ),

            sqrt_one_minus_alpha_cumprod=(
                schedule_device[
                    "sqrt_one_minus_alpha_cumprod"
                ]
            ),
        )

        with torch.amp.autocast(
            device_type="cuda",

            dtype=torch.float16,

            enabled=MIXED_PRECISION,
        ):

            epsilon_pred = model(
                xt,
                timesteps,
                eye_state,
            )

            loss = torch.mean(
                (
                    epsilon_pred.float()
                    - epsilon.float()
                )
                ** 2
            )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Nonfinite Validation "
                "diffusion loss."
            )

        total_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        total_examples += (
            batch_size
        )

    if total_examples != (
        EXPECTED_VALIDATION_WINDOWS
    ):

        raise RuntimeError(
            "Validation window accounting "
            "failed."
        )

    return (
        total_loss
        / total_examples
    )


# ============================================================
# History CSV
# ============================================================

def save_history_csv(
    history: list[dict],
) -> None:

    temp_path = Path(
        str(
            HISTORY_FILE
        )
        + ".tmp"
    )

    fieldnames = [
        "epoch",
        "train_mse",
        "ema_validation_mse",
        "mean_gradient_norm",
        "max_gradient_norm",
        "epoch_seconds",
        "train_windows_per_second",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "best_epoch",
        "best_ema_validation_mse",
    ]

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    os.replace(
        temp_path,
        HISTORY_FILE,
    )


# ============================================================
# Plot helpers
# ============================================================

def save_figures(
    history: list[dict],
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
                "train_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    val_loss = np.asarray(
        [
            row[
                "ema_validation_mse"
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

    throughput = np.asarray(
        [
            row[
                "train_windows_per_second"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    peak_memory = np.asarray(
        [
            row[
                "peak_allocated_mib"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Train / Validation loss
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(10, 6)
    )

    ax = figure.add_subplot(111)

    ax.plot(
        epochs,
        train_loss,
        label="Train epsilon MSE",
    )

    ax.plot(
        epochs,
        val_loss,
        label="EMA Validation epsilon MSE",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "MSE"
    )

    ax.set_title(
        "B0 Full Training — Diffusion Loss"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    figure.tight_layout()

    figure.savefig(
        LOSS_FIGURE_FILE,
        dpi=160,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(10, 6)
    )

    ax = figure.add_subplot(111)

    ax.plot(
        epochs,
        gradient_norm,
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Mean global gradient norm"
    )

    ax.set_title(
        "B0 Full Training — Gradient Norm"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        GRADIENT_FIGURE_FILE,
        dpi=160,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(10, 6)
    )

    ax = figure.add_subplot(111)

    ax.plot(
        epochs,
        throughput,
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Train windows / second"
    )

    ax.set_title(
        "B0 Full Training — Throughput"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        THROUGHPUT_FIGURE_FILE,
        dpi=160,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(10, 6)
    )

    ax = figure.add_subplot(111)

    ax.plot(
        epochs,
        peak_memory,
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Peak allocated GPU memory (MiB)"
    )

    ax.set_title(
        "B0 Full Training — GPU Memory"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        MEMORY_FIGURE_FILE,
        dpi=160,
    )

    plt.close(
        figure
    )


# ============================================================
# Checkpoint builder
# ============================================================

def build_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
    history: list[dict],
    best_epoch: int | None,
    best_ema_validation_mse: float,
    cumulative_training_seconds: float,
    contract: dict,
) -> dict:

    return {
        "status": (
            "B0_FULL_TRAINING_CHECKPOINT"
        ),

        "epoch": int(
            epoch
        ),

        "seed": (
            SEED
        ),

        "validation_seed": (
            VALIDATION_SEED
        ),

        "contract": (
            contract
        ),

        "full_training_protocol": (
            FULL_PROTOCOL
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "ema_state_dict": (
            ema_model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "grad_scaler_state_dict": (
            grad_scaler.state_dict()
        ),

        "train_generator_state": (
            train_generator.get_state()
        ),

        "rng_state": {
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
        },

        "history": (
            history
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_ema_validation_mse": (
            best_ema_validation_mse
        ),

        "cumulative_training_seconds": (
            cumulative_training_seconds
        ),

        "validation_accessed": True,

        "final_test_accessed": False,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "SCENARIO 2 — "
        "B0 FULL DDPM TRAINING"
    )
    print("=" * 80)

    print(
        "\nSCIENTIFIC FULL TRAINING RUN"
    )

    print(
        "\nB0 definition:"
    )

    print(
        "  p(EEG | eye_state)"
    )

    print(
        "\nTraining source:"
    )

    print(
        "  FROM SCRATCH"
    )

    print(
        "  pilot checkpoint = NOT USED"
    )

    print(
        "\nCheckpoint selection:"
    )

    print(
        "  lowest deterministic "
        "EMA Validation epsilon MSE"
    )

    print(
        "\nEarly stopping: NO"
    )

    print(
        "Scientific tuning: NO"
    )

    print(
        "Final Test: SEALED"
    )

    # ========================================================
    # Output directories
    # ========================================================

    for directory in (
        B0_METRIC_DIR,
        B0_FIGURE_DIR,
        B0_LOG_DIR,
        B0_MODEL_DIR,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Freeze full protocol BEFORE training
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — FREEZE FULL-TRAINING PROTOCOL")
    print("=" * 80)

    freeze_full_protocol()

    # ========================================================
    # Step 2 — Environment
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — ENVIRONMENT")
    print("=" * 80)

    set_seed(
        SEED
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA runtime: "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM GiB: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
    )

    # ========================================================
    # Step 3 — Frozen B0 contract
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — LOAD FROZEN B0 CONTRACT")
    print("=" * 80)

    with CONTRACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        contract = json.load(f)

    training_config = (
        contract[
            "training"
        ]
    )

    diffusion_config = (
        contract[
            "diffusion"
        ]
    )

    required_checks = [
        (
            int(
                training_config[
                    "max_epochs"
                ]
            )
            == MAX_EPOCHS,

            "max_epochs",
        ),

        (
            int(
                training_config[
                    "batch_size"
                ]
            )
            == BATCH_SIZE,

            "batch_size",
        ),

        (
            float(
                training_config[
                    "learning_rate"
                ]
            )
            == LEARNING_RATE,

            "learning_rate",
        ),

        (
            float(
                training_config[
                    "weight_decay"
                ]
            )
            == WEIGHT_DECAY,

            "weight_decay",
        ),

        (
            float(
                training_config[
                    "gradient_clip_norm"
                ]
            )
            == GRADIENT_CLIP_NORM,

            "gradient_clip_norm",
        ),

        (
            float(
                training_config[
                    "ema_decay"
                ]
            )
            == EMA_DECAY,

            "ema_decay",
        ),

        (
            int(
                training_config[
                    "checkpoint_every_epochs"
                ]
            )
            == CHECKPOINT_EVERY_EPOCHS,

            "checkpoint_every_epochs",
        ),

        (
            int(
                diffusion_config[
                    "timesteps"
                ]
            )
            == TIMESTEPS,

            "timesteps",
        ),

        (
            diffusion_config[
                "prediction_target"
            ]
            == "epsilon",

            "prediction_target",
        ),

        (
            diffusion_config[
                "beta_schedule"
            ]
            == "cosine",

            "beta_schedule",
        ),

        (
            contract[
                "conditioning"
            ][
                "subject_condition"
            ]
            is False,

            "B0 subject conditioning",
        ),
    ]

    for (
        passed,
        name,
    ) in required_checks:

        if not passed:

            raise RuntimeError(
                f"Frozen B0 contract mismatch: "
                f"{name}"
            )

    print(
        "B0 contract consistency: PASS"
    )

    # ========================================================
    # Step 4 — Diffusion schedule
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — BUILD DIFFUSION SCHEDULE")
    print("=" * 80)

    schedule_cpu = (
        build_cosine_schedule(
            timesteps=TIMESTEPS,
            cosine_s=COSINE_S,
        )
    )

    schedule_device = {
        name: tensor.to(
            device=device,
            dtype=torch.float32,
        )

        for (
            name,
            tensor,
        ) in schedule_cpu.items()
    }

    print(
        f"schedule device: "
        f"{schedule_device['betas'].device}"
    )

    print(
        "schedule: PASS"
    )

    # ========================================================
    # Step 5 — Train / Validation indexes
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — BUILD TRAIN / VALIDATION DATA")
    print("=" * 80)

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        scaler = json.load(f)

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
            "Unexpected Train subject count."
        )

    if (
        train_mapping
        != validation_mapping
    ):

        raise RuntimeError(
            "Train / Validation "
            "subject mapping mismatch."
        )

    train_dataset = (
        CachedRealTrainWindowDataset(
            records=train_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    validation_dataset = (
        CachedRealTrainWindowDataset(
            records=validation_records,
            scaler=scaler,
            max_open_files=8,
        )
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
        "Final Test EEG accessed: NO"
    )

    # ========================================================
    # Step 6 — Model / optimizer / EMA
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — BUILD FULL TRAINING OBJECTS")
    print("=" * 80)

    model = (
        B0StateOnlyUNet1D(
            contract
        )
        .to(device)
    )

    ema_model = copy.deepcopy(
        model
    ).to(
        device
    )

    ema_model.eval()

    for parameter in (
        ema_model.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=MIXED_PRECISION,
    )

    n_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"parameters: "
        f"{n_parameters:,}"
    )

    print(
        "optimizer: AdamW"
    )

    print(
        f"learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        "learning-rate scheduler: NONE"
    )

    print(
        f"EMA decay: "
        f"{EMA_DECAY}"
    )

    # ========================================================
    # Step 7 — Resume detection
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — RESUME CHECK")
    print("=" * 80)

    history: list[dict] = []

    best_epoch = None

    best_ema_validation_mse = (
        float("inf")
    )

    cumulative_training_seconds = 0.0

    start_epoch = 1

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        SEED
    )

    resumed = False

    if (
        RESUME_IF_AVAILABLE
        and LATEST_CHECKPOINT_FILE.exists()
    ):

        print(
            f"latest checkpoint found:"
            f"\n{LATEST_CHECKPOINT_FILE}"
        )

        checkpoint = torch.load(
            LATEST_CHECKPOINT_FILE,
            map_location=device,
            weights_only=False,
        )

        if checkpoint[
            "full_training_protocol"
        ] != FULL_PROTOCOL:

            raise RuntimeError(
                "Resume checkpoint protocol "
                "does not match current protocol."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        ema_model.load_state_dict(
            checkpoint[
                "ema_state_dict"
            ]
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

        train_generator.set_state(
            checkpoint[
                "train_generator_state"
            ]
        )

        random.setstate(
            checkpoint[
                "rng_state"
            ][
                "python"
            ]
        )

        np.random.set_state(
            checkpoint[
                "rng_state"
            ][
                "numpy"
            ]
        )

        torch.set_rng_state(
            checkpoint[
                "rng_state"
            ][
                "torch_cpu"
            ]
        )

        torch.cuda.set_rng_state_all(
            checkpoint[
                "rng_state"
            ][
                "torch_cuda"
            ]
        )

        history = list(
            checkpoint[
                "history"
            ]
        )

        best_epoch = (
            checkpoint[
                "best_epoch"
            ]
        )

        best_ema_validation_mse = float(
            checkpoint[
                "best_ema_validation_mse"
            ]
        )

        cumulative_training_seconds = float(
            checkpoint[
                "cumulative_training_seconds"
            ]
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        resumed = True

        print(
            f"RESUME PASS"
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
            f"best epoch so far: "
            f"{best_epoch}"
        )

        print(
            f"best EMA Validation MSE: "
            f"{best_ema_validation_mse:.6f}"
        )

    else:

        print(
            "No full-training checkpoint found."
        )

        print(
            "Starting B0 FULL training "
            "FROM SCRATCH."
        )

        print(
            "Pilot checkpoint is NOT loaded."
        )

    # ========================================================
    # Step 8 — DataLoaders
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 8 — BUILD DATALOADERS")
    print("=" * 80)

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=True,

        drop_last=False,

        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True,

        drop_last=False,
    )

    print(
        f"Train batches / epoch: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    if len(
        train_loader
    ) != 748:

        raise RuntimeError(
            "Expected exactly 748 "
            "Train batches with drop_last=False."
        )

    # ========================================================
    # Step 9 — Initial deterministic Validation
    # ========================================================

    if not resumed:

        print("\n" + "=" * 80)
        print("STEP 9 — INITIAL EMA VALIDATION")
        print("=" * 80)

        initial_validation_mse = (
            evaluate_validation_mse(
                model=ema_model,

                loader=(
                    validation_loader
                ),

                schedule_device=(
                    schedule_device
                ),

                device=device,
            )
        )

        print(
            f"initial EMA Validation MSE: "
            f"{initial_validation_mse:.6f}"
        )

        print(
            "Validation gradients: NO"
        )

    else:

        initial_validation_mse = None

        print("\n" + "=" * 80)
        print("STEP 9 — INITIAL VALIDATION")
        print("=" * 80)

        print(
            "Skipped because this is "
            "a resumed run."
        )

    # ========================================================
    # Step 10 — Full training
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 10 — FULL B0 TRAINING")
    print("=" * 80)

    if start_epoch > MAX_EPOCHS:

        print(
            "All 200 epochs were already "
            "completed."
        )

    run_start = (
        time.perf_counter()
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

            train_examples = 0

            gradient_norms = []

            # ------------------------------------------------
            # Train epoch
            # ------------------------------------------------

            for (
                batch_index,
                batch,
            ) in enumerate(
                train_loader,
                start=1,
            ):

                x0 = (
                    batch[
                        "x"
                    ]
                    .to(
                        device,
                        non_blocking=True,
                    )
                )

                eye_state = (
                    batch[
                        "y"
                    ]
                    .to(
                        device,
                        non_blocking=True,
                    )
                )

                current_batch_size = int(
                    x0.shape[0]
                )

                timesteps = torch.randint(
                    low=0,
                    high=TIMESTEPS,

                    size=(
                        current_batch_size,
                    ),

                    device=device,

                    dtype=torch.long,
                )

                epsilon = torch.randn_like(
                    x0
                )

                xt = q_sample(
                    x0=x0,

                    timesteps=timesteps,

                    noise=epsilon,

                    sqrt_alpha_cumprod=(
                        schedule_device[
                            "sqrt_alpha_cumprod"
                        ]
                    ),

                    sqrt_one_minus_alpha_cumprod=(
                        schedule_device[
                            "sqrt_one_minus_alpha_cumprod"
                        ]
                    ),
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.amp.autocast(
                    device_type="cuda",

                    dtype=torch.float16,

                    enabled=MIXED_PRECISION,
                ):

                    epsilon_pred = model(
                        xt,
                        timesteps,
                        eye_state,
                    )

                    loss = loss_fn(
                        epsilon_pred,
                        epsilon,
                    )

                if not torch.isfinite(
                    loss
                ):

                    raise RuntimeError(
                        f"Nonfinite Train loss "
                        f"at epoch={epoch}, "
                        f"batch={batch_index}."
                    )

                grad_scaler.scale(
                    loss
                ).backward()

                grad_scaler.unscale_(
                    optimizer
                )

                (
                    _,
                    gradient_norm,
                ) = assert_finite_gradients(
                    model
                )

                gradient_norms.append(
                    float(
                        gradient_norm
                    )
                )

                clipped_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),

                        max_norm=(
                            GRADIENT_CLIP_NORM
                        ),
                    )
                )

                if not torch.isfinite(
                    clipped_norm
                ):

                    raise RuntimeError(
                        "Nonfinite gradient norm."
                    )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()

                update_ema(
                    ema_model=ema_model,

                    model=model,

                    decay=EMA_DECAY,
                )

                train_loss_sum += (
                    float(
                        loss.item()
                    )
                    * current_batch_size
                )

                train_examples += (
                    current_batch_size
                )

                if (
                    batch_index == 1
                    or batch_index
                    % PRINT_EVERY_BATCHES
                    == 0
                    or batch_index
                    == len(train_loader)
                ):

                    current_mean_loss = (
                        train_loss_sum
                        / train_examples
                    )

                    print(
                        f"epoch "
                        f"{epoch:>3}/{MAX_EPOCHS}"
                        f" | batch "
                        f"{batch_index:>3}/"
                        f"{len(train_loader)}"
                        f" | train_MSE="
                        f"{current_mean_loss:.5f}"
                    )

            if train_examples != (
                EXPECTED_TRAIN_WINDOWS
            ):

                raise RuntimeError(
                    f"Train accounting failed: "
                    f"{train_examples}"
                )

            torch.cuda.synchronize()

            train_mse = (
                train_loss_sum
                / train_examples
            )

            # ------------------------------------------------
            # Deterministic EMA Validation
            # ------------------------------------------------

            ema_validation_mse = (
                evaluate_validation_mse(
                    model=ema_model,

                    loader=(
                        validation_loader
                    ),

                    schedule_device=(
                        schedule_device
                    ),

                    device=device,
                )
            )

            # ------------------------------------------------
            # Epoch metrics
            # ------------------------------------------------

            torch.cuda.synchronize()

            epoch_seconds = (
                time.perf_counter()
                - epoch_start
            )

            cumulative_training_seconds += (
                epoch_seconds
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

            train_windows_per_second = (
                EXPECTED_TRAIN_WINDOWS
                / epoch_seconds
            )

            peak_allocated_mib = float(
                torch.cuda
                .max_memory_allocated()
                / 1024**2
            )

            peak_reserved_mib = float(
                torch.cuda
                .max_memory_reserved()
                / 1024**2
            )

            # ------------------------------------------------
            # Best checkpoint rule
            # ------------------------------------------------

            improved = (
                ema_validation_mse
                < best_ema_validation_mse
            )

            if improved:

                best_ema_validation_mse = (
                    ema_validation_mse
                )

                best_epoch = (
                    epoch
                )

            epoch_row = {
                "epoch": int(
                    epoch
                ),

                "train_mse": float(
                    train_mse
                ),

                "ema_validation_mse": float(
                    ema_validation_mse
                ),

                "mean_gradient_norm": float(
                    mean_gradient_norm
                ),

                "max_gradient_norm": float(
                    max_gradient_norm
                ),

                "epoch_seconds": float(
                    epoch_seconds
                ),

                "train_windows_per_second": float(
                    train_windows_per_second
                ),

                "peak_allocated_mib": float(
                    peak_allocated_mib
                ),

                "peak_reserved_mib": float(
                    peak_reserved_mib
                ),

                "best_epoch": int(
                    best_epoch
                ),

                "best_ema_validation_mse": float(
                    best_ema_validation_mse
                ),
            }

            history.append(
                epoch_row
            )

            # ------------------------------------------------
            # Console epoch summary
            # ------------------------------------------------

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"EPOCH {epoch} COMPLETE"
            )

            print(
                f"Train epsilon MSE: "
                f"{train_mse:.6f}"
            )

            print(
                f"EMA Validation epsilon MSE: "
                f"{ema_validation_mse:.6f}"
            )

            print(
                f"Best Validation MSE: "
                f"{best_ema_validation_mse:.6f}"
                f" @ epoch "
                f"{best_epoch}"
            )

            print(
                f"Mean gradient norm: "
                f"{mean_gradient_norm:.4f}"
            )

            print(
                f"Max gradient norm: "
                f"{max_gradient_norm:.4f}"
            )

            print(
                f"Epoch time: "
                f"{epoch_seconds / 60:.2f} min"
            )

            print(
                f"Peak GPU allocated: "
                f"{peak_allocated_mib:.0f} MiB"
            )

            print(
                f"Improved best checkpoint: "
                f"{improved}"
            )

            print(
                "-" * 80
                + "\n"
            )

            # ------------------------------------------------
            # Save logs / figures every epoch
            # ------------------------------------------------

            save_history_csv(
                history
            )

            save_figures(
                history
            )

            # ------------------------------------------------
            # Build checkpoint object
            # ------------------------------------------------

            checkpoint = build_checkpoint(
                epoch=epoch,

                model=model,

                ema_model=ema_model,

                optimizer=optimizer,

                grad_scaler=grad_scaler,

                train_generator=(
                    train_generator
                ),

                history=history,

                best_epoch=(
                    best_epoch
                ),

                best_ema_validation_mse=(
                    best_ema_validation_mse
                ),

                cumulative_training_seconds=(
                    cumulative_training_seconds
                ),

                contract=contract,
            )

            # ------------------------------------------------
            # Latest checkpoint: every epoch
            # ------------------------------------------------

            atomic_torch_save(
                checkpoint,
                LATEST_CHECKPOINT_FILE,
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if improved:

                atomic_torch_save(
                    checkpoint,
                    BEST_CHECKPOINT_FILE,
                )

                print(
                    f"BEST CHECKPOINT UPDATED:"
                    f"\n{BEST_CHECKPOINT_FILE}"
                )

            # ------------------------------------------------
            # Milestone checkpoint every 5 epochs
            # ------------------------------------------------

            if (
                epoch
                % CHECKPOINT_EVERY_EPOCHS
                == 0
            ):

                milestone_file = (
                    B0_MODEL_DIR
                    / (
                        f"b0_full_epoch_"
                        f"{epoch:03d}_v1.pt"
                    )
                )

                atomic_torch_save(
                    checkpoint,
                    milestone_file,
                )

                print(
                    f"milestone checkpoint:"
                    f"\n{milestone_file}"
                )

            # ------------------------------------------------
            # Save live state JSON
            # ------------------------------------------------

            state = {
                "status": (
                    "B0_FULL_TRAINING_IN_PROGRESS"
                ),

                "completed_epoch": (
                    epoch
                ),

                "max_epochs": (
                    MAX_EPOCHS
                ),

                "best_epoch": (
                    best_epoch
                ),

                "best_ema_validation_mse": (
                    best_ema_validation_mse
                ),

                "latest_train_mse": (
                    train_mse
                ),

                "latest_ema_validation_mse": (
                    ema_validation_mse
                ),

                "cumulative_training_hours": (
                    cumulative_training_seconds
                    / 3600.0
                ),

                "latest_checkpoint": str(
                    LATEST_CHECKPOINT_FILE
                ),

                "best_checkpoint": str(
                    BEST_CHECKPOINT_FILE
                ),

                "validation_accessed": True,

                "final_test_accessed": False,
            }

            atomic_json_save(
                state,
                STATE_FILE,
            )

    except KeyboardInterrupt:

        print(
            "\nTRAINING INTERRUPTED BY USER."
        )

        print(
            "The latest COMPLETED epoch "
            "checkpoint remains safe."
        )

        print(
            "Run this script again to "
            "resume automatically."
        )

        raise

    finally:

        train_dataset.close()

        validation_dataset.close()

    # ========================================================
    # Step 11 — Final checkpoint
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 11 — FINALIZE FULL B0 TRAINING")
    print("=" * 80)

    if not history:

        raise RuntimeError(
            "No completed training epochs."
        )

    final_epoch = int(
        history[-1][
            "epoch"
        ]
    )

    if final_epoch != MAX_EPOCHS:

        raise RuntimeError(
            f"Full training incomplete: "
            f"{final_epoch}/{MAX_EPOCHS}"
        )

    latest_checkpoint = torch.load(
        LATEST_CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    atomic_torch_save(
        latest_checkpoint,
        FINAL_CHECKPOINT_FILE,
    )

    # ========================================================
    # Step 12 — Final summary
    # ========================================================

    total_training_hours = (
        cumulative_training_seconds
        / 3600.0
    )

    final_train_mse = float(
        history[-1][
            "train_mse"
        ]
    )

    final_validation_mse = float(
        history[-1][
            "ema_validation_mse"
        ]
    )

    minimum_train_mse = float(
        min(
            row[
                "train_mse"
            ]
            for row in history
        )
    )

    summary = {
        "status": (
            "B0_FULL_TRAINING_COMPLETE"
        ),

        "model": (
            "B0_STATE_ONLY_DDPM"
        ),

        "epochs_completed": (
            MAX_EPOCHS
        ),

        "train_windows_per_epoch": (
            EXPECTED_TRAIN_WINDOWS
        ),

        "validation_windows": (
            EXPECTED_VALIDATION_WINDOWS
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_ema_validation_mse": (
            best_ema_validation_mse
        ),

        "final_train_mse": (
            final_train_mse
        ),

        "final_ema_validation_mse": (
            final_validation_mse
        ),

        "minimum_train_mse": (
            minimum_train_mse
        ),

        "training_hours": (
            total_training_hours
        ),

        "resumed_during_run": (
            resumed
        ),

        "checkpoint_selection": (
            "lowest deterministic "
            "EMA Validation epsilon MSE"
        ),

        "outputs": {
            "protocol": str(
                PROTOCOL_FILE
            ),

            "history_csv": str(
                HISTORY_FILE
            ),

            "state_json": str(
                STATE_FILE
            ),

            "latest_checkpoint": str(
                LATEST_CHECKPOINT_FILE
            ),

            "best_checkpoint": str(
                BEST_CHECKPOINT_FILE
            ),

            "final_checkpoint": str(
                FINAL_CHECKPOINT_FILE
            ),

            "loss_figure": str(
                LOSS_FIGURE_FILE
            ),

            "gradient_figure": str(
                GRADIENT_FIGURE_FILE
            ),

            "throughput_figure": str(
                THROUGHPUT_FIGURE_FILE
            ),

            "memory_figure": str(
                MEMORY_FIGURE_FILE
            ),
        },

        "pilot_checkpoint_used": False,

        "scientific_hyperparameter_tuning": False,

        "synthetic_scientific_metrics_used_for_selection": False,

        "validation_accessed": True,

        "final_test_accessed": False,
    }

    atomic_json_save(
        summary,
        FINAL_RESULT_FILE,
    )

    final_state = {
        "status": (
            "B0_FULL_TRAINING_COMPLETE"
        ),

        "completed_epoch": (
            MAX_EPOCHS
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_ema_validation_mse": (
            best_ema_validation_mse
        ),

        "training_hours": (
            total_training_hours
        ),

        "best_checkpoint": str(
            BEST_CHECKPOINT_FILE
        ),

        "final_checkpoint": str(
            FINAL_CHECKPOINT_FILE
        ),

        "validation_accessed": True,

        "final_test_accessed": False,
    }

    atomic_json_save(
        final_state,
        STATE_FILE,
    )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("B0 FULL TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"epochs: "
        f"{MAX_EPOCHS}"
    )

    print(
        f"best epoch: "
        f"{best_epoch}"
    )

    print(
        f"best EMA Validation MSE: "
        f"{best_ema_validation_mse:.6f}"
    )

    print(
        f"final Train MSE: "
        f"{final_train_mse:.6f}"
    )

    print(
        f"final EMA Validation MSE: "
        f"{final_validation_mse:.6f}"
    )

    print(
        f"training time: "
        f"{total_training_hours:.2f} hours"
    )

    print(
        "Pilot checkpoint used: NO"
    )

    print(
        "Scientific tuning during run: NO"
    )

    print(
        "Validation accessed: YES"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  protocol:"
        f"\n  {PROTOCOL_FILE}"
    )

    print(
        f"  history:"
        f"\n  {HISTORY_FILE}"
    )

    print(
        f"  best checkpoint:"
        f"\n  {BEST_CHECKPOINT_FILE}"
    )

    print(
        f"  final checkpoint:"
        f"\n  {FINAL_CHECKPOINT_FILE}"
    )

    print(
        f"  result JSON:"
        f"\n  {FINAL_RESULT_FILE}"
    )

    print(
        f"  loss figure:"
        f"\n  {LOSS_FIGURE_FILE}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "B0 SAMPLING SANITY "
        "FROM BEST EMA CHECKPOINT"
    )


if __name__ == "__main__":
    main()