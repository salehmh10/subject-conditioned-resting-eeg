from __future__ import annotations

import copy
import csv
import json
import random
import time
from collections import deque
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
# Paths
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


PILOT_RESULT_FILE = (
    B0_METRIC_DIR
    / "b0_small_training_pilot_v1.json"
)

PILOT_HISTORY_FILE = (
    B0_LOG_DIR
    / "b0_small_training_pilot_history_v1.csv"
)

PILOT_CHECKPOINT_FILE = (
    B0_MODEL_DIR
    / "b0_small_training_pilot_v1.pt"
)

LOSS_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_small_training_loss_v1.png"
)

GRADIENT_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_small_training_gradient_norm_v1.png"
)

THROUGHPUT_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_small_training_throughput_v1.png"
)

MEMORY_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_small_training_gpu_memory_v1.png"
)


# ============================================================
# Pilot configuration
# ============================================================

SEED = 20260907

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

BATCH_SIZE = 64

PILOT_STEPS = 500

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

GRADIENT_CLIP_NORM = 1.0

EMA_DECAY = 0.9999

TIMESTEPS = 1000
COSINE_S = 0.008

MIXED_PRECISION = True

ROLLING_WINDOW = 25

PRINT_EVERY = 25


# ============================================================
# Determinism
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


# ============================================================
# EMA
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
            model_parameters[name]
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

    # Copy buffers exactly.
    model_buffers = dict(
        model.named_buffers()
    )

    for (
        name,
        ema_buffer,
    ) in ema_model.named_buffers():

        if name in model_buffers:

            ema_buffer.copy_(
                model_buffers[name]
            )


# ============================================================
# Fixed Train probe
# ============================================================

@torch.no_grad()
def evaluate_fixed_probe(
    model: nn.Module,
    x0: torch.Tensor,
    eye_state: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon: torch.Tensor,
    schedule_device: dict[str, torch.Tensor],
    device: torch.device,
) -> float:

    model.eval()

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
                epsilon_pred
                - epsilon
            ) ** 2
        )

    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Fixed probe loss is nonfinite."
        )

    return float(
        loss.item()
    )


# ============================================================
# Plot helpers
# ============================================================

def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:

    figure = plt.figure(
        figsize=(10, 6)
    )

    ax = figure.add_subplot(111)

    ax.plot(
        x,
        y,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        figure
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "SCENARIO 2 — "
        "B0 SMALL TRAINING PILOT"
    )
    print("=" * 80)

    print(
        "\nEngineering pilot only."
    )

    print(
        f"optimizer steps: "
        f"{PILOT_STEPS}"
    )

    print(
        f"batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        "\nValidation accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nThis pilot must NOT be used "
        "to tune scientific hyperparameters."
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

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
    # Step 1 — Environment
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — ENVIRONMENT")
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
    # Step 2 — Contract
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — LOAD B0 CONTRACT")
    print("=" * 80)

    with CONTRACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        contract = json.load(f)

    training_config = contract[
        "training"
    ]

    diffusion_config = contract[
        "diffusion"
    ]

    checks = [
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
                    "ema_decay"
                ]
            )
            == EMA_DECAY,
            "ema_decay",
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
    ]

    for passed, name in checks:

        if not passed:

            raise RuntimeError(
                f"B0 contract mismatch: "
                f"{name}"
            )

    print(
        "contract consistency: PASS"
    )

    # ========================================================
    # Step 3 — Schedule
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — BUILD DIFFUSION SCHEDULE")
    print("=" * 80)

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    schedule_device = {
        name: tensor.to(
            device=device,
            dtype=torch.float32,
        )
        for name, tensor
        in schedule_cpu.items()
    }

    print(
        f"schedule device: "
        f"{schedule_device['betas'].device}"
    )

    print(
        "schedule: PASS"
    )

    # ========================================================
    # Step 4 — Train data
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — BUILD REAL TRAIN DATASET")
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
            "Normalization not locked."
        )

    records, subject_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    if len(records) != EXPECTED_TRAIN_WINDOWS:

        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(subject_mapping) != EXPECTED_SUBJECTS:

        raise RuntimeError(
            "Unexpected Train subject count."
        )

    dataset = (
        CachedRealTrainWindowDataset(
            records=records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    # --------------------------------------------------------
    # Main shuffled Train loader
    # --------------------------------------------------------

    train_generator = torch.Generator()

    train_generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=True,

        drop_last=True,

        generator=train_generator,
    )

    print(
        f"Train windows: "
        f"{len(dataset)}"
    )

    print(
        f"batches / full epoch: "
        f"{len(train_loader)}"
    )

    # ========================================================
    # Step 5 — Fixed Train probe
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — BUILD FIXED TRAIN PROBE")
    print("=" * 80)

    probe_loader = DataLoader(
        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True,

        drop_last=True,
    )

    probe_batch = next(
        iter(
            probe_loader
        )
    )

    probe_x0 = (
        probe_batch[
            "x"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    probe_state = (
        probe_batch[
            "y"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    # Separate generator keeps the probe
    # fixed independently of Train minibatch RNG.
    probe_generator = torch.Generator(
        device="cuda"
    )

    probe_generator.manual_seed(
        SEED + 1000
    )

    probe_timesteps = torch.randint(
        low=0,
        high=TIMESTEPS,

        size=(
            BATCH_SIZE,
        ),

        device=device,

        dtype=torch.long,

        generator=probe_generator,
    )

    probe_epsilon = torch.randn(
        probe_x0.shape,

        device=device,

        dtype=probe_x0.dtype,

        generator=probe_generator,
    )

    print(
        f"probe shape: "
        f"{tuple(probe_x0.shape)}"
    )

    print(
        f"probe timestep range: "
        f"{int(probe_timesteps.min())}"
        f"–"
        f"{int(probe_timesteps.max())}"
    )

    # ========================================================
    # Step 6 — Model / optimizer / EMA
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — BUILD B0 TRAINING OBJECTS")
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
        f"EMA decay: "
        f"{EMA_DECAY}"
    )

    # --------------------------------------------------------
    # Initial fixed probe
    # --------------------------------------------------------

    initial_probe_loss = (
        evaluate_fixed_probe(
            model=model,

            x0=probe_x0,

            eye_state=probe_state,

            timesteps=(
                probe_timesteps
            ),

            epsilon=(
                probe_epsilon
            ),

            schedule_device=(
                schedule_device
            ),

            device=device,
        )
    )

    initial_ema_probe_loss = (
        evaluate_fixed_probe(
            model=ema_model,

            x0=probe_x0,

            eye_state=probe_state,

            timesteps=(
                probe_timesteps
            ),

            epsilon=(
                probe_epsilon
            ),

            schedule_device=(
                schedule_device
            ),

            device=device,
        )
    )

    print(
        f"initial model probe MSE: "
        f"{initial_probe_loss:.6f}"
    )

    print(
        f"initial EMA probe MSE: "
        f"{initial_ema_probe_loss:.6f}"
    )

    # ========================================================
    # Step 7 — Pilot training
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — RUN SMALL TRAINING PILOT")
    print("=" * 80)

    history: list[
        dict[str, float | int]
    ] = []

    rolling_losses: deque[
        float
    ] = deque(
        maxlen=ROLLING_WINDOW
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    training_start = (
        time.perf_counter()
    )

    train_iterator = iter(
        train_loader
    )

    for step in range(
        1,
        PILOT_STEPS + 1,
    ):

        step_start = (
            time.perf_counter()
        )

        try:

            batch = next(
                train_iterator
            )

        except StopIteration:

            # This is unlikely with 500 steps,
            # but keeps the pilot robust.
            train_iterator = iter(
                train_loader
            )

            batch = next(
                train_iterator
            )

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

        if x0.shape != (
            BATCH_SIZE,
            19,
            512,
        ):

            raise RuntimeError(
                f"Unexpected batch shape "
                f"at step {step}: "
                f"{x0.shape}"
            )

        timesteps = torch.randint(
            low=0,
            high=TIMESTEPS,

            size=(
                BATCH_SIZE,
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

        model.train()

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
                f"Nonfinite loss at "
                f"step {step}."
            )

        grad_scaler.scale(
            loss
        ).backward()

        grad_scaler.unscale_(
            optimizer
        )

        (
            gradient_parameter_count,
            gradient_norm,
        ) = assert_finite_gradients(
            model
        )

        unclipped_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    GRADIENT_CLIP_NORM
                ),
            )
        )

        if not torch.isfinite(
            unclipped_norm
        ):

            raise RuntimeError(
                f"Nonfinite gradient norm "
                f"at step {step}."
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

        torch.cuda.synchronize()

        step_seconds = (
            time.perf_counter()
            - step_start
        )

        loss_value = float(
            loss.item()
        )

        rolling_losses.append(
            loss_value
        )

        rolling_loss = float(
            np.mean(
                rolling_losses
            )
        )

        windows_per_second = (
            BATCH_SIZE
            / step_seconds
        )

        allocated_mib = float(
            torch.cuda.memory_allocated()
            / 1024**2
        )

        reserved_mib = float(
            torch.cuda.memory_reserved()
            / 1024**2
        )

        peak_allocated_mib = float(
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        history.append(
            {
                "step": int(
                    step
                ),

                "loss": (
                    loss_value
                ),

                "rolling_loss": (
                    rolling_loss
                ),

                "gradient_norm": float(
                    gradient_norm
                ),

                "step_seconds": float(
                    step_seconds
                ),

                "windows_per_second": float(
                    windows_per_second
                ),

                "allocated_mib": (
                    allocated_mib
                ),

                "reserved_mib": (
                    reserved_mib
                ),
            }
        )

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == PILOT_STEPS
        ):

            elapsed = (
                time.perf_counter()
                - training_start
            )

            print(
                f"step "
                f"{step:>4}/{PILOT_STEPS}"
                f" | loss="
                f"{loss_value:.5f}"
                f" | roll="
                f"{rolling_loss:.5f}"
                f" | grad="
                f"{gradient_norm:.4f}"
                f" | "
                f"{windows_per_second:.1f} win/s"
                f" | VRAM="
                f"{allocated_mib:.0f} MiB"
                f" | elapsed="
                f"{elapsed / 60:.1f}m"
            )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    # ========================================================
    # Step 8 — Final fixed probes
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 8 — FIXED TRAIN PROBE AFTER PILOT")
    print("=" * 80)

    final_probe_loss = (
        evaluate_fixed_probe(
            model=model,

            x0=probe_x0,

            eye_state=probe_state,

            timesteps=(
                probe_timesteps
            ),

            epsilon=(
                probe_epsilon
            ),

            schedule_device=(
                schedule_device
            ),

            device=device,
        )
    )

    final_ema_probe_loss = (
        evaluate_fixed_probe(
            model=ema_model,

            x0=probe_x0,

            eye_state=probe_state,

            timesteps=(
                probe_timesteps
            ),

            epsilon=(
                probe_epsilon
            ),

            schedule_device=(
                schedule_device
            ),

            device=device,
        )
    )

    print(
        f"model probe:"
        f" {initial_probe_loss:.6f}"
        f" -> "
        f"{final_probe_loss:.6f}"
    )

    print(
        f"EMA probe:"
        f" {initial_ema_probe_loss:.6f}"
        f" -> "
        f"{final_ema_probe_loss:.6f}"
    )

    probe_improvement = (
        initial_probe_loss
        - final_probe_loss
    )

    ema_probe_improvement = (
        initial_ema_probe_loss
        - final_ema_probe_loss
    )

    print(
        f"model probe improvement: "
        f"{probe_improvement:.6f}"
    )

    print(
        f"EMA probe improvement: "
        f"{ema_probe_improvement:.6f}"
    )

    # ========================================================
    # Step 9 — Loss trend
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 9 — TRAINING TREND")
    print("=" * 80)

    losses = np.asarray(
        [
            row["loss"]
            for row in history
        ],
        dtype=np.float64,
    )

    rolling = np.asarray(
        [
            row[
                "rolling_loss"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    gradients = np.asarray(
        [
            row[
                "gradient_norm"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    throughput = np.asarray(
        [
            row[
                "windows_per_second"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    allocated = np.asarray(
        [
            row[
                "allocated_mib"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    initial_mean_loss = float(
        losses[
            :ROLLING_WINDOW
        ].mean()
    )

    final_mean_loss = float(
        losses[
            -ROLLING_WINDOW:
        ].mean()
    )

    trend_improvement = (
        initial_mean_loss
        - final_mean_loss
    )

    trend_fraction = (
        trend_improvement
        / initial_mean_loss
    )

    print(
        f"first {ROLLING_WINDOW} "
        f"mean loss: "
        f"{initial_mean_loss:.6f}"
    )

    print(
        f"last {ROLLING_WINDOW} "
        f"mean loss: "
        f"{final_mean_loss:.6f}"
    )

    print(
        f"absolute improvement: "
        f"{trend_improvement:.6f}"
    )

    print(
        f"relative improvement: "
        f"{100 * trend_fraction:.2f}%"
    )

    if final_mean_loss >= initial_mean_loss:

        raise RuntimeError(
            "Pilot loss did not improve "
            "between initial and final windows."
        )

    print(
        "loss trend: PASS"
    )

    # ========================================================
    # Step 10 — Save CSV history
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 10 — SAVE PILOT HISTORY")
    print("=" * 80)

    with PILOT_HISTORY_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "rolling_loss",
                "gradient_norm",
                "step_seconds",
                "windows_per_second",
                "allocated_mib",
                "reserved_mib",
            ],
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    print(
        f"history CSV: "
        f"{PILOT_HISTORY_FILE}"
    )

    # ========================================================
    # Step 11 — Save figures
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 11 — SAVE PILOT FIGURES")
    print("=" * 80)

    steps = np.arange(
        1,
        PILOT_STEPS + 1,
    )

    # Loss figure:
    # include raw and rolling loss in this figure.
    figure = plt.figure(
        figsize=(10, 6)
    )

    ax = figure.add_subplot(111)

    ax.plot(
        steps,
        losses,
        alpha=0.35,
        label="batch loss",
    )

    ax.plot(
        steps,
        rolling,
        label=(
            f"{ROLLING_WINDOW}-step "
            f"rolling mean"
        ),
    )

    ax.set_title(
        "B0 Small Training Pilot — Diffusion Loss"
    )

    ax.set_xlabel(
        "Optimizer step"
    )

    ax.set_ylabel(
        "Epsilon-prediction MSE"
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

    save_line_plot(
        x=steps,
        y=gradients,

        output_path=(
            GRADIENT_FIGURE_FILE
        ),

        title=(
            "B0 Small Training Pilot "
            "— Gradient Norm"
        ),

        xlabel=(
            "Optimizer step"
        ),

        ylabel=(
            "Global gradient norm "
            "before clipping"
        ),
    )

    save_line_plot(
        x=steps,
        y=throughput,

        output_path=(
            THROUGHPUT_FIGURE_FILE
        ),

        title=(
            "B0 Small Training Pilot "
            "— Throughput"
        ),

        xlabel=(
            "Optimizer step"
        ),

        ylabel=(
            "Windows / second"
        ),
    )

    save_line_plot(
        x=steps,
        y=allocated,

        output_path=(
            MEMORY_FIGURE_FILE
        ),

        title=(
            "B0 Small Training Pilot "
            "— GPU Memory"
        ),

        xlabel=(
            "Optimizer step"
        ),

        ylabel=(
            "Allocated GPU memory (MiB)"
        ),
    )

    print(
        f"loss figure: "
        f"{LOSS_FIGURE_FILE}"
    )

    print(
        f"gradient figure: "
        f"{GRADIENT_FIGURE_FILE}"
    )

    print(
        f"throughput figure: "
        f"{THROUGHPUT_FIGURE_FILE}"
    )

    print(
        f"memory figure: "
        f"{MEMORY_FIGURE_FILE}"
    )

    # ========================================================
    # Step 12 — Save checkpoint
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 12 — SAVE PILOT CHECKPOINT")
    print("=" * 80)

    checkpoint = {
        "status": (
            "B0_SMALL_TRAINING_PILOT"
        ),

        "step": (
            PILOT_STEPS
        ),

        "seed": (
            SEED
        ),

        "contract": (
            contract
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

        "training_seconds": (
            training_seconds
        ),

        "initial_probe_loss": (
            initial_probe_loss
        ),

        "final_probe_loss": (
            final_probe_loss
        ),

        "initial_ema_probe_loss": (
            initial_ema_probe_loss
        ),

        "final_ema_probe_loss": (
            final_ema_probe_loss
        ),

        "full_training": False,

        "validation_accessed": False,

        "final_test_accessed": False,
    }

    torch.save(
        checkpoint,
        PILOT_CHECKPOINT_FILE,
    )

    print(
        f"checkpoint: "
        f"{PILOT_CHECKPOINT_FILE}"
    )

    print(
        f"checkpoint MiB: "
        f"{PILOT_CHECKPOINT_FILE.stat().st_size / 1024**2:.2f}"
    )

    # ========================================================
    # Step 13 — Reload checkpoint verification
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 13 — RELOAD CHECKPOINT VERIFY")
    print("=" * 80)

    loaded = torch.load(
        PILOT_CHECKPOINT_FILE,
        map_location=device,
        weights_only=False,
    )

    reload_model = (
        B0StateOnlyUNet1D(
            contract
        )
        .to(device)
    )

    reload_model.load_state_dict(
        loaded[
            "model_state_dict"
        ]
    )

    reload_probe_loss = (
        evaluate_fixed_probe(
            model=reload_model,

            x0=probe_x0,

            eye_state=probe_state,

            timesteps=(
                probe_timesteps
            ),

            epsilon=(
                probe_epsilon
            ),

            schedule_device=(
                schedule_device
            ),

            device=device,
        )
    )

    reload_difference = abs(
        reload_probe_loss
        - final_probe_loss
    )

    print(
        f"original final probe loss: "
        f"{final_probe_loss:.8f}"
    )

    print(
        f"reloaded probe loss: "
        f"{reload_probe_loss:.8f}"
    )

    print(
        f"difference: "
        f"{reload_difference:.3e}"
    )

    if reload_difference > 1e-6:

        raise RuntimeError(
            "Reloaded checkpoint does not "
            "reproduce final probe loss."
        )

    print(
        "checkpoint reload: PASS"
    )

    del reload_model

    # ========================================================
    # Step 14 — Estimate full-training time
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 14 — FULL TRAINING TIME ESTIMATE")
    print("=" * 80)

    batches_per_epoch = len(
        train_loader
    )

    mean_step_seconds = float(
        np.mean(
            [
                row[
                    "step_seconds"
                ]
                for row in history
            ]
        )
    )

    median_step_seconds = float(
        np.median(
            [
                row[
                    "step_seconds"
                ]
                for row in history
            ]
        )
    )

    estimated_epoch_seconds = (
        mean_step_seconds
        * batches_per_epoch
    )

    contract_max_epochs = int(
        contract[
            "training"
        ][
            "max_epochs"
        ]
    )

    estimated_full_hours = (
        estimated_epoch_seconds
        * contract_max_epochs
        / 3600.0
    )

    print(
        f"mean step seconds: "
        f"{mean_step_seconds:.4f}"
    )

    print(
        f"median step seconds: "
        f"{median_step_seconds:.4f}"
    )

    print(
        f"batches / epoch: "
        f"{batches_per_epoch}"
    )

    print(
        f"estimated epoch minutes: "
        f"{estimated_epoch_seconds / 60:.2f}"
    )

    print(
        f"naive {contract_max_epochs}-epoch "
        f"estimate: "
        f"{estimated_full_hours:.2f} hours"
    )

    print(
        "\nNOTE:"
    )

    print(
        "This is only a pilot throughput estimate."
    )

    # ========================================================
    # Step 15 — Save JSON summary
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 15 — SAVE PILOT SUMMARY")
    print("=" * 80)

    peak_allocated_mib = float(
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    peak_reserved_mib = float(
        torch.cuda.max_memory_reserved()
        / 1024**2
    )

    summary = {
        "status": (
            "B0_SMALL_TRAINING_PILOT_PASS"
        ),

        "seed": (
            SEED
        ),

        "pilot_steps": (
            PILOT_STEPS
        ),

        "batch_size": (
            BATCH_SIZE
        ),

        "window_presentations": (
            PILOT_STEPS
            * BATCH_SIZE
        ),

        "train_windows": (
            EXPECTED_TRAIN_WINDOWS
        ),

        "batches_per_epoch": (
            batches_per_epoch
        ),

        "environment": {
            "torch": (
                torch.__version__
            ),

            "cuda_runtime": (
                torch.version.cuda
            ),

            "gpu": (
                torch.cuda
                .get_device_name(0)
            ),
        },

        "training": {
            "optimizer": (
                "AdamW"
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "weight_decay": (
                WEIGHT_DECAY
            ),

            "mixed_precision": (
                MIXED_PRECISION
            ),

            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),

            "ema_decay": (
                EMA_DECAY
            ),

            "training_seconds": (
                training_seconds
            ),

            "mean_step_seconds": (
                mean_step_seconds
            ),

            "median_step_seconds": (
                median_step_seconds
            ),

            "mean_windows_per_second": float(
                throughput.mean()
            ),
        },

        "loss": {
            "first_window_mean": (
                initial_mean_loss
            ),

            "last_window_mean": (
                final_mean_loss
            ),

            "absolute_improvement": (
                trend_improvement
            ),

            "relative_improvement_fraction": (
                trend_fraction
            ),

            "minimum_batch_loss": float(
                losses.min()
            ),

            "maximum_batch_loss": float(
                losses.max()
            ),
        },

        "fixed_train_probe": {
            "initial_model_mse": (
                initial_probe_loss
            ),

            "final_model_mse": (
                final_probe_loss
            ),

            "improvement": (
                probe_improvement
            ),

            "initial_ema_mse": (
                initial_ema_probe_loss
            ),

            "final_ema_mse": (
                final_ema_probe_loss
            ),

            "ema_improvement": (
                ema_probe_improvement
            ),

            "reload_probe_mse": (
                reload_probe_loss
            ),
        },

        "gpu_memory": {
            "peak_allocated_mib": (
                peak_allocated_mib
            ),

            "peak_reserved_mib": (
                peak_reserved_mib
            ),
        },

        "time_estimate": {
            "estimated_epoch_seconds": (
                estimated_epoch_seconds
            ),

            "contract_max_epochs": (
                contract_max_epochs
            ),

            "naive_full_training_hours": (
                estimated_full_hours
            ),
        },

        "outputs": {
            "checkpoint": str(
                PILOT_CHECKPOINT_FILE
            ),

            "history_csv": str(
                PILOT_HISTORY_FILE
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

        "scientific_hyperparameter_tuning": False,

        "validation_accessed": False,

        "final_test_accessed": False,

        "full_training_started": False,
    }

    with PILOT_RESULT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print(
        f"summary JSON: "
        f"{PILOT_RESULT_FILE}"
    )

    # ========================================================
    # Cleanup
    # ========================================================

    dataset.close()

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("B0 SMALL TRAINING PILOT PASS")
    print("=" * 80)

    print(
        f"steps: "
        f"{PILOT_STEPS}"
    )

    print(
        f"window presentations: "
        f"{PILOT_STEPS * BATCH_SIZE}"
    )

    print(
        f"first {ROLLING_WINDOW}-step "
        f"mean loss: "
        f"{initial_mean_loss:.6f}"
    )

    print(
        f"last {ROLLING_WINDOW}-step "
        f"mean loss: "
        f"{final_mean_loss:.6f}"
    )

    print(
        f"fixed probe: "
        f"{initial_probe_loss:.6f}"
        f" -> "
        f"{final_probe_loss:.6f}"
    )

    print(
        f"EMA probe: "
        f"{initial_ema_probe_loss:.6f}"
        f" -> "
        f"{final_ema_probe_loss:.6f}"
    )

    print(
        f"training time: "
        f"{training_seconds / 60:.2f} min"
    )

    print(
        f"estimated epoch: "
        f"{estimated_epoch_seconds / 60:.2f} min"
    )

    print(
        f"naive full-training estimate: "
        f"{estimated_full_hours:.2f} h"
    )

    print(
        f"peak GPU allocated: "
        f"{peak_allocated_mib:.0f} MiB"
    )

    print(
        "Validation accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  JSON: "
        f"{PILOT_RESULT_FILE}"
    )

    print(
        f"  CSV: "
        f"{PILOT_HISTORY_FILE}"
    )

    print(
        f"  checkpoint: "
        f"{PILOT_CHECKPOINT_FILE}"
    )

    print(
        f"  loss figure: "
        f"{LOSS_FIGURE_FILE}"
    )

    print(
        f"  gradient figure: "
        f"{GRADIENT_FIGURE_FILE}"
    )

    print(
        f"  throughput figure: "
        f"{THROUGHPUT_FIGURE_FILE}"
    )

    print(
        f"  memory figure: "
        f"{MEMORY_FIGURE_FILE}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "REVIEW PILOT BEFORE "
        "FULL B0 TRAINING"
    )


if __name__ == "__main__":
    main()