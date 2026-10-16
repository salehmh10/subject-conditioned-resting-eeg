from __future__ import annotations

import copy
import json
import math
import random
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

SANITY_RESULT_FILE = (
    B0_METRIC_DIR
    / "b0_one_batch_diffusion_sanity_v1.json"
)

SCHEDULE_FIGURE_FILE = (
    B0_FIGURE_DIR
    / "b0_cosine_diffusion_schedule_v1.png"
)


# ============================================================
# Fixed sanity configuration
# ============================================================

SEED = 20260907

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

# Must match the frozen B0 contract.
BATCH_SIZE = 64

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

GRADIENT_CLIP_NORM = 1.0

EMA_DECAY = 0.9999

TIMESTEPS = 1000
COSINE_S = 0.008

MIXED_PRECISION = True


# ============================================================
# Determinism
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Cosine DDPM schedule
# ============================================================

def build_cosine_schedule(
    timesteps: int,
    cosine_s: float,
) -> dict[str, torch.Tensor]:
    """
    Build a cosine DDPM schedule.

    alpha_bar(t) =
        cos^2(
            ((t / T + s) / (1 + s))
            * pi / 2
        )

    The returned tensors are initially stored on CPU
    in float64 for stable schedule construction.
    """

    steps = timesteps + 1

    x = torch.linspace(
        0,
        timesteps,
        steps,
        dtype=torch.float64,
    )

    alpha_bar = torch.cos(
        (
            (
                x / timesteps
                + cosine_s
            )
            / (
                1.0
                + cosine_s
            )
        )
        * math.pi
        / 2.0
    ) ** 2

    alpha_bar = (
        alpha_bar
        / alpha_bar[0]
    )

    betas = (
        1.0
        - (
            alpha_bar[1:]
            / alpha_bar[:-1]
        )
    )

    betas = torch.clamp(
        betas,
        min=1e-8,
        max=0.999,
    )

    alphas = (
        1.0
        - betas
    )

    alpha_cumprod = torch.cumprod(
        alphas,
        dim=0,
    )

    sqrt_alpha_cumprod = torch.sqrt(
        alpha_cumprod
    )

    sqrt_one_minus_alpha_cumprod = torch.sqrt(
        1.0
        - alpha_cumprod
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "sqrt_alpha_cumprod": (
            sqrt_alpha_cumprod
        ),
        "sqrt_one_minus_alpha_cumprod": (
            sqrt_one_minus_alpha_cumprod
        ),
    }


# ============================================================
# q(x_t | x_0)
# ============================================================

def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_cumprod: torch.Tensor,
    sqrt_one_minus_alpha_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Forward diffusion:

        x_t =
            sqrt(alpha_bar_t) * x_0
            +
            sqrt(1-alpha_bar_t) * epsilon
    """

    if x0.shape != noise.shape:
        raise RuntimeError(
            "x0 and noise shapes differ."
        )

    if timesteps.ndim != 1:
        raise RuntimeError(
            "timesteps must have shape (B,)"
        )

    if timesteps.shape[0] != x0.shape[0]:
        raise RuntimeError(
            "timestep batch size mismatch."
        )

    # Schedule tensors and timestep indices MUST
    # be on the same device.
    if (
        sqrt_alpha_cumprod.device
        != timesteps.device
    ):
        raise RuntimeError(
            "sqrt_alpha_cumprod and timesteps "
            "must be on the same device."
        )

    if (
        sqrt_one_minus_alpha_cumprod.device
        != timesteps.device
    ):
        raise RuntimeError(
            "sqrt_one_minus_alpha_cumprod and "
            "timesteps must be on the same device."
        )

    sqrt_alpha = (
        sqrt_alpha_cumprod[
            timesteps
        ]
        .to(
            dtype=x0.dtype
        )
        .view(
            -1,
            1,
            1,
        )
    )

    sqrt_one_minus = (
        sqrt_one_minus_alpha_cumprod[
            timesteps
        ]
        .to(
            dtype=x0.dtype
        )
        .view(
            -1,
            1,
            1,
        )
    )

    xt = (
        sqrt_alpha
        * x0
        + sqrt_one_minus
        * noise
    )

    return xt


# ============================================================
# Gradient helpers
# ============================================================

def assert_finite_gradients(
    model: nn.Module,
) -> tuple[int, float]:
    """
    Verify that gradients exist and remain finite.
    """

    gradient_parameter_count = 0

    squared_norm_sum = 0.0

    for name, parameter in (
        model.named_parameters()
    ):
        if parameter.grad is None:
            continue

        gradient_parameter_count += 1

        if not torch.isfinite(
            parameter.grad
        ).all():
            raise RuntimeError(
                f"Nonfinite gradient: {name}"
            )

        grad_norm = (
            parameter.grad
            .detach()
            .float()
            .norm(2)
            .item()
        )

        squared_norm_sum += (
            grad_norm ** 2
        )

    if gradient_parameter_count == 0:
        raise RuntimeError(
            "No gradients were produced."
        )

    global_norm = (
        squared_norm_sum
        ** 0.5
    )

    return (
        gradient_parameter_count,
        global_norm,
    )


# ============================================================
# Parameter-change helper
# ============================================================

def tensor_change(
    before: torch.Tensor,
    after: torch.Tensor,
) -> float:
    return float(
        torch.mean(
            torch.abs(
                before.float()
                - after.float()
            )
        ).item()
    )


# ============================================================
# Save schedule figure
# ============================================================

def save_schedule_figure(
    schedule: dict[str, torch.Tensor],
) -> None:
    """
    Save a visual check of beta and alpha_bar.
    """

    B0_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(10, 6)
    )

    ax = figure.add_subplot(111)

    timesteps_axis = np.arange(
        TIMESTEPS
    )

    alpha_bar = (
        schedule[
            "alpha_cumprod"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    betas = (
        schedule[
            "betas"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    ax.plot(
        timesteps_axis,
        alpha_bar,
        label="alpha_bar",
    )

    ax.plot(
        timesteps_axis,
        betas,
        label="beta",
    )

    ax.set_xlabel(
        "Diffusion timestep"
    )

    ax.set_ylabel(
        "Value"
    )

    ax.set_title(
        "B0 Cosine DDPM Schedule"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        SCHEDULE_FIGURE_FILE,
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
        "B0 ONE-BATCH DDPM TRAINING SANITY"
    )
    print("=" * 80)

    print(
        "\nThis is NOT full B0 training."
    )

    print(
        "Exactly one real Train mini-batch "
        "and one optimizer step are executed."
    )

    print(
        "\nValidation accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    # --------------------------------------------------------
    # Create output folders early
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

    set_seed(SEED)

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

    # ========================================================
    # Step 2 — Load B0 contract
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — LOAD B0 CONTRACT")
    print("=" * 80)

    if not CONTRACT_FILE.exists():
        raise FileNotFoundError(
            CONTRACT_FILE
        )

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

    if int(
        training_config[
            "batch_size"
        ]
    ) != BATCH_SIZE:
        raise RuntimeError(
            "Batch size differs from frozen contract."
        )

    if float(
        training_config[
            "learning_rate"
        ]
    ) != LEARNING_RATE:
        raise RuntimeError(
            "Learning rate differs from frozen contract."
        )

    if float(
        training_config[
            "weight_decay"
        ]
    ) != WEIGHT_DECAY:
        raise RuntimeError(
            "Weight decay differs from frozen contract."
        )

    if int(
        diffusion_config[
            "timesteps"
        ]
    ) != TIMESTEPS:
        raise RuntimeError(
            "Timestep count differs from contract."
        )

    if diffusion_config[
        "prediction_target"
    ] != "epsilon":
        raise RuntimeError(
            "B0 must predict epsilon."
        )

    if diffusion_config[
        "beta_schedule"
    ] != "cosine":
        raise RuntimeError(
            "B0 contract does not request cosine schedule."
        )

    print(
        "contract consistency: PASS"
    )

    # ========================================================
    # Step 3 — Build cosine DDPM schedule
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — BUILD COSINE DDPM SCHEDULE")
    print("=" * 80)

    schedule = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    betas = schedule[
        "betas"
    ]

    alpha_cumprod = schedule[
        "alpha_cumprod"
    ]

    if betas.shape != (
        TIMESTEPS,
    ):
        raise RuntimeError(
            "Wrong beta schedule shape."
        )

    if not torch.isfinite(
        betas
    ).all():
        raise RuntimeError(
            "Nonfinite betas."
        )

    if torch.any(
        betas <= 0
    ) or torch.any(
        betas >= 1
    ):
        raise RuntimeError(
            "Betas outside (0, 1)."
        )

    if not torch.all(
        alpha_cumprod[1:]
        < alpha_cumprod[:-1]
    ):
        raise RuntimeError(
            "alpha_cumprod is not strictly decreasing."
        )

    print(
        f"beta[0]: "
        f"{betas[0].item():.8f}"
    )

    print(
        f"beta[-1]: "
        f"{betas[-1].item():.8f}"
    )

    print(
        f"alpha_bar[0]: "
        f"{alpha_cumprod[0].item():.8f}"
    )

    print(
        f"alpha_bar[-1]: "
        f"{alpha_cumprod[-1].item():.12f}"
    )

    print(
        "cosine schedule: PASS"
    )

    # --------------------------------------------------------
    # Save schedule visualization
    # --------------------------------------------------------

    save_schedule_figure(
        schedule
    )

    print(
        f"schedule figure saved: "
        f"{SCHEDULE_FIGURE_FILE}"
    )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # move schedule tensors onto GPU ONCE.
    # --------------------------------------------------------

    schedule_device = {
        name: tensor.to(
            device=device,
            dtype=torch.float32,
        )
        for name, tensor
        in schedule.items()
    }

    print(
        f"training schedule device: "
        f"{schedule_device['betas'].device}"
    )

    # ========================================================
    # Step 4 — Load one real Train mini-batch
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — LOAD ONE REAL TRAIN MINI-BATCH")
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
            "Unexpected subject count."
        )

    dataset = (
        CachedRealTrainWindowDataset(
            records=records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )

    batch = next(
        iter(loader)
    )

    x0 = batch[
        "x"
    ].to(
        device,
        non_blocking=True,
    )

    eye_state = batch[
        "y"
    ].to(
        device,
        non_blocking=True,
    )

    if x0.shape != (
        BATCH_SIZE,
        19,
        512,
    ):
        raise RuntimeError(
            f"Wrong real batch shape: "
            f"{x0.shape}"
        )

    if not torch.isfinite(
        x0
    ).all():
        raise RuntimeError(
            "Nonfinite real batch."
        )

    label_counts = torch.bincount(
        eye_state,
        minlength=2,
    )

    print(
        f"x0 shape: "
        f"{tuple(x0.shape)}"
    )

    print(
        f"EO in batch: "
        f"{int(label_counts[0])}"
    )

    print(
        f"EC in batch: "
        f"{int(label_counts[1])}"
    )

    print(
        f"x0 mean: "
        f"{x0.mean().item():.6f}"
    )

    print(
        f"x0 std: "
        f"{x0.std(unbiased=False).item():.6f}"
    )

    # ========================================================
    # Step 5 — Forward diffusion q(x_t | x_0)
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — FORWARD DIFFUSION q(x_t | x_0)")
    print("=" * 80)

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

    if xt.shape != x0.shape:
        raise RuntimeError(
            "x_t shape mismatch."
        )

    if not torch.isfinite(
        xt
    ).all():
        raise RuntimeError(
            "Nonfinite x_t."
        )

    print(
        f"timestep min: "
        f"{int(timesteps.min())}"
    )

    print(
        f"timestep max: "
        f"{int(timesteps.max())}"
    )

    print(
        f"epsilon mean: "
        f"{epsilon.mean().item():.6f}"
    )

    print(
        f"epsilon std: "
        f"{epsilon.std(unbiased=False).item():.6f}"
    )

    print(
        f"x_t mean: "
        f"{xt.mean().item():.6f}"
    )

    print(
        f"x_t std: "
        f"{xt.std(unbiased=False).item():.6f}"
    )

    print(
        "forward diffusion: PASS"
    )

    # ========================================================
    # Step 6 — Build model / optimizer / EMA
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — BUILD TRAINING OBJECTS")
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

    amp_enabled = bool(
        MIXED_PRECISION
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    print(
        "optimizer: AdamW"
    )

    print(
        f"learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"mixed precision: "
        f"{amp_enabled}"
    )

    print(
        f"EMA decay: "
        f"{EMA_DECAY}"
    )

    # --------------------------------------------------------
    # Select one parameter for update proof
    # --------------------------------------------------------

    tracked_name = None
    tracked_parameter = None

    for name, parameter in (
        model.named_parameters()
    ):
        if parameter.requires_grad:
            tracked_name = name
            tracked_parameter = parameter
            break

    if tracked_parameter is None:
        raise RuntimeError(
            "No trainable parameter found."
        )

    parameter_before = (
        tracked_parameter
        .detach()
        .clone()
    )

    ema_parameter_before = (
        dict(
            ema_model.named_parameters()
        )[
            tracked_name
        ]
        .detach()
        .clone()
    )

    # ========================================================
    # Step 7 — ONE forward/backward optimizer step
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — ONE B0 TRAINING STEP")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp_enabled,
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

    if epsilon_pred.shape != x0.shape:
        raise RuntimeError(
            "epsilon prediction shape mismatch."
        )

    if not torch.isfinite(
        epsilon_pred
    ).all():
        raise RuntimeError(
            "Nonfinite epsilon prediction."
        )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Nonfinite diffusion loss."
        )

    print(
        f"initial diffusion MSE: "
        f"{loss.item():.6f}"
    )

    grad_scaler.scale(
        loss
    ).backward()

    # Unscale before inspecting/clipping gradients.
    grad_scaler.unscale_(
        optimizer
    )

    (
        gradient_parameter_count,
        gradient_norm_before_clip,
    ) = assert_finite_gradients(
        model
    )

    clipped_norm = (
        torch.nn.utils.clip_grad_norm_(
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

    print(
        f"parameters with gradients: "
        f"{gradient_parameter_count}"
    )

    print(
        f"global grad norm before clip: "
        f"{gradient_norm_before_clip:.6f}"
    )

    print(
        f"clip threshold: "
        f"{GRADIENT_CLIP_NORM}"
    )

    grad_scaler.step(
        optimizer
    )

    grad_scaler.update()

    # ========================================================
    # Step 8 — EMA update
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 8 — EMA UPDATE")
    print("=" * 80)

    with torch.no_grad():
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
                EMA_DECAY
            )

            ema_parameter.add_(
                model_parameter,
                alpha=(
                    1.0
                    - EMA_DECAY
                ),
            )

    # ========================================================
    # Step 9 — Verify parameter updates
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 9 — VERIFY PARAMETER UPDATES")
    print("=" * 80)

    parameter_after = (
        dict(
            model.named_parameters()
        )[
            tracked_name
        ]
        .detach()
    )

    ema_parameter_after = (
        dict(
            ema_model.named_parameters()
        )[
            tracked_name
        ]
        .detach()
    )

    optimizer_change = tensor_change(
        parameter_before,
        parameter_after,
    )

    ema_change = tensor_change(
        ema_parameter_before,
        ema_parameter_after,
    )

    print(
        f"tracked parameter: "
        f"{tracked_name}"
    )

    print(
        f"mean absolute optimizer update: "
        f"{optimizer_change:.12e}"
    )

    print(
        f"mean absolute EMA update: "
        f"{ema_change:.12e}"
    )

    if optimizer_change <= 0:
        raise RuntimeError(
            "Optimizer step did not change parameters."
        )

    if ema_change <= 0:
        raise RuntimeError(
            "EMA did not change."
        )

    print(
        "optimizer update: PASS"
    )

    print(
        "EMA update: PASS"
    )

    # ========================================================
    # Step 10 — Post-step forward sanity
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 10 — POST-STEP FORWARD SANITY")
    print("=" * 80)

    model.eval()

    with torch.no_grad():

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            epsilon_pred_after = (
                model(
                    xt,
                    timesteps,
                    eye_state,
                )
            )

            loss_after = loss_fn(
                epsilon_pred_after,
                epsilon,
            )

    if not torch.isfinite(
        loss_after
    ):
        raise RuntimeError(
            "Post-step loss is nonfinite."
        )

    print(
        f"post-step MSE on same batch: "
        f"{loss_after.item():.6f}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "A single optimizer step is not expected "
        "to prove model quality."
    )

    print(
        "This step proves finite forward/backward "
        "training mechanics only."
    )

    torch.cuda.synchronize()

    peak_allocated_mib = float(
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    peak_reserved_mib = float(
        torch.cuda.max_memory_reserved()
        / 1024**2
    )

    print(
        f"\npeak GPU allocated: "
        f"{peak_allocated_mib:.2f} MiB"
    )

    print(
        f"peak GPU reserved: "
        f"{peak_reserved_mib:.2f} MiB"
    )

    # ========================================================
    # Step 11 — Save JSON result
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 11 — SAVE SANITY OUTPUTS")
    print("=" * 80)

    sanity_result = {
        "status": (
            "B0_ONE_BATCH_DDPM_TRAINING_SANITY_PASS"
        ),

        "seed": SEED,

        "environment": {
            "torch_version": (
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

        "batch": {
            "batch_size": int(
                BATCH_SIZE
            ),

            "shape": [
                int(value)
                for value
                in x0.shape
            ],

            "eo_count": int(
                label_counts[0]
                .item()
            ),

            "ec_count": int(
                label_counts[1]
                .item()
            ),

            "x0_mean": float(
                x0.mean()
                .item()
            ),

            "x0_std": float(
                x0.std(
                    unbiased=False
                )
                .item()
            ),
        },

        "diffusion": {
            "timesteps": int(
                TIMESTEPS
            ),

            "cosine_s": float(
                COSINE_S
            ),

            "schedule_device": str(
                schedule_device[
                    "betas"
                ].device
            ),

            "beta_first": float(
                betas[0]
                .item()
            ),

            "beta_last": float(
                betas[-1]
                .item()
            ),

            "alpha_bar_first": float(
                alpha_cumprod[0]
                .item()
            ),

            "alpha_bar_last": float(
                alpha_cumprod[-1]
                .item()
            ),

            "sampled_timestep_min": int(
                timesteps.min()
                .item()
            ),

            "sampled_timestep_max": int(
                timesteps.max()
                .item()
            ),

            "epsilon_mean": float(
                epsilon.mean()
                .item()
            ),

            "epsilon_std": float(
                epsilon.std(
                    unbiased=False
                )
                .item()
            ),

            "xt_mean": float(
                xt.mean()
                .item()
            ),

            "xt_std": float(
                xt.std(
                    unbiased=False
                )
                .item()
            ),
        },

        "training_step": {
            "initial_mse": float(
                loss.item()
            ),

            "post_step_mse": float(
                loss_after.item()
            ),

            "gradient_parameter_count": int(
                gradient_parameter_count
            ),

            "gradient_norm_before_clip": float(
                gradient_norm_before_clip
            ),

            "gradient_clip_norm": float(
                GRADIENT_CLIP_NORM
            ),

            "optimizer_parameter_update": float(
                optimizer_change
            ),

            "ema_parameter_update": float(
                ema_change
            ),

            "mixed_precision": bool(
                amp_enabled
            ),

            "optimizer": "AdamW",

            "learning_rate": float(
                LEARNING_RATE
            ),

            "weight_decay": float(
                WEIGHT_DECAY
            ),

            "ema_decay": float(
                EMA_DECAY
            ),
        },

        "memory": {
            "peak_allocated_mib": (
                peak_allocated_mib
            ),

            "peak_reserved_mib": (
                peak_reserved_mib
            ),
        },

        "outputs": {
            "schedule_figure": str(
                SCHEDULE_FIGURE_FILE
            ),

            "sanity_json": str(
                SANITY_RESULT_FILE
            ),
        },

        "validation_accessed": False,

        "final_test_accessed": False,

        "full_training_started": False,
    }

    with SANITY_RESULT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            sanity_result,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print(
        f"sanity JSON saved: "
        f"{SANITY_RESULT_FILE}"
    )

    print(
        f"schedule figure saved: "
        f"{SCHEDULE_FIGURE_FILE}"
    )

    # ========================================================
    # Cleanup
    # ========================================================

    dataset.close()

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("B0 ONE-BATCH DDPM TRAINING SANITY PASS")
    print("=" * 80)

    print(
        f"real Train batch: "
        f"{BATCH_SIZE}"
    )

    print(
        "epsilon target: PASS"
    )

    print(
        "cosine schedule: PASS"
    )

    print(
        "schedule moved to GPU: PASS"
    )

    print(
        "mixed precision forward: PASS"
    )

    print(
        "backward pass: PASS"
    )

    print(
        "finite gradients: PASS"
    )

    print(
        "gradient clipping: PASS"
    )

    print(
        "optimizer step: PASS"
    )

    print(
        "EMA update: PASS"
    )

    print(
        "Validation accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nFull B0 training started: NO"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  JSON: "
        f"{SANITY_RESULT_FILE}"
    )

    print(
        f"  Figure: "
        f"{SCHEDULE_FIGURE_FILE}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "SMALL B0 TRAINING PILOT"
    )


if __name__ == "__main__":
    main()