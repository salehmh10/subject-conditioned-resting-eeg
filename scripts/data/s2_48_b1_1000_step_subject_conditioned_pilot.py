from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
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
    build_subject_diverse_balanced_batch,
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


# ============================================================
# Pilot output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "short_pilot_v1"
)

CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
LOG_DIR = RUN_ROOT / "logs"
METRIC_DIR = RUN_ROOT / "metrics"
GENERATED_DIR = RUN_ROOT / "generated"
FIGURE_DIR = RUN_ROOT / "figures"

PILOT_CHECKPOINT = (
    CHECKPOINT_DIR
    / "b1_short_pilot_final_v1.pt"
)

HISTORY_CSV = (
    LOG_DIR
    / "b1_short_pilot_history_v1.csv"
)

PROTOCOL_SNAPSHOT = (
    METRIC_DIR
    / "b1_short_pilot_protocol_snapshot_v1.json"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "b1_short_pilot_summary_v1.json"
)

GENERATED_NPZ = (
    GENERATED_DIR
    / "b1_short_pilot_samples_v1.npz"
)

LOSS_FIGURE = (
    FIGURE_DIR
    / "b1_short_pilot_loss_v1.png"
)

SUBJECT_GAP_FIGURE = (
    FIGURE_DIR
    / "b1_short_pilot_subject_contrast_v1.png"
)

SAMPLING_SCALE_FIGURE = (
    FIGURE_DIR
    / "b1_short_pilot_sampling_scale_v1.png"
)

SAMPLING_ALPHA_FIGURE = (
    FIGURE_DIR
    / "b1_short_pilot_alpha_effect_v1.png"
)


# ============================================================
# Frozen pilot configuration
# ============================================================

SEED = 20260908
TRAIN_LOADER_SEED = 20260918
TRAIN_NOISE_SEED = 20260919
FIXED_PROBE_SEED = 20260920
SAMPLING_SEED = 20260921

PILOT_SUCCESSFUL_STEPS = 1000
BATCH_SIZE = 64

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

MIXED_PRECISION = True
MAX_TOTAL_AMP_SKIPS = 10
MAX_CONSECUTIVE_AMP_SKIPS = 3

TIMESTEPS = 1000
COSINE_S = 0.008

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

SUBJECT_EMBEDDING_DIM = 128
CONDITION_DIM = 256

SAMPLING_SUBJECT_COUNT = 16

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz",
    "T7", "T8", "C3", "C4", "Cz", "P7", "P8",
    "P3", "P4", "Pz", "O1", "O2",
]

CHANNEL_INDEX = {
    name: index
    for index, name
    in enumerate(CHANNEL_NAMES)
}

ALPHA_BAND = (8.0, 13.0)


# ============================================================
# Reproducibility / IO helpers
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


def save_history_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            "Cannot save empty pilot history."
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


# ============================================================
# Gradient helpers
# ============================================================

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
        else float("nan")
    )

    return (
        finite,
        grad_count,
        norm,
        bad_names,
    )


def parameter_group_grad_norm(
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


# ============================================================
# Reverse diffusion sampler for B1
# ============================================================

def build_reverse_schedule(
    schedule_cpu: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    betas = schedule_cpu[
        "betas"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    alphas = schedule_cpu[
        "alphas"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    alpha_bar = schedule_cpu[
        "alpha_cumprod"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    alpha_bar_prev = torch.cat(
        [
            torch.ones(
                1,
                device=device,
                dtype=torch.float32,
            ),
            alpha_bar[
                :-1
            ],
        ],
        dim=0,
    )

    denominator = (
        1.0
        - alpha_bar
    )

    coef_x0 = (
        betas
        * torch.sqrt(
            alpha_bar_prev
        )
        / denominator
    )

    coef_xt = (
        torch.sqrt(
            alphas
        )
        * (
            1.0
            - alpha_bar_prev
        )
        / denominator
    )

    posterior_variance = (
        betas
        * (
            1.0
            - alpha_bar_prev
        )
        / denominator
    )

    posterior_variance = (
        torch.clamp(
            posterior_variance,
            min=0.0,
        )
    )

    return {
        "alpha_bar": (
            alpha_bar
        ),
        "coef_x0": (
            coef_x0
        ),
        "coef_xt": (
            coef_xt
        ),
        "posterior_variance": (
            posterior_variance
        ),
    }


@torch.inference_mode()
def sample_b1_subjects(
    *,
    model: B1SubjectStateUNet1D,
    subject_indices: list[int],
    eye_state_value: int,
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> np.ndarray:
    if eye_state_value not in (
        0,
        1,
    ):
        raise ValueError(
            "eye_state_value must be 0 or 1."
        )

    batch_size = len(
        subject_indices
    )

    if batch_size <= 0:
        raise ValueError(
            "subject_indices must be non-empty."
        )

    model.eval()

    generator = torch.Generator(
        device="cuda"
    )

    generator.manual_seed(
        seed
    )

    shape = (
        batch_size,
        N_CHANNELS,
        N_TIMES,
    )

    x = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    state = torch.full(
        (
            batch_size,
        ),
        eye_state_value,
        device=device,
        dtype=torch.long,
    )

    subjects = torch.tensor(
        subject_indices,
        device=device,
        dtype=torch.long,
    )

    alpha_bar = reverse_schedule[
        "alpha_bar"
    ]

    coef_x0 = reverse_schedule[
        "coef_x0"
    ]

    coef_xt = reverse_schedule[
        "coef_xt"
    ]

    posterior_variance = reverse_schedule[
        "posterior_variance"
    ]

    for t in reversed(
        range(
            TIMESTEPS
        )
    ):
        timestep = torch.full(
            (
                batch_size,
            ),
            t,
            device=device,
            dtype=torch.long,
        )

        # FP32 sampler by design.
        v_pred = model(
            x,
            timestep,
            state,
            subjects,
        )

        if not torch.isfinite(
            v_pred
        ).all():
            raise RuntimeError(
                "Nonfinite B1 v prediction "
                f"during sampling at t={t}."
            )

        sqrt_ab = torch.sqrt(
            alpha_bar[
                t
            ]
        )

        sqrt_omab = torch.sqrt(
            1.0
            - alpha_bar[
                t
            ]
        )

        x0_pred = (
            sqrt_ab
            * x
            - sqrt_omab
            * v_pred
        )

        model_mean = (
            coef_x0[
                t
            ]
            * x0_pred
            + coef_xt[
                t
            ]
            * x
        )

        if t > 0:
            noise = torch.randn(
                shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

            x = (
                model_mean
                + torch.sqrt(
                    posterior_variance[
                        t
                    ]
                )
                * noise
            )
        else:
            x = model_mean

        if not torch.isfinite(
            x
        ).all():
            raise RuntimeError(
                "Nonfinite B1 reverse state "
                f"during sampling at t={t}."
            )

    return (
        x.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )


# ============================================================
# Spectral / scale helpers
# ============================================================

def inverse_normalize(
    eeg_norm: np.ndarray,
    scaler: dict[str, Any],
) -> np.ndarray:
    means = np.asarray(
        scaler[
            "means_uv"
        ],
        dtype=np.float64,
    )

    stds = np.asarray(
        scaler[
            "stds_uv"
        ],
        dtype=np.float64,
    )

    return (
        eeg_norm.astype(
            np.float64
        )
        * stds[
            None,
            :,
            None,
        ]
        + means[
            None,
            :,
            None,
        ]
    )


def alpha_bandpower_db(
    eeg_uv: np.ndarray,
) -> np.ndarray:
    frequencies, psd = welch(
        eeg_uv,
        fs=SFREQ,
        window="hann",
        nperseg=256,
        noverlap=128,
        detrend="constant",
        scaling="density",
        axis=-1,
    )

    low, high = (
        ALPHA_BAND
    )

    mask = (
        (
            frequencies
            >= low
        )
        & (
            frequencies
            <= high
        )
    )

    power = np.trapezoid(
        psd[
            ...,
            mask
        ],
        frequencies[
            mask
        ],
        axis=-1,
    )

    tiny = np.finfo(
        np.float64
    ).tiny

    return (
        10.0
        * np.log10(
            np.maximum(
                power,
                tiny,
            )
        )
    )


def sampling_metrics(
    eo_norm: np.ndarray,
    ec_norm: np.ndarray,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    eo_uv = inverse_normalize(
        eo_norm,
        scaler,
    )

    ec_uv = inverse_normalize(
        ec_norm,
        scaler,
    )

    eo_alpha = alpha_bandpower_db(
        eo_uv
    )

    ec_alpha = alpha_bandpower_db(
        ec_uv
    )

    alpha_effect = (
        np.median(
            ec_alpha,
            axis=0,
        )
        - np.median(
            eo_alpha,
            axis=0,
        )
    )

    return {
        "eo_normalized_std": float(
            np.std(
                eo_norm,
                dtype=np.float64,
            )
        ),
        "ec_normalized_std": float(
            np.std(
                ec_norm,
                dtype=np.float64,
            )
        ),
        "eo_uv_std": float(
            np.std(
                eo_uv,
                dtype=np.float64,
            )
        ),
        "ec_uv_std": float(
            np.std(
                ec_uv,
                dtype=np.float64,
            )
        ),
        "whole_head_alpha_effect_db": float(
            np.median(
                alpha_effect
            )
        ),
        "o1_alpha_effect_db": float(
            alpha_effect[
                CHANNEL_INDEX[
                    "O1"
                ]
            ]
        ),
        "o2_alpha_effect_db": float(
            alpha_effect[
                CHANNEL_INDEX[
                    "O2"
                ]
            ]
        ),
        "alpha_effect_by_channel_db": [
            float(
                value
            )
            for value
            in alpha_effect
        ],
        "eo_uv": eo_uv.astype(
            np.float32
        ),
        "ec_uv": ec_uv.astype(
            np.float32
        ),
    }


# ============================================================
# Fixed denoising probe
# ============================================================

@torch.inference_mode()
def evaluate_fixed_probe(
    *,
    model: B1SubjectStateUNet1D,
    x0: torch.Tensor,
    state: torch.Tensor,
    subject: torch.Tensor,
    wrong_subject: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    generator = torch.Generator(
        device="cuda"
    )

    generator.manual_seed(
        FIXED_PROBE_SEED
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

    correct_mse = float(
        torch.mean(
            (
                correct_pred.float()
                - target_v.float()
            )
            ** 2
        ).item()
    )

    wrong_mse = float(
        torch.mean(
            (
                wrong_pred.float()
                - target_v.float()
            )
            ** 2
        ).item()
    )

    output_difference_mean = float(
        torch.mean(
            torch.abs(
                correct_pred.float()
                - wrong_pred.float()
            )
        ).item()
    )

    output_difference_max = float(
        torch.max(
            torch.abs(
                correct_pred.float()
                - wrong_pred.float()
            )
        ).item()
    )

    return {
        "correct_subject_v_mse": (
            correct_mse
        ),
        "wrong_subject_v_mse": (
            wrong_mse
        ),
        "wrong_minus_correct_v_mse": (
            wrong_mse
            - correct_mse
        ),
        "correct_vs_wrong_output_mean_abs": (
            output_difference_mean
        ),
        "correct_vs_wrong_output_max_abs": (
            output_difference_max
        ),
    }


@torch.inference_mode()
def subject_condition_statistics(
    model: B1SubjectStateUNet1D,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    subject_ids = torch.arange(
        EXPECTED_SUBJECTS,
        device=device,
        dtype=torch.long,
    )

    residuals = (
        model.subject_condition(
            subject_ids
        )
        .float()
    )

    residual_norms = torch.linalg.vector_norm(
        residuals,
        dim=1,
    )

    feature_std = torch.std(
        residuals,
        dim=0,
        unbiased=False,
    )

    centered = (
        residuals
        - torch.mean(
            residuals,
            dim=0,
            keepdim=True,
        )
    )

    rms_subject_variation = float(
        torch.sqrt(
            torch.mean(
                centered
                * centered
            )
        ).item()
    )

    return {
        "subject_residual_norm_mean": float(
            torch.mean(
                residual_norms
            ).item()
        ),
        "subject_residual_norm_median": float(
            torch.median(
                residual_norms
            ).item()
        ),
        "subject_residual_norm_max": float(
            torch.max(
                residual_norms
            ).item()
        ),
        "subject_residual_feature_std_mean": float(
            torch.mean(
                feature_std
            ).item()
        ),
        "subject_residual_rms_across_subjects": (
            rms_subject_variation
        ),
    }


# ============================================================
# Figures
# ============================================================

def save_loss_figure(
    history: list[dict[str, Any]],
) -> None:
    steps = np.asarray(
        [
            row[
                "successful_step"
            ]
            for row in history
        ],
        dtype=np.int64,
    )

    losses = np.asarray(
        [
            row[
                "v_mse"
            ]
            for row in history
        ],
        dtype=np.float64,
    )

    rolling = pd.Series(
        losses
    ).rolling(
        50,
        min_periods=1,
    ).mean().to_numpy()

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        steps,
        losses,
        alpha=0.35,
        label="step v-MSE",
    )

    ax.plot(
        steps,
        rolling,
        linewidth=2,
        label="50-step rolling mean",
    )

    ax.set_xlabel(
        "Successful optimizer step"
    )

    ax.set_ylabel(
        "v-MSE"
    )

    ax.set_title(
        "B1 1000-Step Pilot — Train v-MSE"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        LOSS_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_subject_gap_figure(
    pre_probe: dict[str, float],
    post_probe: dict[str, float],
) -> None:
    labels = [
        "Correct subject",
        "Wrong subject",
    ]

    pre = [
        pre_probe[
            "correct_subject_v_mse"
        ],
        pre_probe[
            "wrong_subject_v_mse"
        ],
    ]

    post = [
        post_probe[
            "correct_subject_v_mse"
        ],
        post_probe[
            "wrong_subject_v_mse"
        ],
    ]

    x = np.arange(
        len(
            labels
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    ax.bar(
        x
        - width
        / 2,
        pre,
        width,
        label="Before pilot",
    )

    ax.bar(
        x
        + width
        / 2,
        post,
        width,
        label="After pilot",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Fixed-probe v-MSE"
    )

    ax.set_title(
        "B1 Pilot — Correct vs Wrong Subject Denoising"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SUBJECT_GAP_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_sampling_scale_figure(
    pre: dict[str, Any],
    post: dict[str, Any],
) -> None:
    labels = [
        "EO",
        "EC",
    ]

    pre_values = [
        pre[
            "eo_normalized_std"
        ],
        pre[
            "ec_normalized_std"
        ],
    ]

    post_values = [
        post[
            "eo_normalized_std"
        ],
        post[
            "ec_normalized_std"
        ],
    ]

    x = np.arange(
        len(
            labels
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    ax.bar(
        x
        - width
        / 2,
        pre_values,
        width,
        label="Before pilot",
    )

    ax.bar(
        x
        + width
        / 2,
        post_values,
        width,
        label="After pilot",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Normalized std = 1",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Generated normalized std"
    )

    ax.set_title(
        "B1 Pilot — Sampling Scale"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SAMPLING_SCALE_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_sampling_alpha_figure(
    pre: dict[str, Any],
    post: dict[str, Any],
) -> None:
    labels = [
        "Whole head",
        "O1",
        "O2",
    ]

    pre_values = [
        pre[
            "whole_head_alpha_effect_db"
        ],
        pre[
            "o1_alpha_effect_db"
        ],
        pre[
            "o2_alpha_effect_db"
        ],
    ]

    post_values = [
        post[
            "whole_head_alpha_effect_db"
        ],
        post[
            "o1_alpha_effect_db"
        ],
        post[
            "o2_alpha_effect_db"
        ],
    ]

    x = np.arange(
        len(
            labels
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.bar(
        x
        - width
        / 2,
        pre_values,
        width,
        label="Before pilot",
    )

    ax.bar(
        x
        + width
        / 2,
        post_values,
        width,
        label="After pilot",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "EC - EO alpha power (dB)"
    )

    ax.set_title(
        "B1 Pilot — Eye-State Alpha Effect"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SAMPLING_ALPHA_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — B1 1000-STEP SUBJECT-CONDITIONED FINE-TUNING PILOT"
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
        "  B1 architecture + exact B0 warm-start sanity: PASSED"
    )

    print(
        "  B1 1000-step Train-only pilot: NOW"
    )

    print(
        "  B1 full 30-epoch fine-tuning: NOT STARTED"
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
        "\nPilot design:"
    )

    print(
        f"  successful optimizer steps: "
        f"{PILOT_SUCCESSFUL_STEPS}"
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
        "  initialization: ZERO-UPDATE B1 warm-start template"
    )

    print(
        "  optimizer state reused: NO"
    )

    print(
        "  prediction target: v"
    )

    print(
        "  EMA: OFF"
    )

    print(
        "  Validation EEG accessed: NO"
    )

    print(
        "  Final Test accessed: NO"
    )

    print(
        "  pilot checkpoint reusable for scientific full run: NO"
    )

    for directory in (
        CHECKPOINT_DIR,
        LOG_DIR,
        METRIC_DIR,
        GENERATED_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    set_seed(
        SEED
    )

    # ========================================================
    # Step 1 — Audit frozen protocol / warm-start template
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B1 PROTOCOL + ZERO-UPDATE TEMPLATE"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B0_GENERATOR,
        B1_PROTOCOL_FILE,
        WARM_START_TEMPLATE,
        ARCH_SANITY_SUMMARY,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B1_PROTOCOL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        protocol = json.load(
            f
        )

    with ARCH_SANITY_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        architecture_summary = json.load(
            f
        )

    if protocol[
        "status"
    ] != "B1_SUBJECT_STATE_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B1 protocol is not frozen."
        )

    if int(
        protocol[
            "pilot"
        ][
            "steps"
        ]
    ) != PILOT_SUCCESSFUL_STEPS:
        raise RuntimeError(
            "Pilot step count differs from frozen B1 protocol."
        )

    if int(
        protocol[
            "full_fine_tuning"
        ][
            "batch_size"
        ]
    ) != BATCH_SIZE:
        raise RuntimeError(
            "Batch size differs from frozen B1 protocol."
        )

    if float(
        protocol[
            "full_fine_tuning"
        ][
            "learning_rate"
        ]
    ) != LEARNING_RATE:
        raise RuntimeError(
            "Learning rate differs from frozen B1 protocol."
        )

    if protocol[
        "full_fine_tuning"
    ][
        "ema_enabled"
    ] is not False:
        raise RuntimeError(
            "B1 protocol unexpectedly enables EMA."
        )

    if architecture_summary[
        "status"
    ] != "B1_ARCHITECTURE_WARMSTART_SANITY_PASS":
        raise RuntimeError(
            "B1 architecture sanity did not pass."
        )

    template_hash_before = (
        sha256_file(
            WARM_START_TEMPLATE
        )
    )

    b0_hash_before = (
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
            "B1 pilot must start from the zero-update template."
        )

    if template[
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "B1 warm-start template is not v-prediction."
        )

    if int(
        template[
            "num_subjects"
        ]
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Warm-start subject count mismatch."
        )

    if int(
        template[
            "subject_embedding_dim"
        ]
    ) != SUBJECT_EMBEDDING_DIM:
        raise RuntimeError(
            "Warm-start subject embedding mismatch."
        )

    injection_module_name = str(
        template[
            "condition_injection_module"
        ]
    )

    b1_template_state_hash = (
        state_dict_sha256(
            template[
                "model_state_dict"
            ]
        )
    )

    if (
        b1_template_state_hash
        != template[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "B1 template tensor hash mismatch."
        )

    protocol_snapshot = {
        "status": (
            "B1_SHORT_PILOT_PROTOCOL_SNAPSHOT"
        ),
        "frozen_protocol": str(
            B1_PROTOCOL_FILE
        ),
        "frozen_protocol_sha256": (
            sha256_file(
                B1_PROTOCOL_FILE
            )
        ),
        "warm_start_template": str(
            WARM_START_TEMPLATE
        ),
        "warm_start_template_sha256": (
            template_hash_before
        ),
        "warm_start_model_state_sha256": (
            b1_template_state_hash
        ),
        "frozen_B0": str(
            FROZEN_B0_GENERATOR
        ),
        "frozen_B0_sha256": (
            b0_hash_before
        ),
        "condition_injection_module": (
            injection_module_name
        ),
        "successful_optimizer_steps": (
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
        "EMA": False,
        "prediction_target": "v",
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "scientific_full_run_initialization_from_pilot": False,
    }

    atomic_json_save(
        protocol_snapshot,
        PROTOCOL_SNAPSHOT,
    )

    print(
        f"condition injection module: "
        f"{injection_module_name}"
    )

    print(
        f"template SHA256: "
        f"{template_hash_before}"
    )

    print(
        "protocol/template audit: PASS"
    )

    # ========================================================
    # Step 2 — Build Train-only subject-indexed data
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — BUILD B1 TRAIN-ONLY SUBJECT-INDEXED DATA"
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

    records, subject_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    if len(
        records
    ) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            f"Unexpected Train windows: "
            f"{len(records)}"
        )

    if len(
        subject_mapping
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Unexpected subject count: "
            f"{len(subject_mapping)}"
        )

    base_dataset = (
        CachedRealTrainWindowDataset(
            records=records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    train_dataset = (
        SubjectIndexedDataset(
            base_dataset=base_dataset,
            records=records,
        )
    )

    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        TRAIN_LOADER_SEED
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
            f"Expected 748 Train batches, got "
            f"{len(train_loader)}."
        )

    print(
        f"Train windows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Train batches/epoch: "
        f"{len(train_loader)}"
    )

    print(
        f"subjects: "
        f"{len(subject_mapping)}"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    # ========================================================
    # Step 3 — Build B1 / schedule / fixed probe
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD ZERO-UPDATE B1 + FIXED TRAIN PROBE"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    frozen_b0 = torch.load(
        FROZEN_B0_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    b0_contract = frozen_b0[
        "effective_contract"
    ]

    model = B1SubjectStateUNet1D(
        b0_contract=b0_contract,
        injection_module_name=(
            injection_module_name
        ),
        num_subjects=EXPECTED_SUBJECTS,
        subject_embedding_dim=(
            SUBJECT_EMBEDDING_DIM
        ),
        condition_dim=(
            CONDITION_DIM
        ),
    )

    model.load_state_dict(
        template[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    model_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"parameters: "
        f"{model_parameter_count:,}"
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    alpha_bar = schedule_cpu[
        "alpha_cumprod"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    fixed_x0, fixed_state, fixed_subject = (
        build_subject_diverse_balanced_batch(
            train_dataset,
            records,
            device,
        )
    )

    fixed_wrong_subject = (
        fixed_subject
        + 97
    ) % EXPECTED_SUBJECTS

    if torch.any(
        fixed_wrong_subject
        == fixed_subject
    ):
        raise RuntimeError(
            "Wrong-subject mapping accidentally matches "
            "correct subject."
        )

    pre_probe = evaluate_fixed_probe(
        model=model,
        x0=fixed_x0,
        state=fixed_state,
        subject=fixed_subject,
        wrong_subject=(
            fixed_wrong_subject
        ),
        alpha_bar=alpha_bar,
        device=device,
    )

    pre_subject_stats = (
        subject_condition_statistics(
            model,
            device,
        )
    )

    print(
        "fixed probe:"
    )

    print(
        f"  correct-subject v-MSE: "
        f"{pre_probe['correct_subject_v_mse']:.6f}"
    )

    print(
        f"  wrong-subject v-MSE: "
        f"{pre_probe['wrong_subject_v_mse']:.6f}"
    )

    print(
        f"  wrong-correct gap: "
        f"{pre_probe['wrong_minus_correct_v_mse']:.6e}"
    )

    print(
        f"  correct-vs-wrong output mean abs: "
        f"{pre_probe['correct_vs_wrong_output_mean_abs']:.6e}"
    )

    print(
        f"initial subject residual norm mean: "
        f"{pre_subject_stats['subject_residual_norm_mean']:.6e}"
    )

    # ========================================================
    # Step 4 — Pre-pilot paired subject-aware sampling
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — PRE-PILOT SUBJECT-AWARE SAMPLING"
    )

    print(
        "=" * 80
    )

    sampling_subjects = np.linspace(
        0,
        EXPECTED_SUBJECTS - 1,
        SAMPLING_SUBJECT_COUNT,
        dtype=np.int64,
    ).tolist()

    if len(
        set(
            sampling_subjects
        )
    ) != SAMPLING_SUBJECT_COUNT:
        raise RuntimeError(
            "Sampling subject set contains duplicates."
        )

    sampling_start = time.perf_counter()

    pre_eo = sample_b1_subjects(
        model=model,
        subject_indices=(
            sampling_subjects
        ),
        eye_state_value=0,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        seed=SAMPLING_SEED,
    )

    pre_ec = sample_b1_subjects(
        model=model,
        subject_indices=(
            sampling_subjects
        ),
        eye_state_value=1,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        seed=SAMPLING_SEED,
    )

    pre_sampling = sampling_metrics(
        pre_eo,
        pre_ec,
        scaler,
    )

    print(
        f"pre EO normalized std: "
        f"{pre_sampling['eo_normalized_std']:.6f}"
    )

    print(
        f"pre EC normalized std: "
        f"{pre_sampling['ec_normalized_std']:.6f}"
    )

    print(
        f"pre O1 alpha EC-EO: "
        f"{pre_sampling['o1_alpha_effect_db']:.3f} dB"
    )

    print(
        f"pre O2 alpha EC-EO: "
        f"{pre_sampling['o2_alpha_effect_db']:.3f} dB"
    )

    print(
        f"pre sampling time: "
        f"{(time.perf_counter() - sampling_start) / 60:.2f} min"
    )

    # ========================================================
    # Step 5 — Run 1000 successful optimizer steps
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — RUN B1 1000-STEP TRAIN-ONLY PILOT"
    )

    print(
        "=" * 80
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

    train_noise_generator = (
        torch.Generator(
            device="cuda"
        )
    )

    train_noise_generator.manual_seed(
        TRAIN_NOISE_SEED
    )

    model.train()

    torch.cuda.reset_peak_memory_stats()

    history: list[
        dict[str, Any]
    ] = []

    rolling_losses: deque[
        float
    ] = deque(
        maxlen=50
    )

    total_attempted_batches = 0
    successful_steps = 0
    total_amp_skips = 0
    consecutive_amp_skips = 0
    loader_pass = 0
    embedded_grad_seen = False

    run_start = time.perf_counter()

    train_iterator = iter(
        train_loader
    )

    while (
        successful_steps
        < PILOT_SUCCESSFUL_STEPS
    ):
        try:
            batch = next(
                train_iterator
            )
        except StopIteration:
            loader_pass += 1
            train_iterator = iter(
                train_loader
            )
            batch = next(
                train_iterator
            )

        total_attempted_batches += 1

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
                "Nonfinite Train EEG input."
            )

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                current_batch_size,
            ),
            device=device,
            dtype=torch.long,
            generator=(
                train_noise_generator
            ),
        )

        epsilon = torch.randn(
            x0.shape,
            device=device,
            dtype=x0.dtype,
            generator=(
                train_noise_generator
            ),
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
                "Nonfinite B1 pilot v-loss."
            )

        if not torch.isfinite(
            pred_v
        ).all():
            raise RuntimeError(
                "Nonfinite B1 pilot prediction."
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
            parameter_group_grad_norm(
                model.subject_embedding.parameters()
            )
        )

        subject_projection_grad_norm = (
            parameter_group_grad_norm(
                model.subject_projection.parameters()
            )
        )

        if (
            np.isfinite(
                subject_embedding_grad_norm
            )
            and subject_embedding_grad_norm
            > 0.0
        ):
            embedded_grad_seen = True

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
                    "Nonfinite B1 pilot clip norm."
                )

            grad_scaler.step(
                optimizer
            )

            grad_scaler.update()

            successful_steps += 1
            consecutive_amp_skips = 0

            rolling_losses.append(
                loss_value
            )

            scale_after = float(
                grad_scaler.get_scale()
            )

            history.append(
                {
                    "successful_step": (
                        successful_steps
                    ),
                    "attempted_batch": (
                        total_attempted_batches
                    ),
                    "loader_pass": (
                        loader_pass
                    ),
                    "batch_size": (
                        current_batch_size
                    ),
                    "v_mse": (
                        loss_value
                    ),
                    "rolling_50_v_mse": float(
                        np.mean(
                            list(
                                rolling_losses
                            )
                        )
                    ),
                    "gradient_norm": (
                        float(
                            grad_norm
                        )
                    ),
                    "gradient_tensor_count": (
                        grad_count
                    ),
                    "subject_embedding_grad_norm": (
                        subject_embedding_grad_norm
                    ),
                    "subject_projection_grad_norm": (
                        subject_projection_grad_norm
                    ),
                    "amp_scale": (
                        scale_after
                    ),
                    "amp_skips_total": (
                        total_amp_skips
                    ),
                }
            )

            if (
                successful_steps
                == 1
                or successful_steps
                % 50
                == 0
            ):
                elapsed = (
                    time.perf_counter()
                    - run_start
                )

                print(
                    f"step "
                    f"{successful_steps:4d}/"
                    f"{PILOT_SUCCESSFUL_STEPS} "
                    f"| v_MSE={loss_value:.5f} "
                    f"| roll={history[-1]['rolling_50_v_mse']:.5f} "
                    f"| subj_emb_grad={subject_embedding_grad_norm:.2e} "
                    f"| amp_skips={total_amp_skips} "
                    f"| scale={scale_after:g} "
                    f"| elapsed={elapsed / 60:.1f}m"
                )

        else:
            grad_scaler.step(
                optimizer
            )

            grad_scaler.update()

            scale_after = float(
                grad_scaler.get_scale()
            )

            total_amp_skips += 1
            consecutive_amp_skips += 1

            print(
                "AMP SKIP "
                f"| attempted_batch={total_attempted_batches} "
                f"| scale {scale_before:g}->{scale_after:g} "
                f"| bad={bad_names[:3]}"
            )

            if (
                total_amp_skips
                > MAX_TOTAL_AMP_SKIPS
            ):
                raise RuntimeError(
                    "Too many AMP skips in B1 pilot."
                )

            if (
                consecutive_amp_skips
                > MAX_CONSECUTIVE_AMP_SKIPS
            ):
                raise RuntimeError(
                    "Too many consecutive AMP skips "
                    "in B1 pilot."
                )

    training_seconds = (
        time.perf_counter()
        - run_start
    )

    print(
        f"pilot training time: "
        f"{training_seconds / 60:.2f} min"
    )

    print(
        f"successful optimizer steps: "
        f"{successful_steps}"
    )

    print(
        f"attempted batches: "
        f"{total_attempted_batches}"
    )

    print(
        f"AMP skipped steps: "
        f"{total_amp_skips}"
    )

    print(
        f"subject embedding gradient observed: "
        f"{embedded_grad_seen}"
    )

    # ========================================================
    # Step 6 — Post-pilot denoising / subject diagnostics
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — POST-PILOT SUBJECT-CONDITIONING DIAGNOSTICS"
    )

    print(
        "=" * 80
    )

    post_probe = evaluate_fixed_probe(
        model=model,
        x0=fixed_x0,
        state=fixed_state,
        subject=fixed_subject,
        wrong_subject=(
            fixed_wrong_subject
        ),
        alpha_bar=alpha_bar,
        device=device,
    )

    post_subject_stats = (
        subject_condition_statistics(
            model,
            device,
        )
    )

    print(
        f"fixed correct-subject v-MSE: "
        f"{pre_probe['correct_subject_v_mse']:.6f} "
        f"-> "
        f"{post_probe['correct_subject_v_mse']:.6f}"
    )

    print(
        f"fixed wrong-subject v-MSE: "
        f"{pre_probe['wrong_subject_v_mse']:.6f} "
        f"-> "
        f"{post_probe['wrong_subject_v_mse']:.6f}"
    )

    print(
        f"wrong-correct v-MSE gap: "
        f"{pre_probe['wrong_minus_correct_v_mse']:.6e} "
        f"-> "
        f"{post_probe['wrong_minus_correct_v_mse']:.6e}"
    )

    print(
        f"correct-vs-wrong output mean abs: "
        f"{pre_probe['correct_vs_wrong_output_mean_abs']:.6e} "
        f"-> "
        f"{post_probe['correct_vs_wrong_output_mean_abs']:.6e}"
    )

    print(
        f"subject residual norm mean: "
        f"{pre_subject_stats['subject_residual_norm_mean']:.6e} "
        f"-> "
        f"{post_subject_stats['subject_residual_norm_mean']:.6e}"
    )

    print(
        f"subject residual RMS across subjects: "
        f"{pre_subject_stats['subject_residual_rms_across_subjects']:.6e} "
        f"-> "
        f"{post_subject_stats['subject_residual_rms_across_subjects']:.6e}"
    )

    if not embedded_grad_seen:
        raise RuntimeError(
            "Subject embeddings never received gradient "
            "during the 1000-step pilot."
        )

    if (
        post_subject_stats[
            "subject_residual_rms_across_subjects"
        ]
        <= 0.0
    ):
        raise RuntimeError(
            "Subject residuals did not become diverse."
        )

    if (
        post_probe[
            "correct_vs_wrong_output_mean_abs"
        ]
        <= 0.0
    ):
        raise RuntimeError(
            "Subject ID has no post-pilot effect "
            "on B1 output."
        )

    print(
        "subject path learned non-zero subject-specific behavior: PASS"
    )

    # ========================================================
    # Step 7 — Post-pilot paired subject-aware sampling
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — POST-PILOT SUBJECT-AWARE SAMPLING"
    )

    print(
        "=" * 80
    )

    sampling_start = time.perf_counter()

    post_eo = sample_b1_subjects(
        model=model,
        subject_indices=(
            sampling_subjects
        ),
        eye_state_value=0,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        seed=SAMPLING_SEED,
    )

    post_ec = sample_b1_subjects(
        model=model,
        subject_indices=(
            sampling_subjects
        ),
        eye_state_value=1,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        seed=SAMPLING_SEED,
    )

    post_sampling = sampling_metrics(
        post_eo,
        post_ec,
        scaler,
    )

    print(
        f"post EO normalized std: "
        f"{post_sampling['eo_normalized_std']:.6f}"
    )

    print(
        f"post EC normalized std: "
        f"{post_sampling['ec_normalized_std']:.6f}"
    )

    print(
        f"post EO µV std: "
        f"{post_sampling['eo_uv_std']:.3f}"
    )

    print(
        f"post EC µV std: "
        f"{post_sampling['ec_uv_std']:.3f}"
    )

    print(
        f"post whole-head alpha EC-EO: "
        f"{post_sampling['whole_head_alpha_effect_db']:.3f} dB"
    )

    print(
        f"post O1 alpha EC-EO: "
        f"{post_sampling['o1_alpha_effect_db']:.3f} dB"
    )

    print(
        f"post O2 alpha EC-EO: "
        f"{post_sampling['o2_alpha_effect_db']:.3f} dB"
    )

    print(
        f"post sampling time: "
        f"{(time.perf_counter() - sampling_start) / 60:.2f} min"
    )

    # ========================================================
    # Step 8 — Save engineering pilot checkpoint / artifacts
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — SAVE PILOT ARTIFACTS"
    )

    print(
        "=" * 80
    )

    final_model_state = {
        name: tensor.detach()
        .cpu()
        .clone()
        for name, tensor
        in model.state_dict().items()
    }

    final_model_state_hash = (
        state_dict_sha256(
            final_model_state
        )
    )

    checkpoint_payload = {
        "status": (
            "B1_SHORT_PILOT_ENGINEERING_CHECKPOINT"
        ),
        "schema_version": 1,
        "scientific_full_run_initialization_allowed": False,
        "note": (
            "Engineering pilot only. "
            "The scientific full B1 run must restart "
            "from the zero-update warm-start template."
        ),
        "parent_template": str(
            WARM_START_TEMPLATE
        ),
        "parent_template_sha256": (
            template_hash_before
        ),
        "parent_template_model_state_sha256": (
            b1_template_state_hash
        ),
        "frozen_B0": str(
            FROZEN_B0_GENERATOR
        ),
        "frozen_B0_sha256": (
            b0_hash_before
        ),
        "frozen_B1_protocol": str(
            B1_PROTOCOL_FILE
        ),
        "condition_injection_module": (
            injection_module_name
        ),
        "prediction_target": "v",
        "successful_optimizer_steps": (
            successful_steps
        ),
        "attempted_batches": (
            total_attempted_batches
        ),
        "AMP_skipped_steps": (
            total_amp_skips
        ),
        "model_state_dict": (
            final_model_state
        ),
        "model_state_sha256": (
            final_model_state_hash
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "grad_scaler_state_dict": (
            grad_scaler.state_dict()
        ),
        "validation_eeg_accessed": False,
        "final_test_accessed": False,
    }

    atomic_torch_save(
        checkpoint_payload,
        PILOT_CHECKPOINT,
    )

    save_history_csv(
        history,
        HISTORY_CSV,
    )

    np.savez_compressed(
        GENERATED_NPZ,
        subject_indices=np.asarray(
            sampling_subjects,
            dtype=np.int64,
        ),
        pre_eo_norm=pre_eo,
        pre_ec_norm=pre_ec,
        post_eo_norm=post_eo,
        post_ec_norm=post_ec,
        pre_eo_uv=pre_sampling[
            "eo_uv"
        ],
        pre_ec_uv=pre_sampling[
            "ec_uv"
        ],
        post_eo_uv=post_sampling[
            "eo_uv"
        ],
        post_ec_uv=post_sampling[
            "ec_uv"
        ],
    )

    print(
        f"pilot checkpoint: "
        f"{PILOT_CHECKPOINT}"
    )

    print(
        f"history CSV: "
        f"{HISTORY_CSV}"
    )

    print(
        f"generated package: "
        f"{GENERATED_NPZ}"
    )

    # ========================================================
    # Step 9 — Figures
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — SAVE PILOT FIGURES"
    )

    print(
        "=" * 80
    )

    save_loss_figure(
        history
    )

    save_subject_gap_figure(
        pre_probe,
        post_probe,
    )

    save_sampling_scale_figure(
        pre_sampling,
        post_sampling,
    )

    save_sampling_alpha_figure(
        pre_sampling,
        post_sampling,
    )

    print(
        LOSS_FIGURE
    )

    print(
        SUBJECT_GAP_FIGURE
    )

    print(
        SAMPLING_SCALE_FIGURE
    )

    print(
        SAMPLING_ALPHA_FIGURE
    )

    # ========================================================
    # Step 10 — Immutability + summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 10 — VERIFY IMMUTABILITY + SAVE SUMMARY"
    )

    print(
        "=" * 80
    )

    template_hash_after = (
        sha256_file(
            WARM_START_TEMPLATE
        )
    )

    b0_hash_after = (
        sha256_file(
            FROZEN_B0_GENERATOR
        )
    )

    if (
        template_hash_after
        != template_hash_before
    ):
        raise RuntimeError(
            "Zero-update B1 template changed "
            "during pilot."
        )

    if (
        b0_hash_after
        != b0_hash_before
    ):
        raise RuntimeError(
            "Frozen B0 changed during B1 pilot."
        )

    first_50_mean = float(
        np.mean(
            [
                row[
                    "v_mse"
                ]
                for row
                in history[
                    :50
                ]
            ]
        )
    )

    last_50_mean = float(
        np.mean(
            [
                row[
                    "v_mse"
                ]
                for row
                in history[
                    -50:
                ]
            ]
        )
    )

    peak_allocated_mib = float(
        torch.cuda.max_memory_allocated()
        / (
            1024
            ** 2
        )
    )

    peak_reserved_mib = float(
        torch.cuda.max_memory_reserved()
        / (
            1024
            ** 2
        )
    )

    summary = {
        "status": (
            "B1_SHORT_PILOT_COMPLETE"
        ),
        "scientific_interpretation_status": (
            "AWAITING_REVIEW_BEFORE_FULL_B1_RUN"
        ),
        "parent": {
            "zero_update_template": str(
                WARM_START_TEMPLATE
            ),
            "zero_update_template_sha256": (
                template_hash_before
            ),
            "zero_update_template_model_state_sha256": (
                b1_template_state_hash
            ),
            "frozen_B0": str(
                FROZEN_B0_GENERATOR
            ),
            "frozen_B0_sha256": (
                b0_hash_before
            ),
            "condition_injection_module": (
                injection_module_name
            ),
        },
        "pilot": {
            "successful_optimizer_steps": (
                successful_steps
            ),
            "attempted_batches": (
                total_attempted_batches
            ),
            "AMP_skipped_steps": (
                total_amp_skips
            ),
            "training_seconds": (
                training_seconds
            ),
            "training_minutes": (
                training_seconds
                / 60.0
            ),
            "first_50_mean_v_mse": (
                first_50_mean
            ),
            "last_50_mean_v_mse": (
                last_50_mean
            ),
            "subject_embedding_gradient_observed": (
                embedded_grad_seen
            ),
            "peak_allocated_mib": (
                peak_allocated_mib
            ),
            "peak_reserved_mib": (
                peak_reserved_mib
            ),
        },
        "fixed_probe": {
            "before": (
                pre_probe
            ),
            "after": (
                post_probe
            ),
        },
        "subject_condition_statistics": {
            "before": (
                pre_subject_stats
            ),
            "after": (
                post_subject_stats
            ),
        },
        "sampling_subject_indices": (
            sampling_subjects
        ),
        "sampling": {
            "before": {
                key: value
                for key, value
                in pre_sampling.items()
                if key
                not in (
                    "eo_uv",
                    "ec_uv",
                )
            },
            "after": {
                key: value
                for key, value
                in post_sampling.items()
                if key
                not in (
                    "eo_uv",
                    "ec_uv",
                )
            },
        },
        "artifacts": {
            "pilot_checkpoint": str(
                PILOT_CHECKPOINT
            ),
            "pilot_checkpoint_model_state_sha256": (
                final_model_state_hash
            ),
            "history_csv": str(
                HISTORY_CSV
            ),
            "generated_npz": str(
                GENERATED_NPZ
            ),
            "loss_figure": str(
                LOSS_FIGURE
            ),
            "subject_contrast_figure": str(
                SUBJECT_GAP_FIGURE
            ),
            "sampling_scale_figure": str(
                SAMPLING_SCALE_FIGURE
            ),
            "sampling_alpha_figure": str(
                SAMPLING_ALPHA_FIGURE
            ),
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "bulk_synthetic_generation_started": False,
            "pilot_checkpoint_reusable_for_scientific_full_run": False,
            "zero_update_template_unchanged": True,
            "frozen_B0_unchanged": True,
        },
        "next": (
            "Interpret pilot before launching "
            "the frozen 30-epoch B1 full fine-tuning protocol."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
    )

    print(
        "zero-update B1 template unchanged: PASS"
    )

    print(
        "frozen B0 unchanged: PASS"
    )

    print(
        f"summary: "
        f"{SUMMARY_JSON}"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B1 1000-STEP SUBJECT-CONDITIONED PILOT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"first 50-step mean v-MSE: "
        f"{first_50_mean:.6f}"
    )

    print(
        f"last  50-step mean v-MSE: "
        f"{last_50_mean:.6f}"
    )

    print(
        f"fixed correct-subject v-MSE: "
        f"{pre_probe['correct_subject_v_mse']:.6f} "
        f"-> "
        f"{post_probe['correct_subject_v_mse']:.6f}"
    )

    print(
        f"fixed wrong-subject v-MSE: "
        f"{pre_probe['wrong_subject_v_mse']:.6f} "
        f"-> "
        f"{post_probe['wrong_subject_v_mse']:.6f}"
    )

    print(
        f"wrong-correct v-MSE gap: "
        f"{pre_probe['wrong_minus_correct_v_mse']:.6e} "
        f"-> "
        f"{post_probe['wrong_minus_correct_v_mse']:.6e}"
    )

    print(
        f"correct-vs-wrong output mean abs: "
        f"{pre_probe['correct_vs_wrong_output_mean_abs']:.6e} "
        f"-> "
        f"{post_probe['correct_vs_wrong_output_mean_abs']:.6e}"
    )

    print(
        f"subject residual RMS across subjects: "
        f"{pre_subject_stats['subject_residual_rms_across_subjects']:.6e} "
        f"-> "
        f"{post_subject_stats['subject_residual_rms_across_subjects']:.6e}"
    )

    print(
        f"sampling EO normalized std: "
        f"{pre_sampling['eo_normalized_std']:.6f} "
        f"-> "
        f"{post_sampling['eo_normalized_std']:.6f}"
    )

    print(
        f"sampling EC normalized std: "
        f"{pre_sampling['ec_normalized_std']:.6f} "
        f"-> "
        f"{post_sampling['ec_normalized_std']:.6f}"
    )

    print(
        f"O1 alpha EC-EO: "
        f"{pre_sampling['o1_alpha_effect_db']:.3f} "
        f"-> "
        f"{post_sampling['o1_alpha_effect_db']:.3f} dB"
    )

    print(
        f"O2 alpha EC-EO: "
        f"{pre_sampling['o2_alpha_effect_db']:.3f} "
        f"-> "
        f"{post_sampling['o2_alpha_effect_db']:.3f} dB"
    )

    print(
        f"AMP skipped steps: "
        f"{total_amp_skips}"
    )

    print(
        f"training time: "
        f"{training_seconds / 60:.2f} min"
    )

    print(
        f"peak GPU allocated: "
        f"{peak_allocated_mib:.0f} MiB"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Pilot checkpoint reusable for scientific full run: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "INTERPRET B1 PILOT BEFORE STARTING "
        "THE FROZEN 30-EPOCH FULL FINE-TUNING RUN"
    )


if __name__ == "__main__":
    main()
