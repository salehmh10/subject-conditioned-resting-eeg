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
import torch.nn.functional as F
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
from s2_48_b1_1000_step_subject_conditioned_pilot import (
    build_reverse_schedule,
    sample_b1_subjects,
    sampling_metrics,
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

B1_TO_B2_HANDOFF = (
    FROZEN_B1_ROOT
    / "handoff"
    / "b1_to_b2_handoff_v1.json"
)

B2_PROTOCOL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "spectral_regularized_v1"
)

B2_PROTOCOL_FILE = (
    B2_PROTOCOL_ROOT
    / "metrics"
    / "b2_spectral_regularized_protocol_v1.json"
)

B2_PILOT_HANDOFF = (
    B2_PROTOCOL_ROOT
    / "handoff"
    / "b2_lambda_calibration_pilot_handoff_v1.json"
)


# ============================================================
# Pilot output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "lambda_calibration_v1"
)

CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
LOG_DIR = RUN_ROOT / "logs"
METRIC_DIR = RUN_ROOT / "metrics"
GENERATED_DIR = RUN_ROOT / "generated"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_SNAPSHOT = (
    METRIC_DIR
    / "b2_lambda_calibration_protocol_snapshot_v1.json"
)

METRICS_CSV = (
    METRIC_DIR
    / "b2_lambda_calibration_metrics_v1.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "b2_lambda_calibration_summary_v1.json"
)

HISTORY_CSV = (
    LOG_DIR
    / "b2_lambda_calibration_history_v1.csv"
)

GENERATED_NPZ = (
    GENERATED_DIR
    / "b2_lambda_calibration_samples_v1.npz"
)

LOSS_FIGURE = (
    FIGURE_DIR
    / "b2_lambda_calibration_loss_v1.png"
)

PROBE_FIGURE = (
    FIGURE_DIR
    / "b2_lambda_calibration_fixed_probe_v1.png"
)

SCALE_FIGURE = (
    FIGURE_DIR
    / "b2_lambda_calibration_sampling_scale_v1.png"
)

ALPHA_FIGURE = (
    FIGURE_DIR
    / "b2_lambda_calibration_alpha_effect_v1.png"
)

ELIGIBILITY_FIGURE = (
    FIGURE_DIR
    / "b2_lambda_calibration_eligibility_v1.png"
)


# ============================================================
# Frozen protocol expectations
# ============================================================

SEED = 20260908
TRAIN_LOADER_SEED = 20261011
TRAIN_NOISE_SEED = 20261012
FIXED_PROBE_SEED = 20261013
SAMPLING_SEED = 20261014

EXPECTED_SUBJECTS = 198
EXPECTED_TRAIN_WINDOWS = 47_835

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

TIMESTEPS = 1000
COSINE_S = 0.008

LAMBDA_CANDIDATES = [
    0.0,
    0.01,
    0.03,
    0.10,
]

SUCCESSFUL_STEPS_PER_LAMBDA = 500
BATCH_SIZE = 64

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

MIXED_PRECISION = True

SPECTRAL_FMIN_HZ = 1.0
SPECTRAL_FMAX_HZ = 40.0
SPECTRAL_LOG_EPS = 1e-12
SPECTRAL_HUBER_BETA_DB = 1.0

SAMPLING_SUBJECT_COUNT = 16

V_MSE_MAX_RATIO_TO_LAMBDA0 = 1.02
SUBJECT_GAP_MIN_RATIO_TO_LAMBDA0 = 0.90
SAMPLING_STD_MIN = 0.80
SAMPLING_STD_MAX = 1.20
SPECTRAL_IMPROVEMENT_MIN_FRACTION = 0.10


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


# ============================================================
# Spectral objective
# ============================================================

class PhysicalLogPSDLoss(nn.Module):
    """
    Differentiable physical-unit 1-40 Hz log-PSD Huber loss.

    Inputs are normalized EEG tensors [B, C, T].
    The same frozen Train-only global scaler is inverted before PSD.
    """

    def __init__(
        self,
        *,
        means_uv: np.ndarray,
        stds_uv: np.ndarray,
        sampling_rate_hz: int,
        fmin_hz: float,
        fmax_hz: float,
        log_eps: float,
        huber_beta_db: float,
        n_times: int,
    ) -> None:
        super().__init__()

        means = torch.as_tensor(
            means_uv,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
            1,
        )

        stds = torch.as_tensor(
            stds_uv,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
            1,
        )

        if means.shape[
            1
        ] != N_CHANNELS:
            raise RuntimeError(
                "Scaler mean channel count mismatch."
            )

        if stds.shape[
            1
        ] != N_CHANNELS:
            raise RuntimeError(
                "Scaler std channel count mismatch."
            )

        hann = torch.hann_window(
            n_times,
            periodic=True,
            dtype=torch.float32,
        ).reshape(
            1,
            1,
            n_times,
        )

        frequencies = torch.fft.rfftfreq(
            n_times,
            d=(
                1.0
                / sampling_rate_hz
            ),
        )

        mask = (
            (
                frequencies
                >= fmin_hz
            )
            & (
                frequencies
                <= fmax_hz
            )
        )

        if int(
            torch.sum(
                mask
            ).item()
        ) <= 0:
            raise RuntimeError(
                "No spectral bins in requested frequency range."
            )

        self.register_buffer(
            "means_uv",
            means,
            persistent=True,
        )

        self.register_buffer(
            "stds_uv",
            stds,
            persistent=True,
        )

        self.register_buffer(
            "hann",
            hann,
            persistent=True,
        )

        self.register_buffer(
            "frequency_mask",
            mask,
            persistent=True,
        )

        self.sampling_rate_hz = int(
            sampling_rate_hz
        )

        self.log_eps = float(
            log_eps
        )

        self.huber_beta_db = float(
            huber_beta_db
        )

        self.window_power_sum = float(
            torch.sum(
                hann
                * hann
            ).item()
        )

        self.n_times = int(
            n_times
        )

    def _to_uv(
        self,
        x_norm: torch.Tensor,
    ) -> torch.Tensor:
        return (
            x_norm.float()
            * self.stds_uv
            + self.means_uv
        )

    def _log_psd_db(
        self,
        x_norm: torch.Tensor,
    ) -> torch.Tensor:
        x_uv = self._to_uv(
            x_norm
        )

        x_uv = (
            x_uv
            - torch.mean(
                x_uv,
                dim=-1,
                keepdim=True,
            )
        )

        windowed = (
            x_uv
            * self.hann
        )

        spectrum = torch.fft.rfft(
            windowed,
            n=self.n_times,
            dim=-1,
        )

        psd = (
            spectrum.real
            * spectrum.real
            + spectrum.imag
            * spectrum.imag
        ) / (
            self.sampling_rate_hz
            * self.window_power_sum
        )

        # One-sided PSD: double strictly positive,
        # non-Nyquist bins.
        if self.n_times % 2 == 0:
            psd[
                ...,
                1:-1
            ] = (
                psd[
                    ...,
                    1:-1
                ]
                * 2.0
            )
        else:
            psd[
                ...,
                1:
            ] = (
                psd[
                    ...,
                    1:
                ]
                * 2.0
            )

        psd = psd[
            ...,
            self.frequency_mask
        ]

        log_psd_db = (
            10.0
            * torch.log10(
                torch.clamp(
                    psd,
                    min=self.log_eps,
                )
            )
        )

        return log_psd_db

    def forward(
        self,
        x0_hat_norm: torch.Tensor,
        x0_true_norm: torch.Tensor,
    ) -> torch.Tensor:
        predicted_db = self._log_psd_db(
            x0_hat_norm
        )

        with torch.no_grad():
            target_db = self._log_psd_db(
                x0_true_norm
            )

        return F.smooth_l1_loss(
            predicted_db,
            target_db,
            beta=self.huber_beta_db,
            reduction="mean",
        )


def reconstruct_x0_from_v(
    *,
    xt: torch.Tensor,
    v_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    sqrt_ab = torch.sqrt(
        alpha_bar[
            timesteps
        ]
    ).reshape(
        -1,
        1,
        1,
    )

    sqrt_omab = torch.sqrt(
        1.0
        - alpha_bar[
            timesteps
        ]
    ).reshape(
        -1,
        1,
        1,
    )

    return (
        sqrt_ab
        * xt.float()
        - sqrt_omab
        * v_pred.float()
    )


# ============================================================
# Model / data helpers
# ============================================================

def build_b1_model_from_frozen(
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

    model.to(
        device
    )

    return model


def build_train_loader(
    dataset: SubjectIndexedDataset,
) -> DataLoader:
    generator = torch.Generator()

    generator.manual_seed(
        TRAIN_LOADER_SEED
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator,
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


# ============================================================
# Deterministic fixed probe
# ============================================================

@torch.inference_mode()
def build_fixed_probe_noise(
    *,
    x0: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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

    timesteps = torch.randint(
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
        timesteps,
        alpha_bar,
    )

    target_v = make_v_target(
        x0,
        epsilon,
        timesteps,
        alpha_bar,
    )

    return (
        timesteps,
        xt,
        target_v,
    )


@torch.inference_mode()
def evaluate_fixed_probe(
    *,
    model: B1SubjectStateUNet1D,
    x0: torch.Tensor,
    state: torch.Tensor,
    subject: torch.Tensor,
    wrong_subject: torch.Tensor,
    timesteps: torch.Tensor,
    xt: torch.Tensor,
    target_v: torch.Tensor,
    alpha_bar: torch.Tensor,
    spectral_loss_fn: PhysicalLogPSDLoss,
) -> dict[str, float]:
    model.eval()

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

    target_v_f = (
        target_v.float()
    )

    correct_v_mse = float(
        torch.mean(
            (
                correct_pred_f
                - target_v_f
            )
            ** 2
        ).item()
    )

    wrong_v_mse = float(
        torch.mean(
            (
                wrong_pred_f
                - target_v_f
            )
            ** 2
        ).item()
    )

    output_difference = float(
        torch.mean(
            torch.abs(
                correct_pred_f
                - wrong_pred_f
            )
        ).item()
    )

    x0_hat = reconstruct_x0_from_v(
        xt=xt,
        v_pred=correct_pred_f,
        timesteps=timesteps,
        alpha_bar=alpha_bar,
    )

    spectral_loss = float(
        spectral_loss_fn(
            x0_hat,
            x0,
        ).item()
    )

    return {
        "correct_subject_v_mse": (
            correct_v_mse
        ),
        "wrong_subject_v_mse": (
            wrong_v_mse
        ),
        "wrong_minus_correct_v_mse": (
            wrong_v_mse
            - correct_v_mse
        ),
        "correct_vs_wrong_output_mean_abs": (
            output_difference
        ),
        "spectral_loss_db_huber": (
            spectral_loss
        ),
    }


# ============================================================
# Figures
# ============================================================

def save_loss_figure(
    history_rows: list[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for lambda_value in LAMBDA_CANDIDATES:
        rows = [
            row
            for row in history_rows
            if float(
                row[
                    "lambda_spectral"
                ]
            )
            == float(
                lambda_value
            )
        ]

        x = [
            row[
                "successful_step"
            ]
            for row in rows
        ]

        y = [
            row[
                "rolling_50_total_loss"
            ]
            for row in rows
        ]

        ax.plot(
            x,
            y,
            label=f"lambda={lambda_value:g}",
        )

    ax.set_xlabel(
        "Successful optimizer step"
    )

    ax.set_ylabel(
        "50-step rolling total loss"
    )

    ax.set_title(
        "B2 Lambda Calibration — Total Loss"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        LOSS_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_probe_figure(
    metrics_rows: list[dict[str, Any]],
) -> None:
    labels = [
        f"{row['lambda_spectral']:g}"
        for row in metrics_rows
    ]

    spectral = [
        row[
            "post_probe_spectral_loss_db_huber"
        ]
        for row in metrics_rows
    ]

    v_mse = [
        row[
            "post_probe_correct_subject_v_mse"
        ]
        for row in metrics_rows
    ]

    x = np.arange(
        len(
            labels
        )
    )

    fig, ax1 = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax1.bar(
        x,
        spectral,
        width=0.6,
        label="Spectral loss",
    )

    ax1.set_xticks(
        x,
        labels,
    )

    ax1.set_xlabel(
        "lambda_spectral"
    )

    ax1.set_ylabel(
        "Fixed-probe spectral Huber loss (dB)"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        v_mse,
        marker="o",
        label="Correct-subject v-MSE",
    )

    ax2.set_ylabel(
        "Fixed-probe correct-subject v-MSE"
    )

    ax1.set_title(
        "B2 Lambda Calibration — Fixed-Probe Trade-off"
    )

    fig.tight_layout()

    fig.savefig(
        PROBE_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_scale_figure(
    metrics_rows: list[dict[str, Any]],
) -> None:
    labels = [
        f"{row['lambda_spectral']:g}"
        for row in metrics_rows
    ]

    eo = [
        row[
            "post_sampling_eo_normalized_std"
        ]
        for row in metrics_rows
    ]

    ec = [
        row[
            "post_sampling_ec_normalized_std"
        ]
        for row in metrics_rows
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
        eo,
        width,
        label="EO",
    )

    ax.bar(
        x
        + width
        / 2,
        ec,
        width,
        label="EC",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        SAMPLING_STD_MIN,
        linestyle=":",
        linewidth=1,
    )

    ax.axhline(
        SAMPLING_STD_MAX,
        linestyle=":",
        linewidth=1,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        "lambda_spectral"
    )

    ax.set_ylabel(
        "Generated normalized std"
    )

    ax.set_title(
        "B2 Lambda Calibration — Sampling Scale"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SCALE_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_alpha_figure(
    metrics_rows: list[dict[str, Any]],
) -> None:
    labels = [
        f"{row['lambda_spectral']:g}"
        for row in metrics_rows
    ]

    o1 = [
        row[
            "post_sampling_o1_alpha_effect_db"
        ]
        for row in metrics_rows
    ]

    o2 = [
        row[
            "post_sampling_o2_alpha_effect_db"
        ]
        for row in metrics_rows
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
        o1,
        width,
        label="O1",
    )

    ax.bar(
        x
        + width
        / 2,
        o2,
        width,
        label="O2",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        "lambda_spectral"
    )

    ax.set_ylabel(
        "EC - EO alpha power (dB)"
    )

    ax.set_title(
        "B2 Lambda Calibration — Occipital Alpha Effect"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        ALPHA_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_eligibility_figure(
    metrics_rows: list[dict[str, Any]],
) -> None:
    labels = [
        f"{row['lambda_spectral']:g}"
        for row in metrics_rows
        if float(
            row[
                "lambda_spectral"
            ]
        ) > 0.0
    ]

    criteria = [
        "v-MSE",
        "subject gap",
        "EO scale",
        "EC scale",
        "spectral improvement",
    ]

    matrix = []

    for row in metrics_rows:
        if float(
            row[
                "lambda_spectral"
            ]
        ) <= 0.0:
            continue

        matrix.append(
            [
                float(
                    row[
                        "eligibility_v_mse"
                    ]
                ),
                float(
                    row[
                        "eligibility_subject_gap"
                    ]
                ),
                float(
                    row[
                        "eligibility_eo_scale"
                    ]
                ),
                float(
                    row[
                        "eligibility_ec_scale"
                    ]
                ),
                float(
                    row[
                        "eligibility_spectral_improvement"
                    ]
                ),
            ]
        )

    data = np.asarray(
        matrix,
        dtype=np.float64,
    ).T

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    image = ax.imshow(
        data,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        np.arange(
            len(
                labels
            )
        ),
        labels,
    )

    ax.set_yticks(
        np.arange(
            len(
                criteria
            )
        ),
        criteria,
    )

    ax.set_xlabel(
        "lambda_spectral"
    )

    ax.set_title(
        "B2 Lambda Calibration — Eligibility (1=PASS)"
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        ELIGIBILITY_FIGURE,
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
        "SCENARIO 2 — B2 TRAIN-ONLY LAMBDA CALIBRATION PILOT"
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
        "  B2 protocol: FROZEN"
    )

    print(
        "  B2 lambda calibration pilot: NOW"
    )

    print(
        "  B2 scientific full fine-tuning: NOT STARTED"
    )

    print(
        "  Bulk synthetic generation: NOT STARTED"
    )

    print(
        "  Final Test: SEALED"
    )

    print(
        "\nCalibration design:"
    )

    print(
        f"  lambdas: "
        f"{LAMBDA_CANDIDATES}"
    )

    print(
        f"  successful steps / lambda: "
        f"{SUCCESSFUL_STEPS_PER_LAMBDA}"
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
        "  same frozen B1 initialization for every lambda: YES"
    )

    print(
        "  same Train data order across lambdas: YES"
    )

    print(
        "  same timestep/noise stream across lambdas: YES"
    )

    print(
        "  Validation EEG accessed: NO"
    )

    print(
        "  Final Test accessed: NO"
    )

    print(
        "  pilot checkpoints reusable for scientific full run: NO"
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
    # Step 1 — Audit frozen protocol / parent
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B2 PROTOCOL + B1 PARENT"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_TO_B2_HANDOFF,
        B2_PROTOCOL_FILE,
        B2_PILOT_HANDOFF,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B2_PROTOCOL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        b2_protocol = json.load(
            f
        )

    with B2_PILOT_HANDOFF.open(
        "r",
        encoding="utf-8",
    ) as f:
        pilot_handoff = json.load(
            f
        )

    if b2_protocol[
        "status"
    ] != "B2_SPECTRAL_REGULARIZED_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B2 protocol is not frozen."
        )

    if pilot_handoff[
        "status"
    ] != "B2_LAMBDA_CALIBRATION_PILOT_READY":
        raise RuntimeError(
            "B2 lambda-pilot handoff is not ready."
        )

    protocol_lambdas = [
        float(
            value
        )
        for value
        in b2_protocol[
            "lambda_calibration_pilot"
        ][
            "candidate_lambdas"
        ]
    ]

    if (
        protocol_lambdas
        != LAMBDA_CANDIDATES
    ):
        raise RuntimeError(
            "Lambda candidates conflict with frozen B2 protocol."
        )

    if int(
        b2_protocol[
            "lambda_calibration_pilot"
        ][
            "successful_optimizer_steps_per_lambda"
        ]
    ) != SUCCESSFUL_STEPS_PER_LAMBDA:
        raise RuntimeError(
            "Pilot step count conflicts with frozen B2 protocol."
        )

    frozen_b1_hash_before = (
        sha256_file(
            FROZEN_B1_GENERATOR
        )
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    if frozen_b1[
        "status"
    ] != "B1_SUBJECT_STATE_V_FROZEN_GENERATOR":
        raise RuntimeError(
            "Unexpected frozen B1 generator status."
        )

    frozen_b1_state_hash = (
        state_dict_sha256(
            frozen_b1[
                "model_state_dict"
            ]
        )
    )

    if (
        frozen_b1_state_hash
        != frozen_b1[
            "selected_model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B1 model-state SHA256 mismatch."
        )

    protocol_snapshot = {
        "status": (
            "B2_LAMBDA_CALIBRATION_PROTOCOL_SNAPSHOT"
        ),
        "frozen_B2_protocol": str(
            B2_PROTOCOL_FILE
        ),
        "frozen_B2_protocol_sha256": (
            sha256_file(
                B2_PROTOCOL_FILE
            )
        ),
        "frozen_B1_generator": str(
            FROZEN_B1_GENERATOR
        ),
        "frozen_B1_file_sha256": (
            frozen_b1_hash_before
        ),
        "frozen_B1_model_state_sha256": (
            frozen_b1_state_hash
        ),
        "candidate_lambdas": (
            LAMBDA_CANDIDATES
        ),
        "successful_steps_per_lambda": (
            SUCCESSFUL_STEPS_PER_LAMBDA
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
        "same_initialization": True,
        "same_data_order": True,
        "same_timestep_noise_stream": True,
        "spectral_fmin_hz": (
            SPECTRAL_FMIN_HZ
        ),
        "spectral_fmax_hz": (
            SPECTRAL_FMAX_HZ
        ),
        "spectral_log_eps": (
            SPECTRAL_LOG_EPS
        ),
        "spectral_huber_beta_db": (
            SPECTRAL_HUBER_BETA_DB
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "scientific_full_run_initialization_from_pilot": False,
    }

    if PROTOCOL_SNAPSHOT.exists():
        with PROTOCOL_SNAPSHOT.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_snapshot = json.load(
                f
            )

        if existing_snapshot != protocol_snapshot:
            raise RuntimeError(
                "Conflicting B2 lambda-calibration snapshot exists."
            )
    else:
        atomic_json_save(
            protocol_snapshot,
            PROTOCOL_SNAPSHOT,
        )

    print(
        f"frozen B1 SHA256: "
        f"{frozen_b1_hash_before}"
    )

    print(
        f"frozen B1 model-state SHA256: "
        f"{frozen_b1_state_hash}"
    )

    print(
        "protocol / parent audit: PASS"
    )

    # ========================================================
    # Step 2 — Build Train-only data / fixed probe
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — BUILD FIXED TRAIN-ONLY DATA + PROBE"
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

    train_records, subject_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    if len(
        train_records
    ) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            f"Unexpected Train window count: "
            f"{len(train_records)}"
        )

    if len(
        subject_mapping
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Unexpected subject count: "
            f"{len(subject_mapping)}"
        )

    train_base_dataset = (
        CachedRealTrainWindowDataset(
            records=train_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    train_dataset = (
        SubjectIndexedDataset(
            base_dataset=train_base_dataset,
            records=train_records,
        )
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for B2 lambda calibration."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
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

    reverse_schedule = (
        build_reverse_schedule(
            schedule_cpu,
            device,
        )
    )

    fixed_x0, fixed_state, fixed_subject = (
        build_subject_diverse_balanced_batch(
            train_dataset,
            train_records,
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
            "Wrong-subject fixed probe collision."
        )

    (
        fixed_timesteps,
        fixed_xt,
        fixed_target_v,
    ) = build_fixed_probe_noise(
        x0=fixed_x0,
        alpha_bar=alpha_bar,
        device=device,
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
            huber_beta_db=SPECTRAL_HUBER_BETA_DB,
            n_times=N_TIMES,
        )
        .to(
            device
        )
    )

    # Sanity-check differentiable spectral loss against identical input.
    identical_spectral_loss = float(
        spectral_loss_fn(
            fixed_x0,
            fixed_x0,
        ).item()
    )

    if (
        identical_spectral_loss
        > 1e-8
    ):
        raise RuntimeError(
            "Spectral loss identity sanity failed: "
            f"{identical_spectral_loss}"
        )

    print(
        f"Train windows: "
        f"{len(train_dataset)}"
    )

    print(
        f"subjects: "
        f"{len(subject_mapping)}"
    )

    print(
        f"fixed probe shape: "
        f"{tuple(fixed_x0.shape)}"
    )

    print(
        f"spectral identity loss: "
        f"{identical_spectral_loss:.3e}"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    # ========================================================
    # Step 3 — Frozen B1 baseline probe / sampling
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — FROZEN B1 BASELINE DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    parent_model = build_b1_model_from_frozen(
        frozen_b1,
        device,
    )

    parent_probe = evaluate_fixed_probe(
        model=parent_model,
        x0=fixed_x0,
        state=fixed_state,
        subject=fixed_subject,
        wrong_subject=(
            fixed_wrong_subject
        ),
        timesteps=fixed_timesteps,
        xt=fixed_xt,
        target_v=fixed_target_v,
        alpha_bar=alpha_bar,
        spectral_loss_fn=(
            spectral_loss_fn
        ),
    )

    sampling_subjects = (
        np.linspace(
            0,
            EXPECTED_SUBJECTS
            - 1,
            SAMPLING_SUBJECT_COUNT,
            dtype=np.int64,
        )
        .tolist()
    )

    parent_eo = sample_b1_subjects(
        model=parent_model,
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

    parent_ec = sample_b1_subjects(
        model=parent_model,
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

    parent_sampling = sampling_metrics(
        parent_eo,
        parent_ec,
        scaler,
    )

    print(
        f"parent correct-subject v-MSE: "
        f"{parent_probe['correct_subject_v_mse']:.6f}"
    )

    print(
        f"parent wrong-correct gap: "
        f"{parent_probe['wrong_minus_correct_v_mse']:.6e}"
    )

    print(
        f"parent spectral loss: "
        f"{parent_probe['spectral_loss_db_huber']:.6f}"
    )

    print(
        f"parent sampling normalized std EO/EC: "
        f"{parent_sampling['eo_normalized_std']:.3f}"
        " / "
        f"{parent_sampling['ec_normalized_std']:.3f}"
    )

    print(
        f"parent O1/O2 alpha EC-EO: "
        f"{parent_sampling['o1_alpha_effect_db']:.3f}"
        " / "
        f"{parent_sampling['o2_alpha_effect_db']:.3f}"
        " dB"
    )

    del parent_model

    torch.cuda.empty_cache()

    # ========================================================
    # Step 4 — Run matched lambda candidates
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — RUN MATCHED 500-STEP LAMBDA CANDIDATES"
    )

    print(
        "=" * 80
    )

    all_history_rows: list[
        dict[str, Any]
    ] = []

    metrics_rows: list[
        dict[str, Any]
    ] = []

    generated_packages = {
        "sampling_subject_indices": np.asarray(
            sampling_subjects,
            dtype=np.int64,
        ),
        "parent_eo_norm": parent_eo,
        "parent_ec_norm": parent_ec,
    }

    per_lambda_details: dict[
        str,
        dict[str, Any]
    ] = {}

    run_start = (
        time.perf_counter()
    )

    for lambda_index, lambda_value in enumerate(
        LAMBDA_CANDIDATES
    ):
        print(
            "\n"
            + "-" * 80
        )

        print(
            f"LAMBDA {lambda_value:g} "
            f"— MATCHED 500-STEP TRAIN-ONLY PILOT"
        )

        print(
            "-" * 80
        )

        # Exact same model initialization for every lambda.
        set_seed(
            SEED
        )

        model = build_b1_model_from_frozen(
            frozen_b1,
            device,
        )

        initial_model_state_hash = (
            state_dict_sha256(
                model.state_dict()
            )
        )

        if (
            initial_model_state_hash
            != frozen_b1_state_hash
        ):
            raise RuntimeError(
                "Lambda candidate did not start from "
                "the exact frozen B1 model state."
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

        # Fresh loader with identical shuffle seed.
        train_loader = build_train_loader(
            train_dataset
        )

        if len(
            train_loader
        ) != 748:
            raise RuntimeError(
                f"Expected 748 Train batches, got "
                f"{len(train_loader)}."
            )

        train_iterator = iter(
            train_loader
        )

        # Fresh CUDA generator with identical noise seed.
        noise_generator = torch.Generator(
            device="cuda"
        )

        noise_generator.manual_seed(
            TRAIN_NOISE_SEED
        )

        pre_probe = evaluate_fixed_probe(
            model=model,
            x0=fixed_x0,
            state=fixed_state,
            subject=fixed_subject,
            wrong_subject=(
                fixed_wrong_subject
            ),
            timesteps=fixed_timesteps,
            xt=fixed_xt,
            target_v=fixed_target_v,
            alpha_bar=alpha_bar,
            spectral_loss_fn=(
                spectral_loss_fn
            ),
        )

        model.train()

        successful_steps = 0
        attempted_batches = 0

        rolling_total = deque(
            maxlen=50
        )

        rolling_v = deque(
            maxlen=50
        )

        rolling_spectral = deque(
            maxlen=50
        )

        candidate_history: list[
            dict[str, Any]
        ] = []

        candidate_start = (
            time.perf_counter()
        )

        torch.cuda.reset_peak_memory_stats()

        while (
            successful_steps
            < SUCCESSFUL_STEPS_PER_LAMBDA
        ):
            try:
                batch = next(
                    train_iterator
                )
            except StopIteration:
                # 500 steps should remain within the first 748-batch epoch.
                raise RuntimeError(
                    "Unexpected loader exhaustion during "
                    "500-step lambda calibration."
                )

            attempted_batches += 1

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
                generator=(
                    noise_generator
                ),
            )

            epsilon = torch.randn(
                x0.shape,
                device=device,
                dtype=x0.dtype,
                generator=(
                    noise_generator
                ),
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
                + float(
                    lambda_value
                )
                * spectral_loss
            )

            if not (
                torch.isfinite(
                    v_loss
                )
                and torch.isfinite(
                    spectral_loss
                )
                and torch.isfinite(
                    total_loss
                )
            ):
                raise RuntimeError(
                    "Nonfinite B2 lambda-calibration loss."
                )

            grad_scaler.scale(
                total_loss
            ).backward()

            grad_scaler.unscale_(
                optimizer
            )

            (
                gradients_finite,
                grad_count,
                grad_norm,
                bad_names,
            ) = finite_gradient_summary(
                model
            )

            if not gradients_finite:
                # Matched stream fairness is part of the frozen pilot.
                # Do not silently advance only one lambda candidate.
                raise RuntimeError(
                    "AMP/nonfinite gradient occurred during matched "
                    f"lambda calibration for lambda={lambda_value:g}. "
                    "Abort before candidate comparison to preserve "
                    "identical successful-step data/noise streams. "
                    f"bad={bad_names[:5]}"
                )

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
                    "Nonfinite B2 lambda-calibration clip norm."
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

            scale_before = float(
                grad_scaler.get_scale()
            )

            grad_scaler.step(
                optimizer
            )

            grad_scaler.update()

            scale_after = float(
                grad_scaler.get_scale()
            )

            if (
                scale_after
                < scale_before
            ):
                raise RuntimeError(
                    "GradScaler reduced scale despite finite pre-step "
                    "gradient audit. Stop matched lambda calibration "
                    "instead of allowing asymmetric AMP skipping."
                )

            successful_steps += 1

            v_value = float(
                v_loss.item()
            )

            spectral_value = float(
                spectral_loss.item()
            )

            total_value = float(
                total_loss.item()
            )

            rolling_v.append(
                v_value
            )

            rolling_spectral.append(
                spectral_value
            )

            rolling_total.append(
                total_value
            )

            row = {
                "lambda_spectral": float(
                    lambda_value
                ),
                "successful_step": (
                    successful_steps
                ),
                "attempted_batch": (
                    attempted_batches
                ),
                "v_loss": (
                    v_value
                ),
                "spectral_loss_db_huber": (
                    spectral_value
                ),
                "total_loss": (
                    total_value
                ),
                "rolling_50_v_loss": float(
                    np.mean(
                        list(
                            rolling_v
                        )
                    )
                ),
                "rolling_50_spectral_loss": float(
                    np.mean(
                        list(
                            rolling_spectral
                        )
                    )
                ),
                "rolling_50_total_loss": float(
                    np.mean(
                        list(
                            rolling_total
                        )
                    )
                ),
                "gradient_norm": float(
                    grad_norm
                ),
                "gradient_tensor_count": int(
                    grad_count
                ),
                "subject_embedding_grad_norm": float(
                    subject_embedding_grad_norm
                ),
                "subject_projection_grad_norm": float(
                    subject_projection_grad_norm
                ),
                "amp_scale": float(
                    scale_after
                ),
            }

            candidate_history.append(
                row
            )

            all_history_rows.append(
                row
            )

            if (
                successful_steps
                == 1
                or successful_steps
                % 50
                == 0
            ):
                print(
                    f"step "
                    f"{successful_steps:>3}/"
                    f"{SUCCESSFUL_STEPS_PER_LAMBDA} "
                    f"| v={v_value:.5f} "
                    f"| spec={spectral_value:.5f} "
                    f"| total={total_value:.5f} "
                    f"| roll_spec="
                    f"{row['rolling_50_spectral_loss']:.5f} "
                    f"| scale={scale_after:g} "
                    f"| elapsed="
                    f"{(time.perf_counter() - candidate_start)/60:.1f}m"
                )

        training_seconds = (
            time.perf_counter()
            - candidate_start
        )

        post_probe = evaluate_fixed_probe(
            model=model,
            x0=fixed_x0,
            state=fixed_state,
            subject=fixed_subject,
            wrong_subject=(
                fixed_wrong_subject
            ),
            timesteps=fixed_timesteps,
            xt=fixed_xt,
            target_v=fixed_target_v,
            alpha_bar=alpha_bar,
            spectral_loss_fn=(
                spectral_loss_fn
            ),
        )

        model.eval()

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

        first_50_v = float(
            np.mean(
                [
                    row[
                        "v_loss"
                    ]
                    for row in candidate_history[
                        :50
                    ]
                ]
            )
        )

        last_50_v = float(
            np.mean(
                [
                    row[
                        "v_loss"
                    ]
                    for row in candidate_history[
                        -50:
                    ]
                ]
            )
        )

        first_50_spectral = float(
            np.mean(
                [
                    row[
                        "spectral_loss_db_huber"
                    ]
                    for row in candidate_history[
                        :50
                    ]
                ]
            )
        )

        last_50_spectral = float(
            np.mean(
                [
                    row[
                        "spectral_loss_db_huber"
                    ]
                    for row in candidate_history[
                        -50:
                    ]
                ]
            )
        )

        final_state = {
            name: tensor.detach()
            .cpu()
            .clone()
            for name, tensor
            in model.state_dict().items()
        }

        final_state_hash = (
            state_dict_sha256(
                final_state
            )
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / (
                "b2_lambda_"
                + str(
                    lambda_value
                ).replace(
                    ".",
                    "p",
                )
                + "_pilot_final_v1.pt"
            )
        )

        checkpoint_payload = {
            "status": (
                "B2_LAMBDA_CALIBRATION_ENGINEERING_CHECKPOINT"
            ),
            "lambda_spectral": float(
                lambda_value
            ),
            "successful_optimizer_steps": (
                successful_steps
            ),
            "scientific_full_run_initialization_allowed": False,
            "note": (
                "Engineering lambda-calibration checkpoint only. "
                "Scientific B2 full training must restart from "
                "the frozen B1 generator."
            ),
            "frozen_B1_parent": str(
                FROZEN_B1_GENERATOR
            ),
            "frozen_B1_parent_sha256": (
                frozen_b1_hash_before
            ),
            "frozen_B2_protocol": str(
                B2_PROTOCOL_FILE
            ),
            "model_state_dict": (
                final_state
            ),
            "model_state_sha256": (
                final_state_hash
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "grad_scaler_state_dict": (
                grad_scaler.state_dict()
            ),
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
        }

        atomic_torch_save(
            checkpoint_payload,
            checkpoint_path,
        )

        generated_packages[
            (
                "lambda_"
                + str(
                    lambda_value
                ).replace(
                    ".",
                    "p",
                )
                + "_eo_norm"
            )
        ] = post_eo

        generated_packages[
            (
                "lambda_"
                + str(
                    lambda_value
                ).replace(
                    ".",
                    "p",
                )
                + "_ec_norm"
            )
        ] = post_ec

        per_lambda_details[
            str(
                lambda_value
            )
        ] = {
            "pre_probe": (
                pre_probe
            ),
            "post_probe": (
                post_probe
            ),
            "post_sampling": {
                key: value
                for key, value
                in post_sampling.items()
                if key
                not in (
                    "eo_uv",
                    "ec_uv",
                )
            },
            "first_50_mean_v_loss": (
                first_50_v
            ),
            "last_50_mean_v_loss": (
                last_50_v
            ),
            "first_50_mean_spectral_loss": (
                first_50_spectral
            ),
            "last_50_mean_spectral_loss": (
                last_50_spectral
            ),
            "training_seconds": (
                training_seconds
            ),
            "peak_allocated_mib": float(
                torch.cuda.max_memory_allocated()
                / 1024
                ** 2
            ),
            "engineering_checkpoint": str(
                checkpoint_path
            ),
            "engineering_checkpoint_model_state_sha256": (
                final_state_hash
            ),
        }

        print(
            f"post correct-subject v-MSE: "
            f"{post_probe['correct_subject_v_mse']:.6f}"
        )

        print(
            f"post wrong-correct gap: "
            f"{post_probe['wrong_minus_correct_v_mse']:.6e}"
        )

        print(
            f"post spectral loss: "
            f"{post_probe['spectral_loss_db_huber']:.6f}"
        )

        print(
            f"post normalized std EO/EC: "
            f"{post_sampling['eo_normalized_std']:.3f}"
            " / "
            f"{post_sampling['ec_normalized_std']:.3f}"
        )

        print(
            f"post O1/O2 alpha EC-EO: "
            f"{post_sampling['o1_alpha_effect_db']:.3f}"
            " / "
            f"{post_sampling['o2_alpha_effect_db']:.3f}"
            " dB"
        )

        print(
            f"training time: "
            f"{training_seconds/60:.2f} min"
        )

        del model
        del optimizer
        del grad_scaler

        torch.cuda.empty_cache()

    # ========================================================
    # Step 5 — Apply frozen eligibility / selection rule
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — APPLY PREDECLARED LAMBDA ELIGIBILITY RULE"
    )

    print(
        "=" * 80
    )

    lambda0 = per_lambda_details[
        "0.0"
    ]

    lambda0_post_probe = (
        lambda0[
            "post_probe"
        ]
    )

    lambda0_v_mse = float(
        lambda0_post_probe[
            "correct_subject_v_mse"
        ]
    )

    lambda0_gap = float(
        lambda0_post_probe[
            "wrong_minus_correct_v_mse"
        ]
    )

    lambda0_spectral = float(
        lambda0_post_probe[
            "spectral_loss_db_huber"
        ]
    )

    eligible_nonzero = []

    for lambda_value in LAMBDA_CANDIDATES:
        details = per_lambda_details[
            str(
                lambda_value
            )
        ]

        post_probe = details[
            "post_probe"
        ]

        post_sampling = details[
            "post_sampling"
        ]

        if lambda_value == 0.0:
            metrics_rows.append(
                {
                    "lambda_spectral": (
                        lambda_value
                    ),
                    "first_50_mean_v_loss": (
                        details[
                            "first_50_mean_v_loss"
                        ]
                    ),
                    "last_50_mean_v_loss": (
                        details[
                            "last_50_mean_v_loss"
                        ]
                    ),
                    "first_50_mean_spectral_loss": (
                        details[
                            "first_50_mean_spectral_loss"
                        ]
                    ),
                    "last_50_mean_spectral_loss": (
                        details[
                            "last_50_mean_spectral_loss"
                        ]
                    ),
                    "post_probe_correct_subject_v_mse": (
                        post_probe[
                            "correct_subject_v_mse"
                        ]
                    ),
                    "post_probe_wrong_subject_v_mse": (
                        post_probe[
                            "wrong_subject_v_mse"
                        ]
                    ),
                    "post_probe_wrong_minus_correct_v_mse": (
                        post_probe[
                            "wrong_minus_correct_v_mse"
                        ]
                    ),
                    "post_probe_spectral_loss_db_huber": (
                        post_probe[
                            "spectral_loss_db_huber"
                        ]
                    ),
                    "spectral_improvement_vs_lambda0_fraction": 0.0,
                    "post_sampling_eo_normalized_std": (
                        post_sampling[
                            "eo_normalized_std"
                        ]
                    ),
                    "post_sampling_ec_normalized_std": (
                        post_sampling[
                            "ec_normalized_std"
                        ]
                    ),
                    "post_sampling_o1_alpha_effect_db": (
                        post_sampling[
                            "o1_alpha_effect_db"
                        ]
                    ),
                    "post_sampling_o2_alpha_effect_db": (
                        post_sampling[
                            "o2_alpha_effect_db"
                        ]
                    ),
                    "eligibility_v_mse": True,
                    "eligibility_subject_gap": True,
                    "eligibility_eo_scale": True,
                    "eligibility_ec_scale": True,
                    "eligibility_spectral_improvement": True,
                    "eligible_nonzero_candidate": False,
                }
            )

            continue

        candidate_v_mse = float(
            post_probe[
                "correct_subject_v_mse"
            ]
        )

        candidate_gap = float(
            post_probe[
                "wrong_minus_correct_v_mse"
            ]
        )

        candidate_spectral = float(
            post_probe[
                "spectral_loss_db_huber"
            ]
        )

        spectral_improvement_fraction = float(
            (
                lambda0_spectral
                - candidate_spectral
            )
            / lambda0_spectral
        )

        eligibility_v_mse = bool(
            candidate_v_mse
            <= (
                V_MSE_MAX_RATIO_TO_LAMBDA0
                * lambda0_v_mse
            )
        )

        if lambda0_gap > 0.0:
            eligibility_subject_gap = bool(
                candidate_gap
                >= (
                    SUBJECT_GAP_MIN_RATIO_TO_LAMBDA0
                    * lambda0_gap
                )
            )
        else:
            eligibility_subject_gap = bool(
                candidate_gap
                >= 0.0
            )

        eligibility_eo_scale = bool(
            SAMPLING_STD_MIN
            <= float(
                post_sampling[
                    "eo_normalized_std"
                ]
            )
            <= SAMPLING_STD_MAX
        )

        eligibility_ec_scale = bool(
            SAMPLING_STD_MIN
            <= float(
                post_sampling[
                    "ec_normalized_std"
                ]
            )
            <= SAMPLING_STD_MAX
        )

        eligibility_spectral = bool(
            spectral_improvement_fraction
            >= SPECTRAL_IMPROVEMENT_MIN_FRACTION
        )

        eligible = bool(
            eligibility_v_mse
            and eligibility_subject_gap
            and eligibility_eo_scale
            and eligibility_ec_scale
            and eligibility_spectral
        )

        if eligible:
            eligible_nonzero.append(
                float(
                    lambda_value
                )
            )

        metrics_rows.append(
            {
                "lambda_spectral": (
                    lambda_value
                ),
                "first_50_mean_v_loss": (
                    details[
                        "first_50_mean_v_loss"
                    ]
                ),
                "last_50_mean_v_loss": (
                    details[
                        "last_50_mean_v_loss"
                    ]
                ),
                "first_50_mean_spectral_loss": (
                    details[
                        "first_50_mean_spectral_loss"
                    ]
                ),
                "last_50_mean_spectral_loss": (
                    details[
                        "last_50_mean_spectral_loss"
                    ]
                ),
                "post_probe_correct_subject_v_mse": (
                    candidate_v_mse
                ),
                "post_probe_wrong_subject_v_mse": (
                    post_probe[
                        "wrong_subject_v_mse"
                    ]
                ),
                "post_probe_wrong_minus_correct_v_mse": (
                    candidate_gap
                ),
                "post_probe_spectral_loss_db_huber": (
                    candidate_spectral
                ),
                "spectral_improvement_vs_lambda0_fraction": (
                    spectral_improvement_fraction
                ),
                "post_sampling_eo_normalized_std": (
                    post_sampling[
                        "eo_normalized_std"
                    ]
                ),
                "post_sampling_ec_normalized_std": (
                    post_sampling[
                        "ec_normalized_std"
                    ]
                ),
                "post_sampling_o1_alpha_effect_db": (
                    post_sampling[
                        "o1_alpha_effect_db"
                    ]
                ),
                "post_sampling_o2_alpha_effect_db": (
                    post_sampling[
                        "o2_alpha_effect_db"
                    ]
                ),
                "eligibility_v_mse": (
                    eligibility_v_mse
                ),
                "eligibility_subject_gap": (
                    eligibility_subject_gap
                ),
                "eligibility_eo_scale": (
                    eligibility_eo_scale
                ),
                "eligibility_ec_scale": (
                    eligibility_ec_scale
                ),
                "eligibility_spectral_improvement": (
                    eligibility_spectral
                ),
                "eligible_nonzero_candidate": (
                    eligible
                ),
            }
        )

    if eligible_nonzero:
        selected_lambda = min(
            eligible_nonzero,
            key=lambda value: (
                float(
                    per_lambda_details[
                        str(
                            value
                        )
                    ][
                        "post_probe"
                    ][
                        "spectral_loss_db_huber"
                    ]
                ),
                float(
                    value
                ),
            ),
        )

        diagnostic_direction = (
            "NONZERO_B2_LAMBDA_SELECTED"
        )
    else:
        selected_lambda = None

        diagnostic_direction = (
            "NO_NONZERO_B2_LAMBDA_ELIGIBLE_PROTOCOL_AMENDMENT_REQUIRED"
        )

    print(
        f"lambda=0 post correct-subject v-MSE: "
        f"{lambda0_v_mse:.6f}"
    )

    print(
        f"lambda=0 post subject gap: "
        f"{lambda0_gap:.6e}"
    )

    print(
        f"lambda=0 post spectral loss: "
        f"{lambda0_spectral:.6f}"
    )

    for row in metrics_rows:
        lambda_value = float(
            row[
                "lambda_spectral"
            ]
        )

        if lambda_value == 0.0:
            continue

        print(
            "\n"
            f"lambda={lambda_value:g}"
        )

        print(
            f"  v-MSE eligible: "
            f"{row['eligibility_v_mse']}"
        )

        print(
            f"  subject-gap eligible: "
            f"{row['eligibility_subject_gap']}"
        )

        print(
            f"  EO scale eligible: "
            f"{row['eligibility_eo_scale']}"
        )

        print(
            f"  EC scale eligible: "
            f"{row['eligibility_ec_scale']}"
        )

        print(
            f"  spectral improvement: "
            f"{100.0 * row['spectral_improvement_vs_lambda0_fraction']:.2f}%"
        )

        print(
            f"  spectral eligible: "
            f"{row['eligibility_spectral_improvement']}"
        )

        print(
            f"  FINAL ELIGIBLE: "
            f"{row['eligible_nonzero_candidate']}"
        )

    print(
        f"\neligible nonzero lambdas: "
        f"{eligible_nonzero}"
    )

    print(
        f"selected lambda: "
        f"{selected_lambda}"
    )

    print(
        f"diagnostic direction: "
        f"{diagnostic_direction}"
    )

    # ========================================================
    # Step 6 — Save artifacts
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — SAVE CALIBRATION ARTIFACTS"
    )

    print(
        "=" * 80
    )

    save_csv_rows(
        all_history_rows,
        HISTORY_CSV,
    )

    save_csv_rows(
        metrics_rows,
        METRICS_CSV,
    )

    np.savez_compressed(
        GENERATED_NPZ,
        **generated_packages,
    )

    save_loss_figure(
        all_history_rows
    )

    save_probe_figure(
        metrics_rows
    )

    save_scale_figure(
        metrics_rows
    )

    save_alpha_figure(
        metrics_rows
    )

    save_eligibility_figure(
        metrics_rows
    )

    total_seconds = (
        time.perf_counter()
        - run_start
    )

    frozen_b1_hash_after = (
        sha256_file(
            FROZEN_B1_GENERATOR
        )
    )

    if (
        frozen_b1_hash_after
        != frozen_b1_hash_before
    ):
        raise RuntimeError(
            "Frozen B1 changed during B2 lambda calibration."
        )

    summary = {
        "status": (
            "B2_LAMBDA_CALIBRATION_COMPLETE"
        ),
        "scientific_full_B2_started": False,
        "frozen_B1": {
            "path": str(
                FROZEN_B1_GENERATOR
            ),
            "file_sha256": (
                frozen_b1_hash_before
            ),
            "model_state_sha256": (
                frozen_b1_state_hash
            ),
            "unchanged_after_calibration": True,
        },
        "frozen_B2_protocol": str(
            B2_PROTOCOL_FILE
        ),
        "candidate_lambdas": (
            LAMBDA_CANDIDATES
        ),
        "successful_steps_per_lambda": (
            SUCCESSFUL_STEPS_PER_LAMBDA
        ),
        "parent_probe": (
            parent_probe
        ),
        "parent_sampling": {
            key: value
            for key, value
            in parent_sampling.items()
            if key
            not in (
                "eo_uv",
                "ec_uv",
            )
        },
        "per_lambda_details": (
            per_lambda_details
        ),
        "metrics_table": (
            metrics_rows
        ),
        "lambda0_reference": {
            "post_correct_subject_v_mse": (
                lambda0_v_mse
            ),
            "post_wrong_minus_correct_v_mse": (
                lambda0_gap
            ),
            "post_spectral_loss_db_huber": (
                lambda0_spectral
            ),
        },
        "eligible_nonzero_lambdas": (
            eligible_nonzero
        ),
        "selected_lambda": (
            selected_lambda
        ),
        "diagnostic_direction": (
            diagnostic_direction
        ),
        "selection_rule_satisfied": (
            selected_lambda
            is not None
        ),
        "runtime_seconds": (
            total_seconds
        ),
        "runtime_minutes": (
            total_seconds
            / 60.0
        ),
        "artifacts": {
            "protocol_snapshot": str(
                PROTOCOL_SNAPSHOT
            ),
            "history_csv": str(
                HISTORY_CSV
            ),
            "metrics_csv": str(
                METRICS_CSV
            ),
            "generated_npz": str(
                GENERATED_NPZ
            ),
            "loss_figure": str(
                LOSS_FIGURE
            ),
            "probe_figure": str(
                PROBE_FIGURE
            ),
            "scale_figure": str(
                SCALE_FIGURE
            ),
            "alpha_figure": str(
                ALPHA_FIGURE
            ),
            "eligibility_figure": str(
                ELIGIBILITY_FIGURE
            ),
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "bulk_synthetic_generation_started": False,
            "pilot_checkpoints_reusable_for_scientific_full_run": False,
            "scientific_full_run_must_restart_from_frozen_B1": True,
        },
        "next": (
            "If selected_lambda is non-null, freeze the selected "
            "lambda and start B2 scientific full fine-tuning from "
            "frozen B1. Otherwise amend B2 protocol before training."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
    )

    print(
        f"metrics CSV: "
        f"{METRICS_CSV}"
    )

    print(
        f"summary JSON: "
        f"{SUMMARY_JSON}"
    )

    print(
        f"generated package: "
        f"{GENERATED_NPZ}"
    )

    print(
        "frozen B1 unchanged: PASS"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B2 LAMBDA CALIBRATION PILOT COMPLETE"
    )

    print(
        "=" * 80
    )

    for row in metrics_rows:
        print(
            "\n"
            f"lambda={row['lambda_spectral']:g}"
        )

        print(
            f"  fixed correct-subject v-MSE: "
            f"{row['post_probe_correct_subject_v_mse']:.6f}"
        )

        print(
            f"  fixed subject gap: "
            f"{row['post_probe_wrong_minus_correct_v_mse']:.6e}"
        )

        print(
            f"  fixed spectral loss: "
            f"{row['post_probe_spectral_loss_db_huber']:.6f}"
        )

        print(
            f"  spectral improvement vs lambda0: "
            f"{100.0 * row['spectral_improvement_vs_lambda0_fraction']:.2f}%"
        )

        print(
            f"  sampling std EO/EC: "
            f"{row['post_sampling_eo_normalized_std']:.3f}"
            " / "
            f"{row['post_sampling_ec_normalized_std']:.3f}"
        )

        print(
            f"  O1/O2 alpha EC-EO: "
            f"{row['post_sampling_o1_alpha_effect_db']:.3f}"
            " / "
            f"{row['post_sampling_o2_alpha_effect_db']:.3f}"
            " dB"
        )

        print(
            f"  eligible nonzero: "
            f"{row['eligible_nonzero_candidate']}"
        )

    print(
        f"\neligible nonzero lambdas: "
        f"{eligible_nonzero}"
    )

    print(
        f"selected lambda: "
        f"{selected_lambda}"
    )

    print(
        f"diagnostic direction: "
        f"{diagnostic_direction}"
    )

    print(
        f"runtime: "
        f"{total_seconds/60:.2f} min"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "B2 scientific full training started: NO"
    )

    print(
        "Pilot checkpoints reusable for scientific full run: NO"
    )

    print(
        "\nNEXT:"
    )

    if selected_lambda is None:
        print(
            "STOP — INTERPRET FAILURE AND AMEND B2 PROTOCOL "
            "BEFORE ANY FULL B2 TRAINING"
        )
    else:
        print(
            "FREEZE SELECTED B2 LAMBDA AND START "
            "THE 20-EPOCH SCIENTIFIC B2 FULL RUN "
            "FROM FROZEN B1"
        )


if __name__ == "__main__":
    main()
