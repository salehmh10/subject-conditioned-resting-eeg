from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

from s2_32_b0_unet_shape_test import (
    B0StateOnlyUNet1D,
)

from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
    q_sample,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

AMP_SAFE_RUN = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0"
    / "full_amp_safe_v1"
)

LATEST_CHECKPOINT = (
    AMP_SAFE_RUN
    / "checkpoints"
    / "latest.pt"
)

BEST_CHECKPOINT = (
    AMP_SAFE_RUN
    / "checkpoints"
    / "best_ema.pt"
)


# ============================================================
# Outputs
# ============================================================

METRIC_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "b0"
)

FIGURE_DIR = (
    REPO_ROOT
    / "outputs"
    / "figures"
    / "scenario2"
    / "b0"
    / "reverse_diagnostic_v1"
)

LOG_DIR = (
    REPO_ROOT
    / "outputs"
    / "logs"
    / "scenario2"
    / "b0"
)

SUMMARY_FILE = (
    METRIC_DIR
    / "b0_reverse_diagnostic_v1.json"
)

TIMESTEP_CSV = (
    METRIC_DIR
    / "b0_timestep_error_profile_v1.csv"
)

ORACLE_TRAJECTORY_CSV = (
    LOG_DIR
    / "b0_oracle_reverse_trajectory_v1.csv"
)

MODEL_TRAJECTORY_CSV = (
    LOG_DIR
    / "b0_model_reverse_trajectory_v1.csv"
)

EPSILON_FIGURE = (
    FIGURE_DIR
    / "b0_epsilon_mse_by_timestep_v1.png"
)

X0_RMSE_FIGURE = (
    FIGURE_DIR
    / "b0_x0_rmse_by_timestep_v1.png"
)

X0_STD_FIGURE = (
    FIGURE_DIR
    / "b0_x0_predicted_std_by_timestep_v1.png"
)

TRAJECTORY_FIGURE = (
    FIGURE_DIR
    / "b0_reverse_std_trajectory_v1.png"
)

AMPLIFICATION_FIGURE = (
    FIGURE_DIR
    / "b0_epsilon_to_x0_amplification_v1.png"
)


# ============================================================
# Diagnostic configuration
# ============================================================

EXPECTED_LATEST_COMPLETE_EPOCH = 125

SEED = 20260908

TIMESTEPS = 1000
COSINE_S = 0.008

N_CHANNELS = 19
N_TIMES = 512

BALANCED_PER_STATE = 64
DIAGNOSTIC_BATCH_SIZE = 2 * BALANCED_PER_STATE

ORACLE_SUBSET_SIZE = 16
FREE_SAMPLING_PER_STATE = 8

TIMESTEP_GRID = [
    0,
    1,
    5,
    10,
    25,
    50,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    950,
    975,
    990,
    995,
    998,
    999,
]

# These are diagnostic engineering thresholds only.
# They do NOT define scientific acceptance of B0.
ORACLE_FINAL_RMSE_TOL = 1e-4
POSTERIOR_FORMULA_TOL = 1e-4


# ============================================================
# Helpers
# ============================================================


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()



def atomic_json_save(data: dict, path: Path) -> None:
    temporary = Path(str(path) + ".tmp")

    with temporary.open(
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

    os.replace(temporary, path)



def save_csv(
    rows: list[dict],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            f"Cannot save empty CSV: {path}"
        )

    temporary = Path(str(path) + ".tmp")

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)



def subset_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if int(mask.sum().item()) == 0:
        return float("nan")

    value = torch.mean(
        (
            prediction[mask].float()
            - target[mask].float()
        )
        ** 2
    )

    return float(value.item())



def subset_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    mse = subset_mse(
        prediction,
        target,
        mask,
    )

    return float(math.sqrt(mse))


# ============================================================
# Reverse schedule
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
            alpha_bar[:-1],
        ],
        dim=0,
    )

    posterior_variance = (
        betas
        * (
            1.0
            - alpha_bar_prev
        )
        / (
            1.0
            - alpha_bar
        )
    )

    posterior_variance = torch.clamp(
        posterior_variance,
        min=0.0,
    )

    posterior_mean_coef1 = (
        betas
        * torch.sqrt(
            alpha_bar_prev
        )
        / (
            1.0
            - alpha_bar
        )
    )

    posterior_mean_coef2 = (
        (
            1.0
            - alpha_bar_prev
        )
        * torch.sqrt(
            alphas
        )
        / (
            1.0
            - alpha_bar
        )
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "posterior_variance": posterior_variance,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }


# ============================================================
# DDPM reverse helpers
# ============================================================


def reverse_mean_from_epsilon(
    x_t: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    timestep: int,
    reverse_schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    beta_t = reverse_schedule[
        "betas"
    ][timestep]

    alpha_t = reverse_schedule[
        "alphas"
    ][timestep]

    alpha_bar_t = reverse_schedule[
        "alpha_bar"
    ][timestep]

    return (
        x_t
        - (
            beta_t
            / torch.sqrt(
                1.0
                - alpha_bar_t
            )
        )
        * epsilon_prediction
    ) / torch.sqrt(alpha_t)



def oracle_epsilon_from_x0(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    timestep: int,
    reverse_schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    alpha_bar_t = reverse_schedule[
        "alpha_bar"
    ][timestep]

    denominator = torch.sqrt(
        1.0
        - alpha_bar_t
    )

    return (
        x_t
        - torch.sqrt(
            alpha_bar_t
        )
        * x0
    ) / denominator



def closed_form_posterior_mean(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    timestep: int,
    reverse_schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    coef1 = reverse_schedule[
        "posterior_mean_coef1"
    ][timestep]

    coef2 = reverse_schedule[
        "posterior_mean_coef2"
    ][timestep]

    return (
        coef1 * x0
        + coef2 * x_t
    )


# ============================================================
# Fixed balanced Train batch
# ============================================================


def build_balanced_train_batch(
    scaler: dict,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    CachedRealTrainWindowDataset,
]:
    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    records, subject_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    if len(records) != 47_835:
        raise RuntimeError(
            f"Unexpected Train window count: "
            f"{len(records)}"
        )

    if len(subject_mapping) != 198:
        raise RuntimeError(
            f"Unexpected subject count: "
            f"{len(subject_mapping)}"
        )

    eo_records = [
        record
        for record in records
        if int(record.label) == 0
    ][:BALANCED_PER_STATE]

    ec_records = [
        record
        for record in records
        if int(record.label) == 1
    ][:BALANCED_PER_STATE]

    if len(eo_records) != BALANCED_PER_STATE:
        raise RuntimeError(
            "Insufficient EO diagnostic records."
        )

    if len(ec_records) != BALANCED_PER_STATE:
        raise RuntimeError(
            "Insufficient EC diagnostic records."
        )

    selected_records = (
        eo_records
        + ec_records
    )

    dataset = CachedRealTrainWindowDataset(
        records=selected_records,
        scaler=scaler,
        max_open_files=8,
    )

    loader = DataLoader(
        dataset,
        batch_size=DIAGNOSTIC_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    batch = next(iter(loader))

    x0 = batch[
        "x"
    ].to(
        device,
        non_blocking=True,
    )

    labels = batch[
        "y"
    ].to(
        device,
        non_blocking=True,
    )

    if x0.shape != (
        DIAGNOSTIC_BATCH_SIZE,
        N_CHANNELS,
        N_TIMES,
    ):
        raise RuntimeError(
            f"Unexpected diagnostic batch shape: "
            f"{x0.shape}"
        )

    counts = torch.bincount(
        labels,
        minlength=2,
    )

    if int(counts[0].item()) != BALANCED_PER_STATE:
        raise RuntimeError(
            "EO diagnostic count mismatch."
        )

    if int(counts[1].item()) != BALANCED_PER_STATE:
        raise RuntimeError(
            "EC diagnostic count mismatch."
        )

    if not torch.isfinite(x0).all():
        raise RuntimeError(
            "Nonfinite real Train diagnostic batch."
        )

    return (
        x0,
        labels,
        dataset,
    )


# ============================================================
# Diagnostic 1 — model error by timestep
# ============================================================


@torch.inference_mode()
def timestep_error_profile(
    model: torch.nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor,
    schedule_device: dict[str, torch.Tensor],
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[dict], float]:
    generator = torch.Generator(
        device="cuda"
    )
    generator.manual_seed(
        SEED + 100
    )

    # Reuse the SAME epsilon at every tested timestep so the
    # profile is not confounded by different random noise draws.
    epsilon = torch.randn(
        x0.shape,
        device=device,
        dtype=x0.dtype,
        generator=generator,
    )

    eo_mask = labels == 0
    ec_mask = labels == 1

    rows: list[dict] = []

    posterior_formula_max_abs_diff = 0.0

    true_x0_std = float(
        x0.float().std(
            unbiased=False
        ).item()
    )

    for timestep_value in TIMESTEP_GRID:
        timesteps = torch.full(
            (
                x0.shape[0],
            ),
            fill_value=timestep_value,
            device=device,
            dtype=torch.long,
        )

        x_t = q_sample(
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

        epsilon_prediction = model(
            x_t,
            timesteps,
            labels,
        )

        if not torch.isfinite(
            epsilon_prediction
        ).all():
            raise RuntimeError(
                f"Nonfinite epsilon prediction at "
                f"t={timestep_value}."
            )

        alpha_bar_t = reverse_schedule[
            "alpha_bar"
        ][timestep_value]

        sqrt_alpha_bar_t = torch.sqrt(
            alpha_bar_t
        )

        sqrt_one_minus_t = torch.sqrt(
            1.0
            - alpha_bar_t
        )

        x0_prediction = (
            x_t
            - sqrt_one_minus_t
            * epsilon_prediction
        ) / sqrt_alpha_bar_t

        if not torch.isfinite(
            x0_prediction
        ).all():
            raise RuntimeError(
                f"Nonfinite x0 prediction at "
                f"t={timestep_value}."
            )

        model_reverse_mean = (
            reverse_mean_from_epsilon(
                x_t=x_t,
                epsilon_prediction=(
                    epsilon_prediction
                ),
                timestep=timestep_value,
                reverse_schedule=(
                    reverse_schedule
                ),
            )
        )

        oracle_reverse_mean = (
            reverse_mean_from_epsilon(
                x_t=x_t,
                epsilon_prediction=epsilon,
                timestep=timestep_value,
                reverse_schedule=(
                    reverse_schedule
                ),
            )
        )

        closed_form_mean = (
            closed_form_posterior_mean(
                x_t=x_t,
                x0=x0,
                timestep=timestep_value,
                reverse_schedule=(
                    reverse_schedule
                ),
            )
        )

        formula_difference = float(
            torch.max(
                torch.abs(
                    oracle_reverse_mean
                    - closed_form_mean
                )
            )
            .item()
        )

        posterior_formula_max_abs_diff = max(
            posterior_formula_max_abs_diff,
            formula_difference,
        )

        epsilon_mse_all = subset_mse(
            epsilon_prediction,
            epsilon,
            torch.ones_like(
                labels,
                dtype=torch.bool,
            ),
        )

        epsilon_rmse_all = float(
            math.sqrt(
                epsilon_mse_all
            )
        )

        x0_rmse_all = subset_rmse(
            x0_prediction,
            x0,
            torch.ones_like(
                labels,
                dtype=torch.bool,
            ),
        )

        amplification = float(
            (
                sqrt_one_minus_t
                / sqrt_alpha_bar_t
            )
            .item()
        )

        row = {
            "timestep": int(
                timestep_value
            ),
            "alpha_bar": float(
                alpha_bar_t.item()
            ),
            "sqrt_alpha_bar": float(
                sqrt_alpha_bar_t.item()
            ),
            "epsilon_to_x0_error_amplification": (
                amplification
            ),
            "epsilon_mse_all": float(
                epsilon_mse_all
            ),
            "epsilon_rmse_all": float(
                epsilon_rmse_all
            ),
            "epsilon_mse_eo": float(
                subset_mse(
                    epsilon_prediction,
                    epsilon,
                    eo_mask,
                )
            ),
            "epsilon_mse_ec": float(
                subset_mse(
                    epsilon_prediction,
                    epsilon,
                    ec_mask,
                )
            ),
            "x0_rmse_all": float(
                x0_rmse_all
            ),
            "x0_rmse_eo": float(
                subset_rmse(
                    x0_prediction,
                    x0,
                    eo_mask,
                )
            ),
            "x0_rmse_ec": float(
                subset_rmse(
                    x0_prediction,
                    x0,
                    ec_mask,
                )
            ),
            "x0_true_std": float(
                true_x0_std
            ),
            "x0_pred_std": float(
                x0_prediction
                .float()
                .std(
                    unbiased=False
                )
                .item()
            ),
            "x0_pred_std_eo": float(
                x0_prediction[
                    eo_mask
                ]
                .float()
                .std(
                    unbiased=False
                )
                .item()
            ),
            "x0_pred_std_ec": float(
                x0_prediction[
                    ec_mask
                ]
                .float()
                .std(
                    unbiased=False
                )
                .item()
            ),
            "epsilon_pred_std": float(
                epsilon_prediction
                .float()
                .std(
                    unbiased=False
                )
                .item()
            ),
            "model_reverse_mean_std": float(
                model_reverse_mean
                .float()
                .std(
                    unbiased=False
                )
                .item()
            ),
            "oracle_reverse_mean_std": float(
                oracle_reverse_mean
                .float()
                .std(
                    unbiased=False
                )
                .item()
            ),
            "model_vs_oracle_reverse_mean_rmse": float(
                torch.sqrt(
                    torch.mean(
                        (
                            model_reverse_mean.float()
                            - oracle_reverse_mean.float()
                        )
                        ** 2
                    )
                )
                .item()
            ),
            "posterior_formula_max_abs_diff": float(
                formula_difference
            ),
            "x0_rmse_from_epsilon_identity": float(
                amplification
                * epsilon_rmse_all
            ),
        }

        rows.append(row)

        print(
            f"t={timestep_value:>3}"
            f" | eps_MSE={epsilon_mse_all:.6f}"
            f" | eps_RMSE={epsilon_rmse_all:.6f}"
            f" | amp={amplification:>11.2f}"
            f" | x0_RMSE={x0_rmse_all:>11.3f}"
            f" | x0_pred_std={row['x0_pred_std']:>11.3f}"
        )

    return (
        rows,
        posterior_formula_max_abs_diff,
    )


# ============================================================
# Diagnostic 2 — oracle reverse trajectory
# ============================================================


@torch.inference_mode()
def oracle_reverse_trajectory(
    x0: torch.Tensor,
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[dict], float, float]:
    x0 = x0[
        :ORACLE_SUBSET_SIZE
    ].clone()

    generator = torch.Generator(
        device="cuda"
    )
    generator.manual_seed(
        SEED + 200
    )

    forward_epsilon = torch.randn(
        x0.shape,
        device=device,
        dtype=x0.dtype,
        generator=generator,
    )

    alpha_bar_terminal = reverse_schedule[
        "alpha_bar"
    ][TIMESTEPS - 1]

    x = (
        torch.sqrt(
            alpha_bar_terminal
        )
        * x0
        + torch.sqrt(
            1.0
            - alpha_bar_terminal
        )
        * forward_epsilon
    )

    rows: list[dict] = [
        {
            "stage": "initial_xT",
            "timestep": TIMESTEPS,
            "x_mean": float(
                x.mean().item()
            ),
            "x_std": float(
                x.std(
                    unbiased=False
                ).item()
            ),
            "rmse_to_x0": float(
                torch.sqrt(
                    torch.mean(
                        (
                            x.float()
                            - x0.float()
                        )
                        ** 2
                    )
                )
                .item()
            ),
        }
    ]

    for timestep_value in reversed(
        range(TIMESTEPS)
    ):
        epsilon_oracle = (
            oracle_epsilon_from_x0(
                x_t=x,
                x0=x0,
                timestep=timestep_value,
                reverse_schedule=(
                    reverse_schedule
                ),
            )
        )

        model_mean = reverse_mean_from_epsilon(
            x_t=x,
            epsilon_prediction=epsilon_oracle,
            timestep=timestep_value,
            reverse_schedule=reverse_schedule,
        )

        if timestep_value > 0:
            noise = torch.randn(
                x.shape,
                device=device,
                dtype=x.dtype,
                generator=generator,
            )

            variance = reverse_schedule[
                "posterior_variance"
            ][timestep_value]

            x = (
                model_mean
                + torch.sqrt(
                    variance
                )
                * noise
            )

        else:
            x = model_mean

        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"Oracle reverse became nonfinite "
                f"at t={timestep_value}."
            )

        rmse = float(
            torch.sqrt(
                torch.mean(
                    (
                        x.float()
                        - x0.float()
                    )
                    ** 2
                )
            )
            .item()
        )

        rows.append(
            {
                "stage": "after_reverse_step",
                "timestep": int(
                    timestep_value
                ),
                "x_mean": float(
                    x.mean().item()
                ),
                "x_std": float(
                    x.std(
                        unbiased=False
                    ).item()
                ),
                "rmse_to_x0": rmse,
            }
        )

    final_rmse = float(
        torch.sqrt(
            torch.mean(
                (
                    x.float()
                    - x0.float()
                )
                ** 2
            )
        )
        .item()
    )

    final_max_abs_error = float(
        torch.max(
            torch.abs(
                x.float()
                - x0.float()
            )
        )
        .item()
    )

    return (
        rows,
        final_rmse,
        final_max_abs_error,
    )


# ============================================================
# Diagnostic 3 — free model reverse trajectory
# ============================================================


@torch.inference_mode()
def model_free_reverse_trajectory(
    model: torch.nn.Module,
    eye_state_value: int,
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> list[dict]:
    generator = torch.Generator(
        device="cuda"
    )
    generator.manual_seed(seed)

    shape = (
        FREE_SAMPLING_PER_STATE,
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
            FREE_SAMPLING_PER_STATE,
        ),
        fill_value=eye_state_value,
        device=device,
        dtype=torch.long,
    )

    rows: list[dict] = [
        {
            "eye_state": int(
                eye_state_value
            ),
            "stage": "initial_noise",
            "timestep": TIMESTEPS,
            "x_mean": float(
                x.mean().item()
            ),
            "x_std": float(
                x.std(
                    unbiased=False
                ).item()
            ),
            "epsilon_pred_mean": float("nan"),
            "epsilon_pred_std": float("nan"),
        }
    ]

    for timestep_value in reversed(
        range(TIMESTEPS)
    ):
        timestep_tensor = torch.full(
            (
                FREE_SAMPLING_PER_STATE,
            ),
            fill_value=timestep_value,
            device=device,
            dtype=torch.long,
        )

        epsilon_prediction = model(
            x,
            timestep_tensor,
            state,
        )

        if not torch.isfinite(
            epsilon_prediction
        ).all():
            raise RuntimeError(
                f"Model free reverse epsilon became "
                f"nonfinite at t={timestep_value}, "
                f"state={eye_state_value}."
            )

        model_mean = reverse_mean_from_epsilon(
            x_t=x,
            epsilon_prediction=(
                epsilon_prediction
            ),
            timestep=timestep_value,
            reverse_schedule=reverse_schedule,
        )

        if timestep_value > 0:
            noise = torch.randn(
                shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

            variance = reverse_schedule[
                "posterior_variance"
            ][timestep_value]

            x = (
                model_mean
                + torch.sqrt(
                    variance
                )
                * noise
            )
        else:
            x = model_mean

        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"Model free reverse state became "
                f"nonfinite at t={timestep_value}, "
                f"state={eye_state_value}."
            )

        rows.append(
            {
                "eye_state": int(
                    eye_state_value
                ),
                "stage": "after_reverse_step",
                "timestep": int(
                    timestep_value
                ),
                "x_mean": float(
                    x.mean().item()
                ),
                "x_std": float(
                    x.std(
                        unbiased=False
                    ).item()
                ),
                "epsilon_pred_mean": float(
                    epsilon_prediction
                    .mean()
                    .item()
                ),
                "epsilon_pred_std": float(
                    epsilon_prediction
                    .std(
                        unbiased=False
                    )
                    .item()
                ),
            }
        )

    return rows


# ============================================================
# Figures
# ============================================================



def save_timestep_figures(
    rows: list[dict],
) -> None:
    timesteps = np.asarray(
        [
            row["timestep"]
            for row in rows
        ],
        dtype=np.int64,
    )

    epsilon_mse = np.asarray(
        [
            row["epsilon_mse_all"]
            for row in rows
        ],
        dtype=np.float64,
    )

    x0_rmse = np.asarray(
        [
            row["x0_rmse_all"]
            for row in rows
        ],
        dtype=np.float64,
    )

    x0_pred_std = np.asarray(
        [
            row["x0_pred_std"]
            for row in rows
        ],
        dtype=np.float64,
    )

    true_std = float(
        rows[0]["x0_true_std"]
    )

    amplification = np.asarray(
        [
            row[
                "epsilon_to_x0_error_amplification"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    # Epsilon MSE
    figure = plt.figure(
        figsize=(10, 6)
    )
    axis = figure.add_subplot(111)
    axis.plot(
        timesteps,
        epsilon_mse,
        marker="o",
    )
    axis.set_xlabel("Diffusion timestep")
    axis.set_ylabel("Epsilon-prediction MSE")
    axis.set_title(
        "B0 Epsilon Prediction Error by Timestep"
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        EPSILON_FIGURE,
        dpi=160,
    )
    plt.close(figure)

    # x0 RMSE
    figure = plt.figure(
        figsize=(10, 6)
    )
    axis = figure.add_subplot(111)
    axis.semilogy(
        timesteps,
        np.maximum(
            x0_rmse,
            1e-12,
        ),
        marker="o",
    )
    axis.set_xlabel("Diffusion timestep")
    axis.set_ylabel(
        "Predicted x0 RMSE (normalized units)"
    )
    axis.set_title(
        "B0 x0 Reconstruction Error from Epsilon Prediction"
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        X0_RMSE_FIGURE,
        dpi=160,
    )
    plt.close(figure)

    # x0 predicted std
    figure = plt.figure(
        figsize=(10, 6)
    )
    axis = figure.add_subplot(111)
    axis.semilogy(
        timesteps,
        np.maximum(
            x0_pred_std,
            1e-12,
        ),
        marker="o",
        label="Predicted x0 std",
    )
    axis.axhline(
        true_std,
        label="Real Train batch std",
    )
    axis.set_xlabel("Diffusion timestep")
    axis.set_ylabel("Standard deviation")
    axis.set_title(
        "B0 Predicted x0 Scale by Timestep"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        X0_STD_FIGURE,
        dpi=160,
    )
    plt.close(figure)

    # amplification
    figure = plt.figure(
        figsize=(10, 6)
    )
    axis = figure.add_subplot(111)
    axis.semilogy(
        timesteps,
        np.maximum(
            amplification,
            1e-12,
        ),
        marker="o",
    )
    axis.set_xlabel("Diffusion timestep")
    axis.set_ylabel(
        "sqrt(1-alpha_bar_t) / sqrt(alpha_bar_t)"
    )
    axis.set_title(
        "B0 Epsilon Error → x0 Error Amplification"
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        AMPLIFICATION_FIGURE,
        dpi=160,
    )
    plt.close(figure)



def save_trajectory_figure(
    oracle_rows: list[dict],
    model_rows: list[dict],
    real_target_std: float,
) -> None:
    oracle_t = np.asarray(
        [
            row["timestep"]
            for row in oracle_rows
        ],
        dtype=np.int64,
    )

    oracle_std = np.asarray(
        [
            row["x_std"]
            for row in oracle_rows
        ],
        dtype=np.float64,
    )

    figure = plt.figure(
        figsize=(11, 7)
    )

    axis = figure.add_subplot(111)

    axis.plot(
        oracle_t,
        oracle_std,
        label="Oracle reverse from real q(x_T|x0)",
    )

    for state_value, state_name in (
        (0, "Model free reverse EO"),
        (1, "Model free reverse EC"),
    ):
        state_rows = [
            row
            for row in model_rows
            if int(row["eye_state"])
            == state_value
        ]

        state_t = np.asarray(
            [
                row["timestep"]
                for row in state_rows
            ],
            dtype=np.int64,
        )

        state_std = np.asarray(
            [
                row["x_std"]
                for row in state_rows
            ],
            dtype=np.float64,
        )

        axis.plot(
            state_t,
            state_std,
            label=state_name,
        )

    axis.axhline(
        real_target_std,
        label="Real diagnostic x0 std",
    )

    axis.set_yscale("log")
    axis.set_xlabel(
        "Reverse-process timestep"
    )
    axis.set_ylabel(
        "Normalized signal standard deviation"
    )
    axis.set_title(
        "B0 Reverse-Process Scale Trajectory"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.invert_xaxis()

    figure.tight_layout()

    figure.savefig(
        TRAJECTORY_FIGURE,
        dpi=160,
    )

    plt.close(figure)


# ============================================================
# Main
# ============================================================



def main() -> None:
    print("=" * 80)
    print(
        "SCENARIO 2 — "
        "B0 REVERSE-PROCESS DIAGNOSTIC"
    )
    print("=" * 80)

    print(
        "\nGoal: distinguish sampler-equation problems "
        "from learned high-timestep prediction problems."
    )

    print(
        "\nReal data used: TRAIN ONLY"
    )
    print(
        "Validation EEG accessed: NO"
    )
    print(
        "Final Test EEG accessed: NO"
    )
    print(
        "B0 training updates executed: NO"
    )

    for directory in (
        METRIC_DIR,
        FIGURE_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — checkpoint audit
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT B0 CHECKPOINTS")
    print("=" * 80)

    if not LATEST_CHECKPOINT.is_file():
        raise FileNotFoundError(
            LATEST_CHECKPOINT
        )

    if not BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(
            BEST_CHECKPOINT
        )

    latest = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    best = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    latest_epoch = int(
        latest["epoch"]
    )

    best_epoch = int(
        best["epoch"]
    )

    best_validation_mse = float(
        best[
            "best_ema_validation_mse"
        ]
    )

    print(
        f"latest complete epoch: "
        f"{latest_epoch}"
    )

    print(
        f"best EMA epoch: "
        f"{best_epoch}"
    )

    print(
        f"best EMA Validation MSE: "
        f"{best_validation_mse:.6f}"
    )

    if latest_epoch != (
        EXPECTED_LATEST_COMPLETE_EPOCH
    ):
        raise RuntimeError(
            f"Expected latest complete epoch "
            f"{EXPECTED_LATEST_COMPLETE_EPOCH}, "
            f"found {latest_epoch}."
        )

    if best_epoch > latest_epoch:
        raise RuntimeError(
            "Best checkpoint epoch exceeds latest "
            "complete epoch."
        )

    latest_hash = sha256(
        LATEST_CHECKPOINT
    )

    best_hash = sha256(
        BEST_CHECKPOINT
    )

    print(
        f"latest SHA256: {latest_hash}"
    )
    print(
        f"best SHA256: {best_hash}"
    )
    print("checkpoint audit: PASS")

    # ========================================================
    # Step 2 — device and model
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — LOAD BEST EMA MODEL")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for this diagnostic."
        )

    device = torch.device("cuda")

    print(
        f"PyTorch: {torch.__version__}"
    )
    print(
        f"CUDA runtime: {torch.version.cuda}"
    )
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    contract = best["contract"]

    if contract[
        "model_name"
    ] != "B0_STATE_ONLY_DDPM":
        raise RuntimeError(
            "Checkpoint is not B0."
        )

    model = B0StateOnlyUNet1D(
        contract
    ).to(device)

    # best_ema.pt exports EMA weights in both fields.
    model.load_state_dict(
        best["model_state_dict"],
        strict=True,
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"parameters: {parameter_count:,}"
    )
    print("weights source: BEST EMA")

    # ========================================================
    # Step 3 — schedule
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — BUILD FORWARD / REVERSE SCHEDULE")
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

    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    print(
        f"beta[0]: "
        f"{schedule_cpu['betas'][0].item():.8f}"
    )
    print(
        f"beta[-1]: "
        f"{schedule_cpu['betas'][-1].item():.8f}"
    )
    print(
        f"alpha_bar[-1]: "
        f"{schedule_cpu['alpha_cumprod'][-1].item():.12f}"
    )
    print("schedule: PASS")

    # ========================================================
    # Step 4 — frozen Train-only diagnostic batch
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — BUILD BALANCED REAL TRAIN DIAGNOSTIC BATCH")
    print("=" * 80)

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

    x0, labels, dataset = (
        build_balanced_train_batch(
            scaler=scaler,
            device=device,
        )
    )

    try:
        counts = torch.bincount(
            labels,
            minlength=2,
        )

        real_x0_mean = float(
            x0.mean().item()
        )

        real_x0_std = float(
            x0.std(
                unbiased=False
            ).item()
        )

        print(
            f"batch shape: {tuple(x0.shape)}"
        )
        print(
            f"EO: {int(counts[0].item())}"
        )
        print(
            f"EC: {int(counts[1].item())}"
        )
        print(
            f"real normalized mean: "
            f"{real_x0_mean:.6f}"
        )
        print(
            f"real normalized std: "
            f"{real_x0_std:.6f}"
        )

        # ====================================================
        # Step 5 — error profile by timestep
        # ====================================================

        print("\n" + "=" * 80)
        print("STEP 5 — MODEL ERROR PROFILE BY TIMESTEP")
        print("=" * 80)

        (
            timestep_rows,
            posterior_formula_max_abs_diff,
        ) = timestep_error_profile(
            model=model,
            x0=x0,
            labels=labels,
            schedule_device=schedule_device,
            reverse_schedule=reverse_schedule,
            device=device,
        )

        save_csv(
            timestep_rows,
            TIMESTEP_CSV,
        )

        print(
            f"timestep CSV: {TIMESTEP_CSV}"
        )

        print(
            f"max oracle-epsilon vs closed-form "
            f"posterior mean difference: "
            f"{posterior_formula_max_abs_diff:.3e}"
        )

        posterior_formula_pass = (
            posterior_formula_max_abs_diff
            <= POSTERIOR_FORMULA_TOL
        )

        print(
            "posterior-mean algebra: "
            + (
                "PASS"
                if posterior_formula_pass
                else "FAIL"
            )
        )

        # ====================================================
        # Step 6 — oracle reverse test
        # ====================================================

        print("\n" + "=" * 80)
        print("STEP 6 — ORACLE REVERSE-PROCESS TEST")
        print("=" * 80)

        (
            oracle_rows,
            oracle_final_rmse,
            oracle_final_max_abs_error,
        ) = oracle_reverse_trajectory(
            x0=x0,
            reverse_schedule=reverse_schedule,
            device=device,
        )

        save_csv(
            oracle_rows,
            ORACLE_TRAJECTORY_CSV,
        )

        oracle_pass = (
            oracle_final_rmse
            <= ORACLE_FINAL_RMSE_TOL
        )

        print(
            f"oracle final RMSE: "
            f"{oracle_final_rmse:.8e}"
        )
        print(
            f"oracle final max |error|: "
            f"{oracle_final_max_abs_error:.8e}"
        )
        print(
            "oracle reverse: "
            + (
                "PASS"
                if oracle_pass
                else "FAIL"
            )
        )
        print(
            f"oracle trajectory CSV: "
            f"{ORACLE_TRAJECTORY_CSV}"
        )

        # ====================================================
        # Step 7 — model free reverse trajectories
        # ====================================================

        print("\n" + "=" * 80)
        print("STEP 7 — MODEL FREE-REVERSE SCALE TRAJECTORY")
        print("=" * 80)

        model_rows_eo = model_free_reverse_trajectory(
            model=model,
            eye_state_value=0,
            reverse_schedule=reverse_schedule,
            device=device,
            seed=SEED + 300,
        )

        # Same stochastic seed for paired EO/EC trajectory audit.
        model_rows_ec = model_free_reverse_trajectory(
            model=model,
            eye_state_value=1,
            reverse_schedule=reverse_schedule,
            device=device,
            seed=SEED + 300,
        )

        model_rows = (
            model_rows_eo
            + model_rows_ec
        )

        save_csv(
            model_rows,
            MODEL_TRAJECTORY_CSV,
        )

        final_eo_std = float(
            model_rows_eo[-1][
                "x_std"
            ]
        )

        final_ec_std = float(
            model_rows_ec[-1][
                "x_std"
            ]
        )

        print(
            f"free reverse final EO std: "
            f"{final_eo_std:.6f}"
        )
        print(
            f"free reverse final EC std: "
            f"{final_ec_std:.6f}"
        )
        print(
            f"real diagnostic x0 std: "
            f"{real_x0_std:.6f}"
        )
        print(
            f"model trajectory CSV: "
            f"{MODEL_TRAJECTORY_CSV}"
        )

        # ====================================================
        # Step 8 — save figures
        # ====================================================

        print("\n" + "=" * 80)
        print("STEP 8 — SAVE DIAGNOSTIC FIGURES")
        print("=" * 80)

        save_timestep_figures(
            timestep_rows
        )

        save_trajectory_figure(
            oracle_rows=oracle_rows,
            model_rows=model_rows,
            real_target_std=real_x0_std,
        )

        print(
            f"epsilon MSE figure: "
            f"{EPSILON_FIGURE}"
        )
        print(
            f"x0 RMSE figure: "
            f"{X0_RMSE_FIGURE}"
        )
        print(
            f"x0 std figure: "
            f"{X0_STD_FIGURE}"
        )
        print(
            f"amplification figure: "
            f"{AMPLIFICATION_FIGURE}"
        )
        print(
            f"trajectory figure: "
            f"{TRAJECTORY_FIGURE}"
        )

        # ====================================================
        # Step 9 — compact interpretation fields
        # ====================================================

        by_timestep = {
            int(row["timestep"]): row
            for row in timestep_rows
        }

        t999 = by_timestep[999]
        t990 = by_timestep[990]
        t900 = by_timestep[900]
        t500 = by_timestep[500]
        t100 = by_timestep[100]
        t0 = by_timestep[0]

        high_t_x0_scale_ratio = float(
            t999["x0_pred_std"]
            / real_x0_std
        )

        free_eo_scale_ratio = float(
            final_eo_std
            / real_x0_std
        )

        free_ec_scale_ratio = float(
            final_ec_std
            / real_x0_std
        )

        sampler_equations_pass = bool(
            oracle_pass
            and posterior_formula_pass
        )

        # This is a diagnostic flag, not a scientific B0 acceptance rule.
        free_sampling_scale_flag = bool(
            free_eo_scale_ratio > 5.0
            or free_ec_scale_ratio > 5.0
        )

        if sampler_equations_pass and free_sampling_scale_flag:
            diagnostic_direction = (
                "sampler_equations_consistent__"
                "learned_or_accumulated_reverse_dynamics_suspect"
            )
        elif not sampler_equations_pass:
            diagnostic_direction = (
                "sampler_equation_or_reverse_implementation_suspect"
            )
        else:
            diagnostic_direction = (
                "no_major_scale_failure_detected_by_this_diagnostic"
            )

        # ====================================================
        # Step 10 — save summary
        # ====================================================

        print("\n" + "=" * 80)
        print("STEP 10 — SAVE DIAGNOSTIC SUMMARY")
        print("=" * 80)

        summary = {
            "status": (
                "B0_REVERSE_DIAGNOSTIC_COMPLETE"
            ),
            "purpose": (
                "separate reverse-sampler algebra failure "
                "from learned high-timestep / accumulated "
                "reverse-dynamics failure"
            ),
            "checkpoint": {
                "latest_complete_epoch": (
                    latest_epoch
                ),
                "best_ema_epoch": (
                    best_epoch
                ),
                "best_ema_validation_mse": (
                    best_validation_mse
                ),
                "latest_sha256": (
                    latest_hash
                ),
                "best_sha256": (
                    best_hash
                ),
                "best_checkpoint": str(
                    BEST_CHECKPOINT
                ),
            },
            "real_train_diagnostic_batch": {
                "samples": int(
                    x0.shape[0]
                ),
                "eo": int(
                    counts[0].item()
                ),
                "ec": int(
                    counts[1].item()
                ),
                "shape": [
                    int(value)
                    for value in x0.shape
                ],
                "normalized_mean": (
                    real_x0_mean
                ),
                "normalized_std": (
                    real_x0_std
                ),
            },
            "sampler_algebra": {
                "posterior_formula_max_abs_diff": (
                    posterior_formula_max_abs_diff
                ),
                "posterior_formula_tolerance": (
                    POSTERIOR_FORMULA_TOL
                ),
                "posterior_formula_pass": (
                    posterior_formula_pass
                ),
                "oracle_final_rmse": (
                    oracle_final_rmse
                ),
                "oracle_final_max_abs_error": (
                    oracle_final_max_abs_error
                ),
                "oracle_final_rmse_tolerance": (
                    ORACLE_FINAL_RMSE_TOL
                ),
                "oracle_reverse_pass": (
                    oracle_pass
                ),
                "sampler_equations_pass": (
                    sampler_equations_pass
                ),
            },
            "selected_timestep_diagnostics": {
                "t0": t0,
                "t100": t100,
                "t500": t500,
                "t900": t900,
                "t990": t990,
                "t999": t999,
            },
            "high_timestep": {
                "t999_epsilon_mse": float(
                    t999["epsilon_mse_all"]
                ),
                "t999_epsilon_rmse": float(
                    t999["epsilon_rmse_all"]
                ),
                "t999_error_amplification": float(
                    t999[
                        "epsilon_to_x0_error_amplification"
                    ]
                ),
                "t999_x0_rmse": float(
                    t999["x0_rmse_all"]
                ),
                "t999_x0_pred_std": float(
                    t999["x0_pred_std"]
                ),
                "t999_x0_pred_std_over_real": (
                    high_t_x0_scale_ratio
                ),
            },
            "free_reverse": {
                "samples_per_state": (
                    FREE_SAMPLING_PER_STATE
                ),
                "paired_seed": True,
                "final_eo_normalized_std": (
                    final_eo_std
                ),
                "final_ec_normalized_std": (
                    final_ec_std
                ),
                "eo_std_over_real": (
                    free_eo_scale_ratio
                ),
                "ec_std_over_real": (
                    free_ec_scale_ratio
                ),
                "major_scale_failure_flag": (
                    free_sampling_scale_flag
                ),
            },
            "diagnostic_direction": (
                diagnostic_direction
            ),
            "outputs": {
                "timestep_csv": str(
                    TIMESTEP_CSV
                ),
                "oracle_trajectory_csv": str(
                    ORACLE_TRAJECTORY_CSV
                ),
                "model_trajectory_csv": str(
                    MODEL_TRAJECTORY_CSV
                ),
                "epsilon_mse_figure": str(
                    EPSILON_FIGURE
                ),
                "x0_rmse_figure": str(
                    X0_RMSE_FIGURE
                ),
                "x0_std_figure": str(
                    X0_STD_FIGURE
                ),
                "amplification_figure": str(
                    AMPLIFICATION_FIGURE
                ),
                "trajectory_figure": str(
                    TRAJECTORY_FIGURE
                ),
            },
            "real_data_scope": (
                "TRAIN_ONLY"
            ),
            "validation_eeg_accessed": False,
            "final_test_accessed": False,
            "training_updates_executed": False,
            "b0_declared_final": False,
        }

        atomic_json_save(
            summary,
            SUMMARY_FILE,
        )

        print(
            f"summary JSON: {SUMMARY_FILE}"
        )

        # ====================================================
        # Final
        # ====================================================

        print("\n" + "=" * 80)
        print("B0 REVERSE-PROCESS DIAGNOSTIC COMPLETE")
        print("=" * 80)

        print(
            "posterior algebra: "
            + (
                "PASS"
                if posterior_formula_pass
                else "FAIL"
            )
        )

        print(
            "oracle reverse: "
            + (
                "PASS"
                if oracle_pass
                else "FAIL"
            )
        )

        print(
            f"oracle final RMSE: "
            f"{oracle_final_rmse:.8e}"
        )

        print(
            f"t=999 epsilon MSE: "
            f"{t999['epsilon_mse_all']:.6f}"
        )

        print(
            f"t=999 amplification: "
            f"{t999['epsilon_to_x0_error_amplification']:.2f}x"
        )

        print(
            f"t=999 predicted x0 std: "
            f"{t999['x0_pred_std']:.3f}"
        )

        print(
            f"real Train x0 std: "
            f"{real_x0_std:.3f}"
        )

        print(
            f"free reverse EO final std: "
            f"{final_eo_std:.3f}"
        )

        print(
            f"free reverse EC final std: "
            f"{final_ec_std:.3f}"
        )

        print(
            f"diagnostic direction: "
            f"{diagnostic_direction}"
        )

        print(
            "Validation EEG accessed: NO"
        )
        print(
            "Final Test accessed: NO"
        )
        print(
            "Training updates executed: NO"
        )

        print(
            "\nNEXT: INTERPRET THE DIAGNOSTIC "
            "BEFORE CHANGING B0 OR STARTING B1"
        )

    finally:
        dataset.close()


if __name__ == "__main__":
    main()
