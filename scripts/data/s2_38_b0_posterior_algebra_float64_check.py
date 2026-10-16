from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

METRIC_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "b0"
)

LOG_DIR = (
    REPO_ROOT
    / "outputs"
    / "logs"
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


SUMMARY_FILE = (
    METRIC_DIR
    / "b0_posterior_algebra_float64_check_v1.json"
)

CSV_FILE = (
    LOG_DIR
    / "b0_posterior_algebra_float64_check_v1.csv"
)

FIGURE_FILE = (
    FIGURE_DIR
    / "b0_posterior_algebra_precision_v1.png"
)


# ============================================================
# Frozen diffusion configuration
# ============================================================

SEED = 20260908

TIMESTEPS = 1000
COSINE_S = 0.008

BATCH_SIZE = 4
N_CHANNELS = 19
N_TIMES = 512


PRINT_TIMESTEPS = {
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
}


# ============================================================
# Helpers
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_json_save(
    data: dict,
    path: Path,
) -> None:

    temporary = Path(
        str(path)
        + ".tmp"
    )

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

    os.replace(
        temporary,
        path,
    )


# ============================================================
# Posterior coefficients
# ============================================================

def posterior_coefficients(
    *,
    beta_t: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_prev: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Closed-form posterior:

        q(x_{t-1} | x_t, x_0)

    mean =
        coef_x0 * x0
        +
        coef_xt * xt
    """

    denominator = (
        1.0
        - alpha_bar_t
    )

    coef_x0 = (
        beta_t
        * torch.sqrt(
            alpha_bar_prev
        )
        / denominator
    )

    coef_xt = (
        torch.sqrt(
            alpha_t
        )
        * (
            1.0
            - alpha_bar_prev
        )
        / denominator
    )

    return (
        coef_x0,
        coef_xt,
    )


# ============================================================
# Closed-form posterior mean
# ============================================================

def posterior_mean_closed_form(
    *,
    x0: torch.Tensor,
    xt: torch.Tensor,
    beta_t: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_prev: torch.Tensor,
) -> torch.Tensor:

    (
        coef_x0,
        coef_xt,
    ) = posterior_coefficients(
        beta_t=beta_t,
        alpha_t=alpha_t,
        alpha_bar_t=alpha_bar_t,
        alpha_bar_prev=alpha_bar_prev,
    )

    return (
        coef_x0
        * x0
        +
        coef_xt
        * xt
    )


# ============================================================
# Epsilon-form posterior mean
# ============================================================

def posterior_mean_from_epsilon(
    *,
    xt: torch.Tensor,
    epsilon: torch.Tensor,
    beta_t: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """
    Equivalent DDPM epsilon parameterization:

        mu_theta =
            1 / sqrt(alpha_t)
            *
            [
                x_t
                -
                beta_t
                / sqrt(1-alpha_bar_t)
                * epsilon
            ]
    """

    return (
        xt
        - (
            beta_t
            / torch.sqrt(
                1.0
                - alpha_bar_t
            )
        )
        * epsilon
    ) / torch.sqrt(
        alpha_t
    )


# ============================================================
# x_t construction
# ============================================================

def construct_xt(
    *,
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:

    return (
        torch.sqrt(
            alpha_bar_t
        )
        * x0
        +
        torch.sqrt(
            1.0
            - alpha_bar_t
        )
        * epsilon
    )


# ============================================================
# Error metrics
# ============================================================

def tensor_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:

    difference = (
        candidate
        - reference
    )

    max_abs = float(
        torch.max(
            torch.abs(
                difference
            )
        ).item()
    )

    rmse = float(
        torch.sqrt(
            torch.mean(
                difference
                * difference
            )
        ).item()
    )

    reference_rms = float(
        torch.sqrt(
            torch.mean(
                reference
                * reference
            )
        ).item()
    )

    relative_rmse = (
        rmse
        / (
            reference_rms
            + 1e-30
        )
    )

    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_rmse": (
            relative_rmse
        ),
        "reference_rms": (
            reference_rms
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "SCENARIO 2 — "
        "B0 POSTERIOR ALGEBRA FLOAT64 CONFIRMATION"
    )
    print("=" * 80)

    print(
        "\nPurpose:"
    )

    print(
        "  Re-check DDPM posterior mean "
        "equivalence in float64."
    )

    print(
        "  Quantify float32 numerical deviation."
    )

    print(
        "\nTraining updates executed: NO"
    )

    print(
        "Real Train EEG accessed: NO"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test EEG accessed: NO"
    )

    for directory in (
        METRIC_DIR,
        LOG_DIR,
        FIGURE_DIR,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Deterministic synthetic test tensors
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — BUILD CONTROLLED TEST TENSORS")
    print("=" * 80)

    set_seed(
        SEED
    )

    x0_64 = torch.randn(
        (
            BATCH_SIZE,
            N_CHANNELS,
            N_TIMES,
        ),
        dtype=torch.float64,
    )

    epsilon_64 = torch.randn(
        (
            BATCH_SIZE,
            N_CHANNELS,
            N_TIMES,
        ),
        dtype=torch.float64,
    )

    print(
        f"x0 shape: "
        f"{tuple(x0_64.shape)}"
    )

    print(
        f"x0 std: "
        f"{x0_64.std(unbiased=False).item():.6f}"
    )

    print(
        f"epsilon std: "
        f"{epsilon_64.std(unbiased=False).item():.6f}"
    )

    print(
        "synthetic algebra tensors: PASS"
    )

    # ========================================================
    # Step 2 — Schedule
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — BUILD COSINE DDPM SCHEDULE")
    print("=" * 80)

    schedule_64 = (
        build_cosine_schedule(
            timesteps=TIMESTEPS,
            cosine_s=COSINE_S,
        )
    )

    # build_cosine_schedule already returns float64,
    # but enforce explicitly.
    schedule_64 = {
        name: tensor.to(
            dtype=torch.float64
        )

        for (
            name,
            tensor,
        ) in schedule_64.items()
    }

    schedule_32 = {
        name: tensor.to(
            dtype=torch.float32
        )

        for (
            name,
            tensor,
        ) in schedule_64.items()
    }

    alpha_bar_64 = (
        schedule_64[
            "alpha_cumprod"
        ]
    )

    alpha_bar_prev_64 = torch.cat(
        [
            torch.ones(
                1,
                dtype=torch.float64,
            ),

            alpha_bar_64[:-1],
        ],
        dim=0,
    )

    alpha_bar_32 = (
        schedule_32[
            "alpha_cumprod"
        ]
    )

    alpha_bar_prev_32 = torch.cat(
        [
            torch.ones(
                1,
                dtype=torch.float32,
            ),

            alpha_bar_32[:-1],
        ],
        dim=0,
    )

    print(
        f"beta[0]: "
        f"{schedule_64['betas'][0].item():.12f}"
    )

    print(
        f"beta[-1]: "
        f"{schedule_64['betas'][-1].item():.12f}"
    )

    print(
        f"alpha_bar[-1]: "
        f"{alpha_bar_64[-1].item():.15e}"
    )

    print(
        "schedule: PASS"
    )

    # ========================================================
    # Step 3 — Test every timestep
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — CHECK POSTERIOR EQUIVALENCE FOR ALL TIMESTEPS")
    print("=" * 80)

    rows = []

    x0_32 = (
        x0_64.float()
    )

    epsilon_32 = (
        epsilon_64.float()
    )

    max_f64_error = 0.0
    max_f32_error = 0.0

    max_f64_error_t = None
    max_f32_error_t = None

    for t in range(
        TIMESTEPS
    ):

        # ----------------------------------------------------
        # FLOAT64
        # ----------------------------------------------------

        beta64 = (
            schedule_64[
                "betas"
            ][t]
        )

        alpha64 = (
            schedule_64[
                "alphas"
            ][t]
        )

        abar64 = (
            alpha_bar_64[
                t
            ]
        )

        abar_prev64 = (
            alpha_bar_prev_64[
                t
            ]
        )

        xt64 = construct_xt(
            x0=x0_64,
            epsilon=epsilon_64,
            alpha_bar_t=abar64,
        )

        posterior_closed64 = (
            posterior_mean_closed_form(
                x0=x0_64,
                xt=xt64,
                beta_t=beta64,
                alpha_t=alpha64,
                alpha_bar_t=abar64,
                alpha_bar_prev=(
                    abar_prev64
                ),
            )
        )

        posterior_epsilon64 = (
            posterior_mean_from_epsilon(
                xt=xt64,
                epsilon=epsilon_64,
                beta_t=beta64,
                alpha_t=alpha64,
                alpha_bar_t=abar64,
            )
        )

        error64 = (
            tensor_error_metrics(
                posterior_closed64,
                posterior_epsilon64,
            )
        )

        # ----------------------------------------------------
        # FLOAT32
        # ----------------------------------------------------

        beta32 = (
            schedule_32[
                "betas"
            ][t]
        )

        alpha32 = (
            schedule_32[
                "alphas"
            ][t]
        )

        abar32 = (
            alpha_bar_32[
                t
            ]
        )

        abar_prev32 = (
            alpha_bar_prev_32[
                t
            ]
        )

        xt32 = construct_xt(
            x0=x0_32,
            epsilon=epsilon_32,
            alpha_bar_t=abar32,
        )

        posterior_closed32 = (
            posterior_mean_closed_form(
                x0=x0_32,
                xt=xt32,
                beta_t=beta32,
                alpha_t=alpha32,
                alpha_bar_t=abar32,
                alpha_bar_prev=(
                    abar_prev32
                ),
            )
        )

        posterior_epsilon32 = (
            posterior_mean_from_epsilon(
                xt=xt32,
                epsilon=epsilon_32,
                beta_t=beta32,
                alpha_t=alpha32,
                alpha_bar_t=abar32,
            )
        )

        error32 = (
            tensor_error_metrics(
                posterior_closed32,
                posterior_epsilon32,
            )
        )

        # ----------------------------------------------------
        # Float32 versus Float64 epsilon-form result
        # ----------------------------------------------------

        f32_vs_f64 = (
            tensor_error_metrics(
                posterior_epsilon64,
                posterior_epsilon32.double(),
            )
        )

        # ----------------------------------------------------
        # Epsilon -> x0 amplification
        # ----------------------------------------------------

        amplification = float(
            (
                torch.sqrt(
                    1.0
                    - abar64
                )
                / torch.sqrt(
                    abar64
                )
            ).item()
        )

        row = {
            "timestep": int(
                t
            ),

            "beta": float(
                beta64.item()
            ),

            "alpha": float(
                alpha64.item()
            ),

            "alpha_bar": float(
                abar64.item()
            ),

            "epsilon_to_x0_amplification": (
                amplification
            ),

            "float64_max_abs": (
                error64[
                    "max_abs"
                ]
            ),

            "float64_rmse": (
                error64[
                    "rmse"
                ]
            ),

            "float64_relative_rmse": (
                error64[
                    "relative_rmse"
                ]
            ),

            "float32_max_abs": (
                error32[
                    "max_abs"
                ]
            ),

            "float32_rmse": (
                error32[
                    "rmse"
                ]
            ),

            "float32_relative_rmse": (
                error32[
                    "relative_rmse"
                ]
            ),

            "float32_vs_float64_rmse": (
                f32_vs_f64[
                    "rmse"
                ]
            ),
        }

        rows.append(
            row
        )

        if (
            error64[
                "max_abs"
            ]
            > max_f64_error
        ):

            max_f64_error = (
                error64[
                    "max_abs"
                ]
            )

            max_f64_error_t = (
                t
            )

        if (
            error32[
                "max_abs"
            ]
            > max_f32_error
        ):

            max_f32_error = (
                error32[
                    "max_abs"
                ]
            )

            max_f32_error_t = (
                t
            )

        if t in PRINT_TIMESTEPS:

            print(
                f"t={t:>3}"
                f" | amp="
                f"{amplification:>12.2f}"
                f" | f64 max="
                f"{error64['max_abs']:.3e}"
                f" | f64 RMSE="
                f"{error64['rmse']:.3e}"
                f" | f32 max="
                f"{error32['max_abs']:.3e}"
                f" | f32 RMSE="
                f"{error32['rmse']:.3e}"
                f" | f32-vs-f64="
                f"{f32_vs_f64['rmse']:.3e}"
            )

    # ========================================================
    # Step 4 — Save CSV
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — SAVE FULL PRECISION PROFILE")
    print("=" * 80)

    with CSV_FILE.open(
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

    print(
        f"CSV: "
        f"{CSV_FILE}"
    )

    # ========================================================
    # Step 5 — Decide algebra validity
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — ALGEBRA VERDICT")
    print("=" * 80)

    # Float64 should essentially agree to machine precision.
    FLOAT64_MAX_ABS_THRESHOLD = (
        1e-9
    )

    float64_pass = (
        max_f64_error
        <= FLOAT64_MAX_ABS_THRESHOLD
    )

    print(
        f"worst float64 max |difference|: "
        f"{max_f64_error:.6e}"
    )

    print(
        f"worst float64 timestep: "
        f"{max_f64_error_t}"
    )

    print(
        f"float64 threshold: "
        f"{FLOAT64_MAX_ABS_THRESHOLD:.1e}"
    )

    print(
        f"float64 algebra: "
        f"{'PASS' if float64_pass else 'FAIL'}"
    )

    print(
        f"\nworst float32 max |difference|: "
        f"{max_f32_error:.6e}"
    )

    print(
        f"worst float32 timestep: "
        f"{max_f32_error_t}"
    )

    if float64_pass:

        diagnostic_direction = (
            "POSTERIOR_ALGEBRA_CONFIRMED_"
            "PRIOR_FLOAT32_FAIL_WAS_NUMERICAL"
        )

        print(
            "\nINTERPRETATION:"
        )

        print(
            "  Closed-form posterior mean "
            "and epsilon-form posterior mean "
            "are algebraically equivalent."
        )

        print(
            "  The prior float32 discrepancy "
            "was numerical precision/cancellation, "
            "not evidence of a wrong DDPM formula."
        )

        print(
            "  Main remaining issue is "
            "high-timestep epsilon-to-x0 "
            "error amplification."
        )

    else:

        diagnostic_direction = (
            "POSTERIOR_FORMULA_MISMATCH_"
            "STILL_SUSPECT"
        )

        print(
            "\nINTERPRETATION:"
        )

        print(
            "  Float64 equivalence failed."
        )

        print(
            "  Do NOT change prediction "
            "parameterization yet."
        )

        print(
            "  Investigate schedule/posterior "
            "implementation first."
        )

    # ========================================================
    # Step 6 — Plot numerical precision
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — SAVE PRECISION FIGURE")
    print("=" * 80)

    timestep_axis = np.asarray(
        [
            row[
                "timestep"
            ]
            for row in rows
        ],
        dtype=np.int64,
    )

    f64_max = np.asarray(
        [
            row[
                "float64_max_abs"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    f32_max = np.asarray(
        [
            row[
                "float32_max_abs"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    amplification = np.asarray(
        [
            row[
                "epsilon_to_x0_amplification"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    # Avoid zero values on log axis.
    tiny = np.finfo(
        np.float64
    ).tiny

    f64_plot = np.maximum(
        f64_max,
        tiny,
    )

    f32_plot = np.maximum(
        f32_max,
        tiny,
    )

    figure = plt.figure(
        figsize=(
            11,
            7,
        )
    )

    axis = figure.add_subplot(
        111
    )

    axis.semilogy(
        timestep_axis,
        f64_plot,
        label=(
            "Float64 posterior "
            "formula difference"
        ),
    )

    axis.semilogy(
        timestep_axis,
        f32_plot,
        label=(
            "Float32 posterior "
            "formula difference"
        ),
    )

    axis.set_xlabel(
        "Diffusion timestep"
    )

    axis.set_ylabel(
        "Maximum absolute difference"
    )

    axis.set_title(
        "B0 DDPM Posterior Algebra — "
        "Float32 vs Float64 Numerical Precision"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURE_FILE,
        dpi=160,
    )

    plt.close(
        figure
    )

    print(
        f"figure: "
        f"{FIGURE_FILE}"
    )

    # ========================================================
    # Step 7 — Save summary
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — SAVE SUMMARY")
    print("=" * 80)

    row_999 = rows[
        999
    ]

    row_998 = rows[
        998
    ]

    row_995 = rows[
        995
    ]

    summary = {
        "status": (
            "B0_POSTERIOR_ALGEBRA_"
            "FLOAT64_CHECK_COMPLETE"
        ),

        "seed": (
            SEED
        ),

        "timesteps": (
            TIMESTEPS
        ),

        "cosine_s": (
            COSINE_S
        ),

        "test_tensor_shape": [
            BATCH_SIZE,
            N_CHANNELS,
            N_TIMES,
        ],

        "float64": {
            "pass": bool(
                float64_pass
            ),

            "threshold_max_abs": (
                FLOAT64_MAX_ABS_THRESHOLD
            ),

            "worst_max_abs": (
                max_f64_error
            ),

            "worst_timestep": (
                max_f64_error_t
            ),
        },

        "float32": {
            "worst_max_abs": (
                max_f32_error
            ),

            "worst_timestep": (
                max_f32_error_t
            ),
        },

        "terminal_region": {
            "t995_amplification": (
                row_995[
                    "epsilon_to_x0_amplification"
                ]
            ),

            "t998_amplification": (
                row_998[
                    "epsilon_to_x0_amplification"
                ]
            ),

            "t999_amplification": (
                row_999[
                    "epsilon_to_x0_amplification"
                ]
            ),

            "t999_float64_max_abs": (
                row_999[
                    "float64_max_abs"
                ]
            ),

            "t999_float32_max_abs": (
                row_999[
                    "float32_max_abs"
                ]
            ),
        },

        "diagnostic_direction": (
            diagnostic_direction
        ),

        "outputs": {
            "csv": str(
                CSV_FILE
            ),

            "figure": str(
                FIGURE_FILE
            ),
        },

        "training_updates_executed": False,

        "real_train_eeg_accessed": False,

        "validation_eeg_accessed": False,

        "final_test_accessed": False,
    }

    atomic_json_save(
        summary,
        SUMMARY_FILE,
    )

    print(
        f"summary JSON: "
        f"{SUMMARY_FILE}"
    )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "B0 POSTERIOR ALGEBRA "
        "FLOAT64 CHECK COMPLETE"
    )
    print("=" * 80)

    print(
        f"Float64 algebra: "
        f"{'PASS' if float64_pass else 'FAIL'}"
    )

    print(
        f"worst float64 max difference: "
        f"{max_f64_error:.3e}"
    )

    print(
        f"worst float32 max difference: "
        f"{max_f32_error:.3e}"
    )

    print(
        f"t=999 amplification: "
        f"{row_999['epsilon_to_x0_amplification']:.2f}x"
    )

    print(
        f"diagnostic direction: "
        f"{diagnostic_direction}"
    )

    print(
        "Training updates executed: NO"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nNEXT:"
    )

    if float64_pass:

        print(
            "B0 EPSILON -> "
            "V-PREDICTION CONVERSION SANITY"
        )

    else:

        print(
            "STOP AND INVESTIGATE "
            "POSTERIOR IMPLEMENTATION"
        )


if __name__ == "__main__":
    main()