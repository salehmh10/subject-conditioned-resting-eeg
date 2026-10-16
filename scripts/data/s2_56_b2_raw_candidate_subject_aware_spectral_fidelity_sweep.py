from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    build_reverse_schedule,
    inverse_normalize,
    welch_psd,
)
from s2_47_b1_architecture_warmstart_sanity import (
    B1SubjectStateUNet1D,
    state_dict_sha256,
)
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import (
    compute_candidate_metrics,
    sample_b1_candidate_state,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen B1 baseline
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

# Exact B1 subject-aware audit reference and baseline synthetic set.
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

B1_SWEEP_SUMMARY = (
    B1_SWEEP_ROOT
    / "metrics"
    / "b1_raw_candidate_subject_aware_summary_v2.json"
)

# B2 full run
B2_FULL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "full_finetune_v1"
)

B2_FULL_RESULT = (
    B2_FULL_ROOT
    / "metrics"
    / "b2_full_finetune_result_v1.json"
)

B2_CHECKPOINT_DIR = (
    B2_FULL_ROOT
    / "checkpoints"
)

# B2 selected lambda / protocol
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


# ============================================================
# Output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "raw_candidate_subject_aware_spectral_sweep_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
GENERATED_DIR = RUN_ROOT / "generated"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_FILE = (
    METRIC_DIR
    / "b2_raw_candidate_subject_aware_spectral_protocol_v1.json"
)

METRICS_CSV = (
    METRIC_DIR
    / "b2_raw_candidate_subject_aware_spectral_metrics_v1.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "b2_raw_candidate_subject_aware_spectral_summary_v1.json"
)

MATRICES_NPZ = (
    METRIC_DIR
    / "b2_raw_candidate_subject_aware_spectral_matrices_v1.npz"
)

SYNTHETIC_NPZ = (
    GENERATED_DIR
    / "b2_raw_candidate_subject_aware_synthetic_v1.npz"
)

PSD_FIGURE = (
    FIGURE_DIR
    / "b2_raw_candidate_full_psd_mae_v1.png"
)

SCALE_FIGURE = (
    FIGURE_DIR
    / "b2_raw_candidate_scale_preservation_v1.png"
)

SUBJECT_FIGURE = (
    FIGURE_DIR
    / "b2_raw_candidate_subject_preservation_v1.png"
)

GENERIC_FIGURE = (
    FIGURE_DIR
    / "b2_raw_candidate_generic_fidelity_v1.png"
)

ELIGIBILITY_FIGURE = (
    FIGURE_DIR
    / "b2_raw_candidate_eligibility_v1.png"
)

OVERVIEW_FIGURE = (
    FIGURE_DIR
    / "b2_raw_candidate_overview_v1.png"
)


# ============================================================
# Frozen audit design
# ============================================================

CANDIDATE_EPOCHS = [
    5,
    10,
    20,
]

EXPECTED_SELECTED_LAMBDA = 0.01

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

EXPECTED_REAL_SHAPE = (
    SELECTED_SUBJECT_COUNT,
    2,
    REAL_WINDOWS_PER_SUBJECT_STATE,
    19,
    512,
)

EXPECTED_SYNTH_SHAPE = (
    SELECTED_SUBJECT_COUNT,
    2,
    SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
    19,
    512,
)

TIMESTEPS = 1000
COSINE_S = 0.008

# Reuse the exact sampling seed convention from B1 selection audit V2.
SAMPLING_SEED_BASE = 20261001

# Primary B2 requirement:
# improve the full 1-40 Hz log-PSD relative to frozen B1.
MIN_FULL_PSD_IMPROVEMENT_FRACTION = 0.05

# Preservation guardrails relative to frozen B1 on the same
# subject cohort / Real reference / sampling seeds.
PARENT_SCALE_RELATIVE_LOWER = 0.90
PARENT_SCALE_RELATIVE_UPPER = 1.10

MAX_BANDPOWER_MAE_INCREASE_DB = 0.25
MAX_ALPHA_EFFECT_MAE_INCREASE_DB = 0.50

MIN_SPATIAL_CORRELATION_R = 0.98

MIN_SUBJECT_TOP1_FRACTION_OF_B1 = 0.90
MAX_SUBJECT_FINGERPRINT_R_DROP = 0.05
MIN_SUBJECT_MARGIN_FRACTION_OF_B1 = 0.85
MIN_FIXED_WRONG_WIN_FRACTION = 0.90

MAX_WPLI_MEAN_R_DROP = 0.15


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
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


def b2_candidate_path(
    epoch: int,
) -> Path:
    return (
        B2_CHECKPOINT_DIR
        / f"candidate_epoch_{epoch:03d}_raw.pt"
    )


def build_model_from_state(
    *,
    frozen_b1: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
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
        state_dict,
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    return model


def load_reference_packages() -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[str],
]:
    real_package = np.load(
        B1_REAL_REFERENCE_NPZ,
        allow_pickle=False,
    )

    synth_package = np.load(
        B1_SYNTHETIC_REFERENCE_NPZ,
        allow_pickle=False,
    )

    real_z = real_package[
        "real_z"
    ].astype(
        np.float32,
        copy=False,
    )

    selected_subject_indices = (
        real_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    selected_subject_ids = (
        real_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    synth_subject_indices = (
        synth_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    synth_subject_ids = (
        synth_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if (
        selected_subject_indices
        != synth_subject_indices
    ):
        raise RuntimeError(
            "B1 Real/synthetic subject-index cohort mismatch."
        )

    if (
        selected_subject_ids
        != synth_subject_ids
    ):
        raise RuntimeError(
            "B1 Real/synthetic subject-ID cohort mismatch."
        )

    b1_synth_z = synth_package[
        "epoch_010_z"
    ].astype(
        np.float32,
        copy=False,
    )

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(
            f"Unexpected Real reference shape: "
            f"{real_z.shape}"
        )

    if b1_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected frozen-B1 synthetic shape: "
            f"{b1_synth_z.shape}"
        )

    if not np.isfinite(
        real_z
    ).all():
        raise RuntimeError(
            "Real reference contains NaN/Inf."
        )

    if not np.isfinite(
        b1_synth_z
    ).all():
        raise RuntimeError(
            "Frozen-B1 synthetic reference contains NaN/Inf."
        )

    return (
        real_z,
        b1_synth_z,
        selected_subject_indices,
        selected_subject_ids,
    )


def full_log_psd_mae(
    *,
    real_z: np.ndarray,
    synth_z: np.ndarray,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    """
    Cohort-level full 1-40 Hz log-PSD fidelity.

    For each state:
      - inverse normalization to µV
      - Welch PSD
      - retain 1-40 Hz
      - convert to dB
      - median over windows
      - MAE over channels × frequencies
    """

    real_uv = inverse_normalize(
        real_z.reshape(
            -1,
            19,
            512,
        ),
        scaler,
    ).reshape(
        real_z.shape
    )

    synth_uv = inverse_normalize(
        synth_z.reshape(
            -1,
            19,
            512,
        ),
        scaler,
    ).reshape(
        synth_z.shape
    )

    state_results = {}

    all_state_mae = []

    for state_index, state_name in enumerate(
        [
            "EO",
            "EC",
        ]
    ):
        real_state = real_uv[
            :,
            state_index,
            ...,
        ].reshape(
            -1,
            19,
            512,
        )

        synth_state = synth_uv[
            :,
            state_index,
            ...,
        ].reshape(
            -1,
            19,
            512,
        )

        frequencies, real_psd = welch_psd(
            real_state
        )

        synth_frequencies, synth_psd = welch_psd(
            synth_state
        )

        if not np.array_equal(
            frequencies,
            synth_frequencies,
        ):
            raise RuntimeError(
                "Real/Synthetic Welch frequency grid mismatch."
            )

        mask = (
            (
                frequencies
                >= 1.0
            )
            & (
                frequencies
                <= 40.0
            )
        )

        tiny = np.finfo(
            np.float64
        ).tiny

        real_db = (
            10.0
            * np.log10(
                np.maximum(
                    real_psd[
                        ...,
                        mask
                    ],
                    tiny,
                )
            )
        )

        synth_db = (
            10.0
            * np.log10(
                np.maximum(
                    synth_psd[
                        ...,
                        mask
                    ],
                    tiny,
                )
            )
        )

        real_median = np.median(
            real_db,
            axis=0,
        )

        synth_median = np.median(
            synth_db,
            axis=0,
        )

        absolute_error = np.abs(
            synth_median
            - real_median
        )

        state_mae = float(
            np.mean(
                absolute_error
            )
        )

        all_state_mae.append(
            state_mae
        )

        state_results[
            state_name
        ] = {
            "mae_db": (
                state_mae
            ),
            "median_absolute_error_db": float(
                np.median(
                    absolute_error
                )
            ),
        }

    return {
        "EO": (
            state_results[
                "EO"
            ]
        ),
        "EC": (
            state_results[
                "EC"
            ]
        ),
        "mean_mae_db": float(
            np.mean(
                all_state_mae
            )
        ),
    }


def mean_wpli_r(
    metrics: dict[str, Any],
) -> float:
    return float(
        np.nanmean(
            [
                metrics[
                    "spatial"
                ][
                    "EO"
                ][
                    "alpha_dwpli2_r"
                ],
                metrics[
                    "spatial"
                ][
                    "EC"
                ][
                    "alpha_dwpli2_r"
                ],
            ]
        )
    )


def mean_corr_r(
    metrics: dict[str, Any],
) -> float:
    return float(
        np.nanmean(
            [
                metrics[
                    "spatial"
                ][
                    "EO"
                ][
                    "correlation_r"
                ],
                metrics[
                    "spatial"
                ][
                    "EC"
                ][
                    "correlation_r"
                ],
            ]
        )
    )


def evaluate_eligibility(
    *,
    candidate: dict[str, Any],
    b1_baseline: dict[str, Any],
) -> dict[str, Any]:
    b1_psd = float(
        b1_baseline[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    candidate_psd = float(
        candidate[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    psd_improvement_fraction = float(
        (
            b1_psd
            - candidate_psd
        )
        / b1_psd
    )

    candidate_scale_eo = float(
        candidate[
            "generic"
        ][
            "scale"
        ][
            "EO"
        ][
            "normalized_std_ratio"
        ]
    )

    candidate_scale_ec = float(
        candidate[
            "generic"
        ][
            "scale"
        ][
            "EC"
        ][
            "normalized_std_ratio"
        ]
    )

    b1_scale_eo = float(
        b1_baseline[
            "generic"
        ][
            "scale"
        ][
            "EO"
        ][
            "normalized_std_ratio"
        ]
    )

    b1_scale_ec = float(
        b1_baseline[
            "generic"
        ][
            "scale"
        ][
            "EC"
        ][
            "normalized_std_ratio"
        ]
    )

    eo_scale_parent_ratio = float(
        candidate_scale_eo
        / b1_scale_eo
    )

    ec_scale_parent_ratio = float(
        candidate_scale_ec
        / b1_scale_ec
    )

    b1_bandpower = float(
        b1_baseline[
            "generic"
        ][
            "mean_bandpower_mae_db"
        ]
    )

    candidate_bandpower = float(
        candidate[
            "generic"
        ][
            "mean_bandpower_mae_db"
        ]
    )

    b1_alpha = float(
        b1_baseline[
            "generic"
        ][
            "alpha_effect"
        ][
            "channel_mae_db"
        ]
    )

    candidate_alpha = float(
        candidate[
            "generic"
        ][
            "alpha_effect"
        ][
            "channel_mae_db"
        ]
    )

    candidate_corr = mean_corr_r(
        candidate[
            "generic"
        ]
    )

    b1_wpli = mean_wpli_r(
        b1_baseline[
            "generic"
        ]
    )

    candidate_wpli = mean_wpli_r(
        candidate[
            "generic"
        ]
    )

    b1_subject = b1_baseline[
        "generic"
    ][
        "subject_identity"
    ]

    candidate_subject = candidate[
        "generic"
    ][
        "subject_identity"
    ]

    b1_top1 = float(
        b1_subject[
            "subject_retrieval_top1_accuracy"
        ]
    )

    candidate_top1 = float(
        candidate_subject[
            "subject_retrieval_top1_accuracy"
        ]
    )

    b1_subject_r = float(
        b1_subject[
            "subject_fingerprint_pearson_r_median"
        ]
    )

    candidate_subject_r = float(
        candidate_subject[
            "subject_fingerprint_pearson_r_median"
        ]
    )

    b1_margin = float(
        b1_subject[
            "subject_correct_vs_median_wrong_margin_mean"
        ]
    )

    candidate_margin = float(
        candidate_subject[
            "subject_correct_vs_median_wrong_margin_mean"
        ]
    )

    candidate_fixed_wrong = float(
        candidate_subject[
            "subject_correct_better_than_fixed_wrong_fraction"
        ]
    )

    criteria = {
        "full_psd_improvement": bool(
            psd_improvement_fraction
            >= MIN_FULL_PSD_IMPROVEMENT_FRACTION
        ),
        "eo_scale_parent_relative": bool(
            PARENT_SCALE_RELATIVE_LOWER
            <= eo_scale_parent_ratio
            <= PARENT_SCALE_RELATIVE_UPPER
        ),
        "ec_scale_parent_relative": bool(
            PARENT_SCALE_RELATIVE_LOWER
            <= ec_scale_parent_ratio
            <= PARENT_SCALE_RELATIVE_UPPER
        ),
        "bandpower_preserved": bool(
            candidate_bandpower
            <= (
                b1_bandpower
                + MAX_BANDPOWER_MAE_INCREASE_DB
            )
        ),
        "alpha_effect_preserved": bool(
            candidate_alpha
            <= (
                b1_alpha
                + MAX_ALPHA_EFFECT_MAE_INCREASE_DB
            )
        ),
        "spatial_correlation_preserved": bool(
            candidate_corr
            >= MIN_SPATIAL_CORRELATION_R
        ),
        "subject_top1_preserved": bool(
            candidate_top1
            >= (
                MIN_SUBJECT_TOP1_FRACTION_OF_B1
                * b1_top1
            )
        ),
        "subject_fingerprint_r_preserved": bool(
            candidate_subject_r
            >= (
                b1_subject_r
                - MAX_SUBJECT_FINGERPRINT_R_DROP
            )
        ),
        "subject_margin_preserved": bool(
            candidate_margin
            >= (
                MIN_SUBJECT_MARGIN_FRACTION_OF_B1
                * b1_margin
            )
        ),
        "fixed_wrong_subject_win_fraction_preserved": bool(
            candidate_fixed_wrong
            >= MIN_FIXED_WRONG_WIN_FRACTION
        ),
        "wpli_not_catastrophically_degraded": bool(
            candidate_wpli
            >= max(
                0.0,
                b1_wpli
                - MAX_WPLI_MEAN_R_DROP,
            )
        ),
    }

    eligible = bool(
        all(
            criteria.values()
        )
    )

    return {
        "eligible": (
            eligible
        ),
        "criteria": (
            criteria
        ),
        "full_psd_improvement_fraction": (
            psd_improvement_fraction
        ),
        "EO_scale_ratio_to_frozen_B1": (
            eo_scale_parent_ratio
        ),
        "EC_scale_ratio_to_frozen_B1": (
            ec_scale_parent_ratio
        ),
        "mean_spatial_correlation_r": (
            candidate_corr
        ),
        "mean_alpha_dwpli2_r": (
            candidate_wpli
        ),
    }


# ============================================================
# Figures
# ============================================================

def save_figures(
    *,
    b1_baseline: dict[str, Any],
    candidates: dict[int, dict[str, Any]],
    eligibility: dict[int, dict[str, Any]],
) -> None:
    labels = [
        "B1",
        *[
            f"B2-E{epoch}"
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    # Full PSD MAE.
    psd_values = [
        b1_baseline[
            "full_log_psd"
        ][
            "mean_mae_db"
        ],
        *[
            candidates[
                epoch
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.bar(
        np.arange(
            len(labels)
        ),
        psd_values,
    )

    ax.set_xticks(
        np.arange(
            len(labels)
        ),
        labels,
    )

    ax.set_ylabel(
        "1-40 Hz log-PSD MAE (dB)"
    )

    ax.set_title(
        "B2 Candidates vs Frozen B1 — Full Spectral Fidelity"
    )

    fig.tight_layout()

    fig.savefig(
        PSD_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Scale relative to Real.
    eo = [
        b1_baseline[
            "generic"
        ][
            "scale"
        ][
            "EO"
        ][
            "normalized_std_ratio"
        ],
        *[
            candidates[
                epoch
            ][
                "generic"
            ][
                "scale"
            ][
                "EO"
            ][
                "normalized_std_ratio"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    ec = [
        b1_baseline[
            "generic"
        ][
            "scale"
        ][
            "EC"
        ][
            "normalized_std_ratio"
        ],
        *[
            candidates[
                epoch
            ][
                "generic"
            ][
                "scale"
            ][
                "EC"
            ][
                "normalized_std_ratio"
            ]
            for epoch in CANDIDATE_EPOCHS
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
            10,
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

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Synthetic / Real normalized std"
    )

    ax.set_title(
        "B2 Candidates vs Frozen B1 — Scale Fidelity"
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

    # Subject preservation.
    top1 = [
        b1_baseline[
            "generic"
        ][
            "subject_identity"
        ][
            "subject_retrieval_top1_accuracy"
        ],
        *[
            candidates[
                epoch
            ][
                "generic"
            ][
                "subject_identity"
            ][
                "subject_retrieval_top1_accuracy"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    subject_r = [
        b1_baseline[
            "generic"
        ][
            "subject_identity"
        ][
            "subject_fingerprint_pearson_r_median"
        ],
        *[
            candidates[
                epoch
            ][
                "generic"
            ][
                "subject_identity"
            ][
                "subject_fingerprint_pearson_r_median"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        x,
        top1,
        marker="o",
        label="Top-1 subject retrieval",
    )

    ax.plot(
        x,
        subject_r,
        marker="o",
        label="Median subject fingerprint r",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_title(
        "B2 Candidates vs Frozen B1 — Subject Fidelity"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SUBJECT_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Generic metrics.
    bandpower = [
        b1_baseline[
            "generic"
        ][
            "mean_bandpower_mae_db"
        ],
        *[
            candidates[
                epoch
            ][
                "generic"
            ][
                "mean_bandpower_mae_db"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    alpha = [
        b1_baseline[
            "generic"
        ][
            "alpha_effect"
        ][
            "channel_mae_db"
        ],
        *[
            candidates[
                epoch
            ][
                "generic"
            ][
                "alpha_effect"
            ][
                "channel_mae_db"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.plot(
        x,
        bandpower,
        marker="o",
        label="Mean bandpower MAE (dB)",
    )

    ax.plot(
        x,
        alpha,
        marker="o",
        label="Alpha-effect MAE (dB)",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_title(
        "B2 Candidates vs Frozen B1 — Generic Fidelity"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        GENERIC_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Eligibility heatmap.
    criterion_names = list(
        next(
            iter(
                eligibility.values()
            )
        )[
            "criteria"
        ].keys()
    )

    matrix = np.asarray(
        [
            [
                float(
                    eligibility[
                        epoch
                    ][
                        "criteria"
                    ][
                        criterion
                    ]
                )
                for epoch in CANDIDATE_EPOCHS
            ]
            for criterion in criterion_names
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            8,
        )
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        np.arange(
            len(
                CANDIDATE_EPOCHS
            )
        ),
        [
            f"Epoch {epoch}"
            for epoch in CANDIDATE_EPOCHS
        ],
    )

    ax.set_yticks(
        np.arange(
            len(
                criterion_names
            )
        ),
        criterion_names,
    )

    ax.set_title(
        "B2 Candidate Eligibility (1 = PASS)"
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

    # Overview relative improvements.
    rows = [
        "Full PSD improvement ↑",
        "Bandpower MAE ↓",
        "Alpha-effect MAE ↓",
        "Top-1 subject retrieval ↑",
        "Subject fingerprint r ↑",
        "Mean channel correlation r ↑",
        "Mean alpha dwPLI2 r ↑",
    ]

    values = []

    for epoch in CANDIDATE_EPOCHS:
        candidate = candidates[
            epoch
        ]

        values.append(
            [
                eligibility[
                    epoch
                ][
                    "full_psd_improvement_fraction"
                ],
                candidate[
                    "generic"
                ][
                    "mean_bandpower_mae_db"
                ],
                candidate[
                    "generic"
                ][
                    "alpha_effect"
                ][
                    "channel_mae_db"
                ],
                candidate[
                    "generic"
                ][
                    "subject_identity"
                ][
                    "subject_retrieval_top1_accuracy"
                ],
                candidate[
                    "generic"
                ][
                    "subject_identity"
                ][
                    "subject_fingerprint_pearson_r_median"
                ],
                mean_corr_r(
                    candidate[
                        "generic"
                    ]
                ),
                mean_wpli_r(
                    candidate[
                        "generic"
                    ]
                ),
            ]
        )

    matrix = np.asarray(
        values,
        dtype=np.float64,
    ).T

    lower_is_better = {
        1,
        2,
    }

    normalized = np.empty_like(
        matrix
    )

    for row_index in range(
        matrix.shape[0]
    ):
        row = matrix[
            row_index
        ]

        low = np.nanmin(
            row
        )

        high = np.nanmax(
            row
        )

        if abs(
            high
            - low
        ) < 1e-12:
            score = np.full_like(
                row,
                0.5,
            )
        else:
            score = (
                row
                - low
            ) / (
                high
                - low
            )

        if row_index in lower_is_better:
            score = (
                1.0
                - score
            )

        normalized[
            row_index
        ] = score

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    image = ax.imshow(
        normalized,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        np.arange(
            len(
                CANDIDATE_EPOCHS
            )
        ),
        [
            f"Epoch {epoch}"
            for epoch in CANDIDATE_EPOCHS
        ],
    )

    ax.set_yticks(
        np.arange(
            len(
                rows
            )
        ),
        rows,
    )

    ax.set_title(
        "B2 RAW Candidate Relative Overview"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Within-row relative score",
    )

    fig.tight_layout()

    fig.savefig(
        OVERVIEW_FIGURE,
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
        "SCENARIO 2 — B2 RAW EPOCH 5/10/20 SUBJECT-AWARE + SPECTRAL FIDELITY SWEEP"
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
        "  B2 full 20-epoch spectral fine-tuning: COMPLETE"
    )

    print(
        "  B2 RAW epoch 5/10/20 final-selection audit: NOW"
    )

    print(
        "  B2 final freeze: NOT YET"
    )

    print(
        "  Bulk synthetic generation: NOT STARTED"
    )

    print(
        "  Final Test: SEALED"
    )

    print(
        "\nAudit design:"
    )

    print(
        f"  B2 candidates: "
        f"{CANDIDATE_EPOCHS}"
    )

    print(
        "  comparator: exact frozen B1 RAW epoch 10"
    )

    print(
        "  Real reference: exact B1 subject-aware audit V2 reference"
    )

    print(
        f"  subjects: "
        f"{SELECTED_SUBJECT_COUNT}"
    )

    print(
        f"  Real windows/subject/state: "
        f"{REAL_WINDOWS_PER_SUBJECT_STATE}"
    )

    print(
        f"  Synthetic windows/subject/state/model: "
        f"{SYNTHETIC_WINDOWS_PER_SUBJECT_STATE}"
    )

    print(
        "  same subjects + sampling seeds as frozen B1 selection: YES"
    )

    print(
        "  Validation EEG accessed: NO"
    )

    print(
        "  Final Test accessed: NO"
    )

    print(
        "  training updates executed: NO"
    )

    for directory in (
        METRIC_DIR,
        GENERATED_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit frozen sources and B2 candidates
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B1 + B2 CANDIDATES"
    )

    print(
        "=" * 80
    )

    required_files = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_REAL_REFERENCE_NPZ,
        B1_SYNTHETIC_REFERENCE_NPZ,
        B1_SWEEP_SUMMARY,
        B2_FULL_RESULT,
        B2_SELECTED_LAMBDA_FILE,
    ]

    for epoch in CANDIDATE_EPOCHS:
        required_files.append(
            b2_candidate_path(
                epoch
            )
        )

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

    with B1_SWEEP_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        b1_sweep_summary = json.load(
            f
        )

    with B2_FULL_RESULT.open(
        "r",
        encoding="utf-8",
    ) as f:
        b2_result = json.load(
            f
        )

    with B2_SELECTED_LAMBDA_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        selected_lambda = json.load(
            f
        )

    if b1_selection[
        "status"
    ] != "B1_SELECTION_FROZEN":
        raise RuntimeError(
            "B1 is not formally frozen."
        )

    if int(
        b1_selection[
            "selected_epoch"
        ]
    ) != 10:
        raise RuntimeError(
            "Unexpected frozen B1 epoch."
        )

    if b2_result[
        "status"
    ] != "B2_FULL_FINETUNE_20_EPOCHS_COMPLETE":
        raise RuntimeError(
            "B2 full run is not complete."
        )

    if abs(
        float(
            b2_result[
                "lambda_spectral"
            ]
        )
        - EXPECTED_SELECTED_LAMBDA
    ) > 1e-12:
        raise RuntimeError(
            "Unexpected B2 full-run lambda."
        )

    if selected_lambda[
        "status"
    ] != "B2_SELECTED_LAMBDA_FROZEN_V2":
        raise RuntimeError(
            "B2 selected lambda is not frozen."
        )

    if b2_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "B2 full run reports Final Test access."
        )

    frozen_b1_file_hash_before = (
        sha256_file(
            FROZEN_B1_GENERATOR
        )
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
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
            "Frozen B1 model-state hash mismatch."
        )

    candidate_audit = {}

    candidate_hash_before = {}

    for epoch in CANDIDATE_EPOCHS:
        path = b2_candidate_path(
            epoch
        )

        candidate_hash_before[
            epoch
        ] = sha256_file(
            path
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint[
            "status"
        ] != "B2_FULL_FINETUNE_CHECKPOINT":
            raise RuntimeError(
                f"Unexpected B2 checkpoint status at epoch {epoch}."
            )

        if int(
            checkpoint[
                "epoch"
            ]
        ) != epoch:
            raise RuntimeError(
                f"B2 candidate epoch mismatch for {epoch}."
            )

        if abs(
            float(
                checkpoint[
                    "lambda_spectral"
                ]
            )
            - EXPECTED_SELECTED_LAMBDA
        ) > 1e-12:
            raise RuntimeError(
                f"B2 candidate {epoch} lambda mismatch."
            )

        if checkpoint[
            "selected_weight_kind"
        ] != "RAW":
            raise RuntimeError(
                f"B2 candidate {epoch} is not RAW."
            )

        if checkpoint[
            "EMA_enabled"
        ] is not False:
            raise RuntimeError(
                f"B2 candidate {epoch} unexpectedly uses EMA."
            )

        if checkpoint[
            "Final_Test_accessed"
        ] is not False:
            raise RuntimeError(
                f"B2 candidate {epoch} reports Test access."
            )

        candidate_audit[
            epoch
        ] = {
            "path": str(
                path
            ),
            "file_sha256": (
                candidate_hash_before[
                    epoch
                ]
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
            "train_spectral_loss": float(
                checkpoint[
                    "history"
                ][
                    -1
                ][
                    "train_spectral_loss_db_huber"
                ]
            ),
            "validation_v_mse": float(
                checkpoint[
                    "history"
                ][
                    -1
                ][
                    "validation_correct_subject_v_mse"
                ]
            ),
            "validation_spectral_loss": float(
                checkpoint[
                    "history"
                ][
                    -1
                ][
                    "validation_spectral_loss_db_huber"
                ]
            ),
        }

        print(
            f"epoch {epoch:>2} RAW "
            f"| Train-v="
            f"{candidate_audit[epoch]['train_v_mse']:.6f} "
            f"| Train-spec="
            f"{candidate_audit[epoch]['train_spectral_loss']:.4f} "
            f"| Val-v="
            f"{candidate_audit[epoch]['validation_v_mse']:.6f} "
            f"| Val-spec="
            f"{candidate_audit[epoch]['validation_spectral_loss']:.4f} "
            f"| SHA256="
            f"{candidate_audit[epoch]['file_sha256'][:16]}..."
        )

    print(
        "source/candidate audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze final-selection protocol before generation
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE B2 FINAL-SELECTION AUDIT PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "B2_RAW_CANDIDATE_SUBJECT_AWARE_SPECTRAL_PROTOCOL_FROZEN"
        ),
        "candidate_epochs": (
            CANDIDATE_EPOCHS
        ),
        "candidate_weight_kind": (
            "RAW"
        ),
        "lambda_spectral": (
            EXPECTED_SELECTED_LAMBDA
        ),
        "frozen_B1_comparator": str(
            FROZEN_B1_GENERATOR
        ),
        "real_reference": str(
            B1_REAL_REFERENCE_NPZ
        ),
        "frozen_B1_synthetic_reference": str(
            B1_SYNTHETIC_REFERENCE_NPZ
        ),
        "subject_count": (
            SELECTED_SUBJECT_COUNT
        ),
        "real_windows_per_subject_state": (
            REAL_WINDOWS_PER_SUBJECT_STATE
        ),
        "synthetic_windows_per_subject_state": (
            SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
        ),
        "sampling_seed_base": (
            SAMPLING_SEED_BASE
        ),
        "same_subjects_and_seeds_as_B1_final_selection": True,
        "primary_B2_requirement": {
            "metric": (
                "mean 1-40 Hz cohort-median log-PSD MAE"
            ),
            "minimum_improvement_vs_frozen_B1_fraction": (
                MIN_FULL_PSD_IMPROVEMENT_FRACTION
            ),
        },
        "preservation_guardrails": {
            "EO_scale_relative_to_B1": [
                PARENT_SCALE_RELATIVE_LOWER,
                PARENT_SCALE_RELATIVE_UPPER,
            ],
            "EC_scale_relative_to_B1": [
                PARENT_SCALE_RELATIVE_LOWER,
                PARENT_SCALE_RELATIVE_UPPER,
            ],
            "mean_bandpower_MAE_max_increase_dB": (
                MAX_BANDPOWER_MAE_INCREASE_DB
            ),
            "alpha_effect_MAE_max_increase_dB": (
                MAX_ALPHA_EFFECT_MAE_INCREASE_DB
            ),
            "mean_channel_correlation_r_minimum": (
                MIN_SPATIAL_CORRELATION_R
            ),
            "subject_top1_min_fraction_of_B1": (
                MIN_SUBJECT_TOP1_FRACTION_OF_B1
            ),
            "subject_fingerprint_r_max_drop": (
                MAX_SUBJECT_FINGERPRINT_R_DROP
            ),
            "subject_margin_min_fraction_of_B1": (
                MIN_SUBJECT_MARGIN_FRACTION_OF_B1
            ),
            "correct_closer_than_fixed_wrong_min_fraction": (
                MIN_FIXED_WRONG_WIN_FRACTION
            ),
            "mean_alpha_dwPLI2_r_max_drop": (
                MAX_WPLI_MEAN_R_DROP
            ),
        },
        "winner_rule": (
            "Among candidates satisfying ALL eligibility guards, "
            "select lowest full 1-40 Hz log-PSD MAE; "
            "tie -> lower mean bandpower MAE; "
            "tie -> lower alpha-effect MAE; "
            "tie -> earlier epoch."
        ),
        "failure_action": (
            "If no B2 candidate is eligible, do not freeze B2; "
            "interpret whether B2 failed to improve spectrum without "
            "sacrificing B1 fidelity."
        ),
        "matched_lambda0_control": (
            "not required for checkpoint selection, but still required "
            "before causal publication claim about spectral regularization"
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "training_updates_executed": False,
        "bulk_synthetic_generation_started": False,
    }

    if PROTOCOL_FILE.exists():
        with PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != protocol:
            raise RuntimeError(
                "Conflicting B2 final-selection protocol exists."
            )

        print(
            "identical B2 final-selection protocol already exists."
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
    # Step 3 — Load exact frozen-B1 comparison cohort
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — LOAD EXACT B1 SUBJECT-AWARE REAL + SYNTHETIC REFERENCES"
    )

    print(
        "=" * 80
    )

    (
        real_z,
        b1_synth_z,
        selected_subjects,
        selected_subject_ids,
    ) = load_reference_packages()

    # Recover the same scaler used in all prior audits.
    # The exact means/stds are embedded in the frozen B0/B1 workflow,
    # but importing the locked normalizer file is simpler and auditable.
    from s2_23_check_real_train_dataset import (
        NORMALIZATION_FILE,
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

    b1_generic, b1_matrices, _ = (
        compute_candidate_metrics(
            real_z=real_z,
            synth_z=b1_synth_z,
            scaler=scaler,
        )
    )

    b1_full_psd = full_log_psd_mae(
        real_z=real_z,
        synth_z=b1_synth_z,
        scaler=scaler,
    )

    b1_baseline = {
        "generic": (
            b1_generic
        ),
        "full_log_psd": (
            b1_full_psd
        ),
    }

    print(
        f"reference subjects: "
        f"{len(selected_subjects)}"
    )

    print(
        f"Real reference shape: "
        f"{real_z.shape}"
    )

    print(
        f"Frozen B1 synthetic shape: "
        f"{b1_synth_z.shape}"
    )

    print(
        "Frozen B1 baseline metrics:"
    )

    print(
        f"  full 1-40 Hz log-PSD MAE: "
        f"{b1_full_psd['mean_mae_db']:.3f} dB"
    )

    print(
        f"  mean bandpower MAE: "
        f"{b1_generic['mean_bandpower_mae_db']:.3f} dB"
    )

    print(
        f"  alpha-effect MAE: "
        f"{b1_generic['alpha_effect']['channel_mae_db']:.3f} dB"
    )

    print(
        "  scale EO/EC: "
        f"{b1_generic['scale']['EO']['normalized_std_ratio']:.3f}"
        " / "
        f"{b1_generic['scale']['EC']['normalized_std_ratio']:.3f}"
    )

    print(
        "  subject Top1: "
        f"{b1_generic['subject_identity']['subject_retrieval_top1_accuracy']:.3f}"
    )

    print(
        "  subject fingerprint r median: "
        f"{b1_generic['subject_identity']['subject_fingerprint_pearson_r_median']:.3f}"
    )

    # ========================================================
    # Step 4 — Generate B2 candidates
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — GENERATE + EVALUATE B2 RAW CANDIDATES"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required."
        )

    device = torch.device(
        "cuda"
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "sampler: 1000-step ancestral DDPM / v-prediction / FP32"
    )

    candidate_metrics: dict[
        int,
        dict[str, Any],
    ] = {}

    candidate_matrices: dict[
        int,
        dict[str, np.ndarray],
    ] = {}

    synthetic_packages = {
        "selected_subject_indices": np.asarray(
            selected_subjects,
            dtype=np.int64,
        ),
        "selected_subject_ids": np.asarray(
            selected_subject_ids,
            dtype="U32",
        ),
    }

    total_start = time.perf_counter()

    for epoch in CANDIDATE_EPOCHS:
        print(
            "\n"
            + "-" * 80
        )

        print(
            f"B2 RAW EPOCH {epoch} "
            f"— {SELECTED_SUBJECT_COUNT} SUBJECTS × 2 STATES × "
            f"{SYNTHETIC_WINDOWS_PER_SUBJECT_STATE} REPS"
        )

        print(
            "-" * 80
        )

        checkpoint = torch.load(
            b2_candidate_path(
                epoch
            ),
            map_location="cpu",
            weights_only=False,
        )

        model = build_model_from_state(
            frozen_b1=(
                frozen_b1
            ),
            state_dict=(
                checkpoint[
                    "model_state_dict"
                ]
            ),
            device=device,
        )

        candidate_start = time.perf_counter()

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
        )

        generic, matrices, _ = (
            compute_candidate_metrics(
                real_z=real_z,
                synth_z=synth_z,
                scaler=scaler,
            )
        )

        full_psd = full_log_psd_mae(
            real_z=real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

        metrics = {
            "epoch": (
                epoch
            ),
            "generic": (
                generic
            ),
            "full_log_psd": (
                full_psd
            ),
            "candidate_seconds": float(
                time.perf_counter()
                - candidate_start
            ),
            "checkpoint": (
                candidate_audit[
                    epoch
                ]
            ),
        }

        candidate_metrics[
            epoch
        ] = metrics

        candidate_matrices[
            epoch
        ] = matrices

        synthetic_packages[
            f"epoch_{epoch:03d}_z"
        ] = synth_z.astype(
            np.float32,
            copy=False,
        )

        print(
            f"full 1-40 Hz log-PSD MAE: "
            f"{full_psd['mean_mae_db']:.3f} dB"
        )

        print(
            "scale EO/EC: "
            f"{generic['scale']['EO']['normalized_std_ratio']:.3f}"
            " / "
            f"{generic['scale']['EC']['normalized_std_ratio']:.3f}"
        )

        print(
            f"mean bandpower MAE: "
            f"{generic['mean_bandpower_mae_db']:.3f} dB"
        )

        print(
            f"alpha-effect MAE: "
            f"{generic['alpha_effect']['channel_mae_db']:.3f} dB"
        )

        print(
            "correlation r EO/EC: "
            f"{generic['spatial']['EO']['correlation_r']:.3f}"
            " / "
            f"{generic['spatial']['EC']['correlation_r']:.3f}"
        )

        print(
            "alpha dwPLI2 r EO/EC: "
            f"{generic['spatial']['EO']['alpha_dwpli2_r']:.3f}"
            " / "
            f"{generic['spatial']['EC']['alpha_dwpli2_r']:.3f}"
        )

        subject_metrics = generic[
            "subject_identity"
        ]

        print(
            "subject fingerprint r median: "
            f"{subject_metrics['subject_fingerprint_pearson_r_median']:.3f}"
        )

        print(
            "subject retrieval Top1/Top5: "
            f"{subject_metrics['subject_retrieval_top1_accuracy']:.3f}"
            " / "
            f"{subject_metrics['subject_retrieval_top5_accuracy']:.3f}"
        )

        print(
            f"candidate time: "
            f"{metrics['candidate_seconds']/60:.2f} min"
        )

        del model
        torch.cuda.empty_cache()

    runtime_seconds = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # Step 5 — Apply frozen B2 selection rule
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — APPLY FROZEN B2 SPECTRAL-IMPROVEMENT + PRESERVATION RULE"
    )

    print(
        "=" * 80
    )

    eligibility = {}

    eligible_epochs = []

    for epoch in CANDIDATE_EPOCHS:
        result = evaluate_eligibility(
            candidate=(
                candidate_metrics[
                    epoch
                ]
            ),
            b1_baseline=(
                b1_baseline
            ),
        )

        eligibility[
            epoch
        ] = result

        if result[
            "eligible"
        ]:
            eligible_epochs.append(
                epoch
            )

        print(
            "\n"
            f"epoch {epoch}:"
        )

        print(
            "  full PSD improvement vs B1: "
            f"{100.0 * result['full_psd_improvement_fraction']:.2f}%"
        )

        print(
            "  EO/EC scale ratio to B1: "
            f"{result['EO_scale_ratio_to_frozen_B1']:.3f}"
            " / "
            f"{result['EC_scale_ratio_to_frozen_B1']:.3f}"
        )

        for criterion_name, passed in result[
            "criteria"
        ].items():
            print(
                f"  {criterion_name}: "
                f"{passed}"
            )

        print(
            f"  FINAL ELIGIBLE: "
            f"{result['eligible']}"
        )

    if eligible_epochs:
        selected_epoch = min(
            eligible_epochs,
            key=lambda epoch: (
                float(
                    candidate_metrics[
                        epoch
                    ][
                        "full_log_psd"
                    ][
                        "mean_mae_db"
                    ]
                ),
                float(
                    candidate_metrics[
                        epoch
                    ][
                        "generic"
                    ][
                        "mean_bandpower_mae_db"
                    ]
                ),
                float(
                    candidate_metrics[
                        epoch
                    ][
                        "generic"
                    ][
                        "alpha_effect"
                    ][
                        "channel_mae_db"
                    ]
                ),
                int(
                    epoch
                ),
            ),
        )

        diagnostic_direction = (
            f"B2_RAW_EPOCH_{selected_epoch}_PRIMARY_CANDIDATE"
        )
    else:
        selected_epoch = None

        diagnostic_direction = (
            "NO_ELIGIBLE_B2_RAW_CANDIDATE"
        )

    print(
        f"\neligible B2 epochs: "
        f"{eligible_epochs}"
    )

    print(
        f"selected B2 epoch: "
        f"{selected_epoch}"
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
        "STEP 6 — SAVE B2 FINAL-SELECTION AUDIT ARTIFACTS"
    )

    print(
        "=" * 80
    )

    metric_rows = []

    for epoch in CANDIDATE_EPOCHS:
        metrics = candidate_metrics[
            epoch
        ]

        generic = metrics[
            "generic"
        ]

        subject = generic[
            "subject_identity"
        ]

        guard = eligibility[
            epoch
        ]

        metric_rows.append(
            {
                "epoch": (
                    epoch
                ),
                "full_log_psd_mae_db": (
                    metrics[
                        "full_log_psd"
                    ][
                        "mean_mae_db"
                    ]
                ),
                "full_log_psd_improvement_vs_B1_fraction": (
                    guard[
                        "full_psd_improvement_fraction"
                    ]
                ),
                "eo_normalized_std_ratio": (
                    generic[
                        "scale"
                    ][
                        "EO"
                    ][
                        "normalized_std_ratio"
                    ]
                ),
                "ec_normalized_std_ratio": (
                    generic[
                        "scale"
                    ][
                        "EC"
                    ][
                        "normalized_std_ratio"
                    ]
                ),
                "eo_scale_ratio_to_B1": (
                    guard[
                        "EO_scale_ratio_to_frozen_B1"
                    ]
                ),
                "ec_scale_ratio_to_B1": (
                    guard[
                        "EC_scale_ratio_to_frozen_B1"
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
                "subject_margin": (
                    subject[
                        "subject_correct_vs_median_wrong_margin_mean"
                    ]
                ),
                "correct_closer_than_fixed_wrong_fraction": (
                    subject[
                        "subject_correct_better_than_fixed_wrong_fraction"
                    ]
                ),
                "eligible": (
                    guard[
                        "eligible"
                    ]
                ),
                "candidate_seconds": (
                    metrics[
                        "candidate_seconds"
                    ]
                ),
            }
        )

    save_csv_rows(
        metric_rows,
        METRICS_CSV,
    )

    np.savez_compressed(
        SYNTHETIC_NPZ,
        **synthetic_packages,
    )

    matrix_payload = {}

    for key, value in b1_matrices.items():
        matrix_payload[
            f"B1_{key}"
        ] = value

    for epoch in CANDIDATE_EPOCHS:
        for key, value in candidate_matrices[
            epoch
        ].items():
            matrix_payload[
                f"B2_epoch_{epoch:03d}_{key}"
            ] = value

    np.savez_compressed(
        MATRICES_NPZ,
        **matrix_payload,
    )

    save_figures(
        b1_baseline=b1_baseline,
        candidates=(
            candidate_metrics
        ),
        eligibility=(
            eligibility
        ),
    )

    print(
        f"metrics CSV: "
        f"{METRICS_CSV}"
    )

    print(
        f"synthetic package: "
        f"{SYNTHETIC_NPZ}"
    )

    print(
        f"matrix package: "
        f"{MATRICES_NPZ}"
    )

    # ========================================================
    # Step 7 — Immutability + final summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — VERIFY IMMUTABILITY + SAVE SUMMARY"
    )

    print(
        "=" * 80
    )

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != frozen_b1_file_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during B2 audit."
        )

    for epoch in CANDIDATE_EPOCHS:
        if sha256_file(
            b2_candidate_path(
                epoch
            )
        ) != candidate_hash_before[
            epoch
        ]:
            raise RuntimeError(
                f"B2 candidate epoch {epoch} changed during audit."
            )

    summary = {
        "status": (
            "B2_RAW_CANDIDATE_SUBJECT_AWARE_SPECTRAL_SWEEP_COMPLETE"
        ),
        "candidate_epochs": (
            CANDIDATE_EPOCHS
        ),
        "lambda_spectral": (
            EXPECTED_SELECTED_LAMBDA
        ),
        "frozen_B1_baseline": {
            "generator": str(
                FROZEN_B1_GENERATOR
            ),
            "generator_sha256": (
                frozen_b1_file_hash_before
            ),
            "real_reference": str(
                B1_REAL_REFERENCE_NPZ
            ),
            "synthetic_reference": str(
                B1_SYNTHETIC_REFERENCE_NPZ
            ),
            "metrics": (
                b1_baseline
            ),
        },
        "candidate_checkpoint_audit": (
            candidate_audit
        ),
        "metrics_by_epoch": {
            str(
                epoch
            ): candidate_metrics[
                epoch
            ]
            for epoch in CANDIDATE_EPOCHS
        },
        "eligibility_by_epoch": {
            str(
                epoch
            ): eligibility[
                epoch
            ]
            for epoch in CANDIDATE_EPOCHS
        },
        "eligible_epochs": (
            eligible_epochs
        ),
        "recommended_epoch": (
            selected_epoch
        ),
        "diagnostic_direction": (
            diagnostic_direction
        ),
        "runtime_seconds": (
            runtime_seconds
        ),
        "runtime_minutes": (
            runtime_seconds
            / 60.0
        ),
        "matched_lambda0_control_status": (
            "NOT_RUN_YET"
        ),
        "causal_spectral_regularization_claim_ready": False,
        "artifacts": {
            "protocol": str(
                PROTOCOL_FILE
            ),
            "metrics_csv": str(
                METRICS_CSV
            ),
            "synthetic_npz": str(
                SYNTHETIC_NPZ
            ),
            "matrices_npz": str(
                MATRICES_NPZ
            ),
            "psd_figure": str(
                PSD_FIGURE
            ),
            "scale_figure": str(
                SCALE_FIGURE
            ),
            "subject_figure": str(
                SUBJECT_FIGURE
            ),
            "generic_figure": str(
                GENERIC_FIGURE
            ),
            "eligibility_figure": str(
                ELIGIBILITY_FIGURE
            ),
            "overview_figure": str(
                OVERVIEW_FIGURE
            ),
        },
        "governance": {
            "Train_reference_used": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "training_updates_executed": False,
            "bulk_synthetic_generation_started": False,
            "B2_frozen": False,
            "frozen_B1_unchanged": True,
            "B2_candidate_checkpoints_unchanged": True,
        },
        "next": (
            "Interpret B2 candidate sweep. If recommended_epoch is "
            "non-null, formally freeze B2. A matched full lambda=0 "
            "continuation remains required before causal attribution "
            "of improvement to spectral regularization."
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
        "B2 candidate checkpoint immutability: PASS"
    )

    print(
        f"summary JSON: "
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
        "B2 RAW SUBJECT-AWARE + SPECTRAL FIDELITY SWEEP COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nFrozen B1 baseline:"
    )

    print(
        "  full PSD MAE: "
        f"{b1_baseline['full_log_psd']['mean_mae_db']:.3f} dB"
    )

    print(
        "  mean bandpower MAE: "
        f"{b1_baseline['generic']['mean_bandpower_mae_db']:.3f} dB"
    )

    print(
        "  alpha-effect MAE: "
        f"{b1_baseline['generic']['alpha_effect']['channel_mae_db']:.3f} dB"
    )

    print(
        "  subject Top1: "
        f"{b1_baseline['generic']['subject_identity']['subject_retrieval_top1_accuracy']:.3f}"
    )

    print(
        "  subject fingerprint r median: "
        f"{b1_baseline['generic']['subject_identity']['subject_fingerprint_pearson_r_median']:.3f}"
    )

    for epoch in CANDIDATE_EPOCHS:
        metrics = candidate_metrics[
            epoch
        ]

        generic = metrics[
            "generic"
        ]

        subject = generic[
            "subject_identity"
        ]

        guard = eligibility[
            epoch
        ]

        print(
            "\n"
            f"B2 RAW epoch {epoch}:"
        )

        print(
            "  full PSD MAE: "
            f"{metrics['full_log_psd']['mean_mae_db']:.3f} dB"
        )

        print(
            "  full PSD improvement vs B1: "
            f"{100.0 * guard['full_psd_improvement_fraction']:.2f}%"
        )

        print(
            "  scale EO/EC: "
            f"{generic['scale']['EO']['normalized_std_ratio']:.3f}"
            " / "
            f"{generic['scale']['EC']['normalized_std_ratio']:.3f}"
        )

        print(
            "  scale ratio to B1 EO/EC: "
            f"{guard['EO_scale_ratio_to_frozen_B1']:.3f}"
            " / "
            f"{guard['EC_scale_ratio_to_frozen_B1']:.3f}"
        )

        print(
            "  mean bandpower MAE: "
            f"{generic['mean_bandpower_mae_db']:.3f} dB"
        )

        print(
            "  alpha-effect MAE: "
            f"{generic['alpha_effect']['channel_mae_db']:.3f} dB"
        )

        print(
            "  correlation r EO/EC: "
            f"{generic['spatial']['EO']['correlation_r']:.3f}"
            " / "
            f"{generic['spatial']['EC']['correlation_r']:.3f}"
        )

        print(
            "  alpha dwPLI2 r EO/EC: "
            f"{generic['spatial']['EO']['alpha_dwpli2_r']:.3f}"
            " / "
            f"{generic['spatial']['EC']['alpha_dwpli2_r']:.3f}"
        )

        print(
            "  subject Top1/Top5: "
            f"{subject['subject_retrieval_top1_accuracy']:.3f}"
            " / "
            f"{subject['subject_retrieval_top5_accuracy']:.3f}"
        )

        print(
            "  subject fingerprint r median: "
            f"{subject['subject_fingerprint_pearson_r_median']:.3f}"
        )

        print(
            "  eligible: "
            f"{guard['eligible']}"
        )

    print(
        f"\neligible epochs: "
        f"{eligible_epochs}"
    )

    print(
        f"recommended epoch: "
        f"{selected_epoch}"
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
        "B2 frozen: NO"
    )

    print(
        "\nMatched lambda=0 full continuation:"
    )

    print(
        "  NOT RUN YET — still required before causal publication "
        "claim that spectral regularization caused improvement."
    )

    print(
        "\nNEXT:"
    )

    if selected_epoch is None:
        print(
            "STOP — INTERPRET B2 FAILURE / TRADE-OFF BEFORE FREEZE"
        )
    else:
        print(
            "FORMALLY FREEZE RECOMMENDED B2 CANDIDATE, "
            "THEN COMPLETE MATCHED lambda=0 CONTROL BEFORE "
            "CAUSAL ATTRIBUTION / FINAL BULK EXPERIMENTS"
        )


if __name__ == "__main__":
    main()
