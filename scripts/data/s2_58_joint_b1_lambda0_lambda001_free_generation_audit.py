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

from s2_23_check_real_train_dataset import (
    NORMALIZATION_FILE,
)
from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    build_reverse_schedule,
    welch_psd,
)
from s2_47_b1_architecture_warmstart_sanity import (
    state_dict_sha256,
)
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import (
    compute_candidate_metrics,
    sample_b1_candidate_state,
)
from s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep import (
    build_model_from_state,
    evaluate_eligibility,
    full_log_psd_mae,
    mean_corr_r,
    mean_wpli_r,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen B1.
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

# Exact B1 V2 subject-aware reference and cached B1 synthetic samples.
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

# Primary lambda=0.01 B2 full run and its already-completed free-generation audit.
PRIMARY_B2_FULL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "full_finetune_v1"
)

PRIMARY_B2_CHECKPOINT_DIR = (
    PRIMARY_B2_FULL_ROOT
    / "checkpoints"
)

PRIMARY_B2_RESULT = (
    PRIMARY_B2_FULL_ROOT
    / "metrics"
    / "b2_full_finetune_result_v1.json"
)

PRIMARY_B2_SWEEP_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "raw_candidate_subject_aware_spectral_sweep_v1"
)

PRIMARY_B2_SWEEP_SUMMARY = (
    PRIMARY_B2_SWEEP_ROOT
    / "metrics"
    / "b2_raw_candidate_subject_aware_spectral_summary_v1.json"
)

PRIMARY_B2_SYNTHETIC_NPZ = (
    PRIMARY_B2_SWEEP_ROOT
    / "generated"
    / "b2_raw_candidate_subject_aware_synthetic_v1.npz"
)

# Matched lambda=0 continuation control.
CONTROL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "matched_lambda0_control_v1"
)

CONTROL_CHECKPOINT_DIR = (
    CONTROL_ROOT
    / "checkpoints"
)

CONTROL_RESULT = (
    CONTROL_ROOT
    / "metrics"
    / "b2_matched_lambda0_control_result_v1.json"
)


# ============================================================
# s2_58 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "joint_b1_lambda0_lambda001_freegen_audit_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
GENERATED_DIR = RUN_ROOT / "generated"
FIGURE_DIR = RUN_ROOT / "figures"
LOG_DIR = RUN_ROOT / "logs"

PROTOCOL_FILE = (
    METRIC_DIR
    / "s2_58_joint_freegen_protocol_v1.json"
)

METRICS_CSV = (
    METRIC_DIR
    / "s2_58_joint_freegen_metrics_v1.csv"
)

DELTAS_CSV = (
    METRIC_DIR
    / "s2_58_joint_freegen_matched_deltas_v1.csv"
)

SUBJECT_PSD_CSV = (
    METRIC_DIR
    / "s2_58_subject_level_psd_diagnostics_v1.csv"
)

TRAINING_VS_GENERATION_CSV = (
    METRIC_DIR
    / "s2_58_training_vs_generation_v1.csv"
)

BOOTSTRAP_CSV = (
    METRIC_DIR
    / "s2_58_paired_subject_bootstrap_v1.csv"
)

SPECTRAL_DIAGNOSTICS_NPZ = (
    METRIC_DIR
    / "s2_58_spectral_diagnostics_v1.npz"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_58_joint_freegen_summary_v1.json"
)

JOINT_PSD_FIGURE = (
    FIGURE_DIR
    / "s2_58_joint_full_psd_mae_v1.png"
)

JOINT_GENERIC_FIGURE = (
    FIGURE_DIR
    / "s2_58_joint_bandpower_alpha_v1.png"
)

JOINT_SUBJECT_FIGURE = (
    FIGURE_DIR
    / "s2_58_joint_subject_identity_v1.png"
)

PSD_ERROR_FIGURE = (
    FIGURE_DIR
    / "s2_58_signed_psd_error_by_frequency_v1.png"
)


# ============================================================
# Frozen s2_58 design
# ============================================================

CANDIDATE_EPOCHS = [
    5,
    10,
    20,
]

PRIMARY_LAMBDA = 0.01
CONTROL_LAMBDA = 0.0

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

EXPECTED_REAL_SHAPE = (
    SELECTED_SUBJECT_COUNT,
    2,
    REAL_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)

EXPECTED_SYNTH_SHAPE = (
    SELECTED_SUBJECT_COUNT,
    2,
    SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)

TIMESTEPS = 1000
COSINE_S = 0.008

# Exactly match the B1 V2 / s2_56 free-generation sampling convention.
SAMPLING_SEED_BASE = 20261001

# Diagnostic bootstrap only. This does NOT change any frozen
# scientific selection threshold or make a new final-selection decision.
BOOTSTRAP_SEED = 20261158
BOOTSTRAP_REPLICATES = 1000

CHANNEL_NAMES = [
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "F3",
    "F4",
    "Fz",
    "T7",
    "T8",
    "C3",
    "C4",
    "Cz",
    "P7",
    "P8",
    "P3",
    "P4",
    "Pz",
    "O1",
    "O2",
]

STATE_NAMES = [
    "EO",
    "EC",
]


# ============================================================
# IO helpers
# ============================================================

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


def atomic_npz_save(
    path: Path,
    **arrays,
) -> None:
    """
    Save compressed NPZ atomically.

    np.savez_compressed appends .npz when given a string that does not
    already end in .npz, so write through an explicitly-opened file.
    """
    temp = Path(
        str(path)
        + ".tmp"
    )

    with temp.open(
        "wb"
    ) as f:
        np.savez_compressed(
            f,
            **arrays,
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


def primary_candidate_path(
    epoch: int,
) -> Path:
    return (
        PRIMARY_B2_CHECKPOINT_DIR
        / f"candidate_epoch_{epoch:03d}_raw.pt"
    )


def control_candidate_path(
    epoch: int,
) -> Path:
    return (
        CONTROL_CHECKPOINT_DIR
        / f"candidate_epoch_{epoch:03d}_raw.pt"
    )


def control_synthetic_path(
    epoch: int,
) -> Path:
    return (
        GENERATED_DIR
        / f"lambda0_control_epoch_{epoch:03d}_synthetic_v1.npz"
    )


# ============================================================
# Exact reference/cache loading
# ============================================================

def load_exact_b1_reference() -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[str],
]:
    real_package = np.load(
        B1_REAL_REFERENCE_NPZ,
        allow_pickle=False,
    )

    b1_synth_package = np.load(
        B1_SYNTHETIC_REFERENCE_NPZ,
        allow_pickle=False,
    )

    real_z = real_package[
        "real_z"
    ].astype(
        np.float32,
        copy=False,
    )

    real_subject_indices = (
        real_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    real_subject_ids = (
        real_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    synth_subject_indices = (
        b1_synth_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    synth_subject_ids = (
        b1_synth_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if (
        real_subject_indices
        != synth_subject_indices
    ):
        raise RuntimeError(
            "B1 Real/synthetic subject-index cohort mismatch."
        )

    if (
        real_subject_ids
        != synth_subject_ids
    ):
        raise RuntimeError(
            "B1 Real/synthetic subject-ID cohort mismatch."
        )

    b1_synth_z = b1_synth_package[
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
            "Frozen-B1 synthetic cache contains NaN/Inf."
        )

    if len(
        real_subject_indices
    ) != SELECTED_SUBJECT_COUNT:
        raise RuntimeError(
            "Unexpected selected-subject count."
        )

    return (
        real_z,
        b1_synth_z,
        real_subject_indices,
        real_subject_ids,
    )


def load_primary_b2_cache(
    *,
    expected_subject_indices: list[int],
    expected_subject_ids: list[str],
) -> dict[int, np.ndarray]:
    package = np.load(
        PRIMARY_B2_SYNTHETIC_NPZ,
        allow_pickle=False,
    )

    subject_indices = (
        package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    subject_ids = (
        package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if (
        subject_indices
        != expected_subject_indices
    ):
        raise RuntimeError(
            "Primary B2 cached subject-index cohort mismatch."
        )

    if (
        subject_ids
        != expected_subject_ids
    ):
        raise RuntimeError(
            "Primary B2 cached subject-ID cohort mismatch."
        )

    output = {}

    for epoch in CANDIDATE_EPOCHS:
        key = (
            f"epoch_{epoch:03d}_z"
        )

        if key not in package.files:
            raise RuntimeError(
                f"Primary B2 cache is missing key: {key}"
            )

        value = package[
            key
        ].astype(
            np.float32,
            copy=False,
        )

        if value.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"Unexpected primary B2 shape at epoch {epoch}: "
                f"{value.shape}"
            )

        if not np.isfinite(
            value
        ).all():
            raise RuntimeError(
                f"Primary B2 synthetic cache contains NaN/Inf "
                f"at epoch {epoch}."
            )

        output[
            epoch
        ] = value

    return output


# ============================================================
# Lambda=0 control generation / safe resume
# ============================================================

def load_or_generate_control_candidate(
    *,
    epoch: int,
    frozen_b1: dict[str, Any],
    selected_subjects: list[int],
    selected_subject_ids: list[str],
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    checkpoint_path = control_candidate_path(
        epoch
    )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    cache_path = control_synthetic_path(
        epoch
    )

    if cache_path.is_file():
        package = np.load(
            cache_path,
            allow_pickle=False,
        )

        cached_indices = (
            package[
                "selected_subject_indices"
            ]
            .astype(
                np.int64,
                copy=False,
            )
            .tolist()
        )

        cached_ids = (
            package[
                "selected_subject_ids"
            ]
            .astype(
                str
            )
            .tolist()
        )

        cached_hash = str(
            package[
                "checkpoint_sha256"
            ][
                0
            ]
        )

        cached_seed = int(
            package[
                "sampling_seed_base"
            ][
                0
            ]
        )

        synth_z = package[
            "synth_z"
        ].astype(
            np.float32,
            copy=False,
        )

        if (
            cached_indices
            != selected_subjects
        ):
            raise RuntimeError(
                f"Control cache cohort mismatch at epoch {epoch}."
            )

        if (
            cached_ids
            != selected_subject_ids
        ):
            raise RuntimeError(
                f"Control cache subject-ID mismatch at epoch {epoch}."
            )

        if cached_hash != checkpoint_hash:
            raise RuntimeError(
                f"Control cache checkpoint-hash mismatch at epoch {epoch}."
            )

        if cached_seed != SAMPLING_SEED_BASE:
            raise RuntimeError(
                f"Control cache sampling-seed mismatch at epoch {epoch}."
            )

        if synth_z.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"Control cache shape mismatch at epoch {epoch}: "
                f"{synth_z.shape}"
            )

        if not np.isfinite(
            synth_z
        ).all():
            raise RuntimeError(
                f"Control cache contains NaN/Inf at epoch {epoch}."
            )

        print(
            f"CONTROL epoch {epoch}: validated existing cache -> REUSE"
        )

        return synth_z

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint[
        "status"
    ] != "B2_MATCHED_LAMBDA0_CONTROL_CHECKPOINT":
        raise RuntimeError(
            f"Unexpected control checkpoint status at epoch {epoch}."
        )

    if int(
        checkpoint[
            "epoch"
        ]
    ) != epoch:
        raise RuntimeError(
            f"Control checkpoint epoch mismatch for {epoch}."
        )

    if abs(
        float(
            checkpoint[
                "lambda_spectral"
            ]
        )
        - CONTROL_LAMBDA
    ) > 1e-12:
        raise RuntimeError(
            f"Control lambda mismatch at epoch {epoch}."
        )

    if checkpoint[
        "selected_weight_kind"
    ] != "RAW":
        raise RuntimeError(
            f"Control epoch {epoch} is not RAW."
        )

    if checkpoint[
        "EMA_enabled"
    ] is not False:
        raise RuntimeError(
            f"Control epoch {epoch} unexpectedly uses EMA."
        )

    if checkpoint[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            f"Control epoch {epoch} reports Final Test access."
        )

    model = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=(
            checkpoint[
                "model_state_dict"
            ]
        ),
        device=device,
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"GENERATE MATCHED lambda=0 CONTROL RAW EPOCH {epoch}"
    )

    print(
        "-" * 80
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
    ).astype(
        np.float32,
        copy=False,
    )

    if synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected generated control shape at epoch {epoch}: "
            f"{synth_z.shape}"
        )

    if not np.isfinite(
        synth_z
    ).all():
        raise RuntimeError(
            f"Generated control epoch {epoch} contains NaN/Inf."
        )

    atomic_npz_save(
        cache_path,
        selected_subject_indices=np.asarray(
            selected_subjects,
            dtype=np.int64,
        ),
        selected_subject_ids=np.asarray(
            selected_subject_ids,
            dtype="U32",
        ),
        synth_z=synth_z,
        checkpoint_sha256=np.asarray(
            [
                checkpoint_hash
            ],
            dtype="U64",
        ),
        sampling_seed_base=np.asarray(
            [
                SAMPLING_SEED_BASE
            ],
            dtype=np.int64,
        ),
        generation_seconds=np.asarray(
            [
                time.perf_counter()
                - candidate_start
            ],
            dtype=np.float64,
        ),
    )

    print(
        f"control epoch {epoch} generation time: "
        f"{(time.perf_counter() - candidate_start)/60:.2f} min"
    )

    print(
        f"control cache: {cache_path}"
    )

    del model
    torch.cuda.empty_cache()

    return synth_z


# ============================================================
# PSD diagnostics
# ============================================================

def log_psd_tensor(
    eeg_z: np.ndarray,
    scaler: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    [S, 2, R, C, T] ->
      frequencies [F] for 1-40 Hz
      log-PSD [S, 2, R, C, F] in dB

    Uses exactly the same Welch convention as the existing audit helper:
      nperseg=256
      noverlap=128
      Hann
      density scaling
    """

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

    original_shape = eeg_z.shape

    flat_z = eeg_z.reshape(
        (
            -1,
            N_CHANNELS,
            N_TIMES,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    flat_uv = (
        flat_z
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

    frequencies, psd = welch_psd(
        flat_uv
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

    selected_frequencies = frequencies[
        mask
    ]

    tiny = np.finfo(
        np.float64
    ).tiny

    log_psd = (
        10.0
        * np.log10(
            np.maximum(
                psd[
                    ...,
                    mask
                ],
                tiny,
            )
        )
    )

    output = log_psd.reshape(
        (
            original_shape[
                0
            ],
            original_shape[
                1
            ],
            original_shape[
                2
            ],
            original_shape[
                3
            ],
            len(
                selected_frequencies
            ),
        )
    )

    return (
        selected_frequencies,
        output,
    )


def pooled_state_median(
    log_psd: np.ndarray,
) -> np.ndarray:
    """
    [S, 2, R, C, F] -> [2, C, F]

    Pool subject × replicate windows within each state exactly as the
    existing primary full-PSD audit does before taking the median.
    """

    output = []

    for state_index in range(
        2
    ):
        values = log_psd[
            :,
            state_index,
            ...,
        ].reshape(
            (
                -1,
                log_psd.shape[
                    -2
                ],
                log_psd.shape[
                    -1
                ],
            )
        )

        output.append(
            np.median(
                values,
                axis=0,
            )
        )

    return np.stack(
        output,
        axis=0,
    )


def pooled_state_iqr(
    log_psd: np.ndarray,
) -> np.ndarray:
    """
    [S, 2, R, C, F] -> pooled IQR [2, C, F]
    """

    output = []

    for state_index in range(
        2
    ):
        values = log_psd[
            :,
            state_index,
            ...,
        ].reshape(
            (
                -1,
                log_psd.shape[
                    -2
                ],
                log_psd.shape[
                    -1
                ],
            )
        )

        q75 = np.percentile(
            values,
            75,
            axis=0,
        )

        q25 = np.percentile(
            values,
            25,
            axis=0,
        )

        output.append(
            q75
            - q25
        )

    return np.stack(
        output,
        axis=0,
    )


def primary_psd_mae_from_log_psd(
    *,
    real_log_psd: np.ndarray,
    synth_log_psd: np.ndarray,
) -> float:
    real_median = pooled_state_median(
        real_log_psd
    )

    synth_median = pooled_state_median(
        synth_log_psd
    )

    state_maes = np.mean(
        np.abs(
            synth_median
            - real_median
        ),
        axis=(
            1,
            2,
        ),
    )

    return float(
        np.mean(
            state_maes
        )
    )


def spectral_diagnostics(
    *,
    frequencies: np.ndarray,
    real_log_psd: np.ndarray,
    synth_log_psd: np.ndarray,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    real_median = pooled_state_median(
        real_log_psd
    )

    synth_median = pooled_state_median(
        synth_log_psd
    )

    signed_error = (
        synth_median
        - real_median
    )

    state_mae = np.mean(
        np.abs(
            signed_error
        ),
        axis=(
            1,
            2,
        ),
    )

    state_bias = np.mean(
        signed_error,
        axis=(
            1,
            2,
        ),
    )

    # Remove a per-state/per-channel constant offset over frequency.
    # This is DIAGNOSTIC only; it does not alter the primary endpoint.
    channel_offset = np.mean(
        signed_error,
        axis=-1,
    )

    shape_error = (
        signed_error
        - channel_offset[
            ...,
            None
        ]
    )

    shape_mae = float(
        np.mean(
            np.abs(
                shape_error
            )
        )
    )

    log_frequency = np.log10(
        frequencies.astype(
            np.float64
        )
    )

    slopes = np.empty(
        (
            2,
            N_CHANNELS,
        ),
        dtype=np.float64,
    )

    for state_index in range(
        2
    ):
        for channel_index in range(
            N_CHANNELS
        ):
            coefficient = np.polyfit(
                log_frequency,
                signed_error[
                    state_index,
                    channel_index,
                ],
                deg=1,
            )

            slopes[
                state_index,
                channel_index,
            ] = coefficient[
                0
            ]

    real_iqr = pooled_state_iqr(
        real_log_psd
    )

    synth_iqr = pooled_state_iqr(
        synth_log_psd
    )

    iqr_mae = float(
        np.mean(
            np.abs(
                synth_iqr
                - real_iqr
            )
        )
    )

    subject_rows = []

    for subject_index in range(
        SELECTED_SUBJECT_COUNT
    ):
        subject_state_maes = []

        for state_index, state_name in enumerate(
            STATE_NAMES
        ):
            real_subject_median = np.median(
                real_log_psd[
                    subject_index,
                    state_index,
                ],
                axis=0,
            )

            synth_subject_median = np.median(
                synth_log_psd[
                    subject_index,
                    state_index,
                ],
                axis=0,
            )

            mae = float(
                np.mean(
                    np.abs(
                        synth_subject_median
                        - real_subject_median
                    )
                )
            )

            subject_state_maes.append(
                mae
            )

            subject_rows.append(
                {
                    "local_subject_index": (
                        subject_index
                    ),
                    "eye_state": (
                        state_name
                    ),
                    "subject_level_psd_mae_db": (
                        mae
                    ),
                }
            )

        subject_rows.append(
            {
                "local_subject_index": (
                    subject_index
                ),
                "eye_state": (
                    "MEAN"
                ),
                "subject_level_psd_mae_db": float(
                    np.mean(
                        subject_state_maes
                    )
                ),
            }
        )

    diagnostics = {
        "full_psd_mae_recomputed_db": float(
            np.mean(
                state_mae
            )
        ),
        "EO_psd_mae_db": float(
            state_mae[
                0
            ]
        ),
        "EC_psd_mae_db": float(
            state_mae[
                1
            ]
        ),
        "global_signed_psd_bias_db": float(
            np.mean(
                signed_error
            )
        ),
        "EO_signed_psd_bias_db": float(
            state_bias[
                0
            ]
        ),
        "EC_signed_psd_bias_db": float(
            state_bias[
                1
            ]
        ),
        "shape_mae_after_state_channel_offset_db": (
            shape_mae
        ),
        "mean_tilt_db_per_decade": float(
            np.mean(
                slopes
            )
        ),
        "mean_abs_tilt_db_per_decade": float(
            np.mean(
                np.abs(
                    slopes
                )
            )
        ),
        "pooled_iqr_mae_db": (
            iqr_mae
        ),
    }

    arrays = {
        "real_median_log_psd_db": (
            real_median
        ),
        "synthetic_median_log_psd_db": (
            synth_median
        ),
        "signed_error_db": (
            signed_error
        ),
        "shape_error_db": (
            shape_error
        ),
        "channel_offset_db": (
            channel_offset
        ),
        "tilt_db_per_decade": (
            slopes
        ),
        "real_iqr_db": (
            real_iqr
        ),
        "synthetic_iqr_db": (
            synth_iqr
        ),
    }

    return (
        diagnostics,
        arrays,
        subject_rows,
    )


# ============================================================
# Paired subject bootstrap
# ============================================================

def make_bootstrap_subject_indices() -> np.ndarray:
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    return rng.integers(
        low=0,
        high=SELECTED_SUBJECT_COUNT,
        size=(
            BOOTSTRAP_REPLICATES,
            SELECTED_SUBJECT_COUNT,
        ),
        endpoint=False,
        dtype=np.int64,
    )


def bootstrap_primary_mae(
    *,
    real_log_psd: np.ndarray,
    synth_log_psd: np.ndarray,
    subject_indices: np.ndarray,
) -> np.ndarray:
    values = np.empty(
        (
            subject_indices.shape[
                0
            ],
        ),
        dtype=np.float64,
    )

    for bootstrap_index, sampled_subjects in enumerate(
        subject_indices
    ):
        sampled_real = real_log_psd[
            sampled_subjects
        ]

        sampled_synth = synth_log_psd[
            sampled_subjects
        ]

        values[
            bootstrap_index
        ] = primary_psd_mae_from_log_psd(
            real_log_psd=(
                sampled_real
            ),
            synth_log_psd=(
                sampled_synth
            ),
        )

    return values


def bootstrap_difference_summary(
    *,
    candidate_bootstrap: np.ndarray,
    b1_bootstrap: np.ndarray,
) -> dict[str, float]:
    difference = (
        candidate_bootstrap
        - b1_bootstrap
    )

    return {
        "candidate_minus_B1_mean_db": float(
            np.mean(
                difference
            )
        ),
        "candidate_minus_B1_median_db": float(
            np.median(
                difference
            )
        ),
        "candidate_minus_B1_ci95_low_db": float(
            np.percentile(
                difference,
                2.5,
            )
        ),
        "candidate_minus_B1_ci95_high_db": float(
            np.percentile(
                difference,
                97.5,
            )
        ),
        "fraction_candidate_better_than_B1": float(
            np.mean(
                difference
                < 0.0
            )
        ),
    }


# ============================================================
# Metric flattening / matched deltas
# ============================================================

def variant_metric_view(
    bundle: dict[str, Any],
) -> dict[str, float]:
    generic = bundle[
        "generic"
    ]

    subject = generic[
        "subject_identity"
    ]

    return {
        "full_psd_mae_db": float(
            bundle[
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        ),
        "mean_bandpower_mae_db": float(
            generic[
                "mean_bandpower_mae_db"
            ]
        ),
        "alpha_effect_mae_db": float(
            generic[
                "alpha_effect"
            ][
                "channel_mae_db"
            ]
        ),
        "scale_error": float(
            generic[
                "scale_error_mean_abs_ratio_minus_1"
            ]
        ),
        "mean_channel_correlation_r": float(
            mean_corr_r(
                generic
            )
        ),
        "mean_alpha_dwpli2_r": float(
            mean_wpli_r(
                generic
            )
        ),
        "subject_top1": float(
            subject[
                "subject_retrieval_top1_accuracy"
            ]
        ),
        "subject_fingerprint_r_median": float(
            subject[
                "subject_fingerprint_pearson_r_median"
            ]
        ),
        "subject_margin": float(
            subject[
                "subject_correct_vs_median_wrong_margin_mean"
            ]
        ),
        "fixed_wrong_win_fraction": float(
            subject[
                "subject_correct_better_than_fixed_wrong_fraction"
            ]
        ),
    }


DELTA_METRIC_DIRECTIONS = {
    "full_psd_mae_db": "min",
    "mean_bandpower_mae_db": "min",
    "alpha_effect_mae_db": "min",
    "scale_error": "min",
    "mean_channel_correlation_r": "max",
    "mean_alpha_dwpli2_r": "max",
    "subject_top1": "max",
    "subject_fingerprint_r_median": "max",
    "subject_margin": "max",
    "fixed_wrong_win_fraction": "max",
}


def delta_row(
    *,
    epoch: int,
    comparison: str,
    reference_name: str,
    candidate_name: str,
    reference_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []

    for metric_name, direction in DELTA_METRIC_DIRECTIONS.items():
        reference_value = float(
            reference_metrics[
                metric_name
            ]
        )

        candidate_value = float(
            candidate_metrics[
                metric_name
            ]
        )

        raw_delta = (
            candidate_value
            - reference_value
        )

        if direction == "min":
            oriented_improvement = (
                reference_value
                - candidate_value
            )
        elif direction == "max":
            oriented_improvement = (
                candidate_value
                - reference_value
            )
        else:
            raise ValueError(
                direction
            )

        rows.append(
            {
                "epoch": (
                    epoch
                ),
                "comparison": (
                    comparison
                ),
                "reference": (
                    reference_name
                ),
                "candidate": (
                    candidate_name
                ),
                "metric": (
                    metric_name
                ),
                "direction": (
                    direction
                ),
                "reference_value": (
                    reference_value
                ),
                "candidate_value": (
                    candidate_value
                ),
                "candidate_minus_reference": (
                    raw_delta
                ),
                "oriented_improvement_positive_is_better": (
                    oriented_improvement
                ),
            }
        )

    return rows


# ============================================================
# Training-monitor extraction
# ============================================================

def checkpoint_training_monitor(
    *,
    path: Path,
    expected_status: str,
    expected_epoch: int,
    expected_lambda: float,
) -> dict[str, float]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint[
        "status"
    ] != expected_status:
        raise RuntimeError(
            f"Unexpected checkpoint status: {path}"
        )

    if int(
        checkpoint[
            "epoch"
        ]
    ) != expected_epoch:
        raise RuntimeError(
            f"Checkpoint epoch mismatch: {path}"
        )

    if abs(
        float(
            checkpoint[
                "lambda_spectral"
            ]
        )
        - expected_lambda
    ) > 1e-12:
        raise RuntimeError(
            f"Checkpoint lambda mismatch: {path}"
        )

    row = checkpoint[
        "history"
    ][
        -1
    ]

    # The history field names are matched between primary B2 and control.
    return {
        "train_v_mse": float(
            row[
                "train_v_mse"
            ]
        ),
        "train_spectral_loss": float(
            row[
                "train_spectral_loss_db_huber"
            ]
        ),
        "validation_v_mse": float(
            row[
                "validation_correct_subject_v_mse"
            ]
        ),
        "validation_spectral_loss": float(
            row[
                "validation_spectral_loss_db_huber"
            ]
        ),
        "validation_subject_gap": float(
            row[
                "validation_wrong_minus_correct_v_mse"
            ]
        ),
    }


# ============================================================
# Figures
# ============================================================

def save_joint_figures(
    *,
    b1_bundle: dict[str, Any],
    control_bundles: dict[int, dict[str, Any]],
    primary_bundles: dict[int, dict[str, Any]],
    spectral_arrays: dict[str, dict[str, np.ndarray]],
    frequencies: np.ndarray,
) -> None:
    epochs = np.asarray(
        CANDIDATE_EPOCHS,
        dtype=np.int64,
    )

    b1_psd = float(
        b1_bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    control_psd = np.asarray(
        [
            control_bundles[
                epoch
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
        dtype=np.float64,
    )

    primary_psd = np.asarray(
        [
            primary_bundles[
                epoch
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.axhline(
        b1_psd,
        linestyle="--",
        linewidth=1.5,
        label="Frozen B1 RAW10",
    )

    ax.plot(
        epochs,
        control_psd,
        marker="o",
        label="Matched continuation λ=0",
    )

    ax.plot(
        epochs,
        primary_psd,
        marker="o",
        label="Primary B2 λ=0.01",
    )

    ax.set_xticks(
        epochs
    )

    ax.set_xlabel(
        "Continuation epoch"
    )

    ax.set_ylabel(
        "Full 1–40 Hz log-PSD MAE (dB)"
    )

    ax.set_title(
        "s2_58 — Frozen B1 vs Matched λ=0 vs Primary λ=0.01"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        JOINT_PSD_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Generic fidelity.
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    control_bandpower = [
        control_bundles[
            epoch
        ][
            "generic"
        ][
            "mean_bandpower_mae_db"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    primary_bandpower = [
        primary_bundles[
            epoch
        ][
            "generic"
        ][
            "mean_bandpower_mae_db"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    control_alpha = [
        control_bundles[
            epoch
        ][
            "generic"
        ][
            "alpha_effect"
        ][
            "channel_mae_db"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    primary_alpha = [
        primary_bundles[
            epoch
        ][
            "generic"
        ][
            "alpha_effect"
        ][
            "channel_mae_db"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    ax.plot(
        epochs,
        control_bandpower,
        marker="o",
        label="λ=0 bandpower MAE",
    )

    ax.plot(
        epochs,
        primary_bandpower,
        marker="o",
        label="λ=0.01 bandpower MAE",
    )

    ax.plot(
        epochs,
        control_alpha,
        marker="s",
        linestyle="--",
        label="λ=0 alpha-effect MAE",
    )

    ax.plot(
        epochs,
        primary_alpha,
        marker="s",
        linestyle="--",
        label="λ=0.01 alpha-effect MAE",
    )

    ax.set_xticks(
        epochs
    )

    ax.set_xlabel(
        "Continuation epoch"
    )

    ax.set_ylabel(
        "Error (dB)"
    )

    ax.set_title(
        "s2_58 — Generic Spectral Preservation"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        JOINT_GENERIC_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Subject identity.
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    control_top1 = [
        control_bundles[
            epoch
        ][
            "generic"
        ][
            "subject_identity"
        ][
            "subject_retrieval_top1_accuracy"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    primary_top1 = [
        primary_bundles[
            epoch
        ][
            "generic"
        ][
            "subject_identity"
        ][
            "subject_retrieval_top1_accuracy"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    b1_top1 = float(
        b1_bundle[
            "generic"
        ][
            "subject_identity"
        ][
            "subject_retrieval_top1_accuracy"
        ]
    )

    ax.axhline(
        b1_top1,
        linestyle="--",
        linewidth=1.5,
        label="Frozen B1 Top-1",
    )

    ax.plot(
        epochs,
        control_top1,
        marker="o",
        label="Matched λ=0 Top-1",
    )

    ax.plot(
        epochs,
        primary_top1,
        marker="o",
        label="Primary λ=0.01 Top-1",
    )

    ax.set_xticks(
        epochs
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        "Continuation epoch"
    )

    ax.set_ylabel(
        "Subject Top-1 retrieval"
    )

    ax.set_title(
        "s2_58 — Subject Identity Preservation"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        JOINT_SUBJECT_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Mean signed PSD error by frequency, averaged over states/channels.
    fig, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    keys_to_plot = [
        "B1",
        "L0_E005",
        "L0_E010",
        "L0_E020",
        "L001_E005",
        "L001_E010",
        "L001_E020",
    ]

    for key in keys_to_plot:
        error = spectral_arrays[
            key
        ][
            "signed_error_db"
        ]

        curve = np.mean(
            error,
            axis=(
                0,
                1,
            ),
        )

        ax.plot(
            frequencies,
            curve,
            label=key,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "Synthetic − Real median log-PSD (dB)"
    )

    ax.set_title(
        "s2_58 — Signed Spectral Error by Frequency"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        PSD_ERROR_FIGURE,
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
        "SCENARIO 2 — s2_58 JOINT FREE-GENERATION AUDIT"
    )

    print(
        "FROZEN B1 vs MATCHED lambda=0 vs PRIMARY lambda=0.01"
    )

    print(
        "=" * 80
    )

    print(
        "\nScientific role:"
    )

    print(
        "  Diagnostic matched free-generation comparison only."
    )

    print(
        "  This script DOES NOT train a new model."
    )

    print(
        "  This script DOES NOT freeze B2."
    )

    print(
        "  This script DOES NOT change any prior threshold."
    )

    print(
        "  Existing B1 and primary-B2 synthetic caches are reused."
    )

    print(
        "  Only lambda=0 control candidates are newly sampled when needed."
    )

    print(
        "\nComparison:"
    )

    print(
        "  Frozen B1 RAW10"
    )

    print(
        "  Matched lambda=0 RAW5 / RAW10 / RAW20"
    )

    print(
        "  Primary lambda=0.01 B2 RAW5 / RAW10 / RAW20"
    )

    print(
        "\nExact matched audit conditions:"
    )

    print(
        f"  subjects: {SELECTED_SUBJECT_COUNT}"
    )

    print(
        f"  Real windows/subject/state: {REAL_WINDOWS_PER_SUBJECT_STATE}"
    )

    print(
        f"  Synthetic windows/subject/state/model: "
        f"{SYNTHETIC_WINDOWS_PER_SUBJECT_STATE}"
    )

    print(
        f"  sampling seed base: {SAMPLING_SEED_BASE}"
    )

    print(
        "  sampler: 1000-step ancestral DDPM / v-prediction / FP32"
    )

    print(
        "  Final Test: SEALED / NOT ACCESSED"
    )

    for directory in (
        METRIC_DIR,
        GENERATED_DIR,
        FIGURE_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit all frozen source artifacts
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B1, PRIMARY B2, AND MATCHED CONTROL"
    )

    print(
        "=" * 80
    )

    required_files = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_REAL_REFERENCE_NPZ,
        B1_SYNTHETIC_REFERENCE_NPZ,
        PRIMARY_B2_RESULT,
        PRIMARY_B2_SWEEP_SUMMARY,
        PRIMARY_B2_SYNTHETIC_NPZ,
        CONTROL_RESULT,
    ]

    for epoch in CANDIDATE_EPOCHS:
        required_files.extend(
            [
                primary_candidate_path(
                    epoch
                ),
                control_candidate_path(
                    epoch
                ),
            ]
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

    with PRIMARY_B2_RESULT.open(
        "r",
        encoding="utf-8",
    ) as f:
        primary_result = json.load(
            f
        )

    with PRIMARY_B2_SWEEP_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        primary_sweep_summary = json.load(
            f
        )

    with CONTROL_RESULT.open(
        "r",
        encoding="utf-8",
    ) as f:
        control_result = json.load(
            f
        )

    if b1_selection[
        "status"
    ] != "B1_SELECTION_FROZEN":
        raise RuntimeError(
            "B1 selection is not formally frozen."
        )

    if int(
        b1_selection[
            "selected_epoch"
        ]
    ) != 10:
        raise RuntimeError(
            "Unexpected frozen B1 selected epoch."
        )

    if primary_result[
        "status"
    ] != "B2_FULL_FINETUNE_20_EPOCHS_COMPLETE":
        raise RuntimeError(
            "Primary lambda=0.01 B2 full run is not complete."
        )

    if abs(
        float(
            primary_result[
                "lambda_spectral"
            ]
        )
        - PRIMARY_LAMBDA
    ) > 1e-12:
        raise RuntimeError(
            "Primary B2 lambda mismatch."
        )

    if primary_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Primary B2 reports Final Test access."
        )

    if primary_sweep_summary[
        "status"
    ] != "B2_RAW_CANDIDATE_SUBJECT_AWARE_SPECTRAL_SWEEP_COMPLETE":
        raise RuntimeError(
            "Primary B2 free-generation sweep is not complete."
        )

    if primary_sweep_summary[
        "recommended_epoch"
    ] is not None:
        raise RuntimeError(
            "Primary B2 unexpectedly reports an eligible candidate."
        )

    if primary_sweep_summary[
        "diagnostic_direction"
    ] != "NO_ELIGIBLE_B2_RAW_CANDIDATE":
        raise RuntimeError(
            "Unexpected primary B2 failure diagnostic."
        )

    if control_result[
        "status"
    ] != "B2_MATCHED_LAMBDA0_CONTROL_20_EPOCHS_COMPLETE":
        raise RuntimeError(
            "Matched lambda=0 control is not complete."
        )

    if abs(
        float(
            control_result[
                "lambda_spectral"
            ]
        )
        - CONTROL_LAMBDA
    ) > 1e-12:
        raise RuntimeError(
            "Matched control lambda mismatch."
        )

    if control_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Matched control reports Final Test access."
        )

    frozen_b1_file_hash_before = sha256_file(
        FROZEN_B1_GENERATOR
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    frozen_b1_model_hash = state_dict_sha256(
        frozen_b1[
            "model_state_dict"
        ]
    )

    if (
        frozen_b1_model_hash
        != frozen_b1[
            "selected_model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B1 model-state hash mismatch."
        )

    source_hash_before = {
        "frozen_B1_generator": (
            frozen_b1_file_hash_before
        ),
        "B1_real_reference": (
            sha256_file(
                B1_REAL_REFERENCE_NPZ
            )
        ),
        "B1_synthetic_reference": (
            sha256_file(
                B1_SYNTHETIC_REFERENCE_NPZ
            )
        ),
        "primary_B2_synthetic_cache": (
            sha256_file(
                PRIMARY_B2_SYNTHETIC_NPZ
            )
        ),
    }

    primary_checkpoint_hash_before = {
        epoch: sha256_file(
            primary_candidate_path(
                epoch
            )
        )
        for epoch in CANDIDATE_EPOCHS
    }

    control_checkpoint_hash_before = {
        epoch: sha256_file(
            control_candidate_path(
                epoch
            )
        )
        for epoch in CANDIDATE_EPOCHS
    }

    print(
        f"frozen B1 SHA256: "
        f"{frozen_b1_file_hash_before}"
    )

    print(
        f"frozen B1 model-state SHA256: "
        f"{frozen_b1_model_hash}"
    )

    print(
        "source audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze s2_58 diagnostic protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE s2_58 JOINT DIAGNOSTIC PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_58_JOINT_FREEGEN_DIAGNOSTIC_PROTOCOL_FROZEN"
        ),
        "scientific_role": (
            "Matched diagnostic comparison separating observed "
            "continuation effects from incremental spectral-regularization "
            "effects under the exact existing Train-only free-generation audit."
        ),
        "variants": [
            "Frozen B1 RAW epoch 10",
            "Matched lambda=0 RAW epochs 5/10/20",
            "Primary lambda=0.01 B2 RAW epochs 5/10/20",
        ],
        "candidate_epochs": (
            CANDIDATE_EPOCHS
        ),
        "primary_lambda": (
            PRIMARY_LAMBDA
        ),
        "control_lambda": (
            CONTROL_LAMBDA
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
        "sampling": {
            "sampler": (
                "1000-step ancestral DDPM"
            ),
            "prediction_target": (
                "v"
            ),
            "mixed_precision": False,
            "sampling_seed_base": (
                SAMPLING_SEED_BASE
            ),
            "same_subject_order_across_all_variants": True,
            "same_sampling_seed_across_all_variants": True,
            "paired_EO_EC_noise_convention": True,
        },
        "cache_policy": {
            "frozen_B1": (
                "reuse exact B1 V2 synthetic cache"
            ),
            "primary_lambda001_B2": (
                "reuse exact s2_56 synthetic cache"
            ),
            "matched_lambda0_control": (
                "generate once per candidate and atomically cache; "
                "validate checkpoint hash/cohort/seed before reuse"
            ),
        },
        "primary_metric_definition_unchanged": (
            "mean EO/EC cohort-median 1-40 Hz Welch log-PSD MAE; "
            "Welch nperseg=256, noverlap=128, Hann, density"
        ),
        "generic_metric_helpers_unchanged": True,
        "extra_diagnostics": [
            "signed PSD error by state/channel/frequency",
            "state/channel offset-removed spectral shape error",
            "spectral-error tilt over log10 frequency",
            "pooled log-PSD IQR mismatch",
            "subject-level PSD MAE diagnostics",
            "paired subject bootstrap for candidate-minus-B1 primary metric",
            "matched delta decomposition: control-B1, spectral-control, spectral-B1",
            "training-monitor versus free-generation table",
        ],
        "bootstrap": {
            "role": (
                "diagnostic uncertainty estimate only; "
                "does not alter frozen thresholds"
            ),
            "replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "seed": (
                BOOTSTRAP_SEED
            ),
            "resampling_unit": (
                "subject; matched Real/B1/candidate resampling"
            ),
        },
        "interpretation": {
            "control_minus_B1": (
                "observed continuation effect at the matched free-generation audit"
            ),
            "spectral_minus_control": (
                "observed incremental effect associated with lambda=0.01 "
                "under the matched training design; still interpreted cautiously "
                "because AMP skip locations can differ"
            ),
            "spectral_minus_B1": (
                "net observed primary-B2 workflow effect"
            ),
        },
        "selection_policy": (
            "NO new final-model selection is made by s2_58. "
            "Existing B2 eligibility thresholds are recomputed only as a "
            "diagnostic consistency check; they are not changed."
        ),
        "Train_reference_used": True,
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "training_updates_executed": False,
        "bulk_synthetic_generation_started": False,
    }

    if PROTOCOL_FILE.is_file():
        with PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_protocol = json.load(
                f
            )

        if existing_protocol != protocol:
            raise RuntimeError(
                "Conflicting s2_58 protocol already exists."
            )

        print(
            "identical s2_58 protocol already exists."
        )
    else:
        atomic_json_save(
            protocol,
            PROTOCOL_FILE,
        )

        print(
            f"protocol: {PROTOCOL_FILE}"
        )

    # ========================================================
    # Step 3 — Load exact reference and cached comparator samples
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — LOAD EXACT B1 REAL/SYNTHETIC + PRIMARY B2 CACHES"
    )

    print(
        "=" * 80
    )

    (
        real_z,
        b1_synth_z,
        selected_subjects,
        selected_subject_ids,
    ) = load_exact_b1_reference()

    primary_synth_by_epoch = load_primary_b2_cache(
        expected_subject_indices=(
            selected_subjects
        ),
        expected_subject_ids=(
            selected_subject_ids
        ),
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
        "Primary B2 cached candidates: "
        f"{list(primary_synth_by_epoch.keys())}"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    # ========================================================
    # Step 4 — Build sampler and generate matched lambda=0 candidates
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — GENERATE / RESUME MATCHED lambda=0 FREE SAMPLES"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for matched control free-generation sampling."
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

    generation_start = time.perf_counter()

    control_synth_by_epoch = {}

    for epoch in CANDIDATE_EPOCHS:
        control_synth_by_epoch[
            epoch
        ] = load_or_generate_control_candidate(
            epoch=epoch,
            frozen_b1=frozen_b1,
            selected_subjects=(
                selected_subjects
            ),
            selected_subject_ids=(
                selected_subject_ids
            ),
            reverse_schedule=(
                reverse_schedule
            ),
            device=device,
        )

    control_generation_seconds = (
        time.perf_counter()
        - generation_start
    )

    print(
        f"control generation/resume wall time: "
        f"{control_generation_seconds/60:.2f} min"
    )

    # ========================================================
    # Step 5 — Compute exact existing metrics for all seven variants
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — COMPUTE EXACT EXISTING FREE-GENERATION METRICS"
    )

    print(
        "=" * 80
    )

    variant_synth = {
        "B1": (
            b1_synth_z
        ),
    }

    for epoch in CANDIDATE_EPOCHS:
        variant_synth[
            f"L0_E{epoch:03d}"
        ] = control_synth_by_epoch[
            epoch
        ]

        variant_synth[
            f"L001_E{epoch:03d}"
        ] = primary_synth_by_epoch[
            epoch
        ]

    bundles: dict[
        str,
        dict[str, Any],
    ] = {}

    for variant_name, synth_z in variant_synth.items():
        generic, matrices, _ = compute_candidate_metrics(
            real_z=real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

        full_psd = full_log_psd_mae(
            real_z=real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

        bundles[
            variant_name
        ] = {
            "generic": generic,
            "full_log_psd": (
                full_psd
            ),
            "matrices": (
                matrices
            ),
        }

        print(
            "\n"
            f"{variant_name}:"
        )

        print(
            f"  full PSD MAE: "
            f"{full_psd['mean_mae_db']:.6f} dB"
        )

        print(
            f"  mean bandpower MAE: "
            f"{generic['mean_bandpower_mae_db']:.6f} dB"
        )

        print(
            f"  alpha-effect MAE: "
            f"{generic['alpha_effect']['channel_mae_db']:.6f} dB"
        )

        print(
            "  scale EO/EC: "
            f"{generic['scale']['EO']['normalized_std_ratio']:.6f}"
            " / "
            f"{generic['scale']['EC']['normalized_std_ratio']:.6f}"
        )

        print(
            "  subject Top1/Top5: "
            f"{generic['subject_identity']['subject_retrieval_top1_accuracy']:.6f}"
            " / "
            f"{generic['subject_identity']['subject_retrieval_top5_accuracy']:.6f}"
        )

    b1_bundle = bundles[
        "B1"
    ]

    b1_baseline_for_guard = {
        "generic": (
            b1_bundle[
                "generic"
            ]
        ),
        "full_log_psd": (
            b1_bundle[
                "full_log_psd"
            ]
        ),
    }

    # Verify that the reused primary B2 cache reproduces the existing s2_56
    # conclusion rather than silently changing a metric helper/reference.
    for epoch in CANDIDATE_EPOCHS:
        key = (
            f"L001_E{epoch:03d}"
        )

        diagnostic_guard = evaluate_eligibility(
            candidate=(
                {
                    "generic": bundles[
                        key
                    ][
                        "generic"
                    ],
                    "full_log_psd": bundles[
                        key
                    ][
                        "full_log_psd"
                    ],
                }
            ),
            b1_baseline=(
                b1_baseline_for_guard
            ),
        )

        expected_guard = primary_sweep_summary[
            "eligibility_by_epoch"
        ][
            str(
                epoch
            )
        ]

        expected_psd = float(
            primary_sweep_summary[
                "metrics_by_epoch"
            ][
                str(
                    epoch
                )
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        actual_psd = float(
            bundles[
                key
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        if abs(
            expected_psd
            - actual_psd
        ) > 1e-10:
            raise RuntimeError(
                f"s2_58 failed to reproduce s2_56 full-PSD metric "
                f"at primary B2 epoch {epoch}."
            )

        if bool(
            diagnostic_guard[
                "eligible"
            ]
        ) != bool(
            expected_guard[
                "eligible"
            ]
        ):
            raise RuntimeError(
                f"s2_58 failed to reproduce s2_56 eligibility "
                f"at primary B2 epoch {epoch}."
            )

    print(
        "\nExisting s2_56 metric/eligibility reproduction: PASS"
    )

    # ========================================================
    # Step 6 — Add diagnostic spectral decompositions
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — SIGNED PSD / SHAPE / TILT / DISPERSION DIAGNOSTICS"
    )

    print(
        "=" * 80
    )

    frequencies, real_log_psd = log_psd_tensor(
        real_z,
        scaler,
    )

    spectral_diagnostics_by_variant = {}
    spectral_arrays_by_variant = {}
    subject_rows_all = []

    for variant_name, synth_z in variant_synth.items():
        synth_frequencies, synth_log_psd = log_psd_tensor(
            synth_z,
            scaler,
        )

        if not np.array_equal(
            frequencies,
            synth_frequencies,
        ):
            raise RuntimeError(
                f"Frequency-grid mismatch for {variant_name}."
            )

        (
            diagnostic,
            arrays,
            subject_rows,
        ) = spectral_diagnostics(
            frequencies=(
                frequencies
            ),
            real_log_psd=(
                real_log_psd
            ),
            synth_log_psd=(
                synth_log_psd
            ),
        )

        primary_helper_value = float(
            bundles[
                variant_name
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        if abs(
            diagnostic[
                "full_psd_mae_recomputed_db"
            ]
            - primary_helper_value
        ) > 1e-10:
            raise RuntimeError(
                f"Diagnostic PSD implementation disagrees with "
                f"the frozen helper for {variant_name}."
            )

        spectral_diagnostics_by_variant[
            variant_name
        ] = diagnostic

        spectral_arrays_by_variant[
            variant_name
        ] = arrays

        for row in subject_rows:
            local_subject_index = int(
                row[
                    "local_subject_index"
                ]
            )

            subject_rows_all.append(
                {
                    "variant": (
                        variant_name
                    ),
                    "local_subject_index": (
                        local_subject_index
                    ),
                    "subject_index": int(
                        selected_subjects[
                            local_subject_index
                        ]
                    ),
                    "subject_id": (
                        selected_subject_ids[
                            local_subject_index
                        ]
                    ),
                    "eye_state": (
                        row[
                            "eye_state"
                        ]
                    ),
                    "subject_level_psd_mae_db": (
                        row[
                            "subject_level_psd_mae_db"
                        ]
                    ),
                }
            )

        print(
            f"{variant_name}: "
            f"bias={diagnostic['global_signed_psd_bias_db']:+.4f} dB "
            f"| shape-MAE={diagnostic['shape_mae_after_state_channel_offset_db']:.4f} dB "
            f"| |tilt|={diagnostic['mean_abs_tilt_db_per_decade']:.4f} dB/dec "
            f"| IQR-MAE={diagnostic['pooled_iqr_mae_db']:.4f} dB"
        )

    save_csv_rows(
        subject_rows_all,
        SUBJECT_PSD_CSV,
    )

    # Save spectral arrays with explicit variant prefixes.
    spectral_npz_payload = {
        "frequencies_hz": (
            frequencies
        ),
        "channel_names": np.asarray(
            CHANNEL_NAMES,
            dtype="U8",
        ),
        "state_names": np.asarray(
            STATE_NAMES,
            dtype="U2",
        ),
    }

    for variant_name, arrays in spectral_arrays_by_variant.items():
        for key, value in arrays.items():
            spectral_npz_payload[
                f"{variant_name}_{key}"
            ] = value

    atomic_npz_save(
        SPECTRAL_DIAGNOSTICS_NPZ,
        **spectral_npz_payload,
    )

    # ========================================================
    # Step 7 — Paired subject bootstrap versus frozen B1
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — PAIRED SUBJECT BOOTSTRAP (DIAGNOSTIC ONLY)"
    )

    print(
        "=" * 80
    )

    bootstrap_indices = make_bootstrap_subject_indices()

    _, b1_log_psd = log_psd_tensor(
        b1_synth_z,
        scaler,
    )

    b1_bootstrap = bootstrap_primary_mae(
        real_log_psd=(
            real_log_psd
        ),
        synth_log_psd=(
            b1_log_psd
        ),
        subject_indices=(
            bootstrap_indices
        ),
    )

    bootstrap_rows = []

    for variant_name, synth_z in variant_synth.items():
        if variant_name == "B1":
            continue

        _, synth_log_psd = log_psd_tensor(
            synth_z,
            scaler,
        )

        candidate_bootstrap = bootstrap_primary_mae(
            real_log_psd=(
                real_log_psd
            ),
            synth_log_psd=(
                synth_log_psd
            ),
            subject_indices=(
                bootstrap_indices
            ),
        )

        summary = bootstrap_difference_summary(
            candidate_bootstrap=(
                candidate_bootstrap
            ),
            b1_bootstrap=(
                b1_bootstrap
            ),
        )

        bootstrap_rows.append(
            {
                "variant": (
                    variant_name
                ),
                "bootstrap_replicates": (
                    BOOTSTRAP_REPLICATES
                ),
                **summary,
            }
        )

        print(
            f"{variant_name}: "
            f"candidate-B1 mean={summary['candidate_minus_B1_mean_db']:+.4f} dB "
            f"| 95% CI=["
            f"{summary['candidate_minus_B1_ci95_low_db']:+.4f}, "
            f"{summary['candidate_minus_B1_ci95_high_db']:+.4f}]"
        )

    save_csv_rows(
        bootstrap_rows,
        BOOTSTRAP_CSV,
    )

    # ========================================================
    # Step 8 — Existing metrics table + frozen guard diagnostics
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — SAVE JOINT METRIC TABLE"
    )

    print(
        "=" * 80
    )

    metric_rows = []

    def append_metric_row(
        *,
        variant_name: str,
        arm: str,
        epoch: int | None,
    ) -> None:
        bundle = bundles[
            variant_name
        ]

        generic = bundle[
            "generic"
        ]

        subject = generic[
            "subject_identity"
        ]

        diagnostic = spectral_diagnostics_by_variant[
            variant_name
        ]

        if variant_name == "B1":
            guard = None
        else:
            guard = evaluate_eligibility(
                candidate={
                    "generic": (
                        generic
                    ),
                    "full_log_psd": (
                        bundle[
                            "full_log_psd"
                        ]
                    ),
                },
                b1_baseline=(
                    b1_baseline_for_guard
                ),
            )

        b1_psd = float(
            b1_bundle[
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        current_psd = float(
            bundle[
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        improvement_fraction = float(
            (
                b1_psd
                - current_psd
            )
            / b1_psd
        )

        metric_rows.append(
            {
                "variant": (
                    variant_name
                ),
                "arm": (
                    arm
                ),
                "epoch": (
                    ""
                    if epoch is None
                    else epoch
                ),
                "full_log_psd_mae_db": (
                    current_psd
                ),
                "full_log_psd_improvement_vs_B1_fraction": (
                    improvement_fraction
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
                "mean_channel_correlation_r": (
                    mean_corr_r(
                        generic
                    )
                ),
                "mean_alpha_dwpli2_r": (
                    mean_wpli_r(
                        generic
                    )
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
                "subject_fingerprint_rmse_mean": (
                    subject[
                        "subject_fingerprint_standardized_rmse_mean"
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
                "signed_psd_bias_db": (
                    diagnostic[
                        "global_signed_psd_bias_db"
                    ]
                ),
                "shape_mae_after_state_channel_offset_db": (
                    diagnostic[
                        "shape_mae_after_state_channel_offset_db"
                    ]
                ),
                "mean_abs_tilt_db_per_decade": (
                    diagnostic[
                        "mean_abs_tilt_db_per_decade"
                    ]
                ),
                "pooled_iqr_mae_db": (
                    diagnostic[
                        "pooled_iqr_mae_db"
                    ]
                ),
                "existing_B2_guard_all_pass_diagnostic": (
                    ""
                    if guard is None
                    else bool(
                        guard[
                            "eligible"
                        ]
                    )
                ),
            }
        )

    append_metric_row(
        variant_name="B1",
        arm="frozen_B1",
        epoch=None,
    )

    for epoch in CANDIDATE_EPOCHS:
        append_metric_row(
            variant_name=(
                f"L0_E{epoch:03d}"
            ),
            arm="matched_lambda0_control",
            epoch=epoch,
        )

        append_metric_row(
            variant_name=(
                f"L001_E{epoch:03d}"
            ),
            arm="primary_lambda001_B2",
            epoch=epoch,
        )

    save_csv_rows(
        metric_rows,
        METRICS_CSV,
    )

    print(
        f"joint metrics CSV: "
        f"{METRICS_CSV}"
    )

    # ========================================================
    # Step 9 — Matched delta decomposition
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — MATCHED DELTA DECOMPOSITION"
    )

    print(
        "=" * 80
    )

    b1_metric_view = variant_metric_view(
        b1_bundle
    )

    delta_rows = []

    for epoch in CANDIDATE_EPOCHS:
        control_key = (
            f"L0_E{epoch:03d}"
        )

        primary_key = (
            f"L001_E{epoch:03d}"
        )

        control_view = variant_metric_view(
            bundles[
                control_key
            ]
        )

        primary_view = variant_metric_view(
            bundles[
                primary_key
            ]
        )

        delta_rows.extend(
            delta_row(
                epoch=epoch,
                comparison=(
                    "control_minus_B1_observed_continuation_effect"
                ),
                reference_name="B1",
                candidate_name=(
                    control_key
                ),
                reference_metrics=(
                    b1_metric_view
                ),
                candidate_metrics=(
                    control_view
                ),
            )
        )

        delta_rows.extend(
            delta_row(
                epoch=epoch,
                comparison=(
                    "spectral_minus_control_incremental_lambda001_effect"
                ),
                reference_name=(
                    control_key
                ),
                candidate_name=(
                    primary_key
                ),
                reference_metrics=(
                    control_view
                ),
                candidate_metrics=(
                    primary_view
                ),
            )
        )

        delta_rows.extend(
            delta_row(
                epoch=epoch,
                comparison=(
                    "spectral_minus_B1_net_primary_B2_effect"
                ),
                reference_name="B1",
                candidate_name=(
                    primary_key
                ),
                reference_metrics=(
                    b1_metric_view
                ),
                candidate_metrics=(
                    primary_view
                ),
            )
        )

        # Console summary for the primary question.
        print(
            f"\nepoch {epoch}:"
        )

        print(
            "  full-PSD continuation effect "
            "(lambda0 - B1): "
            f"{control_view['full_psd_mae_db'] - b1_metric_view['full_psd_mae_db']:+.6f} dB"
        )

        print(
            "  full-PSD incremental spectral effect "
            "(lambda0.01 - lambda0): "
            f"{primary_view['full_psd_mae_db'] - control_view['full_psd_mae_db']:+.6f} dB"
        )

        print(
            "  full-PSD net primary-B2 effect "
            "(lambda0.01 - B1): "
            f"{primary_view['full_psd_mae_db'] - b1_metric_view['full_psd_mae_db']:+.6f} dB"
        )

    save_csv_rows(
        delta_rows,
        DELTAS_CSV,
    )

    print(
        f"\nmatched deltas CSV: "
        f"{DELTAS_CSV}"
    )

    # ========================================================
    # Step 10 — Join denoising monitors to free-generation results
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 10 — TRAINING-MONITOR vs FREE-GENERATION TABLE"
    )

    print(
        "=" * 80
    )

    training_generation_rows = []

    for epoch in CANDIDATE_EPOCHS:
        control_monitor = checkpoint_training_monitor(
            path=(
                control_candidate_path(
                    epoch
                )
            ),
            expected_status=(
                "B2_MATCHED_LAMBDA0_CONTROL_CHECKPOINT"
            ),
            expected_epoch=(
                epoch
            ),
            expected_lambda=(
                CONTROL_LAMBDA
            ),
        )

        primary_monitor = checkpoint_training_monitor(
            path=(
                primary_candidate_path(
                    epoch
                )
            ),
            expected_status=(
                "B2_FULL_FINETUNE_CHECKPOINT"
            ),
            expected_epoch=(
                epoch
            ),
            expected_lambda=(
                PRIMARY_LAMBDA
            ),
        )

        for arm, key, monitor in (
            (
                "matched_lambda0_control",
                f"L0_E{epoch:03d}",
                control_monitor,
            ),
            (
                "primary_lambda001_B2",
                f"L001_E{epoch:03d}",
                primary_monitor,
            ),
        ):
            bundle = bundles[
                key
            ]

            training_generation_rows.append(
                {
                    "arm": (
                        arm
                    ),
                    "epoch": (
                        epoch
                    ),
                    **monitor,
                    "freegen_full_psd_mae_db": (
                        bundle[
                            "full_log_psd"
                        ][
                            "mean_mae_db"
                        ]
                    ),
                    "freegen_mean_bandpower_mae_db": (
                        bundle[
                            "generic"
                        ][
                            "mean_bandpower_mae_db"
                        ]
                    ),
                    "freegen_alpha_effect_mae_db": (
                        bundle[
                            "generic"
                        ][
                            "alpha_effect"
                        ][
                            "channel_mae_db"
                        ]
                    ),
                    "freegen_subject_top1": (
                        bundle[
                            "generic"
                        ][
                            "subject_identity"
                        ][
                            "subject_retrieval_top1_accuracy"
                        ]
                    ),
                }
            )

    save_csv_rows(
        training_generation_rows,
        TRAINING_VS_GENERATION_CSV,
    )

    print(
        f"training-vs-generation CSV: "
        f"{TRAINING_VS_GENERATION_CSV}"
    )

    # ========================================================
    # Step 11 — Figures
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 11 — SAVE JOINT DIAGNOSTIC FIGURES"
    )

    print(
        "=" * 80
    )

    control_bundles = {
        epoch: bundles[
            f"L0_E{epoch:03d}"
        ]
        for epoch in CANDIDATE_EPOCHS
    }

    primary_bundles = {
        epoch: bundles[
            f"L001_E{epoch:03d}"
        ]
        for epoch in CANDIDATE_EPOCHS
    }

    save_joint_figures(
        b1_bundle=b1_bundle,
        control_bundles=(
            control_bundles
        ),
        primary_bundles=(
            primary_bundles
        ),
        spectral_arrays=(
            spectral_arrays_by_variant
        ),
        frequencies=(
            frequencies
        ),
    )

    for path in (
        JOINT_PSD_FIGURE,
        JOINT_GENERIC_FIGURE,
        JOINT_SUBJECT_FIGURE,
        PSD_ERROR_FIGURE,
    ):
        print(
            path
        )

    # ========================================================
    # Step 12 — Verify source immutability
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 12 — VERIFY SOURCE IMMUTABILITY"
    )

    print(
        "=" * 80
    )

    source_hash_after = {
        "frozen_B1_generator": (
            sha256_file(
                FROZEN_B1_GENERATOR
            )
        ),
        "B1_real_reference": (
            sha256_file(
                B1_REAL_REFERENCE_NPZ
            )
        ),
        "B1_synthetic_reference": (
            sha256_file(
                B1_SYNTHETIC_REFERENCE_NPZ
            )
        ),
        "primary_B2_synthetic_cache": (
            sha256_file(
                PRIMARY_B2_SYNTHETIC_NPZ
            )
        ),
    }

    if (
        source_hash_after
        != source_hash_before
    ):
        raise RuntimeError(
            "A frozen/reference source artifact changed during s2_58."
        )

    for epoch in CANDIDATE_EPOCHS:
        if (
            sha256_file(
                primary_candidate_path(
                    epoch
                )
            )
            != primary_checkpoint_hash_before[
                epoch
            ]
        ):
            raise RuntimeError(
                f"Primary B2 candidate epoch {epoch} changed during s2_58."
            )

        if (
            sha256_file(
                control_candidate_path(
                    epoch
                )
            )
            != control_checkpoint_hash_before[
                epoch
            ]
        ):
            raise RuntimeError(
                f"Matched-control candidate epoch {epoch} changed during s2_58."
            )

    print(
        "frozen/reference source immutability: PASS"
    )

    print(
        "primary B2 checkpoint immutability: PASS"
    )

    print(
        "matched-control checkpoint immutability: PASS"
    )

    # ========================================================
    # Step 13 — Final summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 13 — SAVE s2_58 FINAL SUMMARY"
    )

    print(
        "=" * 80
    )

    serializable_bundles = {}

    for variant_name, bundle in bundles.items():
        generic = bundle[
            "generic"
        ]

        serializable_bundles[
            variant_name
        ] = {
            "full_log_psd": (
                bundle[
                    "full_log_psd"
                ]
            ),
            "generic": (
                generic
            ),
            "spectral_diagnostics": (
                spectral_diagnostics_by_variant[
                    variant_name
                ]
            ),
        }

    bootstrap_by_variant = {
        row[
            "variant"
        ]: {
            key: value
            for key, value
            in row.items()
            if key
            != "variant"
        }
        for row in bootstrap_rows
    }

    matched_effects = {}

    for epoch in CANDIDATE_EPOCHS:
        control_key = (
            f"L0_E{epoch:03d}"
        )

        primary_key = (
            f"L001_E{epoch:03d}"
        )

        b1_view = variant_metric_view(
            bundles[
                "B1"
            ]
        )

        control_view = variant_metric_view(
            bundles[
                control_key
            ]
        )

        primary_view = variant_metric_view(
            bundles[
                primary_key
            ]
        )

        matched_effects[
            str(
                epoch
            )
        ] = {
            "control_minus_B1": {
                metric: float(
                    control_view[
                        metric
                    ]
                    - b1_view[
                        metric
                    ]
                )
                for metric in b1_view
            },
            "spectral_minus_control": {
                metric: float(
                    primary_view[
                        metric
                    ]
                    - control_view[
                        metric
                    ]
                )
                for metric in b1_view
            },
            "spectral_minus_B1": {
                metric: float(
                    primary_view[
                        metric
                    ]
                    - b1_view[
                        metric
                    ]
                )
                for metric in b1_view
            },
        }

    summary = {
        "status": (
            "S2_58_JOINT_B1_LAMBDA0_LAMBDA001_FREEGEN_AUDIT_COMPLETE"
        ),
        "scientific_role": (
            "Matched Train-only free-generation diagnostic; "
            "not Final Test, not a new B2 selection/freeze."
        ),
        "protocol": str(
            PROTOCOL_FILE
        ),
        "variants": (
            serializable_bundles
        ),
        "matched_effects_raw_candidate_minus_reference": (
            matched_effects
        ),
        "bootstrap_candidate_minus_B1": (
            bootstrap_by_variant
        ),
        "source_hashes": (
            source_hash_before
        ),
        "control_generation_or_resume_seconds": (
            control_generation_seconds
        ),
        "artifacts": {
            "joint_metrics_csv": str(
                METRICS_CSV
            ),
            "matched_deltas_csv": str(
                DELTAS_CSV
            ),
            "subject_psd_csv": str(
                SUBJECT_PSD_CSV
            ),
            "training_vs_generation_csv": str(
                TRAINING_VS_GENERATION_CSV
            ),
            "bootstrap_csv": str(
                BOOTSTRAP_CSV
            ),
            "spectral_diagnostics_npz": str(
                SPECTRAL_DIAGNOSTICS_NPZ
            ),
            "joint_psd_figure": str(
                JOINT_PSD_FIGURE
            ),
            "joint_generic_figure": str(
                JOINT_GENERIC_FIGURE
            ),
            "joint_subject_figure": str(
                JOINT_SUBJECT_FIGURE
            ),
            "signed_psd_error_figure": str(
                PSD_ERROR_FIGURE
            ),
        },
        "governance": {
            "Train_reference_used": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "training_updates_executed": False,
            "bulk_synthetic_generation_started": False,
            "B1_frozen_and_unchanged": True,
            "primary_B2_checkpoints_unchanged": True,
            "matched_control_checkpoints_unchanged": True,
            "primary_B2_negative_result_preserved": True,
            "new_B2_freeze_decision_made": False,
            "thresholds_changed": False,
        },
        "next": (
            "Interpret the matched free-generation effects. "
            "Only after interpretation should a redesigned B2 protocol "
            "or denoiser-side diagnostic stage be approved."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
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
        "s2_58 JOINT FREE-GENERATION AUDIT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nFrozen B1:"
    )

    print(
        "  full PSD MAE: "
        f"{bundles['B1']['full_log_psd']['mean_mae_db']:.6f} dB"
    )

    for epoch in CANDIDATE_EPOCHS:
        control_key = (
            f"L0_E{epoch:03d}"
        )

        primary_key = (
            f"L001_E{epoch:03d}"
        )

        control_psd = float(
            bundles[
                control_key
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        primary_psd = float(
            bundles[
                primary_key
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        b1_psd = float(
            bundles[
                "B1"
            ][
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        )

        print(
            "\n"
            f"epoch {epoch}:"
        )

        print(
            f"  lambda=0 control full PSD MAE: "
            f"{control_psd:.6f} dB"
        )

        print(
            f"  lambda=0.01 B2 full PSD MAE: "
            f"{primary_psd:.6f} dB"
        )

        print(
            f"  continuation effect (control-B1): "
            f"{control_psd - b1_psd:+.6f} dB"
        )

        print(
            f"  incremental spectral effect "
            f"(lambda0.01-control): "
            f"{primary_psd - control_psd:+.6f} dB"
        )

        print(
            f"  net primary-B2 effect "
            f"(lambda0.01-B1): "
            f"{primary_psd - b1_psd:+.6f} dB"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  No new model was trained."
    )

    print(
        "  No new B2 candidate was frozen."
    )

    print(
        "  Existing thresholds were NOT modified."
    )

    print(
        "  Primary B2 negative result remains preserved."
    )

    print(
        "  Bootstrap output is diagnostic uncertainty only."
    )

    print(
        "  Causal wording remains cautious because matched arms can "
        "have different AMP-skip locations despite matched design."
    )

    print(
        "\nValidation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Training updates executed: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "INTERPRET s2_58 MATCHED FREE-GENERATION EFFECTS "
        "BEFORE APPROVING ANY B2 REDESIGN"
    )


if __name__ == "__main__":
    main()
