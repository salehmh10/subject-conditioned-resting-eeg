from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    alpha_dwpli2_debiased,
    bandpower_db_from_psd,
    build_reverse_schedule,
    inverse_normalize,
    mean_covariance_and_correlation,
    relative_frobenius,
    upper_triangle_correlation,
    welch_psd,
)
from s2_47_b1_architecture_warmstart_sanity import (
    B1SubjectStateUNet1D,
)
from s2_48_b1_1000_step_subject_conditioned_pilot import (
    sample_b1_subjects,
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

B1_FULL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "full_finetune_v1"
)

B1_FULL_RESULT = (
    B1_FULL_ROOT
    / "metrics"
    / "b1_full_finetune_result_v1.json"
)

CHECKPOINT_DIR = (
    B1_FULL_ROOT
    / "checkpoints"
)


# ============================================================
# Subject-aware fidelity sweep outputs
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "raw_candidate_subject_aware_sweep_v2"
)

METRIC_DIR = RUN_ROOT / "metrics"
LOG_DIR = RUN_ROOT / "logs"
GENERATED_DIR = RUN_ROOT / "generated"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_FILE = (
    METRIC_DIR
    / "b1_raw_candidate_subject_aware_protocol_v2.json"
)

METRICS_CSV = (
    METRIC_DIR
    / "b1_raw_candidate_subject_aware_metrics_v2.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "b1_raw_candidate_subject_aware_summary_v2.json"
)

MATRICES_NPZ = (
    METRIC_DIR
    / "b1_raw_candidate_subject_aware_matrices_v2.npz"
)

FINGERPRINT_CSV = (
    METRIC_DIR
    / "b1_raw_candidate_subject_fingerprint_metrics_v2.csv"
)

REAL_REFERENCE_NPZ = (
    GENERATED_DIR
    / "b1_subject_aware_real_train_reference_v2.npz"
)

SYNTHETIC_NPZ = (
    GENERATED_DIR
    / "b1_raw_candidate_subject_aware_synthetic_v2.npz"
)

REAL_MANIFEST_CSV = (
    LOG_DIR
    / "b1_subject_aware_real_reference_manifest_v2.csv"
)

SCALE_FIGURE = (
    FIGURE_DIR
    / "b1_raw_candidate_scale_ratio_v2.png"
)

BANDPOWER_FIGURE = (
    FIGURE_DIR
    / "b1_raw_candidate_bandpower_mae_v2.png"
)

ALPHA_FIGURE = (
    FIGURE_DIR
    / "b1_raw_candidate_alpha_effect_v2.png"
)

SPATIAL_FIGURE = (
    FIGURE_DIR
    / "b1_raw_candidate_spatial_connectivity_v2.png"
)

SUBJECT_RETRIEVAL_FIGURE = (
    FIGURE_DIR
    / "b1_raw_candidate_subject_retrieval_v2.png"
)

SUBJECT_DISTANCE_FIGURE = (
    FIGURE_DIR
    / "b1_raw_candidate_subject_distance_v2.png"
)

OVERVIEW_FIGURE = (
    FIGURE_DIR
    / "b1_raw_candidate_overview_v2.png"
)


FAILED_V1_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "raw_candidate_subject_aware_sweep_v1"
)

FAILED_V1_PROTOCOL = (
    FAILED_V1_ROOT
    / "metrics"
    / "b1_raw_candidate_subject_aware_protocol_v1.json"
)

AMENDMENT_FILE = (
    METRIC_DIR
    / "b1_raw_candidate_subject_aware_protocol_amendment_v2.json"
)


# ============================================================
# Frozen audit design
# ============================================================

CANDIDATE_EPOCHS = [
    10,
    20,
    30,
]

EXPECTED_SUBJECTS = 198

SELECTED_SUBJECT_COUNT = 64

REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

SAMPLING_BATCH_SIZE = 64

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

TIMESTEPS = 1000
COSINE_S = 0.008

SAMPLING_SEED_BASE = 20261001

FIXED_WRONG_SUBJECT_OFFSET_WITHIN_AUDIT = 17

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

CHANNEL_INDEX = {
    name: index
    for index, name
    in enumerate(
        CHANNEL_NAMES
    )
}

BAND_NAMES = [
    "delta",
    "theta",
    "alpha",
    "beta",
]

STATE_NAMES = [
    "EO",
    "EC",
]

STATE_LABELS = {
    "EO": 0,
    "EC": 1,
}


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
# Deterministic Real Train subject reference
# ============================================================

def build_subject_block_groups(
    records,
) -> dict[
    tuple[int, int],
    dict[str, list[tuple[int, Any]]],
]:
    grouped: dict[
        tuple[int, int],
        dict[str, list[tuple[int, Any]]],
    ] = defaultdict(
        lambda: defaultdict(
            list
        )
    )

    for global_index, record in enumerate(
        records
    ):
        key = (
            int(
                record.subject_index
            ),
            int(
                record.label
            ),
        )

        grouped[
            key
        ][
            str(
                record.block_uid
            )
        ].append(
            (
                global_index,
                record,
            )
        )

    return grouped


def summarize_subject_block_inventory(
    grouped,
) -> dict[str, Any]:
    eo_counts = np.asarray(
        [
            len(
                grouped.get(
                    (
                        subject_index,
                        0,
                    ),
                    {},
                )
            )
            for subject_index in range(
                EXPECTED_SUBJECTS
            )
        ],
        dtype=np.int64,
    )

    ec_counts = np.asarray(
        [
            len(
                grouped.get(
                    (
                        subject_index,
                        1,
                    ),
                    {},
                )
            )
            for subject_index in range(
                EXPECTED_SUBJECTS
            )
        ],
        dtype=np.int64,
    )

    joint_eligible = (
        (
            eo_counts
            >= REAL_WINDOWS_PER_SUBJECT_STATE
        )
        & (
            ec_counts
            >= REAL_WINDOWS_PER_SUBJECT_STATE
        )
    )

    return {
        "EO_min": int(
            np.min(
                eo_counts
            )
        ),
        "EO_median": float(
            np.median(
                eo_counts
            )
        ),
        "EO_max": int(
            np.max(
                eo_counts
            )
        ),
        "EC_min": int(
            np.min(
                ec_counts
            )
        ),
        "EC_median": float(
            np.median(
                ec_counts
            )
        ),
        "EC_max": int(
            np.max(
                ec_counts
            )
        ),
        "joint_subjects_with_at_least_required_blocks": int(
            np.sum(
                joint_eligible
            )
        ),
        "required_distinct_blocks_per_state": int(
            REAL_WINDOWS_PER_SUBJECT_STATE
        ),
        "EO_counts": [
            int(
                value
            )
            for value in eo_counts
        ],
        "EC_counts": [
            int(
                value
            )
            for value in ec_counts
        ],
    }


def choose_selected_subjects(
    grouped,
) -> list[int]:
    eligible = []

    for subject_index in range(
        EXPECTED_SUBJECTS
    ):
        eo_blocks = grouped.get(
            (
                subject_index,
                0,
            ),
            {},
        )

        ec_blocks = grouped.get(
            (
                subject_index,
                1,
            ),
            {},
        )

        if (
            len(
                eo_blocks
            )
            >= REAL_WINDOWS_PER_SUBJECT_STATE
            and len(
                ec_blocks
            )
            >= REAL_WINDOWS_PER_SUBJECT_STATE
        ):
            eligible.append(
                subject_index
            )

    if len(
        eligible
    ) < SELECTED_SUBJECT_COUNT:
        raise RuntimeError(
            "Not enough subjects have the required "
            "number of distinct EO/EC Train blocks. "
            f"eligible={len(eligible)}"
        )

    positions = np.floor(
        np.arange(
            SELECTED_SUBJECT_COUNT,
            dtype=np.float64,
        )
        * len(
            eligible
        )
        / SELECTED_SUBJECT_COUNT
    ).astype(
        np.int64
    )

    selected = [
        int(
            eligible[
                int(
                    position
                )
            ]
        )
        for position in positions
    ]

    if len(
        set(
            selected
        )
    ) != SELECTED_SUBJECT_COUNT:
        raise RuntimeError(
            "Deterministic subject selection produced duplicates."
        )

    return selected


def select_one_central_window_per_block(
    block_map: dict[
        str,
        list[tuple[int, Any]],
    ],
    n_blocks: int,
) -> list[
    tuple[int, Any]
]:
    block_ids = sorted(
        block_map.keys()
    )

    if len(
        block_ids
    ) < n_blocks:
        raise RuntimeError(
            "Insufficient distinct blocks."
        )

    if len(
        block_ids
    ) == n_blocks:
        chosen_block_ids = (
            block_ids
        )
    else:
        positions = np.floor(
            np.arange(
                n_blocks,
                dtype=np.float64,
            )
            * len(
                block_ids
            )
            / n_blocks
        ).astype(
            np.int64
        )

        chosen_block_ids = [
            block_ids[
                int(
                    position
                )
            ]
            for position in positions
        ]

    selected = []

    for block_uid in chosen_block_ids:
        candidates = sorted(
            block_map[
                block_uid
            ],
            key=lambda pair: (
                int(
                    pair[
                        1
                    ].candidate_window_index
                ),
                int(
                    pair[
                        1
                    ].start_sample
                ),
            ),
        )

        chosen = candidates[
            len(
                candidates
            )
            // 2
        ]

        selected.append(
            chosen
        )

    return selected


def build_real_reference_selection(
    *,
    records,
    grouped,
    selected_subjects: list[int],
) -> tuple[
    list,
    list[dict[str, Any]],
]:
    selected_records = []
    manifest_rows = []

    for local_subject_index, subject_index in enumerate(
        selected_subjects
    ):
        for state_name in STATE_NAMES:
            label = STATE_LABELS[
                state_name
            ]

            chosen = (
                select_one_central_window_per_block(
                    grouped[
                        (
                            subject_index,
                            label,
                        )
                    ],
                    REAL_WINDOWS_PER_SUBJECT_STATE,
                )
            )

            for local_rep, (
                global_record_index,
                record,
            ) in enumerate(
                chosen
            ):
                del global_record_index

                selected_records.append(
                    record
                )

                manifest_rows.append(
                    {
                        "local_subject_index": (
                            local_subject_index
                        ),
                        "subject_index": (
                            subject_index
                        ),
                        "subject_id": (
                            record.subject_id
                        ),
                        "eye_state": (
                            state_name
                        ),
                        "label": (
                            label
                        ),
                        "reference_rep": (
                            local_rep
                        ),
                        "block_uid": (
                            record.block_uid
                        ),
                        "block_index": int(
                            record.block_index
                        ),
                        "candidate_window_index": int(
                            record.candidate_window_index
                        ),
                        "start_sample": int(
                            record.start_sample
                        ),
                    }
                )

    expected = (
        SELECTED_SUBJECT_COUNT
        * len(
            STATE_NAMES
        )
        * REAL_WINDOWS_PER_SUBJECT_STATE
    )

    if len(
        selected_records
    ) != expected:
        raise RuntimeError(
            "Real reference selection count mismatch."
        )

    return (
        selected_records,
        manifest_rows,
    )


def load_real_reference(
    *,
    selected_records,
    scaler: dict[str, Any],
) -> np.ndarray:
    dataset = (
        CachedRealTrainWindowDataset(
            records=selected_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    try:
        values = []

        for index in range(
            len(
                dataset
            )
        ):
            item = dataset[
                index
            ]

            x = (
                item[
                    "x"
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            values.append(
                x
            )

        array = np.stack(
            values,
            axis=0,
        )
    finally:
        dataset.close()

    expected_shape = (
        SELECTED_SUBJECT_COUNT,
        len(
            STATE_NAMES
        ),
        REAL_WINDOWS_PER_SUBJECT_STATE,
        N_CHANNELS,
        N_TIMES,
    )

    array = array.reshape(
        expected_shape
    )

    if not np.isfinite(
        array
    ).all():
        raise RuntimeError(
            "Real subject-aware reference contains NaN/Inf."
        )

    return array


# ============================================================
# Candidate loading / sampling
# ============================================================

def candidate_path(
    epoch: int,
) -> Path:
    return (
        CHECKPOINT_DIR
        / f"candidate_epoch_{epoch:03d}_raw.pt"
    )


def build_b1_model(
    *,
    checkpoint: dict[str, Any],
    frozen_b0: dict[str, Any],
    device: torch.device,
) -> B1SubjectStateUNet1D:
    protocol_snapshot = checkpoint[
        "B1_protocol_snapshot"
    ]

    model = B1SubjectStateUNet1D(
        b0_contract=(
            frozen_b0[
                "effective_contract"
            ]
        ),
        injection_module_name=str(
            checkpoint[
                "condition_injection_module"
            ]
        ),
        num_subjects=(
            EXPECTED_SUBJECTS
        ),
        subject_embedding_dim=128,
        condition_dim=256,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    if protocol_snapshot[
        "scientific_checkpoint_weights"
    ] != "RAW":
        raise RuntimeError(
            "Candidate checkpoint is not RAW."
        )

    if checkpoint[
        "EMA_enabled"
    ] is not False:
        raise RuntimeError(
            "Candidate unexpectedly enables EMA."
        )

    model.to(
        device
    )

    model.eval()

    return model


@torch.inference_mode()
def sample_b1_candidate_state(
    *,
    model: B1SubjectStateUNet1D,
    selected_subjects: list[int],
    eye_state_value: int,
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
    base_seed: int,
) -> np.ndarray:
    """
    Generate 4 windows per selected subject/state.

    The exact same subject order, chunking, and seeds are reused for
    every candidate checkpoint. The same base seed is also reused for
    EO and EC, providing a paired diagnostic at the noise level.
    """

    repeated_subjects = []

    for subject_index in selected_subjects:
        repeated_subjects.extend(
            [
                subject_index
            ]
            * SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
        )

    total = len(
        repeated_subjects
    )

    if (
        total
        != SELECTED_SUBJECT_COUNT
        * SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
    ):
        raise RuntimeError(
            "Synthetic subject-list count mismatch."
        )

    batches = []

    start = (
        time.perf_counter()
    )

    for batch_start in range(
        0,
        total,
        SAMPLING_BATCH_SIZE,
    ):
        batch_end = min(
            batch_start
            + SAMPLING_BATCH_SIZE,
            total,
        )

        batch_subjects = (
            repeated_subjects[
                batch_start:
                batch_end
            ]
        )

        batch_index = (
            batch_start
            // SAMPLING_BATCH_SIZE
        )

        batch = sample_b1_subjects(
            model=model,
            subject_indices=(
                batch_subjects
            ),
            eye_state_value=(
                eye_state_value
            ),
            reverse_schedule=(
                reverse_schedule
            ),
            device=device,
            seed=(
                base_seed
                + batch_index
                * 1009
            ),
        )

        batches.append(
            batch
        )

        generated = (
            batch_end
        )

        print(
            f"state={eye_state_value} "
            f"| batch={batch_index + 1}/"
            f"{math.ceil(total / SAMPLING_BATCH_SIZE)} "
            f"| generated={generated}/{total} "
            f"| batch_std={np.std(batch, dtype=np.float64):.4f} "
            f"| elapsed="
            f"{(time.perf_counter() - start)/60:.1f}m"
        )

    output = np.concatenate(
        batches,
        axis=0,
    )

    output = output.reshape(
        (
            SELECTED_SUBJECT_COUNT,
            SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
            N_CHANNELS,
            N_TIMES,
        )
    )

    if not np.isfinite(
        output
    ).all():
        raise RuntimeError(
            "Synthetic B1 candidate contains NaN/Inf."
        )

    return output


# ============================================================
# Feature / fidelity helpers
# ============================================================

def flatten_state_windows(
    array: np.ndarray,
    state_index: int,
) -> np.ndarray:
    """
    [subjects, states, reps, channels, time]
      -> [subjects*reps, channels, time]
    """

    value = array[
        :,
        state_index,
        ...,
    ]

    return value.reshape(
        (
            -1,
            N_CHANNELS,
            N_TIMES,
        )
    )


def flatten_synth_state_windows(
    array: np.ndarray,
    state_index: int,
) -> np.ndarray:
    """
    Synthetic has the same [S, 2, R, C, T] layout.
    """

    return flatten_state_windows(
        array,
        state_index,
    )


def bandpower_tensor_db(
    eeg_uv: np.ndarray,
) -> np.ndarray:
    """
    [N, C, T] -> [N, B, C]
    """

    frequencies, psd = welch_psd(
        eeg_uv
    )

    band_dict = bandpower_db_from_psd(
        frequencies,
        psd,
    )

    tensor = np.stack(
        [
            band_dict[
                band
            ]
            for band in BAND_NAMES
        ],
        axis=1,
    )

    return tensor.astype(
        np.float64,
        copy=False,
    )


def safe_row_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    x = a.astype(
        np.float64
    )

    y = b.astype(
        np.float64
    )

    if (
        np.std(
            x
        )
        <= 0.0
        or np.std(
            y
        )
        <= 0.0
    ):
        return float(
            "nan"
        )

    return float(
        np.corrcoef(
            x,
            y,
        )[
            0,
            1
        ]
    )


def subject_fingerprint_metrics(
    *,
    real_bandpower: np.ndarray,
    synth_bandpower: np.ndarray,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    np.ndarray,
]:
    """
    Inputs:
      real_bandpower  [S, 2, Rreal, B, C]
      synth_bandpower [S, 2, Rsyn,  B, C]

    Each subject fingerprint concatenates EO and EC median
    log-bandpower over all 4 bands and all 19 channels:
      2 × 4 × 19 = 152 dimensions.

    Distances are standardized using only the distribution of the
    Real Train subject fingerprints across the fixed audit cohort.
    """

    real_fingerprint = np.median(
        real_bandpower,
        axis=2,
    ).reshape(
        (
            SELECTED_SUBJECT_COUNT,
            -1,
        )
    )

    synth_fingerprint = np.median(
        synth_bandpower,
        axis=2,
    ).reshape(
        (
            SELECTED_SUBJECT_COUNT,
            -1,
        )
    )

    feature_mean = np.mean(
        real_fingerprint,
        axis=0,
        keepdims=True,
    )

    feature_std = np.std(
        real_fingerprint,
        axis=0,
        ddof=0,
        keepdims=True,
    )

    valid_features = (
        feature_std[
            0
        ]
        > 1e-8
    )

    if int(
        np.sum(
            valid_features
        )
    ) < 100:
        raise RuntimeError(
            "Too few informative subject-fingerprint features."
        )

    real_z = (
        real_fingerprint[
            :,
            valid_features
        ]
        - feature_mean[
            :,
            valid_features
        ]
    ) / feature_std[
        :,
        valid_features
    ]

    synth_z = (
        synth_fingerprint[
            :,
            valid_features
        ]
        - feature_mean[
            :,
            valid_features
        ]
    ) / feature_std[
        :,
        valid_features
    ]

    difference = (
        synth_z[
            :,
            None,
            :,
        ]
        - real_z[
            None,
            :,
            :,
        ]
    )

    distance_matrix = np.sqrt(
        np.mean(
            difference
            * difference,
            axis=-1,
        )
    )

    correct_distance = np.diag(
        distance_matrix
    )

    wrong_local_indices = (
        np.arange(
            SELECTED_SUBJECT_COUNT
        )
        + FIXED_WRONG_SUBJECT_OFFSET_WITHIN_AUDIT
    ) % SELECTED_SUBJECT_COUNT

    fixed_wrong_distance = distance_matrix[
        np.arange(
            SELECTED_SUBJECT_COUNT
        ),
        wrong_local_indices,
    ]

    wrong_mask = ~np.eye(
        SELECTED_SUBJECT_COUNT,
        dtype=bool,
    )

    wrong_values = distance_matrix[
        wrong_mask
    ].reshape(
        (
            SELECTED_SUBJECT_COUNT,
            SELECTED_SUBJECT_COUNT
            - 1,
        )
    )

    median_wrong_distance = np.median(
        wrong_values,
        axis=1,
    )

    margin = (
        median_wrong_distance
        - correct_distance
    )

    order = np.argsort(
        distance_matrix,
        axis=1,
    )

    target = np.arange(
        SELECTED_SUBJECT_COUNT
    )

    top1 = float(
        np.mean(
            order[
                :,
                0
            ]
            == target
        )
    )

    top5 = float(
        np.mean(
            [
                target[
                    index
                ]
                in order[
                    index,
                    :5
                ]
                for index in range(
                    SELECTED_SUBJECT_COUNT
                )
            ]
        )
    )

    correct_better_than_fixed_wrong = float(
        np.mean(
            correct_distance
            < fixed_wrong_distance
        )
    )

    subject_correlations = np.asarray(
        [
            safe_row_correlation(
                synth_fingerprint[
                    index
                ],
                real_fingerprint[
                    index
                ],
            )
            for index in range(
                SELECTED_SUBJECT_COUNT
            )
        ],
        dtype=np.float64,
    )

    per_subject_rows = []

    for local_subject_index in range(
        SELECTED_SUBJECT_COUNT
    ):
        per_subject_rows.append(
            {
                "local_subject_index": (
                    local_subject_index
                ),
                "correct_standardized_distance": float(
                    correct_distance[
                        local_subject_index
                    ]
                ),
                "fixed_wrong_standardized_distance": float(
                    fixed_wrong_distance[
                        local_subject_index
                    ]
                ),
                "median_wrong_standardized_distance": float(
                    median_wrong_distance[
                        local_subject_index
                    ]
                ),
                "median_wrong_minus_correct_margin": float(
                    margin[
                        local_subject_index
                    ]
                ),
                "fingerprint_pearson_r": float(
                    subject_correlations[
                        local_subject_index
                    ]
                ),
                "top1_retrieved_local_subject": int(
                    order[
                        local_subject_index,
                        0
                    ]
                ),
                "correct_is_top1": bool(
                    order[
                        local_subject_index,
                        0
                    ]
                    == local_subject_index
                ),
                "correct_is_top5": bool(
                    local_subject_index
                    in order[
                        local_subject_index,
                        :5
                    ]
                ),
            }
        )

    metrics = {
        "subject_fingerprint_standardized_rmse_mean": float(
            np.mean(
                correct_distance
            )
        ),
        "subject_fingerprint_standardized_rmse_median": float(
            np.median(
                correct_distance
            )
        ),
        "subject_correct_vs_median_wrong_margin_mean": float(
            np.mean(
                margin
            )
        ),
        "subject_correct_vs_median_wrong_margin_median": float(
            np.median(
                margin
            )
        ),
        "subject_correct_better_than_fixed_wrong_fraction": (
            correct_better_than_fixed_wrong
        ),
        "subject_retrieval_top1_accuracy": (
            top1
        ),
        "subject_retrieval_top5_accuracy": (
            top5
        ),
        "subject_retrieval_top1_chance": float(
            1.0
            / SELECTED_SUBJECT_COUNT
        ),
        "subject_retrieval_top5_chance": float(
            5.0
            / SELECTED_SUBJECT_COUNT
        ),
        "subject_fingerprint_pearson_r_median": float(
            np.nanmedian(
                subject_correlations
            )
        ),
        "subject_fingerprint_pearson_r_mean": float(
            np.nanmean(
                subject_correlations
            )
        ),
        "subject_fingerprint_valid_feature_count": int(
            np.sum(
                valid_features
            )
        ),
    }

    return (
        metrics,
        per_subject_rows,
        distance_matrix,
    )


def compute_candidate_metrics(
    *,
    real_z: np.ndarray,
    synth_z: np.ndarray,
    scaler: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    """
    real_z:
      [S, 2, Rreal, C, T]

    synth_z:
      [S, 2, Rsyn, C, T]
    """

    real_uv = inverse_normalize(
        real_z.reshape(
            (
                -1,
                N_CHANNELS,
                N_TIMES,
            )
        ),
        scaler,
    ).reshape(
        real_z.shape
    )

    synth_uv = inverse_normalize(
        synth_z.reshape(
            (
                -1,
                N_CHANNELS,
                N_TIMES,
            )
        ),
        scaler,
    ).reshape(
        synth_z.shape
    )

    # --------------------------------------------------------
    # Scale fidelity
    # --------------------------------------------------------

    pooled_scale = {}

    subject_scale_ratios = {}

    for state_index, state_name in enumerate(
        STATE_NAMES
    ):
        real_state_z = flatten_state_windows(
            real_z,
            state_index,
        )

        synth_state_z = flatten_synth_state_windows(
            synth_z,
            state_index,
        )

        real_state_uv = flatten_state_windows(
            real_uv,
            state_index,
        )

        synth_state_uv = flatten_synth_state_windows(
            synth_uv,
            state_index,
        )

        real_norm_std = float(
            np.std(
                real_state_z,
                dtype=np.float64,
            )
        )

        synth_norm_std = float(
            np.std(
                synth_state_z,
                dtype=np.float64,
            )
        )

        real_uv_std = float(
            np.std(
                real_state_uv,
                dtype=np.float64,
            )
        )

        synth_uv_std = float(
            np.std(
                synth_state_uv,
                dtype=np.float64,
            )
        )

        pooled_scale[
            state_name
        ] = {
            "real_normalized_std": (
                real_norm_std
            ),
            "synthetic_normalized_std": (
                synth_norm_std
            ),
            "normalized_std_ratio": float(
                synth_norm_std
                / real_norm_std
            ),
            "real_uv_std": (
                real_uv_std
            ),
            "synthetic_uv_std": (
                synth_uv_std
            ),
            "uv_std_ratio": float(
                synth_uv_std
                / real_uv_std
            ),
        }

        real_subject_std = np.asarray(
            [
                np.std(
                    real_z[
                        subject_index,
                        state_index,
                    ],
                    dtype=np.float64,
                )
                for subject_index in range(
                    SELECTED_SUBJECT_COUNT
                )
            ],
            dtype=np.float64,
        )

        synth_subject_std = np.asarray(
            [
                np.std(
                    synth_z[
                        subject_index,
                        state_index,
                    ],
                    dtype=np.float64,
                )
                for subject_index in range(
                    SELECTED_SUBJECT_COUNT
                )
            ],
            dtype=np.float64,
        )

        ratio = (
            synth_subject_std
            / real_subject_std
        )

        subject_scale_ratios[
            state_name
        ] = ratio

    scale_error = float(
        np.mean(
            [
                abs(
                    pooled_scale[
                        state_name
                    ][
                        "normalized_std_ratio"
                    ]
                    - 1.0
                )
                for state_name in STATE_NAMES
            ]
        )
    )

    # --------------------------------------------------------
    # Bandpower
    # --------------------------------------------------------

    real_bandpower = np.empty(
        (
            SELECTED_SUBJECT_COUNT,
            2,
            REAL_WINDOWS_PER_SUBJECT_STATE,
            len(
                BAND_NAMES
            ),
            N_CHANNELS,
        ),
        dtype=np.float64,
    )

    synth_bandpower = np.empty(
        (
            SELECTED_SUBJECT_COUNT,
            2,
            SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
            len(
                BAND_NAMES
            ),
            N_CHANNELS,
        ),
        dtype=np.float64,
    )

    bandpower_mae_by_band = {}

    real_cohort_median_bandpower = np.empty(
        (
            2,
            len(
                BAND_NAMES
            ),
            N_CHANNELS,
        ),
        dtype=np.float64,
    )

    synth_cohort_median_bandpower = np.empty_like(
        real_cohort_median_bandpower
    )

    for state_index, state_name in enumerate(
        STATE_NAMES
    ):
        real_state_uv = flatten_state_windows(
            real_uv,
            state_index,
        )

        synth_state_uv = flatten_synth_state_windows(
            synth_uv,
            state_index,
        )

        real_state_bandpower = bandpower_tensor_db(
            real_state_uv
        )

        synth_state_bandpower = bandpower_tensor_db(
            synth_state_uv
        )

        real_bandpower[
            :,
            state_index,
            ...,
        ] = real_state_bandpower.reshape(
            (
                SELECTED_SUBJECT_COUNT,
                REAL_WINDOWS_PER_SUBJECT_STATE,
                len(
                    BAND_NAMES
                ),
                N_CHANNELS,
            )
        )

        synth_bandpower[
            :,
            state_index,
            ...,
        ] = synth_state_bandpower.reshape(
            (
                SELECTED_SUBJECT_COUNT,
                SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
                len(
                    BAND_NAMES
                ),
                N_CHANNELS,
            )
        )

        real_cohort_median_bandpower[
            state_index
        ] = np.median(
            real_state_bandpower,
            axis=0,
        )

        synth_cohort_median_bandpower[
            state_index
        ] = np.median(
            synth_state_bandpower,
            axis=0,
        )

    absolute_bandpower_error = np.abs(
        synth_cohort_median_bandpower
        - real_cohort_median_bandpower
    )

    for band_index, band_name in enumerate(
        BAND_NAMES
    ):
        bandpower_mae_by_band[
            band_name
        ] = float(
            np.mean(
                absolute_bandpower_error[
                    :,
                    band_index,
                    :,
                ]
            )
        )

    mean_bandpower_mae = float(
        np.mean(
            list(
                bandpower_mae_by_band.values()
            )
        )
    )

    # --------------------------------------------------------
    # Subject-macro alpha effect
    # --------------------------------------------------------

    alpha_index = BAND_NAMES.index(
        "alpha"
    )

    real_subject_alpha = np.median(
        real_bandpower[
            :,
            :,
            :,
            alpha_index,
            :,
        ],
        axis=2,
    )

    synth_subject_alpha = np.median(
        synth_bandpower[
            :,
            :,
            :,
            alpha_index,
            :,
        ],
        axis=2,
    )

    real_subject_alpha_effect = (
        real_subject_alpha[
            :,
            1,
            :,
        ]
        - real_subject_alpha[
            :,
            0,
            :,
        ]
    )

    synth_subject_alpha_effect = (
        synth_subject_alpha[
            :,
            1,
            :,
        ]
        - synth_subject_alpha[
            :,
            0,
            :,
        ]
    )

    real_alpha_effect_median = np.median(
        real_subject_alpha_effect,
        axis=0,
    )

    synth_alpha_effect_median = np.median(
        synth_subject_alpha_effect,
        axis=0,
    )

    alpha_effect_mae = float(
        np.mean(
            np.abs(
                synth_alpha_effect_median
                - real_alpha_effect_median
            )
        )
    )

    # --------------------------------------------------------
    # Spatial correlation + alpha dwPLI2
    # --------------------------------------------------------

    spatial = {}

    matrices = {}

    for state_index, state_name in enumerate(
        STATE_NAMES
    ):
        real_state_z = flatten_state_windows(
            real_z,
            state_index,
        )

        synth_state_z = flatten_synth_state_windows(
            synth_z,
            state_index,
        )

        real_state_uv = flatten_state_windows(
            real_uv,
            state_index,
        )

        synth_state_uv = flatten_synth_state_windows(
            synth_uv,
            state_index,
        )

        (
            real_cov,
            real_corr,
        ) = mean_covariance_and_correlation(
            real_state_z
        )

        (
            synth_cov,
            synth_corr,
        ) = mean_covariance_and_correlation(
            synth_state_z
        )

        real_wpli = alpha_dwpli2_debiased(
            real_state_uv
        )

        synth_wpli = alpha_dwpli2_debiased(
            synth_state_uv
        )

        correlation_r = upper_triangle_correlation(
            real_corr,
            synth_corr,
        )

        correlation_relF = relative_frobenius(
            real_corr,
            synth_corr,
        )

        wpli_r = upper_triangle_correlation(
            real_wpli,
            synth_wpli,
        )

        wpli_relF = relative_frobenius(
            real_wpli,
            synth_wpli,
        )

        spatial[
            state_name
        ] = {
            "correlation_r": float(
                correlation_r
            ),
            "correlation_relF": float(
                correlation_relF
            ),
            "alpha_dwpli2_r": float(
                wpli_r
            ),
            "alpha_dwpli2_relF": float(
                wpli_relF
            ),
        }

        matrices[
            f"real_corr_{state_name}"
        ] = real_corr

        matrices[
            f"synth_corr_{state_name}"
        ] = synth_corr

        matrices[
            f"real_dwpli2_{state_name}"
        ] = real_wpli

        matrices[
            f"synth_dwpli2_{state_name}"
        ] = synth_wpli

    mean_spatial_correlation_r = float(
        np.nanmean(
            [
                spatial[
                    state_name
                ][
                    "correlation_r"
                ]
                for state_name in STATE_NAMES
            ]
        )
    )

    mean_dwpli2_r = float(
        np.nanmean(
            [
                spatial[
                    state_name
                ][
                    "alpha_dwpli2_r"
                ]
                for state_name in STATE_NAMES
            ]
        )
    )

    # --------------------------------------------------------
    # Subject identity / fingerprint
    # --------------------------------------------------------

    (
        subject_metrics,
        subject_rows,
        subject_distance_matrix,
    ) = subject_fingerprint_metrics(
        real_bandpower=(
            real_bandpower
        ),
        synth_bandpower=(
            synth_bandpower
        ),
    )

    matrices[
        "subject_distance_matrix"
    ] = subject_distance_matrix

    # --------------------------------------------------------
    # Final metric package
    # --------------------------------------------------------

    metrics = {
        "scale": (
            pooled_scale
        ),
        "scale_error_mean_abs_ratio_minus_1": (
            scale_error
        ),
        "subject_scale_ratio_median": {
            state_name: float(
                np.median(
                    subject_scale_ratios[
                        state_name
                    ]
                )
            )
            for state_name in STATE_NAMES
        },
        "bandpower_mae_db": (
            bandpower_mae_by_band
        ),
        "mean_bandpower_mae_db": (
            mean_bandpower_mae
        ),
        "alpha_effect": {
            "channel_mae_db": (
                alpha_effect_mae
            ),
            "real_whole_head_median_db": float(
                np.median(
                    real_alpha_effect_median
                )
            ),
            "synthetic_whole_head_median_db": float(
                np.median(
                    synth_alpha_effect_median
                )
            ),
            "real_O1_db": float(
                real_alpha_effect_median[
                    CHANNEL_INDEX[
                        "O1"
                    ]
                ]
            ),
            "synthetic_O1_db": float(
                synth_alpha_effect_median[
                    CHANNEL_INDEX[
                        "O1"
                    ]
                ]
            ),
            "real_O2_db": float(
                real_alpha_effect_median[
                    CHANNEL_INDEX[
                        "O2"
                    ]
                ]
            ),
            "synthetic_O2_db": float(
                synth_alpha_effect_median[
                    CHANNEL_INDEX[
                        "O2"
                    ]
                ]
            ),
            "real_by_channel_db": [
                float(
                    value
                )
                for value in real_alpha_effect_median
            ],
            "synthetic_by_channel_db": [
                float(
                    value
                )
                for value in synth_alpha_effect_median
            ],
        },
        "spatial": (
            spatial
        ),
        "mean_spatial_correlation_r": (
            mean_spatial_correlation_r
        ),
        "mean_alpha_dwpli2_r": (
            mean_dwpli2_r
        ),
        "subject_identity": (
            subject_metrics
        ),
    }

    matrices[
        "real_cohort_median_bandpower_db"
    ] = real_cohort_median_bandpower

    matrices[
        "synth_cohort_median_bandpower_db"
    ] = synth_cohort_median_bandpower

    matrices[
        "real_alpha_effect_median_db"
    ] = real_alpha_effect_median

    matrices[
        "synth_alpha_effect_median_db"
    ] = synth_alpha_effect_median

    return (
        metrics,
        matrices,
        subject_rows,
    )


# ============================================================
# Selection rule
# ============================================================

GENERIC_CRITERIA = {
    "scale_fidelity": (
        "scale_error_mean_abs_ratio_minus_1",
        "min",
    ),
    "mean_bandpower_fidelity": (
        "mean_bandpower_mae_db",
        "min",
    ),
    "alpha_effect_fidelity": (
        "alpha_effect.channel_mae_db",
        "min",
    ),
    "spatial_correlation_fidelity": (
        "mean_spatial_correlation_r",
        "max",
    ),
    "alpha_dwpli2_fidelity": (
        "mean_alpha_dwpli2_r",
        "max",
    ),
}

SUBJECT_CRITERIA = {
    "subject_fingerprint_distance": (
        "subject_identity.subject_fingerprint_standardized_rmse_mean",
        "min",
    ),
    "subject_fingerprint_correlation": (
        "subject_identity.subject_fingerprint_pearson_r_median",
        "max",
    ),
    "subject_distance_margin": (
        "subject_identity.subject_correct_vs_median_wrong_margin_mean",
        "max",
    ),
    "subject_top1_retrieval": (
        "subject_identity.subject_retrieval_top1_accuracy",
        "max",
    ),
    "subject_fixed_wrong_win_fraction": (
        "subject_identity.subject_correct_better_than_fixed_wrong_fraction",
        "max",
    ),
}


def nested_get(
    data: dict[str, Any],
    key: str,
) -> float:
    current: Any = data

    for part in key.split(
        "."
    ):
        current = current[
            part
        ]

    return float(
        current
    )


def choose_winners(
    all_metrics: dict[
        int,
        dict[str, Any],
    ],
    criteria: dict[
        str,
        tuple[str, str],
    ],
) -> tuple[
    dict[str, int],
    dict[int, int],
]:
    winners = {}

    win_counts = {
        epoch: 0
        for epoch in CANDIDATE_EPOCHS
    }

    for criterion_name, (
        key,
        direction,
    ) in criteria.items():
        values = {
            epoch: nested_get(
                all_metrics[
                    epoch
                ],
                key,
            )
            for epoch in CANDIDATE_EPOCHS
        }

        finite_values = {
            epoch: value
            for epoch, value
            in values.items()
            if np.isfinite(
                value
            )
        }

        if not finite_values:
            raise RuntimeError(
                f"No finite values for criterion "
                f"{criterion_name}."
            )

        if direction == "min":
            winner = min(
                finite_values,
                key=finite_values.get,
            )
        elif direction == "max":
            winner = max(
                finite_values,
                key=finite_values.get,
            )
        else:
            raise ValueError(
                direction
            )

        winners[
            criterion_name
        ] = int(
            winner
        )

        win_counts[
            winner
        ] += 1

    return (
        winners,
        win_counts,
    )


# ============================================================
# Figures
# ============================================================

def save_scale_figure(
    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ],
) -> None:
    labels = [
        f"E{epoch}"
        for epoch in CANDIDATE_EPOCHS
    ]

    eo = [
        metrics_by_epoch[
            epoch
        ][
            "scale"
        ][
            "EO"
        ][
            "normalized_std_ratio"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    ec = [
        metrics_by_epoch[
            epoch
        ][
            "scale"
        ][
            "EC"
        ][
            "normalized_std_ratio"
        ]
        for epoch in CANDIDATE_EPOCHS
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
        label="Real ratio = 1",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Synthetic / Real normalized std"
    )

    ax.set_title(
        "B1 RAW Candidates — Scale Fidelity"
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


def save_bandpower_figure(
    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ],
) -> None:
    x = np.arange(
        len(
            BAND_NAMES
        )
    )

    width = (
        0.8
        / len(
            CANDIDATE_EPOCHS
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    for candidate_index, epoch in enumerate(
        CANDIDATE_EPOCHS
    ):
        values = [
            metrics_by_epoch[
                epoch
            ][
                "bandpower_mae_db"
            ][
                band
            ]
            for band in BAND_NAMES
        ]

        offset = (
            candidate_index
            - (
                len(
                    CANDIDATE_EPOCHS
                )
                - 1
            )
            / 2
        ) * width

        ax.bar(
            x
            + offset,
            values,
            width,
            label=f"Epoch {epoch}",
        )

    ax.set_xticks(
        x,
        BAND_NAMES,
    )

    ax.set_ylabel(
        "Median bandpower MAE (dB)"
    )

    ax.set_title(
        "B1 RAW Candidates — Bandpower Fidelity"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        BANDPOWER_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_alpha_figure(
    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ],
) -> None:
    real_o1 = metrics_by_epoch[
        CANDIDATE_EPOCHS[
            0
        ]
    ][
        "alpha_effect"
    ][
        "real_O1_db"
    ]

    real_o2 = metrics_by_epoch[
        CANDIDATE_EPOCHS[
            0
        ]
    ][
        "alpha_effect"
    ][
        "real_O2_db"
    ]

    labels = [
        "Real",
        *[
            f"E{epoch}"
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    o1 = [
        real_o1,
        *[
            metrics_by_epoch[
                epoch
            ][
                "alpha_effect"
            ][
                "synthetic_O1_db"
            ]
            for epoch in CANDIDATE_EPOCHS
        ],
    ]

    o2 = [
        real_o2,
        *[
            metrics_by_epoch[
                epoch
            ][
                "alpha_effect"
            ][
                "synthetic_O2_db"
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

    ax.set_ylabel(
        "Subject-macro median EC - EO alpha power (dB)"
    )

    ax.set_title(
        "B1 RAW Candidates — Occipital Alpha State Effect"
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


def save_spatial_figure(
    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ],
) -> None:
    labels = [
        f"E{epoch}"
        for epoch in CANDIDATE_EPOCHS
    ]

    corr = [
        metrics_by_epoch[
            epoch
        ][
            "mean_spatial_correlation_r"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    wpli = [
        metrics_by_epoch[
            epoch
        ][
            "mean_alpha_dwpli2_r"
        ]
        for epoch in CANDIDATE_EPOCHS
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
        corr,
        width,
        label="Mean channel-correlation r",
    )

    ax.bar(
        x
        + width
        / 2,
        wpli,
        width,
        label="Mean alpha dwPLI2 r",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Real-vs-Synthetic upper-triangle r"
    )

    ax.set_title(
        "B1 RAW Candidates — Spatial / Connectivity Fidelity"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SPATIAL_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_subject_retrieval_figure(
    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ],
) -> None:
    labels = [
        f"E{epoch}"
        for epoch in CANDIDATE_EPOCHS
    ]

    top1 = [
        metrics_by_epoch[
            epoch
        ][
            "subject_identity"
        ][
            "subject_retrieval_top1_accuracy"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    top5 = [
        metrics_by_epoch[
            epoch
        ][
            "subject_identity"
        ][
            "subject_retrieval_top5_accuracy"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    fixed_wrong = [
        metrics_by_epoch[
            epoch
        ][
            "subject_identity"
        ][
            "subject_correct_better_than_fixed_wrong_fraction"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    x = np.arange(
        len(
            labels
        )
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.bar(
        x
        - width,
        top1,
        width,
        label="Top-1 retrieval",
    )

    ax.bar(
        x,
        top5,
        width,
        label="Top-5 retrieval",
    )

    ax.bar(
        x
        + width,
        fixed_wrong,
        width,
        label="Correct closer than fixed wrong",
    )

    ax.axhline(
        1.0
        / SELECTED_SUBJECT_COUNT,
        linestyle="--",
        linewidth=1,
        label="Top-1 chance",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "Fraction"
    )

    ax.set_title(
        "B1 RAW Candidates — Subject Identity Retrieval"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SUBJECT_RETRIEVAL_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_subject_distance_figure(
    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ],
) -> None:
    labels = [
        f"E{epoch}"
        for epoch in CANDIDATE_EPOCHS
    ]

    correct_distance = [
        metrics_by_epoch[
            epoch
        ][
            "subject_identity"
        ][
            "subject_fingerprint_standardized_rmse_mean"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    margin = [
        metrics_by_epoch[
            epoch
        ][
            "subject_identity"
        ][
            "subject_correct_vs_median_wrong_margin_mean"
        ]
        for epoch in CANDIDATE_EPOCHS
    ]

    x = np.arange(
        len(
            labels
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.plot(
        x,
        correct_distance,
        marker="o",
        label="Correct-subject standardized RMSE (lower)",
    )

    ax.plot(
        x,
        margin,
        marker="o",
        label="Median-wrong minus correct margin (higher)",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_title(
        "B1 RAW Candidates — Subject Fingerprint Distance"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SUBJECT_DISTANCE_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_overview_figure(
    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ],
) -> None:
    rows = [
        "Scale error ↓",
        "Mean bandpower MAE ↓",
        "Alpha-effect MAE ↓",
        "Spatial r ↑",
        "Alpha dwPLI2 r ↑",
        "Subject fingerprint RMSE ↓",
        "Subject fingerprint r ↑",
        "Subject margin ↑",
        "Top-1 retrieval ↑",
        "Fixed-wrong win ↑",
    ]

    values = []

    for epoch in CANDIDATE_EPOCHS:
        metrics = metrics_by_epoch[
            epoch
        ]

        values.append(
            [
                metrics[
                    "scale_error_mean_abs_ratio_minus_1"
                ],
                metrics[
                    "mean_bandpower_mae_db"
                ],
                metrics[
                    "alpha_effect"
                ][
                    "channel_mae_db"
                ],
                metrics[
                    "mean_spatial_correlation_r"
                ],
                metrics[
                    "mean_alpha_dwpli2_r"
                ],
                metrics[
                    "subject_identity"
                ][
                    "subject_fingerprint_standardized_rmse_mean"
                ],
                metrics[
                    "subject_identity"
                ][
                    "subject_fingerprint_pearson_r_median"
                ],
                metrics[
                    "subject_identity"
                ][
                    "subject_correct_vs_median_wrong_margin_mean"
                ],
                metrics[
                    "subject_identity"
                ][
                    "subject_retrieval_top1_accuracy"
                ],
                metrics[
                    "subject_identity"
                ][
                    "subject_correct_better_than_fixed_wrong_fraction"
                ],
            ]
        )

    matrix = np.asarray(
        values,
        dtype=np.float64,
    ).T

    # Row-wise min-max normalization only for visualization.
    normalized = np.empty_like(
        matrix
    )

    lower_is_better = {
        0,
        1,
        2,
        5,
    }

    for row_index in range(
        matrix.shape[
            0
        ]
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

        if (
            not np.isfinite(
                low
            )
            or not np.isfinite(
                high
            )
            or abs(
                high
                - low
            )
            < 1e-12
        ):
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
            7,
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
        "B1 RAW Candidate Subject-Aware Audit — Relative Overview"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Within-row relative score (higher = better)",
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
        "SCENARIO 2 — B1 RAW EPOCH 10/20/30 SUBJECT-AWARE FIDELITY SWEEP V2"
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
        "  B1 protocol: FROZEN"
    )

    print(
        "  B1 30-epoch full fine-tuning: COMPLETE"
    )

    print(
        "  B1 RAW epoch 10/20/30 subject-aware selection audit: NOW"
    )

    print(
        "  B1 final freeze: NOT YET"
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
        "\nFrozen audit design:"
    )

    print(
        f"  candidates: RAW epochs "
        f"{CANDIDATE_EPOCHS}"
    )

    print(
        f"  subjects: "
        f"{SELECTED_SUBJECT_COUNT}"
    )

    print(
        "  subject selection: deterministic, "
        "Train subjects with >=5 distinct EO and >=5 distinct EC blocks"
    )

    print(
        f"  Real reference per subject/state: "
        f"{REAL_WINDOWS_PER_SUBJECT_STATE} windows "
        f"(one central usable window per distinct Train block)"
    )

    print(
        f"  Synthetic per subject/state/candidate: "
        f"{SYNTHETIC_WINDOWS_PER_SUBJECT_STATE}"
    )

    print(
        "  identical subject order and sampling seeds across candidates: YES"
    )

    print(
        "  same sampling seed for EO and EC paired diagnostic: YES"
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
        LOG_DIR,
        GENERATED_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit frozen sources / candidates
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B1 FULL-RUN CANDIDATES"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B0_GENERATOR,
        B1_PROTOCOL_FILE,
        B1_FULL_RESULT,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B1_PROTOCOL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        b1_protocol = json.load(
            f
        )

    with B1_FULL_RESULT.open(
        "r",
        encoding="utf-8",
    ) as f:
        b1_result = json.load(
            f
        )

    if b1_protocol[
        "status"
    ] != "B1_SUBJECT_STATE_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B1 protocol is not frozen."
        )

    if b1_result[
        "status"
    ] != "B1_FULL_FINETUNE_30_EPOCHS_COMPLETE":
        raise RuntimeError(
            "B1 full run is not complete."
        )

    if b1_result[
        "pilot_checkpoint_used"
    ] is not False:
        raise RuntimeError(
            "Scientific B1 full run unexpectedly used pilot checkpoint."
        )

    if b1_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "B1 full run reports Final Test access."
        )

    if [
        int(
            record[
                "epoch"
            ]
        )
        for record in b1_result[
            "predeclared_RAW_candidates"
        ]
    ] != CANDIDATE_EPOCHS:
        raise RuntimeError(
            "B1 candidate epoch set mismatch."
        )

    candidate_audit = {}

    for epoch in CANDIDATE_EPOCHS:
        path = candidate_path(
            epoch
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

        if checkpoint[
            "status"
        ] != "B1_FULL_FINETUNE_CHECKPOINT":
            raise RuntimeError(
                f"Unexpected candidate checkpoint status at epoch {epoch}."
            )

        if int(
            checkpoint[
                "epoch"
            ]
        ) != epoch:
            raise RuntimeError(
                f"Candidate epoch mismatch for {epoch}."
            )

        if checkpoint[
            "selected_weight_kind"
        ] != "RAW":
            raise RuntimeError(
                f"Candidate {epoch} is not RAW."
            )

        if checkpoint[
            "EMA_enabled"
        ] is not False:
            raise RuntimeError(
                f"Candidate {epoch} unexpectedly uses EMA."
            )

        if checkpoint[
            "final_test_accessed"
        ] is not False:
            raise RuntimeError(
                f"Candidate {epoch} reports Final Test access."
            )

        candidate_audit[
            epoch
        ] = {
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
        }

        print(
            f"epoch {epoch:>2} RAW "
            f"| Train={candidate_audit[epoch]['train_v_mse']:.6f} "
            f"| Val-correct="
            f"{candidate_audit[epoch]['validation_correct_subject_v_mse']:.6f} "
            f"| Val-gap="
            f"{candidate_audit[epoch]['validation_wrong_minus_correct_v_mse']:.3e} "
            f"| SHA256="
            f"{candidate_audit[epoch]['file_sha256'][:16]}..."
        )

    print(
        "candidate audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze audit protocol before generation
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE SUBJECT-AWARE SELECTION PROTOCOL"
    )

    print(
        "=" * 80
    )

    failed_v1_protocol_sha256 = (
        sha256_file(
            FAILED_V1_PROTOCOL
        )
        if FAILED_V1_PROTOCOL.is_file()
        else None
    )

    amendment_payload = {
        "status": (
            "B1_SUBJECT_AWARE_AUDIT_PROTOCOL_AMENDMENT_V2"
        ),
        "amendment_reason": (
            "The v1 audit required >=8 distinct EO Train blocks "
            "and >=8 distinct EC Train blocks per subject. "
            "The actual frozen Train design contains only 10 blocks "
            "per subject total, so that requirement was structurally "
            "impossible; v1 stopped at metadata/index preflight with "
            "eligible=0 before Real EEG reference extraction, "
            "synthetic generation, or candidate evaluation."
        ),
        "v1_protocol_path": str(
            FAILED_V1_PROTOCOL
        ),
        "v1_protocol_sha256": (
            failed_v1_protocol_sha256
        ),
        "v1_protocol_preserved_read_only": True,
        "scientific_change": {
            "old_real_reference": (
                "8 windows per subject/state, each from a distinct Train block"
            ),
            "new_real_reference": (
                "5 windows per subject/state, one central usable window "
                "from each of 5 distinct Train blocks"
            ),
            "selected_subject_count": (
                SELECTED_SUBJECT_COUNT
            ),
            "synthetic_windows_per_subject_state": (
                SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
            ),
            "candidate_epochs_unchanged": (
                CANDIDATE_EPOCHS
            ),
            "selection_criteria_unchanged": True,
            "sampling_seeds_unchanged": True,
        },
        "governance": {
            "amendment_made_before_candidate_generation": True,
            "Validation_EEG_accessed_by_failed_v1": False,
            "Final_Test_accessed_by_failed_v1": False,
            "training_updates_executed_by_failed_v1": False,
            "bulk_synthetic_generation_started": False,
        },
    }

    if AMENDMENT_FILE.exists():
        with AMENDMENT_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_amendment = json.load(
                f
            )

        if existing_amendment != amendment_payload:
            raise RuntimeError(
                "Conflicting B1 subject-aware audit amendment already exists."
            )
    else:
        atomic_json_save(
            amendment_payload,
            AMENDMENT_FILE,
        )

    print(
        f"protocol amendment: "
        f"{AMENDMENT_FILE}"
    )

    protocol_snapshot = {
        "status": (
            "B1_RAW_CANDIDATE_SUBJECT_AWARE_PROTOCOL_V2_FROZEN"
        ),
        "protocol_amendment": str(
            AMENDMENT_FILE
        ),
        "candidate_epochs": (
            CANDIDATE_EPOCHS
        ),
        "weight_kind": (
            "RAW"
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
        "sampling_batch_size": (
            SAMPLING_BATCH_SIZE
        ),
        "sampling_seed_base": (
            SAMPLING_SEED_BASE
        ),
        "paired_EO_EC_noise_seed": True,
        "same_sampling_seeds_across_candidates": True,
        "real_reference": (
            "Real Train only; one central usable window "
            "from each of 5 distinct blocks per subject/state"
        ),
        "generic_selection_criteria": (
            GENERIC_CRITERIA
        ),
        "subject_specific_selection_criteria": (
            SUBJECT_CRITERIA
        ),
        "recommendation_rule": (
            "Recommend a single B1 epoch only if the same epoch "
            "wins at least 3 of 5 generic criteria AND at least "
            "3 of 5 subject-specific criteria. Otherwise report "
            "NO_CLEAR_SINGLE_CANDIDATE and interpret the trade-off "
            "before B1 freeze."
        ),
        "subject_identity_fingerprint": (
            "Concatenated EO+EC median log-bandpower over "
            "delta/theta/alpha/beta × 19 channels = 152 dimensions"
        ),
        "subject_identity_distance": (
            "RMS Euclidean distance after standardizing each "
            "fingerprint feature using only Real Train cohort mean/std"
        ),
        "subject_retrieval_cohort": (
            SELECTED_SUBJECT_COUNT
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "training_updates_executed": False,
        "bulk_synthetic_generation_started": False,
        "B1_frozen": False,
    }

    if PROTOCOL_FILE.exists():
        with PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_protocol = json.load(
                f
            )

        if (
            existing_protocol
            != protocol_snapshot
        ):
            raise RuntimeError(
                "Conflicting B1 subject-aware audit protocol already exists."
            )

        print(
            "identical subject-aware audit protocol already exists."
        )

    else:
        atomic_json_save(
            protocol_snapshot,
            PROTOCOL_FILE,
        )

        print(
            f"protocol: "
            f"{PROTOCOL_FILE}"
        )

    # ========================================================
    # Step 3 — Build fixed Real Train subject reference
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD FIXED SUBJECT-BALANCED REAL TRAIN REFERENCE"
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
        subject_mapping
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Unexpected subject count."
        )

    grouped = build_subject_block_groups(
        records
    )

    block_inventory = summarize_subject_block_inventory(
        grouped
    )

    print(
        "Train distinct-block inventory per subject/state:"
    )

    print(
        "  EO min/median/max: "
        f"{block_inventory['EO_min']} / "
        f"{block_inventory['EO_median']:.1f} / "
        f"{block_inventory['EO_max']}"
    )

    print(
        "  EC min/median/max: "
        f"{block_inventory['EC_min']} / "
        f"{block_inventory['EC_median']:.1f} / "
        f"{block_inventory['EC_max']}"
    )

    print(
        "  subjects with >= "
        f"{REAL_WINDOWS_PER_SUBJECT_STATE} "
        "distinct blocks in BOTH states: "
        f"{block_inventory['joint_subjects_with_at_least_required_blocks']}"
    )

    if (
        block_inventory[
            "joint_subjects_with_at_least_required_blocks"
        ]
        < SELECTED_SUBJECT_COUNT
    ):
        raise RuntimeError(
            "V2 structural preflight still cannot support "
            f"{SELECTED_SUBJECT_COUNT} subjects with "
            f"{REAL_WINDOWS_PER_SUBJECT_STATE} distinct "
            "Train blocks per state. Stop before EEG loading/generation."
        )

    selected_subjects = choose_selected_subjects(
        grouped
    )

    (
        selected_records,
        real_manifest_rows,
    ) = build_real_reference_selection(
        records=records,
        grouped=grouped,
        selected_subjects=(
            selected_subjects
        ),
    )

    real_z = load_real_reference(
        selected_records=(
            selected_records
        ),
        scaler=scaler,
    )

    save_csv_rows(
        real_manifest_rows,
        REAL_MANIFEST_CSV,
    )

    selected_subject_ids = [
        next(
            subject_id
            for subject_id, index
            in subject_mapping.items()
            if int(
                index
            )
            == subject_index
        )
        for subject_index in selected_subjects
    ]

    np.savez_compressed(
        REAL_REFERENCE_NPZ,
        selected_subject_indices=np.asarray(
            selected_subjects,
            dtype=np.int64,
        ),
        selected_subject_ids=np.asarray(
            selected_subject_ids,
            dtype="U32",
        ),
        real_z=real_z,
    )

    print(
        f"selected subjects: "
        f"{len(selected_subjects)}"
    )

    print(
        f"subject index range in audit: "
        f"{min(selected_subjects)} ... "
        f"{max(selected_subjects)}"
    )

    print(
        f"Real reference shape: "
        f"{real_z.shape}"
    )

    print(
        f"Real reference windows total: "
        f"{real_z.shape[0] * real_z.shape[1] * real_z.shape[2]}"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    # ========================================================
    # Step 4 — Build sampler
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — BUILD FIXED B1 V-PREDICTION SAMPLER"
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

    frozen_b0 = torch.load(
        FROZEN_B0_GENERATOR,
        map_location="cpu",
        weights_only=False,
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

    # ========================================================
    # Step 5 — Generate / evaluate candidates
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — GENERATE + EVALUATE RAW B1 CANDIDATES"
    )

    print(
        "=" * 80
    )

    metrics_by_epoch: dict[
        int,
        dict[str, Any],
    ] = {}

    matrices_by_epoch: dict[
        int,
        dict[str, np.ndarray],
    ] = {}

    subject_rows_all: list[
        dict[str, Any]
    ] = []

    synthetic_packages = {}

    total_start = (
        time.perf_counter()
    )

    candidate_file_hash_before = {
        epoch: sha256_file(
            candidate_path(
                epoch
            )
        )
        for epoch in CANDIDATE_EPOCHS
    }

    for epoch in CANDIDATE_EPOCHS:
        print(
            "\n"
            + "-" * 80
        )

        print(
            f"RAW B1 EPOCH {epoch} "
            f"— {SELECTED_SUBJECT_COUNT} SUBJECTS "
            f"× 2 STATES × "
            f"{SYNTHETIC_WINDOWS_PER_SUBJECT_STATE} REPS"
        )

        print(
            "-" * 80
        )

        checkpoint = torch.load(
            candidate_path(
                epoch
            ),
            map_location="cpu",
            weights_only=False,
        )

        model = build_b1_model(
            checkpoint=checkpoint,
            frozen_b0=frozen_b0,
            device=device,
        )

        print(
            f"parameters: "
            f"{sum(parameter.numel() for parameter in model.parameters()):,}"
        )

        candidate_start = (
            time.perf_counter()
        )

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

        (
            metrics,
            matrices,
            subject_rows,
        ) = compute_candidate_metrics(
            real_z=real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

        metrics[
            "epoch"
        ] = int(
            epoch
        )

        metrics[
            "checkpoint"
        ] = (
            candidate_audit[
                epoch
            ]
        )

        metrics[
            "candidate_seconds"
        ] = float(
            time.perf_counter()
            - candidate_start
        )

        metrics_by_epoch[
            epoch
        ] = (
            metrics
        )

        matrices_by_epoch[
            epoch
        ] = (
            matrices
        )

        for local_subject_index, row in enumerate(
            subject_rows
        ):
            subject_rows_all.append(
                {
                    "epoch": (
                        epoch
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
                    **{
                        key: value
                        for key, value
                        in row.items()
                        if key
                        != "local_subject_index"
                    },
                }
            )

        synthetic_packages[
            f"epoch_{epoch:03d}_z"
        ] = synth_z.astype(
            np.float32,
            copy=False,
        )

        print(
            "scale normalized ratio EO/EC: "
            f"{metrics['scale']['EO']['normalized_std_ratio']:.3f}"
            " / "
            f"{metrics['scale']['EC']['normalized_std_ratio']:.3f}"
        )

        print(
            "bandpower MAE delta/theta/alpha/beta: "
            f"{metrics['bandpower_mae_db']['delta']:.3f}"
            " / "
            f"{metrics['bandpower_mae_db']['theta']:.3f}"
            " / "
            f"{metrics['bandpower_mae_db']['alpha']:.3f}"
            " / "
            f"{metrics['bandpower_mae_db']['beta']:.3f}"
            " dB"
        )

        print(
            "alpha effect O1 real/synth: "
            f"{metrics['alpha_effect']['real_O1_db']:.3f}"
            " / "
            f"{metrics['alpha_effect']['synthetic_O1_db']:.3f}"
            " dB"
        )

        print(
            "alpha effect O2 real/synth: "
            f"{metrics['alpha_effect']['real_O2_db']:.3f}"
            " / "
            f"{metrics['alpha_effect']['synthetic_O2_db']:.3f}"
            " dB"
        )

        print(
            "alpha-effect channel MAE: "
            f"{metrics['alpha_effect']['channel_mae_db']:.3f} dB"
        )

        print(
            "channel-correlation r EO/EC: "
            f"{metrics['spatial']['EO']['correlation_r']:.3f}"
            " / "
            f"{metrics['spatial']['EC']['correlation_r']:.3f}"
        )

        print(
            "alpha dwPLI2 r EO/EC: "
            f"{metrics['spatial']['EO']['alpha_dwpli2_r']:.3f}"
            " / "
            f"{metrics['spatial']['EC']['alpha_dwpli2_r']:.3f}"
        )

        subject_metrics = (
            metrics[
                "subject_identity"
            ]
        )

        print(
            "subject fingerprint standardized RMSE mean: "
            f"{subject_metrics['subject_fingerprint_standardized_rmse_mean']:.3f}"
        )

        print(
            "subject fingerprint Pearson r median: "
            f"{subject_metrics['subject_fingerprint_pearson_r_median']:.3f}"
        )

        print(
            "subject correct-vs-median-wrong margin mean: "
            f"{subject_metrics['subject_correct_vs_median_wrong_margin_mean']:.3f}"
        )

        print(
            "subject retrieval top1/top5: "
            f"{subject_metrics['subject_retrieval_top1_accuracy']:.3f}"
            " / "
            f"{subject_metrics['subject_retrieval_top5_accuracy']:.3f}"
        )

        print(
            "correct closer than fixed wrong: "
            f"{subject_metrics['subject_correct_better_than_fixed_wrong_fraction']:.3f}"
        )

        print(
            "candidate time: "
            f"{metrics['candidate_seconds']/60:.2f} min"
        )

        del model

        torch.cuda.empty_cache()

    runtime_seconds = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # Step 6 — Apply predeclared selection rule
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — PREDECLARED GENERIC + SUBJECT-SPECIFIC CRITERION WINS"
    )

    print(
        "=" * 80
    )

    (
        generic_winners,
        generic_win_counts,
    ) = choose_winners(
        metrics_by_epoch,
        GENERIC_CRITERIA,
    )

    (
        subject_winners,
        subject_win_counts,
    ) = choose_winners(
        metrics_by_epoch,
        SUBJECT_CRITERIA,
    )

    total_win_counts = {
        epoch: (
            generic_win_counts[
                epoch
            ]
            + subject_win_counts[
                epoch
            ]
        )
        for epoch in CANDIDATE_EPOCHS
    }

    recommended_epoch = None

    for epoch in CANDIDATE_EPOCHS:
        if (
            generic_win_counts[
                epoch
            ]
            >= 3
            and subject_win_counts[
                epoch
            ]
            >= 3
        ):
            if recommended_epoch is not None:
                raise RuntimeError(
                    "More than one candidate satisfies the "
                    "predeclared recommendation rule."
                )

            recommended_epoch = (
                epoch
            )

    if recommended_epoch is None:
        diagnostic_direction = (
            "NO_CLEAR_SINGLE_B1_CANDIDATE_REQUIRES_INTERPRETATION"
        )
    else:
        diagnostic_direction = (
            f"RAW_EPOCH_{recommended_epoch}_"
            f"PRIMARY_SUBJECT_AWARE_CANDIDATE"
        )

    print(
        "generic criterion winners:"
    )

    for criterion, epoch in generic_winners.items():
        print(
            f"  {criterion}: epoch {epoch}"
        )

    print(
        f"generic wins: "
        f"{generic_win_counts}"
    )

    print(
        "\nsubject-specific criterion winners:"
    )

    for criterion, epoch in subject_winners.items():
        print(
            f"  {criterion}: epoch {epoch}"
        )

    print(
        f"subject-specific wins: "
        f"{subject_win_counts}"
    )

    print(
        f"\ntotal wins: "
        f"{total_win_counts}"
    )

    print(
        f"diagnostic direction: "
        f"{diagnostic_direction}"
    )

    # ========================================================
    # Step 7 — Save metrics / packages
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — SAVE SUBJECT-AWARE SWEEP ARTIFACTS"
    )

    print(
        "=" * 80
    )

    metric_rows = []

    for epoch in CANDIDATE_EPOCHS:
        metrics = metrics_by_epoch[
            epoch
        ]

        subject_metrics = (
            metrics[
                "subject_identity"
            ]
        )

        metric_rows.append(
            {
                "epoch": (
                    epoch
                ),
                "eo_normalized_std_ratio": (
                    metrics[
                        "scale"
                    ][
                        "EO"
                    ][
                        "normalized_std_ratio"
                    ]
                ),
                "ec_normalized_std_ratio": (
                    metrics[
                        "scale"
                    ][
                        "EC"
                    ][
                        "normalized_std_ratio"
                    ]
                ),
                "scale_error_mean_abs_ratio_minus_1": (
                    metrics[
                        "scale_error_mean_abs_ratio_minus_1"
                    ]
                ),
                "delta_bandpower_mae_db": (
                    metrics[
                        "bandpower_mae_db"
                    ][
                        "delta"
                    ]
                ),
                "theta_bandpower_mae_db": (
                    metrics[
                        "bandpower_mae_db"
                    ][
                        "theta"
                    ]
                ),
                "alpha_bandpower_mae_db": (
                    metrics[
                        "bandpower_mae_db"
                    ][
                        "alpha"
                    ]
                ),
                "beta_bandpower_mae_db": (
                    metrics[
                        "bandpower_mae_db"
                    ][
                        "beta"
                    ]
                ),
                "mean_bandpower_mae_db": (
                    metrics[
                        "mean_bandpower_mae_db"
                    ]
                ),
                "alpha_effect_channel_mae_db": (
                    metrics[
                        "alpha_effect"
                    ][
                        "channel_mae_db"
                    ]
                ),
                "real_O1_alpha_effect_db": (
                    metrics[
                        "alpha_effect"
                    ][
                        "real_O1_db"
                    ]
                ),
                "synthetic_O1_alpha_effect_db": (
                    metrics[
                        "alpha_effect"
                    ][
                        "synthetic_O1_db"
                    ]
                ),
                "real_O2_alpha_effect_db": (
                    metrics[
                        "alpha_effect"
                    ][
                        "real_O2_db"
                    ]
                ),
                "synthetic_O2_alpha_effect_db": (
                    metrics[
                        "alpha_effect"
                    ][
                        "synthetic_O2_db"
                    ]
                ),
                "correlation_r_eo": (
                    metrics[
                        "spatial"
                    ][
                        "EO"
                    ][
                        "correlation_r"
                    ]
                ),
                "correlation_r_ec": (
                    metrics[
                        "spatial"
                    ][
                        "EC"
                    ][
                        "correlation_r"
                    ]
                ),
                "alpha_dwpli2_r_eo": (
                    metrics[
                        "spatial"
                    ][
                        "EO"
                    ][
                        "alpha_dwpli2_r"
                    ]
                ),
                "alpha_dwpli2_r_ec": (
                    metrics[
                        "spatial"
                    ][
                        "EC"
                    ][
                        "alpha_dwpli2_r"
                    ]
                ),
                "subject_fingerprint_standardized_rmse_mean": (
                    subject_metrics[
                        "subject_fingerprint_standardized_rmse_mean"
                    ]
                ),
                "subject_fingerprint_pearson_r_median": (
                    subject_metrics[
                        "subject_fingerprint_pearson_r_median"
                    ]
                ),
                "subject_correct_vs_median_wrong_margin_mean": (
                    subject_metrics[
                        "subject_correct_vs_median_wrong_margin_mean"
                    ]
                ),
                "subject_retrieval_top1_accuracy": (
                    subject_metrics[
                        "subject_retrieval_top1_accuracy"
                    ]
                ),
                "subject_retrieval_top5_accuracy": (
                    subject_metrics[
                        "subject_retrieval_top5_accuracy"
                    ]
                ),
                "subject_correct_better_than_fixed_wrong_fraction": (
                    subject_metrics[
                        "subject_correct_better_than_fixed_wrong_fraction"
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

    save_csv_rows(
        subject_rows_all,
        FINGERPRINT_CSV,
    )

    np.savez_compressed(
        SYNTHETIC_NPZ,
        selected_subject_indices=np.asarray(
            selected_subjects,
            dtype=np.int64,
        ),
        selected_subject_ids=np.asarray(
            selected_subject_ids,
            dtype="U32",
        ),
        **synthetic_packages,
    )

    matrix_package = {
        "selected_subject_indices": np.asarray(
            selected_subjects,
            dtype=np.int64,
        ),
    }

    for epoch in CANDIDATE_EPOCHS:
        for key, value in matrices_by_epoch[
            epoch
        ].items():
            matrix_package[
                f"epoch_{epoch:03d}_{key}"
            ] = value

    np.savez_compressed(
        MATRICES_NPZ,
        **matrix_package,
    )

    print(
        f"metrics CSV: "
        f"{METRICS_CSV}"
    )

    print(
        f"subject fingerprint CSV: "
        f"{FINGERPRINT_CSV}"
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
    # Step 8 — Figures
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — SAVE SUBJECT-AWARE SWEEP FIGURES"
    )

    print(
        "=" * 80
    )

    save_scale_figure(
        metrics_by_epoch
    )

    save_bandpower_figure(
        metrics_by_epoch
    )

    save_alpha_figure(
        metrics_by_epoch
    )

    save_spatial_figure(
        metrics_by_epoch
    )

    save_subject_retrieval_figure(
        metrics_by_epoch
    )

    save_subject_distance_figure(
        metrics_by_epoch
    )

    save_overview_figure(
        metrics_by_epoch
    )

    for path in (
        SCALE_FIGURE,
        BANDPOWER_FIGURE,
        ALPHA_FIGURE,
        SPATIAL_FIGURE,
        SUBJECT_RETRIEVAL_FIGURE,
        SUBJECT_DISTANCE_FIGURE,
        OVERVIEW_FIGURE,
    ):
        print(
            path
        )

    # ========================================================
    # Step 9 — Verify checkpoint immutability + save summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — VERIFY IMMUTABILITY + SAVE SUMMARY"
    )

    print(
        "=" * 80
    )

    for epoch in CANDIDATE_EPOCHS:
        hash_after = sha256_file(
            candidate_path(
                epoch
            )
        )

        if (
            hash_after
            != candidate_file_hash_before[
                epoch
            ]
        ):
            raise RuntimeError(
                f"Candidate epoch {epoch} checkpoint changed "
                "during subject-aware sweep."
            )

    summary = {
        "status": (
            "B1_RAW_CANDIDATE_SUBJECT_AWARE_SWEEP_V2_COMPLETE"
        ),
        "candidate_epochs": (
            CANDIDATE_EPOCHS
        ),
        "selected_subject_count": (
            SELECTED_SUBJECT_COUNT
        ),
        "train_distinct_block_inventory": (
            block_inventory
        ),
        "selected_subject_indices": (
            selected_subjects
        ),
        "selected_subject_ids": (
            selected_subject_ids
        ),
        "real_windows_per_subject_state": (
            REAL_WINDOWS_PER_SUBJECT_STATE
        ),
        "synthetic_windows_per_subject_state": (
            SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
        ),
        "candidate_checkpoint_audit": (
            candidate_audit
        ),
        "metrics_by_epoch": {
            str(
                epoch
            ): metrics_by_epoch[
                epoch
            ]
            for epoch in CANDIDATE_EPOCHS
        },
        "generic_criterion_winners": (
            generic_winners
        ),
        "generic_criterion_win_counts": {
            str(
                epoch
            ): generic_win_counts[
                epoch
            ]
            for epoch in CANDIDATE_EPOCHS
        },
        "subject_specific_criterion_winners": (
            subject_winners
        ),
        "subject_specific_criterion_win_counts": {
            str(
                epoch
            ): subject_win_counts[
                epoch
            ]
            for epoch in CANDIDATE_EPOCHS
        },
        "total_criterion_win_counts": {
            str(
                epoch
            ): total_win_counts[
                epoch
            ]
            for epoch in CANDIDATE_EPOCHS
        },
        "recommended_epoch": (
            recommended_epoch
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
        "artifacts": {
            "protocol": str(
                PROTOCOL_FILE
            ),
            "protocol_amendment": str(
                AMENDMENT_FILE
            ),
            "metrics_csv": str(
                METRICS_CSV
            ),
            "subject_fingerprint_csv": str(
                FINGERPRINT_CSV
            ),
            "real_reference_npz": str(
                REAL_REFERENCE_NPZ
            ),
            "real_manifest_csv": str(
                REAL_MANIFEST_CSV
            ),
            "synthetic_npz": str(
                SYNTHETIC_NPZ
            ),
            "matrices_npz": str(
                MATRICES_NPZ
            ),
            "scale_figure": str(
                SCALE_FIGURE
            ),
            "bandpower_figure": str(
                BANDPOWER_FIGURE
            ),
            "alpha_figure": str(
                ALPHA_FIGURE
            ),
            "spatial_figure": str(
                SPATIAL_FIGURE
            ),
            "subject_retrieval_figure": str(
                SUBJECT_RETRIEVAL_FIGURE
            ),
            "subject_distance_figure": str(
                SUBJECT_DISTANCE_FIGURE
            ),
            "overview_figure": str(
                OVERVIEW_FIGURE
            ),
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "training_updates_executed": False,
            "bulk_synthetic_generation_started": False,
            "B1_frozen": False,
            "candidate_checkpoints_unchanged": True,
        },
        "next": (
            "Interpret generic + subject-specific fidelity trade-offs "
            "before formally freezing a B1 RAW candidate."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
    )

    print(
        "candidate checkpoint immutability: PASS"
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
        "B1 RAW SUBJECT-AWARE FIDELITY SWEEP COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"candidates evaluated: "
        f"{len(CANDIDATE_EPOCHS)}"
    )

    print(
        f"subjects: "
        f"{SELECTED_SUBJECT_COUNT}"
    )

    print(
        "synthetic windows per candidate: "
        f"{SELECTED_SUBJECT_COUNT * 2 * SYNTHETIC_WINDOWS_PER_SUBJECT_STATE}"
    )

    print(
        f"runtime: "
        f"{runtime_seconds/60:.2f} min"
    )

    for epoch in CANDIDATE_EPOCHS:
        metrics = metrics_by_epoch[
            epoch
        ]

        subject_metrics = (
            metrics[
                "subject_identity"
            ]
        )

        print(
            "\n"
            f"RAW epoch {epoch}:"
        )

        print(
            "  scale EO/EC: "
            f"{metrics['scale']['EO']['normalized_std_ratio']:.3f}"
            " / "
            f"{metrics['scale']['EC']['normalized_std_ratio']:.3f}"
        )

        print(
            "  mean bandpower MAE: "
            f"{metrics['mean_bandpower_mae_db']:.3f} dB"
        )

        print(
            "  alpha-effect MAE: "
            f"{metrics['alpha_effect']['channel_mae_db']:.3f} dB"
        )

        print(
            "  correlation r EO/EC: "
            f"{metrics['spatial']['EO']['correlation_r']:.3f}"
            " / "
            f"{metrics['spatial']['EC']['correlation_r']:.3f}"
        )

        print(
            "  alpha dwPLI2 r EO/EC: "
            f"{metrics['spatial']['EO']['alpha_dwpli2_r']:.3f}"
            " / "
            f"{metrics['spatial']['EC']['alpha_dwpli2_r']:.3f}"
        )

        print(
            "  subject fingerprint RMSE: "
            f"{subject_metrics['subject_fingerprint_standardized_rmse_mean']:.3f}"
        )

        print(
            "  subject fingerprint r median: "
            f"{subject_metrics['subject_fingerprint_pearson_r_median']:.3f}"
        )

        print(
            "  subject margin: "
            f"{subject_metrics['subject_correct_vs_median_wrong_margin_mean']:.3f}"
        )

        print(
            "  subject retrieval top1/top5: "
            f"{subject_metrics['subject_retrieval_top1_accuracy']:.3f}"
            " / "
            f"{subject_metrics['subject_retrieval_top5_accuracy']:.3f}"
        )

        print(
            "  correct closer than fixed wrong: "
            f"{subject_metrics['subject_correct_better_than_fixed_wrong_fraction']:.3f}"
        )

    print(
        "\nGeneric wins:"
    )

    print(
        generic_win_counts
    )

    print(
        "Subject-specific wins:"
    )

    print(
        subject_win_counts
    )

    print(
        "Total wins:"
    )

    print(
        total_win_counts
    )

    print(
        f"diagnostic direction: "
        f"{diagnostic_direction}"
    )

    print(
        f"recommended epoch: "
        f"{recommended_epoch}"
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
        "B1 frozen: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "INTERPRET SUBJECT-AWARE B1 CANDIDATE SWEEP "
        "BEFORE FORMAL B1 FREEZE"
    )


if __name__ == "__main__":
    main()
