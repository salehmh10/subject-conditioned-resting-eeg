from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from s2_23_check_real_train_dataset import NORMALIZATION_FILE
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import build_reverse_schedule
from s2_47_b1_architecture_warmstart_sanity import state_dict_sha256
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import (
    compute_candidate_metrics,
    sample_b1_candidate_state,
)
from s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep import (
    build_model_from_state,
    full_log_psd_mae,
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

S2_66_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "validation_checkpoint_selector_v1"
)

S2_66_SUMMARY = (
    S2_66_ROOT
    / "metrics"
    / "s2_66_validation_checkpoint_selector_summary_v1.json"
)

VAL_DEV_REAL_NPZ = (
    S2_66_ROOT
    / "references"
    / "s2_66_val_dev_real_reference_v1.npz"
)

VAL_CONF_REAL_NPZ = (
    S2_66_ROOT
    / "references"
    / "s2_66_val_conf_real_reference_v1.npz"
)

S2_66_VAL_DEV_B1 = (
    S2_66_ROOT
    / "generated"
    / "s2_66_val_dev_b1_synthetic_v1.npz"
)

S2_66_VAL_CONF_B1 = (
    S2_66_ROOT
    / "generated"
    / "s2_66_val_conf_b1_synthetic_v1.npz"
)

S2_68_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "replication_e4_scale_failure_diagnostic_v1"
    / "metrics"
    / "s2_68_replication_e4_scale_failure_summary_v1.json"
)


# ============================================================
# s2_69 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "sampling_robust_checkpoint_selector_v2"
)

METRIC_DIR = RUN_ROOT / "metrics"
GENERATED_DIR = RUN_ROOT / "generated"

PROTOCOL_JSON = (
    METRIC_DIR
    / "s2_69_sampling_robust_selector_protocol_v2.json"
)

BASELINE_JSON = (
    METRIC_DIR
    / "s2_69_multiseed_b1_validation_baselines_v2.json"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_69_sampling_robust_selector_summary_v2.json"
)


# ============================================================
# Final selector design
# ============================================================

N_SUBJECTS = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4
N_CHANNELS = 19
N_TIMES = 512

EXPECTED_REAL_SHAPE = (
    N_SUBJECTS,
    2,
    REAL_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)

EXPECTED_SYNTH_SHAPE = (
    N_SUBJECTS,
    2,
    SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)

TIMESTEPS = 1000
COSINE_S = 0.008

# The first seed in each list is the already-locked s2_66 seed.
# The other two are new and frozen BEFORE s2_70 training.
VAL_DEV_SAMPLING_SEEDS = [
    20261511,
    20261911,
    20261912,
]

VAL_CONF_SAMPLING_SEEDS = [
    20261512,
    20261921,
    20261922,
]

# Two completely fresh training streams for the final selector test.
FINAL_PRIMARY_TRAINING_SEEDS = {
    "master_seed": 20262001,
    "loader_seed": 20262002,
    "noise_seed": 20262003,
}

FINAL_REPLICATION_TRAINING_SEEDS = {
    "master_seed": 20262101,
    "loader_seed": 20262102,
    "noise_seed": 20262103,
}

EPOCHS = [
    1,
    2,
    3,
    4,
    5,
]


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda: f.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def atomic_json_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(
        str(
            path
        )
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

        f.write(
            "\n"
        )

    os.replace(
        temp,
        path,
    )


def atomic_npz_save(
    path: Path,
    **arrays,
) -> None:
    temp = Path(
        str(
            path
        )
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


def generated_path(
    cohort_name: str,
    sampling_seed: int,
) -> Path:
    return (
        GENERATED_DIR
        / (
            f"s2_69_{cohort_name.lower()}_b1_"
            f"seed_{sampling_seed}_synthetic_v2.npz"
        )
    )


def load_real_reference(
    path: Path,
) -> tuple[
    np.ndarray,
    list[int],
    list[str],
]:
    package = np.load(
        path,
        allow_pickle=False,
    )

    real_z = package[
        "real_z"
    ].astype(
        np.float32,
        copy=False,
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

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(
            f"Unexpected Real reference shape: {real_z.shape}"
        )

    if len(
        subject_indices
    ) != N_SUBJECTS:
        raise RuntimeError(
            "Unexpected subject-index count."
        )

    if len(
        subject_ids
    ) != N_SUBJECTS:
        raise RuntimeError(
            "Unexpected subject-ID count."
        )

    if not np.isfinite(
        real_z
    ).all():
        raise RuntimeError(
            "Real reference contains NaN/Inf."
        )

    return (
        real_z,
        subject_indices,
        subject_ids,
    )


def validate_existing_s2_66_baseline(
    *,
    path: Path,
    expected_seed: int,
    expected_subject_indices: list[int],
    expected_subject_ids: list[str],
    expected_model_hash: str,
) -> np.ndarray:
    package = np.load(
        path,
        allow_pickle=False,
    )

    subject_indices = (
        package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64
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
            f"Existing s2_66 B1 cohort mismatch: {path}"
        )

    if (
        subject_ids
        != expected_subject_ids
    ):
        raise RuntimeError(
            f"Existing s2_66 B1 subject-ID mismatch: {path}"
        )

    if int(
        package[
            "sampling_seed"
        ][
            0
        ]
    ) != int(
        expected_seed
    ):
        raise RuntimeError(
            f"Existing s2_66 B1 seed mismatch: {path}"
        )

    if str(
        package[
            "model_state_sha256"
        ][
            0
        ]
    ) != expected_model_hash:
        raise RuntimeError(
            f"Existing s2_66 B1 model hash mismatch: {path}"
        )

    synth_z = package[
        "synth_z"
    ].astype(
        np.float32,
        copy=False,
    )

    if synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected s2_66 B1 shape: {synth_z.shape}"
        )

    if not np.isfinite(
        synth_z
    ).all():
        raise RuntimeError(
            "Existing s2_66 B1 baseline contains NaN/Inf."
        )

    return synth_z


def generate_or_reuse_b1(
    *,
    cohort_name: str,
    sampling_seed: int,
    model,
    model_hash: str,
    subject_indices: list[int],
    subject_ids: list[str],
    reverse_schedule,
    device: torch.device,
) -> np.ndarray:
    output_path = generated_path(
        cohort_name,
        sampling_seed,
    )

    if output_path.is_file():
        package = np.load(
            output_path,
            allow_pickle=False,
        )

        if (
            package[
                "selected_subject_indices"
            ]
            .astype(
                np.int64
            )
            .tolist()
            != subject_indices
        ):
            raise RuntimeError(
                f"{cohort_name}/{sampling_seed}: "
                f"cached cohort mismatch."
            )

        if (
            package[
                "selected_subject_ids"
            ]
            .astype(
                str
            )
            .tolist()
            != subject_ids
        ):
            raise RuntimeError(
                f"{cohort_name}/{sampling_seed}: "
                f"cached subject-ID mismatch."
            )

        if int(
            package[
                "sampling_seed"
            ][
                0
            ]
        ) != sampling_seed:
            raise RuntimeError(
                f"{cohort_name}/{sampling_seed}: "
                f"cached sampling seed mismatch."
            )

        if str(
            package[
                "model_state_sha256"
            ][
                0
            ]
        ) != model_hash:
            raise RuntimeError(
                f"{cohort_name}/{sampling_seed}: "
                f"cached model hash mismatch."
            )

        synth_z = package[
            "synth_z"
        ].astype(
            np.float32,
            copy=False,
        )

        if synth_z.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"{cohort_name}/{sampling_seed}: "
                f"cached shape mismatch."
            )

        print(
            f"{cohort_name} seed={sampling_seed}: "
            f"validated cache -> REUSE"
        )

        return synth_z

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"GENERATE FROZEN B1 — "
        f"{cohort_name} — seed={sampling_seed}"
    )

    print(
        "-" * 80
    )

    start = time.perf_counter()

    eo = sample_b1_candidate_state(
        model=model,
        selected_subjects=(
            subject_indices
        ),
        eye_state_value=0,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        base_seed=(
            sampling_seed
        ),
    )

    ec = sample_b1_candidate_state(
        model=model,
        selected_subjects=(
            subject_indices
        ),
        eye_state_value=1,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        base_seed=(
            sampling_seed
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
            f"{cohort_name}/{sampling_seed}: "
            f"unexpected generated shape "
            f"{synth_z.shape}"
        )

    atomic_npz_save(
        output_path,
        selected_subject_indices=np.asarray(
            subject_indices,
            dtype=np.int64,
        ),
        selected_subject_ids=np.asarray(
            subject_ids,
            dtype="U32",
        ),
        synth_z=(
            synth_z
        ),
        sampling_seed=np.asarray(
            [
                sampling_seed
            ],
            dtype=np.int64,
        ),
        model_state_sha256=np.asarray(
            [
                model_hash
            ],
            dtype="U64",
        ),
        generation_seconds=np.asarray(
            [
                time.perf_counter()
                - start
            ],
            dtype=np.float64,
        ),
    )

    print(
        f"{cohort_name} seed={sampling_seed} generation: "
        f"{(time.perf_counter()-start)/60:.2f} min"
    )

    return synth_z


def evaluate_baseline(
    *,
    cohort_name: str,
    sampling_seed: int,
    real_z: np.ndarray,
    synth_z: np.ndarray,
    scaler: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    generic, _, _ = compute_candidate_metrics(
        real_z=real_z,
        synth_z=synth_z,
        scaler=scaler,
    )

    full_psd = full_log_psd_mae(
        real_z=real_z,
        synth_z=synth_z,
        scaler=scaler,
    )

    return {
        "cohort": (
            cohort_name
        ),
        "sampling_seed": int(
            sampling_seed
        ),
        "source_path": str(
            source_path
        ),
        "source_file_sha256": (
            sha256_file(
                source_path
            )
        ),
        "full_psd_mae_db": float(
            full_psd[
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
        "scale_EO": float(
            generic[
                "scale"
            ][
                "EO"
            ][
                "normalized_std_ratio"
            ]
        ),
        "scale_EC": float(
            generic[
                "scale"
            ][
                "EC"
            ][
                "normalized_std_ratio"
            ]
        ),
        "subject_top1": float(
            generic[
                "subject_identity"
            ][
                "subject_retrieval_top1_accuracy"
            ]
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — s2_69 FREEZE SAMPLING-ROBUST "
        "CHECKPOINT SELECTOR V2"
    )

    print(
        "=" * 80
    )

    print(
        "\nScientific role:"
    )

    print(
        "  Freeze the FINAL continuation-B2 selector family "
        "before any new training."
    )

    print(
        "  Robustness is now required across 3 sampling seeds "
        "on VAL-DEV and 3 independent sampling seeds on VAL-CONF."
    )

    print(
        "\nThis stage:"
    )

    print(
        "  NO new continuation training"
    )

    print(
        "  NO candidate checkpoint evaluation"
    )

    print(
        "  YES: generate/lock Frozen-B1 multi-seed Validation baselines"
    )

    print(
        "  Final Test remains SEALED"
    )

    for directory in (
        METRIC_DIR,
        GENERATED_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    required = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        S2_66_SUMMARY,
        VAL_DEV_REAL_NPZ,
        VAL_CONF_REAL_NPZ,
        S2_66_VAL_DEV_B1,
        S2_66_VAL_CONF_B1,
        S2_68_SUMMARY,
        NORMALIZATION_FILE,
    ]

    for path in required:
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

    with S2_66_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_66_summary = json.load(
            f
        )

    with S2_68_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_68_summary = json.load(
            f
        )

    if (
        b1_selection[
            "status"
        ]
        != "B1_SELECTION_FROZEN"
    ):
        raise RuntimeError(
            "B1 is not frozen."
        )

    if (
        s2_66_summary[
            "status"
        ]
        != "S2_66_VALIDATION_CHECKPOINT_SELECTOR_READY"
    ):
        raise RuntimeError(
            "s2_66 is not ready."
        )

    if (
        s2_68_summary[
            "status"
        ]
        != "S2_68_REPLICATION_E4_SCALE_FAILURE_DIAGNOSTIC_COMPLETE"
    ):
        raise RuntimeError(
            "s2_68 is not complete."
        )

    if (
        s2_68_summary[
            "s2_67_failure_still_binding"
        ]
        is not True
    ):
        raise RuntimeError(
            "s2_67 failure is not marked binding."
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

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        scaler = json.load(
            f
        )

    if (
        scaler[
            "status"
        ]
        != "TRAIN_NORMALIZATION_LOCKED"
    ):
        raise RuntimeError(
            "Normalization is not locked."
        )

    # ========================================================
    # Step 1 — Freeze final selector protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — FREEZE FINAL SAMPLING-ROBUST SELECTOR PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_69_SAMPLING_ROBUST_CHECKPOINT_SELECTOR_V2_FROZEN"
        ),
        "scientific_role": (
            "Final continuation-B2 selector family after s2_67 "
            "single-sampling-seed replication failure."
        ),
        "reason_for_change": (
            "s2_68 showed that the nearest failed fresh checkpoint "
            "missed only EO parent-relative scale at point ratio "
            "0.88985912 versus lower guard 0.90, but subject bootstrap "
            "did not justify retroactive acceptance. "
            "Therefore the final family explicitly requires "
            "sampling-seed robustness rather than changing thresholds."
        ),
        "validation_cohorts": {
            "VAL_DEV_reference": str(
                VAL_DEV_REAL_NPZ
            ),
            "VAL_CONF_reference": str(
                VAL_CONF_REAL_NPZ
            ),
            "subjects_per_cohort": (
                N_SUBJECTS
            ),
            "real_windows_per_subject_state": (
                REAL_WINDOWS_PER_SUBJECT_STATE
            ),
            "synthetic_windows_per_subject_state_per_sampling_seed": (
                SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
            ),
        },
        "sampling_seeds": {
            "VAL_DEV": (
                VAL_DEV_SAMPLING_SEEDS
            ),
            "VAL_CONF": (
                VAL_CONF_SAMPLING_SEEDS
            ),
            "important": (
                "Seed 20261511 on VAL-DEV and seed 20261512 on VAL-CONF "
                "reuse the already-locked s2_66 Frozen-B1 baselines. "
                "The remaining four seeds are new and frozen before "
                "any s2_70 continuation training."
            ),
        },
        "future_fresh_training_runs": {
            "PRIMARY": (
                FINAL_PRIMARY_TRAINING_SEEDS
            ),
            "REPLICATION": (
                FINAL_REPLICATION_TRAINING_SEEDS
            ),
            "training_recipe": {
                "initialization": (
                    "exact Frozen B1 RAW epoch10"
                ),
                "lambda_spectral": (
                    0.0
                ),
                "prediction_target": (
                    "v"
                ),
                "epochs": (
                    EPOCHS
                ),
                "batch_size": (
                    64
                ),
                "learning_rate": (
                    5e-5
                ),
                "weight_decay": (
                    1e-4
                ),
                "gradient_clip_norm": (
                    1.0
                ),
                "AMP": (
                    True
                ),
                "EMA": (
                    False
                ),
            },
        },
        "seedwise_guard_rule": (
            "For every candidate checkpoint and every sampling seed, "
            "apply the unchanged existing B2 eligibility helper relative "
            "to the Frozen-B1 baseline generated with the SAME cohort and "
            "the SAME sampling seed. No threshold changes."
        ),
        "VAL_DEV_robust_eligibility_rule": (
            "A checkpoint is robust-eligible only if it passes ALL existing "
            "B2 guards on ALL THREE VAL-DEV sampling seeds (3/3)."
        ),
        "VAL_DEV_selection_rule": (
            "Among robust-eligible E1-E5 checkpoints, select the checkpoint "
            "with lowest MEAN full 1-40 Hz PSD MAE across the three VAL-DEV "
            "sampling seeds. Tie -> lower mean bandpower MAE -> lower mean "
            "alpha-effect MAE -> earlier epoch. If no robust-eligible "
            "checkpoint exists, that training run fails and VAL-CONF is "
            "not accessed for that run."
        ),
        "VAL_CONF_confirmation_rule": (
            "Only the single VAL-DEV-selected checkpoint may be evaluated "
            "on VAL-CONF. It must pass ALL existing B2 guards on ALL THREE "
            "VAL-CONF sampling seeds (3/3). No re-selection, no alternate "
            "checkpoint, and no threshold relaxation."
        ),
        "cross_training_seed_replication_rule": (
            "Both predeclared fresh training runs PRIMARY and REPLICATION "
            "must independently yield a VAL-DEV robust-selected checkpoint "
            "that passes all three VAL-CONF seeds."
        ),
        "formal_freeze_candidate_if_successful": (
            "The predeclared PRIMARY run's selected checkpoint becomes the "
            "sole formal B2 freeze candidate. REPLICATION confirms the "
            "selector mechanism and is not allowed to replace PRIMARY."
        ),
        "terminal_failure_rule": (
            "If either fresh training run fails VAL-DEV robust selection "
            "or 3/3 VAL-CONF confirmation, continuation-B2 is CLOSED. "
            "No additional rescue selector, threshold change, epoch sweep, "
            "or training-seed replacement is allowed. Frozen B1 remains "
            "the robust generator for downstream utility."
        ),
        "Validation_EEG_accessed": (
            True
        ),
        "Final_Test_accessed": (
            False
        ),
        "new_training_executed": (
            False
        ),
        "candidate_evaluated": (
            False
        ),
        "thresholds_changed": (
            False
        ),
        "formal_model_freeze_decision_made": (
            False
        ),
    }

    if PROTOCOL_JSON.is_file():
        with PROTOCOL_JSON.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != protocol:
            raise RuntimeError(
                "Conflicting s2_69 protocol already exists."
            )

        print(
            "identical s2_69 protocol already exists."
        )
    else:
        atomic_json_save(
            protocol,
            PROTOCOL_JSON,
        )

        print(
            f"protocol: "
            f"{PROTOCOL_JSON}"
        )

    # ========================================================
    # Step 2 — Load exact locked Validation references
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — LOAD LOCKED VAL-DEV / VAL-CONF REFERENCES"
    )

    print(
        "=" * 80
    )

    (
        dev_real_z,
        dev_subject_indices,
        dev_subject_ids,
    ) = load_real_reference(
        VAL_DEV_REAL_NPZ
    )

    (
        conf_real_z,
        conf_subject_indices,
        conf_subject_ids,
    ) = load_real_reference(
        VAL_CONF_REAL_NPZ
    )

    if (
        set(
            dev_subject_ids
        )
        & set(
            conf_subject_ids
        )
    ):
        raise RuntimeError(
            "VAL-DEV / VAL-CONF subject overlap."
        )

    print(
        f"VAL-DEV Real shape: "
        f"{dev_real_z.shape}"
    )

    print(
        f"VAL-CONF Real shape: "
        f"{conf_real_z.shape}"
    )

    print(
        "VAL-DEV / VAL-CONF subject overlap: 0"
    )

    # ========================================================
    # Step 3 — Reuse the two historical s2_66 B1 seeds
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — VALIDATE AND REUSE EXISTING s2_66 B1 BASELINES"
    )

    print(
        "=" * 80
    )

    dev_existing_z = validate_existing_s2_66_baseline(
        path=(
            S2_66_VAL_DEV_B1
        ),
        expected_seed=(
            VAL_DEV_SAMPLING_SEEDS[
                0
            ]
        ),
        expected_subject_indices=(
            dev_subject_indices
        ),
        expected_subject_ids=(
            dev_subject_ids
        ),
        expected_model_hash=(
            frozen_b1_model_hash
        ),
    )

    conf_existing_z = validate_existing_s2_66_baseline(
        path=(
            S2_66_VAL_CONF_B1
        ),
        expected_seed=(
            VAL_CONF_SAMPLING_SEEDS[
                0
            ]
        ),
        expected_subject_indices=(
            conf_subject_indices
        ),
        expected_subject_ids=(
            conf_subject_ids
        ),
        expected_model_hash=(
            frozen_b1_model_hash
        ),
    )

    print(
        f"VAL-DEV seed={VAL_DEV_SAMPLING_SEEDS[0]}: "
        f"s2_66 baseline reuse PASS"
    )

    print(
        f"VAL-CONF seed={VAL_CONF_SAMPLING_SEEDS[0]}: "
        f"s2_66 baseline reuse PASS"
    )

    # ========================================================
    # Step 4 — Generate four new B1 baselines
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — GENERATE FOUR NEW FROZEN-B1 SAMPLING BASELINES"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for s2_69 B1 generation."
        )

    device = torch.device(
        "cuda"
    )

    model = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=(
            frozen_b1[
                "model_state_dict"
            ]
        ),
        device=device,
    )

    model.eval()

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

    dev_synthetic_by_seed = {
        int(
            VAL_DEV_SAMPLING_SEEDS[
                0
            ]
        ): (
            dev_existing_z
        ),
    }

    conf_synthetic_by_seed = {
        int(
            VAL_CONF_SAMPLING_SEEDS[
                0
            ]
        ): (
            conf_existing_z
        ),
    }

    for sampling_seed in VAL_DEV_SAMPLING_SEEDS[
        1:
    ]:
        dev_synthetic_by_seed[
            int(
                sampling_seed
            )
        ] = generate_or_reuse_b1(
            cohort_name="VAL_DEV",
            sampling_seed=int(
                sampling_seed
            ),
            model=model,
            model_hash=(
                frozen_b1_model_hash
            ),
            subject_indices=(
                dev_subject_indices
            ),
            subject_ids=(
                dev_subject_ids
            ),
            reverse_schedule=(
                reverse_schedule
            ),
            device=device,
        )

    for sampling_seed in VAL_CONF_SAMPLING_SEEDS[
        1:
    ]:
        conf_synthetic_by_seed[
            int(
                sampling_seed
            )
        ] = generate_or_reuse_b1(
            cohort_name="VAL_CONF",
            sampling_seed=int(
                sampling_seed
            ),
            model=model,
            model_hash=(
                frozen_b1_model_hash
            ),
            subject_indices=(
                conf_subject_indices
            ),
            subject_ids=(
                conf_subject_ids
            ),
            reverse_schedule=(
                reverse_schedule
            ),
            device=device,
        )

    del model

    torch.cuda.empty_cache()

    # ========================================================
    # Step 5 — Compute and lock all 6 B1 baseline metrics
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — COMPUTE AND LOCK ALL MULTI-SEED B1 BASELINE METRICS"
    )

    print(
        "=" * 80
    )

    baseline_records = []

    for sampling_seed in VAL_DEV_SAMPLING_SEEDS:
        sampling_seed = int(
            sampling_seed
        )

        source_path = (
            S2_66_VAL_DEV_B1
            if sampling_seed
            == VAL_DEV_SAMPLING_SEEDS[
                0
            ]
            else generated_path(
                "VAL_DEV",
                sampling_seed,
            )
        )

        record = evaluate_baseline(
            cohort_name="VAL_DEV",
            sampling_seed=(
                sampling_seed
            ),
            real_z=(
                dev_real_z
            ),
            synth_z=(
                dev_synthetic_by_seed[
                    sampling_seed
                ]
            ),
            scaler=scaler,
            source_path=(
                source_path
            ),
        )

        baseline_records.append(
            record
        )

        print(
            f"VAL-DEV seed={sampling_seed}: "
            f"PSD={record['full_psd_mae_db']:.6f} dB "
            f"| band={record['mean_bandpower_mae_db']:.6f} "
            f"| alpha={record['alpha_effect_mae_db']:.6f} "
            f"| Top1={record['subject_top1']:.6f}"
        )

    for sampling_seed in VAL_CONF_SAMPLING_SEEDS:
        sampling_seed = int(
            sampling_seed
        )

        source_path = (
            S2_66_VAL_CONF_B1
            if sampling_seed
            == VAL_CONF_SAMPLING_SEEDS[
                0
            ]
            else generated_path(
                "VAL_CONF",
                sampling_seed,
            )
        )

        record = evaluate_baseline(
            cohort_name="VAL_CONF",
            sampling_seed=(
                sampling_seed
            ),
            real_z=(
                conf_real_z
            ),
            synth_z=(
                conf_synthetic_by_seed[
                    sampling_seed
                ]
            ),
            scaler=scaler,
            source_path=(
                source_path
            ),
        )

        baseline_records.append(
            record
        )

        print(
            f"VAL-CONF seed={sampling_seed}: "
            f"PSD={record['full_psd_mae_db']:.6f} dB "
            f"| band={record['mean_bandpower_mae_db']:.6f} "
            f"| alpha={record['alpha_effect_mae_db']:.6f} "
            f"| Top1={record['subject_top1']:.6f}"
        )

    dev_psd = np.asarray(
        [
            record[
                "full_psd_mae_db"
            ]
            for record in baseline_records
            if record[
                "cohort"
            ]
            == "VAL_DEV"
        ],
        dtype=np.float64,
    )

    conf_psd = np.asarray(
        [
            record[
                "full_psd_mae_db"
            ]
            for record in baseline_records
            if record[
                "cohort"
            ]
            == "VAL_CONF"
        ],
        dtype=np.float64,
    )

    baseline_payload = {
        "status": (
            "S2_69_MULTI_SEED_FROZEN_B1_VALIDATION_BASELINES_LOCKED"
        ),
        "frozen_B1_file_sha256": (
            frozen_b1_file_hash_before
        ),
        "frozen_B1_model_state_sha256": (
            frozen_b1_model_hash
        ),
        "records": (
            baseline_records
        ),
        "aggregate": {
            "VAL_DEV_full_psd_mean_db": float(
                np.mean(
                    dev_psd
                )
            ),
            "VAL_DEV_full_psd_std_ddof0_db": float(
                np.std(
                    dev_psd,
                    ddof=0,
                )
            ),
            "VAL_CONF_full_psd_mean_db": float(
                np.mean(
                    conf_psd
                )
            ),
            "VAL_CONF_full_psd_std_ddof0_db": float(
                np.std(
                    conf_psd,
                    ddof=0,
                )
            ),
        },
        "candidate_checkpoint_evaluated": (
            False
        ),
        "Final_Test_accessed": (
            False
        ),
    }

    atomic_json_save(
        baseline_payload,
        BASELINE_JSON,
    )

    print(
        "\nB1 multi-seed aggregate:"
    )

    print(
        f"  VAL-DEV PSD mean±std: "
        f"{np.mean(dev_psd):.6f} ± "
        f"{np.std(dev_psd, ddof=0):.6f} dB"
    )

    print(
        f"  VAL-CONF PSD mean±std: "
        f"{np.mean(conf_psd):.6f} ± "
        f"{np.std(conf_psd, ddof=0):.6f} dB"
    )

    # ========================================================
    # Step 6 — Immutability + summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — VERIFY IMMUTABILITY + SAVE SUMMARY"
    )

    print(
        "=" * 80
    )

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != frozen_b1_file_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during s2_69."
        )

    summary = {
        "status": (
            "S2_69_SAMPLING_ROBUST_SELECTOR_V2_READY"
        ),
        "protocol": str(
            PROTOCOL_JSON
        ),
        "baseline_metrics": str(
            BASELINE_JSON
        ),
        "VAL_DEV_sampling_seeds": (
            VAL_DEV_SAMPLING_SEEDS
        ),
        "VAL_CONF_sampling_seeds": (
            VAL_CONF_SAMPLING_SEEDS
        ),
        "future_primary_training_seeds": (
            FINAL_PRIMARY_TRAINING_SEEDS
        ),
        "future_replication_training_seeds": (
            FINAL_REPLICATION_TRAINING_SEEDS
        ),
        "robust_VAL_DEV_pass_requirement": (
            "3/3 sampling seeds"
        ),
        "robust_VAL_CONF_pass_requirement": (
            "3/3 sampling seeds"
        ),
        "terminal_failure_policy_locked": (
            True
        ),
        "candidate_selection_executed": (
            False
        ),
        "candidate_checkpoint_evaluated": (
            False
        ),
        "governance": {
            "new_continuation_training_executed": (
                False
            ),
            "Validation_EEG_accessed": (
                True
            ),
            "Final_Test_accessed": (
                False
            ),
            "thresholds_changed": (
                False
            ),
            "formal_model_freeze_decision_made": (
                False
            ),
            "frozen_B1_unchanged": (
                True
            ),
        },
        "next": (
            "Run s2_70 on the two predeclared fresh training streams. "
            "For each E1-E5 checkpoint, evaluate all 3 VAL-DEV sampling "
            "seeds versus matching Frozen-B1 baselines. Select only among "
            "3/3 robust-eligible checkpoints. Confirm the selected checkpoint "
            "on all 3 VAL-CONF seeds. Both fresh training runs must succeed. "
            "If either fails, continuation-B2 is terminally closed."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
    )

    print(
        "Frozen B1 unchanged: PASS"
    )

    print(
        "Candidate evaluated: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        f"summary JSON: "
        f"{SUMMARY_JSON}"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "s2_69 SAMPLING-ROBUST SELECTOR V2 READY"
    )

    print(
        "=" * 80
    )

    print(
        "\nNEXT:"
    )

    print(
        "RUN THE TWO PREDECLARED FRESH TRAINING STREAMS UNDER "
        "THE FINAL 3-SEED VAL-DEV / 3-SEED VAL-CONF SELECTOR."
    )


if __name__ == "__main__":
    main()
