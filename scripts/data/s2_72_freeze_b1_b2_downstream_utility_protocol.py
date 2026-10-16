from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
)
from s2_47_b1_architecture_warmstart_sanity import state_dict_sha256


# ============================================================
# Repository / frozen model paths
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

FROZEN_B1_SELECTION = (
    FROZEN_B1_ROOT
    / "metrics"
    / "b1_frozen_selection_v1.json"
)

FROZEN_B2_ROOT = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b2"
    / "b2_validation_selected_replication_e1_v1"
)

FROZEN_B2_GENERATOR = (
    FROZEN_B2_ROOT
    / "checkpoints"
    / "b2_validation_selected_replication_e1_generator_v1.pt"
)

FROZEN_B2_FULL = (
    FROZEN_B2_ROOT
    / "checkpoints"
    / "b2_validation_selected_replication_e1_full_checkpoint_v1.pt"
)

FROZEN_B2_SELECTION = (
    FROZEN_B2_ROOT
    / "metrics"
    / "b2_frozen_selection_v1.json"
)

FROZEN_B2_MANIFEST = (
    FROZEN_B2_ROOT
    / "metrics"
    / "b2_freeze_manifest_v1.json"
)

FROZEN_B2_HANDOFF = (
    FROZEN_B2_ROOT
    / "handoff"
    / "b2_to_utility_handoff_v1.json"
)


# ============================================================
# s2_72 output tree
# ============================================================

PROTOCOL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "utility"
    / "b1_vs_b2_utility_protocol_v1"
)

METRIC_DIR = PROTOCOL_ROOT / "metrics"

PROTOCOL_JSON = (
    METRIC_DIR
    / "s2_72_b1_vs_b2_downstream_utility_protocol_v1.json"
)

PROTOCOL_TXT = (
    METRIC_DIR
    / "s2_72_b1_vs_b2_downstream_utility_protocol_v1.txt"
)

PROTOCOL_MANIFEST = (
    METRIC_DIR
    / "s2_72_b1_vs_b2_downstream_utility_protocol_manifest_v1.json"
)


# ============================================================
# Frozen data facts
# ============================================================

EXPECTED_SUBJECTS = 198
EXPECTED_TRAIN_ASSIGNED_BLOCKS = 1_980
EXPECTED_TRAIN_ELIGIBLE_BLOCKS = 1_970
EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_VAL_ASSIGNED_BLOCKS = 396
EXPECTED_VAL_WINDOWS = 9_635
EXPECTED_TEST_ASSIGNED_BLOCKS = 792

N_CHANNELS = 19
N_TIMES = 512
SAMPLING_RATE_HZ = 128
WINDOW_SECONDS = 4.0
WINDOW_STRIDE_SAMPLES = 256

EO_LABEL = 0
EC_LABEL = 1


# ============================================================
# Bulk synthetic generation contract
# ============================================================

# One 1x pool per frozen generator.
#
# Pool construction is matched at subject x state level:
# n_synth(subject,state) == n_usable_real_train(subject,state)
#
# Therefore each model produces exactly EXPECTED_TRAIN_WINDOWS samples.
SYNTHETIC_POOL_RATIO = 1.0
EXPECTED_SYNTHETIC_WINDOWS_PER_GENERATOR = EXPECTED_TRAIN_WINDOWS

# 0.5x is a deterministic nested subset of the 1x pool.
# No second diffusion-generation pass is allowed for this ratio.
NESTED_HALF_RATIO = 0.5
HALF_SUBSET_RULE = (
    "For every subject-state stratum with n samples in the frozen 1x pool, "
    "use floor(n/2) indices from a deterministic permutation. "
    "The exact same retained pool indices are used for B1 and B2."
)

# Common-random-number design:
# the complete stochastic sampling stream for a subject/state shard
# must be the same for B1 and B2.
GENERATION_MASTER_SEED = 20_262_201

# subject_index is 0..197 and state_label is 0/1.
GENERATION_SEED_FORMULA = (
    "generation_seed = 20262201 + 2 * subject_index + eye_state_label"
)

HALF_SUBSET_MASTER_SEED = 20_262_251

GENERATION_BATCH_SIZE = 64
DIFFUSION_STEPS = 1000
GENERATION_PRECISION = "FP32"
SAMPLER = "ancestral DDPM"
PREDICTION_TARGET = "v"

# Sharded storage is deliberate: generation is long and must be resumable.
SYNTHETIC_STORAGE = {
    "format": "one compressed NPZ shard per generator x subject x eye_state",
    "dtype": "float32",
    "shape_per_shard": "(n_subject_state, 19, 512)",
    "metadata_required": [
        "generator_name",
        "generator_model_state_sha256",
        "subject_id",
        "subject_index",
        "eye_state",
        "eye_state_label",
        "generation_seed",
        "sample_indices",
        "sampling_rate_hz",
        "n_channels",
        "n_times",
    ],
    "atomic_write": True,
    "resume_rule": (
        "A completed shard may be reused only if its metadata, expected shape, "
        "sample count, generator hash, and file SHA256 agree with its manifest. "
        "A partial or unverifiable shard must be regenerated from its "
        "predeclared subject-state seed."
    ),
    "monolithic_duplicate_pool_files_for_0p5x": False,
    "nested_0p5x_is_index_manifest_only": True,
}


# ============================================================
# Utility classifier contract
# ============================================================

CLASSIFIER_FAMILIES = {
    "LOGBANDPOWER_LOGREG": {
        "role": "classical spectral utility classifier",
        "recipe": (
            "Reuse the exact previously established real-only "
            "LogBandpower + LogisticRegression implementation and fixed "
            "hyperparameters. No new hyperparameter search is allowed."
        ),
        "learned_preprocessing_rule": (
            "Any learned feature scaling must be fit only on that condition's "
            "training data and then applied unchanged to Validation/Test."
        ),
        "training_seeds": [20_262_301],
        "threshold_rule": "fixed classifier default decision threshold; no threshold tuning",
    },
    "OAS_RIEMANN_LINEAR_SVM": {
        "role": "covariance/Riemannian utility classifier",
        "recipe": (
            "Reuse the exact previously established OAS/Riemannian "
            "+ linear-SVM implementation and fixed hyperparameters. "
            "No new hyperparameter search is allowed."
        ),
        "learned_preprocessing_rule": (
            "OAS/covariance transforms, reference geometry, tangent-space "
            "mapping, and any feature scaling must be fit only on that "
            "condition's training data."
        ),
        "training_seeds": [20_262_301],
        "threshold_rule": "fixed zero decision boundary; no threshold tuning",
    },
    "EEGNETV4": {
        "role": "primary downstream utility classifier",
        "recipe": (
            "Reuse the exact previously established EEGNetv4 architecture, "
            "optimizer family, batch size, and fixed non-budget "
            "hyperparameters. No architecture or hyperparameter tuning is "
            "allowed during utility experiments."
        ),
        "training_seeds": [
            20_262_301,
            20_262_302,
            20_262_303,
        ],
        "optimization_budget_rule": (
            "Use a fixed number of optimizer updates for every training "
            "condition. The budget equals 24 complete epochs over the "
            "47,835-window Real Train set using the original EEGNetv4 "
            "baseline batch size. For larger augmented datasets, cycle/"
            "reshuffle the loader only until the same update budget is "
            "reached. Validation must NOT be used for early stopping or "
            "checkpoint selection."
        ),
        "checkpoint_rule": (
            "Only the final model state at the fixed update budget is used. "
            "No best-Validation checkpoint selection."
        ),
        "threshold_rule": "probability threshold 0.5; no threshold tuning",
    },
}

REFERENCE_REAL_ONLY_VALIDATION_ANCHORS = {
    "note": (
        "Historical development anchors only; not pass/fail thresholds "
        "for the new utility experiment."
    ),
    "LOGBANDPOWER_LOGREG_subject_macro_BA": 0.833953,
    "OAS_RIEMANN_LINEAR_SVM_subject_macro_BA": 0.803087,
    "EEGNETV4_subject_macro_BA": 0.868417,
    "EEGNETV4_historical_best_epoch": 24,
}


# ============================================================
# Predeclared training conditions
# ============================================================

UTILITY_CONDITIONS = [
    {
        "id": "REAL_ONLY",
        "short_name": "TRTR",
        "train_sources": ["Real Train"],
        "synthetic_generator": None,
        "synthetic_ratio": 0.0,
        "role": "real-only baseline",
    },
    {
        "id": "REAL_DUPLICATE_1X_CONTROL",
        "short_name": "Real+Real duplicate",
        "train_sources": ["Real Train", "exact duplicate of Real Train"],
        "synthetic_generator": None,
        "synthetic_ratio": 0.0,
        "role": (
            "size/weighting sanity control; adds no new EEG information"
        ),
    },
    {
        "id": "TSTR_B1_1X",
        "short_name": "TSTR-B1",
        "train_sources": ["B1 synthetic 1x"],
        "synthetic_generator": "B1",
        "synthetic_ratio": 1.0,
        "role": "synthetic-only B1 utility",
    },
    {
        "id": "TSTR_B2_1X",
        "short_name": "TSTR-B2",
        "train_sources": ["B2 synthetic 1x"],
        "synthetic_generator": "B2",
        "synthetic_ratio": 1.0,
        "role": "synthetic-only B2 utility",
    },
    {
        "id": "REAL_PLUS_B1_0P5X",
        "short_name": "Real+B1 0.5x",
        "train_sources": ["Real Train", "nested B1 synthetic 0.5x"],
        "synthetic_generator": "B1",
        "synthetic_ratio": 0.5,
        "role": "B1 augmentation dose",
    },
    {
        "id": "REAL_PLUS_B1_1X",
        "short_name": "Real+B1 1x",
        "train_sources": ["Real Train", "B1 synthetic 1x"],
        "synthetic_generator": "B1",
        "synthetic_ratio": 1.0,
        "role": "full B1 augmentation",
    },
    {
        "id": "REAL_PLUS_B2_0P5X",
        "short_name": "Real+B2 0.5x",
        "train_sources": ["Real Train", "nested B2 synthetic 0.5x"],
        "synthetic_generator": "B2",
        "synthetic_ratio": 0.5,
        "role": "B2 augmentation dose",
    },
    {
        "id": "REAL_PLUS_B2_1X",
        "short_name": "Real+B2 1x",
        "train_sources": ["Real Train", "B2 synthetic 1x"],
        "synthetic_generator": "B2",
        "synthetic_ratio": 1.0,
        "role": "primary B2 augmentation condition",
    },
]


# ============================================================
# Evaluation / statistics contract
# ============================================================

PRIMARY_METRIC = "subject-macro balanced accuracy"

SECONDARY_METRICS = [
    "subject-macro ROC-AUC",
    "global balanced accuracy",
    "global ROC-AUC",
    "per-subject balanced-accuracy distribution",
]

PRIMARY_CLASSIFIER = "EEGNETV4"

PRIMARY_COMPARISONS = [
    {
        "id": "P1_TSTR_B2_VS_B1",
        "left": "TSTR_B2_1X",
        "right": "TSTR_B1_1X",
        "interpretation": (
            "Does the selected B2 generator provide more class-relevant "
            "synthetic-only training data than B1?"
        ),
    },
    {
        "id": "P2_AUG_B2_1X_VS_REAL",
        "left": "REAL_PLUS_B2_1X",
        "right": "REAL_ONLY",
        "interpretation": (
            "Does 1x B2 synthetic augmentation improve the primary "
            "downstream classifier over real-only training?"
        ),
    },
    {
        "id": "P3_AUG_B2_1X_VS_B1_1X",
        "left": "REAL_PLUS_B2_1X",
        "right": "REAL_PLUS_B1_1X",
        "interpretation": (
            "Does B2 provide better augmentation utility than B1 at "
            "the same synthetic-data ratio?"
        ),
    },
]

STATISTICS = {
    "bootstrap_unit": "subject",
    "bootstrap_repetitions": 5000,
    "bootstrap_seed": 20_262_401,
    "confidence_interval": "paired nonparametric 95% subject bootstrap",
    "EEGNET_seed_aggregation": (
        "For each condition, compute metrics separately for all 3 frozen "
        "training seeds and report mean +/- population SD. For paired "
        "condition comparisons, each bootstrap replicate resamples subjects, "
        "recomputes the metric for each matched EEGNet seed, averages across "
        "the 3 seeds, and then takes the paired condition difference."
    ),
    "classical_seed_aggregation": "single fixed fit per condition",
    "strong_positive_evidence_rule": (
        "For a predeclared primary comparison, call the evidence strong "
        "positive only if the paired 95% bootstrap CI for the primary "
        "subject-macro BA difference is entirely above 0."
    ),
    "directional_positive_rule": (
        "If the point estimate is above 0 but the paired 95% CI includes 0, "
        "report directional positive evidence, not a definitive improvement."
    ),
    "no_posthoc_threshold_change": True,
}


# ============================================================
# Governance
# ============================================================

FINAL_TEST_POLICY = {
    "current_status": "SEALED",
    "may_be_opened_in_s2_72": False,
    "may_be_opened_during_bulk_generation": False,
    "may_be_opened_during_validation_utility_training": False,
    "opening_preconditions": [
        "B1 1x synthetic pool complete and integrity-verified",
        "B2 1x synthetic pool complete and integrity-verified",
        "nested 0.5x index manifests frozen",
        "all utility classifier artifacts for every predeclared condition frozen",
        "all classifier recipe/hyperparameter audits complete",
        "Validation utility report complete",
        "no condition is added, removed, or retuned based on Validation performance",
        "final-test execution script records hashes of every frozen input artifact",
    ],
    "final_test_conditions": [
        condition["id"]
        for condition in UTILITY_CONDITIONS
    ],
    "final_test_classifier_families": list(
        CLASSIFIER_FAMILIES.keys()
    ),
    "train_plus_validation_refit": False,
    "rule": (
        "All final classifiers remain trained only on the frozen Real Train "
        "and/or frozen synthetic Train pools. Validation is development-only. "
        "The same frozen classifier artifacts are evaluated on Final Test."
    ),
}

NO_TUNING_RULES = [
    "Do not retrain, reselect, or modify frozen B1.",
    "Do not retrain, reselect, or modify frozen B2.",
    "Do not add a new generator after s2_72.",
    "Do not add a 2x synthetic ratio after seeing utility results.",
    "Do not generate a separate 0.5x diffusion pool.",
    "Do not tune classifier hyperparameters on the new utility Validation results.",
    "Do not tune decision thresholds.",
    "Do not use Validation for EEGNet early stopping/checkpoint selection.",
    "Do not drop an unfavorable predeclared condition before Final Test.",
    "Do not use subject identity as a classifier input feature.",
    "Do not fit any learned transform or scaler on Validation or Test.",
    "Do not access Final Test until all preconditions are satisfied.",
]

DATA_HANDLING_RULES = {
    "task": "binary eye-state classification",
    "EO_label": EO_LABEL,
    "EC_label": EC_LABEL,
    "evaluation_regime": (
        "seen-subject, within-subject complete-block holdout"
    ),
    "classifier_input": "EEG only: 19 channels x 512 samples",
    "subject_id_as_classifier_feature": False,
    "subject_id_use": (
        "allowed only for subject-stratified synthetic matching, bookkeeping, "
        "and subject-macro evaluation"
    ),
    "normalization": (
        "the same frozen Train-only global 19-channel mean/std used by "
        "Scenario 2; no per-subject, per-state, or per-window normalization"
    ),
    "real_train_windows": EXPECTED_TRAIN_WINDOWS,
    "real_validation_windows": EXPECTED_VAL_WINDOWS,
    "real_test_access_at_protocol_freeze": False,
    "synthetic_training_only": True,
    "synthetic_samples_never_added_to_validation_or_test": True,
    "subject_state_pool_matching": (
        "For each frozen subject and eye state, synthetic 1x count must equal "
        "the number of usable eligible Real Train windows in that exact "
        "subject-state stratum."
    ),
}


# ============================================================
# Planned remaining stages
# ============================================================

REMAINING_STAGE_PLAN = [
    {
        "stage": "s2_73",
        "name": "resumable_bulk_synthetic_generation",
        "action": (
            "Generate one subject/state-matched 1x pool for frozen B1 and "
            "one for frozen B2 using common random streams and resumable "
            "atomic shards."
        ),
        "Final_Test_access": False,
    },
    {
        "stage": "s2_74",
        "name": "synthetic_pool_integrity_and_nested_subset_freeze",
        "action": (
            "Audit counts, hashes, finite values, scale sanity, subject/state "
            "coverage; freeze deterministic nested 0.5x index manifests."
        ),
        "Final_Test_access": False,
    },
    {
        "stage": "s2_75",
        "name": "utility_classifier_training_and_validation_development_report",
        "action": (
            "Train every predeclared classifier/condition on Train sources "
            "only, freeze classifier artifacts, and evaluate on real "
            "Validation as development evidence. No condition selection."
        ),
        "Final_Test_access": False,
    },
    {
        "stage": "s2_76",
        "name": "final_test_preflight_and_utility_artifact_freeze",
        "action": (
            "Verify hashes, classifier completeness, protocol compliance, "
            "and confirm that no configuration changed after Validation."
        ),
        "Final_Test_access": False,
    },
    {
        "stage": "s2_77",
        "name": "single_final_test_evaluation",
        "action": (
            "Open Final Test for the first time and evaluate all frozen "
            "predeclared classifier artifacts on the real Test set."
        ),
        "Final_Test_access": True,
    },
    {
        "stage": "s2_78",
        "name": "scenario2_final_summary",
        "action": (
            "Produce final B1/B2 fidelity + utility tables, statistical "
            "comparisons, limitations, manifests, and report-ready summary."
        ),
        "Final_Test_access": False,
    },
]


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def atomic_json_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(str(path) + ".tmp")

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


def atomic_text_save(
    text: str,
    path: Path,
) -> None:
    temp = Path(str(path) + ".tmp")

    with temp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(text)

    os.replace(
        temp,
        path,
    )


def save_exact_or_verify_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
    if path.is_file():
        existing = read_json(path)

        if existing != payload:
            raise RuntimeError(
                f"Conflicting frozen protocol artifact already exists: {path}"
            )

        print(
            f"Existing protocol artifact verified: {path.name}"
        )

        return

    atomic_json_save(
        payload,
        path,
    )


# ============================================================
# Frozen model audit
# ============================================================

def audit_frozen_models() -> dict[str, Any]:
    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 1 — AUDIT FROZEN B1 + B2"
    )
    print(
        "=" * 80
    )

    b1_selection = read_json(
        FROZEN_B1_SELECTION
    )

    b2_selection = read_json(
        FROZEN_B2_SELECTION
    )

    b2_manifest = read_json(
        FROZEN_B2_MANIFEST
    )

    b2_handoff = read_json(
        FROZEN_B2_HANDOFF
    )

    if (
        b1_selection.get(
            "status"
        )
        != "B1_SELECTION_FROZEN"
    ):
        raise RuntimeError(
            "B1 selection is not frozen."
        )

    if (
        b2_selection.get(
            "status"
        )
        != "B2_SELECTION_FROZEN"
    ):
        raise RuntimeError(
            "B2 selection is not frozen."
        )

    if (
        b2_manifest.get(
            "status"
        )
        != "B2_FREEZE_MANIFEST_COMPLETE"
    ):
        raise RuntimeError(
            "B2 freeze manifest is incomplete."
        )

    if (
        b2_handoff.get(
            "status"
        )
        != "B2_TO_UTILITY_HANDOFF_FROZEN"
    ):
        raise RuntimeError(
            "B2 utility handoff is not frozen."
        )

    b1_file_hash = sha256_file(
        FROZEN_B1_GENERATOR
    )

    b2_file_hash = sha256_file(
        FROZEN_B2_GENERATOR
    )

    b2_full_hash = sha256_file(
        FROZEN_B2_FULL
    )

    b1_package = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    b2_package = torch.load(
        FROZEN_B2_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    b1_state_hash = state_dict_sha256(
        b1_package[
            "model_state_dict"
        ]
    )

    b2_state_hash = state_dict_sha256(
        b2_package[
            "model_state_dict"
        ]
    )

    if (
        b1_state_hash
        != b1_package[
            "selected_model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B1 model-state hash mismatch."
        )

    if (
        b2_state_hash
        != b2_package[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B2 model-state hash mismatch."
        )

    if (
        b2_state_hash
        != b2_selection[
            "selected_source"
        ][
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "B2 generator / selection model-state disagreement."
        )

    manifest_generator_hash = (
        b2_manifest[
            "artifacts"
        ][
            "frozen_generator"
        ][
            "sha256"
        ]
    )

    if (
        b2_file_hash
        != manifest_generator_hash
    ):
        raise RuntimeError(
            "B2 generator file hash does not match its freeze manifest."
        )

    manifest_full_hash = (
        b2_manifest[
            "artifacts"
        ][
            "frozen_full_checkpoint"
        ][
            "sha256"
        ]
    )

    if (
        b2_full_hash
        != manifest_full_hash
    ):
        raise RuntimeError(
            "B2 full checkpoint file hash mismatch."
        )

    if (
        b2_handoff[
            "utility_stage_rules"
        ][
            "Final_Test_currently_sealed"
        ]
        is not True
    ):
        raise RuntimeError(
            "B2 handoff does not declare Final Test sealed."
        )

    print(
        "Frozen B1: PASS"
    )
    print(
        f"  model-state SHA256 = {b1_state_hash}"
    )

    print(
        "Frozen B2: PASS"
    )
    print(
        f"  model-state SHA256 = {b2_state_hash}"
    )

    print(
        "Final Test handoff status: SEALED"
    )

    return {
        "B1": {
            "generator_path": str(
                FROZEN_B1_GENERATOR
            ),
            "generator_file_sha256": (
                b1_file_hash
            ),
            "model_state_sha256": (
                b1_state_hash
            ),
        },
        "B2": {
            "generator_path": str(
                FROZEN_B2_GENERATOR
            ),
            "generator_file_sha256": (
                b2_file_hash
            ),
            "full_checkpoint_path": str(
                FROZEN_B2_FULL
            ),
            "full_checkpoint_file_sha256": (
                b2_full_hash
            ),
            "model_state_sha256": (
                b2_state_hash
            ),
            "selection_record_path": str(
                FROZEN_B2_SELECTION
            ),
            "selection_record_sha256": sha256_file(
                FROZEN_B2_SELECTION
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )
    print(
        "SCENARIO 2 — s2_72 FREEZE B1 vs B2 DOWNSTREAM UTILITY PROTOCOL"
    )
    print(
        "=" * 80
    )

    print(
        "\nScientific question:"
    )
    print(
        "  Does frozen subject+state-conditioned synthetic EEG from B1/B2"
    )
    print(
        "  transfer to real eye-state classification, and does B2 provide"
    )
    print(
        "  better synthetic-only and augmentation utility than B1?"
    )

    print(
        "\nThis stage:"
    )
    print(
        "  Training: NO"
    )
    print(
        "  Diffusion generation: NO"
    )
    print(
        "  Classifier fitting: NO"
    )
    print(
        "  Final Test access: NO"
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        FROZEN_B1_GENERATOR,
        FROZEN_B1_SELECTION,
        FROZEN_B2_GENERATOR,
        FROZEN_B2_FULL,
        FROZEN_B2_SELECTION,
        FROZEN_B2_MANIFEST,
        FROZEN_B2_HANDOFF,
        Path(
            FINAL_SPLIT_MANIFEST
        ),
        Path(
            NORMALIZATION_FILE
        ),
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    frozen_models = audit_frozen_models()

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 2 — SNAPSHOT FROZEN DATA CONTRACT"
    )
    print(
        "=" * 80
    )

    data_contract = {
        "subjects": EXPECTED_SUBJECTS,
        "Train": {
            "assigned_blocks": EXPECTED_TRAIN_ASSIGNED_BLOCKS,
            "eligible_blocks": EXPECTED_TRAIN_ELIGIBLE_BLOCKS,
            "usable_windows": EXPECTED_TRAIN_WINDOWS,
        },
        "Validation": {
            "assigned_blocks": EXPECTED_VAL_ASSIGNED_BLOCKS,
            "usable_windows": EXPECTED_VAL_WINDOWS,
            "role": (
                "development-only utility evaluation; "
                "not an independent final performance estimate"
            ),
        },
        "Final_Test": {
            "assigned_blocks": EXPECTED_TEST_ASSIGNED_BLOCKS,
            "status_at_s2_72": "SEALED",
        },
        "window": {
            "channels": N_CHANNELS,
            "samples": N_TIMES,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "seconds": WINDOW_SECONDS,
            "stride_samples": WINDOW_STRIDE_SAMPLES,
        },
        "split_manifest": {
            "path": str(
                FINAL_SPLIT_MANIFEST
            ),
            "sha256": sha256_file(
                Path(
                    FINAL_SPLIT_MANIFEST
                )
            ),
        },
        "normalization": {
            "path": str(
                NORMALIZATION_FILE
            ),
            "sha256": sha256_file(
                Path(
                    NORMALIZATION_FILE
                )
            ),
        },
    }

    print(
        "Frozen Train windows: "
        f"{EXPECTED_TRAIN_WINDOWS}"
    )
    print(
        "Frozen Validation windows: "
        f"{EXPECTED_VAL_WINDOWS}"
    )
    print(
        "Final Test assigned blocks: "
        f"{EXPECTED_TEST_ASSIGNED_BLOCKS} (SEALED)"
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 3 — FREEZE SYNTHETIC-POOL DESIGN"
    )
    print(
        "=" * 80
    )

    print(
        "B1 synthetic pool: 1x matched Train distribution"
    )
    print(
        "B2 synthetic pool: 1x matched Train distribution"
    )
    print(
        "0.5x: nested deterministic subset; NO extra diffusion generation"
    )
    print(
        "B1/B2 sampling: common random streams per subject-state shard"
    )
    print(
        "Storage: atomic resumable subject-state NPZ shards"
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 4 — FREEZE CLASSIFIER / CONDITION MATRIX"
    )
    print(
        "=" * 80
    )

    for condition in UTILITY_CONDITIONS:
        print(
            f"  {condition['id']}"
        )

    print(
        "\nClassifier families:"
    )

    for classifier_name in CLASSIFIER_FAMILIES:
        marker = (
            " [PRIMARY]"
            if classifier_name
            == PRIMARY_CLASSIFIER
            else ""
        )

        print(
            f"  {classifier_name}{marker}"
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 5 — FREEZE EVALUATION / FINAL-TEST GOVERNANCE"
    )
    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_72_B1_VS_B2_UTILITY_PROTOCOL_FROZEN"
        ),
        "protocol_version": (
            "b1_vs_b2_utility_protocol_v1"
        ),
        "scientific_question": (
            "Evaluate whether frozen subject+state-conditioned synthetic EEG "
            "transfers to real eye-state classification and whether B2 has "
            "better downstream utility than B1."
        ),
        "frozen_models": frozen_models,
        "data_contract": data_contract,
        "data_handling_rules": DATA_HANDLING_RULES,
        "synthetic_generation": {
            "pool_ratio_per_generator": (
                SYNTHETIC_POOL_RATIO
            ),
            "expected_windows_per_generator": (
                EXPECTED_SYNTHETIC_WINDOWS_PER_GENERATOR
            ),
            "subject_state_count_matching": (
                DATA_HANDLING_RULES[
                    "subject_state_pool_matching"
                ]
            ),
            "generation_master_seed": (
                GENERATION_MASTER_SEED
            ),
            "generation_seed_formula": (
                GENERATION_SEED_FORMULA
            ),
            "common_random_numbers_across_B1_B2": (
                True
            ),
            "common_rng_scope": (
                "entire ancestral reverse-sampling RNG stream for each "
                "matched subject-state shard"
            ),
            "batch_size": (
                GENERATION_BATCH_SIZE
            ),
            "diffusion_steps": (
                DIFFUSION_STEPS
            ),
            "sampler": (
                SAMPLER
            ),
            "prediction_target": (
                PREDICTION_TARGET
            ),
            "precision": (
                GENERATION_PRECISION
            ),
            "half_ratio": (
                NESTED_HALF_RATIO
            ),
            "half_subset_master_seed": (
                HALF_SUBSET_MASTER_SEED
            ),
            "half_subset_rule": (
                HALF_SUBSET_RULE
            ),
            "storage": (
                SYNTHETIC_STORAGE
            ),
            "2x_pool_in_this_protocol": (
                False
            ),
            "reason_for_no_2x": (
                "The 1x pool already supports TSTR, full 1x augmentation, "
                "and nested 0.5x augmentation. A 2x pool would approximately "
                "double diffusion generation cost and is not required for "
                "the predeclared primary B1-vs-B2 utility questions."
            ),
        },
        "classifier_families": (
            CLASSIFIER_FAMILIES
        ),
        "historical_real_only_validation_anchors": (
            REFERENCE_REAL_ONLY_VALIDATION_ANCHORS
        ),
        "utility_conditions": (
            UTILITY_CONDITIONS
        ),
        "primary_classifier": (
            PRIMARY_CLASSIFIER
        ),
        "primary_metric": (
            PRIMARY_METRIC
        ),
        "secondary_metrics": (
            SECONDARY_METRICS
        ),
        "primary_comparisons": (
            PRIMARY_COMPARISONS
        ),
        "statistics": (
            STATISTICS
        ),
        "validation_role": (
            "Development evidence and implementation sanity only. "
            "Because B2 selection already used Validation evidence, "
            "new Validation utility numbers must not be presented as "
            "independent final performance estimates."
        ),
        "final_test_policy": (
            FINAL_TEST_POLICY
        ),
        "no_tuning_rules": (
            NO_TUNING_RULES
        ),
        "remaining_stage_plan": (
            REMAINING_STAGE_PLAN
        ),
        "governance": {
            "B1_frozen_before_protocol": (
                True
            ),
            "B2_frozen_before_protocol": (
                True
            ),
            "new_generator_training_allowed": (
                False
            ),
            "new_generator_selection_allowed": (
                False
            ),
            "new_classifier_hyperparameter_search_allowed": (
                False
            ),
            "post_validation_condition_pruning_allowed": (
                False
            ),
            "Final_Test_accessed_by_s2_72": (
                False
            ),
        },
    }

    save_exact_or_verify_json(
        protocol,
        PROTOCOL_JSON,
    )

    protocol_sha = sha256_file(
        PROTOCOL_JSON
    )

    summary_text = f"""SCENARIO 2 — s2_72 B1 vs B2 DOWNSTREAM UTILITY PROTOCOL
================================================================

STATUS
------
FROZEN.

Scientific question
-------------------
Does synthetic EEG from frozen B1/B2 transfer to real eye-state
classification, and does B2 provide better downstream utility than B1?

Frozen generators
-----------------
B1 model-state SHA256:
  {frozen_models['B1']['model_state_sha256']}

B2 model-state SHA256:
  {frozen_models['B2']['model_state_sha256']}

Data regime
-----------
Subjects:
  {EXPECTED_SUBJECTS}

Real Train usable windows:
  {EXPECTED_TRAIN_WINDOWS}

Real Validation usable windows:
  {EXPECTED_VAL_WINDOWS}

Final Test:
  SEALED

Evaluation regime:
  seen-subject / within-subject complete-block holdout

Synthetic pool design
---------------------
Per generator:
  exactly 1x relative to usable Real Train
  expected total = {EXPECTED_SYNTHETIC_WINDOWS_PER_GENERATOR} windows

Matching:
  exact subject x eye-state Train-window count matching

B1/B2 fairness:
  same complete stochastic sampling stream per subject/state shard

Generation master seed:
  {GENERATION_MASTER_SEED}

Seed formula:
  {GENERATION_SEED_FORMULA}

Sampler:
  ancestral DDPM / v-prediction / {DIFFUSION_STEPS} steps / FP32

Storage:
  atomic resumable NPZ shards, one generator x subject x eye state

0.5x:
  deterministic nested subset of the 1x pool
  NO second diffusion generation

2x:
  NOT PART OF THIS FROZEN PROTOCOL

Utility conditions
------------------
REAL_ONLY
REAL_DUPLICATE_1X_CONTROL
TSTR_B1_1X
TSTR_B2_1X
REAL_PLUS_B1_0P5X
REAL_PLUS_B1_1X
REAL_PLUS_B2_0P5X
REAL_PLUS_B2_1X

Classifier families
-------------------
1. LogBandpower + Logistic Regression
2. OAS/Riemannian + Linear SVM
3. EEGNetv4 [PRIMARY]

Classifier tuning
-----------------
NO new hyperparameter search.
Reuse exact established baseline recipes.

EEGNetv4
--------
Training seeds:
  20262301
  20262302
  20262303

No Validation early stopping.
No best-Validation checkpoint selection.

All conditions use the same optimizer-update budget:
  equivalent to 24 complete Real-Train epochs using the original
  EEGNetv4 baseline batch size.

Primary metric
--------------
subject-macro balanced accuracy

Secondary metrics
-----------------
subject-macro ROC-AUC
global balanced accuracy
global ROC-AUC
per-subject BA distribution

Primary EEGNetv4 comparisons
----------------------------
P1:
  TSTR_B2_1X - TSTR_B1_1X

P2:
  REAL_PLUS_B2_1X - REAL_ONLY

P3:
  REAL_PLUS_B2_1X - REAL_PLUS_B1_1X

Statistics
----------
5000 paired subject bootstrap repetitions
bootstrap seed = 20262401
95% paired bootstrap CI

Final Test rule
---------------
Validation is development-only.

ALL predeclared conditions are carried forward.
No unfavorable condition may be dropped after seeing Validation.

Before Final Test:
  synthetic pools must be frozen
  0.5x subset manifests must be frozen
  all classifier artifacts must be trained and frozen
  protocol compliance must pass
  no hyperparameter/condition change is allowed

No Train+Validation refit.

The exact same frozen classifier artifacts are then evaluated once on
the real Final Test.

Next stages
-----------
s2_73  Resumable bulk B1/B2 1x generation
s2_74  Pool integrity audit + nested 0.5x freeze
s2_75  Utility classifier training + Validation development report
s2_76  Final-Test preflight / artifact freeze
s2_77  Single Final-Test evaluation
s2_78  Final Scenario-2 summary

Protocol JSON SHA256
--------------------
{protocol_sha}
"""

    if PROTOCOL_TXT.is_file():
        existing_text = PROTOCOL_TXT.read_text(
            encoding="utf-8"
        )

        if (
            existing_text
            != summary_text
        ):
            raise RuntimeError(
                "Conflicting frozen protocol text already exists."
            )

        print(
            f"Existing text artifact verified: "
            f"{PROTOCOL_TXT.name}"
        )
    else:
        atomic_text_save(
            summary_text,
            PROTOCOL_TXT,
        )

    manifest = {
        "status": (
            "S2_72_UTILITY_PROTOCOL_MANIFEST_COMPLETE"
        ),
        "protocol_status": (
            protocol[
                "status"
            ]
        ),
        "protocol_json": {
            "path": str(
                PROTOCOL_JSON
            ),
            "sha256": (
                protocol_sha
            ),
        },
        "protocol_text": {
            "path": str(
                PROTOCOL_TXT
            ),
            "sha256": sha256_file(
                PROTOCOL_TXT
            ),
        },
        "frozen_inputs": {
            "B1_generator": {
                "path": str(
                    FROZEN_B1_GENERATOR
                ),
                "sha256": (
                    frozen_models[
                        "B1"
                    ][
                        "generator_file_sha256"
                    ]
                ),
                "model_state_sha256": (
                    frozen_models[
                        "B1"
                    ][
                        "model_state_sha256"
                    ]
                ),
            },
            "B2_generator": {
                "path": str(
                    FROZEN_B2_GENERATOR
                ),
                "sha256": (
                    frozen_models[
                        "B2"
                    ][
                        "generator_file_sha256"
                    ]
                ),
                "model_state_sha256": (
                    frozen_models[
                        "B2"
                    ][
                        "model_state_sha256"
                    ]
                ),
            },
            "B2_selection": {
                "path": str(
                    FROZEN_B2_SELECTION
                ),
                "sha256": sha256_file(
                    FROZEN_B2_SELECTION
                ),
            },
            "split_manifest": (
                data_contract[
                    "split_manifest"
                ]
            ),
            "normalization": (
                data_contract[
                    "normalization"
                ]
            ),
        },
        "locked_decisions": {
            "synthetic_pool_max_ratio": (
                1.0
            ),
            "nested_ratios": [
                0.5,
                1.0,
            ],
            "utility_condition_count": len(
                UTILITY_CONDITIONS
            ),
            "classifier_families": list(
                CLASSIFIER_FAMILIES.keys()
            ),
            "primary_classifier": (
                PRIMARY_CLASSIFIER
            ),
            "primary_metric": (
                PRIMARY_METRIC
            ),
            "Final_Test_conditions_count": len(
                FINAL_TEST_POLICY[
                    "final_test_conditions"
                ]
            ),
            "Final_Test_currently_sealed": (
                True
            ),
        },
    }

    save_exact_or_verify_json(
        manifest,
        PROTOCOL_MANIFEST,
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "s2_72 UTILITY PROTOCOL FREEZE COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nProtocol status: FROZEN"
    )

    print(
        f"Protocol JSON SHA256: {protocol_sha}"
    )

    print(
        f"Conditions frozen: {len(UTILITY_CONDITIONS)}"
    )

    print(
        f"Classifier families frozen: {len(CLASSIFIER_FAMILIES)}"
    )

    print(
        "Synthetic max pool per generator: 1.0x"
    )

    print(
        "Nested 0.5x subset: YES"
    )

    print(
        "2.0x generation: NO"
    )

    print(
        "Common B1/B2 sampling noise streams: YES"
    )

    print(
        "Resumable subject/state generation shards: YES"
    )

    print(
        "Validation-based condition pruning: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "s2_73 — RESUMABLE BULK SYNTHETIC GENERATION "
        "FOR FROZEN B1 AND B2."
    )


if __name__ == "__main__":
    main()
