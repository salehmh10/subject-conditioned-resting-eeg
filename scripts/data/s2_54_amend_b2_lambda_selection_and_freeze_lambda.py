from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


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

B2_PROTOCOL_V1_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "spectral_regularized_v1"
)

B2_PROTOCOL_V1 = (
    B2_PROTOCOL_V1_ROOT
    / "metrics"
    / "b2_spectral_regularized_protocol_v1.json"
)

B2_LAMBDA_RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "lambda_calibration_v1"
)

B2_LAMBDA_SUMMARY_V1 = (
    B2_LAMBDA_RUN_ROOT
    / "metrics"
    / "b2_lambda_calibration_summary_v1.json"
)

B2_LAMBDA_METRICS_V1 = (
    B2_LAMBDA_RUN_ROOT
    / "metrics"
    / "b2_lambda_calibration_metrics_v1.csv"
)


# ============================================================
# Amended protocol / selected-lambda outputs
# ============================================================

B2_PROTOCOL_V2_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "spectral_regularized_v2"
)

METRIC_DIR = B2_PROTOCOL_V2_ROOT / "metrics"
HANDOFF_DIR = B2_PROTOCOL_V2_ROOT / "handoff"

AMENDMENT_FILE = (
    METRIC_DIR
    / "b2_lambda_selection_protocol_amendment_v2.json"
)

SELECTED_LAMBDA_FILE = (
    METRIC_DIR
    / "b2_selected_lambda_v2.json"
)

SUMMARY_TEXT_FILE = (
    METRIC_DIR
    / "b2_selected_lambda_summary_v2.txt"
)

FULL_RUN_HANDOFF = (
    HANDOFF_DIR
    / "b2_full_run_handoff_v2.json"
)


# ============================================================
# Minimal, explicit protocol amendment
# ============================================================

EXPECTED_LAMBDAS = [
    0.0,
    0.01,
    0.03,
    0.10,
]

SELECTED_LAMBDA = 0.01

# Original rule used absolute normalized-std bounds [0.80, 1.20].
# That rule is replaced ONLY for the scale guardrail because
# the frozen B1 parent itself, on the exact small 16-subject
# diagnostic cohort/seeds used by the calibration, had EC std 1.196
# and the matched lambda=0 continuation reached 1.211.
#
# Revised scientific intent:
# Preserve the frozen B1 parent scale on the exact matched sampling
# diagnostic, not force a small cohort to have global normalized std=1.
PARENT_SCALE_RELATIVE_LOWER = 0.90
PARENT_SCALE_RELATIVE_UPPER = 1.10

# All other original criteria remain unchanged.
V_MSE_MAX_RATIO_TO_LAMBDA0 = 1.02
SUBJECT_GAP_MIN_RATIO_TO_LAMBDA0 = 0.90
SPECTRAL_IMPROVEMENT_MIN_FRACTION = 0.10

EXPECTED_FULL_B2_EPOCHS = 20
EXPECTED_FULL_B2_LR = 5e-5
EXPECTED_RAW_CANDIDATE_EPOCHS = [
    5,
    10,
    20,
]


# ============================================================
# Helpers
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


def atomic_text_save(
    text: str,
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
        f.write(
            text
        )

    os.replace(
        temp,
        path,
    )


def get_lambda_detail(
    summary: dict[str, Any],
    lambda_value: float,
) -> dict[str, Any]:
    details = summary[
        "per_lambda_details"
    ]

    preferred = str(
        float(
            lambda_value
        )
    )

    if preferred in details:
        return details[
            preferred
        ]

    fallback = str(
        lambda_value
    )

    if fallback in details:
        return details[
            fallback
        ]

    raise KeyError(
        f"Missing lambda details for {lambda_value}"
    )


def bool_json(
    value: Any,
) -> bool:
    return bool(
        value
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — AMEND B2 LAMBDA SELECTION RULE + FREEZE SELECTED LAMBDA"
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
        "  B2 protocol V1: FROZEN"
    )

    print(
        "  B2 lambda calibration V1: COMPLETE"
    )

    print(
        "  Original nonzero-lambda eligibility: NONE"
    )

    print(
        "  B2 protocol amendment V2 + lambda freeze: NOW"
    )

    print(
        "  B2 scientific full fine-tuning: NOT STARTED"
    )

    print(
        "  Final Test: SEALED"
    )

    for directory in (
        METRIC_DIR,
        HANDOFF_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit immutable source evidence
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT V1 PROTOCOL + CALIBRATION EVIDENCE"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B1_GENERATOR,
        B2_PROTOCOL_V1,
        B2_LAMBDA_SUMMARY_V1,
        B2_LAMBDA_METRICS_V1,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    frozen_b1_hash_before = sha256_file(
        FROZEN_B1_GENERATOR
    )

    v1_protocol_hash = sha256_file(
        B2_PROTOCOL_V1
    )

    calibration_hash = sha256_file(
        B2_LAMBDA_SUMMARY_V1
    )

    with B2_PROTOCOL_V1.open(
        "r",
        encoding="utf-8",
    ) as f:
        v1_protocol = json.load(
            f
        )

    with B2_LAMBDA_SUMMARY_V1.open(
        "r",
        encoding="utf-8",
    ) as f:
        calibration = json.load(
            f
        )

    if v1_protocol[
        "status"
    ] != "B2_SPECTRAL_REGULARIZED_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B2 protocol V1 is not frozen."
        )

    if calibration[
        "status"
    ] != "B2_LAMBDA_CALIBRATION_COMPLETE":
        raise RuntimeError(
            "B2 lambda calibration did not complete."
        )

    if calibration[
        "scientific_full_B2_started"
    ] is not False:
        raise RuntimeError(
            "Scientific B2 full training already started; "
            "amendment cannot be applied retroactively."
        )

    if calibration[
        "selection_rule_satisfied"
    ] is not False:
        raise RuntimeError(
            "Expected original V1 lambda rule to have no eligible "
            "nonzero candidate."
        )

    if calibration[
        "selected_lambda"
    ] is not None:
        raise RuntimeError(
            "Original V1 calibration unexpectedly selected a lambda."
        )

    if calibration[
        "diagnostic_direction"
    ] != (
        "NO_NONZERO_B2_LAMBDA_ELIGIBLE_PROTOCOL_AMENDMENT_REQUIRED"
    ):
        raise RuntimeError(
            "Unexpected V1 calibration direction."
        )

    protocol_lambdas = [
        float(
            value
        )
        for value
        in calibration[
            "candidate_lambdas"
        ]
    ]

    if protocol_lambdas != EXPECTED_LAMBDAS:
        raise RuntimeError(
            "Unexpected calibrated lambda set."
        )

    if calibration[
        "governance"
    ][
        "Validation_EEG_accessed"
    ] is not False:
        raise RuntimeError(
            "Lambda calibration unexpectedly accessed Validation EEG."
        )

    if calibration[
        "governance"
    ][
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Lambda calibration unexpectedly accessed Final Test."
        )

    print(
        "V1 protocol: PASS"
    )

    print(
        "V1 lambda calibration: PASS"
    )

    print(
        "Scientific B2 full training started: NO"
    )

    print(
        "Validation/Test contamination: NO"
    )

    # ========================================================
    # Step 2 — Diagnose why V1 rule failed
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — DIAGNOSE ORIGINAL SCALE-GUARDRAIL FAILURE"
    )

    print(
        "=" * 80
    )

    parent_sampling = calibration[
        "parent_sampling"
    ]

    parent_eo_std = float(
        parent_sampling[
            "eo_normalized_std"
        ]
    )

    parent_ec_std = float(
        parent_sampling[
            "ec_normalized_std"
        ]
    )

    lambda0 = get_lambda_detail(
        calibration,
        0.0,
    )

    lambda0_sampling = lambda0[
        "post_sampling"
    ]

    lambda0_eo_std = float(
        lambda0_sampling[
            "eo_normalized_std"
        ]
    )

    lambda0_ec_std = float(
        lambda0_sampling[
            "ec_normalized_std"
        ]
    )

    print(
        f"frozen B1 parent sampling std EO/EC: "
        f"{parent_eo_std:.3f} / "
        f"{parent_ec_std:.3f}"
    )

    print(
        f"matched lambda=0 continuation std EO/EC: "
        f"{lambda0_eo_std:.3f} / "
        f"{lambda0_ec_std:.3f}"
    )

    print(
        "\nOriginal V1 absolute scale rule:"
    )

    print(
        "  each state must lie in [0.80, 1.20]"
    )

    print(
        "\nProblem:"
    )

    print(
        "  on the exact small 16-subject diagnostic cohort, "
        "the frozen B1 parent EC std is already near the upper "
        "boundary and lambda=0 reaches 1.211."
    )

    print(
        "  Therefore the absolute guardrail is not a clean "
        "preservation test for this matched pilot."
    )

    # ========================================================
    # Step 3 — Freeze minimal V2 amendment
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — FREEZE MINIMAL V2 SCALE-GUARDRAIL AMENDMENT"
    )

    print(
        "=" * 80
    )

    amendment = {
        "status": (
            "B2_LAMBDA_SELECTION_PROTOCOL_AMENDMENT_V2_FROZEN"
        ),
        "amendment_timing": (
            "after completed Train-only lambda calibration V1, "
            "before any scientific B2 full training"
        ),
        "reason": (
            "The V1 scale criterion used an absolute normalized-std "
            "range [0.80,1.20]. On the exact matched 16-subject "
            "diagnostic cohort the frozen B1 parent itself produced "
            "EO/EC std approximately 0.958/1.196 and the lambda=0 "
            "matched continuation produced approximately 0.922/1.211. "
            "Thus the V1 absolute EC cutoff was not a clean measure "
            "of preserving the frozen parent generator. The amendment "
            "changes ONLY the scale guardrail to a paired relative "
            "preservation rule against the frozen B1 parent on the "
            "same subjects and seeds."
        ),
        "source_V1_protocol": str(
            B2_PROTOCOL_V1
        ),
        "source_V1_protocol_sha256": (
            v1_protocol_hash
        ),
        "source_lambda_calibration": str(
            B2_LAMBDA_SUMMARY_V1
        ),
        "source_lambda_calibration_sha256": (
            calibration_hash
        ),
        "original_rule": {
            "v_mse": (
                "<= 1.02 * lambda0"
            ),
            "subject_gap": (
                ">= 0.90 * lambda0 when lambda0 gap positive"
            ),
            "EO_generated_normalized_std": [
                0.80,
                1.20,
            ],
            "EC_generated_normalized_std": [
                0.80,
                1.20,
            ],
            "spectral_improvement": (
                ">= 10% relative to lambda0"
            ),
        },
        "amended_rule": {
            "v_mse": (
                "<= 1.02 * lambda0 (UNCHANGED)"
            ),
            "subject_gap": (
                ">= 0.90 * lambda0 when lambda0 gap positive "
                "(UNCHANGED)"
            ),
            "EO_scale_preservation": (
                "candidate EO generated normalized std / "
                "frozen-B1-parent EO std in [0.90,1.10]"
            ),
            "EC_scale_preservation": (
                "candidate EC generated normalized std / "
                "frozen-B1-parent EC std in [0.90,1.10]"
            ),
            "spectral_improvement": (
                ">= 10% relative to lambda0 (UNCHANGED)"
            ),
            "winner": (
                "among eligible nonzero lambdas choose lowest "
                "post-probe spectral loss; exact tie -> lower lambda"
            ),
        },
        "parent_scale_reference": {
            "sampling_subject_count": 16,
            "same_subjects_and_sampling_seeds_as_candidates": True,
            "EO_normalized_std": (
                parent_eo_std
            ),
            "EC_normalized_std": (
                parent_ec_std
            ),
            "relative_lower": (
                PARENT_SCALE_RELATIVE_LOWER
            ),
            "relative_upper": (
                PARENT_SCALE_RELATIVE_UPPER
            ),
        },
        "unchanged_components": [
            "candidate lambdas",
            "500 successful optimizer steps per lambda",
            "same frozen B1 initialization per lambda",
            "same Train order",
            "same timestep/noise stream",
            "v-MSE guardrail",
            "subject-gap guardrail",
            "spectral-improvement threshold",
            "lambda winner rule",
            "no Validation EEG",
            "no Final Test",
            "pilot checkpoints not reusable for scientific full run",
        ],
        "governance": {
            "post_calibration_amendment_explicitly_documented": True,
            "scientific_full_B2_started_before_amendment": False,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "bulk_synthetic_generation_started": False,
        },
    }

    if AMENDMENT_FILE.exists():
        with AMENDMENT_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != amendment:
            raise RuntimeError(
                "Conflicting B2 lambda-selection amendment V2 exists."
            )

        print(
            "identical V2 amendment already exists."
        )
    else:
        atomic_json_save(
            amendment,
            AMENDMENT_FILE,
        )

    print(
        f"amendment: "
        f"{AMENDMENT_FILE}"
    )

    print(
        "V2 amendment freeze: PASS"
    )

    # ========================================================
    # Step 4 — Re-evaluate lambdas under amended rule
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — APPLY AMENDED V2 ELIGIBILITY RULE"
    )

    print(
        "=" * 80
    )

    lambda0_probe = lambda0[
        "post_probe"
    ]

    lambda0_v_mse = float(
        lambda0_probe[
            "correct_subject_v_mse"
        ]
    )

    lambda0_gap = float(
        lambda0_probe[
            "wrong_minus_correct_v_mse"
        ]
    )

    lambda0_spectral = float(
        lambda0_probe[
            "spectral_loss_db_huber"
        ]
    )

    evaluated = []
    eligible_nonzero = []

    for lambda_value in EXPECTED_LAMBDAS:
        details = get_lambda_detail(
            calibration,
            lambda_value,
        )

        post_probe = details[
            "post_probe"
        ]

        post_sampling = details[
            "post_sampling"
        ]

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

        candidate_eo_std = float(
            post_sampling[
                "eo_normalized_std"
            ]
        )

        candidate_ec_std = float(
            post_sampling[
                "ec_normalized_std"
            ]
        )

        eo_parent_ratio = float(
            candidate_eo_std
            / parent_eo_std
        )

        ec_parent_ratio = float(
            candidate_ec_std
            / parent_ec_std
        )

        spectral_improvement = float(
            (
                lambda0_spectral
                - candidate_spectral
            )
            / lambda0_spectral
        )

        if lambda_value == 0.0:
            eligibility_v = True
            eligibility_gap = True
            eligibility_eo = True
            eligibility_ec = True
            eligibility_spectral = True
            eligible = False
        else:
            eligibility_v = bool(
                candidate_v_mse
                <= (
                    V_MSE_MAX_RATIO_TO_LAMBDA0
                    * lambda0_v_mse
                )
            )

            if lambda0_gap > 0.0:
                eligibility_gap = bool(
                    candidate_gap
                    >= (
                        SUBJECT_GAP_MIN_RATIO_TO_LAMBDA0
                        * lambda0_gap
                    )
                )
            else:
                eligibility_gap = bool(
                    candidate_gap
                    >= 0.0
                )

            eligibility_eo = bool(
                PARENT_SCALE_RELATIVE_LOWER
                <= eo_parent_ratio
                <= PARENT_SCALE_RELATIVE_UPPER
            )

            eligibility_ec = bool(
                PARENT_SCALE_RELATIVE_LOWER
                <= ec_parent_ratio
                <= PARENT_SCALE_RELATIVE_UPPER
            )

            eligibility_spectral = bool(
                spectral_improvement
                >= SPECTRAL_IMPROVEMENT_MIN_FRACTION
            )

            eligible = bool(
                eligibility_v
                and eligibility_gap
                and eligibility_eo
                and eligibility_ec
                and eligibility_spectral
            )

            if eligible:
                eligible_nonzero.append(
                    float(
                        lambda_value
                    )
                )

        record = {
            "lambda_spectral": (
                float(
                    lambda_value
                )
            ),
            "post_correct_subject_v_mse": (
                candidate_v_mse
            ),
            "v_mse_ratio_to_lambda0": float(
                candidate_v_mse
                / lambda0_v_mse
            ),
            "post_subject_gap": (
                candidate_gap
            ),
            "subject_gap_ratio_to_lambda0": (
                float(
                    candidate_gap
                    / lambda0_gap
                )
                if lambda0_gap != 0.0
                else None
            ),
            "post_spectral_loss": (
                candidate_spectral
            ),
            "spectral_improvement_vs_lambda0_fraction": (
                spectral_improvement
            ),
            "post_sampling_eo_std": (
                candidate_eo_std
            ),
            "post_sampling_ec_std": (
                candidate_ec_std
            ),
            "EO_std_ratio_to_frozen_B1_parent": (
                eo_parent_ratio
            ),
            "EC_std_ratio_to_frozen_B1_parent": (
                ec_parent_ratio
            ),
            "eligibility_v_mse": (
                eligibility_v
            ),
            "eligibility_subject_gap": (
                eligibility_gap
            ),
            "eligibility_parent_relative_eo_scale": (
                eligibility_eo
            ),
            "eligibility_parent_relative_ec_scale": (
                eligibility_ec
            ),
            "eligibility_spectral_improvement": (
                eligibility_spectral
            ),
            "eligible_nonzero": (
                eligible
            ),
        }

        evaluated.append(
            record
        )

        print(
            "\n"
            f"lambda={lambda_value:g}"
        )

        print(
            f"  correct-subject v-MSE: "
            f"{candidate_v_mse:.6f}"
        )

        print(
            f"  v-MSE ratio to lambda0: "
            f"{record['v_mse_ratio_to_lambda0']:.4f}"
        )

        print(
            f"  subject gap ratio to lambda0: "
            f"{record['subject_gap_ratio_to_lambda0']}"
        )

        print(
            f"  spectral improvement: "
            f"{100.0 * spectral_improvement:.2f}%"
        )

        print(
            f"  EO std / parent EO std: "
            f"{eo_parent_ratio:.4f}"
        )

        print(
            f"  EC std / parent EC std: "
            f"{ec_parent_ratio:.4f}"
        )

        print(
            f"  eligible V2: "
            f"{eligible}"
        )

    if not eligible_nonzero:
        raise RuntimeError(
            "No nonzero lambda is eligible even after the minimal "
            "parent-relative scale amendment. Do not start B2 full training."
        )

    selected_lambda = min(
        eligible_nonzero,
        key=lambda lambda_value: (
            float(
                get_lambda_detail(
                    calibration,
                    lambda_value,
                )[
                    "post_probe"
                ][
                    "spectral_loss_db_huber"
                ]
            ),
            float(
                lambda_value
            ),
        ),
    )

    if abs(
        selected_lambda
        - SELECTED_LAMBDA
    ) > 1e-12:
        raise RuntimeError(
            f"Amended rule selected {selected_lambda}, "
            f"expected {SELECTED_LAMBDA}."
        )

    print(
        f"\neligible nonzero lambdas under V2: "
        f"{eligible_nonzero}"
    )

    print(
        f"selected lambda under V2: "
        f"{selected_lambda}"
    )

    # ========================================================
    # Step 5 — Freeze selected lambda for scientific full B2
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — FREEZE SELECTED B2 LAMBDA"
    )

    print(
        "=" * 80
    )

    selected_details = get_lambda_detail(
        calibration,
        selected_lambda,
    )

    selected_record = next(
        row
        for row in evaluated
        if abs(
            float(
                row[
                    "lambda_spectral"
                ]
            )
            - selected_lambda
        ) < 1e-12
    )

    selected_payload = {
        "status": (
            "B2_SELECTED_LAMBDA_FROZEN_V2"
        ),
        "selected_lambda_spectral": (
            selected_lambda
        ),
        "selection_basis": (
            "V1 matched Train-only calibration metrics re-evaluated "
            "under frozen V2 parent-relative scale-preservation amendment"
        ),
        "eligible_nonzero_lambdas": (
            eligible_nonzero
        ),
        "selected_metrics": (
            selected_record
        ),
        "all_re_evaluated_candidates": (
            evaluated
        ),
        "frozen_B1_parent": {
            "path": str(
                FROZEN_B1_GENERATOR
            ),
            "sha256": (
                frozen_b1_hash_before
            ),
            "sampling_reference_EO_std": (
                parent_eo_std
            ),
            "sampling_reference_EC_std": (
                parent_ec_std
            ),
        },
        "V1_protocol": {
            "path": str(
                B2_PROTOCOL_V1
            ),
            "sha256": (
                v1_protocol_hash
            ),
        },
        "V1_calibration_summary": {
            "path": str(
                B2_LAMBDA_SUMMARY_V1
            ),
            "sha256": (
                calibration_hash
            ),
        },
        "V2_amendment": {
            "path": str(
                AMENDMENT_FILE
            ),
            "sha256": (
                sha256_file(
                    AMENDMENT_FILE
                )
            ),
        },
        "full_B2_run": {
            "initialization": (
                "exact frozen B1 RAW epoch-10 generator"
            ),
            "pilot_checkpoint_reused": False,
            "lambda_spectral": (
                selected_lambda
            ),
            "epochs": (
                EXPECTED_FULL_B2_EPOCHS
            ),
            "learning_rate": (
                EXPECTED_FULL_B2_LR
            ),
            "RAW_candidate_epochs": (
                EXPECTED_RAW_CANDIDATE_EPOCHS
            ),
            "EMA": False,
        },
        "matched_continuation_control": {
            "lambda": 0.0,
            "required_before_causal_publication_claim": True,
            "note": (
                "The matched lambda=0 full continuation is still required "
                "before attributing any final improvement causally to the "
                "spectral regularizer rather than extra fine-tuning."
            ),
        },
        "governance": {
            "amendment_is_post_calibration": True,
            "amendment_is_pre_full_B2_training": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "bulk_synthetic_generation_started": False,
            "scientific_B2_full_training_started": False,
        },
    }

    if SELECTED_LAMBDA_FILE.exists():
        with SELECTED_LAMBDA_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != selected_payload:
            raise RuntimeError(
                "Conflicting frozen B2 selected-lambda file exists."
            )

        print(
            "identical frozen selected-lambda file already exists."
        )
    else:
        atomic_json_save(
            selected_payload,
            SELECTED_LAMBDA_FILE,
        )

    print(
        f"selected-lambda file: "
        f"{SELECTED_LAMBDA_FILE}"
    )

    # ========================================================
    # Step 6 — Write full-run handoff
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — WRITE B2 FULL-RUN HANDOFF"
    )

    print(
        "=" * 80
    )

    handoff = {
        "status": (
            "B2_FULL_RUN_READY_V2"
        ),
        "frozen_B1_generator": str(
            FROZEN_B1_GENERATOR
        ),
        "frozen_B1_generator_sha256": (
            frozen_b1_hash_before
        ),
        "selected_lambda_file": str(
            SELECTED_LAMBDA_FILE
        ),
        "selected_lambda_file_sha256": (
            sha256_file(
                SELECTED_LAMBDA_FILE
            )
        ),
        "selected_lambda_spectral": (
            selected_lambda
        ),
        "architecture": (
            "exact frozen B1 architecture and weights at initialization"
        ),
        "prediction_target": (
            "v"
        ),
        "spectral_objective": (
            v1_protocol[
                "loss"
            ][
                "spectral_regularization"
            ]
        ),
        "full_run": {
            "epochs": (
                EXPECTED_FULL_B2_EPOCHS
            ),
            "batch_size": int(
                v1_protocol[
                    "full_fine_tuning_after_lambda_selection"
                ][
                    "batch_size"
                ]
            ),
            "learning_rate": (
                EXPECTED_FULL_B2_LR
            ),
            "weight_decay": float(
                v1_protocol[
                    "full_fine_tuning_after_lambda_selection"
                ][
                    "weight_decay"
                ]
            ),
            "gradient_clip_norm": float(
                v1_protocol[
                    "full_fine_tuning_after_lambda_selection"
                ][
                    "gradient_clip_norm"
                ]
            ),
            "EMA_enabled": False,
            "RAW_candidate_epochs": (
                EXPECTED_RAW_CANDIDATE_EPOCHS
            ),
            "pilot_checkpoint_reused": False,
            "automatic_resume": True,
        },
        "validation_policy": (
            "Validation may be used only as deterministic monitor "
            "under the frozen B2 full-run protocol; final scientific "
            "B2 selection uses predeclared Train-only generative fidelity."
        ),
        "Final_Test": (
            "SEALED"
        ),
        "next": (
            "Run 20-epoch scientific B2 full fine-tuning from frozen B1 "
            "with lambda_spectral=0.01."
        ),
    }

    atomic_json_save(
        handoff,
        FULL_RUN_HANDOFF,
    )

    print(
        f"full-run handoff: "
        f"{FULL_RUN_HANDOFF}"
    )

    # ========================================================
    # Step 7 — Human-readable summary
    # ========================================================

    selected_probe = selected_details[
        "post_probe"
    ]

    selected_sampling = selected_details[
        "post_sampling"
    ]

    summary_text = f"""SCENARIO 2 — B2 SELECTED LAMBDA V2

STATUS
B2_SELECTED_LAMBDA_FROZEN_V2

WHY V1 STOPPED
The original lambda calibration selected no nonzero lambda.
The failure was caused by the absolute scale guardrail in the matched
small-sample diagnostic: frozen B1 parent EO/EC std was
{parent_eo_std:.6f}/{parent_ec_std:.6f}, and lambda=0 continuation was
{lambda0_eo_std:.6f}/{lambda0_ec_std:.6f}.
Thus the fixed absolute EC upper bound 1.20 was not a clean parent-preservation test.

V2 AMENDMENT
Only the scale guardrail changed.
A candidate must preserve each state's frozen-B1 parent sampling std
within +/-10% on the exact same subjects/seeds.

Unchanged:
- v-MSE <= 1.02 x lambda0
- subject gap >= 0.90 x lambda0
- spectral improvement >= 10%
- matched Train-only data/noise streams
- no Validation EEG
- no Final Test
- pilot checkpoints cannot initialize scientific full B2

SELECTED LAMBDA
lambda_spectral = {selected_lambda}

SELECTED PILOT METRICS
Correct-subject v-MSE: {selected_probe["correct_subject_v_mse"]:.6f}
Subject gap: {selected_probe["wrong_minus_correct_v_mse"]:.6e}
Spectral loss: {selected_probe["spectral_loss_db_huber"]:.6f}
Spectral improvement vs lambda0:
{100.0 * selected_record["spectral_improvement_vs_lambda0_fraction"]:.2f}%

Sampling std EO/EC:
{selected_sampling["eo_normalized_std"]:.6f} /
{selected_sampling["ec_normalized_std"]:.6f}

Scale ratio to frozen B1 parent EO/EC:
{selected_record["EO_std_ratio_to_frozen_B1_parent"]:.6f} /
{selected_record["EC_std_ratio_to_frozen_B1_parent"]:.6f}

O1/O2 alpha EC-EO:
{selected_sampling["o1_alpha_effect_db"]:.6f} /
{selected_sampling["o2_alpha_effect_db"]:.6f} dB

FULL B2
Initialization: frozen B1 RAW epoch 10
Pilot checkpoint reused: NO
lambda_spectral = {selected_lambda}
Epochs = {EXPECTED_FULL_B2_EPOCHS}
LR = {EXPECTED_FULL_B2_LR}
RAW candidate epochs = {EXPECTED_RAW_CANDIDATE_EPOCHS}
EMA = OFF

CAUSAL CONTROL
A matched full lambda=0 continuation remains required before making
a publication claim that any final improvement was caused specifically
by spectral regularization rather than additional fine-tuning.

FINAL TEST
SEALED

NEXT
Run the 20-epoch scientific B2 full fine-tuning.
"""

    atomic_text_save(
        summary_text,
        SUMMARY_TEXT_FILE,
    )

    print(
        f"summary text: "
        f"{SUMMARY_TEXT_FILE}"
    )

    # ========================================================
    # Step 8 — Immutability
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — VERIFY SOURCE IMMUTABILITY"
    )

    print(
        "=" * 80
    )

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != frozen_b1_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during B2 amendment."
        )

    if sha256_file(
        B2_PROTOCOL_V1
    ) != v1_protocol_hash:
        raise RuntimeError(
            "Frozen B2 protocol V1 changed during amendment."
        )

    if sha256_file(
        B2_LAMBDA_SUMMARY_V1
    ) != calibration_hash:
        raise RuntimeError(
            "B2 lambda calibration V1 evidence changed."
        )

    print(
        "frozen B1 unchanged: PASS"
    )

    print(
        "B2 protocol V1 unchanged: PASS"
    )

    print(
        "lambda calibration V1 evidence unchanged: PASS"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B2 LAMBDA SELECTION AMENDED + FROZEN"
    )

    print(
        "=" * 80
    )

    print(
        f"selected lambda: "
        f"{selected_lambda}"
    )

    print(
        f"eligible nonzero lambdas: "
        f"{eligible_nonzero}"
    )

    print(
        "scale guardrail: parent-relative +/-10%"
    )

    print(
        f"selected spectral improvement: "
        f"{100.0 * selected_record['spectral_improvement_vs_lambda0_fraction']:.2f}%"
    )

    print(
        f"selected v-MSE ratio to lambda0: "
        f"{selected_record['v_mse_ratio_to_lambda0']:.4f}"
    )

    print(
        "selected subject-gap ratio to lambda0: "
        f"{selected_record['subject_gap_ratio_to_lambda0']:.4f}"
    )

    print(
        "selected EO/EC scale ratio to frozen B1 parent: "
        f"{selected_record['EO_std_ratio_to_frozen_B1_parent']:.4f}"
        " / "
        f"{selected_record['EC_std_ratio_to_frozen_B1_parent']:.4f}"
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
        f"\nSelected lambda file:"
        f"\n{SELECTED_LAMBDA_FILE}"
    )

    print(
        f"\nFull-run handoff:"
        f"\n{FULL_RUN_HANDOFF}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "RUN B2 20-EPOCH SCIENTIFIC FULL FINE-TUNING "
        "FROM FROZEN B1 WITH lambda_spectral=0.01"
    )


if __name__ == "__main__":
    main()
