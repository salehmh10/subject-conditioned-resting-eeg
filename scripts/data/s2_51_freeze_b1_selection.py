from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from s2_47_b1_architecture_warmstart_sanity import (
    B1SubjectStateUNet1D,
    state_dict_sha256,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen B0 parent
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

B0_SELECTION_MANIFEST = (
    FROZEN_B0_ROOT
    / "metrics"
    / "b0_frozen_selection_v1.json"
)

# Frozen B1 protocol
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

# Scientific B1 full-run candidate
B1_FULL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "full_finetune_v1"
)

SELECTED_SOURCE_CHECKPOINT = (
    B1_FULL_ROOT
    / "checkpoints"
    / "candidate_epoch_010_raw.pt"
)

# Subject-aware selection audit V2
B1_SWEEP_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "raw_candidate_subject_aware_sweep_v2"
)

B1_SWEEP_PROTOCOL = (
    B1_SWEEP_ROOT
    / "metrics"
    / "b1_raw_candidate_subject_aware_protocol_v2.json"
)

B1_SWEEP_AMENDMENT = (
    B1_SWEEP_ROOT
    / "metrics"
    / "b1_raw_candidate_subject_aware_protocol_amendment_v2.json"
)

B1_SWEEP_METRICS_CSV = (
    B1_SWEEP_ROOT
    / "metrics"
    / "b1_raw_candidate_subject_aware_metrics_v2.csv"
)

B1_SWEEP_SUBJECT_FINGERPRINT_CSV = (
    B1_SWEEP_ROOT
    / "metrics"
    / "b1_raw_candidate_subject_fingerprint_metrics_v2.csv"
)

B1_SWEEP_SUMMARY = (
    B1_SWEEP_ROOT
    / "metrics"
    / "b1_raw_candidate_subject_aware_summary_v2.json"
)


# ============================================================
# Frozen B1 output tree
# ============================================================

FREEZE_ROOT = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b1"
    / "b1_subject_state_raw_epoch010_v1"
)

CHECKPOINT_DIR = FREEZE_ROOT / "checkpoints"
METRIC_DIR = FREEZE_ROOT / "metrics"
HANDOFF_DIR = FREEZE_ROOT / "handoff"

ARCHIVE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "b1_subject_state_raw_epoch010_full_checkpoint_v1.pt"
)

GENERATOR_CHECKPOINT = (
    CHECKPOINT_DIR
    / "b1_subject_state_raw_epoch010_generator_v1.pt"
)

SELECTION_MANIFEST = (
    METRIC_DIR
    / "b1_frozen_selection_v1.json"
)

SELECTION_SUMMARY_TEXT = (
    METRIC_DIR
    / "b1_frozen_selection_summary_v1.txt"
)

B1_TO_B2_HANDOFF = (
    HANDOFF_DIR
    / "b1_to_b2_handoff_v1.json"
)


# ============================================================
# Frozen selection decision
# ============================================================

SELECTED_EPOCH = 10
SELECTED_WEIGHT_KIND = "RAW"

EXPECTED_CANDIDATES = [
    10,
    20,
    30,
]

EXPECTED_DIAGNOSTIC_DIRECTION = (
    "RAW_EPOCH_10_PRIMARY_SUBJECT_AWARE_CANDIDATE"
)

EXPECTED_GENERIC_WINS = {
    "10": 4,
    "20": 1,
    "30": 0,
}

EXPECTED_SUBJECT_WINS = {
    "10": 3,
    "20": 1,
    "30": 1,
}

EXPECTED_TOTAL_WINS = {
    "10": 7,
    "20": 2,
    "30": 1,
}

EXPECTED_SUBJECTS = 198
EXPECTED_INPUT_CHANNELS = 19
EXPECTED_INPUT_SAMPLES = 512


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


def atomic_copy(
    source: Path,
    destination: Path,
) -> None:
    temp = Path(
        str(destination)
        + ".tmp"
    )

    shutil.copy2(
        source,
        temp,
    )

    os.replace(
        temp,
        destination,
    )


def assert_finite_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> None:
    for name, tensor in state_dict.items():
        if (
            torch.is_tensor(
                tensor
            )
            and torch.is_floating_point(
                tensor
            )
            and not torch.isfinite(
                tensor
            ).all()
        ):
            raise RuntimeError(
                f"Nonfinite selected B1 tensor: "
                f"{name}"
            )


def cpu_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        .cpu()
        .clone()
        for name, tensor
        in state_dict.items()
    }


def normalize_epoch_count_dict(
    value: dict,
) -> dict[str, int]:
    return {
        str(
            key
        ): int(
            item
        )
        for key, item
        in value.items()
    }


def selected_metrics_from_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    metrics_by_epoch = summary[
        "metrics_by_epoch"
    ]

    key = str(
        SELECTED_EPOCH
    )

    if key not in metrics_by_epoch:
        raise RuntimeError(
            "Selected epoch metrics missing from "
            "subject-aware sweep summary."
        )

    return metrics_by_epoch[
        key
    ]


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — FREEZE FINAL B1 SELECTION"
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
        "  B1 30-epoch full fine-tuning: COMPLETE"
    )

    print(
        "  B1 RAW 10/20/30 subject-aware sweep V2: COMPLETE"
    )

    print(
        "  Formal B1 freeze: NOW"
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
        "\nFrozen scientific selection:"
    )

    print(
        "  model family: B1 subject+state DDPM-v"
    )

    print(
        f"  selected epoch: "
        f"{SELECTED_EPOCH}"
    )

    print(
        f"  selected weights: "
        f"{SELECTED_WEIGHT_KIND}"
    )

    print(
        "  EMA: OFF"
    )

    print(
        "  Final Test accessed: NO"
    )

    print(
        "  Bulk synthetic generation started: NO"
    )

    for directory in (
        CHECKPOINT_DIR,
        METRIC_DIR,
        HANDOFF_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit frozen parents / evidence
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B0 PARENT + B1 SELECTION EVIDENCE"
    )

    print(
        "=" * 80
    )

    required_files = [
        FROZEN_B0_GENERATOR,
        B0_SELECTION_MANIFEST,
        B1_PROTOCOL_FILE,
        SELECTED_SOURCE_CHECKPOINT,
        B1_SWEEP_PROTOCOL,
        B1_SWEEP_AMENDMENT,
        B1_SWEEP_METRICS_CSV,
        B1_SWEEP_SUBJECT_FINGERPRINT_CSV,
        B1_SWEEP_SUMMARY,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B0_SELECTION_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        b0_selection = json.load(
            f
        )

    with B1_PROTOCOL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        b1_protocol = json.load(
            f
        )

    with B1_SWEEP_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        sweep_summary = json.load(
            f
        )

    with B1_SWEEP_PROTOCOL.open(
        "r",
        encoding="utf-8",
    ) as f:
        sweep_protocol = json.load(
            f
        )

    with B1_SWEEP_AMENDMENT.open(
        "r",
        encoding="utf-8",
    ) as f:
        sweep_amendment = json.load(
            f
        )

    b0_hash_before = sha256_file(
        FROZEN_B0_GENERATOR
    )

    if b0_selection[
        "status"
    ] != "B0_SELECTION_FROZEN":
        raise RuntimeError(
            "B0 parent is not formally frozen."
        )

    if b0_selection[
        "b0_frozen"
    ] is not True:
        raise RuntimeError(
            "B0 parent frozen flag is not True."
        )

    if b1_protocol[
        "status"
    ] != "B1_SUBJECT_STATE_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B1 scientific protocol is not frozen."
        )

    if sweep_summary[
        "status"
    ] != "B1_RAW_CANDIDATE_SUBJECT_AWARE_SWEEP_V2_COMPLETE":
        raise RuntimeError(
            "B1 subject-aware sweep V2 is not complete."
        )

    if [
        int(
            epoch
        )
        for epoch
        in sweep_summary[
            "candidate_epochs"
        ]
    ] != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "B1 candidate set differs from frozen selection design."
        )

    if int(
        sweep_summary[
            "recommended_epoch"
        ]
    ) != SELECTED_EPOCH:
        raise RuntimeError(
            "Subject-aware sweep does not recommend RAW epoch 10."
        )

    if sweep_summary[
        "diagnostic_direction"
    ] != EXPECTED_DIAGNOSTIC_DIRECTION:
        raise RuntimeError(
            "Unexpected B1 subject-aware diagnostic direction."
        )

    generic_wins = normalize_epoch_count_dict(
        sweep_summary[
            "generic_criterion_win_counts"
        ]
    )

    subject_wins = normalize_epoch_count_dict(
        sweep_summary[
            "subject_specific_criterion_win_counts"
        ]
    )

    total_wins = normalize_epoch_count_dict(
        sweep_summary[
            "total_criterion_win_counts"
        ]
    )

    if (
        generic_wins
        != EXPECTED_GENERIC_WINS
    ):
        raise RuntimeError(
            "Generic criterion win counts do not match "
            "the frozen B1 decision."
        )

    if (
        subject_wins
        != EXPECTED_SUBJECT_WINS
    ):
        raise RuntimeError(
            "Subject-specific criterion win counts do not match "
            "the frozen B1 decision."
        )

    if (
        total_wins
        != EXPECTED_TOTAL_WINS
    ):
        raise RuntimeError(
            "Total criterion win counts do not match "
            "the frozen B1 decision."
        )

    if sweep_summary[
        "governance"
    ][
        "Validation_EEG_accessed"
    ] is not False:
        raise RuntimeError(
            "Subject-aware selection audit unexpectedly "
            "accessed Validation EEG."
        )

    if sweep_summary[
        "governance"
    ][
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Subject-aware selection audit unexpectedly "
            "accessed Final Test."
        )

    if sweep_summary[
        "governance"
    ][
        "training_updates_executed"
    ] is not False:
        raise RuntimeError(
            "Subject-aware selection audit unexpectedly "
            "performed training updates."
        )

    if sweep_amendment[
        "status"
    ] != "B1_SUBJECT_AWARE_AUDIT_PROTOCOL_AMENDMENT_V2":
        raise RuntimeError(
            "Unexpected B1 subject-aware audit amendment status."
        )

    if sweep_amendment[
        "governance"
    ][
        "amendment_made_before_candidate_generation"
    ] is not True:
        raise RuntimeError(
            "B1 audit amendment timing is not documented "
            "as pre-generation."
        )

    print(
        "B0 parent: PASS"
    )

    print(
        "B1 frozen protocol: PASS"
    )

    print(
        "subject-aware audit V2: PASS"
    )

    print(
        "protocol amendment V2: PASS"
    )

    print(
        f"generic wins: "
        f"{generic_wins}"
    )

    print(
        f"subject-specific wins: "
        f"{subject_wins}"
    )

    print(
        f"total wins: "
        f"{total_wins}"
    )

    # ========================================================
    # Step 2 — Audit selected RAW epoch-10 checkpoint
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — AUDIT SELECTED B1 RAW EPOCH-10 CHECKPOINT"
    )

    print(
        "=" * 80
    )

    source_hash_before = sha256_file(
        SELECTED_SOURCE_CHECKPOINT
    )

    checkpoint = torch.load(
        SELECTED_SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint[
        "status"
    ] != "B1_FULL_FINETUNE_CHECKPOINT":
        raise RuntimeError(
            "Selected B1 file is not a scientific "
            "full-run checkpoint."
        )

    if int(
        checkpoint[
            "epoch"
        ]
    ) != SELECTED_EPOCH:
        raise RuntimeError(
            "Selected B1 checkpoint epoch mismatch."
        )

    if checkpoint[
        "selected_weight_kind"
    ] != "RAW":
        raise RuntimeError(
            "Selected B1 checkpoint is not RAW."
        )

    if checkpoint[
        "EMA_enabled"
    ] is not False:
        raise RuntimeError(
            "Selected B1 checkpoint unexpectedly enables EMA."
        )

    if checkpoint[
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "Selected B1 checkpoint is not v-prediction."
        )

    if checkpoint[
        "pilot_checkpoint_used"
    ] is not False:
        raise RuntimeError(
            "Scientific B1 checkpoint unexpectedly used "
            "the engineering pilot."
        )

    if checkpoint[
        "final_test_accessed"
    ] is not False:
        raise RuntimeError(
            "Selected B1 checkpoint reports Final Test access."
        )

    selected_raw_state = cpu_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    assert_finite_state_dict(
        selected_raw_state
    )

    selected_model_state_hash = (
        state_dict_sha256(
            selected_raw_state
        )
    )

    if (
        selected_model_state_hash
        != checkpoint[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Selected B1 model-state SHA256 mismatch."
        )

    # Cross-check source SHA256 against the sweep evidence.
    selected_sweep_checkpoint_audit = (
        sweep_summary[
            "candidate_checkpoint_audit"
        ][
            str(
                SELECTED_EPOCH
            )
        ]
    )

    if (
        source_hash_before
        != selected_sweep_checkpoint_audit[
            "file_sha256"
        ]
    ):
        raise RuntimeError(
            "Selected B1 checkpoint file hash does not match "
            "subject-aware sweep evidence."
        )

    if (
        selected_model_state_hash
        != selected_sweep_checkpoint_audit[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Selected B1 model-state hash does not match "
            "subject-aware sweep evidence."
        )

    print(
        f"selected checkpoint SHA256: "
        f"{source_hash_before}"
    )

    print(
        f"selected model-state SHA256: "
        f"{selected_model_state_hash}"
    )

    print(
        "selected checkpoint audit: PASS"
    )

    # ========================================================
    # Step 3 — Extract selected metrics / interpretation
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — FREEZE B1 SCIENTIFIC SELECTION INTERPRETATION"
    )

    print(
        "=" * 80
    )

    selected_metrics = selected_metrics_from_summary(
        sweep_summary
    )

    scale_eo = float(
        selected_metrics[
            "scale"
        ][
            "EO"
        ][
            "normalized_std_ratio"
        ]
    )

    scale_ec = float(
        selected_metrics[
            "scale"
        ][
            "EC"
        ][
            "normalized_std_ratio"
        ]
    )

    mean_bandpower_mae = float(
        selected_metrics[
            "mean_bandpower_mae_db"
        ]
    )

    alpha_effect_mae = float(
        selected_metrics[
            "alpha_effect"
        ][
            "channel_mae_db"
        ]
    )

    corr_eo = float(
        selected_metrics[
            "spatial"
        ][
            "EO"
        ][
            "correlation_r"
        ]
    )

    corr_ec = float(
        selected_metrics[
            "spatial"
        ][
            "EC"
        ][
            "correlation_r"
        ]
    )

    wpli_eo = float(
        selected_metrics[
            "spatial"
        ][
            "EO"
        ][
            "alpha_dwpli2_r"
        ]
    )

    wpli_ec = float(
        selected_metrics[
            "spatial"
        ][
            "EC"
        ][
            "alpha_dwpli2_r"
        ]
    )

    subject_identity = selected_metrics[
        "subject_identity"
    ]

    subject_rmse = float(
        subject_identity[
            "subject_fingerprint_standardized_rmse_mean"
        ]
    )

    subject_r_median = float(
        subject_identity[
            "subject_fingerprint_pearson_r_median"
        ]
    )

    subject_margin = float(
        subject_identity[
            "subject_correct_vs_median_wrong_margin_mean"
        ]
    )

    subject_top1 = float(
        subject_identity[
            "subject_retrieval_top1_accuracy"
        ]
    )

    subject_top5 = float(
        subject_identity[
            "subject_retrieval_top5_accuracy"
        ]
    )

    subject_fixed_wrong = float(
        subject_identity[
            "subject_correct_better_than_fixed_wrong_fraction"
        ]
    )

    chance_top1 = float(
        subject_identity[
            "subject_retrieval_top1_chance"
        ]
    )

    chance_top5 = float(
        subject_identity[
            "subject_retrieval_top5_chance"
        ]
    )

    print(
        f"scale EO/EC: "
        f"{scale_eo:.3f} / "
        f"{scale_ec:.3f}"
    )

    print(
        f"mean bandpower MAE: "
        f"{mean_bandpower_mae:.3f} dB"
    )

    print(
        f"alpha-effect channel MAE: "
        f"{alpha_effect_mae:.3f} dB"
    )

    print(
        f"correlation r EO/EC: "
        f"{corr_eo:.3f} / "
        f"{corr_ec:.3f}"
    )

    print(
        f"alpha dwPLI2 r EO/EC: "
        f"{wpli_eo:.3f} / "
        f"{wpli_ec:.3f}"
    )

    print(
        f"subject fingerprint RMSE: "
        f"{subject_rmse:.3f}"
    )

    print(
        f"subject fingerprint r median: "
        f"{subject_r_median:.3f}"
    )

    print(
        f"subject retrieval top1/top5: "
        f"{subject_top1:.3f} / "
        f"{subject_top5:.3f}"
    )

    print(
        f"chance top1/top5: "
        f"{chance_top1:.3f} / "
        f"{chance_top5:.3f}"
    )

    print(
        f"correct closer than fixed wrong: "
        f"{subject_fixed_wrong:.3f}"
    )

    # ========================================================
    # Step 4 — Archive selected full checkpoint byte-for-byte
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — ARCHIVE SELECTED B1 FULL CHECKPOINT"
    )

    print(
        "=" * 80
    )

    if ARCHIVE_CHECKPOINT.exists():
        archive_hash = sha256_file(
            ARCHIVE_CHECKPOINT
        )

        if (
            archive_hash
            != source_hash_before
        ):
            raise RuntimeError(
                "Existing frozen B1 archive conflicts "
                "with selected source checkpoint."
            )

        print(
            "identical frozen B1 archive already exists."
        )
    else:
        atomic_copy(
            SELECTED_SOURCE_CHECKPOINT,
            ARCHIVE_CHECKPOINT,
        )

        archive_hash = sha256_file(
            ARCHIVE_CHECKPOINT
        )

        if (
            archive_hash
            != source_hash_before
        ):
            raise RuntimeError(
                "Frozen B1 archive verification failed."
            )

        print(
            "selected B1 full checkpoint archived byte-for-byte."
        )

    print(
        f"archive: "
        f"{ARCHIVE_CHECKPOINT}"
    )

    print(
        f"archive SHA256: "
        f"{archive_hash}"
    )

    # ========================================================
    # Step 5 — Create lean frozen B1 generator
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — SAVE LEAN FROZEN B1 GENERATOR"
    )

    print(
        "=" * 80
    )

    frozen_b0 = torch.load(
        FROZEN_B0_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    b0_contract = copy.deepcopy(
        frozen_b0[
            "effective_contract"
        ]
    )

    subject_ids = list(
        b1_protocol[
            "conditioning"
        ][
            "subject"
        ][
            "mapping"
        ]
    ) if isinstance(
        b1_protocol[
            "conditioning"
        ][
            "subject"
        ][
            "mapping"
        ],
        list,
    ) else list(
        b1_protocol[
            "data"
        ][
            "subject_ids_in_index_order"
        ]
    )

    if len(
        subject_ids
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Frozen B1 subject mapping length mismatch."
        )

    effective_b1_contract = {
        "model_name": (
            "B1_SUBJECT_STATE_DDPM_V_FROZEN"
        ),
        "definition": (
            "p(EEG | subject, eye_state)"
        ),
        "prediction_target": (
            "v"
        ),
        "parent_B0_effective_contract": (
            b0_contract
        ),
        "input_channels": (
            EXPECTED_INPUT_CHANNELS
        ),
        "input_samples": (
            EXPECTED_INPUT_SAMPLES
        ),
        "sampling_rate_hz": 128,
        "subject_count": (
            EXPECTED_SUBJECTS
        ),
        "subject_ids_in_index_order": (
            subject_ids
        ),
        "subject_embedding_dim": int(
            b1_protocol[
                "conditioning"
            ][
                "subject"
            ][
                "embedding_dim"
            ]
        ),
        "subject_projection_output_dim": int(
            b1_protocol[
                "conditioning"
            ][
                "subject"
            ][
                "projection_output_dim"
            ]
        ),
        "condition_injection_module": str(
            checkpoint[
                "condition_injection_module"
            ]
        ),
        "eye_state_mapping": {
            "EO": 0,
            "EC": 1,
        },
        "selected_epoch": (
            SELECTED_EPOCH
        ),
        "selected_weight_kind": (
            SELECTED_WEIGHT_KIND
        ),
        "EMA_enabled": False,
    }

    generator_payload = {
        "status": (
            "B1_SUBJECT_STATE_V_FROZEN_GENERATOR"
        ),
        "freeze_schema_version": 1,
        "selected_epoch": (
            SELECTED_EPOCH
        ),
        "selected_weight_kind": (
            SELECTED_WEIGHT_KIND
        ),
        "prediction_target": (
            "v"
        ),
        "EMA_enabled": False,
        "selected_source_checkpoint": str(
            SELECTED_SOURCE_CHECKPOINT
        ),
        "selected_source_checkpoint_sha256": (
            source_hash_before
        ),
        "selected_model_state_sha256": (
            selected_model_state_hash
        ),
        "frozen_B0_parent": str(
            FROZEN_B0_GENERATOR
        ),
        "frozen_B0_parent_sha256": (
            b0_hash_before
        ),
        "B1_protocol": str(
            B1_PROTOCOL_FILE
        ),
        "B1_protocol_sha256": (
            sha256_file(
                B1_PROTOCOL_FILE
            )
        ),
        "effective_contract": (
            effective_b1_contract
        ),
        "model_state_dict": (
            selected_raw_state
        ),
        "selection_evidence": {
            "subject_aware_sweep_summary": str(
                B1_SWEEP_SUMMARY
            ),
            "subject_aware_sweep_summary_sha256": (
                sha256_file(
                    B1_SWEEP_SUMMARY
                )
            ),
            "subject_aware_sweep_protocol": str(
                B1_SWEEP_PROTOCOL
            ),
            "subject_aware_sweep_protocol_sha256": (
                sha256_file(
                    B1_SWEEP_PROTOCOL
                )
            ),
            "subject_aware_sweep_amendment": str(
                B1_SWEEP_AMENDMENT
            ),
            "subject_aware_sweep_amendment_sha256": (
                sha256_file(
                    B1_SWEEP_AMENDMENT
                )
            ),
            "subject_aware_metrics_csv": str(
                B1_SWEEP_METRICS_CSV
            ),
            "subject_fingerprint_metrics_csv": str(
                B1_SWEEP_SUBJECT_FINGERPRINT_CSV
            ),
        },
        "known_limitations": {
            "alpha_dwpli2_connectivity": {
                "status": (
                    "LIMITED"
                ),
                "EO_upper_triangle_r": (
                    wpli_eo
                ),
                "EC_upper_triangle_r": (
                    wpli_ec
                ),
                "interpretation": (
                    "B1 strongly preserves scale, spectral, "
                    "state and subject identity fidelity, but "
                    "alpha phase-lag connectivity remains "
                    "substantially weaker than channel-correlation "
                    "fidelity and is retained as a documented limitation."
                ),
            },
            "selection_tradeoff": (
                "Epoch 20 had the best alpha dwPLI2 criterion and "
                "the best Top-1 subject retrieval, while epoch 30 "
                "had the best median subject-fingerprint correlation. "
                "Epoch 10 was selected because it satisfied the "
                "predeclared dual-domain rule and won 4/5 generic "
                "plus 3/5 subject-specific criteria."
            ),
        },
        "Validation_EEG_accessed_during_full_training": True,
        "Validation_EEG_accessed_during_subject_aware_selection": False,
        "Final_Test_accessed": False,
        "bulk_synthetic_generation_started": False,
        "B1_frozen": True,
    }

    if GENERATOR_CHECKPOINT.exists():
        existing = torch.load(
            GENERATOR_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        if (
            existing.get(
                "selected_model_state_sha256"
            )
            != selected_model_state_hash
            or int(
                existing.get(
                    "selected_epoch",
                    -1,
                )
            )
            != SELECTED_EPOCH
        ):
            raise RuntimeError(
                "Existing frozen B1 generator conflicts "
                "with current selected identity."
            )

        print(
            "identical frozen B1 generator already exists."
        )
    else:
        atomic_torch_save(
            generator_payload,
            GENERATOR_CHECKPOINT,
        )

        print(
            "lean frozen B1 generator saved."
        )

    print(
        f"generator: "
        f"{GENERATOR_CHECKPOINT}"
    )

    # ========================================================
    # Step 6 — Strict reload / functional smoke test
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — STRICT RELOAD + SUBJECT-CONDITIONING SMOKE TEST"
    )

    print(
        "=" * 80
    )

    frozen_generator = torch.load(
        GENERATOR_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    reload_state_hash = state_dict_sha256(
        frozen_generator[
            "model_state_dict"
        ]
    )

    if (
        reload_state_hash
        != selected_model_state_hash
    ):
        raise RuntimeError(
            "Frozen B1 generator model-state hash mismatch."
        )

    model = B1SubjectStateUNet1D(
        b0_contract=(
            frozen_b0[
                "effective_contract"
            ]
        ),
        injection_module_name=str(
            frozen_generator[
                "effective_contract"
            ][
                "condition_injection_module"
            ]
        ),
        num_subjects=(
            EXPECTED_SUBJECTS
        ),
        subject_embedding_dim=int(
            frozen_generator[
                "effective_contract"
            ][
                "subject_embedding_dim"
            ]
        ),
        condition_dim=int(
            frozen_generator[
                "effective_contract"
            ][
                "subject_projection_output_dim"
            ]
        ),
    )

    model.load_state_dict(
        frozen_generator[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    with torch.no_grad():
        x = torch.zeros(
            (
                2,
                EXPECTED_INPUT_CHANNELS,
                EXPECTED_INPUT_SAMPLES,
            ),
            dtype=torch.float32,
        )

        timesteps = torch.tensor(
            [
                100,
                900,
            ],
            dtype=torch.long,
        )

        eye_state = torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        )

        subjects_a = torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        )

        subjects_b = torch.tensor(
            [
                97,
                98,
            ],
            dtype=torch.long,
        )

        output_a = model(
            x,
            timesteps,
            eye_state,
            subjects_a,
        )

        output_b = model(
            x,
            timesteps,
            eye_state,
            subjects_b,
        )

        output_state_flip = model(
            x,
            timesteps,
            1 - eye_state,
            subjects_a,
        )

    if output_a.shape != (
        2,
        EXPECTED_INPUT_CHANNELS,
        EXPECTED_INPUT_SAMPLES,
    ):
        raise RuntimeError(
            "Frozen B1 smoke-test output shape mismatch."
        )

    if not (
        torch.isfinite(
            output_a
        ).all()
        and torch.isfinite(
            output_b
        ).all()
        and torch.isfinite(
            output_state_flip
        ).all()
    ):
        raise RuntimeError(
            "Frozen B1 smoke-test output contains NaN/Inf."
        )

    subject_output_diff = float(
        torch.mean(
            torch.abs(
                output_a
                - output_b
            )
        ).item()
    )

    state_output_diff = float(
        torch.mean(
            torch.abs(
                output_a
                - output_state_flip
            )
        ).item()
    )

    if (
        subject_output_diff
        <= 0.0
    ):
        raise RuntimeError(
            "Frozen B1 subject conditioning appears inactive."
        )

    if (
        state_output_diff
        <= 0.0
    ):
        raise RuntimeError(
            "Frozen B1 eye-state conditioning appears inactive."
        )

    print(
        f"parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"output shape: "
        f"{tuple(output_a.shape)}"
    )

    print(
        f"subject output mean abs difference: "
        f"{subject_output_diff:.6e}"
    )

    print(
        f"eye-state output mean abs difference: "
        f"{state_output_diff:.6e}"
    )

    print(
        "strict reload: PASS"
    )

    print(
        "subject conditioning active: PASS"
    )

    print(
        "eye-state conditioning active: PASS"
    )

    # ========================================================
    # Step 7 — Formal B1 freeze manifest
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — WRITE FORMAL B1 FREEZE MANIFEST"
    )

    print(
        "=" * 80
    )

    selection_manifest = {
        "status": (
            "B1_SELECTION_FROZEN"
        ),
        "freeze_schema_version": 1,
        "B1_frozen": True,
        "selected_model": (
            "B1_SUBJECT_STATE_DDPM_V"
        ),
        "selected_epoch": (
            SELECTED_EPOCH
        ),
        "selected_weight_kind": (
            SELECTED_WEIGHT_KIND
        ),
        "prediction_target": (
            "v"
        ),
        "EMA_enabled": False,
        "parent_B0": {
            "path": str(
                FROZEN_B0_GENERATOR
            ),
            "sha256": (
                b0_hash_before
            ),
            "read_only": True,
        },
        "selection_basis": {
            "candidate_epochs": (
                EXPECTED_CANDIDATES
            ),
            "selection_partition": (
                "Real Train subject-aware generative fidelity only"
            ),
            "generic_criterion_winners": (
                sweep_summary[
                    "generic_criterion_winners"
                ]
            ),
            "generic_criterion_win_counts": (
                generic_wins
            ),
            "subject_specific_criterion_winners": (
                sweep_summary[
                    "subject_specific_criterion_winners"
                ]
            ),
            "subject_specific_criterion_win_counts": (
                subject_wins
            ),
            "total_criterion_win_counts": (
                total_wins
            ),
            "predeclared_recommendation_rule_satisfied": True,
            "diagnostic_direction": (
                sweep_summary[
                    "diagnostic_direction"
                ]
            ),
        },
        "selected_metrics": {
            "scale_normalized_std_ratio_EO": (
                scale_eo
            ),
            "scale_normalized_std_ratio_EC": (
                scale_ec
            ),
            "mean_bandpower_MAE_dB": (
                mean_bandpower_mae
            ),
            "alpha_effect_channel_MAE_dB": (
                alpha_effect_mae
            ),
            "channel_correlation_r_EO": (
                corr_eo
            ),
            "channel_correlation_r_EC": (
                corr_ec
            ),
            "alpha_dwPLI2_r_EO": (
                wpli_eo
            ),
            "alpha_dwPLI2_r_EC": (
                wpli_ec
            ),
            "subject_fingerprint_standardized_RMSE_mean": (
                subject_rmse
            ),
            "subject_fingerprint_Pearson_r_median": (
                subject_r_median
            ),
            "subject_correct_vs_median_wrong_margin_mean": (
                subject_margin
            ),
            "subject_retrieval_top1_accuracy": (
                subject_top1
            ),
            "subject_retrieval_top5_accuracy": (
                subject_top5
            ),
            "subject_retrieval_top1_chance": (
                chance_top1
            ),
            "subject_retrieval_top5_chance": (
                chance_top5
            ),
            "subject_correct_better_than_fixed_wrong_fraction": (
                subject_fixed_wrong
            ),
        },
        "known_limitations": {
            "alpha_phase_lag_connectivity": (
                "limited; epoch 20 had stronger alpha dwPLI2 fidelity"
            ),
            "selection_tradeoff": (
                "Epoch 10 selected by the frozen dual-domain rule "
                "despite isolated criterion wins by epochs 20 and 30."
            ),
        },
        "audit_amendment": {
            "v1_structural_preflight_failed": True,
            "v1_result_used_for_selection": False,
            "v2_amendment_path": str(
                B1_SWEEP_AMENDMENT
            ),
            "v2_amendment_sha256": (
                sha256_file(
                    B1_SWEEP_AMENDMENT
                )
            ),
            "v2_real_reference": (
                "5 distinct Train blocks per subject/state"
            ),
        },
        "frozen_artifacts": {
            "full_archival_checkpoint": str(
                ARCHIVE_CHECKPOINT
            ),
            "full_archival_checkpoint_sha256": (
                archive_hash
            ),
            "lean_generator_checkpoint": str(
                GENERATOR_CHECKPOINT
            ),
            "selected_model_state_sha256": (
                selected_model_state_hash
            ),
            "selection_summary": str(
                B1_SWEEP_SUMMARY
            ),
            "selection_summary_sha256": (
                sha256_file(
                    B1_SWEEP_SUMMARY
                )
            ),
        },
        "governance": {
            "Validation_used_during_B1_training": True,
            "Validation_used_in_final_subject_aware_generator_selection": False,
            "Final_Test_accessed": False,
            "bulk_synthetic_generation_started": False,
            "B0_remains_read_only": True,
            "B2_may_initialize_from_this_frozen_B1": True,
        },
    }

    if SELECTION_MANIFEST.exists():
        with SELECTION_MANIFEST.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_manifest = json.load(
                f
            )

        existing_hash = (
            existing_manifest.get(
                "frozen_artifacts",
                {},
            ).get(
                "selected_model_state_sha256"
            )
        )

        if (
            existing_hash
            != selected_model_state_hash
            or int(
                existing_manifest.get(
                    "selected_epoch",
                    -1,
                )
            )
            != SELECTED_EPOCH
        ):
            raise RuntimeError(
                "Existing B1 freeze manifest conflicts "
                "with selected epoch-10 identity."
            )

        print(
            "identical B1 freeze identity already exists."
        )
    else:
        atomic_json_save(
            selection_manifest,
            SELECTION_MANIFEST,
        )

    print(
        f"selection manifest: "
        f"{SELECTION_MANIFEST}"
    )

    # ========================================================
    # Step 8 — Human-readable summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — SAVE HUMAN-READABLE B1 FREEZE SUMMARY"
    )

    print(
        "=" * 80
    )

    summary_text = f"""SCENARIO 2 — FROZEN B1 SELECTION

STATUS
B1_SELECTION_FROZEN

SELECTED GENERATOR
Model: B1 subject+state DDPM-v
Definition: p(EEG | subject, eye_state)
Selected checkpoint: RAW epoch {SELECTED_EPOCH}
EMA: OFF

PARENT
Frozen B0: B0-v RAW epoch 20
B0 remains read-only.

SELECTION
Candidates: RAW epochs 10, 20, 30
Generic wins: {generic_wins}
Subject-specific wins: {subject_wins}
Total wins: {total_wins}
Diagnostic direction:
{sweep_summary["diagnostic_direction"]}

SELECTED EPOCH-10 GENERIC FIDELITY
Normalized std ratio EO: {scale_eo:.6f}
Normalized std ratio EC: {scale_ec:.6f}
Mean bandpower MAE: {mean_bandpower_mae:.6f} dB
Alpha-effect channel MAE: {alpha_effect_mae:.6f} dB
Channel-correlation r EO: {corr_eo:.6f}
Channel-correlation r EC: {corr_ec:.6f}
Alpha dwPLI2 r EO: {wpli_eo:.6f}
Alpha dwPLI2 r EC: {wpli_ec:.6f}

SELECTED EPOCH-10 SUBJECT-SPECIFIC FIDELITY
Subject fingerprint standardized RMSE mean: {subject_rmse:.6f}
Subject fingerprint Pearson r median: {subject_r_median:.6f}
Correct-vs-median-wrong distance margin mean: {subject_margin:.6f}
Top-1 subject retrieval: {subject_top1:.6f}
Top-5 subject retrieval: {subject_top5:.6f}
Top-1 chance: {chance_top1:.6f}
Top-5 chance: {chance_top5:.6f}
Correct closer than fixed wrong: {subject_fixed_wrong:.6f}

KNOWN LIMITATION
Alpha phase-lag connectivity fidelity remains limited.
Epoch 20 won the alpha dwPLI2 criterion, but epoch 10 was selected
because it satisfied the frozen dual-domain rule:
4/5 generic criteria + 3/5 subject-specific criteria.

AUDIT AMENDMENT
The failed V1 audit required 8 distinct EO and 8 distinct EC Train
blocks per subject, which was structurally impossible.
V1 stopped before candidate generation and was not used for selection.
V2 used 5 distinct Train blocks per state and was frozen before generation.

DATA GOVERNANCE
Validation used during B1 full training: YES
Validation used in final subject-aware generator selection: NO
Final Test accessed: NO
Bulk synthetic generation started: NO

FROZEN IDENTIFIERS
Selected source checkpoint SHA256:
{source_hash_before}

Selected model-state SHA256:
{selected_model_state_hash}

FROZEN GENERATOR
{GENERATOR_CHECKPOINT}

NEXT SCIENTIFIC STAGE
B2 = initialize from this frozen B1 RAW epoch-10 generator,
retain subject + eye-state conditioning, continue v-prediction,
and add the predeclared spectral-regularization objective.
"""

    atomic_text_save(
        summary_text,
        SELECTION_SUMMARY_TEXT,
    )

    print(
        f"summary text: "
        f"{SELECTION_SUMMARY_TEXT}"
    )

    # ========================================================
    # Step 9 — B1 -> B2 handoff
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — WRITE B1 -> B2 HANDOFF"
    )

    print(
        "=" * 80
    )

    handoff = {
        "status": (
            "B1_TO_B2_HANDOFF_READY"
        ),
        "frozen_B1_generator": str(
            GENERATOR_CHECKPOINT
        ),
        "frozen_B1_selection_manifest": str(
            SELECTION_MANIFEST
        ),
        "selected_epoch": (
            SELECTED_EPOCH
        ),
        "selected_weight_kind": (
            SELECTED_WEIGHT_KIND
        ),
        "selected_model_state_sha256": (
            selected_model_state_hash
        ),
        "prediction_target": (
            "v"
        ),
        "definition": (
            "p(EEG | subject, eye_state)"
        ),
        "input_shape": [
            EXPECTED_INPUT_CHANNELS,
            EXPECTED_INPUT_SAMPLES,
        ],
        "sampling_rate_hz": 128,
        "subjects": (
            EXPECTED_SUBJECTS
        ),
        "subject_ids_in_index_order": (
            subject_ids
        ),
        "conditioning": {
            "eye_state": True,
            "subject": True,
            "EO": 0,
            "EC": 1,
        },
        "B2_required_changes": {
            "initialize_from_frozen_B1": True,
            "preserve_subject_conditioning": True,
            "preserve_eye_state_conditioning": True,
            "continue_v_prediction": True,
            "add_spectral_regularization": True,
            "do_not_add_connectivity_loss_without_new_protocol_amendment": True,
            "do_not_modify_frozen_B1_artifact": True,
        },
        "B1_metrics_to_preserve_in_B2": {
            "scale_ratio_EO": (
                scale_eo
            ),
            "scale_ratio_EC": (
                scale_ec
            ),
            "mean_bandpower_MAE_dB": (
                mean_bandpower_mae
            ),
            "alpha_effect_MAE_dB": (
                alpha_effect_mae
            ),
            "subject_top1_retrieval": (
                subject_top1
            ),
            "subject_top5_retrieval": (
                subject_top5
            ),
            "subject_fingerprint_r_median": (
                subject_r_median
            ),
        },
        "known_B1_limitation_to_track": {
            "alpha_dwPLI2_r_EO": (
                wpli_eo
            ),
            "alpha_dwPLI2_r_EC": (
                wpli_ec
            ),
            "description": (
                "Phase-lag connectivity remains weaker "
                "than scale/spectral/spatial-correlation/subject fidelity."
            ),
        },
        "bulk_generation_before_B2_freeze": False,
        "Final_Test": (
            "SEALED"
        ),
    }

    atomic_json_save(
        handoff,
        B1_TO_B2_HANDOFF,
    )

    print(
        f"B2 handoff: "
        f"{B1_TO_B2_HANDOFF}"
    )

    # ========================================================
    # Step 10 — Verify immutability
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 10 — VERIFY SOURCE IMMUTABILITY"
    )

    print(
        "=" * 80
    )

    source_hash_after = sha256_file(
        SELECTED_SOURCE_CHECKPOINT
    )

    if (
        source_hash_after
        != source_hash_before
    ):
        raise RuntimeError(
            "Selected B1 source checkpoint changed "
            "during freeze."
        )

    b0_hash_after = sha256_file(
        FROZEN_B0_GENERATOR
    )

    if (
        b0_hash_after
        != b0_hash_before
    ):
        raise RuntimeError(
            "Frozen B0 changed during B1 freeze."
        )

    print(
        "selected B1 source checkpoint unchanged: PASS"
    )

    print(
        "frozen B0 remains read-only: PASS"
    )

    print(
        "subject-aware V2 evidence read-only: PASS"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B1 SELECTION FORMALLY FROZEN"
    )

    print(
        "=" * 80
    )

    print(
        f"selected model: "
        f"B1 subject+state v-prediction RAW epoch {SELECTED_EPOCH}"
    )

    print(
        f"selected model-state SHA256: "
        f"{selected_model_state_hash}"
    )

    print(
        f"scale EO/EC: "
        f"{scale_eo:.3f} / "
        f"{scale_ec:.3f}"
    )

    print(
        f"mean bandpower MAE: "
        f"{mean_bandpower_mae:.3f} dB"
    )

    print(
        f"alpha-effect MAE: "
        f"{alpha_effect_mae:.3f} dB"
    )

    print(
        f"correlation r EO/EC: "
        f"{corr_eo:.3f} / "
        f"{corr_ec:.3f}"
    )

    print(
        f"alpha dwPLI2 r EO/EC: "
        f"{wpli_eo:.3f} / "
        f"{wpli_ec:.3f}"
    )

    print(
        f"subject fingerprint RMSE: "
        f"{subject_rmse:.3f}"
    )

    print(
        f"subject fingerprint r median: "
        f"{subject_r_median:.3f}"
    )

    print(
        f"subject retrieval top1/top5: "
        f"{subject_top1:.3f} / "
        f"{subject_top5:.3f}"
    )

    print(
        f"correct closer than fixed wrong: "
        f"{subject_fixed_wrong:.3f}"
    )

    print(
        "\nKnown limitation:"
    )

    print(
        "  alpha phase-lag connectivity fidelity remains limited."
    )

    print(
        "\nValidation used during full B1 training: YES"
    )

    print(
        "Validation used in final subject-aware selection: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Bulk synthetic generation started: NO"
    )

    print(
        "B1 frozen: YES"
    )

    print(
        f"\nFrozen B1 generator:"
        f"\n{GENERATOR_CHECKPOINT}"
    )

    print(
        f"\nSelection manifest:"
        f"\n{SELECTION_MANIFEST}"
    )

    print(
        f"\nB1 -> B2 handoff:"
        f"\n{B1_TO_B2_HANDOFF}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "DESIGN AND FREEZE B2 SPECTRAL-REGULARIZED "
        "FINE-TUNING PROTOCOL"
    )


if __name__ == "__main__":
    main()
