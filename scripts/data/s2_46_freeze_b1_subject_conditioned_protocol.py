from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)

from s2_27_logbandpower_logreg_validation import (
    build_validation_window_index,
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

B0_SELECTION_MANIFEST = (
    FROZEN_B0_ROOT
    / "metrics"
    / "b0_frozen_selection_v1.json"
)

B0_TO_B1_HANDOFF = (
    FROZEN_B0_ROOT
    / "handoff"
    / "b0_to_b1_handoff_v1.json"
)


# ============================================================
# B1 protocol outputs
# ============================================================

B1_PROTOCOL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b1"
    / "subject_state_v1"
)

METRIC_DIR = (
    B1_PROTOCOL_ROOT
    / "metrics"
)

HANDOFF_DIR = (
    B1_PROTOCOL_ROOT
    / "handoff"
)

B1_PROTOCOL_FILE = (
    METRIC_DIR
    / "b1_subject_state_finetune_protocol_v1.json"
)

B1_SUMMARY_FILE = (
    METRIC_DIR
    / "b1_subject_state_finetune_protocol_summary_v1.txt"
)

B1_IMPLEMENTATION_HANDOFF = (
    HANDOFF_DIR
    / "b1_architecture_implementation_handoff_v1.json"
)


# ============================================================
# Frozen B1 design
# ============================================================

SCHEMA_VERSION = 1

EXPECTED_SUBJECTS = 198
EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_VALIDATION_WINDOWS = 9_635

INPUT_CHANNELS = 19
INPUT_SAMPLES = 512
SFREQ = 128

# B1-specific conditioning
SUBJECT_EMBEDDING_DIM = 128
CONDITION_DIM = 256

# Pilot
PILOT_STEPS = 1000

# Full fine-tuning
FULL_EPOCHS = 30
BATCH_SIZE = 64

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

CHECKPOINT_EVERY_EPOCHS = 5
SEED = 20260908

# We intentionally do NOT use slow EMA for B1 selection/generation.
# B0 diagnostics showed RAW weights gave materially better
# generative amplitude/spectral fidelity than EMA.
USE_EMA = False


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
    data: dict,
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — FREEZE B1 SUBJECT-CONDITIONED FINE-TUNING PROTOCOL"
    )

    print(
        "=" * 80
    )

    print(
        "\nCurrent project stage:"
    )

    print(
        "  B0 final selection: FROZEN"
    )

    print(
        "  B1 subject+state protocol freeze: NOW"
    )

    print(
        "  B1 architecture implementation: NOT STARTED"
    )

    print(
        "  B1 training: NOT STARTED"
    )

    print(
        "  B2 training: NOT STARTED"
    )

    print(
        "  Bulk synthetic generation: NOT STARTED"
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
    # Step 1 — Audit frozen B0 parent
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B0 PARENT"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B0_GENERATOR,
        B0_SELECTION_MANIFEST,
        B0_TO_B1_HANDOFF,
    ):
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

    with B0_TO_B1_HANDOFF.open(
        "r",
        encoding="utf-8",
    ) as f:
        b0_handoff = json.load(
            f
        )

    if b0_selection[
        "status"
    ] != "B0_SELECTION_FROZEN":
        raise RuntimeError(
            "B0 selection is not formally frozen."
        )

    if b0_selection[
        "b0_frozen"
    ] is not True:
        raise RuntimeError(
            "B0 frozen flag is not True."
        )

    if int(
        b0_selection[
            "selected_epoch"
        ]
    ) != 20:
        raise RuntimeError(
            "B1 must initialize from frozen B0 RAW epoch 20."
        )

    if b0_selection[
        "selected_weight_kind"
    ] != "raw":
        raise RuntimeError(
            "B1 must initialize from frozen B0 RAW weights."
        )

    if b0_selection[
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "B1 must continue v-prediction."
        )

    if b0_selection[
        "governance"
    ][
        "final_test_accessed"
    ] is not False:
        raise RuntimeError(
            "Frozen B0 reports Final Test access."
        )

    if b0_handoff[
        "status"
    ] != "B0_TO_B1_HANDOFF_READY":
        raise RuntimeError(
            "B0 -> B1 handoff is not ready."
        )

    frozen_b0_hash = sha256_file(
        FROZEN_B0_GENERATOR
    )

    frozen_b0 = torch.load(
        FROZEN_B0_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    if frozen_b0[
        "status"
    ] != "B0_V_FROZEN_GENERATOR":
        raise RuntimeError(
            "Unexpected frozen B0 generator status."
        )

    if frozen_b0[
        "selected_weight_kind"
    ] != "raw":
        raise RuntimeError(
            "Frozen B0 generator is not RAW."
        )

    if int(
        frozen_b0[
            "selected_epoch"
        ]
    ) != 20:
        raise RuntimeError(
            "Frozen B0 generator epoch mismatch."
        )

    if frozen_b0[
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "Frozen B0 generator does not use v-prediction."
        )

    print(
        "frozen parent: B0-v RAW epoch 20"
    )

    print(
        f"frozen B0 generator SHA256: "
        f"{frozen_b0_hash}"
    )

    print(
        "B0 parent audit: PASS"
    )

    # ========================================================
    # Step 2 — Resolve subject mapping from locked Train split
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE B1 SUBJECT INDEX MAPPING"
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

    train_records, train_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    validation_records, validation_mapping = (
        build_validation_window_index(
            split
        )
    )

    if len(
        train_records
    ) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(
        validation_records
    ) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            "Unexpected Validation window count."
        )

    if len(
        train_mapping
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Unexpected subject count."
        )

    if (
        train_mapping
        != validation_mapping
    ):
        raise RuntimeError(
            "Train/Validation subject mapping mismatch."
        )

    subject_ids = [
        subject_id
        for (
            subject_id,
            index,
        ) in sorted(
            train_mapping.items(),
            key=lambda item: item[1],
        )
    ]

    if subject_ids[0] != "sub-010002":
        raise RuntimeError(
            "Unexpected first deterministic subject ID."
        )

    if subject_ids[-1] != "sub-010321":
        raise RuntimeError(
            "Unexpected last deterministic subject ID."
        )

    print(
        f"subjects: "
        f"{len(subject_ids)}"
    )

    print(
        f"subject index range: "
        f"0 ... {len(subject_ids)-1}"
    )

    print(
        f"first: "
        f"{subject_ids[0]} -> 0"
    )

    print(
        f"last: "
        f"{subject_ids[-1]} -> "
        f"{len(subject_ids)-1}"
    )

    print(
        "subject mapping freeze: PASS"
    )

    # ========================================================
    # Step 3 — Freeze B1 scientific / architecture contract
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — DEFINE B1 SUBJECT+STATE CONTRACT"
    )

    print(
        "=" * 80
    )

    b0_effective_contract = frozen_b0[
        "effective_contract"
    ]

    network = b0_effective_contract[
        "network"
    ]

    b1_protocol = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "status": (
            "B1_SUBJECT_STATE_PROTOCOL_FROZEN"
        ),
        "model_name": (
            "B1_SUBJECT_STATE_DDPM_V"
        ),
        "role": (
            "Subject- and eye-state-conditioned "
            "resting-EEG generator initialized "
            "from the frozen B0-v baseline."
        ),
        "parent_B0": {
            "frozen_generator": str(
                FROZEN_B0_GENERATOR
            ),
            "frozen_generator_sha256": (
                frozen_b0_hash
            ),
            "selected_epoch": 20,
            "selected_weight_kind": (
                "raw"
            ),
            "prediction_target": (
                "v"
            ),
            "parent_artifact_read_only": True,
        },
        "data": {
            "subjects": (
                EXPECTED_SUBJECTS
            ),
            "subject_ids_in_index_order": (
                subject_ids
            ),
            "train_windows": (
                EXPECTED_TRAIN_WINDOWS
            ),
            "validation_windows": (
                EXPECTED_VALIDATION_WINDOWS
            ),
            "input_channels": (
                INPUT_CHANNELS
            ),
            "input_samples": (
                INPUT_SAMPLES
            ),
            "sampling_rate_hz": (
                SFREQ
            ),
            "normalization": (
                "same locked global Train-only "
                "normalization as B0"
            ),
            "train_partition": (
                "eligible Real Train usable windows"
            ),
            "validation_partition": (
                "Real Validation only under "
                "predeclared evaluation rules"
            ),
            "final_test": (
                "SEALED"
            ),
        },
        "conditioning": {
            "eye_state": {
                "enabled": True,
                "mapping": {
                    "EO": 0,
                    "EC": 1,
                },
                "reuse_B0_state_embedding": True,
            },
            "subject": {
                "enabled": True,
                "num_subjects": (
                    EXPECTED_SUBJECTS
                ),
                "mapping": (
                    "frozen subject_ids_in_index_order"
                ),
                "embedding_dim": (
                    SUBJECT_EMBEDDING_DIM
                ),
                "projection_output_dim": (
                    CONDITION_DIM
                ),
                "fusion": (
                    "additive to existing "
                    "time+eye-state condition vector"
                ),
                "new_branch_initialization": {
                    "subject_embedding": (
                        "PyTorch default embedding "
                        "initialization"
                    ),
                    "subject_projection": (
                        "Linear(128,256) -> SiLU -> "
                        "Linear(256,256)"
                    ),
                    "final_projection_weight": (
                        "zero"
                    ),
                    "final_projection_bias": (
                        "zero"
                    ),
                    "reason": (
                        "At initialization, B1 exactly "
                        "preserves the frozen B0 function; "
                        "subject residual contribution starts "
                        "at zero and is learned during fine-tuning."
                    ),
                },
            },
            "B1_definition": (
                "p(EEG | subject, eye_state)"
            ),
        },
        "network": {
            "backbone": (
                "exact frozen B0 1D U-Net topology"
            ),
            "input_channels": (
                INPUT_CHANNELS
            ),
            "output_channels": (
                INPUT_CHANNELS
            ),
            "base_channels": int(
                network[
                    "base_channels"
                ]
            ),
            "channel_multipliers": list(
                network[
                    "channel_multipliers"
                ]
            ),
            "residual_blocks_per_level": int(
                network[
                    "residual_blocks_per_level"
                ]
            ),
            "time_embedding_dim": int(
                network[
                    "time_embedding_dim"
                ]
            ),
            "state_embedding_dim": int(
                network[
                    "state_embedding_dim"
                ]
            ),
            "subject_embedding_dim": (
                SUBJECT_EMBEDDING_DIM
            ),
            "attention": (
                network[
                    "attention"
                ]
            ),
            "normalization_layer": (
                network[
                    "normalization_layer"
                ]
            ),
            "activation": (
                network[
                    "activation"
                ]
            ),
            "dropout": float(
                network[
                    "dropout"
                ]
            ),
        },
        "diffusion": {
            "type": (
                "DDPM"
            ),
            "prediction_target": (
                "v"
            ),
            "timesteps": 1000,
            "beta_schedule": (
                "cosine"
            ),
            "cosine_s": (
                0.008
            ),
            "sampling_method_for_diagnostics": (
                "1000-step ancestral DDPM"
            ),
        },
        "warm_start": {
            "from_frozen_B0": True,
            "copy_all_matching_B0_weights": True,
            "new_parameters": [
                "subject_embedding",
                "subject_projection",
            ],
            "optimizer_state_reused": False,
            "fresh_optimizer": True,
            "all_backbone_parameters_trainable": True,
            "freeze_backbone": False,
            "pilot_checkpoint_reused_for_full_run": False,
        },
        "pilot": {
            "required_before_full_training": True,
            "steps": (
                PILOT_STEPS
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "partition": (
                "Real Train only"
            ),
            "validation_eeg_accessed": False,
            "final_test_accessed": False,
            "goals": [
                "verify exact B0 function preservation before any update",
                "verify subject conditioning path becomes non-zero",
                "verify eye-state conditioning remains active",
                "verify finite v-loss and gradients",
                "verify v-loss decreases on fixed Train probe",
                "verify no numerical instability",
            ],
        },
        "full_fine_tuning": {
            "epochs": (
                FULL_EPOCHS
            ),
            "early_stopping": False,
            "batch_size": (
                BATCH_SIZE
            ),
            "optimizer": (
                "AdamW"
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "learning_rate_schedule": (
                "none"
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),
            "mixed_precision": True,
            "amp_safe_skip": True,
            "ema_enabled": (
                USE_EMA
            ),
            "selection_weights": (
                "RAW"
            ),
            "checkpoint_every_epochs": (
                CHECKPOINT_EVERY_EPOCHS
            ),
            "automatic_resume": True,
            "seed": (
                SEED
            ),
        },
        "checkpoint_selection": {
            "during_training_monitor": (
                "deterministic RAW Validation v-MSE"
            ),
            "final_generator_selection": (
                "predeclared RAW milestone comparison "
                "after full fine-tuning"
            ),
            "predeclared_RAW_candidate_epochs": [
                10,
                20,
                30,
            ],
            "selection_evidence_partition": (
                "Train-only generative fidelity reference, "
                "with Validation v-MSE reported but not used "
                "as the sole generator-fidelity criterion"
            ),
            "reason": (
                "B0 experiments showed lower EMA Validation "
                "loss did not imply better generative fidelity; "
                "RAW generator checkpoints are therefore "
                "predeclared for B1 scientific selection."
            ),
        },
        "B1_evaluation_focus": {
            "retain_B0_metrics": [
                "amplitude scale",
                "PSD and bandpower",
                "EO/EC alpha effect",
                "channel covariance/correlation",
                "alpha debiased-squared wPLI",
            ],
            "add_subject_specific_metrics": [
                "subject-wise log-bandpower fingerprint similarity",
                "subject-wise channel-power profile similarity",
                "subject-conditioned versus wrong-subject contrast",
                "subject-macro generative fidelity summaries",
            ],
            "connectivity_limitation_from_B0_to_track": (
                "alpha phase-lag connectivity, "
                "especially EO"
            ),
        },
        "scientific_control_note": {
            "matched_compute_B0_continuation": (
                "Recommended if making a strict causal claim "
                "that B1 improvements arise only from subject "
                "conditioning rather than additional fine-tuning."
            ),
            "required_before_starting_B1": False,
            "required_before_final_publication_claim_if_needed": True,
        },
        "prohibitions": [
            "do not modify frozen B0 artifact",
            "do not use Final Test",
            "do not use per-subject normalization",
            "do not add spectral regularization in B1",
            "do not add connectivity loss in B1",
            "do not perform bulk synthetic generation before B0/B1/B2 are all frozen",
            "do not reuse engineering pilot checkpoint as the scientific full-run initialization",
        ],
        "future_B2": {
            "initialize_from_frozen_B1": True,
            "retain_subject_and_eye_state_conditioning": True,
            "continue_v_prediction": True,
            "add_spectral_regularization": True,
        },
        "governance": {
            "protocol_frozen_before_B1_model_updates": True,
            "B0_read_only": True,
            "Validation_allowed_only_under_this_protocol": True,
            "Final_Test_accessed": False,
            "bulk_generation_started": False,
        },
    }

    # ========================================================
    # Step 4 — Validate B1 contract consistency
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — B1 CONTRACT CONSISTENCY CHECK"
    )

    print(
        "=" * 80
    )

    assert (
        b1_protocol[
            "conditioning"
        ][
            "subject"
        ][
            "enabled"
        ]
        is True
    )

    assert (
        b1_protocol[
            "conditioning"
        ][
            "eye_state"
        ][
            "enabled"
        ]
        is True
    )

    assert (
        b1_protocol[
            "diffusion"
        ][
            "prediction_target"
        ]
        == "v"
    )

    assert (
        b1_protocol[
            "warm_start"
        ][
            "from_frozen_B0"
        ]
        is True
    )

    assert (
        b1_protocol[
            "warm_start"
        ][
            "optimizer_state_reused"
        ]
        is False
    )

    assert (
        b1_protocol[
            "full_fine_tuning"
        ][
            "ema_enabled"
        ]
        is False
    )

    assert (
        b1_protocol[
            "full_fine_tuning"
        ][
            "selection_weights"
        ]
        == "RAW"
    )

    assert (
        b1_protocol[
            "future_B2"
        ][
            "add_spectral_regularization"
        ]
        is True
    )

    assert (
        b1_protocol[
            "governance"
        ][
            "Final_Test_accessed"
        ]
        is False
    )

    print(
        "B1 definition:"
    )

    print(
        "  p(EEG | subject, eye_state)"
    )

    print(
        "\nWarm start:"
    )

    print(
        "  frozen B0-v RAW epoch 20"
    )

    print(
        "\nNew conditioning:"
    )

    print(
        f"  subject embedding: "
        f"198 × {SUBJECT_EMBEDDING_DIM}"
    )

    print(
        "  subject contribution initialized at zero"
    )

    print(
        "\nFine-tuning:"
    )

    print(
        f"  pilot = {PILOT_STEPS} steps"
    )

    print(
        f"  full = {FULL_EPOCHS} epochs"
    )

    print(
        f"  LR = {LEARNING_RATE}"
    )

    print(
        "  RAW checkpoint selection"
    )

    print(
        "  EMA disabled"
    )

    print(
        "contract consistency: PASS"
    )

    # ========================================================
    # Step 5 — Save frozen protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — SAVE FROZEN B1 PROTOCOL"
    )

    print(
        "=" * 80
    )

    if B1_PROTOCOL_FILE.exists():
        with B1_PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != b1_protocol:
            raise RuntimeError(
                "A conflicting B1 protocol already exists."
            )

        print(
            "identical B1 protocol already exists."
        )

    else:
        atomic_json_save(
            b1_protocol,
            B1_PROTOCOL_FILE,
        )

    print(
        f"protocol: "
        f"{B1_PROTOCOL_FILE}"
    )

    # ========================================================
    # Step 6 — Save human-readable summary
    # ========================================================

    summary_text = f"""SCENARIO 2 — FROZEN B1 SUBJECT-CONDITIONED PROTOCOL

STATUS
B1_SUBJECT_STATE_PROTOCOL_FROZEN

PARENT
Frozen B0: B0-v RAW epoch 20
Frozen B0 generator:
{FROZEN_B0_GENERATOR}

B1 DEFINITION
p(EEG | subject, eye_state)

DATA
Subjects: {EXPECTED_SUBJECTS}
Train windows: {EXPECTED_TRAIN_WINDOWS}
Validation windows: {EXPECTED_VALIDATION_WINDOWS}
Input: {INPUT_CHANNELS} channels × {INPUT_SAMPLES} samples
Sampling rate: {SFREQ} Hz
Normalization: same global Train-only scaler as B0
Final Test: SEALED

CONDITIONING
EO = 0
EC = 1
Subject index = deterministic 0..197
Subject embedding dimension = {SUBJECT_EMBEDDING_DIM}
Subject projection output = {CONDITION_DIM}
Fusion = additive with existing time+eye-state condition

TRANSFER INITIALIZATION
All matching B0 weights are copied from frozen B0.
New subject branch is initialized so its final residual contribution is exactly zero.
Therefore B1 before the first optimizer update must reproduce the B0 function.
Fresh AdamW optimizer is used.
All backbone parameters remain trainable.

PILOT
{PILOT_STEPS} Train-only optimizer steps
No Validation EEG
No Final Test
Pilot checkpoint is not reused as the scientific full-run initialization.

FULL FINE-TUNING
Epochs = {FULL_EPOCHS}
Batch size = {BATCH_SIZE}
Learning rate = {LEARNING_RATE}
Weight decay = {WEIGHT_DECAY}
Gradient clip = {GRADIENT_CLIP_NORM}
Mixed precision = ON
EMA = OFF
Selection weights = RAW
Checkpoints every {CHECKPOINT_EVERY_EPOCHS} epochs
Automatic resume = ON

PREDECLARED RAW MILESTONE CANDIDATES
Epoch 10
Epoch 20
Epoch 30

B1 EVALUATION
Retain B0 amplitude/spectral/spatial/connectivity metrics.
Add subject-specific fingerprint and wrong-subject contrast metrics.

KNOWN B0 LIMITATION TO TRACK
Alpha phase-lag connectivity fidelity was limited in B0, especially EO.
B1 must be evaluated for whether subject conditioning improves, preserves, or worsens this.

BULK GENERATION
NO bulk synthetic generation until B0, B1, and B2 are all frozen.

NEXT
Implement B1 subject-conditioning architecture and verify exact warm-start equivalence to frozen B0 before training.
"""

    atomic_text_save(
        summary_text,
        B1_SUMMARY_FILE,
    )

    print(
        f"summary: "
        f"{B1_SUMMARY_FILE}"
    )

    # ========================================================
    # Step 7 — Write architecture implementation handoff
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — WRITE B1 ARCHITECTURE HANDOFF"
    )

    print(
        "=" * 80
    )

    implementation_handoff = {
        "status": (
            "B1_ARCHITECTURE_IMPLEMENTATION_READY"
        ),
        "protocol": str(
            B1_PROTOCOL_FILE
        ),
        "frozen_B0_generator": str(
            FROZEN_B0_GENERATOR
        ),
        "B0_state_dict_key": (
            "model_state_dict"
        ),
        "B0_effective_contract_key": (
            "effective_contract"
        ),
        "subject_ids": (
            subject_ids
        ),
        "required_B1_model_changes": {
            "reuse_B0_backbone_topology": True,
            "reuse_B0_time_embedding": True,
            "reuse_B0_eye_state_embedding": True,
            "add_subject_embedding": {
                "num_embeddings": (
                    EXPECTED_SUBJECTS
                ),
                "embedding_dim": (
                    SUBJECT_EMBEDDING_DIM
                ),
            },
            "add_subject_projection": [
                (
                    f"Linear({SUBJECT_EMBEDDING_DIM},"
                    f"{CONDITION_DIM})"
                ),
                "SiLU",
                (
                    f"Linear({CONDITION_DIM},"
                    f"{CONDITION_DIM})"
                ),
            ],
            "zero_initialize_final_subject_projection": True,
            "condition_formula": (
                "time_condition + "
                "eye_state_condition + "
                "subject_condition"
            ),
        },
        "first_required_test": {
            "name": (
                "exact_B0_equivalence_before_training"
            ),
            "description": (
                "For identical x, timestep, and eye state, "
                "B1 outputs must equal frozen B0 outputs "
                "for every subject before the first optimizer "
                "update because the subject residual branch "
                "starts at exactly zero."
            ),
            "tolerance_max_abs": (
                1e-7
            ),
        },
        "second_required_test": {
            "name": (
                "subject_path_learns_after_update"
            ),
            "description": (
                "After a controlled Train-only optimizer "
                "update, changing subject ID must produce "
                "a non-zero output difference while the "
                "source frozen B0 artifact remains unchanged."
            ),
        },
        "training_target": (
            "v"
        ),
        "final_test": (
            "SEALED"
        ),
    }

    atomic_json_save(
        implementation_handoff,
        B1_IMPLEMENTATION_HANDOFF,
    )

    print(
        f"handoff: "
        f"{B1_IMPLEMENTATION_HANDOFF}"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B1 SUBJECT-CONDITIONED PROTOCOL FROZEN"
    )

    print(
        "=" * 80
    )

    print(
        "Parent B0: FROZEN B0-v RAW epoch 20"
    )

    print(
        "B1 target: p(EEG | subject, eye_state)"
    )

    print(
        f"subjects: {EXPECTED_SUBJECTS}"
    )

    print(
        f"subject embedding dim: "
        f"{SUBJECT_EMBEDDING_DIM}"
    )

    print(
        "prediction target: v"
    )

    print(
        f"pilot steps: "
        f"{PILOT_STEPS}"
    )

    print(
        f"full fine-tuning epochs: "
        f"{FULL_EPOCHS}"
    )

    print(
        f"learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        "EMA: OFF"
    )

    print(
        "scientific checkpoint weights: RAW"
    )

    print(
        "Validation EEG accessed in this freeze step: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Bulk synthetic generation started: NO"
    )

    print(
        "B1 training started: NO"
    )

    print(
        f"\nProtocol:"
        f"\n{B1_PROTOCOL_FILE}"
    )

    print(
        f"\nImplementation handoff:"
        f"\n{B1_IMPLEMENTATION_HANDOFF}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "IMPLEMENT B1 SUBJECT-CONDITIONED U-NET "
        "AND VERIFY EXACT B0 WARM-START EQUIVALENCE"
    )


if __name__ == "__main__":
    main()
