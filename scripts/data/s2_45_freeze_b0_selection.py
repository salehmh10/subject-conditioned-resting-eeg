from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

from s2_32_b0_unet_shape_test import B0StateOnlyUNet1D


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

FULL_FINETUNE_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "full_finetune_v1"
)

SELECTED_SOURCE_CHECKPOINT = (
    FULL_FINETUNE_DIR
    / "checkpoints"
    / "epoch_020.pt"
)

CONFIRMATORY_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "raw_candidate_confirmatory_v1"
)

CONFIRMATORY_SUMMARY = (
    CONFIRMATORY_DIR
    / "metrics"
    / "b0_v_raw_candidate_confirmatory_summary_v1.json"
)

CONFIRMATORY_PROTOCOL = (
    CONFIRMATORY_DIR
    / "metrics"
    / "b0_v_raw_candidate_confirmatory_protocol_v1.json"
)

CONFIRMATORY_METRICS_CSV = (
    CONFIRMATORY_DIR
    / "metrics"
    / "b0_v_raw_candidate_confirmatory_metrics_v1.csv"
)


# ============================================================
# Frozen B0 output tree
# ============================================================

FREEZE_ROOT = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b0"
    / "b0_v_raw_epoch020_v1"
)

CHECKPOINT_DIR = (
    FREEZE_ROOT
    / "checkpoints"
)

METRIC_DIR = (
    FREEZE_ROOT
    / "metrics"
)

HANDOFF_DIR = (
    FREEZE_ROOT
    / "handoff"
)

ARCHIVE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "b0_v_raw_epoch020_full_checkpoint_v1.pt"
)

GENERATOR_CHECKPOINT = (
    CHECKPOINT_DIR
    / "b0_v_raw_epoch020_generator_v1.pt"
)

SELECTION_FILE = (
    METRIC_DIR
    / "b0_frozen_selection_v1.json"
)

SUMMARY_TEXT_FILE = (
    METRIC_DIR
    / "b0_frozen_selection_summary_v1.txt"
)

B1_HANDOFF_FILE = (
    HANDOFF_DIR
    / "b0_to_b1_handoff_v1.json"
)


# ============================================================
# Frozen selection decision
# ============================================================

SELECTED_EPOCH = 20
SELECTED_WEIGHT_KIND = "raw"

EXPECTED_RECOMMENDED_EPOCH = 20
EXPECTED_DIAGNOSTIC_DIRECTION = (
    "RAW_EPOCH_20_PRIMARY_CONFIRMATORY_CANDIDATE"
)

EXPECTED_CANDIDATE_EPOCHS = [
    20,
    40,
]

EXPECTED_CRITERION_WINS = {
    "20": 4,
    "40": 1,
}

EXPECTED_SOURCE_EPSILON_EPOCH = 125
EXPECTED_MODEL_CHANNELS = 19
EXPECTED_MODEL_SAMPLES = 512


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


def state_dict_sha256(
    state_dict: dict[str, torch.Tensor],
) -> str:
    """
    Stable hash of model tensors only.

    This identifies the selected RAW generator weights independently
    of optimizer, EMA, RNG, or other full-checkpoint metadata.
    """

    digest = hashlib.sha256()

    for name in sorted(
        state_dict.keys()
    ):
        tensor = (
            state_dict[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            json.dumps(
                list(
                    tensor.shape
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            tensor.numpy()
            .tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


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


def assert_finite_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> None:
    for name, tensor in state_dict.items():
        if (
            torch.is_floating_point(
                tensor
            )
            and not torch.isfinite(
                tensor
            ).all()
        ):
            raise RuntimeError(
                f"Nonfinite selected weight tensor: "
                f"{name}"
            )


def atomic_json_save(
    data: dict,
    path: Path,
) -> None:
    temp_path = Path(
        str(path)
        + ".tmp"
    )

    with temp_path.open(
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
        temp_path,
        path,
    )


def atomic_text_save(
    text: str,
    path: Path,
) -> None:
    temp_path = Path(
        str(path)
        + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(
            text
        )

    os.replace(
        temp_path,
        path,
    )


def atomic_torch_save(
    data: dict,
    path: Path,
) -> None:
    temp_path = Path(
        str(path)
        + ".tmp"
    )

    torch.save(
        data,
        temp_path,
    )

    os.replace(
        temp_path,
        path,
    )


def atomic_copy(
    source: Path,
    destination: Path,
) -> None:
    temp_path = Path(
        str(destination)
        + ".tmp"
    )

    shutil.copy2(
        source,
        temp_path,
    )

    os.replace(
        temp_path,
        destination,
    )


def get_candidate_row(
    summary: dict,
    epoch: int,
) -> dict:
    matches = [
        row
        for row in summary[
            "candidate_metrics"
        ]
        if int(
            row[
                "epoch"
            ]
        )
        == epoch
        and row[
            "weight_kind"
        ]
        == "raw"
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one RAW candidate "
            f"row for epoch {epoch}; "
            f"found {len(matches)}."
        )

    return matches[0]


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — FREEZE FINAL B0 SELECTION"
    )

    print(
        "=" * 80
    )

    print(
        "\nCurrent project stage:"
    )

    print(
        "  B0 epsilon pretraining: COMPLETE"
    )

    print(
        "  B0-v fine-tuning: COMPLETE"
    )

    print(
        "  RAW vs EMA diagnostic: COMPLETE"
    )

    print(
        "  RAW epoch-20 vs epoch-40 "
        "confirmatory audit: COMPLETE"
    )

    print(
        "  Formal B0 freeze: NOW"
    )

    print(
        "\nFrozen scientific selection:"
    )

    print(
        "  model family: B0-v state-only DDPM"
    )

    print(
        f"  selected fine-tune epoch: "
        f"{SELECTED_EPOCH}"
    )

    print(
        f"  selected weights: "
        f"{SELECTED_WEIGHT_KIND.upper()}"
    )

    print(
        "  EMA selected for generation: NO"
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
    # Step 1 — Audit confirmatory evidence
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT CONFIRMATORY SELECTION EVIDENCE"
    )

    print(
        "=" * 80
    )

    required_files = [
        SELECTED_SOURCE_CHECKPOINT,
        CONFIRMATORY_SUMMARY,
        CONFIRMATORY_PROTOCOL,
        CONFIRMATORY_METRICS_CSV,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with CONFIRMATORY_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        audit_summary = json.load(
            f
        )

    with CONFIRMATORY_PROTOCOL.open(
        "r",
        encoding="utf-8",
    ) as f:
        audit_protocol = json.load(
            f
        )

    if audit_summary[
        "status"
    ] != (
        "B0_V_RAW_CANDIDATE_"
        "CONFIRMATORY_AUDIT_COMPLETE"
    ):
        raise RuntimeError(
            "Unexpected confirmatory audit status."
        )

    if list(
        audit_summary[
            "candidate_epochs"
        ]
    ) != EXPECTED_CANDIDATE_EPOCHS:
        raise RuntimeError(
            "Confirmatory candidate epoch set "
            "does not match the frozen decision."
        )

    if audit_summary[
        "weight_kind"
    ] != "raw":
        raise RuntimeError(
            "Confirmatory audit did not evaluate "
            "RAW candidate weights."
        )

    if int(
        audit_summary[
            "recommended_epoch"
        ]
    ) != EXPECTED_RECOMMENDED_EPOCH:
        raise RuntimeError(
            "Confirmatory audit does not recommend "
            "RAW epoch 20."
        )

    if audit_summary[
        "diagnostic_direction"
    ] != EXPECTED_DIAGNOSTIC_DIRECTION:
        raise RuntimeError(
            "Unexpected confirmatory diagnostic "
            "direction."
        )

    normalized_win_counts = {
        str(
            key
        ): int(
            value
        )
        for key, value
        in audit_summary[
            "criterion_win_counts"
        ].items()
    }

    if normalized_win_counts != EXPECTED_CRITERION_WINS:
        raise RuntimeError(
            "Confirmatory criterion-win counts "
            "do not match expected 4:1 decision."
        )

    if audit_summary[
        "validation_eeg_accessed"
    ] is not False:
        raise RuntimeError(
            "Confirmatory audit unexpectedly "
            "accessed Validation EEG."
        )

    if audit_summary[
        "final_test_accessed"
    ] is not False:
        raise RuntimeError(
            "Confirmatory audit unexpectedly "
            "accessed Final Test."
        )

    if audit_summary[
        "training_updates_executed"
    ] is not False:
        raise RuntimeError(
            "Confirmatory audit unexpectedly "
            "performed training updates."
        )

    selected_metrics = get_candidate_row(
        audit_summary,
        SELECTED_EPOCH,
    )

    alternative_metrics = get_candidate_row(
        audit_summary,
        40,
    )

    print(
        "confirmatory audit: PASS"
    )

    print(
        f"recommended candidate: "
        f"RAW epoch "
        f"{audit_summary['recommended_epoch']}"
    )

    print(
        "criterion wins: "
        f"{normalized_win_counts}"
    )

    # ========================================================
    # Step 2 — Audit selected epoch-20 RAW checkpoint
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — AUDIT SELECTED RAW EPOCH-20 CHECKPOINT"
    )

    print(
        "=" * 80
    )

    source_hash_before = sha256_file(
        SELECTED_SOURCE_CHECKPOINT
    )

    expected_hash = (
        audit_summary[
            "checkpoint_hashes"
        ][
            str(
                SELECTED_EPOCH
            )
        ]
    )

    if (
        source_hash_before
        != expected_hash
    ):
        raise RuntimeError(
            "Selected epoch-20 checkpoint SHA256 "
            "does not match confirmatory evidence."
        )

    checkpoint = torch.load(
        SELECTED_SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint[
        "status"
    ] != "B0_V_FULL_FINETUNE_CHECKPOINT":
        raise RuntimeError(
            "Selected file is not a B0-v "
            "fine-tuning checkpoint."
        )

    if int(
        checkpoint[
            "epoch"
        ]
    ) != SELECTED_EPOCH:
        raise RuntimeError(
            "Selected checkpoint epoch mismatch."
        )

    if checkpoint[
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "Selected checkpoint is not "
            "v-prediction."
        )

    if int(
        checkpoint[
            "source_epoch"
        ]
    ) != EXPECTED_SOURCE_EPSILON_EPOCH:
        raise RuntimeError(
            "Unexpected epsilon-pretraining "
            "source epoch."
        )

    if checkpoint[
        "final_test_accessed"
    ] is not False:
        raise RuntimeError(
            "Selected checkpoint reports "
            "Final Test access."
        )

    if checkpoint[
        "pilot_checkpoint_used"
    ] is not False:
        raise RuntimeError(
            "Scientific B0-v run unexpectedly "
            "used the engineering pilot checkpoint."
        )

    selected_raw_state = cpu_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    assert_finite_state_dict(
        selected_raw_state
    )

    selected_weights_hash = (
        state_dict_sha256(
            selected_raw_state
        )
    )

    print(
        f"selected checkpoint SHA256: "
        f"{source_hash_before}"
    )

    print(
        f"selected RAW tensor SHA256: "
        f"{selected_weights_hash}"
    )

    print(
        "selected checkpoint audit: PASS"
    )

    # ========================================================
    # Step 3 — Create effective frozen B0-v contract
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD EFFECTIVE FROZEN B0-V CONTRACT"
    )

    print(
        "=" * 80
    )

    original_source_contract = copy.deepcopy(
        checkpoint[
            "source_contract"
        ]
    )

    effective_contract = copy.deepcopy(
        original_source_contract
    )

    effective_contract[
        "model_name"
    ] = (
        "B0_STATE_ONLY_DDPM_V_FROZEN"
    )

    effective_contract[
        "diffusion"
    ][
        "prediction_target"
    ] = "v"

    effective_contract[
        "training"
    ][
        "train_from_scratch"
    ] = False

    effective_contract[
        "training"
    ][
        "pretrained_model"
    ] = True

    effective_contract[
        "frozen_selection"
    ] = {
        "selected_epoch": (
            SELECTED_EPOCH
        ),
        "weight_kind": (
            SELECTED_WEIGHT_KIND
        ),
        "epsilon_pretraining_source_epoch": (
            EXPECTED_SOURCE_EPSILON_EPOCH
        ),
        "v_finetune_epochs_executed": 40,
        "selected_from_v_finetune_epoch": (
            SELECTED_EPOCH
        ),
        "selection_basis": (
            "Train-only confirmatory generative "
            "fidelity audit"
        ),
    }

    if int(
        effective_contract[
            "data"
        ][
            "input_channels"
        ]
    ) != EXPECTED_MODEL_CHANNELS:
        raise RuntimeError(
            "Frozen contract channel count mismatch."
        )

    if int(
        effective_contract[
            "data"
        ][
            "input_samples"
        ]
    ) != EXPECTED_MODEL_SAMPLES:
        raise RuntimeError(
            "Frozen contract sample-length mismatch."
        )

    print(
        "effective prediction target: v"
    )

    print(
        "subject conditioning: NO"
    )

    print(
        "eye-state conditioning: YES"
    )

    print(
        "effective frozen contract: PASS"
    )

    # ========================================================
    # Step 4 — Archive full source checkpoint byte-for-byte
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — ARCHIVE SELECTED FULL CHECKPOINT"
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
                "Existing frozen archive conflicts "
                "with selected source checkpoint."
            )

        print(
            "identical frozen archive already exists."
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
                "Frozen archive SHA256 verification "
                "failed after copy."
            )

        print(
            "full checkpoint archived byte-for-byte."
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
    # Step 5 — Save lean frozen generator checkpoint
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — SAVE LEAN FROZEN B0 GENERATOR"
    )

    print(
        "=" * 80
    )

    generator_payload = {
        "status": (
            "B0_V_FROZEN_GENERATOR"
        ),
        "freeze_schema_version": 1,
        "model_name": (
            "B0_STATE_ONLY_DDPM_V_FROZEN"
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
        "selected_source_checkpoint": str(
            SELECTED_SOURCE_CHECKPOINT
        ),
        "selected_source_checkpoint_sha256": (
            source_hash_before
        ),
        "selected_model_state_sha256": (
            selected_weights_hash
        ),
        "epsilon_pretraining_source_epoch": int(
            checkpoint[
                "source_epoch"
            ]
        ),
        "epsilon_pretraining_source_sha256": (
            checkpoint[
                "source_sha256"
            ]
        ),
        "effective_contract": (
            effective_contract
        ),
        "original_source_contract": (
            original_source_contract
        ),
        "v_finetune_protocol": (
            checkpoint[
                "fine_tune_protocol"
            ]
        ),
        "model_state_dict": (
            selected_raw_state
        ),
        "selection_evidence": {
            "confirmatory_summary": str(
                CONFIRMATORY_SUMMARY
            ),
            "confirmatory_summary_sha256": (
                sha256_file(
                    CONFIRMATORY_SUMMARY
                )
            ),
            "confirmatory_protocol": str(
                CONFIRMATORY_PROTOCOL
            ),
            "confirmatory_protocol_sha256": (
                sha256_file(
                    CONFIRMATORY_PROTOCOL
                )
            ),
            "confirmatory_metrics_csv": str(
                CONFIRMATORY_METRICS_CSV
            ),
            "confirmatory_metrics_csv_sha256": (
                sha256_file(
                    CONFIRMATORY_METRICS_CSV
                )
            ),
        },
        "known_limitations": {
            "alpha_dwpli2_connectivity_fidelity": (
                "limited_and_state_asymmetric"
            ),
            "eo_alpha_dwpli2_upper_triangle_r": float(
                selected_metrics[
                    "alpha_dwpli2_upper_triangle_r_eo"
                ]
            ),
            "ec_alpha_dwpli2_upper_triangle_r": float(
                selected_metrics[
                    "alpha_dwpli2_upper_triangle_r_ec"
                ]
            ),
            "eo_alpha_dwpli2_relative_frobenius": float(
                selected_metrics[
                    "alpha_dwpli2_relF_eo"
                ]
            ),
            "ec_alpha_dwpli2_relative_frobenius": float(
                selected_metrics[
                    "alpha_dwpli2_relF_ec"
                ]
            ),
            "interpretation": (
                "B0 is frozen as a state-only baseline "
                "with strong amplitude/spectral/spatial-"
                "correlation fidelity but limited phase-lag "
                "connectivity fidelity, especially EO. "
                "No claim of complete connectivity fidelity "
                "is made."
            ),
        },
        "validation_eeg_accessed_during_selection": False,
        "final_test_accessed": False,
        "bulk_synthetic_generation_started": False,
        "b0_frozen": True,
    }

    if GENERATOR_CHECKPOINT.exists():
        existing_generator = torch.load(
            GENERATOR_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        if (
            existing_generator.get(
                "selected_model_state_sha256"
            )
            != selected_weights_hash
            or int(
                existing_generator.get(
                    "selected_epoch",
                    -1,
                )
            )
            != SELECTED_EPOCH
            or existing_generator.get(
                "selected_weight_kind"
            )
            != SELECTED_WEIGHT_KIND
        ):
            raise RuntimeError(
                "Existing frozen generator conflicts "
                "with selected B0 identity."
            )

        print(
            "identical frozen generator already exists."
        )

    else:
        atomic_torch_save(
            generator_payload,
            GENERATOR_CHECKPOINT,
        )

        print(
            "lean frozen generator saved."
        )

    print(
        f"generator: "
        f"{GENERATOR_CHECKPOINT}"
    )

    # ========================================================
    # Step 6 — Strict reload / forward smoke test
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — STRICT RELOAD + FORWARD SMOKE TEST"
    )

    print(
        "=" * 80
    )

    frozen_generator = torch.load(
        GENERATOR_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    reloaded_hash = state_dict_sha256(
        frozen_generator[
            "model_state_dict"
        ]
    )

    if (
        reloaded_hash
        != selected_weights_hash
    ):
        raise RuntimeError(
            "Frozen generator tensor hash mismatch."
        )

    model = B0StateOnlyUNet1D(
        frozen_generator[
            "effective_contract"
        ]
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
                EXPECTED_MODEL_CHANNELS,
                EXPECTED_MODEL_SAMPLES,
            ),
            dtype=torch.float32,
        )

        timesteps = torch.tensor(
            [
                10,
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

        output = model(
            x,
            timesteps,
            eye_state,
        )

    if output.shape != (
        2,
        EXPECTED_MODEL_CHANNELS,
        EXPECTED_MODEL_SAMPLES,
    ):
        raise RuntimeError(
            f"Frozen generator output shape "
            f"mismatch: {output.shape}"
        )

    if not torch.isfinite(
        output
    ).all():
        raise RuntimeError(
            "Frozen generator smoke-test "
            "output contains NaN/Inf."
        )

    print(
        f"parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        "strict reload: PASS"
    )

    print(
        "forward smoke test: PASS"
    )

    # ========================================================
    # Step 7 — Build formal selection manifest
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — WRITE FORMAL B0 FREEZE MANIFEST"
    )

    print(
        "=" * 80
    )

    selection_manifest = {
        "status": (
            "B0_SELECTION_FROZEN"
        ),
        "freeze_schema_version": 1,
        "b0_frozen": True,
        "selected_model": (
            "B0_STATE_ONLY_DDPM_V"
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
        "selection_timeline": [
            {
                "stage": (
                    "epsilon_pretraining"
                ),
                "epoch": (
                    EXPECTED_SOURCE_EPSILON_EPOCH
                ),
                "role": (
                    "representation and state-conditioned "
                    "diffusion pretraining"
                ),
            },
            {
                "stage": (
                    "v_prediction_finetuning"
                ),
                "epochs_executed": 40,
                "selected_epoch": (
                    SELECTED_EPOCH
                ),
                "selected_weight_kind": (
                    SELECTED_WEIGHT_KIND
                ),
            },
            {
                "stage": (
                    "raw_vs_ema_diagnostic"
                ),
                "result": (
                    "RAW weights consistently outperformed "
                    "slow EMA on key generator-fidelity "
                    "diagnostics"
                ),
            },
            {
                "stage": (
                    "confirmatory_candidate_audit"
                ),
                "candidates": [
                    20,
                    40,
                ],
                "recommended_epoch": (
                    EXPECTED_RECOMMENDED_EPOCH
                ),
                "criterion_win_counts": (
                    normalized_win_counts
                ),
            },
        ],
        "selection_basis": {
            "partition": (
                "Real Train reference only"
            ),
            "validation_eeg_used_for_fidelity_selection": False,
            "final_test_used": False,
            "confirmatory_synthetic_per_candidate": {
                "EO": 256,
                "EC": 256,
            },
            "principal_named_criteria": [
                "scale_fidelity",
                "alpha_effect_fidelity",
                "alpha_bandpower_fidelity",
                "spatial_correlation_fidelity",
                "alpha_dwpli2_fidelity",
            ],
            "criterion_winners": (
                audit_summary[
                    "criterion_winners"
                ]
            ),
            "criterion_win_counts": (
                normalized_win_counts
            ),
            "decision_rule": (
                "Primary candidate auto-recommended "
                "when winning at least 3 of 5 "
                "predefined confirmatory criteria."
            ),
        },
        "selected_confirmatory_metrics": {
            "normalized_std_ratio_EO": float(
                selected_metrics[
                    "eo_normalized_std_ratio"
                ]
            ),
            "normalized_std_ratio_EC": float(
                selected_metrics[
                    "ec_normalized_std_ratio"
                ]
            ),
            "uv_std_ratio_EO": float(
                selected_metrics[
                    "eo_uv_std_ratio"
                ]
            ),
            "uv_std_ratio_EC": float(
                selected_metrics[
                    "ec_uv_std_ratio"
                ]
            ),
            "delta_bandpower_MAE_dB": float(
                selected_metrics[
                    "delta_bandpower_mae_db"
                ]
            ),
            "theta_bandpower_MAE_dB": float(
                selected_metrics[
                    "theta_bandpower_mae_db"
                ]
            ),
            "alpha_bandpower_MAE_dB": float(
                selected_metrics[
                    "alpha_bandpower_mae_db"
                ]
            ),
            "beta_bandpower_MAE_dB": float(
                selected_metrics[
                    "beta_bandpower_mae_db"
                ]
            ),
            "alpha_effect_channel_MAE_dB": float(
                selected_metrics[
                    "alpha_effect_channel_mae_db"
                ]
            ),
            "O1_real_alpha_effect_dB": float(
                selected_metrics[
                    "o1_real_alpha_effect_db"
                ]
            ),
            "O1_synthetic_alpha_effect_dB": float(
                selected_metrics[
                    "o1_synthetic_alpha_effect_db"
                ]
            ),
            "O2_real_alpha_effect_dB": float(
                selected_metrics[
                    "o2_real_alpha_effect_db"
                ]
            ),
            "O2_synthetic_alpha_effect_dB": float(
                selected_metrics[
                    "o2_synthetic_alpha_effect_db"
                ]
            ),
            "correlation_r_EO": float(
                selected_metrics[
                    "correlation_upper_triangle_r_eo"
                ]
            ),
            "correlation_r_EC": float(
                selected_metrics[
                    "correlation_upper_triangle_r_ec"
                ]
            ),
            "alpha_dwPLI2_r_EO": float(
                selected_metrics[
                    "alpha_dwpli2_upper_triangle_r_eo"
                ]
            ),
            "alpha_dwPLI2_r_EC": float(
                selected_metrics[
                    "alpha_dwpli2_upper_triangle_r_ec"
                ]
            ),
            "alpha_dwPLI2_relF_EO": float(
                selected_metrics[
                    "alpha_dwpli2_relF_eo"
                ]
            ),
            "alpha_dwPLI2_relF_EC": float(
                selected_metrics[
                    "alpha_dwpli2_relF_ec"
                ]
            ),
        },
        "alternative_epoch40_metrics_recorded_for_transparency": {
            "normalized_std_ratio_EO": float(
                alternative_metrics[
                    "eo_normalized_std_ratio"
                ]
            ),
            "normalized_std_ratio_EC": float(
                alternative_metrics[
                    "ec_normalized_std_ratio"
                ]
            ),
            "alpha_effect_channel_MAE_dB": float(
                alternative_metrics[
                    "alpha_effect_channel_mae_db"
                ]
            ),
            "alpha_bandpower_MAE_dB": float(
                alternative_metrics[
                    "alpha_bandpower_mae_db"
                ]
            ),
            "alpha_dwPLI2_r_EO": float(
                alternative_metrics[
                    "alpha_dwpli2_upper_triangle_r_eo"
                ]
            ),
            "alpha_dwPLI2_r_EC": float(
                alternative_metrics[
                    "alpha_dwpli2_upper_triangle_r_ec"
                ]
            ),
        },
        "known_limitations": {
            "connectivity": {
                "status": (
                    "LIMITED"
                ),
                "description": (
                    "Alpha debiased-squared wPLI fidelity "
                    "is weak/state-asymmetric, especially EO. "
                    "B0 is therefore not claimed to reproduce "
                    "complete phase-lag connectivity structure."
                ),
                "EO_upper_triangle_r": float(
                    selected_metrics[
                        "alpha_dwpli2_upper_triangle_r_eo"
                    ]
                ),
                "EC_upper_triangle_r": float(
                    selected_metrics[
                        "alpha_dwpli2_upper_triangle_r_ec"
                    ]
                ),
                "EO_relative_frobenius": float(
                    selected_metrics[
                        "alpha_dwpli2_relF_eo"
                    ]
                ),
                "EC_relative_frobenius": float(
                    selected_metrics[
                        "alpha_dwpli2_relF_ec"
                    ]
                ),
            },
            "interpretation_boundary": (
                "B0 is the state-only baseline generator. "
                "Its known connectivity limitation is retained "
                "rather than adding subject/spectral/connectivity "
                "mechanisms that would blur B0/B1/B2 roles."
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
                selected_weights_hash
            ),
            "confirmatory_summary": str(
                CONFIRMATORY_SUMMARY
            ),
            "confirmatory_summary_sha256": (
                sha256_file(
                    CONFIRMATORY_SUMMARY
                )
            ),
            "confirmatory_protocol": str(
                CONFIRMATORY_PROTOCOL
            ),
            "confirmatory_metrics_csv": str(
                CONFIRMATORY_METRICS_CSV
            ),
        },
        "governance": {
            "selected_after_train_only_generative_fidelity_audit": True,
            "post_training_weight_selection_documented": True,
            "validation_eeg_accessed_during_confirmatory_selection": False,
            "final_test_accessed": False,
            "bulk_synthetic_generation_started": False,
            "b1_may_initialize_from_this_frozen_B0": True,
        },
    }

    if SELECTION_FILE.exists():
        with SELECTION_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_selection = json.load(
                f
            )

        existing_identity = (
            existing_selection.get(
                "frozen_artifacts",
                {},
            ).get(
                "selected_model_state_sha256"
            )
        )

        if (
            existing_identity
            != selected_weights_hash
            or int(
                existing_selection.get(
                    "selected_epoch",
                    -1,
                )
            )
            != SELECTED_EPOCH
        ):
            raise RuntimeError(
                "Existing B0 freeze manifest conflicts "
                "with the selected model."
            )

        print(
            "identical B0 freeze identity already exists."
        )

    else:
        atomic_json_save(
            selection_manifest,
            SELECTION_FILE,
        )

    print(
        f"selection manifest: "
        f"{SELECTION_FILE}"
    )

    # ========================================================
    # Step 8 — Human-readable freeze summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — SAVE HUMAN-READABLE FREEZE SUMMARY"
    )

    print(
        "=" * 80
    )

    summary_text = f"""SCENARIO 2 — FROZEN B0 SELECTION

STATUS
B0_SELECTION_FROZEN

SELECTED GENERATOR
Model: B0-v state-only DDPM
Prediction target: v
Selected fine-tuning checkpoint: RAW epoch {SELECTED_EPOCH}
EMA selected for final B0 generation: NO

PROVENANCE
B0 epsilon pretraining source: epoch {EXPECTED_SOURCE_EPSILON_EPOCH}
Then: v-prediction fine-tuning for 40 epochs
Final selected weights: RAW epoch {SELECTED_EPOCH}

CONFIRMATORY SELECTION
Candidate epochs: 20 and 40
Criterion wins: epoch 20 = {normalized_win_counts["20"]}, epoch 40 = {normalized_win_counts["40"]}
Confirmatory diagnostic direction:
{audit_summary["diagnostic_direction"]}

SELECTED EPOCH-20 METRICS
Normalized std ratio EO: {selected_metrics["eo_normalized_std_ratio"]:.6f}
Normalized std ratio EC: {selected_metrics["ec_normalized_std_ratio"]:.6f}
Physical std ratio EO: {selected_metrics["eo_uv_std_ratio"]:.6f}
Physical std ratio EC: {selected_metrics["ec_uv_std_ratio"]:.6f}
Alpha bandpower MAE: {selected_metrics["alpha_bandpower_mae_db"]:.6f} dB
Alpha-effect channel MAE: {selected_metrics["alpha_effect_channel_mae_db"]:.6f} dB
O1 alpha EC-EO — Real: {selected_metrics["o1_real_alpha_effect_db"]:.6f} dB
O1 alpha EC-EO — Synthetic: {selected_metrics["o1_synthetic_alpha_effect_db"]:.6f} dB
O2 alpha EC-EO — Real: {selected_metrics["o2_real_alpha_effect_db"]:.6f} dB
O2 alpha EC-EO — Synthetic: {selected_metrics["o2_synthetic_alpha_effect_db"]:.6f} dB
Spatial correlation r EO: {selected_metrics["correlation_upper_triangle_r_eo"]:.6f}
Spatial correlation r EC: {selected_metrics["correlation_upper_triangle_r_ec"]:.6f}

KNOWN LIMITATION
Alpha debiased-squared wPLI fidelity remains limited and state-asymmetric.
EO alpha dwPLI2 r: {selected_metrics["alpha_dwpli2_upper_triangle_r_eo"]:.6f}
EC alpha dwPLI2 r: {selected_metrics["alpha_dwpli2_upper_triangle_r_ec"]:.6f}
EO alpha dwPLI2 relF: {selected_metrics["alpha_dwpli2_relF_eo"]:.6f}
EC alpha dwPLI2 relF: {selected_metrics["alpha_dwpli2_relF_ec"]:.6f}

This limitation is explicitly retained because B0 is the state-only baseline.
No claim is made that B0 fully reproduces phase-lag connectivity.

DATA GOVERNANCE
Validation EEG used in confirmatory fidelity selection: NO
Final Test accessed: NO
Bulk synthetic generation started: NO

FROZEN IDENTIFIERS
Source checkpoint SHA256:
{source_hash_before}

Selected RAW model-state SHA256:
{selected_weights_hash}

FROZEN GENERATOR
{GENERATOR_CHECKPOINT}

NEXT SCIENTIFIC STAGE
B1 = initialize from this frozen B0 RAW generator,
add subject conditioning, and fine-tune without changing the frozen B0 artifact.
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
    # Step 9 — B0 -> B1 handoff
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — WRITE B0 -> B1 HANDOFF"
    )

    print(
        "=" * 80
    )

    b1_handoff = {
        "status": (
            "B0_TO_B1_HANDOFF_READY"
        ),
        "frozen_b0_generator": str(
            GENERATOR_CHECKPOINT
        ),
        "frozen_b0_selection_manifest": str(
            SELECTION_FILE
        ),
        "selected_model_state_sha256": (
            selected_weights_hash
        ),
        "load_key": (
            "model_state_dict"
        ),
        "effective_contract_key": (
            "effective_contract"
        ),
        "prediction_target": (
            "v"
        ),
        "input_shape": [
            EXPECTED_MODEL_CHANNELS,
            EXPECTED_MODEL_SAMPLES,
        ],
        "sampling_rate_hz": 128,
        "current_conditioning": {
            "eye_state": True,
            "subject": False,
            "EO": 0,
            "EC": 1,
        },
        "b1_required_change": {
            "preserve_B0_backbone_initialization": True,
            "add_subject_conditioning": True,
            "subject_count": 198,
            "continue_v_prediction": True,
            "do_not_modify_frozen_B0_artifact": True,
        },
        "known_B0_limitation_to_track_in_B1": {
            "alpha_dwpli2_connectivity": (
                "limited, especially EO"
            ),
            "B1_should_be_evaluated_for_subject_specific_and_connectivity_changes": True,
        },
        "bulk_generation_before_B1_B2_freeze": False,
        "validation_test_policy": {
            "Validation": (
                "may be used only according to "
                "predeclared B1 selection protocol"
            ),
            "Final_Test": (
                "SEALED"
            ),
        },
    }

    atomic_json_save(
        b1_handoff,
        B1_HANDOFF_FILE,
    )

    print(
        f"B1 handoff: "
        f"{B1_HANDOFF_FILE}"
    )

    # ========================================================
    # Step 10 — Verify all source evidence remained immutable
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
            "Selected source checkpoint changed "
            "during freeze operation."
        )

    print(
        "selected source checkpoint unchanged: PASS"
    )

    print(
        "confirmatory evidence read-only: PASS"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B0 SELECTION FORMALLY FROZEN"
    )

    print(
        "=" * 80
    )

    print(
        f"selected model: "
        f"B0-v RAW epoch {SELECTED_EPOCH}"
    )

    print(
        f"selected RAW state SHA256: "
        f"{selected_weights_hash}"
    )

    print(
        f"normalized std ratio EO/EC: "
        f"{selected_metrics['eo_normalized_std_ratio']:.3f}"
        f" / "
        f"{selected_metrics['ec_normalized_std_ratio']:.3f}"
    )

    print(
        f"alpha-effect MAE: "
        f"{selected_metrics['alpha_effect_channel_mae_db']:.3f} dB"
    )

    print(
        f"alpha bandpower MAE: "
        f"{selected_metrics['alpha_bandpower_mae_db']:.3f} dB"
    )

    print(
        f"spatial correlation r EO/EC: "
        f"{selected_metrics['correlation_upper_triangle_r_eo']:.3f}"
        f" / "
        f"{selected_metrics['correlation_upper_triangle_r_ec']:.3f}"
    )

    print(
        f"alpha dwPLI2 r EO/EC: "
        f"{selected_metrics['alpha_dwpli2_upper_triangle_r_eo']:.3f}"
        f" / "
        f"{selected_metrics['alpha_dwpli2_upper_triangle_r_ec']:.3f}"
    )

    print(
        "\nKnown limitation:"
    )

    print(
        "  alpha phase-lag connectivity fidelity "
        "remains limited, especially EO."
    )

    print(
        "\nValidation EEG used in confirmatory selection: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Bulk synthetic generation started: NO"
    )

    print(
        "B0 frozen: YES"
    )

    print(
        f"\nFrozen generator:"
        f"\n{GENERATOR_CHECKPOINT}"
    )

    print(
        f"\nSelection manifest:"
        f"\n{SELECTION_FILE}"
    )

    print(
        f"\nB0 -> B1 handoff:"
        f"\n{B1_HANDOFF_FILE}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "DESIGN AND FREEZE B1 "
        "SUBJECT-CONDITIONED FINE-TUNING PROTOCOL"
    )


if __name__ == "__main__":
    main()
