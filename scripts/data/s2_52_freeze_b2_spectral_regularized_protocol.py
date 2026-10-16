from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


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

B1_TO_B2_HANDOFF = (
    FROZEN_B1_ROOT
    / "handoff"
    / "b1_to_b2_handoff_v1.json"
)


# ============================================================
# B2 protocol outputs
# ============================================================

B2_PROTOCOL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b2"
    / "spectral_regularized_v1"
)

METRIC_DIR = B2_PROTOCOL_ROOT / "metrics"
HANDOFF_DIR = B2_PROTOCOL_ROOT / "handoff"

B2_PROTOCOL_FILE = (
    METRIC_DIR
    / "b2_spectral_regularized_protocol_v1.json"
)

B2_SUMMARY_FILE = (
    METRIC_DIR
    / "b2_spectral_regularized_protocol_summary_v1.txt"
)

B2_PILOT_HANDOFF = (
    HANDOFF_DIR
    / "b2_lambda_calibration_pilot_handoff_v1.json"
)


# ============================================================
# Frozen B2 scientific design
# ============================================================

SCHEMA_VERSION = 1

EXPECTED_PARENT_EPOCH = 10
EXPECTED_PARENT_WEIGHT_KIND = "RAW"
EXPECTED_SUBJECTS = 198
EXPECTED_CHANNELS = 19
EXPECTED_SAMPLES = 512
SFREQ = 128

# Spectral objective
SPECTRAL_FMIN_HZ = 1.0
SPECTRAL_FMAX_HZ = 40.0
SPECTRAL_LOG_EPS = 1e-12
SPECTRAL_HUBER_BETA_DB = 1.0

# Predeclared Train-only lambda calibration.
LAMBDA_CANDIDATES = [
    0.0,
    0.01,
    0.03,
    0.10,
]

PILOT_STEPS_PER_LAMBDA = 500
PILOT_BATCH_SIZE = 64
PILOT_SAMPLING_SUBJECTS = 16

# B2 full run after lambda calibration.
FULL_EPOCHS = 20
FULL_BATCH_SIZE = 64
FULL_LEARNING_RATE = 5e-5
FULL_WEIGHT_DECAY = 1e-4
FULL_GRAD_CLIP_NORM = 1.0

FULL_RAW_CANDIDATE_EPOCHS = [
    5,
    10,
    20,
]

MIXED_PRECISION = True
EMA_ENABLED = False

SEED = 20260908


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


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — FREEZE B2 SPECTRAL-REGULARIZED PROTOCOL"
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
        "  B1 final generator: FROZEN"
    )

    print(
        "  B2 spectral-regularized protocol freeze: NOW"
    )

    print(
        "  B2 lambda calibration pilot: NOT STARTED"
    )

    print(
        "  B2 full fine-tuning: NOT STARTED"
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
    # Step 1 — Audit frozen B1 parent
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B1 PARENT"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_TO_B2_HANDOFF,
    ):
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

    with B1_TO_B2_HANDOFF.open(
        "r",
        encoding="utf-8",
    ) as f:
        b1_handoff = json.load(
            f
        )

    frozen_b1_hash = sha256_file(
        FROZEN_B1_GENERATOR
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    if b1_selection[
        "status"
    ] != "B1_SELECTION_FROZEN":
        raise RuntimeError(
            "B1 selection is not frozen."
        )

    if b1_selection[
        "B1_frozen"
    ] is not True:
        raise RuntimeError(
            "B1 frozen flag is not True."
        )

    if int(
        b1_selection[
            "selected_epoch"
        ]
    ) != EXPECTED_PARENT_EPOCH:
        raise RuntimeError(
            "B2 must initialize from frozen B1 RAW epoch 10."
        )

    if b1_selection[
        "selected_weight_kind"
    ] != EXPECTED_PARENT_WEIGHT_KIND:
        raise RuntimeError(
            "B2 must initialize from frozen B1 RAW weights."
        )

    if b1_selection[
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "B2 must continue v-prediction."
        )

    if b1_selection[
        "governance"
    ][
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Frozen B1 selection reports Final Test access."
        )

    if b1_handoff[
        "status"
    ] != "B1_TO_B2_HANDOFF_READY":
        raise RuntimeError(
            "B1 -> B2 handoff is not ready."
        )

    if frozen_b1[
        "status"
    ] != "B1_SUBJECT_STATE_V_FROZEN_GENERATOR":
        raise RuntimeError(
            "Unexpected frozen B1 generator status."
        )

    if int(
        frozen_b1[
            "selected_epoch"
        ]
    ) != EXPECTED_PARENT_EPOCH:
        raise RuntimeError(
            "Frozen B1 generator epoch mismatch."
        )

    if frozen_b1[
        "selected_weight_kind"
    ] != EXPECTED_PARENT_WEIGHT_KIND:
        raise RuntimeError(
            "Frozen B1 generator is not RAW."
        )

    if frozen_b1[
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "Frozen B1 generator is not v-prediction."
        )

    print(
        "frozen parent: B1 subject+state v-prediction RAW epoch 10"
    )

    print(
        f"frozen B1 generator SHA256: "
        f"{frozen_b1_hash}"
    )

    print(
        "B1 parent audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze B2 scientific definition
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — DEFINE B2 SPECTRAL-REGULARIZED OBJECTIVE"
    )

    print(
        "=" * 80
    )

    parent_contract = frozen_b1[
        "effective_contract"
    ]

    if int(
        parent_contract[
            "subject_count"
        ]
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Unexpected B1 subject count."
        )

    if int(
        parent_contract[
            "input_channels"
        ]
    ) != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Unexpected B1 channel count."
        )

    if int(
        parent_contract[
            "input_samples"
        ]
    ) != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Unexpected B1 input length."
        )

    b2_protocol = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "status": (
            "B2_SPECTRAL_REGULARIZED_PROTOCOL_FROZEN"
        ),
        "model_name": (
            "B2_SUBJECT_STATE_SPECTRAL_DDPM_V"
        ),
        "definition": (
            "p(EEG | subject, eye_state)"
        ),
        "role": (
            "Fine-tune frozen B1 subject+state generator "
            "with an additional differentiable spectral-fidelity "
            "regularization term while preserving v-prediction."
        ),
        "parent_B1": {
            "generator": str(
                FROZEN_B1_GENERATOR
            ),
            "generator_sha256": (
                frozen_b1_hash
            ),
            "selected_epoch": (
                EXPECTED_PARENT_EPOCH
            ),
            "selected_weight_kind": (
                EXPECTED_PARENT_WEIGHT_KIND
            ),
            "prediction_target": (
                "v"
            ),
            "read_only": True,
        },
        "architecture": {
            "topology_change": False,
            "new_trainable_parameters": 0,
            "reuse_entire_B1_model_state": True,
            "retain_subject_conditioning": True,
            "retain_eye_state_conditioning": True,
            "subject_count": (
                EXPECTED_SUBJECTS
            ),
            "input_channels": (
                EXPECTED_CHANNELS
            ),
            "input_samples": (
                EXPECTED_SAMPLES
            ),
            "sampling_rate_hz": (
                SFREQ
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
            "ancestral_sampling_steps": (
                1000
            ),
        },
        "loss": {
            "total": (
                "L_total = L_v + lambda_spectral * L_spectral"
            ),
            "L_v": (
                "mean squared error between predicted v and true v"
            ),
            "x0_reconstruction_from_v": (
                "x0_hat = sqrt(alpha_bar_t) * x_t "
                "- sqrt(1-alpha_bar_t) * v_pred"
            ),
            "spectral_regularization": {
                "domain": (
                    "physical microvolts after inverse global Train-only normalization"
                ),
                "temporal_demean_before_fft": True,
                "window": (
                    "Hann"
                ),
                "fft": (
                    "rFFT length 512"
                ),
                "one_sided_psd_normalization": (
                    "|FFT|^2 / (fs * sum(window^2)); "
                    "double positive-frequency non-Nyquist bins"
                ),
                "frequency_range_hz": [
                    SPECTRAL_FMIN_HZ,
                    SPECTRAL_FMAX_HZ,
                ],
                "representation": (
                    "10*log10(PSD + epsilon)"
                ),
                "epsilon": (
                    SPECTRAL_LOG_EPS
                ),
                "distance": (
                    "SmoothL1 / Huber in dB"
                ),
                "huber_beta_db": (
                    SPECTRAL_HUBER_BETA_DB
                ),
                "aggregation": (
                    "mean over batch, channels, and 1-40 Hz frequency bins"
                ),
                "x0_clipping": False,
                "alpha_specific_weighting": False,
                "reason": (
                    "Regularize the full physiological spectrum rather "
                    "than directly optimizing only the EO/EC alpha effect."
                ),
            },
        },
        "lambda_calibration_pilot": {
            "required_before_full_B2_run": True,
            "candidate_lambdas": (
                LAMBDA_CANDIDATES
            ),
            "successful_optimizer_steps_per_lambda": (
                PILOT_STEPS_PER_LAMBDA
            ),
            "batch_size": (
                PILOT_BATCH_SIZE
            ),
            "optimizer": (
                "fresh AdamW for every lambda candidate"
            ),
            "learning_rate": (
                FULL_LEARNING_RATE
            ),
            "weight_decay": (
                FULL_WEIGHT_DECAY
            ),
            "gradient_clip_norm": (
                FULL_GRAD_CLIP_NORM
            ),
            "mixed_precision": (
                MIXED_PRECISION
            ),
            "initialization_for_every_lambda": (
                "exact frozen B1 RAW epoch-10 generator"
            ),
            "same_data_order_across_lambdas": True,
            "same_timestep_noise_stream_across_lambdas": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "small_sampling_diagnostic": {
                "subjects": (
                    PILOT_SAMPLING_SUBJECTS
                ),
                "states": [
                    "EO",
                    "EC",
                ],
                "same_sampling_seeds_across_lambdas": True,
                "purpose": (
                    "guard against scale/state/subject-fidelity collapse"
                ),
            },
            "predeclared_lambda_selection_rule": {
                "reference_candidate": (
                    "lambda=0 matched continuation"
                ),
                "eligible_nonzero_lambda_constraints": {
                    "fixed_probe_correct_subject_v_mse": (
                        "<= 1.02 * lambda0"
                    ),
                    "fixed_probe_subject_gap": (
                        ">= 0.90 * lambda0 when lambda0 gap is positive"
                    ),
                    "generated_normalized_std_EO": [
                        0.80,
                        1.20,
                    ],
                    "generated_normalized_std_EC": [
                        0.80,
                        1.20,
                    ],
                    "spectral_fixed_probe_loss_improvement": (
                        ">= 10% lower than lambda0"
                    ),
                },
                "winner": (
                    "among eligible nonzero lambdas, choose lowest "
                    "fixed-probe spectral loss; exact tie -> lower lambda"
                ),
                "failure_action": (
                    "if no nonzero lambda is eligible, stop and amend "
                    "the B2 protocol before any full B2 run"
                ),
            },
            "pilot_checkpoint_reused_for_full_run": False,
        },
        "full_fine_tuning_after_lambda_selection": {
            "initialization": (
                "exact frozen B1 RAW epoch-10 generator, not pilot checkpoint"
            ),
            "epochs": (
                FULL_EPOCHS
            ),
            "early_stopping": False,
            "batch_size": (
                FULL_BATCH_SIZE
            ),
            "optimizer": (
                "fresh AdamW"
            ),
            "learning_rate": (
                FULL_LEARNING_RATE
            ),
            "learning_rate_schedule": (
                "constant"
            ),
            "weight_decay": (
                FULL_WEIGHT_DECAY
            ),
            "gradient_clip_norm": (
                FULL_GRAD_CLIP_NORM
            ),
            "mixed_precision": (
                MIXED_PRECISION
            ),
            "EMA_enabled": (
                EMA_ENABLED
            ),
            "scientific_weights": (
                "RAW"
            ),
            "predeclared_RAW_candidate_epochs": (
                FULL_RAW_CANDIDATE_EPOCHS
            ),
            "automatic_resume": True,
            "seed": (
                SEED
            ),
            "Validation": (
                "may be used only as deterministic training monitor; "
                "final B2 generator selection uses predeclared "
                "Train-only generative-fidelity audit"
            ),
            "Final_Test": (
                "SEALED"
            ),
        },
        "B2_final_evaluation": {
            "retain_B1_generic_metrics": [
                "amplitude scale",
                "delta/theta/alpha/beta bandpower",
                "EO/EC alpha effect",
                "channel covariance/correlation",
                "alpha debiased-squared wPLI",
            ],
            "retain_B1_subject_metrics": [
                "subject fingerprint standardized distance",
                "subject fingerprint correlation",
                "subject retrieval Top-1",
                "subject retrieval Top-5",
                "correct-vs-wrong subject distance margin",
            ],
            "spectral_primary_metrics": [
                "mean 1-40 Hz log-PSD error",
                "mean bandpower MAE",
                "alpha-effect MAE",
            ],
            "must_not_sacrifice": [
                "subject identity fidelity",
                "EO/EC state fidelity",
                "amplitude scale",
            ],
        },
        "matched_continuation_control": {
            "definition": (
                "same frozen B1 initialization, same optimizer, "
                "same LR, same epochs, lambda_spectral=0"
            ),
            "required_for_causal_attribution": True,
            "reason": (
                "Separates the effect of spectral regularization "
                "from the effect of simply giving B1 additional "
                "fine-tuning updates."
            ),
            "may_be_run_after_primary_B2_training": True,
            "must_exist_before_publication_claim_that_spectral_loss_caused_improvement": True,
        },
        "prohibitions": [
            "do not modify frozen B1 artifact",
            "do not modify frozen B0 artifact",
            "do not use Final Test",
            "do not add connectivity loss in B2 without a new protocol amendment",
            "do not add per-subject normalization",
            "do not reuse B2 lambda-pilot checkpoints for the scientific full run",
            "do not bulk-generate final synthetic datasets before B0/B1/B2 are all frozen",
        ],
        "governance": {
            "protocol_frozen_before_B2_model_updates": True,
            "B0_read_only": True,
            "B1_read_only": True,
            "Validation_allowed_only_under_declared_full_run_monitoring": True,
            "Final_Test_accessed": False,
            "bulk_generation_started": False,
            "B2_training_started": False,
        },
    }

    # ========================================================
    # Step 3 — Contract consistency
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — B2 CONTRACT CONSISTENCY CHECK"
    )

    print(
        "=" * 80
    )

    assert (
        b2_protocol[
            "architecture"
        ][
            "topology_change"
        ]
        is False
    )

    assert int(
        b2_protocol[
            "architecture"
        ][
            "new_trainable_parameters"
        ]
    ) == 0

    assert (
        b2_protocol[
            "diffusion"
        ][
            "prediction_target"
        ]
        == "v"
    )

    assert (
        b2_protocol[
            "architecture"
        ][
            "retain_subject_conditioning"
        ]
        is True
    )

    assert (
        b2_protocol[
            "architecture"
        ][
            "retain_eye_state_conditioning"
        ]
        is True
    )

    assert (
        b2_protocol[
            "full_fine_tuning_after_lambda_selection"
        ][
            "EMA_enabled"
        ]
        is False
    )

    assert (
        b2_protocol[
            "governance"
        ][
            "Final_Test_accessed"
        ]
        is False
    )

    if 0.0 not in LAMBDA_CANDIDATES:
        raise RuntimeError(
            "B2 lambda pilot must contain lambda=0 control."
        )

    nonzero = [
        value
        for value in LAMBDA_CANDIDATES
        if value > 0.0
    ]

    if not nonzero:
        raise RuntimeError(
            "B2 lambda pilot has no nonzero spectral candidate."
        )

    print(
        "B2 definition:"
    )

    print(
        "  same B1 subject+state generator"
    )

    print(
        "  + spectral regularization on reconstructed x0"
    )

    print(
        "\nSpectral loss:"
    )

    print(
        "  physical µV"
    )

    print(
        "  demean -> Hann -> rFFT"
    )

    print(
        "  1-40 Hz one-sided PSD"
    )

    print(
        "  dB SmoothL1/Huber"
    )

    print(
        "\nLambda candidates:"
    )

    print(
        f"  {LAMBDA_CANDIDATES}"
    )

    print(
        "\nFull B2:"
    )

    print(
        f"  epochs = {FULL_EPOCHS}"
    )

    print(
        f"  LR = {FULL_LEARNING_RATE}"
    )

    print(
        f"  candidate epochs = "
        f"{FULL_RAW_CANDIDATE_EPOCHS}"
    )

    print(
        "  EMA = OFF"
    )

    print(
        "contract consistency: PASS"
    )

    # ========================================================
    # Step 4 — Save frozen B2 protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — SAVE FROZEN B2 PROTOCOL"
    )

    print(
        "=" * 80
    )

    if B2_PROTOCOL_FILE.exists():
        with B2_PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != b2_protocol:
            raise RuntimeError(
                "A conflicting B2 protocol already exists."
            )

        print(
            "identical B2 protocol already exists."
        )
    else:
        atomic_json_save(
            b2_protocol,
            B2_PROTOCOL_FILE,
        )

    print(
        f"protocol: "
        f"{B2_PROTOCOL_FILE}"
    )

    # ========================================================
    # Step 5 — Save human-readable summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — SAVE HUMAN-READABLE B2 SUMMARY"
    )

    print(
        "=" * 80
    )

    summary_text = f"""SCENARIO 2 — FROZEN B2 SPECTRAL-REGULARIZED PROTOCOL

STATUS
B2_SPECTRAL_REGULARIZED_PROTOCOL_FROZEN

PARENT
Frozen B1: subject+state v-prediction RAW epoch 10
Parent remains read-only.

B2 DEFINITION
Same p(EEG | subject, eye_state) architecture as B1.
No new trainable parameters.
Continue v-prediction.
Add differentiable spectral regularization.

TOTAL LOSS
L_total = L_v + lambda_spectral * L_spectral

L_v
MSE between predicted v and true v.

L_spectral
1) reconstruct x0_hat from v prediction
2) inverse global Train-only normalization to physical microvolts
3) temporal demean
4) Hann window
5) 512-point rFFT
6) one-sided PSD
7) retain {SPECTRAL_FMIN_HZ:.1f}-{SPECTRAL_FMAX_HZ:.1f} Hz
8) convert to dB with epsilon={SPECTRAL_LOG_EPS}
9) SmoothL1/Huber distance with beta={SPECTRAL_HUBER_BETA_DB:.1f} dB
10) mean over batch, channels, frequencies

No x0 clipping.
No alpha-specific weighting.
No connectivity loss.

LAMBDA CALIBRATION PILOT
Candidates: {LAMBDA_CANDIDATES}
Steps per lambda: {PILOT_STEPS_PER_LAMBDA}
Batch size: {PILOT_BATCH_SIZE}
Every candidate restarts from the exact same frozen B1.
Every candidate uses a fresh optimizer.
Same data order and timestep/noise streams across candidates.
Validation EEG: NO
Final Test: NO

ELIGIBILITY FOR NONZERO LAMBDA
Correct-subject v-MSE <= 1.02 x lambda0
Subject gap >= 0.90 x lambda0 when positive
Generated normalized std EO and EC each in [0.80, 1.20]
Fixed-probe spectral loss at least 10% lower than lambda0

WINNER
Among eligible nonzero lambdas, lowest spectral fixed-probe loss.
Exact tie -> lower lambda.
If none eligible, STOP and amend protocol.

FULL B2
Initialization: frozen B1 RAW epoch 10, never pilot checkpoint
Epochs: {FULL_EPOCHS}
Batch size: {FULL_BATCH_SIZE}
LR: {FULL_LEARNING_RATE}
Weight decay: {FULL_WEIGHT_DECAY}
Gradient clip: {FULL_GRAD_CLIP_NORM}
AMP: ON
EMA: OFF
RAW candidate epochs: {FULL_RAW_CANDIDATE_EPOCHS}

MATCHED CONTINUATION CONTROL
lambda=0 with identical compute is required before making a causal
publication claim that spectral regularization itself caused an improvement.

FINAL TEST
SEALED

BULK GENERATION
NO final bulk generation until B0, B1, and B2 are all frozen.

NEXT
Run the frozen Train-only B2 lambda-calibration pilot.
"""

    atomic_text_save(
        summary_text,
        B2_SUMMARY_FILE,
    )

    print(
        f"summary: "
        f"{B2_SUMMARY_FILE}"
    )

    # ========================================================
    # Step 6 — Write pilot implementation handoff
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — WRITE B2 LAMBDA-PILOT HANDOFF"
    )

    print(
        "=" * 80
    )

    handoff = {
        "status": (
            "B2_LAMBDA_CALIBRATION_PILOT_READY"
        ),
        "frozen_B2_protocol": str(
            B2_PROTOCOL_FILE
        ),
        "frozen_B1_generator": str(
            FROZEN_B1_GENERATOR
        ),
        "frozen_B1_generator_sha256": (
            frozen_b1_hash
        ),
        "load_key": (
            "model_state_dict"
        ),
        "effective_contract_key": (
            "effective_contract"
        ),
        "candidate_lambdas": (
            LAMBDA_CANDIDATES
        ),
        "successful_steps_per_lambda": (
            PILOT_STEPS_PER_LAMBDA
        ),
        "same_initialization_per_lambda": True,
        "same_optimizer_hyperparameters_per_lambda": True,
        "same_data_and_noise_streams_per_lambda": True,
        "prediction_target": (
            "v"
        ),
        "spectral_loss": (
            b2_protocol[
                "loss"
            ][
                "spectral_regularization"
            ]
        ),
        "selection_rule": (
            b2_protocol[
                "lambda_calibration_pilot"
            ][
                "predeclared_lambda_selection_rule"
            ]
        ),
        "updated_pilot_models_scientific_full_run_reusable": False,
        "Validation_EEG": (
            "DO NOT ACCESS"
        ),
        "Final_Test": (
            "SEALED"
        ),
        "next_if_pass": (
            "Freeze selected lambda, then start the 20-epoch "
            "B2 scientific full fine-tuning from frozen B1."
        ),
    }

    atomic_json_save(
        handoff,
        B2_PILOT_HANDOFF,
    )

    print(
        f"handoff: "
        f"{B2_PILOT_HANDOFF}"
    )

    # ========================================================
    # Step 7 — Verify parent immutability
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — VERIFY FROZEN B1 IMMUTABILITY"
    )

    print(
        "=" * 80
    )

    if (
        sha256_file(
            FROZEN_B1_GENERATOR
        )
        != frozen_b1_hash
    ):
        raise RuntimeError(
            "Frozen B1 changed during B2 protocol freeze."
        )

    print(
        "frozen B1 unchanged: PASS"
    )

    print(
        "Final Test accessed: NO"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B2 SPECTRAL-REGULARIZED PROTOCOL FROZEN"
    )

    print(
        "=" * 80
    )

    print(
        "Parent: frozen B1 RAW epoch 10"
    )

    print(
        "Architecture change: NO"
    )

    print(
        "Prediction target: v"
    )

    print(
        "Loss: L_v + lambda * L_spectral"
    )

    print(
        f"lambda pilot candidates: "
        f"{LAMBDA_CANDIDATES}"
    )

    print(
        f"pilot steps per lambda: "
        f"{PILOT_STEPS_PER_LAMBDA}"
    )

    print(
        f"full B2 epochs: "
        f"{FULL_EPOCHS}"
    )

    print(
        f"full B2 LR: "
        f"{FULL_LEARNING_RATE}"
    )

    print(
        "EMA: OFF"
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
        "B2 training started: NO"
    )

    print(
        f"\nProtocol:"
        f"\n{B2_PROTOCOL_FILE}"
    )

    print(
        f"\nPilot handoff:"
        f"\n{B2_PILOT_HANDOFF}"
    )

    print(
        "\nNEXT:"
    )

    print(
        "RUN B2 TRAIN-ONLY LAMBDA CALIBRATION PILOT "
        "FOR [0, 0.01, 0.03, 0.10]"
    )


if __name__ == "__main__":
    main()
