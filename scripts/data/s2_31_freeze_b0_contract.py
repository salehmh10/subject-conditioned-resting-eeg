from __future__ import annotations

import json
from pathlib import Path

import torch


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "b0_state_only_ddpm_contract_v1.json"
)


# ============================================================
# Frozen B0 scientific contract
# ============================================================

CONTRACT = {
    "schema_version": 1,

    "model_name": "B0_STATE_ONLY_DDPM",

    "role": (
        "Primary unconditional-subject baseline generator "
        "conditioned only on eye state."
    ),

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    "data": {
        "subjects": 198,

        "input_channels": 19,

        "input_samples": 512,

        "sampling_rate_hz": 128,

        "window_seconds": 4.0,

        "train_windows": 47835,

        "validation_windows": 9635,

        "input_representation": (
            "locked global Train-normalized EEG"
        ),

        "normalization": (
            "train_normalization.json"
        ),

        "train_partition": (
            "eligible Real Train usable windows"
        ),

        "validation_partition": (
            "Real Validation only"
        ),

        "final_test": "SEALED",
    },

    # --------------------------------------------------------
    # Conditioning
    # --------------------------------------------------------

    "conditioning": {
        "eye_state": True,

        "eye_state_mapping": {
            "EO": 0,
            "EC": 1,
        },

        "subject_condition": False,

        "subject_embedding": False,

        "B0_definition": (
            "p(EEG | eye_state)"
        ),
    },

    # --------------------------------------------------------
    # Diffusion
    # --------------------------------------------------------

    "diffusion": {
        "type": "DDPM",

        "prediction_target": "epsilon",

        "timesteps": 1000,

        "beta_schedule": "cosine",

        "cosine_s": 0.008,

        "training_objective": "MSE(epsilon_pred, epsilon)",

        "sampling_method": "ancestral_DDPM",

        "training_noise": "standard_normal",
    },

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    "network": {
        "type": "1D_UNet",

        "signal_axis": "time",

        "EEG_channels_as_input_channels": True,

        "base_channels": 64,

        "channel_multipliers": [
            1,
            2,
            4,
            4,
        ],

        "residual_blocks_per_level": 2,

        "time_embedding_dim": 256,

        "state_embedding_dim": 64,

        "normalization_layer": (
            "GroupNorm"
        ),

        "activation": "SiLU",

        "dropout": 0.0,

        "attention": {
            "enabled": True,

            "levels": [
                2,
                3,
            ],

            "heads": 4,
        },

        "output_channels": 19,
    },

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    "training": {
        "train_from_scratch": True,

        "pretrained_model": False,

        "optimizer": "AdamW",

        "learning_rate": 2e-4,

        "weight_decay": 1e-4,

        "batch_size": 64,

        "gradient_clip_norm": 1.0,

        "mixed_precision": True,

        "ema": True,

        "ema_decay": 0.9999,

        "seed": 20260907,

        "max_epochs": 200,

        "checkpoint_every_epochs": 5,
    },

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    "validation": {
        "used_for_training_weights": False,

        "used_for_checkpoint_selection": True,

        "final_test_used": False,

        "initial_checkpoint_metrics": [
            "validation_diffusion_loss",
            "train_diffusion_loss",
        ],

        "scientific_sample_metrics": (
            "not activated until sampling/evaluation phase"
        ),
    },

    # --------------------------------------------------------
    # B0 boundaries
    # --------------------------------------------------------

    "prohibitions": [
        "no subject conditioning",
        "no spectral regularization",
        "no connectivity loss",
        "no Test access",
        "no pretrained checkpoint",
        "no per-subject scaler",
        "no Validation normalization fit",
    ],

    "future_models": {
        "B1": (
            "same core model plus subject + eye-state conditioning"
        ),

        "B2": (
            "B1 plus multi-resolution spectral regularization"
        ),
    },
}


# ============================================================
# Validation
# ============================================================

def validate_contract() -> None:

    assert CONTRACT[
        "model_name"
    ] == "B0_STATE_ONLY_DDPM"

    assert CONTRACT[
        "conditioning"
    ][
        "subject_condition"
    ] is False

    assert CONTRACT[
        "conditioning"
    ][
        "eye_state"
    ] is True

    assert CONTRACT[
        "diffusion"
    ][
        "prediction_target"
    ] == "epsilon"

    assert CONTRACT[
        "training"
    ][
        "train_from_scratch"
    ] is True

    assert CONTRACT[
        "data"
    ][
        "final_test"
    ] == "SEALED"

    assert CONTRACT[
        "network"
    ][
        "output_channels"
    ] == 19

    assert CONTRACT[
        "data"
    ][
        "input_samples"
    ] == 512


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — FREEZE B0 STATE-ONLY DDPM CONTRACT")
    print("=" * 80)

    print(
        "\nThis step does NOT train a model."
    )

    print(
        "It freezes the initial B0 "
        "scientific/engineering contract "
        "before architecture implementation."
    )

    # --------------------------------------------------------
    # Step 1 — GPU environment
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 1 — EXECUTION ENVIRONMENT")
    print("=" * 80)

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA runtime: "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = (
            torch.cuda
            .get_device_properties(0)
        )

        print(
            f"VRAM GiB: "
            f"{props.total_memory / 1024**3:.2f}"
        )

    # --------------------------------------------------------
    # Step 2 — Validate contract
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 2 — CONTRACT CHECK")
    print("=" * 80)

    validate_contract()

    print("contract consistency: PASS")

    # --------------------------------------------------------
    # Step 3 — Print key scientific definition
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 3 — B0 DEFINITION")
    print("=" * 80)

    print(
        "Target distribution:"
    )

    print(
        "  p(EEG | eye_state)"
    )

    print(
        "\nInput:"
    )

    print(
        "  shape = (19, 512)"
    )

    print(
        "  unit = locked normalized EEG"
    )

    print(
        "\nConditions:"
    )

    print(
        "  EO = 0"
    )

    print(
        "  EC = 1"
    )

    print(
        "  subject identity = NOT USED"
    )

    print(
        "\nDiffusion:"
    )

    print(
        "  DDPM"
    )

    print(
        "  epsilon prediction"
    )

    print(
        "  1000 timesteps"
    )

    print(
        "  cosine beta schedule"
    )

    print(
        "\nNetwork:"
    )

    print(
        "  1D U-Net"
    )

    print(
        "  EEG channels treated as "
        "input feature channels"
    )

    print(
        "\nTraining:"
    )

    print(
        "  FROM SCRATCH"
    )

    print(
        "  AdamW"
    )

    print(
        "  mixed precision"
    )

    print(
        "  EMA"
    )

    print(
        "\nB0 does NOT include:"
    )

    print(
        "  subject conditioning"
    )

    print(
        "  spectral loss"
    )

    print(
        "  connectivity loss"
    )

    # --------------------------------------------------------
    # Step 4 — Save
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 4 — SAVE CONTRACT")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_FILE.exists():

        with OUTPUT_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            existing = json.load(f)

        if existing != CONTRACT:

            raise RuntimeError(
                "A conflicting B0 contract "
                "already exists."
            )

        print(
            "identical contract already exists."
        )

    else:

        with OUTPUT_FILE.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:

            json.dump(
                CONTRACT,
                f,
                indent=2,
                sort_keys=True,
            )

            f.write("\n")

    print(
        f"saved: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("B0 CONTRACT FROZEN")
    print("=" * 80)

    print(
        "Training started: NO"
    )

    print(
        "Validation used: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "BUILD B0 1D U-NET "
        "AND RUN SHAPE / PARAMETER TESTS"
    )


if __name__ == "__main__":
    main()