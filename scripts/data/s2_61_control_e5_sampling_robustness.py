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

# Exact B1 subject-aware V2 Train-only reference + seed-20261001 cache.
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

# Strong current development baseline: lambda=0 Control E5.
CONTROL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "matched_lambda0_control_v1"
)

CONTROL_E5_CHECKPOINT = (
    CONTROL_ROOT
    / "checkpoints"
    / "candidate_epoch_005_raw.pt"
)

CONTROL_RESULT = (
    CONTROL_ROOT
    / "metrics"
    / "b2_matched_lambda0_control_result_v1.json"
)

# Existing seed-20261001 Control-E5 synthetic cache from s2_58.
S2_58_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "joint_b1_lambda0_lambda001_freegen_audit_v1"
)

S2_58_SUMMARY = (
    S2_58_ROOT
    / "metrics"
    / "s2_58_joint_freegen_summary_v1.json"
)

CONTROL_E5_SEED1_SYNTHETIC = (
    S2_58_ROOT
    / "generated"
    / "lambda0_control_epoch_005_synthetic_v1.npz"
)

# s2_59 and s2_60 must both be complete before s2_61.
S2_59_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "denoiser_mechanism_diagnostics_v1"
    / "metrics"
    / "s2_59_denoiser_diagnostics_summary_v1.json"
)

S2_60_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "corrective_low_t_spectral_pilot_v1"
    / "metrics"
    / "s2_60_corrective_low_t_spectral_summary_v1.json"
)


# ============================================================
# s2_61 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "control_e5_sampling_robustness_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
GENERATED_DIR = RUN_ROOT / "generated"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_FILE = (
    METRIC_DIR
    / "s2_61_control_e5_sampling_robustness_protocol_v1.json"
)

SEED_METRICS_CSV = (
    METRIC_DIR
    / "s2_61_seedwise_metrics_v1.csv"
)

DELTA_CSV = (
    METRIC_DIR
    / "s2_61_seedwise_control_minus_b1_deltas_v1.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_61_control_e5_sampling_robustness_summary_v1.json"
)

PSD_FIGURE = (
    FIGURE_DIR
    / "s2_61_full_psd_mae_by_sampling_seed_v1.png"
)

DELTA_FIGURE = (
    FIGURE_DIR
    / "s2_61_control_minus_b1_psd_delta_by_seed_v1.png"
)


# ============================================================
# Frozen robustness design
# ============================================================

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

N_CHANNELS = 19
N_TIMES = 512
TIMESTEPS = 1000
COSINE_S = 0.008

# Seed 20261001 is the exact historical audit seed.
# 20261002 / 20261003 are the two new, predeclared matched seeds.
SAMPLING_SEEDS = [
    20261001,
    20261002,
    20261003,
]

HISTORICAL_SEED = 20261001

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


# ============================================================
# IO helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

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
    temp = Path(
        str(path)
        + ".tmp"
    )

    with temp.open("wb") as f:
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


def synthetic_cache_path(
    model_name: str,
    sampling_seed: int,
) -> Path:
    safe_name = model_name.lower()

    return (
        GENERATED_DIR
        / f"s2_61_{safe_name}_seed_{sampling_seed}_synthetic_v1.npz"
    )


# ============================================================
# Exact historical reference/cache loading
# ============================================================

def load_exact_reference_and_seed1_caches() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[str],
]:
    real_package = np.load(
        B1_REAL_REFERENCE_NPZ,
        allow_pickle=False,
    )

    b1_package = np.load(
        B1_SYNTHETIC_REFERENCE_NPZ,
        allow_pickle=False,
    )

    control_package = np.load(
        CONTROL_E5_SEED1_SYNTHETIC,
        allow_pickle=False,
    )

    real_z = real_package[
        "real_z"
    ].astype(
        np.float32,
        copy=False,
    )

    subject_indices = (
        real_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    subject_ids = (
        real_package[
            "selected_subject_ids"
        ]
        .astype(str)
        .tolist()
    )

    b1_indices = (
        b1_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    b1_ids = (
        b1_package[
            "selected_subject_ids"
        ]
        .astype(str)
        .tolist()
    )

    control_indices = (
        control_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    control_ids = (
        control_package[
            "selected_subject_ids"
        ]
        .astype(str)
        .tolist()
    )

    if not (
        subject_indices
        == b1_indices
        == control_indices
    ):
        raise RuntimeError(
            "Historical audit subject-index cohort mismatch."
        )

    if not (
        subject_ids
        == b1_ids
        == control_ids
    ):
        raise RuntimeError(
            "Historical audit subject-ID cohort mismatch."
        )

    b1_seed1_z = b1_package[
        "epoch_010_z"
    ].astype(
        np.float32,
        copy=False,
    )

    control_seed1_z = control_package[
        "synth_z"
    ].astype(
        np.float32,
        copy=False,
    )

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(
            f"Unexpected Real reference shape: {real_z.shape}"
        )

    if b1_seed1_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected B1 seed1 shape: {b1_seed1_z.shape}"
        )

    if control_seed1_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected Control-E5 seed1 shape: "
            f"{control_seed1_z.shape}"
        )

    return (
        real_z,
        b1_seed1_z,
        control_seed1_z,
        subject_indices,
        subject_ids,
    )


# ============================================================
# Model builders
# ============================================================

def build_models(
    *,
    frozen_b1: dict[str, Any],
    control_e5: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    models = {
        "B1": build_model_from_state(
            frozen_b1=frozen_b1,
            state_dict=(
                frozen_b1[
                    "model_state_dict"
                ]
            ),
            device=device,
        ),
        "CONTROL_E5": build_model_from_state(
            frozen_b1=frozen_b1,
            state_dict=(
                control_e5[
                    "model_state_dict"
                ]
            ),
            device=device,
        ),
    }

    for model in models.values():
        model.eval()

    return models


# ============================================================
# Safe generation / cache reuse
# ============================================================

def load_or_generate(
    *,
    model_name: str,
    sampling_seed: int,
    model: torch.nn.Module,
    model_source_path: Path,
    expected_model_state_hash: str,
    selected_subjects: list[int],
    selected_subject_ids: list[str],
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    output_path = synthetic_cache_path(
        model_name,
        sampling_seed,
    )

    source_file_hash = sha256_file(
        model_source_path
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
            != selected_subjects
        ):
            raise RuntimeError(
                f"Cached cohort mismatch: "
                f"{model_name}, seed={sampling_seed}"
            )

        if (
            package[
                "selected_subject_ids"
            ]
            .astype(str)
            .tolist()
            != selected_subject_ids
        ):
            raise RuntimeError(
                f"Cached subject-ID mismatch: "
                f"{model_name}, seed={sampling_seed}"
            )

        if str(
            package[
                "source_file_sha256"
            ][0]
        ) != source_file_hash:
            raise RuntimeError(
                f"Cached source-file hash mismatch: "
                f"{model_name}, seed={sampling_seed}"
            )

        if str(
            package[
                "model_state_sha256"
            ][0]
        ) != expected_model_state_hash:
            raise RuntimeError(
                f"Cached model-state hash mismatch: "
                f"{model_name}, seed={sampling_seed}"
            )

        if int(
            package[
                "sampling_seed"
            ][0]
        ) != sampling_seed:
            raise RuntimeError(
                f"Cached seed mismatch: "
                f"{model_name}, seed={sampling_seed}"
            )

        synth_z = package[
            "synth_z"
        ].astype(
            np.float32,
            copy=False,
        )

        if synth_z.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"Cached shape mismatch: "
                f"{model_name}, seed={sampling_seed}"
            )

        print(
            f"{model_name} seed={sampling_seed}: "
            f"validated existing cache -> REUSE"
        )

        return synth_z

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"GENERATE {model_name} — SAMPLING SEED {sampling_seed}"
    )

    print(
        "-" * 80
    )

    start = time.perf_counter()

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
            sampling_seed
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
            f"Unexpected generated shape: "
            f"{model_name}, seed={sampling_seed}, "
            f"shape={synth_z.shape}"
        )

    generation_seconds = (
        time.perf_counter()
        - start
    )

    atomic_npz_save(
        output_path,
        selected_subject_indices=np.asarray(
            selected_subjects,
            dtype=np.int64,
        ),
        selected_subject_ids=np.asarray(
            selected_subject_ids,
            dtype="U32",
        ),
        synth_z=synth_z,
        source_file_sha256=np.asarray(
            [
                source_file_hash
            ],
            dtype="U64",
        ),
        model_state_sha256=np.asarray(
            [
                expected_model_state_hash
            ],
            dtype="U64",
        ),
        sampling_seed=np.asarray(
            [
                sampling_seed
            ],
            dtype=np.int64,
        ),
        generation_seconds=np.asarray(
            [
                generation_seconds
            ],
            dtype=np.float64,
        ),
    )

    print(
        f"{model_name} seed={sampling_seed} generation: "
        f"{generation_seconds/60:.2f} min"
    )

    print(
        f"cache: {output_path}"
    )

    return synth_z


# ============================================================
# Metric helpers
# ============================================================

def evaluate_bundle(
    *,
    real_z: np.ndarray,
    synth_z: np.ndarray,
    scaler: dict[str, Any],
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
        "generic": (
            generic
        ),
        "full_log_psd": (
            full_psd
        ),
    }


def flatten_metrics(
    *,
    model_name: str,
    sampling_seed: int,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    generic = bundle[
        "generic"
    ]

    subject = generic[
        "subject_identity"
    ]

    return {
        "model": (
            model_name
        ),
        "sampling_seed": (
            sampling_seed
        ),
        "full_psd_mae_db": float(
            bundle[
                "full_log_psd"
            ][
                "mean_mae_db"
            ]
        ),
        "eo_scale": float(
            generic[
                "scale"
            ][
                "EO"
            ][
                "normalized_std_ratio"
            ]
        ),
        "ec_scale": float(
            generic[
                "scale"
            ][
                "EC"
            ][
                "normalized_std_ratio"
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
        "subject_top5": float(
            subject[
                "subject_retrieval_top5_accuracy"
            ]
        ),
        "subject_fingerprint_r_median": float(
            subject[
                "subject_fingerprint_pearson_r_median"
            ]
        ),
        "subject_fingerprint_rmse_mean": float(
            subject[
                "subject_fingerprint_standardized_rmse_mean"
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


# ============================================================
# Figures
# ============================================================

def save_figures(
    *,
    seed_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    seeds = np.asarray(
        SAMPLING_SEEDS,
        dtype=np.int64,
    )

    b1_values = np.asarray(
        [
            next(
                row[
                    "full_psd_mae_db"
                ]
                for row in seed_rows
                if (
                    row[
                        "model"
                    ]
                    == "B1"
                    and int(
                        row[
                            "sampling_seed"
                        ]
                    )
                    == seed
                )
            )
            for seed in SAMPLING_SEEDS
        ],
        dtype=np.float64,
    )

    control_values = np.asarray(
        [
            next(
                row[
                    "full_psd_mae_db"
                ]
                for row in seed_rows
                if (
                    row[
                        "model"
                    ]
                    == "CONTROL_E5"
                    and int(
                        row[
                            "sampling_seed"
                        ]
                    )
                    == seed
                )
            )
            for seed in SAMPLING_SEEDS
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.plot(
        seeds,
        b1_values,
        marker="o",
        label="Frozen B1",
    )

    ax.plot(
        seeds,
        control_values,
        marker="o",
        label="Control E5",
    )

    ax.set_xlabel(
        "Sampling seed"
    )

    ax.set_ylabel(
        "Full 1–40 Hz log-PSD MAE (dB)"
    )

    ax.set_title(
        "s2_61 — Sampling-Seed Robustness"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PSD_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    deltas = np.asarray(
        [
            next(
                row[
                    "control_minus_b1_full_psd_mae_db"
                ]
                for row in delta_rows
                if int(
                    row[
                        "sampling_seed"
                    ]
                )
                == seed
            )
            for seed in SAMPLING_SEEDS
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.bar(
        np.arange(
            len(
                seeds
            )
        ),
        deltas,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        np.arange(
            len(
                seeds
            )
        ),
        [
            str(
                seed
            )
            for seed in seeds
        ],
    )

    ax.set_xlabel(
        "Sampling seed"
    )

    ax.set_ylabel(
        "Control E5 − B1 full PSD MAE (dB)"
    )

    ax.set_title(
        "s2_61 — Matched Control-E5 Improvement by Sampling Seed"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        DELTA_FIGURE,
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
        "SCENARIO 2 — s2_61 CONTROL-E5 SAMPLING ROBUSTNESS"
    )

    print(
        "=" * 80
    )

    print(
        "\nScientific question:"
    )

    print(
        "  Is the Control-E5 advantage over frozen B1 stable across"
    )

    print(
        "  multiple matched free-generation sampling seeds?"
    )

    print(
        "\nNo training is executed."
    )

    print(
        "No threshold is changed."
    )

    print(
        "No Final Test EEG is accessed."
    )

    print(
        "No model is formally frozen by this stage."
    )

    print(
        f"\nMatched sampling seeds: "
        f"{SAMPLING_SEEDS}"
    )

    for directory in (
        METRIC_DIR,
        GENERATED_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit all required prior-stage artifacts
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B1 / CONTROL-E5 / s2_58 / s2_59 / s2_60"
    )

    print(
        "=" * 80
    )

    required_files = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_REAL_REFERENCE_NPZ,
        B1_SYNTHETIC_REFERENCE_NPZ,
        CONTROL_E5_CHECKPOINT,
        CONTROL_RESULT,
        CONTROL_E5_SEED1_SYNTHETIC,
        S2_58_SUMMARY,
        S2_59_SUMMARY,
        S2_60_SUMMARY,
    ]

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

    with CONTROL_RESULT.open(
        "r",
        encoding="utf-8",
    ) as f:
        control_result = json.load(
            f
        )

    with S2_58_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_58_summary = json.load(
            f
        )

    with S2_59_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_59_summary = json.load(
            f
        )

    with S2_60_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_60_summary = json.load(
            f
        )

    if b1_selection[
        "status"
    ] != "B1_SELECTION_FROZEN":
        raise RuntimeError(
            "B1 is not frozen."
        )

    if int(
        b1_selection[
            "selected_epoch"
        ]
    ) != 10:
        raise RuntimeError(
            "Unexpected frozen B1 epoch."
        )

    if control_result[
        "status"
    ] != "B2_MATCHED_LAMBDA0_CONTROL_20_EPOCHS_COMPLETE":
        raise RuntimeError(
            "Matched lambda=0 control is not complete."
        )

    if s2_58_summary[
        "status"
    ] != "S2_58_JOINT_B1_LAMBDA0_LAMBDA001_FREEGEN_AUDIT_COMPLETE":
        raise RuntimeError(
            "s2_58 is not complete."
        )

    if s2_59_summary[
        "status"
    ] != "S2_59_CONTROL_E5_GUARDRAILS_AND_DENOISER_DIAGNOSTICS_COMPLETE":
        raise RuntimeError(
            "s2_59 is not complete."
        )

    if s2_60_summary[
        "status"
    ] != "S2_60_CORRECTIVE_LOW_T_SPECTRAL_500STEP_PILOT_COMPLETE":
        raise RuntimeError(
            "s2_60 is not complete."
        )

    if s2_60_summary[
        "recommended_gated_mechanism"
    ] is not None:
        raise RuntimeError(
            "s2_60 unexpectedly recommended a gated mechanism."
        )

    if control_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Control result reports Final Test access."
        )

    frozen_b1_hash_before = sha256_file(
        FROZEN_B1_GENERATOR
    )

    control_e5_hash_before = sha256_file(
        CONTROL_E5_CHECKPOINT
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    control_e5 = torch.load(
        CONTROL_E5_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    frozen_b1_state_hash = state_dict_sha256(
        frozen_b1[
            "model_state_dict"
        ]
    )

    control_e5_state_hash = state_dict_sha256(
        control_e5[
            "model_state_dict"
        ]
    )

    if (
        frozen_b1_state_hash
        != frozen_b1[
            "selected_model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B1 model-state hash mismatch."
        )

    if (
        control_e5_state_hash
        != control_e5[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Control E5 model-state hash mismatch."
        )

    print(
        f"Frozen B1 SHA256: "
        f"{frozen_b1_hash_before}"
    )

    print(
        f"Control E5 SHA256: "
        f"{control_e5_hash_before}"
    )

    print(
        "source audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze s2_61 robustness protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE s2_61 ROBUSTNESS PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_61_CONTROL_E5_SAMPLING_ROBUSTNESS_PROTOCOL_FROZEN"
        ),
        "scientific_role": (
            "Train-only sampling-seed robustness diagnostic for the strong "
            "Control-E5 development baseline. No final model freeze."
        ),
        "comparators": {
            "B1": (
                "exact frozen B1 RAW epoch 10"
            ),
            "CONTROL_E5": (
                "exact matched lambda=0 Control E5"
            ),
        },
        "sampling_seeds": (
            SAMPLING_SEEDS
        ),
        "historical_seed": (
            HISTORICAL_SEED
        ),
        "historical_seed_cache_policy": (
            "reuse exact existing B1 V2 and s2_58 Control-E5 caches"
        ),
        "new_seed_cache_policy": (
            "generate each B1/Control-E5 seed once and cache with "
            "model/file hashes and cohort identity"
        ),
        "audit": {
            "subject_count": (
                SELECTED_SUBJECT_COUNT
            ),
            "real_windows_per_subject_state": (
                REAL_WINDOWS_PER_SUBJECT_STATE
            ),
            "synthetic_windows_per_subject_state_per_seed": (
                SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
            ),
            "sampler": (
                "1000-step ancestral DDPM / v-prediction / FP32"
            ),
            "exact_existing_metric_helpers_reused": True,
            "same_subject_order_between_models_and_seeds": True,
            "paired_EO_EC_noise_seed_convention": True,
        },
        "development_anchor_support_rule": (
            "Control E5 is considered ROBUSTLY SUPPORTED as the development "
            "anchor only if, for EVERY predeclared sampling seed: "
            "(1) Control E5 full PSD MAE < matched B1 full PSD MAE, and "
            "(2) Control E5 passes ALL existing frozen B2 guards when "
            "evaluated against that same-seed matched B1 baseline. "
            "This is not a final scientific freeze."
        ),
        "important_limit": (
            "No new threshold is introduced for final model selection. "
            "This stage only tests sampling robustness of an already-observed "
            "Train-only development advantage."
        ),
        "Train_reference_used": True,
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "training_updates_executed": False,
        "thresholds_changed": False,
        "formal_model_freeze_decision_made": False,
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
                "Conflicting s2_61 protocol already exists."
            )

        print(
            "identical s2_61 protocol already exists."
        )
    else:
        atomic_json_save(
            protocol,
            PROTOCOL_FILE,
        )

        print(
            f"protocol: "
            f"{PROTOCOL_FILE}"
        )

    # ========================================================
    # Step 3 — Load exact reference and historical seed-1 caches
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — LOAD EXACT AUDIT REFERENCE + HISTORICAL SEED CACHE"
    )

    print(
        "=" * 80
    )

    (
        real_z,
        b1_seed1_z,
        control_seed1_z,
        selected_subject_indices,
        selected_subject_ids,
    ) = load_exact_reference_and_seed1_caches()

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
        f"Real reference shape: "
        f"{real_z.shape}"
    )

    print(
        f"Audit subjects: "
        f"{len(selected_subject_indices)}"
    )

    print(
        "Historical seed cache reuse: READY"
    )

    # ========================================================
    # Step 4 — Build fixed models and sampler
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — BUILD FROZEN B1 / CONTROL-E5 + FIXED SAMPLER"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for s2_61 sampling."
        )

    device = torch.device(
        "cuda"
    )

    models = build_models(
        frozen_b1=frozen_b1,
        control_e5=control_e5,
        device=device,
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
    # Step 5 — Assemble all matched seed samples
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — REUSE HISTORICAL SEED + GENERATE TWO NEW MATCHED SEEDS"
    )

    print(
        "=" * 80
    )

    samples: dict[
        tuple[
            str,
            int,
        ],
        np.ndarray,
    ] = {
        (
            "B1",
            HISTORICAL_SEED,
        ): (
            b1_seed1_z
        ),
        (
            "CONTROL_E5",
            HISTORICAL_SEED,
        ): (
            control_seed1_z
        ),
    }

    for sampling_seed in SAMPLING_SEEDS:
        if sampling_seed == HISTORICAL_SEED:
            print(
                f"B1 seed={sampling_seed}: "
                f"historical cache -> REUSE"
            )

            print(
                f"CONTROL_E5 seed={sampling_seed}: "
                f"historical cache -> REUSE"
            )

            continue

        samples[
            (
                "B1",
                sampling_seed,
            )
        ] = load_or_generate(
            model_name="B1",
            sampling_seed=(
                sampling_seed
            ),
            model=models[
                "B1"
            ],
            model_source_path=(
                FROZEN_B1_GENERATOR
            ),
            expected_model_state_hash=(
                frozen_b1_state_hash
            ),
            selected_subjects=(
                selected_subject_indices
            ),
            selected_subject_ids=(
                selected_subject_ids
            ),
            reverse_schedule=(
                reverse_schedule
            ),
            device=device,
        )

        samples[
            (
                "CONTROL_E5",
                sampling_seed,
            )
        ] = load_or_generate(
            model_name="CONTROL_E5",
            sampling_seed=(
                sampling_seed
            ),
            model=models[
                "CONTROL_E5"
            ],
            model_source_path=(
                CONTROL_E5_CHECKPOINT
            ),
            expected_model_state_hash=(
                control_e5_state_hash
            ),
            selected_subjects=(
                selected_subject_indices
            ),
            selected_subject_ids=(
                selected_subject_ids
            ),
            reverse_schedule=(
                reverse_schedule
            ),
            device=device,
        )

    # ========================================================
    # Step 6 — Evaluate exact existing metrics for every seed
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — EVALUATE MATCHED SEEDWISE FIDELITY"
    )

    print(
        "=" * 80
    )

    bundles: dict[
        tuple[
            str,
            int,
        ],
        dict[str, Any],
    ] = {}

    seed_rows = []

    delta_rows = []

    seed_support = {}

    for sampling_seed in SAMPLING_SEEDS:
        b1_bundle = evaluate_bundle(
            real_z=real_z,
            synth_z=samples[
                (
                    "B1",
                    sampling_seed,
                )
            ],
            scaler=scaler,
        )

        control_bundle = evaluate_bundle(
            real_z=real_z,
            synth_z=samples[
                (
                    "CONTROL_E5",
                    sampling_seed,
                )
            ],
            scaler=scaler,
        )

        bundles[
            (
                "B1",
                sampling_seed,
            )
        ] = b1_bundle

        bundles[
            (
                "CONTROL_E5",
                sampling_seed,
            )
        ] = control_bundle

        b1_row = flatten_metrics(
            model_name="B1",
            sampling_seed=(
                sampling_seed
            ),
            bundle=b1_bundle,
        )

        control_row = flatten_metrics(
            model_name="CONTROL_E5",
            sampling_seed=(
                sampling_seed
            ),
            bundle=control_bundle,
        )

        seed_rows.extend(
            [
                b1_row,
                control_row,
            ]
        )

        guard = evaluate_eligibility(
            candidate=(
                control_bundle
            ),
            b1_baseline=(
                b1_bundle
            ),
        )

        full_psd_delta = float(
            control_row[
                "full_psd_mae_db"
            ]
            - b1_row[
                "full_psd_mae_db"
            ]
        )

        full_psd_improvement_fraction = float(
            (
                b1_row[
                    "full_psd_mae_db"
                ]
                - control_row[
                    "full_psd_mae_db"
                ]
            )
            / b1_row[
                "full_psd_mae_db"
            ]
        )

        delta_row = {
            "sampling_seed": (
                sampling_seed
            ),
            "b1_full_psd_mae_db": (
                b1_row[
                    "full_psd_mae_db"
                ]
            ),
            "control_e5_full_psd_mae_db": (
                control_row[
                    "full_psd_mae_db"
                ]
            ),
            "control_minus_b1_full_psd_mae_db": (
                full_psd_delta
            ),
            "control_improvement_vs_b1_fraction": (
                full_psd_improvement_fraction
            ),
            "control_passes_all_existing_B2_guards_vs_matched_B1": bool(
                guard[
                    "eligible"
                ]
            ),
        }

        delta_rows.append(
            delta_row
        )

        full_psd_better = bool(
            full_psd_delta
            < 0.0
        )

        guards_pass = bool(
            guard[
                "eligible"
            ]
        )

        supported = bool(
            full_psd_better
            and guards_pass
        )

        seed_support[
            str(
                sampling_seed
            )
        ] = {
            "full_psd_better_than_matched_B1": (
                full_psd_better
            ),
            "passes_all_existing_B2_guards_vs_matched_B1": (
                guards_pass
            ),
            "development_anchor_supported_this_seed": (
                supported
            ),
            "control_minus_B1_full_psd_mae_db": (
                full_psd_delta
            ),
            "control_improvement_vs_B1_fraction": (
                full_psd_improvement_fraction
            ),
            "guard_detail": (
                guard
            ),
        }

        print(
            "\n"
            f"sampling seed {sampling_seed}:"
        )

        print(
            f"  B1 full PSD MAE: "
            f"{b1_row['full_psd_mae_db']:.6f} dB"
        )

        print(
            f"  Control E5 full PSD MAE: "
            f"{control_row['full_psd_mae_db']:.6f} dB"
        )

        print(
            f"  Control-B1 delta: "
            f"{full_psd_delta:+.6f} dB"
        )

        print(
            f"  improvement vs B1: "
            f"{100.0 * full_psd_improvement_fraction:+.2f}%"
        )

        print(
            f"  passes all existing B2 guards: "
            f"{guards_pass}"
        )

        print(
            f"  anchor supported this seed: "
            f"{supported}"
        )

    save_csv_rows(
        seed_rows,
        SEED_METRICS_CSV,
    )

    save_csv_rows(
        delta_rows,
        DELTA_CSV,
    )

    # ========================================================
    # Step 7 — Aggregate robustness
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — AGGREGATE SAMPLING-SEED ROBUSTNESS"
    )

    print(
        "=" * 80
    )

    b1_psd_values = np.asarray(
        [
            next(
                row[
                    "full_psd_mae_db"
                ]
                for row in seed_rows
                if (
                    row[
                        "model"
                    ]
                    == "B1"
                    and int(
                        row[
                            "sampling_seed"
                        ]
                    )
                    == seed
                )
            )
            for seed in SAMPLING_SEEDS
        ],
        dtype=np.float64,
    )

    control_psd_values = np.asarray(
        [
            next(
                row[
                    "full_psd_mae_db"
                ]
                for row in seed_rows
                if (
                    row[
                        "model"
                    ]
                    == "CONTROL_E5"
                    and int(
                        row[
                            "sampling_seed"
                        ]
                    )
                    == seed
                )
            )
            for seed in SAMPLING_SEEDS
        ],
        dtype=np.float64,
    )

    deltas = (
        control_psd_values
        - b1_psd_values
    )

    improvement_fractions = (
        (
            b1_psd_values
            - control_psd_values
        )
        / b1_psd_values
    )

    development_anchor_robustly_supported = bool(
        all(
            item[
                "development_anchor_supported_this_seed"
            ]
            for item in seed_support.values()
        )
    )

    aggregate = {
        "B1_full_psd_mae_db": {
            "mean": float(
                np.mean(
                    b1_psd_values
                )
            ),
            "std_ddof0": float(
                np.std(
                    b1_psd_values,
                    ddof=0,
                )
            ),
            "min": float(
                np.min(
                    b1_psd_values
                )
            ),
            "max": float(
                np.max(
                    b1_psd_values
                )
            ),
        },
        "Control_E5_full_psd_mae_db": {
            "mean": float(
                np.mean(
                    control_psd_values
                )
            ),
            "std_ddof0": float(
                np.std(
                    control_psd_values,
                    ddof=0,
                )
            ),
            "min": float(
                np.min(
                    control_psd_values
                )
            ),
            "max": float(
                np.max(
                    control_psd_values
                )
            ),
        },
        "Control_minus_B1_full_psd_mae_db": {
            "mean": float(
                np.mean(
                    deltas
                )
            ),
            "std_ddof0": float(
                np.std(
                    deltas,
                    ddof=0,
                )
            ),
            "min": float(
                np.min(
                    deltas
                )
            ),
            "max": float(
                np.max(
                    deltas
                )
            ),
        },
        "Control_improvement_vs_B1_fraction": {
            "mean": float(
                np.mean(
                    improvement_fractions
                )
            ),
            "std_ddof0": float(
                np.std(
                    improvement_fractions,
                    ddof=0,
                )
            ),
            "min": float(
                np.min(
                    improvement_fractions
                )
            ),
            "max": float(
                np.max(
                    improvement_fractions
                )
            ),
        },
        "development_anchor_robustly_supported": (
            development_anchor_robustly_supported
        ),
    }

    print(
        "B1 full-PSD MAE "
        f"mean±std: "
        f"{aggregate['B1_full_psd_mae_db']['mean']:.6f} ± "
        f"{aggregate['B1_full_psd_mae_db']['std_ddof0']:.6f} dB"
    )

    print(
        "Control-E5 full-PSD MAE "
        f"mean±std: "
        f"{aggregate['Control_E5_full_psd_mae_db']['mean']:.6f} ± "
        f"{aggregate['Control_E5_full_psd_mae_db']['std_ddof0']:.6f} dB"
    )

    print(
        "Control-B1 PSD delta "
        f"mean±std: "
        f"{aggregate['Control_minus_B1_full_psd_mae_db']['mean']:+.6f} ± "
        f"{aggregate['Control_minus_B1_full_psd_mae_db']['std_ddof0']:.6f} dB"
    )

    print(
        "Control improvement vs B1 "
        f"mean±std: "
        f"{100.0 * aggregate['Control_improvement_vs_B1_fraction']['mean']:.2f}% ± "
        f"{100.0 * aggregate['Control_improvement_vs_B1_fraction']['std_ddof0']:.2f}%"
    )

    print(
        "\nDEVELOPMENT ANCHOR ROBUSTLY SUPPORTED: "
        f"{development_anchor_robustly_supported}"
    )

    # ========================================================
    # Step 8 — Figures
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — SAVE SAMPLING ROBUSTNESS FIGURES"
    )

    print(
        "=" * 80
    )

    save_figures(
        seed_rows=seed_rows,
        delta_rows=delta_rows,
    )

    print(
        PSD_FIGURE
    )

    print(
        DELTA_FIGURE
    )

    # ========================================================
    # Step 9 — Source immutability + summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — VERIFY IMMUTABILITY + SAVE FINAL SUMMARY"
    )

    print(
        "=" * 80
    )

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != frozen_b1_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during s2_61."
        )

    if sha256_file(
        CONTROL_E5_CHECKPOINT
    ) != control_e5_hash_before:
        raise RuntimeError(
            "Control E5 checkpoint changed during s2_61."
        )

    summary = {
        "status": (
            "S2_61_CONTROL_E5_SAMPLING_ROBUSTNESS_COMPLETE"
        ),
        "scientific_role": (
            "Train-only matched sampling-seed robustness diagnostic. "
            "No final model freeze."
        ),
        "sampling_seeds": (
            SAMPLING_SEEDS
        ),
        "seedwise_support": (
            seed_support
        ),
        "aggregate": (
            aggregate
        ),
        "development_anchor_robustly_supported": (
            development_anchor_robustly_supported
        ),
        "artifacts": {
            "protocol": str(
                PROTOCOL_FILE
            ),
            "seed_metrics_csv": str(
                SEED_METRICS_CSV
            ),
            "delta_csv": str(
                DELTA_CSV
            ),
            "psd_figure": str(
                PSD_FIGURE
            ),
            "delta_figure": str(
                DELTA_FIGURE
            ),
        },
        "governance": {
            "Train_reference_used": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "training_updates_executed": False,
            "thresholds_changed": False,
            "formal_model_freeze_decision_made": False,
            "Frozen_B1_unchanged": True,
            "Control_E5_checkpoint_unchanged": True,
        },
        "next": (
            "If development_anchor_robustly_supported is True, use "
            "Control E5 as the strong Train-only development anchor for "
            "a separately frozen B2 corrective-adapter objective. "
            "If False, do not build the adapter yet; first interpret the "
            "seed instability."
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
        "Control E5 unchanged: PASS"
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
        "s2_61 CONTROL-E5 SAMPLING ROBUSTNESS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"development anchor robustly supported: "
        f"{development_anchor_robustly_supported}"
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
        "Formal model freeze decision made: NO"
    )

    print(
        "\nNEXT:"
    )

    if development_anchor_robustly_supported:
        print(
            "PROCEED TO A SEPARATELY FROZEN CONTROL-E5-PRESERVING "
            "CORRECTIVE ADAPTER PROTOCOL."
        )
    else:
        print(
            "STOP AND INTERPRET SAMPLING-SEED INSTABILITY BEFORE "
            "ANY NEW B2 REDESIGN."
        )


if __name__ == "__main__":
    main()
