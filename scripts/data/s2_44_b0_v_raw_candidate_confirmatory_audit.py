from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

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
from s2_24_full_train_loader_sanity import CachedRealTrainWindowDataset
from s2_32_b0_unet_shape_test import B0StateOnlyUNet1D
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    alpha_dwpli2_debiased,
    bandpower_db_from_psd,
    build_reverse_schedule,
    mean_covariance_and_correlation,
    relative_frobenius,
    sample_v_batches,
    upper_triangle_correlation,
    welch_psd,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

B0V_FULL_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "full_finetune_v1"
)

CHECKPOINT_DIR = B0V_FULL_DIR / "checkpoints"

CANDIDATE_CHECKPOINTS = {
    20: CHECKPOINT_DIR / "epoch_020.pt",
    40: CHECKPOINT_DIR / "epoch_040.pt",
}

AUDIT_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "raw_candidate_confirmatory_v1"
)

GENERATED_DIR = AUDIT_DIR / "generated"
LOG_DIR = AUDIT_DIR / "logs"
METRIC_DIR = AUDIT_DIR / "metrics"
FIGURE_DIR = AUDIT_DIR / "figures"

PROTOCOL_FILE = METRIC_DIR / "b0_v_raw_candidate_confirmatory_protocol_v1.json"
SUMMARY_FILE = METRIC_DIR / "b0_v_raw_candidate_confirmatory_summary_v1.json"
CANDIDATE_METRICS_FILE = METRIC_DIR / "b0_v_raw_candidate_confirmatory_metrics_v1.csv"
REAL_MANIFEST_FILE = LOG_DIR / "b0_v_confirmatory_real_train_reference_manifest_v1.csv"
MATRICES_FILE = METRIC_DIR / "b0_v_raw_candidate_confirmatory_matrices_v1.npz"

REAL_REFERENCE_FILE = GENERATED_DIR / "b0_v_confirmatory_real_train_reference_v1.npz"
E20_SYNTHETIC_FILE = GENERATED_DIR / "b0_v_epoch020_raw_confirmatory_synthetic_v1.npz"
E40_SYNTHETIC_FILE = GENERATED_DIR / "b0_v_epoch040_raw_confirmatory_synthetic_v1.npz"

SCALE_FIGURE = FIGURE_DIR / "b0_v_raw_candidate_scale_ratio_v1.png"
ALPHA_FIGURE = FIGURE_DIR / "b0_v_raw_candidate_alpha_effect_v1.png"
BANDPOWER_FIGURE = FIGURE_DIR / "b0_v_raw_candidate_bandpower_mae_v1.png"
CORRELATION_FIGURE = FIGURE_DIR / "b0_v_raw_candidate_correlation_v1.png"
WPLI_FIGURE = FIGURE_DIR / "b0_v_raw_candidate_alpha_dwpli2_v1.png"
OVERVIEW_FIGURE = FIGURE_DIR / "b0_v_raw_candidate_overview_v1.png"


# ============================================================
# Frozen confirmatory design
# ============================================================

SEED = 20260916
REAL_REFERENCE_SEED = 20260917
EO_SAMPLING_SEED = 20260918
EC_SAMPLING_SEED = 20260919

CANDIDATE_EPOCHS = (20, 40)

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198
EXPECTED_SOURCE_EPOCH = 125

N_REAL_PER_STATE = 256
N_SYNTH_PER_STATE = 256
SAMPLING_BATCH_SIZE = 64

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128
TIMESTEPS = 1000
COSINE_S = 0.008

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8",
    "C3", "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2",
]
CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


# ============================================================
# Utilities
# ============================================================


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def atomic_json_save(data: dict, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)



def inverse_normalize(eeg_z: np.ndarray, scaler: dict) -> np.ndarray:
    means = np.asarray(scaler["means_uv"], dtype=np.float64)
    stds = np.asarray(scaler["stds_uv"], dtype=np.float64)
    return (
        eeg_z.astype(np.float64) * stds[None, :, None]
        + means[None, :, None]
    ).astype(np.float32)



def select_subject_balanced_records(records, label: int, n: int, seed: int):
    grouped = defaultdict(list)
    for record in records:
        if int(record.label) == label:
            grouped[record.subject_id].append(record)

    subjects = sorted(grouped)
    if len(subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} subjects for label {label}, "
            f"found {len(subjects)}"
        )

    rng = np.random.default_rng(seed)
    for subject in subjects:
        order = rng.permutation(len(grouped[subject]))
        grouped[subject] = [grouped[subject][int(i)] for i in order]

    selected = []
    round_index = 0
    while len(selected) < n:
        added = 0
        for subject in subjects:
            if round_index < len(grouped[subject]):
                selected.append(grouped[subject][round_index])
                added += 1
                if len(selected) == n:
                    break
        if added == 0:
            break
        round_index += 1

    if len(selected) != n:
        raise RuntimeError(
            f"Could only select {len(selected)} records for label {label}; wanted {n}"
        )

    return selected



def load_reference_array(selected_records, scaler: dict) -> np.ndarray:
    dataset = CachedRealTrainWindowDataset(
        records=selected_records,
        scaler=scaler,
        max_open_files=8,
    )
    try:
        xs = []
        for index in range(len(dataset)):
            item = dataset[index]
            x = item["x"].detach().cpu().numpy().astype(np.float32, copy=False)
            xs.append(x)
        array = np.stack(xs, axis=0)
    finally:
        dataset.close()

    expected = (len(selected_records), N_CHANNELS, N_TIMES)
    if array.shape != expected:
        raise RuntimeError(f"Unexpected Real reference shape: {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("Real reference contains NaN/Inf")
    return array



def save_real_manifest(eo_records, ec_records) -> None:
    rows = []
    for label, eye_state, state_records in (
        (0, "EO", eo_records),
        (1, "EC", ec_records),
    ):
        for local_index, record in enumerate(state_records):
            rows.append(
                {
                    "reference_index": local_index,
                    "eye_state": eye_state,
                    "label": label,
                    "subject_id": record.subject_id,
                    "subject_index": int(record.subject_index),
                    "block_uid": record.block_uid,
                    "block_index": int(record.block_index),
                    "candidate_window_index": int(record.candidate_window_index),
                    "start_sample": int(record.start_sample),
                }
            )

    with REAL_MANIFEST_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Metric helpers
# ============================================================


def compute_bandpower(eeg_uv: np.ndarray) -> dict[str, np.ndarray]:
    freqs, psd = welch_psd(eeg_uv)
    return bandpower_db_from_psd(freqs, psd)



def candidate_metrics(
    *,
    epoch: int,
    real_z: dict[str, np.ndarray],
    real_uv: dict[str, np.ndarray],
    synth_z: dict[str, np.ndarray],
    synth_uv: dict[str, np.ndarray],
    real_band: dict[str, dict[str, np.ndarray]],
    real_corr: dict[str, np.ndarray],
    real_wpli: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    synth_band = {
        state: compute_bandpower(synth_uv[state])
        for state in ("EO", "EC")
    }

    band_mae = {}
    for band in BANDS:
        errors = []
        for state in ("EO", "EC"):
            real_med = np.median(real_band[state][band], axis=0)
            synth_med = np.median(synth_band[state][band], axis=0)
            errors.extend(np.abs(synth_med - real_med).tolist())
        band_mae[band] = float(np.mean(errors))

    real_alpha_delta = (
        np.median(real_band["EC"]["alpha"], axis=0)
        - np.median(real_band["EO"]["alpha"], axis=0)
    )
    synth_alpha_delta = (
        np.median(synth_band["EC"]["alpha"], axis=0)
        - np.median(synth_band["EO"]["alpha"], axis=0)
    )

    alpha_effect_mae = float(np.mean(np.abs(synth_alpha_delta - real_alpha_delta)))

    synth_corr = {}
    corr_metrics = {}
    for state in ("EO", "EC"):
        _, corr = mean_covariance_and_correlation(synth_z[state])
        synth_corr[state] = corr
        corr_metrics[state] = {
            "r": upper_triangle_correlation(real_corr[state], corr),
            "relF": relative_frobenius(real_corr[state], corr),
        }

    synth_wpli = {
        state: alpha_dwpli2_debiased(synth_uv[state])
        for state in ("EO", "EC")
    }
    wpli_metrics = {
        state: {
            "r": upper_triangle_correlation(real_wpli[state], synth_wpli[state]),
            "relF": relative_frobenius(real_wpli[state], synth_wpli[state]),
        }
        for state in ("EO", "EC")
    }

    row = {
        "epoch": int(epoch),
        "weight_kind": "raw",
        "eo_normalized_std": float(synth_z["EO"].std()),
        "ec_normalized_std": float(synth_z["EC"].std()),
        "eo_real_normalized_std": float(real_z["EO"].std()),
        "ec_real_normalized_std": float(real_z["EC"].std()),
        "eo_normalized_std_ratio": float(synth_z["EO"].std() / real_z["EO"].std()),
        "ec_normalized_std_ratio": float(synth_z["EC"].std() / real_z["EC"].std()),
        "eo_uv_std_ratio": float(synth_uv["EO"].std() / real_uv["EO"].std()),
        "ec_uv_std_ratio": float(synth_uv["EC"].std() / real_uv["EC"].std()),
        "delta_bandpower_mae_db": band_mae["delta"],
        "theta_bandpower_mae_db": band_mae["theta"],
        "alpha_bandpower_mae_db": band_mae["alpha"],
        "beta_bandpower_mae_db": band_mae["beta"],
        "alpha_effect_channel_mae_db": alpha_effect_mae,
        "whole_head_real_alpha_effect_db": float(np.mean(real_alpha_delta)),
        "whole_head_synthetic_alpha_effect_db": float(np.mean(synth_alpha_delta)),
        "o1_real_alpha_effect_db": float(real_alpha_delta[CHANNEL_INDEX["O1"]]),
        "o1_synthetic_alpha_effect_db": float(synth_alpha_delta[CHANNEL_INDEX["O1"]]),
        "o2_real_alpha_effect_db": float(real_alpha_delta[CHANNEL_INDEX["O2"]]),
        "o2_synthetic_alpha_effect_db": float(synth_alpha_delta[CHANNEL_INDEX["O2"]]),
        "correlation_upper_triangle_r_eo": float(corr_metrics["EO"]["r"]),
        "correlation_upper_triangle_r_ec": float(corr_metrics["EC"]["r"]),
        "correlation_relF_eo": float(corr_metrics["EO"]["relF"]),
        "correlation_relF_ec": float(corr_metrics["EC"]["relF"]),
        "alpha_dwpli2_upper_triangle_r_eo": float(wpli_metrics["EO"]["r"]),
        "alpha_dwpli2_upper_triangle_r_ec": float(wpli_metrics["EC"]["r"]),
        "alpha_dwpli2_relF_eo": float(wpli_metrics["EO"]["relF"]),
        "alpha_dwpli2_relF_ec": float(wpli_metrics["EC"]["relF"]),
    }

    alpha_arrays = {
        "real_delta": real_alpha_delta,
        "synth_delta": synth_alpha_delta,
    }

    return row, synth_corr, synth_wpli, {"band": synth_band, "alpha": alpha_arrays}


# ============================================================
# Figures
# ============================================================


def save_scale_figure(rows: list[dict]) -> None:
    labels = [f"E{row['epoch']} RAW" for row in rows]
    x = np.arange(len(rows))
    width = 0.36

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, [r["eo_normalized_std_ratio"] for r in rows], width, label="EO")
    ax.bar(x + width / 2, [r["ec_normalized_std_ratio"] for r in rows], width, label="EC")
    ax.axhline(1.0, linewidth=1.5, label="Perfect scale")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Synthetic / Real normalized std")
    ax.set_title("B0-v Confirmatory Audit — RAW Candidate Scale Fidelity")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCALE_FIGURE, dpi=160)
    plt.close(fig)



def save_alpha_figure(rows: list[dict]) -> None:
    labels = [f"E{row['epoch']} RAW" for row in rows]
    x = np.arange(len(rows))
    width = 0.22

    real_o1 = rows[0]["o1_real_alpha_effect_db"]
    real_o2 = rows[0]["o2_real_alpha_effect_db"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, channel, real_value, field in (
        (axes[0], "O1", real_o1, "o1_synthetic_alpha_effect_db"),
        (axes[1], "O2", real_o2, "o2_synthetic_alpha_effect_db"),
    ):
        ax.bar(x, [r[field] for r in rows], width=0.55, label="Synthetic RAW")
        ax.axhline(real_value, linewidth=1.5, label=f"Real {channel}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(channel)
        ax.set_ylabel("EC − EO alpha power (dB)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("B0-v Confirmatory Audit — Occipital Eye-State Alpha Effect")
    fig.tight_layout()
    fig.savefig(ALPHA_FIGURE, dpi=160)
    plt.close(fig)



def save_bandpower_figure(rows: list[dict]) -> None:
    labels = [f"E{row['epoch']} RAW" for row in rows]
    bands = ("delta", "theta", "alpha", "beta")
    matrix = np.asarray(
        [[row[f"{band}_bandpower_mae_db"] for band in bands] for row in rows],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    image = ax.imshow(matrix, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_title("B0-v Confirmatory Audit — Median Bandpower MAE")
    fig.colorbar(image, ax=ax, label="MAE (dB)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(BANDPOWER_FIGURE, dpi=160)
    plt.close(fig)



def save_matrix_comparison(
    *,
    real: dict[str, np.ndarray],
    candidates: dict[int, dict[str, np.ndarray]],
    output: Path,
    title: str,
) -> None:
    # 2 states x (Real + candidate epochs)
    columns = ["Real"] + [f"E{epoch} RAW" for epoch in CANDIDATE_EPOCHS]
    fig, axes = plt.subplots(2, len(columns), figsize=(16, 10))

    all_matrices = []
    for state in ("EO", "EC"):
        all_matrices.append(real[state])
        for epoch in CANDIDATE_EPOCHS:
            all_matrices.append(candidates[epoch][state])

    vmin = float(min(np.nanmin(m) for m in all_matrices))
    vmax = float(max(np.nanmax(m) for m in all_matrices))

    image = None
    for row_idx, state in enumerate(("EO", "EC")):
        matrices = [real[state]] + [candidates[e][state] for e in CANDIDATE_EPOCHS]
        for col_idx, (label, matrix) in enumerate(zip(columns, matrices)):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(matrix, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} — {state}")
            ax.set_xticks(np.arange(N_CHANNELS))
            ax.set_yticks(np.arange(N_CHANNELS))
            ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
            ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.90, wspace=0.25, hspace=0.25)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)



def save_overview_figure(rows: list[dict]) -> None:
    labels = [f"E{row['epoch']} RAW" for row in rows]
    fields = [
        ("EO scale err", lambda r: abs(r["eo_normalized_std_ratio"] - 1.0)),
        ("EC scale err", lambda r: abs(r["ec_normalized_std_ratio"] - 1.0)),
        ("alpha BP MAE", lambda r: r["alpha_bandpower_mae_db"]),
        ("alpha effect MAE", lambda r: r["alpha_effect_channel_mae_db"]),
        ("1−corr EO", lambda r: 1.0 - r["correlation_upper_triangle_r_eo"]),
        ("1−corr EC", lambda r: 1.0 - r["correlation_upper_triangle_r_ec"]),
        ("1−wPLI EO r", lambda r: 1.0 - r["alpha_dwpli2_upper_triangle_r_eo"]),
        ("1−wPLI EC r", lambda r: 1.0 - r["alpha_dwpli2_upper_triangle_r_ec"]),
    ]

    raw = np.asarray([[fn(row) for _, fn in fields] for row in rows], dtype=np.float64)
    column_max = np.maximum(raw.max(axis=0, keepdims=True), np.finfo(float).eps)
    display = raw / column_max

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    image = ax.imshow(display, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels([name for name, _ in fields], rotation=35, ha="right")
    ax.set_title("B0-v Confirmatory Audit — Candidate Overview (column-normalized error)")
    fig.colorbar(image, ax=ax, label="Relative error within metric")
    fig.tight_layout()
    fig.savefig(OVERVIEW_FIGURE, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — B0-V RAW CANDIDATE CONFIRMATORY AUDIT")
    print("=" * 80)

    print("\nCurrent project stage:")
    print("  B0-v full fine-tuning: COMPLETE")
    print("  RAW-vs-EMA sweep: COMPLETE")
    print("  RAW epoch-20 vs epoch-40 confirmatory audit: NOW")

    print("\nScientific purpose:")
    print("  Confirm candidate selection with larger synthetic samples and alpha wPLI.")
    print("  This script does NOT freeze B0 automatically.")

    print("\nDesign:")
    print(f"  Candidates: RAW epochs {list(CANDIDATE_EPOCHS)}")
    print(f"  Synthetic per candidate: {N_SYNTH_PER_STATE} EO + {N_SYNTH_PER_STATE} EC")
    print(f"  Real Train reference: {N_REAL_PER_STATE} EO + {N_REAL_PER_STATE} EC")
    print("  Same EO/EC sampling seeds across both candidates: YES")
    print("  Validation EEG accessed: NO")
    print("  Final Test EEG accessed: NO")
    print("  Training updates executed: NO")

    for directory in (GENERATED_DIR, LOG_DIR, METRIC_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Step 1 — Freeze audit protocol and audit candidates
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 1 — FREEZE CONFIRMATORY AUDIT PROTOCOL + AUDIT CHECKPOINTS")
    print("=" * 80)

    protocol = {
        "status": "B0_V_RAW_CANDIDATE_CONFIRMATORY_PROTOCOL_FROZEN",
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "weight_kind": "raw",
        "synthetic_per_state_per_candidate": N_SYNTH_PER_STATE,
        "real_train_per_state": N_REAL_PER_STATE,
        "sampler": "1000_step_ancestral_DDPM_v_prediction",
        "sampling_batch_size": SAMPLING_BATCH_SIZE,
        "eo_sampling_seed": EO_SAMPLING_SEED,
        "ec_sampling_seed": EC_SAMPLING_SEED,
        "same_sampling_seeds_across_candidates": True,
        "metrics": [
            "normalized_scale",
            "uV_scale",
            "delta_theta_alpha_beta_bandpower",
            "channelwise_alpha_state_effect",
            "mean_channel_correlation",
            "alpha_debiased_squared_wPLI",
        ],
        "validation_eeg_accessed": False,
        "final_test_accessed": False,
        "training_updates_executed": False,
    }
    if PROTOCOL_FILE.exists():
        with PROTOCOL_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != protocol:
            raise RuntimeError("Conflicting confirmatory protocol already exists")
    else:
        atomic_json_save(protocol, PROTOCOL_FILE)

    checkpoints = {}
    hashes_before = {}
    source_contract = None

    for epoch in CANDIDATE_EPOCHS:
        path = CANDIDATE_CHECKPOINTS[epoch]
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Epoch mismatch in {path}")
        if checkpoint.get("prediction_target") != "v":
            raise RuntimeError(f"Not a v-prediction checkpoint: {path}")
        if bool(checkpoint.get("final_test_accessed", True)):
            raise RuntimeError(f"Final Test invariant failed in {path}")
        if int(checkpoint.get("source_epoch", EXPECTED_SOURCE_EPOCH)) != EXPECTED_SOURCE_EPOCH:
            raise RuntimeError(f"Unexpected epsilon source epoch in {path}")

        current_contract = checkpoint["source_contract"]
        if source_contract is None:
            source_contract = current_contract
        elif current_contract != source_contract:
            raise RuntimeError("Source contract mismatch between candidates")

        hashes_before[epoch] = sha256(path)
        checkpoints[epoch] = checkpoint
        history_last = checkpoint["history"][-1]
        print(
            f"epoch {epoch:>2} RAW | train_v_MSE={history_last['train_v_mse']:.6f} | "
            f"EMA Val v-MSE={history_last['ema_validation_v_mse']:.6f} | "
            f"SHA256={hashes_before[epoch][:16]}..."
        )

    if source_contract is None:
        raise RuntimeError("Could not resolve common source contract")

    print("checkpoint audit: PASS")
    print(f"protocol: {PROTOCOL_FILE}")

    # ========================================================
    # Step 2 — Build fixed Real Train reference
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 2 — BUILD FIXED SUBJECT-BALANCED REAL TRAIN REFERENCE")
    print("=" * 80)

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)
    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked")

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)
    records, subject_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )
    if len(records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(f"Unexpected Train windows: {len(records)}")
    if len(subject_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Unexpected subject count: {len(subject_mapping)}")

    eo_records = select_subject_balanced_records(
        records, label=0, n=N_REAL_PER_STATE, seed=REAL_REFERENCE_SEED
    )
    ec_records = select_subject_balanced_records(
        records, label=1, n=N_REAL_PER_STATE, seed=REAL_REFERENCE_SEED + 1
    )
    save_real_manifest(eo_records, ec_records)

    real_z = {
        "EO": load_reference_array(eo_records, scaler),
        "EC": load_reference_array(ec_records, scaler),
    }
    real_uv = {state: inverse_normalize(real_z[state], scaler) for state in ("EO", "EC")}

    np.savez_compressed(
        REAL_REFERENCE_FILE,
        eo_z=real_z["EO"],
        ec_z=real_z["EC"],
        eo_uv=real_uv["EO"],
        ec_uv=real_uv["EC"],
        channel_names=np.asarray(CHANNEL_NAMES),
        sfreq=np.asarray([SFREQ], dtype=np.int64),
    )

    real_band = {state: compute_bandpower(real_uv[state]) for state in ("EO", "EC")}
    real_corr = {}
    real_wpli = {}
    for state in ("EO", "EC"):
        _, corr = mean_covariance_and_correlation(real_z[state])
        real_corr[state] = corr
        real_wpli[state] = alpha_dwpli2_debiased(real_uv[state])

    real_alpha_delta = (
        np.median(real_band["EC"]["alpha"], axis=0)
        - np.median(real_band["EO"]["alpha"], axis=0)
    )

    print(f"Real EO normalized std: {real_z['EO'].std():.6f}")
    print(f"Real EC normalized std: {real_z['EC'].std():.6f}")
    print(f"Real EO µV std: {real_uv['EO'].std():.3f}")
    print(f"Real EC µV std: {real_uv['EC'].std():.3f}")
    print(f"Real O1 alpha EC-EO: {real_alpha_delta[CHANNEL_INDEX['O1']]:.3f} dB")
    print(f"Real O2 alpha EC-EO: {real_alpha_delta[CHANNEL_INDEX['O2']]:.3f} dB")
    print(f"reference: {REAL_REFERENCE_FILE}")

    # ========================================================
    # Step 3 — Build sampler once
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 3 — BUILD FIXED V-PREDICTION SAMPLER")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    schedule_cpu = build_cosine_schedule(timesteps=TIMESTEPS, cosine_s=COSINE_S)
    reverse_schedule = build_reverse_schedule(schedule_cpu, device)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("sampler: 1000-step ancestral DDPM / v-prediction / FP32")

    # ========================================================
    # Step 4 — Generate and evaluate candidates
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 4 — GENERATE + EVALUATE RAW CANDIDATES")
    print("=" * 80)

    rows = []
    candidate_corr = {}
    candidate_wpli = {}
    candidate_alpha = {}
    synthetic_files = {20: E20_SYNTHETIC_FILE, 40: E40_SYNTHETIC_FILE}

    overall_start = time.perf_counter()

    for epoch in CANDIDATE_EPOCHS:
        print("\n" + "-" * 80)
        print(f"RAW EPOCH {epoch} — CONFIRMATORY 256 EO + 256 EC")
        print("-" * 80)

        model = B0StateOnlyUNet1D(source_contract).to(device)
        model.load_state_dict(checkpoints[epoch]["model_state_dict"], strict=True)
        model.eval()

        start = time.perf_counter()
        eo_z = sample_v_batches(
            model=model,
            eye_state_value=0,
            n_samples=N_SYNTH_PER_STATE,
            batch_size=SAMPLING_BATCH_SIZE,
            reverse_schedule=reverse_schedule,
            device=device,
            seed=EO_SAMPLING_SEED,
        )
        ec_z = sample_v_batches(
            model=model,
            eye_state_value=1,
            n_samples=N_SYNTH_PER_STATE,
            batch_size=SAMPLING_BATCH_SIZE,
            reverse_schedule=reverse_schedule,
            device=device,
            seed=EC_SAMPLING_SEED,
        )

        synth_z = {"EO": eo_z, "EC": ec_z}
        synth_uv = {state: inverse_normalize(synth_z[state], scaler) for state in ("EO", "EC")}

        if not all(np.isfinite(synth_z[state]).all() for state in ("EO", "EC")):
            raise RuntimeError(f"Nonfinite synthetic data for epoch {epoch}")

        np.savez_compressed(
            synthetic_files[epoch],
            eo_z=synth_z["EO"],
            ec_z=synth_z["EC"],
            eo_uv=synth_uv["EO"],
            ec_uv=synth_uv["EC"],
            channel_names=np.asarray(CHANNEL_NAMES),
            sfreq=np.asarray([SFREQ], dtype=np.int64),
            checkpoint_epoch=np.asarray([epoch], dtype=np.int64),
            weight_kind=np.asarray(["raw"]),
            eo_sampling_seed=np.asarray([EO_SAMPLING_SEED], dtype=np.int64),
            ec_sampling_seed=np.asarray([EC_SAMPLING_SEED], dtype=np.int64),
        )

        row, corr, wpli, extra = candidate_metrics(
            epoch=epoch,
            real_z=real_z,
            real_uv=real_uv,
            synth_z=synth_z,
            synth_uv=synth_uv,
            real_band=real_band,
            real_corr=real_corr,
            real_wpli=real_wpli,
        )
        row["sampling_and_metrics_minutes"] = float((time.perf_counter() - start) / 60.0)
        rows.append(row)
        candidate_corr[epoch] = corr
        candidate_wpli[epoch] = wpli
        candidate_alpha[epoch] = extra["alpha"]

        print(
            f"normalized std ratio EO/EC: "
            f"{row['eo_normalized_std_ratio']:.3f} / {row['ec_normalized_std_ratio']:.3f}"
        )
        print(
            f"µV std ratio EO/EC: "
            f"{row['eo_uv_std_ratio']:.3f} / {row['ec_uv_std_ratio']:.3f}"
        )
        print(
            f"O1 alpha EC-EO: real={row['o1_real_alpha_effect_db']:.3f} | "
            f"synthetic={row['o1_synthetic_alpha_effect_db']:.3f} dB"
        )
        print(
            f"O2 alpha EC-EO: real={row['o2_real_alpha_effect_db']:.3f} | "
            f"synthetic={row['o2_synthetic_alpha_effect_db']:.3f} dB"
        )
        print(f"alpha-effect channel MAE: {row['alpha_effect_channel_mae_db']:.3f} dB")
        print(
            f"bandpower MAE delta/theta/alpha/beta: "
            f"{row['delta_bandpower_mae_db']:.3f} / "
            f"{row['theta_bandpower_mae_db']:.3f} / "
            f"{row['alpha_bandpower_mae_db']:.3f} / "
            f"{row['beta_bandpower_mae_db']:.3f} dB"
        )
        print(
            f"correlation r EO/EC: "
            f"{row['correlation_upper_triangle_r_eo']:.3f} / "
            f"{row['correlation_upper_triangle_r_ec']:.3f}"
        )
        print(
            f"alpha dwPLI2 r EO/EC: "
            f"{row['alpha_dwpli2_upper_triangle_r_eo']:.3f} / "
            f"{row['alpha_dwpli2_upper_triangle_r_ec']:.3f}"
        )
        print(
            f"alpha dwPLI2 relF EO/EC: "
            f"{row['alpha_dwpli2_relF_eo']:.3f} / "
            f"{row['alpha_dwpli2_relF_ec']:.3f}"
        )
        print(f"candidate time: {row['sampling_and_metrics_minutes']:.2f} min")

        del model
        torch.cuda.empty_cache()

    runtime_minutes = (time.perf_counter() - overall_start) / 60.0

    # ========================================================
    # Step 5 — Save metrics and matrices
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 5 — SAVE CONFIRMATORY METRICS")
    print("=" * 80)

    with CANDIDATE_METRICS_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        MATRICES_FILE,
        real_corr_eo=real_corr["EO"],
        real_corr_ec=real_corr["EC"],
        real_wpli_eo=real_wpli["EO"],
        real_wpli_ec=real_wpli["EC"],
        epoch20_corr_eo=candidate_corr[20]["EO"],
        epoch20_corr_ec=candidate_corr[20]["EC"],
        epoch40_corr_eo=candidate_corr[40]["EO"],
        epoch40_corr_ec=candidate_corr[40]["EC"],
        epoch20_wpli_eo=candidate_wpli[20]["EO"],
        epoch20_wpli_ec=candidate_wpli[20]["EC"],
        epoch40_wpli_eo=candidate_wpli[40]["EO"],
        epoch40_wpli_ec=candidate_wpli[40]["EC"],
    )

    print(f"metrics CSV: {CANDIDATE_METRICS_FILE}")
    print(f"matrix package: {MATRICES_FILE}")

    # ========================================================
    # Step 6 — Figures
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 6 — SAVE CONFIRMATORY FIGURES")
    print("=" * 80)

    save_scale_figure(rows)
    save_alpha_figure(rows)
    save_bandpower_figure(rows)
    save_matrix_comparison(
        real=real_corr,
        candidates=candidate_corr,
        output=CORRELATION_FIGURE,
        title="B0-v Confirmatory Audit — Mean Channel Correlation",
    )
    save_matrix_comparison(
        real=real_wpli,
        candidates=candidate_wpli,
        output=WPLI_FIGURE,
        title="B0-v Confirmatory Audit — Alpha Debiased Squared wPLI",
    )
    save_overview_figure(rows)

    for path in (
        SCALE_FIGURE,
        ALPHA_FIGURE,
        BANDPOWER_FIGURE,
        CORRELATION_FIGURE,
        WPLI_FIGURE,
        OVERVIEW_FIGURE,
    ):
        print(path)

    # ========================================================
    # Step 7 — Candidate summary, no automatic freeze
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 7 — CONFIRMATORY CANDIDATE SUMMARY")
    print("=" * 80)

    by_epoch = {int(row["epoch"]): row for row in rows}

    def scale_error(row):
        return abs(row["eo_normalized_std_ratio"] - 1.0) + abs(row["ec_normalized_std_ratio"] - 1.0)

    def mean_corr(row):
        return 0.5 * (
            row["correlation_upper_triangle_r_eo"]
            + row["correlation_upper_triangle_r_ec"]
        )

    def mean_wpli(row):
        return 0.5 * (
            row["alpha_dwpli2_upper_triangle_r_eo"]
            + row["alpha_dwpli2_upper_triangle_r_ec"]
        )

    best_scale_epoch = min(CANDIDATE_EPOCHS, key=lambda e: scale_error(by_epoch[e]))
    best_alpha_effect_epoch = min(
        CANDIDATE_EPOCHS,
        key=lambda e: by_epoch[e]["alpha_effect_channel_mae_db"],
    )
    best_alpha_bp_epoch = min(
        CANDIDATE_EPOCHS,
        key=lambda e: by_epoch[e]["alpha_bandpower_mae_db"],
    )
    best_corr_epoch = max(CANDIDATE_EPOCHS, key=lambda e: mean_corr(by_epoch[e]))
    best_wpli_epoch = max(CANDIDATE_EPOCHS, key=lambda e: mean_wpli(by_epoch[e]))

    print(
        f"best scale fidelity: epoch {best_scale_epoch} RAW | "
        f"EO/EC ratio={by_epoch[best_scale_epoch]['eo_normalized_std_ratio']:.3f}/"
        f"{by_epoch[best_scale_epoch]['ec_normalized_std_ratio']:.3f}"
    )
    print(
        f"best alpha-effect fidelity: epoch {best_alpha_effect_epoch} RAW | "
        f"MAE={by_epoch[best_alpha_effect_epoch]['alpha_effect_channel_mae_db']:.3f} dB"
    )
    print(
        f"best alpha absolute-bandpower fidelity: epoch {best_alpha_bp_epoch} RAW | "
        f"MAE={by_epoch[best_alpha_bp_epoch]['alpha_bandpower_mae_db']:.3f} dB"
    )
    print(
        f"best spatial-correlation fidelity: epoch {best_corr_epoch} RAW | "
        f"r EO/EC={by_epoch[best_corr_epoch]['correlation_upper_triangle_r_eo']:.3f}/"
        f"{by_epoch[best_corr_epoch]['correlation_upper_triangle_r_ec']:.3f}"
    )
    print(
        f"best alpha dwPLI2 fidelity: epoch {best_wpli_epoch} RAW | "
        f"r EO/EC={by_epoch[best_wpli_epoch]['alpha_dwpli2_upper_triangle_r_eo']:.3f}/"
        f"{by_epoch[best_wpli_epoch]['alpha_dwpli2_upper_triangle_r_ec']:.3f}"
    )

    # A candidate is only auto-recommended when it wins at least 3/5 named criteria.
    criterion_winners = [
        best_scale_epoch,
        best_alpha_effect_epoch,
        best_alpha_bp_epoch,
        best_corr_epoch,
        best_wpli_epoch,
    ]
    winner_counts = {e: criterion_winners.count(e) for e in CANDIDATE_EPOCHS}
    if max(winner_counts.values()) >= 3:
        recommended_epoch = max(CANDIDATE_EPOCHS, key=lambda e: winner_counts[e])
        diagnostic_direction = f"RAW_EPOCH_{recommended_epoch}_PRIMARY_CONFIRMATORY_CANDIDATE"
    else:
        recommended_epoch = None
        diagnostic_direction = "MIXED_CONFIRMATORY_RESULT_REQUIRES_INTERPRETATION"

    print(f"criterion wins: {winner_counts}")
    print(f"diagnostic direction: {diagnostic_direction}")
    print("B0 frozen: NO")

    # ========================================================
    # Step 8 — Checkpoint immutability + summary
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 8 — VERIFY CHECKPOINT IMMUTABILITY + SAVE SUMMARY")
    print("=" * 80)

    hashes_after = {epoch: sha256(CANDIDATE_CHECKPOINTS[epoch]) for epoch in CANDIDATE_EPOCHS}
    immutable = hashes_after == hashes_before
    if not immutable:
        raise RuntimeError("Candidate checkpoint immutability FAILED")
    print("checkpoint immutability: PASS")

    summary = {
        "status": "B0_V_RAW_CANDIDATE_CONFIRMATORY_AUDIT_COMPLETE",
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "weight_kind": "raw",
        "candidate_metrics": rows,
        "criterion_winners": {
            "scale_fidelity": best_scale_epoch,
            "alpha_effect_fidelity": best_alpha_effect_epoch,
            "alpha_bandpower_fidelity": best_alpha_bp_epoch,
            "spatial_correlation_fidelity": best_corr_epoch,
            "alpha_dwpli2_fidelity": best_wpli_epoch,
        },
        "criterion_win_counts": winner_counts,
        "recommended_epoch": recommended_epoch,
        "diagnostic_direction": diagnostic_direction,
        "runtime_minutes": float(runtime_minutes),
        "checkpoint_hashes": {
            str(epoch): hashes_before[epoch] for epoch in CANDIDATE_EPOCHS
        },
        "outputs": {
            "protocol": str(PROTOCOL_FILE),
            "metrics_csv": str(CANDIDATE_METRICS_FILE),
            "real_reference": str(REAL_REFERENCE_FILE),
            "real_manifest": str(REAL_MANIFEST_FILE),
            "epoch20_synthetic": str(E20_SYNTHETIC_FILE),
            "epoch40_synthetic": str(E40_SYNTHETIC_FILE),
            "matrices": str(MATRICES_FILE),
            "scale_figure": str(SCALE_FIGURE),
            "alpha_figure": str(ALPHA_FIGURE),
            "bandpower_figure": str(BANDPOWER_FIGURE),
            "correlation_figure": str(CORRELATION_FIGURE),
            "wpli_figure": str(WPLI_FIGURE),
            "overview_figure": str(OVERVIEW_FIGURE),
        },
        "checkpoint_immutability": True,
        "validation_eeg_accessed": False,
        "final_test_accessed": False,
        "training_updates_executed": False,
        "b0_frozen": False,
    }
    atomic_json_save(summary, SUMMARY_FILE)
    print(f"summary JSON: {SUMMARY_FILE}")

    # ========================================================
    # Final
    # ========================================================
    print("\n" + "=" * 80)
    print("B0-V RAW CANDIDATE CONFIRMATORY AUDIT COMPLETE")
    print("=" * 80)
    print(f"candidates evaluated: {len(CANDIDATE_EPOCHS)}")
    print(f"synthetic per candidate: {N_SYNTH_PER_STATE} EO + {N_SYNTH_PER_STATE} EC")
    print(f"runtime: {runtime_minutes:.2f} min")
    print(f"diagnostic direction: {diagnostic_direction}")
    print("Validation EEG accessed: NO")
    print("Final Test accessed: NO")
    print("Training updates executed: NO")
    print("B0 frozen: NO")
    print("\nNEXT:")
    print("INTERPRET CONFIRMATORY RAW CANDIDATE AUDIT BEFORE FREEZING B0")


if __name__ == "__main__":
    main()
