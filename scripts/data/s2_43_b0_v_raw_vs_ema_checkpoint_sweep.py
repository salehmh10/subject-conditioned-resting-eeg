from __future__ import annotations

import csv
import hashlib
import json
import os
import time
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
from s2_32_b0_unet_shape_test import B0StateOnlyUNet1D
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    BANDS,
    CHANNEL_INDEX,
    CHANNEL_NAMES,
    COSINE_S,
    REAL_REFERENCE_SEED,
    TIMESTEPS,
    bandpower_db_from_psd,
    build_reverse_schedule,
    inverse_normalize,
    load_reference_array,
    mean_covariance_and_correlation,
    sample_v_batches,
    select_subject_balanced_records,
    upper_triangle_correlation,
    welch_psd,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

FULL_RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "full_finetune_v1"
)
CHECKPOINT_DIR = FULL_RUN_DIR / "checkpoints"

SWEEP_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "raw_vs_ema_sweep_v1"
)
GENERATED_DIR = SWEEP_DIR / "generated"
LOG_DIR = SWEEP_DIR / "logs"
METRIC_DIR = SWEEP_DIR / "metrics"
FIGURE_DIR = SWEEP_DIR / "figures"

METRIC_CSV = METRIC_DIR / "b0_v_raw_vs_ema_sweep_metrics_v1.csv"
SUMMARY_JSON = METRIC_DIR / "b0_v_raw_vs_ema_sweep_summary_v1.json"
REAL_REFERENCE_FILE = GENERATED_DIR / "b0_v_raw_vs_ema_real_train_reference_v1.npz"
SYNTHETIC_FILE = GENERATED_DIR / "b0_v_raw_vs_ema_synthetic_v1.npz"
REAL_MANIFEST_FILE = LOG_DIR / "b0_v_raw_vs_ema_real_train_reference_manifest_v1.csv"

SCALE_FIGURE = FIGURE_DIR / "b0_v_raw_vs_ema_scale_ratio_v1.png"
ALPHA_O1_O2_FIGURE = FIGURE_DIR / "b0_v_raw_vs_ema_occipital_alpha_effect_v1.png"
ALPHA_MAE_FIGURE = FIGURE_DIR / "b0_v_raw_vs_ema_alpha_effect_mae_v1.png"
BANDPOWER_FIGURE = FIGURE_DIR / "b0_v_raw_vs_ema_bandpower_mae_v1.png"
CORRELATION_FIGURE = FIGURE_DIR / "b0_v_raw_vs_ema_correlation_v1.png"
OVERVIEW_FIGURE = FIGURE_DIR / "b0_v_raw_vs_ema_overview_heatmap_v1.png"


# ============================================================
# Frozen diagnostic design
# ============================================================

EPOCHS = (10, 20, 30, 40)
WEIGHT_KINDS = ("raw", "ema")

# Small but useful sweep. The same stochastic seeds are reused for every
# checkpoint/weight type so differences mainly reflect model weights.
N_SYNTH_PER_STATE = 32
SAMPLING_BATCH_SIZE = 32
EO_SAMPLING_SEED = 20260914
EC_SAMPLING_SEED = 20260915

# Keep the Real reference more stable than the synthetic diagnostic set.
N_REAL_PER_STATE = 256
EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128


# ============================================================
# Helpers
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



def checkpoint_path(epoch: int) -> Path:
    return CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"



def save_reference_manifest(eo_records, ec_records) -> None:
    rows = []
    for label, state, records in (
        (0, "EO", eo_records),
        (1, "EC", ec_records),
    ):
        for reference_index, record in enumerate(records):
            rows.append(
                {
                    "reference_index": int(reference_index),
                    "eye_state": state,
                    "label": int(label),
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



def compute_real_reference(scaler: dict):
    split = pd.read_csv(FINAL_SPLIT_MANIFEST)
    records, subject_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )

    if len(records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(f"Unexpected Train window count: {len(records)}")
    if len(subject_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Unexpected subject count: {len(subject_mapping)}")

    eo_records = select_subject_balanced_records(
        records,
        label=0,
        n=N_REAL_PER_STATE,
        seed=REAL_REFERENCE_SEED,
    )
    ec_records = select_subject_balanced_records(
        records,
        label=1,
        n=N_REAL_PER_STATE,
        seed=REAL_REFERENCE_SEED + 1,
    )

    real_z = {
        "EO": load_reference_array(eo_records, scaler),
        "EC": load_reference_array(ec_records, scaler),
    }
    real_uv = {
        state: inverse_normalize(array, scaler)
        for state, array in real_z.items()
    }

    save_reference_manifest(eo_records, ec_records)

    np.savez_compressed(
        REAL_REFERENCE_FILE,
        eo_z=real_z["EO"],
        ec_z=real_z["EC"],
        eo_uv=real_uv["EO"],
        ec_uv=real_uv["EC"],
        channel_names=np.asarray(CHANNEL_NAMES),
        sfreq=np.asarray([SFREQ], dtype=np.int64),
    )

    return real_z, real_uv



def compute_band_metrics(real_uv, synth_uv):
    real_bands = {}
    synth_bands = {}

    for state in ("EO", "EC"):
        f_real, p_real = welch_psd(real_uv[state])
        f_synth, p_synth = welch_psd(synth_uv[state])
        if not np.array_equal(f_real, f_synth):
            raise RuntimeError("Real/Synthetic PSD frequency grids differ")
        real_bands[state] = bandpower_db_from_psd(f_real, p_real)
        synth_bands[state] = bandpower_db_from_psd(f_synth, p_synth)

    band_mae = {}
    for band in BANDS:
        errors = []
        for state in ("EO", "EC"):
            real_median = np.median(real_bands[state][band], axis=0)
            synth_median = np.median(synth_bands[state][band], axis=0)
            errors.extend(np.abs(synth_median - real_median).tolist())
        band_mae[band] = float(np.mean(errors))

    real_alpha_delta = (
        np.median(real_bands["EC"]["alpha"], axis=0)
        - np.median(real_bands["EO"]["alpha"], axis=0)
    )
    synth_alpha_delta = (
        np.median(synth_bands["EC"]["alpha"], axis=0)
        - np.median(synth_bands["EO"]["alpha"], axis=0)
    )

    alpha_effect_mae = float(
        np.mean(np.abs(synth_alpha_delta - real_alpha_delta))
    )

    whole_head_real = float(np.median(real_alpha_delta))
    whole_head_synth = float(np.median(synth_alpha_delta))

    return {
        "band_mae": band_mae,
        "real_alpha_delta": real_alpha_delta,
        "synth_alpha_delta": synth_alpha_delta,
        "alpha_effect_mae": alpha_effect_mae,
        "whole_head_real": whole_head_real,
        "whole_head_synth": whole_head_synth,
    }



def compute_spatial_metrics(real_z, synth_z):
    output = {}
    for state in ("EO", "EC"):
        _, real_corr = mean_covariance_and_correlation(real_z[state])
        _, synth_corr = mean_covariance_and_correlation(synth_z[state])
        output[state] = upper_triangle_correlation(real_corr, synth_corr)
    return output



def metric_row(
    *,
    epoch: int,
    weight_kind: str,
    checkpoint: Path,
    checkpoint_hash: str,
    real_z,
    real_uv,
    synth_z,
    synth_uv,
):
    bands = compute_band_metrics(real_uv, synth_uv)
    spatial = compute_spatial_metrics(real_z, synth_z)

    eo_norm_ratio = float(
        synth_z["EO"].std()
        / max(real_z["EO"].std(), np.finfo(float).eps)
    )
    ec_norm_ratio = float(
        synth_z["EC"].std()
        / max(real_z["EC"].std(), np.finfo(float).eps)
    )
    eo_uv_ratio = float(
        synth_uv["EO"].std()
        / max(real_uv["EO"].std(), np.finfo(float).eps)
    )
    ec_uv_ratio = float(
        synth_uv["EC"].std()
        / max(real_uv["EC"].std(), np.finfo(float).eps)
    )

    real_alpha_delta = bands["real_alpha_delta"]
    synth_alpha_delta = bands["synth_alpha_delta"]
    o1 = CHANNEL_INDEX["O1"]
    o2 = CHANNEL_INDEX["O2"]

    return {
        "epoch": int(epoch),
        "weight_kind": weight_kind,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "synthetic_samples_per_state": int(N_SYNTH_PER_STATE),
        "eo_normalized_std": float(synth_z["EO"].std()),
        "ec_normalized_std": float(synth_z["EC"].std()),
        "eo_normalized_std_ratio": eo_norm_ratio,
        "ec_normalized_std_ratio": ec_norm_ratio,
        "eo_uv_std": float(synth_uv["EO"].std()),
        "ec_uv_std": float(synth_uv["EC"].std()),
        "eo_uv_std_ratio": eo_uv_ratio,
        "ec_uv_std_ratio": ec_uv_ratio,
        "delta_bandpower_mae_db": bands["band_mae"]["delta"],
        "theta_bandpower_mae_db": bands["band_mae"]["theta"],
        "alpha_bandpower_mae_db": bands["band_mae"]["alpha"],
        "beta_bandpower_mae_db": bands["band_mae"]["beta"],
        "all_bandpower_mae_db": float(np.mean(list(bands["band_mae"].values()))),
        "whole_head_real_alpha_effect_db": bands["whole_head_real"],
        "whole_head_synthetic_alpha_effect_db": bands["whole_head_synth"],
        "o1_real_alpha_effect_db": float(real_alpha_delta[o1]),
        "o1_synthetic_alpha_effect_db": float(synth_alpha_delta[o1]),
        "o2_real_alpha_effect_db": float(real_alpha_delta[o2]),
        "o2_synthetic_alpha_effect_db": float(synth_alpha_delta[o2]),
        "alpha_effect_channel_mae_db": bands["alpha_effect_mae"],
        "correlation_upper_triangle_r_eo": float(spatial["EO"]),
        "correlation_upper_triangle_r_ec": float(spatial["EC"]),
    }



def save_figures(rows: list[dict]) -> None:
    def values(kind: str, field: str):
        subset = sorted(
            [row for row in rows if row["weight_kind"] == kind],
            key=lambda row: row["epoch"],
        )
        return (
            np.asarray([row["epoch"] for row in subset]),
            np.asarray([row[field] for row in subset], dtype=np.float64),
        )

    # --------------------------------------------------------
    # Scale ratios
    # --------------------------------------------------------
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    for kind in WEIGHT_KINDS:
        epochs, eo = values(kind, "eo_normalized_std_ratio")
        _, ec = values(kind, "ec_normalized_std_ratio")
        ax.plot(epochs, eo, marker="o", label=f"{kind.upper()} EO")
        ax.plot(epochs, ec, marker="o", linestyle="--", label=f"{kind.upper()} EC")
    ax.axhline(1.0, linewidth=1, label="Real target ratio = 1")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Synthetic / Real normalized std")
    ax.set_title("B0-v Raw vs EMA — Scale Fidelity")
    ax.set_xticks(EPOCHS)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(SCALE_FIGURE, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # O1/O2 alpha effects
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for ax, channel in zip(axes, ("O1", "O2")):
        real_field = f"{channel.lower()}_real_alpha_effect_db"
        synth_field = f"{channel.lower()}_synthetic_alpha_effect_db"
        real_value = float(rows[0][real_field])
        ax.axhline(real_value, linewidth=1.5, label=f"Real {channel}")
        for kind in WEIGHT_KINDS:
            epochs, synth = values(kind, synth_field)
            ax.plot(epochs, synth, marker="o", label=kind.upper())
        ax.set_ylabel(f"{channel} EC−EO alpha (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Fine-tuning epoch")
    axes[-1].set_xticks(EPOCHS)
    fig.suptitle("B0-v Raw vs EMA — Occipital Eye-State Alpha Effect")
    fig.tight_layout()
    fig.savefig(ALPHA_O1_O2_FIGURE, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Alpha-effect channel MAE
    # --------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for kind in WEIGHT_KINDS:
        epochs, y = values(kind, "alpha_effect_channel_mae_db")
        ax.plot(epochs, y, marker="o", label=kind.upper())
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Channel-wise alpha-effect MAE (dB)")
    ax.set_title("B0-v Raw vs EMA — Eye-State Alpha Topography Error")
    ax.set_xticks(EPOCHS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ALPHA_MAE_FIGURE, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Bandpower MAE
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, band in zip(axes.flat, ("delta", "theta", "alpha", "beta")):
        field = f"{band}_bandpower_mae_db"
        for kind in WEIGHT_KINDS:
            epochs, y = values(kind, field)
            ax.plot(epochs, y, marker="o", label=kind.upper())
        ax.set_title(band)
        ax.set_ylabel("Median-bandpower MAE (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(EPOCHS)
    axes[1, 0].set_xlabel("Fine-tuning epoch")
    axes[1, 1].set_xlabel("Fine-tuning epoch")
    axes[0, 0].legend()
    fig.suptitle("B0-v Raw vs EMA — Absolute Bandpower Fidelity")
    fig.tight_layout()
    fig.savefig(BANDPOWER_FIGURE, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Correlation fidelity
    # --------------------------------------------------------
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    for kind in WEIGHT_KINDS:
        epochs, eo = values(kind, "correlation_upper_triangle_r_eo")
        _, ec = values(kind, "correlation_upper_triangle_r_ec")
        ax.plot(epochs, eo, marker="o", label=f"{kind.upper()} EO")
        ax.plot(epochs, ec, marker="o", linestyle="--", label=f"{kind.upper()} EC")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Real/Synthetic mean-correlation upper-triangle r")
    ax.set_title("B0-v Raw vs EMA — Spatial Correlation Fidelity")
    ax.set_xticks(EPOCHS)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(CORRELATION_FIGURE, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Compact overview heatmap
    # Lower is better for errors; std closeness is encoded as |ratio-1|.
    # --------------------------------------------------------
    ordered = sorted(rows, key=lambda r: (r["epoch"], r["weight_kind"]))
    labels = [f"E{row['epoch']}-{row['weight_kind'].upper()}" for row in ordered]
    matrix = np.asarray(
        [
            [
                abs(row["eo_normalized_std_ratio"] - 1.0),
                abs(row["ec_normalized_std_ratio"] - 1.0),
                row["alpha_bandpower_mae_db"],
                row["alpha_effect_channel_mae_db"],
                1.0 - row["correlation_upper_triangle_r_eo"],
                1.0 - row["correlation_upper_triangle_r_ec"],
            ]
            for row in ordered
        ],
        dtype=np.float64,
    )

    # Normalize each column only for visualization, preserving the raw CSV.
    column_max = np.maximum(matrix.max(axis=0, keepdims=True), np.finfo(float).eps)
    display = matrix / column_max

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    image = ax.imshow(display, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(
        [
            "|EO std ratio−1|",
            "|EC std ratio−1|",
            "alpha BP MAE",
            "alpha effect MAE",
            "1−corr r EO",
            "1−corr r EC",
        ],
        rotation=35,
        ha="right",
    )
    ax.set_title("B0-v Raw vs EMA — Diagnostic Overview (column-normalized error)")
    fig.colorbar(image, ax=ax, label="Relative error within metric")
    fig.tight_layout()
    fig.savefig(OVERVIEW_FIGURE, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — B0-V RAW VS EMA CHECKPOINT SWEEP")
    print("=" * 80)

    print("\nCurrent project stage:")
    print("  B0-v 40-epoch fine-tuning: COMPLETE")
    print("  B0-v final fidelity audit: COMPLETE")
    print("  Raw-vs-EMA checkpoint diagnostic: NOW")

    print("\nDiagnostic question:")
    print("  Is the under-dispersion / weak alpha effect mainly caused by slow EMA,")
    print("  or is it also present in the raw fine-tuned model weights?")

    print("\nSweep:")
    print(f"  epochs: {list(EPOCHS)}")
    print(f"  weights per epoch: {list(WEIGHT_KINDS)}")
    print(f"  synthetic per variant: {N_SYNTH_PER_STATE} EO + {N_SYNTH_PER_STATE} EC")
    print(f"  Real Train reference: {N_REAL_PER_STATE} EO + {N_REAL_PER_STATE} EC")
    print("  identical EO/EC sampling seeds reused across all variants: YES")
    print("  Validation EEG accessed: NO")
    print("  Final Test accessed: NO")
    print("  Training updates executed: NO")

    for directory in (GENERATED_DIR, LOG_DIR, METRIC_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Step 1 — Audit milestone checkpoints
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT MILESTONE CHECKPOINTS")
    print("=" * 80)

    paths = {epoch: checkpoint_path(epoch) for epoch in EPOCHS}
    hashes_before = {}
    checkpoint_meta = {}
    source_contract = None

    for epoch, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["epoch"]) != epoch:
            raise RuntimeError(
                f"Checkpoint epoch mismatch for {path}: {checkpoint['epoch']}"
            )
        if checkpoint.get("prediction_target") != "v":
            raise RuntimeError(f"Checkpoint is not v-prediction: {path}")
        if bool(checkpoint.get("final_test_accessed", True)):
            raise RuntimeError(f"Final Test invariant failed in {path}")

        current_contract = checkpoint["source_contract"]
        if source_contract is None:
            source_contract = current_contract
        elif current_contract != source_contract:
            raise RuntimeError("Source-contract mismatch across milestone checkpoints")

        hashes_before[epoch] = sha256(path)
        checkpoint_meta[epoch] = {
            "path": str(path),
            "sha256": hashes_before[epoch],
            "ema_validation_v_mse": float(checkpoint["history"][-1]["ema_validation_v_mse"]),
            "train_v_mse": float(checkpoint["history"][-1]["train_v_mse"]),
        }

        print(
            f"epoch {epoch:>2}: train_v_MSE="
            f"{checkpoint_meta[epoch]['train_v_mse']:.6f} | "
            f"EMA Val v-MSE={checkpoint_meta[epoch]['ema_validation_v_mse']:.6f} | "
            f"SHA256={hashes_before[epoch][:16]}..."
        )

    if source_contract is None:
        raise RuntimeError("Could not resolve source contract")

    print("checkpoint audit: PASS")

    # ========================================================
    # Step 2 — Build deterministic Real Train reference
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 2 — BUILD FIXED REAL TRAIN REFERENCE")
    print("=" * 80)

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)
    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked")

    real_z, real_uv = compute_real_reference(scaler)

    print(f"Real EO normalized std: {real_z['EO'].std():.6f}")
    print(f"Real EC normalized std: {real_z['EC'].std():.6f}")
    print(f"Real EO µV std: {real_uv['EO'].std():.3f}")
    print(f"Real EC µV std: {real_uv['EC'].std():.3f}")
    print(f"Real reference package: {REAL_REFERENCE_FILE}")

    # Compute Real alpha effect once for console context.
    real_band_context = compute_band_metrics(real_uv, real_uv)
    o1 = CHANNEL_INDEX["O1"]
    o2 = CHANNEL_INDEX["O2"]
    print(
        f"Real O1 alpha EC-EO: "
        f"{real_band_context['real_alpha_delta'][o1]:.3f} dB"
    )
    print(
        f"Real O2 alpha EC-EO: "
        f"{real_band_context['real_alpha_delta'][o2]:.3f} dB"
    )

    # ========================================================
    # Step 3 — Device + sampler
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 3 — BUILD FIXED V-PREDICTION SAMPLER")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this sweep")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    schedule_cpu = build_cosine_schedule(TIMESTEPS, COSINE_S)
    reverse_schedule = build_reverse_schedule(schedule_cpu, device)
    print("sampler: 1000-step ancestral DDPM with v-prediction")
    print("sampling precision: FP32")

    # ========================================================
    # Step 4 — Sweep RAW and EMA at each milestone
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 4 — SAMPLE RAW VS EMA MILESTONES")
    print("=" * 80)

    rows: list[dict] = []
    synthetic_package: dict[str, np.ndarray] = {}
    total_start = time.perf_counter()

    model = B0StateOnlyUNet1D(source_contract).to(device)
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    for epoch in EPOCHS:
        path = paths[epoch]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        for weight_kind in WEIGHT_KINDS:
            print("\n" + "-" * 80)
            print(f"EPOCH {epoch} — {weight_kind.upper()} WEIGHTS")
            print("-" * 80)

            state_dict_key = (
                "model_state_dict" if weight_kind == "raw" else "ema_state_dict"
            )
            model.load_state_dict(checkpoint[state_dict_key], strict=True)
            model.eval()

            torch.cuda.empty_cache()
            variant_start = time.perf_counter()

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
            synth_uv = {
                state: inverse_normalize(array, scaler)
                for state, array in synth_z.items()
            }

            row = metric_row(
                epoch=epoch,
                weight_kind=weight_kind,
                checkpoint=path,
                checkpoint_hash=hashes_before[epoch],
                real_z=real_z,
                real_uv=real_uv,
                synth_z=synth_z,
                synth_uv=synth_uv,
            )
            rows.append(row)

            prefix = f"e{epoch:03d}_{weight_kind}"
            synthetic_package[f"{prefix}_eo_z"] = eo_z
            synthetic_package[f"{prefix}_ec_z"] = ec_z
            synthetic_package[f"{prefix}_eo_uv"] = synth_uv["EO"]
            synthetic_package[f"{prefix}_ec_uv"] = synth_uv["EC"]

            elapsed = time.perf_counter() - variant_start
            print(
                f"normalized std ratio EO/EC: "
                f"{row['eo_normalized_std_ratio']:.3f} / "
                f"{row['ec_normalized_std_ratio']:.3f}"
            )
            print(
                f"µV std ratio EO/EC: "
                f"{row['eo_uv_std_ratio']:.3f} / "
                f"{row['ec_uv_std_ratio']:.3f}"
            )
            print(
                f"O1 alpha EC-EO: real={row['o1_real_alpha_effect_db']:.3f} "
                f"| synth={row['o1_synthetic_alpha_effect_db']:.3f} dB"
            )
            print(
                f"O2 alpha EC-EO: real={row['o2_real_alpha_effect_db']:.3f} "
                f"| synth={row['o2_synthetic_alpha_effect_db']:.3f} dB"
            )
            print(
                f"alpha-effect channel MAE: "
                f"{row['alpha_effect_channel_mae_db']:.3f} dB"
            )
            print(
                f"alpha bandpower MAE: "
                f"{row['alpha_bandpower_mae_db']:.3f} dB"
            )
            print(
                f"correlation r EO/EC: "
                f"{row['correlation_upper_triangle_r_eo']:.3f} / "
                f"{row['correlation_upper_triangle_r_ec']:.3f}"
            )
            print(f"variant sampling+metrics time: {elapsed / 60:.2f} min")

    del model
    torch.cuda.empty_cache()

    # ========================================================
    # Step 5 — Save metrics + generated diagnostic samples
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 5 — SAVE SWEEP OUTPUTS")
    print("=" * 80)

    with METRIC_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    synthetic_package["channel_names"] = np.asarray(CHANNEL_NAMES)
    synthetic_package["epochs"] = np.asarray(EPOCHS, dtype=np.int64)
    synthetic_package["eo_sampling_seed"] = np.asarray(
        [EO_SAMPLING_SEED], dtype=np.int64
    )
    synthetic_package["ec_sampling_seed"] = np.asarray(
        [EC_SAMPLING_SEED], dtype=np.int64
    )
    np.savez_compressed(SYNTHETIC_FILE, **synthetic_package)

    save_figures(rows)

    print(f"metrics CSV: {METRIC_CSV}")
    print(f"synthetic diagnostic package: {SYNTHETIC_FILE}")
    print(f"scale figure: {SCALE_FIGURE}")
    print(f"occipital alpha figure: {ALPHA_O1_O2_FIGURE}")
    print(f"alpha MAE figure: {ALPHA_MAE_FIGURE}")
    print(f"bandpower figure: {BANDPOWER_FIGURE}")
    print(f"correlation figure: {CORRELATION_FIGURE}")
    print(f"overview figure: {OVERVIEW_FIGURE}")

    # ========================================================
    # Step 6 — Diagnostic interpretation summary
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 6 — RAW VS EMA SUMMARY")
    print("=" * 80)

    def best_row_by(field: str, target: float | None = None):
        if target is None:
            return min(rows, key=lambda row: float(row[field]))
        return min(rows, key=lambda row: abs(float(row[field]) - target))

    # Aggregate scale error across EO/EC.
    for row in rows:
        row["mean_normalized_scale_abs_error"] = float(
            0.5
            * (
                abs(row["eo_normalized_std_ratio"] - 1.0)
                + abs(row["ec_normalized_std_ratio"] - 1.0)
            )
        )

    best_scale = min(rows, key=lambda row: row["mean_normalized_scale_abs_error"])
    best_alpha_effect = best_row_by("alpha_effect_channel_mae_db")
    best_alpha_power = best_row_by("alpha_bandpower_mae_db")
    best_corr = max(
        rows,
        key=lambda row: 0.5
        * (
            row["correlation_upper_triangle_r_eo"]
            + row["correlation_upper_triangle_r_ec"]
        ),
    )

    print(
        "best scale fidelity: "
        f"epoch {best_scale['epoch']} {best_scale['weight_kind'].upper()} "
        f"| EO/EC ratio={best_scale['eo_normalized_std_ratio']:.3f}/"
        f"{best_scale['ec_normalized_std_ratio']:.3f}"
    )
    print(
        "best alpha-effect fidelity: "
        f"epoch {best_alpha_effect['epoch']} {best_alpha_effect['weight_kind'].upper()} "
        f"| MAE={best_alpha_effect['alpha_effect_channel_mae_db']:.3f} dB"
    )
    print(
        "best alpha absolute-bandpower fidelity: "
        f"epoch {best_alpha_power['epoch']} {best_alpha_power['weight_kind'].upper()} "
        f"| MAE={best_alpha_power['alpha_bandpower_mae_db']:.3f} dB"
    )
    print(
        "best spatial-correlation fidelity: "
        f"epoch {best_corr['epoch']} {best_corr['weight_kind'].upper()} "
        f"| r EO/EC={best_corr['correlation_upper_triangle_r_eo']:.3f}/"
        f"{best_corr['correlation_upper_triangle_r_ec']:.3f}"
    )

    # Simple evidence statement — not a scientific freeze decision.
    raw_rows = [row for row in rows if row["weight_kind"] == "raw"]
    ema_rows = [row for row in rows if row["weight_kind"] == "ema"]
    raw_scale_error = float(
        np.mean([row["mean_normalized_scale_abs_error"] for row in raw_rows])
    )
    ema_scale_error = float(
        np.mean([row["mean_normalized_scale_abs_error"] for row in ema_rows])
    )
    raw_alpha_effect_mae = float(
        np.mean([row["alpha_effect_channel_mae_db"] for row in raw_rows])
    )
    ema_alpha_effect_mae = float(
        np.mean([row["alpha_effect_channel_mae_db"] for row in ema_rows])
    )

    if (
        raw_scale_error < ema_scale_error
        and raw_alpha_effect_mae < ema_alpha_effect_mae
    ):
        direction = "RAW_WEIGHTS_CONSISTENTLY_OUTPERFORM_EMA_ON_KEY_DIAGNOSTICS"
    elif (
        ema_scale_error < raw_scale_error
        and ema_alpha_effect_mae < raw_alpha_effect_mae
    ):
        direction = "EMA_WEIGHTS_CONSISTENTLY_OUTPERFORM_RAW_ON_KEY_DIAGNOSTICS"
    else:
        direction = "RAW_VS_EMA_TRADEOFF_REQUIRES_METRIC_LEVEL_INTERPRETATION"

    print(f"diagnostic direction: {direction}")

    # ========================================================
    # Step 7 — Checkpoint immutability + summary JSON
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 7 — VERIFY CHECKPOINT IMMUTABILITY + SAVE SUMMARY")
    print("=" * 80)

    hashes_after = {epoch: sha256(path) for epoch, path in paths.items()}
    immutable = hashes_before == hashes_after
    if not immutable:
        raise RuntimeError("One or more milestone checkpoints changed during sweep")

    total_seconds = time.perf_counter() - total_start

    summary = {
        "status": "B0_V_RAW_VS_EMA_SWEEP_COMPLETE",
        "question": (
            "Determine whether B0-v under-dispersion and weak alpha-state effect "
            "are primarily associated with slow EMA weights versus raw weights."
        ),
        "epochs": list(EPOCHS),
        "weight_kinds": list(WEIGHT_KINDS),
        "synthetic_samples_per_state_per_variant": N_SYNTH_PER_STATE,
        "real_reference_samples_per_state": N_REAL_PER_STATE,
        "sampling": {
            "sampler": "1000-step ancestral DDPM v-prediction",
            "batch_size": SAMPLING_BATCH_SIZE,
            "eo_seed": EO_SAMPLING_SEED,
            "ec_seed": EC_SAMPLING_SEED,
            "same_seeds_across_variants": True,
        },
        "checkpoint_metadata": checkpoint_meta,
        "rows": rows,
        "aggregate": {
            "mean_raw_scale_abs_error": raw_scale_error,
            "mean_ema_scale_abs_error": ema_scale_error,
            "mean_raw_alpha_effect_mae_db": raw_alpha_effect_mae,
            "mean_ema_alpha_effect_mae_db": ema_alpha_effect_mae,
            "diagnostic_direction": direction,
            "best_scale_variant": {
                "epoch": best_scale["epoch"],
                "weight_kind": best_scale["weight_kind"],
            },
            "best_alpha_effect_variant": {
                "epoch": best_alpha_effect["epoch"],
                "weight_kind": best_alpha_effect["weight_kind"],
            },
            "best_alpha_bandpower_variant": {
                "epoch": best_alpha_power["epoch"],
                "weight_kind": best_alpha_power["weight_kind"],
            },
            "best_spatial_correlation_variant": {
                "epoch": best_corr["epoch"],
                "weight_kind": best_corr["weight_kind"],
            },
        },
        "outputs": {
            "metrics_csv": str(METRIC_CSV),
            "real_reference_npz": str(REAL_REFERENCE_FILE),
            "real_reference_manifest": str(REAL_MANIFEST_FILE),
            "synthetic_npz": str(SYNTHETIC_FILE),
            "scale_figure": str(SCALE_FIGURE),
            "occipital_alpha_figure": str(ALPHA_O1_O2_FIGURE),
            "alpha_mae_figure": str(ALPHA_MAE_FIGURE),
            "bandpower_figure": str(BANDPOWER_FIGURE),
            "correlation_figure": str(CORRELATION_FIGURE),
            "overview_figure": str(OVERVIEW_FIGURE),
        },
        "runtime_minutes": total_seconds / 60.0,
        "checkpoint_immutability": immutable,
        "validation_eeg_accessed": False,
        "final_test_accessed": False,
        "training_updates_executed": False,
        "b0_frozen": False,
    }
    atomic_json_save(summary, SUMMARY_JSON)

    print(f"checkpoint immutability: {'PASS' if immutable else 'FAIL'}")
    print(f"summary JSON: {SUMMARY_JSON}")

    # ========================================================
    # Final
    # ========================================================
    print("\n" + "=" * 80)
    print("B0-V RAW VS EMA CHECKPOINT SWEEP COMPLETE")
    print("=" * 80)
    print(f"variants evaluated: {len(rows)}")
    print(f"runtime: {total_seconds / 60:.2f} min")
    print(f"diagnostic direction: {direction}")
    print("Validation EEG accessed: NO")
    print("Final Test accessed: NO")
    print("Training updates executed: NO")
    print("B0 frozen: NO")
    print("\nNEXT:")
    print("INTERPRET RAW VS EMA SWEEP BEFORE SELECTING OR MODIFYING B0-V")


if __name__ == "__main__":
    main()
