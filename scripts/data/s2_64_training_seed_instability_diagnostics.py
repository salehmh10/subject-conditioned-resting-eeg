from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from s2_23_check_real_train_dataset import NORMALIZATION_FILE
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_39_b0_epsilon_to_v_conversion_sanity import make_v_target, q_sample
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import inverse_normalize, welch_psd
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import compute_candidate_metrics
from s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep import (
    build_model_from_state,
    evaluate_eligibility,
    full_log_psd_mae,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

FROZEN_B1_GENERATOR = (
    REPO_ROOT / "outputs" / "frozen" / "scenario2" / "b1"
    / "b1_subject_state_raw_epoch010_v1" / "checkpoints"
    / "b1_subject_state_raw_epoch010_generator_v1.pt"
)

B1_SELECTION_MANIFEST = (
    REPO_ROOT / "outputs" / "frozen" / "scenario2" / "b1"
    / "b1_subject_state_raw_epoch010_v1" / "metrics"
    / "b1_frozen_selection_v1.json"
)

B1_SWEEP_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b1"
    / "raw_candidate_subject_aware_sweep_v2"
)

B1_REAL_REFERENCE_NPZ = (
    B1_SWEEP_ROOT / "generated" / "b1_subject_aware_real_train_reference_v2.npz"
)

B1_SYNTHETIC_REFERENCE_NPZ = (
    B1_SWEEP_ROOT / "generated" / "b1_raw_candidate_subject_aware_synthetic_v2.npz"
)

CONTROL_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "matched_lambda0_control_v1"
)

CONTROL_E5_CHECKPOINT = CONTROL_ROOT / "checkpoints" / "candidate_epoch_005_raw.pt"

S2_58_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "joint_b1_lambda0_lambda001_freegen_audit_v1"
)

CONTROL_E5_SYNTHETIC_NPZ = (
    S2_58_ROOT / "generated" / "lambda0_control_epoch_005_synthetic_v1.npz"
)

S2_63_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "control_e5_training_seed_replication_v1"
)

S2_63_SUMMARY = (
    S2_63_ROOT / "metrics" / "s2_63_control_e5_training_seed_replication_summary_v1.json"
)

REPLICA_A_CHECKPOINT = (
    S2_63_ROOT / "checkpoints" / "s2_63_replica_a_epoch005_raw.pt"
)

REPLICA_B_CHECKPOINT = (
    S2_63_ROOT / "checkpoints" / "s2_63_replica_b_epoch005_raw.pt"
)

REPLICA_A_SYNTHETIC = (
    S2_63_ROOT / "generated" / "s2_63_replica_a_seed_20261001_synthetic_v1.npz"
)

REPLICA_B_SYNTHETIC = (
    S2_63_ROOT / "generated" / "s2_63_replica_b_seed_20261001_synthetic_v1.npz"
)

RUN_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "training_seed_instability_diagnostics_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_FILE = METRIC_DIR / "s2_64_training_seed_instability_protocol_v1.json"
GUARDRAIL_CSV = METRIC_DIR / "s2_64_exact_guardrail_comparison_v1.csv"
SPECTRAL_CSV = METRIC_DIR / "s2_64_freegen_spectral_decomposition_v1.csv"
PARAMETER_CSV = METRIC_DIR / "s2_64_parameter_drift_v1.csv"
DENOISER_CSV = METRIC_DIR / "s2_64_fixed_probe_denoiser_diagnostics_v1.csv"
TRAINING_CSV = METRIC_DIR / "s2_64_training_trajectory_comparison_v1.csv"
SUMMARY_JSON = METRIC_DIR / "s2_64_training_seed_instability_summary_v1.json"

TRAINING_FIGURE = FIGURE_DIR / "s2_64_train_v_by_epoch_v1.png"
PSD_FIGURE = FIGURE_DIR / "s2_64_freegen_full_psd_by_training_seed_v1.png"
DRIFT_FIGURE = FIGURE_DIR / "s2_64_parameter_drift_from_b1_v1.png"
DENOISER_FIGURE = FIGURE_DIR / "s2_64_denoiser_output_distance_vs_timestep_v1.png"

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

EXPECTED_REAL_SHAPE = (
    SELECTED_SUBJECT_COUNT, 2, REAL_WINDOWS_PER_SUBJECT_STATE, N_CHANNELS, N_TIMES
)
EXPECTED_SYNTH_SHAPE = (
    SELECTED_SUBJECT_COUNT, 2, SYNTHETIC_WINDOWS_PER_SUBJECT_STATE, N_CHANNELS, N_TIMES
)

TIMESTEPS = 1000
COSINE_S = 0.008

PROBE_NOISE_SEED = 20261401
PROBE_TIMESTEPS = [25, 100, 300, 500, 800, 950, 999]
PROBE_BATCH_SIZE = 64

MODEL_NAMES = ["B1", "CANONICAL_E5", "REPLICA_A", "REPLICA_B"]
CHECKPOINT_MODEL_NAMES = ["CANONICAL_E5", "REPLICA_A", "REPLICA_B"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp, path)


def save_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Cannot save empty CSV: {path}")
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def load_exact_audit_data():
    real_package = np.load(B1_REAL_REFERENCE_NPZ, allow_pickle=False)
    b1_package = np.load(B1_SYNTHETIC_REFERENCE_NPZ, allow_pickle=False)
    canonical_package = np.load(CONTROL_E5_SYNTHETIC_NPZ, allow_pickle=False)
    replica_a_package = np.load(REPLICA_A_SYNTHETIC, allow_pickle=False)
    replica_b_package = np.load(REPLICA_B_SYNTHETIC, allow_pickle=False)

    real_z = real_package["real_z"].astype(np.float32, copy=False)
    subject_indices = (
        real_package["selected_subject_indices"].astype(np.int64, copy=False).tolist()
    )
    subject_ids = real_package["selected_subject_ids"].astype(str).tolist()

    packages = {
        "B1": (b1_package, "epoch_010_z"),
        "CANONICAL_E5": (canonical_package, "synth_z"),
        "REPLICA_A": (replica_a_package, "synth_z"),
        "REPLICA_B": (replica_b_package, "synth_z"),
    }

    synthetic = {}

    for model_name, (package, key) in packages.items():
        package_indices = (
            package["selected_subject_indices"].astype(np.int64, copy=False).tolist()
        )
        package_ids = package["selected_subject_ids"].astype(str).tolist()

        if package_indices != subject_indices:
            raise RuntimeError(f"Subject-index mismatch for {model_name}.")
        if package_ids != subject_ids:
            raise RuntimeError(f"Subject-ID mismatch for {model_name}.")

        values = package[key].astype(np.float32, copy=False)

        if values.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"Unexpected synthetic shape for {model_name}: {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"NaN/Inf in synthetic cache for {model_name}.")

        synthetic[model_name] = values

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(f"Unexpected Real shape: {real_z.shape}")

    return real_z, synthetic, subject_indices, subject_ids


def evaluate_bundle(*, real_z, synth_z, scaler):
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
    return {"generic": generic, "full_log_psd": full_psd}


def build_guardrail_rows(*, bundles):
    b1_bundle = bundles["B1"]
    rows = []
    details = {}

    for model_name in CHECKPOINT_MODEL_NAMES:
        eligibility = evaluate_eligibility(
            candidate=bundles[model_name],
            b1_baseline=b1_bundle,
        )

        criteria = eligibility["criteria"]
        failed_criteria = [
            criterion for criterion, passed in criteria.items() if not bool(passed)
        ]

        details[model_name] = {
            "eligible": bool(eligibility["eligible"]),
            "failed_criteria": failed_criteria,
            "criterion_count": len(criteria),
            "failed_count": len(failed_criteria),
            "full_detail": eligibility,
        }

        generic = bundles[model_name]["generic"]
        subject = generic["subject_identity"]

        rows.append(
            {
                "model": model_name,
                "eligible_vs_B1": bool(eligibility["eligible"]),
                "failed_count": len(failed_criteria),
                "failed_criteria": ";".join(failed_criteria),
                "full_psd_mae_db": float(
                    bundles[model_name]["full_log_psd"]["mean_mae_db"]
                ),
                "mean_bandpower_mae_db": float(generic["mean_bandpower_mae_db"]),
                "alpha_effect_mae_db": float(
                    generic["alpha_effect"]["channel_mae_db"]
                ),
                "scale_EO": float(generic["scale"]["EO"]["normalized_std_ratio"]),
                "scale_EC": float(generic["scale"]["EC"]["normalized_std_ratio"]),
                "correlation_r_EO": float(
                    generic["spatial"]["EO"]["correlation_r"]
                ),
                "correlation_r_EC": float(
                    generic["spatial"]["EC"]["correlation_r"]
                ),
                "alpha_dwpli2_r_EO": float(
                    generic["spatial"]["EO"]["alpha_dwpli2_r"]
                ),
                "alpha_dwpli2_r_EC": float(
                    generic["spatial"]["EC"]["alpha_dwpli2_r"]
                ),
                "subject_top1": float(
                    subject["subject_retrieval_top1_accuracy"]
                ),
                "subject_top5": float(
                    subject["subject_retrieval_top5_accuracy"]
                ),
                "subject_fingerprint_r_median": float(
                    subject["subject_fingerprint_pearson_r_median"]
                ),
                "subject_fingerprint_rmse_mean": float(
                    subject["subject_fingerprint_standardized_rmse_mean"]
                ),
                "subject_margin": float(
                    subject["subject_correct_vs_median_wrong_margin_mean"]
                ),
                "fixed_wrong_win_fraction": float(
                    subject["subject_correct_better_than_fixed_wrong_fraction"]
                ),
            }
        )

    return rows, details


def log_psd_tensor(eeg_z, scaler):
    original_shape = eeg_z.shape
    flat_z = eeg_z.reshape((-1, N_CHANNELS, N_TIMES))
    flat_uv = inverse_normalize(flat_z, scaler)
    frequencies, psd = welch_psd(flat_uv)

    mask = (frequencies >= 1.0) & (frequencies <= 40.0)
    selected_frequencies = frequencies[mask]
    tiny = np.finfo(np.float64).tiny

    log_psd = 10.0 * np.log10(np.maximum(psd[..., mask], tiny))
    output = log_psd.reshape(
        (
            original_shape[0],
            original_shape[1],
            original_shape[2],
            original_shape[3],
            len(selected_frequencies),
        )
    )
    return selected_frequencies, output


def pooled_state_median(log_psd):
    output = []
    for state_index in range(2):
        values = log_psd[:, state_index, ...].reshape(
            (-1, N_CHANNELS, log_psd.shape[-1])
        )
        output.append(np.median(values, axis=0))
    return np.stack(output, axis=0)


def pooled_state_iqr(log_psd):
    output = []
    for state_index in range(2):
        values = log_psd[:, state_index, ...].reshape(
            (-1, N_CHANNELS, log_psd.shape[-1])
        )
        q75 = np.percentile(values, 75, axis=0)
        q25 = np.percentile(values, 25, axis=0)
        output.append(q75 - q25)
    return np.stack(output, axis=0)


def spectral_decomposition_row(
    *, model_name, frequencies, real_log_psd, synth_log_psd
):
    real_median = pooled_state_median(real_log_psd)
    synth_median = pooled_state_median(synth_log_psd)
    signed_error = synth_median - real_median

    channel_offset = np.mean(signed_error, axis=-1)
    shape_error = signed_error - channel_offset[..., None]

    log_frequency = np.log10(frequencies.astype(np.float64))
    slopes = np.empty((2, N_CHANNELS), dtype=np.float64)

    for state_index in range(2):
        for channel_index in range(N_CHANNELS):
            slopes[state_index, channel_index] = np.polyfit(
                log_frequency,
                signed_error[state_index, channel_index],
                deg=1,
            )[0]

    real_iqr = pooled_state_iqr(real_log_psd)
    synth_iqr = pooled_state_iqr(synth_log_psd)

    return {
        "model": model_name,
        "global_signed_psd_bias_db": float(np.mean(signed_error)),
        "EO_signed_psd_bias_db": float(np.mean(signed_error[0])),
        "EC_signed_psd_bias_db": float(np.mean(signed_error[1])),
        "shape_mae_after_state_channel_offset_db": float(
            np.mean(np.abs(shape_error))
        ),
        "mean_tilt_db_per_decade": float(np.mean(slopes)),
        "mean_abs_tilt_db_per_decade": float(np.mean(np.abs(slopes))),
        "pooled_iqr_mae_db": float(np.mean(np.abs(synth_iqr - real_iqr))),
    }


def parameter_distance_summary(
    *, model_name, state_dict, b1_state, canonical_state
):
    b1_sq = canonical_sq = 0.0
    delta_b1_sq = delta_canonical_sq = 0.0
    dot_b1 = dot_canonical = 0.0
    model_sq = 0.0
    max_abs_delta_b1 = max_abs_delta_canonical = 0.0
    floating_tensors = 0
    floating_values = 0

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if not torch.is_floating_point(value):
            continue

        current = value.detach().cpu().double()
        b1 = b1_state[key].detach().cpu().double()
        canonical = canonical_state[key].detach().cpu().double()

        if current.shape != b1.shape or current.shape != canonical.shape:
            raise RuntimeError(f"State shape mismatch for key: {key}")

        current_flat = current.reshape(-1)
        b1_flat = b1.reshape(-1)
        canonical_flat = canonical.reshape(-1)

        delta_b1 = current_flat - b1_flat
        delta_canonical = current_flat - canonical_flat

        model_sq += float(torch.dot(current_flat, current_flat).item())
        b1_sq += float(torch.dot(b1_flat, b1_flat).item())
        canonical_sq += float(torch.dot(canonical_flat, canonical_flat).item())
        delta_b1_sq += float(torch.dot(delta_b1, delta_b1).item())
        delta_canonical_sq += float(torch.dot(delta_canonical, delta_canonical).item())
        dot_b1 += float(torch.dot(current_flat, b1_flat).item())
        dot_canonical += float(torch.dot(current_flat, canonical_flat).item())

        max_abs_delta_b1 = max(
            max_abs_delta_b1,
            float(torch.max(torch.abs(delta_b1)).item()),
        )
        max_abs_delta_canonical = max(
            max_abs_delta_canonical,
            float(torch.max(torch.abs(delta_canonical)).item()),
        )

        floating_tensors += 1
        floating_values += int(current.numel())

    eps = np.finfo(np.float64).eps

    return {
        "model": model_name,
        "floating_state_tensors": floating_tensors,
        "floating_values": floating_values,
        "relative_l2_distance_from_B1": float(
            math.sqrt(delta_b1_sq) / max(math.sqrt(b1_sq), eps)
        ),
        "relative_l2_distance_from_CANONICAL_E5": float(
            math.sqrt(delta_canonical_sq) / max(math.sqrt(canonical_sq), eps)
        ),
        "cosine_similarity_to_B1": float(
            dot_b1 / max(math.sqrt(model_sq) * math.sqrt(b1_sq), eps)
        ),
        "cosine_similarity_to_CANONICAL_E5": float(
            dot_canonical / max(math.sqrt(model_sq) * math.sqrt(canonical_sq), eps)
        ),
        "max_abs_value_delta_from_B1": float(max_abs_delta_b1),
        "max_abs_value_delta_from_CANONICAL_E5": float(
            max_abs_delta_canonical
        ),
    }


def build_probe_arrays(*, real_z, selected_subject_indices):
    x0 = real_z.reshape((-1, N_CHANNELS, N_TIMES))

    state = []
    subject = []

    for global_subject_index in selected_subject_indices:
        for state_index in range(2):
            for _ in range(REAL_WINDOWS_PER_SUBJECT_STATE):
                state.append(state_index)
                subject.append(int(global_subject_index))

    state_array = np.asarray(state, dtype=np.int64)
    subject_array = np.asarray(subject, dtype=np.int64)

    if x0.shape[0] != len(state_array):
        raise RuntimeError("Probe state accounting mismatch.")

    return x0, state_array, subject_array


def make_fixed_probe_noise(shape):
    generator = torch.Generator()
    generator.manual_seed(PROBE_NOISE_SEED)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32)
        .numpy()
        .astype(np.float32, copy=False)
    )


def run_fixed_denoiser_probe(
    *, models, real_z, selected_subject_indices, device
):
    x0_np, state_np, subject_np = build_probe_arrays(
        real_z=real_z,
        selected_subject_indices=selected_subject_indices,
    )

    epsilon_np = make_fixed_probe_noise(x0_np.shape)

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    alpha_bar = schedule_cpu["alpha_cumprod"].to(
        device=device,
        dtype=torch.float32,
    )

    rows = []
    n_examples = x0_np.shape[0]
    n_batches = math.ceil(n_examples / PROBE_BATCH_SIZE)

    for timestep_value in PROBE_TIMESTEPS:
        accumulators = {
            model_name: {
                "v_loss_sum": 0.0,
                "pred_energy_sum": 0.0,
                "presented_values": 0,
                "delta_to_b1_sq": 0.0,
                "delta_to_canonical_sq": 0.0,
            }
            for model_name in MODEL_NAMES
        }

        for batch_index in range(n_batches):
            start = batch_index * PROBE_BATCH_SIZE
            end = min(start + PROBE_BATCH_SIZE, n_examples)

            x0 = torch.from_numpy(x0_np[start:end]).to(
                device=device,
                dtype=torch.float32,
            )
            epsilon = torch.from_numpy(epsilon_np[start:end]).to(
                device=device,
                dtype=torch.float32,
            )
            state = torch.from_numpy(state_np[start:end]).to(
                device=device,
                dtype=torch.long,
            )
            subject = torch.from_numpy(subject_np[start:end]).to(
                device=device,
                dtype=torch.long,
            )

            batch_size = int(x0.shape[0])
            timesteps = torch.full(
                (batch_size,),
                int(timestep_value),
                device=device,
                dtype=torch.long,
            )

            xt = q_sample(
                x0,
                epsilon,
                timesteps,
                alpha_bar,
            )
            target_v = make_v_target(
                x0,
                epsilon,
                timesteps,
                alpha_bar,
            ).float()

            predictions = {}
            with torch.no_grad():
                for model_name, model in models.items():
                    predictions[model_name] = model(
                        xt,
                        timesteps,
                        state,
                        subject,
                    ).float()

            b1_pred = predictions["B1"]
            canonical_pred = predictions["CANONICAL_E5"]

            for model_name, pred_v in predictions.items():
                v_loss = torch.mean((pred_v - target_v) ** 2)
                delta_to_b1 = pred_v - b1_pred
                delta_to_canonical = pred_v - canonical_pred

                accumulator = accumulators[model_name]
                value_count = int(pred_v.numel())

                accumulator["v_loss_sum"] += float(v_loss.item()) * batch_size
                accumulator["pred_energy_sum"] += float(
                    torch.sum(pred_v * pred_v).item()
                )
                accumulator["delta_to_b1_sq"] += float(
                    torch.sum(delta_to_b1 * delta_to_b1).item()
                )
                accumulator["delta_to_canonical_sq"] += float(
                    torch.sum(delta_to_canonical * delta_to_canonical).item()
                )
                accumulator["presented_values"] += value_count

        for model_name in MODEL_NAMES:
            accumulator = accumulators[model_name]
            value_count = accumulator["presented_values"]

            rows.append(
                {
                    "model": model_name,
                    "timestep": timestep_value,
                    "alpha_bar": float(alpha_bar[timestep_value].item()),
                    "v_mse": float(accumulator["v_loss_sum"] / n_examples),
                    "pred_v_rms": float(
                        math.sqrt(accumulator["pred_energy_sum"] / value_count)
                    ),
                    "pred_v_rms_difference_from_B1": float(
                        math.sqrt(accumulator["delta_to_b1_sq"] / value_count)
                    ),
                    "pred_v_rms_difference_from_CANONICAL_E5": float(
                        math.sqrt(
                            accumulator["delta_to_canonical_sq"] / value_count
                        )
                    ),
                }
            )

    return rows


def extract_training_history(*, model_name, checkpoint):
    history = checkpoint["history"]
    rows = []

    for row in history:
        epoch = int(row["epoch"])
        if epoch > 5:
            continue

        rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_v_mse": float(row["train_v_mse"]),
                "successful_optimizer_steps": int(
                    row["successful_optimizer_steps"]
                ),
                "amp_skipped_steps": int(row["amp_skipped_steps"]),
            }
        )

    if len(rows) != 5:
        raise RuntimeError(
            f"Expected 5 history rows for {model_name}; got {len(rows)}."
        )

    return rows


def save_figures(
    *, training_rows, guardrail_rows, parameter_rows, denoiser_rows
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in CHECKPOINT_MODEL_NAMES:
        rows = [row for row in training_rows if row["model"] == model_name]
        ax.plot(
            [row["epoch"] for row in rows],
            [row["train_v_mse"] for row in rows],
            marker="o",
            label=model_name,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train v-MSE")
    ax.set_title("s2_64 — Training Trajectory Across Training Seeds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRAINING_FIGURE, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [row["model"] for row in guardrail_rows]
    values = [row["full_psd_mae_db"] for row in guardrail_rows]
    ax.bar(np.arange(len(names)), values)
    ax.set_xticks(np.arange(len(names)), names, rotation=15)
    ax.set_ylabel("Full 1–40 Hz PSD MAE (dB)")
    ax.set_title("s2_64 — Free-Generation PSD Across Training Seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PSD_FIGURE, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    parameter_plot_rows = [
        row for row in parameter_rows if row["model"] != "B1"
    ]
    names = [row["model"] for row in parameter_plot_rows]
    values = [row["relative_l2_distance_from_B1"] for row in parameter_plot_rows]
    ax.bar(np.arange(len(names)), values)
    ax.set_xticks(np.arange(len(names)), names, rotation=15)
    ax.set_ylabel("Relative L2 distance from frozen B1")
    ax.set_title("s2_64 — Parameter Drift from B1")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DRIFT_FIGURE, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in ("REPLICA_A", "REPLICA_B"):
        rows = [row for row in denoiser_rows if row["model"] == model_name]
        ax.plot(
            [row["timestep"] for row in rows],
            [
                row["pred_v_rms_difference_from_CANONICAL_E5"]
                for row in rows
            ],
            marker="o",
            label=model_name,
        )
    ax.set_xlabel("Diffusion timestep")
    ax.set_ylabel("RMS(pred_v − canonical Control-E5 pred_v)")
    ax.set_title("s2_64 — Denoiser Function Divergence Across Training Seeds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DENOISER_FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — s2_64 TRAINING-SEED INSTABILITY DIAGNOSTICS")
    print("=" * 80)

    print("\nScientific role:")
    print("  Diagnose why the 5-epoch lambda=0 continuation recipe")
    print("  replicated for one independent training seed but failed for another.")

    print("\nThis stage executes:")
    print("  NO training")
    print("  NO new free generation")
    print("  NO threshold changes")
    print("  NO model freeze")
    print("  NO Validation EEG")
    print("  NO Final Test EEG")

    for directory in (METRIC_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT B1 / CANONICAL E5 / REPLICA A / REPLICA B")
    print("=" * 80)

    required_files = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_REAL_REFERENCE_NPZ,
        B1_SYNTHETIC_REFERENCE_NPZ,
        CONTROL_E5_CHECKPOINT,
        CONTROL_E5_SYNTHETIC_NPZ,
        S2_63_SUMMARY,
        REPLICA_A_CHECKPOINT,
        REPLICA_B_CHECKPOINT,
        REPLICA_A_SYNTHETIC,
        REPLICA_B_SYNTHETIC,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    with B1_SELECTION_MANIFEST.open("r", encoding="utf-8") as f:
        b1_selection = json.load(f)

    with S2_63_SUMMARY.open("r", encoding="utf-8") as f:
        s2_63_summary = json.load(f)

    if b1_selection["status"] != "B1_SELECTION_FROZEN":
        raise RuntimeError("B1 is not frozen.")

    if (
        s2_63_summary["status"]
        != "S2_63_CONTROL_E5_TRAINING_SEED_REPLICATION_COMPLETE"
    ):
        raise RuntimeError("s2_63 is not complete.")

    if s2_63_summary["training_recipe_independently_replicated"] is not False:
        raise RuntimeError(
            "s2_64 expects the observed s2_63 replication failure."
        )

    frozen_b1_hash_before = sha256_file(FROZEN_B1_GENERATOR)
    canonical_hash_before = sha256_file(CONTROL_E5_CHECKPOINT)
    replica_a_hash_before = sha256_file(REPLICA_A_CHECKPOINT)
    replica_b_hash_before = sha256_file(REPLICA_B_CHECKPOINT)

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )
    canonical = torch.load(
        CONTROL_E5_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    replica_a = torch.load(
        REPLICA_A_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    replica_b = torch.load(
        REPLICA_B_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        replica_a["status"]
        != "S2_63_CONTROL_E5_TRAINING_SEED_REPLICA_CHECKPOINT"
    ):
        raise RuntimeError("Unexpected Replica A checkpoint status.")

    if (
        replica_b["status"]
        != "S2_63_CONTROL_E5_TRAINING_SEED_REPLICA_CHECKPOINT"
    ):
        raise RuntimeError("Unexpected Replica B checkpoint status.")

    print(f"Frozen B1 SHA256: {frozen_b1_hash_before}")
    print(f"Canonical E5 SHA256: {canonical_hash_before}")
    print(f"Replica A SHA256: {replica_a_hash_before}")
    print(f"Replica B SHA256: {replica_b_hash_before}")
    print("source audit: PASS")

    print("\n" + "=" * 80)
    print("STEP 2 — FREEZE s2_64 DIAGNOSTIC PROTOCOL")
    print("=" * 80)

    protocol = {
        "status": "S2_64_TRAINING_SEED_INSTABILITY_DIAGNOSTIC_PROTOCOL_FROZEN",
        "scientific_role": (
            "Post-replication diagnostic only; explains the already-observed "
            "training-seed instability without generating new outcomes."
        ),
        "models": MODEL_NAMES,
        "inputs": (
            "existing checkpoints and existing seed-20261001 synthetic caches only"
        ),
        "diagnostics": [
            "exact existing B2 guardrail failures and pass/fail breadth",
            "free-generation signed PSD bias / spectral shape / tilt / IQR",
            "parameter-space relative L2 drift from B1 and canonical E5",
            "five-epoch Train-v trajectories and AMP skip accounting",
            "fixed Real-Train denoiser probe at predeclared timesteps",
            "RMS denoiser-output divergence from B1 and canonical E5",
        ],
        "fixed_probe": {
            "noise_seed": PROBE_NOISE_SEED,
            "timesteps": PROBE_TIMESTEPS,
            "examples": (
                SELECTED_SUBJECT_COUNT
                * 2
                * REAL_WINDOWS_PER_SUBJECT_STATE
            ),
        },
        "interpretation_limit": (
            "This stage can characterize seed instability but does not "
            "authorize a new checkpoint-selection rule or new hyperparameter."
        ),
        "training_updates_executed": False,
        "new_sampling_executed": False,
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "thresholds_changed": False,
        "formal_model_freeze_decision_made": False,
    }

    if PROTOCOL_FILE.is_file():
        with PROTOCOL_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != protocol:
            raise RuntimeError("Conflicting s2_64 protocol already exists.")
        print("identical s2_64 protocol already exists.")
    else:
        atomic_json_save(protocol, PROTOCOL_FILE)
        print(f"protocol: {PROTOCOL_FILE}")

    print("\n" + "=" * 80)
    print("STEP 3 — EXACT GUARDRAIL FAILURE DECOMPOSITION")
    print("=" * 80)

    real_z, synthetic, selected_subject_indices, _ = load_exact_audit_data()

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked.")

    bundles = {}
    for model_name in MODEL_NAMES:
        bundles[model_name] = evaluate_bundle(
            real_z=real_z,
            synth_z=synthetic[model_name],
            scaler=scaler,
        )

    guardrail_rows, guardrail_detail = build_guardrail_rows(
        bundles=bundles
    )

    save_csv_rows(guardrail_rows, GUARDRAIL_CSV)

    for row in guardrail_rows:
        print(f"\n{row['model']}:")
        print(f"  eligible vs B1: {row['eligible_vs_B1']}")
        print(f"  failed criteria: {row['failed_criteria'] or 'NONE'}")
        print(f"  Full PSD: {row['full_psd_mae_db']:.6f} dB")
        print(f"  Bandpower: {row['mean_bandpower_mae_db']:.6f} dB")
        print(f"  Alpha-effect: {row['alpha_effect_mae_db']:.6f} dB")
        print(f"  Top1: {row['subject_top1']:.6f}")

    print("\n" + "=" * 80)
    print("STEP 4 — FREE-GENERATION SPECTRAL ERROR DECOMPOSITION")
    print("=" * 80)

    frequencies, real_log_psd = log_psd_tensor(real_z, scaler)

    spectral_rows = []
    for model_name in MODEL_NAMES:
        synth_frequencies, synth_log_psd = log_psd_tensor(
            synthetic[model_name],
            scaler,
        )

        if not np.array_equal(frequencies, synth_frequencies):
            raise RuntimeError(
                f"Frequency-grid mismatch for {model_name}."
            )

        row = spectral_decomposition_row(
            model_name=model_name,
            frequencies=frequencies,
            real_log_psd=real_log_psd,
            synth_log_psd=synth_log_psd,
        )

        spectral_rows.append(row)

        print(
            f"{model_name}: "
            f"bias={row['global_signed_psd_bias_db']:+.4f} dB "
            f"| shape={row['shape_mae_after_state_channel_offset_db']:.4f} dB "
            f"| |tilt|={row['mean_abs_tilt_db_per_decade']:.4f} dB/dec "
            f"| IQR={row['pooled_iqr_mae_db']:.4f} dB"
        )

    save_csv_rows(spectral_rows, SPECTRAL_CSV)

    print("\n" + "=" * 80)
    print("STEP 5 — PARAMETER-SPACE DRIFT DIAGNOSTICS")
    print("=" * 80)

    state_dicts = {
        "B1": frozen_b1["model_state_dict"],
        "CANONICAL_E5": canonical["model_state_dict"],
        "REPLICA_A": replica_a["model_state_dict"],
        "REPLICA_B": replica_b["model_state_dict"],
    }

    b1_state = state_dicts["B1"]
    canonical_state = state_dicts["CANONICAL_E5"]

    parameter_rows = []
    for model_name in MODEL_NAMES:
        row = parameter_distance_summary(
            model_name=model_name,
            state_dict=state_dicts[model_name],
            b1_state=b1_state,
            canonical_state=canonical_state,
        )

        parameter_rows.append(row)

        print(
            f"{model_name}: "
            f"rel-L2 from B1={row['relative_l2_distance_from_B1']:.6e} "
            f"| rel-L2 from canonical="
            f"{row['relative_l2_distance_from_CANONICAL_E5']:.6e}"
        )

    save_csv_rows(parameter_rows, PARAMETER_CSV)

    print("\n" + "=" * 80)
    print("STEP 6 — TRAINING TRAJECTORY COMPARISON")
    print("=" * 80)

    training_rows = []

    for model_name, checkpoint in (
        ("CANONICAL_E5", canonical),
        ("REPLICA_A", replica_a),
        ("REPLICA_B", replica_b),
    ):
        rows = extract_training_history(
            model_name=model_name,
            checkpoint=checkpoint,
        )

        training_rows.extend(rows)
        final_row = rows[-1]

        print(
            f"{model_name}: "
            f"E5 Train-v={final_row['train_v_mse']:.6f} "
            f"| total E1-E5 skips="
            f"{sum(row['amp_skipped_steps'] for row in rows)}"
        )

    save_csv_rows(training_rows, TRAINING_CSV)

    print("\n" + "=" * 80)
    print("STEP 7 — FIXED REAL-TRAIN DENOISER FUNCTION PROBE")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the denoiser probe.")

    device = torch.device("cuda")

    models = {
        model_name: build_model_from_state(
            frozen_b1=frozen_b1,
            state_dict=state_dicts[model_name],
            device=device,
        )
        for model_name in MODEL_NAMES
    }

    for model in models.values():
        model.eval()

    denoiser_rows = run_fixed_denoiser_probe(
        models=models,
        real_z=real_z,
        selected_subject_indices=selected_subject_indices,
        device=device,
    )

    save_csv_rows(denoiser_rows, DENOISER_CSV)

    for timestep_value in PROBE_TIMESTEPS:
        row_a = next(
            row
            for row in denoiser_rows
            if row["model"] == "REPLICA_A"
            and row["timestep"] == timestep_value
        )
        row_b = next(
            row
            for row in denoiser_rows
            if row["model"] == "REPLICA_B"
            and row["timestep"] == timestep_value
        )

        print(
            f"t={timestep_value}: "
            f"A-canonical RMS="
            f"{row_a['pred_v_rms_difference_from_CANONICAL_E5']:.6f} "
            f"| B-canonical RMS="
            f"{row_b['pred_v_rms_difference_from_CANONICAL_E5']:.6f}"
        )

    print("\n" + "=" * 80)
    print("STEP 8 — SAVE DIAGNOSTIC FIGURES")
    print("=" * 80)

    save_figures(
        training_rows=training_rows,
        guardrail_rows=guardrail_rows,
        parameter_rows=parameter_rows,
        denoiser_rows=denoiser_rows,
    )

    for path in (
        TRAINING_FIGURE,
        PSD_FIGURE,
        DRIFT_FIGURE,
        DENOISER_FIGURE,
    ):
        print(path)

    print("\n" + "=" * 80)
    print("STEP 9 — VERIFY IMMUTABILITY + SAVE FINAL SUMMARY")
    print("=" * 80)

    if sha256_file(FROZEN_B1_GENERATOR) != frozen_b1_hash_before:
        raise RuntimeError("Frozen B1 changed during s2_64.")
    if sha256_file(CONTROL_E5_CHECKPOINT) != canonical_hash_before:
        raise RuntimeError("Canonical E5 changed during s2_64.")
    if sha256_file(REPLICA_A_CHECKPOINT) != replica_a_hash_before:
        raise RuntimeError("Replica A changed during s2_64.")
    if sha256_file(REPLICA_B_CHECKPOINT) != replica_b_hash_before:
        raise RuntimeError("Replica B changed during s2_64.")

    summary = {
        "status": "S2_64_TRAINING_SEED_INSTABILITY_DIAGNOSTICS_COMPLETE",
        "scientific_role": (
            "Diagnostic characterization of the s2_63 training-seed "
            "replication failure; no new outcome generation."
        ),
        "guardrail_detail": guardrail_detail,
        "spectral_decomposition": {
            row["model"]: {
                key: value
                for key, value in row.items()
                if key != "model"
            }
            for row in spectral_rows
        },
        "parameter_drift": {
            row["model"]: {
                key: value
                for key, value in row.items()
                if key != "model"
            }
            for row in parameter_rows
        },
        "training_final": {
            model_name: next(
                row
                for row in reversed(training_rows)
                if row["model"] == model_name
            )
            for model_name in CHECKPOINT_MODEL_NAMES
        },
        "fixed_probe_timesteps": PROBE_TIMESTEPS,
        "artifacts": {
            "protocol": str(PROTOCOL_FILE),
            "guardrail_csv": str(GUARDRAIL_CSV),
            "spectral_csv": str(SPECTRAL_CSV),
            "parameter_csv": str(PARAMETER_CSV),
            "denoiser_csv": str(DENOISER_CSV),
            "training_csv": str(TRAINING_CSV),
            "training_figure": str(TRAINING_FIGURE),
            "psd_figure": str(PSD_FIGURE),
            "drift_figure": str(DRIFT_FIGURE),
            "denoiser_figure": str(DENOISER_FIGURE),
        },
        "governance": {
            "Train_reference_used": True,
            "new_training_executed": False,
            "new_sampling_executed": False,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "thresholds_changed": False,
            "formal_model_freeze_decision_made": False,
            "all_source_checkpoints_unchanged": True,
        },
        "next": (
            "Interpret whether the replication failure is broad versus "
            "single-guard/near-threshold, and whether it is accompanied by "
            "distinct parameter/function drift. Only then decide whether "
            "a new versioned robust checkpoint-selection protocol is justified "
            "or whether Control E5 should remain unfrozen."
        ),
    }

    atomic_json_save(summary, SUMMARY_JSON)

    print("source checkpoint immutability: PASS")
    print(f"summary JSON: {SUMMARY_JSON}")

    print("\n" + "=" * 80)
    print("s2_64 TRAINING-SEED INSTABILITY DIAGNOSTICS COMPLETE")
    print("=" * 80)

    print("\nNo training executed: YES")
    print("No new sampling executed: YES")
    print("Validation EEG accessed: NO")
    print("Final Test accessed: NO")
    print("Formal model freeze decision made: NO")

    print("\nNEXT:")
    print(
        "INTERPRET THE SEED-INSTABILITY MECHANISM BEFORE CHANGING "
        "ANY TRAINING OR CHECKPOINT-SELECTION RULE."
    )


if __name__ == "__main__":
    main()
