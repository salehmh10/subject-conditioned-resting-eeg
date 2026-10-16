from __future__ import annotations

import csv
import hashlib
import json
import math
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
from scipy.signal import welch

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)
from s2_24_full_train_loader_sanity import CachedRealTrainWindowDataset
from s2_32_b0_unet_shape_test import B0StateOnlyUNet1D
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

B0V_RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "full_finetune_v1"
)

BEST_CHECKPOINT = B0V_RUN_DIR / "checkpoints" / "best_ema.pt"
RESULT_JSON = B0V_RUN_DIR / "metrics" / "b0_v_full_finetune_result_v1.json"

AUDIT_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "final_fidelity_audit_v1"
)

GENERATED_DIR = AUDIT_DIR / "generated"
LOG_DIR = AUDIT_DIR / "logs"
METRIC_DIR = AUDIT_DIR / "metrics"
FIGURE_DIR = AUDIT_DIR / "figures"

SYNTHETIC_FILE = GENERATED_DIR / "b0_v_final_audit_synthetic_v1.npz"
REAL_MANIFEST_FILE = LOG_DIR / "b0_v_real_train_reference_manifest_v1.csv"
BANDPOWER_FILE = METRIC_DIR / "b0_v_bandpower_fidelity_v1.csv"
AMPLITUDE_FILE = METRIC_DIR / "b0_v_amplitude_fidelity_v1.csv"
SUMMARY_FILE = METRIC_DIR / "b0_v_final_fidelity_audit_v1.json"
MATRICES_FILE = METRIC_DIR / "b0_v_spatial_connectivity_matrices_v1.npz"

WAVEFORM_FIGURE = FIGURE_DIR / "b0_v_real_vs_synthetic_waveforms_v1.png"
AMPLITUDE_FIGURE = FIGURE_DIR / "b0_v_real_vs_synthetic_amplitudes_v1.png"
PSD_FIGURE = FIGURE_DIR / "b0_v_real_vs_synthetic_occipital_psd_v1.png"
BANDPOWER_FIGURE = FIGURE_DIR / "b0_v_bandpower_error_heatmap_v1.png"
ALPHA_EFFECT_FIGURE = FIGURE_DIR / "b0_v_alpha_state_effect_real_vs_synthetic_v1.png"
CORRELATION_FIGURE = FIGURE_DIR / "b0_v_mean_channel_correlation_v1.png"
WPLI_FIGURE = FIGURE_DIR / "b0_v_alpha_dwpli2_v1.png"
STD_RATIO_FIGURE = FIGURE_DIR / "b0_v_channel_std_ratio_v1.png"


# ============================================================
# Frozen audit configuration
# ============================================================

SEED = 20260912
REAL_REFERENCE_SEED = 20260913
EO_SAMPLING_SEED = 20260914
EC_SAMPLING_SEED = 20260915

EXPECTED_BEST_EPOCH = 40
EXPECTED_SOURCE_EPOCH = 125
EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

N_PER_STATE = 256
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



def relative_frobenius(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = float(np.linalg.norm(candidate - reference, ord="fro"))
    denominator = float(np.linalg.norm(reference, ord="fro"))
    return numerator / max(denominator, np.finfo(float).eps)



def upper_triangle_correlation(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.triu(np.ones_like(a, dtype=bool), k=1)
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# Stable v-prediction DDPM sampler
# ============================================================


def build_reverse_schedule(
    schedule_cpu: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    betas = schedule_cpu["betas"].to(device=device, dtype=torch.float32)
    alphas = schedule_cpu["alphas"].to(device=device, dtype=torch.float32)
    alpha_bar = schedule_cpu["alpha_cumprod"].to(
        device=device, dtype=torch.float32
    )
    alpha_bar_prev = torch.cat(
        [
            torch.ones(1, device=device, dtype=torch.float32),
            alpha_bar[:-1],
        ],
        dim=0,
    )

    denominator = 1.0 - alpha_bar
    coef_x0 = betas * torch.sqrt(alpha_bar_prev) / denominator
    coef_xt = torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / denominator
    posterior_variance = betas * (1.0 - alpha_bar_prev) / denominator
    posterior_variance = torch.clamp(posterior_variance, min=0.0)

    return {
        "alpha_bar": alpha_bar,
        "coef_x0": coef_x0,
        "coef_xt": coef_xt,
        "posterior_variance": posterior_variance,
    }


@torch.inference_mode()
def sample_v_batches(
    *,
    model: torch.nn.Module,
    eye_state_value: int,
    n_samples: int,
    batch_size: int,
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> np.ndarray:
    if eye_state_value not in (0, 1):
        raise ValueError("eye_state_value must be 0 or 1")

    if n_samples % batch_size != 0:
        raise ValueError("n_samples must be divisible by batch_size for this audit")

    model.eval()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    alpha_bar = reverse_schedule["alpha_bar"]
    coef_x0 = reverse_schedule["coef_x0"]
    coef_xt = reverse_schedule["coef_xt"]
    posterior_variance = reverse_schedule["posterior_variance"]

    batches = []
    start = time.perf_counter()
    n_batches = n_samples // batch_size

    for batch_index in range(n_batches):
        shape = (batch_size, N_CHANNELS, N_TIMES)
        x = torch.randn(
            shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        state = torch.full(
            (batch_size,),
            eye_state_value,
            device=device,
            dtype=torch.long,
        )

        for t in reversed(range(TIMESTEPS)):
            timestep = torch.full(
                (batch_size,),
                t,
                device=device,
                dtype=torch.long,
            )

            # Intentionally FP32 sampling: no AMP in the sampler.
            v_pred = model(x, timestep, state)
            if not torch.isfinite(v_pred).all():
                raise RuntimeError(
                    f"Nonfinite v prediction at state={eye_state_value}, "
                    f"batch={batch_index}, timestep={t}"
                )

            sqrt_ab = torch.sqrt(alpha_bar[t])
            sqrt_omab = torch.sqrt(1.0 - alpha_bar[t])
            x0_pred = sqrt_ab * x - sqrt_omab * v_pred

            model_mean = coef_x0[t] * x0_pred + coef_xt[t] * x

            if t > 0:
                noise = torch.randn(
                    shape,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                x = model_mean + torch.sqrt(posterior_variance[t]) * noise
            else:
                x = model_mean

            if not torch.isfinite(x).all():
                raise RuntimeError(
                    f"Nonfinite reverse state at state={eye_state_value}, "
                    f"batch={batch_index}, timestep={t}"
                )

        batches.append(x.detach().cpu().numpy().astype(np.float32, copy=False))
        elapsed = time.perf_counter() - start
        print(
            f"state={eye_state_value} | batch={batch_index + 1}/{n_batches} "
            f"| generated={(batch_index + 1) * batch_size}/{n_samples} "
            f"| final_std={x.std(unbiased=False).item():.4f} "
            f"| elapsed={elapsed / 60:.1f}m"
        )

    return np.concatenate(batches, axis=0)


# ============================================================
# Deterministic subject-balanced Real Train reference
# ============================================================


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

    if array.shape != (len(selected_records), N_CHANNELS, N_TIMES):
        raise RuntimeError(f"Unexpected Real reference shape: {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("Real reference contains NaN/Inf")
    return array



def save_real_reference_manifest(eo_records, ec_records) -> None:
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
# Spectral metrics
# ============================================================


def welch_psd(eeg_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, psd = welch(
        eeg_uv,
        fs=SFREQ,
        window="hann",
        nperseg=256,
        noverlap=128,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    return frequencies, psd



def bandpower_db_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {}
    tiny = np.finfo(np.float64).tiny
    for band_name, (low, high) in BANDS.items():
        # Match previous project convention: inclusive edges.
        mask = (frequencies >= low) & (frequencies <= high)
        power = np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)
        output[band_name] = 10.0 * np.log10(np.maximum(power, tiny))
    return output



def make_bandpower_rows(
    real_by_state: dict[str, dict[str, np.ndarray]],
    synthetic_by_state: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    rows = []
    for state in ("EO", "EC"):
        for band in BANDS:
            real_values = real_by_state[state][band]
            synth_values = synthetic_by_state[state][band]
            for channel_index, channel in enumerate(CHANNEL_NAMES):
                real_median = float(np.median(real_values[:, channel_index]))
                synth_median = float(np.median(synth_values[:, channel_index]))
                rows.append(
                    {
                        "eye_state": state,
                        "band": band,
                        "channel": channel,
                        "real_median_db": real_median,
                        "synthetic_median_db": synth_median,
                        "synthetic_minus_real_db": synth_median - real_median,
                        "real_iqr_db": float(
                            np.percentile(real_values[:, channel_index], 75)
                            - np.percentile(real_values[:, channel_index], 25)
                        ),
                        "synthetic_iqr_db": float(
                            np.percentile(synth_values[:, channel_index], 75)
                            - np.percentile(synth_values[:, channel_index], 25)
                        ),
                    }
                )
    return rows


# ============================================================
# Spatial covariance / correlation
# ============================================================


def mean_covariance_and_correlation(eeg_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # demean per window/channel over time
    centered = eeg_z.astype(np.float64) - eeg_z.astype(np.float64).mean(
        axis=-1, keepdims=True
    )
    covariance = np.einsum("ect,edt->ecd", centered, centered) / (N_TIMES - 1)

    variances = np.diagonal(covariance, axis1=1, axis2=2)
    std = np.sqrt(np.maximum(variances, np.finfo(float).eps))
    denominator = std[:, :, None] * std[:, None, :]
    correlation = covariance / denominator

    return covariance.mean(axis=0), correlation.mean(axis=0)


# ============================================================
# Alpha-band debiased squared wPLI across windows
# ============================================================


def alpha_dwpli2_debiased(eeg_uv: np.ndarray) -> np.ndarray:
    """
    Debiased squared weighted phase-lag index (wPLI2-debiased).

    Observations are 4-second windows. For each alpha frequency bin, the
    imaginary cross-spectrum is accumulated across windows, the debiased
    squared wPLI estimator is computed, then alpha bins are averaged.

    This is a state-level connectivity summary, not a per-window metric.
    """
    x = eeg_uv.astype(np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    window = np.hanning(N_TIMES).astype(np.float64)
    spectrum = np.fft.rfft(x * window[None, None, :], axis=-1)
    frequencies = np.fft.rfftfreq(N_TIMES, d=1.0 / SFREQ)
    alpha_bins = np.where((frequencies >= 8.0) & (frequencies <= 13.0))[0]

    matrices = []
    eps = np.finfo(np.float64).eps

    for freq_index in alpha_bins:
        z = spectrum[:, :, freq_index]
        cross = z[:, :, None] * np.conj(z[:, None, :])
        imag = np.imag(cross)

        sum_im = imag.sum(axis=0)
        sum_abs = np.abs(imag).sum(axis=0)
        sum_sq = np.square(imag).sum(axis=0)

        numerator = np.square(sum_im) - sum_sq
        denominator = np.square(sum_abs) - sum_sq

        matrix = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float64)
        valid = np.abs(denominator) > eps
        matrix[valid] = numerator[valid] / denominator[valid]
        np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)

    result = np.mean(np.stack(matrices, axis=0), axis=0)
    return result


# ============================================================
# Amplitude metrics
# ============================================================


def amplitude_summary_rows(
    real_z: dict[str, np.ndarray],
    synth_z: dict[str, np.ndarray],
    real_uv: dict[str, np.ndarray],
    synth_uv: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for state in ("EO", "EC"):
        for representation, real_values, synth_values in (
            ("normalized", real_z[state], synth_z[state]),
            ("uV", real_uv[state], synth_uv[state]),
        ):
            for channel_index, channel in enumerate(CHANNEL_NAMES):
                r = real_values[:, channel_index, :].reshape(-1).astype(np.float64)
                s = synth_values[:, channel_index, :].reshape(-1).astype(np.float64)
                real_std = float(np.std(r))
                synth_std = float(np.std(s))
                rows.append(
                    {
                        "eye_state": state,
                        "representation": representation,
                        "channel": channel,
                        "real_mean": float(np.mean(r)),
                        "synthetic_mean": float(np.mean(s)),
                        "real_std": real_std,
                        "synthetic_std": synth_std,
                        "synthetic_to_real_std_ratio": synth_std
                        / max(real_std, np.finfo(float).eps),
                        "real_p01": float(np.percentile(r, 1)),
                        "synthetic_p01": float(np.percentile(s, 1)),
                        "real_p50": float(np.percentile(r, 50)),
                        "synthetic_p50": float(np.percentile(s, 50)),
                        "real_p99": float(np.percentile(r, 99)),
                        "synthetic_p99": float(np.percentile(s, 99)),
                    }
                )
    return rows


# ============================================================
# Figure helpers
# ============================================================


def save_waveform_figure(real_uv, synthetic_uv) -> None:
    time_axis = np.arange(N_TIMES) / SFREQ
    channels = ["Fz", "Cz", "Pz", "O1", "O2"]

    fig, axes = plt.subplots(len(channels), 2, figsize=(14, 11), sharex=True)
    for row, channel in enumerate(channels):
        idx = CHANNEL_INDEX[channel]
        for col, state in enumerate(("EO", "EC")):
            ax = axes[row, col]
            ax.plot(time_axis, real_uv[state][0, idx], label="Real")
            ax.plot(time_axis, synthetic_uv[state][0, idx], label="Synthetic")
            ax.set_ylabel(f"{channel}\nµV")
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(state)
            if row == len(channels) - 1:
                ax.set_xlabel("Time (s)")
    axes[0, 0].legend()
    fig.suptitle("B0-v Final Audit — Real Train vs Synthetic Waveforms")
    fig.tight_layout()
    fig.savefig(WAVEFORM_FIGURE, dpi=160)
    plt.close(fig)



def save_amplitude_figure(real_uv, synthetic_uv) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, state in zip(axes, ("EO", "EC")):
        ax.hist(
            real_uv[state].reshape(-1),
            bins=180,
            density=True,
            histtype="step",
            label="Real Train",
        )
        ax.hist(
            synthetic_uv[state].reshape(-1),
            bins=180,
            density=True,
            histtype="step",
            label="Synthetic B0-v",
        )
        ax.set_ylabel(f"{state}\nDensity")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("EEG amplitude (µV)")
    fig.suptitle("B0-v Final Audit — Amplitude Distributions")
    fig.tight_layout()
    fig.savefig(AMPLITUDE_FIGURE, dpi=160)
    plt.close(fig)



def save_psd_figure(freqs, real_psd, synth_psd) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for row, channel in enumerate(("O1", "O2")):
        idx = CHANNEL_INDEX[channel]
        for col, state in enumerate(("EO", "EC")):
            ax = axes[row, col]
            real_curve = np.median(real_psd[state][:, idx, :], axis=0)
            synth_curve = np.median(synth_psd[state][:, idx, :], axis=0)
            ax.semilogy(freqs, real_curve, label="Real Train")
            ax.semilogy(freqs, synth_curve, label="Synthetic B0-v")
            ax.axvspan(8, 13, alpha=0.12)
            ax.set_xlim(1, 40)
            ax.set_ylabel(f"{channel}\nPSD")
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(state)
            if row == 1:
                ax.set_xlabel("Frequency (Hz)")
    axes[0, 0].legend()
    fig.suptitle("B0-v Final Audit — Occipital PSD")
    fig.tight_layout()
    fig.savefig(PSD_FIGURE, dpi=160)
    plt.close(fig)



def save_alpha_effect_figure(real_band, synth_band) -> None:
    real_alpha = real_band["alpha"]
    synth_alpha = synth_band["alpha"]

    real_delta = np.median(real_alpha["EC"], axis=0) - np.median(
        real_alpha["EO"], axis=0
    )
    synth_delta = np.median(synth_alpha["EC"], axis=0) - np.median(
        synth_alpha["EO"], axis=0
    )

    x = np.arange(N_CHANNELS)
    width = 0.38
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, real_delta, width=width, label="Real Train")
    ax.bar(x + width / 2, synth_delta, width=width, label="Synthetic B0-v")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Median EC − EO alpha power (dB)")
    ax.set_title("B0-v Final Audit — Eye-State Alpha Effect")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ALPHA_EFFECT_FIGURE, dpi=160)
    plt.close(fig)



def save_bandpower_error_figure(band_rows) -> None:
    # Mean absolute Synthetic-Real median error across EO/EC for each band/channel.
    matrix = np.zeros((len(BANDS), N_CHANNELS), dtype=np.float64)
    band_names = list(BANDS)
    for band_index, band in enumerate(band_names):
        for channel_index, channel in enumerate(CHANNEL_NAMES):
            values = [
                abs(row["synthetic_minus_real_db"])
                for row in band_rows
                if row["band"] == band and row["channel"] == channel
            ]
            matrix[band_index, channel_index] = np.mean(values)

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)
    image = ax.imshow(matrix, aspect="auto")
    ax.set_yticks(np.arange(len(band_names)))
    ax.set_yticklabels(band_names)
    ax.set_xticks(np.arange(N_CHANNELS))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right")
    ax.set_title("B0-v Final Audit — Median Bandpower Absolute Error (dB)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Band")
    fig.colorbar(image, ax=ax, label="Absolute error (dB)")
    fig.tight_layout()
    fig.savefig(BANDPOWER_FIGURE, dpi=160)
    plt.close(fig)



def save_matrix_figure(real_matrices, synth_matrices, output_file, title) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    entries = [
        ("Real EO", real_matrices["EO"]),
        ("Synthetic EO", synth_matrices["EO"]),
        ("Real EC", real_matrices["EC"]),
        ("Synthetic EC", synth_matrices["EC"]),
    ]
    image = None
    for ax, (label, matrix) in zip(axes.flat, entries):
        image = ax.imshow(matrix, aspect="equal")
        ax.set_title(label)
        ax.set_xticks(np.arange(N_CHANNELS))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(N_CHANNELS))
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)
    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)



def save_std_ratio_figure(amplitude_rows) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    x = np.arange(N_CHANNELS)
    width = 0.38
    for offset, state in ((-width / 2, "EO"), (width / 2, "EC")):
        ratios = [
            row["synthetic_to_real_std_ratio"]
            for row in amplitude_rows
            if row["eye_state"] == state and row["representation"] == "normalized"
        ]
        ax.bar(x + offset, ratios, width=width, label=state)
    ax.axhline(1.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Synthetic / Real normalized std")
    ax.set_title("B0-v Final Audit — Channel-wise Scale Ratio")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(STD_RATIO_FIGURE, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — B0-V FINAL SAMPLING + REAL-TRAIN FIDELITY AUDIT")
    print("=" * 80)

    print("\nCurrent project stage:")
    print("  B0 epsilon pretraining: COMPLETE")
    print("  epsilon sampling failure diagnosis: COMPLETE")
    print("  B0-v conversion + pilot: COMPLETE")
    print("  B0-v 40-epoch full fine-tuning: COMPLETE")
    print("  B0-v final fidelity audit: NOW")

    print("\nAudit design:")
    print(f"  Synthetic: {N_PER_STATE} EO + {N_PER_STATE} EC")
    print(f"  Real Train reference: {N_PER_STATE} EO + {N_PER_STATE} EC")
    print("  Real reference selection: deterministic subject-balanced")
    print("  Sampler: 1000-step ancestral DDPM with v-prediction")
    print("  Validation EEG accessed: NO")
    print("  Final Test accessed: NO")
    print("  No training updates are executed in this script")

    for directory in (GENERATED_DIR, LOG_DIR, METRIC_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Step 1 — Audit final B0-v checkpoint
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT FINAL B0-V CHECKPOINT")
    print("=" * 80)

    if not BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(BEST_CHECKPOINT)
    if not RESULT_JSON.is_file():
        raise FileNotFoundError(RESULT_JSON)

    checkpoint_hash_before = sha256(BEST_CHECKPOINT)
    checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    with RESULT_JSON.open("r", encoding="utf-8") as f:
        training_result = json.load(f)

    best_epoch = int(checkpoint["epoch"])
    best_val_mse = float(checkpoint["best_ema_validation_v_mse"])
    source_epoch = int(checkpoint["source_epoch"])

    if best_epoch != EXPECTED_BEST_EPOCH:
        raise RuntimeError(
            f"Expected best B0-v epoch {EXPECTED_BEST_EPOCH}, found {best_epoch}"
        )
    if source_epoch != EXPECTED_SOURCE_EPOCH:
        raise RuntimeError(
            f"Expected epsilon source epoch {EXPECTED_SOURCE_EPOCH}, found {source_epoch}"
        )
    if checkpoint.get("prediction_target") != "v":
        raise RuntimeError("Checkpoint prediction target is not v")
    if checkpoint.get("final_test_accessed") is not False:
        raise RuntimeError("Final Test invariant failed in checkpoint")

    print(f"best B0-v epoch: {best_epoch}")
    print(f"best EMA Validation v-MSE: {best_val_mse:.6f}")
    print(f"epsilon source epoch: {source_epoch}")
    print(f"checkpoint SHA256: {checkpoint_hash_before}")
    print("checkpoint audit: PASS")

    # ========================================================
    # Step 2 — Build generator
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 2 — LOAD BEST EMA B0-V GENERATOR")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    source_contract = checkpoint["source_contract"]
    model = B0StateOnlyUNet1D(source_contract).to(device)
    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    model.eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("weights source: BEST EMA B0-v")

    schedule_cpu = build_cosine_schedule(TIMESTEPS, COSINE_S)
    reverse_schedule = build_reverse_schedule(schedule_cpu, device)

    # ========================================================
    # Step 3 — Generate independent synthetic audit set
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 3 — GENERATE FINAL-AUDIT SYNTHETIC SET")
    print("=" * 80)

    generation_start = time.perf_counter()
    eo_synth_z = sample_v_batches(
        model=model,
        eye_state_value=0,
        n_samples=N_PER_STATE,
        batch_size=SAMPLING_BATCH_SIZE,
        reverse_schedule=reverse_schedule,
        device=device,
        seed=EO_SAMPLING_SEED,
    )
    ec_synth_z = sample_v_batches(
        model=model,
        eye_state_value=1,
        n_samples=N_PER_STATE,
        batch_size=SAMPLING_BATCH_SIZE,
        reverse_schedule=reverse_schedule,
        device=device,
        seed=EC_SAMPLING_SEED,
    )
    generation_seconds = time.perf_counter() - generation_start

    if not np.isfinite(eo_synth_z).all() or not np.isfinite(ec_synth_z).all():
        raise RuntimeError("Synthetic data contains NaN/Inf")

    print(f"generation time: {generation_seconds / 60:.2f} min")
    print(f"synthetic EO normalized std: {eo_synth_z.std():.6f}")
    print(f"synthetic EC normalized std: {ec_synth_z.std():.6f}")

    # ========================================================
    # Step 4 — Build subject-balanced Real Train reference
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 4 — BUILD SUBJECT-BALANCED REAL TRAIN REFERENCE")
    print("=" * 80)

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)
    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked")

    train_records, subject_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )
    if len(train_records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError("Train window count mismatch")
    if len(subject_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError("Train subject count mismatch")

    eo_real_records = select_subject_balanced_records(
        train_records, 0, N_PER_STATE, REAL_REFERENCE_SEED
    )
    ec_real_records = select_subject_balanced_records(
        train_records, 1, N_PER_STATE, REAL_REFERENCE_SEED + 1
    )

    save_real_reference_manifest(eo_real_records, ec_real_records)

    eo_real_z = load_reference_array(eo_real_records, scaler)
    ec_real_z = load_reference_array(ec_real_records, scaler)

    real_z = {"EO": eo_real_z, "EC": ec_real_z}
    synth_z = {"EO": eo_synth_z, "EC": ec_synth_z}

    real_uv = {
        state: inverse_normalize(array, scaler) for state, array in real_z.items()
    }
    synth_uv = {
        state: inverse_normalize(array, scaler) for state, array in synth_z.items()
    }

    print(f"Real EO normalized std: {real_z['EO'].std():.6f}")
    print(f"Real EC normalized std: {real_z['EC'].std():.6f}")
    print(f"Synthetic EO normalized std: {synth_z['EO'].std():.6f}")
    print(f"Synthetic EC normalized std: {synth_z['EC'].std():.6f}")
    print(f"Real EO µV std: {real_uv['EO'].std():.3f}")
    print(f"Real EC µV std: {real_uv['EC'].std():.3f}")
    print(f"Synthetic EO µV std: {synth_uv['EO'].std():.3f}")
    print(f"Synthetic EC µV std: {synth_uv['EC'].std():.3f}")

    # Save synthetic package only. Real source is represented by manifest.
    np.savez_compressed(
        SYNTHETIC_FILE,
        eo_z=eo_synth_z,
        ec_z=ec_synth_z,
        eo_uv=synth_uv["EO"],
        ec_uv=synth_uv["EC"],
        channel_names=np.asarray(CHANNEL_NAMES),
        sfreq=np.asarray([SFREQ], dtype=np.int64),
        checkpoint_epoch=np.asarray([best_epoch], dtype=np.int64),
        eo_seed=np.asarray([EO_SAMPLING_SEED], dtype=np.int64),
        ec_seed=np.asarray([EC_SAMPLING_SEED], dtype=np.int64),
    )

    # ========================================================
    # Step 5 — Amplitude fidelity
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 5 — AMPLITUDE / SCALE FIDELITY")
    print("=" * 80)

    amplitude_rows = amplitude_summary_rows(real_z, synth_z, real_uv, synth_uv)
    with AMPLITUDE_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(amplitude_rows[0].keys()))
        writer.writeheader()
        writer.writerows(amplitude_rows)

    global_scale = {}
    for state in ("EO", "EC"):
        global_scale[state] = {
            "real_normalized_std": float(real_z[state].std()),
            "synthetic_normalized_std": float(synth_z[state].std()),
            "normalized_std_ratio": float(
                synth_z[state].std() / max(real_z[state].std(), np.finfo(float).eps)
            ),
            "real_uv_std": float(real_uv[state].std()),
            "synthetic_uv_std": float(synth_uv[state].std()),
            "uv_std_ratio": float(
                synth_uv[state].std()
                / max(real_uv[state].std(), np.finfo(float).eps)
            ),
            "synthetic_abs_gt_100uv_fraction": float(
                np.mean(np.abs(synth_uv[state]) > 100.0)
            ),
            "real_abs_gt_100uv_fraction": float(
                np.mean(np.abs(real_uv[state]) > 100.0)
            ),
        }
        print(
            f"{state}: normalized std ratio="
            f"{global_scale[state]['normalized_std_ratio']:.3f} "
            f"| µV std ratio={global_scale[state]['uv_std_ratio']:.3f}"
        )

    # ========================================================
    # Step 6 — PSD / bandpower fidelity
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 6 — PSD / BANDPOWER FIDELITY")
    print("=" * 80)

    real_psd = {}
    synth_psd = {}
    real_bands_by_band = {band: {} for band in BANDS}
    synth_bands_by_band = {band: {} for band in BANDS}

    frequency_grid = None
    real_by_state = {}
    synth_by_state = {}

    for state in ("EO", "EC"):
        f_real, p_real = welch_psd(real_uv[state])
        f_synth, p_synth = welch_psd(synth_uv[state])
        if not np.array_equal(f_real, f_synth):
            raise RuntimeError("PSD frequency grids differ")
        if frequency_grid is None:
            frequency_grid = f_real
        real_psd[state] = p_real
        synth_psd[state] = p_synth

        real_state_bands = bandpower_db_from_psd(f_real, p_real)
        synth_state_bands = bandpower_db_from_psd(f_synth, p_synth)
        real_by_state[state] = real_state_bands
        synth_by_state[state] = synth_state_bands
        for band in BANDS:
            real_bands_by_band[band][state] = real_state_bands[band]
            synth_bands_by_band[band][state] = synth_state_bands[band]

    band_rows = make_bandpower_rows(real_by_state, synth_by_state)
    with BANDPOWER_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(band_rows[0].keys()))
        writer.writeheader()
        writer.writerows(band_rows)

    band_mae = {}
    for band in BANDS:
        errors = [
            abs(row["synthetic_minus_real_db"])
            for row in band_rows
            if row["band"] == band
        ]
        band_mae[band] = float(np.mean(errors))
        print(f"{band} median-bandpower MAE: {band_mae[band]:.3f} dB")

    real_alpha_delta = np.median(real_by_state["EC"]["alpha"], axis=0) - np.median(
        real_by_state["EO"]["alpha"], axis=0
    )
    synth_alpha_delta = np.median(
        synth_by_state["EC"]["alpha"], axis=0
    ) - np.median(synth_by_state["EO"]["alpha"], axis=0)
    alpha_effect_mae = float(np.mean(np.abs(synth_alpha_delta - real_alpha_delta)))

    for channel in ("O1", "O2"):
        idx = CHANNEL_INDEX[channel]
        print(
            f"{channel} alpha EC-EO | real={real_alpha_delta[idx]:.3f} dB "
            f"| synth={synth_alpha_delta[idx]:.3f} dB"
        )
    print(f"channel-wise alpha-effect MAE: {alpha_effect_mae:.3f} dB")

    # ========================================================
    # Step 7 — Spatial covariance/correlation fidelity
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 7 — SPATIAL COVARIANCE / CORRELATION FIDELITY")
    print("=" * 80)

    real_cov = {}
    synth_cov = {}
    real_corr = {}
    synth_corr = {}
    spatial_metrics = {}

    for state in ("EO", "EC"):
        real_cov[state], real_corr[state] = mean_covariance_and_correlation(real_z[state])
        synth_cov[state], synth_corr[state] = mean_covariance_and_correlation(synth_z[state])

        spatial_metrics[state] = {
            "covariance_relative_frobenius_error": relative_frobenius(
                real_cov[state], synth_cov[state]
            ),
            "correlation_relative_frobenius_error": relative_frobenius(
                real_corr[state], synth_corr[state]
            ),
            "correlation_upper_triangle_pearson": upper_triangle_correlation(
                real_corr[state], synth_corr[state]
            ),
        }

        print(
            f"{state}: corr upper-triangle r="
            f"{spatial_metrics[state]['correlation_upper_triangle_pearson']:.3f} "
            f"| corr relF="
            f"{spatial_metrics[state]['correlation_relative_frobenius_error']:.3f}"
        )

    # ========================================================
    # Step 8 — Alpha debiased squared wPLI fidelity
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 8 — ALPHA DEBIASED-SQUARED wPLI FIDELITY")
    print("=" * 80)

    real_wpli = {}
    synth_wpli = {}
    connectivity_metrics = {}

    for state in ("EO", "EC"):
        real_wpli[state] = alpha_dwpli2_debiased(real_uv[state])
        synth_wpli[state] = alpha_dwpli2_debiased(synth_uv[state])

        connectivity_metrics[state] = {
            "relative_frobenius_error": relative_frobenius(
                real_wpli[state], synth_wpli[state]
            ),
            "upper_triangle_pearson": upper_triangle_correlation(
                real_wpli[state], synth_wpli[state]
            ),
            "real_mean_upper_triangle": float(
                np.mean(real_wpli[state][np.triu_indices(N_CHANNELS, k=1)])
            ),
            "synthetic_mean_upper_triangle": float(
                np.mean(synth_wpli[state][np.triu_indices(N_CHANNELS, k=1)])
            ),
        }

        print(
            f"{state}: alpha dwPLI2 upper-triangle r="
            f"{connectivity_metrics[state]['upper_triangle_pearson']:.3f} "
            f"| relF={connectivity_metrics[state]['relative_frobenius_error']:.3f}"
        )

    np.savez_compressed(
        MATRICES_FILE,
        real_cov_eo=real_cov["EO"],
        synth_cov_eo=synth_cov["EO"],
        real_cov_ec=real_cov["EC"],
        synth_cov_ec=synth_cov["EC"],
        real_corr_eo=real_corr["EO"],
        synth_corr_eo=synth_corr["EO"],
        real_corr_ec=real_corr["EC"],
        synth_corr_ec=synth_corr["EC"],
        real_dwpli2_eo=real_wpli["EO"],
        synth_dwpli2_eo=synth_wpli["EO"],
        real_dwpli2_ec=real_wpli["EC"],
        synth_dwpli2_ec=synth_wpli["EC"],
        channel_names=np.asarray(CHANNEL_NAMES),
    )

    # ========================================================
    # Step 9 — Save visual audit
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 9 — SAVE FINAL FIDELITY FIGURES")
    print("=" * 80)

    save_waveform_figure(real_uv, synth_uv)
    save_amplitude_figure(real_uv, synth_uv)
    save_psd_figure(frequency_grid, real_psd, synth_psd)
    save_alpha_effect_figure(real_bands_by_band, synth_bands_by_band)
    save_bandpower_error_figure(band_rows)
    save_matrix_figure(
        real_corr,
        synth_corr,
        CORRELATION_FIGURE,
        "B0-v Final Audit — Mean Channel Correlation",
    )
    save_matrix_figure(
        real_wpli,
        synth_wpli,
        WPLI_FIGURE,
        "B0-v Final Audit — Alpha Debiased Squared wPLI",
    )
    save_std_ratio_figure(amplitude_rows)

    print(f"waveforms: {WAVEFORM_FIGURE}")
    print(f"amplitude: {AMPLITUDE_FIGURE}")
    print(f"PSD: {PSD_FIGURE}")
    print(f"bandpower: {BANDPOWER_FIGURE}")
    print(f"alpha effect: {ALPHA_EFFECT_FIGURE}")
    print(f"correlation: {CORRELATION_FIGURE}")
    print(f"alpha dwPLI2: {WPLI_FIGURE}")
    print(f"channel std ratios: {STD_RATIO_FIGURE}")

    # ========================================================
    # Step 10 — Check checkpoint immutability and save summary
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 10 — SAVE FINAL AUDIT SUMMARY")
    print("=" * 80)

    checkpoint_hash_after = sha256(BEST_CHECKPOINT)
    checkpoint_unchanged = checkpoint_hash_before == checkpoint_hash_after
    if not checkpoint_unchanged:
        raise RuntimeError("B0-v source checkpoint changed during audit")

    summary = {
        "status": "B0_V_FINAL_REAL_TRAIN_FIDELITY_AUDIT_COMPLETE",
        "scientific_role": (
            "Generator-fidelity audit after 40-epoch B0-v fine-tuning; "
            "not a Final-Test evaluation."
        ),
        "checkpoint": {
            "path": str(BEST_CHECKPOINT),
            "epoch": best_epoch,
            "best_ema_validation_v_mse": best_val_mse,
            "epsilon_source_epoch": source_epoch,
            "sha256": checkpoint_hash_before,
            "unchanged_during_audit": checkpoint_unchanged,
        },
        "sample_sizes": {
            "real_train_per_state": N_PER_STATE,
            "synthetic_per_state": N_PER_STATE,
            "real_reference_selection": "deterministic_subject_balanced",
        },
        "sampling": {
            "prediction_target": "v",
            "sampler": "ancestral_DDPM",
            "timesteps": TIMESTEPS,
            "cosine_s": COSINE_S,
            "batch_size": SAMPLING_BATCH_SIZE,
            "eo_seed": EO_SAMPLING_SEED,
            "ec_seed": EC_SAMPLING_SEED,
            "generation_seconds": generation_seconds,
            "mixed_precision": False,
            "x0_clipping": False,
        },
        "global_scale": global_scale,
        "spectral": {
            "median_bandpower_mae_db": band_mae,
            "alpha_state_effect_channel_mae_db": alpha_effect_mae,
            "real_alpha_ec_minus_eo_db": {
                channel: float(real_alpha_delta[idx])
                for idx, channel in enumerate(CHANNEL_NAMES)
            },
            "synthetic_alpha_ec_minus_eo_db": {
                channel: float(synth_alpha_delta[idx])
                for idx, channel in enumerate(CHANNEL_NAMES)
            },
        },
        "spatial": spatial_metrics,
        "alpha_dwpli2_debiased": connectivity_metrics,
        "outputs": {
            "synthetic_npz": str(SYNTHETIC_FILE),
            "real_reference_manifest": str(REAL_MANIFEST_FILE),
            "amplitude_csv": str(AMPLITUDE_FILE),
            "bandpower_csv": str(BANDPOWER_FILE),
            "spatial_connectivity_npz": str(MATRICES_FILE),
            "waveform_figure": str(WAVEFORM_FIGURE),
            "amplitude_figure": str(AMPLITUDE_FIGURE),
            "psd_figure": str(PSD_FIGURE),
            "bandpower_figure": str(BANDPOWER_FIGURE),
            "alpha_effect_figure": str(ALPHA_EFFECT_FIGURE),
            "correlation_figure": str(CORRELATION_FIGURE),
            "wpli_figure": str(WPLI_FIGURE),
            "std_ratio_figure": str(STD_RATIO_FIGURE),
        },
        "training_updates_executed": False,
        "real_train_eeg_accessed": True,
        "validation_eeg_accessed": False,
        "final_test_accessed": False,
        "b0_declared_final": False,
        "acceptance_decision": "PENDING_INTERPRETATION",
    }

    atomic_json_save(summary, SUMMARY_FILE)

    print(f"summary JSON: {SUMMARY_FILE}")
    print(f"synthetic package: {SYNTHETIC_FILE}")
    print(f"Real reference manifest: {REAL_MANIFEST_FILE}")
    print("checkpoint immutability: PASS")

    # ========================================================
    # Final
    # ========================================================
    print("\n" + "=" * 80)
    print("B0-V FINAL REAL-TRAIN FIDELITY AUDIT COMPLETE")
    print("=" * 80)

    print(f"best checkpoint epoch: {best_epoch}")
    print(f"best EMA Validation v-MSE: {best_val_mse:.6f}")
    print(f"Synthetic windows: {N_PER_STATE} EO + {N_PER_STATE} EC")
    print(f"Real reference windows: {N_PER_STATE} EO + {N_PER_STATE} EC")
    print(
        f"normalized std ratio EO: {global_scale['EO']['normalized_std_ratio']:.3f}"
    )
    print(
        f"normalized std ratio EC: {global_scale['EC']['normalized_std_ratio']:.3f}"
    )
    print(f"alpha-effect channel MAE: {alpha_effect_mae:.3f} dB")
    print(
        f"correlation r EO/EC: "
        f"{spatial_metrics['EO']['correlation_upper_triangle_pearson']:.3f} / "
        f"{spatial_metrics['EC']['correlation_upper_triangle_pearson']:.3f}"
    )
    print(
        f"alpha dwPLI2 r EO/EC: "
        f"{connectivity_metrics['EO']['upper_triangle_pearson']:.3f} / "
        f"{connectivity_metrics['EC']['upper_triangle_pearson']:.3f}"
    )
    print("Validation EEG accessed: NO")
    print("Final Test accessed: NO")
    print("Training updates executed: NO")

    print("\nNEXT:")
    print("INTERPRET B0-V FIDELITY BEFORE FREEZING B0 OR STARTING B1")


if __name__ == "__main__":
    main()
