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
import torch
from scipy.signal import welch

from s2_23_check_real_train_dataset import (
    NORMALIZATION_FILE,
)

from s2_32_b0_unet_shape_test import (
    B0StateOnlyUNet1D,
)

from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

AMP_SAFE_RUN = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0"
    / "full_amp_safe_v1"
)

LATEST_CHECKPOINT = (
    AMP_SAFE_RUN
    / "checkpoints"
    / "latest.pt"
)

BEST_CHECKPOINT = (
    AMP_SAFE_RUN
    / "checkpoints"
    / "best_ema.pt"
)


# ============================================================
# Outputs
# ============================================================

GENERATED_DIR = (
    REPO_ROOT
    / "outputs"
    / "generated"
    / "scenario2"
    / "b0"
    / "sampling_sanity_v1"
)

METRIC_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "b0"
)

FIGURE_DIR = (
    REPO_ROOT
    / "outputs"
    / "figures"
    / "scenario2"
    / "b0"
    / "sampling_sanity_v1"
)

SYNTHETIC_FILE = (
    GENERATED_DIR
    / "b0_paired_synthetic_128_v1.npz"
)

SUMMARY_FILE = (
    METRIC_DIR
    / "b0_sampling_sanity_v1.json"
)

ALPHA_CSV_FILE = (
    METRIC_DIR
    / "b0_sampling_alpha_channel_summary_v1.csv"
)

WAVEFORM_FIGURE = (
    FIGURE_DIR
    / "b0_waveform_pairs_v1.png"
)

PSD_FIGURE = (
    FIGURE_DIR
    / "b0_occipital_psd_v1.png"
)

ALPHA_FIGURE = (
    FIGURE_DIR
    / "b0_alpha_state_effect_v1.png"
)

AMPLITUDE_FIGURE = (
    FIGURE_DIR
    / "b0_amplitude_distribution_v1.png"
)


# ============================================================
# Sampling configuration
# ============================================================

EXPECTED_PAUSE_EPOCH = 125

N_PER_STATE = 64

SAMPLING_SEED = 20260908

TIMESTEPS = 1000

COSINE_S = 0.008

SFREQ = 128

N_CHANNELS = 19

N_TIMES = 512


CHANNEL_NAMES = [
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "F3",
    "F4",
    "Fz",
    "T7",
    "T8",
    "C3",
    "C4",
    "Cz",
    "P7",
    "P8",
    "P3",
    "P4",
    "Pz",
    "O1",
    "O2",
]

CHANNEL_INDEX = {
    channel: index
    for index, channel
    in enumerate(CHANNEL_NAMES)
}


# ============================================================
# Helpers
# ============================================================

def sha256(
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

    temporary = Path(
        str(path)
        + ".tmp"
    )

    with temporary.open(
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
        temporary,
        path,
    )


# ============================================================
# DDPM reverse-process quantities
# ============================================================

def build_reverse_schedule(
    schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:

    betas = (
        schedule[
            "betas"
        ]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    alphas = (
        schedule[
            "alphas"
        ]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    alpha_bar = (
        schedule[
            "alpha_cumprod"
        ]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    alpha_bar_prev = torch.cat(
        [
            torch.ones(
                1,
                device=device,
                dtype=torch.float32,
            ),
            alpha_bar[:-1],
        ],
        dim=0,
    )

    posterior_variance = (
        betas
        * (
            1.0
            - alpha_bar_prev
        )
        / (
            1.0
            - alpha_bar
        )
    )

    posterior_variance = torch.clamp(
        posterior_variance,
        min=0.0,
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": (
            alpha_bar_prev
        ),
        "posterior_variance": (
            posterior_variance
        ),
    }


# ============================================================
# DDPM ancestral sampling
# ============================================================

@torch.inference_mode()
def sample_state(
    model: torch.nn.Module,
    eye_state_value: int,
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> np.ndarray:

    if eye_state_value not in (
        0,
        1,
    ):
        raise ValueError(
            "eye_state must be 0 or 1"
        )

    generator = torch.Generator(
        device="cuda"
    )

    generator.manual_seed(
        seed
    )

    shape = (
        N_PER_STATE,
        N_CHANNELS,
        N_TIMES,
    )

    # --------------------------------------------------------
    # x_T ~ N(0, I)
    #
    # The same seed will later be reused for EO and EC,
    # creating paired stochastic trajectories.
    # --------------------------------------------------------

    x = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    state = torch.full(
        (
            N_PER_STATE,
        ),
        fill_value=eye_state_value,
        device=device,
        dtype=torch.long,
    )

    betas = reverse_schedule[
        "betas"
    ]

    alphas = reverse_schedule[
        "alphas"
    ]

    alpha_bar = reverse_schedule[
        "alpha_bar"
    ]

    posterior_variance = (
        reverse_schedule[
            "posterior_variance"
        ]
    )

    model.eval()

    start = time.perf_counter()

    for t in reversed(
        range(
            TIMESTEPS
        )
    ):

        timestep = torch.full(
            (
                N_PER_STATE,
            ),
            fill_value=t,
            device=device,
            dtype=torch.long,
        )

        epsilon_pred = model(
            x,
            timestep,
            state,
        )

        if not torch.isfinite(
            epsilon_pred
        ).all():

            raise RuntimeError(
                f"Nonfinite epsilon prediction "
                f"at timestep {t}"
            )

        beta_t = betas[t]

        alpha_t = alphas[t]

        alpha_bar_t = alpha_bar[t]

        # ----------------------------------------------------
        # DDPM posterior mean using epsilon prediction
        #
        # mu_theta =
        #
        # 1/sqrt(alpha_t) *
        # [
        #   x_t -
        #   beta_t/sqrt(1-alpha_bar_t)
        #   * epsilon_theta
        # ]
        # ----------------------------------------------------

        model_mean = (
            x
            - (
                beta_t
                / torch.sqrt(
                    1.0
                    - alpha_bar_t
                )
            )
            * epsilon_pred
        ) / torch.sqrt(
            alpha_t
        )

        if t > 0:

            noise = torch.randn(
                shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

            variance = (
                posterior_variance[t]
            )

            x = (
                model_mean
                + torch.sqrt(
                    variance
                )
                * noise
            )

        else:

            x = model_mean

        if not torch.isfinite(
            x
        ).all():

            raise RuntimeError(
                f"Nonfinite generated state "
                f"at timestep {t}"
            )

        if (
            t == 999
            or t % 100 == 0
            or t == 0
        ):

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"state={eye_state_value}"
                f" | timestep={t:>3}"
                f" | mean={x.mean().item():>10.5f}"
                f" | std={x.std(unbiased=False).item():>10.5f}"
                f" | elapsed={elapsed:>7.1f}s"
            )

    torch.cuda.synchronize()

    print(
        f"state={eye_state_value} "
        f"sampling seconds: "
        f"{time.perf_counter() - start:.2f}"
    )

    return (
        x.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )


# ============================================================
# PSD helper
# ============================================================

def compute_alpha_logpower(
    eeg_uv: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

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

    alpha_mask = (
        (frequencies >= 8.0)
        & (frequencies <= 13.0)
    )

    alpha_power = np.trapezoid(
        psd[
            ...,
            alpha_mask
        ],
        frequencies[
            alpha_mask
        ],
        axis=-1,
    )

    alpha_power = np.maximum(
        alpha_power,
        np.finfo(
            np.float64
        ).tiny,
    )

    alpha_db = (
        10.0
        * np.log10(
            alpha_power
        )
    )

    return (
        frequencies,
        psd,
        alpha_db,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "SCENARIO 2 — "
        "B0 BEST-EMA SAMPLING SANITY"
    )
    print("=" * 80)

    print(
        "\nPurpose:"
    )

    print(
        "  Generate a SMALL paired "
        "EO/EC synthetic set."
    )

    print(
        "  This is NOT the final "
        "synthetic dataset."
    )

    print(
        "\nValidation EEG accessed: NO"
    )

    print(
        "Final Test EEG accessed: NO"
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    for directory in (
        GENERATED_DIR,
        METRIC_DIR,
        FIGURE_DIR,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit checkpoint
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT PAUSED B0 CHECKPOINT")
    print("=" * 80)

    if not LATEST_CHECKPOINT.is_file():

        raise FileNotFoundError(
            LATEST_CHECKPOINT
        )

    if not BEST_CHECKPOINT.is_file():

        raise FileNotFoundError(
            BEST_CHECKPOINT
        )

    latest = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    best = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    latest_epoch = int(
        latest[
            "epoch"
        ]
    )

    best_epoch = int(
        best[
            "epoch"
        ]
    )

    best_mse = float(
        best[
            "best_ema_validation_mse"
        ]
    )

    print(
        f"latest completed epoch: "
        f"{latest_epoch}"
    )

    print(
        f"best EMA epoch: "
        f"{best_epoch}"
    )

    print(
        f"best EMA Validation MSE: "
        f"{best_mse:.6f}"
    )

    if latest_epoch != (
        EXPECTED_PAUSE_EPOCH
    ):

        raise RuntimeError(
            f"Expected latest complete epoch "
            f"{EXPECTED_PAUSE_EPOCH}, "
            f"found {latest_epoch}. "
            f"Do not sample an unintended run."
        )

    if best_epoch > latest_epoch:

        raise RuntimeError(
            "Best epoch exceeds latest "
            "complete epoch."
        )

    if best_epoch != (
        EXPECTED_PAUSE_EPOCH
    ):

        print(
            "\nNOTE:"
        )

        print(
            f"Latest epoch is "
            f"{latest_epoch}, but the "
            f"Validation-selected best EMA "
            f"is epoch {best_epoch}."
        )

        print(
            "Sampling will correctly use "
            "the best EMA checkpoint."
        )

    latest_hash = sha256(
        LATEST_CHECKPOINT
    )

    best_hash = sha256(
        BEST_CHECKPOINT
    )

    print(
        f"latest SHA256: "
        f"{latest_hash}"
    )

    print(
        f"best SHA256: "
        f"{best_hash}"
    )

    print(
        "checkpoint audit: PASS"
    )

    # ========================================================
    # Step 2 — Device / model
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — LOAD BEST EMA GENERATOR")
    print("=" * 80)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA runtime: "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    contract = best[
        "contract"
    ]

    if contract[
        "model_name"
    ] != "B0_STATE_ONLY_DDPM":

        raise RuntimeError(
            "Checkpoint is not B0."
        )

    model = (
        B0StateOnlyUNet1D(
            contract
        )
        .to(device)
    )

    state_dict = (
        best[
            "ema_state_dict"
        ]
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"parameters: "
        f"{parameter_count:,}"
    )

    print(
        "weights source: BEST EMA"
    )

    print(
        f"checkpoint epoch: "
        f"{best_epoch}"
    )

    # ========================================================
    # Step 3 — Diffusion schedule
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — BUILD ANCESTRAL DDPM SAMPLER")
    print("=" * 80)

    schedule_cpu = (
        build_cosine_schedule(
            timesteps=TIMESTEPS,
            cosine_s=COSINE_S,
        )
    )

    reverse_schedule = (
        build_reverse_schedule(
            schedule_cpu,
            device,
        )
    )

    print(
        f"sampling steps: "
        f"{TIMESTEPS}"
    )

    print(
        "sampler: ancestral DDPM"
    )

    print(
        "x0 clipping: NO"
    )

    print(
        "mixed precision sampling: NO"
    )

    print(
        "sampler: PASS"
    )

    # ========================================================
    # Step 4 — Generate EO
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — GENERATE 64 EO WINDOWS")
    print("=" * 80)

    torch.cuda.empty_cache()

    eo_z = sample_state(
        model=model,
        eye_state_value=0,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        seed=SAMPLING_SEED,
    )

    # ========================================================
    # Step 5 — Generate EC with IDENTICAL stochastic seed
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — GENERATE 64 EC WINDOWS")
    print("=" * 80)

    print(
        "Using the SAME stochastic seed "
        "as EO for paired conditioning sanity."
    )

    torch.cuda.empty_cache()

    ec_z = sample_state(
        model=model,
        eye_state_value=1,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        seed=SAMPLING_SEED,
    )

    if eo_z.shape != (
        N_PER_STATE,
        N_CHANNELS,
        N_TIMES,
    ):

        raise RuntimeError(
            f"Unexpected EO shape: "
            f"{eo_z.shape}"
        )

    if ec_z.shape != (
        N_PER_STATE,
        N_CHANNELS,
        N_TIMES,
    ):

        raise RuntimeError(
            f"Unexpected EC shape: "
            f"{ec_z.shape}"
        )

    if not np.isfinite(
        eo_z
    ).all():

        raise RuntimeError(
            "EO synthetic contains NaN/Inf."
        )

    if not np.isfinite(
        ec_z
    ).all():

        raise RuntimeError(
            "EC synthetic contains NaN/Inf."
        )

    print(
        "synthetic tensor integrity: PASS"
    )

    # ========================================================
    # Step 6 — Inverse normalization to physical µV
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — RETURN SYNTHETIC EEG TO µV")
    print("=" * 80)

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        scaler = json.load(f)

    if scaler[
        "status"
    ] != "TRAIN_NORMALIZATION_LOCKED":

        raise RuntimeError(
            "Normalization is not locked."
        )

    means_uv = np.asarray(
        scaler[
            "means_uv"
        ],
        dtype=np.float64,
    )

    stds_uv = np.asarray(
        scaler[
            "stds_uv"
        ],
        dtype=np.float64,
    )

    if means_uv.shape != (
        N_CHANNELS,
    ):

        raise RuntimeError(
            "Wrong scaler mean shape."
        )

    if stds_uv.shape != (
        N_CHANNELS,
    ):

        raise RuntimeError(
            "Wrong scaler std shape."
        )

    eo_uv = (
        eo_z.astype(
            np.float64
        )
        * stds_uv[
            None,
            :,
            None,
        ]
        + means_uv[
            None,
            :,
            None,
        ]
    ).astype(
        np.float32
    )

    ec_uv = (
        ec_z.astype(
            np.float64
        )
        * stds_uv[
            None,
            :,
            None,
        ]
        + means_uv[
            None,
            :,
            None,
        ]
    ).astype(
        np.float32
    )

    print(
        f"EO normalized: "
        f"mean={eo_z.mean():.6f}, "
        f"std={eo_z.std():.6f}, "
        f"min={eo_z.min():.3f}, "
        f"max={eo_z.max():.3f}"
    )

    print(
        f"EC normalized: "
        f"mean={ec_z.mean():.6f}, "
        f"std={ec_z.std():.6f}, "
        f"min={ec_z.min():.3f}, "
        f"max={ec_z.max():.3f}"
    )

    print(
        f"EO µV: "
        f"mean={eo_uv.mean():.6f}, "
        f"std={eo_uv.std():.6f}, "
        f"min={eo_uv.min():.3f}, "
        f"max={eo_uv.max():.3f}"
    )

    print(
        f"EC µV: "
        f"mean={ec_uv.mean():.6f}, "
        f"std={ec_uv.std():.6f}, "
        f"min={ec_uv.min():.3f}, "
        f"max={ec_uv.max():.3f}"
    )

    paired_mean_abs_difference = float(
        np.mean(
            np.abs(
                ec_z
                - eo_z
            )
        )
    )

    print(
        f"paired mean |EC-EO| "
        f"(normalized): "
        f"{paired_mean_abs_difference:.6f}"
    )

    if paired_mean_abs_difference <= 0:

        raise RuntimeError(
            "EO and EC generation are identical."
        )

    # ========================================================
    # Step 7 — Save synthetic package
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — SAVE SYNTHETIC SANITY PACKAGE")
    print("=" * 80)

    eeg_z = np.concatenate(
        [
            eo_z,
            ec_z,
        ],
        axis=0,
    )

    eeg_uv = np.concatenate(
        [
            eo_uv,
            ec_uv,
        ],
        axis=0,
    )

    eye_state = np.concatenate(
        [
            np.zeros(
                N_PER_STATE,
                dtype=np.int64,
            ),
            np.ones(
                N_PER_STATE,
                dtype=np.int64,
            ),
        ]
    )

    pair_id = np.concatenate(
        [
            np.arange(
                N_PER_STATE,
                dtype=np.int64,
            ),
            np.arange(
                N_PER_STATE,
                dtype=np.int64,
            ),
        ]
    )

    temporary_npz = Path(
        str(
            SYNTHETIC_FILE
        )
        + ".tmp.npz"
    )

    np.savez_compressed(
        temporary_npz,
        eeg_z=eeg_z,
        eeg_uv=eeg_uv,
        eye_state=eye_state,
        pair_id=pair_id,
        channel_names=np.asarray(
            CHANNEL_NAMES
        ),
        sampling_seed=np.asarray(
            [
                SAMPLING_SEED
            ],
            dtype=np.int64,
        ),
        checkpoint_epoch=np.asarray(
            [
                best_epoch
            ],
            dtype=np.int64,
        ),
        sfreq=np.asarray(
            [
                SFREQ
            ],
            dtype=np.int64,
        ),
    )

    os.replace(
        temporary_npz,
        SYNTHETIC_FILE,
    )

    print(
        f"saved: "
        f"{SYNTHETIC_FILE}"
    )

    print(
        f"size MiB: "
        f"{SYNTHETIC_FILE.stat().st_size / 1024**2:.2f}"
    )

    # ========================================================
    # Step 8 — Synthetic spectral sanity
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 8 — SYNTHETIC ALPHA SANITY")
    print("=" * 80)

    (
        frequencies_eo,
        psd_eo,
        alpha_db_eo,
    ) = compute_alpha_logpower(
        eo_uv
    )

    (
        frequencies_ec,
        psd_ec,
        alpha_db_ec,
    ) = compute_alpha_logpower(
        ec_uv
    )

    if not np.array_equal(
        frequencies_eo,
        frequencies_ec,
    ):

        raise RuntimeError(
            "PSD frequency grids differ."
        )

    frequencies = (
        frequencies_eo
    )

    alpha_delta = (
        alpha_db_ec
        - alpha_db_eo
    )

    channel_rows = []

    for channel_index, channel in enumerate(
        CHANNEL_NAMES
    ):

        delta = (
            alpha_delta[
                :,
                channel_index
            ]
        )

        row = {
            "channel": channel,

            "eo_alpha_db_median": float(
                np.median(
                    alpha_db_eo[
                        :,
                        channel_index
                    ]
                )
            ),

            "ec_alpha_db_median": float(
                np.median(
                    alpha_db_ec[
                        :,
                        channel_index
                    ]
                )
            ),

            "paired_ec_minus_eo_db_median": float(
                np.median(
                    delta
                )
            ),

            "paired_ec_minus_eo_db_mean": float(
                np.mean(
                    delta
                )
            ),

            "fraction_ec_gt_eo": float(
                np.mean(
                    delta
                    > 0
                )
            ),
        }

        channel_rows.append(
            row
        )

    whole_head_subject_delta = (
        np.mean(
            alpha_delta,
            axis=1,
        )
    )

    whole_head_median_delta = float(
        np.median(
            whole_head_subject_delta
        )
    )

    print(
        f"whole-head paired median "
        f"EC-EO alpha difference: "
        f"{whole_head_median_delta:.3f} dB"
    )

    for channel in (
        "O1",
        "O2",
    ):

        index = (
            CHANNEL_INDEX[
                channel
            ]
        )

        delta = (
            alpha_delta[
                :,
                index
            ]
        )

        print(
            f"{channel}: "
            f"median EC-EO="
            f"{np.median(delta):.3f} dB"
            f" | mean="
            f"{np.mean(delta):.3f} dB"
            f" | P(EC>EO)="
            f"{np.mean(delta > 0):.3f}"
        )

    # --------------------------------------------------------
    # Save alpha CSV
    # --------------------------------------------------------

    with ALPHA_CSV_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                channel_rows[0]
                .keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            channel_rows
        )

    print(
        f"alpha CSV: "
        f"{ALPHA_CSV_FILE}"
    )

    # ========================================================
    # Step 9 — Figures
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 9 — SAVE VISUAL SANITY FIGURES")
    print("=" * 80)

    time_axis = (
        np.arange(
            N_TIMES
        )
        / SFREQ
    )

    # --------------------------------------------------------
    # Figure 1 — paired waveform
    # --------------------------------------------------------

    selected_channels = [
        "Fz",
        "Cz",
        "Pz",
        "O1",
        "O2",
    ]

    figure, axes = plt.subplots(
        len(
            selected_channels
        ),
        1,
        figsize=(
            12,
            10,
        ),
        sharex=True,
    )

    for axis, channel in zip(
        axes,
        selected_channels,
    ):

        index = (
            CHANNEL_INDEX[
                channel
            ]
        )

        axis.plot(
            time_axis,
            eo_uv[
                0,
                index,
                :,
            ],
            label="EO",
        )

        axis.plot(
            time_axis,
            ec_uv[
                0,
                index,
                :,
            ],
            label="EC",
        )

        axis.set_ylabel(
            f"{channel}\nµV"
        )

        axis.grid(
            True,
            alpha=0.25,
        )

    axes[0].legend()

    axes[-1].set_xlabel(
        "Time (s)"
    )

    figure.suptitle(
        "B0 Paired Synthetic EEG — "
        "Same Stochastic Trajectory, "
        "Different Eye-State Condition"
    )

    figure.tight_layout()

    figure.savefig(
        WAVEFORM_FIGURE,
        dpi=160,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # Figure 2 — O1/O2 PSD
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(
            10,
            8,
        ),
        sharex=True,
    )

    for axis, channel in zip(
        axes,
        [
            "O1",
            "O2",
        ],
    ):

        index = (
            CHANNEL_INDEX[
                channel
            ]
        )

        mean_psd_eo = np.mean(
            psd_eo[
                :,
                index,
                :,
            ],
            axis=0,
        )

        mean_psd_ec = np.mean(
            psd_ec[
                :,
                index,
                :,
            ],
            axis=0,
        )

        axis.semilogy(
            frequencies,
            mean_psd_eo,
            label="EO",
        )

        axis.semilogy(
            frequencies,
            mean_psd_ec,
            label="EC",
        )

        axis.axvspan(
            8,
            13,
            alpha=0.15,
            label=(
                "Alpha 8–13 Hz"
                if channel == "O1"
                else None
            ),
        )

        axis.set_ylabel(
            f"{channel}\nPSD"
        )

        axis.set_xlim(
            1,
            40,
        )

        axis.grid(
            True,
            alpha=0.25,
        )

    axes[0].legend()

    axes[-1].set_xlabel(
        "Frequency (Hz)"
    )

    figure.suptitle(
        "B0 Synthetic Occipital PSD"
    )

    figure.tight_layout()

    figure.savefig(
        PSD_FIGURE,
        dpi=160,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # Figure 3 — alpha EC-EO by channel
    # --------------------------------------------------------

    channel_delta_medians = np.asarray(
        [
            row[
                "paired_ec_minus_eo_db_median"
            ]
            for row in channel_rows
        ],
        dtype=np.float64,
    )

    figure = plt.figure(
        figsize=(
            13,
            6,
        )
    )

    axis = figure.add_subplot(
        111
    )

    axis.bar(
        np.arange(
            N_CHANNELS
        ),
        channel_delta_medians,
    )

    axis.axhline(
        0.0,
        linewidth=1,
    )

    axis.set_xticks(
        np.arange(
            N_CHANNELS
        )
    )

    axis.set_xticklabels(
        CHANNEL_NAMES,
        rotation=45,
        ha="right",
    )

    axis.set_ylabel(
        "Median paired EC − EO alpha power (dB)"
    )

    axis.set_title(
        "B0 Synthetic Eye-State Alpha Effect"
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        ALPHA_FIGURE,
        dpi=160,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # Figure 4 — amplitude distributions
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(
            10,
            6,
        )
    )

    axis = figure.add_subplot(
        111
    )

    axis.hist(
        eo_uv.reshape(
            -1
        ),
        bins=150,
        density=True,
        histtype="step",
        label="EO",
    )

    axis.hist(
        ec_uv.reshape(
            -1
        ),
        bins=150,
        density=True,
        histtype="step",
        label="EC",
    )

    axis.set_xlabel(
        "Synthetic EEG amplitude (µV)"
    )

    axis.set_ylabel(
        "Density"
    )

    axis.set_title(
        "B0 Synthetic EEG Amplitude Distribution"
    )

    axis.legend()

    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        AMPLITUDE_FIGURE,
        dpi=160,
    )

    plt.close(
        figure
    )

    print(
        f"waveforms: "
        f"{WAVEFORM_FIGURE}"
    )

    print(
        f"PSD: "
        f"{PSD_FIGURE}"
    )

    print(
        f"alpha effect: "
        f"{ALPHA_FIGURE}"
    )

    print(
        f"amplitudes: "
        f"{AMPLITUDE_FIGURE}"
    )

    # ========================================================
    # Step 10 — Save summary
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 10 — SAVE SAMPLING SUMMARY")
    print("=" * 80)

    summary = {
        "status": (
            "B0_SAMPLING_SANITY_COMPLETE"
        ),

        "role": (
            "interim generator-quality "
            "assessment after epoch-125 pause"
        ),

        "checkpoint": {
            "latest_complete_epoch": (
                latest_epoch
            ),

            "selected_best_ema_epoch": (
                best_epoch
            ),

            "best_ema_validation_mse": (
                best_mse
            ),

            "latest_checkpoint": str(
                LATEST_CHECKPOINT
            ),

            "best_checkpoint": str(
                BEST_CHECKPOINT
            ),

            "latest_sha256": (
                latest_hash
            ),

            "best_sha256": (
                best_hash
            ),
        },

        "sampling": {
            "sampler": (
                "ancestral_DDPM"
            ),

            "timesteps": (
                TIMESTEPS
            ),

            "cosine_s": (
                COSINE_S
            ),

            "samples_per_state": (
                N_PER_STATE
            ),

            "total_samples": (
                2
                * N_PER_STATE
            ),

            "sampling_seed": (
                SAMPLING_SEED
            ),

            "paired_stochastic_trajectory": True,

            "x0_clipping": False,

            "mixed_precision": False,
        },

        "synthetic_normalized": {
            "eo_mean": float(
                eo_z.mean()
            ),

            "eo_std": float(
                eo_z.std()
            ),

            "eo_min": float(
                eo_z.min()
            ),

            "eo_max": float(
                eo_z.max()
            ),

            "ec_mean": float(
                ec_z.mean()
            ),

            "ec_std": float(
                ec_z.std()
            ),

            "ec_min": float(
                ec_z.min()
            ),

            "ec_max": float(
                ec_z.max()
            ),

            "paired_mean_absolute_difference": (
                paired_mean_abs_difference
            ),
        },

        "synthetic_uv": {
            "eo_mean": float(
                eo_uv.mean()
            ),

            "eo_std": float(
                eo_uv.std()
            ),

            "eo_min": float(
                eo_uv.min()
            ),

            "eo_max": float(
                eo_uv.max()
            ),

            "ec_mean": float(
                ec_uv.mean()
            ),

            "ec_std": float(
                ec_uv.std()
            ),

            "ec_min": float(
                ec_uv.min()
            ),

            "ec_max": float(
                ec_uv.max()
            ),
        },

        "alpha_sanity": {
            "whole_head_paired_median_ec_minus_eo_db": (
                whole_head_median_delta
            ),

            "O1": next(
                row
                for row
                in channel_rows
                if row[
                    "channel"
                ]
                == "O1"
            ),

            "O2": next(
                row
                for row
                in channel_rows
                if row[
                    "channel"
                ]
                == "O2"
            ),
        },

        "outputs": {
            "synthetic_npz": str(
                SYNTHETIC_FILE
            ),

            "alpha_csv": str(
                ALPHA_CSV_FILE
            ),

            "waveform_figure": str(
                WAVEFORM_FIGURE
            ),

            "psd_figure": str(
                PSD_FIGURE
            ),

            "alpha_figure": str(
                ALPHA_FIGURE
            ),

            "amplitude_figure": str(
                AMPLITUDE_FIGURE
            ),
        },

        "validation_eeg_accessed": False,

        "final_test_accessed": False,

        "b0_training_resumable": True,

        "b0_declared_final": False,
    }

    atomic_json_save(
        summary,
        SUMMARY_FILE,
    )

    print(
        f"summary JSON: "
        f"{SUMMARY_FILE}"
    )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("B0 SAMPLING SANITY COMPLETE")
    print("=" * 80)

    print(
        f"selected checkpoint epoch: "
        f"{best_epoch}"
    )

    print(
        f"EO synthetic windows: "
        f"{N_PER_STATE}"
    )

    print(
        f"EC synthetic windows: "
        f"{N_PER_STATE}"
    )

    print(
        f"total synthetic windows: "
        f"{2 * N_PER_STATE}"
    )

    print(
        f"paired |EC-EO| z difference: "
        f"{paired_mean_abs_difference:.6f}"
    )

    print(
        f"whole-head alpha EC-EO: "
        f"{whole_head_median_delta:.3f} dB"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These 128 windows are diagnostic "
        "samples, NOT the final synthetic dataset."
    )

    print(
        "Do NOT start B1 yet."
    )

    print(
        "\nNEXT:"
    )

    print(
        "COMPARE B0 SYNTHETIC "
        "WITH REAL TRAIN EEG"
    )


if __name__ == "__main__":
    main()