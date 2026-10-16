from __future__ import annotations

import csv
import hashlib
import json
import math
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
from s2_39_b0_epsilon_to_v_conversion_sanity import (
    make_v_target,
    q_sample,
)
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    inverse_normalize,
    welch_psd,
)
from s2_47_b1_architecture_warmstart_sanity import (
    state_dict_sha256,
)
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import (
    compute_candidate_metrics,
)
from s2_53_b2_lambda_calibration_pilot import (
    PhysicalLogPSDLoss,
    reconstruct_x0_from_v,
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

# Exact B1 subject-aware V2 reference and baseline synthetic samples.
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

# Matched lambda=0 control.
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

# Primary lambda=0.01 B2.
PRIMARY_B2_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "full_finetune_v1"
)

PRIMARY_B2_E5_CHECKPOINT = (
    PRIMARY_B2_ROOT
    / "checkpoints"
    / "candidate_epoch_005_raw.pt"
)

PRIMARY_B2_RESULT = (
    PRIMARY_B2_ROOT
    / "metrics"
    / "b2_full_finetune_result_v1.json"
)

# Completed s2_58 joint audit.
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

S2_58_CONTROL_E5_SYNTHETIC = (
    S2_58_ROOT
    / "generated"
    / "lambda0_control_epoch_005_synthetic_v1.npz"
)


# ============================================================
# s2_59 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "denoiser_mechanism_diagnostics_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
FIGURE_DIR = RUN_ROOT / "figures"

PROTOCOL_FILE = (
    METRIC_DIR
    / "s2_59_denoiser_diagnostics_protocol_v1.json"
)

CONTROL_GUARDRAIL_JSON = (
    METRIC_DIR
    / "s2_59_control_e5_guardrail_audit_v1.json"
)

TIMESTEP_CSV = (
    METRIC_DIR
    / "s2_59_timestep_denoiser_diagnostics_v1.csv"
)

BAND_PROJECTION_CSV = (
    METRIC_DIR
    / "s2_59_noise_projection_by_band_v1.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_59_denoiser_diagnostics_summary_v1.json"
)

PSD_VS_TIMESTEP_FIGURE = (
    FIGURE_DIR
    / "s2_59_one_step_psd_mae_vs_timestep_v1.png"
)

SPECTRAL_LOSS_FIGURE = (
    FIGURE_DIR
    / "s2_59_spectral_huber_vs_timestep_v1.png"
)

GRADIENT_FIGURE = (
    FIGURE_DIR
    / "s2_59_output_gradient_conflict_vs_timestep_v1.png"
)

NOISE_PROJECTION_FIGURE = (
    FIGURE_DIR
    / "s2_59_noise_projection_vs_timestep_v1.png"
)


# ============================================================
# Frozen diagnostic design
# ============================================================

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

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

TIMESTEPS = 1000
COSINE_S = 0.008

CONTROL_LAMBDA = 0.0
PRIMARY_LAMBDA = 0.01

# Use the exact Train-only B1 V2 reference already used by s2_56/s2_58.
# One fixed epsilon tensor is reused across ALL timestep probes and ALL models.
PROBE_NOISE_SEED = 20261159

# Chosen before diagnostic execution. Includes very low, low, middle,
# high, and extreme diffusion-noise levels.
PROBE_TIMESTEPS = [
    0,
    25,
    50,
    100,
    200,
    300,
    400,
    500,
    650,
    800,
    950,
    999,
]

BATCH_SIZE = 64

MIXED_PRECISION = True

SPECTRAL_FMIN_HZ = 1.0
SPECTRAL_FMAX_HZ = 40.0
SPECTRAL_LOG_EPS = 1e-12
SPECTRAL_HUBER_BETA_DB = 1.0

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

STATE_NAMES = [
    "EO",
    "EC",
]

# Diagnostic bands only. These do not change any project scientific metric.
DIAGNOSTIC_BANDS = {
    "delta_1_4": (
        1.0,
        4.0,
    ),
    "theta_4_8": (
        4.0,
        8.0,
    ),
    "alpha_8_13": (
        8.0,
        13.0,
    ),
    "beta_13_30": (
        13.0,
        30.0,
    ),
    "high_30_40": (
        30.0,
        40.0,
    ),
}


# ============================================================
# IO helpers
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


# ============================================================
# Exact reference/cache loading
# ============================================================

def load_exact_b1_reference() -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[str],
]:
    real_package = np.load(
        B1_REAL_REFERENCE_NPZ,
        allow_pickle=False,
    )

    b1_synth_package = np.load(
        B1_SYNTHETIC_REFERENCE_NPZ,
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
        .astype(
            str
        )
        .tolist()
    )

    synth_subject_indices = (
        b1_synth_package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    synth_subject_ids = (
        b1_synth_package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if subject_indices != synth_subject_indices:
        raise RuntimeError(
            "B1 Real/synthetic subject-index cohort mismatch."
        )

    if subject_ids != synth_subject_ids:
        raise RuntimeError(
            "B1 Real/synthetic subject-ID cohort mismatch."
        )

    b1_synth_z = b1_synth_package[
        "epoch_010_z"
    ].astype(
        np.float32,
        copy=False,
    )

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(
            f"Unexpected Real reference shape: "
            f"{real_z.shape}"
        )

    if b1_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected B1 synthetic shape: "
            f"{b1_synth_z.shape}"
        )

    if len(
        subject_indices
    ) != SELECTED_SUBJECT_COUNT:
        raise RuntimeError(
            "Unexpected selected-subject count."
        )

    if not np.isfinite(
        real_z
    ).all():
        raise RuntimeError(
            "Real reference contains NaN/Inf."
        )

    if not np.isfinite(
        b1_synth_z
    ).all():
        raise RuntimeError(
            "B1 synthetic reference contains NaN/Inf."
        )

    return (
        real_z,
        b1_synth_z,
        subject_indices,
        subject_ids,
    )


def load_control_e5_synthetic(
    *,
    expected_subject_indices: list[int],
    expected_subject_ids: list[str],
) -> np.ndarray:
    package = np.load(
        S2_58_CONTROL_E5_SYNTHETIC,
        allow_pickle=False,
    )

    subject_indices = (
        package[
            "selected_subject_indices"
        ]
        .astype(
            np.int64,
            copy=False,
        )
        .tolist()
    )

    subject_ids = (
        package[
            "selected_subject_ids"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if subject_indices != expected_subject_indices:
        raise RuntimeError(
            "Control E5 cache subject-index mismatch."
        )

    if subject_ids != expected_subject_ids:
        raise RuntimeError(
            "Control E5 cache subject-ID mismatch."
        )

    expected_checkpoint_hash = sha256_file(
        CONTROL_E5_CHECKPOINT
    )

    cached_checkpoint_hash = str(
        package[
            "checkpoint_sha256"
        ][
            0
        ]
    )

    if cached_checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError(
            "Control E5 cache checkpoint hash mismatch."
        )

    synth_z = package[
        "synth_z"
    ].astype(
        np.float32,
        copy=False,
    )

    if synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected Control E5 synthetic shape: "
            f"{synth_z.shape}"
        )

    if not np.isfinite(
        synth_z
    ).all():
        raise RuntimeError(
            "Control E5 synthetic cache contains NaN/Inf."
        )

    return synth_z


# ============================================================
# Model / probe helpers
# ============================================================

def build_probe_arrays(
    *,
    real_z: np.ndarray,
    selected_subject_indices: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Real reference layout:
      [S, state, rep, C, T]

    Flatten in exact C-order:
      subject -> state -> rep.

    Returns:
      x0 [640, 19, 512]
      eye_state [640]
      subject_index [640]
    """

    x0 = real_z.reshape(
        (
            -1,
            N_CHANNELS,
            N_TIMES,
        )
    )

    state = []

    subject = []

    for local_subject_index, global_subject_index in enumerate(
        selected_subject_indices
    ):
        del local_subject_index

        for state_index in range(
            2
        ):
            for _ in range(
                REAL_WINDOWS_PER_SUBJECT_STATE
            ):
                state.append(
                    state_index
                )

                subject.append(
                    int(
                        global_subject_index
                    )
                )

    state_array = np.asarray(
        state,
        dtype=np.int64,
    )

    subject_array = np.asarray(
        subject,
        dtype=np.int64,
    )

    if len(
        state_array
    ) != x0.shape[
        0
    ]:
        raise RuntimeError(
            "Probe state count mismatch."
        )

    if len(
        subject_array
    ) != x0.shape[
        0
    ]:
        raise RuntimeError(
            "Probe subject count mismatch."
        )

    return (
        x0,
        state_array,
        subject_array,
    )


def make_fixed_probe_noise(
    shape: tuple[
        int,
        int,
        int,
    ],
) -> np.ndarray:
    generator = torch.Generator()

    generator.manual_seed(
        PROBE_NOISE_SEED
    )

    epsilon = torch.randn(
        shape,
        dtype=torch.float32,
        generator=generator,
    )

    return epsilon.numpy().astype(
        np.float32,
        copy=False,
    )


def build_model_variants(
    *,
    frozen_b1: dict[str, Any],
    device: torch.device,
) -> dict[
    str,
    torch.nn.Module,
]:
    control_e5 = torch.load(
        CONTROL_E5_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    primary_e5 = torch.load(
        PRIMARY_B2_E5_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if control_e5[
        "status"
    ] != "B2_MATCHED_LAMBDA0_CONTROL_CHECKPOINT":
        raise RuntimeError(
            "Unexpected Control E5 checkpoint status."
        )

    if int(
        control_e5[
            "epoch"
        ]
    ) != 5:
        raise RuntimeError(
            "Control E5 checkpoint epoch mismatch."
        )

    if abs(
        float(
            control_e5[
                "lambda_spectral"
            ]
        )
        - CONTROL_LAMBDA
    ) > 1e-12:
        raise RuntimeError(
            "Control E5 lambda mismatch."
        )

    if primary_e5[
        "status"
    ] != "B2_FULL_FINETUNE_CHECKPOINT":
        raise RuntimeError(
            "Unexpected Primary B2 E5 checkpoint status."
        )

    if int(
        primary_e5[
            "epoch"
        ]
    ) != 5:
        raise RuntimeError(
            "Primary B2 E5 checkpoint epoch mismatch."
        )

    if abs(
        float(
            primary_e5[
                "lambda_spectral"
            ]
        )
        - PRIMARY_LAMBDA
    ) > 1e-12:
        raise RuntimeError(
            "Primary B2 E5 lambda mismatch."
        )

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
        "L0_E005": build_model_from_state(
            frozen_b1=frozen_b1,
            state_dict=(
                control_e5[
                    "model_state_dict"
                ]
            ),
            device=device,
        ),
        "L001_E005": build_model_from_state(
            frozen_b1=frozen_b1,
            state_dict=(
                primary_e5[
                    "model_state_dict"
                ]
            ),
            device=device,
        ),
    }

    for model in models.values():
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(
                False
            )

    return models


# ============================================================
# Numeric diagnostic helpers
# ============================================================

def output_gradient_metrics(
    *,
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
    spectral_loss_fn: PhysicalLogPSDLoss,
) -> dict[str, float]:
    """
    Compute conflict in MODEL-OUTPUT SPACE, not parameter space.

    This is intentionally cheap and does not backpropagate through the model.
    The model output is detached and used as an independent leaf tensor.

    Therefore:
      cosine < 0
        -> the two losses locally prefer opposing changes in pred_v output.
      lambda_weighted_norm_ratio
        -> magnitude of lambda*grad_spec relative to grad_v in output space.

    This is diagnostic only. Parameter-space gradient conflict can differ
    because of the network Jacobian.
    """

    pred_leaf = (
        pred_v
        .detach()
        .float()
        .requires_grad_(
            True
        )
    )

    target_v_f = target_v.float()

    v_loss = torch.mean(
        (
            pred_leaf
            - target_v_f
        )
        ** 2
    )

    x0_hat = reconstruct_x0_from_v(
        xt=xt.float(),
        v_pred=pred_leaf,
        timesteps=timesteps,
        alpha_bar=alpha_bar,
    )

    spectral_loss = spectral_loss_fn(
        x0_hat,
        x0.float(),
    )

    grad_v = torch.autograd.grad(
        v_loss,
        pred_leaf,
        retain_graph=True,
        create_graph=False,
    )[
        0
    ]

    grad_spec = torch.autograd.grad(
        spectral_loss,
        pred_leaf,
        retain_graph=False,
        create_graph=False,
    )[
        0
    ]

    grad_v_f = grad_v.float().reshape(
        -1
    )

    grad_spec_f = grad_spec.float().reshape(
        -1
    )

    dot = torch.dot(
        grad_v_f,
        grad_spec_f,
    )

    norm_v = torch.linalg.vector_norm(
        grad_v_f
    )

    norm_spec = torch.linalg.vector_norm(
        grad_spec_f
    )

    denominator = (
        norm_v
        * norm_spec
    )

    cosine = (
        float(
            (
                dot
                / denominator
            ).item()
        )
        if float(
            denominator.item()
        )
        > 0.0
        else float(
            "nan"
        )
    )

    norm_ratio = float(
        (
            norm_spec
            / torch.clamp(
                norm_v,
                min=torch.finfo(
                    torch.float32
                ).eps,
            )
        ).item()
    )

    return {
        "v_loss": float(
            v_loss.item()
        ),
        "spectral_loss_db_huber": float(
            spectral_loss.item()
        ),
        "grad_dot": float(
            dot.item()
        ),
        "grad_v_norm": float(
            norm_v.item()
        ),
        "grad_spec_norm": float(
            norm_spec.item()
        ),
        "grad_cosine_output_space": (
            cosine
        ),
        "grad_spec_to_v_norm_ratio_output_space": (
            norm_ratio
        ),
        "lambda001_weighted_grad_norm_ratio_output_space": float(
            PRIMARY_LAMBDA
            * norm_ratio
        ),
    }


def multiple_projection_coefficients(
    *,
    xhat: np.ndarray,
    x0: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """
    Descriptive two-predictor least-squares projection:

      xhat ~= c_x0 * x0 + c_eps * epsilon

    Each window/channel is first demeaned over time.

    IMPORTANT:
    c_eps is NOT expected to be zero even for an ideal MMSE denoiser.
    It is interpreted only RELATIVELY across matched B1 / lambda0 /
    lambda0.01 models on the exact same x0, epsilon, and timestep.
    """

    y = xhat.astype(
        np.float64,
        copy=False,
    )

    x = x0.astype(
        np.float64,
        copy=False,
    )

    e = epsilon.astype(
        np.float64,
        copy=False,
    )

    y = (
        y
        - np.mean(
            y,
            axis=-1,
            keepdims=True,
        )
    )

    x = (
        x
        - np.mean(
            x,
            axis=-1,
            keepdims=True,
        )
    )

    e = (
        e
        - np.mean(
            e,
            axis=-1,
            keepdims=True,
        )
    )

    y = y.reshape(
        -1
    )

    x = x.reshape(
        -1
    )

    e = e.reshape(
        -1
    )

    xx = float(
        np.dot(
            x,
            x,
        )
    )

    xe = float(
        np.dot(
            x,
            e,
        )
    )

    ee = float(
        np.dot(
            e,
            e,
        )
    )

    xy = float(
        np.dot(
            x,
            y,
        )
    )

    ey = float(
        np.dot(
            e,
            y,
        )
    )

    determinant = (
        xx
        * ee
        - xe
        * xe
    )

    if abs(
        determinant
    ) <= np.finfo(
        np.float64
    ).eps:
        return (
            float(
                "nan"
            ),
            float(
                "nan"
            ),
        )

    c_x0 = (
        xy
        * ee
        - ey
        * xe
    ) / determinant

    c_eps = (
        ey
        * xx
        - xy
        * xe
    ) / determinant

    return (
        float(
            c_x0
        ),
        float(
            c_eps
        ),
    )


def fft_band_component(
    array: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """
    Zero-phase FFT band extraction for DIAGNOSTIC projection only.
    Does not modify any scientific model/evaluation metric.

    Uses [low, high) except that 40 Hz is included by floating grid.
    """

    values = array.astype(
        np.float64,
        copy=False,
    )

    spectrum = np.fft.rfft(
        values,
        axis=-1,
    )

    frequencies = np.fft.rfftfreq(
        N_TIMES,
        d=1.0 / SFREQ,
    )

    if abs(
        high_hz
        - 40.0
    ) < 1e-12:
        mask = (
            (
                frequencies
                >= low_hz
            )
            & (
                frequencies
                <= high_hz
            )
        )
    else:
        mask = (
            (
                frequencies
                >= low_hz
            )
            & (
                frequencies
                < high_hz
            )
        )

    filtered = np.zeros_like(
        spectrum
    )

    filtered[
        ...,
        mask
    ] = spectrum[
        ...,
        mask
    ]

    return np.fft.irfft(
        filtered,
        n=N_TIMES,
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )


def normalized_scale_ratio(
    *,
    xhat: np.ndarray,
    x0: np.ndarray,
) -> float:
    xhat_std = float(
        np.std(
            xhat,
            dtype=np.float64,
        )
    )

    x0_std = float(
        np.std(
            x0,
            dtype=np.float64,
        )
    )

    return float(
        xhat_std
        / max(
            x0_std,
            np.finfo(
                np.float64
            ).eps,
        )
    )


def signed_psd_bias_from_arrays(
    *,
    real_z: np.ndarray,
    synth_z: np.ndarray,
    scaler: dict[str, Any],
) -> float:
    """
    Exact project Welch convention; pooled state median.
    Returns mean signed (synthetic - real) log-PSD bias in dB.
    """

    real_shape = real_z.shape

    synth_shape = synth_z.shape

    real_uv = inverse_normalize(
        real_z.reshape(
            (
                -1,
                N_CHANNELS,
                N_TIMES,
            )
        ),
        scaler,
    )

    synth_uv = inverse_normalize(
        synth_z.reshape(
            (
                -1,
                N_CHANNELS,
                N_TIMES,
            )
        ),
        scaler,
    )

    frequencies, real_psd = welch_psd(
        real_uv
    )

    synth_frequencies, synth_psd = welch_psd(
        synth_uv
    )

    if not np.array_equal(
        frequencies,
        synth_frequencies,
    ):
        raise RuntimeError(
            "Real/synthetic Welch grid mismatch."
        )

    mask = (
        (
            frequencies
            >= 1.0
        )
        & (
            frequencies
            <= 40.0
        )
    )

    tiny = np.finfo(
        np.float64
    ).tiny

    real_db = (
        10.0
        * np.log10(
            np.maximum(
                real_psd[
                    ...,
                    mask
                ],
                tiny,
            )
        )
    ).reshape(
        (
            real_shape[
                0
            ],
            real_shape[
                1
            ],
            real_shape[
                2
            ],
            N_CHANNELS,
            -1,
        )
    )

    synth_db = (
        10.0
        * np.log10(
            np.maximum(
                synth_psd[
                    ...,
                    mask
                ],
                tiny,
            )
        )
    ).reshape(
        (
            synth_shape[
                0
            ],
            synth_shape[
                1
            ],
            synth_shape[
                2
            ],
            N_CHANNELS,
            -1,
        )
    )

    state_biases = []

    for state_index in range(
        2
    ):
        real_state = real_db[
            :,
            state_index,
            ...,
        ].reshape(
            (
                -1,
                N_CHANNELS,
                real_db.shape[
                    -1
                ],
            )
        )

        synth_state = synth_db[
            :,
            state_index,
            ...,
        ].reshape(
            (
                -1,
                N_CHANNELS,
                synth_db.shape[
                    -1
                ],
            )
        )

        real_median = np.median(
            real_state,
            axis=0,
        )

        synth_median = np.median(
            synth_state,
            axis=0,
        )

        state_biases.append(
            float(
                np.mean(
                    synth_median
                    - real_median
                )
            )
        )

    return float(
        np.mean(
            state_biases
        )
    )


# ============================================================
# Control E5 guardrail audit
# ============================================================

def audit_control_e5_guardrails(
    *,
    real_z: np.ndarray,
    b1_synth_z: np.ndarray,
    control_e5_synth_z: np.ndarray,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    b1_generic, _, _ = compute_candidate_metrics(
        real_z=real_z,
        synth_z=b1_synth_z,
        scaler=scaler,
    )

    b1_full_psd = full_log_psd_mae(
        real_z=real_z,
        synth_z=b1_synth_z,
        scaler=scaler,
    )

    control_generic, _, _ = compute_candidate_metrics(
        real_z=real_z,
        synth_z=control_e5_synth_z,
        scaler=scaler,
    )

    control_full_psd = full_log_psd_mae(
        real_z=real_z,
        synth_z=control_e5_synth_z,
        scaler=scaler,
    )

    b1_baseline = {
        "generic": (
            b1_generic
        ),
        "full_log_psd": (
            b1_full_psd
        ),
    }

    candidate = {
        "generic": (
            control_generic
        ),
        "full_log_psd": (
            control_full_psd
        ),
    }

    eligibility = evaluate_eligibility(
        candidate=candidate,
        b1_baseline=b1_baseline,
    )

    subject = control_generic[
        "subject_identity"
    ]

    result = {
        "status": (
            "CONTROL_E5_EXISTING_B2_GUARDRAIL_DIAGNOSTIC_COMPLETE"
        ),
        "scientific_role": (
            "Diagnostic application of the already-frozen B2 preservation "
            "guardrails to matched lambda=0 Control E5. This does NOT "
            "turn Control E5 into a B2 spectral candidate and does NOT "
            "freeze it."
        ),
        "B1": {
            "full_psd_mae_db": (
                b1_full_psd[
                    "mean_mae_db"
                ]
            ),
            "mean_bandpower_mae_db": (
                b1_generic[
                    "mean_bandpower_mae_db"
                ]
            ),
            "alpha_effect_mae_db": (
                b1_generic[
                    "alpha_effect"
                ][
                    "channel_mae_db"
                ]
            ),
            "mean_channel_correlation_r": (
                mean_corr_r(
                    b1_generic
                )
            ),
            "mean_alpha_dwpli2_r": (
                mean_wpli_r(
                    b1_generic
                )
            ),
            "subject_top1": (
                b1_generic[
                    "subject_identity"
                ][
                    "subject_retrieval_top1_accuracy"
                ]
            ),
            "subject_fingerprint_r_median": (
                b1_generic[
                    "subject_identity"
                ][
                    "subject_fingerprint_pearson_r_median"
                ]
            ),
            "subject_margin": (
                b1_generic[
                    "subject_identity"
                ][
                    "subject_correct_vs_median_wrong_margin_mean"
                ]
            ),
            "fixed_wrong_win_fraction": (
                b1_generic[
                    "subject_identity"
                ][
                    "subject_correct_better_than_fixed_wrong_fraction"
                ]
            ),
        },
        "Control_E5": {
            "full_psd_mae_db": (
                control_full_psd[
                    "mean_mae_db"
                ]
            ),
            "mean_bandpower_mae_db": (
                control_generic[
                    "mean_bandpower_mae_db"
                ]
            ),
            "alpha_effect_mae_db": (
                control_generic[
                    "alpha_effect"
                ][
                    "channel_mae_db"
                ]
            ),
            "scale_EO": (
                control_generic[
                    "scale"
                ][
                    "EO"
                ][
                    "normalized_std_ratio"
                ]
            ),
            "scale_EC": (
                control_generic[
                    "scale"
                ][
                    "EC"
                ][
                    "normalized_std_ratio"
                ]
            ),
            "mean_channel_correlation_r": (
                mean_corr_r(
                    control_generic
                )
            ),
            "mean_alpha_dwpli2_r": (
                mean_wpli_r(
                    control_generic
                )
            ),
            "subject_top1": (
                subject[
                    "subject_retrieval_top1_accuracy"
                ]
            ),
            "subject_top5": (
                subject[
                    "subject_retrieval_top5_accuracy"
                ]
            ),
            "subject_fingerprint_rmse_mean": (
                subject[
                    "subject_fingerprint_standardized_rmse_mean"
                ]
            ),
            "subject_fingerprint_r_median": (
                subject[
                    "subject_fingerprint_pearson_r_median"
                ]
            ),
            "subject_margin": (
                subject[
                    "subject_correct_vs_median_wrong_margin_mean"
                ]
            ),
            "fixed_wrong_win_fraction": (
                subject[
                    "subject_correct_better_than_fixed_wrong_fraction"
                ]
            ),
        },
        "existing_B2_guardrail_diagnostic": (
            eligibility
        ),
        "all_existing_B2_guards_pass_diagnostic": bool(
            eligibility[
                "eligible"
            ]
        ),
        "formal_freeze_decision_made": False,
    }

    return result


# ============================================================
# Main denoiser diagnostic
# ============================================================

def run_denoiser_diagnostics(
    *,
    models: dict[
        str,
        torch.nn.Module,
    ],
    real_z: np.ndarray,
    selected_subject_indices: list[int],
    scaler: dict[str, Any],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    (
        x0_np,
        state_np,
        subject_np,
    ) = build_probe_arrays(
        real_z=real_z,
        selected_subject_indices=(
            selected_subject_indices
        ),
    )

    epsilon_np = make_fixed_probe_noise(
        x0_np.shape
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    alpha_bar = (
        schedule_cpu[
            "alpha_cumprod"
        ]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    spectral_loss_fn = (
        PhysicalLogPSDLoss(
            means_uv=np.asarray(
                scaler[
                    "means_uv"
                ],
                dtype=np.float32,
            ),
            stds_uv=np.asarray(
                scaler[
                    "stds_uv"
                ],
                dtype=np.float32,
            ),
            sampling_rate_hz=SFREQ,
            fmin_hz=SPECTRAL_FMIN_HZ,
            fmax_hz=SPECTRAL_FMAX_HZ,
            log_eps=SPECTRAL_LOG_EPS,
            huber_beta_db=(
                SPECTRAL_HUBER_BETA_DB
            ),
            n_times=N_TIMES,
        )
        .to(
            device
        )
    )

    timestep_rows = []

    band_rows = []

    n_examples = int(
        x0_np.shape[
            0
        ]
    )

    n_batches = math.ceil(
        n_examples
        / BATCH_SIZE
    )

    print(
        f"Train-only denoiser probe examples: "
        f"{n_examples}"
    )

    print(
        f"probe batches/model/timestep: "
        f"{n_batches}"
    )

    print(
        f"probe timesteps: "
        f"{PROBE_TIMESTEPS}"
    )

    for timestep_value in PROBE_TIMESTEPS:
        print(
            "\n"
            + "-" * 80
        )

        print(
            f"TIMESTEP {timestep_value}"
        )

        print(
            "-" * 80
        )

        model_x0hat = {
            model_name: []
            for model_name in models
        }

        # Aggregators for output-space gradients and exact one-step losses.
        gradient_accumulators = {
            model_name: {
                "v_loss_weighted": 0.0,
                "spec_loss_weighted": 0.0,
                "dot": 0.0,
                "v_norm_sq": 0.0,
                "spec_norm_sq": 0.0,
                "negative_cosine_batches": 0,
                "cosine_batches": 0,
                "presented": 0,
            }
            for model_name in models
        }

        for batch_index in range(
            n_batches
        ):
            start = (
                batch_index
                * BATCH_SIZE
            )

            end = min(
                start
                + BATCH_SIZE,
                n_examples,
            )

            x0 = torch.from_numpy(
                x0_np[
                    start:
                    end
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            epsilon = torch.from_numpy(
                epsilon_np[
                    start:
                    end
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            state = torch.from_numpy(
                state_np[
                    start:
                    end
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            subject = torch.from_numpy(
                subject_np[
                    start:
                    end
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            batch_size = int(
                x0.shape[
                    0
                ]
            )

            timesteps = torch.full(
                (
                    batch_size,
                ),
                int(
                    timestep_value
                ),
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
            )

            for model_name, model in models.items():
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=MIXED_PRECISION,
                    ):
                        pred_v = model(
                            xt,
                            timesteps,
                            state,
                            subject,
                        )

                pred_v_f = pred_v.float()

                x0_hat = reconstruct_x0_from_v(
                    xt=xt,
                    v_pred=pred_v_f,
                    timesteps=timesteps,
                    alpha_bar=alpha_bar,
                )

                if not torch.isfinite(
                    x0_hat
                ).all():
                    raise RuntimeError(
                        f"Nonfinite x0_hat: "
                        f"{model_name}, t={timestep_value}"
                    )

                model_x0hat[
                    model_name
                ].append(
                    x0_hat
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                gradient_metrics = output_gradient_metrics(
                    pred_v=pred_v_f,
                    target_v=target_v,
                    xt=xt,
                    x0=x0,
                    timesteps=timesteps,
                    alpha_bar=alpha_bar,
                    spectral_loss_fn=(
                        spectral_loss_fn
                    ),
                )

                accumulator = gradient_accumulators[
                    model_name
                ]

                accumulator[
                    "v_loss_weighted"
                ] += (
                    gradient_metrics[
                        "v_loss"
                    ]
                    * batch_size
                )

                accumulator[
                    "spec_loss_weighted"
                ] += (
                    gradient_metrics[
                        "spectral_loss_db_huber"
                    ]
                    * batch_size
                )

                accumulator[
                    "dot"
                ] += (
                    gradient_metrics[
                        "grad_dot"
                    ]
                )

                accumulator[
                    "v_norm_sq"
                ] += (
                    gradient_metrics[
                        "grad_v_norm"
                    ]
                    ** 2
                )

                accumulator[
                    "spec_norm_sq"
                ] += (
                    gradient_metrics[
                        "grad_spec_norm"
                    ]
                    ** 2
                )

                cosine = gradient_metrics[
                    "grad_cosine_output_space"
                ]

                if np.isfinite(
                    cosine
                ):
                    accumulator[
                        "cosine_batches"
                    ] += 1

                    if cosine < 0.0:
                        accumulator[
                            "negative_cosine_batches"
                        ] += 1

                accumulator[
                    "presented"
                ] += batch_size

        # Full probe arrays and per-model diagnostics at this timestep.
        for model_name in models:
            x0hat_np = np.concatenate(
                model_x0hat[
                    model_name
                ],
                axis=0,
            )

            if x0hat_np.shape != x0_np.shape:
                raise RuntimeError(
                    "x0_hat probe shape mismatch."
                )

            # Restore [S, 2, 5, C, T] for exact free-generation-style
            # 1-40 Hz Welch diagnostic on one-step reconstructions.
            x0hat_structured = x0hat_np.reshape(
                EXPECTED_REAL_SHAPE
            )

            one_step_full_psd = full_log_psd_mae(
                real_z=real_z,
                synth_z=x0hat_structured,
                scaler=scaler,
            )

            one_step_signed_bias = signed_psd_bias_from_arrays(
                real_z=real_z,
                synth_z=x0hat_structured,
                scaler=scaler,
            )

            scale_ratio = normalized_scale_ratio(
                xhat=x0hat_np,
                x0=x0_np,
            )

            c_x0, c_eps = multiple_projection_coefficients(
                xhat=x0hat_np,
                x0=x0_np,
                epsilon=epsilon_np,
            )

            accumulator = gradient_accumulators[
                model_name
            ]

            presented = int(
                accumulator[
                    "presented"
                ]
            )

            total_grad_v_norm = math.sqrt(
                accumulator[
                    "v_norm_sq"
                ]
            )

            total_grad_spec_norm = math.sqrt(
                accumulator[
                    "spec_norm_sq"
                ]
            )

            grad_denominator = (
                total_grad_v_norm
                * total_grad_spec_norm
            )

            gradient_cosine = (
                accumulator[
                    "dot"
                ]
                / grad_denominator
                if grad_denominator
                > 0.0
                else float(
                    "nan"
                )
            )

            norm_ratio = (
                total_grad_spec_norm
                / max(
                    total_grad_v_norm,
                    np.finfo(
                        np.float64
                    ).eps,
                )
            )

            negative_fraction = (
                accumulator[
                    "negative_cosine_batches"
                ]
                / max(
                    accumulator[
                        "cosine_batches"
                    ],
                    1,
                )
            )

            timestep_rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "timestep": int(
                        timestep_value
                    ),
                    "alpha_bar": float(
                        alpha_bar[
                            timestep_value
                        ].item()
                    ),
                    "one_step_v_mse": float(
                        accumulator[
                            "v_loss_weighted"
                        ]
                        / presented
                    ),
                    "one_step_current_spectral_huber": float(
                        accumulator[
                            "spec_loss_weighted"
                        ]
                        / presented
                    ),
                    "one_step_welch_full_psd_mae_db": float(
                        one_step_full_psd[
                            "mean_mae_db"
                        ]
                    ),
                    "one_step_welch_psd_mae_EO_db": float(
                        one_step_full_psd[
                            "EO"
                        ][
                            "mae_db"
                        ]
                    ),
                    "one_step_welch_psd_mae_EC_db": float(
                        one_step_full_psd[
                            "EC"
                        ][
                            "mae_db"
                        ]
                    ),
                    "one_step_signed_psd_bias_db": float(
                        one_step_signed_bias
                    ),
                    "one_step_normalized_std_ratio": float(
                        scale_ratio
                    ),
                    "projection_c_x0": (
                        c_x0
                    ),
                    "projection_c_epsilon": (
                        c_eps
                    ),
                    "output_grad_cosine_v_vs_spec": float(
                        gradient_cosine
                    ),
                    "output_grad_spec_to_v_norm_ratio": float(
                        norm_ratio
                    ),
                    "output_grad_lambda001_spec_to_v_norm_ratio": float(
                        PRIMARY_LAMBDA
                        * norm_ratio
                    ),
                    "fraction_batches_with_negative_output_grad_cosine": float(
                        negative_fraction
                    ),
                    "probe_examples": (
                        presented
                    ),
                }
            )

            print(
                f"{model_name}: "
                f"v={accumulator['v_loss_weighted']/presented:.6f} "
                f"| spec={accumulator['spec_loss_weighted']/presented:.4f} "
                f"| WelchPSD={one_step_full_psd['mean_mae_db']:.4f} dB "
                f"| bias={one_step_signed_bias:+.4f} dB "
                f"| c_eps={c_eps:+.4f} "
                f"| gradcos={gradient_cosine:+.4f} "
                f"| λgrad/v={PRIMARY_LAMBDA * norm_ratio:.4f}"
            )

            # Band-specific descriptive projection.
            for band_name, (
                low_hz,
                high_hz,
            ) in DIAGNOSTIC_BANDS.items():
                xhat_band = fft_band_component(
                    x0hat_np,
                    low_hz,
                    high_hz,
                )

                x0_band = fft_band_component(
                    x0_np,
                    low_hz,
                    high_hz,
                )

                epsilon_band = fft_band_component(
                    epsilon_np,
                    low_hz,
                    high_hz,
                )

                band_c_x0, band_c_eps = multiple_projection_coefficients(
                    xhat=xhat_band,
                    x0=x0_band,
                    epsilon=epsilon_band,
                )

                band_rows.append(
                    {
                        "model": (
                            model_name
                        ),
                        "timestep": int(
                            timestep_value
                        ),
                        "band": (
                            band_name
                        ),
                        "low_hz": (
                            low_hz
                        ),
                        "high_hz": (
                            high_hz
                        ),
                        "projection_c_x0": (
                            band_c_x0
                        ),
                        "projection_c_epsilon": (
                            band_c_eps
                        ),
                    }
                )

    return (
        timestep_rows,
        band_rows,
    )


# ============================================================
# Figures
# ============================================================

def save_diagnostic_figures(
    *,
    timestep_rows: list[dict[str, Any]],
) -> None:
    model_names = [
        "B1",
        "L0_E005",
        "L001_E005",
    ]

    # PSD vs timestep.
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for model_name in model_names:
        rows = [
            row
            for row in timestep_rows
            if row[
                "model"
            ]
            == model_name
        ]

        ax.plot(
            [
                row[
                    "timestep"
                ]
                for row in rows
            ],
            [
                row[
                    "one_step_welch_full_psd_mae_db"
                ]
                for row in rows
            ],
            marker="o",
            label=(
                model_name
            ),
        )

    ax.set_xlabel(
        "Diffusion timestep"
    )

    ax.set_ylabel(
        "One-step reconstruction 1–40 Hz Welch PSD MAE (dB)"
    )

    ax.set_title(
        "s2_59 — One-Step Spectral Fidelity vs Diffusion Timestep"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PSD_VS_TIMESTEP_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Current spectral Huber monitor.
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for model_name in model_names:
        rows = [
            row
            for row in timestep_rows
            if row[
                "model"
            ]
            == model_name
        ]

        ax.plot(
            [
                row[
                    "timestep"
                ]
                for row in rows
            ],
            [
                row[
                    "one_step_current_spectral_huber"
                ]
                for row in rows
            ],
            marker="o",
            label=(
                model_name
            ),
        )

    ax.set_xlabel(
        "Diffusion timestep"
    )

    ax.set_ylabel(
        "Current PhysicalLogPSDLoss (dB Huber)"
    )

    ax.set_title(
        "s2_59 — Current Spectral Training Loss vs Timestep"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SPECTRAL_LOSS_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Output-space gradient conflict.
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for model_name in model_names:
        rows = [
            row
            for row in timestep_rows
            if row[
                "model"
            ]
            == model_name
        ]

        ax.plot(
            [
                row[
                    "timestep"
                ]
                for row in rows
            ],
            [
                row[
                    "output_grad_cosine_v_vs_spec"
                ]
                for row in rows
            ],
            marker="o",
            label=(
                model_name
            ),
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Diffusion timestep"
    )

    ax.set_ylabel(
        "Cosine(∇pred L_v, ∇pred L_spec)"
    )

    ax.set_title(
        "s2_59 — Output-Space Gradient Alignment"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        GRADIENT_FIGURE,
        dpi=180,
    )

    plt.close(
        fig
    )

    # Descriptive noise projection.
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for model_name in model_names:
        rows = [
            row
            for row in timestep_rows
            if row[
                "model"
            ]
            == model_name
        ]

        ax.plot(
            [
                row[
                    "timestep"
                ]
                for row in rows
            ],
            [
                row[
                    "projection_c_epsilon"
                ]
                for row in rows
            ],
            marker="o",
            label=(
                model_name
            ),
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Diffusion timestep"
    )

    ax.set_ylabel(
        "Descriptive c_epsilon in x0_hat ≈ c_x0*x0 + c_eps*epsilon"
    )

    ax.set_title(
        "s2_59 — Matched Noise-Projection Diagnostic"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        NOISE_PROJECTION_FIGURE,
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
        "SCENARIO 2 — s2_59 CONTROL-E5 GUARDRAILS + DENOISER DIAGNOSTICS"
    )

    print(
        "=" * 80
    )

    print(
        "\nScientific role:"
    )

    print(
        "  1) Apply the already-frozen B2 guardrails to matched λ=0 Control E5"
    )

    print(
        "     as a DIAGNOSTIC ONLY."
    )

    print(
        "  2) Compare B1 vs matched λ=0 E5 vs primary λ=0.01 E5"
    )

    print(
        "     on the exact same Train-only real/noise/timestep probe."
    )

    print(
        "\nNo training updates are executed."
    )

    print(
        "No Final Test EEG is accessed."
    )

    print(
        "No threshold is changed."
    )

    print(
        "No model is frozen by this script."
    )

    for directory in (
        METRIC_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit sources / completed prior stages
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B1 + E5 SOURCES + COMPLETED s2_58"
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
        PRIMARY_B2_E5_CHECKPOINT,
        PRIMARY_B2_RESULT,
        S2_58_SUMMARY,
        S2_58_CONTROL_E5_SYNTHETIC,
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

    with PRIMARY_B2_RESULT.open(
        "r",
        encoding="utf-8",
    ) as f:
        primary_result = json.load(
            f
        )

    with S2_58_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_58_summary = json.load(
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

    if primary_result[
        "status"
    ] != "B2_FULL_FINETUNE_20_EPOCHS_COMPLETE":
        raise RuntimeError(
            "Primary B2 full run is not complete."
        )

    if s2_58_summary[
        "status"
    ] != "S2_58_JOINT_B1_LAMBDA0_LAMBDA001_FREEGEN_AUDIT_COMPLETE":
        raise RuntimeError(
            "s2_58 joint free-generation audit is not complete."
        )

    if control_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Control result reports Final Test access."
        )

    if primary_result[
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Primary B2 result reports Final Test access."
        )

    frozen_b1_hash_before = sha256_file(
        FROZEN_B1_GENERATOR
    )

    control_e5_hash_before = sha256_file(
        CONTROL_E5_CHECKPOINT
    )

    primary_e5_hash_before = sha256_file(
        PRIMARY_B2_E5_CHECKPOINT
    )

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    frozen_b1_state_hash = state_dict_sha256(
        frozen_b1[
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

    print(
        f"frozen B1 SHA256: "
        f"{frozen_b1_hash_before}"
    )

    print(
        f"frozen B1 model-state SHA256: "
        f"{frozen_b1_state_hash}"
    )

    print(
        f"control E5 SHA256: "
        f"{control_e5_hash_before}"
    )

    print(
        f"primary B2 E5 SHA256: "
        f"{primary_e5_hash_before}"
    )

    print(
        "source audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze s2_59 protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE s2_59 DIAGNOSTIC PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_59_CONTROL_E5_GUARDRAILS_AND_DENOISER_DIAGNOSTICS_PROTOCOL_FROZEN"
        ),
        "scientific_role": (
            "Train-only mechanism diagnostic after completed s2_58. "
            "No training, no Final Test, no new model-selection threshold."
        ),
        "models": {
            "B1": (
                "exact frozen B1 RAW epoch 10"
            ),
            "L0_E005": (
                "matched lambda=0 continuation RAW epoch 5"
            ),
            "L001_E005": (
                "primary lambda=0.01 B2 RAW epoch 5"
            ),
        },
        "control_E5_guardrail_role": (
            "Apply the already-frozen s2_56 B2 eligibility guards as a "
            "diagnostic consistency check only. Control E5 is not declared "
            "a spectral-B2 candidate or frozen by this script."
        ),
        "real_probe": (
            "exact B1 V2 64-subject Real Train reference; "
            "5 distinct Train windows per subject/state"
        ),
        "probe_examples": (
            SELECTED_SUBJECT_COUNT
            * 2
            * REAL_WINDOWS_PER_SUBJECT_STATE
        ),
        "probe_noise_seed": (
            PROBE_NOISE_SEED
        ),
        "probe_noise_policy": (
            "one fixed epsilon tensor reused across all probe timesteps "
            "and all three models"
        ),
        "probe_timesteps": (
            PROBE_TIMESTEPS
        ),
        "model_forward": (
            "same AMP convention as B2 Train/Validation: FP16 autocast "
            "model forward, then float32 diagnostics"
        ),
        "current_spectral_loss": {
            "class": (
                "PhysicalLogPSDLoss from s2_53"
            ),
            "fmin_hz": (
                SPECTRAL_FMIN_HZ
            ),
            "fmax_hz": (
                SPECTRAL_FMAX_HZ
            ),
            "huber_beta_db": (
                SPECTRAL_HUBER_BETA_DB
            ),
        },
        "diagnostics": [
            "exact one-step v-MSE by timestep",
            "exact current one-step spectral Huber by timestep",
            "one-step reconstruction Welch full-PSD MAE by timestep",
            "one-step signed PSD bias by timestep",
            "one-step normalized std ratio by timestep",
            "descriptive x0/epsilon multiple-projection coefficients",
            "descriptive band-specific x0/epsilon projection coefficients",
            "output-space gradient cosine between v loss and spectral loss",
            "output-space spectral-to-v gradient norm ratio",
            "lambda=0.01 weighted output-gradient norm ratio",
        ],
        "important_interpretation_limits": [
            "noise projection coefficient c_epsilon is not expected to be zero "
            "for an ideal MMSE denoiser; compare matched models relatively",
            "gradient conflict is measured in pred_v output space, not parameter space",
            "this stage does not implement or assume the closed-form SNR/MMSE "
            "spectral-bias hypothesis from the external review",
            "this stage does not approve any new B2 redesign",
        ],
        "Train_reference_used": True,
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "training_updates_executed": False,
        "thresholds_changed": False,
        "formal_freeze_decision_made": False,
    }

    if PROTOCOL_FILE.is_file():
        with PROTOCOL_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        if existing != protocol:
            raise RuntimeError(
                "Conflicting s2_59 protocol already exists."
            )

        print(
            "identical s2_59 protocol already exists."
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
    # Step 3 — Load exact Train-only audit reference
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — LOAD EXACT B1 V2 REAL/SYNTHETIC + CONTROL E5 CACHE"
    )

    print(
        "=" * 80
    )

    (
        real_z,
        b1_synth_z,
        selected_subject_indices,
        selected_subject_ids,
    ) = load_exact_b1_reference()

    control_e5_synth_z = load_control_e5_synthetic(
        expected_subject_indices=(
            selected_subject_indices
        ),
        expected_subject_ids=(
            selected_subject_ids
        ),
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

    print(
        f"Real reference shape: "
        f"{real_z.shape}"
    )

    print(
        f"B1 synthetic shape: "
        f"{b1_synth_z.shape}"
    )

    print(
        f"Control E5 synthetic shape: "
        f"{control_e5_synth_z.shape}"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    # ========================================================
    # Step 4 — Formal diagnostic check of Control E5 guardrails
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — APPLY EXISTING B2 GUARDRAILS TO CONTROL E5 (DIAGNOSTIC ONLY)"
    )

    print(
        "=" * 80
    )

    control_guardrail = audit_control_e5_guardrails(
        real_z=real_z,
        b1_synth_z=b1_synth_z,
        control_e5_synth_z=(
            control_e5_synth_z
        ),
        scaler=scaler,
    )

    atomic_json_save(
        control_guardrail,
        CONTROL_GUARDRAIL_JSON,
    )

    print(
        "\nB1:"
    )

    for key, value in control_guardrail[
        "B1"
    ].items():
        print(
            f"  {key}: "
            f"{value}"
        )

    print(
        "\nControl E5:"
    )

    for key, value in control_guardrail[
        "Control_E5"
    ].items():
        print(
            f"  {key}: "
            f"{value}"
        )

    print(
        "\nExisting B2 guardrail criteria:"
    )

    for criterion, passed in control_guardrail[
        "existing_B2_guardrail_diagnostic"
    ][
        "criteria"
    ].items():
        print(
            f"  {criterion}: "
            f"{passed}"
        )

    print(
        "\nALL EXISTING B2 GUARDS PASS "
        "(DIAGNOSTIC): "
        f"{control_guardrail['all_existing_B2_guards_pass_diagnostic']}"
    )

    print(
        "NOTE: this does NOT freeze Control E5 and does NOT relabel it as B2."
    )

    # ========================================================
    # Step 5 — Build exact three denoiser models
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — BUILD B1 / CONTROL-E5 / PRIMARY-B2-E5 DENOISERS"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for denoiser diagnostics."
        )

    device = torch.device(
        "cuda"
    )

    models = build_model_variants(
        frozen_b1=frozen_b1,
        device=device,
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    for model_name, model in models.items():
        print(
            f"{model_name} parameters: "
            f"{sum(p.numel() for p in model.parameters()):,}"
        )

    # ========================================================
    # Step 6 — Run Train-only timestep/noise/gradient diagnostics
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — RUN MATCHED DENOISER MECHANISM DIAGNOSTICS"
    )

    print(
        "=" * 80
    )

    diagnostic_start = time.perf_counter()

    (
        timestep_rows,
        band_projection_rows,
    ) = run_denoiser_diagnostics(
        models=models,
        real_z=real_z,
        selected_subject_indices=(
            selected_subject_indices
        ),
        scaler=scaler,
        device=device,
    )

    diagnostic_seconds = (
        time.perf_counter()
        - diagnostic_start
    )

    save_csv_rows(
        timestep_rows,
        TIMESTEP_CSV,
    )

    save_csv_rows(
        band_projection_rows,
        BAND_PROJECTION_CSV,
    )

    print(
        f"\ntimestep diagnostic CSV: "
        f"{TIMESTEP_CSV}"
    )

    print(
        f"band projection CSV: "
        f"{BAND_PROJECTION_CSV}"
    )

    print(
        f"diagnostic runtime: "
        f"{diagnostic_seconds/60:.2f} min"
    )

    # ========================================================
    # Step 7 — Save figures
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — SAVE DENOISER DIAGNOSTIC FIGURES"
    )

    print(
        "=" * 80
    )

    save_diagnostic_figures(
        timestep_rows=timestep_rows,
    )

    for path in (
        PSD_VS_TIMESTEP_FIGURE,
        SPECTRAL_LOSS_FIGURE,
        GRADIENT_FIGURE,
        NOISE_PROJECTION_FIGURE,
    ):
        print(
            path
        )

    # ========================================================
    # Step 8 — Source immutability
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — VERIFY SOURCE IMMUTABILITY"
    )

    print(
        "=" * 80
    )

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != frozen_b1_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during s2_59."
        )

    if sha256_file(
        CONTROL_E5_CHECKPOINT
    ) != control_e5_hash_before:
        raise RuntimeError(
            "Control E5 checkpoint changed during s2_59."
        )

    if sha256_file(
        PRIMARY_B2_E5_CHECKPOINT
    ) != primary_e5_hash_before:
        raise RuntimeError(
            "Primary B2 E5 checkpoint changed during s2_59."
        )

    print(
        "frozen B1 unchanged: PASS"
    )

    print(
        "control E5 unchanged: PASS"
    )

    print(
        "primary B2 E5 unchanged: PASS"
    )

    # ========================================================
    # Step 9 — Final summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — SAVE s2_59 FINAL SUMMARY"
    )

    print(
        "=" * 80
    )

    # Compact per-model snapshots at selected low/mid/high timesteps.
    selected_snapshot_timesteps = [
        100,
        500,
        950,
    ]

    snapshots = {}

    for model_name in (
        "B1",
        "L0_E005",
        "L001_E005",
    ):
        snapshots[
            model_name
        ] = {}

        for timestep_value in selected_snapshot_timesteps:
            matching = [
                row
                for row in timestep_rows
                if (
                    row[
                        "model"
                    ]
                    == model_name
                    and int(
                        row[
                            "timestep"
                        ]
                    )
                    == timestep_value
                )
            ]

            if len(
                matching
            ) != 1:
                raise RuntimeError(
                    "Snapshot row mismatch."
                )

            row = matching[
                0
            ]

            snapshots[
                model_name
            ][
                str(
                    timestep_value
                )
            ] = {
                "v_mse": (
                    row[
                        "one_step_v_mse"
                    ]
                ),
                "spectral_huber": (
                    row[
                        "one_step_current_spectral_huber"
                    ]
                ),
                "welch_psd_mae_db": (
                    row[
                        "one_step_welch_full_psd_mae_db"
                    ]
                ),
                "signed_psd_bias_db": (
                    row[
                        "one_step_signed_psd_bias_db"
                    ]
                ),
                "projection_c_epsilon": (
                    row[
                        "projection_c_epsilon"
                    ]
                ),
                "output_grad_cosine": (
                    row[
                        "output_grad_cosine_v_vs_spec"
                    ]
                ),
                "lambda001_weighted_grad_norm_ratio": (
                    row[
                        "output_grad_lambda001_spec_to_v_norm_ratio"
                    ]
                ),
            }

    summary = {
        "status": (
            "S2_59_CONTROL_E5_GUARDRAILS_AND_DENOISER_DIAGNOSTICS_COMPLETE"
        ),
        "scientific_role": (
            "Train-only mechanism diagnostic after s2_58; "
            "no new training/freeze/test access."
        ),
        "control_E5_guardrail_diagnostic": (
            control_guardrail
        ),
        "probe": {
            "real_reference": str(
                B1_REAL_REFERENCE_NPZ
            ),
            "subjects": (
                SELECTED_SUBJECT_COUNT
            ),
            "examples": (
                SELECTED_SUBJECT_COUNT
                * 2
                * REAL_WINDOWS_PER_SUBJECT_STATE
            ),
            "noise_seed": (
                PROBE_NOISE_SEED
            ),
            "timesteps": (
                PROBE_TIMESTEPS
            ),
        },
        "selected_timestep_snapshots": (
            snapshots
        ),
        "runtime_seconds": (
            diagnostic_seconds
        ),
        "artifacts": {
            "protocol": str(
                PROTOCOL_FILE
            ),
            "control_guardrail_json": str(
                CONTROL_GUARDRAIL_JSON
            ),
            "timestep_csv": str(
                TIMESTEP_CSV
            ),
            "band_projection_csv": str(
                BAND_PROJECTION_CSV
            ),
            "psd_vs_timestep_figure": str(
                PSD_VS_TIMESTEP_FIGURE
            ),
            "spectral_loss_figure": str(
                SPECTRAL_LOSS_FIGURE
            ),
            "gradient_figure": str(
                GRADIENT_FIGURE
            ),
            "noise_projection_figure": str(
                NOISE_PROJECTION_FIGURE
            ),
        },
        "interpretation_limits": [
            "Control E5 passing/failing existing B2 guards is diagnostic only.",
            "c_epsilon is a relative projection diagnostic, not a zero-target leakage proof.",
            "gradient cosine/norm ratio are measured in pred_v output space.",
            "no closed-form spectral MMSE/SNR claim is tested in s2_59.",
        ],
        "governance": {
            "Train_reference_used": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "training_updates_executed": False,
            "thresholds_changed": False,
            "formal_freeze_decision_made": False,
            "frozen_B1_unchanged": True,
            "control_E5_checkpoint_unchanged": True,
            "primary_B2_E5_checkpoint_unchanged": True,
        },
        "next": (
            "Interpret Control E5 guardrails plus timestep-dependent "
            "spectral/noise/gradient diagnostics before approving a "
            "new B2 corrective protocol."
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_JSON,
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
        "s2_59 CONTROL-E5 GUARDRAIL + DENOISER DIAGNOSTICS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nControl E5 all existing B2 guards pass "
        "(diagnostic only): "
        f"{control_guardrail['all_existing_B2_guards_pass_diagnostic']}"
    )

    print(
        "\nKey interpretation rule:"
    )

    print(
        "  Compare B2 spectral E5 against BOTH B1 and matched lambda=0 E5."
    )

    print(
        "  A non-zero c_epsilon alone is NOT evidence of pathological noise leakage."
    )

    print(
        "  Output-gradient conflict is NOT automatically parameter-gradient conflict."
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
        "Thresholds changed: NO"
    )

    print(
        "Formal freeze decision made: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "INTERPRET s2_59 BEFORE DEFINING THE NEW B2 CORRECTIVE OBJECTIVE"
    )


if __name__ == "__main__":
    main()
