from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)
from s2_24_full_train_loader_sanity import (
    CachedRealTrainWindowDataset,
)
from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)
from s2_39_b0_epsilon_to_v_conversion_sanity import (
    make_v_target,
    q_sample,
)
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import (
    build_reverse_schedule,
)
from s2_47_b1_architecture_warmstart_sanity import (
    SubjectIndexedDataset,
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
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen B1 is needed for the architecture/effective contract.
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

# Exact subject-aware V2 audit reference / historical B1 synthetic cache.
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

# Strong development anchor: exact matched lambda=0 Control E5.
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

# Historical Control-E5 seed-20261001 synthetic cache from s2_58.
S2_58_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "joint_b1_lambda0_lambda001_freegen_audit_v1"
)

CONTROL_E5_SYNTHETIC_NPZ = (
    S2_58_ROOT
    / "generated"
    / "lambda0_control_epoch_005_synthetic_v1.npz"
)

# Prior diagnostic stages must be complete.
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

S2_61_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "control_e5_sampling_robustness_v1"
    / "metrics"
    / "s2_61_control_e5_sampling_robustness_summary_v1.json"
)


# ============================================================
# s2_62 output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b2"
    / "control_e5_preserving_distributional_adapter_pilot_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
GENERATED_DIR = RUN_ROOT / "generated"
LOG_DIR = RUN_ROOT / "logs"

PROTOCOL_FILE = (
    METRIC_DIR
    / "s2_62_distributional_adapter_protocol_v1.json"
)

TRAINING_CSV = (
    LOG_DIR
    / "s2_62_distributional_adapter_training_v1.csv"
)

FREEGEN_METRICS_CSV = (
    METRIC_DIR
    / "s2_62_distributional_adapter_freegen_metrics_v1.csv"
)

SUMMARY_JSON = (
    METRIC_DIR
    / "s2_62_distributional_adapter_summary_v1.json"
)


# ============================================================
# Frozen pilot design
# ============================================================

MASTER_SEED = 20261170
TRAIN_LOADER_SEED = 20261171
MAIN_NOISE_SEED = 20261172
SPEC_TIMESTEP_SEED = 20261173
SPEC_NOISE_SEED = 20261174

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128

TIMESTEPS = 1000
COSINE_S = 0.008

BATCH_SIZE = 64
PILOT_SUCCESSFUL_STEPS = 500

# Adapter-only optimizer.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

# Intentional FP32 matched pilot.
MIXED_PRECISION = False

# Preserve parent function explicitly.
DELTA_ANCHOR_WEIGHT = 0.10

# Distributional spectral auxiliary branch:
# all spectral-path samples are deliberately low-noise.
SPEC_TIMESTEP_MAX = 100

# Spectral gradient is bounded relative to the preservation gradient
# on the small adapter parameter set.
TARGET_SPECTRAL_GRAD_RATIO = 0.10
SPECTRAL_LAMBDA_CAP = 0.01

WELCH_SEGMENT_LENGTH = 256
WELCH_OVERLAP = 128
WELCH_STEP = (
    WELCH_SEGMENT_LENGTH
    - WELCH_OVERLAP
)
SPECTRAL_FMIN_HZ = 1.0
SPECTRAL_FMAX_HZ = 40.0
SPECTRAL_HUBER_BETA_DB = 1.0
SPECTRAL_LOG_EPS = 1e-12

# Tiny zero-initialized adapter.
ADAPTER_HIDDEN_CHANNELS = 64
ADAPTER_TIME_EMBED_DIM = 32
ADAPTER_STATE_EMBED_DIM = 16

# Historical free-generation seed only for mechanism pilot.
# Multi-seed confirmation is deferred until a mechanism actually advances.
SAMPLING_SEED_BASE = 20261001

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4

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

ARM_CONFIG = {
    "ADAPTER_L0": {
        "use_distributional_spectral": False,
        "role": (
            "Zero-init frozen-backbone adapter control: "
            "v-MSE + parent-output anchor only."
        ),
    },
    "ADAPTER_DIST": {
        "use_distributional_spectral": True,
        "role": (
            "Same zero-init frozen-backbone adapter plus low-t "
            "state-grouped distribution-level Welch spectral correction."
        ),
    },
}


# ============================================================
# Reproducibility / IO
# ============================================================

def set_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
        str(
            path
        )
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

        f.write(
            "\n"
        )

    os.replace(
        temp,
        path,
    )


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temp = Path(
        str(
            path
        )
        + ".tmp"
    )

    torch.save(
        payload,
        temp,
    )

    os.replace(
        temp,
        path,
    )


def atomic_npz_save(
    path: Path,
    **arrays,
) -> None:
    temp = Path(
        str(
            path
        )
        + ".tmp"
    )

    with temp.open(
        "wb"
    ) as f:
        np.savez_compressed(
            f,
            **arrays,
        )

    os.replace(
        temp,
        path,
    )


def save_csv_rows(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            f"Cannot save empty CSV: "
            f"{path}"
        )

    temp = Path(
        str(
            path
        )
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
                rows[
                    0
                ].keys()
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


def arm_checkpoint_path(
    arm_name: str,
) -> Path:
    return (
        CHECKPOINT_DIR
        / f"s2_62_{arm_name.lower()}_step500.pt"
    )


def arm_synthetic_path(
    arm_name: str,
) -> Path:
    return (
        GENERATED_DIR
        / f"s2_62_{arm_name.lower()}_step500_synthetic_v1.npz"
    )


# ============================================================
# Exact audit references
# ============================================================

def load_exact_audit_reference() -> tuple[
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
        CONTROL_E5_SYNTHETIC_NPZ,
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
        .astype(
            str
        )
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
        .astype(
            str
        )
        .tolist()
    )

    if not (
        subject_indices
        == b1_indices
        == control_indices
    ):
        raise RuntimeError(
            "Audit subject-index cohort mismatch."
        )

    if not (
        subject_ids
        == b1_ids
        == control_ids
    ):
        raise RuntimeError(
            "Audit subject-ID cohort mismatch."
        )

    b1_synth_z = b1_package[
        "epoch_010_z"
    ].astype(
        np.float32,
        copy=False,
    )

    control_synth_z = control_package[
        "synth_z"
    ].astype(
        np.float32,
        copy=False,
    )

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(
            f"Unexpected Real audit shape: "
            f"{real_z.shape}"
        )

    if b1_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected B1 synthetic shape: "
            f"{b1_synth_z.shape}"
        )

    if control_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected Control-E5 synthetic shape: "
            f"{control_synth_z.shape}"
        )

    return (
        real_z,
        b1_synth_z,
        control_synth_z,
        subject_indices,
        subject_ids,
    )


# ============================================================
# Adapter architecture
# ============================================================

def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    half = (
        embedding_dim
        // 2
    )

    if half < 1:
        raise ValueError(
            "embedding_dim must be >= 2."
        )

    frequencies = torch.exp(
        -math.log(
            max_period
        )
        * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / max(
            half
            - 1,
            1,
        )
    )

    arguments = (
        timesteps.float()[
            :,
            None,
        ]
        * frequencies[
            None,
            :,
        ]
    )

    embedding = torch.cat(
        [
            torch.sin(
                arguments
            ),
            torch.cos(
                arguments
            ),
        ],
        dim=-1,
    )

    if embedding.shape[
        1
    ] < embedding_dim:
        embedding = torch.nn.functional.pad(
            embedding,
            (
                0,
                embedding_dim
                - embedding.shape[
                    1
                ],
            ),
        )

    return embedding


class ZeroInitResidualVAdapter(
    nn.Module,
):
    """
    Small state/timestep-aware residual adapter.

    Input:
      x_t      [B, 19, 512]
      base_v   [B, 19, 512]
      t        [B]
      state    [B]

    Output:
      delta_v  [B, 19, 512]

    The final 1x1 convolution is exactly zero initialized, therefore
    delta_v == 0 at initialization and the wrapped model starts exactly
    equal to the frozen Control-E5 parent.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        input_channels = (
            2
            * N_CHANNELS
        )

        self.conv_in = nn.Conv1d(
            input_channels,
            ADAPTER_HIDDEN_CHANNELS,
            kernel_size=5,
            padding=2,
        )

        self.norm_in = nn.GroupNorm(
            num_groups=8,
            num_channels=(
                ADAPTER_HIDDEN_CHANNELS
            ),
        )

        self.conv_mid = nn.Conv1d(
            ADAPTER_HIDDEN_CHANNELS,
            ADAPTER_HIDDEN_CHANNELS,
            kernel_size=5,
            padding=2,
        )

        self.norm_mid = nn.GroupNorm(
            num_groups=8,
            num_channels=(
                ADAPTER_HIDDEN_CHANNELS
            ),
        )

        self.state_embedding = nn.Embedding(
            2,
            ADAPTER_STATE_EMBED_DIM,
        )

        condition_input_dim = (
            ADAPTER_TIME_EMBED_DIM
            + ADAPTER_STATE_EMBED_DIM
        )

        self.condition_projection = nn.Sequential(
            nn.Linear(
                condition_input_dim,
                ADAPTER_HIDDEN_CHANNELS,
            ),
            nn.SiLU(),
            nn.Linear(
                ADAPTER_HIDDEN_CHANNELS,
                ADAPTER_HIDDEN_CHANNELS,
            ),
        )

        self.conv_out = nn.Conv1d(
            ADAPTER_HIDDEN_CHANNELS,
            N_CHANNELS,
            kernel_size=1,
        )

        # Exact parent equality at zero update.
        nn.init.zeros_(
            self.conv_out.weight
        )

        nn.init.zeros_(
            self.conv_out.bias
        )

    def forward(
        self,
        x_t: torch.Tensor,
        base_v: torch.Tensor,
        timesteps: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat(
            [
                x_t,
                base_v,
            ],
            dim=1,
        )

        hidden = self.conv_in(
            values
        )

        hidden = self.norm_in(
            hidden
        )

        hidden = torch.nn.functional.silu(
            hidden
        )

        time_embedding = sinusoidal_timestep_embedding(
            timesteps,
            ADAPTER_TIME_EMBED_DIM,
        )

        state_embedding = self.state_embedding(
            state
        )

        condition = torch.cat(
            [
                time_embedding,
                state_embedding,
            ],
            dim=-1,
        )

        condition = self.condition_projection(
            condition
        )

        hidden = (
            hidden
            + condition[
                :,
                :,
                None,
            ]
        )

        hidden = self.conv_mid(
            hidden
        )

        hidden = self.norm_mid(
            hidden
        )

        hidden = torch.nn.functional.silu(
            hidden
        )

        delta_v = self.conv_out(
            hidden
        )

        return delta_v


class FrozenParentWithAdapter(
    nn.Module,
):
    """
    Sampling-compatible wrapper:
      forward(x_t, t, state, subject) -> base_v + adapter_delta.
    """

    def __init__(
        self,
        *,
        frozen_parent: nn.Module,
        adapter: ZeroInitResidualVAdapter,
    ) -> None:
        super().__init__()

        self.frozen_parent = (
            frozen_parent
        )

        self.adapter = (
            adapter
        )

        self.frozen_parent.eval()

        for parameter in self.frozen_parent.parameters():
            parameter.requires_grad_(
                False
            )

    def forward_with_delta(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        state: torch.Tensor,
        subject: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            base_v = self.frozen_parent(
                x_t,
                timesteps,
                state,
                subject,
            ).float()

        delta_v = self.adapter(
            x_t.float(),
            base_v,
            timesteps,
            state,
        )

        corrected_v = (
            base_v
            + delta_v
        )

        return (
            corrected_v,
            base_v,
            delta_v,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        state: torch.Tensor,
        subject: torch.Tensor,
    ) -> torch.Tensor:
        corrected_v, _, _ = self.forward_with_delta(
            x_t,
            timesteps,
            state,
            subject,
        )

        return corrected_v


# ============================================================
# Distribution-level differentiable Welch spectral loss
# ============================================================

class StateGroupedDistributionWelchLoss(
    nn.Module,
):
    """
    Distribution-level spectral correction.

    Important differences from the failed paired spectral objective:
      1) No per-sample x0_hat-vs-paired-x0 PSD matching.
      2) Welch-like 256/128 segmentation, close to the audit estimator.
      3) Periodograms are averaged in LINEAR power within each state group.
      4) Log transform happens only AFTER the group mean.
      5) Auxiliary spectral timesteps are explicitly low-t (0..100).

    This remains a Train-only proxy and is evaluated by full free generation.
    """

    def __init__(
        self,
        *,
        means_uv: np.ndarray,
        stds_uv: np.ndarray,
    ) -> None:
        super().__init__()

        means = torch.as_tensor(
            means_uv,
            dtype=torch.float32,
        ).view(
            1,
            N_CHANNELS,
            1,
        )

        stds = torch.as_tensor(
            stds_uv,
            dtype=torch.float32,
        ).view(
            1,
            N_CHANNELS,
            1,
        )

        self.register_buffer(
            "means_uv",
            means,
        )

        self.register_buffer(
            "stds_uv",
            stds,
        )

        window = torch.hann_window(
            WELCH_SEGMENT_LENGTH,
            periodic=True,
            dtype=torch.float32,
        )

        self.register_buffer(
            "window",
            window,
        )

        frequencies = torch.fft.rfftfreq(
            WELCH_SEGMENT_LENGTH,
            d=1.0 / SFREQ,
        )

        frequency_mask = (
            (
                frequencies
                >= SPECTRAL_FMIN_HZ
            )
            & (
                frequencies
                <= SPECTRAL_FMAX_HZ
            )
        )

        self.register_buffer(
            "frequency_mask",
            frequency_mask,
        )

        self.register_buffer(
            "frequency_grid_hz",
            frequencies,
        )

        self.window_power = float(
            torch.sum(
                window
                * window
            ).item()
        )

    def inverse_normalize(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            values.float()
            * self.stds_uv
            + self.means_uv
        )

    def welch_psd(
        self,
        values_uv: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, C, 512] -> [B, C, F]
        """

        segments = values_uv.unfold(
            dimension=-1,
            size=WELCH_SEGMENT_LENGTH,
            step=WELCH_STEP,
        )

        # Detrend='constant' equivalent.
        segments = (
            segments
            - torch.mean(
                segments,
                dim=-1,
                keepdim=True,
            )
        )

        segments = (
            segments
            * self.window[
                None,
                None,
                None,
                :,
            ]
        )

        spectrum = torch.fft.rfft(
            segments,
            dim=-1,
        )

        power = (
            spectrum.real
            * spectrum.real
            + spectrum.imag
            * spectrum.imag
        )

        power = (
            power
            / (
                SFREQ
                * self.window_power
            )
        )

        # One-sided density scaling for real signals:
        # double non-DC/non-Nyquist bins.
        if power.shape[
            -1
        ] > 2:
            power[
                ...,
                1:
                -1
            ] = (
                2.0
                * power[
                    ...,
                    1:
                    -1
                ]
            )

        power = torch.mean(
            power,
            dim=2,
        )

        return power

    def forward(
        self,
        x0_hat_z: torch.Tensor,
        x0_real_z: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        x0_hat_uv = self.inverse_normalize(
            x0_hat_z
        )

        x0_real_uv = self.inverse_normalize(
            x0_real_z
        )

        pred_psd = self.welch_psd(
            x0_hat_uv
        )

        real_psd = self.welch_psd(
            x0_real_uv
        )

        losses = []

        for state_value in (
            0,
            1,
        ):
            mask = (
                state
                == state_value
            )

            count = int(
                torch.sum(
                    mask
                ).item()
            )

            if count < 2:
                raise RuntimeError(
                    "Distributional spectral batch does not "
                    "contain at least 2 examples for each eye state."
                )

            pred_group_power = torch.mean(
                pred_psd[
                    mask
                ],
                dim=0,
            )

            real_group_power = torch.mean(
                real_psd[
                    mask
                ],
                dim=0,
            )

            pred_group_power = pred_group_power[
                ...,
                self.frequency_mask
            ]

            real_group_power = real_group_power[
                ...,
                self.frequency_mask
            ]

            pred_db = (
                10.0
                * torch.log10(
                    torch.clamp(
                        pred_group_power,
                        min=SPECTRAL_LOG_EPS,
                    )
                )
            )

            real_db = (
                10.0
                * torch.log10(
                    torch.clamp(
                        real_group_power,
                        min=SPECTRAL_LOG_EPS,
                    )
                )
            )

            loss = torch.nn.functional.smooth_l1_loss(
                pred_db,
                real_db,
                beta=SPECTRAL_HUBER_BETA_DB,
                reduction="mean",
            )

            losses.append(
                loss
            )

        return torch.mean(
            torch.stack(
                losses
            )
        )


# ============================================================
# Gradient helpers
# ============================================================

def trainable_parameters(
    adapter: nn.Module,
) -> list[
    nn.Parameter
]:
    return [
        parameter
        for parameter in adapter.parameters()
        if parameter.requires_grad
    ]


def gradient_norm_from_grads(
    grads,
) -> float:
    squared_sum = 0.0

    for grad in grads:
        if grad is None:
            continue

        if not torch.isfinite(
            grad
        ).all():
            raise RuntimeError(
                "Nonfinite gradient in s2_62."
            )

        squared_sum += float(
            torch.sum(
                grad.float()
                * grad.float()
            ).item()
        )

    return float(
        math.sqrt(
            squared_sum
        )
    )


def parameter_gradient_norm(
    parameters: list[
        nn.Parameter
    ],
) -> float:
    squared_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        if not torch.isfinite(
            grad
        ).all():
            raise RuntimeError(
                "Nonfinite parameter gradient in s2_62."
            )

        squared_sum += float(
            torch.sum(
                grad.float()
                * grad.float()
            ).item()
        )

    return float(
        math.sqrt(
            squared_sum
        )
    )


def compute_dynamic_spectral_weight(
    *,
    preserve_loss: torch.Tensor,
    spectral_loss: torch.Tensor,
    parameters: list[
        nn.Parameter
    ],
) -> tuple[
    float,
    float,
    float,
]:
    """
    Bound spectral-gradient magnitude on adapter parameters:

      ||lambda_eff * grad_spec|| <= rho * ||grad_preserve||

    with lambda_eff <= SPECTRAL_LAMBDA_CAP.

    Returns:
      lambda_eff
      preserve_grad_norm
      spec_grad_norm
    """

    preserve_grads = torch.autograd.grad(
        preserve_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    spectral_grads = torch.autograd.grad(
        spectral_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    preserve_norm = gradient_norm_from_grads(
        preserve_grads
    )

    spectral_norm = gradient_norm_from_grads(
        spectral_grads
    )

    if spectral_norm <= 0.0:
        lambda_eff = 0.0
    else:
        ratio_bound = (
            TARGET_SPECTRAL_GRAD_RATIO
            * preserve_norm
            / spectral_norm
        )

        lambda_eff = min(
            SPECTRAL_LAMBDA_CAP,
            ratio_bound,
        )

    return (
        float(
            lambda_eff
        ),
        float(
            preserve_norm
        ),
        float(
            spectral_norm
        ),
    )


# ============================================================
# Matched-stream audit
# ============================================================

def update_stream_digest(
    digest,
    *,
    x0: torch.Tensor,
    state: torch.Tensor,
    subject: torch.Tensor,
    main_t: torch.Tensor,
    main_epsilon: torch.Tensor,
    spec_t: torch.Tensor,
    spec_epsilon: torch.Tensor,
) -> None:
    digest.update(
        state.detach().cpu().numpy().tobytes()
    )

    digest.update(
        subject.detach().cpu().numpy().tobytes()
    )

    digest.update(
        main_t.detach().cpu().numpy().tobytes()
    )

    digest.update(
        spec_t.detach().cpu().numpy().tobytes()
    )

    digest.update(
        x0[
            ...,
            :8
        ]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    )

    digest.update(
        main_epsilon[
            ...,
            :8
        ]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    )

    digest.update(
        spec_epsilon[
            ...,
            :8
        ]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    )


# ============================================================
# Model / adapter builders
# ============================================================

def build_frozen_parent(
    *,
    frozen_b1: dict[str, Any],
    control_e5: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    parent = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=(
            control_e5[
                "model_state_dict"
            ]
        ),
        device=device,
    )

    parent.eval()

    for parameter in parent.parameters():
        parameter.requires_grad_(
            False
        )

    return parent


def build_zero_init_adapter(
    *,
    device: torch.device,
) -> ZeroInitResidualVAdapter:
    adapter = ZeroInitResidualVAdapter().to(
        device
    )

    adapter.train()

    # Explicit exact-zero output sanity.
    if not torch.all(
        adapter.conv_out.weight
        == 0
    ):
        raise RuntimeError(
            "Adapter final weight is not zero initialized."
        )

    if not torch.all(
        adapter.conv_out.bias
        == 0
    ):
        raise RuntimeError(
            "Adapter final bias is not zero initialized."
        )

    return adapter


# ============================================================
# Pilot training
# ============================================================

def run_arm_training(
    *,
    arm_name: str,
    arm_config: dict[str, Any],
    frozen_b1: dict[str, Any],
    control_e5: dict[str, Any],
    train_dataset: SubjectIndexedDataset,
    scaler: dict[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TRAIN ARM {arm_name}"
    )

    print(
        "=" * 80
    )

    print(
        f"role: "
        f"{arm_config['role']}"
    )

    print(
        "frozen parent: Control E5"
    )

    print(
        f"use distributional spectral objective: "
        f"{arm_config['use_distributional_spectral']}"
    )

    # Full reset to guarantee matched loader/noise streams and exact
    # identical adapter initialization between arms.
    set_seed(
        MASTER_SEED
    )

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        TRAIN_LOADER_SEED
    )

    main_noise_generator = torch.Generator(
        device="cuda"
    )

    main_noise_generator.manual_seed(
        MAIN_NOISE_SEED
    )

    spec_t_generator = torch.Generator(
        device="cuda"
    )

    spec_t_generator.manual_seed(
        SPEC_TIMESTEP_SEED
    )

    spec_noise_generator = torch.Generator(
        device="cuda"
    )

    spec_noise_generator.manual_seed(
        SPEC_NOISE_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    if len(
        train_loader
    ) != 748:
        raise RuntimeError(
            "Expected 748 Train batches."
        )

    frozen_parent = build_frozen_parent(
        frozen_b1=frozen_b1,
        control_e5=control_e5,
        device=device,
    )

    adapter = build_zero_init_adapter(
        device=device
    )

    adapter_parameter_list = trainable_parameters(
        adapter
    )

    adapter_parameter_count = sum(
        parameter.numel()
        for parameter in adapter_parameter_list
    )

    optimizer = torch.optim.AdamW(
        adapter_parameter_list,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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

    distributional_loss_fn = (
        StateGroupedDistributionWelchLoss(
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
        )
        .to(
            device
        )
    )

    # Zero-update exact parent equality sanity on first Train batch.
    sanity_loader_generator = torch.Generator()

    sanity_loader_generator.manual_seed(
        TRAIN_LOADER_SEED
    )

    sanity_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=sanity_loader_generator,
    )

    sanity_batch = next(
        iter(
            sanity_loader
        )
    )

    sanity_x0 = sanity_batch[
        "x"
    ].to(
        device,
        dtype=torch.float32,
    )

    sanity_state = sanity_batch[
        "y"
    ].to(
        device,
        dtype=torch.long,
    )

    sanity_subject = sanity_batch[
        "subject_index"
    ].to(
        device,
        dtype=torch.long,
    )

    sanity_t = torch.full(
        (
            sanity_x0.shape[
                0
            ],
        ),
        100,
        device=device,
        dtype=torch.long,
    )

    sanity_epsilon = torch.zeros_like(
        sanity_x0
    )

    sanity_xt = q_sample(
        sanity_x0,
        sanity_epsilon,
        sanity_t,
        alpha_bar,
    )

    with torch.no_grad():
        sanity_base_v = frozen_parent(
            sanity_xt,
            sanity_t,
            sanity_state,
            sanity_subject,
        ).float()

        sanity_delta = adapter(
            sanity_xt,
            sanity_base_v,
            sanity_t,
            sanity_state,
        )

    max_abs_zero_delta = float(
        torch.max(
            torch.abs(
                sanity_delta
            )
        ).item()
    )

    if max_abs_zero_delta != 0.0:
        raise RuntimeError(
            "Zero-init adapter does not exactly reproduce parent."
        )

    print(
        f"adapter parameters: "
        f"{adapter_parameter_count:,}"
    )

    print(
        f"zero-init max |delta_v|: "
        f"{max_abs_zero_delta:.3e}"
    )

    stream_digest = hashlib.sha256()

    successful_steps = 0
    presented_examples = 0

    v_sum = 0.0
    delta_anchor_sum = 0.0
    distributional_spec_sum = 0.0
    total_sum = 0.0

    lambda_values = []
    preserve_grad_norm_values = []
    spec_grad_norm_values = []
    preclip_grad_norm_values = []

    loader_iterator = iter(
        train_loader
    )

    arm_start = time.perf_counter()

    use_distributional_spectral = bool(
        arm_config[
            "use_distributional_spectral"
        ]
    )

    while successful_steps < PILOT_SUCCESSFUL_STEPS:
        try:
            batch = next(
                loader_iterator
            )
        except StopIteration:
            raise RuntimeError(
                "s2_62 exhausted one Train epoch before 500 steps."
            )

        x0 = batch[
            "x"
        ].to(
            device,
            non_blocking=True,
        ).float()

        state = batch[
            "y"
        ].to(
            device,
            non_blocking=True,
        ).long()

        subject = batch[
            "subject_index"
        ].to(
            device,
            non_blocking=True,
        ).long()

        current_batch_size = int(
            x0.shape[
                0
            ]
        )

        # ----------------------------------------------------
        # Main preservation path: full diffusion timestep range
        # ----------------------------------------------------

        main_t = torch.randint(
            0,
            TIMESTEPS,
            (
                current_batch_size,
            ),
            device=device,
            dtype=torch.long,
            generator=main_noise_generator,
        )

        main_epsilon = torch.randn(
            x0.shape,
            device=device,
            dtype=x0.dtype,
            generator=main_noise_generator,
        )

        main_xt = q_sample(
            x0,
            main_epsilon,
            main_t,
            alpha_bar,
        )

        main_target_v = make_v_target(
            x0,
            main_epsilon,
            main_t,
            alpha_bar,
        ).float()

        with torch.no_grad():
            main_base_v = frozen_parent(
                main_xt,
                main_t,
                state,
                subject,
            ).float()

        main_delta_v = adapter(
            main_xt,
            main_base_v,
            main_t,
            state,
        )

        main_corrected_v = (
            main_base_v
            + main_delta_v
        )

        v_loss = torch.mean(
            (
                main_corrected_v
                - main_target_v
            )
            ** 2
        )

        delta_anchor_loss = torch.mean(
            main_delta_v
            * main_delta_v
        )

        preserve_loss = (
            v_loss
            + DELTA_ANCHOR_WEIGHT
            * delta_anchor_loss
        )

        # ----------------------------------------------------
        # Auxiliary distributional spectral path:
        # every example receives a low-noise t in [0, 100].
        # ----------------------------------------------------

        spec_t = torch.randint(
            0,
            SPEC_TIMESTEP_MAX
            + 1,
            (
                current_batch_size,
            ),
            device=device,
            dtype=torch.long,
            generator=spec_t_generator,
        )

        spec_epsilon = torch.randn(
            x0.shape,
            device=device,
            dtype=x0.dtype,
            generator=spec_noise_generator,
        )

        update_stream_digest(
            stream_digest,
            x0=x0,
            state=state,
            subject=subject,
            main_t=main_t,
            main_epsilon=main_epsilon,
            spec_t=spec_t,
            spec_epsilon=spec_epsilon,
        )

        spec_xt = q_sample(
            x0,
            spec_epsilon,
            spec_t,
            alpha_bar,
        )

        with torch.no_grad():
            spec_base_v = frozen_parent(
                spec_xt,
                spec_t,
                state,
                subject,
            ).float()

        spec_delta_v = adapter(
            spec_xt,
            spec_base_v,
            spec_t,
            state,
        )

        spec_corrected_v = (
            spec_base_v
            + spec_delta_v
        )

        spec_x0_hat = (
            torch.sqrt(
                alpha_bar[
                    spec_t
                ]
            )[
                :,
                None,
                None,
            ]
            * spec_xt
            - torch.sqrt(
                1.0
                - alpha_bar[
                    spec_t
                ]
            )[
                :,
                None,
                None,
            ]
            * spec_corrected_v
        )

        distributional_spec_loss = (
            distributional_loss_fn(
                spec_x0_hat,
                x0,
                state,
            )
        )

        if not (
            torch.isfinite(
                v_loss
            )
            and torch.isfinite(
                delta_anchor_loss
            )
            and torch.isfinite(
                preserve_loss
            )
            and torch.isfinite(
                distributional_spec_loss
            )
        ):
            raise RuntimeError(
                f"Nonfinite s2_62 loss in arm {arm_name}."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_distributional_spectral:
            (
                lambda_eff,
                preserve_grad_norm,
                spec_grad_norm,
            ) = compute_dynamic_spectral_weight(
                preserve_loss=(
                    preserve_loss
                ),
                spectral_loss=(
                    distributional_spec_loss
                ),
                parameters=(
                    adapter_parameter_list
                ),
            )
        else:
            # Still compute the spectral monitor but give it no gradient.
            preserve_grads = torch.autograd.grad(
                preserve_loss,
                adapter_parameter_list,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            preserve_grad_norm = gradient_norm_from_grads(
                preserve_grads
            )

            spec_grad_norm = float(
                "nan"
            )

            lambda_eff = 0.0

        total_loss = (
            preserve_loss
            + float(
                lambda_eff
            )
            * distributional_spec_loss
        )

        total_loss.backward()

        preclip_grad_norm = parameter_gradient_norm(
            adapter_parameter_list
        )

        clip_return = torch.nn.utils.clip_grad_norm_(
            adapter_parameter_list,
            GRADIENT_CLIP_NORM,
        )

        if not torch.isfinite(
            clip_return
        ):
            raise RuntimeError(
                f"Nonfinite adapter clip norm in arm {arm_name}."
            )

        optimizer.step()

        successful_steps += 1
        presented_examples += current_batch_size

        v_sum += (
            float(
                v_loss.item()
            )
            * current_batch_size
        )

        delta_anchor_sum += (
            float(
                delta_anchor_loss.item()
            )
            * current_batch_size
        )

        distributional_spec_sum += (
            float(
                distributional_spec_loss.item()
            )
            * current_batch_size
        )

        total_sum += (
            float(
                total_loss.item()
            )
            * current_batch_size
        )

        lambda_values.append(
            float(
                lambda_eff
            )
        )

        preserve_grad_norm_values.append(
            float(
                preserve_grad_norm
            )
        )

        spec_grad_norm_values.append(
            float(
                spec_grad_norm
            )
        )

        preclip_grad_norm_values.append(
            float(
                preclip_grad_norm
            )
        )

        if (
            successful_steps == 1
            or successful_steps
            % 50
            == 0
            or successful_steps
            == PILOT_SUCCESSFUL_STEPS
        ):
            finite_spec_grad_values = [
                value
                for value in spec_grad_norm_values
                if np.isfinite(
                    value
                )
            ]

            mean_spec_grad = (
                float(
                    np.mean(
                        finite_spec_grad_values
                    )
                )
                if finite_spec_grad_values
                else float(
                    "nan"
                )
            )

            print(
                f"{arm_name} "
                f"| step={successful_steps:>3}/"
                f"{PILOT_SUCCESSFUL_STEPS} "
                f"| v={v_sum/presented_examples:.6f} "
                f"| delta={delta_anchor_sum/presented_examples:.6f} "
                f"| dist-spec={distributional_spec_sum/presented_examples:.4f} "
                f"| lambda-eff={np.mean(lambda_values):.6f} "
                f"| preserve-grad={np.mean(preserve_grad_norm_values):.4f} "
                f"| spec-grad={mean_spec_grad:.4f}"
            )

    arm_seconds = (
        time.perf_counter()
        - arm_start
    )

    stream_sha256 = stream_digest.hexdigest()

    adapter_state = adapter.state_dict()

    checkpoint = {
        "status": (
            "S2_62_CONTROL_E5_PRESERVING_ADAPTER_PILOT_CHECKPOINT"
        ),
        "arm": (
            arm_name
        ),
        "parent_checkpoint": str(
            CONTROL_E5_CHECKPOINT
        ),
        "parent_model_state_sha256": (
            control_e5[
                "model_state_sha256"
            ]
        ),
        "adapter_architecture": (
            "ZeroInitResidualVAdapter"
        ),
        "adapter_parameter_count": (
            adapter_parameter_count
        ),
        "adapter_state_dict": (
            adapter_state
        ),
        "adapter_state_sha256": (
            state_dict_sha256(
                adapter_state
            )
        ),
        "successful_optimizer_steps": (
            successful_steps
        ),
        "learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "delta_anchor_weight": (
            DELTA_ANCHOR_WEIGHT
        ),
        "use_distributional_spectral": (
            use_distributional_spectral
        ),
        "spec_timestep_max": (
            SPEC_TIMESTEP_MAX
        ),
        "target_spectral_grad_ratio": (
            TARGET_SPECTRAL_GRAD_RATIO
        ),
        "spectral_lambda_cap": (
            SPECTRAL_LAMBDA_CAP
        ),
        "matched_stream_sha256": (
            stream_sha256
        ),
        "training_summary": {
            "mean_v_mse": float(
                v_sum
                / presented_examples
            ),
            "mean_delta_anchor_mse": float(
                delta_anchor_sum
                / presented_examples
            ),
            "mean_distributional_spectral_loss": float(
                distributional_spec_sum
                / presented_examples
            ),
            "mean_effective_lambda": float(
                np.mean(
                    lambda_values
                )
            ),
            "max_effective_lambda": float(
                np.max(
                    lambda_values
                )
            ),
            "mean_preserve_grad_norm": float(
                np.mean(
                    preserve_grad_norm_values
                )
            ),
            "mean_spec_grad_norm": (
                float(
                    np.mean(
                        [
                            value
                            for value in spec_grad_norm_values
                            if np.isfinite(
                                value
                            )
                        ]
                    )
                )
                if any(
                    np.isfinite(
                        value
                    )
                    for value in spec_grad_norm_values
                )
                else None
            ),
            "mean_preclip_adapter_grad_norm": float(
                np.mean(
                    preclip_grad_norm_values
                )
            ),
            "runtime_seconds": (
                arm_seconds
            ),
        },
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "formal_candidate_selected": False,
    }

    history_row = {
        "arm": (
            arm_name
        ),
        "use_distributional_spectral": (
            use_distributional_spectral
        ),
        "successful_steps": (
            successful_steps
        ),
        "adapter_parameter_count": (
            adapter_parameter_count
        ),
        "mean_v_mse": (
            checkpoint[
                "training_summary"
            ][
                "mean_v_mse"
            ]
        ),
        "mean_delta_anchor_mse": (
            checkpoint[
                "training_summary"
            ][
                "mean_delta_anchor_mse"
            ]
        ),
        "mean_distributional_spectral_loss": (
            checkpoint[
                "training_summary"
            ][
                "mean_distributional_spectral_loss"
            ]
        ),
        "mean_effective_lambda": (
            checkpoint[
                "training_summary"
            ][
                "mean_effective_lambda"
            ]
        ),
        "max_effective_lambda": (
            checkpoint[
                "training_summary"
            ][
                "max_effective_lambda"
            ]
        ),
        "mean_preserve_grad_norm": (
            checkpoint[
                "training_summary"
            ][
                "mean_preserve_grad_norm"
            ]
        ),
        "mean_spec_grad_norm": (
            ""
            if checkpoint[
                "training_summary"
            ][
                "mean_spec_grad_norm"
            ]
            is None
            else checkpoint[
                "training_summary"
            ][
                "mean_spec_grad_norm"
            ]
        ),
        "mean_preclip_adapter_grad_norm": (
            checkpoint[
                "training_summary"
            ][
                "mean_preclip_adapter_grad_norm"
            ]
        ),
        "runtime_seconds": (
            arm_seconds
        ),
        "matched_stream_sha256": (
            stream_sha256
        ),
        "adapter_state_sha256": (
            checkpoint[
                "adapter_state_sha256"
            ]
        ),
    }

    del frozen_parent
    del adapter

    torch.cuda.empty_cache()

    return (
        checkpoint,
        history_row,
    )


# ============================================================
# Sampling wrapper / generation
# ============================================================

def build_sampling_wrapper(
    *,
    frozen_b1: dict[str, Any],
    control_e5: dict[str, Any],
    adapter_checkpoint: dict[str, Any],
    device: torch.device,
) -> FrozenParentWithAdapter:
    frozen_parent = build_frozen_parent(
        frozen_b1=frozen_b1,
        control_e5=control_e5,
        device=device,
    )

    adapter = ZeroInitResidualVAdapter().to(
        device
    )

    adapter.load_state_dict(
        adapter_checkpoint[
            "adapter_state_dict"
        ],
        strict=True,
    )

    adapter.eval()

    wrapper = FrozenParentWithAdapter(
        frozen_parent=frozen_parent,
        adapter=adapter,
    ).to(
        device
    )

    wrapper.eval()

    return wrapper


def generate_arm_samples(
    *,
    arm_name: str,
    arm_checkpoint: dict[str, Any],
    frozen_b1: dict[str, Any],
    control_e5: dict[str, Any],
    selected_subjects: list[int],
    selected_subject_ids: list[str],
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    checkpoint_path = arm_checkpoint_path(
        arm_name
    )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    output_path = arm_synthetic_path(
        arm_name
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
                f"Cached cohort mismatch for {arm_name}."
            )

        if (
            package[
                "selected_subject_ids"
            ]
            .astype(
                str
            )
            .tolist()
            != selected_subject_ids
        ):
            raise RuntimeError(
                f"Cached subject-ID mismatch for {arm_name}."
            )

        if str(
            package[
                "checkpoint_sha256"
            ][0]
        ) != checkpoint_hash:
            raise RuntimeError(
                f"Cached checkpoint hash mismatch for {arm_name}."
            )

        if int(
            package[
                "sampling_seed"
            ][0]
        ) != SAMPLING_SEED_BASE:
            raise RuntimeError(
                f"Cached sampling seed mismatch for {arm_name}."
            )

        synth_z = package[
            "synth_z"
        ].astype(
            np.float32,
            copy=False,
        )

        if synth_z.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError(
                f"Cached synthetic shape mismatch for {arm_name}."
            )

        print(
            f"{arm_name}: validated synthetic cache -> REUSE"
        )

        return synth_z

    wrapper = build_sampling_wrapper(
        frozen_b1=frozen_b1,
        control_e5=control_e5,
        adapter_checkpoint=(
            arm_checkpoint
        ),
        device=device,
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"FREE GENERATION — {arm_name}"
    )

    print(
        "-" * 80
    )

    start = time.perf_counter()

    eo = sample_b1_candidate_state(
        model=wrapper,
        selected_subjects=(
            selected_subjects
        ),
        eye_state_value=0,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        base_seed=(
            SAMPLING_SEED_BASE
        ),
    )

    ec = sample_b1_candidate_state(
        model=wrapper,
        selected_subjects=(
            selected_subjects
        ),
        eye_state_value=1,
        reverse_schedule=(
            reverse_schedule
        ),
        device=device,
        base_seed=(
            SAMPLING_SEED_BASE
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
            f"Unexpected synthetic shape for {arm_name}: "
            f"{synth_z.shape}"
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
        checkpoint_sha256=np.asarray(
            [
                checkpoint_hash
            ],
            dtype="U64",
        ),
        sampling_seed=np.asarray(
            [
                SAMPLING_SEED_BASE
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
        f"{arm_name} generation: "
        f"{generation_seconds/60:.2f} min"
    )

    del wrapper

    torch.cuda.empty_cache()

    return synth_z


# ============================================================
# Evaluation
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
    variant_name: str,
    bundle: dict[str, Any],
    b1_bundle: dict[str, Any],
    parent_bundle: dict[str, Any],
) -> dict[str, Any]:
    generic = bundle[
        "generic"
    ]

    subject = generic[
        "subject_identity"
    ]

    full_psd = float(
        bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    b1_psd = float(
        b1_bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    parent_psd = float(
        parent_bundle[
            "full_log_psd"
        ][
            "mean_mae_db"
        ]
    )

    if variant_name == "B1":
        guard = None
    else:
        guard = evaluate_eligibility(
            candidate=bundle,
            b1_baseline=b1_bundle,
        )

    return {
        "variant": (
            variant_name
        ),
        "full_psd_mae_db": (
            full_psd
        ),
        "improvement_vs_B1_fraction": float(
            (
                b1_psd
                - full_psd
            )
            / b1_psd
        ),
        "improvement_vs_ControlE5_fraction": float(
            (
                parent_psd
                - full_psd
            )
            / parent_psd
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
            np.nanmean(
                [
                    generic[
                        "spatial"
                    ][
                        "EO"
                    ][
                        "correlation_r"
                    ],
                    generic[
                        "spatial"
                    ][
                        "EC"
                    ][
                        "correlation_r"
                    ],
                ]
            )
        ),
        "mean_alpha_dwpli2_r": float(
            np.nanmean(
                [
                    generic[
                        "spatial"
                    ][
                        "EO"
                    ][
                        "alpha_dwpli2_r"
                    ],
                    generic[
                        "spatial"
                    ][
                        "EC"
                    ][
                        "alpha_dwpli2_r"
                    ],
                ]
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
        "passes_existing_B2_guards_vs_B1": (
            ""
            if guard is None
            else bool(
                guard[
                    "eligible"
                ]
            )
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — s2_62 CONTROL-E5-PRESERVING DISTRIBUTIONAL ADAPTER PILOT"
    )

    print(
        "=" * 80
    )

    print(
        "\nScientific question:"
    )

    print(
        "  Can a tiny zero-initialized residual adapter improve the"
    )

    print(
        "  already-strong Control-E5 generator without allowing the"
    )

    print(
        "  8.5M-parameter diffusion backbone to drift?"
    )

    print(
        "\nObjective-family change:"
    )

    print(
        "  FAILED family removed: paired per-sample spectral matching."
    )

    print(
        "  NEW family: state-grouped distribution-level Welch correction"
    )

    print(
        "  on a dedicated low-t auxiliary path."
    )

    print(
        "\nArms:"
    )

    for arm_name, arm_config in ARM_CONFIG.items():
        print(
            f"  {arm_name}: "
            f"{arm_config['role']}"
        )

    print(
        "\nNo backbone parameter is trainable."
    )

    print(
        "No Validation EEG is accessed."
    )

    print(
        "No Final Test EEG is accessed."
    )

    print(
        "Passing this pilot does NOT freeze a new B2."
    )

    for directory in (
        METRIC_DIR,
        CHECKPOINT_DIR,
        GENERATED_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Step 1 — Audit prerequisite evidence
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT B1 / CONTROL-E5 / s2_59 / s2_60 / s2_61"
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
        CONTROL_E5_SYNTHETIC_NPZ,
        S2_59_SUMMARY,
        S2_60_SUMMARY,
        S2_61_SUMMARY,
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

    with S2_61_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        s2_61_summary = json.load(
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
            "Control continuation run is not complete."
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
            "s2_60 unexpectedly recommended the old gated family."
        )

    if s2_61_summary[
        "status"
    ] != "S2_61_CONTROL_E5_SAMPLING_ROBUSTNESS_COMPLETE":
        raise RuntimeError(
            "s2_61 is not complete."
        )

    if s2_61_summary[
        "development_anchor_robustly_supported"
    ] is not True:
        raise RuntimeError(
            "Control E5 is not robustly supported as development anchor."
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

    control_e5_state_hash = state_dict_sha256(
        control_e5[
            "model_state_dict"
        ]
    )

    if (
        control_e5_state_hash
        != control_e5[
            "model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Control-E5 model-state hash mismatch."
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
        f"Control E5 model-state SHA256: "
        f"{control_e5_state_hash}"
    )

    print(
        "prerequisite audit: PASS"
    )

    # ========================================================
    # Step 2 — Freeze s2_62 protocol
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — FREEZE s2_62 ADAPTER PILOT PROTOCOL"
    )

    print(
        "=" * 80
    )

    protocol = {
        "status": (
            "S2_62_CONTROL_E5_PRESERVING_DISTRIBUTIONAL_ADAPTER_PROTOCOL_FROZEN"
        ),
        "scientific_role": (
            "Matched Train-only corrective-adapter mechanism pilot. "
            "Not a final B2 freeze."
        ),
        "parent": {
            "model": (
                "exact matched lambda=0 Control E5"
            ),
            "checkpoint": str(
                CONTROL_E5_CHECKPOINT
            ),
            "file_sha256": (
                control_e5_hash_before
            ),
            "model_state_sha256": (
                control_e5_state_hash
            ),
            "backbone_trainable": False,
        },
        "adapter": {
            "architecture": (
                "ZeroInitResidualVAdapter"
            ),
            "input": (
                "concat(x_t, frozen_parent_v)"
            ),
            "conditioning": (
                "diffusion timestep + eye state; no subject-specific adapter parameters"
            ),
            "hidden_channels": (
                ADAPTER_HIDDEN_CHANNELS
            ),
            "zero_init_final_projection": True,
            "zero_update_exact_parent_equality_required": True,
        },
        "arms": (
            ARM_CONFIG
        ),
        "training": {
            "successful_steps_per_arm": (
                PILOT_SUCCESSFUL_STEPS
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "adapter_learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),
            "mixed_precision": (
                MIXED_PRECISION
            ),
            "delta_anchor_weight": (
                DELTA_ANCHOR_WEIGHT
            ),
            "same_initial_adapter_state_across_arms": True,
            "same_loader_main_noise_spec_noise_streams_across_arms": True,
        },
        "distributional_spectral_objective": {
            "paired_per_sample_target": False,
            "spectral_path_timestep_range": [
                0,
                SPEC_TIMESTEP_MAX,
            ],
            "estimator": (
                "differentiable Welch-like Hann 256 / overlap 128 / density"
            ),
            "aggregation_order": (
                "periodograms -> average segments -> average LINEAR power "
                "within eye-state group -> log10 -> SmoothL1"
            ),
            "grouping": (
                "eye state EO/EC"
            ),
            "frequency_range_hz": [
                SPECTRAL_FMIN_HZ,
                SPECTRAL_FMAX_HZ,
            ],
            "huber_beta_db": (
                SPECTRAL_HUBER_BETA_DB
            ),
            "gradient_balance": {
                "target_spec_to_preserve_ratio": (
                    TARGET_SPECTRAL_GRAD_RATIO
                ),
                "lambda_cap": (
                    SPECTRAL_LAMBDA_CAP
                ),
                "measured_on": (
                    "adapter parameters only"
                ),
            },
        },
        "free_generation_audit": {
            "sampling_seed": (
                SAMPLING_SEED_BASE
            ),
            "sampler": (
                "1000-step ancestral DDPM / v-prediction / FP32"
            ),
            "subjects": (
                SELECTED_SUBJECT_COUNT
            ),
            "real_windows_per_subject_state": (
                REAL_WINDOWS_PER_SUBJECT_STATE
            ),
            "synthetic_windows_per_subject_state": (
                SYNTHETIC_WINDOWS_PER_SUBJECT_STATE
            ),
            "exact_existing_metric_helpers_reused": True,
        },
        "pilot_advancement_rule": (
            "ADAPTER_DIST advances only if: "
            "(1) it passes ALL existing frozen B2 guards versus frozen B1, "
            "(2) its full 1-40 Hz PSD MAE is lower than the robust Control-E5 parent, "
            "and (3) its full PSD MAE is lower than matched ADAPTER_L0. "
            "Passing this rule only approves a separately frozen follow-up protocol."
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
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
                "Conflicting s2_62 protocol already exists."
            )

        print(
            "identical s2_62 protocol already exists."
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
    # Step 3 — Train dataset + exact audit reference
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD REAL TRAIN DATA + EXACT AUDIT REFERENCE"
    )

    print(
        "=" * 80
    )

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
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

    train_records, train_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )

    if len(
        train_records
    ) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(
        train_mapping
    ) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Unexpected subject count."
        )

    train_base_dataset = CachedRealTrainWindowDataset(
        records=train_records,
        scaler=scaler,
        max_open_files=8,
    )

    train_dataset = SubjectIndexedDataset(
        base_dataset=train_base_dataset,
        records=train_records,
    )

    (
        real_z,
        b1_synth_z,
        control_parent_synth_z,
        selected_subject_indices,
        selected_subject_ids,
    ) = load_exact_audit_reference()

    print(
        f"Train windows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Audit subjects: "
        f"{len(selected_subject_indices)}"
    )

    # ========================================================
    # Step 4 — Run matched adapter pilot arms
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — RUN MATCHED ZERO-INIT ADAPTER PILOT ARMS"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for s2_62."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    arm_checkpoints = {}
    history_rows = []

    for arm_name, arm_config in ARM_CONFIG.items():
        (
            checkpoint,
            history_row,
        ) = run_arm_training(
            arm_name=arm_name,
            arm_config=arm_config,
            frozen_b1=frozen_b1,
            control_e5=control_e5,
            train_dataset=train_dataset,
            scaler=scaler,
            device=device,
        )

        atomic_torch_save(
            checkpoint,
            arm_checkpoint_path(
                arm_name
            ),
        )

        arm_checkpoints[
            arm_name
        ] = checkpoint

        history_rows.append(
            history_row
        )

        print(
            f"checkpoint: "
            f"{arm_checkpoint_path(arm_name)}"
        )

    save_csv_rows(
        history_rows,
        TRAINING_CSV,
    )

    stream_hashes = {
        row[
            "arm"
        ]: row[
            "matched_stream_sha256"
        ]
        for row in history_rows
    }

    if len(
        set(
            stream_hashes.values()
        )
    ) != 1:
        raise RuntimeError(
            "Matched training streams differ across s2_62 arms."
        )

    matched_stream_hash = next(
        iter(
            stream_hashes.values()
        )
    )

    print(
        "\nMatched training-stream audit: PASS"
    )

    print(
        f"stream SHA256: "
        f"{matched_stream_hash}"
    )

    # ========================================================
    # Step 5 — Matched free generation
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — 1000-STEP FREE GENERATION FOR BOTH ADAPTER ARMS"
    )

    print(
        "=" * 80
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    arm_synth = {}

    for arm_name in ARM_CONFIG:
        arm_synth[
            arm_name
        ] = generate_arm_samples(
            arm_name=arm_name,
            arm_checkpoint=(
                arm_checkpoints[
                    arm_name
                ]
            ),
            frozen_b1=frozen_b1,
            control_e5=control_e5,
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
    # Step 6 — Existing metric audit
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — FREE-GENERATION FIDELITY AUDIT"
    )

    print(
        "=" * 80
    )

    bundles = {
        "B1": evaluate_bundle(
            real_z=real_z,
            synth_z=b1_synth_z,
            scaler=scaler,
        ),
        "CONTROL_E5_PARENT": evaluate_bundle(
            real_z=real_z,
            synth_z=control_parent_synth_z,
            scaler=scaler,
        ),
    }

    for arm_name, synth_z in arm_synth.items():
        bundles[
            arm_name
        ] = evaluate_bundle(
            real_z=real_z,
            synth_z=synth_z,
            scaler=scaler,
        )

    b1_bundle = bundles[
        "B1"
    ]

    parent_bundle = bundles[
        "CONTROL_E5_PARENT"
    ]

    metric_rows = []

    for variant_name in (
        "B1",
        "CONTROL_E5_PARENT",
        "ADAPTER_L0",
        "ADAPTER_DIST",
    ):
        row = flatten_metrics(
            variant_name=variant_name,
            bundle=(
                bundles[
                    variant_name
                ]
            ),
            b1_bundle=b1_bundle,
            parent_bundle=parent_bundle,
        )

        metric_rows.append(
            row
        )

        print(
            "\n"
            f"{variant_name}:"
        )

        print(
            f"  full PSD MAE: "
            f"{row['full_psd_mae_db']:.6f} dB"
        )

        print(
            f"  vs B1 improvement: "
            f"{100.0 * row['improvement_vs_B1_fraction']:+.2f}%"
        )

        print(
            f"  vs Control-E5 improvement: "
            f"{100.0 * row['improvement_vs_ControlE5_fraction']:+.2f}%"
        )

        print(
            f"  mean bandpower MAE: "
            f"{row['mean_bandpower_mae_db']:.6f} dB"
        )

        print(
            f"  alpha-effect MAE: "
            f"{row['alpha_effect_mae_db']:.6f} dB"
        )

        print(
            f"  subject Top1: "
            f"{row['subject_top1']:.6f}"
        )

        print(
            f"  existing B2 guards vs B1: "
            f"{row['passes_existing_B2_guards_vs_B1']}"
        )

    save_csv_rows(
        metric_rows,
        FREEGEN_METRICS_CSV,
    )

    # ========================================================
    # Step 7 — Apply pilot advancement rule
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — APPLY PREDECLARED ADAPTER ADVANCEMENT RULE"
    )

    print(
        "=" * 80
    )

    rows_by_variant = {
        row[
            "variant"
        ]: row
        for row in metric_rows
    }

    dist_row = rows_by_variant[
        "ADAPTER_DIST"
    ]

    l0_row = rows_by_variant[
        "ADAPTER_L0"
    ]

    parent_row = rows_by_variant[
        "CONTROL_E5_PARENT"
    ]

    advancement_conditions = {
        "passes_existing_B2_guards_vs_B1": bool(
            dist_row[
                "passes_existing_B2_guards_vs_B1"
            ]
        ),
        "beats_Control_E5_parent_full_PSD": bool(
            float(
                dist_row[
                    "full_psd_mae_db"
                ]
            )
            < float(
                parent_row[
                    "full_psd_mae_db"
                ]
            )
        ),
        "beats_matched_ADAPTER_L0_full_PSD": bool(
            float(
                dist_row[
                    "full_psd_mae_db"
                ]
            )
            < float(
                l0_row[
                    "full_psd_mae_db"
                ]
            )
        ),
    }

    adapter_dist_advances = bool(
        all(
            advancement_conditions.values()
        )
    )

    for condition_name, passed in advancement_conditions.items():
        print(
            f"{condition_name}: "
            f"{passed}"
        )

    print(
        f"\nADAPTER_DIST ADVANCES: "
        f"{adapter_dist_advances}"
    )

    # ========================================================
    # Step 8 — Source immutability + summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — VERIFY IMMUTABILITY + SAVE FINAL SUMMARY"
    )

    print(
        "=" * 80
    )

    if sha256_file(
        FROZEN_B1_GENERATOR
    ) != frozen_b1_hash_before:
        raise RuntimeError(
            "Frozen B1 changed during s2_62."
        )

    if sha256_file(
        CONTROL_E5_CHECKPOINT
    ) != control_e5_hash_before:
        raise RuntimeError(
            "Control E5 changed during s2_62."
        )

    summary = {
        "status": (
            "S2_62_CONTROL_E5_PRESERVING_DISTRIBUTIONAL_ADAPTER_PILOT_COMPLETE"
        ),
        "scientific_role": (
            "Matched Train-only zero-init corrective-adapter pilot. "
            "No final B2 freeze."
        ),
        "parent": {
            "checkpoint": str(
                CONTROL_E5_CHECKPOINT
            ),
            "file_sha256": (
                control_e5_hash_before
            ),
            "model_state_sha256": (
                control_e5_state_hash
            ),
            "robust_anchor_supported_by_s2_61": True,
        },
        "matched_training_stream_sha256": (
            matched_stream_hash
        ),
        "training": {
            row[
                "arm"
            ]: row
            for row in history_rows
        },
        "free_generation_metrics": {
            row[
                "variant"
            ]: {
                key: value
                for key, value in row.items()
                if key
                != "variant"
            }
            for row in metric_rows
        },
        "advancement_conditions": (
            advancement_conditions
        ),
        "adapter_dist_advances": (
            adapter_dist_advances
        ),
        "artifacts": {
            "protocol": str(
                PROTOCOL_FILE
            ),
            "training_csv": str(
                TRAINING_CSV
            ),
            "freegen_metrics_csv": str(
                FREEGEN_METRICS_CSV
            ),
            "checkpoints": {
                arm_name: str(
                    arm_checkpoint_path(
                        arm_name
                    )
                )
                for arm_name in ARM_CONFIG
            },
            "synthetic": {
                arm_name: str(
                    arm_synthetic_path(
                        arm_name
                    )
                )
                for arm_name in ARM_CONFIG
            },
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "thresholds_changed": False,
            "formal_model_freeze_decision_made": False,
            "Frozen_B1_unchanged": True,
            "Control_E5_unchanged": True,
        },
        "next": (
            "If adapter_dist_advances is True, freeze a separate multi-seed "
            "follow-up protocol before any longer training. "
            "If False, do not tune this adapter objective post-hoc; interpret "
            "whether the next family should be post-generation correction, "
            "a different distributional statistic, or preserving Control E5 as final "
            "development generator."
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

    train_base_dataset.close()

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "s2_62 CONTROL-E5-PRESERVING DISTRIBUTIONAL ADAPTER PILOT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"ADAPTER_DIST advances: "
        f"{adapter_dist_advances}"
    )

    print(
        "\nValidation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "Formal model freeze decision made: NO"
    )

    print(
        "\nNEXT:"
    )

    if adapter_dist_advances:
        print(
            "FREEZE A SEPARATE MULTI-SEED FOLLOW-UP PROTOCOL FOR THE "
            "DISTRIBUTIONAL CORRECTIVE ADAPTER BEFORE LONGER TRAINING."
        )
    else:
        print(
            "DO NOT POST-HOC TUNE THIS ADAPTER OBJECTIVE. "
            "INTERPRET THE FAILURE BEFORE CHOOSING A NEW CORRECTIVE FAMILY."
        )


if __name__ == "__main__":
    main()
