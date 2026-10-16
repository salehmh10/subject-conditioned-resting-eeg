from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)
from s2_24_full_train_loader_sanity import (
    CachedRealTrainWindowDataset,
)
from s2_32_b0_unet_shape_test import (
    B0StateOnlyUNet1D,
)
from s2_33_b0_one_batch_diffusion_sanity import (
    build_cosine_schedule,
)
from s2_39_b0_epsilon_to_v_conversion_sanity import (
    make_v_target,
    q_sample,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

FROZEN_B0_ROOT = (
    REPO_ROOT
    / "outputs"
    / "frozen"
    / "scenario2"
    / "b0"
    / "b0_v_raw_epoch020_v1"
)

FROZEN_B0_GENERATOR = (
    FROZEN_B0_ROOT
    / "checkpoints"
    / "b0_v_raw_epoch020_generator_v1.pt"
)

B1_PROTOCOL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "protocols"
    / "scenario2"
    / "b1"
    / "subject_state_v1"
)

B1_PROTOCOL_FILE = (
    B1_PROTOCOL_ROOT
    / "metrics"
    / "b1_subject_state_finetune_protocol_v1.json"
)

B1_IMPLEMENTATION_HANDOFF = (
    B1_PROTOCOL_ROOT
    / "handoff"
    / "b1_architecture_implementation_handoff_v1.json"
)


# ============================================================
# Engineering sanity output tree
# ============================================================

RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b1"
    / "architecture_sanity_v1"
)

CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
METRIC_DIR = RUN_ROOT / "metrics"
LOG_DIR = RUN_ROOT / "logs"

TEMPLATE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "b1_warm_start_template_v1.pt"
)

SUMMARY_FILE = (
    METRIC_DIR
    / "b1_architecture_warmstart_sanity_v1.json"
)

DISCOVERY_FILE = (
    LOG_DIR
    / "b1_condition_injection_discovery_v1.json"
)


# ============================================================
# Frozen protocol expectations
# ============================================================

SEED = 20260908

N_SUBJECTS = 198
SUBJECT_EMBEDDING_DIM = 128
CONDITION_DIM = 256

N_CHANNELS = 19
N_TIMES = 512

TIMESTEPS = 1000
COSINE_S = 0.008

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

EQUIVALENCE_TOL = 1e-7
SUBJECT_EFFECT_MIN = 1e-10

# This script performs exactly one controlled Train-only
# optimizer step after saving the zero-update warm-start template.
# The updated model is intentionally NOT saved.
ENGINEERING_UPDATE_STEPS = 1


# ============================================================
# Reproducibility helpers
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def state_dict_sha256(
    state_dict: dict[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()

    for name in sorted(state_dict.keys()):
        tensor = (
            state_dict[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(
            json.dumps(
                list(tensor.shape)
            ).encode("utf-8")
        )
        digest.update(
            tensor.numpy().tobytes(order="C")
        )

    return digest.hexdigest()


def atomic_json_save(
    data: dict[str, Any],
    path: Path,
) -> None:
    tmp = Path(str(path) + ".tmp")

    with tmp.open(
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

    os.replace(tmp, path)


def atomic_torch_save(
    data: dict[str, Any],
    path: Path,
) -> None:
    tmp = Path(str(path) + ".tmp")

    torch.save(
        data,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


# ============================================================
# Subject-indexed Train dataset wrapper
# ============================================================

class SubjectIndexedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        records: list,
    ) -> None:
        if len(base_dataset) != len(records):
            raise RuntimeError(
                "Base dataset / record length mismatch."
            )

        self.base_dataset = base_dataset
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        item = self.base_dataset[index]
        record = self.records[index]

        return {
            "x": item["x"],
            "y": item["y"],
            "subject_index": torch.tensor(
                int(record.subject_index),
                dtype=torch.long,
            ),
        }


# ============================================================
# B0 conditioning-path discovery
# ============================================================

def _capture_2d_condition_outputs(
    model: nn.Module,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    eye_state: torch.Tensor,
    expected_dim: int,
) -> dict[str, list[torch.Tensor]]:
    captures: dict[str, list[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(
            module: nn.Module,
            inputs: tuple,
            output: Any,
        ):
            del module, inputs

            if not torch.is_tensor(output):
                return

            if output.ndim != 2:
                return

            if output.shape[0] != x.shape[0]:
                return

            if output.shape[1] != expected_dim:
                return

            captures.setdefault(
                name,
                [],
            ).append(
                output.detach().cpu().clone()
            )

        return hook

    for name, module in model.named_modules():
        if name == "":
            continue

        handles.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )

    try:
        with torch.no_grad():
            _ = model(
                x,
                timesteps,
                eye_state,
            )
    finally:
        for handle in handles:
            handle.remove()

    return captures


def discover_condition_injection_target(
    model: nn.Module,
    device: torch.device,
    expected_dim: int,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Discover the B0 module whose 256-D output represents the
    state-conditioning vector (preferred) or the already fused
    time+state conditioning vector (fallback).

    We do not silently alter the scientific contract:
    - target must emit exactly [B, 256]
    - target must be called exactly once per forward
    - output must depend on eye-state
    - state-only candidate is preferred over fused candidate
    """

    model.eval()

    generator = torch.Generator(
        device=device.type
    )
    generator.manual_seed(
        SEED + 101
    )

    x = torch.randn(
        (
            4,
            N_CHANNELS,
            N_TIMES,
        ),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    t_base = torch.tensor(
        [10, 250, 700, 990],
        device=device,
        dtype=torch.long,
    )

    t_changed = torch.tensor(
        [147, 387, 837, 127],
        device=device,
        dtype=torch.long,
    )

    state_base = torch.tensor(
        [0, 1, 0, 1],
        device=device,
        dtype=torch.long,
    )

    state_flipped = 1 - state_base

    base = _capture_2d_condition_outputs(
        model,
        x,
        t_base,
        state_base,
        expected_dim,
    )

    changed_state = _capture_2d_condition_outputs(
        model,
        x,
        t_base,
        state_flipped,
        expected_dim,
    )

    changed_time = _capture_2d_condition_outputs(
        model,
        x,
        t_changed,
        state_base,
        expected_dim,
    )

    module_map = dict(
        model.named_modules()
    )

    common_names = sorted(
        set(base)
        & set(changed_state)
        & set(changed_time)
    )

    rows: list[dict[str, Any]] = []

    for name in common_names:
        if not (
            len(base[name]) == 1
            and len(changed_state[name]) == 1
            and len(changed_time[name]) == 1
        ):
            continue

        base_output = base[name][0]
        state_output = changed_state[name][0]
        time_output = changed_time[name][0]

        state_delta = float(
            torch.max(
                torch.abs(
                    base_output
                    - state_output
                )
            ).item()
        )

        time_delta = float(
            torch.max(
                torch.abs(
                    base_output
                    - time_output
                )
            ).item()
        )

        lname = name.lower()

        semantic_tokens = [
            "state",
            "condition",
            "cond",
            "class",
            "label",
            "eye",
        ]

        semantic_score = sum(
            1
            for token in semantic_tokens
            if token in lname
        )

        state_dependent = (
            state_delta > 1e-9
        )

        time_dependent = (
            time_delta > 1e-9
        )

        if state_dependent and not time_dependent:
            category = (
                "STATE_ONLY_256D"
            )
            category_rank = 0
        elif state_dependent and time_dependent:
            category = (
                "FUSED_TIME_STATE_256D"
            )
            category_rank = 1
        else:
            category = (
                "NOT_STATE_CONDITION_TARGET"
            )
            category_rank = 9

        rows.append(
            {
                "module_name": name,
                "module_type": (
                    module_map[
                        name
                    ].__class__.__name__
                ),
                "state_delta_max_abs": (
                    state_delta
                ),
                "time_delta_max_abs": (
                    time_delta
                ),
                "semantic_score": (
                    semantic_score
                ),
                "name_depth": (
                    name.count(".")
                ),
                "category": category,
                "_category_rank": (
                    category_rank
                ),
            }
        )

    valid = [
        row
        for row in rows
        if row[
            "_category_rank"
        ] < 9
    ]

    if not valid:
        diagnostic = [
            {
                key: value
                for key, value
                in row.items()
                if not key.startswith("_")
            }
            for row in rows
        ]

        raise RuntimeError(
            "Could not discover a unique state-dependent "
            "256-D B0 conditioning module. "
            "The frozen B1 architecture contract must not "
            "be silently changed. Candidate diagnostics:\n"
            + json.dumps(
                diagnostic,
                indent=2,
            )
        )

    # Priority:
    #   1) state-only output over fused output
    #   2) semantically named module
    #   3) outer/shallow module rather than internal child
    #   4) stronger state dependence
    valid.sort(
        key=lambda row: (
            row[
                "_category_rank"
            ],
            -row[
                "semantic_score"
            ],
            row[
                "name_depth"
            ],
            -row[
                "state_delta_max_abs"
            ],
            row[
                "module_name"
            ],
        )
    )

    selected = valid[0]

    # Ambiguity guard:
    # If top two candidates have identical priority features
    # except name, stop rather than guessing.
    if len(valid) > 1:
        first = valid[0]
        second = valid[1]

        first_signature = (
            first[
                "_category_rank"
            ],
            first[
                "semantic_score"
            ],
            first[
                "name_depth"
            ],
        )

        second_signature = (
            second[
                "_category_rank"
            ],
            second[
                "semantic_score"
            ],
            second[
                "name_depth"
            ],
        )

        if (
            first_signature
            == second_signature
            and abs(
                first[
                    "state_delta_max_abs"
                ]
                - second[
                    "state_delta_max_abs"
                ]
            )
            < 1e-12
        ):
            raise RuntimeError(
                "Condition target discovery is ambiguous. "
                "Top candidates:\n"
                + json.dumps(
                    [
                        first,
                        second,
                    ],
                    indent=2,
                )
            )

    clean_rows = [
        {
            key: value
            for key, value
            in row.items()
            if not key.startswith("_")
        }
        for row in rows
    ]

    return (
        str(
            selected[
                "module_name"
            ]
        ),
        clean_rows,
    )


# ============================================================
# B1 architecture
# ============================================================

class B1SubjectStateUNet1D(nn.Module):
    """
    B1 = frozen-B0 topology + additive subject conditioning.

    The subject branch is injected into the discovered 256-D
    state/fused conditioning vector through a differentiable
    PyTorch forward hook.

    The final subject-projection layer is initialized exactly to
    zero, so before the first optimizer update the model function
    is exactly identical to B0 for every subject ID.
    """

    def __init__(
        self,
        b0_contract: dict,
        injection_module_name: str,
        num_subjects: int = N_SUBJECTS,
        subject_embedding_dim: int = (
            SUBJECT_EMBEDDING_DIM
        ),
        condition_dim: int = CONDITION_DIM,
    ) -> None:
        super().__init__()

        self.backbone = (
            B0StateOnlyUNet1D(
                b0_contract
            )
        )

        self.num_subjects = int(
            num_subjects
        )

        self.subject_embedding_dim = int(
            subject_embedding_dim
        )

        self.condition_dim = int(
            condition_dim
        )

        self.injection_module_name = str(
            injection_module_name
        )

        self.subject_embedding = nn.Embedding(
            self.num_subjects,
            self.subject_embedding_dim,
        )

        self.subject_projection = nn.Sequential(
            nn.Linear(
                self.subject_embedding_dim,
                self.condition_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.condition_dim,
                self.condition_dim,
            ),
        )

        # Critical transfer-learning invariant:
        # subject contribution starts at exactly zero.
        final_linear = (
            self.subject_projection[-1]
        )

        if not isinstance(
            final_linear,
            nn.Linear,
        ):
            raise RuntimeError(
                "Unexpected subject-projection final module."
            )

        nn.init.zeros_(
            final_linear.weight
        )

        nn.init.zeros_(
            final_linear.bias
        )

        module_map = dict(
            self.backbone.named_modules()
        )

        if (
            self.injection_module_name
            not in module_map
        ):
            raise RuntimeError(
                "Requested B1 injection module does not "
                "exist in B0 backbone: "
                f"{self.injection_module_name}"
            )

        self._injection_module = module_map[
            self.injection_module_name
        ]

        self._active_subject_condition: (
            torch.Tensor
            | None
        ) = None

        self._hook_calls = 0

        self._hook_handle = (
            self._injection_module
            .register_forward_hook(
                self._inject_subject_condition
            )
        )

    def load_b0_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.backbone.load_state_dict(
            state_dict,
            strict=True,
        )

    def subject_condition(
        self,
        subject_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.subject_projection(
            self.subject_embedding(
                subject_index
            )
        )

    def _inject_subject_condition(
        self,
        module: nn.Module,
        inputs: tuple,
        output: Any,
    ) -> Any:
        del module, inputs

        if (
            self._active_subject_condition
            is None
        ):
            return output

        if not torch.is_tensor(
            output
        ):
            raise RuntimeError(
                "B1 conditioning injection target "
                "did not return a tensor."
            )

        if output.ndim != 2:
            raise RuntimeError(
                "B1 conditioning injection target "
                f"output ndim={output.ndim}, expected 2."
            )

        if (
            output.shape
            != self._active_subject_condition.shape
        ):
            raise RuntimeError(
                "B1 subject condition shape mismatch: "
                f"target={tuple(output.shape)} "
                f"subject={tuple(self._active_subject_condition.shape)}"
            )

        self._hook_calls += 1

        return (
            output
            + self._active_subject_condition.to(
                dtype=output.dtype
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        eye_state: torch.Tensor,
        subject_index: torch.Tensor,
    ) -> torch.Tensor:
        if subject_index.ndim != 1:
            raise ValueError(
                "subject_index must have shape [B]."
            )

        if subject_index.shape[0] != x.shape[0]:
            raise ValueError(
                "subject_index batch size mismatch."
            )

        if subject_index.dtype != torch.long:
            raise ValueError(
                "subject_index must be torch.long."
            )

        if torch.any(
            subject_index < 0
        ) or torch.any(
            subject_index
            >= self.num_subjects
        ):
            raise ValueError(
                "subject_index outside frozen mapping."
            )

        subject_condition = (
            self.subject_condition(
                subject_index
            )
        )

        self._active_subject_condition = (
            subject_condition
        )

        self._hook_calls = 0

        try:
            output = self.backbone(
                x,
                timesteps,
                eye_state,
            )
        finally:
            self._active_subject_condition = None

        if self._hook_calls != 1:
            raise RuntimeError(
                "B1 conditioning target was expected "
                "to be called exactly once per forward; "
                f"observed {self._hook_calls} calls."
            )

        return output


# ============================================================
# Fixed balanced / subject-diverse Train batch
# ============================================================

def build_subject_diverse_balanced_batch(
    dataset: SubjectIndexedDataset,
    records: list,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    64 deterministic Train windows:
      subjects 0..31: one EO window each
      subjects 32..63: one EC window each

    This guarantees:
      - 64 unique subjects
      - 32 EO / 32 EC
      - Train-only
    """

    lookup: dict[
        tuple[int, int],
        int,
    ] = {}

    for index, record in enumerate(
        records
    ):
        key = (
            int(
                record.subject_index
            ),
            int(
                record.label
            ),
        )

        if key not in lookup:
            lookup[
                key
            ] = index

    selected_indices = []

    for subject_index in range(
        0,
        32,
    ):
        key = (
            subject_index,
            0,
        )

        if key not in lookup:
            raise RuntimeError(
                f"No EO Train window for subject "
                f"index {subject_index}."
            )

        selected_indices.append(
            lookup[
                key
            ]
        )

    for subject_index in range(
        32,
        64,
    ):
        key = (
            subject_index,
            1,
        )

        if key not in lookup:
            raise RuntimeError(
                f"No EC Train window for subject "
                f"index {subject_index}."
            )

        selected_indices.append(
            lookup[
                key
            ]
        )

    x_list = []
    state_list = []
    subject_list = []

    for index in selected_indices:
        item = dataset[
            index
        ]

        x_list.append(
            item[
                "x"
            ]
        )

        state_list.append(
            item[
                "y"
            ]
        )

        subject_list.append(
            item[
                "subject_index"
            ]
        )

    x = torch.stack(
        x_list,
        dim=0,
    ).to(
        device,
        non_blocking=True,
    )

    state = torch.stack(
        state_list,
        dim=0,
    ).to(
        device,
        non_blocking=True,
    )

    subject = torch.stack(
        subject_list,
        dim=0,
    ).to(
        device,
        non_blocking=True,
    )

    if x.shape != (
        BATCH_SIZE,
        N_CHANNELS,
        N_TIMES,
    ):
        raise RuntimeError(
            f"Unexpected controlled batch shape: "
            f"{tuple(x.shape)}"
        )

    if int(
        torch.sum(
            state == 0
        ).item()
    ) != 32:
        raise RuntimeError(
            "Controlled batch EO count mismatch."
        )

    if int(
        torch.sum(
            state == 1
        ).item()
    ) != 32:
        raise RuntimeError(
            "Controlled batch EC count mismatch."
        )

    if int(
        torch.unique(
            subject
        ).numel()
    ) != 64:
        raise RuntimeError(
            "Controlled batch subject diversity mismatch."
        )

    return (
        x,
        state,
        subject,
    )


# ============================================================
# Gradient helpers
# ============================================================

def grad_l2_norm(
    parameters,
) -> float:
    squares = []

    for parameter in parameters:
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        if not torch.isfinite(
            grad
        ).all():
            raise RuntimeError(
                "Nonfinite gradient encountered."
            )

        squares.append(
            torch.sum(
                grad.float()
                * grad.float()
            )
        )

    if not squares:
        return 0.0

    total = torch.sum(
        torch.stack(
            squares
        )
    )

    return float(
        torch.sqrt(
            total
        ).item()
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "SCENARIO 2 — B1 ARCHITECTURE + B0 WARM-START SANITY"
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
        "  B1 protocol: FROZEN"
    )

    print(
        "  B1 architecture implementation + warm-start sanity: NOW"
    )

    print(
        "  B1 1000-step pilot: NOT STARTED"
    )

    print(
        "  B1 full fine-tuning: NOT STARTED"
    )

    print(
        "  B2: NOT STARTED"
    )

    print(
        "  Bulk generation: NOT STARTED"
    )

    print(
        "  Final Test: SEALED"
    )

    for directory in (
        CHECKPOINT_DIR,
        METRIC_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    set_seed(
        SEED
    )

    # ========================================================
    # Step 1 — Audit protocol and frozen B0
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — AUDIT FROZEN B1 PROTOCOL + B0 PARENT"
    )

    print(
        "=" * 80
    )

    for path in (
        FROZEN_B0_GENERATOR,
        B1_PROTOCOL_FILE,
        B1_IMPLEMENTATION_HANDOFF,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    with B1_PROTOCOL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        protocol = json.load(
            f
        )

    with B1_IMPLEMENTATION_HANDOFF.open(
        "r",
        encoding="utf-8",
    ) as f:
        handoff = json.load(
            f
        )

    if protocol[
        "status"
    ] != "B1_SUBJECT_STATE_PROTOCOL_FROZEN":
        raise RuntimeError(
            "B1 protocol is not frozen."
        )

    if protocol[
        "governance"
    ][
        "Final_Test_accessed"
    ] is not False:
        raise RuntimeError(
            "Frozen B1 protocol reports Final Test access."
        )

    if protocol[
        "diffusion"
    ][
        "prediction_target"
    ] != "v":
        raise RuntimeError(
            "B1 protocol must use v-prediction."
        )

    if protocol[
        "full_fine_tuning"
    ][
        "ema_enabled"
    ] is not False:
        raise RuntimeError(
            "B1 protocol unexpectedly enables EMA."
        )

    if int(
        protocol[
            "conditioning"
        ][
            "subject"
        ][
            "num_subjects"
        ]
    ) != N_SUBJECTS:
        raise RuntimeError(
            "Frozen B1 subject count mismatch."
        )

    if int(
        protocol[
            "conditioning"
        ][
            "subject"
        ][
            "embedding_dim"
        ]
    ) != SUBJECT_EMBEDDING_DIM:
        raise RuntimeError(
            "Frozen subject embedding dimension mismatch."
        )

    if int(
        protocol[
            "conditioning"
        ][
            "subject"
        ][
            "projection_output_dim"
        ]
    ) != CONDITION_DIM:
        raise RuntimeError(
            "Frozen subject projection dimension mismatch."
        )

    if handoff[
        "status"
    ] != "B1_ARCHITECTURE_IMPLEMENTATION_READY":
        raise RuntimeError(
            "B1 architecture handoff not ready."
        )

    b0_file_hash_before = (
        sha256_file(
            FROZEN_B0_GENERATOR
        )
    )

    frozen_b0 = torch.load(
        FROZEN_B0_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    if frozen_b0[
        "status"
    ] != "B0_V_FROZEN_GENERATOR":
        raise RuntimeError(
            "Unexpected frozen B0 generator status."
        )

    b0_state_dict = frozen_b0[
        "model_state_dict"
    ]

    b0_state_hash = (
        state_dict_sha256(
            b0_state_dict
        )
    )

    if (
        b0_state_hash
        != frozen_b0[
            "selected_model_state_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen B0 tensor hash mismatch."
        )

    b0_contract = frozen_b0[
        "effective_contract"
    ]

    print(
        "B1 protocol audit: PASS"
    )

    print(
        "frozen parent: B0-v RAW epoch 20"
    )

    print(
        f"B0 model-state SHA256: "
        f"{b0_state_hash}"
    )

    # ========================================================
    # Step 2 — Discover exact conditioning injection point
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 2 — DISCOVER B0 256-D CONDITIONING INJECTION POINT"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this architecture sanity step."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    b0_discovery_model = (
        B0StateOnlyUNet1D(
            b0_contract
        )
    )

    b0_discovery_model.load_state_dict(
        b0_state_dict,
        strict=True,
    )

    b0_discovery_model.to(
        device
    )

    injection_module_name, discovery_rows = (
        discover_condition_injection_target(
            b0_discovery_model,
            device,
            CONDITION_DIM,
        )
    )

    del b0_discovery_model

    torch.cuda.empty_cache()

    discovery_payload = {
        "status": (
            "B1_CONDITION_INJECTION_DISCOVERY_COMPLETE"
        ),
        "selected_module": (
            injection_module_name
        ),
        "expected_condition_dim": (
            CONDITION_DIM
        ),
        "candidates": (
            discovery_rows
        ),
        "final_test_accessed": False,
    }

    atomic_json_save(
        discovery_payload,
        DISCOVERY_FILE,
    )

    selected_row = next(
        row
        for row in discovery_rows
        if row[
            "module_name"
        ]
        == injection_module_name
    )

    print(
        f"selected injection module: "
        f"{injection_module_name}"
    )

    print(
        f"module category: "
        f"{selected_row['category']}"
    )

    print(
        f"eye-state delta: "
        f"{selected_row['state_delta_max_abs']:.6e}"
    )

    print(
        f"time delta: "
        f"{selected_row['time_delta_max_abs']:.6e}"
    )

    print(
        "conditioning-path discovery: PASS"
    )

    # ========================================================
    # Step 3 — Build B0 reference + B1 warm start
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — BUILD B1 FROM FROZEN B0"
    )

    print(
        "=" * 80
    )

    set_seed(
        SEED
    )

    b0_reference = (
        B0StateOnlyUNet1D(
            b0_contract
        )
    )

    b0_reference.load_state_dict(
        b0_state_dict,
        strict=True,
    )

    b0_reference.to(
        device
    )

    b0_reference.eval()

    set_seed(
        SEED
    )

    b1 = B1SubjectStateUNet1D(
        b0_contract=b0_contract,
        injection_module_name=(
            injection_module_name
        ),
        num_subjects=N_SUBJECTS,
        subject_embedding_dim=(
            SUBJECT_EMBEDDING_DIM
        ),
        condition_dim=(
            CONDITION_DIM
        ),
    )

    b1.load_b0_state_dict(
        b0_state_dict
    )

    b1.to(
        device
    )

    b1.eval()

    b0_parameter_count = sum(
        parameter.numel()
        for parameter
        in b0_reference.parameters()
    )

    b1_parameter_count = sum(
        parameter.numel()
        for parameter
        in b1.parameters()
    )

    subject_branch_parameter_count = (
        b1_parameter_count
        - b0_parameter_count
    )

    print(
        f"B0 parameters: "
        f"{b0_parameter_count:,}"
    )

    print(
        f"B1 parameters: "
        f"{b1_parameter_count:,}"
    )

    print(
        f"new subject-branch parameters: "
        f"{subject_branch_parameter_count:,}"
    )

    # ========================================================
    # Step 4 — Exact B0 equivalence before training
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — VERIFY EXACT B0 FUNCTION PRESERVATION"
    )

    print(
        "=" * 80
    )

    generator = torch.Generator(
        device="cuda"
    )

    generator.manual_seed(
        SEED + 202
    )

    x_probe = torch.randn(
        (
            4,
            N_CHANNELS,
            N_TIMES,
        ),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    t_probe = torch.tensor(
        [0, 137, 731, 999],
        device=device,
        dtype=torch.long,
    )

    state_probe = torch.tensor(
        [0, 1, 0, 1],
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():
        b0_output = b0_reference(
            x_probe,
            t_probe,
            state_probe,
        )

    tested_subject_ids = [
        0,
        1,
        99,
        197,
    ]

    equivalence_diffs = {}

    with torch.no_grad():
        for subject_id in tested_subject_ids:
            subject_probe = torch.full(
                (
                    x_probe.shape[0],
                ),
                subject_id,
                device=device,
                dtype=torch.long,
            )

            b1_output = b1(
                x_probe,
                t_probe,
                state_probe,
                subject_probe,
            )

            diff = float(
                torch.max(
                    torch.abs(
                        b1_output
                        - b0_output
                    )
                ).item()
            )

            equivalence_diffs[
                str(
                    subject_id
                )
            ] = diff

    max_equivalence_diff = max(
        equivalence_diffs.values()
    )

    with torch.no_grad():
        output_subject_0 = b1(
            x_probe,
            t_probe,
            state_probe,
            torch.zeros(
                x_probe.shape[0],
                device=device,
                dtype=torch.long,
            ),
        )

        output_subject_197 = b1(
            x_probe,
            t_probe,
            state_probe,
            torch.full(
                (
                    x_probe.shape[0],
                ),
                197,
                device=device,
                dtype=torch.long,
            ),
        )

    pre_update_subject_diff = float(
        torch.max(
            torch.abs(
                output_subject_0
                - output_subject_197
            )
        ).item()
    )

    with torch.no_grad():
        initial_subject_residual = (
            b1.subject_condition(
                torch.tensor(
                    [0, 1, 99, 197],
                    device=device,
                    dtype=torch.long,
                )
            )
        )

    initial_subject_residual_max = float(
        torch.max(
            torch.abs(
                initial_subject_residual
            )
        ).item()
    )

    print(
        f"max |B1 - B0| across tested subjects: "
        f"{max_equivalence_diff:.3e}"
    )

    print(
        f"pre-update subject-to-subject output diff: "
        f"{pre_update_subject_diff:.3e}"
    )

    print(
        f"initial subject residual max abs: "
        f"{initial_subject_residual_max:.3e}"
    )

    if (
        max_equivalence_diff
        > EQUIVALENCE_TOL
    ):
        raise RuntimeError(
            "B1 does not preserve frozen B0 function "
            "at initialization."
        )

    if (
        pre_update_subject_diff
        > EQUIVALENCE_TOL
    ):
        raise RuntimeError(
            "B1 unexpectedly depends on subject before "
            "the first update."
        )

    if (
        initial_subject_residual_max
        != 0.0
    ):
        raise RuntimeError(
            "Subject residual is not exactly zero at "
            "initialization."
        )

    print(
        "exact warm-start equivalence: PASS"
    )

    # ========================================================
    # Step 5 — Save zero-update B1 template
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — SAVE ZERO-UPDATE B1 WARM-START TEMPLATE"
    )

    print(
        "=" * 80
    )

    b1_initial_state_hash = (
        state_dict_sha256(
            b1.state_dict()
        )
    )

    template_payload = {
        "status": (
            "B1_ZERO_UPDATE_WARM_START_TEMPLATE"
        ),
        "schema_version": 1,
        "parent_B0_generator": str(
            FROZEN_B0_GENERATOR
        ),
        "parent_B0_file_sha256": (
            b0_file_hash_before
        ),
        "parent_B0_model_state_sha256": (
            b0_state_hash
        ),
        "B1_protocol": str(
            B1_PROTOCOL_FILE
        ),
        "B1_protocol_sha256": (
            sha256_file(
                B1_PROTOCOL_FILE
            )
        ),
        "prediction_target": (
            "v"
        ),
        "num_subjects": (
            N_SUBJECTS
        ),
        "subject_embedding_dim": (
            SUBJECT_EMBEDDING_DIM
        ),
        "condition_dim": (
            CONDITION_DIM
        ),
        "condition_injection_module": (
            injection_module_name
        ),
        "seed": (
            SEED
        ),
        "model_state_dict": {
            name: tensor.detach()
            .cpu()
            .clone()
            for name, tensor
            in b1.state_dict().items()
        },
        "model_state_sha256": (
            b1_initial_state_hash
        ),
        "exact_B0_equivalence_max_abs": (
            max_equivalence_diff
        ),
        "training_updates_executed": 0,
        "validation_eeg_accessed": False,
        "final_test_accessed": False,
        "scientific_full_run_initialization_allowed": True,
        "note": (
            "This is a zero-update deterministic B1 "
            "initialization template. It is not a pilot-trained "
            "checkpoint and contains no optimizer state."
        ),
    }

    if TEMPLATE_CHECKPOINT.exists():
        existing_template = torch.load(
            TEMPLATE_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        if (
            existing_template.get(
                "model_state_sha256"
            )
            != b1_initial_state_hash
        ):
            raise RuntimeError(
                "Existing B1 warm-start template conflicts "
                "with current frozen initialization."
            )

        print(
            "identical zero-update template already exists."
        )
    else:
        atomic_torch_save(
            template_payload,
            TEMPLATE_CHECKPOINT,
        )

        print(
            "zero-update template saved."
        )

    print(
        f"template: "
        f"{TEMPLATE_CHECKPOINT}"
    )

    # ========================================================
    # Step 6 — Build deterministic Train-only controlled batch
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — BUILD SUBJECT-DIVERSE TRAIN-ONLY BATCH"
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

    records, subject_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    if len(records) != 47_835:
        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(subject_mapping) != N_SUBJECTS:
        raise RuntimeError(
            "Unexpected Train subject count."
        )

    base_dataset = (
        CachedRealTrainWindowDataset(
            records=records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    dataset = SubjectIndexedDataset(
        base_dataset=base_dataset,
        records=records,
    )

    x0, state, subject = (
        build_subject_diverse_balanced_batch(
            dataset,
            records,
            device,
        )
    )

    print(
        f"batch shape: "
        f"{tuple(x0.shape)}"
    )

    print(
        "EO / EC: 32 / 32"
    )

    print(
        "unique subjects: 64"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    # ========================================================
    # Step 7 — One controlled v-prediction update
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — ONE CONTROLLED TRAIN-ONLY B1 UPDATE"
    )

    print(
        "=" * 80
    )

    b1.train()

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

    train_generator = (
        torch.Generator(
            device="cuda"
        )
    )

    train_generator.manual_seed(
        SEED + 303
    )

    timesteps = torch.randint(
        0,
        TIMESTEPS,
        (
            BATCH_SIZE,
        ),
        device=device,
        dtype=torch.long,
        generator=train_generator,
    )

    epsilon = torch.randn(
        x0.shape,
        device=device,
        dtype=x0.dtype,
        generator=train_generator,
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

    optimizer = torch.optim.AdamW(
        b1.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    prediction_pre = b1(
        xt,
        timesteps,
        state,
        subject,
    )

    loss_pre = torch.mean(
        (
            prediction_pre
            - target_v
        )
        ** 2
    )

    if not torch.isfinite(
        loss_pre
    ):
        raise RuntimeError(
            "Initial B1 controlled loss is nonfinite."
        )

    loss_pre.backward()

    final_subject_linear = (
        b1.subject_projection[-1]
    )

    subject_final_grad_norm = (
        grad_l2_norm(
            final_subject_linear.parameters()
        )
    )

    subject_embedding_grad_norm = (
        grad_l2_norm(
            b1.subject_embedding.parameters()
        )
    )

    backbone_grad_norm = (
        grad_l2_norm(
            b1.backbone.parameters()
        )
    )

    total_grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            b1.parameters(),
            max_norm=(
                GRADIENT_CLIP_NORM
            ),
        ).item()
    )

    if not np.isfinite(
        total_grad_norm
    ):
        raise RuntimeError(
            "Total gradient norm is nonfinite."
        )

    if (
        subject_final_grad_norm
        <= 0.0
    ):
        raise RuntimeError(
            "New subject branch received no gradient."
        )

    optimizer.step()

    with torch.no_grad():
        prediction_post = b1(
            xt,
            timesteps,
            state,
            subject,
        )

        loss_post = torch.mean(
            (
                prediction_post
                - target_v
            )
            ** 2
        )

    if not torch.isfinite(
        loss_post
    ):
        raise RuntimeError(
            "Post-update B1 controlled loss is nonfinite."
        )

    print(
        f"v-MSE: "
        f"{loss_pre.item():.6f} "
        f"-> "
        f"{loss_post.item():.6f}"
    )

    print(
        f"subject final-projection grad norm: "
        f"{subject_final_grad_norm:.6e}"
    )

    print(
        f"subject embedding grad norm "
        f"(expected ~0 on first step): "
        f"{subject_embedding_grad_norm:.6e}"
    )

    print(
        f"backbone grad norm: "
        f"{backbone_grad_norm:.6e}"
    )

    print(
        f"total pre-clip grad norm: "
        f"{total_grad_norm:.6f}"
    )

    # ========================================================
    # Step 8 — Verify subject path becomes functional
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 8 — VERIFY SUBJECT CONDITIONING ACTIVATES AFTER UPDATE"
    )

    print(
        "=" * 80
    )

    b1.eval()

    probe_xt = xt[
        0:1
    ]

    probe_t = timesteps[
        0:1
    ]

    probe_state = state[
        0:1
    ]

    subject_a = torch.tensor(
        [0],
        device=device,
        dtype=torch.long,
    )

    subject_b = torch.tensor(
        [1],
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():
        residual_a = (
            b1.subject_condition(
                subject_a
            )
        )

        residual_b = (
            b1.subject_condition(
                subject_b
            )
        )

        output_a = b1(
            probe_xt,
            probe_t,
            probe_state,
            subject_a,
        )

        output_b = b1(
            probe_xt,
            probe_t,
            probe_state,
            subject_b,
        )

        output_eye_flipped = b1(
            probe_xt,
            probe_t,
            1 - probe_state,
            subject_a,
        )

        output_time_changed = b1(
            probe_xt,
            (
                probe_t
                + 137
            )
            % TIMESTEPS,
            probe_state,
            subject_a,
        )

    subject_residual_difference = float(
        torch.max(
            torch.abs(
                residual_a
                - residual_b
            )
        ).item()
    )

    subject_output_difference_max = float(
        torch.max(
            torch.abs(
                output_a
                - output_b
            )
        ).item()
    )

    subject_output_difference_mean = float(
        torch.mean(
            torch.abs(
                output_a
                - output_b
            )
        ).item()
    )

    eye_state_output_difference = float(
        torch.mean(
            torch.abs(
                output_a
                - output_eye_flipped
            )
        ).item()
    )

    timestep_output_difference = float(
        torch.mean(
            torch.abs(
                output_a
                - output_time_changed
            )
        ).item()
    )

    print(
        f"subject residual difference max abs: "
        f"{subject_residual_difference:.6e}"
    )

    print(
        f"subject output difference max abs: "
        f"{subject_output_difference_max:.6e}"
    )

    print(
        f"subject output difference mean abs: "
        f"{subject_output_difference_mean:.6e}"
    )

    print(
        f"eye-state output difference mean abs: "
        f"{eye_state_output_difference:.6e}"
    )

    print(
        f"timestep output difference mean abs: "
        f"{timestep_output_difference:.6e}"
    )

    if (
        subject_residual_difference
        <= SUBJECT_EFFECT_MIN
    ):
        raise RuntimeError(
            "Subject residual did not become subject-specific "
            "after the controlled update."
        )

    if (
        subject_output_difference_max
        <= SUBJECT_EFFECT_MIN
    ):
        raise RuntimeError(
            "Changing subject ID does not affect B1 output "
            "after the controlled update."
        )

    if (
        eye_state_output_difference
        <= 0.0
    ):
        raise RuntimeError(
            "Eye-state conditioning appears inactive."
        )

    if (
        timestep_output_difference
        <= 0.0
    ):
        raise RuntimeError(
            "Timestep conditioning appears inactive."
        )

    print(
        "post-update subject conditioning: PASS"
    )

    print(
        "eye-state conditioning retained: PASS"
    )

    print(
        "timestep conditioning retained: PASS"
    )

    # ========================================================
    # Step 9 — Verify frozen parent immutability
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 9 — VERIFY FROZEN B0 IMMUTABILITY"
    )

    print(
        "=" * 80
    )

    b0_file_hash_after = (
        sha256_file(
            FROZEN_B0_GENERATOR
        )
    )

    if (
        b0_file_hash_after
        != b0_file_hash_before
    ):
        raise RuntimeError(
            "Frozen B0 generator file changed."
        )

    b0_reference_state_hash_after = (
        state_dict_sha256(
            b0_reference.state_dict()
        )
    )

    if (
        b0_reference_state_hash_after
        != b0_state_hash
    ):
        raise RuntimeError(
            "B0 reference model changed during B1 sanity."
        )

    print(
        "frozen B0 file unchanged: PASS"
    )

    print(
        "B0 reference tensors unchanged: PASS"
    )

    # ========================================================
    # Step 10 — Save summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 10 — SAVE B1 ARCHITECTURE SANITY SUMMARY"
    )

    print(
        "=" * 80
    )

    summary = {
        "status": (
            "B1_ARCHITECTURE_WARMSTART_SANITY_PASS"
        ),
        "stage": (
            "B1 architecture implementation before "
            "1000-step pilot"
        ),
        "parent_B0": {
            "path": str(
                FROZEN_B0_GENERATOR
            ),
            "file_sha256": (
                b0_file_hash_before
            ),
            "model_state_sha256": (
                b0_state_hash
            ),
            "selected_epoch": 20,
            "selected_weight_kind": (
                "raw"
            ),
            "prediction_target": (
                "v"
            ),
            "unchanged_after_test": True,
        },
        "B1": {
            "definition": (
                "p(EEG | subject, eye_state)"
            ),
            "num_subjects": (
                N_SUBJECTS
            ),
            "subject_embedding_dim": (
                SUBJECT_EMBEDDING_DIM
            ),
            "condition_dim": (
                CONDITION_DIM
            ),
            "condition_injection_module": (
                injection_module_name
            ),
            "condition_injection_category": (
                selected_row[
                    "category"
                ]
            ),
            "B0_parameter_count": (
                b0_parameter_count
            ),
            "B1_parameter_count": (
                b1_parameter_count
            ),
            "new_subject_branch_parameters": (
                subject_branch_parameter_count
            ),
            "zero_update_template": str(
                TEMPLATE_CHECKPOINT
            ),
            "zero_update_template_state_sha256": (
                b1_initial_state_hash
            ),
        },
        "warm_start_equivalence": {
            "tested_subject_ids": (
                tested_subject_ids
            ),
            "per_subject_max_abs_diff": (
                equivalence_diffs
            ),
            "max_abs_diff": (
                max_equivalence_diff
            ),
            "tolerance": (
                EQUIVALENCE_TOL
            ),
            "pre_update_subject_output_diff": (
                pre_update_subject_diff
            ),
            "initial_subject_residual_max_abs": (
                initial_subject_residual_max
            ),
            "pass": True,
        },
        "controlled_train_only_update": {
            "steps": (
                ENGINEERING_UPDATE_STEPS
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "EO": 32,
            "EC": 32,
            "unique_subjects": 64,
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),
            "v_mse_before": float(
                loss_pre.item()
            ),
            "v_mse_after_same_batch": float(
                loss_post.item()
            ),
            "subject_final_projection_grad_norm": (
                subject_final_grad_norm
            ),
            "subject_embedding_grad_norm_first_step": (
                subject_embedding_grad_norm
            ),
            "backbone_grad_norm": (
                backbone_grad_norm
            ),
            "total_pre_clip_grad_norm": (
                total_grad_norm
            ),
            "updated_model_saved": False,
        },
        "post_update_conditioning": {
            "subject_residual_difference_max_abs": (
                subject_residual_difference
            ),
            "subject_output_difference_max_abs": (
                subject_output_difference_max
            ),
            "subject_output_difference_mean_abs": (
                subject_output_difference_mean
            ),
            "eye_state_output_difference_mean_abs": (
                eye_state_output_difference
            ),
            "timestep_output_difference_mean_abs": (
                timestep_output_difference
            ),
            "subject_conditioning_active": True,
            "eye_state_conditioning_active": True,
            "timestep_conditioning_active": True,
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "bulk_synthetic_generation_started": False,
            "B1_1000_step_pilot_started": False,
            "B1_full_training_started": False,
            "engineering_update_not_saved": True,
        },
        "next": (
            "B1 1000-step Train-only subject-conditioned "
            "fine-tuning pilot from zero-update warm-start template"
        ),
    }

    atomic_json_save(
        summary,
        SUMMARY_FILE,
    )

    print(
        f"summary: "
        f"{SUMMARY_FILE}"
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "B1 ARCHITECTURE + WARM-START SANITY COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"condition injection module: "
        f"{injection_module_name}"
    )

    print(
        f"B0 parameters: "
        f"{b0_parameter_count:,}"
    )

    print(
        f"B1 parameters: "
        f"{b1_parameter_count:,}"
    )

    print(
        f"new subject parameters: "
        f"{subject_branch_parameter_count:,}"
    )

    print(
        f"exact B0 equivalence max abs: "
        f"{max_equivalence_diff:.3e}"
    )

    print(
        f"controlled v-MSE: "
        f"{loss_pre.item():.6f} "
        f"-> "
        f"{loss_post.item():.6f}"
    )

    print(
        f"post-update subject output diff max abs: "
        f"{subject_output_difference_max:.6e}"
    )

    print(
        f"eye-state output diff mean abs: "
        f"{eye_state_output_difference:.6e}"
    )

    print(
        f"timestep output diff mean abs: "
        f"{timestep_output_difference:.6e}"
    )

    print(
        "frozen B0 unchanged: YES"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "B1 pilot started: NO"
    )

    print(
        "\nZero-update B1 template:"
    )

    print(
        TEMPLATE_CHECKPOINT
    )

    print(
        "\nNEXT:"
    )

    print(
        "RUN B1 1000-STEP TRAIN-ONLY "
        "SUBJECT-CONDITIONED FINE-TUNING PILOT"
    )


if __name__ == "__main__":
    main()
