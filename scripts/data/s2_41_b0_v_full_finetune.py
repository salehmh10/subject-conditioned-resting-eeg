from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from s2_24_full_train_loader_sanity import CachedRealTrainWindowDataset
from s2_27_logbandpower_logreg_validation import build_validation_window_index
from s2_32_b0_unet_shape_test import B0StateOnlyUNet1D
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_39_b0_epsilon_to_v_conversion_sanity import (
    make_v_target,
    q_sample,
    finite_gradient_summary,
)


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0"
    / "full_amp_safe_v1"
    / "checkpoints"
    / "best_ema.pt"
)

# Engineering pilot is deliberately NOT a source for this run.
PILOT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "short_pilot_v1"
    / "checkpoints"
    / "b0_v_short_pilot_final_v1.pt"
)

RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "full_finetune_v1"
)

CHECKPOINT_DIR = RUN_DIR / "checkpoints"
LOG_DIR = RUN_DIR / "logs"
METRIC_DIR = RUN_DIR / "metrics"
FIGURE_DIR = RUN_DIR / "figures"

PROTOCOL_FILE = METRIC_DIR / "b0_v_full_finetune_protocol_v1.json"
STATE_FILE = METRIC_DIR / "b0_v_full_finetune_state_v1.json"
RESULT_FILE = METRIC_DIR / "b0_v_full_finetune_result_v1.json"
HISTORY_FILE = LOG_DIR / "b0_v_full_finetune_history_v1.csv"
EVENT_FILE = LOG_DIR / "b0_v_full_finetune_events_v1.jsonl"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_ema.pt"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "final.pt"

LOSS_FIGURE = FIGURE_DIR / "b0_v_full_finetune_loss_v1.png"
GRADIENT_FIGURE = FIGURE_DIR / "b0_v_full_finetune_gradient_norm_v1.png"
AMP_SCALE_FIGURE = FIGURE_DIR / "b0_v_full_finetune_amp_scale_v1.png"
AMP_SKIP_FIGURE = FIGURE_DIR / "b0_v_full_finetune_amp_skips_v1.png"
THROUGHPUT_FIGURE = FIGURE_DIR / "b0_v_full_finetune_throughput_v1.png"
MEMORY_FIGURE = FIGURE_DIR / "b0_v_full_finetune_gpu_memory_v1.png"


# ============================================================
# Frozen scientific / engineering protocol
# ============================================================

SEED = 20260910
VALIDATION_SEED = 20260911
EXPECTED_SOURCE_EPOCH = 125
EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_VALIDATION_WINDOWS = 9_635
EXPECTED_SUBJECTS = 198

N_CHANNELS = 19
N_TIMES = 512
TIMESTEPS = 1000
COSINE_S = 0.008

BATCH_SIZE = 64
MAX_EPOCHS = 40  # USER-REQUESTED: exactly 40 complete fine-tuning epochs.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.9999
MIXED_PRECISION = True

CHECKPOINT_EVERY_EPOCHS = 5
PRINT_EVERY_BATCHES = 100
MAX_AMP_SKIPS_PER_EPOCH = 5
MAX_CONSECUTIVE_AMP_SKIPS = 3

# No post-hoc early stopping: execute all 40 epochs and keep the
# Validation-selected best EMA checkpoint.
EARLY_STOPPING = False


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



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



def atomic_torch_save(data: dict, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)



def append_event(kind: str, **fields) -> None:
    payload = {
        "time_unix": time.time(),
        "kind": kind,
        **fields,
    }
    with EVENT_FILE.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")



def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }



def assert_finite_state_dict(module: nn.Module, label: str) -> None:
    for name, tensor in module.state_dict().items():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Nonfinite tensor in {label}: {name}")



def capture_rng(train_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "train_generator": train_generator.get_state(),
    }



def restore_rng(state: dict, train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_generator"])


# ============================================================
# EMA
# ============================================================


@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    model_params = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        parameter = model_params[name]
        ema_parameter.mul_(decay)
        ema_parameter.add_(parameter, alpha=1.0 - decay)

    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        if name in model_buffers:
            ema_buffer.copy_(model_buffers[name])


# ============================================================
# Deterministic Validation v-MSE
# ============================================================


@torch.inference_mode()
def evaluate_validation_v_mse(
    model: nn.Module,
    loader: DataLoader,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Reinitialize the CUDA generator on every call so that every epoch
    sees exactly the same Validation timesteps and epsilon draws.
    """
    model.eval()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(VALIDATION_SEED)

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        x0 = batch["x"].to(device, non_blocking=True)
        state = batch["y"].to(device, non_blocking=True)
        batch_size = int(x0.shape[0])

        t = torch.randint(
            0,
            TIMESTEPS,
            (batch_size,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )
        epsilon = torch.randn(
            x0.shape,
            device=device,
            dtype=x0.dtype,
            generator=generator,
        )
        xt = q_sample(x0, epsilon, t, alpha_bar)
        target_v = make_v_target(x0, epsilon, t, alpha_bar)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=MIXED_PRECISION,
        ):
            pred_v = model(xt, t, state)

        loss = torch.mean((pred_v.float() - target_v.float()) ** 2)

        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite deterministic Validation v-MSE.")

        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            f"Validation accounting mismatch: {total_examples} != "
            f"{EXPECTED_VALIDATION_WINDOWS}"
        )

    return total_loss / total_examples


# ============================================================
# History / figures
# ============================================================


HISTORY_FIELDS = [
    "epoch",
    "train_v_mse",
    "ema_validation_v_mse",
    "mean_gradient_norm",
    "max_gradient_norm",
    "amp_skipped_steps",
    "amp_scale_end",
    "successful_optimizer_steps",
    "epoch_seconds",
    "train_windows_per_second",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "best_epoch",
    "best_ema_validation_v_mse",
]



def save_history(history: list[dict]) -> None:
    tmp = Path(str(HISTORY_FILE) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, HISTORY_FILE)



def save_figures(history: list[dict]) -> None:
    if not history:
        return

    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
    train_loss = np.asarray([row["train_v_mse"] for row in history], dtype=float)
    val_loss = np.asarray(
        [row["ema_validation_v_mse"] for row in history], dtype=float
    )
    grad = np.asarray([row["mean_gradient_norm"] for row in history], dtype=float)
    amp_scale = np.asarray([row["amp_scale_end"] for row in history], dtype=float)
    amp_skips = np.asarray([row["amp_skipped_steps"] for row in history], dtype=float)
    throughput = np.asarray(
        [row["train_windows_per_second"] for row in history], dtype=float
    )
    memory = np.asarray([row["peak_allocated_mib"] for row in history], dtype=float)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_loss, label="Train v-MSE")
    ax.plot(epochs, val_loss, label="EMA Validation v-MSE")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("MSE")
    ax.set_title("B0-v Full Fine-Tuning — Train and Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_FIGURE, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, grad)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Mean global gradient norm")
    ax.set_title("B0-v Full Fine-Tuning — Gradient Norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRADIENT_FIGURE, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, amp_scale)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("GradScaler scale")
    ax.set_title("B0-v Full Fine-Tuning — AMP Scale")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AMP_SCALE_FIGURE, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(epochs, amp_skips)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Skipped optimizer steps")
    ax.set_title("B0-v Full Fine-Tuning — AMP Skips")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(AMP_SKIP_FIGURE, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, throughput)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Train windows / second")
    ax.set_title("B0-v Full Fine-Tuning — Throughput")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(THROUGHPUT_FIGURE, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, memory)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Peak allocated GPU memory (MiB)")
    ax.set_title("B0-v Full Fine-Tuning — GPU Memory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEMORY_FIGURE, dpi=160)
    plt.close(fig)


# ============================================================
# Checkpoint payload
# ============================================================



def make_checkpoint(
    *,
    epoch: int,
    source_checkpoint: str,
    source_sha256: str,
    source_epoch: int,
    source_contract: dict,
    protocol: dict,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
    history: list[dict],
    best_epoch: int,
    best_ema_validation_v_mse: float,
    cumulative_seconds: float,
    total_amp_skips: int,
) -> dict:
    return {
        "status": "B0_V_FULL_FINETUNE_CHECKPOINT",
        "epoch": int(epoch),
        "source_checkpoint": source_checkpoint,
        "source_sha256": source_sha256,
        "source_epoch": int(source_epoch),
        "source_contract": source_contract,
        "fine_tune_protocol": protocol,
        "prediction_target": "v",
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": grad_scaler.state_dict(),
        "rng_state": capture_rng(train_generator),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_ema_validation_v_mse": float(best_ema_validation_v_mse),
        "cumulative_training_seconds": float(cumulative_seconds),
        "total_amp_skips": int(total_amp_skips),
        "validation_accessed": True,
        "final_test_accessed": False,
        "pilot_checkpoint_used": False,
    }


# ============================================================
# Main
# ============================================================



def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — B0-V FULL FINE-TUNING")
    print("=" * 80)

    print("\nCurrent project stage:")
    print("  B0 epsilon epoch-125 pretraining: COMPLETE")
    print("  B0-v conversion sanity: COMPLETE")
    print("  B0-v 1000-step engineering pilot: COMPLETE")
    print("  B0-v scientific full fine-tuning: NOW")

    print("\nFrozen run definition:")
    print(f"  max fine-tuning epochs: {MAX_EPOCHS}")
    print("  early stopping: NO — execute all 40 epochs")
    print("  initialization: B0 epsilon epoch-125 BEST EMA")
    print("  pilot checkpoint used: NO")
    print("  prediction target: v")
    print(f"  batch size: {BATCH_SIZE}")
    print(f"  AdamW LR: {LEARNING_RATE}")
    print(f"  EMA decay: {EMA_DECAY}")
    print("  checkpoint selection: lowest deterministic EMA Validation v-MSE")
    print("  Final Test: SEALED")

    for directory in (CHECKPOINT_DIR, LOG_DIR, METRIC_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for B0-v full fine-tuning.")
    device = torch.device("cuda")

    # ========================================================
    # Step 1 — Audit source and freeze protocol
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT SOURCE + FREEZE 40-EPOCH PROTOCOL")
    print("=" * 80)

    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)

    source_hash_before = sha256(SOURCE_CHECKPOINT)
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    source_epoch = int(source["epoch"])
    source_epsilon_mse = float(source["best_ema_validation_mse"])
    source_contract = source["contract"]

    if source_epoch != EXPECTED_SOURCE_EPOCH:
        raise RuntimeError(
            f"Expected B0 epsilon source epoch {EXPECTED_SOURCE_EPOCH}, "
            f"found {source_epoch}."
        )

    protocol = {
        "status": "B0_V_FULL_FINETUNE_PROTOCOL_FROZEN",
        "scientific_run": True,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_sha256": source_hash_before,
        "source_epoch": source_epoch,
        "source_ema_validation_epsilon_mse": source_epsilon_mse,
        "initialization": "B0 epsilon BEST EMA weights",
        "pilot_checkpoint_used": False,
        "pilot_checkpoint_path_record_only": str(PILOT_CHECKPOINT),
        "prediction_target": "v",
        "timesteps": TIMESTEPS,
        "beta_schedule": "cosine",
        "cosine_s": COSINE_S,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": EARLY_STOPPING,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "optimizer_state_reused": False,
        "learning_rate": LEARNING_RATE,
        "learning_rate_schedule": "constant",
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mixed_precision": MIXED_PRECISION,
        "ema": True,
        "ema_decay": EMA_DECAY,
        "checkpoint_selection": "lowest deterministic EMA Validation v-MSE",
        "validation_noise_and_timesteps": "fixed deterministic across epochs",
        "validation_seed": VALIDATION_SEED,
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "automatic_resume": True,
        "seed": SEED,
        "final_test_accessed": False,
    }

    if PROTOCOL_FILE.exists():
        with PROTOCOL_FILE.open("r", encoding="utf-8") as f:
            existing_protocol = json.load(f)
        if existing_protocol != protocol:
            raise RuntimeError(
                "Conflicting B0-v full fine-tuning protocol already exists."
            )
        print("identical 40-epoch protocol already frozen.")
    else:
        atomic_json_save(protocol, PROTOCOL_FILE)
        print(f"protocol frozen: {PROTOCOL_FILE}")

    print(f"source epoch: {source_epoch}")
    print(f"source epsilon EMA Validation MSE: {source_epsilon_mse:.6f}")
    print(f"source SHA256: {source_hash_before}")
    print("pilot checkpoint loaded: NO")

    # ========================================================
    # Step 2 — Build Train / Validation data
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 2 — BUILD REAL TRAIN / VALIDATION DATA")
    print("=" * 80)

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)
    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler_data = json.load(f)

    if scaler_data["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked.")

    train_records, train_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler_data,
    )
    validation_records, validation_mapping = build_validation_window_index(split)

    if len(train_records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(f"Unexpected Train windows: {len(train_records)}")
    if len(validation_records) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            f"Unexpected Validation windows: {len(validation_records)}"
        )
    if len(train_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Unexpected subject count: {len(train_mapping)}")
    if train_mapping != validation_mapping:
        raise RuntimeError("Train / Validation subject mapping mismatch.")

    train_dataset = CachedRealTrainWindowDataset(
        records=train_records,
        scaler=scaler_data,
        max_open_files=8,
    )
    validation_dataset = CachedRealTrainWindowDataset(
        records=validation_records,
        scaler=scaler_data,
        max_open_files=8,
    )

    print(f"Train windows: {len(train_dataset)}")
    print(f"Validation windows: {len(validation_dataset)}")
    print("Final Test EEG accessed: NO")

    # ========================================================
    # Step 3 — Build model / schedule / resume state
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 3 — BUILD OR RESUME B0-V FULL RUN")
    print("=" * 80)

    schedule_cpu = build_cosine_schedule(TIMESTEPS, COSINE_S)
    alpha_bar = schedule_cpu["alpha_cumprod"].to(
        device=device,
        dtype=torch.float32,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    model = B0StateOnlyUNet1D(source_contract).to(device)
    ema_model = B0StateOnlyUNet1D(source_contract).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    best_epoch = 0
    best_ema_validation_v_mse = float("inf")
    cumulative_seconds = 0.0
    total_amp_skips = 0
    start_epoch = 1
    resumed = False

    if LATEST_CHECKPOINT.exists():
        print(f"resume checkpoint found: {LATEST_CHECKPOINT}")
        checkpoint = torch.load(
            LATEST_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("fine_tune_protocol") != protocol:
            raise RuntimeError("Resume checkpoint protocol mismatch.")
        if checkpoint.get("source_sha256") != source_hash_before:
            raise RuntimeError("Resume checkpoint source hash mismatch.")
        if checkpoint.get("prediction_target") != "v":
            raise RuntimeError("Resume checkpoint is not v-prediction.")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        history = list(checkpoint["history"])
        best_epoch = int(checkpoint["best_epoch"])
        best_ema_validation_v_mse = float(
            checkpoint["best_ema_validation_v_mse"]
        )
        cumulative_seconds = float(checkpoint["cumulative_training_seconds"])
        total_amp_skips = int(checkpoint.get("total_amp_skips", 0))
        start_epoch = int(checkpoint["epoch"]) + 1

        restore_rng(checkpoint["rng_state"], train_generator)
        resumed = True

        print("RESUME PASS")
        print(f"completed fine-tuning epoch: {start_epoch - 1}")
        print(f"next epoch: {start_epoch}")
        print(f"best epoch so far: {best_epoch}")
        print(
            "best EMA Validation v-MSE: "
            f"{best_ema_validation_v_mse:.6f}"
        )
    else:
        # Scientific full run starts again from the epsilon epoch-125 EMA,
        # NOT from the engineering pilot checkpoint.
        model.load_state_dict(source["ema_state_dict"], strict=True)
        ema_model.load_state_dict(source["ema_state_dict"], strict=True)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

        print("Starting full B0-v fine-tuning from epsilon epoch-125 BEST EMA.")
        print("Fresh AdamW optimizer: YES")
        print("Pilot checkpoint used: NO")

    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build loaders only after restoring the dedicated Train generator.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    if len(train_loader) != 748:
        raise RuntimeError(f"Expected 748 Train batches, got {len(train_loader)}")
    if len(validation_loader) != 151:
        raise RuntimeError(
            f"Expected 151 Validation batches, got {len(validation_loader)}"
        )

    # ========================================================
    # Step 4 — Validation baseline / replay
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 4 — DETERMINISTIC EMA VALIDATION CHECK")
    print("=" * 80)

    validation_now = evaluate_validation_v_mse(
        ema_model,
        validation_loader,
        alpha_bar,
        device,
    )

    if resumed:
        expected = float(history[-1]["ema_validation_v_mse"])
        difference = abs(validation_now - expected)
        print(f"saved Validation v-MSE: {expected:.8f}")
        print(f"replayed Validation v-MSE: {validation_now:.8f}")
        print(f"difference: {difference:.3e}")
        if difference > 1e-6:
            raise RuntimeError("Deterministic Validation replay mismatch.")
        print("Validation replay: PASS")
    else:
        print(f"initial source-EMA Validation v-MSE: {validation_now:.6f}")
        print("This value is pre-fine-tuning and is NOT a selected checkpoint.")

    # ========================================================
    # Step 5 — 40-epoch full fine-tuning
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 5 — RUN 40-EPOCH B0-V FULL FINE-TUNING")
    print("=" * 80)

    append_event(
        "run_started_or_resumed",
        start_epoch=start_epoch,
        max_epochs=MAX_EPOCHS,
        resumed=resumed,
        source_epoch=source_epoch,
        source_sha256=source_hash_before,
    )

    try:
        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

            model.train()
            train_loss_sum = 0.0
            presented = 0
            gradient_norms: list[float] = []
            amp_skips = 0
            consecutive_amp_skips = 0
            successful_optimizer_steps = 0

            for batch_index, batch in enumerate(train_loader, start=1):
                x0 = batch["x"].to(device, non_blocking=True)
                state = batch["y"].to(device, non_blocking=True)
                current_batch_size = int(x0.shape[0])

                if not torch.isfinite(x0).all():
                    raise RuntimeError(
                        f"Nonfinite Train input at epoch={epoch}, batch={batch_index}."
                    )

                t = torch.randint(
                    0,
                    TIMESTEPS,
                    (current_batch_size,),
                    device=device,
                    dtype=torch.long,
                )
                epsilon = torch.randn_like(x0)
                xt = q_sample(x0, epsilon, t, alpha_bar)
                target_v = make_v_target(x0, epsilon, t, alpha_bar)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=MIXED_PRECISION,
                ):
                    pred_v = model(xt, t, state)
                    loss = loss_fn(pred_v.float(), target_v.float())

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Nonfinite v-loss at epoch={epoch}, batch={batch_index}."
                    )
                if not torch.isfinite(pred_v).all():
                    raise RuntimeError(
                        f"Nonfinite v prediction at epoch={epoch}, batch={batch_index}."
                    )

                loss_value = float(loss.item())
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)

                finite, grad_count, grad_norm, bad_names = finite_gradient_summary(model)
                scale_before = float(grad_scaler.get_scale())

                if finite:
                    clip_return = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRADIENT_CLIP_NORM,
                    )
                    if not torch.isfinite(clip_return):
                        raise RuntimeError(
                            "Nonfinite clip norm after finite gradient inspection."
                        )

                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    update_ema(ema_model, model, EMA_DECAY)

                    successful_optimizer_steps += 1
                    gradient_norms.append(float(grad_norm))
                    consecutive_amp_skips = 0
                else:
                    # GradScaler already recorded found_inf during unscale_.
                    # Calling step() lets it safely skip the optimizer update.
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scale_after = float(grad_scaler.get_scale())

                    amp_skips += 1
                    total_amp_skips += 1
                    consecutive_amp_skips += 1

                    print(
                        f"AMP SKIP | epoch={epoch} batch={batch_index} | "
                        f"scale {scale_before:g}->{scale_after:g} | "
                        "optimizer=SKIP EMA=SKIP"
                    )
                    append_event(
                        "amp_step_skipped",
                        epoch=epoch,
                        batch=batch_index,
                        loss=loss_value,
                        scale_before=scale_before,
                        scale_after=scale_after,
                        bad_parameter_names=bad_names[:20],
                    )

                    if amp_skips > MAX_AMP_SKIPS_PER_EPOCH:
                        raise RuntimeError(
                            f"Too many AMP skips in epoch {epoch}: {amp_skips}."
                        )
                    if consecutive_amp_skips >= MAX_CONSECUTIVE_AMP_SKIPS:
                        raise RuntimeError(
                            f"Too many consecutive AMP skips in epoch {epoch}: "
                            f"{consecutive_amp_skips}."
                        )

                presented += current_batch_size
                train_loss_sum += loss_value * current_batch_size

                if (
                    batch_index == 1
                    or batch_index % PRINT_EVERY_BATCHES == 0
                    or batch_index == len(train_loader)
                ):
                    print(
                        f"epoch {epoch:>2}/{MAX_EPOCHS}"
                        f" | batch {batch_index:>3}/{len(train_loader)}"
                        f" | train_v_MSE={train_loss_sum/presented:.6f}"
                        f" | amp_skips={amp_skips}"
                        f" | scale={grad_scaler.get_scale():g}"
                    )

            if presented != EXPECTED_TRAIN_WINDOWS:
                raise RuntimeError(
                    f"Train presentation accounting failed: {presented}"
                )
            if successful_optimizer_steps + amp_skips != len(train_loader):
                raise RuntimeError("Optimizer-step accounting failed.")
            if not gradient_norms:
                raise RuntimeError("No successful optimizer steps in epoch.")

            assert_finite_state_dict(model, "live model")
            assert_finite_state_dict(ema_model, "live EMA")

            ema_validation_v_mse = evaluate_validation_v_mse(
                ema_model,
                validation_loader,
                alpha_bar,
                device,
            )

            torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - epoch_start
            cumulative_seconds += epoch_seconds

            train_v_mse = train_loss_sum / presented
            mean_gradient_norm = float(np.mean(gradient_norms))
            max_gradient_norm = float(np.max(gradient_norms))
            peak_allocated_mib = float(
                torch.cuda.max_memory_allocated() / 1024**2
            )
            peak_reserved_mib = float(
                torch.cuda.max_memory_reserved() / 1024**2
            )
            throughput = presented / epoch_seconds

            improved = ema_validation_v_mse < best_ema_validation_v_mse
            if improved:
                best_ema_validation_v_mse = float(ema_validation_v_mse)
                best_epoch = int(epoch)

            row = {
                "epoch": int(epoch),
                "train_v_mse": float(train_v_mse),
                "ema_validation_v_mse": float(ema_validation_v_mse),
                "mean_gradient_norm": mean_gradient_norm,
                "max_gradient_norm": max_gradient_norm,
                "amp_skipped_steps": int(amp_skips),
                "amp_scale_end": float(grad_scaler.get_scale()),
                "successful_optimizer_steps": int(successful_optimizer_steps),
                "epoch_seconds": float(epoch_seconds),
                "train_windows_per_second": float(throughput),
                "peak_allocated_mib": peak_allocated_mib,
                "peak_reserved_mib": peak_reserved_mib,
                "best_epoch": int(best_epoch),
                "best_ema_validation_v_mse": float(best_ema_validation_v_mse),
            }
            history.append(row)

            print("\n" + "-" * 80)
            print(f"B0-V EPOCH {epoch} COMPLETE")
            print(f"Train v-MSE: {train_v_mse:.6f}")
            print(f"EMA Validation v-MSE: {ema_validation_v_mse:.6f}")
            print(
                f"Best EMA Validation v-MSE: "
                f"{best_ema_validation_v_mse:.6f} @ epoch {best_epoch}"
            )
            print(f"Mean gradient norm: {mean_gradient_norm:.4f}")
            print(f"Max gradient norm: {max_gradient_norm:.4f}")
            print(f"AMP skipped steps: {amp_skips}")
            print(f"GradScaler scale end: {grad_scaler.get_scale():g}")
            print(f"Epoch time: {epoch_seconds/60:.2f} min")
            print(f"Peak GPU allocated: {peak_allocated_mib:.0f} MiB")
            print(f"Improved best checkpoint: {improved}")
            print("-" * 80 + "\n")

            checkpoint = make_checkpoint(
                epoch=epoch,
                source_checkpoint=str(SOURCE_CHECKPOINT),
                source_sha256=source_hash_before,
                source_epoch=source_epoch,
                source_contract=source_contract,
                protocol=protocol,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                train_generator=train_generator,
                history=history,
                best_epoch=best_epoch,
                best_ema_validation_v_mse=best_ema_validation_v_mse,
                cumulative_seconds=cumulative_seconds,
                total_amp_skips=total_amp_skips,
            )

            # Numerical state first; plots/reporting second.
            atomic_torch_save(checkpoint, LATEST_CHECKPOINT)

            if improved or not BEST_CHECKPOINT.exists():
                atomic_torch_save(checkpoint, BEST_CHECKPOINT)
                print(f"BEST EMA CHECKPOINT UPDATED: {BEST_CHECKPOINT}")

            if epoch % CHECKPOINT_EVERY_EPOCHS == 0:
                milestone = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
                atomic_torch_save(checkpoint, milestone)
                print(f"milestone checkpoint: {milestone}")

            save_history(history)
            save_figures(history)

            state = {
                "status": "B0_V_FULL_FINETUNE_IN_PROGRESS",
                "completed_epoch": int(epoch),
                "max_epochs": MAX_EPOCHS,
                "best_epoch": int(best_epoch),
                "best_ema_validation_v_mse": float(
                    best_ema_validation_v_mse
                ),
                "latest_train_v_mse": float(train_v_mse),
                "latest_ema_validation_v_mse": float(
                    ema_validation_v_mse
                ),
                "total_amp_skips": int(total_amp_skips),
                "cumulative_hours": float(cumulative_seconds / 3600.0),
                "latest_checkpoint": str(LATEST_CHECKPOINT),
                "best_checkpoint": str(BEST_CHECKPOINT),
                "source_checkpoint": str(SOURCE_CHECKPOINT),
                "source_sha256": source_hash_before,
                "pilot_checkpoint_used": False,
                "validation_accessed": True,
                "final_test_accessed": False,
            }
            atomic_json_save(state, STATE_FILE)

            append_event(
                "epoch_committed",
                epoch=epoch,
                train_v_mse=train_v_mse,
                ema_validation_v_mse=ema_validation_v_mse,
                best_epoch=best_epoch,
                best_ema_validation_v_mse=best_ema_validation_v_mse,
                amp_skips=amp_skips,
            )

    except KeyboardInterrupt:
        print("\nB0-v FINE-TUNING INTERRUPTED BY USER.")
        print("The latest COMPLETED epoch checkpoint remains safe.")
        print("Run THIS SAME script again to resume automatically.")
        append_event("keyboard_interrupt")
        raise
    finally:
        train_dataset.close()
        validation_dataset.close()

    # ========================================================
    # Step 6 — Finalize after exactly 40 epochs
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 6 — FINALIZE 40-EPOCH B0-V FINE-TUNING")
    print("=" * 80)

    if not history:
        raise RuntimeError("No completed B0-v fine-tuning epochs.")

    final_epoch = int(history[-1]["epoch"])
    if final_epoch != MAX_EPOCHS:
        raise RuntimeError(
            f"B0-v full fine-tuning incomplete: {final_epoch}/{MAX_EPOCHS}"
        )

    final_checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    atomic_torch_save(final_checkpoint, FINAL_CHECKPOINT)

    source_hash_after = sha256(SOURCE_CHECKPOINT)
    source_unchanged = source_hash_after == source_hash_before
    if not source_unchanged:
        raise RuntimeError("Source B0 epsilon checkpoint changed unexpectedly.")

    final_train_v_mse = float(history[-1]["train_v_mse"])
    final_validation_v_mse = float(history[-1]["ema_validation_v_mse"])

    result = {
        "status": "B0_V_FULL_FINETUNE_COMPLETE",
        "epochs_completed": MAX_EPOCHS,
        "prediction_target": "v",
        "source": {
            "checkpoint": str(SOURCE_CHECKPOINT),
            "epoch": source_epoch,
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "unchanged": source_unchanged,
            "epsilon_ema_validation_mse": source_epsilon_mse,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "mixed_precision": MIXED_PRECISION,
            "ema_decay": EMA_DECAY,
            "early_stopping": EARLY_STOPPING,
            "total_amp_skips": int(total_amp_skips),
            "training_hours": float(cumulative_seconds / 3600.0),
        },
        "selection": {
            "metric": "deterministic EMA Validation v-MSE",
            "best_epoch": int(best_epoch),
            "best_ema_validation_v_mse": float(best_ema_validation_v_mse),
            "final_train_v_mse": final_train_v_mse,
            "final_ema_validation_v_mse": final_validation_v_mse,
        },
        "outputs": {
            "protocol": str(PROTOCOL_FILE),
            "history_csv": str(HISTORY_FILE),
            "state_json": str(STATE_FILE),
            "latest_checkpoint": str(LATEST_CHECKPOINT),
            "best_checkpoint": str(BEST_CHECKPOINT),
            "final_checkpoint": str(FINAL_CHECKPOINT),
            "loss_figure": str(LOSS_FIGURE),
            "gradient_figure": str(GRADIENT_FIGURE),
            "amp_scale_figure": str(AMP_SCALE_FIGURE),
            "amp_skip_figure": str(AMP_SKIP_FIGURE),
            "throughput_figure": str(THROUGHPUT_FIGURE),
            "memory_figure": str(MEMORY_FIGURE),
        },
        "pilot_checkpoint_used": False,
        "validation_accessed": True,
        "final_test_accessed": False,
        "ready_for_final_sampling_audit": True,
        "b0_declared_final": False,
    }
    atomic_json_save(result, RESULT_FILE)

    final_state = {
        "status": "B0_V_FULL_FINETUNE_COMPLETE",
        "completed_epoch": MAX_EPOCHS,
        "best_epoch": int(best_epoch),
        "best_ema_validation_v_mse": float(best_ema_validation_v_mse),
        "best_checkpoint": str(BEST_CHECKPOINT),
        "final_checkpoint": str(FINAL_CHECKPOINT),
        "source_checkpoint_unchanged": True,
        "validation_accessed": True,
        "final_test_accessed": False,
    }
    atomic_json_save(final_state, STATE_FILE)

    print("\n" + "=" * 80)
    print("B0-V 40-EPOCH FULL FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"completed epochs: {MAX_EPOCHS}")
    print(f"best epoch: {best_epoch}")
    print(f"best EMA Validation v-MSE: {best_ema_validation_v_mse:.6f}")
    print(f"final Train v-MSE: {final_train_v_mse:.6f}")
    print(f"final EMA Validation v-MSE: {final_validation_v_mse:.6f}")
    print(f"total AMP skipped steps: {total_amp_skips}")
    print(f"training time: {cumulative_seconds/60:.2f} min")
    print("source epsilon checkpoint unchanged: YES")
    print("pilot checkpoint used: NO")
    print("Validation accessed: YES")
    print("Final Test accessed: NO")
    print(f"\nbest checkpoint:\n{BEST_CHECKPOINT}")
    print(f"result JSON:\n{RESULT_FILE}")
    print("\nNEXT:")
    print("B0-V FINAL SAMPLING + REAL-TRAIN FIDELITY AUDIT")


if __name__ == "__main__":
    main()
