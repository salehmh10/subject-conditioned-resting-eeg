from __future__ import annotations

import hashlib
import json
import math
import os
import random
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
from s2_32_b0_unet_shape_test import B0StateOnlyUNet1D
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0"
    / "full_amp_safe_v1"
)
SOURCE_BEST_CHECKPOINT = SOURCE_RUN_DIR / "checkpoints" / "best_ema.pt"

OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "conversion_sanity_v1"
)
METRIC_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"

SUMMARY_FILE = METRIC_DIR / "b0_epsilon_to_v_conversion_sanity_v1.json"
PROFILE_FILE = LOG_DIR / "b0_v_zero_shot_timestep_profile_v1.csv"
AMPLIFICATION_FIGURE = FIGURE_DIR / "b0_epsilon_vs_v_x0_error_gain_v1.png"
PROFILE_FIGURE = FIGURE_DIR / "b0_v_zero_shot_x0_profile_v1.png"
STEP_FIGURE = FIGURE_DIR / "b0_v_one_step_sanity_v1.png"


# ============================================================
# Engineering-only sanity configuration
# ============================================================

SEED = 20260908
EXPECTED_SOURCE_EPOCH = 125
EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

N_PER_STATE = 64
BATCH_SIZE = 2 * N_PER_STATE
N_CHANNELS = 19
N_TIMES = 512

TIMESTEPS = 1000
COSINE_S = 0.008

# This LR is ONLY for one in-memory engineering step.
# It is NOT the full B0-v fine-tuning protocol.
SANITY_LEARNING_RATE = 1e-4
SANITY_WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MIXED_PRECISION = True

PROFILE_TIMESTEPS = [
    0, 1, 5, 10, 25, 50, 100, 200, 300, 400,
    500, 600, 700, 800, 900, 950, 975, 990, 995, 998, 999,
]


# ============================================================
# Helpers
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


def extract(values: torch.Tensor, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return values[timesteps].to(device=like.device, dtype=like.dtype).view(-1, 1, 1)


# ============================================================
# v-parameterization identities
# ============================================================

def make_v_target(
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """
    v = sqrt(alpha_bar) * epsilon - sqrt(1-alpha_bar) * x0
    """
    sqrt_a = torch.sqrt(extract(alpha_bar, timesteps, x0))
    sqrt_oma = torch.sqrt(1.0 - extract(alpha_bar, timesteps, x0))
    return sqrt_a * epsilon - sqrt_oma * x0


def q_sample(
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    sqrt_a = torch.sqrt(extract(alpha_bar, timesteps, x0))
    sqrt_oma = torch.sqrt(1.0 - extract(alpha_bar, timesteps, x0))
    return sqrt_a * x0 + sqrt_oma * epsilon


def x0_from_v(
    xt: torch.Tensor,
    v: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """
    x0 = sqrt(alpha_bar) * xt - sqrt(1-alpha_bar) * v
    """
    sqrt_a = torch.sqrt(extract(alpha_bar, timesteps, xt))
    sqrt_oma = torch.sqrt(1.0 - extract(alpha_bar, timesteps, xt))
    return sqrt_a * xt - sqrt_oma * v


def epsilon_from_v(
    xt: torch.Tensor,
    v: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """
    epsilon = sqrt(1-alpha_bar) * xt + sqrt(alpha_bar) * v
    """
    sqrt_a = torch.sqrt(extract(alpha_bar, timesteps, xt))
    sqrt_oma = torch.sqrt(1.0 - extract(alpha_bar, timesteps, xt))
    return sqrt_oma * xt + sqrt_a * v


# ============================================================
# Gradients
# ============================================================

def finite_gradient_summary(model: nn.Module) -> tuple[bool, int, float, list[str]]:
    count = 0
    squared = 0.0
    bad: list[str] = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        count += 1
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            bad.append(name)
            continue
        norm = float(grad.float().norm(2).item())
        squared += norm * norm

    if count == 0:
        raise RuntimeError("No gradients were produced.")

    finite = len(bad) == 0
    global_norm = math.sqrt(squared) if finite else float("nan")
    return finite, count, global_norm, bad


# ============================================================
# Data selection
# ============================================================

def build_balanced_train_dataset(split: pd.DataFrame, scaler: dict):
    records, subject_map = build_train_window_index(split_manifest=split, scaler=scaler)

    if len(records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(f"Unexpected Train windows: {len(records)}")
    if len(subject_map) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Unexpected subjects: {len(subject_map)}")

    eo_records = [record for record in records if int(record.label) == 0]
    ec_records = [record for record in records if int(record.label) == 1]

    rng = np.random.default_rng(SEED)
    eo_indices = rng.choice(len(eo_records), size=N_PER_STATE, replace=False)
    ec_indices = rng.choice(len(ec_records), size=N_PER_STATE, replace=False)

    selected = [eo_records[int(i)] for i in eo_indices] + [ec_records[int(i)] for i in ec_indices]

    dataset = CachedRealTrainWindowDataset(records=selected, scaler=scaler, max_open_files=8)
    return dataset, subject_map


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — B0 EPSILON -> V-PREDICTION CONVERSION SANITY")
    print("=" * 80)

    print("\nPurpose:")
    print("  Reuse the epoch-125 B0 epsilon EMA weights as a warm start.")
    print("  Verify v-parameterization algebra, bounded x0 reconstruction,")
    print("  and exactly ONE in-memory v-loss optimizer step.")

    print("\nIMPORTANT:")
    print("  This is NOT full B0-v fine-tuning.")
    print("  The one-step updated model is NOT saved as a scientific checkpoint.")
    print("  Source B0 epsilon checkpoint is READ ONLY.")
    print("\nReal data used: TRAIN ONLY")
    print("Validation EEG accessed: NO")
    print("Final Test EEG accessed: NO")

    for directory in (METRIC_DIR, FIGURE_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Step 1 — Source checkpoint
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT B0 EPSILON SOURCE CHECKPOINT")
    print("=" * 80)

    if not SOURCE_BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_BEST_CHECKPOINT)

    source_hash_before = sha256(SOURCE_BEST_CHECKPOINT)
    source = torch.load(SOURCE_BEST_CHECKPOINT, map_location="cpu", weights_only=False)

    source_epoch = int(source["epoch"])
    source_mse = float(source["best_ema_validation_mse"])

    if source_epoch != EXPECTED_SOURCE_EPOCH:
        raise RuntimeError(
            f"Expected B0 epsilon best epoch {EXPECTED_SOURCE_EPOCH}, found {source_epoch}."
        )

    contract = source["contract"]
    if contract["model_name"] != "B0_STATE_ONLY_DDPM":
        raise RuntimeError("Wrong source model.")
    if contract["diffusion"]["prediction_target"] != "epsilon":
        raise RuntimeError("Source checkpoint is not epsilon-prediction B0.")

    print(f"source checkpoint: {SOURCE_BEST_CHECKPOINT}")
    print(f"source epoch: {source_epoch}")
    print(f"source EMA Validation epsilon MSE: {source_mse:.6f}")
    print(f"source SHA256: {source_hash_before}")
    print("source checkpoint audit: PASS")

    # ========================================================
    # Step 2 — Environment / model
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 2 — LOAD WARM-START MODEL")
    print("=" * 80)

    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    device = torch.device("cuda")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = B0StateOnlyUNet1D(contract).to(device)
    # best_ema.pt exports EMA weights in both fields; use explicit EMA field.
    model.load_state_dict(source["ema_state_dict"], strict=True)
    model.eval()

    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("warm start: B0 epsilon BEST EMA")

    # ========================================================
    # Step 3 — Schedule and exact v algebra
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 3 — VERIFY V-PARAMETERIZATION IDENTITIES")
    print("=" * 80)

    schedule_cpu = build_cosine_schedule(timesteps=TIMESTEPS, cosine_s=COSINE_S)
    alpha_bar = schedule_cpu["alpha_cumprod"].to(device=device, dtype=torch.float64)

    algebra_generator = torch.Generator(device="cuda")
    algebra_generator.manual_seed(SEED + 1)

    x0_alg = torch.randn(
        (8, N_CHANNELS, N_TIMES), device=device, dtype=torch.float64, generator=algebra_generator
    )
    eps_alg = torch.randn(
        x0_alg.shape, device=device, dtype=torch.float64, generator=algebra_generator
    )
    t_alg = torch.tensor([0, 10, 100, 500, 900, 990, 998, 999], device=device, dtype=torch.long)

    xt_alg = q_sample(x0_alg, eps_alg, t_alg, alpha_bar)
    v_alg = make_v_target(x0_alg, eps_alg, t_alg, alpha_bar)
    x0_reconstructed = x0_from_v(xt_alg, v_alg, t_alg, alpha_bar)
    eps_reconstructed = epsilon_from_v(xt_alg, v_alg, t_alg, alpha_bar)

    x0_algebra_error = float(torch.max(torch.abs(x0_reconstructed - x0_alg)).item())
    eps_algebra_error = float(torch.max(torch.abs(eps_reconstructed - eps_alg)).item())

    print(f"max |x0_from_v - x0|: {x0_algebra_error:.3e}")
    print(f"max |epsilon_from_v - epsilon|: {eps_algebra_error:.3e}")

    if x0_algebra_error > 1e-10 or eps_algebra_error > 1e-10:
        raise RuntimeError("v-parameterization algebra failed.")

    print("v algebra: PASS")

    # ========================================================
    # Step 4 — Train-only balanced batch
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 4 — BUILD BALANCED REAL TRAIN SANITY BATCH")
    print("=" * 80)

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)
    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization not locked.")

    dataset, _ = build_balanced_train_dataset(split, scaler)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    batch = next(iter(loader))
    x0 = batch["x"].to(device, non_blocking=True)
    eye_state = batch["y"].to(device, non_blocking=True)

    if x0.shape != (BATCH_SIZE, N_CHANNELS, N_TIMES):
        raise RuntimeError(f"Unexpected batch shape: {x0.shape}")
    if not torch.isfinite(x0).all():
        raise RuntimeError("Nonfinite Train batch.")

    counts = torch.bincount(eye_state, minlength=2)
    print(f"batch shape: {tuple(x0.shape)}")
    print(f"EO: {int(counts[0])}")
    print(f"EC: {int(counts[1])}")
    print(f"real normalized mean: {x0.mean().item():.6f}")
    print(f"real normalized std: {x0.std(unbiased=False).item():.6f}")

    # float32 schedule for model diagnostics/training.
    alpha_bar32 = schedule_cpu["alpha_cumprod"].to(device=device, dtype=torch.float32)

    # ========================================================
    # Step 5 — Error-gain comparison
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 5 — EPSILON VS V X0-ERROR GAIN")
    print("=" * 80)

    abar_np = schedule_cpu["alpha_cumprod"].cpu().numpy().astype(np.float64)
    epsilon_gain = np.sqrt(1.0 - abar_np) / np.sqrt(abar_np)
    v_gain = np.sqrt(1.0 - abar_np)

    print(f"epsilon error gain @ t=999: {epsilon_gain[-1]:.2f}x")
    print(f"v error gain @ t=999: {v_gain[-1]:.6f}x")

    if v_gain[-1] > 1.000001:
        raise RuntimeError("Unexpected v error gain > 1.")

    figure = plt.figure(figsize=(11, 7))
    ax = figure.add_subplot(111)
    ax.semilogy(np.arange(TIMESTEPS), epsilon_gain, label="epsilon prediction: error gain to x0")
    ax.semilogy(np.arange(TIMESTEPS), np.maximum(v_gain, 1e-12), label="v prediction: error gain to x0")
    ax.set_xlabel("Diffusion timestep")
    ax.set_ylabel("Multiplicative x0 error gain")
    ax.set_title("B0 Parameterization Conditioning: Epsilon vs V")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(AMPLIFICATION_FIGURE, dpi=160)
    plt.close(figure)

    # ========================================================
    # Step 6 — Zero-shot reinterpretation profile
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 6 — ZERO-SHOT V-SEMANTICS PROFILE OF EPSILON WARM START")
    print("=" * 80)

    profile_generator = torch.Generator(device="cuda")
    profile_generator.manual_seed(SEED + 2)
    profile_rows: list[dict] = []

    model.eval()
    with torch.inference_mode():
        for t_value in PROFILE_TIMESTEPS:
            t = torch.full((BATCH_SIZE,), t_value, device=device, dtype=torch.long)
            epsilon = torch.randn(
                x0.shape, device=device, dtype=x0.dtype, generator=profile_generator
            )
            xt = q_sample(x0, epsilon, t, alpha_bar32)
            v_true = make_v_target(x0, epsilon, t, alpha_bar32)

            # Same numeric network output as the epsilon model, now inspected under v semantics.
            raw_output = model(xt, t, eye_state)
            if not torch.isfinite(raw_output).all():
                raise RuntimeError(f"Nonfinite warm-start output at t={t_value}")

            v_mse = float(torch.mean((raw_output.float() - v_true.float()) ** 2).item())
            x0_pred = x0_from_v(xt, raw_output, t, alpha_bar32)
            x0_rmse = float(torch.sqrt(torch.mean((x0_pred.float() - x0.float()) ** 2)).item())
            x0_pred_std = float(x0_pred.float().std(unbiased=False).item())
            output_std = float(raw_output.float().std(unbiased=False).item())

            row = {
                "timestep": int(t_value),
                "v_mse_zero_shot": v_mse,
                "x0_rmse_zero_shot": x0_rmse,
                "x0_pred_std_zero_shot": x0_pred_std,
                "raw_output_std": output_std,
                "epsilon_error_gain": float(epsilon_gain[t_value]),
                "v_error_gain": float(v_gain[t_value]),
            }
            profile_rows.append(row)

            print(
                f"t={t_value:>3} | zero-shot v_MSE={v_mse:>9.5f}"
                f" | x0_RMSE={x0_rmse:>8.4f}"
                f" | x0_pred_std={x0_pred_std:>8.4f}"
                f" | v_gain={v_gain[t_value]:>7.4f}"
            )

    import csv
    with PROFILE_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    profile_t = np.asarray([r["timestep"] for r in profile_rows])
    profile_rmse = np.asarray([r["x0_rmse_zero_shot"] for r in profile_rows])
    profile_std = np.asarray([r["x0_pred_std_zero_shot"] for r in profile_rows])

    figure = plt.figure(figsize=(11, 7))
    ax = figure.add_subplot(111)
    ax.plot(profile_t, profile_rmse, marker="o", label="zero-shot x0 RMSE")
    ax.plot(profile_t, profile_std, marker="o", label="zero-shot predicted x0 std")
    ax.axhline(float(x0.std(unbiased=False).item()), linestyle="--", label="real Train batch std")
    ax.set_xlabel("Diffusion timestep")
    ax.set_ylabel("Normalized scale")
    ax.set_title("B0 Epsilon Warm Start Interpreted as V: Bounded x0 Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(PROFILE_FIGURE, dpi=160)
    plt.close(figure)

    terminal_row = next(r for r in profile_rows if r["timestep"] == 999)
    if not math.isfinite(terminal_row["x0_pred_std_zero_shot"]):
        raise RuntimeError("Nonfinite t=999 v-semantics x0 std.")

    # ========================================================
    # Step 7 — One in-memory v-loss optimizer step
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 7 — ONE IN-MEMORY V-LOSS FINE-TUNING STEP")
    print("=" * 80)

    # Stratified timesteps guarantee coverage of the full noise trajectory,
    # including the terminal region, in this engineering sanity batch.
    timesteps = torch.linspace(
        0, TIMESTEPS - 1, BATCH_SIZE, device=device, dtype=torch.float32
    ).round().long()

    train_generator = torch.Generator(device="cuda")
    train_generator.manual_seed(SEED + 3)
    epsilon = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=train_generator)
    xt = q_sample(x0, epsilon, timesteps, alpha_bar32)
    v_target = make_v_target(x0, epsilon, timesteps, alpha_bar32)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SANITY_LEARNING_RATE, weight_decay=SANITY_WEIGHT_DECAY
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION)
    loss_fn = nn.MSELoss()

    # Track one parameter but never persist this updated model.
    tracked_name, tracked_parameter = next(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    tracked_before = tracked_parameter.detach().clone()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(
        device_type="cuda", dtype=torch.float16, enabled=MIXED_PRECISION
    ):
        v_pred = model(xt, timesteps, eye_state)
        pre_step_loss = loss_fn(v_pred, v_target)

    if not torch.isfinite(pre_step_loss):
        raise RuntimeError("Nonfinite pre-step v loss.")

    scaler_amp.scale(pre_step_loss).backward()
    scaler_amp.unscale_(optimizer)

    gradients_finite, gradient_tensor_count, grad_norm, bad_gradients = finite_gradient_summary(model)
    if not gradients_finite:
        raise RuntimeError(f"Nonfinite gradients during v sanity: {bad_gradients[:10]}")

    clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    if not torch.isfinite(clip_return):
        raise RuntimeError("Nonfinite clipped gradient norm.")

    scaler_amp.step(optimizer)
    scaler_amp.update()

    tracked_after = dict(model.named_parameters())[tracked_name].detach()
    parameter_update = float(torch.mean(torch.abs(tracked_after.float() - tracked_before.float())).item())
    if parameter_update <= 0:
        raise RuntimeError("One-step v fine-tuning did not update parameters.")

    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=MIXED_PRECISION
        ):
            v_pred_after = model(xt, timesteps, eye_state)
            post_step_loss = loss_fn(v_pred_after, v_target)

    if not torch.isfinite(post_step_loss):
        raise RuntimeError("Nonfinite post-step v loss.")

    print(f"pre-step v MSE: {pre_step_loss.item():.6f}")
    print(f"post-step v MSE on same batch: {post_step_loss.item():.6f}")
    print(f"parameters with gradients: {gradient_tensor_count}")
    print(f"global gradient norm before clip: {grad_norm:.6f}")
    print(f"tracked parameter: {tracked_name}")
    print(f"mean absolute one-step update: {parameter_update:.12e}")
    print("one-step v training mechanics: PASS")

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot(111)
    ax.bar(["Before one step", "After one step"], [pre_step_loss.item(), post_step_loss.item()])
    ax.set_ylabel("V-prediction MSE")
    ax.set_title("B0 Epsilon Warm Start — One-Step V-Loss Sanity")
    ax.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(STEP_FIGURE, dpi=160)
    plt.close(figure)

    # ========================================================
    # Step 8 — Verify source checkpoint unchanged
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 8 — VERIFY SOURCE CHECKPOINT REMAINED UNCHANGED")
    print("=" * 80)

    source_hash_after = sha256(SOURCE_BEST_CHECKPOINT)
    print(f"SHA256 before: {source_hash_before}")
    print(f"SHA256 after:  {source_hash_after}")

    if source_hash_after != source_hash_before:
        raise RuntimeError("Source B0 epsilon checkpoint changed unexpectedly.")

    print("source checkpoint immutability: PASS")

    # ========================================================
    # Step 9 — Save summary
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 9 — SAVE CONVERSION SANITY OUTPUTS")
    print("=" * 80)

    summary = {
        "status": "B0_EPSILON_TO_V_CONVERSION_SANITY_PASS",
        "source": {
            "checkpoint": str(SOURCE_BEST_CHECKPOINT),
            "epoch": source_epoch,
            "ema_validation_epsilon_mse": source_mse,
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "source_unchanged": True,
        },
        "parameterization": {
            "source_prediction_target": "epsilon",
            "sanity_target": "v",
            "v_definition": "sqrt(alpha_bar)*epsilon - sqrt(1-alpha_bar)*x0",
            "x0_from_v": "sqrt(alpha_bar)*xt - sqrt(1-alpha_bar)*v",
            "epsilon_from_v": "sqrt(1-alpha_bar)*xt + sqrt(alpha_bar)*v",
            "max_x0_algebra_error_float64": x0_algebra_error,
            "max_epsilon_algebra_error_float64": eps_algebra_error,
            "epsilon_error_gain_t999": float(epsilon_gain[-1]),
            "v_error_gain_t999": float(v_gain[-1]),
        },
        "train_batch": {
            "shape": list(x0.shape),
            "eo": int(counts[0].item()),
            "ec": int(counts[1].item()),
            "normalized_mean": float(x0.mean().item()),
            "normalized_std": float(x0.std(unbiased=False).item()),
        },
        "zero_shot_terminal": terminal_row,
        "one_step_engineering_sanity": {
            "learning_rate": SANITY_LEARNING_RATE,
            "weight_decay": SANITY_WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "mixed_precision": MIXED_PRECISION,
            "pre_step_v_mse": float(pre_step_loss.item()),
            "post_step_v_mse_same_batch": float(post_step_loss.item()),
            "gradient_tensor_count": gradient_tensor_count,
            "gradient_norm_before_clip": grad_norm,
            "tracked_parameter": tracked_name,
            "mean_absolute_parameter_update": parameter_update,
            "updated_model_persisted": False,
        },
        "outputs": {
            "profile_csv": str(PROFILE_FILE),
            "amplification_figure": str(AMPLIFICATION_FIGURE),
            "zero_shot_profile_figure": str(PROFILE_FIGURE),
            "one_step_figure": str(STEP_FIGURE),
            "summary_json": str(SUMMARY_FILE),
        },
        "scientific_full_v_protocol_frozen": False,
        "source_b0_epsilon_checkpoint_modified": False,
        "validation_eeg_accessed": False,
        "final_test_accessed": False,
    }

    atomic_json_save(summary, SUMMARY_FILE)

    print(f"profile CSV: {PROFILE_FILE}")
    print(f"amplification figure: {AMPLIFICATION_FIGURE}")
    print(f"zero-shot figure: {PROFILE_FIGURE}")
    print(f"one-step figure: {STEP_FIGURE}")
    print(f"summary JSON: {SUMMARY_FILE}")

    dataset.close()

    # ========================================================
    # Final
    # ========================================================
    print("\n" + "=" * 80)
    print("B0 EPSILON -> V-PREDICTION CONVERSION SANITY PASS")
    print("=" * 80)
    print(f"source epoch: {source_epoch}")
    print(f"epsilon x0-error gain @ t=999: {epsilon_gain[-1]:.2f}x")
    print(f"v x0-error gain @ t=999: {v_gain[-1]:.6f}x")
    print(f"zero-shot v x0 std @ t=999: {terminal_row['x0_pred_std_zero_shot']:.6f}")
    print(f"zero-shot v x0 RMSE @ t=999: {terminal_row['x0_rmse_zero_shot']:.6f}")
    print(f"pre-step v MSE: {pre_step_loss.item():.6f}")
    print(f"post-step v MSE: {post_step_loss.item():.6f}")
    print("source epsilon checkpoint unchanged: YES")
    print("one-step updated model saved: NO")
    print("Validation EEG accessed: NO")
    print("Final Test accessed: NO")
    print("\nNEXT:")
    print("FREEZE AND RUN A SHORT B0-V FINE-TUNING PILOT FROM THE EPOCH-125 EMA WARM START")


if __name__ == "__main__":
    main()
