from __future__ import annotations

import csv
import hashlib
import json
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
from scipy.signal import welch
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
from s2_39_b0_epsilon_to_v_conversion_sanity import (
    make_v_target,
    q_sample,
    x0_from_v,
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

RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "scenario2"
    / "b0_v"
    / "short_pilot_v1"
)

CHECKPOINT_DIR = RUN_DIR / "checkpoints"
LOG_DIR = RUN_DIR / "logs"
METRIC_DIR = RUN_DIR / "metrics"
FIGURE_DIR = RUN_DIR / "figures"
GENERATED_DIR = RUN_DIR / "generated"

PROTOCOL_FILE = METRIC_DIR / "b0_v_short_pilot_protocol_v1.json"
SUMMARY_FILE = METRIC_DIR / "b0_v_short_pilot_summary_v1.json"
HISTORY_FILE = LOG_DIR / "b0_v_short_pilot_history_v1.csv"
PROFILE_FILE = LOG_DIR / "b0_v_timestep_profile_pre_post_v1.csv"
CHECKPOINT_FILE = CHECKPOINT_DIR / "b0_v_short_pilot_final_v1.pt"
GENERATED_FILE = GENERATED_DIR / "b0_v_short_pilot_samples_v1.npz"

LOSS_FIGURE = FIGURE_DIR / "b0_v_short_pilot_loss_v1.png"
PROFILE_FIGURE = FIGURE_DIR / "b0_v_short_pilot_x0_profile_v1.png"
SAMPLING_FIGURE = FIGURE_DIR / "b0_v_short_pilot_sampling_scale_v1.png"
ALPHA_FIGURE = FIGURE_DIR / "b0_v_short_pilot_alpha_v1.png"


# ============================================================
# Engineering pilot configuration
# ============================================================

SEED = 20260908
SAMPLING_SEED = 20260909
EXPECTED_SOURCE_EPOCH = 125
EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

N_CHANNELS = 19
N_TIMES = 512
SFREQ = 128
TIMESTEPS = 1000
COSINE_S = 0.008

BATCH_SIZE = 64
PILOT_STEPS = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MIXED_PRECISION = True
PRINT_EVERY = 50
ROLLING_WINDOW = 50

# Deliberately small: this is only a quality/mechanics pilot.
SAMPLES_PER_STATE = 16

PROFILE_TIMESTEPS = [
    0, 10, 50, 100, 200, 300, 400, 500,
    600, 700, 800, 900, 950, 975, 990, 995, 998, 999,
]

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8",
    "C3", "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2",
]
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}


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


def inverse_normalize(eeg_z: np.ndarray, scaler: dict) -> np.ndarray:
    means = np.asarray(scaler["means_uv"], dtype=np.float64)
    stds = np.asarray(scaler["stds_uv"], dtype=np.float64)
    return (
        eeg_z.astype(np.float64) * stds[None, :, None]
        + means[None, :, None]
    ).astype(np.float32)


# ============================================================
# Stable v-prediction DDPM reverse sampler
# ============================================================

def build_reverse_schedule(
    schedule_cpu: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    betas = schedule_cpu["betas"].to(device=device, dtype=torch.float32)
    alphas = schedule_cpu["alphas"].to(device=device, dtype=torch.float32)
    alpha_bar = schedule_cpu["alpha_cumprod"].to(device=device, dtype=torch.float32)
    alpha_bar_prev = torch.cat(
        [torch.ones(1, device=device, dtype=torch.float32), alpha_bar[:-1]], dim=0
    )

    denominator = 1.0 - alpha_bar
    coef_x0 = betas * torch.sqrt(alpha_bar_prev) / denominator
    coef_xt = torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / denominator
    posterior_variance = betas * (1.0 - alpha_bar_prev) / denominator
    posterior_variance = torch.clamp(posterior_variance, min=0.0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "coef_x0": coef_x0,
        "coef_xt": coef_xt,
        "posterior_variance": posterior_variance,
    }


@torch.inference_mode()
def sample_v_model(
    model: nn.Module,
    eye_state_value: int,
    reverse_schedule: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> np.ndarray:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    shape = (SAMPLES_PER_STATE, N_CHANNELS, N_TIMES)
    x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    state = torch.full(
        (SAMPLES_PER_STATE,), eye_state_value, device=device, dtype=torch.long
    )

    alpha_bar = reverse_schedule["alpha_bar"]
    coef_x0 = reverse_schedule["coef_x0"]
    coef_xt = reverse_schedule["coef_xt"]
    posterior_variance = reverse_schedule["posterior_variance"]

    model.eval()

    for t in reversed(range(TIMESTEPS)):
        timestep = torch.full(
            (SAMPLES_PER_STATE,), t, device=device, dtype=torch.long
        )

        # FP32 sampling: no AMP here.
        v_pred = model(x, timestep, state)
        if not torch.isfinite(v_pred).all():
            raise RuntimeError(f"Nonfinite v prediction at timestep {t}")

        sqrt_ab = torch.sqrt(alpha_bar[t])
        sqrt_omab = torch.sqrt(1.0 - alpha_bar[t])

        # Stable v -> x0 identity. No division by sqrt(alpha_bar).
        x0_pred = sqrt_ab * x - sqrt_omab * v_pred

        model_mean = coef_x0[t] * x0_pred + coef_xt[t] * x

        if t > 0:
            noise = torch.randn(
                shape, device=device, dtype=torch.float32, generator=generator
            )
            x = model_mean + torch.sqrt(posterior_variance[t]) * noise
        else:
            x = model_mean

        if not torch.isfinite(x).all():
            raise RuntimeError(f"Nonfinite reverse state at timestep {t}")

    return x.detach().cpu().numpy().astype(np.float32, copy=False)


# ============================================================
# Balanced fixed Train probe
# ============================================================

def build_balanced_probe(
    dataset: CachedRealTrainWindowDataset,
    records,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    eo_indices = [i for i, record in enumerate(records) if int(record.label) == 0][:32]
    ec_indices = [i for i, record in enumerate(records) if int(record.label) == 1][:32]

    if len(eo_indices) != 32 or len(ec_indices) != 32:
        raise RuntimeError("Could not build 32 EO + 32 EC fixed Train probe.")

    indices = eo_indices + ec_indices
    xs = []
    ys = []

    for index in indices:
        item = dataset[index]
        xs.append(item["x"])
        ys.append(int(item["y"]))

    x0 = torch.stack(xs, dim=0).to(device)
    y = torch.tensor(ys, dtype=torch.long, device=device)

    if x0.shape != (64, N_CHANNELS, N_TIMES):
        raise RuntimeError(f"Unexpected fixed probe shape: {x0.shape}")

    return x0, y


@torch.no_grad()
def fixed_probe_v_mse(
    model: nn.Module,
    x0: torch.Tensor,
    eye_state: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> float:
    model.eval()
    xt = q_sample(x0, epsilon, timesteps, alpha_bar)
    target = make_v_target(x0, epsilon, timesteps, alpha_bar)
    with torch.amp.autocast(
        device_type="cuda", dtype=torch.float16, enabled=MIXED_PRECISION
    ):
        pred = model(xt, timesteps, eye_state)
        loss = torch.mean((pred.float() - target.float()) ** 2)
    return float(loss.item())


@torch.no_grad()
def timestep_profile(
    model: nn.Module,
    x0: torch.Tensor,
    eye_state: torch.Tensor,
    alpha_bar: torch.Tensor,
    seed: int,
    stage: str,
) -> list[dict]:
    model.eval()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    rows = []
    for t_value in PROFILE_TIMESTEPS:
        epsilon = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
        )
        t = torch.full(
            (x0.shape[0],), t_value, device=x0.device, dtype=torch.long
        )
        xt = q_sample(x0, epsilon, t, alpha_bar)
        target_v = make_v_target(x0, epsilon, t, alpha_bar)

        v_pred = model(xt, t, eye_state)
        x0_pred = x0_from_v(xt, v_pred, t, alpha_bar)

        v_mse = float(torch.mean((v_pred.float() - target_v.float()) ** 2).item())
        x0_rmse = float(
            torch.sqrt(torch.mean((x0_pred.float() - x0.float()) ** 2)).item()
        )
        x0_std = float(x0_pred.float().std(unbiased=False).item())

        rows.append(
            {
                "stage": stage,
                "timestep": t_value,
                "v_mse": v_mse,
                "x0_rmse": x0_rmse,
                "x0_pred_std": x0_std,
            }
        )

    return rows


# ============================================================
# Alpha sanity
# ============================================================

def alpha_summary(eeg_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    mask = (frequencies >= 8.0) & (frequencies <= 13.0)
    power = np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)
    power = np.maximum(power, np.finfo(np.float64).tiny)
    alpha_db = 10.0 * np.log10(power)
    return frequencies, psd, alpha_db


def paired_alpha_metrics(eo_uv: np.ndarray, ec_uv: np.ndarray) -> dict:
    _, _, eo_alpha = alpha_summary(eo_uv)
    _, _, ec_alpha = alpha_summary(ec_uv)
    delta = ec_alpha - eo_alpha
    whole_head = np.mean(delta, axis=1)

    result = {
        "whole_head_median_ec_minus_eo_db": float(np.median(whole_head)),
    }
    for channel in ("O1", "O2"):
        idx = CHANNEL_INDEX[channel]
        values = delta[:, idx]
        result[channel] = {
            "median_ec_minus_eo_db": float(np.median(values)),
            "mean_ec_minus_eo_db": float(np.mean(values)),
            "fraction_ec_gt_eo": float(np.mean(values > 0)),
        }
    return result


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — B0-V SHORT FINE-TUNING PILOT")
    print("=" * 80)

    print("\nCurrent project stage:")
    print("  B0 epsilon epoch-125 pretraining: COMPLETE")
    print("  epsilon sampling amplitude fidelity: FAILED")
    print("  v conversion algebra/mechanics: PASSED")
    print("  B0-v short fine-tuning pilot: NOW")

    print("\nThis is an ENGINEERING PILOT, not final B0-v training.")
    print(f"Steps: {PILOT_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"LR: {LEARNING_RATE}")
    print("Initialization: epoch-125 B0 epsilon BEST EMA")
    print("Optimizer state reused from epsilon run: NO")
    print("Prediction target: v")
    print("Validation EEG accessed: NO")
    print("Final Test EEG accessed: NO")

    for directory in (
        CHECKPOINT_DIR, LOG_DIR, METRIC_DIR, FIGURE_DIR, GENERATED_DIR
    ):
        directory.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    # ========================================================
    # Step 1 — Audit/freeze source + pilot protocol
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 1 — AUDIT SOURCE AND FREEZE PILOT PROTOCOL")
    print("=" * 80)

    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)

    source_hash_before = sha256(SOURCE_CHECKPOINT)
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)

    source_epoch = int(source["epoch"])
    source_mse = float(source["best_ema_validation_mse"])
    if source_epoch != EXPECTED_SOURCE_EPOCH:
        raise RuntimeError(
            f"Expected source epoch {EXPECTED_SOURCE_EPOCH}, found {source_epoch}."
        )

    protocol = {
        "status": "B0_V_SHORT_FINE_TUNING_PILOT_PROTOCOL",
        "engineering_pilot_only": True,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_sha256": source_hash_before,
        "source_epoch": source_epoch,
        "source_ema_validation_epsilon_mse": source_mse,
        "initialization": "B0 epsilon best EMA weights",
        "optimizer_state_reused": False,
        "prediction_target": "v",
        "timesteps": TIMESTEPS,
        "beta_schedule": "cosine",
        "cosine_s": COSINE_S,
        "pilot_steps": PILOT_STEPS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mixed_precision": MIXED_PRECISION,
        "ema_used": False,
        "validation_accessed": False,
        "final_test_accessed": False,
        "seed": SEED,
    }

    if PROTOCOL_FILE.exists():
        with PROTOCOL_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != protocol:
            raise RuntimeError("Conflicting B0-v pilot protocol already exists.")
    else:
        atomic_json_save(protocol, PROTOCOL_FILE)

    print(f"source epoch: {source_epoch}")
    print(f"source epsilon EMA Validation MSE: {source_mse:.6f}")
    print(f"source SHA256: {source_hash_before}")
    print(f"protocol: {PROTOCOL_FILE}")

    # ========================================================
    # Step 2 — Model and schedule
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 2 — BUILD B0-V WARM START")
    print("=" * 80)

    contract = source["contract"]
    model = B0StateOnlyUNet1D(contract).to(device)
    model.load_state_dict(source["ema_state_dict"], strict=True)

    schedule_cpu = build_cosine_schedule(TIMESTEPS, COSINE_S)
    alpha_bar = schedule_cpu["alpha_cumprod"].to(device=device, dtype=torch.float32)
    reverse_schedule = build_reverse_schedule(schedule_cpu, device)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("warm-start weights: PASS")

    # ========================================================
    # Step 3 — Train data + fixed probe
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 3 — BUILD REAL TRAIN PILOT DATA")
    print("=" * 80)

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)
    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked.")

    records, subject_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )
    if len(records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError("Unexpected Train window count.")
    if len(subject_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError("Unexpected Train subject count.")

    dataset = CachedRealTrainWindowDataset(
        records=records,
        scaler=scaler,
        max_open_files=8,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )

    probe_x0, probe_state = build_balanced_probe(dataset, records, device)

    probe_generator = torch.Generator(device="cuda")
    probe_generator.manual_seed(SEED + 1000)
    probe_timesteps = torch.randint(
        0,
        TIMESTEPS,
        (probe_x0.shape[0],),
        device=device,
        dtype=torch.long,
        generator=probe_generator,
    )
    probe_epsilon = torch.randn(
        probe_x0.shape,
        device=device,
        dtype=probe_x0.dtype,
        generator=probe_generator,
    )

    initial_probe_mse = fixed_probe_v_mse(
        model,
        probe_x0,
        probe_state,
        probe_timesteps,
        probe_epsilon,
        alpha_bar,
    )
    pre_profile = timestep_profile(
        model,
        probe_x0,
        probe_state,
        alpha_bar,
        SEED + 2000,
        "pre",
    )

    print(f"fixed balanced probe shape: {tuple(probe_x0.shape)}")
    print(f"initial fixed v MSE: {initial_probe_mse:.6f}")
    print(
        "pre t=999: "
        f"v_MSE={pre_profile[-1]['v_mse']:.6f}, "
        f"x0_RMSE={pre_profile[-1]['x0_rmse']:.6f}, "
        f"x0_std={pre_profile[-1]['x0_pred_std']:.6f}"
    )

    # ========================================================
    # Step 4 — Zero-shot v sampling before pilot
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 4 — ZERO-SHOT V-SAMPLING BEFORE FINE-TUNING")
    print("=" * 80)

    pre_eo_z = sample_v_model(model, 0, reverse_schedule, device, SAMPLING_SEED)
    pre_ec_z = sample_v_model(model, 1, reverse_schedule, device, SAMPLING_SEED)

    print(f"pre EO normalized std: {pre_eo_z.std():.6f}")
    print(f"pre EC normalized std: {pre_ec_z.std():.6f}")

    # ========================================================
    # Step 5 — New optimizer + 1000-step v fine-tuning pilot
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 5 — RUN 1000-STEP B0-V FINE-TUNING PILOT")
    print("=" * 80)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION)
    loss_fn = nn.MSELoss()

    history = []
    train_iterator = iter(loader)
    rolling_losses: list[float] = []
    amp_skips = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for step in range(1, PILOT_STEPS + 1):
        step_start = time.perf_counter()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(loader)
            batch = next(train_iterator)

        x0 = batch["x"].to(device, non_blocking=True)
        state = batch["y"].to(device, non_blocking=True)

        t = torch.randint(
            0, TIMESTEPS, (x0.shape[0],), device=device, dtype=torch.long
        )
        epsilon = torch.randn_like(x0)
        xt = q_sample(x0, epsilon, t, alpha_bar)
        v_target = make_v_target(x0, epsilon, t, alpha_bar)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=MIXED_PRECISION
        ):
            v_pred = model(xt, t, state)
            loss = loss_fn(v_pred.float(), v_target.float())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite v loss at pilot step {step}.")

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)

        finite, grad_count, grad_norm, bad_names = finite_gradient_summary(model)
        scale_before = float(scaler_amp.get_scale())

        if finite:
            clip_return = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(clip_return):
                raise RuntimeError("Nonfinite clip norm despite finite gradient check.")
            scaler_amp.step(optimizer)
        else:
            amp_skips += 1
            # Let GradScaler observe found_inf and skip optimizer internally.
            scaler_amp.step(optimizer)

        scaler_amp.update()
        scale_after = float(scaler_amp.get_scale())

        if not finite:
            print(
                f"AMP SKIP | step={step} | scale {scale_before:g}->{scale_after:g} "
                f"| bad={bad_names[:3]}"
            )
            if amp_skips > 5:
                raise RuntimeError("More than 5 AMP skips in the short B0-v pilot.")

        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_start
        loss_value = float(loss.item())
        rolling_losses.append(loss_value)
        if len(rolling_losses) > ROLLING_WINDOW:
            rolling_losses.pop(0)
        rolling = float(np.mean(rolling_losses))

        history.append(
            {
                "step": step,
                "loss": loss_value,
                "rolling_loss": rolling,
                "gradient_norm": float(grad_norm) if finite else float("nan"),
                "amp_skipped": int(not finite),
                "amp_scale": scale_after,
                "step_seconds": step_seconds,
            }
        )

        if step == 1 or step % PRINT_EVERY == 0 or step == PILOT_STEPS:
            print(
                f"step {step:>4}/{PILOT_STEPS}"
                f" | v_MSE={loss_value:.5f}"
                f" | roll={rolling:.5f}"
                f" | amp_skips={amp_skips}"
                f" | scale={scale_after:g}"
                f" | elapsed={(time.perf_counter()-start)/60:.1f}m"
            )

    training_seconds = time.perf_counter() - start

    # ========================================================
    # Step 6 — Fixed probe/profile after pilot
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 6 — POST-PILOT V DIAGNOSTICS")
    print("=" * 80)

    final_probe_mse = fixed_probe_v_mse(
        model,
        probe_x0,
        probe_state,
        probe_timesteps,
        probe_epsilon,
        alpha_bar,
    )
    post_profile = timestep_profile(
        model,
        probe_x0,
        probe_state,
        alpha_bar,
        SEED + 2000,
        "post",
    )

    print(f"fixed v MSE: {initial_probe_mse:.6f} -> {final_probe_mse:.6f}")
    print(
        "post t=999: "
        f"v_MSE={post_profile[-1]['v_mse']:.6f}, "
        f"x0_RMSE={post_profile[-1]['x0_rmse']:.6f}, "
        f"x0_std={post_profile[-1]['x0_pred_std']:.6f}"
    )

    # ========================================================
    # Step 7 — Post-pilot v sampling
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 7 — POST-PILOT V-SAMPLING")
    print("=" * 80)

    post_eo_z = sample_v_model(model, 0, reverse_schedule, device, SAMPLING_SEED)
    post_ec_z = sample_v_model(model, 1, reverse_schedule, device, SAMPLING_SEED)

    post_eo_uv = inverse_normalize(post_eo_z, scaler)
    post_ec_uv = inverse_normalize(post_ec_z, scaler)
    alpha = paired_alpha_metrics(post_eo_uv, post_ec_uv)

    print(f"post EO normalized std: {post_eo_z.std():.6f}")
    print(f"post EC normalized std: {post_ec_z.std():.6f}")
    print(f"post EO µV std: {post_eo_uv.std():.3f}")
    print(f"post EC µV std: {post_ec_uv.std():.3f}")
    print(
        "post alpha whole-head EC-EO: "
        f"{alpha['whole_head_median_ec_minus_eo_db']:.3f} dB"
    )
    print(
        f"post O1 EC-EO: {alpha['O1']['median_ec_minus_eo_db']:.3f} dB"
    )
    print(
        f"post O2 EC-EO: {alpha['O2']['median_ec_minus_eo_db']:.3f} dB"
    )

    # ========================================================
    # Step 8 — Save history/profile/samples/checkpoint
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 8 — SAVE PILOT ARTIFACTS")
    print("=" * 80)

    with HISTORY_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    all_profile = pre_profile + post_profile
    with PROFILE_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_profile[0].keys()))
        writer.writeheader()
        writer.writerows(all_profile)

    np.savez_compressed(
        GENERATED_FILE,
        pre_eo_z=pre_eo_z,
        pre_ec_z=pre_ec_z,
        post_eo_z=post_eo_z,
        post_ec_z=post_ec_z,
        post_eo_uv=post_eo_uv,
        post_ec_uv=post_ec_uv,
        channel_names=np.asarray(CHANNEL_NAMES),
        sampling_seed=np.asarray([SAMPLING_SEED], dtype=np.int64),
    )

    torch.save(
        {
            "status": "B0_V_SHORT_FINE_TUNING_PILOT_COMPLETE",
            "engineering_pilot_only": True,
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_sha256": source_hash_before,
            "source_epoch": source_epoch,
            "contract": contract,
            "prediction_target": "v",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": scaler_amp.state_dict(),
            "pilot_steps": PILOT_STEPS,
            "validation_accessed": False,
            "final_test_accessed": False,
        },
        CHECKPOINT_FILE,
    )

    # ========================================================
    # Step 9 — Figures
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 9 — SAVE PILOT FIGURES")
    print("=" * 80)

    steps = np.asarray([row["step"] for row in history])
    losses = np.asarray([row["loss"] for row in history])
    rolling = np.asarray([row["rolling_loss"] for row in history])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(steps, losses, alpha=0.3, label="batch v-MSE")
    ax.plot(steps, rolling, label=f"{ROLLING_WINDOW}-step rolling")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("v-prediction MSE")
    ax.set_title("B0-v Short Fine-Tuning Pilot")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_FIGURE, dpi=160)
    plt.close(fig)

    pre_by_t = {row["timestep"]: row for row in pre_profile}
    post_by_t = {row["timestep"]: row for row in post_profile}
    ts = np.asarray(PROFILE_TIMESTEPS)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.semilogy(ts, [pre_by_t[t]["x0_rmse"] for t in ts], label="pre x0 RMSE")
    ax.semilogy(ts, [post_by_t[t]["x0_rmse"] for t in ts], label="post x0 RMSE")
    ax.set_xlabel("Diffusion timestep")
    ax.set_ylabel("x0 reconstruction RMSE")
    ax.set_title("B0-v x0 Reconstruction Before vs After Pilot")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROFILE_FIGURE, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    labels = ["pre EO", "pre EC", "post EO", "post EC", "real probe"]
    values = [
        float(pre_eo_z.std()),
        float(pre_ec_z.std()),
        float(post_eo_z.std()),
        float(post_ec_z.std()),
        float(probe_x0.detach().cpu().numpy().std()),
    ]
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Normalized EEG standard deviation")
    ax.set_title("B0-v Sampling Scale Sanity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAMPLING_FIGURE, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    alpha_labels = ["Whole head", "O1", "O2"]
    alpha_values = [
        alpha["whole_head_median_ec_minus_eo_db"],
        alpha["O1"]["median_ec_minus_eo_db"],
        alpha["O2"]["median_ec_minus_eo_db"],
    ]
    ax.bar(np.arange(3), alpha_values)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(alpha_labels)
    ax.set_ylabel("EC - EO alpha power (dB)")
    ax.set_title("B0-v Short Pilot Synthetic Alpha Effect")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ALPHA_FIGURE, dpi=160)
    plt.close(fig)

    # ========================================================
    # Step 10 — Source immutability + summary
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 10 — VERIFY SOURCE + SAVE SUMMARY")
    print("=" * 80)

    source_hash_after = sha256(SOURCE_CHECKPOINT)
    source_unchanged = source_hash_after == source_hash_before
    if not source_unchanged:
        raise RuntimeError("Source epsilon checkpoint changed during B0-v pilot.")

    first_window_mean = float(np.mean(losses[:ROLLING_WINDOW]))
    last_window_mean = float(np.mean(losses[-ROLLING_WINDOW:]))
    peak_allocated_mib = float(torch.cuda.max_memory_allocated() / 1024**2)

    summary = {
        "status": "B0_V_SHORT_FINE_TUNING_PILOT_COMPLETE",
        "engineering_pilot_only": True,
        "source": {
            "epoch": source_epoch,
            "epsilon_ema_validation_mse": source_mse,
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "unchanged": source_unchanged,
        },
        "training": {
            "steps": PILOT_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "mixed_precision": MIXED_PRECISION,
            "amp_skips": amp_skips,
            "training_seconds": training_seconds,
            "first_rolling_window_mean_v_mse": first_window_mean,
            "last_rolling_window_mean_v_mse": last_window_mean,
            "initial_fixed_probe_v_mse": initial_probe_mse,
            "final_fixed_probe_v_mse": final_probe_mse,
        },
        "t999": {
            "pre_v_mse": pre_profile[-1]["v_mse"],
            "pre_x0_rmse": pre_profile[-1]["x0_rmse"],
            "pre_x0_pred_std": pre_profile[-1]["x0_pred_std"],
            "post_v_mse": post_profile[-1]["v_mse"],
            "post_x0_rmse": post_profile[-1]["x0_rmse"],
            "post_x0_pred_std": post_profile[-1]["x0_pred_std"],
        },
        "sampling": {
            "samples_per_state": SAMPLES_PER_STATE,
            "pre_eo_normalized_std": float(pre_eo_z.std()),
            "pre_ec_normalized_std": float(pre_ec_z.std()),
            "post_eo_normalized_std": float(post_eo_z.std()),
            "post_ec_normalized_std": float(post_ec_z.std()),
            "post_eo_uv_std": float(post_eo_uv.std()),
            "post_ec_uv_std": float(post_ec_uv.std()),
            "alpha": alpha,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_mib": peak_allocated_mib,
        },
        "outputs": {
            "protocol": str(PROTOCOL_FILE),
            "history_csv": str(HISTORY_FILE),
            "profile_csv": str(PROFILE_FILE),
            "checkpoint": str(CHECKPOINT_FILE),
            "generated_npz": str(GENERATED_FILE),
            "loss_figure": str(LOSS_FIGURE),
            "profile_figure": str(PROFILE_FIGURE),
            "sampling_figure": str(SAMPLING_FIGURE),
            "alpha_figure": str(ALPHA_FIGURE),
        },
        "validation_accessed": False,
        "final_test_accessed": False,
        "scientific_checkpoint_declared_final": False,
    }
    atomic_json_save(summary, SUMMARY_FILE)

    dataset.close()

    print(f"source checkpoint unchanged: {source_unchanged}")
    print(f"summary: {SUMMARY_FILE}")
    print(f"pilot checkpoint: {CHECKPOINT_FILE}")

    print("\n" + "=" * 80)
    print("B0-V SHORT FINE-TUNING PILOT COMPLETE")
    print("=" * 80)
    print(f"first {ROLLING_WINDOW}-step mean v-MSE: {first_window_mean:.6f}")
    print(f"last  {ROLLING_WINDOW}-step mean v-MSE: {last_window_mean:.6f}")
    print(f"fixed probe: {initial_probe_mse:.6f} -> {final_probe_mse:.6f}")
    print(
        f"t=999 x0 RMSE: {pre_profile[-1]['x0_rmse']:.6f} "
        f"-> {post_profile[-1]['x0_rmse']:.6f}"
    )
    print(
        f"t=999 x0 std: {pre_profile[-1]['x0_pred_std']:.6f} "
        f"-> {post_profile[-1]['x0_pred_std']:.6f}"
    )
    print(f"sampling EO normalized std: {pre_eo_z.std():.6f} -> {post_eo_z.std():.6f}")
    print(f"sampling EC normalized std: {pre_ec_z.std():.6f} -> {post_ec_z.std():.6f}")
    print(f"post O1 alpha EC-EO: {alpha['O1']['median_ec_minus_eo_db']:.3f} dB")
    print(f"post O2 alpha EC-EO: {alpha['O2']['median_ec_minus_eo_db']:.3f} dB")
    print(f"AMP skipped steps: {amp_skips}")
    print(f"training time: {training_seconds / 60:.2f} min")
    print("Validation EEG accessed: NO")
    print("Final Test accessed: NO")
    print("\nNEXT: INTERPRET PILOT BEFORE FREEZING THE FULL B0-V FINE-TUNING PROTOCOL")


if __name__ == "__main__":
    main()
