from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)
from s2_24_full_train_loader_sanity import CachedRealTrainWindowDataset
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_39_b0_epsilon_to_v_conversion_sanity import make_v_target, q_sample
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
from s2_57_b2_matched_lambda0_full_continuation_control import (
    build_b2_model,
    finite_gradient_summary,
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

CANONICAL_E5_CHECKPOINT = CONTROL_ROOT / "checkpoints" / "candidate_epoch_005_raw.pt"

S2_58_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "joint_b1_lambda0_lambda001_freegen_audit_v1"
)

CANONICAL_E5_SYNTHETIC = (
    S2_58_ROOT / "generated" / "lambda0_control_epoch_005_synthetic_v1.npz"
)

S2_63_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "control_e5_training_seed_replication_v1"
)

REPLICA_A_E5_CHECKPOINT = (
    S2_63_ROOT / "checkpoints" / "s2_63_replica_a_epoch005_raw.pt"
)

REPLICA_B_E5_CHECKPOINT = (
    S2_63_ROOT / "checkpoints" / "s2_63_replica_b_epoch005_raw.pt"
)

REPLICA_A_E5_SYNTHETIC = (
    S2_63_ROOT / "generated" / "s2_63_replica_a_seed_20261001_synthetic_v1.npz"
)

REPLICA_B_E5_SYNTHETIC = (
    S2_63_ROOT / "generated" / "s2_63_replica_b_seed_20261001_synthetic_v1.npz"
)

S2_64_SUMMARY = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "training_seed_instability_diagnostics_v1" / "metrics"
    / "s2_64_training_seed_instability_summary_v1.json"
)

RUN_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "early_epoch_training_seed_checkpoint_landscape_v1"
)

METRIC_DIR = RUN_ROOT / "metrics"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
GENERATED_DIR = RUN_ROOT / "generated"
LOG_DIR = RUN_ROOT / "logs"

PROTOCOL_FILE = (
    METRIC_DIR / "s2_65_early_epoch_checkpoint_landscape_protocol_v1.json"
)
TRAINING_CSV = LOG_DIR / "s2_65_reconstructed_training_history_v1.csv"
LANDSCAPE_CSV = METRIC_DIR / "s2_65_checkpoint_landscape_metrics_v1.csv"
SUMMARY_JSON = METRIC_DIR / "s2_65_checkpoint_landscape_summary_v1.json"

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

TIMESTEPS = 1000
COSINE_S = 0.008

EPOCHS = [1, 2, 3, 4, 5]
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MIXED_PRECISION = True
EMA_ENABLED = False
LAMBDA_SPECTRAL = 0.0

MAX_AMP_SKIPS_PER_EPOCH = 5
MAX_CONSECUTIVE_AMP_SKIPS = 3

TRAINING_STREAMS = {
    "CANONICAL": {
        "master_seed": 20260908,
        "loader_seed": 20261021,
        "noise_seed": 20261022,
        "historical_e5_checkpoint": str(CANONICAL_E5_CHECKPOINT),
        "historical_e5_synthetic": str(CANONICAL_E5_SYNTHETIC),
    },
    "REPLICA_A": {
        "master_seed": 20261201,
        "loader_seed": 20261202,
        "noise_seed": 20261203,
        "historical_e5_checkpoint": str(REPLICA_A_E5_CHECKPOINT),
        "historical_e5_synthetic": str(REPLICA_A_E5_SYNTHETIC),
    },
    "REPLICA_B": {
        "master_seed": 20261301,
        "loader_seed": 20261302,
        "noise_seed": 20261303,
        "historical_e5_checkpoint": str(REPLICA_B_E5_CHECKPOINT),
        "historical_e5_synthetic": str(REPLICA_B_E5_SYNTHETIC),
    },
}

SAMPLING_SEED_BASE = 20261001

SELECTED_SUBJECT_COUNT = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4
N_CHANNELS = 19
N_TIMES = 512

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
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def atomic_npz_save(path: Path, **arrays) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
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


def checkpoint_path(stream_name: str, epoch: int) -> Path:
    return (
        CHECKPOINT_DIR
        / f"s2_65_{stream_name.lower()}_epoch_{epoch:03d}_raw.pt"
    )


def synthetic_path(stream_name: str, epoch: int) -> Path:
    return (
        GENERATED_DIR
        / f"s2_65_{stream_name.lower()}_epoch_{epoch:03d}"
        f"_seed_{SAMPLING_SEED_BASE}_synthetic_v1.npz"
    )


def load_audit_reference():
    real_package = np.load(B1_REAL_REFERENCE_NPZ, allow_pickle=False)
    b1_package = np.load(B1_SYNTHETIC_REFERENCE_NPZ, allow_pickle=False)

    real_z = real_package["real_z"].astype(np.float32, copy=False)
    b1_synth_z = b1_package["epoch_010_z"].astype(np.float32, copy=False)

    subject_indices = (
        real_package["selected_subject_indices"].astype(np.int64).tolist()
    )
    subject_ids = real_package["selected_subject_ids"].astype(str).tolist()

    if (
        b1_package["selected_subject_indices"].astype(np.int64).tolist()
        != subject_indices
    ):
        raise RuntimeError("B1 audit subject-index mismatch.")

    if b1_package["selected_subject_ids"].astype(str).tolist() != subject_ids:
        raise RuntimeError("B1 audit subject-ID mismatch.")

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(f"Unexpected Real reference shape: {real_z.shape}")

    if b1_synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(f"Unexpected B1 synthetic shape: {b1_synth_z.shape}")

    return real_z, b1_synth_z, subject_indices, subject_ids


def historical_e5_paths(stream_name: str) -> tuple[Path, Path]:
    cfg = TRAINING_STREAMS[stream_name]
    return (
        Path(cfg["historical_e5_checkpoint"]),
        Path(cfg["historical_e5_synthetic"]),
    )


def load_historical_e5_synthetic(
    *, stream_name: str, selected_subjects, selected_subject_ids
) -> np.ndarray:
    _, synthetic_file = historical_e5_paths(stream_name)

    package = np.load(synthetic_file, allow_pickle=False)

    if (
        package["selected_subject_indices"].astype(np.int64).tolist()
        != selected_subjects
    ):
        raise RuntimeError(f"Historical E5 cohort mismatch for {stream_name}.")

    if package["selected_subject_ids"].astype(str).tolist() != selected_subject_ids:
        raise RuntimeError(f"Historical E5 subject-ID mismatch for {stream_name}.")

    key = "synth_z"

    values = package[key].astype(np.float32, copy=False)

    if values.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Historical E5 synthetic shape mismatch for {stream_name}."
        )

    return values


def train_stream(
    *,
    stream_name: str,
    config: dict[str, Any],
    frozen_b1: dict[str, Any],
    train_dataset,
    device,
):
    print("\n" + "=" * 80)
    print(f"RECONSTRUCT TRAINING STREAM {stream_name}")
    print("=" * 80)

    set_seed(int(config["master_seed"]))

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config["loader_seed"]))

    noise_generator = torch.Generator(device="cuda")
    noise_generator.manual_seed(int(config["noise_seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    if len(train_loader) != 748:
        raise RuntimeError("Expected 748 Train batches.")

    model = build_b2_model(frozen_b1, device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=MIXED_PRECISION,
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )

    alpha_bar = schedule_cpu["alpha_cumprod"].to(
        device=device,
        dtype=torch.float32,
    )

    history_rows = []
    epoch_checkpoints = {}

    total_amp_skips = 0

    for epoch in EPOCHS:
        epoch_start = time.perf_counter()

        model.train()

        train_v_sum = 0.0
        presented = 0
        successful_steps = 0
        amp_skips = 0
        consecutive_amp_skips = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            x0 = batch["x"].to(device, non_blocking=True)
            state = batch["y"].to(device, non_blocking=True)
            subject = batch["subject_index"].to(device, non_blocking=True)

            current_batch_size = int(x0.shape[0])

            timesteps = torch.randint(
                0,
                TIMESTEPS,
                (current_batch_size,),
                device=device,
                dtype=torch.long,
                generator=noise_generator,
            )

            epsilon = torch.randn(
                x0.shape,
                device=device,
                dtype=x0.dtype,
                generator=noise_generator,
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

            optimizer.zero_grad(set_to_none=True)

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

            v_loss = torch.mean(
                (
                    pred_v.float()
                    - target_v.float()
                )
                ** 2
            )

            if not torch.isfinite(v_loss):
                raise RuntimeError(
                    f"Nonfinite v-loss in {stream_name}, epoch {epoch}."
                )

            v_value = float(v_loss.item())

            grad_scaler.scale(v_loss).backward()
            grad_scaler.unscale_(optimizer)

            finite, _, _, bad_names = finite_gradient_summary(model)

            scale_before = float(grad_scaler.get_scale())

            if finite:
                clip_return = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP_NORM,
                )

                if not torch.isfinite(clip_return):
                    raise RuntimeError(
                        f"Nonfinite clip norm in {stream_name}."
                    )

                grad_scaler.step(optimizer)
                grad_scaler.update()

                successful_steps += 1
                consecutive_amp_skips = 0
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()

                amp_skips += 1
                total_amp_skips += 1
                consecutive_amp_skips += 1

                print(
                    f"AMP SKIP | stream={stream_name} "
                    f"| epoch={epoch} | batch={batch_index} "
                    f"| scale={scale_before:g}->{grad_scaler.get_scale():g} "
                    f"| bad={bad_names[:3]}"
                )

                if amp_skips > MAX_AMP_SKIPS_PER_EPOCH:
                    raise RuntimeError("Too many AMP skips.")
                if consecutive_amp_skips >= MAX_CONSECUTIVE_AMP_SKIPS:
                    raise RuntimeError("Too many consecutive AMP skips.")

            train_v_sum += v_value * current_batch_size
            presented += current_batch_size

        if presented != EXPECTED_TRAIN_WINDOWS:
            raise RuntimeError("Train accounting mismatch.")

        if successful_steps + amp_skips != len(train_loader):
            raise RuntimeError("Optimizer-step accounting mismatch.")

        epoch_seconds = time.perf_counter() - epoch_start

        model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        model_hash = state_dict_sha256(model_state)

        checkpoint = {
            "status": "S2_65_EARLY_EPOCH_RECONSTRUCTED_CHECKPOINT",
            "stream": stream_name,
            "epoch": epoch,
            "lambda_spectral": 0.0,
            "model_state_dict": model_state,
            "model_state_sha256": model_hash,
            "seeds": {
                "master_seed": int(config["master_seed"]),
                "loader_seed": int(config["loader_seed"]),
                "noise_seed": int(config["noise_seed"]),
            },
            "train_v_mse": float(train_v_sum / presented),
            "successful_optimizer_steps": successful_steps,
            "amp_skipped_steps": amp_skips,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
        }

        atomic_torch_save(
            checkpoint,
            checkpoint_path(stream_name, epoch),
        )

        epoch_checkpoints[epoch] = checkpoint

        history_rows.append(
            {
                "stream": stream_name,
                "epoch": epoch,
                "train_v_mse": checkpoint["train_v_mse"],
                "successful_optimizer_steps": successful_steps,
                "amp_skipped_steps": amp_skips,
                "model_state_sha256": model_hash,
                "epoch_seconds": epoch_seconds,
            }
        )

        print(
            f"{stream_name} E{epoch} "
            f"| Train-v={checkpoint['train_v_mse']:.6f} "
            f"| skips={amp_skips} "
            f"| hash={model_hash[:16]}..."
        )

    historical_e5_checkpoint, _ = historical_e5_paths(stream_name)

    historical_e5 = torch.load(
        historical_e5_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    historical_hash = historical_e5["model_state_sha256"]

    reconstructed_hash = epoch_checkpoints[5]["model_state_sha256"]

    if reconstructed_hash != historical_hash:
        raise RuntimeError(
            f"E5 hash reproduction FAILED for {stream_name}: "
            f"reconstructed={reconstructed_hash}, "
            f"historical={historical_hash}"
        )

    print(
        f"{stream_name} E5 historical hash reproduction: PASS"
    )

    del model
    torch.cuda.empty_cache()

    return epoch_checkpoints, history_rows


def generate_checkpoint(
    *,
    stream_name,
    epoch,
    checkpoint,
    frozen_b1,
    selected_subjects,
    selected_subject_ids,
    reverse_schedule,
    device,
):
    output_path = synthetic_path(stream_name, epoch)

    if output_path.is_file():
        package = np.load(output_path, allow_pickle=False)

        if (
            package["selected_subject_indices"].astype(np.int64).tolist()
            != selected_subjects
        ):
            raise RuntimeError("Cached cohort mismatch.")

        if package["selected_subject_ids"].astype(str).tolist() != selected_subject_ids:
            raise RuntimeError("Cached subject-ID mismatch.")

        if str(package["model_state_sha256"][0]) != checkpoint["model_state_sha256"]:
            raise RuntimeError("Cached model-state hash mismatch.")

        values = package["synth_z"].astype(np.float32, copy=False)

        if values.shape != EXPECTED_SYNTH_SHAPE:
            raise RuntimeError("Cached synthetic shape mismatch.")

        print(
            f"{stream_name} E{epoch}: validated cache -> REUSE"
        )

        return values

    model = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=checkpoint["model_state_dict"],
        device=device,
    )

    print("\n" + "-" * 80)
    print(f"FREE GENERATION — {stream_name} E{epoch}")
    print("-" * 80)

    start = time.perf_counter()

    eo = sample_b1_candidate_state(
        model=model,
        selected_subjects=selected_subjects,
        eye_state_value=0,
        reverse_schedule=reverse_schedule,
        device=device,
        base_seed=SAMPLING_SEED_BASE,
    )

    ec = sample_b1_candidate_state(
        model=model,
        selected_subjects=selected_subjects,
        eye_state_value=1,
        reverse_schedule=reverse_schedule,
        device=device,
        base_seed=SAMPLING_SEED_BASE,
    )

    synth_z = np.stack([eo, ec], axis=1).astype(np.float32, copy=False)

    if synth_z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(
            f"Unexpected synthetic shape: {synth_z.shape}"
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
        model_state_sha256=np.asarray(
            [checkpoint["model_state_sha256"]],
            dtype="U64",
        ),
        sampling_seed=np.asarray(
            [SAMPLING_SEED_BASE],
            dtype=np.int64,
        ),
        generation_seconds=np.asarray(
            [time.perf_counter() - start],
            dtype=np.float64,
        ),
    )

    print(
        f"{stream_name} E{epoch} generation: "
        f"{(time.perf_counter()-start)/60:.2f} min"
    )

    del model
    torch.cuda.empty_cache()

    return synth_z


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

    return {
        "generic": generic,
        "full_log_psd": full_psd,
    }


def landscape_row(
    *,
    stream_name,
    epoch,
    bundle,
    b1_bundle,
):
    eligibility = evaluate_eligibility(
        candidate=bundle,
        b1_baseline=b1_bundle,
    )

    generic = bundle["generic"]
    subject = generic["subject_identity"]

    failed_criteria = [
        criterion
        for criterion, passed
        in eligibility["criteria"].items()
        if not bool(passed)
    ]

    return {
        "stream": stream_name,
        "epoch": epoch,
        "eligible_vs_B1": bool(eligibility["eligible"]),
        "failed_criteria": ";".join(failed_criteria),
        "full_psd_mae_db": float(
            bundle["full_log_psd"]["mean_mae_db"]
        ),
        "mean_bandpower_mae_db": float(
            generic["mean_bandpower_mae_db"]
        ),
        "alpha_effect_mae_db": float(
            generic["alpha_effect"]["channel_mae_db"]
        ),
        "scale_EO": float(
            generic["scale"]["EO"]["normalized_std_ratio"]
        ),
        "scale_EC": float(
            generic["scale"]["EC"]["normalized_std_ratio"]
        ),
        "subject_top1": float(
            subject["subject_retrieval_top1_accuracy"]
        ),
        "subject_fingerprint_r_median": float(
            subject["subject_fingerprint_pearson_r_median"]
        ),
        "subject_margin": float(
            subject["subject_correct_vs_median_wrong_margin_mean"]
        ),
    }


def main():
    print("=" * 80)
    print("SCENARIO 2 — s2_65 EARLY-EPOCH TRAINING-SEED CHECKPOINT LANDSCAPE")
    print("=" * 80)

    print("\nScientific question:")
    print("  Is s2_63 instability caused by a seed-dependent optimal stopping time?")
    print("  We reconstruct E1-E5 for the same 3 known training streams.")

    print("\nThis is a DEVELOPMENT DIAGNOSTIC:")
    print("  It does NOT freeze a model.")
    print("  It does NOT change thresholds.")
    print("  It does NOT use Validation or Final Test.")
    print("  Any epoch chosen here must later be confirmed on fresh training seeds.")

    for directory in (
        METRIC_DIR,
        CHECKPOINT_DIR,
        GENERATED_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    required_files = [
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_REAL_REFERENCE_NPZ,
        B1_SYNTHETIC_REFERENCE_NPZ,
        CANONICAL_E5_CHECKPOINT,
        CANONICAL_E5_SYNTHETIC,
        REPLICA_A_E5_CHECKPOINT,
        REPLICA_B_E5_CHECKPOINT,
        REPLICA_A_E5_SYNTHETIC,
        REPLICA_B_E5_SYNTHETIC,
        S2_64_SUMMARY,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    with S2_64_SUMMARY.open("r", encoding="utf-8") as f:
        s2_64_summary = json.load(f)

    if (
        s2_64_summary["status"]
        != "S2_64_TRAINING_SEED_INSTABILITY_DIAGNOSTICS_COMPLETE"
    ):
        raise RuntimeError("s2_64 is not complete.")

    frozen_b1_file_hash_before = sha256_file(FROZEN_B1_GENERATOR)

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    protocol = {
        "status": "S2_65_EARLY_EPOCH_CHECKPOINT_LANDSCAPE_PROTOCOL_FROZEN",
        "scientific_role": (
            "Development-only diagnosis of whether training-seed instability "
            "reflects seed-dependent checkpoint timing."
        ),
        "training_streams": TRAINING_STREAMS,
        "epochs": EPOCHS,
        "training_recipe": {
            "initialization": "exact frozen B1 RAW epoch 10",
            "lambda_spectral": 0.0,
            "prediction_target": "v",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "mixed_precision": MIXED_PRECISION,
            "EMA_enabled": EMA_ENABLED,
        },
        "required_reconstruction_check": (
            "For each stream, reconstructed E5 model-state SHA256 must exactly "
            "match the historical E5 checkpoint before any landscape interpretation."
        ),
        "sampling": {
            "seed_fixed": SAMPLING_SEED_BASE,
            "sampler": "1000-step ancestral DDPM / v / FP32",
            "historical_E5_samples_reused": True,
            "new_samples_required": "E1-E4 only for each stream",
        },
        "development_rule": (
            "Identify all fixed epochs in E1-E5 that pass ALL existing frozen "
            "B2 criteria for ALL THREE known training streams. "
            "If multiple common eligible epochs exist, choose the one with lowest "
            "mean full PSD MAE across the three streams, tie-break by earlier epoch. "
            "This choice is development-only and MUST be confirmed on fresh seeds."
        ),
        "if_no_common_epoch": (
            "Do not manually choose per-seed epochs. Conclude fixed-epoch recipe "
            "is not robust in this range and design a genuine checkpoint-selection mechanism."
        ),
        "Validation_EEG_accessed": False,
        "Final_Test_accessed": False,
        "thresholds_changed": False,
        "formal_model_freeze_decision_made": False,
    }

    if PROTOCOL_FILE.is_file():
        with PROTOCOL_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != protocol:
            raise RuntimeError("Conflicting s2_65 protocol already exists.")
        print("identical s2_65 protocol already exists.")
    else:
        atomic_json_save(protocol, PROTOCOL_FILE)
        print(f"protocol: {PROTOCOL_FILE}")

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked.")

    train_records, train_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )

    if len(train_records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError("Unexpected Train window count.")
    if len(train_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError("Unexpected subject count.")

    train_base_dataset = CachedRealTrainWindowDataset(
        records=train_records,
        scaler=scaler,
        max_open_files=8,
    )

    train_dataset = SubjectIndexedDataset(
        base_dataset=train_base_dataset,
        records=train_records,
    )

    real_z, b1_synth_z, selected_subjects, selected_subject_ids = (
        load_audit_reference()
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for s2_65.")

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_checkpoints = {}
    all_history_rows = []

    for stream_name, config in TRAINING_STREAMS.items():
        checkpoints, history_rows = train_stream(
            stream_name=stream_name,
            config=config,
            frozen_b1=frozen_b1,
            train_dataset=train_dataset,
            device=device,
        )

        all_checkpoints[stream_name] = checkpoints
        all_history_rows.extend(history_rows)

    save_csv_rows(all_history_rows, TRAINING_CSV)

    print("\nALL THREE E5 HASH REPRODUCTIONS PASSED.")

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )
    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    synthetic = {}

    for stream_name in TRAINING_STREAMS:
        for epoch in EPOCHS:
            if epoch == 5:
                values = load_historical_e5_synthetic(
                    stream_name=stream_name,
                    selected_subjects=selected_subjects,
                    selected_subject_ids=selected_subject_ids,
                )

                print(
                    f"{stream_name} E5 historical synthetic cache -> REUSE"
                )
            else:
                values = generate_checkpoint(
                    stream_name=stream_name,
                    epoch=epoch,
                    checkpoint=all_checkpoints[stream_name][epoch],
                    frozen_b1=frozen_b1,
                    selected_subjects=selected_subjects,
                    selected_subject_ids=selected_subject_ids,
                    reverse_schedule=reverse_schedule,
                    device=device,
                )

            synthetic[
                (
                    stream_name,
                    epoch,
                )
            ] = values

    b1_bundle = evaluate_bundle(
        real_z=real_z,
        synth_z=b1_synth_z,
        scaler=scaler,
    )

    landscape_rows = []

    print("\n" + "=" * 80)
    print("CHECKPOINT LANDSCAPE")
    print("=" * 80)

    for stream_name in TRAINING_STREAMS:
        for epoch in EPOCHS:
            bundle = evaluate_bundle(
                real_z=real_z,
                synth_z=synthetic[(stream_name, epoch)],
                scaler=scaler,
            )

            row = landscape_row(
                stream_name=stream_name,
                epoch=epoch,
                bundle=bundle,
                b1_bundle=b1_bundle,
            )

            landscape_rows.append(row)

            print(
                f"{stream_name} E{epoch}: "
                f"eligible={row['eligible_vs_B1']} "
                f"| PSD={row['full_psd_mae_db']:.6f} "
                f"| band={row['mean_bandpower_mae_db']:.6f} "
                f"| alpha={row['alpha_effect_mae_db']:.6f} "
                f"| failed={row['failed_criteria'] or 'NONE'}"
            )

    save_csv_rows(landscape_rows, LANDSCAPE_CSV)

    common_eligible_epochs = []

    epoch_mean_psd = {}

    for epoch in EPOCHS:
        epoch_rows = [
            row
            for row in landscape_rows
            if row["epoch"] == epoch
        ]

        if len(epoch_rows) != len(TRAINING_STREAMS):
            raise RuntimeError("Landscape epoch accounting mismatch.")

        if all(
            bool(row["eligible_vs_B1"])
            for row in epoch_rows
        ):
            common_eligible_epochs.append(epoch)

        epoch_mean_psd[str(epoch)] = float(
            np.mean(
                [
                    row["full_psd_mae_db"]
                    for row in epoch_rows
                ]
            )
        )

    development_fixed_epoch = None

    if common_eligible_epochs:
        development_fixed_epoch = min(
            common_eligible_epochs,
            key=lambda epoch: (
                epoch_mean_psd[str(epoch)],
                epoch,
            ),
        )

    per_stream_eligible_epochs = {
        stream_name: [
            row["epoch"]
            for row in landscape_rows
            if (
                row["stream"] == stream_name
                and bool(row["eligible_vs_B1"])
            )
        ]
        for stream_name in TRAINING_STREAMS
    }

    print("\n" + "=" * 80)
    print("DEVELOPMENT INTERPRETATION")
    print("=" * 80)

    print(
        f"per-stream eligible epochs: "
        f"{per_stream_eligible_epochs}"
    )

    print(
        f"common eligible epochs: "
        f"{common_eligible_epochs}"
    )

    print(
        f"development fixed-epoch candidate: "
        f"{development_fixed_epoch}"
    )

    if sha256_file(FROZEN_B1_GENERATOR) != frozen_b1_file_hash_before:
        raise RuntimeError("Frozen B1 changed during s2_65.")

    summary = {
        "status": "S2_65_EARLY_EPOCH_CHECKPOINT_LANDSCAPE_COMPLETE",
        "scientific_role": (
            "Development-only early-checkpoint landscape across the three "
            "already-known training streams."
        ),
        "all_E5_historical_hash_reproductions_passed": True,
        "per_stream_eligible_epochs": per_stream_eligible_epochs,
        "common_eligible_epochs": common_eligible_epochs,
        "epoch_mean_full_psd_mae_db": epoch_mean_psd,
        "development_fixed_epoch_candidate": development_fixed_epoch,
        "candidate_is_final_or_frozen": False,
        "required_next_if_candidate_exists": (
            "Test this single fixed epoch on fresh independent training seeds "
            "under a separately frozen protocol."
            if development_fixed_epoch is not None
            else None
        ),
        "required_next_if_no_candidate": (
            "Design a genuine checkpoint-selection mechanism; do not manually "
            "choose a different epoch for each seed."
            if development_fixed_epoch is None
            else None
        ),
        "artifacts": {
            "protocol": str(PROTOCOL_FILE),
            "training_csv": str(TRAINING_CSV),
            "landscape_csv": str(LANDSCAPE_CSV),
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": False,
            "Final_Test_accessed": False,
            "thresholds_changed": False,
            "formal_model_freeze_decision_made": False,
            "frozen_B1_unchanged": True,
        },
        "next": (
            "If development_fixed_epoch_candidate exists: confirm it on fresh "
            "training seeds. Otherwise design a checkpoint-selection mechanism."
        ),
    }

    atomic_json_save(summary, SUMMARY_JSON)

    train_base_dataset.close()

    print("\n" + "=" * 80)
    print("s2_65 EARLY-EPOCH TRAINING-SEED CHECKPOINT LANDSCAPE COMPLETE")
    print("=" * 80)

    print(f"common eligible epochs: {common_eligible_epochs}")
    print(
        f"development fixed-epoch candidate: "
        f"{development_fixed_epoch}"
    )

    print("\nValidation EEG accessed: NO")
    print("Final Test accessed: NO")
    print("Formal model freeze decision made: NO")

    print("\nNEXT:")

    if development_fixed_epoch is None:
        print(
            "NO COMMON FIXED EPOCH — DESIGN A GENUINE ROBUST "
            "CHECKPOINT-SELECTION MECHANISM."
        )
    else:
        print(
            "CONFIRM THIS SINGLE DEVELOPMENT EPOCH ON FRESH "
            "INDEPENDENT TRAINING SEEDS BEFORE ANY FREEZE."
        )


if __name__ == "__main__":
    main()
