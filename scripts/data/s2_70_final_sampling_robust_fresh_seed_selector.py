from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import s2_67_fresh_seed_validation_checkpoint_selection as base

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)
from s2_24_full_train_loader_sanity import CachedRealTrainWindowDataset
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import build_reverse_schedule
from s2_47_b1_architecture_warmstart_sanity import (
    SubjectIndexedDataset,
    state_dict_sha256,
)


# ============================================================
# Paths
# ============================================================

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

S2_66_ROOT = (
    REPO_ROOT / "outputs" / "protocols" / "scenario2" / "b2"
    / "validation_checkpoint_selector_v1"
)

VAL_DEV_REAL_NPZ = (
    S2_66_ROOT / "references" / "s2_66_val_dev_real_reference_v1.npz"
)

VAL_CONF_REAL_NPZ = (
    S2_66_ROOT / "references" / "s2_66_val_conf_real_reference_v1.npz"
)

S2_66_VAL_DEV_B1 = (
    S2_66_ROOT / "generated" / "s2_66_val_dev_b1_synthetic_v1.npz"
)

S2_66_VAL_CONF_B1 = (
    S2_66_ROOT / "generated" / "s2_66_val_conf_b1_synthetic_v1.npz"
)

S2_69_ROOT = (
    REPO_ROOT / "outputs" / "protocols" / "scenario2" / "b2"
    / "sampling_robust_checkpoint_selector_v2"
)

S2_69_PROTOCOL = (
    S2_69_ROOT / "metrics" / "s2_69_sampling_robust_selector_protocol_v2.json"
)

S2_69_SUMMARY = (
    S2_69_ROOT / "metrics" / "s2_69_sampling_robust_selector_summary_v2.json"
)

S2_69_BASELINES = (
    S2_69_ROOT / "metrics" / "s2_69_multiseed_b1_validation_baselines_v2.json"
)

RUN_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "final_sampling_robust_fresh_seed_selector_v1"
)

CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
GENERATED_DEV_DIR = RUN_ROOT / "generated" / "val_dev"
GENERATED_CONF_DIR = RUN_ROOT / "generated" / "val_conf"
METRIC_DIR = RUN_ROOT / "metrics"
LOG_DIR = RUN_ROOT / "logs"

PROTOCOL_SNAPSHOT = METRIC_DIR / "s2_70_final_selector_protocol_snapshot_v1.json"
TRAINING_HISTORY_CSV = LOG_DIR / "s2_70_training_history_v1.csv"
VAL_DEV_SEEDWISE_CSV = METRIC_DIR / "s2_70_val_dev_seedwise_metrics_v1.csv"
VAL_DEV_AGGREGATE_CSV = METRIC_DIR / "s2_70_val_dev_checkpoint_aggregate_v1.csv"
VAL_CONF_SEEDWISE_CSV = METRIC_DIR / "s2_70_val_conf_seedwise_metrics_v1.csv"
DECISION_JSON = METRIC_DIR / "s2_70_final_selector_decision_v1.json"
SUMMARY_JSON = METRIC_DIR / "s2_70_final_sampling_robust_selector_summary_v1.json"


# ============================================================
# Frozen constants — must match s2_69 / prior continuation recipe
# ============================================================

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

EPOCHS = [1, 2, 3, 4, 5]

N_SUBJECTS = 64
REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4
N_CHANNELS = 19
N_TIMES = 512

EXPECTED_REAL_SHAPE = (
    N_SUBJECTS,
    2,
    REAL_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)

EXPECTED_SYNTH_SHAPE = (
    N_SUBJECTS,
    2,
    SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
    N_CHANNELS,
    N_TIMES,
)


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp, path)


def save_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def checkpoint_path(run_name: str, epoch: int) -> Path:
    return (
        CHECKPOINT_DIR
        / f"s2_70_{run_name.lower()}_epoch_{epoch:03d}_raw.pt"
    )


def candidate_synthetic_path(
    stage: str,
    run_name: str,
    epoch: int,
    sampling_seed: int,
) -> Path:
    root = GENERATED_DEV_DIR if stage == "VAL_DEV" else GENERATED_CONF_DIR

    return (
        root
        / (
            f"s2_70_{run_name.lower()}_epoch_{epoch:03d}_"
            f"{stage.lower()}_seed_{sampling_seed}_synthetic_v1.npz"
        )
    )


def b1_baseline_path(stage: str, sampling_seed: int) -> Path:
    if stage == "VAL_DEV":
        if sampling_seed == 20261511:
            return S2_66_VAL_DEV_B1
        return (
            S2_69_ROOT / "generated"
            / f"s2_69_val_dev_b1_seed_{sampling_seed}_synthetic_v2.npz"
        )

    if stage == "VAL_CONF":
        if sampling_seed == 20261512:
            return S2_66_VAL_CONF_B1
        return (
            S2_69_ROOT / "generated"
            / f"s2_69_val_conf_b1_seed_{sampling_seed}_synthetic_v2.npz"
        )

    raise ValueError(stage)


def load_real_reference(path: Path):
    package = np.load(path, allow_pickle=False)

    real_z = package["real_z"].astype(np.float32, copy=False)
    subject_indices = package["selected_subject_indices"].astype(np.int64).tolist()
    subject_ids = package["selected_subject_ids"].astype(str).tolist()

    if real_z.shape != EXPECTED_REAL_SHAPE:
        raise RuntimeError(f"Unexpected Real shape: {real_z.shape}")

    return real_z, subject_indices, subject_ids


def load_b1_baseline(
    *,
    stage: str,
    sampling_seed: int,
    expected_subject_indices: list[int],
    expected_subject_ids: list[str],
    expected_model_hash: str,
) -> np.ndarray:
    path = b1_baseline_path(stage, sampling_seed)

    if not path.is_file():
        raise FileNotFoundError(path)

    package = np.load(path, allow_pickle=False)

    indices = package["selected_subject_indices"].astype(np.int64).tolist()
    ids = package["selected_subject_ids"].astype(str).tolist()

    if indices != expected_subject_indices:
        raise RuntimeError(f"{stage}/{sampling_seed}: B1 cohort mismatch.")

    if ids != expected_subject_ids:
        raise RuntimeError(f"{stage}/{sampling_seed}: B1 ID mismatch.")

    if int(package["sampling_seed"][0]) != sampling_seed:
        raise RuntimeError(f"{stage}/{sampling_seed}: B1 sampling-seed mismatch.")

    if str(package["model_state_sha256"][0]) != expected_model_hash:
        raise RuntimeError(f"{stage}/{sampling_seed}: B1 model-hash mismatch.")

    z = package["synth_z"].astype(np.float32, copy=False)

    if z.shape != EXPECTED_SYNTH_SHAPE:
        raise RuntimeError(f"{stage}/{sampling_seed}: B1 shape mismatch {z.shape}.")

    return z


def evaluate_bundle(real_z, synth_z, scaler):
    return base.evaluate_bundle(
        real_z=real_z,
        synth_z=synth_z,
        scaler=scaler,
    )


def seedwise_metric_row(
    *,
    run_name: str,
    stage: str,
    epoch: int,
    sampling_seed: int,
    candidate_bundle: dict[str, Any],
    b1_bundle: dict[str, Any],
):
    row, eligibility = base.flatten_candidate_metrics(
        run_name=run_name,
        epoch=epoch,
        stage=stage,
        bundle=candidate_bundle,
        b1_bundle=b1_bundle,
    )

    # Rename the eligibility column to emphasize matching-seed B1 comparison.
    row["sampling_seed"] = int(sampling_seed)
    row["eligible_vs_matching_B1"] = bool(row.pop("eligible_vs_B1"))

    return row, eligibility


def aggregate_val_dev_rows(
    *,
    run_name: str,
    seedwise_rows: list[dict[str, Any]],
    sampling_seeds: list[int],
):
    aggregate_rows = []

    for epoch in EPOCHS:
        rows = [
            row
            for row in seedwise_rows
            if (
                row["run"] == run_name
                and row["stage"] == "VAL_DEV"
                and int(row["epoch"]) == epoch
            )
        ]

        if len(rows) != len(sampling_seeds):
            raise RuntimeError(
                f"{run_name} E{epoch}: expected {len(sampling_seeds)} "
                f"seedwise rows, got {len(rows)}."
            )

        pass_count = sum(
            bool(row["eligible_vs_matching_B1"])
            for row in rows
        )

        robust = pass_count == len(sampling_seeds)

        aggregate_rows.append(
            {
                "run": run_name,
                "epoch": epoch,
                "robust_eligible_3_of_3": bool(robust),
                "seed_pass_count": int(pass_count),
                "failed_sampling_seeds": ";".join(
                    str(int(row["sampling_seed"]))
                    for row in rows
                    if not bool(row["eligible_vs_matching_B1"])
                ),
                "mean_full_psd_mae_db": float(
                    np.mean([row["full_psd_mae_db"] for row in rows])
                ),
                "std_full_psd_mae_db_ddof0": float(
                    np.std(
                        [row["full_psd_mae_db"] for row in rows],
                        ddof=0,
                    )
                ),
                "mean_bandpower_mae_db": float(
                    np.mean([row["mean_bandpower_mae_db"] for row in rows])
                ),
                "mean_alpha_effect_mae_db": float(
                    np.mean([row["alpha_effect_mae_db"] for row in rows])
                ),
            }
        )

    return aggregate_rows


def select_robust_epoch(rows: list[dict[str, Any]]) -> int | None:
    eligible = [
        row
        for row in rows
        if bool(row["robust_eligible_3_of_3"])
    ]

    if not eligible:
        return None

    selected = min(
        eligible,
        key=lambda row: (
            float(row["mean_full_psd_mae_db"]),
            float(row["mean_bandpower_mae_db"]),
            float(row["mean_alpha_effect_mae_db"]),
            int(row["epoch"]),
        ),
    )

    return int(selected["epoch"])


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — s2_70 FINAL SAMPLING-ROBUST FRESH-SEED SELECTOR")
    print("=" * 80)

    print("\nScientific question:")
    print(
        "  Can the FINAL pre-frozen continuation-B2 selector replicate "
        "across both fresh training seeds and 3 sampling seeds per "
        "Validation stage?"
    )

    print("\nTerminal governance:")
    print("  Threshold changes: NO")
    print("  Training-seed replacement: NO")
    print("  Additional rescue after failure: NO")
    print("  Final Test access: NO")

    for directory in (
        CHECKPOINT_DIR,
        GENERATED_DEV_DIR,
        GENERATED_CONF_DIR,
        METRIC_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for path in (
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        S2_69_PROTOCOL,
        S2_69_SUMMARY,
        S2_69_BASELINES,
        VAL_DEV_REAL_NPZ,
        VAL_CONF_REAL_NPZ,
        NORMALIZATION_FILE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with B1_SELECTION_MANIFEST.open("r", encoding="utf-8") as f:
        b1_selection = json.load(f)

    with S2_69_PROTOCOL.open("r", encoding="utf-8") as f:
        protocol = json.load(f)

    with S2_69_SUMMARY.open("r", encoding="utf-8") as f:
        s2_69_summary = json.load(f)

    with S2_69_BASELINES.open("r", encoding="utf-8") as f:
        s2_69_baselines = json.load(f)

    if b1_selection["status"] != "B1_SELECTION_FROZEN":
        raise RuntimeError("B1 is not frozen.")

    if (
        protocol["status"]
        != "S2_69_SAMPLING_ROBUST_CHECKPOINT_SELECTOR_V2_FROZEN"
    ):
        raise RuntimeError("Unexpected s2_69 protocol status.")

    if s2_69_summary["status"] != "S2_69_SAMPLING_ROBUST_SELECTOR_V2_READY":
        raise RuntimeError("s2_69 is not ready.")

    if (
        s2_69_baselines["status"]
        != "S2_69_MULTI_SEED_FROZEN_B1_VALIDATION_BASELINES_LOCKED"
    ):
        raise RuntimeError("s2_69 B1 baselines are not locked.")

    val_dev_seeds = [
        int(value)
        for value in protocol["sampling_seeds"]["VAL_DEV"]
    ]

    val_conf_seeds = [
        int(value)
        for value in protocol["sampling_seeds"]["VAL_CONF"]
    ]

    if len(val_dev_seeds) != 3 or len(val_conf_seeds) != 3:
        raise RuntimeError("Expected exactly 3 sampling seeds per cohort.")

    run_seeds = {
        "PRIMARY": {
            key: int(value)
            for key, value in protocol[
                "future_fresh_training_runs"
            ]["PRIMARY"].items()
        },
        "REPLICATION": {
            key: int(value)
            for key, value in protocol[
                "future_fresh_training_runs"
            ]["REPLICATION"].items()
        },
    }

    protocol_snapshot = {
        "status": "S2_70_FINAL_SELECTOR_EXECUTION_PROTOCOL_SNAPSHOT",
        "source_protocol": str(S2_69_PROTOCOL),
        "source_protocol_sha256": sha256_file(S2_69_PROTOCOL),
        "VAL_DEV_sampling_seeds": val_dev_seeds,
        "VAL_CONF_sampling_seeds": val_conf_seeds,
        "run_seeds": run_seeds,
        "VAL_DEV_robust_rule": protocol["VAL_DEV_robust_eligibility_rule"],
        "VAL_DEV_selection_rule": protocol["VAL_DEV_selection_rule"],
        "VAL_CONF_confirmation_rule": protocol["VAL_CONF_confirmation_rule"],
        "cross_training_seed_rule": protocol[
            "cross_training_seed_replication_rule"
        ],
        "terminal_failure_rule": protocol["terminal_failure_rule"],
        "Final_Test_accessed": False,
    }

    if PROTOCOL_SNAPSHOT.is_file():
        with PROTOCOL_SNAPSHOT.open("r", encoding="utf-8") as f:
            existing = json.load(f)

        if existing != protocol_snapshot:
            raise RuntimeError("Conflicting s2_70 protocol snapshot.")
    else:
        atomic_json_save(protocol_snapshot, PROTOCOL_SNAPSHOT)

    b1_file_hash_before = sha256_file(FROZEN_B1_GENERATOR)

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    b1_model_hash = state_dict_sha256(frozen_b1["model_state_dict"])

    if b1_model_hash != frozen_b1["selected_model_state_sha256"]:
        raise RuntimeError("Frozen B1 model-state hash mismatch.")

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked.")

    # ========================================================
    # Helper compatibility assertions
    # ========================================================

    if base.BATCH_SIZE != 64:
        raise RuntimeError("s2_67 helper batch-size mismatch.")
    if base.LEARNING_RATE != 5e-5:
        raise RuntimeError("s2_67 helper learning-rate mismatch.")
    if base.WEIGHT_DECAY != 1e-4:
        raise RuntimeError("s2_67 helper weight-decay mismatch.")
    if base.GRADIENT_CLIP_NORM != 1.0:
        raise RuntimeError("s2_67 helper clip mismatch.")
    if base.LAMBDA_SPECTRAL != 0.0:
        raise RuntimeError("s2_67 helper lambda mismatch.")
    if base.EMA_ENABLED is not False:
        raise RuntimeError("s2_67 helper EMA mismatch.")

    # Redirect s2_67 training helper to s2_70 checkpoint directory.
    base.CHECKPOINT_DIR = CHECKPOINT_DIR

    # ========================================================
    # Step 1 — Frozen Train data
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — BUILD FROZEN REAL TRAIN DATA")
    print("=" * 80)

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)

    train_records, train_mapping = build_train_window_index(
        split_manifest=split,
        scaler=scaler,
    )

    if len(train_records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError("Unexpected Train window count.")

    if len(train_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError("Unexpected Train subject count.")

    train_base_dataset = CachedRealTrainWindowDataset(
        records=train_records,
        scaler=scaler,
        max_open_files=8,
    )

    train_dataset = SubjectIndexedDataset(
        base_dataset=train_base_dataset,
        records=train_records,
    )

    print(f"Train windows: {len(train_dataset)}")

    # ========================================================
    # Step 2 — Locked Validation references + B1 baselines
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — LOAD LOCKED MULTI-SEED VALIDATION BASELINES")
    print("=" * 80)

    (
        dev_real_z,
        dev_subject_indices,
        dev_subject_ids,
    ) = load_real_reference(VAL_DEV_REAL_NPZ)

    (
        conf_real_z,
        conf_subject_indices,
        conf_subject_ids,
    ) = load_real_reference(VAL_CONF_REAL_NPZ)

    dev_b1_bundle_by_seed = {}

    for sampling_seed in val_dev_seeds:
        z = load_b1_baseline(
            stage="VAL_DEV",
            sampling_seed=sampling_seed,
            expected_subject_indices=dev_subject_indices,
            expected_subject_ids=dev_subject_ids,
            expected_model_hash=b1_model_hash,
        )

        dev_b1_bundle_by_seed[sampling_seed] = evaluate_bundle(
            dev_real_z,
            z,
            scaler,
        )

        print(
            f"VAL-DEV B1 seed={sampling_seed}: "
            f"PSD="
            f"{dev_b1_bundle_by_seed[sampling_seed]['full_log_psd']['mean_mae_db']:.6f}"
        )

    conf_b1_bundle_by_seed = {}

    for sampling_seed in val_conf_seeds:
        z = load_b1_baseline(
            stage="VAL_CONF",
            sampling_seed=sampling_seed,
            expected_subject_indices=conf_subject_indices,
            expected_subject_ids=conf_subject_ids,
            expected_model_hash=b1_model_hash,
        )

        conf_b1_bundle_by_seed[sampling_seed] = evaluate_bundle(
            conf_real_z,
            z,
            scaler,
        )

        print(
            f"VAL-CONF B1 seed={sampling_seed}: "
            f"PSD="
            f"{conf_b1_bundle_by_seed[sampling_seed]['full_log_psd']['mean_mae_db']:.6f}"
        )

    # ========================================================
    # Step 3 — Two final fresh training streams
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — TRAIN TWO FINAL PREDECLARED FRESH RUNS")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for s2_70.")

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoints_by_run = {}
    training_rows = []

    for run_name in ("PRIMARY", "REPLICATION"):
        checkpoints, history = base.train_fresh_run(
            run_name=run_name,
            seeds=run_seeds[run_name],
            frozen_b1=frozen_b1,
            train_dataset=train_dataset,
            device=device,
        )

        # Re-label metadata for s2_70 without changing model state.
        for epoch, checkpoint in checkpoints.items():
            checkpoint["status"] = "S2_70_FINAL_FRESH_SELECTOR_CHECKPOINT"
            checkpoint["s2_70_origin"] = {
                "source_selector_protocol": str(S2_69_PROTOCOL),
                "source_selector_protocol_sha256": sha256_file(S2_69_PROTOCOL),
            }

            base.atomic_torch_save(
                checkpoint,
                checkpoint_path(run_name, epoch),
            )

        checkpoints_by_run[run_name] = checkpoints
        training_rows.extend(history)

    save_csv_rows(training_rows, TRAINING_HISTORY_CSV)

    # ========================================================
    # Step 4 — 3-seed VAL-DEV robust selection
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — 3-SEED VAL-DEV ROBUST CHECKPOINT SELECTION")
    print("=" * 80)

    schedule_cpu = build_cosine_schedule(
        timesteps=base.TIMESTEPS,
        cosine_s=base.COSINE_S,
    )

    reverse_schedule = build_reverse_schedule(
        schedule_cpu,
        device,
    )

    dev_seedwise_rows = []
    dev_aggregate_rows = []
    decisions = {}

    for run_name in ("PRIMARY", "REPLICATION"):
        print("\n" + "#" * 80)
        print(f"ROBUST VAL-DEV — {run_name}")
        print("#" * 80)

        for epoch in EPOCHS:
            for sampling_seed in val_dev_seeds:
                output_path = candidate_synthetic_path(
                    "VAL_DEV",
                    run_name,
                    epoch,
                    sampling_seed,
                )

                synth_z = base.generate_candidate(
                    run_name=run_name,
                    epoch=epoch,
                    checkpoint=checkpoints_by_run[run_name][epoch],
                    output_path=output_path,
                    frozen_b1=frozen_b1,
                    selected_subjects=dev_subject_indices,
                    selected_subject_ids=dev_subject_ids,
                    sampling_seed=sampling_seed,
                    reverse_schedule=reverse_schedule,
                    device=device,
                    cohort_name="VAL_DEV",
                )

                bundle = evaluate_bundle(
                    dev_real_z,
                    synth_z,
                    scaler,
                )

                row, eligibility = seedwise_metric_row(
                    run_name=run_name,
                    stage="VAL_DEV",
                    epoch=epoch,
                    sampling_seed=sampling_seed,
                    candidate_bundle=bundle,
                    b1_bundle=dev_b1_bundle_by_seed[sampling_seed],
                )

                dev_seedwise_rows.append(row)

                print(
                    f"{run_name} E{epoch} seed={sampling_seed}: "
                    f"eligible={row['eligible_vs_matching_B1']} "
                    f"| PSD={row['full_psd_mae_db']:.6f} "
                    f"| band={row['mean_bandpower_mae_db']:.6f} "
                    f"| alpha={row['alpha_effect_mae_db']:.6f} "
                    f"| failed={row['failed_criteria'] or 'NONE'}"
                )

        aggregates = aggregate_val_dev_rows(
            run_name=run_name,
            seedwise_rows=dev_seedwise_rows,
            sampling_seeds=val_dev_seeds,
        )

        dev_aggregate_rows.extend(aggregates)

        print(f"\n{run_name} robust VAL-DEV summary:")

        for row in aggregates:
            print(
                f"  E{row['epoch']}: "
                f"{row['seed_pass_count']}/3 "
                f"| robust={row['robust_eligible_3_of_3']} "
                f"| mean PSD={row['mean_full_psd_mae_db']:.6f} "
                f"| failed seeds={row['failed_sampling_seeds'] or 'NONE'}"
            )

        selected_epoch = select_robust_epoch(aggregates)

        decisions[run_name] = {
            "selected_epoch": selected_epoch,
            "VAL_DEV_robust_candidate_exists": selected_epoch is not None,
        }

        print(
            f"\n{run_name} robust VAL-DEV selected epoch: "
            f"{selected_epoch}"
        )

    save_csv_rows(dev_seedwise_rows, VAL_DEV_SEEDWISE_CSV)
    save_csv_rows(dev_aggregate_rows, VAL_DEV_AGGREGATE_CSV)

    # ========================================================
    # Step 5 — 3-seed VAL-CONF confirmation
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — 3-SEED VAL-CONF CONFIRMATION")
    print("=" * 80)

    conf_seedwise_rows = []

    for run_name in ("PRIMARY", "REPLICATION"):
        selected_epoch = decisions[run_name]["selected_epoch"]

        if selected_epoch is None:
            decisions[run_name]["VAL_CONF_evaluated"] = False
            decisions[run_name]["VAL_CONF_3_of_3_pass"] = False

            print(
                f"{run_name}: no robust VAL-DEV candidate -> "
                f"VAL-CONF NOT ACCESSED."
            )
            continue

        selected_checkpoint = checkpoints_by_run[run_name][selected_epoch]
        run_conf_rows = []

        print("\n" + "#" * 80)
        print(f"VAL-CONF — {run_name} selected E{selected_epoch}")
        print("#" * 80)

        for sampling_seed in val_conf_seeds:
            output_path = candidate_synthetic_path(
                "VAL_CONF",
                run_name,
                selected_epoch,
                sampling_seed,
            )

            synth_z = base.generate_candidate(
                run_name=run_name,
                epoch=selected_epoch,
                checkpoint=selected_checkpoint,
                output_path=output_path,
                frozen_b1=frozen_b1,
                selected_subjects=conf_subject_indices,
                selected_subject_ids=conf_subject_ids,
                sampling_seed=sampling_seed,
                reverse_schedule=reverse_schedule,
                device=device,
                cohort_name="VAL_CONF",
            )

            bundle = evaluate_bundle(
                conf_real_z,
                synth_z,
                scaler,
            )

            row, eligibility = seedwise_metric_row(
                run_name=run_name,
                stage="VAL_CONF",
                epoch=selected_epoch,
                sampling_seed=sampling_seed,
                candidate_bundle=bundle,
                b1_bundle=conf_b1_bundle_by_seed[sampling_seed],
            )

            conf_seedwise_rows.append(row)
            run_conf_rows.append(row)

            print(
                f"{run_name} E{selected_epoch} seed={sampling_seed}: "
                f"eligible={row['eligible_vs_matching_B1']} "
                f"| PSD={row['full_psd_mae_db']:.6f} "
                f"| band={row['mean_bandpower_mae_db']:.6f} "
                f"| alpha={row['alpha_effect_mae_db']:.6f} "
                f"| failed={row['failed_criteria'] or 'NONE'}"
            )

        pass_count = sum(
            bool(row["eligible_vs_matching_B1"])
            for row in run_conf_rows
        )

        conf_pass = pass_count == 3

        decisions[run_name]["VAL_CONF_evaluated"] = True
        decisions[run_name]["VAL_CONF_seed_pass_count"] = int(pass_count)
        decisions[run_name]["VAL_CONF_3_of_3_pass"] = bool(conf_pass)
        decisions[run_name]["selected_checkpoint_path"] = str(
            checkpoint_path(run_name, selected_epoch)
        )
        decisions[run_name]["selected_model_state_sha256"] = (
            selected_checkpoint["model_state_sha256"]
        )

        print(
            f"{run_name} VAL-CONF result: "
            f"{pass_count}/3 | 3/3 PASS={conf_pass}"
        )

    save_csv_rows(conf_seedwise_rows, VAL_CONF_SEEDWISE_CSV)

    # ========================================================
    # Step 6 — Terminal decision
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — APPLY TERMINAL CROSS-TRAINING-SEED RULE")
    print("=" * 80)

    primary_success = bool(
        decisions["PRIMARY"].get("VAL_CONF_3_of_3_pass", False)
    )

    replication_success = bool(
        decisions["REPLICATION"].get("VAL_CONF_3_of_3_pass", False)
    )

    selector_replicated = bool(
        primary_success
        and replication_success
    )

    continuation_closed = not selector_replicated

    formal_freeze_candidate = None

    if selector_replicated:
        primary_epoch = int(
            decisions["PRIMARY"]["selected_epoch"]
        )

        formal_freeze_candidate = {
            "run": "PRIMARY",
            "epoch": primary_epoch,
            "checkpoint_path": str(
                checkpoint_path("PRIMARY", primary_epoch)
            ),
            "model_state_sha256": (
                decisions["PRIMARY"]["selected_model_state_sha256"]
            ),
            "reason": (
                "PRIMARY was 3/3 robust-selected on VAL-DEV, "
                "3/3 confirmed on VAL-CONF, and the complete selector "
                "mechanism independently replicated on REPLICATION."
            ),
        }

    decision = {
        "status": "S2_70_FINAL_SELECTOR_DECISION_COMPLETE",
        "PRIMARY_success": primary_success,
        "REPLICATION_success": replication_success,
        "selector_mechanism_replicated": selector_replicated,
        "continuation_B2_terminally_closed": bool(continuation_closed),
        "decisions": decisions,
        "formal_freeze_candidate": formal_freeze_candidate,
        "candidate_is_frozen": False,
        "Final_Test_accessed": False,
    }

    atomic_json_save(decision, DECISION_JSON)

    print(f"PRIMARY success: {primary_success}")
    print(f"REPLICATION success: {replication_success}")
    print(
        f"FINAL SELECTOR MECHANISM REPLICATED: "
        f"{selector_replicated}"
    )
    print(
        f"CONTINUATION-B2 TERMINALLY CLOSED: "
        f"{continuation_closed}"
    )

    # ========================================================
    # Step 7 — Immutability + final summary
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — VERIFY IMMUTABILITY + SAVE FINAL SUMMARY")
    print("=" * 80)

    if sha256_file(FROZEN_B1_GENERATOR) != b1_file_hash_before:
        raise RuntimeError("Frozen B1 changed during s2_70.")

    summary = {
        "status": "S2_70_FINAL_SAMPLING_ROBUST_SELECTOR_COMPLETE",
        "selector_mechanism_replicated": selector_replicated,
        "continuation_B2_terminally_closed": bool(continuation_closed),
        "PRIMARY_success": primary_success,
        "REPLICATION_success": replication_success,
        "decisions": decisions,
        "formal_freeze_candidate": formal_freeze_candidate,
        "candidate_is_frozen": False,
        "artifacts": {
            "protocol_snapshot": str(PROTOCOL_SNAPSHOT),
            "training_history_csv": str(TRAINING_HISTORY_CSV),
            "VAL_DEV_seedwise_csv": str(VAL_DEV_SEEDWISE_CSV),
            "VAL_DEV_aggregate_csv": str(VAL_DEV_AGGREGATE_CSV),
            "VAL_CONF_seedwise_csv": (
                str(VAL_CONF_SEEDWISE_CSV)
                if conf_seedwise_rows
                else None
            ),
            "decision_json": str(DECISION_JSON),
        },
        "governance": {
            "Train_EEG_accessed": True,
            "Validation_EEG_accessed": True,
            "Final_Test_accessed": False,
            "thresholds_changed": False,
            "training_seed_replacement": False,
            "additional_rescue_after_failure_allowed": False,
            "formal_model_freeze_decision_made": False,
            "Frozen_B1_unchanged": True,
        },
        "next": (
            "If selector_mechanism_replicated=True: s2_71 formal B2 freeze "
            "of the predeclared PRIMARY selected checkpoint. "
            "If False: continuation-B2 is terminally closed and Frozen B1 "
            "is the robust generator for downstream utility."
        ),
    }

    atomic_json_save(summary, SUMMARY_JSON)

    train_base_dataset.close()

    print("Frozen B1 unchanged: PASS")
    print("Final Test accessed: NO")
    print("Formal B2 freeze performed: NO")
    print(f"summary JSON: {SUMMARY_JSON}")

    print("\n" + "=" * 80)
    print("s2_70 FINAL SAMPLING-ROBUST SELECTOR COMPLETE")
    print("=" * 80)

    if selector_replicated:
        print(
            f"PRIMARY selected epoch: "
            f"{formal_freeze_candidate['epoch']}"
        )
        print("NEXT: s2_71 FORMAL B2 FREEZE.")
    else:
        print("\nFINAL OUTCOME:")
        print(
            "CONTINUATION-B2 CLOSED. "
            "NO ADDITIONAL RESCUE IS ALLOWED."
        )
        print(
            "Frozen B1 remains the robust generator "
            "for downstream utility."
        )


if __name__ == "__main__":
    main()
