from __future__ import annotations

import csv, hashlib, json, os, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from s2_23_check_real_train_dataset import FINAL_SPLIT_MANIFEST, NORMALIZATION_FILE
from s2_24_full_train_loader_sanity import CachedRealTrainWindowDataset
from s2_27_logbandpower_logreg_validation import build_validation_window_index
from s2_33_b0_one_batch_diffusion_sanity import build_cosine_schedule
from s2_42_b0_v_final_sampling_real_train_fidelity_audit import build_reverse_schedule
from s2_47_b1_architecture_warmstart_sanity import state_dict_sha256
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import (
    compute_candidate_metrics,
    sample_b1_candidate_state,
)
from s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep import (
    build_model_from_state,
    full_log_psd_mae,
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
B1_REAL_REFERENCE_NPZ = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b1"
    / "raw_candidate_subject_aware_sweep_v2" / "generated"
    / "b1_subject_aware_real_train_reference_v2.npz"
)
S2_65_SUMMARY = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "early_epoch_training_seed_checkpoint_landscape_v1" / "metrics"
    / "s2_65_checkpoint_landscape_summary_v1.json"
)

RUN_ROOT = (
    REPO_ROOT / "outputs" / "protocols" / "scenario2" / "b2"
    / "validation_checkpoint_selector_v1"
)
METRIC_DIR = RUN_ROOT / "metrics"
REFERENCE_DIR = RUN_ROOT / "references"
GENERATED_DIR = RUN_ROOT / "generated"
MANIFEST_DIR = RUN_ROOT / "manifests"

PROTOCOL_FILE = METRIC_DIR / "s2_66_validation_checkpoint_selector_protocol_v1.json"
COHORT_MANIFEST = MANIFEST_DIR / "s2_66_validation_selector_subject_cohorts_v1.csv"
DEV_REFERENCE_MANIFEST = MANIFEST_DIR / "s2_66_val_dev_real_reference_manifest_v1.csv"
CONF_REFERENCE_MANIFEST = MANIFEST_DIR / "s2_66_val_conf_real_reference_manifest_v1.csv"
DEV_REAL_REFERENCE_NPZ = REFERENCE_DIR / "s2_66_val_dev_real_reference_v1.npz"
CONF_REAL_REFERENCE_NPZ = REFERENCE_DIR / "s2_66_val_conf_real_reference_v1.npz"
DEV_B1_SYNTHETIC_NPZ = GENERATED_DIR / "s2_66_val_dev_b1_synthetic_v1.npz"
CONF_B1_SYNTHETIC_NPZ = GENERATED_DIR / "s2_66_val_conf_b1_synthetic_v1.npz"
BASELINE_JSON = METRIC_DIR / "s2_66_validation_b1_baselines_v1.json"
SUMMARY_JSON = METRIC_DIR / "s2_66_validation_checkpoint_selector_summary_v1.json"

EXPECTED_SUBJECTS = 198
EXPECTED_VALIDATION_WINDOWS = 9635
DEV_SUBJECT_COUNT = 64
CONF_SUBJECT_COUNT = 64
RESERVE_SUBJECT_COUNT = 6

REAL_WINDOWS_PER_SUBJECT_STATE = 5
SYNTHETIC_WINDOWS_PER_SUBJECT_STATE = 4
N_CHANNELS = 19
N_TIMES = 512

TIMESTEPS = 1000
COSINE_S = 0.008

COHORT_SPLIT_SEED = 20261501
DEV_SAMPLING_SEED = 20261511
CONF_SAMPLING_SEED = 20261512

FUTURE_PRIMARY_TRAINING_SEEDS = {
    "master_seed": 20261601,
    "loader_seed": 20261602,
    "noise_seed": 20261603,
}
FUTURE_REPLICATION_TRAINING_SEEDS = {
    "master_seed": 20261701,
    "loader_seed": 20261702,
    "noise_seed": 20261703,
}

STATE_NAMES = ["EO", "EC"]
STATE_LABELS = {"EO": 0, "EC": 1}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def atomic_npz_save(path: Path, **arrays) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def save_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Cannot save empty CSV: {path}")
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def deterministic_subject_order(subject_ids: list[str]) -> list[str]:
    scored = []
    for subject_id in subject_ids:
        score = hashlib.sha256(
            f"{COHORT_SPLIT_SEED}:{subject_id}".encode("utf-8")
        ).hexdigest()
        scored.append((score, subject_id))
    scored.sort()
    return [subject_id for _, subject_id in scored]


def build_reference_selection(
    *, records, subject_mapping, selected_subject_ids, cohort_name
):
    grouped = defaultdict(list)
    for record in records:
        grouped[(int(record.subject_index), int(record.label))].append(record)

    selected_records = []
    manifest_rows = []

    for local_subject_index, subject_id in enumerate(selected_subject_ids):
        subject_index = int(subject_mapping[subject_id])

        for state_name in STATE_NAMES:
            label = STATE_LABELS[state_name]
            candidates = sorted(
                grouped[(subject_index, label)],
                key=lambda r: (
                    int(r.candidate_window_index),
                    int(r.start_sample),
                ),
            )

            if len(candidates) < REAL_WINDOWS_PER_SUBJECT_STATE:
                raise RuntimeError(
                    f"{cohort_name}: {subject_id}/{state_name} only has "
                    f"{len(candidates)} usable Validation windows."
                )

            positions = np.rint(
                np.linspace(
                    0,
                    len(candidates) - 1,
                    REAL_WINDOWS_PER_SUBJECT_STATE,
                )
            ).astype(np.int64)

            if len(np.unique(positions)) != REAL_WINDOWS_PER_SUBJECT_STATE:
                raise RuntimeError("Duplicate reference positions.")

            for reference_rep, position in enumerate(positions.tolist()):
                record = candidates[int(position)]
                selected_records.append(record)
                manifest_rows.append(
                    {
                        "cohort": cohort_name,
                        "local_subject_index": local_subject_index,
                        "subject_index": subject_index,
                        "subject_id": subject_id,
                        "eye_state": state_name,
                        "label": label,
                        "reference_rep": reference_rep,
                        "block_uid": record.block_uid,
                        "block_index": int(record.block_index),
                        "candidate_window_index": int(record.candidate_window_index),
                        "start_sample": int(record.start_sample),
                    }
                )

    expected = len(selected_subject_ids) * 2 * REAL_WINDOWS_PER_SUBJECT_STATE
    if len(selected_records) != expected:
        raise RuntimeError("Validation reference count mismatch.")

    return selected_records, manifest_rows


def load_real_reference(*, selected_records, scaler, subject_count):
    dataset = CachedRealTrainWindowDataset(
        records=selected_records,
        scaler=scaler,
        max_open_files=8,
    )
    try:
        values = []
        for index in range(len(dataset)):
            item = dataset[index]
            values.append(
                item["x"].detach().cpu().numpy().astype(np.float32, copy=False)
            )
        array = np.stack(values, axis=0)
    finally:
        dataset.close()

    array = array.reshape(
        (
            subject_count,
            2,
            REAL_WINDOWS_PER_SUBJECT_STATE,
            N_CHANNELS,
            N_TIMES,
        )
    )

    if not np.isfinite(array).all():
        raise RuntimeError("Validation reference contains NaN/Inf.")

    return array


def generate_b1_baseline(
    *,
    cohort_name,
    model,
    subject_indices,
    subject_ids,
    output_path,
    sampling_seed,
    reverse_schedule,
    device,
):
    model_hash = state_dict_sha256(model.state_dict())

    if output_path.is_file():
        package = np.load(output_path, allow_pickle=False)

        if (
            package["selected_subject_indices"].astype(np.int64).tolist()
            != subject_indices
        ):
            raise RuntimeError(f"{cohort_name}: cached cohort mismatch.")

        if package["selected_subject_ids"].astype(str).tolist() != subject_ids:
            raise RuntimeError(f"{cohort_name}: cached ID mismatch.")

        if int(package["sampling_seed"][0]) != sampling_seed:
            raise RuntimeError(f"{cohort_name}: cached seed mismatch.")

        if str(package["model_state_sha256"][0]) != model_hash:
            raise RuntimeError(f"{cohort_name}: cached model hash mismatch.")

        synth_z = package["synth_z"].astype(np.float32, copy=False)
        print(f"{cohort_name}: validated B1 cache -> REUSE")
        return synth_z

    print("\n" + "-" * 80)
    print(f"GENERATE FROZEN B1 BASELINE — {cohort_name}")
    print("-" * 80)

    start = time.perf_counter()

    eo = sample_b1_candidate_state(
        model=model,
        selected_subjects=subject_indices,
        eye_state_value=0,
        reverse_schedule=reverse_schedule,
        device=device,
        base_seed=sampling_seed,
    )
    ec = sample_b1_candidate_state(
        model=model,
        selected_subjects=subject_indices,
        eye_state_value=1,
        reverse_schedule=reverse_schedule,
        device=device,
        base_seed=sampling_seed,
    )

    synth_z = np.stack([eo, ec], axis=1).astype(np.float32, copy=False)

    expected_shape = (
        len(subject_indices),
        2,
        SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
        N_CHANNELS,
        N_TIMES,
    )
    if synth_z.shape != expected_shape:
        raise RuntimeError(
            f"{cohort_name}: unexpected B1 synthetic shape {synth_z.shape}"
        )

    atomic_npz_save(
        output_path,
        selected_subject_indices=np.asarray(subject_indices, dtype=np.int64),
        selected_subject_ids=np.asarray(subject_ids, dtype="U32"),
        synth_z=synth_z,
        sampling_seed=np.asarray([sampling_seed], dtype=np.int64),
        model_state_sha256=np.asarray([model_hash], dtype="U64"),
        generation_seconds=np.asarray(
            [time.perf_counter() - start],
            dtype=np.float64,
        ),
    )

    print(
        f"{cohort_name} B1 generation: "
        f"{(time.perf_counter()-start)/60:.2f} min"
    )
    return synth_z


def evaluate_baseline(*, cohort_name, real_z, synth_z, scaler):
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
        "cohort": cohort_name,
        "full_log_psd": full_psd,
        "generic": generic,
    }


def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — s2_66 FREEZE VALIDATION CHECKPOINT SELECTOR")
    print("=" * 80)

    print("\nScientific role:")
    print("  Fixed-epoch selection failed in s2_65.")
    print("  Build and freeze a held-out Validation-based checkpoint selector.")
    print("\nNo new model training.")
    print("No candidate checkpoint selection.")
    print("No Final Test access.")

    for directory in (METRIC_DIR, REFERENCE_DIR, GENERATED_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for path in (
        FROZEN_B1_GENERATOR,
        B1_SELECTION_MANIFEST,
        B1_REAL_REFERENCE_NPZ,
        S2_65_SUMMARY,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with B1_SELECTION_MANIFEST.open("r", encoding="utf-8") as f:
        b1_selection = json.load(f)
    with S2_65_SUMMARY.open("r", encoding="utf-8") as f:
        s2_65_summary = json.load(f)

    if b1_selection["status"] != "B1_SELECTION_FROZEN":
        raise RuntimeError("B1 is not frozen.")
    if (
        s2_65_summary["status"]
        != "S2_65_EARLY_EPOCH_CHECKPOINT_LANDSCAPE_COMPLETE"
    ):
        raise RuntimeError("s2_65 is not complete.")
    if s2_65_summary["development_fixed_epoch_candidate"] is not None:
        raise RuntimeError("s2_66 expects no fixed-epoch candidate.")

    b1_hash_before = sha256_file(FROZEN_B1_GENERATOR)

    frozen_b1 = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    if (
        state_dict_sha256(frozen_b1["model_state_dict"])
        != frozen_b1["selected_model_state_sha256"]
    ):
        raise RuntimeError("Frozen B1 model-state hash mismatch.")

    protocol = {
        "status": "S2_66_VALIDATION_CHECKPOINT_SELECTOR_PROTOCOL_FROZEN",
        "scientific_role": (
            "Versioned Validation checkpoint-selection mechanism after "
            "fixed-epoch failure. Final Test remains sealed."
        ),
        "subject_partition": {
            "exclude_historical_train_audit_subjects": 64,
            "remaining_expected": 134,
            "VAL_DEV_subjects": DEV_SUBJECT_COUNT,
            "VAL_CONF_subjects": CONF_SUBJECT_COUNT,
            "reserve_subjects": RESERVE_SUBJECT_COUNT,
            "deterministic_hash_seed": COHORT_SPLIT_SEED,
        },
        "real_reference": {
            "partition": "held-out Validation blocks",
            "windows_per_subject_state": REAL_WINDOWS_PER_SUBJECT_STATE,
            "selection": (
                "five deterministic evenly-spaced usable windows "
                "from the single Validation block/state"
            ),
            "normalization": "frozen Train-only global scaler",
        },
        "B1_baseline_sampling": {
            "synthetic_windows_per_subject_state": SYNTHETIC_WINDOWS_PER_SUBJECT_STATE,
            "VAL_DEV_seed": DEV_SAMPLING_SEED,
            "VAL_CONF_seed": CONF_SAMPLING_SEED,
            "sampler": "1000-step ancestral DDPM / v / FP32",
        },
        "future_training_runs": {
            "PRIMARY": FUTURE_PRIMARY_TRAINING_SEEDS,
            "REPLICATION": FUTURE_REPLICATION_TRAINING_SEEDS,
            "recipe": {
                "init": "frozen B1 RAW10",
                "lambda_spectral": 0.0,
                "epochs": [1, 2, 3, 4, 5],
                "batch": 64,
                "lr": 5e-5,
                "wd": 1e-4,
                "clip": 1.0,
                "AMP": True,
                "EMA": False,
            },
        },
        "VAL_DEV_selection_rule": (
            "For each fresh training run independently: evaluate E1-E5 "
            "against the frozen B1 VAL-DEV baseline. Eligible checkpoints "
            "must pass ALL existing B2 guards. Choose lowest full PSD MAE; "
            "tie -> lower bandpower MAE; tie -> lower alpha-effect MAE; "
            "tie -> earlier epoch. If none eligible, the run fails."
        ),
        "VAL_CONF_confirmation_rule": (
            "Only the single VAL-DEV-selected checkpoint may be evaluated "
            "on VAL-CONF. It must pass ALL existing B2 guards relative to "
            "the frozen B1 VAL-CONF baseline. No re-selection on VAL-CONF."
        ),
        "replication_success_rule": (
            "Both predeclared PRIMARY and REPLICATION runs must produce a "
            "VAL-DEV-selected checkpoint that independently passes VAL-CONF."
        ),
        "final_candidate_if_successful": (
            "The predeclared PRIMARY run is the candidate for formal freeze; "
            "REPLICATION is confirmation only."
        ),
        "important_limit": (
            "Validation EEG was used previously for denoising metrics. "
            "These are new subject-aware free-generation selector/confirmation "
            "cohorts, not globally untouched data."
        ),
        "Validation_EEG_accessed": True,
        "Final_Test_accessed": False,
        "training_updates_executed": False,
        "thresholds_changed": False,
        "formal_model_freeze_decision_made": False,
    }

    if PROTOCOL_FILE.is_file():
        with PROTOCOL_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != protocol:
            raise RuntimeError("Conflicting s2_66 protocol already exists.")
        print("identical s2_66 protocol already exists.")
    else:
        atomic_json_save(protocol, PROTOCOL_FILE)
        print(f"protocol: {PROTOCOL_FILE}")

    print("\n" + "=" * 80)
    print("STEP 1 — BUILD VALIDATION INDEX + SUBJECT COHORTS")
    print("=" * 80)

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)
    validation_records, validation_mapping = build_validation_window_index(split)

    if len(validation_records) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            f"Unexpected Validation windows: {len(validation_records)}"
        )
    if len(validation_mapping) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Unexpected Validation subjects: {len(validation_mapping)}"
        )

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)
    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked.")

    old_package = np.load(B1_REAL_REFERENCE_NPZ, allow_pickle=False)
    old_subject_ids = set(old_package["selected_subject_ids"].astype(str).tolist())

    remaining = [
        subject_id
        for subject_id in sorted(validation_mapping.keys())
        if subject_id not in old_subject_ids
    ]

    if len(remaining) != 134:
        raise RuntimeError(
            f"Expected 134 subjects after exclusion; got {len(remaining)}"
        )

    ordered = deterministic_subject_order(remaining)
    dev_subject_ids = ordered[:64]
    conf_subject_ids = ordered[64:128]
    reserve_subject_ids = ordered[128:]

    if len(reserve_subject_ids) != 6:
        raise RuntimeError("Reserve subject count mismatch.")
    if set(dev_subject_ids) & set(conf_subject_ids):
        raise RuntimeError("VAL-DEV/VAL-CONF overlap.")

    cohort_rows = []
    for cohort_name, ids in (
        ("VAL_DEV", dev_subject_ids),
        ("VAL_CONF", conf_subject_ids),
        ("RESERVE", reserve_subject_ids),
    ):
        for local_index, subject_id in enumerate(ids):
            cohort_rows.append(
                {
                    "cohort": cohort_name,
                    "local_subject_index": local_index,
                    "subject_id": subject_id,
                    "subject_index": int(validation_mapping[subject_id]),
                }
            )
    save_csv_rows(cohort_rows, COHORT_MANIFEST)

    print(f"VAL-DEV: {len(dev_subject_ids)} subjects")
    print(f"VAL-CONF: {len(conf_subject_ids)} subjects")
    print(f"RESERVE: {len(reserve_subject_ids)} subjects")
    print("Historical Train-audit overlap: 0")

    print("\n" + "=" * 80)
    print("STEP 2 — MATERIALIZE REAL VALIDATION REFERENCES")
    print("=" * 80)

    dev_records, dev_manifest = build_reference_selection(
        records=validation_records,
        subject_mapping=validation_mapping,
        selected_subject_ids=dev_subject_ids,
        cohort_name="VAL_DEV",
    )
    conf_records, conf_manifest = build_reference_selection(
        records=validation_records,
        subject_mapping=validation_mapping,
        selected_subject_ids=conf_subject_ids,
        cohort_name="VAL_CONF",
    )

    save_csv_rows(dev_manifest, DEV_REFERENCE_MANIFEST)
    save_csv_rows(conf_manifest, CONF_REFERENCE_MANIFEST)

    dev_real_z = load_real_reference(
        selected_records=dev_records,
        scaler=scaler,
        subject_count=64,
    )
    conf_real_z = load_real_reference(
        selected_records=conf_records,
        scaler=scaler,
        subject_count=64,
    )

    dev_subject_indices = [int(validation_mapping[x]) for x in dev_subject_ids]
    conf_subject_indices = [int(validation_mapping[x]) for x in conf_subject_ids]

    atomic_npz_save(
        DEV_REAL_REFERENCE_NPZ,
        selected_subject_indices=np.asarray(dev_subject_indices, dtype=np.int64),
        selected_subject_ids=np.asarray(dev_subject_ids, dtype="U32"),
        real_z=dev_real_z,
    )
    atomic_npz_save(
        CONF_REAL_REFERENCE_NPZ,
        selected_subject_indices=np.asarray(conf_subject_indices, dtype=np.int64),
        selected_subject_ids=np.asarray(conf_subject_ids, dtype="U32"),
        real_z=conf_real_z,
    )

    print(f"VAL-DEV Real shape: {dev_real_z.shape}")
    print(f"VAL-CONF Real shape: {conf_real_z.shape}")

    print("\n" + "=" * 80)
    print("STEP 3 — GENERATE FROZEN-B1 VALIDATION BASELINES")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for B1 Validation baselines.")

    device = torch.device("cuda")

    model = build_model_from_state(
        frozen_b1=frozen_b1,
        state_dict=frozen_b1["model_state_dict"],
        device=device,
    )

    schedule_cpu = build_cosine_schedule(
        timesteps=TIMESTEPS,
        cosine_s=COSINE_S,
    )
    reverse_schedule = build_reverse_schedule(schedule_cpu, device)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    dev_b1_z = generate_b1_baseline(
        cohort_name="VAL_DEV",
        model=model,
        subject_indices=dev_subject_indices,
        subject_ids=dev_subject_ids,
        output_path=DEV_B1_SYNTHETIC_NPZ,
        sampling_seed=DEV_SAMPLING_SEED,
        reverse_schedule=reverse_schedule,
        device=device,
    )

    conf_b1_z = generate_b1_baseline(
        cohort_name="VAL_CONF",
        model=model,
        subject_indices=conf_subject_indices,
        subject_ids=conf_subject_ids,
        output_path=CONF_B1_SYNTHETIC_NPZ,
        sampling_seed=CONF_SAMPLING_SEED,
        reverse_schedule=reverse_schedule,
        device=device,
    )

    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("STEP 4 — LOCK B1 VALIDATION BASELINES")
    print("=" * 80)

    dev_baseline = evaluate_baseline(
        cohort_name="VAL_DEV",
        real_z=dev_real_z,
        synth_z=dev_b1_z,
        scaler=scaler,
    )
    conf_baseline = evaluate_baseline(
        cohort_name="VAL_CONF",
        real_z=conf_real_z,
        synth_z=conf_b1_z,
        scaler=scaler,
    )

    atomic_json_save(
        {
            "status": "S2_66_VALIDATION_B1_BASELINES_LOCKED",
            "VAL_DEV": dev_baseline,
            "VAL_CONF": conf_baseline,
            "candidate_checkpoint_evaluated": False,
        },
        BASELINE_JSON,
    )

    print(
        "VAL-DEV B1: "
        f"PSD={dev_baseline['full_log_psd']['mean_mae_db']:.6f} dB "
        f"| band={dev_baseline['generic']['mean_bandpower_mae_db']:.6f} dB "
        f"| alpha={dev_baseline['generic']['alpha_effect']['channel_mae_db']:.6f} dB"
    )
    print("VAL-CONF B1 baseline materialized and locked.")

    print("\n" + "=" * 80)
    print("STEP 5 — VERIFY IMMUTABILITY + SAVE SUMMARY")
    print("=" * 80)

    if sha256_file(FROZEN_B1_GENERATOR) != b1_hash_before:
        raise RuntimeError("Frozen B1 changed during s2_66.")

    summary = {
        "status": "S2_66_VALIDATION_CHECKPOINT_SELECTOR_READY",
        "protocol": str(PROTOCOL_FILE),
        "cohorts": {
            "VAL_DEV": 64,
            "VAL_CONF": 64,
            "RESERVE": 6,
            "historical_train_audit_overlap": 0,
        },
        "references": {
            "VAL_DEV": str(DEV_REAL_REFERENCE_NPZ),
            "VAL_CONF": str(CONF_REAL_REFERENCE_NPZ),
        },
        "B1_baselines": {
            "VAL_DEV_synthetic": str(DEV_B1_SYNTHETIC_NPZ),
            "VAL_CONF_synthetic": str(CONF_B1_SYNTHETIC_NPZ),
            "metrics": str(BASELINE_JSON),
        },
        "future_primary_training_seeds": FUTURE_PRIMARY_TRAINING_SEEDS,
        "future_replication_training_seeds": FUTURE_REPLICATION_TRAINING_SEEDS,
        "candidate_selection_executed": False,
        "governance": {
            "Validation_EEG_accessed": True,
            "Final_Test_accessed": False,
            "training_updates_executed": False,
            "thresholds_changed": False,
            "formal_model_freeze_decision_made": False,
            "frozen_B1_unchanged": True,
        },
        "next": (
            "Run two predeclared fresh training seeds. "
            "Select E1-E5 using VAL-DEV only; evaluate only the selected "
            "checkpoint on VAL-CONF."
        ),
    }

    atomic_json_save(summary, SUMMARY_JSON)

    print("Frozen B1 unchanged: PASS")
    print("Final Test accessed: NO")
    print("Candidate checkpoint selected: NO")
    print(f"summary JSON: {SUMMARY_JSON}")

    print("\n" + "=" * 80)
    print("s2_66 VALIDATION CHECKPOINT SELECTOR READY")
    print("=" * 80)
    print("\nNEXT:")
    print(
        "RUN THE TWO PREDECLARED FRESH TRAINING SEEDS WITH "
        "VAL-DEV SELECTION + SINGLE VAL-CONF CONFIRMATION."
    )


if __name__ == "__main__":
    main()
