from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from s2_47_b1_architecture_warmstart_sanity import state_dict_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen B1 parent
FROZEN_B1_ROOT = (
    REPO_ROOT / "outputs" / "frozen" / "scenario2" / "b1"
    / "b1_subject_state_raw_epoch010_v1"
)
FROZEN_B1_GENERATOR = (
    FROZEN_B1_ROOT / "checkpoints"
    / "b1_subject_state_raw_epoch010_generator_v1.pt"
)

# s2_70 source
S2_70_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "final_sampling_robust_fresh_seed_selector_v1"
)
SOURCE_CHECKPOINT = (
    S2_70_ROOT / "checkpoints"
    / "s2_70_replication_epoch_001_raw.pt"
)
S2_70_DEV_SEEDWISE = (
    S2_70_ROOT / "metrics"
    / "s2_70_val_dev_seedwise_metrics_v1.csv"
)
S2_70_DEV_AGGREGATE = (
    S2_70_ROOT / "metrics"
    / "s2_70_val_dev_checkpoint_aggregate_v1.csv"
)
S2_70_CONF_SEEDWISE = (
    S2_70_ROOT / "metrics"
    / "s2_70_val_conf_seedwise_metrics_v1.csv"
)
S2_70_DECISION = (
    S2_70_ROOT / "metrics"
    / "s2_70_final_selector_decision_v1.json"
)
S2_70_SUMMARY = (
    S2_70_ROOT / "metrics"
    / "s2_70_final_sampling_robust_selector_summary_v1.json"
)

# s2_69 B1 baselines
S2_69_BASELINES = (
    REPO_ROOT / "outputs" / "protocols" / "scenario2" / "b2"
    / "sampling_robust_checkpoint_selector_v2" / "metrics"
    / "s2_69_multiseed_b1_validation_baselines_v2.json"
)

# Formal B2 freeze location
FROZEN_B2_ROOT = (
    REPO_ROOT / "outputs" / "frozen" / "scenario2" / "b2"
    / "b2_validation_selected_replication_e1_v1"
)
CHECKPOINT_DIR = FROZEN_B2_ROOT / "checkpoints"
METRIC_DIR = FROZEN_B2_ROOT / "metrics"
HANDOFF_DIR = FROZEN_B2_ROOT / "handoff"

FROZEN_FULL = (
    CHECKPOINT_DIR
    / "b2_validation_selected_replication_e1_full_checkpoint_v1.pt"
)
FROZEN_GENERATOR = (
    CHECKPOINT_DIR
    / "b2_validation_selected_replication_e1_generator_v1.pt"
)
SELECTION_JSON = METRIC_DIR / "b2_frozen_selection_v1.json"
SUMMARY_TXT = METRIC_DIR / "b2_frozen_selection_summary_v1.txt"
MANIFEST_JSON = METRIC_DIR / "b2_freeze_manifest_v1.json"
HANDOFF_JSON = HANDOFF_DIR / "b2_to_utility_handoff_v1.json"

MODEL_NAME = "B2"
MODEL_VERSION = "b2_validation_selected_replication_e1_v1"
SOURCE_RUN = "REPLICATION"
SOURCE_EPOCH = 1

EXPECTED_DEV_SEEDS = {20261511, 20261911, 20261912}
EXPECTED_CONF_SEEDS = {20261512, 20261921, 20261922}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no", ""}:
        return False
    raise ValueError(value)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp, path)


def atomic_text_save(text: str, path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(temp, path)


def atomic_exact_copy(source: Path, destination: Path) -> None:
    source_hash = sha256_file(source)

    if destination.is_file():
        if sha256_file(destination) != source_hash:
            raise RuntimeError(
                "Existing frozen full checkpoint differs from source."
            )
        print("Frozen full checkpoint already exists -> verified.")
        return

    temp = Path(str(destination) + ".tmp")
    shutil.copy2(source, temp)

    if sha256_file(temp) != source_hash:
        temp.unlink(missing_ok=True)
        raise RuntimeError("Byte-exact checkpoint copy failed.")

    os.replace(temp, destination)


def audit_s2_70() -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(S2_70_SUMMARY)
    decision = read_json(S2_70_DECISION)

    if summary["status"] != "S2_70_FINAL_SAMPLING_ROBUST_SELECTOR_COMPLETE":
        raise RuntimeError("Unexpected s2_70 summary status.")

    if decision["status"] != "S2_70_FINAL_SELECTOR_DECISION_COMPLETE":
        raise RuntimeError("Unexpected s2_70 decision status.")

    # Preserve original result exactly.
    if summary["selector_mechanism_replicated"] is not False:
        raise RuntimeError("Expected original selector replication failure.")

    if summary["continuation_B2_terminally_closed"] is not True:
        raise RuntimeError("Expected original continuation closure flag.")

    if summary["PRIMARY_success"] is not False:
        raise RuntimeError("Unexpected PRIMARY result.")

    if summary["REPLICATION_success"] is not True:
        raise RuntimeError("REPLICATION did not succeed.")

    if summary["decisions"] != decision["decisions"]:
        raise RuntimeError("s2_70 summary/decision mismatch.")

    replication = summary["decisions"][SOURCE_RUN]

    if int(replication["selected_epoch"]) != SOURCE_EPOCH:
        raise RuntimeError("REPLICATION selected epoch is not E1.")

    if replication["VAL_DEV_robust_candidate_exists"] is not True:
        raise RuntimeError("REPLICATION E1 was not robust-selected.")

    if replication["VAL_CONF_evaluated"] is not True:
        raise RuntimeError("REPLICATION E1 was not VAL-CONF evaluated.")

    if int(replication["VAL_CONF_seed_pass_count"]) != 3:
        raise RuntimeError("REPLICATION E1 did not pass 3/3 VAL-CONF seeds.")

    if replication["VAL_CONF_3_of_3_pass"] is not True:
        raise RuntimeError("REPLICATION E1 VAL-CONF was not 3/3 PASS.")

    if Path(replication["selected_checkpoint_path"]).resolve() != SOURCE_CHECKPOINT.resolve():
        raise RuntimeError("Selected checkpoint path mismatch.")

    print("Original s2_70 cross-training-seed rule: FAIL (preserved)")
    print("REPLICATION E1 model-level evidence: PASS")

    return summary, replication


def audit_metrics() -> dict[str, Any]:
    dev_rows = [
        r for r in read_csv(S2_70_DEV_SEEDWISE)
        if (
            r["run"] == SOURCE_RUN
            and int(r["epoch"]) == SOURCE_EPOCH
            and r["stage"] == "VAL_DEV"
        )
    ]

    conf_rows = [
        r for r in read_csv(S2_70_CONF_SEEDWISE)
        if (
            r["run"] == SOURCE_RUN
            and int(r["epoch"]) == SOURCE_EPOCH
            and r["stage"] == "VAL_CONF"
        )
    ]

    agg_rows = [
        r for r in read_csv(S2_70_DEV_AGGREGATE)
        if (
            r["run"] == SOURCE_RUN
            and int(r["epoch"]) == SOURCE_EPOCH
        )
    ]

    if len(dev_rows) != 3 or len(conf_rows) != 3 or len(agg_rows) != 1:
        raise RuntimeError("Unexpected s2_70 metric row counts.")

    if {int(r["sampling_seed"]) for r in dev_rows} != EXPECTED_DEV_SEEDS:
        raise RuntimeError("Unexpected VAL-DEV seeds.")

    if {int(r["sampling_seed"]) for r in conf_rows} != EXPECTED_CONF_SEEDS:
        raise RuntimeError("Unexpected VAL-CONF seeds.")

    if not all(parse_bool(r["eligible_vs_matching_B1"]) for r in dev_rows):
        raise RuntimeError("A REPLICATION E1 VAL-DEV seed failed.")

    if not all(parse_bool(r["eligible_vs_matching_B1"]) for r in conf_rows):
        raise RuntimeError("A REPLICATION E1 VAL-CONF seed failed.")

    agg = agg_rows[0]

    if not parse_bool(agg["robust_eligible_3_of_3"]):
        raise RuntimeError("REPLICATION E1 aggregate is not robust 3/3.")

    if int(agg["seed_pass_count"]) != 3:
        raise RuntimeError("REPLICATION E1 aggregate pass count is not 3.")

    def mean_std(rows, key):
        values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        return float(np.mean(values)), float(np.std(values, ddof=0))

    dev_psd_mean, dev_psd_std = mean_std(dev_rows, "full_psd_mae_db")
    conf_psd_mean, conf_psd_std = mean_std(conf_rows, "full_psd_mae_db")

    evidence = {
        "VAL_DEV": {
            "seed_pass_count": 3,
            "seed_count": 3,
            "sampling_seeds": sorted(EXPECTED_DEV_SEEDS),
            "full_psd_mae_db_mean": dev_psd_mean,
            "full_psd_mae_db_std_ddof0": dev_psd_std,
            "mean_bandpower_mae_db_mean": float(
                np.mean([float(r["mean_bandpower_mae_db"]) for r in dev_rows])
            ),
            "alpha_effect_mae_db_mean": float(
                np.mean([float(r["alpha_effect_mae_db"]) for r in dev_rows])
            ),
        },
        "VAL_CONF": {
            "seed_pass_count": 3,
            "seed_count": 3,
            "sampling_seeds": sorted(EXPECTED_CONF_SEEDS),
            "full_psd_mae_db_mean": conf_psd_mean,
            "full_psd_mae_db_std_ddof0": conf_psd_std,
            "mean_bandpower_mae_db_mean": float(
                np.mean([float(r["mean_bandpower_mae_db"]) for r in conf_rows])
            ),
            "alpha_effect_mae_db_mean": float(
                np.mean([float(r["alpha_effect_mae_db"]) for r in conf_rows])
            ),
        },
    }

    print(
        f"REPLICATION E1 VAL-DEV: 3/3 PASS | "
        f"PSD={evidence['VAL_DEV']['full_psd_mae_db_mean']:.6f} dB"
    )
    print(
        f"REPLICATION E1 VAL-CONF: 3/3 PASS | "
        f"PSD={evidence['VAL_CONF']['full_psd_mae_db_mean']:.6f} dB"
    )

    return evidence


def audit_source_checkpoint(replication_decision: dict[str, Any]) -> dict[str, Any]:
    source_hash = sha256_file(SOURCE_CHECKPOINT)

    checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["status"] != "S2_70_FINAL_FRESH_SELECTOR_CHECKPOINT":
        raise RuntimeError("Unexpected source checkpoint status.")

    if checkpoint["run"] != SOURCE_RUN or int(checkpoint["epoch"]) != SOURCE_EPOCH:
        raise RuntimeError("Source checkpoint identity mismatch.")

    computed_state_hash = state_dict_sha256(checkpoint["model_state_dict"])

    if computed_state_hash != checkpoint["model_state_sha256"]:
        raise RuntimeError("Source checkpoint model-state hash mismatch.")

    if computed_state_hash != replication_decision["selected_model_state_sha256"]:
        raise RuntimeError("Decision/source model hash mismatch.")

    return {
        "checkpoint": checkpoint,
        "file_sha256": source_hash,
        "model_state_sha256": computed_state_hash,
    }


def load_b1_baseline_summary() -> dict[str, float]:
    payload = read_json(S2_69_BASELINES)

    if payload["status"] != "S2_69_MULTI_SEED_FROZEN_B1_VALIDATION_BASELINES_LOCKED":
        raise RuntimeError("Unexpected s2_69 baseline status.")

    agg = payload["aggregate"]

    return {
        "VAL_DEV_full_psd_mean_db": float(agg["VAL_DEV_full_psd_mean_db"]),
        "VAL_CONF_full_psd_mean_db": float(agg["VAL_CONF_full_psd_mean_db"]),
    }


def build_or_verify_generator(
    source_audit: dict[str, Any],
    b1_file_hash: str,
    b1_model_hash: str,
    frozen_full_hash: str,
) -> str:
    source_checkpoint = source_audit["checkpoint"]

    payload = {
        "status": "B2_VALIDATION_SELECTED_GENERATOR_FROZEN",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "selection_role": "post-s2_70 model-level validation selection",
        "source": {
            "stage": "s2_70",
            "run": SOURCE_RUN,
            "epoch": SOURCE_EPOCH,
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_checkpoint_file_sha256": source_audit["file_sha256"],
            "frozen_full_checkpoint_file_sha256": frozen_full_hash,
        },
        "lineage": {
            "parent_model": "Frozen B1 RAW epoch10",
            "parent_generator_path": str(FROZEN_B1_GENERATOR),
            "parent_generator_file_sha256": b1_file_hash,
            "parent_model_state_sha256": b1_model_hash,
            "architecture_relation": (
                "same B1 subject+state conditional architecture; "
                "short lambda=0 continuation"
            ),
        },
        "training_recipe": source_checkpoint["training_recipe"],
        "training_seeds": source_checkpoint["seeds"],
        "train_v_mse_at_selected_epoch": source_checkpoint["train_v_mse"],
        "model_state_dict": source_checkpoint["model_state_dict"],
        "model_state_sha256": source_audit["model_state_sha256"],
        "scientific_scope": {
            "official_B2_checkpoint": True,
            "training_recipe_reproducibility_claim": False,
            "Final_Test_accessed_before_freeze": False,
        },
    }

    if FROZEN_GENERATOR.is_file():
        existing = torch.load(
            FROZEN_GENERATOR,
            map_location="cpu",
            weights_only=False,
        )

        if existing["model_state_sha256"] != payload["model_state_sha256"]:
            raise RuntimeError("Existing B2 generator model hash mismatch.")

        if state_dict_sha256(existing["model_state_dict"]) != payload["model_state_sha256"]:
            raise RuntimeError("Existing B2 generator state mismatch.")

        print("Frozen lean generator already exists -> verified.")
    else:
        temp = Path(str(FROZEN_GENERATOR) + ".tmp")
        torch.save(payload, temp)

        check = torch.load(
            temp,
            map_location="cpu",
            weights_only=False,
        )

        if state_dict_sha256(check["model_state_dict"]) != payload["model_state_sha256"]:
            temp.unlink(missing_ok=True)
            raise RuntimeError("Lean generator round-trip verification failed.")

        os.replace(temp, FROZEN_GENERATOR)

    return sha256_file(FROZEN_GENERATOR)


def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — s2_71 FORMAL B2 FREEZE: REPLICATION E1")
    print("=" * 80)

    print("\nScientific role:")
    print("  Freeze the best successful s2_70 checkpoint as official B2.")
    print("  No training, no sampling, no threshold changes.")
    print("  Original s2_70 cross-training-seed failure remains preserved.")
    print("  Final Test remains sealed.")

    for directory in (CHECKPOINT_DIR, METRIC_DIR, HANDOFF_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    required = [
        FROZEN_B1_GENERATOR,
        SOURCE_CHECKPOINT,
        S2_70_DEV_SEEDWISE,
        S2_70_DEV_AGGREGATE,
        S2_70_CONF_SEEDWISE,
        S2_70_DECISION,
        S2_70_SUMMARY,
        S2_69_BASELINES,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # B1 parent audit
    b1_file_hash_before = sha256_file(FROZEN_B1_GENERATOR)

    b1_package = torch.load(
        FROZEN_B1_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    b1_model_hash = state_dict_sha256(b1_package["model_state_dict"])

    if b1_model_hash != b1_package["selected_model_state_sha256"]:
        raise RuntimeError("Frozen B1 model-state hash mismatch.")

    print("\nSTEP 1 — AUDIT s2_70 DECISION")
    _, replication_decision = audit_s2_70()

    print("\nSTEP 2 — AUDIT MULTI-SEED VALIDATION EVIDENCE")
    evidence = audit_metrics()

    print("\nSTEP 3 — AUDIT EXACT SOURCE CHECKPOINT")
    source_audit = audit_source_checkpoint(replication_decision)

    print(f"Source file SHA256: {source_audit['file_sha256']}")
    print(f"Model-state SHA256: {source_audit['model_state_sha256']}")

    b1_baseline = load_b1_baseline_summary()

    # New model-level decision is intentionally explicit and post-s2_70.
    print("\nSTEP 4 — REGISTER POST-s2_70 MODEL-LEVEL B2 SELECTION")
    print("  Official B2 = s2_70 REPLICATION E1")
    print("  Original selector-replication FAIL remains unchanged.")

    print("\nSTEP 5 — FREEZE BYTE-EXACT FULL CHECKPOINT")

    atomic_exact_copy(
        SOURCE_CHECKPOINT,
        FROZEN_FULL,
    )

    frozen_full_hash = sha256_file(FROZEN_FULL)

    if frozen_full_hash != source_audit["file_sha256"]:
        raise RuntimeError("Frozen full checkpoint is not byte-identical.")

    print(f"Frozen full checkpoint SHA256: {frozen_full_hash}")

    print("\nSTEP 6 — BUILD LEAN B2 GENERATOR")

    generator_hash = build_or_verify_generator(
        source_audit=source_audit,
        b1_file_hash=b1_file_hash_before,
        b1_model_hash=b1_model_hash,
        frozen_full_hash=frozen_full_hash,
    )

    print(f"Frozen lean generator SHA256: {generator_hash}")

    conf_b2_psd = evidence["VAL_CONF"]["full_psd_mae_db_mean"]
    conf_b1_psd = b1_baseline["VAL_CONF_full_psd_mean_db"]

    relative_reduction = float(
        (conf_b1_psd - conf_b2_psd) / conf_b1_psd
    )

    selection_record = {
        "status": "B2_SELECTION_FROZEN",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "selected_source": {
            "stage": "s2_70",
            "run": SOURCE_RUN,
            "epoch": SOURCE_EPOCH,
            "checkpoint": str(SOURCE_CHECKPOINT),
            "source_file_sha256": source_audit["file_sha256"],
            "model_state_sha256": source_audit["model_state_sha256"],
        },
        "formal_frozen_artifacts": {
            "full_checkpoint": str(FROZEN_FULL),
            "full_checkpoint_sha256": frozen_full_hash,
            "lean_generator": str(FROZEN_GENERATOR),
            "lean_generator_sha256": generator_hash,
        },
        "parent_B1": {
            "generator_path": str(FROZEN_B1_GENERATOR),
            "generator_file_sha256": b1_file_hash_before,
            "model_state_sha256": b1_model_hash,
        },
        "selection_basis": {
            "decision_type": "POST_S2_70_MODEL_LEVEL_VALIDATION_SELECTION",
            "decision_timing": "after observing s2_70 validation results",
            "VAL_DEV": evidence["VAL_DEV"],
            "VAL_CONF": evidence["VAL_CONF"],
            "B1_VAL_CONF_full_psd_mean_db": conf_b1_psd,
            "B2_VAL_CONF_full_psd_mean_db": conf_b2_psd,
            "relative_reduction_in_mean_VAL_CONF_full_psd_vs_B1": relative_reduction,
        },
        "original_s2_70_protocol_result": {
            "cross_training_seed_selector_replicated": False,
            "PRIMARY_success": False,
            "REPLICATION_success": True,
            "continuation_B2_terminally_closed_under_original_rule": True,
            "record_preserved_and_not_rewritten": True,
        },
        "interpretation": {
            "B2_is_official_frozen_model_checkpoint": True,
            "training_recipe_reproducibility_claim": False,
            "B2_final_test_superiority_established": False,
            "VAL_DEV_and_VAL_CONF_are_development_data_for_this_selection": True,
            "Final_Test_remains_sealed_at_freeze": True,
        },
    }

    if SELECTION_JSON.is_file():
        existing = read_json(SELECTION_JSON)
        if existing != selection_record:
            raise RuntimeError("Conflicting frozen B2 selection record.")
    else:
        atomic_json_save(selection_record, SELECTION_JSON)

    summary_text = f"""SCENARIO 2 — FORMAL B2 FREEZE SUMMARY
========================================

Official model:
  B2

Frozen version:
  {MODEL_VERSION}

Selected source:
  stage = s2_70
  run   = {SOURCE_RUN}
  epoch = E{SOURCE_EPOCH}

Selection evidence:
  VAL-DEV  = 3/3 sampling seeds PASS
  VAL-CONF = 3/3 sampling seeds PASS

Mean full PSD MAE:
  B2 VAL-DEV  = {evidence['VAL_DEV']['full_psd_mae_db_mean']:.6f} dB
  B2 VAL-CONF = {evidence['VAL_CONF']['full_psd_mae_db_mean']:.6f} dB
  B1 VAL-CONF = {conf_b1_psd:.6f} dB

Relative reduction in mean VAL-CONF full PSD MAE vs B1:
  {100.0 * relative_reduction:.2f}%

Important interpretation:
  - This exact checkpoint is now the official frozen B2 model.
  - Its selection used development validation evidence.
  - The original s2_70 cross-training-seed replication rule FAILED.
  - That failure is preserved and not rewritten.
  - Therefore this freeze supports a model-level claim for this selected
    checkpoint, NOT a claim that the training recipe succeeds reliably
    across arbitrary training seeds.
  - Final Test was not accessed before this freeze.

Next:
  Freeze the downstream utility protocol before opening Final Test.
  Compare frozen B1 and frozen B2 in utility experiments.
"""

    atomic_text_save(summary_text, SUMMARY_TXT)

    handoff = {
        "status": "B2_TO_UTILITY_HANDOFF_FROZEN",
        "official_models": {
            "B1": {
                "role": "frozen baseline/comparator generator",
                "generator_path": str(FROZEN_B1_GENERATOR),
            },
            "B2": {
                "role": "official validation-selected upgraded generator",
                "generator_path": str(FROZEN_GENERATOR),
                "full_checkpoint_path": str(FROZEN_FULL),
                "selection_record": str(SELECTION_JSON),
                "model_state_sha256": source_audit["model_state_sha256"],
            },
        },
        "B2_generation_contract": {
            "architecture": "same subject+state conditional diffusion architecture as B1",
            "conditioning": "subject index + eye state",
            "prediction_target": "v",
            "sampler_family": "ancestral DDPM / v-prediction / FP32",
            "normalization": "same locked Train-only global 19-channel scaler",
            "no_per_subject_or_window_normalization": True,
        },
        "utility_stage_rules": {
            "do_not_retrain_or_reselect_B2": True,
            "do_not_use_VAL_DEV_or_VAL_CONF_to_choose_a_new_B2": True,
            "freeze_utility_protocol_before_Final_Test": True,
            "Final_Test_currently_sealed": True,
        },
        "scientific_limit": (
            "B2 is a formally frozen checkpoint selected using development "
            "validation evidence. This checkpoint passed all 3 VAL-DEV and "
            "all 3 VAL-CONF sampling seeds. This does not establish that the "
            "short-continuation training recipe is reproducibly successful "
            "across arbitrary training seeds."
        ),
    }

    if HANDOFF_JSON.is_file():
        existing = read_json(HANDOFF_JSON)
        if existing != handoff:
            raise RuntimeError("Conflicting B2 utility handoff.")
    else:
        atomic_json_save(handoff, HANDOFF_JSON)

    manifest = {
        "status": "B2_FREEZE_MANIFEST_COMPLETE",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_state_sha256": source_audit["model_state_sha256"],
        "artifacts": {
            "source_checkpoint": {
                "path": str(SOURCE_CHECKPOINT),
                "sha256": source_audit["file_sha256"],
            },
            "frozen_full_checkpoint": {
                "path": str(FROZEN_FULL),
                "sha256": frozen_full_hash,
            },
            "frozen_generator": {
                "path": str(FROZEN_GENERATOR),
                "sha256": generator_hash,
            },
            "selection_record": {
                "path": str(SELECTION_JSON),
                "sha256": sha256_file(SELECTION_JSON),
            },
            "selection_summary": {
                "path": str(SUMMARY_TXT),
                "sha256": sha256_file(SUMMARY_TXT),
            },
            "utility_handoff": {
                "path": str(HANDOFF_JSON),
                "sha256": sha256_file(HANDOFF_JSON),
            },
            "parent_B1_generator": {
                "path": str(FROZEN_B1_GENERATOR),
                "sha256": b1_file_hash_before,
            },
        },
        "governance": {
            "new_training_executed": False,
            "new_sampling_executed": False,
            "thresholds_changed": False,
            "source_s2_70_artifacts_modified": False,
            "parent_B1_modified": False,
            "Final_Test_accessed": False,
            "formal_B2_model_frozen": True,
        },
    }

    if MANIFEST_JSON.is_file():
        existing = read_json(MANIFEST_JSON)
        if existing != manifest:
            raise RuntimeError("Conflicting B2 freeze manifest.")
    else:
        atomic_json_save(manifest, MANIFEST_JSON)

    print("\nSTEP 7 — VERIFY FINAL IMMUTABILITY")

    if sha256_file(FROZEN_B1_GENERATOR) != b1_file_hash_before:
        raise RuntimeError("Frozen B1 changed during s2_71.")

    if sha256_file(SOURCE_CHECKPOINT) != source_audit["file_sha256"]:
        raise RuntimeError("s2_70 source checkpoint changed.")

    if sha256_file(FROZEN_FULL) != source_audit["file_sha256"]:
        raise RuntimeError("Frozen full checkpoint byte identity failed.")

    generator_check = torch.load(
        FROZEN_GENERATOR,
        map_location="cpu",
        weights_only=False,
    )

    if state_dict_sha256(generator_check["model_state_dict"]) != source_audit["model_state_sha256"]:
        raise RuntimeError("Frozen B2 generator state identity failed.")

    print("Frozen B1 unchanged: PASS")
    print("s2_70 source checkpoint unchanged: PASS")
    print("Frozen B2 full checkpoint byte identity: PASS")
    print("Frozen B2 lean generator state identity: PASS")

    print("\n" + "=" * 80)
    print("s2_71 FORMAL B2 FREEZE COMPLETE")
    print("=" * 80)

    print(f"\nOFFICIAL B2 VERSION: {MODEL_VERSION}")
    print(f"SELECTED SOURCE: s2_70 {SOURCE_RUN} E{SOURCE_EPOCH}")
    print("VAL-DEV: 3/3 sampling seeds PASS")
    print("VAL-CONF: 3/3 sampling seeds PASS")
    print(f"B2 model-state SHA256: {source_audit['model_state_sha256']}")
    print(f"Frozen full checkpoint: {FROZEN_FULL}")
    print(f"Frozen lean generator: {FROZEN_GENERATOR}")

    print("\nIMPORTANT:")
    print("  Official B2 checkpoint frozen: YES")
    print("  Original s2_70 cross-training-seed replication PASS: NO")
    print("  Training-recipe reproducibility claim: NO")
    print("  Model-level validation-selected B2 claim: YES")
    print("  Final Test accessed: NO")

    print("\nNEXT:")
    print(
        "FREEZE THE DOWNSTREAM UTILITY PROTOCOL FOR B1 vs B2 "
        "BEFORE OPENING FINAL TEST."
    )


if __name__ == "__main__":
    main()
