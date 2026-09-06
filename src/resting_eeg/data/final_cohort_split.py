"""Deterministically lock Scenario 2 cohort membership and unchanged block roles.

Only frozen manifests, receipts, HDF5 metadata/QC masks, and access registries are
read.  The ``eeg_uv`` datasets are never loaded.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from resting_eeg.common import canonical_hash, sha256_file
from resting_eeg.data.paths import load_paths


ACCEPTED_STREAMING_HEAD = "77a6c3745408acc7c40260794a731123dc43c93b"
FINAL_STATUS = "FINAL_COHORT_SPLIT_LOCKED"
MINIMUM_FINAL_TEST_COUNT = 398

FINAL_LOCK_TEST_NAMES = {
    "test_duplicate_subject_detection",
    "test_missing_subject_detection",
    "test_altered_split_assignment_detection",
    "test_pilot_exposure_violation_detection",
    "test_source_hash_mismatch_detection",
    "test_unexpected_cohort_count_detection",
    "test_non_idempotent_regeneration_detection",
    "test_byte_identical_artifact_validation",
    "test_ineligible_block_reason_is_deterministic",
    "test_junit_summary_parses_guarded_counts",
    "test_tiny_junit_cannot_be_accepted",
    "test_incomplete_full_cohort_verifier_cannot_be_accepted",
    "test_unresolved_review_cannot_be_accepted",
    "test_merge_commit_topology_cannot_be_accepted",
    "test_alternate_acceptance_record_cannot_be_mislabeled",
}

SCHEMA_VERSIONS = {
    "cohort": "scenario2-final-cohort-manifest-v1",
    "block_split": "scenario2-final-block-split-manifest-v1",
    "summary": "scenario2-final-cohort-split-summary-v1",
    "lock": "scenario2-final-cohort-split-lock-v1",
}

EXPECTED_STRUCTURAL_EXCLUSIONS = {
    "sub-010015",
    "sub-010100",
    "sub-010203",
    "sub-010285",
}
EXPECTED_POST_DOWNLOAD_EXCLUSIONS = {
    "sub-010012",
    "sub-010030",
    "sub-010146",
    "sub-010157",
    "sub-010163",
    "sub-010200",
    "sub-010226",
    "sub-010234",
    "sub-010250",
    "sub-010260",
    "sub-010280",
    "sub-010291",
    "sub-010316",
}

EXPECTED_INCLUDED_INELIGIBLE_TRAIN = {
    "sub-010020_block-16",
    "sub-010024_block-14",
    "sub-010066_block-16",
    "sub-010196_block-05",
    "sub-010232_block-02",
    "sub-010263_block-12",
    "sub-010263_block-16",
    "sub-010276_block-02",
    "sub-010305_block-02",
    "sub-010315_block-14",
}

SOURCE_PATHS = {
    "metadata_inventory": "outputs/metrics/scenario2/lemon_metadata_inventory.csv",
    "structural_cohort_manifest": "outputs/metrics/scenario2/lemon_structural_cohort_manifest.csv",
    "full_cohort_subject_manifest": "outputs/metrics/scenario2/batches/full_cohort_streaming_subject_manifest.csv",
    "full_cohort_qc_summary": "outputs/metrics/scenario2/batches/full_cohort_streaming_qc_summary.json",
    "full_cohort_completion_audit": "outputs/metrics/scenario2/batches/full_cohort_streaming_completion_audit.json",
    "provisional_split_manifest": "outputs/metrics/scenario2/provisional_block_split_manifest.csv",
    "block_inventory": "outputs/metrics/scenario2/lemon_block_inventory.csv",
    "pilot_waveform_access_registry": "outputs/metrics/scenario2/pilot_waveform_access_registry.csv",
    "streaming_train_access_registry": "outputs/metrics/scenario2/streaming_train_access_registry.csv",
    "mechanical_heldout_access_registry": "outputs/metrics/scenario2/pilot_mechanical_heldout_access_registry.csv",
    "preprocessing_config": "configs/preprocessing/frozen_preprocessing_v1.yaml",
    "preprocessing_config_semantic_hash": "configs/preprocessing/frozen_preprocessing_v1.sha256",
    "mechanical_access_config": "configs/access/mechanical_heldout_v1.yaml",
    "mechanical_access_config_semantic_hash": "configs/access/mechanical_heldout_v1.sha256",
}

# Byte identities frozen at the accepted full-cohort streaming HEAD.  These are
# verification expectations; membership itself is derived from the source rows.
EXPECTED_SOURCE_SHA256 = {
    "outputs/metrics/scenario2/lemon_metadata_inventory.csv": "bf7b1f27ed9f1d7c0dcd137c4e52009a758803fec72d2fb702ab59963912457c",
    "outputs/metrics/scenario2/lemon_structural_cohort_manifest.csv": "85bf4786b2edf06b0c6653c94bc5c3b71e5fa116c0c95d0d826e71cfabac43a3",
    "outputs/metrics/scenario2/batches/full_cohort_streaming_subject_manifest.csv": "81705601b558d5429de50cb9d26084de530d99708dbec0867c1f6f6c3fd52be6",
    "outputs/metrics/scenario2/batches/full_cohort_streaming_qc_summary.json": "c7bdbc41171e85cbf5c4a63008ea5978f1145d4e6709842124b832b58dcdae05",
    "outputs/metrics/scenario2/batches/full_cohort_streaming_completion_audit.json": "76af2a13a3a592e2b1035d9f36cb2c3c7578733ca17e2707cbba91f4bc03d738",
    "outputs/metrics/scenario2/provisional_block_split_manifest.csv": "6adc10b108a7e7b587757cbda630a3c4909c97f8cc2faf24320b0a6e89af0b26",
    "outputs/metrics/scenario2/lemon_block_inventory.csv": "d5deff471d02baa55ac60539986595b82594d5137fcf1bffac6a5ec982b28742",
    "outputs/metrics/scenario2/pilot_waveform_access_registry.csv": "cc0ff1542f7cb026e491c36d1211ee119095d6a9a6334b81002dea6aec489fed",
    "outputs/metrics/scenario2/streaming_train_access_registry.csv": "60bc00030f6f5e910776b8d7ad90e6f5fbbe12b57f2eede7b5e984a1dc811da9",
    "outputs/metrics/scenario2/pilot_mechanical_heldout_access_registry.csv": "50ffc3a04886ae7211a51162e664d8c34ac4a6342a93f3de9ff54312e85b389b",
    "configs/preprocessing/frozen_preprocessing_v1.yaml": "da57489746e66fa819da16217e585cdd0ca3b4e5cbe4491603e831be98889110",
    "configs/preprocessing/frozen_preprocessing_v1.sha256": "4f58fdf9cc66e9487f76842eade2346cecdb403a9fa7edd2e646957dfdf6ac6d",
    "configs/access/mechanical_heldout_v1.yaml": "4fbf4106c7f42957325cd12ba66294fd2d1a43e435e5f7282a318e8102976053",
    "configs/access/mechanical_heldout_v1.sha256": "fee08f305ea02f2f4427926c906cbbb1aed3605a02027821554c93b747c28f0c",
}

COHORT_COLUMNS = [
    "subject_id",
    "structural_status",
    "technical_complete",
    "train_qc_pass",
    "provisional_feasible",
    "final_cohort_included",
    "exclusion_stage",
    "exclusion_reason",
    "evidence_source",
]

BLOCK_COLUMNS = [
    "subject_id",
    "eye_state",
    "block_uid",
    "block_index",
    "state_block_index",
    "split",
    "block_eligible",
    "eligibility_reason",
    "candidate_windows",
    "usable_windows",
    "usable_fraction",
    "window_rejection_reason_counts",
    "window_usable_mask_sha256",
    "window_qc_reason_sha256",
    "source_output_sha256",
    "pilot_exposed",
    "final_lock_status",
]

ARTIFACT_FILENAMES = {
    "cohort": "final_cohort_manifest.csv",
    "block_split": "final_block_split_manifest.csv",
    "summary": "final_cohort_split_summary.json",
    "lock": "final_cohort_split_lock.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def truth(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Boolean evidence is missing")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid Boolean evidence: {value!r}")


def read_json(path):
    # PowerShell's historical acceptance writer emitted a UTF-8 BOM.  The BOM
    # is serialization only; exact file bytes are still bound separately.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        require(set(row) == set(columns), "CSV row does not match deterministic schema")
        writer.writerow({column: "" if row[column] is None else row[column] for column in columns})
    return stream.getvalue().encode("utf-8")


def byte_sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def validate_file_hash(path, expected, label="source"):
    actual = sha256_file(path)
    require(actual == expected, f"Frozen {label} hash mismatch: {path}")
    return actual


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def default_acceptance_path(repo_root):
    runtime = load_paths(repo_root)["runtime_root"]
    return runtime / "docs/final_acceptance_20260907_resume/final_acceptance.json"


def validate_acceptance_path(provided, canonical):
    canonical = Path(canonical).resolve()
    selected = Path(provided).resolve() if provided is not None else canonical
    require(selected == canonical, "Accepted-record override must resolve to the canonical external record")
    return selected


def validate_source_hashes(repo_root):
    root = Path(repo_root).resolve()
    bindings = {}
    for name, relative in SOURCE_PATHS.items():
        path = root / relative
        require(path.is_file(), f"Missing frozen source: {relative}")
        actual = validate_file_hash(path, EXPECTED_SOURCE_SHA256[relative])
        bindings[name] = {
            "path": relative,
            "sha256": actual,
            "bytes": path.stat().st_size,
        }

    preprocessing = yaml.safe_load((root / SOURCE_PATHS["preprocessing_config"]).read_text(encoding="utf-8"))
    mechanical = yaml.safe_load((root / SOURCE_PATHS["mechanical_access_config"]).read_text(encoding="utf-8"))
    preprocessing_semantic = canonical_hash(preprocessing)
    mechanical_semantic = canonical_hash(mechanical)
    require(
        (root / SOURCE_PATHS["preprocessing_config_semantic_hash"]).read_text(encoding="utf-8").strip()
        == preprocessing_semantic,
        "Preprocessing semantic hash mismatch",
    )
    require(
        (root / SOURCE_PATHS["mechanical_access_config_semantic_hash"]).read_text(encoding="utf-8").strip()
        == mechanical_semantic,
        "Mechanical-access semantic hash mismatch",
    )
    require(
        mechanical["required_preprocessing_config_sha256"] == preprocessing_semantic,
        "Mechanical authorization does not bind the frozen preprocessing configuration",
    )
    bindings["preprocessing_config"]["semantic_sha256"] = preprocessing_semantic
    bindings["mechanical_access_config"]["semantic_sha256"] = mechanical_semantic
    return bindings, preprocessing, mechanical


def validate_acceptance_record(path, source_bindings):
    path = Path(path).resolve()
    require(path.is_file(), "Accepted full-cohort record is missing")
    record = read_json(path)
    require(record.get("state") == "FULL_COHORT_STREAMING_COMPLETE", "Full-cohort acceptance state is not complete")
    require(record.get("final_committed_head") == ACCEPTED_STREAMING_HEAD, "Accepted streaming HEAD mismatch")
    require(record.get("worktree_clean") is True, "Accepted streaming worktree was not clean")
    require(record.get("all_acceptance_steps_passed") is True, "Full-cohort acceptance did not pass")
    require(record.get("postcommit_verification", {}).get("verification") == "passed", "Post-commit streaming verification did not pass")
    require(record.get("cumulative", {}).get("technical_subjects") == 211, "Accepted technical subject count mismatch")
    require(record.get("cumulative", {}).get("blocks") == 3376, "Accepted block count mismatch")
    require(record.get("cumulative", {}).get("train_qc_pass") == 205, "Accepted Train-QC count mismatch")
    require(record.get("cumulative", {}).get("provisional_feasibility_pass") == 198, "Accepted feasibility count mismatch")
    require(record.get("terminal", {}).get("batch_006_exists") is False, "Accepted record reports Batch 006")
    require(record.get("normalization_fitted") is False, "Accepted record reports fitted normalization")
    require(record.get("scientific_heldout_analysis") is False, "Accepted record reports scientific held-out analysis")

    recorded = {
        Path(item["path"]).resolve(): item["sha256"]
        for item in record.get("evidence_file_hashes", [])
        if isinstance(item, dict) and item.get("path") and item.get("sha256")
    }
    for key in (
        "full_cohort_subject_manifest",
        "full_cohort_qc_summary",
        "full_cohort_completion_audit",
        "provisional_split_manifest",
        "block_inventory",
        "preprocessing_config",
        "mechanical_access_config",
        "streaming_train_access_registry",
        "mechanical_heldout_access_registry",
    ):
        source = source_bindings[key]
        resolved = (Path.cwd() / source["path"]).resolve()
        # Callers execute from the repository root; fall back to suffix lookup for
        # direct library use with another current directory.
        matches = [digest for candidate, digest in recorded.items() if candidate.as_posix().endswith(source["path"])]
        digest = recorded.get(resolved, matches[0] if len(matches) == 1 else None)
        require(digest == source["sha256"], f"External acceptance source binding mismatch: {key}")

    return record, {
        "path": "runtime/docs/final_acceptance_20260907_resume/final_acceptance.json",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "accepted_streaming_head": ACCEPTED_STREAMING_HEAD,
        "state": record["state"],
    }


def validate_subject_sets(structural, completed):
    required_structural = {"subject_id", "structurally_eligible", "primary_reason", "secondary_reason"}
    required_completed = {
        "subject_id",
        "technically_complete",
        "blocks",
        "train_qc_pass",
        "provisional_5_1_2_feasible",
        "feasibility_reason",
        "output_sha256",
        "receipt_sha256",
    }
    require(required_structural <= set(structural.columns), "Structural manifest schema mismatch")
    require(required_completed <= set(completed.columns), "Full-cohort subject manifest schema mismatch")
    require(not structural.subject_id.duplicated().any(), "Duplicate subject in structural manifest")
    require(not completed.subject_id.duplicated().any(), "Duplicate subject in full-cohort manifest")

    all_ids = set(structural.subject_id)
    eligible_ids = set(structural.loc[structural.structurally_eligible.map(truth), "subject_id"])
    completed_ids = set(completed.subject_id)
    require(completed_ids == eligible_ids, "Missing or unexpected technical subject evidence")
    require(completed_ids <= all_ids, "Technical evidence contains a non-dataset subject")
    require(completed.technically_complete.map(truth).all(), "A structural candidate is not technically complete")
    require((pd.to_numeric(completed.blocks) == 16).all(), "A technical subject does not have 16 stored blocks")
    return all_ids, eligible_ids


def derive_cohort(structural, completed):
    all_ids, eligible_ids = validate_subject_sets(structural, completed)
    structural_by_id = structural.set_index("subject_id", drop=False)
    completed_by_id = completed.set_index("subject_id", drop=False)
    rows = []
    for subject in sorted(all_ids):
        structural_row = structural_by_id.loc[subject]
        structurally_eligible = truth(structural_row["structurally_eligible"])
        if not structurally_eligible:
            reason = str(structural_row["primary_reason"])
            secondary = str(structural_row.get("secondary_reason", "")).strip()
            if secondary:
                reason += ";" + secondary
            rows.append(
                {
                    "subject_id": subject,
                    "structural_status": "excluded",
                    "technical_complete": None,
                    "train_qc_pass": None,
                    "provisional_feasible": None,
                    "final_cohort_included": False,
                    "exclusion_stage": "pre_download_structural",
                    "exclusion_reason": reason,
                    "evidence_source": SOURCE_PATHS["structural_cohort_manifest"],
                }
            )
            continue

        require(subject in eligible_ids, f"Missing completed evidence for {subject}")
        completed_row = completed_by_id.loc[subject]
        technical = truth(completed_row["technically_complete"])
        train_qc = truth(completed_row["train_qc_pass"])
        feasible = truth(completed_row["provisional_5_1_2_feasible"])
        included = technical and train_qc and feasible
        reason = "" if included else str(completed_row["feasibility_reason"]).strip()
        require(included or reason, f"Excluded subject lacks a frozen reason: {subject}")
        rows.append(
            {
                "subject_id": subject,
                "structural_status": "eligible",
                "technical_complete": technical,
                "train_qc_pass": train_qc,
                "provisional_feasible": feasible,
                "final_cohort_included": included,
                "exclusion_stage": "" if included else "post_download_qc_or_feasibility",
                "exclusion_reason": reason,
                "evidence_source": ";".join(
                    [
                        SOURCE_PATHS["structural_cohort_manifest"],
                        SOURCE_PATHS["full_cohort_subject_manifest"],
                        SOURCE_PATHS["full_cohort_qc_summary"],
                    ]
                ),
            }
        )
    return rows


def validate_expected_cohort(cohort_rows, expected_total=215, expected_included=198):
    require(len(cohort_rows) == expected_total, "Unexpected original cohort count")
    subjects = [row["subject_id"] for row in cohort_rows]
    require(len(subjects) == len(set(subjects)), "Duplicate subject in final cohort manifest")
    included = {row["subject_id"] for row in cohort_rows if row["final_cohort_included"]}
    structural = {row["subject_id"] for row in cohort_rows if row["exclusion_stage"] == "pre_download_structural"}
    post_download = {
        row["subject_id"]
        for row in cohort_rows
        if row["exclusion_stage"] == "post_download_qc_or_feasibility"
    }
    require(len(included) == expected_included, "Unexpected final included-subject count")
    require(structural == EXPECTED_STRUCTURAL_EXCLUSIONS, "Structural exclusions contradict accepted evidence")
    require(post_download == EXPECTED_POST_DOWNLOAD_EXCLUSIONS, "Post-download exclusions contradict accepted evidence")
    require(not (included & structural or included & post_download or structural & post_download), "Cohort partitions overlap")
    require(len(included | structural | post_download) == expected_total, "Dataset subjects are not accounted for exactly once")
    return included, structural, post_download


def _numeric_equal(first, second):
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1e-9)


def validate_provisional_assignments(provisional, inventory, candidate_ids):
    required_split = {
        "subject_id",
        "block_index",
        "acquisition_segment_id",
        "state",
        "state_block_index",
        "start_sec",
        "end_estimate_capped_sec",
        "split",
        "split_status",
    }
    required_inventory = required_split - {"split", "split_status"} | {
        "is_complete",
        "is_study_block",
        "is_interruption_fragment",
    }
    require(required_split <= set(provisional.columns), "Provisional split schema mismatch")
    require(required_inventory <= set(inventory.columns), "Block inventory schema mismatch")
    keys = ["subject_id", "block_index"]
    require(not provisional.duplicated(keys).any(), "Duplicate provisional assignment")
    require(not inventory.duplicated(keys).any(), "Duplicate block inventory key")
    require(set(provisional.subject_id) == set(candidate_ids), "Provisional split subject coverage mismatch")
    require(len(provisional) == len(candidate_ids) * 16, "Provisional assignment count mismatch")
    require(set(provisional.split_status) == {"PROVISIONAL_METADATA_ONLY"}, "Unexpected provisional split status")
    require(set(provisional.split) == {"Train", "Validation", "Test"}, "Unexpected provisional role")

    inventory_index = inventory.set_index(keys, drop=False)
    for row in provisional.to_dict("records"):
        key = (row["subject_id"], row["block_index"])
        require(key in inventory_index.index, f"Assigned block absent from inventory: {key}")
        source = inventory_index.loc[key]
        require(truth(source["is_complete"]), f"Assigned block is incomplete: {key}")
        require(truth(source["is_study_block"]), f"Assigned block is not a study block: {key}")
        require(not truth(source["is_interruption_fragment"]), f"Assigned block is an interruption fragment: {key}")
        require(str(source["state"]) == str(row["state"]), f"Split/inventory field mismatch for {key}: state")
        for field in ("acquisition_segment_id", "state_block_index"):
            require(_numeric_equal(source[field], row[field]), f"Split/inventory field mismatch for {key}: {field}")
        for field in ("start_sec", "end_estimate_capped_sec"):
            require(_numeric_equal(source[field], row[field]), f"Split/inventory time mismatch for {key}: {field}")

    expected = {"Train": 5, "Validation": 1, "Test": 2}
    grouped = provisional.groupby(["subject_id", "state", "split"]).size()
    for subject in sorted(candidate_ids):
        for state in ("EO", "EC"):
            actual = {split: int(grouped.get((subject, state, split), 0)) for split in expected}
            require(actual == expected, f"Provisional 5/1/2 role count mismatch: {subject} {state}")


def _registry_keys(frame):
    return {(str(row.subject_id), int(row.block_index)) for row in frame.itertuples(index=False)}


def validate_access_registries(repo_root, provisional, mechanical_config):
    root = Path(repo_root)
    pilot = pd.read_csv(root / SOURCE_PATHS["pilot_waveform_access_registry"], keep_default_na=False)
    streaming = pd.read_csv(root / SOURCE_PATHS["streaming_train_access_registry"], keep_default_na=False)
    mechanical = pd.read_csv(root / SOURCE_PATHS["mechanical_heldout_access_registry"], keep_default_na=False)
    assignment = {
        (str(row.subject_id), int(row.block_index)): str(row.split)
        for row in provisional.itertuples(index=False)
    }
    assignment_rows = {
        (str(row.subject_id), int(row.block_index)): row
        for row in provisional.itertuples(index=False)
    }
    train_keys = {key for key, split in assignment.items() if split == "Train"}
    heldout_keys = {key for key, split in assignment.items() if split in {"Validation", "Test"}}

    for name, frame in (("pilot", pilot), ("streaming", streaming)):
        require(not frame.duplicated(["subject_id", "block_index"]).any(), f"Duplicate {name} Train access row")
        require(frame.access_allowed.map(truth).all(), f"Denied row present in {name} registry")
        require(set(frame.split) == {"Train"}, f"Non-Train role present in {name} registry")
        require(set(frame.split_manifest_sha256) == {EXPECTED_SOURCE_SHA256[SOURCE_PATHS["provisional_split_manifest"]]}, f"{name} split hash mismatch")
        require(set(frame.block_inventory_sha256) == {EXPECTED_SOURCE_SHA256[SOURCE_PATHS["block_inventory"]]}, f"{name} inventory hash mismatch")
        for row in frame.itertuples(index=False):
            key = (str(row.subject_id), int(row.block_index))
            require(assignment.get(key) == "Train", f"{name} access role contradicts provisional split: {key}")
            source = assignment_rows[key]
            require(str(row.block_uid) == f"{key[0]}_block-{key[1]:02d}", f"{name} block UID mismatch: {key}")
            require(str(row.state) == str(source.state), f"{name} state mismatch: {key}")
            require(_numeric_equal(row.start_sec, source.start_sec), f"{name} start mismatch: {key}")
            require(_numeric_equal(row.end_sec, source.end_estimate_capped_sec), f"{name} end mismatch: {key}")

    pilot_keys, streaming_keys = _registry_keys(pilot), _registry_keys(streaming)
    require(len(pilot_keys) == 150, "Pilot exposure count mismatch")
    require(not pilot_keys & streaming_keys, "Pilot and streaming Train registries overlap")
    require(pilot_keys | streaming_keys == train_keys, "Train access registries do not cover the frozen Train assignments exactly")

    require(not mechanical.duplicated(["subject_id", "block_index"]).any(), "Duplicate mechanical held-out access row")
    require(mechanical.access_allowed.map(truth).all(), "Denied row present in mechanical registry")
    mechanical_keys = _registry_keys(mechanical)
    require(mechanical_keys == heldout_keys, "Mechanical registry does not cover held-out assignments exactly")
    allowed_purposes = set(mechanical_config["allowed_purposes"])
    forbidden_purposes = set(mechanical_config["forbidden_purposes"])
    require(set(mechanical.purpose) <= allowed_purposes, "Mechanical access used an unauthorized purpose")
    require(not set(mechanical.purpose) & forbidden_purposes, "Scientific held-out purpose appears in access registry")
    require(
        set(mechanical.preprocessing_config_sha256)
        == {mechanical_config["required_preprocessing_config_sha256"]},
        "Mechanical registry preprocessing hash mismatch",
    )
    require(
        set(mechanical.mechanical_access_config_sha256)
        == {(root / SOURCE_PATHS["mechanical_access_config_semantic_hash"]).read_text(encoding="utf-8").strip()},
        "Mechanical registry authorization hash mismatch",
    )
    for row in mechanical.itertuples(index=False):
        key = (str(row.subject_id), int(row.block_index))
        require(assignment.get(key) == str(row.provisional_split), f"Mechanical access role mismatch: {key}")
        source = assignment_rows[key]
        require(str(row.block_uid) == f"{key[0]}_block-{key[1]:02d}", f"Mechanical block UID mismatch: {key}")
        require(str(row.state) == str(source.state), f"Mechanical state mismatch: {key}")
        require(_numeric_equal(row.start_sec, source.start_sec), f"Mechanical start mismatch: {key}")
        require(_numeric_equal(row.end_sec, source.end_estimate_capped_sec), f"Mechanical end mismatch: {key}")
    return pilot_keys, {
        "pilot_unique_train_blocks": len(pilot_keys),
        "streaming_unique_train_blocks": len(streaming_keys),
        "all_train_assignments_covered": len(train_keys),
        "mechanical_unique_heldout_blocks": len(mechanical_keys),
        "scientific_heldout_purposes": [],
    }


def _decode_reasons(dataset):
    try:
        values = dataset.asstr()[:]
    except AttributeError:
        values = dataset[:].astype(str)
    return [str(value) for value in values.tolist()]


def _reason_counts(reasons):
    counts = Counter()
    for cell in reasons:
        if cell == "usable":
            continue
        for reason in cell.split("|"):
            if reason and reason != "usable":
                counts[reason] += 1
    return dict(sorted(counts.items()))


def _eligibility_reason(eligible, usable, candidates, minimum_windows, minimum_fraction):
    if eligible:
        return "frozen_qc_eligible"
    reasons = []
    if usable < minimum_windows:
        reasons.append(f"usable_windows_below_{minimum_windows}")
    if candidates == 0 or usable / candidates < minimum_fraction:
        reasons.append(f"usable_fraction_below_{minimum_fraction:g}")
    require(reasons, "Ineligible block lacks a frozen criterion failure")
    return ";".join(reasons)


def collect_processed_evidence(repo_root, completed, final_assignments, included_ids, pilot_keys, preprocessing):
    paths = load_paths(repo_root)
    output_root = paths["processed_root"] / "all-blocks-v1"
    completed_by_id = completed.set_index("subject_id", drop=False)
    assignments_by_subject = {
        subject: group.sort_values("block_index", kind="stable")
        for subject, group in final_assignments.groupby("subject_id", sort=False)
    }
    minimum_windows = int(preprocessing["qc"]["minimum_usable_windows_per_block"])
    minimum_fraction = float(preprocessing["qc"]["minimum_usable_fraction_per_block"])
    preprocessing_semantic = (Path(repo_root) / SOURCE_PATHS["preprocessing_config_semantic_hash"]).read_text(encoding="utf-8").strip()
    mechanical_semantic = (Path(repo_root) / SOURCE_PATHS["mechanical_access_config_semantic_hash"]).read_text(encoding="utf-8").strip()
    output_hashes, receipt_hashes, block_rows = {}, {}, []
    normalizations_checked = 0

    for subject in sorted(completed_by_id.index):
        manifest_row = completed_by_id.loc[subject]
        output = output_root / f"{subject}_all_blocks.h5"
        receipt_path = output.with_suffix(".provenance.json")
        require(output.is_file() and receipt_path.is_file(), f"Missing processed evidence: {subject}")
        output_hash = sha256_file(output)
        receipt_hash = sha256_file(receipt_path)
        require(output_hash == manifest_row["output_sha256"], f"All-block output hash mismatch: {subject}")
        require(receipt_hash == manifest_row["receipt_sha256"], f"Receipt hash mismatch: {subject}")
        receipt = read_json(receipt_path)
        require(receipt.get("subject_id") == subject, f"Receipt subject mismatch: {subject}")
        require(receipt.get("output_sha256") == output_hash, f"Receipt/output binding mismatch: {subject}")
        require(receipt.get("fit_parameters_from") == "Train_only", f"Receipt is not Train-only fitted: {subject}")
        require(receipt.get("heldout_parameters_are") == "apply_only", f"Receipt held-out provenance mismatch: {subject}")
        require(truth(receipt.get("qc_pass")) == truth(manifest_row["train_qc_pass"]), f"Receipt Train-QC mismatch: {subject}")
        require(
            truth(receipt.get("provisional_feasibility", {}).get("provisional_5_1_2_feasible"))
            == truth(manifest_row["provisional_5_1_2_feasible"]),
            f"Receipt feasibility mismatch: {subject}",
        )
        output_hashes[subject], receipt_hashes[subject] = output_hash, receipt_hash

        with h5py.File(output, "r") as handle:
            require(handle.attrs.get("scope") == "all_blocks_mechanical_provisional_v1", f"Unexpected HDF5 scope: {subject}")
            require(not truth(handle.attrs.get("normalization_fitted")), f"Normalization is fitted: {subject}")
            require(handle.attrs.get("preprocessing_config_sha256") == preprocessing_semantic, f"HDF5 preprocessing hash mismatch: {subject}")
            require(handle.attrs.get("mechanical_access_config_sha256") == mechanical_semantic, f"HDF5 access hash mismatch: {subject}")
            parameters = json.loads(handle.attrs.get("train_parameters"))
            require(parameters.get("fit_parameters_from") == "Train_only", f"HDF5 is not Train-only fitted: {subject}")
            normalizations_checked += 1
            if subject not in included_ids:
                continue

            require(subject in assignments_by_subject, f"Included subject has no final assignments: {subject}")
            for assignment in assignments_by_subject[subject].itertuples(index=False):
                uid = f"{subject}_block-{int(assignment.block_index):02d}"
                require(uid in handle["blocks"], f"Final assigned block absent from HDF5: {uid}")
                group = handle["blocks"][uid]
                require("eeg_uv" in group, f"Compact dataset is missing: {uid}")
                # Shape/dtype inspection is metadata-only.  Never index eeg_uv.
                require(group["eeg_uv"].shape[0] == 19 and group["eeg_uv"].dtype == np.float32, f"Compact metadata mismatch: {uid}")
                require(str(group.attrs.get("provisional_split")) == str(assignment.split), f"Altered split assignment: {uid}")
                require(str(group.attrs.get("state")) == str(assignment.state), f"HDF5 state mismatch: {uid}")
                require(int(group.attrs.get("block_index")) == int(assignment.block_index), f"HDF5 block index mismatch: {uid}")
                mask = np.asarray(group["window_usable_mask"][:], dtype=bool)
                reasons = _decode_reasons(group["window_qc_reason"])
                require(len(mask) == len(reasons), f"Mask/reason length mismatch: {uid}")
                require(np.array_equal(mask, np.asarray([reason == "usable" for reason in reasons])), f"Mask/reason contradiction: {uid}")
                candidates, usable = int(len(mask)), int(mask.sum())
                eligible = usable >= minimum_windows and candidates > 0 and usable / candidates >= minimum_fraction
                require(eligible == truth(group.attrs.get("block_eligible")), f"Frozen block eligibility mismatch: {uid}")
                block_rows.append(
                    {
                        "subject_id": subject,
                        "eye_state": str(assignment.state),
                        "block_uid": uid,
                        "block_index": int(assignment.block_index),
                        "state_block_index": int(assignment.state_block_index),
                        "split": str(assignment.split),
                        "block_eligible": eligible,
                        "eligibility_reason": _eligibility_reason(
                            eligible, usable, candidates, minimum_windows, minimum_fraction
                        ),
                        "candidate_windows": candidates,
                        "usable_windows": usable,
                        "usable_fraction": f"{usable / candidates:.12f}" if candidates else "0.000000000000",
                        "window_rejection_reason_counts": json.dumps(
                            _reason_counts(reasons), sort_keys=True, separators=(",", ":")
                        ),
                        "window_usable_mask_sha256": canonical_hash(mask.tolist()),
                        "window_qc_reason_sha256": canonical_hash(reasons),
                        "source_output_sha256": output_hash,
                        "pilot_exposed": (subject, int(assignment.block_index)) in pilot_keys,
                        "final_lock_status": "LOCKED_UNCHANGED_PROVISIONAL_ROLE",
                    }
                )

    require(normalizations_checked == 211, "Normalization state was not checked for every technical subject")
    block_rows.sort(key=lambda row: (row["subject_id"], row["block_index"]))
    return block_rows, {
        "all_block_output_count": len(output_hashes),
        "all_block_output_hash_bundle_sha256": canonical_hash(output_hashes),
        "receipt_count": len(receipt_hashes),
        "receipt_hash_bundle_sha256": canonical_hash(receipt_hashes),
        "normalization_flags_checked": normalizations_checked,
        "eeg_uv_arrays_loaded": False,
    }


def validate_final_roles(block_rows, provisional, included_ids):
    final_keys = {(row["subject_id"], int(row["block_index"])): row["split"] for row in block_rows}
    expected = {
        (str(row.subject_id), int(row.block_index)): str(row.split)
        for row in provisional.loc[provisional.subject_id.isin(included_ids)].itertuples(index=False)
    }
    require(final_keys == expected, "Final assignments do not exactly match provisional roles")
    require(len(block_rows) == 3168, "Unexpected final assignment count")
    counts = Counter(row["split"] for row in block_rows)
    require(counts == {"Train": 1980, "Validation": 396, "Test": 792}, "Unexpected final role counts")
    grouped = Counter((row["subject_id"], row["eye_state"], row["split"]) for row in block_rows)
    for subject in sorted(included_ids):
        for state in ("EO", "EC"):
            require(
                {split: grouped[(subject, state, split)] for split in ("Train", "Validation", "Test")}
                == {"Train": 5, "Validation": 1, "Test": 2},
                f"Final 5/1/2 role count mismatch: {subject} {state}",
            )
    require(
        not [row for row in block_rows if row["split"] in {"Validation", "Test"} and not row["block_eligible"]],
        "Included held-out assignment is ineligible",
    )
    return dict(counts)


def validate_pilot_exposure(block_rows, cohort_rows, pilot_keys):
    included_roles = {
        (row["subject_id"], int(row["block_index"])): row["split"] for row in block_rows
    }
    excluded_subjects = {
        row["subject_id"] for row in cohort_rows if not row["final_cohort_included"]
    }
    violations = []
    included_train, excluded = 0, 0
    for key in sorted(pilot_keys):
        subject, _ = key
        if subject in excluded_subjects:
            excluded += 1
        elif included_roles.get(key) == "Train":
            included_train += 1
        else:
            violations.append(key)
    require(not violations, f"Pilot exposure violation: {violations}")
    require(included_train + excluded == 150, "Pilot exposure accounting mismatch")
    return {
        "exposed_blocks": 150,
        "included_subject_train_blocks": included_train,
        "excluded_subject_blocks": excluded,
        "validation_or_test_violations": 0,
        "passed": True,
    }


def _git_output(repo_root, *args):
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def validate_repository_boundaries(repo_root, require_clean=False):
    root = Path(repo_root).resolve()
    current_head = _git_output(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ACCEPTED_STREAMING_HEAD, current_head],
        cwd=root,
        check=False,
    ).returncode == 0
    require(ancestor, "Accepted streaming HEAD is not an ancestor of current HEAD")
    if require_clean:
        require(not _git_output(root, "status", "--porcelain"), "Worktree is not clean")

    tracked = _git_output(root, "ls-files").splitlines()
    signal_suffixes = {".h5", ".hdf5", ".eeg", ".vhdr", ".vmrk", ".edf", ".bdf", ".set", ".fdt", ".npy", ".npz"}
    signal_payloads = [path for path in tracked if Path(path).suffix.lower() in signal_suffixes]
    require(not signal_payloads, f"Signal payload is tracked in Git: {signal_payloads}")

    paths = load_paths(root)
    production_batch_roots = [
        root / "outputs/metrics/scenario2/batches",
        paths["runtime_root"] / "batches",
    ]
    batch_006 = []
    for base in production_batch_roots:
        if base.exists():
            batch_006.extend(path for path in base.iterdir() if "batch_006" in path.name.lower())
    require(not batch_006, f"Batch 006 exists: {batch_006}")
    return {
        "current_head": current_head,
        "accepted_streaming_head_is_ancestor": True,
        "worktree_clean_required": require_clean,
        "worktree_clean": not bool(_git_output(root, "status", "--porcelain")),
        "tracked_signal_payloads": 0,
        "batch_006_exists": False,
    }


def _exclusion_records(cohort_rows, stage):
    return [
        {
            "subject_id": row["subject_id"],
            "reason": row["exclusion_reason"],
        }
        for row in cohort_rows
        if row["exclusion_stage"] == stage
    ]


def build_artifacts(repo_root, acceptance_path=None):
    root = Path(repo_root).resolve()
    source_bindings, preprocessing, mechanical_config = validate_source_hashes(root)
    acceptance_path = validate_acceptance_path(acceptance_path, default_acceptance_path(root))
    acceptance_record, acceptance_binding = validate_acceptance_record(acceptance_path, source_bindings)
    source_bindings["accepted_full_cohort_record"] = acceptance_binding

    structural = pd.read_csv(root / SOURCE_PATHS["structural_cohort_manifest"], keep_default_na=False)
    metadata_inventory = pd.read_csv(root / SOURCE_PATHS["metadata_inventory"], keep_default_na=False)
    completed = pd.read_csv(root / SOURCE_PATHS["full_cohort_subject_manifest"], keep_default_na=False)
    qc = read_json(root / SOURCE_PATHS["full_cohort_qc_summary"])
    completion = read_json(root / SOURCE_PATHS["full_cohort_completion_audit"])
    cohort_rows = derive_cohort(structural, completed)
    included_ids, structural_excluded, post_download_excluded = validate_expected_cohort(cohort_rows)
    require(not metadata_inventory.subject_id.duplicated().any(), "Duplicate subject in metadata inventory")
    require(set(metadata_inventory.subject_id) == set(structural.subject_id), "Metadata and structural subject universes differ")
    require(set(qc["provisional_feasibility_fail"]) == post_download_excluded, "QC summary exclusion set mismatch")
    require(set(qc["provisional_feasibility_pass"]) == included_ids, "QC summary included set mismatch")
    require(set(qc["train_qc_fail"]) == set(completed.loc[~completed.train_qc_pass.map(truth), "subject_id"]), "QC summary Train-fail set mismatch")
    require(set(qc["train_qc_pass"]) == set(completed.loc[completed.train_qc_pass.map(truth), "subject_id"]), "QC summary Train-pass set mismatch")
    require(completion.get("normalization_fitted") is False, "Completion audit reports fitted normalization")
    require(completion.get("scientific_heldout_analysis") is False, "Completion audit reports scientific held-out analysis")
    require(completion.get("no_batch_006") is True, "Completion audit does not seal Batch 006")

    provisional = pd.read_csv(root / SOURCE_PATHS["provisional_split_manifest"], keep_default_na=False)
    inventory = pd.read_csv(root / SOURCE_PATHS["block_inventory"], keep_default_na=False)
    validate_provisional_assignments(provisional, inventory, set(completed.subject_id))
    pilot_keys, registry_results = validate_access_registries(root, provisional, mechanical_config)
    final_assignments = provisional.loc[provisional.subject_id.isin(included_ids)].copy()
    block_rows, processed_evidence = collect_processed_evidence(
        root, completed, final_assignments, included_ids, pilot_keys, preprocessing
    )
    split_counts = validate_final_roles(block_rows, provisional, included_ids)
    pilot = validate_pilot_exposure(block_rows, cohort_rows, pilot_keys)

    ineligible_train = [
        {"subject_id": row["subject_id"], "block_uid": row["block_uid"], "eye_state": row["eye_state"]}
        for row in block_rows
        if row["split"] == "Train" and not row["block_eligible"]
    ]
    require(
        {row["block_uid"] for row in ineligible_train} == EXPECTED_INCLUDED_INELIGIBLE_TRAIN,
        "Included ineligible Train-block set contradicts frozen evidence",
    )
    cohort_payload = csv_bytes(cohort_rows, COHORT_COLUMNS)
    block_payload = csv_bytes(block_rows, BLOCK_COLUMNS)
    summary = {
        "schema_version": SCHEMA_VERSIONS["summary"],
        "status": FINAL_STATUS,
        "accepted_streaming_head": ACCEPTED_STREAMING_HEAD,
        "cohort_rule": "technical_complete AND train_qc_pass AND provisional_5_1_2_feasible",
        "counts": {
            "original_dataset_subjects": 215,
            "structural_candidates": 211,
            "included_subjects": 198,
            "excluded_subjects": 17,
            "structural_exclusions": 4,
            "post_download_exclusions": 13,
            "total_block_assignments": 3168,
            "Train": split_counts["Train"],
            "Validation": split_counts["Validation"],
            "Test": split_counts["Test"],
            "ineligible_assigned_Train_blocks": len(ineligible_train),
            "ineligible_assigned_Validation_blocks": 0,
            "ineligible_assigned_Test_blocks": 0,
        },
        "structural_exclusions": _exclusion_records(cohort_rows, "pre_download_structural"),
        "post_download_exclusions": _exclusion_records(cohort_rows, "post_download_qc_or_feasibility"),
        "ineligible_assigned_Train_blocks": ineligible_train,
        "pilot_exposure": pilot,
        "registry_verification": registry_results,
        "processed_evidence": processed_evidence,
        "artifact_hashes": {
            ARTIFACT_FILENAMES["cohort"]: byte_sha256(cohort_payload),
            ARTIFACT_FILENAMES["block_split"]: byte_sha256(block_payload),
        },
        "verification_results": {
            "all_215_subjects_accounted_once": True,
            "inclusion_rule_derived_from_frozen_evidence": True,
            "expected_exclusion_sets_match": True,
            "final_roles_match_provisional_exactly": True,
            "every_final_block_exists_in_inventory": True,
            "per_subject_state_5_1_2": True,
            "ineligible_train_blocks_preserved_without_replacement": True,
            "pilot_exposure_train_or_subject_excluded": True,
            "frozen_source_hashes_match": True,
            "all_output_and_receipt_hashes_match": True,
            "normalization_unfitted": True,
            "scientific_heldout_outcomes_accessed": False,
            "eeg_uv_arrays_loaded": False,
            "final_test_sealed": True,
        },
    }
    summary_payload = json_bytes(summary)
    lock = {
        "schema_version": SCHEMA_VERSIONS["lock"],
        "status": FINAL_STATUS,
        "accepted_streaming_head": ACCEPTED_STREAMING_HEAD,
        "artifact_bindings": {
            ARTIFACT_FILENAMES["cohort"]: {
                "schema_version": SCHEMA_VERSIONS["cohort"],
                "sha256": byte_sha256(cohort_payload),
                "bytes": len(cohort_payload),
                "rows": len(cohort_rows),
            },
            ARTIFACT_FILENAMES["block_split"]: {
                "schema_version": SCHEMA_VERSIONS["block_split"],
                "sha256": byte_sha256(block_payload),
                "bytes": len(block_payload),
                "rows": len(block_rows),
            },
            ARTIFACT_FILENAMES["summary"]: {
                "schema_version": SCHEMA_VERSIONS["summary"],
                "sha256": byte_sha256(summary_payload),
                "bytes": len(summary_payload),
            },
        },
        "source_bindings": dict(sorted(source_bindings.items())),
        "processed_evidence_bindings": processed_evidence,
        "scientific_boundaries": {
            "cohort_inputs": "frozen_outcome_blind_engineering_evidence_only",
            "block_roles": "unchanged_from_provisional_metadata_only_assignment",
            "normalization_unfitted": True,
            "final_test_sealed": True,
            "scientific_heldout_outcomes_accessed": False,
            "eeg_uv_arrays_loaded": False,
        },
        "deterministic_serialization": {
            "json": "UTF-8, LF, sorted keys, indent=2, allow_nan=false, terminal newline",
            "csv": "UTF-8, LF, fixed schema column order, stable subject/block order",
            "timestamps_included": False,
        },
    }
    lock_payload = json_bytes(lock)
    return {
        ARTIFACT_FILENAMES["cohort"]: cohort_payload,
        ARTIFACT_FILENAMES["block_split"]: block_payload,
        ARTIFACT_FILENAMES["summary"]: summary_payload,
        ARTIFACT_FILENAMES["lock"]: lock_payload,
    }, summary, lock


def write_artifacts(repo_root, output_root=None, acceptance_path=None):
    root = Path(repo_root).resolve()
    output = Path(output_root or root / "outputs/metrics/scenario2")
    payloads, summary, lock = build_artifacts(root, acceptance_path=acceptance_path)
    for filename, payload in payloads.items():
        atomic_write(output / filename, payload)
    validate_artifact_bytes(output, payloads)
    return summary, lock


def validate_artifact_bytes(output_root, expected_payloads):
    output = Path(output_root)
    for filename, payload in expected_payloads.items():
        path = output / filename
        require(path.is_file(), f"Missing final artifact: {filename}")
        require(path.read_bytes() == payload, f"Artifact differs from deterministic regeneration: {filename}")


def verify_artifacts(repo_root, output_root=None, acceptance_path=None, require_clean=False):
    root = Path(repo_root).resolve()
    repository = validate_repository_boundaries(root, require_clean=require_clean)
    output = Path(output_root or root / "outputs/metrics/scenario2")
    payloads, summary, lock = build_artifacts(root, acceptance_path=acceptance_path)
    repeated_payloads, repeated_summary, repeated_lock = build_artifacts(
        root, acceptance_path=acceptance_path
    )
    require(payloads == repeated_payloads, "Non-idempotent in-memory regeneration")
    require(summary == repeated_summary and lock == repeated_lock, "Non-idempotent canonical content")
    validate_artifact_bytes(output, payloads)
    return {
        "status": FINAL_STATUS,
        "verification": "passed",
        "repository": repository,
        "counts": summary["counts"],
        "pilot_exposure": summary["pilot_exposure"],
        "artifact_sha256": {name: byte_sha256(payload) for name, payload in payloads.items()},
        "normalization_unfitted": True,
        "final_test_sealed": True,
        "scientific_heldout_outcomes_accessed": False,
        "eeg_uv_arrays_loaded": False,
        "deterministic_regeneration": "byte_identical",
    }


def junit_summary(path):
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))
    def total(field):
        if root.get(field) is not None:
            return int(root.get(field, 0))
        return sum(int(suite.get(field, 0)) for suite in suites)
    cases = list(root.findall(".//testcase"))
    final_lock_names = sorted(
        case.get("name")
        for case in cases
        if "test_final_cohort_split" in case.get("classname", "")
    )
    return {
        "tests": total("tests"),
        "failures": total("failures"),
        "errors": total("errors"),
        "skipped": total("skipped"),
        "time_seconds": float(root.get("time", sum(float(suite.get("time", 0)) for suite in suites))),
        "sha256": sha256_file(path),
        "testcase_nodes": len(cases),
        "failure_nodes": len(root.findall(".//failure")),
        "error_nodes": len(root.findall(".//error")),
        "skipped_nodes": len(root.findall(".//skipped")),
        "final_lock_test_names": final_lock_names,
    }


def validate_full_junit(result, minimum_tests=MINIMUM_FINAL_TEST_COUNT):
    require(result["failures"] == result["errors"] == result["skipped"] == 0, "Final JUnit report is not all-pass")
    require(result["tests"] >= minimum_tests, "JUnit report is not the guarded full test suite")
    require(result["testcase_nodes"] == result["tests"], "JUnit aggregate count does not match testcase nodes")
    require(result["failure_nodes"] == result["failures"], "JUnit failure nodes contradict aggregate")
    require(result["error_nodes"] == result["errors"], "JUnit error nodes contradict aggregate")
    require(result["skipped_nodes"] == result["skipped"], "JUnit skipped nodes contradict aggregate")
    require(FINAL_LOCK_TEST_NAMES <= set(result["final_lock_test_names"]), "Final-lock tests are absent from JUnit evidence")


def validate_full_cohort_verifier(result, head, pytest_tests):
    require(result.get("verification") == "passed", "Full-cohort verifier log does not pass")
    require(result.get("batch_id") == "005", "Full-cohort verifier did not check Batch 005")
    require(result.get("subjects") == 65 and result.get("blocks") == 1040, "Full-cohort Batch 005 counts mismatch")
    require(result.get("terminal_subjects") == 65, "Full-cohort terminal-subject count mismatch")
    require(result.get("technical_failed_subjects") == [], "Full-cohort verifier reports a technical failure")
    require(result.get("all_subjects_technically_complete") is True, "Full-cohort technical completion is false")
    require(result.get("git_head") == head, "Full-cohort verifier checked another commit")
    require(result.get("worktree_clean") is True, "Full-cohort verifier did not check a clean worktree")
    require(result.get("pytest_count") == pytest_tests, "Full-cohort verifier/test count mismatch")
    require(result.get("no_future_batch") is True, "Full-cohort verifier reports a future batch")
    require(result.get("prior_batch_unchanged") is True, "Full-cohort verifier reports prior-batch drift")
    require(result.get("no_normalization") is True, "Full-cohort verifier reports fitted normalization")
    require(result.get("no_scientific_heldout_analysis") is True, "Full-cohort verifier reports held-out analysis")
    require(result.get("metadata_clone_clean") is True, "Full-cohort verifier reports metadata-clone drift")


def validate_review_results(reviews):
    expected_reviews = {"data_reconciliation", "leakage_split", "artifact_test"}
    require(set(reviews) == expected_reviews, "Independent review evidence is incomplete")
    for name, result in reviews.items():
        require(result.get("status") == "passed", f"Independent review did not pass: {name}")
        findings = result.get("findings")
        resolutions = result.get("resolutions")
        unresolved = result.get("unresolved_findings")
        require(isinstance(findings, list), f"Review findings missing: {name}")
        require(isinstance(resolutions, list), f"Review resolutions missing: {name}")
        require(unresolved == [], f"Review has unresolved findings: {name}")
        require(set(findings) <= set(resolutions), f"Review finding lacks a resolution: {name}")


def validate_commit_topology(parent, commit_count, parent_field_count):
    require(parent == ACCEPTED_STREAMING_HEAD, "Final lock is not exactly one commit after accepted streaming")
    require(commit_count == 1, "More than one final-lock commit exists")
    require(parent_field_count == 2, "Final lock commit is a merge")


def write_external_acceptance(repo_root, evidence_directory):
    """Write the post-commit record outside Git after all command evidence exists."""
    root = Path(repo_root).resolve()
    paths = load_paths(root)
    evidence = Path(evidence_directory).resolve()
    docs_root = (paths["runtime_root"] / "docs").resolve()
    require(docs_root in evidence.parents, "Acceptance directory must be under external runtime/docs")
    require("final_cohort_split_lock" in evidence.name.lower(), "Acceptance directory name is not phase-specific")
    require(evidence.is_dir(), "Acceptance evidence directory is missing")

    required_results = [
        "pytest",
        "compileall",
        "full_cohort_verifier",
        "final_lock_verifier_precommit",
        "final_lock_verifier_postcommit",
    ]
    exit_codes = {}
    for name in required_results:
        exit_path = evidence / f"{name}_exit_code.txt"
        require(exit_path.is_file(), f"Missing acceptance exit code: {name}")
        exit_codes[name] = int(exit_path.read_text(encoding="utf-8-sig").strip())
        require(exit_codes[name] == 0, f"Acceptance command failed: {name}")

    verification = verify_artifacts(root, require_clean=True)
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    parent = _git_output(root, "rev-parse", "HEAD^")
    require(branch == "main", "Final lock commit is not on main")
    require(head != ACCEPTED_STREAMING_HEAD, "Final lock commit is missing")
    validate_commit_topology(
        parent,
        int(_git_output(root, "rev-list", "--count", f"{ACCEPTED_STREAMING_HEAD}..{head}")),
        len(_git_output(root, "rev-list", "--parents", "-n", "1", head).split()),
    )

    metrics = root / "outputs/metrics/scenario2"
    summary = read_json(metrics / ARTIFACT_FILENAMES["summary"])
    lock_record = read_json(metrics / ARTIFACT_FILENAMES["lock"])
    report = root / "outputs/logs/pytest.xml"
    pytest = junit_summary(report)
    validate_full_junit(pytest)

    full_cohort = read_json(evidence / "full_cohort_verifier_stdout.json")
    validate_full_cohort_verifier(full_cohort, head, pytest["tests"])

    precommit_verifier = read_json(evidence / "final_lock_verifier_precommit_stdout.json")
    postcommit_verifier = read_json(evidence / "final_lock_verifier_postcommit_stdout.json")
    for name, result in (("precommit", precommit_verifier), ("postcommit", postcommit_verifier)):
        require(result.get("status") == FINAL_STATUS and result.get("verification") == "passed", f"Final-lock {name} verifier log does not pass")
        require(result.get("deterministic_regeneration") == "byte_identical", f"Final-lock {name} verifier lacks determinism proof")
    require(postcommit_verifier.get("repository", {}).get("current_head") == head, "Post-commit verifier checked another commit")
    require(postcommit_verifier.get("repository", {}).get("worktree_clean") is True, "Post-commit verifier did not check clean state")

    reviews = read_json(evidence / "review_results.json")
    validate_review_results(reviews)

    import compileall
    require(
        all(compileall.compile_dir(root / directory, quiet=1) for directory in ("scripts", "src", "tests")),
        "Post-commit compileall verification failed",
    )

    artifacts = []
    for filename in ARTIFACT_FILENAMES.values():
        path = metrics / filename
        artifacts.append(
            {
                "path": path.resolve().as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    evidence_files = []
    for path in sorted(evidence.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "final_acceptance.json":
            evidence_files.append(
                {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            )

    record = {
        "status": FINAL_STATUS,
        "accepted_utc": datetime.now(timezone.utc).isoformat(),
        "starting_head": ACCEPTED_STREAMING_HEAD,
        "final_committed_head": head,
        "commit_parent": parent,
        "branch": branch,
        "one_new_local_commit": True,
        "worktree_clean": verification["repository"]["worktree_clean"],
        "git_push_performed": False,
        "no_push_confirmation": "No git push command was executed; application-level confirmation.",
        "cohort_and_split_counts": summary["counts"],
        "structural_exclusions": summary["structural_exclusions"],
        "post_download_exclusions": summary["post_download_exclusions"],
        "pilot_exposure": summary["pilot_exposure"],
        "artifact_files": artifacts,
        "artifact_lock_sha256": sha256_file(metrics / ARTIFACT_FILENAMES["lock"]),
        "accepted_source_bindings": lock_record["source_bindings"],
        "test_and_verifier_results": {
            "exit_codes": exit_codes,
            "pytest": pytest,
            "final_lock_verification": "passed",
            "full_cohort_preservation_verification": "passed",
            "deterministic_regeneration": verification["deterministic_regeneration"],
            "independent_reviewers": reviews,
        },
        "evidence_file_hashes": evidence_files,
        "normalization_unfitted": True,
        "final_test_sealed": True,
        "scientific_heldout_outcomes_accessed": False,
        "eeg_uv_arrays_loaded_for_lock": False,
        "batch_006_exists": False,
        "tracked_signal_payloads": 0,
        "next_action": "TRAIN-ONLY NORMALIZATION",
    }
    target = evidence / "final_acceptance.json"
    atomic_write(target, json_bytes(record))
    return record
