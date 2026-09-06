from pathlib import Path

import pandas as pd
import pytest

from resting_eeg.data import final_cohort_split as lock


def _structural(subjects, eligible=True):
    return pd.DataFrame(
        [
            {
                "subject_id": subject,
                "structurally_eligible": eligible,
                "primary_reason": "meets" if eligible else "excluded",
                "secondary_reason": "",
            }
            for subject in subjects
        ]
    )


def _completed(subjects):
    return pd.DataFrame(
        [
            {
                "subject_id": subject,
                "technically_complete": True,
                "blocks": 16,
                "train_qc_pass": True,
                "provisional_5_1_2_feasible": True,
                "feasibility_reason": "",
                "output_sha256": "0" * 64,
                "receipt_sha256": "1" * 64,
            }
            for subject in subjects
        ]
    )


def test_duplicate_subject_detection():
    structural = _structural(["sub-001", "sub-001"])
    with pytest.raises(ValueError, match="Duplicate subject"):
        lock.validate_subject_sets(structural, _completed(["sub-001"]))


def test_missing_subject_detection():
    structural = _structural(["sub-001", "sub-002"])
    with pytest.raises(ValueError, match="Missing or unexpected technical subject evidence"):
        lock.validate_subject_sets(structural, _completed(["sub-001"]))


def test_altered_split_assignment_detection():
    provisional = pd.DataFrame(
        [{"subject_id": "sub-001", "block_index": 1, "split": "Train"}]
    )
    final = [
        {
            "subject_id": "sub-001",
            "block_index": 1,
            "split": "Validation",
            "eye_state": "EO",
            "block_eligible": True,
        }
    ]
    with pytest.raises(ValueError, match="exactly match provisional roles"):
        lock.validate_final_roles(final, provisional, {"sub-001"})


def test_pilot_exposure_violation_detection():
    cohort = [{"subject_id": "sub-001", "final_cohort_included": True}]
    final = [{"subject_id": "sub-001", "block_index": 1, "split": "Validation"}]
    with pytest.raises(ValueError, match="Pilot exposure violation"):
        lock.validate_pilot_exposure(final, cohort, {("sub-001", 1)})


def test_source_hash_mismatch_detection(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("frozen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source hash mismatch"):
        lock.validate_file_hash(source, "0" * 64)


def test_unexpected_cohort_count_detection():
    malformed = [
        {
            "subject_id": "sub-001",
            "final_cohort_included": True,
            "exclusion_stage": "",
        }
    ]
    with pytest.raises(ValueError, match="Unexpected original cohort count"):
        lock.validate_expected_cohort(malformed)


def test_non_idempotent_regeneration_detection(tmp_path):
    expected = {"artifact.json": b'{"stable":true}\n'}
    (tmp_path / "artifact.json").write_bytes(b'{"stable":false}\n')
    with pytest.raises(ValueError, match="differs from deterministic regeneration"):
        lock.validate_artifact_bytes(tmp_path, expected)


def test_byte_identical_artifact_validation(tmp_path):
    expected = {"artifact.json": b'{"stable":true}\n'}
    (tmp_path / "artifact.json").write_bytes(expected["artifact.json"])
    lock.validate_artifact_bytes(tmp_path, expected)


def test_ineligible_block_reason_is_deterministic():
    assert (
        lock._eligibility_reason(False, 0, 25, minimum_windows=10, minimum_fraction=0.5)
        == "usable_windows_below_10;usable_fraction_below_0.5"
    )
    assert lock._eligibility_reason(True, 25, 25, 10, 0.5) == "frozen_qc_eligible"


def test_junit_summary_parses_guarded_counts(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text(
        '<testsuites tests="1" failures="0" errors="0" skipped="0" time="0.5">'
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_final_cohort_split" name="sample" />'
        '</testsuite></testsuites>',
        encoding="utf-8",
    )
    summary = lock.junit_summary(report)
    assert summary["tests"] == summary["testcase_nodes"] == 1
    assert summary["failures"] == summary["errors"] == summary["skipped"] == 0


def test_tiny_junit_cannot_be_accepted():
    result = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "testcase_nodes": 0,
        "failure_nodes": 0,
        "error_nodes": 0,
        "skipped_nodes": 0,
        "final_lock_test_names": [],
    }
    with pytest.raises(ValueError, match="not the guarded full test suite"):
        lock.validate_full_junit(result, minimum_tests=1)


def test_incomplete_full_cohort_verifier_cannot_be_accepted():
    with pytest.raises(ValueError, match="Batch 005 counts mismatch"):
        lock.validate_full_cohort_verifier({"verification": "passed", "batch_id": "005"}, "head", 398)


def test_unresolved_review_cannot_be_accepted():
    reviews = {
        name: {
            "status": "passed",
            "findings": ["critical"],
            "resolutions": [],
            "unresolved_findings": ["critical"],
        }
        for name in ("data_reconciliation", "leakage_split", "artifact_test")
    }
    with pytest.raises(ValueError, match="unresolved findings"):
        lock.validate_review_results(reviews)


def test_merge_commit_topology_cannot_be_accepted():
    with pytest.raises(ValueError, match="merge"):
        lock.validate_commit_topology(lock.ACCEPTED_STREAMING_HEAD, 1, 3)


def test_alternate_acceptance_record_cannot_be_mislabeled(tmp_path):
    canonical = tmp_path / "canonical.json"
    alternate = tmp_path / "alternate.json"
    with pytest.raises(ValueError, match="canonical external record"):
        lock.validate_acceptance_path(alternate, canonical)
