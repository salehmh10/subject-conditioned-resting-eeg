from pathlib import Path

import pandas as pd
import pytest

from scripts.audit.s2_03_build_structural_cohort import classify
from scripts.audit.s2_05_plan_splits_and_downloads import build_split_manifest
from scripts.data.s2_06_download_manifest import safe_destination, validate_manifest


CRITERIA = {
    "require_all_19_target_channels": True,
    "require_alternating_study_blocks": True,
    "required_study_blocks_per_state": 8,
    "required_complete_blocks_per_state": 8,
}


SPLIT_CONFIG = {
    "status": "PROVISIONAL_METADATA_ONLY",
    "algorithm_version": "test-v1",
    "subject_order_seed": "test-v1",
    "base_roles": ["Test", "Train", "Train", "Validation", "Train", "Test", "Train", "Train"],
    "state_offsets": {"EO": 0, "EC": 4},
    "requirements": {"blocks_per_subject_state": 8, "train_blocks": 5, "validation_blocks": 1, "test_blocks": 2},
}


def eligible_row(**updates):
    row = {
        "parser_status": "ok",
        "channels_readable": True,
        "target_19_present": True,
        "n_EO_blocks": 8,
        "n_EC_blocks": 8,
        "n_complete_EO_blocks": 8,
        "n_complete_EC_blocks": 8,
        "alternating": True,
    }
    row.update(updates)
    return pd.Series(row)


def test_structural_eligibility_reasons_are_machine_readable() -> None:
    assert classify(eligible_row(), CRITERIA)[0] is True
    eligible, reason, _ = classify(eligible_row(n_complete_EO_blocks=7), CRITERIA)
    assert eligible is False
    assert reason == "insufficient_complete_blocks_for_5_1_2"
    eligible, reason, _ = classify(
        eligible_row(parser_status="events_file_missing_or_ambiguous"), CRITERIA
    )
    assert eligible is False
    assert reason == "state_annotations_unavailable_missing_events"


def synthetic_blocks(subjects):
    rows = []
    for subject in subjects:
        block_index = 0
        for state in ["EO", "EC"]:
            for ordinal in range(1, 9):
                block_index += 1
                rows.append(
                    {
                        "subject_id": subject,
                        "block_index": block_index,
                        "acquisition_segment_id": 1,
                        "state": state,
                        "state_block_index": ordinal,
                        "start_sec": block_index * 60.0,
                        "end_estimate_capped_sec": block_index * 60.0 + 60.0,
                        "n_markers": 30,
                        "is_study_block": True,
                        "is_complete": True,
                    }
                )
    return pd.DataFrame(rows)


def test_provisional_split_is_deterministic_disjoint_and_order_balanced() -> None:
    subjects = [f"sub-{index:03d}" for index in range(9)]
    cohort = pd.DataFrame({"subject_id": subjects, "structurally_eligible": True})
    blocks = synthetic_blocks(subjects)
    first = build_split_manifest(blocks, cohort, SPLIT_CONFIG)
    second = build_split_manifest(blocks, cohort, SPLIT_CONFIG)
    pd.testing.assert_frame_equal(first, second)
    counts = first.groupby(["subject_id", "state", "split"]).size().unstack(fill_value=0)
    assert (counts[["Train", "Validation", "Test"]] == [5, 1, 2]).all().all()
    for (_, split), group in first.groupby(["state", "split"]):
        per_ordinal = group["state_block_index"].value_counts()
        assert per_ordinal.max() - per_ordinal.min() <= 1


def test_download_manifest_validation_is_network_free_and_path_safe(tmp_path: Path) -> None:
    digest = "a" * 64
    frame = pd.DataFrame(
        [
            {
                "subject_id": "sub-001",
                "relative_path": "sub-001/eeg/sub-001_task-resting_eeg.eeg",
                "source_url": "https://nemar.s3.us-east-2.amazonaws.com/nm000179/objects/key",
                "payload_size_bytes": 123,
                "payload_sha256": digest,
            }
        ]
    )
    validated = validate_manifest(frame, tmp_path)
    assert len(validated) == 1
    assert validated[0][1].is_relative_to(tmp_path)
    with pytest.raises(ValueError):
        safe_destination(tmp_path, "../escape.eeg")
