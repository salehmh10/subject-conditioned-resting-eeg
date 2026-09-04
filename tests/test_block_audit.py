import pandas as pd

from scripts.audit.s2_02_audit_blocks import build_blocks


STANDARD = {"Stimulus/S200": "EO", "Stimulus/S210": "EC"}
THRESHOLDS = {
    "expected_block_duration_sec": 60.0,
    "minimum_typical_block_duration_sec": 55.0,
    "maximum_typical_block_duration_sec": 65.0,
    "expected_markers_per_block": 30.0,
    "same_state_gap_warning_sec": 4.5,
    "cross_state_gap_warning_sec": 30.0,
}


def events(rows):
    return pd.DataFrame(rows, columns=["onset", "trial_type", "value"])


def parse(frame, subject="sub-test", override=None, recording_duration=None):
    return build_blocks(
        frame,
        subject,
        STANDARD,
        override or {},
        THRESHOLDS,
        "synthetic override evidence",
        recording_duration,
    )


def test_normal_repeated_markers_make_two_blocks_and_use_trial_type() -> None:
    frame = events(
        [(0, "Stimulus/S210", 999), (2, "Stimulus/S210", 999), (4, "Stimulus/S210", 999),
         (10, "Stimulus/S200", 4), (12, "Stimulus/S200", 4), (14, "Stimulus/S200", 4)]
    )
    blocks, status, _ = parse(frame)
    assert status == "ok"
    assert [block["state"] for block in blocks] == ["EC", "EO"]
    assert [block["n_markers"] for block in blocks] == [3, 3]


def test_s208_maps_only_with_subject_override() -> None:
    frame = events([(0, "Stimulus/S208", 3), (2, "Stimulus/S208", 3), (4, "Stimulus/S200", 2)])
    blocks, _, anomalies = parse(frame, subject="sub-override", override={"Stimulus/S208": "EC"})
    assert [block["state"] for block in blocks] == ["EC", "EO"]
    assert any(row["anomaly_type"] == "subject_specific_override_applied" for row in anomalies)


def test_s208_without_override_is_ignored_and_flagged() -> None:
    frame = events([(0, "Stimulus/S208", 3), (2, "Stimulus/S208", 3), (4, "Stimulus/S200", 2)])
    blocks, _, anomalies = parse(frame)
    assert [block["state"] for block in blocks] == ["EO"]
    assert any(row["anomaly_type"] == "unmapped_state_like_code" for row in anomalies)


def test_missing_events() -> None:
    blocks, status, anomalies = parse(None)
    assert blocks == []
    assert status == "events_file_missing_or_ambiguous"
    assert anomalies[0]["anomaly_type"] == "events_missing_or_unreadable"


def test_no_state_markers() -> None:
    blocks, status, anomalies = parse(events([(0, "Comment/no USB Connection to actiCAP", 1)]))
    assert blocks == []
    assert status == "no_state_markers"
    assert anomalies[-1]["anomaly_type"] == "no_state_markers"


def test_truncated_final_block_is_detected_and_capped_to_recording() -> None:
    normal_ec = [(2 * index, "Stimulus/S210", 4) for index in range(30)]
    short_eo = [(70 + 2 * index, "Stimulus/S200", 3) for index in range(10)]
    blocks, _, anomalies = parse(events(normal_ec + short_eo), recording_duration=88.25)
    assert len(blocks) == 2
    assert blocks[-1]["is_complete"] is False
    assert blocks[-1]["recorded_span_upper_bound_sec"] == 18.25
    assert any(row["anomaly_type"] == "truncated_final_block" for row in anomalies)


def test_large_within_state_gap_does_not_split_or_impute() -> None:
    frame = events([(0, "Stimulus/S210", 4), (2, "Stimulus/S210", 4), (10, "Stimulus/S210", 4)])
    blocks, _, anomalies = parse(frame)
    assert len(blocks) == 1
    assert blocks[0]["n_abnormal_same_state_gaps"] == 1
    assert any(row["anomaly_type"] == "abnormal_same_state_marker_gap" for row in anomalies)


def test_alternating_state_sequence() -> None:
    frame = events([(0, "Stimulus/S210", 4), (2, "Stimulus/S200", 3), (4, "Stimulus/S210", 4)])
    blocks, _, _ = parse(frame)
    assert [block["state"] for block in blocks] == ["EC", "EO", "EC"]


def test_repeated_same_state_markers_are_not_separate_blocks() -> None:
    frame = events([(2 * index, "Stimulus/S200", 3) for index in range(30)])
    blocks, _, _ = parse(frame)
    assert len(blocks) == 1
    assert blocks[0]["is_complete"] is True


def test_new_segment_preserves_aborted_fragment_and_starts_new_run() -> None:
    fragment = [(0, "Stimulus/S210", 4), (2, "Stimulus/S210", 4), (4, "Stimulus/S210", 4),
                (10, "New Segment/", 2)]
    complete_ec = [(20 + 2 * index, "Stimulus/S210", 4) for index in range(30)]
    complete_eo = [(90 + 2 * index, "Stimulus/S200", 3) for index in range(30)]
    blocks, _, anomalies = parse(events(fragment + complete_ec + complete_eo))
    assert len(blocks) == 3
    assert blocks[0]["is_interruption_fragment"] is True
    assert blocks[0]["is_study_block"] is False
    assert blocks[1]["state_block_index"] == 1
    assert any(row["anomaly_type"] == "same_state_gap_across_acquisition_segment" for row in anomalies)
