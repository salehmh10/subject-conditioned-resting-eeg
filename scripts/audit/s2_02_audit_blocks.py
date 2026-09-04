"""Audit MPI-LEMON EO/EC marker runs without reading EEG waveforms."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ANOMALY_COLUMNS = [
    "subject_id",
    "anomaly_type",
    "severity",
    "block_index",
    "state",
    "previous_onset_sec",
    "onset_sec",
    "gap_sec",
    "trial_type",
    "context",
    "interpretation",
]


def load_event_config(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, float], dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    standard = {str(key): str(value) for key, value in raw.get("standard_mapping", {}).items()}
    overrides: dict[str, dict[str, str]] = {}
    evidence: dict[str, str] = {}
    for subject_id, spec in raw.get("subject_overrides", {}).items():
        overrides[str(subject_id)] = {
            str(key): str(value) for key, value in (spec.get("mapping", {}) or {}).items()
        }
        evidence[str(subject_id)] = str(spec.get("evidence", ""))
    thresholds = {str(key): float(value) for key, value in raw.get("audit_thresholds", {}).items()}
    return standard, overrides, thresholds, evidence


def load_events(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        return None, "events_file_missing_or_ambiguous"
    try:
        return pd.read_csv(path, sep="\t"), ""
    except Exception as exc:
        return None, f"events_unreadable:{type(exc).__name__}:{exc}"


def anomaly(
    subject_id: str,
    anomaly_type: str,
    severity: str,
    interpretation: str,
    **kwargs: Any,
) -> dict[str, Any]:
    row = {column: "" for column in ANOMALY_COLUMNS}
    row.update(
        subject_id=subject_id,
        anomaly_type=anomaly_type,
        severity=severity,
        interpretation=interpretation,
    )
    row.update(kwargs)
    return row


def event_context(events: pd.DataFrame, start: float, end: float, limit: int = 12) -> str:
    nearby = events.loc[events["_numeric_onset"].between(start, end, inclusive="both")]
    values = [
        f"{trial_type}@{onset:.4f}"
        for trial_type, onset in nearby[["trial_type", "_numeric_onset"]].itertuples(index=False, name=None)
    ]
    if len(values) > limit:
        values = values[: limit // 2] + [f"...{len(values) - limit} omitted..."] + values[-limit // 2 :]
    return " | ".join(values)


def _summarize_group(
    subject_id: str,
    block_index: int,
    group: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    onsets = np.asarray([float(row["onset"]) for row in group], dtype=float)
    intervals = np.diff(onsets)
    normal_intervals = intervals[intervals <= thresholds["same_state_gap_warning_sec"]]
    cadence = float(np.median(normal_intervals)) if len(normal_intervals) else 2.0
    elapsed = float(onsets[-1] - onsets[0] + cadence)
    nominal_coverage = float(len(onsets) * cadence)
    large_gaps = intervals[intervals > thresholds["same_state_gap_warning_sec"]]
    marker_types = [str(row["trial_type"]) for row in group]
    sources = sorted(set(str(row["mapping_source"]) for row in group))
    complete = (
        len(onsets) == int(thresholds["expected_markers_per_block"])
        and thresholds["minimum_typical_block_duration_sec"] <= elapsed
        <= thresholds["maximum_typical_block_duration_sec"]
        and not len(large_gaps)
    )
    return {
        "subject_id": subject_id,
        "block_index": block_index,
        "acquisition_segment_id": int(group[0]["acquisition_segment_id"]),
        "state": str(group[0]["state"]),
        "start_sec": float(onsets[0]),
        "last_marker_sec": float(onsets[-1]),
        "end_estimate_sec": float(onsets[-1] + cadence),
        "elapsed_span_sec": elapsed,
        "nominal_marker_coverage_sec": nominal_coverage,
        "n_markers": len(onsets),
        "median_marker_interval_sec": cadence,
        "max_marker_interval_sec": float(intervals.max()) if len(intervals) else np.nan,
        "n_abnormal_same_state_gaps": int(len(large_gaps)),
        "first_marker_type": marker_types[0],
        "last_marker_type": marker_types[-1],
        "marker_types": ";".join(sorted(set(marker_types))),
        "mapping_source": "+".join(sources),
        "contains_s208": "Stimulus/S208" in marker_types,
        "is_complete": complete,
    }


def build_blocks(
    events: pd.DataFrame | None,
    subject_id: str,
    standard_mapping: dict[str, str],
    subject_override: dict[str, str] | None,
    thresholds: dict[str, float],
    override_evidence: str = "",
    recording_duration_sec: float | None = None,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    anomalies: list[dict[str, Any]] = []
    if events is None:
        anomalies.append(
            anomaly(subject_id, "events_missing_or_unreadable", "exclude", "EO/EC annotations are unavailable.")
        )
        return [], "events_file_missing_or_ambiguous", anomalies
    if not {"trial_type", "onset"}.issubset(events.columns):
        anomalies.append(
            anomaly(subject_id, "required_event_columns_missing", "exclude", "Both onset and trial_type are required.")
        )
        return [], "required_columns_missing", anomalies

    raw = events.copy()
    raw["_source_row"] = np.arange(len(raw))
    raw["trial_type"] = raw["trial_type"].fillna("<missing>").astype(str)
    raw["_numeric_onset"] = pd.to_numeric(raw["onset"], errors="coerce")

    mapping = dict(standard_mapping)
    subject_override = subject_override or {}
    mapping.update(subject_override)
    state_like = raw["trial_type"].str.match(r"^Stimulus/S20[08]$|^Stimulus/S210$")
    unmapped = raw.loc[state_like & ~raw["trial_type"].isin(mapping)]
    for _, row in unmapped.iterrows():
        anomalies.append(
            anomaly(
                subject_id,
                "unmapped_state_like_code",
                "warning",
                "Code is ignored; no global S208 interpretation is permitted.",
                onset_sec=row["_numeric_onset"],
                trial_type=row["trial_type"],
            )
        )

    invalid_state = raw.loc[raw["trial_type"].isin(mapping) & raw["_numeric_onset"].isna()]
    for _, row in invalid_state.iterrows():
        anomalies.append(
            anomaly(
                subject_id,
                "invalid_state_marker_onset",
                "warning",
                "Mapped marker is ignored because onset is not numeric.",
                trial_type=row["trial_type"],
            )
        )

    valid = raw.loc[raw["_numeric_onset"].notna()].sort_values(
        ["_numeric_onset", "_source_row"], kind="stable"
    )
    if valid.duplicated(["_numeric_onset", "trial_type"]).any():
        for _, row in valid.loc[
            valid.duplicated(["_numeric_onset", "trial_type"], keep=False)
        ].iterrows():
            anomalies.append(
                anomaly(
                    subject_id,
                    "duplicate_event",
                    "warning",
                    "Duplicate onset/trial_type entry requires later waveform-alignment review.",
                    onset_sec=row["_numeric_onset"],
                    trial_type=row["trial_type"],
                )
            )

    acquisition_segment_id = 1
    mapped_rows: list[dict[str, Any]] = []
    for _, row in valid.iterrows():
        if row["trial_type"] == "New Segment/":
            acquisition_segment_id += 1
            anomalies.append(
                anomaly(
                    subject_id,
                    "explicit_acquisition_segment",
                    "info",
                    "Explicit New Segment marker starts a new acquisition segment; state runs cannot bridge it.",
                    onset_sec=row["_numeric_onset"],
                    trial_type=row["trial_type"],
                )
            )
            continue
        if row["trial_type"] not in mapping:
            continue
        mapped_rows.append(
            {
                "onset": float(row["_numeric_onset"]),
                "trial_type": row["trial_type"],
                "state": mapping[row["trial_type"]],
                "mapping_source": "override" if row["trial_type"] in subject_override else "standard",
                "acquisition_segment_id": acquisition_segment_id,
            }
        )

    if subject_override and any(row["mapping_source"] == "override" for row in mapped_rows):
        anomalies.append(
            anomaly(
                subject_id,
                "subject_specific_override_applied",
                "info",
                override_evidence or "Documented subject-specific event-code override applied.",
                trial_type=";".join(sorted(subject_override)),
            )
        )
    if not mapped_rows:
        anomalies.append(
            anomaly(subject_id, "no_state_markers", "exclude", "No mapped EO/EC trial_type markers are present.")
        )
        return [], "no_state_markers", anomalies

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in mapped_rows:
        if current and (
            row["state"] != current[-1]["state"]
            or row["acquisition_segment_id"] != current[-1]["acquisition_segment_id"]
        ):
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)

    blocks = [
        _summarize_group(subject_id, index + 1, group, thresholds) for index, group in enumerate(groups)
    ]
    for index, block in enumerate(blocks):
        previous = blocks[index - 1] if index else None
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if previous is None:
            started_by = "recording_start"
        elif previous["acquisition_segment_id"] != block["acquisition_segment_id"]:
            started_by = "acquisition_segment"
        else:
            started_by = "state_change"
        if following is None:
            ended_by = "recording_end"
        elif following["acquisition_segment_id"] != block["acquisition_segment_id"]:
            ended_by = "acquisition_segment"
        else:
            ended_by = "state_change"
        block["started_by"] = started_by
        block["ended_by"] = ended_by
        capped_end = block["end_estimate_sec"]
        if ended_by == "recording_end" and recording_duration_sec is not None:
            capped_end = min(capped_end, recording_duration_sec)
        block["end_estimate_capped_sec"] = capped_end
        block["recorded_span_upper_bound_sec"] = max(0.0, capped_end - block["start_sec"])
        block["is_interruption_fragment"] = bool(
            not block["is_complete"]
            and (started_by == "acquisition_segment" or ended_by == "acquisition_segment")
        )
        block["is_study_block"] = not block["is_interruption_fragment"]
        block["state_block_index"] = pd.NA

    state_indices: Counter[str] = Counter()
    for block in blocks:
        if block["is_study_block"]:
            state_indices[block["state"]] += 1
            block["state_block_index"] = state_indices[block["state"]]

    for block in blocks:
        block_index = block["block_index"]
        if block["n_abnormal_same_state_gaps"]:
            group = groups[block_index - 1]
            for previous, current_row in zip(group[:-1], group[1:]):
                gap = current_row["onset"] - previous["onset"]
                if gap > thresholds["same_state_gap_warning_sec"]:
                    anomalies.append(
                        anomaly(
                            subject_id,
                            "abnormal_same_state_marker_gap",
                            "warning",
                            "Gap is retained within one run; no missing state is imputed.",
                            block_index=block_index,
                            state=block["state"],
                            previous_onset_sec=previous["onset"],
                            onset_sec=current_row["onset"],
                            gap_sec=gap,
                            context=event_context(raw, previous["onset"], current_row["onset"]),
                        )
                    )
        if block["n_markers"] != int(thresholds["expected_markers_per_block"]):
            kind = "interrupted_state_fragment" if block["is_interruption_fragment"] else "unusual_marker_count"
            anomalies.append(
                anomaly(
                    subject_id,
                    kind,
                    "warning" if block["is_interruption_fragment"] else "exclude",
                    (
                        "Explicit acquisition boundary isolates an aborted fragment; it is not counted as a study block."
                        if block["is_interruption_fragment"]
                        else "Study block has a nonstandard marker count and is incomplete."
                    ),
                    block_index=block_index,
                    state=block["state"],
                    onset_sec=block["start_sec"],
                    context=(
                        f"n_markers={block['n_markers']};elapsed_span_sec={block['elapsed_span_sec']:.4f};"
                        f"recorded_span_upper_bound_sec={block['recorded_span_upper_bound_sec']:.4f}"
                    ),
                )
            )
        if not (
            thresholds["minimum_typical_block_duration_sec"]
            <= block["elapsed_span_sec"]
            <= thresholds["maximum_typical_block_duration_sec"]
        ):
            anomalies.append(
                anomaly(
                    subject_id,
                    "unusual_block_duration",
                    "warning" if block["is_interruption_fragment"] else "exclude",
                    "Elapsed state-marker span falls outside the prespecified 55--65 s structural range.",
                    block_index=block_index,
                    state=block["state"],
                    onset_sec=block["start_sec"],
                    context=(
                        f"elapsed_span_sec={block['elapsed_span_sec']:.4f};"
                        f"recorded_span_upper_bound_sec={block['recorded_span_upper_bound_sec']:.4f}"
                    ),
                )
            )
        if not block["is_complete"] and block["ended_by"] == "recording_end" and block["is_study_block"]:
            anomalies.append(
                anomaly(
                    subject_id,
                    "truncated_final_block",
                    "exclude",
                    "Incomplete final marker run ends with the event file; no completion is inferred.",
                    block_index=block_index,
                    state=block["state"],
                    onset_sec=block["start_sec"],
                )
            )

    # Gap audit uses mapped markers across state changes and acquisition segments.
    for previous, current_row in zip(mapped_rows[:-1], mapped_rows[1:]):
        gap = current_row["onset"] - previous["onset"]
        segment_changed = current_row["acquisition_segment_id"] != previous["acquisition_segment_id"]
        if segment_changed and current_row["state"] == previous["state"]:
            anomalies.append(
                anomaly(
                    subject_id,
                    "same_state_gap_across_acquisition_segment",
                    "warning",
                    "Explicit acquisition interruption separates same-state fragments; the interval is unlabeled.",
                    state=current_row["state"],
                    previous_onset_sec=previous["onset"],
                    onset_sec=current_row["onset"],
                    gap_sec=gap,
                    context=event_context(raw, previous["onset"], current_row["onset"]),
                )
            )
        elif (
            not segment_changed
            and current_row["state"] != previous["state"]
            and gap > thresholds["cross_state_gap_warning_sec"]
        ):
            anomalies.append(
                anomaly(
                    subject_id,
                    "abnormal_cross_state_gap",
                    "warning",
                    "Long transition interval is unlabeled; neither adjacent state is extended into it.",
                    state=f"{previous['state']}->{current_row['state']}",
                    previous_onset_sec=previous["onset"],
                    onset_sec=current_row["onset"],
                    gap_sec=gap,
                    context=event_context(raw, previous["onset"], current_row["onset"]),
                )
            )

    return blocks, "ok", anomalies


def summarize_subject(
    subject_id: str,
    status: str,
    blocks: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    has_override: bool,
) -> dict[str, Any]:
    study = [block for block in blocks if block["is_study_block"]]
    complete = [block for block in study if block["is_complete"]]
    states = [block["state"] for block in study]
    eo = [block for block in study if block["state"] == "EO"]
    ec = [block for block in study if block["state"] == "EC"]
    complete_eo = [block for block in complete if block["state"] == "EO"]
    complete_ec = [block for block in complete if block["state"] == "EC"]
    return {
        "subject_id": subject_id,
        "parser_status": status,
        "n_detected_runs_total": len(blocks),
        "n_interruption_fragments": sum(block["is_interruption_fragment"] for block in blocks),
        "n_blocks_total": len(study),
        "n_EO_blocks": len(eo),
        "n_EC_blocks": len(ec),
        "n_complete_blocks_total": len(complete),
        "n_complete_EO_blocks": len(complete_eo),
        "n_complete_EC_blocks": len(complete_ec),
        "alternating": bool(states) and all(a != b for a, b in zip(states[:-1], states[1:])),
        "median_EO_elapsed_sec": float(np.median([b["elapsed_span_sec"] for b in eo])) if eo else np.nan,
        "median_EC_elapsed_sec": float(np.median([b["elapsed_span_sec"] for b in ec])) if ec else np.nan,
        "has_s208": any(block["contains_s208"] for block in blocks),
        "has_event_override": has_override,
        "n_abnormal_same_state_gaps": sum(a["anomaly_type"] == "abnormal_same_state_marker_gap" for a in anomalies),
        "n_abnormal_cross_state_gaps": sum(a["anomaly_type"] == "abnormal_cross_state_gap" for a in anomalies),
        "has_timing_uncertainty": any(
            a["anomaly_type"]
            in {
                "abnormal_same_state_marker_gap",
                "abnormal_cross_state_gap",
                "same_state_gap_across_acquisition_segment",
                "interrupted_state_fragment",
            }
            for a in anomalies
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="D:/SHARIF/PROJECT/Datasets/MPI-LEMON")
    parser.add_argument("--output-dir", default="outputs/metrics/scenario2")
    parser.add_argument("--event-config", default="configs/audit/event_code_overrides.yaml")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    standard, overrides, thresholds, evidence = load_event_config(Path(args.event_config))

    all_blocks: list[dict[str, Any]] = []
    all_anomalies: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    subject_dirs = sorted(path for path in root.glob("sub-*") if path.is_dir())
    for subject_dir in subject_dirs:
        event_files = sorted((subject_dir / "eeg").glob("*_task-resting_events.tsv"))
        events, load_status = load_events(event_files[0] if len(event_files) == 1 else None)
        recording_duration_sec: float | None = None
        json_files = sorted((subject_dir / "eeg").glob("*_task-resting_eeg.json"))
        if len(json_files) == 1:
            try:
                sidecar = json.loads(json_files[0].read_text(encoding="utf-8-sig"))
                recording_duration_sec = float(sidecar["RecordingDuration"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                recording_duration_sec = None
        blocks, status, anomalies = build_blocks(
            events,
            subject_dir.name,
            standard,
            overrides.get(subject_dir.name),
            thresholds,
            evidence.get(subject_dir.name, ""),
            recording_duration_sec,
        )
        if load_status and events is None:
            status = load_status
        all_blocks.extend(blocks)
        all_anomalies.extend(anomalies)
        summaries.append(
            summarize_subject(subject_dir.name, status, blocks, anomalies, subject_dir.name in overrides)
        )

    block_df = pd.DataFrame(all_blocks)
    summary_df = pd.DataFrame(summaries)
    anomaly_df = pd.DataFrame(all_anomalies, columns=ANOMALY_COLUMNS)
    block_df.to_csv(out / "lemon_block_inventory.csv", index=False)
    summary_df.to_csv(out / "lemon_subject_block_summary.csv", index=False)
    anomaly_df.to_csv(out / "lemon_event_anomalies.csv", index=False)

    print("=== MPI-LEMON EVENT/BLOCK AUDIT ===")
    print(f"Subjects discovered: {len(subject_dirs)}")
    print("Parser status:")
    print(summary_df["parser_status"].value_counts(dropna=False).to_string())
    print("Study block counts (EO, EC):")
    print(summary_df.groupby(["n_EO_blocks", "n_EC_blocks"]).size().to_string())
    print("Complete block counts (EO, EC):")
    print(summary_df.groupby(["n_complete_EO_blocks", "n_complete_EC_blocks"]).size().to_string())
    print(f"Anomaly evidence rows: {len(anomaly_df)}")


if __name__ == "__main__":
    main()
