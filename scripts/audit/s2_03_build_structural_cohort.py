"""Build the pre-download structural cohort from reproducible audit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def classify(row: pd.Series, criteria: dict[str, Any]) -> tuple[bool, str, str]:
    if row["parser_status"] == "events_file_missing_or_ambiguous":
        return False, "state_annotations_unavailable_missing_events", "events.tsv is absent; vmrk has zero marker entries"
    if row["parser_status"] == "no_state_markers":
        return False, "state_annotations_unavailable_no_state_markers", "events.tsv contains no mapped EO/EC trial_type markers"
    if not bool(row["channels_readable"]):
        return False, "channels_metadata_unreadable", str(row.get("channels_error", ""))
    if criteria["require_all_19_target_channels"] and not bool(row["target_19_present"]):
        return False, "required_target_channels_missing", str(row.get("missing_target_channels", ""))
    required_runs = int(criteria["required_study_blocks_per_state"])
    required_complete = int(criteria["required_complete_blocks_per_state"])
    if (
        int(row["n_EO_blocks"]) < required_runs
        or int(row["n_EC_blocks"]) < required_runs
        or int(row["n_complete_EO_blocks"]) < required_complete
        or int(row["n_complete_EC_blocks"]) < required_complete
    ):
        detail = (
            f"study_runs_EO={int(row['n_EO_blocks'])};study_runs_EC={int(row['n_EC_blocks'])};"
            f"complete_EO={int(row['n_complete_EO_blocks'])};complete_EC={int(row['n_complete_EC_blocks'])}"
        )
        return False, "insufficient_complete_blocks_for_5_1_2", detail
    if criteria["require_alternating_study_blocks"] and not bool(row["alternating"]):
        return False, "nonalternating_state_structure", "retained study runs do not alternate EO/EC"
    return True, "meets_pre_download_structural_criteria", ""


def subject_notes(row: pd.Series, eligible: bool) -> str:
    notes: list[str] = []
    if int(row.get("n_interruption_fragments", 0)):
        notes.append(
            f"{int(row['n_interruption_fragments'])} pre-sequence acquisition-interruption fragment(s) excluded from study blocks"
        )
    if bool(row.get("has_event_override", False)):
        notes.append("documented subject-specific S208-to-EC compatibility override")
    if bool(row.get("has_timing_uncertainty", False)):
        notes.append("unlabeled transition/interruption interval(s) preserved; no state imputation")
    if float(row.get("sampling_frequency", 2500)) != 2500:
        notes.append(f"native sampling rate {float(row['sampling_frequency']):g} Hz; resampling must be validated")
    if eligible:
        notes.append("waveform integrity, signal QC, bad channels, and artifacts remain unassessed")
    return "; ".join(notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/audit/cohort_eligibility.yaml")
    parser.add_argument("--input-dir", default="outputs/metrics/scenario2")
    parser.add_argument("--output-dir", default="outputs/metrics/scenario2")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    criteria = config["criteria"]
    source = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(source / "lemon_metadata_inventory.csv")
    blocks = pd.read_csv(source / "lemon_subject_block_summary.csv")
    merged = metadata.merge(blocks, on="subject_id", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for _, row in merged.sort_values("subject_id").iterrows():
        eligible, primary, secondary = classify(row, criteria)
        complete_structure = bool(
            int(row["n_EO_blocks"]) == int(criteria["required_study_blocks_per_state"])
            and int(row["n_EC_blocks"]) == int(criteria["required_study_blocks_per_state"])
            and int(row["n_complete_EO_blocks"]) == int(criteria["required_complete_blocks_per_state"])
            and int(row["n_complete_EC_blocks"]) == int(criteria["required_complete_blocks_per_state"])
            and bool(row["alternating"])
        )
        rows.append(
            {
                "subject_id": row["subject_id"],
                "structurally_eligible": eligible,
                "cohort_status": config["cohort_status"],
                "primary_reason": primary,
                "secondary_reason": secondary,
                "n_EO_blocks": int(row["n_EO_blocks"]),
                "n_EC_blocks": int(row["n_EC_blocks"]),
                "n_complete_EO_blocks": int(row["n_complete_EO_blocks"]),
                "n_complete_EC_blocks": int(row["n_complete_EC_blocks"]),
                "n_interruption_fragments": int(row["n_interruption_fragments"]),
                "block_structure_complete": complete_structure,
                "state_annotations_available": row["parser_status"] == "ok",
                "all_19_channels_present": bool(row["target_19_present"]),
                "sampling_frequency": float(row["sampling_frequency"]),
                "has_event_override": bool(row["has_event_override"]),
                "has_timing_uncertainty": bool(row["has_timing_uncertainty"]),
                "requires_waveform_qc": eligible,
                "notes": subject_notes(row, eligible),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "lemon_structural_cohort_manifest.csv", index=False)
    summary = {
        "cohort_status": config["cohort_status"],
        "n_full_metadata_cohort": len(manifest),
        "n_structurally_eligible": int(manifest["structurally_eligible"].sum()),
        "n_structurally_excluded": int((~manifest["structurally_eligible"]).sum()),
        "exclusion_reason_counts": {
            str(key): int(value)
            for key, value in manifest.loc[~manifest["structurally_eligible"], "primary_reason"].value_counts().items()
        },
        "excluded_subjects": manifest.loc[
            ~manifest["structurally_eligible"], ["subject_id", "primary_reason", "secondary_reason"]
        ].to_dict(orient="records"),
        "all_retained_require_waveform_qc": bool(
            manifest.loc[manifest["structurally_eligible"], "requires_waveform_qc"].all()
        ),
    }
    (out / "lemon_structural_cohort_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("=== PRE-DOWNLOAD STRUCTURAL COHORT ===")
    print(f"Full metadata cohort: {summary['n_full_metadata_cohort']}")
    print(f"Structurally eligible: {summary['n_structurally_eligible']}")
    print(f"Structurally excluded: {summary['n_structurally_excluded']}")
    print(manifest.loc[~manifest["structurally_eligible"], ["subject_id", "primary_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
