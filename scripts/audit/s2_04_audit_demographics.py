"""Compare distributed participant metadata for full and structural cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MISSING_TOKENS = {"", "n/a", "na", "nan", "none", "null"}


def normalized_values(series: pd.Series) -> pd.Series:
    values = series.fillna("<missing>").astype(str).str.strip()
    return values.mask(values.str.lower().isin(MISSING_TOKENS), "<missing>")


def summarize_group(frame: pd.DataFrame, cohort_name: str, variables: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        values = normalized_values(frame[variable])
        n_missing = int(values.eq("<missing>").sum())
        n_nonmissing = len(values) - n_missing
        for level, count in values.value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "cohort": cohort_name,
                    "n_subjects": len(frame),
                    "variable": variable,
                    "level": level,
                    "count": int(count),
                    "percent_of_cohort": 100.0 * count / len(frame) if len(frame) else 0.0,
                    "n_nonmissing": n_nonmissing,
                    "n_missing": n_missing,
                    "percent_of_nonmissing": (
                        100.0 * count / n_nonmissing if n_nonmissing and level != "<missing>" else pd.NA
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="D:/SHARIF/PROJECT/Datasets/MPI-LEMON")
    parser.add_argument("--input-dir", default="outputs/metrics/scenario2")
    parser.add_argument("--output-dir", default="outputs/metrics/scenario2")
    args = parser.parse_args()

    participants = pd.read_csv(Path(args.dataset_root) / "participants.tsv", sep="\t", dtype=str)
    cohort = pd.read_csv(Path(args.input_dir) / "lemon_structural_cohort_manifest.csv")
    if participants["participant_id"].duplicated().any():
        raise ValueError("participants.tsv contains duplicate participant_id values")
    participant_ids = set(participants["participant_id"])
    cohort_ids = set(cohort["subject_id"])
    if participant_ids != cohort_ids:
        raise ValueError(
            f"participants/cohort ID mismatch: participants_only={sorted(participant_ids-cohort_ids)}, "
            f"cohort_only={sorted(cohort_ids-participant_ids)}"
        )

    merged = participants.merge(
        cohort[["subject_id", "structurally_eligible"]],
        left_on="participant_id",
        right_on="subject_id",
        validate="one_to_one",
    )
    variables = [column for column in participants.columns if column != "participant_id"]
    rows = summarize_group(merged, "full_metadata_cohort", variables)
    rows.extend(
        summarize_group(
            merged.loc[merged["structurally_eligible"]], "pre_download_structurally_eligible", variables
        )
    )
    summary_df = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out / "lemon_demographic_cohort_summary.csv", index=False)

    nonmissing = {
        variable: int(normalized_values(participants[variable]).ne("<missing>").sum()) for variable in variables
    }
    result = {
        "n_participant_rows": len(participants),
        "n_structurally_eligible": int(cohort["structurally_eligible"].sum()),
        "participant_id_sets_match": True,
        "available_variables": variables,
        "nonmissing_counts_full_cohort": nonmissing,
        "comparison_possible": any(nonmissing.values()),
        "interpretation": (
            "All distributed participant-level demographic fields are n/a; structural-exclusion distortion "
            "cannot be assessed from participants.tsv. Dataset README aggregates are not substituted because "
            "they are not linked to subject IDs and their stated age-group counts sum to 227, not 215."
        ),
    }
    (out / "lemon_demographic_cohort_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("=== DEMOGRAPHIC/METADATA COHORT AUDIT ===")
    print(f"Participant rows: {len(participants)}")
    print(f"ID sets match: {result['participant_id_sets_match']}")
    print("Nonmissing participant-level values:")
    print(pd.Series(nonmissing).to_string())
    print(f"Full-vs-eligible demographic comparison possible: {result['comparison_possible']}")


if __name__ == "__main__":
    main()
