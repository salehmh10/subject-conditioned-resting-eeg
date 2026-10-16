from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

SCENARIO2_DATA_ROOT = Path(
    r"D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2"
)

FINAL_COHORT_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "final_cohort_manifest.csv"
)

FINAL_SPLIT_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "final_block_split_manifest.csv"
)

NORMALIZATION_FILE = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "train_normalization.json"
)

ALL_BLOCKS_ROOT = (
    SCENARIO2_DATA_ROOT
    / "processed"
    / "all-blocks-v1"
)


def print_hdf5_tree(group: h5py.Group, prefix: str = "") -> None:
    """Recursively print HDF5 groups/datasets without reading EEG arrays."""
    for key in group.keys():
        obj = group[key]
        path = f"{prefix}/{key}"

        if isinstance(obj, h5py.Dataset):
            print(
                f"DATASET {path} | "
                f"shape={obj.shape} | "
                f"dtype={obj.dtype}"
            )
        elif isinstance(obj, h5py.Group):
            print(f"GROUP   {path}")
            print_hdf5_tree(obj, path)


def print_attrs(obj, title: str) -> None:
    print(f"\n--- {title} ATTRIBUTES ---")

    if not obj.attrs:
        print("(none)")
        return

    for key in sorted(obj.attrs.keys()):
        value = obj.attrs[key]

        # Avoid ugly giant output.
        if hasattr(value, "shape") and getattr(value, "size", 1) > 30:
            print(
                f"{key}: <array shape={value.shape} dtype={value.dtype}>"
            )
        else:
            print(f"{key}: {value}")


def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — REAL BASELINE INPUT INSPECTION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Load final cohort
    # ------------------------------------------------------------------

    cohort = pd.read_csv(FINAL_COHORT_MANIFEST)

    print("\n[1] FINAL COHORT MANIFEST")
    print(f"path: {FINAL_COHORT_MANIFEST}")
    print(f"rows: {len(cohort)}")
    print(f"columns: {list(cohort.columns)}")

    inclusion_candidates = [
        "final_cohort_included",
        "included",
        "final_included",
    ]

    inclusion_column = next(
        (c for c in inclusion_candidates if c in cohort.columns),
        None,
    )

    if inclusion_column is None:
        raise RuntimeError(
            "Could not identify final cohort inclusion column."
        )

    included = cohort[
        cohort[inclusion_column].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    ].copy()

    print(f"inclusion column: {inclusion_column}")
    print(f"included subjects: {len(included)}")

    if len(included) != 198:
        raise RuntimeError(
            f"Expected 198 included subjects, found {len(included)}."
        )

    subject_column = "subject_id"

    if subject_column not in included.columns:
        raise RuntimeError("subject_id column not found.")

    example_subject = sorted(
        included[subject_column].astype(str).tolist()
    )[0]

    print(f"example included subject: {example_subject}")

    # ------------------------------------------------------------------
    # Step 2: Inspect final split rows for this subject
    # ------------------------------------------------------------------

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)

    subject_split = split[
        split["subject_id"].astype(str) == example_subject
    ].copy()

    print("\n[2] FINAL SPLIT MANIFEST")
    print(f"path: {FINAL_SPLIT_MANIFEST}")
    print(f"rows total: {len(split)}")
    print(f"columns: {list(split.columns)}")
    print(f"rows for {example_subject}: {len(subject_split)}")

    print("\nExample subject split rows:")
    print(subject_split.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 3: Inspect frozen normalization artifact
    # ------------------------------------------------------------------

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        normalization = json.load(f)

    print("\n[3] TRAIN NORMALIZATION")
    print(f"path: {NORMALIZATION_FILE}")
    print(f"top-level keys: {sorted(normalization.keys())}")

    # Print only compact scalar/list information.
    for key in [
        "status",
        "ddof",
        "included_subject_count",
        "assigned_train_block_count",
        "eligible_train_block_count",
        "unique_sample_count",
        "unique_duration_seconds",
        "unique_duration_hours",
    ]:
        if key in normalization:
            print(f"{key}: {normalization[key]}")

    for key in [
        "channel_order",
        "channels",
        "means_uv",
        "stds_uv",
        "mean_uv",
        "std_uv",
    ]:
        if key in normalization:
            print(f"{key}: {normalization[key]}")

    # ------------------------------------------------------------------
    # Step 4: Locate example subject HDF5
    # ------------------------------------------------------------------

    expected_candidates = [
        ALL_BLOCKS_ROOT / f"{example_subject}_all_blocks.h5",
        ALL_BLOCKS_ROOT / f"{example_subject}.h5",
        ALL_BLOCKS_ROOT / f"{example_subject}_preprocessed.h5",
    ]

    h5_path = next(
        (p for p in expected_candidates if p.exists()),
        None,
    )

    if h5_path is None:
        matches = sorted(
            ALL_BLOCKS_ROOT.glob(f"*{example_subject}*.h5")
        )

        if len(matches) == 1:
            h5_path = matches[0]
        elif len(matches) > 1:
            print("\nPossible HDF5 matches:")
            for p in matches:
                print(p)

            raise RuntimeError(
                "Multiple HDF5 files match subject."
            )
        else:
            raise FileNotFoundError(
                f"No HDF5 found for {example_subject}"
            )

    print("\n[4] EXAMPLE HDF5")
    print(f"path: {h5_path}")
    print(f"size bytes: {h5_path.stat().st_size}")

    # ------------------------------------------------------------------
    # Step 5: Print HDF5 schema WITHOUT loading EEG samples
    # ------------------------------------------------------------------

    with h5py.File(h5_path, "r") as h5:
        print_attrs(h5, "ROOT")

        print("\n--- HDF5 TREE ---")
        print_hdf5_tree(h5)

        print("\n--- TOP-LEVEL OBJECT ATTRIBUTES ---")

        for key in h5.keys():
            obj = h5[key]
            print_attrs(obj, key)

            # Only inspect one additional level of attributes.
            if isinstance(obj, h5py.Group):
                for child_key in obj.keys():
                    child = obj[child_key]
                    if isinstance(child, h5py.Group):
                        print_attrs(
                            child,
                            f"{key}/{child_key}",
                        )

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("No EEG dataset values were loaded.")
    print("=" * 80)


if __name__ == "__main__":
    main()