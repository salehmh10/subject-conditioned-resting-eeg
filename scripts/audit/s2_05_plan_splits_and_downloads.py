"""Create a provisional block allocation and exact future download manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ordered_subjects(subject_ids: list[str], seed: str) -> list[str]:
    return sorted(subject_ids, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())


def assign_split(ordinal: int, rotation: int, state: str, config: dict[str, Any]) -> str:
    roles = config["base_roles"]
    offset = int(config["state_offsets"][state])
    return str(roles[((ordinal - 1) + rotation + offset) % len(roles)])


def build_split_manifest(
    blocks: pd.DataFrame, cohort: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    eligible_ids = cohort.loc[cohort["structurally_eligible"], "subject_id"].tolist()
    subject_order = ordered_subjects(eligible_ids, str(config["subject_order_seed"]))
    rotations = {subject_id: rank % 8 for rank, subject_id in enumerate(subject_order)}
    selected = blocks.loc[
        blocks["subject_id"].isin(eligible_ids) & blocks["is_study_block"] & blocks["is_complete"]
    ].copy()
    selected["state_block_index"] = selected["state_block_index"].astype(int)
    selected["split"] = selected.apply(
        lambda row: assign_split(
            int(row["state_block_index"]), rotations[row["subject_id"]], row["state"], config
        ),
        axis=1,
    )
    selected["split_status"] = config["status"]
    selected["algorithm_version"] = config["algorithm_version"]
    selected["subject_rotation"] = selected["subject_id"].map(rotations)
    selected["post_download_qc_required"] = True
    selected["analytical_access_before_final_lock"] = selected["split"].map(
        {"Train": "pilot_method_development_allowed", "Validation": "do_not_inspect", "Test": "do_not_inspect"}
    )
    columns = [
        "subject_id",
        "block_index",
        "acquisition_segment_id",
        "state",
        "state_block_index",
        "start_sec",
        "end_estimate_capped_sec",
        "n_markers",
        "split",
        "split_status",
        "algorithm_version",
        "subject_rotation",
        "post_download_qc_required",
        "analytical_access_before_final_lock",
    ]
    result = selected[columns].sort_values(
        ["subject_id", "state", "state_block_index"], kind="stable"
    )
    validate_split(result, config)
    return result


def validate_split(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    required = config["requirements"]
    expected = {
        "Train": int(required["train_blocks"]),
        "Validation": int(required["validation_blocks"]),
        "Test": int(required["test_blocks"]),
    }
    if frame.duplicated(["subject_id", "state", "state_block_index"]).any():
        raise ValueError("duplicate subject/state/block assignment")
    for (subject_id, state), group in frame.groupby(["subject_id", "state"]):
        counts = group["split"].value_counts().to_dict()
        if counts != expected:
            raise ValueError(f"invalid allocation for {subject_id}/{state}: {counts}")
        ordinals = sorted(group["state_block_index"].tolist())
        if ordinals != list(range(1, int(required["blocks_per_subject_state"]) + 1)):
            raise ValueError(f"invalid block ordinals for {subject_id}/{state}: {ordinals}")


def pilot_subject_reasons(
    metadata: pd.DataFrame, cohort: pd.DataFrame, config: dict[str, Any]
) -> dict[str, str]:
    eligible = cohort.loc[cohort["structurally_eligible"]].copy()
    if "sampling_frequency" not in eligible.columns:
        eligible = eligible.merge(
            metadata[["subject_id", "sampling_frequency"]], on="subject_id", validate="one_to_one"
        )
    reasons: dict[str, str] = {}
    for _, row in eligible.loc[eligible["sampling_frequency"] != 2500].sort_values("subject_id").iterrows():
        reasons[row["subject_id"]] = f"rare native sampling rate: {float(row['sampling_frequency']):g} Hz"
    for subject_id in eligible.loc[eligible["has_event_override"], "subject_id"]:
        reasons.setdefault(subject_id, "documented subject-specific S208-to-EC override")
    for subject_id in eligible.loc[eligible["n_interruption_fragments"] > 0, "subject_id"]:
        reasons.setdefault(subject_id, "explicit acquisition restart with excluded pre-sequence fragment")

    ordinary = eligible.loc[
        (eligible["sampling_frequency"] == 2500)
        & ~eligible["subject_id"].isin(reasons)
        & ~eligible["has_event_override"]
        & (eligible["n_interruption_fragments"] == 0),
        "subject_id",
    ].tolist()
    ordinary_order = ordered_subjects(ordinary, str(config["pilot"]["algorithm_version"]))
    for subject_id in ordinary_order[: int(config["pilot"]["ordinary_2500_hz_slots"])]:
        reasons[subject_id] = "deterministic ordinary 2500 Hz baseline"
    if len(reasons) != int(config["pilot"]["n_subjects"]):
        raise ValueError(f"pilot selection produced {len(reasons)} rather than {config['pilot']['n_subjects']} subjects")
    return dict(sorted(reasons.items()))


def asset_rows(
    metadata: pd.DataFrame,
    subject_reasons: dict[str, str],
    base_url: str,
    manifest_scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indexed = metadata.set_index("subject_id")
    for subject_id, reason in sorted(subject_reasons.items()):
        row = indexed.loc[subject_id]
        for extension, role, key_column, size_column, hash_column in [
            ("eeg", "waveform", "eeg_pointer_key", "eeg_payload_size_bytes", "eeg_payload_sha256"),
            ("vhdr", "brainvision_header", "vhdr_pointer_key", "vhdr_payload_size_bytes", "vhdr_payload_sha256"),
        ]:
            key = str(row[key_column])
            if not key.startswith("SHA256E-s"):
                raise ValueError(f"invalid {extension} pointer for {subject_id}: {key}")
            relative = f"{subject_id}/eeg/{subject_id}_task-resting_eeg.{extension}"
            rows.append(
                {
                    "manifest_scope": manifest_scope,
                    "subject_id": subject_id,
                    "selection_reason": reason,
                    "file_role": role,
                    "relative_path": relative,
                    "annex_key": key,
                    "source_url": f"{base_url.rstrip('/')}/{key}",
                    "payload_size_bytes": int(row[size_column]),
                    "payload_sha256": str(row[hash_column]),
                    "currently_materialized": False,
                    "future_destination_root": "data_raw/MPI-LEMON",
                    "requires_size_and_sha256_verification": True,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/audit/provisional_split.yaml")
    parser.add_argument("--input-dir", default="outputs/metrics/scenario2")
    parser.add_argument("--output-dir", default="outputs/metrics/scenario2")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    source = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(source / "lemon_metadata_inventory.csv")
    cohort = pd.read_csv(source / "lemon_structural_cohort_manifest.csv")
    blocks = pd.read_csv(source / "lemon_block_inventory.csv")

    split = build_split_manifest(blocks, cohort, config)
    split.to_csv(out / "provisional_block_split_manifest.csv", index=False)

    pilot_reasons = pilot_subject_reasons(metadata, cohort, config)
    eligible_reasons = {
        subject_id: "pre-download structurally eligible full candidate"
        for subject_id in cohort.loc[cohort["structurally_eligible"], "subject_id"]
    }
    base_url = str(config["backing_store"]["base_url"])
    pilot = pd.DataFrame(asset_rows(metadata, pilot_reasons, base_url, "pilot"))
    full = pd.DataFrame(asset_rows(metadata, eligible_reasons, base_url, "full_candidate"))
    pilot.to_csv(out / "pilot_download_manifest.csv", index=False)
    full.to_csv(out / "full_candidate_download_manifest.csv", index=False)

    order_balance = (
        split.groupby(["state", "state_block_index", "split"]).size().rename("n_blocks").reset_index()
    )
    summary = {
        "split_status": config["status"],
        "algorithm_version": config["algorithm_version"],
        "n_eligible_subjects": split["subject_id"].nunique(),
        "allocation_per_subject_state": {"Train": 5, "Validation": 1, "Test": 2},
        "five_one_two_structurally_feasible": True,
        "n_provisional_block_assignments": len(split),
        "order_balance": order_balance.to_dict(orient="records"),
        "final_lock_blockers": [
            "waveform integrity and BrainVision linkage",
            "prespecified channel/signal/artifact QC",
            "usable duration and window counts after preprocessing",
            "outcome-blind final cohort and block eligibility",
            "final manifest checksum and commit",
        ],
        "pilot_subjects": pilot_reasons,
        "pilot_n_subjects": len(pilot_reasons),
        "pilot_n_assets": len(pilot),
        "pilot_payload_bytes": int(pilot["payload_size_bytes"].sum()),
        "full_n_subjects": full["subject_id"].nunique(),
        "full_n_assets": len(full),
        "full_payload_bytes": int(full["payload_size_bytes"].sum()),
        "commands_not_run": [
            "python scripts/data/s2_06_download_manifest.py --manifest outputs/metrics/scenario2/pilot_download_manifest.csv --destination data_raw/MPI-LEMON --metadata-root D:/SHARIF/PROJECT/Datasets/MPI-LEMON --execute",
            "python scripts/data/s2_06_download_manifest.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --destination data_raw/MPI-LEMON --metadata-root D:/SHARIF/PROJECT/Datasets/MPI-LEMON --execute",
        ],
    }
    (out / "lemon_split_and_download_plan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("=== PROVISIONAL SPLIT AND DOWNLOAD PLANS ===")
    print(f"5/1/2 structurally feasible: {summary['five_one_two_structurally_feasible']}")
    print(f"Provisional assignments: {len(split)}")
    print(f"Pilot: {summary['pilot_n_subjects']} subjects, {summary['pilot_n_assets']} assets, {summary['pilot_payload_bytes']} bytes")
    print(f"Full: {summary['full_n_subjects']} subjects, {summary['full_n_assets']} assets, {summary['full_payload_bytes']} bytes")
    print("No download command was executed.")


if __name__ == "__main__":
    main()
