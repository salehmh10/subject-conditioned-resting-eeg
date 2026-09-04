"""Metadata-only MPI-LEMON inventory.

This module deliberately never opens a materialized EEG waveform. A small
NEMAR pointer file may be inspected to recover its annex key and declared
payload size.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


POINTER_RE = re.compile(
    r"^/annex/objects/(?P<key>SHA256E-s(?P<size>\d+)--(?P<digest>[0-9a-f]+)\.(?P<ext>[^/]+))$"
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern)) if folder.is_dir() else []


def only_file(files: list[Path]) -> Path | None:
    return files[0] if len(files) == 1 else None


def read_tsv(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        return None, "missing_or_ambiguous"
    try:
        return pd.read_csv(path, sep="\t"), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "missing_or_ambiguous"
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            return None, "top_level_json_is_not_an_object"
        return value, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def inspect_small_pointer(path: Path | None) -> dict[str, Any]:
    result = {
        "is_pointer": False,
        "pointer_key": "",
        "payload_size_bytes": pd.NA,
        "payload_sha256": "",
        "payload_present": False,
    }
    if path is None or not path.is_file():
        return result
    # A real .eeg can be hundreds of MB. Never open it in a metadata audit.
    if path.stat().st_size > 512:
        result["payload_present"] = True
        return result
    try:
        line = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError):
        return result
    match = POINTER_RE.fullmatch(line)
    if match:
        result.update(
            is_pointer=True,
            pointer_key=match.group("key"),
            payload_size_bytes=int(match.group("size")),
            payload_sha256=match.group("digest"),
        )
    return result


def count_vmrk_entries(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return sum(bool(re.match(r"^Mk\d+=", line)) for line in handle)
    except OSError:
        return None


def channel_type_counts(channels: pd.DataFrame | None) -> Counter[str]:
    if channels is None or "type" not in channels.columns:
        return Counter()
    return Counter(channels["type"].fillna("MISSING").astype(str).str.strip().str.upper())


def status_summary(channels: pd.DataFrame | None, target_channels: list[str]) -> str:
    if channels is None or not {"name", "status"}.issubset(channels.columns):
        return "unavailable"
    names = channels["name"].astype(str).str.strip()
    selected = channels.loc[names.isin(target_channels), "status"]
    counts = selected.fillna("missing").astype(str).str.strip().str.lower().value_counts()
    return ";".join(f"{key}:{int(value)}" for key, value in sorted(counts.items()))


def audit_subject(
    subject_dir: Path, target_channels: list[str]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    subject_id = subject_dir.name
    eeg_dir = subject_dir / "eeg"

    channels_files = find_files(eeg_dir, "*_task-resting_channels.tsv")
    events_files = find_files(eeg_dir, "*_task-resting_events.tsv")
    eeg_json_files = find_files(eeg_dir, "*_task-resting_eeg.json")
    vhdr_files = find_files(eeg_dir, "*_task-resting_eeg.vhdr")
    vmrk_files = find_files(eeg_dir, "*_task-resting_eeg.vmrk")
    eeg_files = find_files(eeg_dir, "*_task-resting_eeg.eeg")

    channels_path = only_file(channels_files)
    events_path = only_file(events_files)
    eeg_json_path = only_file(eeg_json_files)
    vhdr_path = only_file(vhdr_files)
    vmrk_path = only_file(vmrk_files)
    eeg_path = only_file(eeg_files)

    channels, channels_error = read_tsv(channels_path)
    events, events_error = read_tsv(events_path)
    eeg_json, eeg_json_error = read_json(eeg_json_path)

    channel_names: list[str] = []
    if channels is not None and "name" in channels.columns:
        channel_names = channels["name"].dropna().astype(str).str.strip().tolist()
    channel_set = set(channel_names)
    channel_name_counts = Counter(channel_names)
    missing_target = [name for name in target_channels if name not in channel_set]
    duplicate_target = [name for name in target_channels if channel_name_counts[name] > 1]
    type_counts = channel_type_counts(channels)
    total_channels = len(channels) if channels is not None else pd.NA
    typed_channels = sum(type_counts.values())
    misc_channels = typed_channels - type_counts.get("EEG", 0) - type_counts.get("EOG", 0)

    event_columns = list(events.columns) if events is not None else []
    trial_counts: Counter[str] = Counter()
    event_value_rows: list[dict[str, Any]] = []
    if events is not None and "trial_type" in events.columns:
        trial_counts = Counter(events["trial_type"].fillna("<missing>").astype(str))
        for value, count in sorted(trial_counts.items()):
            event_value_rows.append(
                {"subject_id": subject_id, "column": "trial_type", "value": value, "count": count}
            )

    eeg_pointer = inspect_small_pointer(eeg_path)
    vhdr_pointer = inspect_small_pointer(vhdr_path)
    sampling_frequency = eeg_json.get("SamplingFrequency") if eeg_json else pd.NA

    inventory = {
        "subject_id": subject_id,
        "has_eeg_dir": eeg_dir.is_dir(),
        "has_channels_tsv": len(channels_files) == 1,
        "has_events_tsv": len(events_files) == 1,
        "has_eeg_json": len(eeg_json_files) == 1,
        "has_vhdr": len(vhdr_files) == 1,
        "has_vmrk": len(vmrk_files) == 1,
        "has_eeg_file_entry": len(eeg_files) == 1,
        "has_eeg_pointer": bool(eeg_pointer["is_pointer"]),
        "eeg_payload_present": bool(eeg_pointer["payload_present"]),
        "eeg_pointer_key": eeg_pointer["pointer_key"],
        "eeg_payload_size_bytes": eeg_pointer["payload_size_bytes"],
        "eeg_payload_sha256": eeg_pointer["payload_sha256"],
        "vhdr_is_pointer": bool(vhdr_pointer["is_pointer"]),
        "vhdr_pointer_key": vhdr_pointer["pointer_key"],
        "vhdr_payload_size_bytes": vhdr_pointer["payload_size_bytes"],
        "vhdr_payload_sha256": vhdr_pointer["payload_sha256"],
        "n_vmrk_entries": count_vmrk_entries(vmrk_path),
        "channels_readable": channels is not None,
        "channels_error": channels_error,
        "n_total_channels": total_channels,
        "n_eeg_channels": type_counts.get("EEG", 0) if channels is not None else pd.NA,
        "n_eog_channels": type_counts.get("EOG", 0) if channels is not None else pd.NA,
        "n_misc_channels": misc_channels if channels is not None else pd.NA,
        "channel_type_summary": ";".join(f"{key}:{value}" for key, value in sorted(type_counts.items())),
        "target_19_present": bool(channel_names) and not missing_target and not duplicate_target,
        "missing_target_channels": ";".join(missing_target),
        "duplicate_target_channels": ";".join(duplicate_target),
        "target_channel_status_summary": status_summary(channels, target_channels),
        "sampling_frequency": sampling_frequency,
        "eeg_reference": eeg_json.get("EEGReference", "") if eeg_json else "",
        "power_line_frequency": eeg_json.get("PowerLineFrequency", pd.NA) if eeg_json else pd.NA,
        "eeg_json_readable": eeg_json is not None,
        "eeg_json_error": eeg_json_error,
        "events_readable": events is not None,
        "events_error": events_error,
        "event_schema": ";".join(event_columns),
        "event_schema_has_required": {"onset", "trial_type"}.issubset(event_columns),
        "n_events": len(events) if events is not None else pd.NA,
        "n_trial_type_s200": trial_counts.get("Stimulus/S200", 0),
        "n_trial_type_s210": trial_counts.get("Stimulus/S210", 0),
        "n_trial_type_s208": trial_counts.get("Stimulus/S208", 0),
        "n_other_events": (
            len(events)
            - trial_counts.get("Stimulus/S200", 0)
            - trial_counts.get("Stimulus/S210", 0)
            - trial_counts.get("Stimulus/S208", 0)
            if events is not None
            else pd.NA
        ),
    }
    schema = {
        "subject_id": subject_id,
        "events_readable": events is not None,
        "columns": ";".join(event_columns),
        "n_columns": len(event_columns),
        "has_onset": "onset" in event_columns,
        "has_duration": "duration" in event_columns,
        "has_trial_type": "trial_type" in event_columns,
        "has_value": "value" in event_columns,
        "has_sample": "sample" in event_columns,
        "error": events_error,
    }
    return inventory, schema, event_value_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/audit/metadata_audit.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    root = Path(args.dataset_root or config["dataset_root"])
    out = Path(args.output_dir or config["output_dir"])
    target_channels = [str(value) for value in config["target_channels"]]
    out.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(path for path in root.glob("sub-*") if path.is_dir())
    inventories: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    event_values: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        inventory, schema, values = audit_subject(subject_dir, target_channels)
        inventories.append(inventory)
        schemas.append(schema)
        event_values.extend(values)

    inventory_df = pd.DataFrame(inventories)
    schema_df = pd.DataFrame(schemas)
    values_df = pd.DataFrame(event_values, columns=["subject_id", "column", "value", "count"])
    inventory_df.to_csv(out / "lemon_metadata_inventory.csv", index=False)
    schema_df.to_csv(out / "lemon_event_schema.csv", index=False)
    values_df.to_csv(out / "lemon_event_values.csv", index=False)

    summary = {
        "audit_scope": "metadata_only_no_waveform_read",
        "dataset_root": str(root.resolve()),
        "n_subjects": len(inventory_df),
        "n_channels_readable": int(inventory_df["channels_readable"].sum()),
        "n_events_readable": int(inventory_df["events_readable"].sum()),
        "n_target_19_present": int(inventory_df["target_19_present"].sum()),
        "channel_count_combinations": (
            inventory_df.groupby(
                ["n_total_channels", "n_eeg_channels", "n_eog_channels", "n_misc_channels"], dropna=False
            )
            .size()
            .rename("n_subjects")
            .reset_index()
            .to_dict(orient="records")
        ),
        "sampling_frequency_counts": {
            str(key): int(value)
            for key, value in inventory_df["sampling_frequency"].value_counts(dropna=False).sort_index().items()
        },
        "eeg_pointer_payload_total_bytes": int(inventory_df["eeg_payload_size_bytes"].sum()),
        "n_materialized_eeg_payloads": int(inventory_df["eeg_payload_present"].sum()),
    }
    (out / "lemon_metadata_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("=== MPI-LEMON METADATA-ONLY AUDIT ===")
    print(f"Subjects discovered: {summary['n_subjects']}")
    print(f"Readable channels.tsv: {summary['n_channels_readable']}")
    print(f"Readable events.tsv: {summary['n_events_readable']}")
    print(f"All 19 targets present: {summary['n_target_19_present']}")
    print("Channel combinations:")
    print(pd.DataFrame(summary["channel_count_combinations"]).to_string(index=False))
    print("Sampling frequencies:")
    print(inventory_df["sampling_frequency"].value_counts(dropna=False).sort_index().to_string())
    print(f"Materialized EEG payloads: {summary['n_materialized_eeg_payloads']} (must remain 0 pre-download)")


if __name__ == "__main__":
    main()
