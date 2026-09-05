"""Reverify only the staged 15-subject pilot using opaque hashes and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from resting_eeg.data.integrity import TARGET_CHANNELS, audit_pilot_inventory, audit_subject


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/metrics/scenario2/pilot_download_manifest.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data_raw/MPI-LEMON"))
    parser.add_argument("--blocks", type=Path, default=Path("outputs/metrics/scenario2/lemon_block_inventory.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics/scenario2"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    blocks = pd.read_csv(args.blocks)
    summary = audit_pilot_inventory(args.raw_root, manifest)
    if not summary["inventory_pass"]:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "pilot_loader_integrity_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print("Pilot inventory failed; refusing to open recordings outside the exact pilot inventory.", flush=True)
        return 1
    rows = []
    for subject_id in sorted(manifest.subject_id.unique()):
        row = audit_subject(subject_id, manifest, args.raw_root, blocks, TARGET_CHANNELS)
        rows.append(row)
        print(f"{subject_id}: integrity_pass={row['integrity_pass']} {row['failure_reason']}", flush=True)
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "pilot_loader_integrity.csv", index=False)
    summary.update(
        n_integrity_pass=int(frame.integrity_pass.sum()),
        n_integrity_fail=int((~frame.integrity_pass).sum()),
        all_integrity_pass=bool(frame.integrity_pass.all()),
        native_sfreq_counts={str(key): int(value) for key, value in frame.get("native_sfreq", pd.Series(dtype=float)).value_counts().sort_index().items()},
        total_waveform_value_accesses=0,
        source_manifest=str(args.manifest), authoritative_block_inventory=str(args.blocks),
        duration_convention="n_times/sfreq is exclusive end; metadata may describe last sample within one native sample",
        montage_limitations="standard_1005 template positions; individual measured electrode coordinates unavailable",
        warnings_by_subject={row["subject_id"]: row["warnings"] for row in rows if row["warnings"]},
    )
    (args.output_dir / "pilot_loader_integrity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["all_integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
