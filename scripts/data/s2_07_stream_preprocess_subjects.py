"""Prepare or execute one-subject-at-a-time provisional-Train preprocessing.

The default is a read-only dry run. Future network staging requires --execute;
raw deletion additionally requires --delete-raw-after-success. No held-out split
can be selected. This phase must not execute downloads of non-pilot subjects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for source_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import pandas as pd

from resting_eeg.data.streaming import stream_subjects
from scripts.data.s2_06_download_manifest import filter_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "outputs/metrics/scenario2/full_candidate_download_manifest.csv")
    parser.add_argument("--destination", "--raw-root", dest="raw_root", type=Path, default=REPO_ROOT / "data_raw/MPI-LEMON")
    parser.add_argument("--metadata-root", type=Path, default=Path("D:/SHARIF/PROJECT/Datasets/MPI-LEMON"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/preprocessing/frozen_preprocessing_v1.yaml")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data_processed/MPI-LEMON")
    parser.add_argument("--metrics-root", type=Path, default=REPO_ROOT / "outputs/metrics/scenario2")
    parser.add_argument("--journal", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    retention = parser.add_mutually_exclusive_group()
    retention.add_argument("--keep-raw", action="store_true", help="retain raw data (the default)")
    retention.add_argument("--delete-raw-after-success", action="store_true")
    parser.add_argument("--subject-id", action="append", default=[])
    parser.add_argument("--subjects-file", type=Path)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--local-existing-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.retries < 0 or args.timeout <= 0:
            raise ValueError("retries must be nonnegative and timeout must be positive")
        frame = filter_manifest(pd.read_csv(args.manifest), subject_ids=args.subject_id,
                                subjects_file=args.subjects_file, max_subjects=args.max_subjects)
        results = stream_subjects(
            frame, repo_root=REPO_ROOT, raw_root=args.raw_root, metadata_root=args.metadata_root,
            output_root=args.output_root, config_path=args.config, metrics_root=args.metrics_root,
            execute=args.execute, local_existing_only=args.local_existing_only,
            delete_raw_after_success=args.delete_raw_after_success,
            continue_on_error=args.continue_on_error, retries=args.retries, timeout=args.timeout,
            journal_path=args.journal,
        )
        for result in results:
            print(json.dumps(result, sort_keys=True))
        if not args.execute:
            print("DRY RUN: no downloads, waveform access, journal writes, or raw deletion. Scope: provisional Train only.")
        return int(any(result["status"] not in {"planned", "complete"} for result in results))
    except Exception as exc:
        print(f"Streaming failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
