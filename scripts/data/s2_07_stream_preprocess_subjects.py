"""Prepare or execute one-subject-at-a-time provisional-Train preprocessing.

The default is a read-only dry run. Future network staging requires --execute;
raw deletion additionally requires --delete-raw-after-success. Held-out access is
frozen mechanical apply-only. Batch execution requires immutable membership and
a durable authorization. Execution never plans or starts another batch.
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
from resting_eeg.data.paths import load_paths
from scripts.data.s2_06_download_manifest import filter_manifest


def build_parser() -> argparse.ArgumentParser:
    storage = load_paths(REPO_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "outputs/metrics/scenario2/full_candidate_download_manifest.csv")
    parser.add_argument("--destination", "--raw-root", dest="raw_root", type=Path, default=storage['raw_root'])
    parser.add_argument("--metadata-root", type=Path, default=storage['metadata_root'])
    parser.add_argument("--temp-root", type=Path, default=storage['temp_root'])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/preprocessing/frozen_preprocessing_v1.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--processing-scope", choices=["train-only", "all-blocks-mechanical"], default="train-only")
    parser.add_argument("--mechanical-access-config", type=Path, default=REPO_ROOT / "configs/access/mechanical_heldout_v1.yaml")
    parser.add_argument("--train-output-root", type=Path, default=storage['processed_root']/'train-only-v1')
    parser.add_argument("--fit-parameters-from", choices=["Train_only"], default="Train_only")
    parser.add_argument("--metrics-root", type=Path, default=REPO_ROOT / "outputs/metrics/scenario2")
    parser.add_argument("--journal", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--plan-next-batch", action="store_true")
    mode.add_argument("--execute-batch", action="store_true")
    mode.add_argument("--authorize-batch", action="store_true")
    mode.add_argument("--cleanup-pilot", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--batch-max-gib")
    parser.add_argument("--batch-manifest", type=Path)
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
        if any((args.plan_next_batch, args.execute_batch, args.authorize_batch, args.cleanup_pilot)):
            from resting_eeg.data import batches
            if args.processing_scope != 'all-blocks-mechanical':
                raise ValueError('Batch operations require all-blocks-mechanical scope')
            p = batches.paths(REPO_ROOT)
            for key in ('raw_root','metadata_root','config_path','metrics_root','train_output_root','temp_root'):
                if getattr(args, 'config' if key == 'config_path' else key).resolve() != p[key].resolve():
                    raise ValueError('Batch operations require canonical repository paths')
            if args.output_root and args.output_root.resolve() != p['output_root'].resolve():
                raise ValueError('Batch output path must be canonical')
            if args.mechanical_access_config.resolve() != p['access_config_path'].resolve():
                raise ValueError('Batch access configuration must be canonical')
            if args.subjects_file or args.max_subjects or args.journal or args.local_existing_only:
                raise ValueError('Batch membership and journals cannot be overridden')
            directory = args.metrics_root/'batches'
            if args.cleanup_pilot:
                if not args.delete_raw_after_success:
                    raise ValueError('Pilot cleanup requires --delete-raw-after-success')
                print(json.dumps(batches.cleanup_pilot(REPO_ROOT), indent=2))
                return 0
            batches.batch_name(args.batch_id)
            cap = batches.cap_bytes(args.batch_max_gib) if args.batch_max_gib is not None else None
            if args.plan_next_batch:
                if cap is None or args.subject_id:
                    raise ValueError('Planning requires an explicit cap and authoritative membership')
                if args.manifest.resolve() != (args.metrics_root/'full_candidate_download_manifest.csv').resolve():
                    raise ValueError('Planning requires the authoritative full candidate manifest')
                print(json.dumps(batches.freeze_batch(directory,args.batch_id,cap,pd.read_csv(args.manifest),
                    batches.verified_completed(REPO_ROOT),batches.git_head(REPO_ROOT),batches.frozen_hashes(REPO_ROOT)),indent=2))
            elif args.authorize_batch:
                print(json.dumps(batches.authorize(REPO_ROOT,args.batch_id,directory,args.batch_manifest,cap),indent=2))
            else:
                if args.batch_manifest is None:
                    raise ValueError('Execution requires --batch-manifest')
                result = batches.execute_batch(REPO_ROOT,args.batch_id,directory,manifest_path=args.batch_manifest,
                    requested_cap=cap,delete_raw=args.delete_raw_after_success,continue_on_error=args.continue_on_error,
                    retries=args.retries,timeout=args.timeout,selected=args.subject_id)
                print(json.dumps(result,indent=2))
                return int(bool(result['technical_failures']))
            return 0
        if args.batch_id or args.batch_max_gib or args.batch_manifest:
            raise ValueError('Batch options require a batch operation')
        frame = filter_manifest(pd.read_csv(args.manifest), subject_ids=args.subject_id,
                                subjects_file=args.subjects_file, max_subjects=args.max_subjects)
        results = stream_subjects(
            frame, repo_root=REPO_ROOT, raw_root=args.raw_root, metadata_root=args.metadata_root,
            output_root=args.output_root or load_paths(REPO_ROOT)['processed_root'] / ("all-blocks-v1" if args.processing_scope == "all-blocks-mechanical" else "train-only-v1"), config_path=args.config, metrics_root=args.metrics_root,
            execute=args.execute, local_existing_only=args.local_existing_only,
            delete_raw_after_success=args.delete_raw_after_success,
            continue_on_error=args.continue_on_error, retries=args.retries, timeout=args.timeout,
            journal_path=args.journal or load_paths(REPO_ROOT)['runtime_root']/('streaming_all_block_status.jsonl' if args.processing_scope=='all-blocks-mechanical' else 'streaming_subject_status.jsonl'), processing_scope=args.processing_scope,
            access_config_path=args.mechanical_access_config,train_output_root=args.train_output_root,
            fit_parameters_from=args.fit_parameters_from,
            temp_root=args.temp_root, download_journal_path=load_paths(REPO_ROOT)['runtime_root']/'streaming_download_status.jsonl',
        )
        for result in results:
            print(json.dumps(result, sort_keys=True))
        if not args.execute:
            print("DRY RUN: no downloads, waveform access, journal writes, or raw deletion. Scope follows --processing-scope.")
        return int(any(result["status"] not in {"planned", "complete"} for result in results))
    except Exception as exc:
        print(f"Streaming failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
