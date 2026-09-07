"""Verify s2_18 independently; no held-out EEG is read and no scaler is changed.

Input: normalization lock/scaler/block audit and the accepted final sources.
Output: PASS/FAIL JSON; optional evidence JSON outside the tracked artifacts.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from resting_eeg.data import train_normalization_verify as check
from resting_eeg.data import train_normalization as norm
from resting_eeg.data import final_cohort_split as frozen

OUTPUT_ROOT = ROOT / norm.METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    progress = lambda s: print(s, file=sys.stderr)
    # 1. Validate scaler/lock and independently reconstruct final Train inputs.
    repository = frozen.validate_repository_boundaries(ROOT, require_clean=args.require_clean)
    inputs = norm.load_locked_inputs(ROOT)
    check.verify_locked_artifact_hashes(OUTPUT_ROOT)
    scaler = frozen.read_json(OUTPUT_ROOT / norm.SCALER_FILE)
    # 2. Rebuild per-block Boolean sample coverage; check all assigned blocks.
    # 3. Read only eligible covered Train samples and check z mean/std (ddof=0).
    second_pass = check.independent_second_pass(
        inputs, scaler, (OUTPUT_ROOT / norm.BLOCK_FILE).read_bytes(), progress)
    # 4. Refit deterministically and require byte-identical canonical artifacts.
    mean, std, blocks, volume = norm.fit_train(inputs, progress)
    payloads = norm.build_payloads(inputs, mean, std, blocks, volume)
    frozen.validate_artifact_bytes(OUTPUT_ROOT, payloads)
    result = check.verification_report(OUTPUT_ROOT, repository, second_pass, volume, payloads)
    if args.report:
        frozen.atomic_write(args.report, frozen.json_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"TRAIN_NORMALIZATION_BLOCKED: {error}", file=sys.stderr)
        raise
