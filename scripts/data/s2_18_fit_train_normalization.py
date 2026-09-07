"""Fit one global scaler on locked eligible Real Train; next run s2_19.

Input: accepted final cohort/split, physical-unit HDF5 blocks and frozen masks.
Output: scaler JSON, all-assigned-Train block CSV, and normalization lock JSON.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from resting_eeg.data import train_normalization as norm
from resting_eeg.data.final_cohort_split import validate_repository_boundaries

OUTPUT_ROOT = ROOT / norm.METRICS


def main():
    # 1. Bind final membership, assignment, preprocessing and source identities.
    validate_repository_boundaries(ROOT)
    inputs = norm.load_locked_inputs(ROOT)
    # 2. Walk subjects/blocks in order; skip ineligible EEG; union usable windows.
    # 3. Accumulate float64 centered moments chunkwise and calculate ddof=0 std.
    mean, std, blocks, volume = norm.fit_train(inputs, progress=lambda s: print(s, file=sys.stderr))
    # 4. Account for every assignment and freeze the single shared scaler.
    payloads = norm.build_payloads(inputs, mean, std, blocks, volume)
    norm.write_payloads(OUTPUT_ROOT, payloads)
    print(json.dumps({"status": "TRAIN_NORMALIZATION_FITTED_AWAITING_VERIFICATION",
                      "volume": {k: v for k, v in volume.items() if k != "per_subject_unique_sample_counts"},
                      "next_script": "scripts/data/s2_19_verify_train_normalization.py"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"TRAIN_NORMALIZATION_BLOCKED: {error}", file=sys.stderr)
        raise
