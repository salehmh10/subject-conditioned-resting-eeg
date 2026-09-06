"""Write durable post-commit acceptance for the Scenario 2 cohort/split lock."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from resting_eeg.data.final_cohort_split import write_external_acceptance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    args = parser.parse_args()
    result = write_external_acceptance(ROOT, args.evidence_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
