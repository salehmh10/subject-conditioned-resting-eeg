"""Generate or verify the deterministic Scenario 2 final cohort/split lock."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from resting_eeg.data.final_cohort_split import verify_artifacts, write_artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the four deterministic tracked artifacts")
    parser.add_argument("--require-clean", action="store_true", help="Require a clean Git worktree while verifying")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--accepted-record", type=Path)
    args = parser.parse_args()
    if args.write and args.require_clean:
        parser.error("--write and --require-clean are mutually exclusive")
    if args.write:
        summary, _ = write_artifacts(
            ROOT, output_root=args.output_root, acceptance_path=args.accepted_record
        )
        result = {
            "status": summary["status"],
            "write": "passed",
            "counts": summary["counts"],
            "pilot_exposure": summary["pilot_exposure"],
        }
    else:
        result = verify_artifacts(
            ROOT,
            output_root=args.output_root,
            acceptance_path=args.accepted_record,
            require_clean=args.require_clean,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
