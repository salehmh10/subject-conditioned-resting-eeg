"""Authorize or supervise automatic recovery of the already frozen Batch 004."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from resting_eeg.data import autonomous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--authorize-resume', action='store_true')
    mode.add_argument('--worker', action='store_true')
    parser.add_argument('--supervision-token', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.authorize_resume:
        print(json.dumps(autonomous.authorize(ROOT), indent=2))
    else:
        sys.exit(autonomous.worker(ROOT, args.supervision_token) if args.worker else autonomous.supervisor(ROOT))
