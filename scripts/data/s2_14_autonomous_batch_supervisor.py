"""Authorize or supervise automatic recovery of a frozen Batch 004 or 005."""
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
    mode.add_argument('--authorize-batch', action='store_true')
    parser.add_argument('--batch-id', choices=['004','005'], default='004')
    parser.add_argument('--supervision-token', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.authorize_resume:
        if args.batch_id != '004':
            parser.error('Use the immutable Batch 005 authorization and same frozen worker')
        print(json.dumps(autonomous.authorize(ROOT), indent=2))
    elif args.authorize_batch:
        if args.batch_id != '005':
            parser.error('New authorization is limited to final Batch 005')
        from resting_eeg.data import batches as b
        print(json.dumps(b.authorize(ROOT,'005',b.paths(ROOT)['metrics_root']/'batches'),indent=2))
    else:
        sys.exit(autonomous.worker(ROOT,args.supervision_token,args.batch_id) if args.worker else autonomous.supervisor(ROOT,args.batch_id))
