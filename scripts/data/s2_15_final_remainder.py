"""Freeze or audit the explicitly authorized final Batch 005. No network calls."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from resting_eeg.data import final_remainder as final

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--write-results',action='store_true')
    args=parser.parse_args()
    result=final.freeze(ROOT) if args.freeze else final.completion_audit(ROOT,write=args.write_results)
    print(json.dumps(result,indent=2))
