"""Read-only acceptance verification; never downloads or opens raw waveform values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

import pandas as pd

from resting_eeg.common import sha256_file
from resting_eeg.data.access import TrainAccessGate, sample_interval
from resting_eeg.data.integrity import audit_pilot_inventory
from resting_eeg.preprocessing.pipeline import load_config, verify_output


def verify(repo_root=REPO, require_clean=False, test_report=None):
    repo_root = Path(repo_root)
    metrics = repo_root / 'outputs/metrics/scenario2'
    pilot = pd.read_csv(metrics/'pilot_download_manifest.csv')
    inventory = audit_pilot_inventory(repo_root/'data_raw/MPI-LEMON', pilot)
    assert inventory['inventory_pass'] and inventory['n_subjects'] == 15
    config, digest = load_config(repo_root/'configs/preprocessing/frozen_preprocessing_v1.yaml')
    gate = TrainAccessGate(metrics/'provisional_block_split_manifest.csv', metrics/'lemon_block_inventory.csv',
                           metrics/'pilot_waveform_access_registry.csv', repo_root)
    access = pd.read_csv(gate.registry_path)
    assert access.subject_id.isin(pilot.subject_id).all()
    assert (access['split'] == 'Train').all() and access.access_allowed.all()
    for row in access.to_dict('records'):
        block = gate.authorize(row['subject_id'], row['block_index'])
        assert block['block_uid'] == row['block_uid'] and block['state'] == row['state']
        assert row['split_manifest_sha256'] == gate.split_hash and row['block_inventory_sha256'] == gate.inventory_hash
        assert (row['start_sample'], row['stop_sample_exclusive']) == sample_interval(block['start_sec'], block['end_estimate_capped_sec'], row['native_sfreq'])
    assert access.block_uid.nunique() == 150
    manifest = pd.read_csv(metrics/'pilot_processed_manifest.csv')
    assert len(manifest) == 15 and set(manifest.subject_id) == set(pilot.subject_id)
    for subject in sorted(pilot.subject_id.unique()):
        path = repo_root/'data_processed/MPI-LEMON'/f'{subject}_preprocessed.h5'
        receipt = json.loads(path.with_suffix('.provenance.json').read_text())
        expected = [b['block_uid'] for b in gate.train_blocks(subject)]
        result = verify_output(path, digest, expected)
        assert result['output_sha256'] == receipt['output_sha256']
        assert manifest.set_index('subject_id').loc[subject, 'output_sha256'] == result['output_sha256']
        assert receipt['source_hash_verified'] and receipt['processed_verified']
    # Git ignore rules are part of the acceptance gate, independently of file existence.
    for path in ['data_raw/MPI-LEMON/x.eeg', 'data_processed/MPI-LEMON/x.h5', 'data_normalized/x.npy', '.venv/pyvenv.cfg']:
        assert subprocess.run(['git','check-ignore','-q',path],cwd=repo_root).returncode == 0
    tracked = subprocess.check_output(['git','ls-files'],cwd=repo_root,text=True).splitlines()
    forbidden = {'.eeg','.vhdr','.vmrk','.h5','.hdf5','.npy','.npz','.pt','.pth','.ckpt'}
    assert not any(Path(p).suffix.lower() in forbidden for p in tracked)
    for name in ['pilot_loader_integrity.csv','pilot_loader_integrity_summary.json','pilot_environment.json',
                 'pilot_train_signal_sanity.csv','pilot_train_signal_sanity_summary.json',
                 'pilot_bad_channel_metrics.csv','pilot_bad_channel_decisions.csv','pilot_bad_channel_summary.json',
                 'pilot_qc_candidate_comparison.csv','pilot_train_window_qc.csv','pilot_train_block_qc.csv',
                 'pilot_train_subject_qc.csv','pilot_resampling_validation.csv','pilot_resampling_summary.json',
                 'pilot_processed_manifest.csv','pilot_storage_summary.json','pilot_raw_cleanup_manifest.csv',
                 'streaming_subject_status.jsonl']:
        assert (metrics/name).is_file(), name
    assert pd.read_csv(metrics/'pilot_loader_integrity.csv').integrity_pass.all()
    assert pd.read_csv(metrics/'pilot_resampling_validation.csv').pass_validation.all()
    for name in ['README.md','docs/PROTOCOL.md','docs/DECISIONS.md','docs/PILOT_PREPROCESSING_HANDOFF.md']:
        assert (repo_root/name).is_file()
    tests = None
    if test_report:
        root = ET.parse(test_report).getroot()
        suites = list(root.iter('testsuite'))
        assert suites and all(int(s.attrib['failures']) == int(s.attrib['errors']) == 0 for s in suites)
        tests = sum(int(s.attrib['tests']) for s in suites)
    clean = not subprocess.check_output(['git','status','--porcelain'],cwd=repo_root,text=True).strip()
    if require_clean:
        assert clean, 'Final acceptance requires a clean worktree'
    return dict(project_state='PREPROCESSING_FROZEN_READY_FOR_STREAMING' if clean else 'VERIFIED_PENDING_LOCAL_COMMIT',
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo_root,text=True).strip(),
                worktree_clean=clean, integrity_pass=15, accessed_Train_blocks=150,
                raw_access_records=len(access), pytest_count=tests, config_sha256=digest,
                target_sfreq=128, window_sec=4, stride_sec=2, storage_format='HDF5_float32_uV',
                pilot_raw_bytes=inventory['raw_bytes'], no_nonpilot_raw_present=True,
                no_heldout_or_interrupted_access=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-clean', action='store_true')
    parser.add_argument('--test-report', type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(require_clean=args.require_clean,test_report=args.test_report),indent=2))


if __name__ == '__main__':
    main()
