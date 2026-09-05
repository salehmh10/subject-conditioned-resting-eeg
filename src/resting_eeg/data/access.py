"""The only real waveform-value entry point: complete provisional Train blocks.

Checks precede I/O, authorization is durably recorded before read, and no public
API exposes the underlying MNE Raw object. Header and cryptographic byte checks
are separately permitted. This is an application boundary, not an OS ACL.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from resting_eeg.common import append_csv, exclusive_lock, git_head, sha256_file, utc_now


class WaveformAccessError(PermissionError):
    pass


def sample_interval(start_sec, end_sec, sfreq):
    """Half-open [start,end): first sample >= start, stop at first sample >= end."""
    rate = Decimal(str(sfreq))
    # Decimal avoids float under/over-rounding of metadata sample-grid times.
    start = int((Decimal(str(start_sec)) * rate).to_integral_value(rounding=ROUND_CEILING))
    stop = int((Decimal(str(end_sec)) * rate).to_integral_value(rounding=ROUND_CEILING))
    if start < 0 or stop <= start:
        raise WaveformAccessError('Invalid block interval')
    return start, stop


def block_uid(subject, block_index):
    return f'{subject}_block-{int(block_index):02d}'


def truth(value):
    return str(value).lower() == 'true'


class TrainAccessGate:
    def __init__(self, split_path, inventory_path, registry_path, repo_root):
        self.split_path, self.inventory_path = Path(split_path), Path(inventory_path)
        self.split_hash = sha256_file(self.split_path)
        self.inventory_hash = sha256_file(self.inventory_path)
        self.split = pd.read_csv(self.split_path)
        self.inventory = pd.read_csv(self.inventory_path)
        self.registry_path, self.repo_root = Path(registry_path), Path(repo_root)
        for frame in (self.split, self.inventory):
            if frame.duplicated(['subject_id', 'block_index']).any():
                raise WaveformAccessError('Duplicate authoritative block key')

    def authorize(self, subject, index):
        if sha256_file(self.split_path) != self.split_hash or sha256_file(self.inventory_path) != self.inventory_hash:
            raise WaveformAccessError('Authoritative metadata changed during processing')
        rows = self.inventory.loc[(self.inventory.subject_id == subject) & (self.inventory.block_index == int(index))]
        if len(rows) != 1:
            raise WaveformAccessError('Unknown block')
        block = rows.iloc[0].to_dict()
        if not truth(block['is_study_block']) or truth(block['is_interruption_fragment']):
            raise WaveformAccessError('Interrupted/non-study fragment refused')
        if not truth(block['is_complete']):
            raise WaveformAccessError('Incomplete block refused')
        rows = self.split.loc[(self.split.subject_id == subject) & (self.split.block_index == int(index))]
        if len(rows) != 1 or rows.iloc[0]['split'] != 'Train':
            raise WaveformAccessError('Only provisional Train waveform access is allowed')
        assignment = rows.iloc[0]
        if assignment['split_status'] != 'PROVISIONAL_METADATA_ONLY':
            raise WaveformAccessError('Unexpected split status')
        for column in ['state', 'start_sec', 'end_estimate_capped_sec', 'acquisition_segment_id']:
            if block[column] != assignment[column]:
                raise WaveformAccessError(f'Split/inventory disagreement: {column}')
        block.update(split='Train', block_uid=block_uid(subject, index))
        return block

    def train_blocks(self, subject):
        indices = self.split.loc[(self.split.subject_id == subject) & (self.split.split == 'Train'), 'block_index']
        return [self.authorize(subject, index) for index in sorted(indices)]

    def record(self, block, sfreq, start, stop, purpose):
        if not purpose:
            raise WaveformAccessError('Value access requires a purpose')
        append_csv(self.registry_path, dict(
            timestamp_utc=utc_now(), subject_id=block['subject_id'], state=block['state'],
            block_index=block['block_index'], block_uid=block['block_uid'], split='Train',
            start_sec=block['start_sec'], end_sec=block['end_estimate_capped_sec'],
            native_sfreq=sfreq, start_sample=start, stop_sample_exclusive=stop,
            purpose=purpose, script_name='resting_eeg.data.access.TrainBlockLoader',
            git_head=git_head(self.repo_root), split_manifest_sha256=self.split_hash,
            block_inventory_sha256=self.inventory_hash, access_allowed=True,
            reason='complete_study_Train_authorized_before_read'))


@dataclass
class PhysicalBlock:
    values_v: np.ndarray
    channel_names: list[str]
    channel_types: list[str]
    sfreq: float
    start_sample: int
    stop_sample: int
    metadata: dict
    unit: str = 'V'


class TrainBlockLoader:
    def __init__(self, subject, vhdr, gate, integrity, lock_path):
        if not integrity.get('integrity_pass') or integrity.get('subject_id') != subject:
            raise WaveformAccessError('Subject integrity must pass before preprocessing')
        self.subject, self.vhdr, self.gate = subject, Path(vhdr), gate
        if (str(self.vhdr.resolve()) != integrity.get('verified_header_path') or
                self.vhdr.name != f'{subject}_task-resting_eeg.vhdr' or
                self.vhdr.parent.parent.name != subject or
                sha256_file(self.vhdr) != integrity.get('header_sha256')):
            raise WaveformAccessError('Loader header is not the verified subject recording')
        self._verified_header_sha256 = integrity['header_sha256']
        self.lock_path = Path(lock_path)
        self.__raw = None

    def __enter__(self):
        self._lock = exclusive_lock(self.lock_path)
        self._lock.__enter__()
        try:
            if sha256_file(self.vhdr) != self._verified_header_sha256:
                raise WaveformAccessError('Verified header changed before opening')
            self.__raw = mne.io.read_raw_brainvision(self.vhdr, preload=False, eog=['VEOG'], verbose='ERROR')
            self.__raw.set_montage('standard_1005', on_missing='raise', verbose='ERROR')
        except Exception:
            self._lock.__exit__(None, None, None)
            raise
        return self

    def read(self, index, purpose):
        if self.__raw is None:
            raise WaveformAccessError('Loader is closed')
        block = self.gate.authorize(self.subject, index)
        rate = float(self.__raw.info['sfreq'])
        start, stop = sample_interval(block['start_sec'], block['end_estimate_capped_sec'], rate)
        if stop > self.__raw.n_times or stop - start > int(65 * rate):
            raise WaveformAccessError('Block outside recording or 65-second memory bound')
        types = self.__raw.get_channel_types()
        if types.count('eeg') != 61 or types.count('eog') != 1:
            raise WaveformAccessError('Unexpected channel composition')
        self.gate.record(block, rate, start, stop, purpose)
        # MNE reads exactly [start, stop), all EEG and VEOG in original order.
        data = self.__raw.get_data(picks=list(range(62)), start=start, stop=stop)
        return PhysicalBlock(data, self.__raw.ch_names.copy(), types, rate, start, stop, block)

    def __exit__(self, *args):
        if self.__raw is not None:
            self.__raw.close()
            self.__raw = None
        self._lock.__exit__(*args)
