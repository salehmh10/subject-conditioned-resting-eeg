"""Separate apply-only access boundary; the Train gate remains unchanged.

Application policy, not an operating-system sandbox. Raw objects remain private
in the bounded loader; the durable record is written before every value read.
"""
from pathlib import Path
import subprocess
import yaml

from resting_eeg.common import canonical_hash, sha256_file, append_csv, utc_now, git_head
from resting_eeg.data.access import TrainAccessGate, TrainBlockLoader, WaveformAccessError, truth, block_uid, sample_interval
from resting_eeg.preprocessing.pipeline import load_config

FROZEN_SHA = '5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b'
PURPOSES = {'frozen_preprocessing', 'fixed_waveform_qc', 'compact_storage', 'output_verification'}


def load_access_config(path, preprocessing_path):
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding='utf-8'))
    digest = canonical_hash(spec)
    if path.with_suffix('.sha256').read_text().strip() != digest:
        raise WaveformAccessError('Mechanical config hash mismatch')
    _, frozen = load_config(preprocessing_path)
    if frozen != FROZEN_SHA or spec['required_preprocessing_config_sha256'] != frozen:
        raise WaveformAccessError('Preprocessing config hash mismatch')
    expected = dict(scope_name='scenario2-mechanical-heldout-v1', allowed_splits=['Validation', 'Test'],
                    required_split_status='PROVISIONAL_METADATA_ONLY', require_complete_study_block=True,
                    allow_interruption_fragments=False, allow_incomplete_blocks=False,
                    fit_parameters_from='Train_only', heldout_parameters_are='apply_only')
    if any(spec.get(k) != v for k, v in expected.items()) or set(spec['allowed_purposes']) != PURPOSES:
        raise WaveformAccessError('Mechanical policy cannot widen access or fit held-out parameters')
    return spec, digest


def require_committed_clean(repo_root):
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_root, text=True).splitlines()
    # Append-only execution evidence may grow after the initial clean preflight.
    if any(not line[3:].replace('\\', '/').startswith('outputs/') for line in dirty):
        raise WaveformAccessError('Commit code/authorization and clean worktree before new held-out reads')
    return git_head(repo_root)


class MechanicalHeldoutAccessGate(TrainAccessGate):
    def __init__(self, split_path, inventory_path, registry_path, repo_root, access_config_path, config_path):
        super().__init__(split_path, inventory_path, registry_path, repo_root)
        self.access_path, self.config_path = Path(access_config_path), Path(config_path)
        self.spec, self.access_hash = load_access_config(self.access_path, self.config_path)
        if self.split_hash != self.spec['split_manifest_sha256'] or self.inventory_hash != self.spec['block_inventory_sha256']:
            raise WaveformAccessError('Authoritative manifest hash mismatch')

    def authorize(self, subject, index, purpose='frozen_preprocessing'):
        if purpose not in PURPOSES:
            raise WaveformAccessError('Scientific, fitting, plotting or unknown purpose refused')
        if load_access_config(self.access_path, self.config_path)[1] != self.access_hash:
            raise WaveformAccessError('Mechanical config changed')
        if sha256_file(self.split_path) != self.split_hash or sha256_file(self.inventory_path) != self.inventory_hash:
            raise WaveformAccessError('Authoritative metadata changed')
        rows = self.inventory.loc[(self.inventory.subject_id == subject) & (self.inventory.block_index == int(index))]
        if len(rows) != 1:
            raise WaveformAccessError('Unknown block')
        block = rows.iloc[0].to_dict()
        if not truth(block['is_study_block']) or truth(block['is_interruption_fragment']):
            raise WaveformAccessError('Interrupted/non-study fragment refused')
        if not truth(block['is_complete']):
            raise WaveformAccessError('Incomplete block refused')
        rows = self.split.loc[(self.split.subject_id == subject) & (self.split.block_index == int(index))]
        if len(rows) != 1 or rows.iloc[0]['split'] not in ('Validation', 'Test'):
            raise WaveformAccessError('Only Validation/Test permitted by mechanical gate')
        assignment = rows.iloc[0]
        if assignment['split_status'] != self.spec['required_split_status']:
            raise WaveformAccessError('Unexpected split status')
        for key in ('state', 'start_sec', 'end_estimate_capped_sec', 'acquisition_segment_id'):
            if block[key] != assignment[key]:
                raise WaveformAccessError(f'Split/inventory disagreement: {key}')
        return dict(block, split=assignment['split'], block_uid=block_uid(subject, index))

    def record(self, block, sfreq, start, stop, purpose):
        checked = self.authorize(block['subject_id'], block['block_index'], purpose)
        if checked != block or (start, stop) != sample_interval(block['start_sec'], block['end_estimate_capped_sec'], sfreq):
            raise WaveformAccessError('Authorization interval mismatch')
        append_csv(self.registry_path, dict(timestamp_utc=utc_now(), subject_id=block['subject_id'],
            state=block['state'], block_index=block['block_index'], block_uid=block['block_uid'],
            provisional_split=block['split'], start_sec=block['start_sec'], end_sec=block['end_estimate_capped_sec'],
            native_sfreq=sfreq, native_start_sample=start, native_stop_sample=stop, purpose=purpose,
            preprocessing_config_sha256=FROZEN_SHA, mechanical_access_config_sha256=self.access_hash,
            code_git_head=git_head(self.repo_root), split_manifest_sha256=self.split_hash,
            block_inventory_sha256=self.inventory_hash, access_allowed=True,
            authorization_reason='complete_study_heldout_frozen_apply_only_before_read',
            script_name='resting_eeg.data.mechanical_access.MechanicalHeldoutBlockLoader'))


class MechanicalHeldoutBlockLoader(TrainBlockLoader):
    def __init__(self, subject, vhdr, gate, integrity, lock_path):
        if not isinstance(gate, MechanicalHeldoutAccessGate):
            raise WaveformAccessError('Mechanical loader requires separate held-out gate')
        super().__init__(subject, vhdr, gate, integrity, lock_path)

    def read(self, index, purpose):
        self.gate.authorize(self.subject, index, purpose)
        return super().read(index, purpose)
