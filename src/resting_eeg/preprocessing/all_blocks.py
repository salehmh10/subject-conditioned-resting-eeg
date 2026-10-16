"""Frozen mechanical application with verified Train reuse and bounded memory."""
from pathlib import Path
import json
import os

import h5py
import numpy as np
import pandas as pd

from resting_eeg.common import canonical_hash, sha256_file, write_json, utc_now, git_head, exclusive_lock
from resting_eeg.data.access import block_uid, truth, sample_interval
from resting_eeg.data.mechanical_access import (MechanicalHeldoutAccessGate, MechanicalHeldoutBlockLoader,
                                              load_access_config, require_committed_clean)
from resting_eeg.data.integrity import audit_subject
from resting_eeg.preprocessing.pipeline import load_config, verify_output, process_subject, native_block_failure
from resting_eeg.preprocessing.signal import (filter_resample, reference_and_select, window_features,
                                            apply_window_qc, sanity_metrics, expected_samples)

SCOPE = 'all_blocks_mechanical_provisional_v1'


def expected_subject_blocks(repo_root, subject):
    root = Path(repo_root) / 'outputs/metrics/scenario2'
    split = pd.read_csv(root / 'provisional_block_split_manifest.csv')
    inventory = pd.read_csv(root / 'lemon_block_inventory.csv')
    selected = split.loc[split.subject_id == subject].sort_values('block_index')
    if len(selected) != 16 or selected.block_index.duplicated().any():
        raise ValueError('Exactly 16 unique study blocks required')
    blocks = []
    for row in selected.to_dict('records'):
        match = inventory.loc[(inventory.subject_id == subject) & (inventory.block_index == row['block_index'])]
        if len(match) != 1:
            raise ValueError('Missing authoritative block')
        b = match.iloc[0].to_dict()
        if not truth(b['is_complete']) or not truth(b['is_study_block']) or truth(b['is_interruption_fragment']):
            raise ValueError('Invalid study block')
        for k in ('state', 'start_sec', 'end_estimate_capped_sec', 'acquisition_segment_id'):
            if b[k] != row[k]:
                raise ValueError('Split/inventory mismatch')
        if row['split_status'] != 'PROVISIONAL_METADATA_ONLY':
            raise ValueError('Unexpected split status')
        blocks.append(dict(b, split=row['split'], block_uid=block_uid(subject, row['block_index'])))
    for state in ('EO', 'EC'):
        if {s: sum(b['state'] == state and b['split'] == s for b in blocks) for s in ('Train','Validation','Test')} != {'Train':5,'Validation':1,'Test':2}:
            raise ValueError('Expected unchanged 5/1/2 allocation')
    return blocks


def verified_train_parameters(path, config, digest, blocks, metrics_root):
    """Cross-check receipt, HDF5, per-subject QC, and published pilot evidence."""
    path, metrics_root = Path(path), Path(metrics_root)
    expected = [b['block_uid'] for b in blocks if b['split'] == 'Train']
    checked = verify_output(path, digest, expected)
    receipt = json.loads(path.with_suffix('.provenance.json').read_text())
    if receipt['output_sha256'] != checked['output_sha256'] or not receipt['source_hash_verified']:
        raise ValueError('Train receipt/hash mismatch')
    subject = receipt['subject_id']
    summary_path = metrics_root / 'subjects' / subject / 'subject_qc.json'
    summary = json.loads(summary_path.read_text())
    if summary != receipt['qc_summary'] or summary['fit_pool'] != 'pooled_EO_EC_Train':
        raise ValueError('Train parameter evidence missing/inconsistent')
    with h5py.File(path, 'r') as h:
        if h.attrs['scope'] != 'provisional_Train_only' or h.attrs['normalization_fitted']:
            raise ValueError('Unexpected Train scope/normalization')
        for attr, key in [('qc_thresholds','thresholds'), ('bad_channels','bad_channels')]:
            if json.loads(h.attrs[attr]) != summary[key]:
                raise ValueError('Train HDF5 parameters disagree')
        if bool(h.attrs['bad_channel_limits_pass']) != summary['bad_channel_limits_pass'] or bool(h.attrs['qc_pass']) != summary['qc_pass']:
            raise ValueError('Train QC disagreement')
        if json.loads(h.attrs['target_channels']) != config['target_channels']:
            raise ValueError('Train target order mismatch')
        names, types = json.loads(h.attrs['original_channel_order']), json.loads(h.attrs['original_channel_types'])
        sources = json.loads(h.attrs['source_manifest_rows'])
        if sources != receipt['source_manifest_rows']:
            raise ValueError('Train source provenance mismatch')
    decisions = pd.read_csv(metrics_root/'subjects'/subject/'bad_channel_decisions.csv')
    if decisions.loc[decisions.bad, 'channel'].tolist() != summary['bad_channels'] or set(decisions.fit_pool) != {'pooled_EO_EC_Train'}:
        raise ValueError('Train bad-channel decisions disagree')
    aggregate = metrics_root / 'pilot_train_subject_qc.csv'
    if aggregate.exists():
        rows = pd.read_csv(aggregate).query('subject_id == @subject')
        if len(rows):
            row = rows.iloc[0]
            for key in ('thresholds','bad_channels','usable_blocks_by_state'):
                if json.loads(row[key]) != summary[key]:
                    raise ValueError('Committed Train parameter evidence disagrees')
            if bool(row.qc_pass) != summary['qc_pass']:
                raise ValueError('Committed Train QC disagrees')
            manifest = pd.read_csv(metrics_root/'pilot_processed_manifest.csv').set_index('subject_id')
            if manifest.loc[subject,'output_sha256'] != checked['output_sha256']:
                raise ValueError('Committed Train output hash disagrees')
    parameters = dict(summary=summary, channel_names=names, channel_types=types,
                      train_output_sha256=checked['output_sha256'], source_manifest_rows=sources,
                      source_sidecar_sha256=receipt['source_sidecar_sha256'],
                      preprocessing_config_sha256=digest, fit_block_uids=expected, fit_parameters_from='Train_only')
    return parameters, canonical_hash(parameters)


def verify_all_block_output(path, config_hash, expected_blocks, access_hash, parameter_hash=None):
    """Output verification is a permitted mechanical operation, one block at a time."""
    if len(expected_blocks) != 16:
        raise ValueError('All-block verification requires 16 authoritative blocks')
    with h5py.File(path, 'r') as h:
        if h.attrs['scope'] != SCOPE or h.attrs['normalization_fitted']:
            raise ValueError('Wrong all-block scope/normalization')
        if h.attrs['preprocessing_config_sha256'] != config_hash or h.attrs['mechanical_access_config_sha256'] != access_hash:
            raise ValueError('All-block config hash mismatch')
        params = json.loads(h.attrs['train_parameters'])
        actual_parameter_hash = canonical_hash(params)
        if h.attrs['train_parameter_source_hash'] != actual_parameter_hash or (parameter_hash and parameter_hash != actual_parameter_hash):
            raise ValueError('Train parameter hash mismatch')
        if params['fit_parameters_from'] != 'Train_only' or set(params['fit_block_uids']) != {b['block_uid'] for b in expected_blocks if b['split']=='Train'}:
            raise ValueError('Held-out data in parameter fitting provenance')
        if set(h['blocks']) != {b['block_uid'] for b in expected_blocks}:
            raise ValueError('Missing or unexpected all-block group')
        names = json.loads(h.attrs['target_channels'])
        source_hashes = {Path(r['relative_path']).suffix:r['payload_sha256'] for r in params['source_manifest_rows']}
        for b in expected_blocks:
            g = h['blocks'][b['block_uid']]
            for key, value in dict(subject_id=b['subject_id'],state=b['state'],block_index=b['block_index'],block_uid=b['block_uid'],
                                   provisional_split=b['split'],is_study_block=True,is_complete=True,
                                   original_start_sec=b['start_sec'],original_end_sec=b['end_estimate_capped_sec'],
                                   processed_sfreq=128,physical_unit='uV',edge_trim_sec=3.3828125,
                                   preprocessing_config_sha256=config_hash,mechanical_access_config_sha256=access_hash,
                                   train_parameter_source_hash=actual_parameter_hash,
                                   source_eeg_sha256=source_hashes['.eeg'],source_vhdr_sha256=source_hashes['.vhdr']).items():
                if g.attrs.get(key) != value:
                    raise ValueError(f'All-block attribute mismatch: {key}')
            if not g.attrs.get('processing_code_head') or json.loads(g.attrs['channel_names']) != names or names != json.loads(h.attrs['target_channels']):
                raise ValueError('Channel/provenance mismatch')
            x = g['eeg_uv']
            start, stop = sample_interval(b['start_sec'], b['end_estimate_capped_sec'], g.attrs['native_sfreq'])
            n = expected_samples(stop-start, g.attrs['native_sfreq']) - 866
            if x.shape != (19,n) or x.dtype != np.float32 or not np.isfinite(x[:]).all():
                raise ValueError('Invalid compact shape/dtype/finite status')
            starts, mask = g['window_start_samples'][:], g['window_usable_mask'][:]
            reasons = g['window_qc_reason'][:].astype(str)
            if not np.array_equal(starts,np.arange(0,max(0,n-512+1),256)) or mask.shape != starts.shape or reasons.shape != starts.shape:
                raise ValueError('Window geometry crosses block or is invalid')
            if not np.array_equal(mask,reasons=='usable'):
                raise ValueError('Mask/reason mismatch')
            eligible = int(mask.sum()) >= 10 and float(mask.mean()) >= .5
            if bool(g.attrs['block_eligible']) != eligible:
                raise ValueError('Block eligibility mismatch')
    return dict(output_path=str(Path(path).resolve()),output_sha256=sha256_file(path),output_bytes=Path(path).stat().st_size,
                config_sha256=config_hash,expected_blocks=[b['block_uid'] for b in expected_blocks],processed_verified=True)


def apply_heldout_block(physical, parameters, config):
    """No fit callable is imported or invoked here. Parameters are immutable inputs."""
    before = canonical_hash(parameters)
    summary = parameters['summary']
    if physical.metadata['split'] not in ('Validation','Test') or parameters['fit_parameters_from'] != 'Train_only':
        raise ValueError('Held-out apply-only scope required')
    if physical.channel_names != parameters['channel_names'] or physical.channel_types != parameters['channel_types']:
        raise ValueError('Native channel order differs from Train')
    uid = physical.metadata['block_uid']
    sanity = pd.DataFrame(sanity_metrics(physical.values_v, physical.channel_names))
    sanity['block_uid'] = uid
    bads = summary['bad_channels'] if summary['bad_channel_limits_pass'] else []
    failed = native_block_failure(sanity,uid,bads,config)
    np.nan_to_num(physical.values_v,copy=False,nan=0.,posinf=0.,neginf=0.)
    compact = filter_resample(physical.values_v,physical.sfreq,config)
    target, veog = reference_and_select(compact,physical.channel_names,physical.channel_types,bads,config)
    guard = config['edge']['guard_samples_at_128']
    target, veog = target[:,guard:-guard], veog[guard:-guard]
    features = window_features(target,veog,config)
    features['provisional_split'] = physical.metadata['split']
    features['native_integrity_failure'] = failed
    if failed:
        features['finite'] = False
    features['qc_reference_failed'] = not summary['threshold_fit_available']
    mask, reasons = apply_window_qc(features,summary['thresholds'],summary['bad_channel_limits_pass'])
    if canonical_hash(parameters) != before:
        raise ValueError('Train parameters mutated during held-out application')
    return target.astype(np.float32),features.window_start_sample.to_numpy(),mask,np.asarray(reasons,dtype='S192')


def subject_feasibility(path, train_summary):
    counts = {}
    with h5py.File(path, 'r') as h:
        for split in ('Train', 'Validation', 'Test'):
            for state in ('EO', 'EC'):
                groups = [g for g in h['blocks'].values() if g.attrs['provisional_split']==split and g.attrs['state']==state]
                counts[f'{state}_{split}_eligible_blocks'] = sum(bool(g.attrs['block_eligible']) for g in groups)
    failures = [] if train_summary['qc_pass'] else ['frozen_Train_QC_failed']
    for split,n in [('Validation',1),('Test',2)]:
        for state in ('EO','EC'):
            if counts[f'{state}_{split}_eligible_blocks']!=n:
                failures.append(f'{state}_{split}_assigned_block_ineligible')
    return dict(counts,train_criterion_pass=train_summary['qc_pass'],provisional_5_1_2_feasible=not failures,
                label='pilot_all_block_qc_not_feasible' if failures else 'pilot_all_block_qc_feasible',reason=';'.join(failures))


def process_all_blocks(subject_id, *, repo_root, raw_root, output_root, config_path, metrics_root,
                       access_config_path, train_output_root=None, fit_parameters_from='Train_only', temp_root=None):
    if fit_parameters_from != 'Train_only':
        raise ValueError('Held-out parameter fitting is prohibited')
    repo_root, raw_root, output_root, metrics_root = map(Path,(repo_root,raw_root,output_root,metrics_root))
    workspace = Path(temp_root) if temp_root is not None else output_root.parent/'temp'
    workspace.mkdir(parents=True, exist_ok=True)
    config,digest = load_config(config_path)
    spec,access_hash = load_access_config(access_config_path,config_path)
    authoritative = repo_root/'outputs/metrics/scenario2'
    blocks = expected_subject_blocks(repo_root,subject_id)
    gate = MechanicalHeldoutAccessGate(authoritative/'provisional_block_split_manifest.csv',authoritative/'lemon_block_inventory.csv',
        metrics_root/'pilot_mechanical_heldout_access_registry.csv',repo_root,access_config_path,config_path)
    output = output_root/f'{subject_id}_all_blocks.h5'
    receipt_path = output.with_suffix('.provenance.json')
    if output.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        checked = verify_all_block_output(output,digest,blocks,access_hash,receipt['train_parameter_source_hash'])
        if checked['output_sha256'] != receipt['output_sha256']:
            raise ValueError('All-block output differs from durable receipt')
        return dict(receipt,**{'idempotent_reuse':True})
    # New value access is permitted only from committed, clean code. Existing
    # receipts above support read-only verification after documentation commits.
    head = require_committed_clean(repo_root)
    if output.exists() or receipt_path.exists() or (workspace/(output.name + '.tmp')).exists() or output.with_suffix('.h5.tmp').exists():
        raise ValueError('Partial all-block transaction: investigate and invalidate explicitly before restart')
    train_root = Path(train_output_root) if train_output_root else output_root.parent/'train-only-v1'
    train_path = train_root/f'{subject_id}_preprocessed.h5'
    if not train_path.exists():
        process_subject(subject_id,repo_root=repo_root,raw_root=raw_root,output_root=train_root,config_path=config_path,metrics_root=metrics_root,
                        train_registry_path=metrics_root/'streaming_train_access_registry.csv', temp_root=workspace)
    params,param_hash = verified_train_parameters(train_path,config,digest,blocks,metrics_root)
    manifest = pd.read_csv(authoritative/'full_candidate_download_manifest.csv')
    inventory = pd.read_csv(authoritative/'lemon_block_inventory.csv')
    integrity = audit_subject(subject_id,manifest,raw_root,inventory,config['target_channels'])
    if not integrity['integrity_pass']:
        raise ValueError('Source/header integrity failed')
    sources = manifest.loc[manifest.subject_id==subject_id].to_dict('records')
    if sources != params['source_manifest_rows']:
        raise ValueError('Source manifest differs from Train provenance')
    folder = raw_root/subject_id/'eeg'
    sidecars = {p.name:sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix not in ('.eeg','.vhdr')}
    if sidecars != params['source_sidecar_sha256']:
        raise ValueError('Sidecars differ from verified Train provenance')
    source_hashes = {Path(r['relative_path']).suffix:r['payload_sha256'] for r in sources}
    header = raw_root/next(r['relative_path'] for r in sources if r['relative_path'].endswith('.vhdr'))
    output_root.mkdir(parents=True,exist_ok=True)
    temporary = workspace/(output.name + '.tmp')
    with exclusive_lock(output.with_suffix('.lock')):
        with h5py.File(train_path,'r') as train, h5py.File(temporary,'w') as h:
            for key,value in train.attrs.items():
                h.attrs[key] = value
            h.attrs.update(scope=SCOPE,mechanical_access_config_sha256=access_hash,train_parameters=json.dumps(params,sort_keys=True),
                           train_parameter_source_hash=param_hash,processing_code_head=head,created_utc=utc_now())
            root = h.create_group('blocks')
            for b in blocks:
                if b['split']=='Train':
                    train.copy(train['blocks'][b['block_uid']],root,name=b['block_uid'])
            with MechanicalHeldoutBlockLoader(subject_id,header,gate,integrity,workspace/'waveform_processing.lock') as loader:
                for b in blocks:
                    if b['split']=='Train':
                        continue
                    physical = loader.read(b['block_index'],'frozen_preprocessing')
                    rate = physical.sfreq
                    target,starts,mask,reasons = apply_heldout_block(physical,params,config)
                    del physical
                    g = root.create_group(b['block_uid'])
                    g.create_dataset('eeg_uv',data=target,chunks=(19,min(1024,target.shape[1])),compression='gzip',compression_opts=4,shuffle=True,track_times=False)
                    for key,data in [('window_start_samples',starts),('window_usable_mask',mask),('window_qc_reason',reasons)]:
                        g.create_dataset(key,data=data,track_times=False)
                    g.attrs.update(subject_id=subject_id,state=b['state'],block_index=b['block_index'],block_uid=b['block_uid'],
                        original_start_sec=b['start_sec'],original_end_sec=b['end_estimate_capped_sec'],native_sfreq=rate,processed_sfreq=128,
                        channel_names=json.dumps(config['target_channels']),physical_unit='uV',edge_trim_sec=config['edge']['guard_sec'],
                        provisional_split=b['split'],preprocessing_config_sha256=digest,code_git_head=head,
                        source_eeg_sha256=source_hashes['.eeg'],source_vhdr_sha256=source_hashes['.vhdr'],
                        block_eligible=bool(mask.sum()>=10 and mask.mean()>=.5))
                    del target
            for b in blocks:
                root[b['block_uid']].attrs.update(is_complete=True,is_study_block=True,mechanical_access_config_sha256=access_hash,
                    train_parameter_source_hash=param_hash,processing_code_head=head)
        checked = verify_all_block_output(temporary,digest,blocks,access_hash,param_hash)
        # Persist both files before promotion; a partial pair fails closed on restart.
        receipt = dict(checked,subject_id=subject_id,source_hash_verified=True,source_manifest_rows=sources,source_sidecar_sha256=sidecars,
            scope=SCOPE,authoritative_blocks=blocks,mechanical_access_config_sha256=access_hash,train_parameter_source_hash=param_hash,
            train_parameters=params,fit_parameters_from='Train_only',heldout_parameters_are='apply_only',
            qc_pass=params['summary']['qc_pass'],qc_summary=params['summary'],code_git_head=head,created_utc=utc_now(),
            train_blocks_reused=10,heldout_blocks_processed=6,idempotent_reuse=False,
            provisional_feasibility=subject_feasibility(temporary,params['summary']))
        receipt['output_path'] = str(output.resolve())
        pending_receipt = receipt_path.with_suffix('.pending.json')
        write_json(pending_receipt,receipt)
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary,output)
        os.replace(pending_receipt,receipt_path)
        return receipt
