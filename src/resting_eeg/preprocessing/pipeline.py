"""Sequential subject preprocessing with a disk-backed, bounded block cache."""
from __future__ import annotations

import json
import importlib.metadata
import os
import uuid
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from resting_eeg.common import canonical_hash, exclusive_lock, git_head, sha256_file, utc_now, write_json
from resting_eeg.data.access import TrainAccessGate, TrainBlockLoader, sample_interval
from resting_eeg.data.integrity import audit_subject
from resting_eeg.preprocessing.signal import (
    apply_window_qc, channel_metrics, filter_resample, fit_bad_channels,
    fit_window_thresholds, reference_and_select, sanity_metrics, window_features,
)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    digest = canonical_hash(config)
    expected = Path(path).with_suffix('.sha256')
    if not expected.exists() or expected.read_text().strip() != digest:
        raise ValueError('Frozen configuration digest absent or mismatched')
    for package, version in config['software_versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Frozen software version mismatch: {package}')
    return config, digest


def verify_output(path, config_hash, expected_blocks):
    path = Path(path)
    with h5py.File(path, 'r') as handle:
        if handle.attrs['preprocessing_config_sha256'] != config_hash:
            raise ValueError('Output config hash mismatch')
        if set(handle['blocks']) != set(expected_blocks):
            raise ValueError('Expected blocks missing or unexpected blocks present')
        names = json.loads(handle.attrs['target_channels'])
        for uid in expected_blocks:
            group = handle['blocks'][uid]
            values = group['eeg_uv']
            starts = group['window_start_samples'][:]
            mask = group['window_usable_mask'][:]
            reasons = group['window_qc_reason'][:].astype(str)
            if values.dtype != np.float32 or values.shape[0] != 19 or len(names) != 19:
                raise ValueError('Invalid compact EEG shape/dtype')
            if group.attrs['physical_unit'] != 'uV' or group.attrs['provisional_split'] != 'Train':
                raise ValueError('Invalid output unit/access scope')
            if not bool(group.attrs['is_study_block']) or group.attrs['processed_sfreq'] != 128:
                raise ValueError('Invalid study block')
            if json.loads(group.attrs['channel_names']) != names:
                raise ValueError('Non-deterministic target channel order')
            expected_starts = np.arange(0, max(0, values.shape[1]-512+1), 256)
            if not np.array_equal(starts, expected_starts) or mask.shape != starts.shape or reasons.shape != starts.shape:
                raise ValueError('Invalid window geometry')
            if not np.array_equal(mask, reasons == 'usable'):
                raise ValueError('QC reason/mask mismatch')
            # At most one compact block in memory; all are already authorized Train.
            if not np.isfinite(values[:]).all():
                raise ValueError('Nonfinite compact values')
            for key in ('source_eeg_sha256', 'source_vhdr_sha256', 'code_git_head'):
                if not group.attrs[key]:
                    raise ValueError('Missing provenance')
    return dict(output_path=str(path.resolve()), output_sha256=sha256_file(path),
                output_bytes=path.stat().st_size, config_sha256=config_hash,
                expected_blocks=list(expected_blocks), processed_verified=True)


def code_digest(repo_root):
    files = sorted((repo_root / 'src' / 'resting_eeg').rglob('*.py'))
    return canonical_hash({str(path.relative_to(repo_root)): sha256_file(path) for path in files})


def native_block_failure(sanity, uid, bads, config):
    rows = sanity.loc[(sanity.block_uid == uid) & ~sanity.channel.isin(bads)]
    # A block-local flat channel can acquire apparent variation through CAR.
    # Persistent bad EEG is repaired by interpolation; unrepaired failures reject.
    return bool((rows.finite_fraction < 1).any() or
                (rows.robust_scale < config['bad_channels']['catastrophic_flat_scale_uv']).any() or
                (rows.flat_fraction > config['qc']['maximum_repeated_difference_fraction']).any())


def process_subject(subject_id, *, repo_root, raw_root, output_root, config_path, metrics_root, train_registry_path=None, temp_root=None):
    repo_root, raw_root, output_root, metrics_root = map(Path, (repo_root, raw_root, output_root, metrics_root))
    config, config_hash = load_config(config_path)
    authoritative = repo_root / 'outputs/metrics/scenario2'
    manifest = pd.read_csv(authoritative / 'full_candidate_download_manifest.csv')
    inventory = pd.read_csv(authoritative / 'lemon_block_inventory.csv')
    source_rows = manifest.loc[manifest.subject_id == subject_id]
    if len(source_rows) != 2:
        raise ValueError('Subject absent or incomplete in authoritative candidate manifest')
    integrity = audit_subject(subject_id, manifest, raw_root, inventory, config['target_channels'])
    if not integrity['integrity_pass']:
        raise ValueError(f"Subject integrity failed: {integrity['failure_reason']}")
    gate = TrainAccessGate(authoritative/'provisional_block_split_manifest.csv',
                           authoritative/'lemon_block_inventory.csv',
                           train_registry_path or metrics_root/'pilot_waveform_access_registry.csv', repo_root)
    blocks = gate.train_blocks(subject_id)
    if len(blocks) != 10 or {state: sum(b['state'] == state for b in blocks) for state in ('EO','EC')} != {'EO':5, 'EC':5}:
        raise ValueError('Expected five provisional Train blocks per state')
    expected = [b['block_uid'] for b in blocks]
    sources = source_rows.to_dict('records')
    source_hashes = {Path(row['relative_path']).suffix: row['payload_sha256'] for row in sources}
    source_directory = raw_root / subject_id / 'eeg'
    sidecar_hashes = {p.name: sha256_file(p) for p in sorted(source_directory.iterdir())
                     if p.is_file() and p.suffix not in ('.eeg', '.vhdr')}
    signature = canonical_hash(dict(config=config_hash, sources=sources, sidecars=sidecar_hashes,
                                    split=gate.split_hash, inventory=gate.inventory_hash,
                                    code=code_digest(repo_root)))
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f'{subject_id}_preprocessed.h5'
    receipt_path = output.with_suffix('.provenance.json')
    subject_metrics = metrics_root / 'subjects' / subject_id
    subject_metrics.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output.with_suffix('.lock')):
        if output.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['input_signature'] != signature:
                raise ValueError('Existing output has different source/config; explicit new output directory required')
            checked = verify_output(output, config_hash, expected)
            if receipt['output_sha256'] != checked['output_sha256']:
                raise ValueError('Existing output hash differs from durable receipt')
            receipt.update(checked, idempotent_reuse=True)
            return receipt
        if output.exists() or receipt_path.exists():
            # Preserve an interrupted two-file transaction and regenerate from verified raw.
            # Never overwrite an ambiguous output or use it as success evidence.
            token = uuid.uuid4().hex
            for path in (output, receipt_path):
                if path.exists():
                    os.replace(path, path.with_name(path.name + '.orphan-' + token))
        workspace = Path(temp_root) if temp_root is not None else output_root.parent/'temp'
        workspace.mkdir(parents=True, exist_ok=True)
        temporary = workspace/(output.name + '.tmp')
        cache_path = workspace/(output.stem + '.cache.h5')
        native_sanity, bad_metrics = [], []
        veog_nonfinite = set()
        created = utc_now()
        head = git_head(repo_root)
        guard = config['edge']['guard_samples_at_128']
        vhdr = raw_root / next(row['relative_path'] for row in sources if row['relative_path'].endswith('.vhdr'))
        try:
            with h5py.File(cache_path, 'w') as cache:
                with TrainBlockLoader(subject_id, vhdr, gate, integrity,
                                       workspace/'waveform_processing.lock') as loader:
                    for block in blocks:
                        physical = loader.read(block['block_index'], 'pilot_Train_sanity_and_frozen_preprocessing')
                        names, types, rate = physical.channel_names, physical.channel_types, physical.sfreq
                        eeg_indices = [i for i, kind in enumerate(types) if kind == 'eeg']
                        rows = sanity_metrics(physical.values_v, names)
                        if not np.isfinite(physical.values_v[names.index('VEOG')]).all():
                            veog_nonfinite.add(block['block_uid'])
                        for row in rows:
                            row.update(subject_id=subject_id, block_uid=block['block_uid'], state=block['state'], native_sfreq=rate)
                        native_sanity.extend(rows)
                        # Nonfinite channels are catastrophic bad-channel candidates.
                        # Replacement only enables interpolation; raw metrics preserve the failure.
                        np.nan_to_num(physical.values_v, copy=False, nan=0., posinf=0., neginf=0.)
                        compact = filter_resample(physical.values_v, rate, config)
                        del physical
                        cache.create_dataset(block['block_uid'], data=compact, track_times=False)
                        metrics = channel_metrics(compact[eeg_indices, guard:-guard] * 1e6,
                                                  [names[i] for i in eeg_indices])
                        metrics['subject_id'], metrics['block_uid'], metrics['state'] = subject_id, block['block_uid'], block['state']
                        bad_metrics.append(metrics)
                        del compact
                sanity = pd.DataFrame(native_sanity)
                metric_frame = pd.concat(bad_metrics, ignore_index=True)
                sanity['provisional_split'] = 'Train'
                metric_frame['provisional_split'] = 'Train'
                eeg_names = [names[i] for i in eeg_indices]
                bads, bad_eligible, decisions = fit_bad_channels(metric_frame, sanity, eeg_names,
                                                                config['target_channels'], config)
                decisions['subject_id'] = subject_id
                sanity.to_csv(subject_metrics/'signal_sanity.csv', index=False)
                metric_frame.to_csv(subject_metrics/'bad_channel_metrics.csv', index=False)
                decisions.to_csv(subject_metrics/'bad_channel_decisions.csv', index=False)
                all_features = []
                with h5py.File(temporary, 'w') as result:
                    result.attrs.update(subject_id=subject_id, preprocessing_config_sha256=config_hash,
                                        target_channels=json.dumps(config['target_channels']),
                                        source_manifest_rows=json.dumps(sources, sort_keys=True),
                                        original_channel_order=json.dumps(names), original_channel_types=json.dumps(types),
                                        source_eeg_sha256=source_hashes['.eeg'], source_vhdr_sha256=source_hashes['.vhdr'],
                                        split_manifest_sha256=gate.split_hash, block_inventory_sha256=gate.inventory_hash,
                                        code_git_head=head, code_content_sha256=code_digest(repo_root),
                                        created_utc=created, input_signature=signature,
                                        bad_channels=json.dumps(bads), bad_channel_limits_pass=bad_eligible,
                                        scope='provisional_Train_only', normalization_fitted=False)
                    root = result.create_group('blocks')
                    for block in blocks:
                        data_v = cache[block['block_uid']][:]
                        # If interpolation limits fail, retain a labeled diagnostic CAR output.
                        # Every candidate is rejected; no invalid channel is silently accepted.
                        target, veog = reference_and_select(data_v, names, types, bads if bad_eligible else [], config)
                        del data_v
                        target, veog = target[:, guard:-guard], veog[guard:-guard]
                        features = window_features(target, veog, config)
                        if block['block_uid'] in veog_nonfinite or native_block_failure(sanity, block['block_uid'], bads if bad_eligible else [], config):
                            features['finite'] = False
                            features['native_integrity_failure'] = True
                        else:
                            features['native_integrity_failure'] = False
                        features['block_uid'], features['state'], features['subject_id'] = block['block_uid'], block['state'], subject_id
                        all_features.append(features)
                        group = root.create_group(block['block_uid'])
                        group.create_dataset('eeg_uv', data=target.astype(np.float32),
                                             chunks=(19, min(1024, target.shape[1])),
                                             compression='gzip', compression_opts=4, shuffle=True, track_times=False)
                        group.attrs.update(subject_id=subject_id, state=block['state'], block_index=block['block_index'],
                                           block_uid=block['block_uid'], original_start_sec=block['start_sec'],
                                           original_end_sec=block['end_estimate_capped_sec'],
                                           native_sfreq=rate, processed_sfreq=128, channel_names=json.dumps(config['target_channels']),
                                           physical_unit='uV', edge_trim_sec=config['edge']['guard_sec'],
                                           first_processed_sample_sec=sample_interval(block['start_sec'],block['end_estimate_capped_sec'],rate)[0]/rate+config['edge']['guard_sec'],
                                           provisional_split='Train', is_study_block=True,
                                           preprocessing_config_sha256=config_hash,
                                           source_eeg_sha256=source_hashes['.eeg'], source_vhdr_sha256=source_hashes['.vhdr'],
                                           code_git_head=head)
                        del target, veog
                    feature_frame = pd.concat(all_features, ignore_index=True)
                    feature_frame['provisional_split'] = 'Train'
                    reference_available = bool((feature_frame.finite & ~feature_frame.flat).any())
                    thresholds = (fit_window_thresholds(feature_frame, config) if reference_available
                                  else dict(zip(config['qc']['features'], config['qc']['threshold_caps'])))
                    feature_frame['qc_reference_failed'] = not reference_available
                    mask, reasons = apply_window_qc(feature_frame, thresholds, bad_eligible)
                    feature_frame['usable'], feature_frame['reason'] = mask, reasons
                    candidates = []
                    for k in config['qc']['multiplier_candidates']:
                        threshold = fit_window_thresholds(feature_frame, config, k) if reference_available else thresholds
                        kmask, _ = apply_window_qc(feature_frame, threshold, bad_eligible)
                        for state in ('EO','EC'):
                            state_mask = feature_frame.state.to_numpy() == state
                            candidates.append(dict(subject_id=subject_id, native_sfreq=rate, multiplier=k, state=state,
                                                   candidate_windows=int(state_mask.sum()), usable_windows=int(kmask[state_mask].sum()),
                                                   retained_fraction=float(kmask[state_mask].mean()), selected=k == config['qc']['multiplier']))
                    block_qc = []
                    for uid, features in feature_frame.groupby('block_uid', sort=False):
                        group = root[uid]
                        group.create_dataset('window_start_samples', data=features.window_start_sample.to_numpy(), track_times=False)
                        group.create_dataset('window_usable_mask', data=features.usable.to_numpy(), track_times=False)
                        group.create_dataset('window_qc_reason', data=np.asarray(features.reason, dtype='S192'), track_times=False)
                        n, retained = len(features), int(features.usable.sum())
                        eligible = retained >= config['qc']['minimum_usable_windows_per_block'] and retained/n >= config['qc']['minimum_usable_fraction_per_block']
                        group.attrs['block_eligible'] = eligible
                        block_qc.append(dict(subject_id=subject_id, block_uid=uid, state=features.state.iloc[0],
                                             native_sfreq=rate, candidate_windows=n, usable_windows=retained,
                                             retained_fraction=retained/n, block_eligible=eligible))
                    blocks_frame = pd.DataFrame(block_qc)
                    min_blocks = config['qc']['minimum_usable_Train_blocks_per_state']
                    usable_by_state = blocks_frame.groupby('state').block_eligible.sum().to_dict()
                    qc_pass = reference_available and bad_eligible and all(usable_by_state[state] >= min_blocks for state in ('EO','EC'))
                    result.attrs.update(qc_pass=qc_pass, qc_thresholds=json.dumps(thresholds, sort_keys=True))
                feature_frame.to_csv(subject_metrics/'window_qc.csv', index=False)
                blocks_frame.to_csv(subject_metrics/'block_qc.csv', index=False)
                pd.DataFrame(candidates).to_csv(subject_metrics/'qc_candidates.csv', index=False)
                summary = dict(subject_id=subject_id, native_sfreq=rate, bad_channels=bads,
                               n_bad_eeg=len(bads), n_bad_target=len(set(bads)&set(config['target_channels'])),
                               bad_channel_limits_pass=bad_eligible, candidate_windows=len(feature_frame),
                               usable_windows=int(feature_frame.usable.sum()), retained_fraction=float(feature_frame.usable.mean()),
                               usable_blocks=int(blocks_frame.block_eligible.sum()), usable_blocks_by_state={k:int(v) for k,v in usable_by_state.items()},
                               qc_pass=bool(qc_pass), thresholds=thresholds, threshold_fit_available=reference_available, fit_pool='pooled_EO_EC_Train',
                               failure_reason='' if qc_pass else 'bad_channel_limits_or_fewer_than_3_usable_Train_blocks_per_state')
                write_json(subject_metrics/'subject_qc.json', summary)
            verify_output(temporary, config_hash, expected)
            os.replace(temporary, output)
            checked = verify_output(output, config_hash, expected)
            receipt = dict(**checked, subject_id=subject_id, source_hash_verified=True,
                           source_manifest_rows=sources, source_sidecar_sha256=sidecar_hashes, input_signature=signature,
                           code_git_head=head, code_content_sha256=code_digest(repo_root), created_utc=created,
                           qc_pass=bool(qc_pass), idempotent_reuse=False, qc_summary=summary)
            write_json(receipt_path, receipt)
            return receipt
        finally:
            # Only task-created cache/temporary paths; verified raw files are untouched.
            for path in (cache_path, temporary):
                if path.exists():
                    path.unlink()
