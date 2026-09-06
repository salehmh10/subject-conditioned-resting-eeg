"""Engineering-only final remainder planning, coverage, and completion audit."""
from pathlib import Path
import subprocess

import pandas as pd

from resting_eeg.common import canonical_hash, git_head, sha256_file, utc_now, write_json
from resting_eeg.data import batches as b

EXPECTED_REMAINDER_BYTES = 42976220335
EXPECTED_REMAINDER_SUBJECTS = 65
EXPECTED_CANDIDATES = 211
STRUCTURAL_EXCLUSIONS = {'sub-010015', 'sub-010100', 'sub-010203', 'sub-010285'}


def remainder_membership(frame, completed, assigned):
    """Derive every remaining pair in authoritative order, then reconcile totals."""
    b.st._manifest_rows(frame, Path('synthetic_manifest_validation'))
    candidates = frame.subject_id.drop_duplicates().tolist()
    completed, assigned = list(completed), list(assigned)
    if len(candidates) != EXPECTED_CANDIDATES or set(candidates) & STRUCTURAL_EXCLUSIONS:
        raise ValueError('Structural candidate identity/count contradiction')
    if len(completed) != 146 or len(completed) != len(set(completed)):
        raise ValueError('Expected 146 unique verified pre-Batch-005 completions')
    if len(assigned) != len(set(assigned)) or not set(assigned) <= set(completed):
        raise ValueError('Duplicate assignment or unresolved previous batch')
    if not set(completed) <= set(candidates):
        raise ValueError('Completed subject outside structural candidates')
    remaining = frame.loc[~frame.subject_id.isin(set(completed) | set(assigned))]
    total = sum(int(v) for v in remaining.payload_size_bytes)
    ids = remaining.subject_id.drop_duplicates().tolist()
    if len(ids) != EXPECTED_REMAINDER_SUBJECTS or total != EXPECTED_REMAINDER_BYTES:
        raise ValueError(f'Final remainder reconciliation failed: {len(ids)} subjects, {total} bytes')
    if ids[0] != 'sub-010242' or ids[-1] != candidates[-1]:
        raise ValueError('Final remainder authoritative endpoints disagree')
    selected, plan = b.plan_membership(frame, total, completed, assigned)
    if plan['subject_ids'] != ids or plan['unused_capacity_bytes'] != 0 or len(selected) != 130:
        raise ValueError('Final remainder omitted a subject or complete source pair')
    return selected, plan


def coverage_record(root):
    directory = b.paths(root)['metrics_root']/'batches'
    frame = pd.read_csv(directory.parent/'full_candidate_download_manifest.csv')
    before = b.read_json(directory/'batch_005_starting_state.json')
    prior = [b.verify_frozen(directory, f'{i:03}')[1] for i in range(1, 5)]
    assigned = [s for plan in prior for s in plan['subject_ids']]
    selected, expected = remainder_membership(frame, before['completed_subjects'], assigned)
    frozen, plan, entry = b.verify_frozen(directory, '005')
    if frozen.to_dict('records') != selected.to_dict('records'):
        raise ValueError('Frozen final remainder differs from authoritative source')
    for key in ('subject_ids', 'subjects', 'subject_count', 'asset_count', 'cap_bytes',
                'declared_bytes', 'unused_capacity_bytes', 'first_not_included'):
        if plan[key] != expected[key]:
            raise ValueError(f'Final remainder plan invariant changed: {key}')
    if plan.get('mode') != 'final_remainder' or plan.get('final_remainder') is not True or plan.get('future_batch_allowed') is not False:
        raise ValueError('Final remainder mode or future boundary changed')
    combined = before['completed_subjects'] + plan['subject_ids']
    candidates = frame.subject_id.drop_duplicates().tolist()
    if len(combined) != 211 or len(set(combined)) != 211 or set(combined) != set(candidates):
        raise ValueError('Full candidate coverage has overlap, duplicates, or omissions')
    if list(directory.glob('batch_006*')) or list(b.runtime_directory(root).glob('batch_006*')):
        raise ValueError('Batch 006 is prohibited')
    if any(int(e['batch_id']) > 5 for e in b.ledger_read(directory)['batches']):
        raise ValueError('Future batch assignment is prohibited')
    return dict(passed=True, batch_id='005', candidate_count=211, candidate_ids=candidates,
        previous_completed_count=146, previous_completed_ids=before['completed_subjects'],
        final_remainder_count=65, subject_ids=plan['subject_ids'], asset_count=130,
        first_subject=plan['subject_ids'][0], last_subject=plan['subject_ids'][-1],
        declared_bytes=expected['declared_bytes'], cap_bytes=expected['cap_bytes'],
        unused_capacity_bytes=0, remaining_unassigned_count=0, overlap=[], omissions=[], duplicates=[],
        structural_exclusions=sorted(STRUCTURAL_EXCLUSIONS), structural_exclusions_in_batch=[],
        retained_complete_subjects_excluded=before['retained_raw_subjects'],
        future_batch_allowed=False, no_batch_006=True,
        source_manifest_sha256=sha256_file(directory.parent/'full_candidate_download_manifest.csv'),
        manifest_sha256=entry['manifest_sha256'], plan_sha256=entry['plan_sha256'],
        plan_file_sha256=entry['plan_file_sha256'],
        starting_state_sha256=sha256_file(directory/'batch_005_starting_state.json'))


def validate_coverage(root):
    expected = coverage_record(root)
    stored = b.read_json(b.paths(root)['metrics_root']/'batches/batch_005_coverage_audit.json')
    if stored != expected:
        raise ValueError('Final coverage audit differs from frozen evidence')
    return stored


def freeze(root):
    directory = b.paths(root)['metrics_root']/'batches'
    before = b.read_json(directory/'batch_005_starting_state.json')
    b.verify_previous_state(root, '005', full=True)
    source = pd.read_csv(directory.parent/'full_candidate_download_manifest.csv')
    assigned = [s for e in before['prior_ledger_entries'] for s in b.verify_frozen(directory,e['batch_id'])[1]['subject_ids']]
    _, expected = remainder_membership(source, before['completed_subjects'], assigned)
    plan = b.freeze_batch(directory, '005', expected['declared_bytes'], source,
        before['completed_subjects'], git_head(root), b.frozen_hashes(root), final_remainder=True)
    audit = coverage_record(root)
    audit_path = directory/'batch_005_coverage_audit.json'
    if audit_path.exists() and b.read_json(audit_path) != audit:
        raise ValueError('Existing coverage audit is immutable')
    if not audit_path.exists():
        write_json(audit_path, audit)
    inventory = pd.DataFrame([dict(order=i, **row) for i,row in enumerate(plan['subjects'],1)])
    path = directory/'batch_005_final_remainder_inventory.csv'
    if path.exists():
        if pd.read_csv(path).to_dict('records') != inventory.to_dict('records'):
            raise ValueError('Existing final remainder inventory changed')
    else:
        b.csv_write(path, inventory)
    p = b.paths(root)
    frame, _, entry = b.verify_frozen(directory,'005')
    report = b.network_preflight(frame,plan,p['raw_root'],p['output_root'])
    report.update(**b.frozen_hashes(root),manifest_sha256=entry['manifest_sha256'],
        plan_sha256=entry['plan_sha256'],plan_file_sha256=entry['plan_file_sha256'],
        coverage_audit_passed=True,coverage_audit_sha256=sha256_file(audit_path),
        mode='final_remainder',future_batch_allowed=False)
    if not (directory/'batch_005_execution_authorization.json').exists():
        write_json(directory/'batch_005_network_preflight.json',report)
    return audit


def require_full_completion(candidates, completed, batch_summary, parts):
    if len(candidates) != 211 or len(completed) != 211 or len(set(completed)) != 211 or set(candidates) != set(completed):
        raise ValueError('Final completion requires 211 unique verified outputs without omissions')
    if batch_summary['subjects_completed'] != 65 or batch_summary['technical_failures'] or parts:
        raise ValueError('Final remainder still has unresolved outputs/downloads/partials')


def single_active_subject(raw_root, current_subject, retained_subjects):
    unexpected = [p.name for p in Path(raw_root).iterdir()
                  if p.is_dir() and p.name not in set(retained_subjects) | {current_subject}]
    if unexpected:
        raise ValueError(f'More than one active raw subject or unexplained staging: {unexpected}')


def verify_source_provenance(receipt, source_rows, metadata_directory):
    if receipt['source_manifest_rows'] != source_rows:
        raise ValueError('Receipt source manifest differs from authoritative source')
    for name,digest in receipt['source_sidecar_sha256'].items():
        path=Path(metadata_directory)/name
        if path.parent.resolve()!=Path(metadata_directory).resolve() or sha256_file(path)!=digest:
            raise ValueError('Source metadata sidecar identity changed')
    if not receipt['source_hash_verified'] or receipt['fit_parameters_from']!='Train_only' or receipt['heldout_parameters_are']!='apply_only':
        raise ValueError('Source verification or Train/apply-only provenance failed')


def completion_audit(root, *, write=False):
    """Reopen/check containers and provenance; no native waveform reprocessing."""
    p = b.paths(root); directory = p['metrics_root']/'batches'; runtime = b.runtime_directory(root)
    coverage = validate_coverage(root)
    b.verify_previous_state(root,'005',full=True)
    frame, plan, auth = b.validate_authorization(root,'005',directory)
    completed = b.verified_completed(root)
    summary = b.read_json(directory/'batch_005_summary.json')
    parts = list(p['raw_root'].rglob('*.part'))
    require_full_completion(coverage['candidate_ids'],completed,summary,parts)
    assert b.ledger_read(directory)['batches'][-1]['batch_status']=='complete'
    before = b.read_json(directory/'batch_005_starting_state.json')
    metadata = p['metadata_root']
    git = ['git','-c',f'safe.directory={metadata.as_posix()}','-C',str(metadata)]
    assert not subprocess.check_output(git+['status','--porcelain'],text=True).strip()
    assert subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()==before['metadata_head']
    source = pd.read_csv(directory.parent/'full_candidate_download_manifest.csv')
    rows=[]
    for subject in coverage['candidate_ids']:
        output=p['output_root']/f'{subject}_all_blocks.h5'
        receipt_path=output.with_suffix('.provenance.json'); receipt=b.read_json(receipt_path)
        verify_source_provenance(receipt,source.loc[source.subject_id==subject].to_dict('records'),metadata/subject/'eeg')
        raw=p['raw_root']/subject
        assert bool(raw.exists()) == (not receipt['qc_pass'])
        raw_bytes=b.directory_bytes(raw)
        for asset,target in b.st._manifest_rows(source.loc[source.subject_id==subject],p['raw_root']):
            if raw.exists():
                b.st.downloader.verify_file(target,int(asset['payload_size_bytes']),asset['payload_sha256'])
        train=p['train_output_root']/f'{subject}_preprocessed.h5'
        params,param_hash=b.ab.verified_train_parameters(train,b.load_config(p['config_path'])[0],
            b.FROZEN_SHA,receipt['authoritative_blocks'],p['metrics_root'])
        assert receipt['train_parameters']==params and receipt['train_parameter_source_hash']==param_hash
        rows.append(dict(subject_id=subject,technically_complete=True,blocks=16,
            train_qc_pass=receipt['qc_pass'],provisional_5_1_2_feasible=receipt['provisional_feasibility']['provisional_5_1_2_feasible'],
            feasibility_reason=receipt['provisional_feasibility']['reason'],
            raw_deleted=not raw.exists(),raw_retained=raw.exists(),raw_bytes_retained=raw_bytes,
            output_bytes=output.stat().st_size,output_sha256=sha256_file(output),
            receipt_sha256=sha256_file(receipt_path),train_reference_bytes=train.stat().st_size,
            train_reference_sha256=sha256_file(train)))
    qc=dict(subjects=211,train_qc_pass=[r['subject_id'] for r in rows if r['train_qc_pass']],
        train_qc_fail=[r['subject_id'] for r in rows if not r['train_qc_pass']],
        provisional_feasibility_pass=[r['subject_id'] for r in rows if r['provisional_5_1_2_feasible']],
        provisional_feasibility_fail=[r['subject_id'] for r in rows if not r['provisional_5_1_2_feasible']])
    assert {p.name for p in p['raw_root'].iterdir() if p.is_dir()}==set(qc['train_qc_fail'])
    cumulative_freed=sum(b.read_json(directory/f'batch_{i:03}_summary.json')['raw_bytes_freed'] for i in range(1,6))
    # Pilot field naming is preserved in its original immutable summary.
    pilot_cleanup=pd.read_csv(directory/'pilot_cleanup_execution.csv')
    cumulative_freed += int(pilot_cleanup.loc[pilot_cleanup.raw_deleted,'bytes_freed'].sum())
    storage=dict(raw_deleted_ids=[r['subject_id'] for r in rows if r['raw_deleted']],
        raw_retained_ids=[r['subject_id'] for r in rows if r['raw_retained']],
        raw_bytes_freed=cumulative_freed,raw_retained_bytes=b.directory_bytes(p['raw_root']),
        all_block_hdf5_bytes=sum(r['output_bytes'] for r in rows),
        train_reference_hdf5_bytes=sum(r['train_reference_bytes'] for r in rows),
        receipt_bytes=sum(f.stat().st_size for f in p['output_root'].parent.rglob('*.provenance.json')),
        total_processed_bytes=b.directory_bytes(p['output_root'].parent),
        batch_005_compact_added_bytes=summary['compact_bytes_added'],
        batch_005_raw_bytes_freed=summary['raw_bytes_freed'])
    events=b.read_journal(runtime/'batch_005_download_journal.jsonl')
    recovery=[r for r in events if r['status']=='connectivity_restored']
    network=dict(expected_bytes=auth['expected_new_network_bytes'],
        measured_response_body_bytes=sum(r.get('received_body_bytes',0) for r in events),
        resumed_prefix_bytes=sum(r.get('resumed_prefix_bytes',0) for r in events if r['status']=='response_validated'),
        outage_count=sum(r['status']=='waiting_for_network' for r in events),
        outage_wait_seconds=sum(r.get('outage_seconds',0) for r in recovery),
        measurement='Application response-body bytes; HTTP/TLS/ISP overhead unmeasured')
    audit=dict(state='FULL_COHORT_STREAMING_AWAITING_FINAL_ACCEPTANCE',timestamp_utc=utc_now(),
        unique_technically_complete_subjects=211,technical_completion_count=211,expected_blocks=3376,
        verified_blocks=sum(r['blocks'] for r in rows),missing_subjects=[],duplicate_subjects=[],
        remaining_unassigned_subjects=[],no_batch_006=True,source_receipt_hash_verification=True,
        prior_protected_artifacts_verified=len(before['files']),metadata_clone_clean=True,
        frozen_hashes=b.frozen_hashes(root),scientific_heldout_analysis=False,normalization_fitted=False,
        final_cohort_locked=False,final_split_locked=False,git_push_performed=False,
        qc=qc,storage=storage,network=network,coverage_audit_sha256=sha256_file(directory/'batch_005_coverage_audit.json'))
    if write:
        b.csv_write(directory/'full_cohort_streaming_subject_manifest.csv',pd.DataFrame(rows))
        write_json(directory/'full_cohort_streaming_qc_summary.json',qc)
        write_json(directory/'full_cohort_streaming_storage_summary.json',storage)
        write_json(directory/'full_cohort_streaming_completion_audit.json',audit)
    return audit
