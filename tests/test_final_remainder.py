"""Offline final remainder reconciliation, boundaries, heartbeat, and completion."""
from pathlib import Path
import urllib.error
import pandas as pd
import pytest

from resting_eeg.data import batches as b, final_remainder as f, autonomous as a, network_wait as nw

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    directory=ROOT/'outputs/metrics/scenario2/batches'
    source=pd.read_csv(directory.parent/'full_candidate_download_manifest.csv')
    before=b.read_json(directory/'batch_005_starting_state.json')
    assigned=[s for i in range(1,5) for s in b.read_json(directory/f'batch_{i:03}_plan.json')['subject_ids']]
    return source,before,assigned


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    monkeypatch.setattr(nw.urllib.request,'urlopen',lambda *a,**k:pytest.fail('Real EEG request forbidden'))


def test_exact_all_remaining_coverage_order_and_retained_exclusion(evidence):
    source,before,assigned=evidence
    selected,plan=f.remainder_membership(source,before['completed_subjects'],assigned)
    expected=[s for s in source.subject_id.drop_duplicates() if s not in before['completed_subjects']]
    assert plan['subject_ids']==expected
    assert plan['subject_count']==65 and len(selected)==130
    assert plan['declared_bytes']==plan['cap_bytes']==42976220335
    assert plan['unused_capacity_bytes']==0 and plan['first_not_included'] is None
    assert plan['subject_ids'][0]=='sub-010242'
    assert plan['subject_ids'][-1]==source.subject_id.iloc[-1]
    assert not set(expected)&set(before['retained_raw_subjects'])
    assert not set(expected)&f.STRUCTURAL_EXCLUSIONS
    assert len(before['completed_subjects']+expected)==len(set(before['completed_subjects']+expected))==211


@pytest.mark.parametrize('mutation',['bytes','missing','duplicate_asset','duplicate_assignment','incomplete_prior','exclusion','completed_duplicate'])
def test_reconciliation_contradictions_fail_before_network(evidence,mutation):
    source,before,assigned=evidence
    completed=before['completed_subjects'].copy()
    if mutation=='bytes':source.loc[source.subject_id=='sub-010242','payload_size_bytes']+=1
    if mutation=='missing':source=source.loc[source.subject_id!='sub-010321']
    if mutation=='duplicate_asset':source=pd.concat([source,source.iloc[:1]])
    if mutation=='duplicate_assignment':assigned.append(assigned[0])
    if mutation=='incomplete_prior':assigned.append('sub-010242')
    if mutation=='exclusion':source.loc[source.subject_id=='sub-010242','subject_id']='sub-010015'
    if mutation=='completed_duplicate':completed[-1]=completed[0]
    with pytest.raises(ValueError):f.remainder_membership(source,completed,assigned)


def test_freeze_exact_cap_immutable_prior_records_and_no_next(tmp_path,evidence):
    source,before,assigned=evidence
    import shutil
    directory=ROOT/'outputs/metrics/scenario2/batches'
    for pattern in ('batch_00[1234]_manifest.csv','batch_00[1234]_plan.json','batch_ledger.json'):
        for path in directory.glob(pattern):shutil.copyfile(path,tmp_path/path.name)
    # The real ledger can already contain Batch 005 when tests run after freezing.
    ledger=b.ledger_read(tmp_path);ledger['batches']=ledger['batches'][:4]
    b.write_json(tmp_path/'batch_ledger.json',ledger)
    old={p.name:p.read_bytes() for p in tmp_path.glob('batch_00[1234]*')}
    with pytest.raises(ValueError,match='requires final_remainder'):
        b.freeze_batch(tmp_path,'005',42976220335,source,before['completed_subjects'],'test',{})
    with pytest.raises(ValueError,match='exact declared total'):
        b.freeze_batch(tmp_path,'005',41*b.GIB,source,before['completed_subjects'],'test',{},final_remainder=True)
    plan=b.freeze_batch(tmp_path,'005',42976220335,source,before['completed_subjects'],'test',{},final_remainder=True)
    assert plan['final_remainder'] and not plan['future_batch_allowed']
    assert b.ledger_read(tmp_path)['batches'][:4]==ledger['batches']
    assert all((tmp_path/name).read_bytes()==data for name,data in old.items())
    assert b.freeze_batch(tmp_path,'005',42976220335,source,[],'later',{},final_remainder=True)==plan
    with pytest.raises(ValueError,match='prohibited'):
        b.freeze_batch(tmp_path,'006',100,source,[],'test',{})
    assert not list(tmp_path.glob('batch_006*'))


def test_final_network_total_is_not_simultaneous_disk_need(tmp_path,monkeypatch,evidence):
    source,before,assigned=evidence
    frame,plan=f.remainder_membership(source,before['completed_subjects'],assigned);plan['batch_id']='005'
    monkeypatch.setattr(b.st,'_free_bytes',lambda *a:4*b.GIB)
    report=b.network_preflight(frame,plan,tmp_path,tmp_path)
    assert report['declared_bytes']>40*b.GIB and report['required_free_bytes']<4*b.GIB and report['disk_ok']


def test_only_current_staging_and_completed_qc_failure_raw_allowed(tmp_path):
    for name in ('retained','current'):(tmp_path/name).mkdir()
    f.single_active_subject(tmp_path,'current',['retained'])
    (tmp_path/'next').mkdir()
    with pytest.raises(ValueError,match='one active'):f.single_active_subject(tmp_path,'current',['retained'])


def test_batch005_heartbeat_during_network_wait(tmp_path):
    import time
    plan=dict(batch_id='005',subject_ids=['sub-001'],manifest_sha256='frozen')
    with a.WorkerState(tmp_path,tmp_path,plan,[],0,interval=.02) as state:
        state.update(state='waiting_for_network',current_subject='sub-001',expected_asset_bytes=100,
            partial_bytes=25,consecutive_connectivity_failures=4)
        time.sleep(.07)
        first=b.read_json(tmp_path/'batch_005_autoresume_worker_state.json')
        time.sleep(.05)
        second=b.read_json(tmp_path/'batch_005_autoresume_worker_state.json')
    assert first['timestamp_utc']!=second['timestamp_utc']
    assert second['state']=='waiting_for_network' and second['batch_id']=='005'
    assert second['subjects_completed']==0 and second['subjects_remaining']==1
    assert second['asset_progress_percent']==25 and second['connectivity_failures']==4
    assert second['manifest_sha256']=='frozen'


@pytest.mark.parametrize('code',[408,425,429,500,502,503,504])
def test_retry_after_honored_for_all_transient_http(code):
    error=urllib.error.HTTPError('https://synthetic',code,'temporary',{'Retry-After':'600'},None)
    assert nw.retry_after(error)==600


@pytest.mark.parametrize('count,failed,parts',[(210,False,[]),(211,True,[]),(211,False,['preserved.part'])])
def test_final_completion_never_accepted_early(count,failed,parts):
    candidates=[f'sub-{i}' for i in range(211)]
    with pytest.raises(ValueError):f.require_full_completion(candidates,candidates[:count],dict(subjects_completed=65,technical_failures=['failure'] if failed else []),parts)


def test_final_completion_accepts_exact_211_and_65():
    ids=[f'sub-{i}' for i in range(211)]
    f.require_full_completion(ids,ids,dict(subjects_completed=65,technical_failures=[]),[])


def test_batch005_supervisor_mismatched_exit_is_hard_blocked():
    actual=a.supervisor_heartbeat(dict(runner_pid=1,batch_id='005'),{},2,'token',0)
    assert actual['project_state']=='FINAL_REMAINDER_HARD_BLOCKED'


@pytest.mark.parametrize('mutation',['source','sidecar','fit','heldout'])
def test_source_and_train_provenance_contradictions_rejected(tmp_path,mutation):
    sidecar=tmp_path/'sidecar.tsv';sidecar.write_text('synthetic metadata')
    receipt=dict(source_manifest_rows=[dict(payload_sha256='expected')],
        source_sidecar_sha256={'sidecar.tsv':b.sha256_file(sidecar)},source_hash_verified=True,
        fit_parameters_from='Train_only',heldout_parameters_are='apply_only')
    rows=[dict(payload_sha256='expected')]
    f.verify_source_provenance(receipt,rows,tmp_path)
    if mutation=='source':rows[0]['payload_sha256']='changed'
    if mutation=='sidecar':sidecar.write_text('changed')
    if mutation=='fit':receipt['fit_parameters_from']='Validation'
    if mutation=='heldout':receipt['heldout_parameters_are']='refit'
    with pytest.raises(ValueError):f.verify_source_provenance(receipt,rows,tmp_path)
