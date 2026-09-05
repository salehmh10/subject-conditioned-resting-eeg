"""Offline 45-GiB membership, streaming disk accounting, and durable progress."""
from pathlib import Path
import pandas as pd
import pytest
from resting_eeg.data import batches as b
from test_batches import manifest

ROOT=Path(__file__).resolve().parents[1]


def test_batch004_contiguous_exact_cap_and_prior_evidence_immutable(tmp_path):
    directory=ROOT/'outputs/metrics/scenario2/batches'
    source=pd.read_csv(directory.parent/'full_candidate_download_manifest.csv')
    before=b.read_json(directory/'batch_004_starting_state.json')
    prior=[b.read_json(directory/f'batch_{i:03}_plan.json') for i in range(1,4)]
    assigned=[s for p in prior for s in p['subject_ids']]
    pilot=[s for s in before['completed_subjects'] if s not in assigned]
    complete=list(pilot)
    for i,p in enumerate(prior,1):
        b.freeze_batch(tmp_path,f'{i:03}',p['cap_bytes'],source,complete,'synthetic',{})
        complete+=p['subject_ids']
    old={f.name:f.read_bytes() for f in tmp_path.glob('batch_00[123]*') if f.is_file()}
    entries=b.ledger_read(tmp_path)['batches']
    cap=b.cap_bytes('45')
    assert cap==48318382080
    plan=b.freeze_batch(tmp_path,'004',cap,source,before['completed_subjects'],'synthetic',{})
    remaining=[s for s in source.subject_id.drop_duplicates() if s not in set(complete)]
    assert remaining[0]==plan['subject_ids'][0]=='sub-010075'
    assert plan['subject_count']==74 and plan['declared_bytes']==48277992566
    assert plan['unused_capacity_bytes']==40389514
    assert plan['subject_ids']==remaining[:plan['subject_count']]
    assert len(set(plan['subject_ids']))==plan['subject_count']
    assert plan['asset_count']==2*plan['subject_count']
    assert not set(plan['subject_ids']) & set(before['completed_subjects'])
    assert not set(plan['subject_ids']) & {'sub-010012','sub-010030','sub-010250'}
    assert plan['first_not_included']['subject_id']==remaining[plan['subject_count']]
    assert plan['first_not_included']['subject_id']=='sub-010242'
    assert plan['declared_bytes']<=cap<plan['declared_bytes']+plan['first_not_included']['declared_bytes']
    assert all((tmp_path/n).read_bytes()==data for n,data in old.items())
    assert b.ledger_read(tmp_path)['batches'][:3]==entries
    assert not list(tmp_path.glob('batch_005*'))
    assert b.freeze_batch(tmp_path,'004',cap,source,remaining,'different-head',{})==plan
    with pytest.raises(ValueError,match='cap cannot change'):
        b.freeze_batch(tmp_path,'004',cap-1,source,[],'synthetic',{})


@pytest.mark.parametrize('mutation',['001','002','003','005'])
def test_batch004_prior_ledger_and_future_boundary(tmp_path,monkeypatch,mutation):
    directory=tmp_path/'batches';directory.mkdir()
    prior=[dict(batch_id=f'{i:03}',batch_status='complete') for i in range(1,4)]
    b.write_json(directory/'batch_ledger.json',dict(batches=prior))
    b.write_json(directory/'batch_004_starting_state.json',dict(prior_ledger_entries=prior,files=[],registries={},completed_subjects=[]))
    monkeypatch.setattr(b,'paths',lambda root:dict(metrics_root=tmp_path))
    monkeypatch.setattr(b,'runtime_directory',lambda root:tmp_path/'runtime')
    assert b.verify_previous_state(tmp_path,'004',full=True)
    if mutation=='005':
        (directory/'batch_005_plan.json').write_text('prohibited synthetic artifact')
    else:
        prior[int(mutation)-1]['batch_status']='changed'
        b.write_json(directory/'batch_ledger.json',dict(batches=prior))
    with pytest.raises(ValueError):b.verify_previous_state(tmp_path,'004',full=True)


def test_45_gib_network_cap_does_not_require_simultaneous_disk(tmp_path,monkeypatch):
    source=pd.read_csv(ROOT/'outputs/metrics/scenario2/full_candidate_download_manifest.csv')
    frame,plan=b.plan_membership(source,b.cap_bytes('45'))
    plan['batch_id']='004'
    monkeypatch.setattr(b.st,'_free_bytes',lambda p:4*b.GIB)
    report=b.network_preflight(frame,plan,tmp_path,tmp_path)
    assert report['declared_bytes']>40*b.GIB and report['disk_ok']
    assert report['required_free_bytes']==report['largest_subject_bytes']+b.st.PROCESSING_OVERHEAD_BYTES+b.st.COMPACT_OUTPUT_ALLOWANCE_BYTES+b.st.SAFETY_MARGIN_BYTES
    assert report['expected_new_network_bytes']==report['declared_bytes']
    monkeypatch.setattr(b.st,'_free_bytes',lambda p:report['required_free_bytes']-1)
    with pytest.raises(ValueError,match='disk guard'):
        b.network_preflight(frame,plan,tmp_path,tmp_path)


def test_existing_raw_and_prefix_reduce_only_additional_space(tmp_path):
    frame,plan=b.plan_membership(manifest(),100);plan['batch_id']='004'
    rows=b.st._manifest_rows(frame,tmp_path)
    target=rows[2][1];target.parent.mkdir(parents=True)
    target.with_name(target.name+'.part').write_bytes(b'x'*30)
    retained=tmp_path/'sub-retained';retained.mkdir();(retained/'evidence').write_bytes(b'x'*100)
    report=b.network_preflight(frame,plan,tmp_path,tmp_path)
    assert report['remaining_payload_bytes_by_subject']=={'sub-001':40,'sub-002':20}
    assert report['retained_raw_bytes']==130
    assert report['required_free_bytes']==40+b.st.PROCESSING_OVERHEAD_BYTES+b.st.COMPACT_OUTPUT_ALLOWANCE_BYTES+b.st.SAFETY_MARGIN_BYTES
    assert report['peak_footprint_including_retained_raw_bytes']==130+report['required_free_bytes']


def test_checkpoints_preserve_terminal_failure_and_exact_measured_counts(tmp_path,monkeypatch):
    frame,plan=b.plan_membership(manifest((10,)*11),120);plan['batch_id']='004'
    runtime=tmp_path/'runtime';raw=tmp_path/'raw';raw.mkdir();(raw/'retained').write_bytes(b'x'*7)
    runtime.mkdir()
    monkeypatch.setattr(b,'paths',lambda root:dict(raw_root=raw))
    monkeypatch.setattr(b,'runtime_directory',lambda root:runtime)
    monkeypatch.setattr(b,'verify_frozen',lambda *a:(frame,plan,{'manifest_sha256':'frozen'}))
    for i,subject in enumerate(plan['subject_ids'][:10]):
        result=dict(output_bytes=3,qc_pass=i!=1)
        b.st.append_status(runtime/'batch_004_streaming_journal.jsonl','run',subject,
            'complete' if i<9 else 'download_interrupted',result=result)
        if i==0:
            b.st.append_status(runtime/'batch_004_streaming_journal.jsonl','run',subject,'raw_deleted',bytes_freed=10)
        b.st.append_status(runtime/'batch_004_download_journal.jsonl','run',subject,'response_body',received_body_bytes=5)
        record=b.progress_checkpoint(tmp_path,'004',tmp_path,subject)
    assert record['terminal_count']==record['attempted_count']==10
    assert record['completed_count']==9 and record['technical_failed_count']==1 and record['qc_failed_count']==1
    assert record['measured_response_body_bytes']==50 and record['raw_deleted_bytes']==10
    assert record['raw_retained_bytes']==7 and record['compact_output_bytes']==27
    assert record['next_subject']=='sub-011'
    assert b.read_json(runtime/'batch_004_progress_010.json')==record
    assert len(b.read_journal(runtime/'batch_004_progress_journal.jsonl'))==10
    final=b.progress_checkpoint(tmp_path,'004',tmp_path,final=True)
    assert b.read_json(runtime/'batch_004_progress_final.json')==final and final['final']
    assert not list(tmp_path.rglob('batch_005*'))
