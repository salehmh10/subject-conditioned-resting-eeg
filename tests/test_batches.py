"""Network-free variable batch, immutable authorization, and accounting checks."""
from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest

from resting_eeg.data import batches as b


def manifest(sizes=(40, 50, 20, 30)):
    rows = []
    for i, size in enumerate(sizes, 1):
        subject = f'sub-{i:03}'
        for suffix, role, n in [('.eeg', 'waveform', size-1), ('.vhdr', 'brainvision_header', 1)]:
            rows.append(dict(subject_id=subject, file_role=role,
                relative_path=f'{subject}/eeg/{subject}_task-resting_eeg{suffix}',
                payload_size_bytes=n, payload_sha256=hashlib.sha256(b'x'*n).hexdigest(),
                source_url='https://nemar.s3.us-east-2.amazonaws.com/synthetic-test-only'))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(b.st.downloader.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network forbidden'))


@pytest.mark.parametrize('gib,expected', [('1',1073741824),('5',5368709120),('7',7516192768),('10',10737418240),('0.5',536870912)])
def test_exact_variable_caps(gib, expected):
    assert b.cap_bytes(gib) == expected


@pytest.mark.parametrize('value', [None, 'NaN', 'Infinity', '-1', '0', '0.0000000001'])
def test_invalid_caps(value):
    with pytest.raises(ValueError): b.cap_bytes(value)


def test_authoritative_contiguous_no_packing():
    f, p = b.plan_membership(manifest(), 100)
    assert p['subject_ids'] == ['sub-001','sub-002']
    assert p['declared_bytes'] == 90 and p['unused_capacity_bytes'] == 10
    assert p['first_not_included'] == dict(subject_id='sub-003',declared_bytes=20,reason='next_subject_would_exceed_batch_cap')
    assert len(f) == 4


def test_do_not_skip_large_next():
    _, p = b.plan_membership(manifest((40,80,20)), 100)
    assert p['subject_ids'] == ['sub-001']


def test_preserve_nonsorted_authoritative_order():
    frame = manifest().iloc[[4,5,0,1,2,3,6,7]].reset_index(drop=True)
    _, p = b.plan_membership(frame, 100)
    assert p['subject_ids'] == ['sub-003','sub-001']


def test_completed_pilot_and_assigned_excluded():
    _, p = b.plan_membership(manifest(), 100, completed=['sub-001'], assigned=['sub-002'])
    assert p['subject_ids'] == ['sub-003','sub-004']


def test_no_duplicate_assets():
    f = manifest()
    with pytest.raises(ValueError): b.plan_membership(pd.concat([f,f.iloc[:1]]), 100)


def test_single_subject_over_cap():
    with pytest.raises(ValueError, match='Single subject'): b.plan_membership(manifest((110,20)), 100)


def freeze(tmp_path, cap=100):
    return b.freeze_batch(tmp_path,'001',cap,manifest(),[],'head',{})


def test_freeze_and_rerun_idempotent(tmp_path):
    plan = freeze(tmp_path)
    digest = b.sha256_file(tmp_path/'batch_001_manifest.csv')
    assert b.freeze_batch(tmp_path,'001',100,manifest(),['sub-001'],'new-head',{}) == plan
    assert digest == b.sha256_file(tmp_path/'batch_001_manifest.csv')
    assert len(b.ledger_read(tmp_path)['batches']) == 1
    assert not (tmp_path/'batch_002_manifest.csv').exists()


def test_future_cap_may_differ_no_duplicate_subjects(tmp_path):
    first = freeze(tmp_path)
    second = b.freeze_batch(tmp_path,'002',60,manifest(),[],'head',{})
    assert second['cap_bytes'] == 60 and first['cap_bytes'] == 100
    assert second['subject_ids'] == ['sub-003','sub-004']
    assert not set(first['subject_ids']) & set(second['subject_ids'])


def test_frozen_cap_cannot_change(tmp_path):
    freeze(tmp_path)
    with pytest.raises(ValueError, match='cap cannot change'): freeze(tmp_path, 99)
    with pytest.raises(ValueError, match='cap cannot change'): b.verify_frozen(tmp_path,'001',requested_cap=99)


def test_frozen_membership_cannot_change(tmp_path):
    freeze(tmp_path)
    path = tmp_path/'batch_001_manifest.csv'
    path.write_text(path.read_text().replace('sub-001','sub-999'))
    with pytest.raises(ValueError, match='manifest changed'): b.verify_frozen(tmp_path,'001')


def test_frozen_plan_cap_cannot_change(tmp_path):
    freeze(tmp_path)
    path = tmp_path/'batch_001_plan.json'
    data = b.read_json(path); data['cap_bytes'] = 200; b.write_json(path,data)
    with pytest.raises(ValueError, match='plan/cap changed'): b.verify_frozen(tmp_path,'001')


def test_unfrozen_refused(tmp_path):
    with pytest.raises(ValueError, match='not frozen'): b.verify_frozen(tmp_path,'001')


def test_changed_manifest_path_refused(tmp_path):
    freeze(tmp_path)
    with pytest.raises(ValueError, match='manifest path'): b.verify_frozen(tmp_path,'001',tmp_path/'other.csv')


def test_immutable_ledger_fields(tmp_path):
    freeze(tmp_path)
    with pytest.raises(ValueError,match='Immutable'): b.update_ledger(tmp_path,'001',cap_bytes=200)


def test_local_and_partial_bytes_affect_transfer_only(tmp_path):
    frame, plan = b.plan_membership(manifest(),100)
    plan['batch_id'] = '001'
    before = b.network_preflight(frame,plan,tmp_path,tmp_path)
    rows = b.st._manifest_rows(frame,tmp_path)
    rows[0][1].parent.mkdir(parents=True)
    rows[0][1].write_bytes(b'x'*39)
    rows[1][1].with_name(rows[1][1].name+'.part').write_bytes(b'x')
    rows[2][1].parent.mkdir(parents=True)
    rows[2][1].with_name(rows[2][1].name+'.part').write_bytes(b'x'*10)
    after = b.network_preflight(frame,plan,tmp_path,tmp_path)
    assert before['expected_new_network_bytes'] == 90
    assert after['verified_existing_bytes'] == 39 and after['valid_partial_bytes'] == 11
    assert after['expected_new_network_bytes'] == 40
    assert after['declared_bytes'] == 39 + 11 + 40
    assert b.plan_membership(manifest(),100)[1]['subject_ids'] == plan['subject_ids']


def test_corrupt_existing_refused(tmp_path):
    frame, plan = b.plan_membership(manifest(),100); plan['batch_id']='001'
    _, target = b.st._manifest_rows(frame,tmp_path)[0]
    target.parent.mkdir(parents=True); target.write_bytes(b'y'*39)
    with pytest.raises(ValueError): b.network_preflight(frame,plan,tmp_path,tmp_path)


def test_disk_guard(tmp_path,monkeypatch):
    frame, plan = b.plan_membership(manifest(),100); plan['batch_id']='001'
    monkeypatch.setattr(b.st,'_free_bytes',lambda p:0)
    with pytest.raises(ValueError,match='disk guard'): b.network_preflight(frame,plan,tmp_path,tmp_path)


def execution_mocks(tmp_path,monkeypatch,fail=False):
    freeze(tmp_path)
    frame, plan, _ = b.verify_frozen(tmp_path,'001')
    auth = {'authorization_code_head':'head'}
    monkeypatch.setattr(b,'validate_authorization',lambda *a:(frame,plan,auth))
    monkeypatch.setattr(b,'require_committed_clean',lambda *a:'head')
    monkeypatch.setattr(b,'paths',lambda *a:dict(raw_root=tmp_path/'raw',output_root=tmp_path/'out',config_path=tmp_path/'config'))
    monkeypatch.setattr(b,'runtime_directory',lambda *a:tmp_path/'runtime')
    monkeypatch.setattr(b,'network_preflight',lambda *a:{})
    monkeypatch.setattr(b,'frozen_hashes',lambda *a:{})
    monkeypatch.setattr(b.st,'_verify_receipt',lambda *a:{})
    monkeypatch.setattr(b,'summarize',lambda *a:{'subjects_completed':0 if fail else 2})
    calls=[]
    def process(frame,**kwargs):
        ids=frame.subject_id.unique().tolist(); calls.extend(ids)
        assert len(ids)==1
        if fail: return [dict(subject_id=ids[0],status='preprocessing_failed',error='synthetic failure')]
        return [dict(subject_id=ids[0],status='complete',raw_retained=True,
                     result=dict(qc_pass=False,output_sha256='hash'))]
    monkeypatch.setattr(b.st,'stream_subjects',process)
    return calls


def test_execution_sequential_stops_at_batch_no_automatic_next(tmp_path,monkeypatch):
    calls=execution_mocks(tmp_path,monkeypatch)
    b.execute_batch(tmp_path,'001',tmp_path,manifest_path=tmp_path/'batch_001_manifest.csv')
    assert calls==['sub-001','sub-002']
    assert not (tmp_path/'batch_002_manifest.csv').exists()
    assert b.read_json(tmp_path/'batch_001_canary_verification.json')['passed']


def test_canary_failure_no_replacement(tmp_path,monkeypatch):
    calls=execution_mocks(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError,match='Stopped frozen batch'):
        b.execute_batch(tmp_path,'001',tmp_path,manifest_path=tmp_path/'batch_001_manifest.csv',continue_on_error=True)
    assert calls==['sub-001']
    assert b.verify_frozen(tmp_path,'001')[1]['subject_ids']==['sub-001','sub-002']
    assert not (tmp_path/'batch_002_manifest.csv').exists()


def test_outside_subject_refused(tmp_path,monkeypatch):
    calls=execution_mocks(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='outside frozen batch'):
        b.execute_batch(tmp_path,'001',tmp_path,manifest_path=tmp_path/'batch_001_manifest.csv',selected=['sub-003'])
    assert calls==[]


def test_failure_remains_assigned(tmp_path):
    freeze(tmp_path)
    b.update_ledger(tmp_path,'001',batch_status='failed_or_partial')
    next_plan=b.freeze_batch(tmp_path,'002',60,manifest(),[],'head',{})
    assert next_plan['subject_ids']==['sub-003','sub-004']
