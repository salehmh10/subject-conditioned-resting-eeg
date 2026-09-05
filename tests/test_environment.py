"""D-only runtime policy, system-drive guard, and safe interrupted lifecycle."""
from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
import pytest
from resting_eeg.data import environment as env, streaming as st
from resting_eeg.data.paths import load_paths

ROOT=Path(__file__).resolve().parents[1]


def test_real_project_environment_and_pytest_basetemp(pytestconfig):
    paths=load_paths(ROOT)
    assert Path(tempfile.gettempdir()).resolve()==paths['temp_root']
    assert Path(pytestconfig.option.basetemp).resolve()==paths['temp_root']/'pytest'
    for key in ('raw_root','processed_root','runtime_root','temp_root'):
        assert paths[key].drive.upper()=='D:'
    for key,value in env.environment_values(paths).items():
        assert os.environ[key]==value


@pytest.mark.parametrize('free',[0,env.MINIMUM_SYSTEM_FREE_BYTES-1])
def test_low_system_drive_guard(free,tmp_path,monkeypatch):
    monkeypatch.setenv('SCENARIO2_RUNTIME_ROOT',str(tmp_path/'runtime'))
    with pytest.raises(env.SystemDriveLowSpaceError,match='c_drive_low_space_guard'):
        env.system_drive_guard('test',windows=True,disk_usage=lambda p:SimpleNamespace(free=free))
    assert (tmp_path/'runtime/c_drive_low_space_guard.json').is_file()


def test_system_drive_threshold_exact_boundary():
    result=env.system_drive_guard('test',windows=True,disk_usage=lambda p:SimpleNamespace(free=env.MINIMUM_SYSTEM_FREE_BYTES))
    assert result['free_bytes']==3*1073741824


def test_low_space_preserves_partial_without_request(tmp_path,monkeypatch):
    from test_download_resume import asset,PAYLOAD
    target=tmp_path/'source.eeg';part=target.with_name(target.name+'.part');part.write_bytes(PAYLOAD[:12])
    def low(*a,**k):raise env.SystemDriveLowSpaceError('c_drive_low_space_guard')
    monkeypatch.setattr(env,'system_drive_guard',low)
    monkeypatch.setattr(st.downloader.urllib.request,'urlopen',lambda *a,**k:pytest.fail('network forbidden'))
    journal=st.downloader.DownloadJournal(tmp_path/'journal.jsonl')
    with pytest.raises(env.SystemDriveLowSpaceError):st.downloader.fetch_one(asset(),target,journal=journal)
    assert part.read_bytes()==PAYLOAD[:12] and not target.exists()
    assert 'c_drive_low_space_guard' in journal.path.read_text()


def test_stream_guard_stops_before_next_subject(staged,monkeypatch):
    frame,paths,calls=staged
    def low(*a,**k):raise env.SystemDriveLowSpaceError('c_drive_low_space_guard')
    monkeypatch.setattr(st,'system_drive_guard',low)
    records=st.stream_subjects(frame,**paths,execute=True,local_existing_only=True,continue_on_error=True)
    assert len(records)==1 and records[0]['status']=='c_drive_low_space_guard'
    assert not calls and (paths['raw_root']/'sub-001').is_dir()


def test_guard_after_download_preserves_raw(staged,monkeypatch):
    frame,paths,calls=staged
    def guard(stage):
        if stage=='start_subject_preprocessing':raise env.SystemDriveLowSpaceError('c_drive_low_space_guard')
    monkeypatch.setattr(st,'system_drive_guard',guard)
    records=st.stream_subjects(frame,**paths,execute=True,local_existing_only=True,continue_on_error=True)
    assert records[0]['status']=='c_drive_low_space_guard' and not calls
    assert (paths['raw_root']/'sub-001').is_dir()


# Reuse the established synthetic streaming fixture without real waveform I/O.
from test_streaming import staged


def test_resume_authorization_preserves_freeze(tmp_path,monkeypatch):
    from resting_eeg.data import batches as b
    from test_batches import manifest
    hashes=dict(preprocessing_config_sha256=b.FROZEN_SHA,mechanical_access_config_sha256=b.ACCESS_SHA,variable_batch_config_sha256='config')
    b.freeze_batch(tmp_path,'001',100,manifest(),[],'original',hashes)
    frame,plan,entry=b.verify_frozen(tmp_path,'001')
    p=dict(raw_root=tmp_path/'raw',output_root=tmp_path/'processed')
    monkeypatch.setattr(b,'paths',lambda *a:p)
    monkeypatch.setattr(b,'frozen_hashes',lambda *a:hashes)
    monkeypatch.setattr(b,'require_committed_clean',lambda *a:'hardened')
    monkeypatch.setattr(b,'verified_completed',lambda *a:['sub-001'])
    monkeypatch.setattr(env,'configure_environment',lambda *a:{'tempfile_gettempdir':'D:/synthetic'})
    selected=[]
    def preflight(frame,remaining_plan,*a):
        selected.extend(frame.subject_id.unique())
        assert remaining_plan['subject_ids']==['sub-002']
        assert remaining_plan['subject_count']==1
        return {'valid_partial_bytes':12,'expected_new_network_bytes':38}
    monkeypatch.setattr(b,'network_preflight',preflight)
    b.write_json(tmp_path/'batch_001_network_preflight.json',{})
    auth={k:plan[k] for k in ('batch_id','cap_bytes','declared_bytes','subject_ids')}
    auth.update(hashes,manifest_sha256=entry['manifest_sha256'],plan_sha256=entry['plan_sha256'],
        authorization_code_head='original',expected_new_network_bytes=90,authorization_utc='2026-01-01T00:00:00Z',
        resolved_paths={k:str(v) for k,v in p.items()},network_preflight_sha256=b.sha256_file(tmp_path/'batch_001_network_preflight.json'))
    b.write_json(tmp_path/'batch_001_execution_authorization.json',auth)
    protected=['batch_001_manifest.csv','batch_001_plan.json','batch_ledger.json','batch_001_execution_authorization.json']
    before={f:b.sha256_file(tmp_path/f) for f in protected}
    resumed=b.authorize_resume(tmp_path,'001',tmp_path)
    assert resumed['authorization_code_head']=='hardened'
    assert resumed['completed_subjects_reverified']==['sub-001'] and selected==['sub-002']
    assert resumed['unresolved_subject_ids']==['sub-002']
    assert before=={f:b.sha256_file(tmp_path/f) for f in protected}
    assert b.validate_authorization(tmp_path,'001',tmp_path)[2]['authorization_code_head']=='hardened'
    assert not (tmp_path/'batch_002_manifest.csv').exists()
