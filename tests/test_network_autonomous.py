"""Offline recovery, progress, preservation, and supervision regressions."""
import errno
import hashlib
import http.client
import json
from pathlib import Path
import socket
import ssl
import threading
import time
import urllib.error

import pytest

from scripts.data import s2_06_download_manifest as download
from resting_eeg.data import autonomous as auto, batches as b, network_wait as nw
from test_download_resume import PAYLOAD, Response, asset, make_partial
from test_streaming import staged, journal_records

ROOT = Path(__file__).resolve().parents[1]


class NetworkResponse(Response):
    def geturl(self):
        return asset()['source_url']

    def read1(self, size=-1):
        return self.read(size)


def resumed(offset, **kwargs):
    return NetworkResponse(PAYLOAD[offset:], 206,
        {'Content-Range': f'bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}'}, **kwargs)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('Real network forbidden in recovery tests')
    monkeypatch.setattr(nw.urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(nw.socket, 'getaddrinfo', forbidden)


def install(monkeypatch, *responses):
    pending = iter(responses); calls = []
    def open_response(request, timeout):
        calls.append(request)
        response = next(pending)
        if isinstance(response, BaseException):
            raise response
        return response
    monkeypatch.setattr(nw.urllib.request, 'urlopen', open_response)
    return calls


def controller(tmp_path, monkeypatch):
    updates = []; delays = []; probes = []
    control = nw.NetworkWait(tmp_path, tmp_path/'STOP', update=lambda **kw: updates.append(kw), jitter=lambda: 0)
    monkeypatch.setattr(control, 'pause', lambda delay: delays.append(delay))
    monkeypatch.setattr(control, 'probe', lambda row, emit: probes.append(row['subject_id']))
    return control, updates, delays, probes


@pytest.mark.parametrize('failure', [socket.gaierror('getaddrinfo failed'),
    urllib.error.URLError(socket.gaierror('DNS')), TimeoutError('read timed out'),
    ConnectionResetError('reset'), ConnectionAbortedError('aborted'), ssl.SSLError('temporary handshake'),
    http.client.RemoteDisconnected('remote closed'), http.client.IncompleteRead(b'', 2)])
def test_transient_preserves_prefix_and_automatically_resumes_same_asset(tmp_path, monkeypatch, failure):
    target, part = make_partial(tmp_path, PAYLOAD[:11])
    calls = install(monkeypatch, failure, resumed(11))
    control, updates, delays, probes = controller(tmp_path, monkeypatch)
    journal = download.DownloadJournal(tmp_path/'journal.jsonl')
    download.fetch_one(asset(), target, retries=0, network_wait=control, journal=journal)
    assert target.read_bytes() == PAYLOAD and not part.exists()
    assert [r.get_header('Range') for r in calls] == ['bytes=11-', 'bytes=11-']
    assert probes == ['sub-001']*2
    records = b.read_journal(journal.path)
    waiting = next(r for r in records if r['status']=='waiting_for_network')
    assert waiting['partial_bytes'] == 11 and waiting['partial_sha256'] == hashlib.sha256(PAYLOAD[:11]).hexdigest()
    assert waiting['terminal'] is False and not any(r['status']=='failed' for r in records)
    assert any(r['status']=='connectivity_restored' for r in records)


def test_more_than_finite_retry_budget_never_advances_subject(tmp_path, monkeypatch):
    calls = install(monkeypatch, *[socket.gaierror('DNS') for _ in range(12)], NetworkResponse(PAYLOAD))
    control, updates, _, probes = controller(tmp_path, monkeypatch)
    download.fetch_one(asset(), tmp_path/'asset.eeg', retries=0, network_wait=control)
    assert len(calls)==13 and set(probes)=={'sub-001'}
    assert {u['current_subject'] for u in updates if 'current_subject' in u} == {'sub-001'}


def test_backoff_caps_indefinitely():
    assert [nw.backoff_seconds(i, lambda:0) for i in range(1,8)] == [15,30,60,120,300,300,300]
    assert nw.backoff_seconds(10**9, lambda:1) == 300


def test_failed_probes_need_two_consecutive_successes(tmp_path, monkeypatch):
    control, updates, delays, _ = controller(tmp_path, monkeypatch)
    outcomes = iter([None, socket.gaierror('flap'), None, None]); calls=[]; events=[]
    def probe(row, emit):
        calls.append(row['subject_id']); error=next(outcomes)
        if error: raise error
    monkeypatch.setattr(control,'probe',probe)
    control.recover(socket.gaierror('offline'),asset(),tmp_path/'a.part',lambda status,**kw:events.append(status))
    assert len(calls)==4 and events.count('connectivity_restored')==1
    assert delays == [15,5,30,5]


def test_head_probe_reads_zero_body_and_exact_host(tmp_path, monkeypatch):
    control=nw.NetworkWait(tmp_path,tmp_path/'STOP'); names=[]
    monkeypatch.setattr(nw.socket,'getaddrinfo',lambda name,*a,**k:names.append(name))
    response=NetworkResponse(b'',headers={'Content-Length':str(len(PAYLOAD))})
    monkeypatch.setattr(response,'read',lambda *a:pytest.fail('HEAD must not read body'))
    calls=install(monkeypatch,response)
    control.probe(asset(),lambda *a,**k:None)
    assert names==['nemar.s3.us-east-2.amazonaws.com'] and calls[0].get_method()=='HEAD'


def test_unsupported_head_probe_is_one_byte_and_measured(tmp_path,monkeypatch):
    monkeypatch.setattr(nw.socket,'getaddrinfo',lambda *a,**k:[])
    response=NetworkResponse(b'x',206,{'Content-Range':f'bytes 0-0/{len(PAYLOAD)}'})
    calls=install(monkeypatch,urllib.error.HTTPError(asset()['source_url'],405,'HEAD unsupported',{},None),response)
    events=[]; nw.NetworkWait(tmp_path,tmp_path/'STOP').probe(asset(),lambda status,**kw:events.append(kw))
    assert calls[1].get_header('Range')=='bytes=0-0'
    assert sum(e.get('received_body_bytes',0) for e in events)==1


@pytest.mark.parametrize('code',[408,425,429,500,502,503,504])
def test_http_transient_waits_and_429_honors_retry_after(tmp_path,monkeypatch,code):
    error=urllib.error.HTTPError(asset()['source_url'],code,'temporary',{'Retry-After':'777'},None)
    install(monkeypatch,error,NetworkResponse(PAYLOAD))
    control,_,delays,_=controller(tmp_path,monkeypatch)
    download.fetch_one(asset(),tmp_path/'a.eeg',network_wait=control,retries=0)
    assert delays[0]==(777 if code==429 else 15)


@pytest.mark.parametrize('code',[403,404])
def test_persistent_missing_or_forbidden_object_is_hard(tmp_path,monkeypatch,code):
    error=urllib.error.HTTPError(asset()['source_url'],code,'contradiction',{},None)
    control,_,_,_=controller(tmp_path,monkeypatch)
    monkeypatch.setattr(control,'probe',lambda *a:(_ for _ in ()).throw(error))
    with pytest.raises(nw.SourceIdentityError,match='Persistent'):
        control.recover(error,asset(),tmp_path/'part',lambda *a,**k:None)


@pytest.mark.parametrize('failure',[OSError(errno.ENOSPC,'disk full'),PermissionError('filesystem'),
    ssl.SSLCertVerificationError('bad certificate'),ValueError('invariant')])
def test_non_network_failures_do_not_enter_unlimited_wait(tmp_path,failure):
    assert not nw.transient(failure)
    with pytest.raises(type(failure)):
        nw.NetworkWait(tmp_path,tmp_path/'STOP').recover(failure,asset(),tmp_path/'part',lambda *a,**k:None)


def test_range_200_preserves_quarantine_and_never_appends(tmp_path,monkeypatch):
    target,part=make_partial(tmp_path,PAYLOAD[:11])
    control,_,_,_=controller(tmp_path,monkeypatch)
    install(monkeypatch,NetworkResponse(PAYLOAD))
    journal=download.DownloadJournal(tmp_path/'journal.jsonl')
    download.fetch_one(asset(),target,network_wait=control,journal=journal)
    assert target.read_bytes()==PAYLOAD
    assert list((tmp_path/'quarantine/sub-001').glob('*'))[0].read_bytes()==PAYLOAD[:11]
    events=b.read_journal(journal.path)
    assert next(e for e in events if e['status']=='range_ignored_restarting')['extra_required_body_bytes']==11
    assert sum(e.get('received_body_bytes',0) for e in events)==len(PAYLOAD)


def test_wrong_range_preserves_partial_without_retry(tmp_path,monkeypatch):
    target,part=make_partial(tmp_path,PAYLOAD[:11]);control,*_=controller(tmp_path,monkeypatch)
    calls=install(monkeypatch,resumed(12))
    with pytest.raises(download.DownloadProtocolError): download.fetch_one(asset(),target,network_wait=control)
    assert part.read_bytes()==PAYLOAD[:11] and len(calls)==1


def test_complete_hash_mismatch_controlled_redownload_once(tmp_path,monkeypatch):
    control,*_=controller(tmp_path,monkeypatch)
    calls=install(monkeypatch,NetworkResponse(b'x'*len(PAYLOAD)),NetworkResponse(PAYLOAD))
    target=tmp_path/'a.eeg';download.fetch_one(asset(),target,network_wait=control)
    assert target.read_bytes()==PAYLOAD and len(calls)==2


def test_repeated_complete_hash_mismatch_hard_and_evidence_retained(tmp_path,monkeypatch):
    control,*_=controller(tmp_path,monkeypatch)
    calls=install(monkeypatch,*[NetworkResponse(b'x'*len(PAYLOAD)) for _ in range(2)])
    with pytest.raises(download.DownloadIntegrityError): download.fetch_one(asset(),tmp_path/'a.eeg',network_wait=control)
    assert len(calls)==2 and len(list((tmp_path/'quarantine/sub-001').glob('*')))==2
    assert not (tmp_path/'a.eeg').exists()


def test_actual_stall_watchdog_closes_response_and_preserves_chunk(tmp_path,monkeypatch):
    target=tmp_path/'a.eeg';control,*_=controller(tmp_path,monkeypatch)
    monkeypatch.setitem(nw.POLICY,'stall_timeout_seconds',.01)
    closed=threading.Event()
    class Blocked(NetworkResponse):
        def read1(self,size):
            if self.tell()==0: return self.read(11)
            assert closed.wait(3)
            raise OSError('closed by watchdog')
        def close(self):
            closed.set();super().close()
    calls=install(monkeypatch,Blocked(PAYLOAD),resumed(11))
    journal=download.DownloadJournal(tmp_path/'journal.jsonl')
    download.fetch_one(asset(),target,network_wait=control,journal=journal)
    assert target.read_bytes()==PAYLOAD and calls[1].get_header('Range')=='bytes=11-'
    waiting=next(r for r in b.read_journal(journal.path) if r['status']=='waiting_for_network')
    assert waiting['error_type']=='TransferStalled' and waiting['partial_bytes']==11


def test_safe_stop_during_download_fsyncs_and_has_no_failure(tmp_path,monkeypatch):
    stop=tmp_path/'STOP'; control=nw.NetworkWait(tmp_path,stop)
    class Stopping(NetworkResponse):
        def read1(self,size):
            chunk=self.read(11);stop.touch();return chunk
    install(monkeypatch,Stopping(PAYLOAD))
    target=tmp_path/'a.eeg';journal=download.DownloadJournal(tmp_path/'journal.jsonl')
    with pytest.raises(nw.ManuallyPaused): download.fetch_one(asset(),target,network_wait=control,journal=journal)
    assert target.with_name('a.eeg.part').read_bytes()==PAYLOAD[:11]
    assert not any(r['status']=='failed' for r in b.read_journal(journal.path))


def test_safe_stop_during_backoff_is_responsive(tmp_path):
    stop=tmp_path/'STOP';delays=[]
    def sleep(seconds): delays.append(seconds);stop.touch()
    control=nw.NetworkWait(tmp_path,stop,sleep=sleep)
    with pytest.raises(nw.ManuallyPaused): control.pause(300)
    assert delays==[1]


def test_single_instance_os_lock_rejects_second_runner(tmp_path):
    with b.st._journal_lock(tmp_path/auto.NAME):
        with pytest.raises(RuntimeError,match='another streaming process'):
            with b.st._journal_lock(tmp_path/auto.NAME): pytest.fail('second runner acquired lock')
    with b.st._journal_lock(tmp_path/auto.NAME): pass


def test_heartbeat_advances_while_network_waits(tmp_path):
    with auto.WorkerState(tmp_path,tmp_path,{'subject_ids':['sub-001']},[],0,interval=.03) as state:
        state.update(state='waiting_for_network',current_subject='sub-001',network_state='waiting_for_network')
        state.persist();first=b.read_json(tmp_path/f'{auto.NAME}_worker_state.json')
        time.sleep(.08)
        second=b.read_json(tmp_path/f'{auto.NAME}_worker_state.json')
    assert second['timestamp_utc']>first['timestamp_utc'] and second['state']=='waiting_for_network'
    assert second['current_subject']=='sub-001' and second['unresolved_subjects']==1


def test_existing_complete_asset_reused_without_request(tmp_path,monkeypatch):
    target=tmp_path/'a.eeg';target.write_bytes(PAYLOAD);control,*_=controller(tmp_path,monkeypatch)
    assert download.fetch_one(asset(),target,network_wait=control)=='verified_existing'


def test_frozen_order_excludes_completed_and_never_adds_replacements():
    plan=b.read_json(ROOT/'outputs/metrics/scenario2/batches/batch_004_plan.json')
    remaining=auto.unresolved_in_order(plan,plan['subject_ids'][:12])
    assert len(remaining)==62 and remaining[0]=='sub-010090' and remaining[-1]=='sub-010241'
    assert 'sub-010242' not in remaining


def test_disk_guard_failure_preserves_partial_and_stops(tmp_path,monkeypatch):
    target,part=make_partial(tmp_path,PAYLOAD[:11]);control,*_=controller(tmp_path,monkeypatch)
    control.guard=lambda remaining:(_ for _ in ()).throw(OSError(errno.ENOSPC,'D-drive safety guard'))
    with pytest.raises(OSError,match='D-drive'):download.fetch_one(asset(),target,network_wait=control)
    assert part.read_bytes()==PAYLOAD[:11]


def test_system_drive_guard_remains_hard(tmp_path,monkeypatch):
    from resting_eeg.data import environment
    target,part=make_partial(tmp_path,PAYLOAD[:11]);control,*_=controller(tmp_path,monkeypatch)
    monkeypatch.setattr(environment,'system_drive_guard',lambda *a:(_ for _ in ()).throw(environment.SystemDriveLowSpaceError('C guard')))
    with pytest.raises(environment.SystemDriveLowSpaceError):download.fetch_one(asset(),target,network_wait=control)
    assert part.read_bytes()==PAYLOAD[:11]


def test_science_hashes_and_future_boundary_remain_frozen():
    assert b.frozen_hashes(ROOT)['preprocessing_config_sha256']==b.FROZEN_SHA
    auto.validate_freeze(ROOT)
    baseline=b.read_json(ROOT/'outputs/metrics/scenario2/batches/batch_004_starting_state.json')
    for record in baseline['files']:
        if 'preprocessing' in record['path']:
            assert b.sha256_file(Path(record['path']))==record['sha256']


def test_engineering_modules_do_not_call_scientific_analysis_or_fitting():
    import ast
    for module in (auto,nw):
        tree=ast.parse(Path(module.__file__).read_text())
        calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree)
            if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        assert not calls & {'compute_psd','plot','wpli','fit_transform','fit','StandardScaler','process_all_blocks'}


def test_streaming_safe_pause_is_nonterminal_and_never_advances(staged,monkeypatch):
    frame,paths,processing_calls=staged
    calls=[]
    def fetch(row,*args,**kwargs):
        calls.append(row['subject_id'])
        raise nw.ManuallyPaused('explicit synthetic stop')
    monkeypatch.setattr(b.st.downloader,'fetch_one',fetch)
    with pytest.raises(nw.ManuallyPaused):
        b.st.stream_subjects(frame,**paths,execute=True,continue_on_error=True)
    records=journal_records(paths)
    assert calls==['sub-001'] and processing_calls==[]
    assert records[-1]['status']=='manually_paused' and records[-1]['terminal'] is False
    assert {r['subject_id'] for r in records}=={'sub-001'}


def test_batch_waits_for_current_asset_before_second_subject(tmp_path,monkeypatch):
    from test_batches import execution_mocks
    execution_mocks(tmp_path,monkeypatch)
    control,updates,_,_=controller(tmp_path,monkeypatch)
    started=[]; probe_subjects=[]
    def probe(row,emit):
        assert started==['sub-001']
        probe_subjects.append(row['subject_id'])
    monkeypatch.setattr(control,'probe',probe)
    install(monkeypatch,socket.gaierror('offline'),NetworkResponse(PAYLOAD),NetworkResponse(PAYLOAD))
    def process(frame,**kwargs):
        subject=frame.subject_id.iloc[0];started.append(subject)
        download.fetch_one(asset(subject),tmp_path/f'{subject}.eeg',network_wait=kwargs['network_wait'],retries=0)
        return [dict(subject_id=subject,status='complete',raw_retained=True,result=dict(qc_pass=False,output_sha256='hash'))]
    monkeypatch.setattr(b.st,'stream_subjects',process)
    b.execute_batch(tmp_path,'001',tmp_path,manifest_path=tmp_path/'batch_001_manifest.csv',network_wait=control)
    assert started==['sub-001','sub-002'] and probe_subjects==['sub-001']*2


def test_response_identity_rejects_redirect_and_wrong_object_key():
    response=NetworkResponse(b'');row=asset()
    row['source_url'] += '-different'
    with pytest.raises(nw.SourceIdentityError,match='URL'):nw.response_identity(response,row)
    row=asset();row['annex_key']='SHA256E-s1--wrong.eeg'
    with pytest.raises(nw.SourceIdentityError,match='key'):nw.response_identity(response,row)


def test_read_timeout_is_set_on_actual_socket(tmp_path):
    from types import SimpleNamespace
    values=[];response=NetworkResponse(b'')
    response.fp=SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=values.append)))
    with nw.NetworkWait(tmp_path,tmp_path/'STOP').watch(response):pass
    assert values==[120]


@pytest.mark.parametrize('state,exit_code', [
    ('waiting_for_network', None), ('batch_complete', 0),
    ('manually_paused', 20), ('hard_blocked', 1),
])
def test_supervisor_accepts_actual_worker_behind_windows_launcher(state, exit_code):
    base = dict(runner_pid=10, state='starting')
    status = dict(supervision_token='current-run', worker_pid=30, state=state,
        current_subject='sub-001', partial_bytes=11, completed_subjects=12,
        unresolved_subjects=62, network_state='waiting_for_network')
    actual = auto.supervisor_heartbeat(base, status, 20, 'current-run', exit_code)
    assert actual['state'] == state and actual['worker_exit_code'] == exit_code
    assert actual['runner_pid'] == 10 and actual['worker_launcher_pid'] == 20
    assert actual['worker_pid'] == 30 and actual['partial_bytes'] == 11
    assert actual['unresolved_subjects'] == 62


def test_supervisor_ignores_durable_state_from_an_older_worker_run():
    old = dict(supervision_token='old-run', worker_pid=20, state='batch_complete')
    actual = auto.supervisor_heartbeat(dict(runner_pid=10, state='starting'), old, 20, 'new-run', None)
    assert actual['state'] == 'starting' and 'supervision_token' not in actual


@pytest.mark.parametrize('state,exit_code', [('downloading', 0), ('batch_complete', 1), ('manually_paused', 0)])
def test_supervisor_rejects_exit_without_matching_durable_terminal_state(state, exit_code):
    status = dict(supervision_token='run', worker_pid=30, state=state)
    actual = auto.supervisor_heartbeat(dict(runner_pid=10), status, 20, 'run', exit_code)
    assert actual['state'] == 'hard_blocked' and actual['project_state'] == auto.HARD_BLOCKED
    assert 'matching durable' in actual['error']


def test_worker_state_clears_finished_asset_and_preserves_last_network_time(tmp_path):
    state = auto.WorkerState(tmp_path, tmp_path, {'subject_ids':['sub-001']}, [], 100,
        supervision_token='run', last_network_success='2026-09-06T00:00:00+00:00')
    state.update(current_subject='sub-001', current_asset='a.eeg', expected_asset_bytes=100, partial_bytes=11)
    state.update(completed_subject='sub-001', current_subject=None, current_asset=None, state='batch_complete')
    actual = state.snapshot()
    assert actual['expected_asset_bytes'] == actual['partial_bytes'] == actual['unresolved_subjects'] == 0
    assert actual['completed_subjects'] == 1 and actual['supervision_token'] == 'run'
    assert actual['last_successful_network_time'] == '2026-09-06T00:00:00+00:00'


@pytest.mark.parametrize('failure', [TimeoutError('mid-body timeout'), ConnectionResetError('mid-body reset')])
def test_mid_body_network_failure_fsyncs_and_resumes_exact_received_prefix(tmp_path, monkeypatch, failure):
    class Interrupted(NetworkResponse):
        def read1(self, size):
            if self.tell() == 0:
                return self.read(11)
            raise failure
    calls = install(monkeypatch, Interrupted(PAYLOAD), resumed(11))
    control, _, _, _ = controller(tmp_path, monkeypatch)
    target = tmp_path/'a.eeg'
    journal = download.DownloadJournal(tmp_path/'journal.jsonl')
    download.fetch_one(asset(), target, retries=0, network_wait=control, journal=journal)
    assert target.read_bytes() == PAYLOAD and calls[1].get_header('Range') == 'bytes=11-'
    events = b.read_journal(journal.path)
    waiting = next(r for r in events if r['status'] == 'waiting_for_network')
    assert waiting['partial_bytes'] == 11 and waiting['partial_sha256'] == hashlib.sha256(PAYLOAD[:11]).hexdigest()
    assert sum(r.get('received_body_bytes', 0) for r in events) == len(PAYLOAD)
    assert not any(r['status'] == 'failed' for r in events)
