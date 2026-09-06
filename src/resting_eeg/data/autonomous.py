"""Process supervision and engineering heartbeat for authorized frozen batches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import traceback
from uuid import uuid4

from resting_eeg.common import canonical_hash, sha256_file, utc_now, write_json
from resting_eeg.data import batches as b
from resting_eeg.data.environment import configure_environment, system_drive_guard
from resting_eeg.data.network_wait import ManuallyPaused, NetworkWait, POLICY

BATCH = '004'
NAME = 'batch_004_autoresume'
MANIFEST_SHA = '0be1d41165aa55e30dc86e72724a752270de4a8bec7a50fbad5595a158e37fb5'
COMPLETE = 'BATCH_004_AUTONOMOUS_RESUME_COMPLETE'
HARD_BLOCKED = 'BATCH_004_HARD_BLOCKED'


def unresolved_in_order(plan, completed):
    return [s for s in plan['subject_ids'] if s not in completed]


def verify_baseline(root, *, partial=False, batch_id=BATCH):
    if batch_id == '005':
        return b.verify_previous_state(root, batch_id, full=True)
    runtime = b.runtime_directory(root)
    baseline = b.read_json(runtime/'batch_004_autonomous_starting_state.json')
    for artifact in baseline['protected_files'] + baseline['progress_archives']:
        path = Path(artifact['path'])
        if path.stat().st_size != artifact['bytes'] or sha256_file(path) != artifact['sha256']:
            raise ValueError(f'Protected starting artifact changed: {path}')
    for artifact in baseline['journal_prefixes']:
        with Path(artifact['path']).open('rb') as handle:
            digest = hashlib.sha256(handle.read(artifact['bytes'])).hexdigest()
        if digest != artifact['sha256']:
            raise ValueError(f'Historical journal prefix changed: {artifact["path"]}')
    if partial:
        artifact = baseline['partial']
        path = Path(artifact['path'])
        if path.stat().st_size != artifact['bytes'] or sha256_file(path) != artifact['sha256']:
            raise ValueError('Preserved sub-010232 partial changed before authorization')
    return baseline


def validate_freeze(root, batch_id=BATCH):
    directory = b.paths(root)['metrics_root']/'batches'
    frame, plan, auth = b.validate_authorization(root, batch_id, directory)
    if batch_id == '005':
        from resting_eeg.data.final_remainder import validate_coverage
        validate_coverage(root)
        return directory, frame, plan, auth
    if batch_id != BATCH:
        raise ValueError('Autonomous execution is limited to Batch 004 or final Batch 005')
    if auth['manifest_sha256'] != MANIFEST_SHA or plan['cap_bytes'] != 45*b.GIB or plan['subject_count'] != 74:
        raise ValueError('Autonomous runner requires the existing exact Batch 004 freeze')
    if list(directory.glob('batch_005*')) or list(b.runtime_directory(root).glob('batch_005*')):
        raise ValueError('Batch 005 is prohibited')
    return directory, frame, plan, auth


def authorize(root):
    directory, frame, plan, auth = validate_freeze(root)
    baseline = verify_baseline(root, partial=True)
    completed = b.verified_completed(root)
    unresolved = unresolved_in_order(plan, completed)
    if unresolved != baseline['unresolved_subject_ids'] or len(unresolved) != 62 or len(completed) != 84:
        raise ValueError('Expected exactly 84 preserved completions and 62 frozen unresolved subjects')
    record = b.authorize_resume(root, BATCH, directory, network_policy=POLICY)
    if record['completed_subjects_reverified'] != baseline['completed_batch_subjects']:
        raise ValueError('Completed Batch 004 membership changed')
    return record


class WorkerState:
    """Memory updates per read; durable snapshot independent of network waits."""
    def __init__(self, runtime, raw_root, plan, completed, received, *, interval=30,
                 supervision_token=None, last_network_success=None):
        self.runtime, self.raw_root = Path(runtime), Path(raw_root)
        self.plan, self.completed = plan, set(completed)
        self.batch_id = plan.get('batch_id', BATCH)
        self.name = f'batch_{self.batch_id}_autoresume'
        self.interval = interval
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.persistence_error = None
        self.value = dict(batch_id=self.batch_id, worker_pid=os.getpid(), state='preflight', supervision_token=supervision_token,
            current_subject=None, current_asset=None, partial_bytes=0, expected_asset_bytes=0,
            network_state='not_requested', consecutive_connectivity_failures=0,
            current_backoff_seconds=0, last_successful_network_time=last_network_success,
            received_response_body_bytes=received, manifest_sha256=plan.get('manifest_sha256', MANIFEST_SHA))
        self.thread = threading.Thread(target=self._run, daemon=True, name='worker-heartbeat')

    def update(self, **details):
        if self.persistence_error is not None:
            raise RuntimeError('Worker heartbeat persistence failed') from self.persistence_error
        with self.lock:
            if 'current_asset' in details and details['current_asset'] is None:
                details.setdefault('partial_bytes', 0)
                details.setdefault('expected_asset_bytes', 0)
            if subject := details.pop('completed_subject', None):
                self.completed.add(subject)
            self.value['received_response_body_bytes'] += details.pop('received_delta', 0)
            self.value.update(details)

    def snapshot(self):
        with self.lock:
            value = dict(self.value, timestamp_utc=utc_now(),
                completed_subjects=len(self.completed), unresolved_subjects=len(self.plan['subject_ids'])-len(self.completed))
        subject = value['current_subject']
        assets = self.plan.get('subject_payload_bytes', {})
        # File sizes are engineering metadata, never numerical EEG inspection.
        total = assets.get(subject, 0) if isinstance(assets, dict) else 0
        if subject and total:
            local = 0
            for file in (self.raw_root/subject).rglob('*'):
                try:
                    if file.is_file() and file.suffix in {'.eeg', '.vhdr', '.part'}:
                        local += file.stat().st_size
                except FileNotFoundError:
                    pass  # Guarded cleanup may finish while heartbeat reads sizes.
            value['subject_progress_percent'] = min(100.0, 100*local/total)
        else:
            value['subject_progress_percent'] = 100.0 if subject in self.completed else 0.0
        if asset := value['current_asset']:
            target = self.raw_root/asset
            part = target.with_name(target.name+'.part')
            try:
                value['partial_bytes'] = part.stat().st_size
            except FileNotFoundError:
                value['partial_bytes'] = 0  # Atomic promotion or guarded cleanup may finish concurrently.
        value['C_free_GiB'] = shutil.disk_usage('C:/').free/b.GIB if os.name == 'nt' else None
        value['D_free_GiB'] = b.st._free_bytes(self.raw_root)/b.GIB
        if self.batch_id == '005':
            value.update(subjects_completed=value['completed_subjects'], subjects_remaining=value['unresolved_subjects'],
                connectivity_failures=value.get('consecutive_connectivity_failures',0),
                asset_progress_percent=(min(100.,100*value['partial_bytes']/value['expected_asset_bytes'])
                    if value['expected_asset_bytes'] else 0.))
            progress_path=self.runtime/'batch_005_progress.json'
            progress=b.read_json(progress_path) if progress_path.exists() else {}
            value['raw_deleted_bytes']=progress.get('raw_deleted_bytes',0)
            value['raw_retained_bytes']=progress.get('raw_retained_bytes',b.directory_bytes(self.raw_root))
        return value

    def persist(self):
        write_json(self.runtime/f'{self.name}_worker_state.json', self.snapshot())

    def _run(self):
        while not self.done.wait(self.interval):
            try:
                self.persist()
            except Exception as exc:
                self.persistence_error = exc
                return

    def __enter__(self):
        self.persist()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=5)
        self.persist()


def worker(root, supervision_token=None, batch_id=BATCH):
    p = b.paths(root); runtime = b.runtime_directory(root)
    directory, frame, plan, auth = validate_freeze(root, batch_id)
    name = f'batch_{batch_id}_autoresume'
    if auth.get('network_wait_policy') != POLICY or auth.get('network_wait_policy_sha256') != canonical_hash(POLICY):
        raise ValueError('Committed network policy requires a matching numbered authorization')
    if batch_id == BATCH and auth.get('autonomous_starting_state_sha256') != sha256_file(runtime/'batch_004_autonomous_starting_state.json'):
        raise ValueError('Autonomous preservation baseline differs from authorization')
    completed = auth['completed_subjects_reverified']
    plan = dict(plan, manifest_sha256=auth['manifest_sha256'], subject_payload_bytes={str(s):int(g.payload_size_bytes.sum()) for s,g in frame.groupby('subject_id', sort=False)})
    events = b.read_journal(runtime/f'batch_{batch_id}_download_journal.jsonl')
    received = sum(r.get('received_body_bytes', 0) for r in events)
    last_network = next((r['timestamp_utc'] for r in reversed(events) if r.get('received_body_bytes', 0)), None)
    with WorkerState(runtime, p['raw_root'], plan, completed, received,
                     supervision_token=supervision_token, last_network_success=last_network) as state:
        def disk_guard(remaining):
            system_drive_guard('autonomous_begin_or_resume_asset')
            required = remaining+b.st.PROCESSING_OVERHEAD_BYTES+b.st.COMPACT_OUTPUT_ALLOWANCE_BYTES+b.st.SAFETY_MARGIN_BYTES
            if b.st._free_bytes(p['raw_root']) < required:
                raise OSError(f'D-drive safety guard: require {required} free bytes for current asset and processing')
        control = NetworkWait(runtime, runtime/f'STOP_BATCH_{batch_id}', update=state.update, guard=disk_guard)
        previous = signal.signal(signal.SIGINT, lambda *_: control.stop_requested.set())
        try:
            configure_environment(root)
            verify_baseline(root, batch_id=batch_id)
            if batch_id == BATCH:
                b.verify_previous_state(root, batch_id, full=True)
            verified = b.verified_completed(root)
            remaining = unresolved_in_order(plan, verified)
            if not set(remaining) <= set(auth['unresolved_subject_ids']):
                raise ValueError('Unresolved membership exceeds numbered authorization')
            state.completed = set(plan['subject_ids']) & set(verified)
            if remaining:
                b.execute_batch(root, batch_id, directory, manifest_path=directory/f'batch_{batch_id}_manifest.csv',
                    requested_cap=plan['cap_bytes'], delete_raw=True, continue_on_error=False, selected=remaining, network_wait=control)
            elif batch_id == BATCH:
                b.summarize(root, batch_id, directory)
            control.check_stop()
            summary = b.read_json(directory/f'batch_{batch_id}_summary.json')
            if summary['subjects_completed'] != plan['subject_count'] or summary['technical_failures']:
                raise ValueError(f'Frozen Batch {batch_id} did not reach all verified completions')
            for subject in plan['subject_ids']:
                if list((p['raw_root']/subject).rglob('*.part')):
                    raise ValueError(f'Unresolved partial after completion: {subject}')
            verify_baseline(root, batch_id=batch_id)
            state.update(state='batch_complete', project_state=f'BATCH_{batch_id}_AWAITING_FINAL_ACCEPTANCE', current_subject=None,
                         current_asset=None, network_state='complete')
            return 0
        except (ManuallyPaused, KeyboardInterrupt) as exc:
            state.update(state='manually_paused', error=str(exc), network_state='paused')
            b.st.append_status(runtime/f'{name}_journal.jsonl', name,
                state.value.get('current_subject') or '', 'manually_paused', terminal=False, error=str(exc))
            return 20
        except Exception as exc:
            traceback.print_exc()
            state.update(state='hard_blocked', project_state='FINAL_REMAINDER_HARD_BLOCKED' if batch_id=='005' else HARD_BLOCKED,
                         error=f'{type(exc).__name__}: {exc}')
            b.st.append_status(runtime/f'{name}_journal.jsonl', name,
                state.value.get('current_subject') or '', 'hard_blocked', error=f'{type(exc).__name__}: {exc}')
            return 1
        finally:
            signal.signal(signal.SIGINT, previous)


def supervisor_heartbeat(base, status, launcher_pid, token, exit_code):
    """Match one worker run even when the Windows venv launcher has another PID."""
    if not token or status.get('supervision_token') != token:
        status = {}
    heartbeat = dict(base, **status)
    heartbeat.update(timestamp_utc=utc_now(), runner_pid=base['runner_pid'],
        worker_launcher_pid=launcher_pid, worker_pid=status.get('worker_pid', launcher_pid),
        worker_exit_code=exit_code)
    expected = {'batch_complete': 0, 'manually_paused': 20, 'hard_blocked': 1}
    if exit_code is not None and (status.get('state') not in expected or expected[status['state']] != exit_code):
        heartbeat.update(state='hard_blocked', project_state='FINAL_REMAINDER_HARD_BLOCKED' if base.get('batch_id')=='005' else HARD_BLOCKED,
            error=f'Worker exited without matching durable completion/pause: exit {exit_code}; inspect stderr')
    return heartbeat


def supervisor(root, batch_id=BATCH):
    directory, frame, plan, auth = validate_freeze(root, batch_id)
    name = f'batch_{batch_id}_autoresume'
    runtime = b.runtime_directory(root)
    runtime.mkdir(parents=True, exist_ok=True)
    with b.st._journal_lock(runtime/name):
        # A surviving old worker must finish before a new supervisor can launch one.
        with b.st._journal_lock(runtime/'batch_execution'):
            pass
        pid_path = runtime/f'{name}.pid'
        with pid_path.open('w', encoding='ascii') as handle:
            handle.write(str(os.getpid())); handle.flush(); os.fsync(handle.fileno())
        base = dict(timestamp_utc=utc_now(), batch_id=batch_id, runner_pid=os.getpid(),
                    state='starting', manifest_sha256=auth['manifest_sha256'],
                    current_subject=None, current_asset=None, partial_bytes=0, expected_asset_bytes=0,
                    subject_progress_percent=0, completed_subjects=None, unresolved_subjects=None,
                    network_state='not_requested', consecutive_connectivity_failures=0,
                    current_backoff_seconds=0, last_successful_network_time=None,
                    received_response_body_bytes=None, C_free_GiB=None, D_free_GiB=None)
        write_json(runtime/f'{name}_heartbeat.json', base)
        token = uuid4().hex
        with (runtime/f'{name}_stdout.log').open('a', encoding='utf-8') as out, (runtime/f'{name}_stderr.log').open('a', encoding='utf-8') as err:
            child = subprocess.Popen([sys.executable, '-u', str(root/'scripts/data/s2_14_autonomous_batch_supervisor.py'),
                '--worker', '--supervision-token', token, '--batch-id', batch_id],
                cwd=root, stdout=out, stderr=err, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
            old_handler = signal.signal(signal.SIGINT, lambda *_: (runtime/f'STOP_BATCH_{batch_id}').touch())
            try:
                while True:
                    try:
                        code = child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        code = None
                    path = runtime/f'{name}_worker_state.json'
                    status = b.read_json(path) if path.exists() else {}
                    heartbeat = supervisor_heartbeat(base, status, child.pid, token, code)
                    write_json(runtime/f'{name}_heartbeat.json', heartbeat)
                    if code is not None:
                        return code
            finally:
                signal.signal(signal.SIGINT, old_handler)
