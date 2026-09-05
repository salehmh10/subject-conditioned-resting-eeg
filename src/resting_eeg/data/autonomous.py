"""Process supervision and engineering heartbeat for the existing Batch 004."""
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


def verify_baseline(root, *, partial=False):
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


def validate_freeze(root):
    directory = b.paths(root)['metrics_root']/'batches'
    frame, plan, auth = b.validate_authorization(root, BATCH, directory)
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
    def __init__(self, runtime, raw_root, plan, completed, received, *, interval=30):
        self.runtime, self.raw_root = Path(runtime), Path(raw_root)
        self.plan, self.completed = plan, set(completed)
        self.interval = interval
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.persistence_error = None
        self.value = dict(batch_id=BATCH, worker_pid=os.getpid(), state='preflight',
            current_subject=None, current_asset=None, partial_bytes=0, expected_asset_bytes=0,
            network_state='not_requested', consecutive_connectivity_failures=0,
            current_backoff_seconds=0, last_successful_network_time=None,
            received_response_body_bytes=received, manifest_sha256=MANIFEST_SHA)
        self.thread = threading.Thread(target=self._run, daemon=True, name='worker-heartbeat')

    def update(self, **details):
        if self.persistence_error is not None:
            raise RuntimeError('Worker heartbeat persistence failed') from self.persistence_error
        with self.lock:
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
            value['partial_bytes'] = part.stat().st_size if part.exists() else 0
        value['C_free_GiB'] = shutil.disk_usage('C:/').free/b.GIB if os.name == 'nt' else None
        value['D_free_GiB'] = b.st._free_bytes(self.raw_root)/b.GIB
        return value

    def persist(self):
        write_json(self.runtime/f'{NAME}_worker_state.json', self.snapshot())

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


def worker(root):
    p = b.paths(root); runtime = b.runtime_directory(root)
    directory, frame, plan, auth = validate_freeze(root)
    if auth.get('network_wait_policy') != POLICY or auth.get('network_wait_policy_sha256') != canonical_hash(POLICY):
        raise ValueError('Committed network policy requires a matching numbered authorization')
    if auth.get('autonomous_starting_state_sha256') != sha256_file(runtime/'batch_004_autonomous_starting_state.json'):
        raise ValueError('Autonomous preservation baseline differs from authorization')
    completed = auth['completed_subjects_reverified']
    plan = dict(plan, subject_payload_bytes={str(s):int(g.payload_size_bytes.sum()) for s,g in frame.groupby('subject_id', sort=False)})
    received = sum(r.get('received_body_bytes', 0) for r in b.read_journal(runtime/'batch_004_download_journal.jsonl'))
    with WorkerState(runtime, p['raw_root'], plan, completed, received) as state:
        def disk_guard(remaining):
            system_drive_guard('autonomous_begin_or_resume_asset')
            required = remaining+b.st.PROCESSING_OVERHEAD_BYTES+b.st.COMPACT_OUTPUT_ALLOWANCE_BYTES+b.st.SAFETY_MARGIN_BYTES
            if b.st._free_bytes(p['raw_root']) < required:
                raise OSError(f'D-drive safety guard: require {required} free bytes for current asset and processing')
        control = NetworkWait(runtime, runtime/'STOP_BATCH_004', update=state.update, guard=disk_guard)
        previous = signal.signal(signal.SIGINT, lambda *_: control.stop_requested.set())
        try:
            configure_environment(root)
            verify_baseline(root)
            b.verify_previous_state(root, BATCH, full=True)
            verified = b.verified_completed(root)
            remaining = unresolved_in_order(plan, verified)
            if not set(remaining) <= set(auth['unresolved_subject_ids']):
                raise ValueError('Unresolved membership exceeds numbered authorization')
            state.completed = set(plan['subject_ids']) & set(verified)
            if remaining:
                b.execute_batch(root, BATCH, directory, manifest_path=directory/'batch_004_manifest.csv',
                    requested_cap=45*b.GIB, delete_raw=True, continue_on_error=False, selected=remaining, network_wait=control)
            else:
                b.summarize(root, BATCH, directory)
            control.check_stop()
            summary = b.read_json(directory/'batch_004_summary.json')
            if summary['subjects_completed'] != 74 or summary['technical_failures']:
                raise ValueError('Frozen Batch 004 did not reach 74 verified completions')
            for subject in plan['subject_ids']:
                if list((p['raw_root']/subject).rglob('*.part')):
                    raise ValueError(f'Unresolved partial after completion: {subject}')
            verify_baseline(root)
            state.update(state='batch_complete', project_state='BATCH_004_AWAITING_FINAL_ACCEPTANCE', current_subject=None,
                         current_asset=None, network_state='complete')
            return 0
        except (ManuallyPaused, KeyboardInterrupt) as exc:
            state.update(state='manually_paused', error=str(exc), network_state='paused')
            b.st.append_status(runtime/'batch_004_autoresume_journal.jsonl', NAME,
                state.value.get('current_subject') or '', 'manually_paused', terminal=False, error=str(exc))
            return 20
        except Exception as exc:
            traceback.print_exc()
            state.update(state='hard_blocked', project_state=HARD_BLOCKED,
                         error=f'{type(exc).__name__}: {exc}')
            b.st.append_status(runtime/'batch_004_autoresume_journal.jsonl', NAME,
                state.value.get('current_subject') or '', 'hard_blocked', error=f'{type(exc).__name__}: {exc}')
            return 1
        finally:
            signal.signal(signal.SIGINT, previous)


def supervisor(root):
    runtime = b.runtime_directory(root)
    runtime.mkdir(parents=True, exist_ok=True)
    with b.st._journal_lock(runtime/NAME):
        # A surviving old worker must finish before a new supervisor can launch one.
        with b.st._journal_lock(runtime/'batch_execution'):
            pass
        pid_path = runtime/f'{NAME}.pid'
        with pid_path.open('w', encoding='ascii') as handle:
            handle.write(str(os.getpid())); handle.flush(); os.fsync(handle.fileno())
        base = dict(timestamp_utc=utc_now(), batch_id=BATCH, runner_pid=os.getpid(),
                    state='starting', manifest_sha256=MANIFEST_SHA)
        write_json(runtime/f'{NAME}_heartbeat.json', base)
        with (runtime/f'{NAME}_stdout.log').open('a', encoding='utf-8') as out, (runtime/f'{NAME}_stderr.log').open('a', encoding='utf-8') as err:
            child = subprocess.Popen([sys.executable, '-u', str(root/'scripts/data/s2_14_autonomous_batch_supervisor.py'), '--worker'],
                cwd=root, stdout=out, stderr=err, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
            old_handler = signal.signal(signal.SIGINT, lambda *_: (runtime/'STOP_BATCH_004').touch())
            try:
                while True:
                    try:
                        code = child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        code = None
                    path = runtime/f'{NAME}_worker_state.json'
                    status = b.read_json(path) if path.exists() else {}
                    if status.get('worker_pid') != child.pid:
                        status = {}
                    heartbeat = dict(base, **{k:v for k,v in status.items() if k not in {'runner_pid'}})
                    heartbeat.update(timestamp_utc=utc_now(), worker_pid=child.pid, worker_exit_code=code)
                    if code is not None and status.get('state') not in {'batch_complete', 'manually_paused', 'hard_blocked'}:
                        heartbeat.update(state='hard_blocked', project_state=HARD_BLOCKED,
                            error=f'Worker exited without durable completion/pause: exit {code}; inspect stderr')
                    write_json(runtime/f'{NAME}_heartbeat.json', heartbeat)
                    if code is not None:
                        return code
            finally:
                signal.signal(signal.SIGINT, old_handler)
