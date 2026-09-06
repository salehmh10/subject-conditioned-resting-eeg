"""Sticky, unbounded transport recovery. No waveform processing lives here."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import errno
import http.client
import os
from pathlib import Path
import random
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from resting_eeg.common import sha256_file, utc_now

POLICY = dict(network_outage_policy='wait_and_resume', transient_network_retry_limit='unlimited',
    stay_on_current_subject_during_outage=True, advance_to_next_subject_during_outage=False,
    preserve_partial_download=True, automatic_resume_after_connectivity_returns=True,
    require_new_user_prompt=False, backoff_seconds=[15, 30, 60, 120, 300],
    maximum_backoff_seconds=300, jitter_fraction=0.05, connect_timeout_seconds=45,
    read_timeout_seconds=120, stall_timeout_seconds=180, restoration_probes=2,
    restoration_probe_separation_seconds=5, heartbeat_seconds=30)
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


class ManuallyPaused(BaseException):
    """Deliberately bypass terminal subject failure handling."""


class SourceIdentityError(ValueError):
    pass


class TransferStalled(TimeoutError):
    pass


def transient(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP
    if isinstance(exc, urllib.error.URLError):
        return transient(exc.reason) if isinstance(exc.reason, BaseException) else any(
            token in str(exc.reason).lower() for token in
            ('getaddrinfo', 'name resolution', 'timed out', 'timeout', 'connection reset',
             'connection aborted', 'temporary failure', 'handshake'))
    if isinstance(exc, ssl.SSLCertVerificationError):
        return False
    if isinstance(exc, (socket.gaierror, TimeoutError, ConnectionError, ssl.SSLError,
                        http.client.IncompleteRead, http.client.RemoteDisconnected)):
        return True
    return isinstance(exc, OSError) and exc.errno in {
        errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT, errno.ENETDOWN,
        errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPIPE}


def retry_after(exc):
    if not isinstance(exc, urllib.error.HTTPError) or exc.code not in TRANSIENT_HTTP:
        return 0.0
    value = exc.headers.get('Retry-After', '')
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.0


def backoff_seconds(failures, jitter=random.random):
    base = POLICY['backoff_seconds'][min(max(failures - 1, 0), 4)]
    return min(300.0, base * (1 + .05 * jitter()))


def response_identity(response, row):
    if response.geturl() != str(row['source_url']):
        raise SourceIdentityError('Response URL differs from exact frozen object URL')
    key = row.get('annex_key')
    if key and (str(key) != urllib.parse.urlsplit(str(row['source_url'])).path.rsplit('/', 1)[-1]
                or f"--{row['payload_sha256']}." not in str(key)
                or not str(key).startswith(f"SHA256E-s{row['payload_size_bytes']}--")):
        raise SourceIdentityError('Frozen object key differs from manifest size/SHA256')
    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise SourceIdentityError('Unexpected encoded source object')


class NetworkWait:
    def __init__(self, runtime, stop_file, *, update=None, sleep=time.sleep,
                 monotonic=time.monotonic, jitter=random.random, guard=None):
        self.runtime = Path(runtime)
        self.stop_file = Path(stop_file)
        self.update = update or (lambda **kw: None)
        self.sleep = sleep
        self.monotonic = monotonic
        self.jitter = jitter
        self.guard = guard or (lambda remaining: None)
        self.stop_requested = threading.Event()
        self.failures = 0
        self.last_success = None

    def check_stop(self):
        if self.stop_requested.is_set() or self.stop_file.exists():
            raise ManuallyPaused('Safe stop requested; current subject and partial preserved')

    def pause(self, seconds):
        # Small intervals keep stop responsive even during an hours-long outage.
        remaining = seconds
        while remaining > 0:
            self.check_stop()
            self.update()  # Surface a failed heartbeat writer as a hard filesystem error.
            interval = min(1.0, remaining)
            self.sleep(interval)
            remaining -= interval
        self.check_stop()

    def begin(self, row, target):
        self.check_stop()
        self.row, self.target = row, Path(target)
        self.update(state='downloading', current_subject=str(row['subject_id']),
                    current_asset=str(row['relative_path']), expected_asset_bytes=int(row['payload_size_bytes']))

    def received(self, count, partial_bytes):
        self.last_success = utc_now()
        self.update(state='downloading', network_state='connected', partial_bytes=partial_bytes,
                    last_successful_network_time=self.last_success, received_delta=count)

    def quarantine(self, partial, emit, reason):
        directory = self.runtime/'quarantine'/str(self.row['subject_id'])
        directory.mkdir(parents=True, exist_ok=True)
        target = directory/(partial.name + '.' + uuid.uuid4().hex)
        size, digest = partial.stat().st_size, sha256_file(partial)
        os.replace(partial, target)
        emit('partial_quarantined', quarantine=str(target), partial_bytes=size,
             partial_sha256=digest, reason=reason)
        return target

    def probe(self, row, emit):
        self.check_stop()
        url = str(row['source_url'])
        hostname = urllib.parse.urlsplit(url).hostname
        # HEAD performs TCP/TLS (including certificate validation) after exact DNS.
        socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'scenario2-network-wait/1'}
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, headers=headers, method='HEAD'),
                                              timeout=POLICY['connect_timeout_seconds'])
        except urllib.error.HTTPError as exc:
            if exc.code not in {405, 501}:
                raise
            exc.close()
            headers['Range'] = 'bytes=0-0'
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=POLICY['connect_timeout_seconds']) as response:
                response_identity(response, row)
                if response.status != 206 or response.headers.get('Content-Range') != f"bytes 0-0/{row['payload_size_bytes']}":
                    raise SourceIdentityError('One-byte probe requires exact 206 Content-Range')
                body = response.read(1)
                emit('response_body', received_body_bytes=len(body), probe=True)
                self.update(received_delta=len(body))
                if len(body) != 1:
                    raise http.client.IncompleteRead(body, 1)
            return
        with response:
            response_identity(response, row)
            if response.status != 200 or response.headers.get('Content-Length') != str(row['payload_size_bytes']):
                raise SourceIdentityError('HEAD status/size differs from frozen source object')
            # Never call read() on a HEAD response.

    def recover(self, exc, row, partial, emit):
        persistent = isinstance(exc, urllib.error.HTTPError) and exc.code in {403, 404}
        if not transient(exc) and not persistent:
            raise exc
        started = self.monotonic()
        size = partial.stat().st_size if partial.exists() else 0
        digest = sha256_file(partial) if partial.exists() else None
        emit('waiting_for_network', error=str(exc), error_type=type(exc).__name__,
             partial_bytes=size, partial_sha256=digest, terminal=False)
        while True:
            self.failures += 1
            delay = max(backoff_seconds(self.failures, self.jitter), retry_after(exc))
            self.update(state='waiting_for_network', network_state='waiting_for_network',
                        consecutive_connectivity_failures=self.failures, current_backoff_seconds=delay,
                        partial_bytes=size)
            emit('network_retry_scheduled', delay_seconds=delay, consecutive_failures=self.failures)
            self.pause(delay)
            try:
                self.probe(row, emit)
                self.pause(POLICY['restoration_probe_separation_seconds'])
                self.probe(row, emit)
            except Exception as failure:
                if persistent and isinstance(failure, urllib.error.HTTPError) and failure.code in {403, 404}:
                    raise SourceIdentityError(f'Persistent HTTP {failure.code} contradicts frozen manifest') from failure
                if not transient(failure):
                    if isinstance(failure, urllib.error.HTTPError) and failure.code in {403, 404}:
                        persistent = True
                    else:
                        raise
                emit('connectivity_probe_failed', error=str(failure), terminal=False)
                exc = failure
                continue
            self.last_success = utc_now()
            emit('connectivity_restored', successful_probes=2, outage_seconds=self.monotonic()-started,
                 partial_bytes=size, consecutive_failures=self.failures)
            self.failures = 0
            self.update(state='resuming', network_state='connected', consecutive_connectivity_failures=0,
                        current_backoff_seconds=0, last_successful_network_time=self.last_success)
            return

    @contextmanager
    def watch(self, response):
        """Track delivered bytes; interrupt a blocked socket without losing the prefix."""
        done = threading.Event()
        state = dict(last=self.monotonic(), stalled=False)
        sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
        if sock is not None:
            sock.settimeout(POLICY['read_timeout_seconds'])

        def monitor():
            while not done.wait(1):
                stopping = self.stop_requested.is_set() or self.stop_file.exists()
                stalled = self.monotonic() - state['last'] >= POLICY['stall_timeout_seconds']
                if stopping or stalled:
                    state['stalled'] = stalled
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    response.close()
                    return

        thread = threading.Thread(target=monitor, daemon=True, name='download-progress-watchdog')
        thread.start()
        try:
            yield lambda: state.update(last=self.monotonic())
        except BaseException:
            self.check_stop()
            if state['stalled']:
                raise TransferStalled('No response-body byte progress for 180 seconds') from None
            raise
        finally:
            done.set()
            thread.join(timeout=2)
        self.check_stop()
        if state['stalled']:
            raise TransferStalled('No response-body byte progress for 180 seconds')
