"""Network-free regression coverage for manifest download/resume safety."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from scripts.data import s2_06_download_manifest as download


PAYLOAD = b"audited waveform payload used only in synthetic unit tests"


def test_measured_body_accounting_with_retry(tmp_path,monkeypatch):
    prefix=10
    install_responses(monkeypatch,Response(PAYLOAD,fail_after=prefix),resume_response(prefix))
    journal=download.DownloadJournal(tmp_path/'events.jsonl')
    download.fetch_one(asset(),tmp_path/'asset.eeg',journal=journal,backoff=0)
    records=[json.loads(line) for line in journal.path.read_text().splitlines()]
    assert sum(r.get('received_body_bytes',0) for r in records)==len(PAYLOAD)
    assert sum(r['status']=='retry_scheduled' for r in records)==1
    assert sum(r['status']=='download_interrupted' for r in records)==1
    assert [r['resume_offset'] for r in records if r['status']=='downloading']==[0,prefix]


def asset(subject="sub-001", payload=PAYLOAD):
    return pd.Series({
        "subject_id": subject,
        "relative_path": f"{subject}/eeg/{subject}_task-resting_eeg.eeg",
        "source_url": "https://nemar.s3.us-east-2.amazonaws.com/nm000179/objects/synthetic",
        "payload_size_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    })


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None, fail_after=None, failure=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.fail_after = fail_after
        self.failure = failure or ConnectionResetError("synthetic interrupted connection")

    def read(self, size=-1):
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise self.failure
        if self.fail_after is not None:
            size = min(size, self.fail_after - self.tell())
        return super().read(size)


@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This test must not issue an external network request")
    monkeypatch.setattr(download.urllib.request, "urlopen", forbidden)


def install_responses(monkeypatch, *responses):
    pending = iter(responses)
    calls = []

    def opener(request, timeout):
        calls.append((dict(request.header_items()), timeout))
        response = next(pending)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(download.urllib.request, "urlopen", opener)
    return calls


def make_partial(tmp_path, prefix):
    target = tmp_path / "asset.eeg"
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(prefix)
    return target, partial


def resume_response(offset, payload=PAYLOAD, **kwargs):
    return Response(payload[offset:], status=206, headers={
        "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
        "Content-Length": str(len(payload) - offset),
    }, **kwargs)


def test_complete_existing_is_verified_without_request(tmp_path):
    target = tmp_path / "asset.eeg"
    target.write_bytes(PAYLOAD)
    journal = download.DownloadJournal(tmp_path / "events.jsonl")
    assert download.fetch_one(asset(), target, journal=journal) == "verified_existing"
    record = json.loads(journal.path.read_text())
    assert record["status"] == "verified_existing"
    assert record["expected_sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


@pytest.mark.parametrize("corrupt", [b"short", b"x" * len(PAYLOAD)])
def test_corrupt_final_is_refused_and_preserved(tmp_path, corrupt):
    target = tmp_path / "asset.eeg"
    target.write_bytes(corrupt)
    with pytest.raises(download.DownloadIntegrityError):
        download.fetch_one(asset(), target)
    assert target.read_bytes() == corrupt


def test_valid_partial_sends_range_and_atomically_promotes(monkeypatch, tmp_path):
    target, partial = make_partial(tmp_path, PAYLOAD[:11])
    calls = install_responses(monkeypatch, resume_response(11))
    real_replace = download.os.replace
    replacements = []

    def replace(source, destination):
        assert Path(source) == partial
        assert not target.exists()
        assert Path(source).read_bytes() == PAYLOAD
        replacements.append((source, destination))
        return real_replace(source, destination)

    monkeypatch.setattr(download.os, "replace", replace)
    assert download.fetch_one(asset(), target, timeout=4.5, chunk_size=7) == "resumed_and_verified"
    assert calls == [({"User-agent": "scenario2-audited-downloader/2", "Accept-encoding": "identity", "Range": "bytes=11-"}, 4.5)]
    assert target.read_bytes() == PAYLOAD
    assert not partial.exists()
    assert len(replacements) == 1


def test_server_ignoring_range_restarts_without_appending(monkeypatch, tmp_path):
    target, partial = make_partial(tmp_path, b"corrupt prefix")
    calls = install_responses(monkeypatch, Response(PAYLOAD, headers={"Content-Length": str(len(PAYLOAD))}))
    messages = []
    assert download.fetch_one(asset(), target, progress=messages.append) == "downloaded_and_verified"
    assert calls[0][0]["Range"] == "bytes=14-"
    assert target.read_bytes() == PAYLOAD
    assert not partial.exists()
    assert any("range_ignored_restarting" in message for message in messages)


@pytest.mark.parametrize("content_range", [
    "", "bytes 0-55/56", "bytes 11-56/*", "items 11-56/57",
    f"bytes 12-{len(PAYLOAD)-1}/{len(PAYLOAD)}",
    f"bytes 11-{len(PAYLOAD)-2}/{len(PAYLOAD)}",
    f"bytes 11-{len(PAYLOAD)-1}/{len(PAYLOAD)+1}",
])
def test_wrong_content_range_preserves_prefix(monkeypatch, tmp_path, content_range):
    target, partial = make_partial(tmp_path, PAYLOAD[:11])
    calls = install_responses(monkeypatch, Response(PAYLOAD[11:], 206, {"Content-Range": content_range}))
    with pytest.raises(download.DownloadProtocolError, match="Content-Range"):
        download.fetch_one(asset(), target)
    assert partial.read_bytes() == PAYLOAD[:11]
    assert not target.exists()
    assert len(calls) == 1


@pytest.mark.parametrize("headers", [
    {"Content-Length": "1"}, {"Content-Length": "garbage"},
    {"Content-Encoding": "gzip"}, {"Content-Range": "bytes 0-1/2"},
])
def test_invalid_full_response_headers_preserve_prefix(monkeypatch, tmp_path, headers):
    target, partial = make_partial(tmp_path, PAYLOAD[:11])
    install_responses(monkeypatch, Response(PAYLOAD, headers=headers))
    with pytest.raises(download.DownloadProtocolError):
        download.fetch_one(asset(), target)
    assert partial.read_bytes() == PAYLOAD[:11]
    assert not target.exists()


def test_empty_partial_requests_zero_range(monkeypatch, tmp_path):
    target, partial = make_partial(tmp_path, b"")
    calls = install_responses(monkeypatch, resume_response(0))
    assert download.fetch_one(asset(), target) == "downloaded_and_verified"
    assert calls[0][0]["Range"] == "bytes=0-"
    assert target.read_bytes() == PAYLOAD


def test_completed_valid_partial_promotes_without_network(tmp_path):
    target, partial = make_partial(tmp_path, PAYLOAD)
    assert download.fetch_one(asset(), target) == "resumed_and_verified"
    assert target.read_bytes() == PAYLOAD
    assert not partial.exists()


@pytest.mark.parametrize("prefix", [PAYLOAD + b"too long", b"x" * len(PAYLOAD)])
def test_oversize_or_completed_corrupt_partial_is_quarantined_then_restarted(monkeypatch, tmp_path, prefix):
    target, partial = make_partial(tmp_path, prefix)
    calls = install_responses(monkeypatch, Response(PAYLOAD))
    assert download.fetch_one(asset(), target) == "downloaded_and_verified"
    assert "Range" not in calls[0][0]
    assert target.read_bytes() == PAYLOAD
    quarantined = list(tmp_path.glob("asset.eeg.part.invalid-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == prefix
    assert not partial.exists()


def test_wrong_sha256_quarantines_payload_without_promoting(monkeypatch, tmp_path):
    target = tmp_path / "asset.eeg"
    install_responses(monkeypatch, Response(b"x" * len(PAYLOAD)))
    with pytest.raises(download.DownloadIntegrityError, match="SHA256"):
        download.fetch_one(asset(), target)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()
    assert list(tmp_path.glob("*.invalid-*"))[0].read_bytes() == b"x" * len(PAYLOAD)


def test_corrupt_prefix_is_detected_by_final_hash(monkeypatch, tmp_path):
    target, partial = make_partial(tmp_path, b"x" * 11)
    install_responses(monkeypatch, resume_response(11))
    with pytest.raises(download.DownloadIntegrityError, match="SHA256"):
        download.fetch_one(asset(), target)
    assert not target.exists()
    assert not partial.exists()
    assert list(tmp_path.glob("*.invalid-*"))[0].read_bytes() == b"x" * 11 + PAYLOAD[11:]


def test_short_response_preserves_received_prefix(monkeypatch, tmp_path):
    target = tmp_path / "asset.eeg"
    install_responses(monkeypatch, Response(PAYLOAD[:13]))
    with pytest.raises(http.client.IncompleteRead):
        download.fetch_one(asset(), target, retries=0)
    assert not target.exists()
    assert target.with_name(target.name + ".part").read_bytes() == PAYLOAD[:13]


def test_oversize_response_never_promotes(monkeypatch, tmp_path):
    target = tmp_path / "asset.eeg"
    install_responses(monkeypatch, Response(PAYLOAD + b"extra"))
    with pytest.raises(download.DownloadIntegrityError, match="exceeds"):
        download.fetch_one(asset(), target)
    assert not target.exists()
    assert len(list(tmp_path.glob("*.invalid-*"))) == 1


def test_interrupted_request_preserves_prefix_and_next_call_resumes(monkeypatch, tmp_path):
    target, partial = make_partial(tmp_path, PAYLOAD[:11])
    calls = install_responses(monkeypatch, resume_response(11, fail_after=7))
    with pytest.raises(ConnectionResetError):
        download.fetch_one(asset(), target, retries=0, chunk_size=7)
    assert calls[0][0]["Range"] == "bytes=11-"
    assert partial.read_bytes() == PAYLOAD[:18]
    calls = install_responses(monkeypatch, resume_response(18))
    assert download.fetch_one(asset(), target) == "resumed_and_verified"
    assert calls[0][0]["Range"] == "bytes=18-"


def test_incomplete_read_exception_payload_is_preserved(monkeypatch, tmp_path):
    target = tmp_path / "asset.eeg"
    response = Response(b"", fail_after=0, failure=http.client.IncompleteRead(PAYLOAD[:9], len(PAYLOAD)-9))
    install_responses(monkeypatch, response)
    with pytest.raises(http.client.IncompleteRead):
        download.fetch_one(asset(), target, retries=0)
    assert target.with_name(target.name + ".part").read_bytes() == PAYLOAD[:9]


def test_complete_partial_after_interrupted_eof_is_verified_without_another_request(monkeypatch, tmp_path):
    target = tmp_path / "asset.eeg"
    calls = install_responses(monkeypatch, Response(PAYLOAD, fail_after=len(PAYLOAD)))
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    assert download.fetch_one(asset(), target, retries=1) == "resumed_and_verified"
    assert len(calls) == 1
    assert target.read_bytes() == PAYLOAD


def test_transient_retry_resumes_and_backoff_is_bounded(monkeypatch, tmp_path):
    target = tmp_path / "asset.eeg"
    calls = install_responses(monkeypatch,
        Response(PAYLOAD, fail_after=10),
        urllib.error.URLError("synthetic timeout"),
        resume_response(10),
    )
    delays = []
    monkeypatch.setattr(download.time, "sleep", delays.append)
    journal = download.DownloadJournal(tmp_path / "journal.jsonl")
    assert download.fetch_one(asset(), target, retries=2, backoff=40, journal=journal) == "resumed_and_verified"
    assert delays == [40, 60]
    assert len(calls) == 3
    assert calls[1][0]["Range"] == calls[2][0]["Range"] == "bytes=10-"
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert sum(event["status"] == "download_interrupted" for event in events) == 2
    assert events[-1]["status"] == "resumed_and_verified"


def test_retry_exhaustion_has_exact_attempt_bound_and_preserves_partial(monkeypatch, tmp_path):
    target, partial = make_partial(tmp_path, PAYLOAD[:11])
    calls = install_responses(monkeypatch, *[TimeoutError("synthetic timeout") for _ in range(4)])
    delays = []
    monkeypatch.setattr(download.time, "sleep", delays.append)
    with pytest.raises(TimeoutError):
        download.fetch_one(asset(), target, retries=3, backoff=0.5)
    assert len(calls) == 4
    assert delays == [0.5, 1.0, 2.0]
    assert partial.read_bytes() == PAYLOAD[:11]


@pytest.mark.parametrize("code,retry", [(404, False), (416, False), (429, True), (503, True)])
def test_http_status_controls_retry(monkeypatch, tmp_path, code, retry):
    target = tmp_path / "asset.eeg"
    failure = urllib.error.HTTPError("https://example.invalid", code, "synthetic", {}, None)
    calls = install_responses(monkeypatch, failure, Response(PAYLOAD))
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    if retry:
        assert download.fetch_one(asset(), target, retries=1) == "downloaded_and_verified"
        assert len(calls) == 2
    else:
        with pytest.raises(urllib.error.HTTPError):
            download.fetch_one(asset(), target, retries=1)
        assert len(calls) == 1


@pytest.mark.parametrize("relative", ["../escape.eeg", "/absolute.eeg", "C:/drive.eeg", "a/../../escape", "a\\..\\escape", "a/./b", "a//b", "a/file:stream", "a/name.", "a/name ", ""])
def test_safe_destination_rejects_aliases_and_escapes(tmp_path, relative):
    with pytest.raises(ValueError):
        download.safe_destination(tmp_path, relative)


@pytest.mark.parametrize("url", ["http://nemar.s3.us-east-2.amazonaws.com/x", "https://nemar.s3.evil.example/x", "https://nemar.s3.us-east-2.amazonaws.com.evil.example/x", "https://user@nemar.s3.us-east-2.amazonaws.com/x"])
def test_manifest_rejects_untrusted_sources(tmp_path, url):
    row = asset()
    row["source_url"] = url
    with pytest.raises(ValueError, match="NEMAR"):
        download.validate_manifest(pd.DataFrame([row]), tmp_path)


def test_filters_preserve_manifest_order_and_accept_text_and_csv(tmp_path):
    frame = pd.DataFrame([asset("sub-002"), asset("sub-001"), asset("sub-003")])
    text_file = tmp_path / "subjects.txt"
    text_file.write_text("# comment\nsub-001\n\nsub-003\n")
    chosen = download.filter_manifest(frame, ["sub-002"], text_file, 2)
    assert chosen["subject_id"].tolist() == ["sub-002", "sub-001"]
    csv_file = tmp_path / "subjects.csv"
    pd.DataFrame({"subject_id": ["sub-003", "sub-001"]}).to_csv(csv_file, index=False)
    assert download.filter_manifest(frame, subjects_file=csv_file)["subject_id"].tolist() == ["sub-001", "sub-003"]
    with pytest.raises(ValueError, match="absent"):
        download.filter_manifest(frame, ["sub-404"])
    with pytest.raises(ValueError, match="positive"):
        download.filter_manifest(frame, max_subjects=0)


@pytest.mark.parametrize("explicit_dry_run", [False, True])
def test_cli_defaults_to_write_free_network_free_dry_run(tmp_path, explicit_dry_run):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([asset()]).to_csv(manifest, index=False)
    destination = tmp_path / "raw"
    journal = tmp_path / "journal.jsonl"
    arguments = ["--manifest", str(manifest), "--destination", str(destination), "--journal", str(journal)]
    if explicit_dry_run:
        arguments.append("--dry-run")
    assert download.main(arguments) == 0
    assert not destination.exists()
    assert not journal.exists()


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_cli_failure_exit_and_continue_policy(monkeypatch, tmp_path, continue_on_error):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([asset("sub-001"), asset("sub-002")]).to_csv(manifest, index=False)
    journal = tmp_path / "journal.jsonl"
    calls = []
    copied = []

    def fetch(row, target, **kwargs):
        calls.append(row["subject_id"])
        if row["subject_id"] == "sub-001":
            raise TimeoutError("synthetic")
        return "verified_existing"

    monkeypatch.setattr(download, "fetch_one", fetch)
    monkeypatch.setattr(download, "copy_sidecars", lambda subjects, *args: copied.extend(subjects) or 0)
    arguments = ["--manifest", str(manifest), "--destination", str(tmp_path / "raw"), "--journal", str(journal), "--execute"]
    if continue_on_error:
        arguments.append("--continue-on-error")
    assert download.main(arguments) == 1
    assert calls == (["sub-001", "sub-002"] if continue_on_error else ["sub-001"])
    assert copied == (["sub-002"] if continue_on_error else [])
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert events[0]["status"] == "asset_failed"


def test_cli_invalid_manifest_has_validation_exit_code(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject_id\nsub-001\n")
    assert download.main(["--manifest", str(manifest)]) == 2


def test_sidecar_copy_is_idempotent_and_refuses_differences(tmp_path):
    metadata = tmp_path / "metadata"
    source = metadata / "sub-001" / "eeg"
    source.mkdir(parents=True)
    (source / "sub-001_task-resting_eeg.vmrk").write_text("synthetic marker metadata")
    (source / "sub-001_task-resting_eeg.eeg").write_bytes(b"excluded payload")
    raw = tmp_path / "raw"
    assert download.copy_sidecars(["sub-001"], metadata, raw) == 1
    assert download.copy_sidecars(["sub-001"], metadata, raw) == 0
    (source / "sub-001_task-resting_eeg.vmrk").write_text("changed metadata")
    with pytest.raises(ValueError, match="differing sidecar"):
        download.copy_sidecars(["sub-001"], metadata, raw)
    assert not (raw / "sub-001" / "eeg" / "sub-001_task-resting_eeg.eeg").exists()


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"timeout": float("nan")}, {"retries": -1}, {"backoff": -1}, {"chunk_size": 0}])
def test_invalid_retry_options_fail_before_request(tmp_path, kwargs):
    with pytest.raises(ValueError):
        download.fetch_one(asset(), tmp_path / "asset.eeg", **kwargs)
