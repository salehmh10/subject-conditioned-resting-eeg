"""Stage audited NEMAR payloads with verified HTTP Range resume.

The command is a network-free, write-free dry run unless ``--execute`` is set.
The full manifest remains an inventory; prefer subject selection or the streaming
orchestrator for future execution. Retries are additional attempts, not a total.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd


REQUIRED_COLUMNS = {
    "subject_id", "relative_path", "source_url", "payload_size_bytes", "payload_sha256",
}
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


class DownloadIntegrityError(ValueError):
    """A completed payload does not agree with the audited manifest."""


class DownloadProtocolError(ValueError):
    """The HTTP response cannot safely be appended to the partial payload."""


class DownloadJournal:
    """Durable append-only JSONL events for the sequential download workflow."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def record(self, **event: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def safe_destination(root: Path, relative_path: str) -> Path:
    # Reject Windows separators/drive letters and aliases even on POSIX test hosts.
    relative = PurePosixPath(relative_path)
    pieces = relative_path.split("/")
    if (
        relative.is_absolute() or not relative.parts or "\\" in relative_path
        or ":" in relative_path or any(part in {"", ".", ".."} for part in pieces)
        or any(part.rstrip(" .") != part for part in pieces)
    ):
        raise ValueError(f"unsafe manifest path: {relative_path!r}")
    root_resolved = root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if root_resolved not in target.parents:
        raise ValueError(f"manifest path escapes destination: {relative_path!r}")
    return target


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, digest: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise DownloadIntegrityError(f"size mismatch for {path}: expected {size}, found {actual_size}")
    actual_digest = sha256_file(path)
    if actual_digest.lower() != digest.lower():
        raise DownloadIntegrityError(f"SHA256 mismatch for {path}: expected {digest}, found {actual_digest}")


def validate_manifest(frame: pd.DataFrame, destination: Path) -> list[tuple[pd.Series, Path]]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest lacks columns: {sorted(missing)}")
    if frame["relative_path"].duplicated().any():
        raise ValueError("manifest contains duplicate relative_path values")
    validated: list[tuple[pd.Series, Path]] = []
    for _, row in frame.iterrows():
        url = str(row["source_url"])
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in {None, 443} or parsed.fragment
            or not re.fullmatch(r"nemar\.s3\.[a-z0-9-]+\.amazonaws\.com", parsed.hostname or "")
        ):
            raise ValueError(f"unexpected non-NEMAR HTTPS URL: {url}")
        subject = str(row["subject_id"])
        relative = str(row["relative_path"])
        target = safe_destination(destination, relative)
        if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject) or PurePosixPath(relative).parts[0] != subject:
            raise ValueError(f"invalid subject/path association: {subject!r}, {relative!r}")
        digest = str(row["payload_sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"invalid SHA256 for {relative}")
        value = str(row["payload_size_bytes"])
        if not re.fullmatch(r"[0-9]+", value) or int(value) <= 0:
            raise ValueError(f"invalid payload size for {relative}")
        validated.append((row, target))
    return validated


def filter_manifest(
    frame: pd.DataFrame,
    subject_ids: list[str] | None = None,
    subjects_file: Path | str | None = None,
    max_subjects: int | None = None,
) -> pd.DataFrame:
    """Select in manifest order; repeated IDs and file entries form a union.

    Files are newline-delimited IDs (blank/comment lines ignored) or CSV with a
    ``subject_id`` column. Unknown IDs are errors, never silent omissions.
    """
    if max_subjects is not None and max_subjects <= 0:
        raise ValueError("max_subjects must be positive")
    selected = list(subject_ids or [])
    if subjects_file is not None:
        path = Path(subjects_file)
        if path.suffix.lower() == ".csv":
            listed = pd.read_csv(path, dtype=str)
            if "subject_id" not in listed:
                raise ValueError("subjects CSV must contain subject_id")
            selected.extend(listed["subject_id"].dropna().tolist())
        else:
            selected.extend(
                line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        if not selected:
            raise ValueError("subjects file contains no subject IDs")
    known = frame["subject_id"].astype(str).drop_duplicates().tolist()
    if unknown := set(selected) - set(known):
        raise ValueError(f"subject IDs absent from manifest: {sorted(unknown)}")
    subjects = [subject for subject in known if not selected or subject in selected]
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    return frame.loc[frame["subject_id"].astype(str).isin(subjects)].copy()


def _quarantine(partial: Path) -> Path:
    quarantine = partial.with_name(partial.name + ".invalid-" + uuid.uuid4().hex)
    os.replace(partial, quarantine)
    return quarantine


def _validate_response(response: object, offset: int, size: int) -> tuple[str, int]:
    status = response.status
    encoding = response.headers.get("Content-Encoding", "identity").lower()
    if encoding != "identity":
        raise DownloadProtocolError(f"unexpected Content-Encoding: {encoding}")
    if status == 206:
        value = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", value)
        if not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
            raise DownloadProtocolError(f"invalid Content-Range for offset {offset}, expected total {size}: {value!r}")
        mode, expected = ("ab" if offset else "wb"), size - offset
    elif status == 200:
        # A server may ignore Range. Validate headers before discarding the prefix.
        if response.headers.get("Content-Range") is not None:
            raise DownloadProtocolError("200 response unexpectedly contains Content-Range")
        mode, expected = "wb", size
    else:
        raise DownloadProtocolError(f"unexpected HTTP status: {status}")
    length = response.headers.get("Content-Length")
    if length is not None and (not re.fullmatch(r"[0-9]+", length) or int(length) != expected):
        raise DownloadProtocolError(f"Content-Length does not match expected response size {expected}: {length!r}")
    return mode, expected


def fetch_one(
    row: pd.Series,
    target: Path,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 1.0,
    journal: DownloadJournal | None = None,
    progress: Callable[[str], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Verify/fetch one asset; preserve incomplete prefixes on transient failures.

    Existing corrupt final files are preserved and refused. Oversize or completed
    corrupt partials are quarantined before a fresh request. A checksum failure
    discovered after receiving a body quarantines it and raises; rerunning starts
    cleanly. Protocol errors preserve the prefix and never append response bytes.
    ``timeout`` bounds socket operations; retry delays are capped at 60 seconds.
    """
    if not math.isfinite(timeout) or timeout <= 0 or retries < 0:
        raise ValueError("timeout must be positive and finite; retries must be nonnegative")
    if not math.isfinite(backoff) or backoff < 0 or chunk_size <= 0:
        raise ValueError("backoff must be nonnegative and finite; chunk_size must be positive")
    size = int(row["payload_size_bytes"])
    digest = str(row["payload_sha256"])
    partial = target.with_name(target.name + ".part")

    def emit(status: str, **details: object) -> None:
        if journal is not None:
            journal.record(
                status=status, subject_id=str(row["subject_id"]), relative_path=str(row["relative_path"]),
                target=str(target), expected_bytes=size, expected_sha256=digest, **details,
            )
        if progress is not None:
            progress(f"{row['subject_id']} {target.name}: {status}" + (f" {details}" if details else ""))

    def promote(status: str) -> str:
        # Only verified bytes enter the final filename, via one same-directory rename.
        verify_file(partial, size, digest)
        os.replace(partial, target)
        emit(status, bytes=size)
        return status

    if target.exists():
        try:
            verify_file(target, size, digest)
        except DownloadIntegrityError as exc:
            emit("failed", error=str(exc), final_preserved=True)
            raise
        emit("verified_existing", bytes=size)
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.is_symlink():
        raise ValueError(f"refusing symlink partial: {partial}")
    for attempt in range(1, retries + 2):
        # The previous read can fail immediately after delivering all bytes. Check
        # for a complete prefix on every attempt instead of requesting bytes=N-.
        if partial.exists() and partial.stat().st_size >= size:
            if partial.stat().st_size == size:
                try:
                    return promote("resumed_and_verified")
                except DownloadIntegrityError:
                    pass
            quarantine = _quarantine(partial)
            emit("partial_quarantined", quarantine=str(quarantine))
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "scenario2-audited-downloader/2", "Accept-Encoding": "identity"}
        if partial.exists():
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(str(row["source_url"]), headers=headers)
        from resting_eeg.data.environment import system_drive_guard, SystemDriveLowSpaceError
        try:
            system_drive_guard('begin_or_resume_asset_download')
        except SystemDriveLowSpaceError as exc:
            emit('c_drive_low_space_guard', error=str(exc), partial_preserved=partial.exists())
            raise
        emit("downloading", attempt=attempt, resume_offset=offset)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                mode, expected = _validate_response(response, offset, size)
                emit('response_validated', attempt=attempt, http_status=response.status,
                     content_range=response.headers.get('Content-Range'), resume_offset=offset,
                     resumed_prefix_bytes=offset if mode=='ab' else 0)
                if offset and mode == "wb":
                    emit("range_ignored_restarting", attempt=attempt, resume_offset=offset)
                received = 0
                next_report = 64 * 1024 * 1024
                with partial.open(mode) as handle:
                    while True:
                        interrupted = None
                        try:
                            chunk = response.read(min(chunk_size, expected - received + 1))
                        except http.client.IncompleteRead as exc:
                            # Some HTTP readers carry the final valid prefix in the exception.
                            chunk, interrupted = exc.partial, exc
                        if chunk:
                            # Application body bytes, including retries and bytes that
                            # subsequently fail integrity. Never equate file size with traffic.
                            emit("response_body", attempt=attempt, received_body_bytes=len(chunk))
                        if received + len(chunk) > expected:
                            raise DownloadIntegrityError(f"response body exceeds expected {expected} bytes")
                        if chunk:
                            handle.write(chunk)
                            received += len(chunk)
                            if received >= next_report:
                                emit("progress", attempt=attempt, bytes=(offset if mode == "ab" else 0) + received)
                                next_report = received + 64 * 1024 * 1024
                        if interrupted is not None:
                            raise interrupted
                        if not chunk:
                            break
                    handle.flush()
                    os.fsync(handle.fileno())
                if received != expected:
                    raise http.client.IncompleteRead(b"", expected - received)
            return promote("resumed_and_verified" if offset and mode == "ab" else "downloaded_and_verified")
        except DownloadIntegrityError as exc:
            quarantine = _quarantine(partial) if partial.exists() else None
            emit("failed", attempt=attempt, error=str(exc), quarantine=str(quarantine) if quarantine else None)
            raise
        except DownloadProtocolError as exc:
            emit("failed", attempt=attempt, error=str(exc), partial_preserved=partial.exists())
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in RETRYABLE_HTTP_CODES
            emit("download_interrupted", attempt=attempt, error=str(exc), partial_bytes=partial.stat().st_size if partial.exists() else 0)
            if not retryable or attempt > retries:
                emit("failed", attempt=attempt, error=str(exc), partial_preserved=partial.exists())
                raise
            delay = min(60.0, backoff * (2 ** min(attempt - 1, 16)))
            emit("retry_scheduled", attempt=attempt, delay_seconds=delay)
            time.sleep(delay)
    raise AssertionError("unreachable retry state")


def copy_sidecars(subject_ids: list[str], metadata_root: Path, destination: Path) -> int:
    copied = 0
    for subject_id in subject_ids:
        if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject_id):
            raise ValueError(f"invalid subject ID: {subject_id!r}")
        source_dir = safe_destination(metadata_root, f"{subject_id}/eeg")
        if not source_dir.is_dir():
            raise FileNotFoundError(f"missing metadata subject directory: {source_dir}")
        target_dir = safe_destination(destination, f"{subject_id}/eeg")
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.glob(f"{subject_id}_task-resting_*")):
            if source.suffix.lower() in {".eeg", ".vhdr"}:
                continue
            if not source.is_file():
                raise ValueError(f"metadata sidecar is not a materialized file: {source}")
            target = safe_destination(destination, f"{subject_id}/eeg/{source.name}")
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f"refusing to overwrite differing sidecar: {target}")
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root/"src")]
    from resting_eeg.data.paths import load_paths
    storage = load_paths(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", default=str(storage["raw_root"]))
    parser.add_argument("--metadata-root", default=str(storage["metadata_root"]))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="perform downloads and local writes")
    mode.add_argument("--dry-run", action="store_true", help="validate and plan only (the default)")
    parser.add_argument("--subject-id", action="append", default=[], help="select one subject; repeat to select several")
    parser.add_argument("--subjects-file", help="newline-delimited IDs or CSV with subject_id column")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--timeout", type=float, default=60.0, help="socket timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="maximum additional attempts per asset")
    parser.add_argument("--backoff", type=float, default=1.0, help="initial exponential retry delay, capped at 60 seconds")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--journal", default=str(storage["runtime_root"]/"download_journal.jsonl"))
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.retries < 0 or not math.isfinite(args.backoff) or args.backoff < 0:
        parser.error("timeout must be positive and finite; retries/backoff must be nonnegative and finite")
    try:
        manifest = pd.read_csv(args.manifest)
        destination = Path(args.destination)
        validate_manifest(manifest, destination)
        selected = filter_manifest(manifest, args.subject_id, args.subjects_file, args.max_subjects)
        rows = validate_manifest(selected, destination)
        if not rows:
            raise ValueError("no manifest assets selected")
    except (ValueError, OSError) as exc:
        print(f"Manifest validation failed: {exc}", file=sys.stderr)
        return 2
    total_bytes = int(selected["payload_size_bytes"].sum())
    print(f"Validated {len(rows)} assets for {selected['subject_id'].nunique()} subjects ({total_bytes} bytes).")
    if not args.execute:
        for row, target in rows:
            print(f"planned: {row['subject_id']} {target} ({row['payload_size_bytes']} bytes)")
        print("DRY RUN ONLY: no network request or filesystem write was made.")
        return 0

    journal = DownloadJournal(args.journal)
    failed_subjects: set[str] = set()
    for index, (row, target) in enumerate(rows, start=1):
        try:
            status = fetch_one(row, target, timeout=args.timeout, retries=args.retries, backoff=args.backoff, journal=journal, progress=print)
            print(f"[{index}/{len(rows)}] {status}: {target}")
        except (ValueError, OSError, http.client.HTTPException) as exc:
            failed_subjects.add(str(row["subject_id"]))
            print(f"[{index}/{len(rows)}] FAILED: {target}: {exc}", file=sys.stderr)
            journal.record(status="asset_failed", subject_id=str(row["subject_id"]), relative_path=str(row["relative_path"]), error=str(exc))
            if not args.continue_on_error:
                return 1
    copied = 0
    for subject in selected["subject_id"].drop_duplicates():
        if subject in failed_subjects:
            continue
        try:
            count = copy_sidecars([subject], Path(args.metadata_root), destination)
            copied += count
            journal.record(status="subject_complete", subject_id=str(subject), sidecars_copied=count)
        except (ValueError, OSError) as exc:
            failed_subjects.add(str(subject))
            journal.record(status="sidecar_failed", subject_id=str(subject), error=str(exc))
            print(f"FAILED sidecars: {subject}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
    print(f"Finished: {len(failed_subjects)} failed subjects; copied {copied} local metadata sidecars.")
    return 1 if failed_subjects else 0


if __name__ == "__main__":
    raise SystemExit(main())
