"""Sequential Train fitting followed by optional frozen mechanical held-out application.

Planning is read-only. Execution and raw cleanup are separate explicit choices.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Iterator
from uuid import uuid4

import pandas as pd
import yaml

from resting_eeg.common import canonical_hash
from resting_eeg.data.environment import system_drive_guard, SystemDriveLowSpaceError
from resting_eeg.data.network_wait import ManuallyPaused
from scripts.data import s2_06_download_manifest as downloader


PROCESSING_OVERHEAD_BYTES = 512 * 1024**2
COMPACT_OUTPUT_ALLOWANCE_BYTES = 32 * 1024**2
SAFETY_MARGIN_BYTES = 1024**3
REQUIRED_SIDECARS = (
    "channels.tsv", "eeg.json", "eeg.vmrk", "events.json", "events.tsv",
)


def process_subject(subject_id: str, **kwargs: Any) -> dict[str, Any]:
    """Lazy import keeps dry runs independent of the waveform implementation."""
    from resting_eeg.preprocessing.pipeline import process_subject as process

    return process(subject_id, **kwargs)


def verify_output(path: Path, config_hash: str, expected_blocks: list) -> dict:
    from resting_eeg.preprocessing.pipeline import verify_output as verify

    return verify(path, config_hash, expected_blocks)


def _frozen_config_hash(config_path: Path) -> str:
    """Match the pipeline's canonical parsed-config digest and frozen receipt."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    digest = canonical_hash(config)
    frozen_digest = config_path.with_suffix(".sha256")
    if not frozen_digest.is_file() or frozen_digest.read_text(encoding="utf-8").strip() != digest:
        raise ValueError("Frozen configuration digest absent or mismatched")
    return digest


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _assert_no_links(path: Path) -> None:
    """Refuse symlinks and Windows junction/reparse points, including ancestors."""
    for item in (path, *path.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError(f"symlink or reparse point is not a safe staging path: {item}")


def safe_subject_directory(raw_root: Path, metadata_root: Path, subject_id: str) -> Path:
    if not re.fullmatch(r"sub-[0-9]+", subject_id):
        raise ValueError(f"invalid subject ID: {subject_id!r}")
    _assert_no_links(raw_root.absolute())
    target = raw_root.absolute() / subject_id
    _assert_no_links(target)
    root = raw_root.resolve()
    metadata = metadata_root.resolve()
    target = target.resolve()
    if target.parent != root or target == root:
        raise ValueError("raw subject directory must be a direct child of raw root")
    if _is_within(target, metadata) or _is_within(metadata, target) or root == metadata:
        raise ValueError("raw staging path overlaps the read-only metadata clone")
    return target


def validate_metadata(subject_id: str, metadata_root: Path) -> None:
    directory = metadata_root / subject_id / "eeg"
    for suffix in REQUIRED_SIDECARS:
        source = directory / f"{subject_id}_task-resting_{suffix}"
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"required metadata sidecar unavailable: {source}")


def _free_bytes(path: Path) -> int:
    existing = path.resolve()
    while not existing.exists():
        if existing.parent == existing:
            raise FileNotFoundError(f"no existing parent for disk-space check: {path}")
        existing = existing.parent
    return int(shutil.disk_usage(existing).free)


def _manifest_rows(frame: pd.DataFrame, raw_root: Path) -> list:
    if frame.empty:
        raise ValueError("no subjects selected for streaming")
    rows = downloader.validate_manifest(frame, raw_root)
    roles: dict[str, list[str]] = {}
    for row, target in rows:
        _assert_no_links(target)
        subject_id = str(row["subject_id"])
        relative = PurePosixPath(str(row["relative_path"]))
        _assert_no_links(raw_root.joinpath(*relative.parts).absolute())
        if len(relative.parts) != 3 or relative.parts[:2] != (subject_id, "eeg"):
            raise ValueError(f"payload path does not belong to subject {subject_id}")
        if relative.name != f"{subject_id}_task-resting_eeg{target.suffix.lower()}":
            raise ValueError(f"unexpected subject payload filename: {relative.name}")
        roles.setdefault(subject_id, []).append(target.suffix.lower())
    for subject_id, suffixes in roles.items():
        if sorted(suffixes) != [".eeg", ".vhdr"]:
            raise ValueError(f"subject {subject_id} must have exactly one EEG and one header payload")
    return rows


def _subject_plan(subject_id: str, rows: list, raw_root: Path, output_root: Path) -> dict:
    missing = 0
    existing = True
    source_payloads = []
    for row, target in rows:
        size = int(row["payload_size_bytes"])
        source_payloads.append({
            "relative_path": str(row["relative_path"]),
            "payload_size_bytes": size,
            "payload_sha256": str(row["payload_sha256"]).lower(),
        })
        if target.is_file() and target.stat().st_size == size:
            continue
        existing = False
        partial = target.with_name(target.name + ".part")
        _assert_no_links(partial)
        retained = partial.stat().st_size if partial.is_file() else 0
        missing += size - retained if 0 <= retained <= size else size
    required = missing + PROCESSING_OVERHEAD_BYTES + COMPACT_OUTPUT_ALLOWANCE_BYTES + SAFETY_MARGIN_BYTES
    raw_free = _free_bytes(raw_root)
    output_free = _free_bytes(output_root)
    return {
        "subject_id": subject_id,
        "scope": "provisional_Train_only",
        "status": "planned",
        "source_payloads": source_payloads,
        "expected_payload_bytes": sum(item["payload_size_bytes"] for item in source_payloads),
        "missing_payload_bytes": missing,
        "processing_overhead_bytes": PROCESSING_OVERHEAD_BYTES,
        "compact_output_allowance_bytes": COMPACT_OUTPUT_ALLOWANCE_BYTES,
        "safety_margin_bytes": SAFETY_MARGIN_BYTES,
        "required_free_bytes": required,
        "available_raw_free_bytes": raw_free,
        "available_output_free_bytes": output_free,
        "disk_ok": raw_free >= required and output_free >= COMPACT_OUTPUT_ALLOWANCE_BYTES + SAFETY_MARGIN_BYTES,
        "local_payloads_present": existing,
    }


@contextmanager
def _journal_lock(path: Path) -> Iterator[None]:
    """An OS lock is released automatically after a crash; stale files are safe."""
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another streaming process holds {lock_path}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _recover_journal(path: Path) -> list[dict]:
    """Only discard an incomplete final JSON write; interior corruption is fatal."""
    if not path.exists():
        return []
    records = []
    with path.open("r+b") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("journal records must be objects")
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise ValueError(f"corrupt complete journal record at byte {offset}")
                handle.seek(offset)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                break
            records.append(record)
            if not line.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                break
    return records


def append_status(journal: Path, run_id: str, subject_id: str, status: str, **details: Any) -> dict:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "subject_id": subject_id,
        "status": status,
        "scope": details.get("result", {}).get("scope", details.get("plan", {}).get("scope", "provisional_Train_only")),
        **details,
    }
    payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with journal.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _verify_receipt(result: dict, subject_id: str, config_path: Path, output_root: Path) -> dict:
    required = {"subject_id", "output_path", "output_sha256", "output_bytes", "config_sha256",
                "source_hash_verified", "processed_verified", "expected_blocks", "qc_pass"}
    if required - result.keys():
        raise ValueError(f"processing result lacks verification fields: {sorted(required - result.keys())}")
    if result["subject_id"] != subject_id:
        raise ValueError("processing receipt belongs to a different subject")
    if result["source_hash_verified"] is not True or result["processed_verified"] is not True:
        raise ValueError("source and processed verification must both pass")
    config_hash = _frozen_config_hash(config_path)
    if result["config_sha256"] != config_hash:
        raise ValueError("processing receipt config hash does not match current frozen config")
    from resting_eeg.data.paths import relocated_path
    output = relocated_path(result["output_path"])
    if not _is_within(output, output_root.resolve()) or output == output_root.resolve():
        raise ValueError("compact output escapes output root")
    _assert_no_links(output)
    if not result["expected_blocks"] or len(result["expected_blocks"]) != len(set(result["expected_blocks"])):
        raise ValueError("processing receipt needs a nonempty, unique expected-block list")
    downloader.verify_file(output, int(result["output_bytes"]), str(result["output_sha256"]))
    if result.get("scope") == "all_blocks_mechanical_provisional_v1":
        from resting_eeg.preprocessing.all_blocks import verify_all_block_output
        checked = verify_all_block_output(output, config_hash, result["authoritative_blocks"],
                                          result["mechanical_access_config_sha256"], result["train_parameter_source_hash"])
        for key in ("fit_parameters_from", "heldout_parameters_are", "train_parameter_source_hash",
                    "mechanical_access_config_sha256", "authoritative_blocks", "scope"):
            required.add(key)
        if result["fit_parameters_from"] != "Train_only" or result["heldout_parameters_are"] != "apply_only":
            raise ValueError("Held-out parameters must be apply-only")
    else:
        checked = verify_output(output, config_hash, result["expected_blocks"])
    if checked.get("processed_verified") is not True:
        raise ValueError("compact output verification did not affirm success")
    receipt_path = output.with_suffix(".provenance.json")
    _assert_no_links(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key in required:
        if key == 'output_path' and relocated_path(receipt.get(key, '')) == relocated_path(result[key]):
            continue
        if receipt.get(key) != result[key]:
            raise ValueError(f"persisted provenance receipt disagrees on {key}")
    return {**result, "output_path": str(output), "receipt_path": str(receipt_path)}


def delete_verified_raw(
    subject_id: str, *, raw_root: Path, metadata_root: Path, output_root: Path,
    config_path: Path, journal: Path, run_id: str, result: dict,
    source_payloads: list[dict], explicitly_requested: bool,
) -> None:
    """Delete only after current verification and a durable completion receipt."""
    if not explicitly_requested:
        raise PermissionError("raw deletion requires --delete-raw-after-success")
    if result.get("scope") != "all_blocks_mechanical_provisional_v1" or len(result.get("expected_blocks", [])) != 16:
        raise ValueError("Raw deletion requires verified all-block output; Train-only output is insufficient")
    checked = _verify_receipt(result, subject_id, config_path, output_root)
    if checked["qc_pass"] is not True:
        raise ValueError("raw cleanup requires passing subject QC")
    validate_metadata(subject_id, metadata_root)
    target = safe_subject_directory(raw_root, metadata_root, subject_id)
    for path in (output_root.resolve(), journal.resolve(), config_path.resolve()):
        if _is_within(path, target):
            raise ValueError("cleanup would delete output, configuration, or provenance journal")
    if not target.is_dir():
        raise FileNotFoundError(f"staged raw subject directory is unavailable: {target}")
    for item in target.rglob("*"):
        _assert_no_links(item)
        if item.name.endswith(".part"):
            raise ValueError("unresolved partial payload blocks raw cleanup")
    if not source_payloads:
        raise ValueError("raw cleanup requires current source manifest fingerprints")
    for row in source_payloads:
        source = downloader.safe_destination(raw_root, row["relative_path"])
        if not _is_within(source, target):
            raise ValueError("source payload does not belong to cleanup subject")
        downloader.verify_file(source, int(row["payload_size_bytes"]), row["payload_sha256"])
    completed = _recover_journal(journal)
    if not any(record.get("subject_id") == subject_id and record.get("run_id") == run_id
               and record.get("status") == "complete" and record.get("result") == checked
               and record.get("source_payloads") == source_payloads for record in completed):
        raise ValueError("raw cleanup requires a durable verified completion journal record")
    bytes_freed = sum(item.stat().st_size for item in target.rglob('*') if item.is_file())
    append_status(journal, run_id, subject_id, "raw_delete_pending", raw_directory=str(target), bytes_to_free=bytes_freed)
    shutil.rmtree(target)
    if target.exists():
        raise OSError('Guarded raw deletion did not remove subject directory')
    append_status(journal, run_id, subject_id, "raw_deleted", raw_directory=str(target), bytes_freed=bytes_freed)


def stream_subjects(
    manifest: pd.DataFrame, *, repo_root: Path, raw_root: Path, metadata_root: Path,
    output_root: Path, config_path: Path, metrics_root: Path, execute: bool = False,
    local_existing_only: bool = False, delete_raw_after_success: bool = False,
    continue_on_error: bool = False, retries: int = 3, timeout: float = 60,
    journal_path: Path | None = None,
    processing_scope: str = "train-only", access_config_path: Path | None = None,
    train_output_root: Path | None = None, fit_parameters_from: str = "Train_only",
    download_journal_path: Path | None = None,
    temp_root: Path | None = None,
    network_wait=None,
) -> list[dict]:
    """Plan or process each selected subject sequentially; return terminal records.

    Existing raw is re-hashed even on reruns. The preprocessing callable validates
    and reuses an intact output under the same config. A previously cleaned subject
    may be skipped only with durable matching source fingerprints and reverified
    output/provenance, so an interrupted cleanup does not trigger redownloading.
    """
    repo_root, raw_root, metadata_root = Path(repo_root), Path(raw_root), Path(metadata_root)
    output_root, config_path, metrics_root = Path(output_root), Path(config_path), Path(metrics_root)
    for writable in (output_root, metrics_root, *([Path(train_output_root)] if train_output_root else [])):
        _assert_no_links(writable.absolute())
        if _is_within(writable.resolve(), metadata_root.resolve()):
            raise ValueError("output and metrics paths must not modify the read-only metadata clone")
    if processing_scope not in ("train-only", "all-blocks-mechanical") or fit_parameters_from != "Train_only":
        raise ValueError("Invalid processing scope or attempted held-out fitting")
    if processing_scope == "all-blocks-mechanical":
        from resting_eeg.data.mechanical_access import load_access_config
        if access_config_path is None:
            raise ValueError("All-block mode requires mechanical access configuration")
        load_access_config(access_config_path, config_path)
    validated = _manifest_rows(manifest, raw_root)
    grouped: dict[str, list] = {}
    for row, target in validated:
        grouped.setdefault(str(row["subject_id"]), []).append((row, target))
    for subject_id in grouped:
        subject_dir = safe_subject_directory(raw_root, metadata_root, subject_id)
        for protected in (output_root.resolve(), metrics_root.resolve(), config_path.resolve(), *([Path(train_output_root).resolve()] if train_output_root else [])):
            if _is_within(protected, subject_dir):
                raise ValueError("output, metrics, and config must be outside raw subject directories")
    if not execute:
        return [dict(_subject_plan(subject_id, rows, raw_root, output_root), processing_scope=processing_scope,
                     scope="all_blocks_mechanical_provisional_v1" if processing_scope == "all-blocks-mechanical" else "provisional_Train_only")
                for subject_id, rows in grouped.items()]
    if not config_path.is_file():
        raise FileNotFoundError(f"frozen preprocessing configuration unavailable: {config_path}")
    _frozen_config_hash(config_path)
    journal = Path(journal_path) if journal_path is not None else metrics_root / ("streaming_all_block_status.jsonl" if processing_scope == "all-blocks-mechanical" else "streaming_subject_status.jsonl")
    _assert_no_links(journal.absolute())
    _assert_no_links(journal.with_name(journal.name + ".lock").absolute())
    if _is_within(journal.resolve(), metadata_root.resolve()):
        raise ValueError("status journal must not modify the read-only metadata clone")
    for subject_id in grouped:
        if _is_within(journal.resolve(), (raw_root / subject_id).resolve()):
            raise ValueError("status journal must be outside raw subject directories")
    journal.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    terminal = []
    def record_status(*args, **details):
        details.setdefault('scope', 'all_blocks_mechanical_provisional_v1' if processing_scope == 'all-blocks-mechanical' else 'provisional_Train_only')
        return append_status(*args, **details)
    with _journal_lock(journal):
        history = _recover_journal(journal)
        download_journal = downloader.DownloadJournal(download_journal_path or metrics_root / "streaming_download_status.jsonl")
        for subject_id, rows in grouped.items():
            stage = "planning"
            try:
                plan = _subject_plan(subject_id, rows, raw_root, output_root)
                if processing_scope == "all-blocks-mechanical":
                    plan["scope"] = "all_blocks_mechanical_provisional_v1"
                record_status(journal, run_id, subject_id, "planned", plan=plan)
                validate_metadata(subject_id, metadata_root)
                subject_dir = safe_subject_directory(raw_root, metadata_root, subject_id)
                prior = next((record for record in reversed(history)
                              if record.get("subject_id") == subject_id and record.get("status") == "complete"
                              and record.get("source_payloads") == plan["source_payloads"]
                              and (record.get("result", {}).get("scope") == "all_blocks_mechanical_provisional_v1") == (processing_scope == "all-blocks-mechanical")), None)
                if not subject_dir.exists() and prior is not None and prior.get("cleanup_requested"):
                    result = _verify_receipt(prior["result"], subject_id, config_path, output_root)
                    terminal.append(record_status(journal, run_id, subject_id, "complete", result=result,
                                                  source_payloads=plan["source_payloads"], raw_retained=False,
                                                  cleanup_requested=True, reused_after_cleanup=True))
                    continue
                if not plan["disk_ok"]:
                    raise OSError(f"insufficient free disk: need {plan['required_free_bytes']} bytes at raw destination")
                if local_existing_only and not plan["local_payloads_present"]:
                    raise FileNotFoundError(f"local-existing-only requires both complete payloads for {subject_id}")
                stage = "downloading"
                system_drive_guard('begin_or_resume_subject_download')
                record_status(journal, run_id, subject_id, "downloading", local_existing_only=local_existing_only)
                for row, target in rows:
                    if local_existing_only:
                        downloader.verify_file(target, int(row["payload_size_bytes"]), str(row["payload_sha256"]))
                    else:
                        options = {'network_wait': network_wait} if network_wait is not None else {}
                        downloader.fetch_one(row, target, retries=retries, timeout=timeout, journal=download_journal, **options)
                downloader.copy_sidecars([subject_id], metadata_root, raw_root)
                record_status(journal, run_id, subject_id, "download_verified", source_payloads=plan["source_payloads"])
                stage = "preprocessing"
                if network_wait is not None:
                    network_wait.check_stop()
                    network_wait.update(state='preprocessing', current_asset=None)
                system_drive_guard('start_subject_preprocessing')
                record_status(journal, run_id, subject_id, "preprocessing")
                kwargs = dict(repo_root=repo_root, raw_root=raw_root, output_root=output_root,
                              config_path=config_path, metrics_root=metrics_root)
                if temp_root is not None:
                    kwargs['temp_root'] = Path(temp_root)
                if processing_scope == "all-blocks-mechanical":
                    from resting_eeg.preprocessing.all_blocks import process_all_blocks
                    result = process_all_blocks(subject_id, **kwargs, access_config_path=access_config_path,
                                                train_output_root=train_output_root, fit_parameters_from=fit_parameters_from)
                else:
                    result = process_subject(subject_id, **kwargs)
                result = _verify_receipt(result, subject_id, config_path, output_root)
                record_status(journal, run_id, subject_id, "processed_verified", result=result)
                complete = record_status(journal, run_id, subject_id, "complete", result=result,
                                         source_payloads=plan["source_payloads"], raw_retained=True,
                                         cleanup_requested=delete_raw_after_success)
                if delete_raw_after_success:
                    stage = "cleanup"
                    if result["qc_pass"] is not True:
                        terminal.append(record_status(journal, run_id, subject_id, "complete", result=result,
                            source_payloads=plan["source_payloads"], raw_retained=True,
                            cleanup_requested=True, cleanup_blocker="frozen_Train_QC_failed"))
                        continue
                    delete_verified_raw(subject_id, raw_root=raw_root, metadata_root=metadata_root,
                                        output_root=output_root, config_path=config_path, journal=journal,
                                        run_id=run_id, result=result, source_payloads=plan["source_payloads"],
                                        explicitly_requested=True)
                    complete = record_status(journal, run_id, subject_id, "complete", result=result,
                                             source_payloads=plan["source_payloads"], raw_retained=False,
                                             cleanup_requested=True)
                terminal.append(complete)
            except (KeyboardInterrupt, ManuallyPaused):
                record_status(journal, run_id, subject_id, 'manually_paused', raw_preserved=True, terminal=False)
                raise
            except Exception as exc:
                status = {"downloading": "download_interrupted", "preprocessing": "preprocessing_failed",
                          "cleanup": "cleanup_failed"}.get(stage, "planning_failed")
                if isinstance(exc, SystemDriveLowSpaceError):
                    status = 'c_drive_low_space_guard'
                terminal.append(record_status(journal, run_id, subject_id, status,
                                             error_type=type(exc).__name__, error=str(exc),
                                             raw_preserved=stage != "cleanup",
                                             raw_directory_exists=(raw_root / subject_id).is_dir()))
                if isinstance(exc, SystemDriveLowSpaceError) or not continue_on_error:
                    break
    return terminal
