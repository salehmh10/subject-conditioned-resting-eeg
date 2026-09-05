"""Streaming tests use only tiny temporary payloads and synthetic HDF5 outputs."""

import hashlib
import json
from pathlib import Path

import h5py
import pandas as pd
import pytest
import yaml

from resting_eeg.common import canonical_hash
from resting_eeg.data import streaming


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def config_digest(path: Path) -> str:
    return canonical_hash(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    paths = {"repo_root": tmp_path, "raw_root": tmp_path / "raw", "metadata_root": tmp_path / "metadata",
             "output_root": tmp_path / "processed", "config_path": tmp_path / "frozen.yaml",
             "metrics_root": tmp_path / "metrics"}
    paths["config_path"].write_text("scope: Train\n", encoding="utf-8")
    paths["config_path"].with_suffix(".sha256").write_text(config_digest(paths["config_path"]) + "\n", encoding="utf-8")
    rows = []
    for subject in ("sub-001", "sub-002"):
        for suffix in (".eeg", ".vhdr"):
            relative = f"{subject}/eeg/{subject}_task-resting_eeg{suffix}"
            payload = (subject + suffix).encode()
            target = paths["raw_root"] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            rows.append({"subject_id": subject, "relative_path": relative,
                         "source_url": "https://nemar.s3.us-east-2.amazonaws.com/synthetic-test-only",
                         "payload_size_bytes": len(payload), "payload_sha256": digest(payload)})
        metadata = paths["metadata_root"] / subject / "eeg"
        metadata.mkdir(parents=True)
        for suffix in streaming.REQUIRED_SIDECARS:
            (metadata / f"{subject}_task-resting_{suffix}").write_text("synthetic sidecar\n")

    calls = []

    def process(subject_id, **kwargs):
        calls.append(subject_id)
        output = kwargs["output_root"] / f"{subject_id}.h5"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            with h5py.File(output, "w") as handle:
                handle.attrs["config_sha256"] = config_digest(paths["config_path"])
                handle.create_dataset("Train/EO/block-01", data=[[0.0, 1.0]], dtype="float32")
                handle.create_dataset("Train/EC/block-02", data=[[2.0, 3.0]], dtype="float32")
        result = {"subject_id": subject_id, "output_path": str(output), "output_sha256": digest(output.read_bytes()),
                  "output_bytes": output.stat().st_size, "config_sha256": config_digest(paths["config_path"]),
                  "source_hash_verified": True, "processed_verified": True,
                  "expected_blocks": ["Train/EO/block-01", "Train/EC/block-02"], "qc_pass": True}
        output.with_suffix(".provenance.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    def verify(path, config_hash, expected_blocks):
        with h5py.File(path, "r") as handle:
            if handle.attrs["config_sha256"] != config_hash or any(block not in handle for block in expected_blocks):
                raise ValueError("synthetic HDF5 verification failed")
        return {"processed_verified": True}

    def no_network(*args, **kwargs):
        pytest.fail("tests must never request EEG payloads")

    monkeypatch.setattr(streaming, "process_subject", process)
    monkeypatch.setattr(streaming, "verify_output", verify)
    monkeypatch.setattr(streaming.downloader, "fetch_one", no_network)
    monkeypatch.setattr(streaming, "_free_bytes", lambda path: 10 * 1024**3)
    return pd.DataFrame(rows), paths, calls


def one(frame):
    return frame.loc[frame["subject_id"] == "sub-001"].copy()


def journal_records(paths):
    return [json.loads(line) for line in (paths["metrics_root"] / "streaming_subject_status.jsonl").read_text().splitlines()]


def test_dry_run_is_read_only_even_when_delete_flag_is_selected(staged):
    frame, paths, calls = staged
    before = {str(path.relative_to(paths["repo_root"])) for path in paths["repo_root"].rglob("*")}
    results = streaming.stream_subjects(frame, **paths, delete_raw_after_success=True)
    after = {str(path.relative_to(paths["repo_root"])) for path in paths["repo_root"].rglob("*")}
    assert before == after
    assert calls == []
    assert [result["status"] for result in results] == ["planned", "planned"]
    assert all(result["scope"] == "provisional_Train_only" for result in results)
    assert all(result["missing_payload_bytes"] == 0 for result in results)


def test_local_existing_processes_sequentially_and_keeps_raw_by_default(staged):
    frame, paths, calls = staged
    results = streaming.stream_subjects(frame, **paths, execute=True, local_existing_only=True)
    assert calls == ["sub-001", "sub-002"]
    assert all(result["status"] == "complete" and result["raw_retained"] for result in results)
    assert all((paths["raw_root"] / subject).is_dir() for subject in calls)
    assert [record["status"] for record in journal_records(paths)][:5] == [
        "planned", "downloading", "download_verified", "preprocessing", "processed_verified"]


def test_existing_output_is_reverified_on_restart(staged):
    frame, paths, calls = staged
    first = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True)[0]
    output = Path(first["result"]["output_path"])
    modified = output.stat().st_mtime_ns
    second = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True)[0]
    assert second["result"]["output_sha256"] == first["result"]["output_sha256"]
    assert output.stat().st_mtime_ns == modified
    assert calls == ["sub-001", "sub-001"]
    payload = paths["raw_root"] / str(frame.iloc[0]["relative_path"])
    payload.write_bytes(b"X" * payload.stat().st_size)
    failed = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True)[0]
    assert failed["status"] == "download_interrupted"
    assert "SHA256 mismatch" in failed["error"]
    assert calls == ["sub-001", "sub-001"]


def test_semantically_identical_yaml_keeps_canonical_frozen_digest(staged):
    frame, paths, calls = staged
    first = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True)[0]
    initial_config_bytes = paths["config_path"].read_bytes()
    paths["config_path"].write_text('# Formatting changes do not alter the frozen method.\nscope: "Train"\n', encoding="utf-8")
    assert paths["config_path"].read_bytes() != initial_config_bytes
    second = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True)[0]
    assert first["status"] == second["status"] == "complete"
    assert first["result"]["config_sha256"] == second["result"]["config_sha256"]
    assert first["result"]["output_sha256"] == second["result"]["output_sha256"]
    assert calls == ["sub-001", "sub-001"]


@pytest.mark.parametrize("alteration", ["missing", "mismatched", "changed_config"])
def test_frozen_digest_is_required_before_processing(staged, alteration):
    frame, paths, calls = staged
    frozen_digest = paths["config_path"].with_suffix(".sha256")
    if alteration == "missing":
        frozen_digest.unlink()
    elif alteration == "mismatched":
        frozen_digest.write_text("0" * 64, encoding="utf-8")
    else:
        paths["config_path"].write_text("scope: heldout\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen configuration digest"):
        streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True,
                                  delete_raw_after_success=True)
    assert calls == []
    assert (paths["raw_root"] / "sub-001").is_dir()
    assert not paths["output_root"].exists()


def test_missing_local_asset_preserves_partial_and_never_downloads(staged):
    frame, paths, calls = staged
    target = paths["raw_root"] / str(frame.iloc[0]["relative_path"])
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(target.read_bytes()[:4])
    target.unlink()
    result = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True)[0]
    assert result["status"] == "planning_failed"
    assert partial.read_bytes() == b"sub-"
    assert calls == []
    plan = streaming.stream_subjects(one(frame), **paths)[0]
    assert plan["missing_payload_bytes"] == int(frame.iloc[0]["payload_size_bytes"]) - 4


def test_preprocessing_failure_preserves_raw_and_partial_and_can_continue(staged, monkeypatch):
    frame, paths, calls = staged
    process = streaming.process_subject

    def fail_first(subject_id, **kwargs):
        if subject_id == "sub-001":
            raise RuntimeError("synthetic processing failure")
        return process(subject_id, **kwargs)

    monkeypatch.setattr(streaming, "process_subject", fail_first)
    partial = paths["raw_root"] / "sub-001/eeg/interrupted.part"
    partial.write_bytes(b"keep")
    results = streaming.stream_subjects(frame, **paths, execute=True, local_existing_only=True,
                                       delete_raw_after_success=True, continue_on_error=True)
    assert [result["status"] for result in results] == ["preprocessing_failed", "complete"]
    assert (paths["raw_root"] / "sub-001").is_dir()
    assert partial.read_bytes() == b"keep"
    assert not (paths["raw_root"] / "sub-002").exists()


def test_disk_shortage_stops_before_processing(staged, monkeypatch):
    frame, paths, calls = staged
    monkeypatch.setattr(streaming, "_free_bytes", lambda path: 1024)
    result = streaming.stream_subjects(frame, **paths, execute=True, local_existing_only=True)[0]
    assert result["status"] == "planning_failed"
    assert "insufficient free disk" in result["error"]
    assert calls == []


def test_explicit_deletion_happens_after_durable_complete_and_is_restartable(staged, monkeypatch):
    frame, paths, calls = staged
    original_delete = streaming.shutil.rmtree

    def checked_delete(path):
        records = journal_records(paths)
        assert records[-1]["status"] == "raw_delete_pending"
        assert records[-2]["status"] == "complete"
        assert records[-2]["result"]["source_hash_verified"]
        assert records[-2]["result"]["processed_verified"]
        original_delete(path)

    monkeypatch.setattr(streaming.shutil, "rmtree", checked_delete)
    result = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True,
                                      delete_raw_after_success=True)[0]
    assert result["status"] == "complete" and result["raw_retained"] is False
    assert not (paths["raw_root"] / "sub-001").exists()
    again = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True)[0]
    assert again["reused_after_cleanup"] is True
    assert calls == ["sub-001"]
    assert (paths["metadata_root"] / "sub-001/eeg").is_dir()


def test_delete_helper_requires_explicit_flag(staged):
    frame, paths, _ = staged
    with pytest.raises(PermissionError, match="explicit|requires"):
        streaming.delete_verified_raw("sub-001", raw_root=paths["raw_root"], metadata_root=paths["metadata_root"],
                                      output_root=paths["output_root"], config_path=paths["config_path"],
                                      journal=paths["metrics_root"] / "journal.jsonl", run_id="test", result={},
                                      source_payloads=[], explicitly_requested=False)


@pytest.mark.parametrize("alteration", ["qc", "source", "processed", "hash", "receipt", "blocks", "config", "raw_config_hash", "freeze", "metadata", "partial"])
def test_deletion_requires_every_verification(staged, monkeypatch, alteration):
    frame, paths, _ = staged
    process = streaming.process_subject

    def altered(subject_id, **kwargs):
        result = process(subject_id, **kwargs)
        receipt = Path(result["output_path"]).with_suffix(".provenance.json")
        if alteration == "qc":
            result["qc_pass"] = False
        elif alteration == "source":
            result["source_hash_verified"] = False
        elif alteration == "processed":
            result["processed_verified"] = False
        elif alteration == "hash":
            result["output_sha256"] = "0" * 64
        elif alteration == "blocks":
            result["expected_blocks"] = ["missing-block"]
        elif alteration == "config":
            paths["config_path"].write_text("changed: true\n")
        elif alteration == "raw_config_hash":
            result["config_sha256"] = digest(paths["config_path"].read_bytes())
        elif alteration == "freeze":
            paths["config_path"].with_suffix(".sha256").unlink()
        elif alteration == "metadata":
            (paths["metadata_root"] / subject_id / "eeg" / f"{subject_id}_task-resting_eeg.vmrk").unlink()
        elif alteration == "partial":
            (paths["raw_root"] / subject_id / "eeg" / "unfinished.part").write_bytes(b"preserve")
        receipt.write_text(json.dumps(result), encoding="utf-8")
        if alteration == "receipt":
            receipt.write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(streaming, "process_subject", altered)
    result = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True,
                                      delete_raw_after_success=True)[0]
    assert result["status"] in {"preprocessing_failed", "cleanup_failed"}
    assert (paths["raw_root"] / "sub-001").is_dir()
    assert not any(record["status"] == "raw_deleted" for record in journal_records(paths))


def test_path_safety_rejects_metadata_overlap_and_subject_escape(staged):
    _, paths, _ = staged
    with pytest.raises(ValueError, match="metadata clone"):
        streaming.safe_subject_directory(paths["metadata_root"], paths["metadata_root"], "sub-001")
    with pytest.raises(ValueError, match="subject ID"):
        streaming.safe_subject_directory(paths["raw_root"], paths["metadata_root"], "../metadata")


def test_manifest_subject_path_mismatch_is_rejected(staged):
    frame, paths, _ = staged
    frame.loc[0, "subject_id"] = "sub-999"
    with pytest.raises(ValueError, match="subject"):
        streaming.stream_subjects(frame, **paths)


def test_empty_manifest_cannot_claim_success(staged):
    frame, paths, _ = staged
    with pytest.raises(ValueError, match="no subjects selected"):
        streaming.stream_subjects(frame.iloc[:0], **paths)


def test_symlink_staging_is_refused(staged):
    _, paths, _ = staged
    link = paths["repo_root"] / "raw-link"
    try:
        link.symlink_to(paths["raw_root"], target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit creating test symlinks")
    with pytest.raises(ValueError, match="symlink|reparse"):
        streaming.safe_subject_directory(link, paths["metadata_root"], "sub-001")


def test_nested_symlink_payload_path_is_refused(staged):
    frame, paths, _ = staged
    original = paths["raw_root"] / "sub-001/eeg"
    moved = paths["raw_root"] / "sub-001/materialized"
    original.rename(moved)
    try:
        original.symlink_to(moved, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit creating test symlinks")
    with pytest.raises(ValueError, match="symlink|reparse"):
        streaming.stream_subjects(one(frame), **paths)


def test_incomplete_final_journal_write_recovers_but_interior_corruption_fails(tmp_path):
    journal = tmp_path / "status.jsonl"
    journal.write_bytes(b'{"status":"complete"}\n{"status":')
    assert streaming._recover_journal(journal) == [{"status": "complete"}]
    streaming.append_status(journal, "test", "sub-001", "planned")
    assert len(streaming._recover_journal(journal)) == 2
    journal.write_bytes(b'{bad complete record}\n')
    with pytest.raises(ValueError, match="corrupt complete journal"):
        streaming._recover_journal(journal)


def test_journal_write_failure_prevents_cleanup(staged, monkeypatch):
    frame, paths, _ = staged
    append = streaming.append_status

    def fail_complete(journal, run_id, subject_id, status, **kwargs):
        if status == "complete":
            raise OSError("synthetic durable-journal failure")
        return append(journal, run_id, subject_id, status, **kwargs)

    monkeypatch.setattr(streaming, "append_status", fail_complete)
    result = streaming.stream_subjects(one(frame), **paths, execute=True, local_existing_only=True,
                                      delete_raw_after_success=True)[0]
    assert result["status"] == "preprocessing_failed"
    assert (paths["raw_root"] / "sub-001").is_dir()


def test_cli_defaults_and_mutually_exclusive_flags():
    from scripts.data.s2_07_stream_preprocess_subjects import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert not args.execute and not args.delete_raw_after_success
    assert args.manifest.name == "full_candidate_download_manifest.csv"
    with pytest.raises(SystemExit):
        parser.parse_args(["--execute", "--dry-run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--keep-raw", "--delete-raw-after-success"])


def test_cli_filters_subjects_and_propagates_failure(staged, capsys):
    from scripts.data.s2_07_stream_preprocess_subjects import main

    frame, paths, calls = staged
    manifest = paths["repo_root"] / "manifest.csv"
    frame.to_csv(manifest, index=False)
    args = ["--manifest", str(manifest), "--destination", str(paths["raw_root"]),
            "--metadata-root", str(paths["metadata_root"]), "--output-root", str(paths["output_root"]),
            "--metrics-root", str(paths["metrics_root"]), "--config", str(paths["config_path"]),
            "--subject-id", "sub-002"]
    assert main(args) == 0
    captured = capsys.readouterr().out
    assert '"subject_id": "sub-002"' in captured
    assert '"subject_id": "sub-001"' not in captured
    assert calls == []
    (paths["raw_root"] / str(frame.iloc[2]["relative_path"])).unlink()
    assert main(args + ["--execute", "--local-existing-only"]) == 1
    assert calls == []
