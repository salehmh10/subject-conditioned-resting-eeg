"""Synthetic-only normalization tests; production EEG is never a fixture."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from resting_eeg.common import canonical_hash, sha256_file
from resting_eeg.data import final_cohort_split as frozen
from resting_eeg.data import train_normalization as norm
from resting_eeg.data import train_normalization_verify as verify
from scripts.data import s2_20_accept_train_normalization as acceptance


@pytest.fixture
def inputs(tmp_path):
    subject = "sub-000001"
    processed = tmp_path / "processed"
    path = processed / "all-blocks-v1" / f"{subject}_all_blocks.h5"
    path.parent.mkdir(parents=True)
    rows = []
    with h5py.File(path, "w") as h:
        h.attrs.update(subject_id=subject, scope="all_blocks_mechanical_provisional_v1",
                       normalization_fitted=False, preprocessing_config_sha256="frozen",
                       target_channels=json.dumps(norm.CHANNELS),
                       train_parameters=json.dumps({"fit_parameters_from": "Train_only"}))
        groups = h.create_group("blocks")
        for block, split, eligible in [(1, "Train", True), (2, "Train", False), (3, "Validation", True), (4, "Test", True)]:
            uid = f"{subject}_block-{block:02d}"
            g = groups.create_group(uid)
            g.attrs.update(subject_id=subject, block_uid=uid, block_index=block,
                           provisional_split=split, state="EO", physical_unit="uV", processed_sfreq=128,
                           channel_names=json.dumps(norm.CHANNELS), block_eligible=eligible)
            x = np.random.default_rng(block).normal(3, 7, (19, 1536)).astype(np.float32)
            if block != 1:
                x[:] = np.nan  # Any forbidden access that reaches statistics fails.
            g.create_dataset("eeg_uv", data=x)
            starts = np.arange(0, 1025, 256)
            mask = np.array([True, True, False, True, False])
            g.create_dataset("window_start_samples", data=starts)
            g.create_dataset("window_usable_mask", data=mask)
            if split == "Train":
                rows.append(dict(subject_id=subject, block_index=block, block_uid=uid, split=split,
                                 eye_state="EO", block_eligible=eligible, candidate_windows=5, usable_windows=3,
                                 window_usable_mask_sha256=canonical_hash(mask.tolist()),
                                 eligibility_reason="frozen_qc_eligible" if eligible else "frozen_ineligible"))
    receipt_path = path.with_suffix(".provenance.json")
    frozen.atomic_write(receipt_path, frozen.json_bytes(dict(subject_id=subject, output_sha256=sha256_file(path),
                        fit_parameters_from="Train_only", heldout_parameters_are="apply_only")))
    source = dict(output_sha256=sha256_file(path), receipt_sha256=sha256_file(receipt_path))
    for row in rows:
        row["source_output_sha256"] = source["output_sha256"]
    for name in norm.IMPLEMENTATION_FILES:
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# synthetic implementation identity\n")
    return dict(root=tmp_path, paths={"processed_root": processed}, included={subject}, train=rows,
                keys={(subject, 1), (subject, 2)}, completed={subject: source}, preprocessing_sha256="frozen",
                source_hashes={}, accepted_record_sha256="accepted", accepted_bundle={})


def source_path(inputs):
    return inputs["paths"]["processed_root"] / "all-blocks-v1/sub-000001_all_blocks.h5"


def rebind_fixture(inputs):
    path = source_path(inputs)
    receipt_path = path.with_suffix(".provenance.json")
    receipt = frozen.read_json(receipt_path)
    receipt["output_sha256"] = sha256_file(path)
    frozen.atomic_write(receipt_path, frozen.json_bytes(receipt))
    source = inputs["completed"]["sub-000001"]
    source.update(output_sha256=sha256_file(path), receipt_sha256=sha256_file(receipt_path))
    for row in inputs["train"]:
        row["source_output_sha256"] = source["output_sha256"]


def test_overlapping_windows_count_once():
    assert norm.union_intervals(np.array([0, 256]), np.ones(2, bool), 768) == [[0, 768]]


def test_disjoint_window_unions():
    assert norm.union_intervals(np.array([0, 1024]), np.ones(2, bool), 1536) == [[0, 512], [1024, 1536]]


def test_partially_overlapping_chains():
    assert norm.union_intervals(np.array([768, 0, 400]), np.ones(3, bool), 1280) == [[0, 1280]]


def test_ineligible_train_contributes_zero(inputs):
    mean, std, rows, volume = norm.fit_train(inputs)
    assert rows[1]["unique_sample_count_used"] == rows[1]["fitting_window_count"] == 0
    assert rows[1]["usable_window_count"] == 3
    assert rows[1]["sample_intervals"] == "[]"
    assert volume["unique_sample_count"] == 1280
    with h5py.File(source_path(inputs)) as h:
        expected = h["blocks/sub-000001_block-01/eeg_uv"][:, :1280].astype(np.float64)
    np.testing.assert_allclose(mean, expected.mean(axis=1), rtol=0, atol=1e-14)
    np.testing.assert_allclose(std, expected.std(axis=1, ddof=0), rtol=0, atol=1e-14)


@pytest.mark.parametrize("split", ["Validation", "Test"])
def test_heldout_block_rejected(inputs, split):
    inputs["train"][0]["split"] = split
    with pytest.raises(ValueError, match="Validation/Test"):
        norm.fit_train(inputs)


def test_excluded_subject_rejected(inputs):
    inputs["included"] = set()
    with pytest.raises(ValueError, match="Excluded subject"):
        norm.fit_train(inputs)


@pytest.mark.parametrize("at_header", [True, False])
def test_wrong_channel_order_fails(inputs, at_header):
    with h5py.File(source_path(inputs), "r+") as h:
        obj = h if at_header else h["blocks/sub-000001_block-01"]
        obj.attrs["target_channels" if at_header else "channel_names"] = json.dumps(norm.CHANNELS[::-1])
    rebind_fixture(inputs)
    with pytest.raises(ValueError, match="Wrong channel order"):
        norm.fit_train(inputs)


@pytest.mark.parametrize("missing", ["hdf5", "receipt"])
def test_missing_hdf5_provenance_fails(inputs, missing):
    path = source_path(inputs)
    (path if missing == "hdf5" else path.with_suffix(".provenance.json")).unlink()
    with pytest.raises(ValueError, match="Missing HDF5/provenance"):
        norm.fit_train(inputs)


def test_source_hash_mismatch_fails(inputs):
    inputs["completed"]["sub-000001"]["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        norm.fit_train(inputs)


def test_nonfinite_eeg_fails(inputs):
    with h5py.File(source_path(inputs), "r+") as h:
        h["blocks/sub-000001_block-01/eeg_uv"][0, 0] = np.inf
    rebind_fixture(inputs)
    with pytest.raises(ValueError, match="Nonfinite EEG sample"):
        norm.fit_train(inputs)


def test_tiny_std_fails():
    moments = norm.StreamingMoments()
    moments.add(np.tile([1e-8, -1e-8], (19, 1)))
    with pytest.raises(ValueError, match="std < 1e-6"):
        moments.scaler()


def test_duplicate_block_accounting_fails(inputs):
    inputs["train"].append(copy.deepcopy(inputs["train"][0]))
    with pytest.raises(ValueError, match="Duplicate block"):
        norm.fit_train(inputs)


def test_missing_train_assignment_fails(inputs):
    inputs["train"].pop()
    with pytest.raises(ValueError, match="Missing Train assignment"):
        norm.fit_train(inputs)


def test_deterministic_rerun_canonical_artifacts(inputs):
    first = norm.build_payloads(inputs, *norm.fit_train(inputs))
    second = norm.build_payloads(inputs, *norm.fit_train(inputs))
    assert first == second
    output = inputs["root"] / "artifacts"
    norm.write_payloads(output, first)
    norm.write_payloads(output, second)
    assert {name: (output / name).read_bytes() for name in first} == first
    verify.verify_locked_artifact_hashes(output)


def test_independent_second_pass_and_forbidden_reads(inputs, monkeypatch):
    original = h5py.Dataset.__getitem__
    reads = []
    def guard(dataset, key):
        if dataset.name.endswith("/eeg_uv"):
            assert dataset.name == "/blocks/sub-000001_block-01/eeg_uv"
            reads.append(dataset.name)
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guard)
    payloads = norm.build_payloads(inputs, *norm.fit_train(inputs))
    result = verify.independent_second_pass(inputs, json.loads(payloads[norm.SCALER_FILE]), payloads[norm.BLOCK_FILE])
    assert reads and result["eeg_blocks_read"] == 1 and result["unique_sample_count"] == 1280
    assert result["maximum_absolute_normalized_std_deviation"] < 1e-12


def test_independent_pass_rejects_double_count(inputs):
    payloads = norm.build_payloads(inputs, *norm.fit_train(inputs))
    scaler = json.loads(payloads[norm.SCALER_FILE])
    scaler["unique_sample_count"] += 256
    with pytest.raises(ValueError, match="Volume accounting"):
        verify.independent_second_pass(inputs, scaler, payloads[norm.BLOCK_FILE])


def test_independent_pass_rejects_wrong_scaler(inputs):
    payloads = norm.build_payloads(inputs, *norm.fit_train(inputs))
    scaler = json.loads(payloads[norm.SCALER_FILE])
    scaler["means_uv"][0] += 0.01
    with pytest.raises(ValueError, match="tolerance failed"):
        verify.independent_second_pass(inputs, scaler, payloads[norm.BLOCK_FILE])


def test_changed_mask_fails(inputs):
    with h5py.File(source_path(inputs), "r+") as h:
        h["blocks/sub-000001_block-01/window_usable_mask"][:] = [False, True, True, True, False]
    rebind_fixture(inputs)
    with pytest.raises(ValueError, match="mask hash mismatch"):
        norm.fit_train(inputs)


def test_conflicting_artifact_cannot_be_overwritten(tmp_path):
    norm.write_payloads(tmp_path, {norm.SCALER_FILE: b"accepted"})
    with pytest.raises(ValueError, match="Conflicting normalization"):
        norm.write_payloads(tmp_path, {norm.SCALER_FILE: b"changed"})
    assert (tmp_path / norm.SCALER_FILE).read_bytes() == b"accepted"


def test_numerical_stability_with_large_offset():
    x = 1e10 + np.tile(np.arange(-512, 512, dtype=np.float64), (19, 1)) / 16
    moments = norm.StreamingMoments()
    for chunk in np.array_split(x, 4, axis=1):
        moments.add(chunk)
    mean, std = moments.scaler()
    np.testing.assert_allclose(mean, x.mean(axis=1), rtol=0, atol=1e-6)
    np.testing.assert_allclose(std, x.std(axis=1), rtol=0, atol=1e-12)


def test_locked_input_loader_integration_with_committed_metadata(tmp_path, monkeypatch):
    # Real committed metadata is portable evidence; external acceptance is mocked.
    # This exercises the existing source-validator tuple API without EEG access.
    root = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "runtime"
    acceptance = runtime / "docs/final_cohort_split_lock_20260907/final_acceptance.json"
    frozen.atomic_write(acceptance, frozen.json_bytes(dict(
        status=frozen.FINAL_STATUS, final_committed_head=norm.STARTING_HEAD,
        worktree_clean=True, final_test_sealed=True, normalization_unfitted=True,
        artifact_files=[{"path": name, "sha256": digest} for name, digest in norm.FINAL_HASHES.items()])))
    monkeypatch.setattr(norm, "load_paths", lambda _: {"runtime_root": runtime, "processed_root": tmp_path / "processed"})
    result = norm.load_locked_inputs(root)
    assert len(result["included"]) == 198 and len(result["train"]) == 1980
    assert len(result["keys"]) == 1980 and len(result["source_hashes"]) == len(frozen.SOURCE_PATHS)


def test_acceptance_rejects_unresolved_review():
    passed = {"status": "passed", "findings": [], "resolutions": [], "unresolved_findings": []}
    reviews = {"data_leakage": copy.deepcopy(passed), "code_test": copy.deepcopy(passed)}
    reviews["code_test"]["unresolved_findings"] = ["leakage"]
    with pytest.raises(ValueError, match="Unresolved reviewer"):
        acceptance.validate_reviews(reviews)


@pytest.mark.parametrize("change", ["wrong_head", "dirty", "clean_not_required"])
def test_acceptance_rejects_stale_or_unclean_report(change):
    report = {"verification": "passed", "final_test_sealed": True, "repository": {
        "current_head": "final", "worktree_clean": True, "worktree_clean_required": True}}
    field, value = {"wrong_head": ("current_head", "old"), "dirty": ("worktree_clean", False),
                    "clean_not_required": ("worktree_clean_required", False)}[change]
    report["repository"][field] = value
    with pytest.raises(ValueError, match="clean committed HEAD"):
        acceptance.validate_postcommit_report(report, "final")
