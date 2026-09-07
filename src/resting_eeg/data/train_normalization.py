"""One global scaler from unique usable samples of locked, eligible Real Train.

No raw loader or held-out EEG reader is used. HDF5 files remain read-only.
Whole-container hashes are opaque integrity reads, not held-out sample access.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np

from resting_eeg.common import canonical_hash, sha256_file
from resting_eeg.data import final_cohort_split as frozen
from resting_eeg.data.paths import load_paths

STARTING_HEAD = "9fa49dba1fda8fd0e37852ccdce642372aecca70"
STATUS = "TRAIN_NORMALIZATION_LOCKED"
VERSION = "scenario2-train-normalization-v1"
ALGORITHM = "float64-parallel-centered-moments-interval-union-v1"
CHANNELS = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8",
            "C3", "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2"]
SFREQ, WINDOW, STRIDE, CHUNK = 128, 512, 256, 4096
MIN_STD_UV = 1e-6
METRICS = Path("outputs/metrics/scenario2")
SCALER_FILE = "train_normalization.json"
BLOCK_FILE = "train_normalization_block_manifest.csv"
LOCK_FILE = "train_normalization_lock.json"
FINAL_HASHES = {
    "final_cohort_manifest.csv": "e9405aacd8109700ed55ef3df49b8b5ec0187a65b1c3d79049597fa27b490bc9",
    "final_block_split_manifest.csv": "a58605010f7371fe1805a7834c205f2da7be816515779e15f2ba082bfd201f8b",
    "final_cohort_split_summary.json": "fe90564149a5a7740d66952134473ec45122dd1c5acf51f7d7b5e6a0c0551adb",
    "final_cohort_split_lock.json": "c591a46aafaa11cda0cb5b8fd3f39d1a5df61ac3fa8973fae96f804857edfe0f",
}
IMPLEMENTATION_FILES = [
    "src/resting_eeg/data/train_normalization.py",
    "src/resting_eeg/data/train_normalization_verify.py",
    "scripts/data/s2_18_fit_train_normalization.py",
    "scripts/data/s2_19_verify_train_normalization.py",
]
BLOCK_COLUMNS = [
    "subject_id", "eye_state", "block_id", "block_index", "split", "block_eligible",
    "candidate_window_count", "usable_window_count", "fitting_window_count",
    "unique_sample_count_used", "unique_duration_seconds_used", "usable_mask_sha256",
    "window_starts_sha256", "sample_intervals", "hdf5_path", "hdf5_sha256",
    "receipt_sha256", "exclusion_reason",
]
require = frozen.require


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_assignments(rows, included, expected_keys):
    """Reject unexpected, duplicate, held-out, excluded, or missing inputs."""
    seen = set()
    for row in rows:
        require(row["subject_id"] in included, "Excluded subject cannot contribute")
        require(row["split"] == "Train", "Validation/Test block rejected from fit")
        key = (row["subject_id"], int(row["block_index"]))
        require(key in expected_keys, "Unexpected Train assignment")
        require(key not in seen, "Duplicate block accounting")
        require(row["block_uid"] == f"{key[0]}_block-{key[1]:02d}", "Block identity mismatch")
        seen.add(key)
    require(seen == expected_keys, "Missing Train assignment")


def load_locked_inputs(repo_root):
    """Pin the accepted cohort/split and preprocessing before EEG access."""
    root = Path(repo_root).resolve()
    sources, _, _ = frozen.validate_source_hashes(root)
    for name, digest in FINAL_HASHES.items():
        frozen.validate_file_hash(root / METRICS / name, digest, "final cohort/split")
    paths = load_paths(root)
    acceptance_path = paths["runtime_root"] / "docs/final_cohort_split_lock_20260907/final_acceptance.json"
    accepted = frozen.read_json(acceptance_path)
    require(accepted.get("status") == frozen.FINAL_STATUS, "Final cohort is not accepted")
    require(accepted.get("final_committed_head") == STARTING_HEAD, "Accepted starting HEAD mismatch")
    require(accepted.get("worktree_clean") is True and accepted.get("final_test_sealed") is True,
            "Accepted lock is not clean/sealed")
    require(accepted.get("normalization_unfitted") is True, "Conflicting accepted normalization state")
    require({Path(a["path"]).name: a["sha256"] for a in accepted["artifact_files"]} == FINAL_HASHES,
            "Accepted artifact hash bindings changed")
    cohort = read_csv(root / METRICS / "final_cohort_manifest.csv")
    included = {r["subject_id"] for r in cohort if frozen.truth(r["final_cohort_included"])}
    require(len(cohort) == len({r["subject_id"] for r in cohort}) == 215 and len(included) == 198,
            "Final subject accounting mismatch")
    all_rows = read_csv(root / METRICS / "final_block_split_manifest.csv")
    require(Counter(r["split"] for r in all_rows) == {"Train": 1980, "Validation": 396, "Test": 792},
            "Final assigned block counts changed")
    require(all(r["subject_id"] in included for r in all_rows), "Excluded final assignment")
    require(all(frozen.truth(r["block_eligible"]) for r in all_rows if r["split"] != "Train"),
            "Frozen held-out eligibility changed")
    train = sorted((r for r in all_rows if r["split"] == "Train"),
                   key=lambda r: (r["subject_id"], int(r["block_index"])))
    keys = {(r["subject_id"], int(r["block_index"])) for r in train}
    validate_assignments(train, included, keys)
    require(Counter((r["subject_id"], r["eye_state"]) for r in train)
            == {(s, state): 5 for s in included for state in ("EO", "EC")}, "Train 5/5 assignment changed")
    require({r["block_uid"] for r in train if not frozen.truth(r["block_eligible"])}
            == frozen.EXPECTED_INCLUDED_INELIGIBLE_TRAIN, "Frozen ineligible Train blocks changed")
    completed = read_csv(root / METRICS / "batches/full_cohort_streaming_subject_manifest.csv")
    require(len(completed) == len({r["subject_id"] for r in completed}) == 211, "Processed subject accounting mismatch")
    final_lock = frozen.read_json(root / METRICS / "final_cohort_split_lock.json")
    bundle = final_lock["processed_evidence_bindings"]
    require(canonical_hash({r["subject_id"]: r["output_sha256"] for r in completed})
            == bundle["all_block_output_hash_bundle_sha256"], "HDF5 bundle mismatch")
    require(canonical_hash({r["subject_id"]: r["receipt_sha256"] for r in completed})
            == bundle["receipt_hash_bundle_sha256"], "Receipt bundle mismatch")
    return dict(root=root, paths=paths, included=included, train=train, keys=keys,
                completed={r["subject_id"]: r for r in completed},
                source_hashes={k: v["sha256"] for k, v in sources.items()},
                preprocessing_sha256=sources["preprocessing_config"]["semantic_sha256"],
                accepted_record_sha256=sha256_file(acceptance_path), accepted_bundle=bundle)


def validate_hdf5_header(handle, subject, preprocessing_sha256):
    require(handle.attrs.get("subject_id") == subject, "HDF5 subject mismatch")
    require(handle.attrs.get("scope") == "all_blocks_mechanical_provisional_v1", "HDF5 scope mismatch")
    require(not frozen.truth(handle.attrs.get("normalization_fitted")), "HDF5 already normalized")
    require(handle.attrs.get("preprocessing_config_sha256") == preprocessing_sha256, "Preprocessing hash mismatch")
    require(json.loads(handle.attrs["target_channels"]) == CHANNELS, "Wrong channel order")
    require(json.loads(handle.attrs["train_parameters"])["fit_parameters_from"] == "Train_only",
            "Missing Train-only provenance")


def validate_block_metadata(group, row):
    """Validate frozen metadata; do not read EEG here or recalculate QC."""
    for attr, value in [("subject_id", row["subject_id"]), ("block_uid", row["block_uid"]),
                        ("provisional_split", "Train"), ("state", row["eye_state"]),
                        ("block_index", int(row["block_index"])), ("physical_unit", "uV"),
                        ("processed_sfreq", SFREQ)]:
        require(group.attrs.get(attr) == value, f"HDF5 {attr} mismatch: {row['block_uid']}")
    require(json.loads(group.attrs["channel_names"]) == CHANNELS, "Wrong channel order")
    require(frozen.truth(group.attrs.get("block_eligible")) == frozen.truth(row["block_eligible"]),
            "Frozen block eligibility changed")
    data = group["eeg_uv"]
    require(data.ndim == 2 and data.shape[0] == 19 and data.dtype == np.float32, "EEG schema mismatch")
    starts, mask = group["window_start_samples"][:], group["window_usable_mask"][:]
    require(starts.dtype.kind in "iu" and mask.dtype.kind == "b", "Window metadata dtype mismatch")
    require(np.array_equal(starts, np.arange(0, data.shape[1] - WINDOW + 1, STRIDE)), "Window geometry mismatch")
    require(starts.shape == mask.shape == (int(row["candidate_windows"]),), "Candidate window count mismatch")
    require(int(mask.sum()) == int(row["usable_windows"]), "Usable window count mismatch")
    require(canonical_hash(mask.tolist()) == row["window_usable_mask_sha256"], "Usable mask hash mismatch")
    return starts, mask


def iter_train_blocks(inputs):
    """Open one verified subject at a time; yield only final Train groups.

    The consumer must skip EEG indexing when block_eligible is false. Hashes
    are checked both before and after each subject to detect source mutation.
    """
    validate_assignments(inputs["train"], inputs["included"], inputs["keys"])
    for subject, rows in groupby(inputs["train"], key=lambda r: r["subject_id"]):
        source = inputs["completed"][subject]
        path = inputs["paths"]["processed_root"] / "all-blocks-v1" / f"{subject}_all_blocks.h5"
        receipt_path = path.with_suffix(".provenance.json")
        require(path.is_file() and receipt_path.is_file(), f"Missing HDF5/provenance: {subject}")
        frozen.validate_file_hash(path, source["output_sha256"], "HDF5")
        frozen.validate_file_hash(receipt_path, source["receipt_sha256"], "receipt")
        receipt = frozen.read_json(receipt_path)
        require(receipt.get("subject_id") == subject and receipt.get("output_sha256") == source["output_sha256"],
                "Receipt/HDF5 identity mismatch")
        require(receipt.get("fit_parameters_from") == "Train_only"
                and receipt.get("heldout_parameters_are") == "apply_only", "Missing Train-only provenance")
        with h5py.File(path, "r") as handle:
            validate_hdf5_header(handle, subject, inputs["preprocessing_sha256"])
            for row in rows:
                require(row["source_output_sha256"] == source["output_sha256"], "Assignment source hash mismatch")
                group = handle["blocks"][row["block_uid"]]
                starts, mask = validate_block_metadata(group, row)
                yield row, group, starts, mask, source
        frozen.validate_file_hash(path, source["output_sha256"], "HDF5 after read")
        frozen.validate_file_hash(receipt_path, source["receipt_sha256"], "receipt after read")


def union_intervals(starts, mask, sample_count):
    """Sorted half-open union; called separately for each recording block."""
    starts, mask = np.asarray(starts), np.asarray(mask)
    require(starts.ndim == 1 and starts.dtype.kind in "iu" and starts.shape == mask.shape,
            "Invalid window intervals")
    require(mask.dtype.kind == "b", "Usability must be Boolean")
    require(np.all(starts >= 0) and np.all(starts + WINDOW <= sample_count), "Window outside block")
    merged = []
    for start in sorted(int(s) for s in starts[mask]):
        stop = start + WINDOW
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def fitting_intervals(row, starts, mask, sample_count):
    require(row["split"] == "Train", "Validation/Test block rejected from fit")
    if not frozen.truth(row["block_eligible"]):
        return []
    intervals = union_intervals(starts, mask, sample_count)
    require(bool(intervals), "Eligible Train block has no fitting samples")
    return intervals


class StreamingMoments:
    """Float64 centered chunk moments combined by the parallel variance rule."""

    def __init__(self):
        self.count = 0
        self.mean = np.zeros(19, dtype=np.float64)
        self.m2 = np.zeros(19, dtype=np.float64)

    def add(self, values):
        x = np.asarray(values, dtype=np.float64)
        require(x.ndim == 2 and x.shape[0] == 19 and x.shape[1] > 0, "Invalid EEG chunk")
        require(np.isfinite(x).all(), "Nonfinite EEG sample")
        n = x.shape[1]
        mean = x.mean(axis=1, dtype=np.float64)
        m2 = np.sum((x - mean[:, None]) ** 2, axis=1, dtype=np.float64)
        total = self.count + n
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total
        require(np.isfinite(self.mean).all() and np.isfinite(self.m2).all(), "Nonfinite accumulated statistics")

    def scaler(self):
        require(self.count > 0, "No fitting samples")
        std = np.sqrt(self.m2 / self.count)  # Population variance, ddof=0.
        require(np.isfinite(std).all() and np.all(std >= MIN_STD_UV), "Channel std < 1e-6 uV or nonfinite")
        return self.mean.copy(), std


def block_record(row, starts, intervals, source):
    count = sum(stop - start for start, stop in intervals)
    eligible = frozen.truth(row["block_eligible"])
    return dict(subject_id=row["subject_id"], eye_state=row["eye_state"], block_id=row["block_uid"],
                block_index=int(row["block_index"]), split="Train", block_eligible=eligible,
                candidate_window_count=int(row["candidate_windows"]), usable_window_count=int(row["usable_windows"]),
                fitting_window_count=int(row["usable_windows"]) if eligible else 0,
                unique_sample_count_used=count, unique_duration_seconds_used=count / SFREQ,
                usable_mask_sha256=row["window_usable_mask_sha256"], window_starts_sha256=canonical_hash(starts.tolist()),
                sample_intervals=json.dumps(intervals, separators=(",", ":")),
                hdf5_path=f"processed/all-blocks-v1/{row['subject_id']}_all_blocks.h5",
                hdf5_sha256=source["output_sha256"], receipt_sha256=source["receipt_sha256"],
                exclusion_reason="" if eligible else row["eligibility_reason"])


def volume_summary(records):
    """Descriptive bookkeeping only; mask-usable and fitting windows differ."""
    def tally(rows):
        candidate = sum(r["candidate_window_count"] for r in rows)
        usable = sum(r["usable_window_count"] for r in rows)
        count = sum(r["unique_sample_count_used"] for r in rows)
        eligible = sum(r["block_eligible"] for r in rows)
        return dict(assigned_train_block_count=len(rows), eligible_train_block_count=eligible,
                    ineligible_train_block_count=len(rows) - eligible, candidate_train_window_count=candidate,
                    usable_train_window_count=usable, fitting_train_window_count=sum(r["fitting_window_count"] for r in rows),
                    usable_window_fraction=usable / candidate if candidate else 0.0,
                    unique_sample_count=count, unique_duration_seconds=count / SFREQ,
                    unique_duration_hours=count / SFREQ / 3600)
    per_subject = defaultdict(int)
    for row in records:
        per_subject[row["subject_id"]] += row["unique_sample_count_used"]
    seconds = np.array(list(per_subject.values()), dtype=np.float64) / SFREQ
    result = tally(records)
    result.update(included_subject_count=len(per_subject),
                  by_eye_state={state: tally([r for r in records if r["eye_state"] == state]) for state in ("EO", "EC")},
                  per_subject_duration_seconds={"min": float(seconds.min()), "median": float(np.median(seconds)),
                                                "mean": float(seconds.mean()), "max": float(seconds.max())},
                  per_subject_unique_sample_counts=dict(sorted(per_subject.items())))
    return result


def fit_train(inputs, progress=None):
    moments, records = StreamingMoments(), []
    for row, group, starts, mask, source in iter_train_blocks(inputs):
        intervals = fitting_intervals(row, starts, mask, group["eeg_uv"].shape[1])
        # Only merged eligible Train intervals are indexed. Ineligible = zero reads.
        for start, stop in intervals:
            for first in range(start, stop, CHUNK):
                moments.add(group["eeg_uv"][:, first:min(first + CHUNK, stop)])
        records.append(block_record(row, starts, intervals, source))
        if progress and len(records) % 200 == 0:
            progress(f"Fitted {len(records)}/{len(inputs['train'])} assigned Train blocks")
    mean, std = moments.scaler()
    volume = volume_summary(records)
    require(volume["unique_sample_count"] == moments.count, "Duplicate/missing sample accounting")
    return mean, std, records, volume


def build_payloads(inputs, mean, std, records, volume):
    manifest = frozen.csv_bytes(records, BLOCK_COLUMNS)
    input_bundle = {s: {"hdf5_sha256": inputs["completed"][s]["output_sha256"],
                        "receipt_sha256": inputs["completed"][s]["receipt_sha256"]} for s in sorted(inputs["included"])}
    implementation = {p: sha256_file(inputs["root"] / p) for p in IMPLEMENTATION_FILES}
    scaler = dict(schema_version=VERSION, status=STATUS, algorithm=ALGORITHM,
                  channel_order=CHANNELS, means_uv=mean.tolist(), stds_uv=std.tolist(), ddof=0,
                  scaler_count=1, fit_split="Train", fit_population="eligible_final_included_real_train_unique_usable_samples",
                  shared_across=["subjects", "EO", "EC", "blocks", "windows"],
                  apply_unchanged_to=["Real Train", "Real Validation", "Real Test", "Synthetic EEG"],
                  sampling_rate_hz=SFREQ, window_length_samples=WINDOW, window_stride_samples=STRIDE,
                  chunk_samples=CHUNK, accumulation_dtype="float64", minimum_std_uv=MIN_STD_UV,
                  source_hashes=inputs["source_hashes"], final_input_hashes=FINAL_HASHES,
                  final_cohort_split_lock_sha256=FINAL_HASHES["final_cohort_split_lock.json"],
                  preprocessing_semantic_sha256=inputs["preprocessing_sha256"],
                  block_manifest_sha256=frozen.byte_sha256(manifest),
                  input_hdf5_receipt_bundle_sha256=canonical_hash(input_bundle),
                  window_accounting_definition="usable includes all assigned masks; fitting excludes ineligible blocks",
                  **volume)
    scaler_bytes = frozen.json_bytes(scaler)
    lock = dict(schema_version=VERSION, status=STATUS, algorithm=ALGORITHM, starting_head=STARTING_HEAD,
                final_input_hashes=FINAL_HASHES, source_hashes=inputs["source_hashes"],
                accepted_final_cohort_record_sha256=inputs["accepted_record_sha256"],
                preprocessing_semantic_sha256=inputs["preprocessing_sha256"],
                accepted_all_subject_hdf5_receipt_bundle=inputs["accepted_bundle"],
                input_hdf5_receipt_bundle=input_bundle, input_hdf5_receipt_bundle_sha256=canonical_hash(input_bundle),
                artifact_sha256={SCALER_FILE: frozen.byte_sha256(scaler_bytes), BLOCK_FILE: frozen.byte_sha256(manifest)},
                implementation_sha256=implementation, final_test_sealed=True,
                validation_sample_count_used=0, test_sample_count_used=0, excluded_subject_sample_count_used=0,
                ineligible_block_sample_count_used=0, deterministic_serialization="UTF-8 LF sorted JSON fixed CSV; no timestamps")
    return {SCALER_FILE: scaler_bytes, BLOCK_FILE: manifest, LOCK_FILE: frozen.json_bytes(lock)}


def write_payloads(output_root, payloads):
    """Refuse conflicting existing artifacts; identical reruns are no-ops."""
    root = Path(output_root)
    for name, payload in payloads.items():
        if (root / name).exists():
            require((root / name).read_bytes() == payload, f"Conflicting normalization artifact: {name}")
    for name, payload in payloads.items():
        if not (root / name).exists():
            frozen.atomic_write(root / name, payload)
