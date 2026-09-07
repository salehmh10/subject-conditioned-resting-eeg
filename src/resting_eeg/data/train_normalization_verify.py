"""Independent Boolean coverage and compensated second-pass Train statistics.

Input identity/schema validation is shared with fitting. Coverage and numerical
verification use a different algorithm; a separate refit tests exact determinism.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from resting_eeg.common import sha256_file
from resting_eeg.data import final_cohort_split as frozen
from resting_eeg.data import train_normalization as norm

NORMALIZED_TOLERANCE = 1e-12


def sample_coverage(starts, mask, sample_count, eligible):
    """Independent union via sample bitmap, bounded to one processed block."""
    covered = np.zeros(sample_count, dtype=bool)
    if eligible:
        for index, start in enumerate(starts):
            if mask[index]:
                covered[int(start):int(start) + norm.WINDOW] = True
    return covered


def bitmap_intervals(covered):
    edges = np.diff(np.r_[False, covered, False].astype(np.int8))
    return np.column_stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))).tolist()


class CompensatedSum:
    """Kahan sum of channelwise chunk reductions, independent of fit moments."""

    def __init__(self):
        self.total = np.zeros(19, dtype=np.float64)
        self.correction = np.zeros(19, dtype=np.float64)

    def add(self, vector):
        adjusted = vector - self.correction
        total = self.total + adjusted
        self.correction = (total - self.total) - adjusted
        self.total = total


def independent_second_pass(inputs, scaler, manifest_bytes, progress=None):
    mean, std = np.array(scaler["means_uv"]), np.array(scaler["stds_uv"])
    norm.require(scaler["channel_order"] == norm.CHANNELS and mean.shape == std.shape == (19,), "Wrong channel order/scaler shape")
    norm.require(np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std >= norm.MIN_STD_UV), "Invalid scaler std/mean")
    norm.require(scaler["ddof"] == 0 and scaler["scaler_count"] == 1 and scaler["fit_split"] == "Train", "Scaler is not single global Train population fit")
    sums, z_sums, z_squares = CompensatedSum(), CompensatedSum(), CompensatedSum()
    count, records, read_blocks = 0, [], []
    for row, group, starts, mask, source in norm.iter_train_blocks(inputs):
        eligible = frozen.truth(row["block_eligible"])
        covered = sample_coverage(starts, mask, group["eeg_uv"].shape[1], eligible)
        intervals = bitmap_intervals(covered)
        records.append(norm.block_record(row, starts, intervals, source))
        if not eligible:
            norm.require(not covered.any(), "Ineligible Train block contributed")
            continue
        read_blocks.append(row["block_uid"])
        # Read only covered samples (never a disallowed gap), in a distinct
        # 2,048-sample chunk partition from the fitting pass.
        for first in range(0, len(covered), 2048):
            indices = np.flatnonzero(covered[first:first + 2048]) + first
            if not len(indices):
                continue
            x = np.asarray(group["eeg_uv"][:, indices], dtype=np.float64)
            norm.require(np.isfinite(x).all(), "Nonfinite EEG sample in independent pass")
            z = (x - mean[:, None]) / std[:, None]
            sums.add(np.sum(x, axis=1, dtype=np.float64))
            z_sums.add(np.sum(z, axis=1, dtype=np.float64))
            z_squares.add(np.sum(z * z, axis=1, dtype=np.float64))
            count += x.shape[1]
        if progress and len(records) % 200 == 0:
            progress(f"Independently checked {len(records)}/{len(inputs['train'])} Train assignments")
    norm.require(frozen.csv_bytes(records, norm.BLOCK_COLUMNS) == manifest_bytes,
                 "Independent Train block/overlap accounting differs from manifest")
    volume = norm.volume_summary(records)
    norm.require(all(scaler[k] == v for k, v in volume.items()), "Volume accounting mismatch")
    norm.require(count == scaler["unique_sample_count"] and len(read_blocks) == scaler["eligible_train_block_count"],
                 "Independent sample/block count mismatch")
    z_mean = z_sums.total / count
    z_std = np.sqrt(z_squares.total / count - z_mean ** 2)
    independent_mean = sums.total / count
    independent_std = std * z_std  # Centered second moment with mean correction.
    max_mean = float(np.max(np.abs(z_mean)))
    max_std = float(np.max(np.abs(z_std - 1)))
    norm.require(np.isfinite(z_mean).all() and np.isfinite(z_std).all(), "Nonfinite second-pass statistics")
    norm.require(max_mean <= NORMALIZED_TOLERANCE and max_std <= NORMALIZED_TOLERANCE,
                 "Independent normalized mean/std tolerance failed")
    norm.require(np.allclose(independent_mean, mean, rtol=0, atol=NORMALIZED_TOLERANCE), "Independent raw mean mismatch")
    return dict(verification="passed", algorithm="boolean-sample-coverage-kahan-z-moments-v1",
                unique_sample_count=count, eeg_blocks_read=len(read_blocks),
                normalized_channel_means=z_mean.tolist(), normalized_channel_population_stds=z_std.tolist(),
                independent_means_uv=independent_mean.tolist(), independent_stds_uv=independent_std.tolist(),
                maximum_absolute_normalized_mean=max_mean,
                maximum_absolute_normalized_std_deviation=max_std, absolute_tolerance=NORMALIZED_TOLERANCE,
                validation_samples_read=0, test_samples_read=0, ineligible_block_samples_read=0,
                excluded_subject_samples_read=0, overlap_deduplication="independent_bitmap_matches_all_intervals")


def verify_locked_artifact_hashes(root):
    root = Path(root)
    lock = frozen.read_json(root / norm.LOCK_FILE)
    norm.require(lock["schema_version"] == norm.VERSION and lock["status"] == norm.STATUS, "Wrong normalization lock schema/status")
    norm.require(set(lock["artifact_sha256"]) == {norm.SCALER_FILE, norm.BLOCK_FILE}, "Missing/unexpected scaler artifacts")
    for name, digest in lock["artifact_sha256"].items():
        frozen.validate_file_hash(root / name, digest, "normalization artifact")
    return lock


def verification_report(output, repository, second_pass, volume, payloads):
    return dict(status=norm.STATUS, verification="passed", repository=repository,
                starting_head=norm.STARTING_HEAD, final_input_hashes=norm.FINAL_HASHES,
                artifact_sha256={name: sha256_file(output / name) for name in payloads},
                second_pass=second_pass, deterministic_regeneration="byte_identical",
                counts_and_duration={k: v for k, v in volume.items() if k != "per_subject_unique_sample_counts"},
                final_test_sealed=True, scientific_heldout_outcomes_accessed=False)
