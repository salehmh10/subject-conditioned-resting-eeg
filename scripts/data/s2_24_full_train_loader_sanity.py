from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Reuse the already-verified deterministic index builder from s2_23.
from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    WINDOW_LENGTH,
    WindowRecord,
    build_train_window_index,
)


# ============================================================
# Configuration
# ============================================================

EXPECTED_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

BATCH_SIZE = 128

# Keep only a few HDF5 subject files open at once.
MAX_OPEN_H5_FILES = 8


# ============================================================
# Cached Dataset
# ============================================================

class CachedRealTrainWindowDataset(Dataset):
    """
    Real Train EEG dataset with a small LRU HDF5 cache.

    Scientific behavior is unchanged:
    - final Train only
    - eligible blocks only
    - usable windows only
    - one frozen global scaler
    """

    def __init__(
        self,
        records: list[WindowRecord],
        scaler: dict,
        max_open_files: int = MAX_OPEN_H5_FILES,
    ) -> None:

        self.records = records

        self.means = np.asarray(
            scaler["means_uv"],
            dtype=np.float64,
        )

        self.stds = np.asarray(
            scaler["stds_uv"],
            dtype=np.float64,
        )

        self.window_length = int(
            scaler["window_length_samples"]
        )

        self.max_open_files = int(
            max_open_files
        )

        # Path -> open h5py.File
        self._cache: OrderedDict[
            Path,
            h5py.File,
        ] = OrderedDict()

        if self.means.shape != (19,):
            raise RuntimeError(
                f"Unexpected mean shape: {self.means.shape}"
            )

        if self.stds.shape != (19,):
            raise RuntimeError(
                f"Unexpected std shape: {self.stds.shape}"
            )

        if self.window_length != WINDOW_LENGTH:
            raise RuntimeError(
                "Unexpected window length."
            )

    def __len__(self) -> int:
        return len(self.records)

    # --------------------------------------------------------
    # Small LRU cache
    # --------------------------------------------------------

    def _get_h5(
        self,
        path: Path,
    ) -> h5py.File:

        path = Path(path)

        if path in self._cache:

            h5 = self._cache.pop(path)

            # Move to most-recently-used end.
            self._cache[path] = h5

            return h5

        h5 = h5py.File(
            path,
            "r",
        )

        self._cache[path] = h5

        # Remove least-recently-used file.
        while len(self._cache) > self.max_open_files:

            _, old_h5 = self._cache.popitem(
                last=False
            )

            old_h5.close()

        return h5

    def close(self) -> None:

        for h5 in self._cache.values():
            try:
                h5.close()
            except Exception:
                pass

        self._cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # Dataset access
    # --------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ):

        record = self.records[index]

        h5 = self._get_h5(
            record.h5_path
        )

        eeg = h5[
            f"blocks/{record.block_uid}/eeg_uv"
        ]

        start = record.start_sample
        stop = start + self.window_length

        window_uv = np.asarray(
            eeg[:, start:stop],
            dtype=np.float32,
        )

        if window_uv.shape != (19, 512):
            raise RuntimeError(
                f"Unexpected EEG shape: "
                f"{record.block_uid}: "
                f"{window_uv.shape}"
            )

        if not np.isfinite(window_uv).all():
            raise RuntimeError(
                f"Nonfinite EEG detected: "
                f"{record.block_uid}"
            )

        # ----------------------------------------------------
        # ONE frozen global Train-only normalization
        # ----------------------------------------------------

        normalized = (
            (
                window_uv.astype(np.float64)
                - self.means[:, None]
            )
            / self.stds[:, None]
        ).astype(np.float32)

        if not np.isfinite(normalized).all():
            raise RuntimeError(
                f"Nonfinite normalized EEG: "
                f"{record.block_uid}"
            )

        return {
            "x": torch.from_numpy(
                normalized
            ),

            "y": torch.tensor(
                record.label,
                dtype=torch.long,
            ),

            "subject_index": torch.tensor(
                record.subject_index,
                dtype=torch.long,
            ),
        }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — FULL REAL TRAIN LOADER SANITY")
    print("=" * 80)

    # ========================================================
    # Step 1 — Load frozen artifacts
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — LOAD LOCKED TRAIN DEFINITION")
    print("=" * 80)

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError(
            "Normalization is not locked."
        )

    print(
        "normalization status:",
        scaler["status"],
    )

    # ========================================================
    # Step 2 — Rebuild deterministic index
    # ========================================================

    records, subject_to_index = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    if len(records) != EXPECTED_WINDOWS:
        raise RuntimeError(
            f"Expected {EXPECTED_WINDOWS} windows, "
            f"got {len(records)}"
        )

    if len(subject_to_index) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"got {len(subject_to_index)}"
        )

    # --------------------------------------------------------
    # Expected counts directly from deterministic records
    # --------------------------------------------------------

    expected_labels = np.bincount(
        np.asarray(
            [r.label for r in records],
            dtype=np.int64,
        ),
        minlength=2,
    )

    expected_subject_counts = np.bincount(
        np.asarray(
            [
                r.subject_index
                for r in records
            ],
            dtype=np.int64,
        ),
        minlength=EXPECTED_SUBJECTS,
    )

    print("\nEXPECTED DATASET COUNTS")

    print(
        f"EO windows: "
        f"{int(expected_labels[0])}"
    )

    print(
        f"EC windows: "
        f"{int(expected_labels[1])}"
    )

    # ========================================================
    # Step 3 — Build cached Dataset
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — CREATE CACHED TRAIN DATASET")
    print("=" * 80)

    dataset = CachedRealTrainWindowDataset(
        records=records,
        scaler=scaler,
        max_open_files=MAX_OPEN_H5_FILES,
    )

    print(
        f"dataset length: {len(dataset)}"
    )

    print(
        f"HDF5 cache size: "
        f"{MAX_OPEN_H5_FILES} subject files"
    )

    # ========================================================
    # Step 4 — Create deterministic full-pass DataLoader
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — FULL TRAIN PASS")
    print("=" * 80)

    loader = DataLoader(
        dataset,

        batch_size=BATCH_SIZE,

        # Full deterministic audit.
        shuffle=False,

        # Keep this simple on Windows for now.
        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        drop_last=False,
    )

    total_windows = 0

    observed_labels = torch.zeros(
        2,
        dtype=torch.long,
    )

    observed_subject_counts = torch.zeros(
        EXPECTED_SUBJECTS,
        dtype=torch.long,
    )

    total_values = 0

    sum_z = 0.0
    sumsq_z = 0.0

    global_min = float("inf")
    global_max = float("-inf")

    first_batch_x = None
    first_batch_y = None

    start_time = time.perf_counter()

    n_batches = len(loader)

    for batch_index, batch in enumerate(loader):

        x = batch["x"]
        y = batch["y"]

        subject_index = batch[
            "subject_index"
        ]

        # ----------------------------------------------------
        # Structural checks
        # ----------------------------------------------------

        if x.ndim != 3:
            raise RuntimeError(
                f"Expected 3D EEG batch, got {x.shape}"
            )

        if x.shape[1:] != (19, 512):
            raise RuntimeError(
                f"Unexpected EEG batch shape: {x.shape}"
            )

        if x.dtype != torch.float32:
            raise RuntimeError(
                f"Unexpected dtype: {x.dtype}"
            )

        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"Nonfinite value in batch {batch_index}"
            )

        batch_size = int(
            x.shape[0]
        )

        total_windows += batch_size

        # ----------------------------------------------------
        # Label accounting
        # ----------------------------------------------------

        observed_labels += torch.bincount(
            y,
            minlength=2,
        ).cpu()

        observed_subject_counts += torch.bincount(
            subject_index,
            minlength=EXPECTED_SUBJECTS,
        ).cpu()

        # ----------------------------------------------------
        # Window-weighted diagnostic statistics
        #
        # These are NOT the normalization fitting statistics,
        # because overlapping windows appear repeatedly here.
        # ----------------------------------------------------

        total_values += int(
            x.numel()
        )

        sum_z += float(
            x.sum(
                dtype=torch.float64
            ).item()
        )

        # float32 square, float64 reduction.
        sumsq_z += float(
            (x * x).sum(
                dtype=torch.float64
            ).item()
        )

        global_min = min(
            global_min,
            float(x.min().item()),
        )

        global_max = max(
            global_max,
            float(x.max().item()),
        )

        if first_batch_x is None:

            first_batch_x = x.clone()

            first_batch_y = y.clone()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index == 0
            or (batch_index + 1) % 50 == 0
            or batch_index + 1 == n_batches
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            rate = (
                total_windows / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"batch "
                f"{batch_index + 1:>4}/{n_batches}"
                f" | windows={total_windows:>6}"
                f" | {rate:>8.1f} windows/s"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    dataset.close()

    # ========================================================
    # Step 5 — Verify full-pass accounting
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — VERIFY FULL TRAIN ACCOUNTING")
    print("=" * 80)

    if total_windows != EXPECTED_WINDOWS:
        raise RuntimeError(
            f"Observed {total_windows} windows; "
            f"expected {EXPECTED_WINDOWS}"
        )

    expected_labels_t = torch.tensor(
        expected_labels,
        dtype=torch.long,
    )

    if not torch.equal(
        observed_labels,
        expected_labels_t,
    ):
        raise RuntimeError(
            "Observed class counts do not "
            "match deterministic index."
        )

    expected_subject_counts_t = torch.tensor(
        expected_subject_counts,
        dtype=torch.long,
    )

    if not torch.equal(
        observed_subject_counts,
        expected_subject_counts_t,
    ):
        raise RuntimeError(
            "Observed subject counts do not "
            "match deterministic index."
        )

    print("window count: PASS")
    print("EO/EC accounting: PASS")
    print("subject accounting: PASS")
    print("finite EEG check: PASS")

    print(
        f"\nEO windows: "
        f"{int(observed_labels[0])}"
    )

    print(
        f"EC windows: "
        f"{int(observed_labels[1])}"
    )

    print(
        f"total windows: "
        f"{total_windows}"
    )

    # ========================================================
    # Step 6 — Dataset descriptive accounting
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — TRAIN DATASET DESCRIPTION")
    print("=" * 80)

    subject_counts_np = (
        observed_subject_counts.numpy()
    )

    print(
        "windows per subject:"
    )

    print(
        f"  min    = "
        f"{int(subject_counts_np.min())}"
    )

    print(
        f"  median = "
        f"{float(np.median(subject_counts_np)):.1f}"
    )

    print(
        f"  mean   = "
        f"{float(subject_counts_np.mean()):.2f}"
    )

    print(
        f"  max    = "
        f"{int(subject_counts_np.max())}"
    )

    eo_fraction = (
        int(observed_labels[0])
        / total_windows
    )

    ec_fraction = (
        int(observed_labels[1])
        / total_windows
    )

    print(
        f"\nEO fraction: "
        f"{eo_fraction:.6f}"
    )

    print(
        f"EC fraction: "
        f"{ec_fraction:.6f}"
    )

    # --------------------------------------------------------
    # Window-weighted z statistics
    # --------------------------------------------------------

    window_weighted_mean = (
        sum_z / total_values
    )

    window_weighted_var = (
        sumsq_z / total_values
        - window_weighted_mean ** 2
    )

    window_weighted_std = (
        max(window_weighted_var, 0.0)
        ** 0.5
    )

    print(
        "\nWINDOW-WEIGHTED NORMALIZED DIAGNOSTIC"
    )

    print(
        f"mean z: "
        f"{window_weighted_mean:.8f}"
    )

    print(
        f"std z:  "
        f"{window_weighted_std:.8f}"
    )

    print(
        f"minimum z: "
        f"{global_min:.6f}"
    )

    print(
        f"maximum z: "
        f"{global_max:.6f}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "These statistics weight overlapping windows "
        "multiple times."
    )

    print(
        "They are a loader sanity diagnostic, "
        "NOT the locked normalization statistics."
    )

    # ========================================================
    # Step 7 — Throughput
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — LOADER THROUGHPUT")
    print("=" * 80)

    windows_per_second = (
        total_windows / elapsed
    )

    mb_read_equivalent = (
        total_windows
        * 19
        * 512
        * 4
        / 1024**2
    )

    print(
        f"elapsed seconds: "
        f"{elapsed:.2f}"
    )

    print(
        f"windows / second: "
        f"{windows_per_second:.2f}"
    )

    print(
        f"normalized tensor volume: "
        f"{mb_read_equivalent:.2f} MiB"
    )

    # ========================================================
    # Step 8 — GPU batch confirmation
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 8 — GPU CONFIRMATION")
    print("=" * 80)

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        device = torch.device("cuda")

        x_gpu = first_batch_x.to(
            device,
            non_blocking=True,
        )

        y_gpu = first_batch_y.to(
            device,
            non_blocking=True,
        )

        energy = (
            x_gpu.square().mean()
        )

        torch.cuda.synchronize()

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"batch shape: "
            f"{tuple(x_gpu.shape)}"
        )

        print(
            f"batch energy: "
            f"{energy.item():.6f}"
        )

        print(
            f"peak allocated MB: "
            f"{torch.cuda.max_memory_allocated() / 1024**2:.2f}"
        )

        print(
            "FULL TRAIN LOADER GPU PASS"
        )

    else:

        print(
            "CUDA unavailable."
        )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("FULL REAL TRAIN LOADER SANITY PASS")
    print("=" * 80)

    print(
        f"subjects: {EXPECTED_SUBJECTS}"
    )

    print(
        f"windows: {total_windows}"
    )

    print(
        f"EO: {int(observed_labels[0])}"
    )

    print(
        f"EC: {int(observed_labels[1])}"
    )

    print(
        f"throughput: "
        f"{windows_per_second:.2f} windows/s"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Test EEG accessed: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "REAL-DATA SCIENTIFIC BASELINE"
    )


if __name__ == "__main__":
    main()