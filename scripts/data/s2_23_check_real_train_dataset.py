from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Fixed project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    r"D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2"
)

ALL_BLOCKS_ROOT = (
    DATA_ROOT
    / "processed"
    / "all-blocks-v1"
)

FINAL_SPLIT_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "final_block_split_manifest.csv"
)

NORMALIZATION_FILE = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "train_normalization.json"
)


# ============================================================
# Frozen technical conventions
# ============================================================

STATE_TO_LABEL = {
    "EO": 0,
    "EC": 1,
}

WINDOW_LENGTH = 512
EXPECTED_TRAIN_BLOCKS = 1980
EXPECTED_ELIGIBLE_TRAIN_BLOCKS = 1970
EXPECTED_TRAIN_WINDOWS = 47835
EXPECTED_SUBJECTS = 198


# ============================================================
# Helpers
# ============================================================

def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def parse_json_text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        return json.loads(value)

    return value


# ============================================================
# Window record
# ============================================================

@dataclass(frozen=True)
class WindowRecord:
    subject_id: str
    subject_index: int

    block_uid: str
    block_index: int

    eye_state: str
    label: int

    candidate_window_index: int
    start_sample: int

    h5_path: Path


# ============================================================
# Step 1 — Build deterministic Train-window index
# ============================================================

def build_train_window_index(
    split_manifest: pd.DataFrame,
    scaler: dict,
) -> tuple[list[WindowRecord], dict[str, int]]:

    print("\n" + "=" * 80)
    print("STEP 1 — BUILD REAL TRAIN WINDOW INDEX")
    print("=" * 80)

    train = split_manifest[
        split_manifest["split"].astype(str).eq("Train")
    ].copy()

    if len(train) != EXPECTED_TRAIN_BLOCKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_BLOCKS} Train blocks, "
            f"found {len(train)}"
        )

    eligible = train[
        train["block_eligible"].map(as_bool)
    ].copy()

    if len(eligible) != EXPECTED_ELIGIBLE_TRAIN_BLOCKS:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_TRAIN_BLOCKS} "
            f"eligible Train blocks, found {len(eligible)}"
        )

    # Deterministic subject IDs.
    subjects = sorted(
        eligible["subject_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"found {len(subjects)}"
        )

    subject_to_index = {
        subject_id: index
        for index, subject_id in enumerate(subjects)
    }

    print(f"Train blocks assigned: {len(train)}")
    print(f"Train blocks eligible: {len(eligible)}")
    print(f"Subjects: {len(subjects)}")

    print(
        f"subject index range: "
        f"0 ... {len(subjects) - 1}"
    )

    print(
        f"first subject mapping: "
        f"{subjects[0]} -> {subject_to_index[subjects[0]]}"
    )

    print(
        f"last subject mapping: "
        f"{subjects[-1]} -> {subject_to_index[subjects[-1]]}"
    )

    eligible = eligible.sort_values(
        ["subject_id", "block_index"]
    )

    records: list[WindowRecord] = []

    block_counter = 0
    manifest_window_counter = 0

    state_counts = {
        "EO": 0,
        "EC": 0,
    }

    # --------------------------------------------------------
    # Open each subject HDF5 once while constructing the index
    # --------------------------------------------------------

    for subject_id, subject_rows in eligible.groupby(
        "subject_id",
        sort=True,
    ):

        subject_id = str(subject_id)

        h5_path = (
            ALL_BLOCKS_ROOT
            / f"{subject_id}_all_blocks.h5"
        )

        if not h5_path.exists():
            raise FileNotFoundError(h5_path)

        with h5py.File(h5_path, "r") as h5:

            for _, row in subject_rows.sort_values(
                "block_index"
            ).iterrows():

                block_uid = str(row["block_uid"])
                block_index = int(row["block_index"])
                eye_state = str(row["eye_state"])

                if eye_state not in STATE_TO_LABEL:
                    raise RuntimeError(
                        f"Unknown eye state: {eye_state}"
                    )

                group_path = f"blocks/{block_uid}"

                if group_path not in h5:
                    raise RuntimeError(
                        f"Missing HDF5 group: {group_path}"
                    )

                block = h5[group_path]

                # ------------------------------------------------
                # Verify metadata
                # ------------------------------------------------

                h5_subject = as_text(
                    block.attrs["subject_id"]
                )

                h5_block_uid = as_text(
                    block.attrs["block_uid"]
                )

                h5_state = as_text(
                    block.attrs["state"]
                )

                h5_split = as_text(
                    block.attrs["provisional_split"]
                )

                h5_eligible = as_bool(
                    block.attrs["block_eligible"]
                )

                h5_channels = parse_json_text(
                    block.attrs["channel_names"]
                )

                h5_unit = as_text(
                    block.attrs["physical_unit"]
                )

                h5_sfreq = float(
                    block.attrs["processed_sfreq"]
                )

                if h5_subject != subject_id:
                    raise RuntimeError(
                        f"Subject mismatch: {block_uid}"
                    )

                if h5_block_uid != block_uid:
                    raise RuntimeError(
                        f"Block UID mismatch: {block_uid}"
                    )

                if h5_state != eye_state:
                    raise RuntimeError(
                        f"Eye-state mismatch: {block_uid}"
                    )

                if h5_split != "Train":
                    raise RuntimeError(
                        f"Non-Train block entered index: "
                        f"{block_uid}"
                    )

                if not h5_eligible:
                    raise RuntimeError(
                        f"Ineligible Train block entered index: "
                        f"{block_uid}"
                    )

                if h5_channels != scaler["channel_order"]:
                    raise RuntimeError(
                        f"Channel mismatch: {block_uid}"
                    )

                if h5_unit != "uV":
                    raise RuntimeError(
                        f"Wrong unit: {block_uid}"
                    )

                if h5_sfreq != 128.0:
                    raise RuntimeError(
                        f"Wrong sampling rate: {block_uid}"
                    )

                eeg = block["eeg_uv"]

                if eeg.shape[0] != 19:
                    raise RuntimeError(
                        f"Wrong EEG shape: "
                        f"{block_uid}: {eeg.shape}"
                    )

                starts = block[
                    "window_start_samples"
                ][:]

                usable = block[
                    "window_usable_mask"
                ][:]

                if len(starts) != len(usable):
                    raise RuntimeError(
                        f"Window metadata length mismatch: "
                        f"{block_uid}"
                    )

                manifest_usable = int(
                    row["usable_windows"]
                )

                actual_usable = int(
                    usable.sum()
                )

                if actual_usable != manifest_usable:
                    raise RuntimeError(
                        f"Usable-window mismatch for "
                        f"{block_uid}: "
                        f"manifest={manifest_usable}, "
                        f"HDF5={actual_usable}"
                    )

                manifest_window_counter += manifest_usable
                block_counter += 1

                # ------------------------------------------------
                # Create one record per usable Train window
                # ------------------------------------------------

                for candidate_index in np.flatnonzero(usable):

                    candidate_index = int(
                        candidate_index
                    )

                    start = int(
                        starts[candidate_index]
                    )

                    stop = start + WINDOW_LENGTH

                    if start < 0:
                        raise RuntimeError(
                            f"Negative window start: {block_uid}"
                        )

                    if stop > eeg.shape[1]:
                        raise RuntimeError(
                            f"Window exceeds EEG block: "
                            f"{block_uid}"
                        )

                    record = WindowRecord(
                        subject_id=subject_id,
                        subject_index=subject_to_index[
                            subject_id
                        ],
                        block_uid=block_uid,
                        block_index=block_index,
                        eye_state=eye_state,
                        label=STATE_TO_LABEL[eye_state],
                        candidate_window_index=candidate_index,
                        start_sample=start,
                        h5_path=h5_path,
                    )

                    records.append(record)

                    state_counts[eye_state] += 1

    # --------------------------------------------------------
    # Global checks
    # --------------------------------------------------------

    if block_counter != EXPECTED_ELIGIBLE_TRAIN_BLOCKS:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_TRAIN_BLOCKS} "
            f"eligible blocks, indexed {block_counter}"
        )

    if manifest_window_counter != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            f"Manifest usable windows = "
            f"{manifest_window_counter}; "
            f"expected {EXPECTED_TRAIN_WINDOWS}"
        )

    if len(records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            f"Dataset index contains {len(records)} windows; "
            f"expected {EXPECTED_TRAIN_WINDOWS}"
        )

    if len(records) != int(
        scaler["fitting_train_window_count"]
    ):
        raise RuntimeError(
            "Window index count differs from "
            "locked normalization artifact."
        )

    print("\nINDEX AUDIT PASS")
    print(f"Eligible blocks indexed: {block_counter}")
    print(f"Usable windows indexed: {len(records)}")
    print(f"EO windows: {state_counts['EO']}")
    print(f"EC windows: {state_counts['EC']}")

    print("\nFirst record:")
    print(records[0])

    print("\nLast record:")
    print(records[-1])

    return records, subject_to_index


# ============================================================
# Step 2 — PyTorch Dataset
# ============================================================

class RealTrainWindowDataset(Dataset):

    def __init__(
        self,
        records: list[WindowRecord],
        scaler: dict,
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

        if self.means.shape != (19,):
            raise RuntimeError(
                f"Unexpected mean shape: "
                f"{self.means.shape}"
            )

        if self.stds.shape != (19,):
            raise RuntimeError(
                f"Unexpected std shape: "
                f"{self.stds.shape}"
            )

        if self.window_length != WINDOW_LENGTH:
            raise RuntimeError(
                "Unexpected window length."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):

        record = self.records[index]

        with h5py.File(
            record.h5_path,
            "r",
        ) as h5:

            eeg = h5[
                f"blocks/{record.block_uid}/eeg_uv"
            ]

            start = record.start_sample
            stop = start + self.window_length

            window_uv = np.asarray(
                eeg[:, start:stop],
                dtype=np.float32,
            )

        if window_uv.shape != (
            19,
            self.window_length,
        ):
            raise RuntimeError(
                f"Unexpected window shape: "
                f"{window_uv.shape}"
            )

        if not np.isfinite(window_uv).all():
            raise RuntimeError(
                f"Nonfinite EEG in "
                f"{record.block_uid}"
            )

        # ----------------------------------------------------
        # Apply ONE frozen global Train-only scaler
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
                "Nonfinite normalized EEG."
            )

        x = torch.from_numpy(normalized)

        y = torch.tensor(
            record.label,
            dtype=torch.long,
        )

        subject_index = torch.tensor(
            record.subject_index,
            dtype=torch.long,
        )

        return {
            "x": x,
            "y": y,

            # Stable deterministic technical index.
            "subject_index": subject_index,

            # Metadata for audit/debugging.
            "subject_id": record.subject_id,
            "eye_state": record.eye_state,
            "block_uid": record.block_uid,
            "candidate_window_index": (
                record.candidate_window_index
            ),
        }


# ============================================================
# Step 3 — Main inspection
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — REAL TRAIN DATASET CHECK")
    print("=" * 80)

    # --------------------------------------------------------
    # Load locked scaler
    # --------------------------------------------------------

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError(
            "Normalization is not locked."
        )

    # --------------------------------------------------------
    # Load final split manifest
    # --------------------------------------------------------

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    # --------------------------------------------------------
    # Build deterministic index
    # --------------------------------------------------------

    records, subject_to_index = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    # --------------------------------------------------------
    # Construct Dataset
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 2 — CREATE PYTORCH DATASET")
    print("=" * 80)

    dataset = RealTrainWindowDataset(
        records=records,
        scaler=scaler,
    )

    print(f"dataset length: {len(dataset)}")

    if len(dataset) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            "Unexpected dataset length."
        )

    # --------------------------------------------------------
    # Inspect one sample
    # --------------------------------------------------------

    sample = dataset[0]

    print("\nFIRST DATASET SAMPLE")
    print(
        f"x shape: "
        f"{tuple(sample['x'].shape)}"
    )
    print(
        f"x dtype: "
        f"{sample['x'].dtype}"
    )
    print(
        f"label: "
        f"{sample['y'].item()}"
    )
    print(
        f"subject_index: "
        f"{sample['subject_index'].item()}"
    )
    print(
        f"subject_id: "
        f"{sample['subject_id']}"
    )
    print(
        f"eye_state: "
        f"{sample['eye_state']}"
    )
    print(
        f"block_uid: "
        f"{sample['block_uid']}"
    )
    print(
        f"window index: "
        f"{sample['candidate_window_index']}"
    )

    # --------------------------------------------------------
    # Build first real mini-batch
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 3 — CREATE FIRST REAL MINI-BATCH")
    print("=" * 80)

    loader = DataLoader(
        dataset,
        batch_size=32,

        # Deterministic smoke test.
        shuffle=False,

        # Correctness first.
        # We optimize workers later.
        num_workers=0,

        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    batch = next(
        iter(loader)
    )

    x = batch["x"]
    y = batch["y"]
    subject_index = batch[
        "subject_index"
    ]

    print(f"batch x shape: {tuple(x.shape)}")
    print(f"batch x dtype: {x.dtype}")
    print(f"batch y shape: {tuple(y.shape)}")
    print(
        f"subject_index shape: "
        f"{tuple(subject_index.shape)}"
    )

    if x.shape != (
        32,
        19,
        512,
    ):
        raise RuntimeError(
            f"Unexpected batch shape: {x.shape}"
        )

    label_counts = torch.bincount(
        y,
        minlength=2,
    )

    print(
        f"EO labels in batch: "
        f"{int(label_counts[0])}"
    )
    print(
        f"EC labels in batch: "
        f"{int(label_counts[1])}"
    )

    print(
        f"batch normalized mean: "
        f"{x.mean().item():.6f}"
    )

    print(
        f"batch normalized std: "
        f"{x.std(unbiased=False).item():.6f}"
    )

    # --------------------------------------------------------
    # GPU smoke test on a REAL mini-batch
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 4 — REAL MINI-BATCH GPU CHECK")
    print("=" * 80)

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        x_gpu = x.to(
            device,
            non_blocking=True,
        )

        y_gpu = y.to(
            device,
            non_blocking=True,
        )

        subject_gpu = subject_index.to(
            device,
            non_blocking=True,
        )

        # Small real computation.
        batch_energy = torch.mean(
            x_gpu ** 2
        )

        channel_energy = torch.mean(
            x_gpu ** 2,
            dim=(0, 2),
        )

        torch.cuda.synchronize()

        print(f"device: {x_gpu.device}")

        print(
            f"x GPU shape: "
            f"{tuple(x_gpu.shape)}"
        )

        print(
            f"batch energy: "
            f"{batch_energy.item():.6f}"
        )

        print(
            f"channel energy shape: "
            f"{tuple(channel_energy.shape)}"
        )

        print(
            f"GPU memory allocated MB: "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f}"
        )

        print(
            f"GPU memory reserved MB: "
            f"{torch.cuda.memory_reserved() / 1024**2:.2f}"
        )

        print("REAL MINI-BATCH GPU PASS")

    else:
        print(
            "CUDA unavailable — CPU only."
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("REAL TRAIN DATASET PASS")
    print("=" * 80)

    print(
        f"subjects: "
        f"{len(subject_to_index)}"
    )

    print(
        f"eligible Train blocks: "
        f"{EXPECTED_ELIGIBLE_TRAIN_BLOCKS}"
    )

    print(
        f"usable Train windows: "
        f"{len(dataset)}"
    )

    print(
        "sample shape: (19, 512)"
    )

    print(
        "mini-batch shape: (32, 19, 512)"
    )

    print(
        "normalization: locked global Train-only"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Test EEG accessed: NO"
    )


if __name__ == "__main__":
    main()