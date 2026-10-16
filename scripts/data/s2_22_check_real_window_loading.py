from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


# ============================================================
# Paths
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
# Small helpers
# ============================================================

def parse_json_attr(value):
    """
    HDF5 attributes such as channel_names are stored as JSON text.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        return json.loads(value)

    return value


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("SCENARIO 2 — FIRST REAL TRAIN WINDOW")
    print("=" * 80)

    # --------------------------------------------------------
    # Step 1 — Load frozen final split
    # --------------------------------------------------------

    split = pd.read_csv(FINAL_SPLIT_MANIFEST)

    train = split[
        split["split"].astype(str).eq("Train")
    ].copy()

    eligible_train = train[
        train["block_eligible"].map(as_bool)
    ].copy()

    eligible_train = eligible_train[
        eligible_train["usable_windows"] > 0
    ].copy()

    eligible_train = eligible_train.sort_values(
        ["subject_id", "block_index"]
    )

    print("\n[1] FINAL TRAIN SELECTION")
    print(f"all final rows: {len(split)}")
    print(f"assigned Train blocks: {len(train)}")
    print(f"eligible Train blocks: {len(eligible_train)}")

    if len(train) != 1980:
        raise RuntimeError(
            f"Expected 1980 Train blocks, got {len(train)}"
        )

    if len(eligible_train) != 1970:
        raise RuntimeError(
            f"Expected 1970 eligible Train blocks, "
            f"got {len(eligible_train)}"
        )

    row = eligible_train.iloc[0]

    subject_id = str(row["subject_id"])
    block_uid = str(row["block_uid"])
    eye_state = str(row["eye_state"])

    print(f"\nselected subject: {subject_id}")
    print(f"selected block:   {block_uid}")
    print(f"eye state:        {eye_state}")
    print(f"manifest usable windows: {int(row['usable_windows'])}")

    # --------------------------------------------------------
    # Step 2 — Load frozen normalization
    # --------------------------------------------------------

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError(
            "Normalization artifact is not locked."
        )

    channels = list(scaler["channel_order"])

    means = np.asarray(
        scaler["means_uv"],
        dtype=np.float64,
    )

    stds = np.asarray(
        scaler["stds_uv"],
        dtype=np.float64,
    )

    window_length = int(
        scaler["window_length_samples"]
    )

    sampling_rate = float(
        scaler["sampling_rate_hz"]
    )

    if len(channels) != 19:
        raise RuntimeError(
            f"Expected 19 channels, got {len(channels)}"
        )

    if means.shape != (19,):
        raise RuntimeError(
            f"Unexpected mean shape: {means.shape}"
        )

    if stds.shape != (19,):
        raise RuntimeError(
            f"Unexpected std shape: {stds.shape}"
        )

    if window_length != 512:
        raise RuntimeError(
            f"Expected window length 512, got {window_length}"
        )

    if sampling_rate != 128:
        raise RuntimeError(
            f"Expected sampling rate 128, got {sampling_rate}"
        )

    print("\n[2] LOCKED NORMALIZATION")
    print(f"channels: {channels}")
    print(f"window length: {window_length}")
    print(f"sampling rate: {sampling_rate} Hz")
    print("scaler status: TRAIN_NORMALIZATION_LOCKED")

    # --------------------------------------------------------
    # Step 3 — Open selected subject HDF5
    # --------------------------------------------------------

    h5_path = (
        ALL_BLOCKS_ROOT
        / f"{subject_id}_all_blocks.h5"
    )

    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    print("\n[3] HDF5 INPUT")
    print(f"path: {h5_path}")

    with h5py.File(h5_path, "r") as h5:

        group_path = f"blocks/{block_uid}"

        if group_path not in h5:
            raise RuntimeError(
                f"Missing group: {group_path}"
            )

        block = h5[group_path]

        # ----------------------------------------------------
        # Step 4 — Verify block metadata
        # ----------------------------------------------------

        h5_subject = str(
            block.attrs["subject_id"]
        )

        h5_state = str(
            block.attrs["state"]
        )

        h5_split = str(
            block.attrs["provisional_split"]
        )

        h5_eligible = as_bool(
            block.attrs["block_eligible"]
        )

        h5_channels = parse_json_attr(
            block.attrs["channel_names"]
        )

        h5_unit = str(
            block.attrs["physical_unit"]
        )

        h5_sfreq = float(
            block.attrs["processed_sfreq"]
        )

        if h5_subject != subject_id:
            raise RuntimeError(
                "Subject mismatch."
            )

        if h5_state != eye_state:
            raise RuntimeError(
                "Eye-state mismatch."
            )

        if h5_split != "Train":
            raise RuntimeError(
                f"Refusing non-Train block: {h5_split}"
            )

        if not h5_eligible:
            raise RuntimeError(
                "Refusing ineligible Train block."
            )

        if h5_channels != channels:
            raise RuntimeError(
                "Channel order does not match scaler."
            )

        if h5_unit != "uV":
            raise RuntimeError(
                f"Expected uV, got {h5_unit}"
            )

        if h5_sfreq != sampling_rate:
            raise RuntimeError(
                "Sampling-rate mismatch."
            )

        print("\n[4] BLOCK VALIDATION")
        print("subject match:       PASS")
        print("state match:         PASS")
        print("Train-only check:    PASS")
        print("eligible check:      PASS")
        print("channel order:       PASS")
        print("physical unit:       PASS")
        print("sampling rate:       PASS")

        # ----------------------------------------------------
        # Step 5 — Read window metadata
        # ----------------------------------------------------

        eeg = block["eeg_uv"]

        starts = block[
            "window_start_samples"
        ][:]

        usable = block[
            "window_usable_mask"
        ][:]

        print("\n[5] WINDOW METADATA")
        print(f"EEG shape: {eeg.shape}")
        print(f"EEG dtype: {eeg.dtype}")
        print(f"candidate windows: {len(starts)}")
        print(f"usable windows: {int(usable.sum())}")

        if eeg.shape[0] != 19:
            raise RuntimeError(
                f"Expected 19 EEG channels, got {eeg.shape}"
            )

        if len(starts) != 25:
            raise RuntimeError(
                f"Expected 25 candidate windows, got {len(starts)}"
            )

        if int(usable.sum()) != int(row["usable_windows"]):
            raise RuntimeError(
                "Usable-window count differs from final manifest."
            )

        usable_indices = np.flatnonzero(usable)

        if len(usable_indices) == 0:
            raise RuntimeError(
                "Selected block contains no usable windows."
            )

        candidate_index = int(
            usable_indices[0]
        )

        start = int(
            starts[candidate_index]
        )

        stop = start + window_length

        if stop > eeg.shape[1]:
            raise RuntimeError(
                "Window exceeds EEG block."
            )

        print(
            f"selected usable window index: "
            f"{candidate_index}"
        )
        print(
            f"sample interval: [{start}, {stop})"
        )

        # ----------------------------------------------------
        # Step 6 — Read EXACTLY ONE real Train window
        # ----------------------------------------------------

        raw_window_uv = np.asarray(
            eeg[:, start:stop],
            dtype=np.float32,
        )

    # HDF5 is closed here.

    if raw_window_uv.shape != (19, 512):
        raise RuntimeError(
            f"Unexpected raw shape: {raw_window_uv.shape}"
        )

    if not np.isfinite(raw_window_uv).all():
        raise RuntimeError(
            "Nonfinite EEG sample detected."
        )

    print("\n[6] RAW REAL TRAIN WINDOW")
    print(f"shape: {raw_window_uv.shape}")
    print(f"dtype: {raw_window_uv.dtype}")
    print(
        f"minimum uV: {raw_window_uv.min():.6f}"
    )
    print(
        f"maximum uV: {raw_window_uv.max():.6f}"
    )
    print(
        f"mean uV: {raw_window_uv.mean():.6f}"
    )
    print(
        f"std uV: {raw_window_uv.std():.6f}"
    )

    # --------------------------------------------------------
    # Step 7 — Apply the ONE locked global scaler
    # --------------------------------------------------------

    normalized = (
        (
            raw_window_uv.astype(np.float64)
            - means[:, None]
        )
        / stds[:, None]
    )

    if not np.isfinite(normalized).all():
        raise RuntimeError(
            "Nonfinite normalized EEG detected."
        )

    normalized = normalized.astype(
        np.float32
    )

    print("\n[7] NORMALIZED WINDOW")
    print(f"shape: {normalized.shape}")
    print(f"dtype: {normalized.dtype}")
    print(
        f"minimum z: {normalized.min():.6f}"
    )
    print(
        f"maximum z: {normalized.max():.6f}"
    )
    print(
        f"window mean z: {normalized.mean():.6f}"
    )
    print(
        f"window std z: {normalized.std():.6f}"
    )

    print(
        "\nNOTE: this single window is NOT expected "
        "to have exactly mean=0 and std=1."
    )
    print(
        "The locked scaler is global over the complete "
        "eligible Real Train population."
    )

    # --------------------------------------------------------
    # Step 8 — Convert to PyTorch tensor
    # --------------------------------------------------------

    x = torch.from_numpy(
        normalized
    )

    # Binary eye-state label for future classifiers.
    # EO = 0
    # EC = 1
    label = (
        0 if eye_state == "EO" else 1
    )

    y = torch.tensor(
        label,
        dtype=torch.long,
    )

    print("\n[8] PYTORCH SAMPLE")
    print(f"x shape: {tuple(x.shape)}")
    print(f"x dtype: {x.dtype}")
    print(f"label: {label}")
    print(
        f"label meaning: "
        f"{'EO' if label == 0 else 'EC'}"
    )

    # --------------------------------------------------------
    # Step 9 — GPU smoke test
    # --------------------------------------------------------

    print("\n[9] GPU SMOKE TEST")

    if torch.cuda.is_available():

        device = torch.device("cuda")

        x_gpu = x.to(
            device,
            non_blocking=False,
        )

        y_gpu = y.to(device)

        # Tiny real operation on the EEG tensor.
        energy = torch.mean(
            x_gpu ** 2
        )

        torch.cuda.synchronize()

        print(
            f"device: {x_gpu.device}"
        )
        print(
            f"x GPU shape: {tuple(x_gpu.shape)}"
        )
        print(
            f"y GPU: {int(y_gpu.item())}"
        )
        print(
            f"mean squared normalized amplitude: "
            f"{energy.item():.6f}"
        )
        print("GPU REAL EEG PASS")

    else:
        print(
            "CUDA unavailable — CPU path only."
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("REAL TRAIN WINDOW LOADING PASS")
    print("=" * 80)

    print(f"subject_id : {subject_id}")
    print(f"block_uid  : {block_uid}")
    print(f"eye_state  : {eye_state}")
    print(
        f"window     : candidate #{candidate_index}"
    )
    print(
        f"input      : 19 channels × 512 samples"
    )
    print(
        "normalizer : locked global Train-only scaler"
    )
    print(
        "held-out EEG accessed: NO"
    )


if __name__ == "__main__":
    main()