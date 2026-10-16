from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    WINDOW_LENGTH,
    build_train_window_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# ============================================================
# Fixed settings for this baseline
# ============================================================

SFREQ = 128.0

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128

EXPECTED_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

EPS = 1e-12


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — TRAIN LOG-BANDPOWER FEATURES")
    print("=" * 80)

    print(
        "\nScope:"
        "\n- Real Train only"
        "\n- Physical-unit EEG (uV)"
        "\n- Only usable windows from eligible Train blocks"
        "\n- Validation EEG: NOT ACCESSED"
        "\n- Test EEG: NOT ACCESSED"
        "\n- No classifier fitted yet"
    )

    # ========================================================
    # Step 1 — Load locked Train definition
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

    channels = list(
        scaler["channel_order"]
    )

    if len(channels) != 19:
        raise RuntimeError(
            f"Expected 19 channels, got {len(channels)}"
        )

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

    print(f"\nsubjects: {len(subject_to_index)}")
    print(f"windows: {len(records)}")
    print(f"channels: {len(channels)}")
    print(f"bands: {list(BANDS.keys())}")

    n_features = (
        len(channels)
        * len(BANDS)
    )

    print(
        f"feature count per window: "
        f"{n_features}"
    )

    # ========================================================
    # Step 2 — Allocate feature matrix
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — COMPUTE LOG-BANDPOWER")
    print("=" * 80)

    n_windows = len(records)

    X = np.empty(
        (
            n_windows,
            n_features,
        ),
        dtype=np.float32,
    )

    y = np.empty(
        n_windows,
        dtype=np.int64,
    )

    subject_index = np.empty(
        n_windows,
        dtype=np.int64,
    )

    block_index = np.empty(
        n_windows,
        dtype=np.int64,
    )

    # ========================================================
    # Step 3 — Process block-by-block
    # ========================================================

    current_h5_path: Path | None = None
    current_h5: h5py.File | None = None

    i = 0
    block_counter = 0

    start_time = time.perf_counter()

    band_items = list(
        BANDS.items()
    )

    try:

        while i < n_windows:

            first = records[i]

            # ------------------------------------------------
            # Gather contiguous windows from same block
            # ------------------------------------------------

            j = i + 1

            while (
                j < n_windows
                and records[j].subject_id
                == first.subject_id
                and records[j].block_uid
                == first.block_uid
            ):
                j += 1

            block_records = records[i:j]

            # ------------------------------------------------
            # Open subject HDF5
            # ------------------------------------------------

            if current_h5_path != first.h5_path:

                if current_h5 is not None:
                    current_h5.close()

                current_h5_path = first.h5_path

                current_h5 = h5py.File(
                    current_h5_path,
                    "r",
                )

            eeg = current_h5[
                f"blocks/{first.block_uid}/eeg_uv"
            ]

            # ------------------------------------------------
            # Read usable windows for this block
            # ------------------------------------------------

            windows = np.stack(
                [
                    np.asarray(
                        eeg[
                            :,
                            r.start_sample:
                            r.start_sample
                            + WINDOW_LENGTH
                        ],
                        dtype=np.float64,
                    )
                    for r in block_records
                ],
                axis=0,
            )

            if windows.shape[1:] != (
                19,
                WINDOW_LENGTH,
            ):
                raise RuntimeError(
                    f"Unexpected shape: "
                    f"{first.block_uid} "
                    f"{windows.shape}"
                )

            if not np.isfinite(
                windows
            ).all():
                raise RuntimeError(
                    f"Nonfinite EEG: "
                    f"{first.block_uid}"
                )

            # ------------------------------------------------
            # Welch PSD
            # ------------------------------------------------

            freqs, psd = welch(
                windows,
                fs=SFREQ,
                window="hann",
                nperseg=WELCH_NPERSEG,
                noverlap=WELCH_NOVERLAP,
                detrend="constant",
                scaling="density",
                axis=-1,
            )

            # Shape:
            # block_windows × 19 × frequencies

            # ------------------------------------------------
            # Bandpower features
            # ------------------------------------------------

            features = []

            for (
                band_name,
                (low_hz, high_hz),
            ) in band_items:

                mask = (
                    (freqs >= low_hz)
                    & (freqs < high_hz)
                )

                # Include 30 Hz for beta endpoint.
                if band_name == "beta":
                    mask = (
                        (freqs >= low_hz)
                        & (freqs <= high_hz)
                    )

                band_freqs = freqs[
                    mask
                ]

                band_psd = psd[
                    ...,
                    mask
                ]

                power = np.trapezoid(
                    band_psd,
                    x=band_freqs,
                    axis=-1,
                )

                if not np.isfinite(
                    power
                ).all():
                    raise RuntimeError(
                        f"Nonfinite {band_name} power."
                    )

                if np.any(
                    power <= 0
                ):
                    raise RuntimeError(
                        f"Non-positive {band_name} power."
                    )

                # log10(uV^2)
                log_power = np.log10(
                    np.maximum(
                        power,
                        EPS,
                    )
                )

                features.append(
                    log_power
                )

            # ------------------------------------------------
            # Feature order:
            #
            # delta: 19 channels
            # theta: 19 channels
            # alpha: 19 channels
            # beta:  19 channels
            # ------------------------------------------------

            block_X = np.concatenate(
                features,
                axis=1,
            )

            if block_X.shape != (
                len(block_records),
                n_features,
            ):
                raise RuntimeError(
                    f"Unexpected feature shape: "
                    f"{block_X.shape}"
                )

            X[i:j] = block_X.astype(
                np.float32
            )

            y[i:j] = first.label

            subject_index[i:j] = (
                first.subject_index
            )

            block_index[i:j] = (
                first.block_index
            )

            block_counter += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                block_counter == 1
                or block_counter % 200 == 0
                or j == n_windows
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"blocks={block_counter:>4}"
                    f" | windows={j:>6}/{n_windows}"
                    f" | elapsed={elapsed:>7.1f}s"
                )

            i = j

    finally:

        if current_h5 is not None:
            current_h5.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\nFeature extraction finished "
        f"in {elapsed:.2f} seconds"
    )

    # ========================================================
    # Step 4 — Structural verification
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — VERIFY FEATURE MATRIX")
    print("=" * 80)

    if X.shape != (
        EXPECTED_WINDOWS,
        n_features,
    ):
        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Feature matrix contains NaN/Inf."
        )

    eo_count = int(
        np.sum(y == 0)
    )

    ec_count = int(
        np.sum(y == 1)
    )

    if eo_count != 23_302:
        raise RuntimeError(
            f"Unexpected EO count: {eo_count}"
        )

    if ec_count != 24_533:
        raise RuntimeError(
            f"Unexpected EC count: {ec_count}"
        )

    print(f"X shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"EO windows: {eo_count}")
    print(f"EC windows: {ec_count}")

    print(
        f"unique subjects: "
        f"{len(np.unique(subject_index))}"
    )

    # ========================================================
    # Step 5 — Feature names
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — FEATURE DEFINITION")
    print("=" * 80)

    feature_names = []

    for band_name in BANDS:

        for channel in channels:

            feature_names.append(
                f"{band_name}_{channel}"
            )

    if len(feature_names) != n_features:
        raise RuntimeError(
            "Feature-name count mismatch."
        )

    print(
        f"feature count: "
        f"{len(feature_names)}"
    )

    print(
        "\nfirst 10 features:"
    )

    for name in feature_names[:10]:
        print(f"  {name}")

    print(
        "\nlast 10 features:"
    )

    for name in feature_names[-10:]:
        print(f"  {name}")

    # ========================================================
    # Step 6 — Descriptive feature sanity
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — FEATURE SANITY")
    print("=" * 80)

    feature_mean = X.mean(
        axis=0,
        dtype=np.float64,
    )

    feature_std = X.std(
        axis=0,
        dtype=np.float64,
    )

    print(
        f"global feature min: "
        f"{X.min():.6f}"
    )

    print(
        f"global feature max: "
        f"{X.max():.6f}"
    )

    print(
        f"minimum feature std: "
        f"{feature_std.min():.6f}"
    )

    print(
        f"maximum feature std: "
        f"{feature_std.max():.6f}"
    )

    if np.any(
        feature_std <= 0
    ):
        raise RuntimeError(
            "At least one feature has zero variance."
        )

    # --------------------------------------------------------
    # Alpha O1/O2 sanity from feature matrix itself
    # --------------------------------------------------------

    alpha_start = (
        2
        * len(channels)
    )

    o1_index = channels.index(
        "O1"
    )

    o2_index = channels.index(
        "O2"
    )

    alpha_o1_feature = (
        alpha_start
        + o1_index
    )

    alpha_o2_feature = (
        alpha_start
        + o2_index
    )

    for (
        name,
        feature_index,
    ) in [
        (
            "alpha_O1",
            alpha_o1_feature,
        ),
        (
            "alpha_O2",
            alpha_o2_feature,
        ),
    ]:

        eo_values = X[
            y == 0,
            feature_index,
        ]

        ec_values = X[
            y == 1,
            feature_index,
        ]

        difference_db = (
            10.0
            * (
                np.median(ec_values)
                - np.median(eo_values)
            )
        )

        print(
            f"\n{name}"
        )

        print(
            f"median EO log10 power: "
            f"{np.median(eo_values):.6f}"
        )

        print(
            f"median EC log10 power: "
            f"{np.median(ec_values):.6f}"
        )

        print(
            f"window-pooled median difference: "
            f"{difference_db:.3f} dB"
        )

    print(
        "\nNOTE:"
    )

    print(
        "The window-pooled EC/EO difference above "
        "is only a feature sanity check."
    )

    print(
        "The previous subject-level Train alpha analysis "
        "is the correct descriptive subject-aware summary."
    )

    # ========================================================
    # Step 7 — Save temporary feature package
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — SAVE TRAIN FEATURE PACKAGE")
    print("=" * 80)

    output_dir = (
        REPO_ROOT
        / "outputs"
        / "features"
        / "scenario2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "real_train_logbandpower_v1.npz"
    )

    np.savez_compressed(
        output_path,

        X=X,
        y=y,

        subject_index=subject_index,
        block_index=block_index,

        feature_names=np.asarray(
            feature_names,
            dtype="U64",
        ),

        channel_names=np.asarray(
            channels,
            dtype="U16",
        ),

        band_names=np.asarray(
            list(BANDS.keys()),
            dtype="U16",
        ),

        sampling_rate_hz=np.asarray(
            SFREQ,
            dtype=np.float64,
        ),

        window_length_samples=np.asarray(
            WINDOW_LENGTH,
            dtype=np.int64,
        ),
    )

    print(f"saved: {output_path}")

    print(
        f"size MiB: "
        f"{output_path.stat().st_size / 1024**2:.2f}"
    )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAIN LOG-BANDPOWER FEATURE EXTRACTION PASS")
    print("=" * 80)

    print(
        f"subjects: {EXPECTED_SUBJECTS}"
    )

    print(
        f"windows: {EXPECTED_WINDOWS}"
    )

    print(
        f"features/window: {n_features}"
    )

    print(
        f"X shape: {X.shape}"
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
        "CONTROLLED REAL VALIDATION + "
        "LOGISTIC REGRESSION BASELINE"
    )


if __name__ == "__main__":
    main()