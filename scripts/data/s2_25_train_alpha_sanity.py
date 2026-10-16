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


# ============================================================
# Frozen / descriptive settings
# ============================================================

SFREQ = 128.0

ALPHA_LOW_HZ = 8.0
ALPHA_HIGH_HZ = 13.0

# Exploratory Train-only Welch settings.
# These are NOT yet being declared as the final paper metric.
WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128

EXPECTED_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — TRAIN-ONLY ALPHA SANITY BASELINE")
    print("=" * 80)

    print(
        "\nIMPORTANT:"
        "\n- Real Train only"
        "\n- Physical-unit EEG (uV)"
        "\n- Validation EEG: NOT ACCESSED"
        "\n- Test EEG: NOT ACCESSED"
        "\n- No model fitting"
        "\n- No final statistical inference"
    )

    # ========================================================
    # Step 1 — Load frozen Train definition
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — LOAD FINAL TRAIN DEFINITION")
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

    print("\nScientific input:")
    print(f"subjects: {len(subject_to_index)}")
    print(f"Train windows: {len(records)}")
    print(f"channels: {len(channels)}")
    print("physical unit: uV")

    # ========================================================
    # Step 2 — Allocate result arrays
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — COMPUTE WINDOW-LEVEL ALPHA POWER")
    print("=" * 80)

    n_windows = len(records)

    # log10(alpha power in uV^2)
    log_alpha = np.empty(
        (n_windows, 19),
        dtype=np.float64,
    )

    labels = np.empty(
        n_windows,
        dtype=np.int64,
    )

    subject_indices = np.empty(
        n_windows,
        dtype=np.int64,
    )

    # ========================================================
    # Step 3 — Process block by block
    # ========================================================

    start_time = time.perf_counter()

    current_h5_path: Path | None = None
    current_h5: h5py.File | None = None

    block_count = 0

    i = 0

    try:

        while i < n_windows:

            first = records[i]

            # ------------------------------------------------
            # Find contiguous records belonging to same block
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
            # Open subject HDF5 only when needed
            # ------------------------------------------------

            if (
                current_h5_path
                != first.h5_path
            ):

                if current_h5 is not None:
                    current_h5.close()

                current_h5_path = (
                    first.h5_path
                )

                current_h5 = h5py.File(
                    current_h5_path,
                    "r",
                )

            eeg = current_h5[
                f"blocks/{first.block_uid}/eeg_uv"
            ]

            # ------------------------------------------------
            # Read all usable windows from this block
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

            # Shape:
            # (usable_windows_in_block, 19, 512)

            if (
                windows.ndim != 3
                or windows.shape[1:] != (19, 512)
            ):
                raise RuntimeError(
                    f"Unexpected window shape for "
                    f"{first.block_uid}: "
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

            # psd:
            # windows × channels × frequencies
            #
            # Unit:
            # uV^2 / Hz

            alpha_mask = (
                (freqs >= ALPHA_LOW_HZ)
                & (freqs <= ALPHA_HIGH_HZ)
            )

            alpha_freqs = freqs[
                alpha_mask
            ]

            alpha_psd = psd[
                ...,
                alpha_mask
            ]

            # ------------------------------------------------
            # Integrate alpha-band PSD
            # ------------------------------------------------

            alpha_power = np.trapezoid(
                alpha_psd,
                x=alpha_freqs,
                axis=-1,
            )

            # Unit: uV^2

            if not np.isfinite(
                alpha_power
            ).all():
                raise RuntimeError(
                    "Nonfinite alpha power."
                )

            if np.any(
                alpha_power <= 0
            ):
                raise RuntimeError(
                    "Non-positive alpha power."
                )

            block_log_alpha = np.log10(
                alpha_power
            )

            n_block_windows = (
                len(block_records)
            )

            log_alpha[
                i:j
            ] = block_log_alpha

            labels[
                i:j
            ] = first.label

            subject_indices[
                i:j
            ] = first.subject_index

            block_count += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                block_count == 1
                or block_count % 200 == 0
                or j == n_windows
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"blocks={block_count:>4}"
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
        f"\nPSD computation finished in "
        f"{elapsed:.2f} seconds"
    )

    # ========================================================
    # Step 4 — Basic accounting
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — VERIFY SPECTRAL ACCOUNTING")
    print("=" * 80)

    if not np.isfinite(
        log_alpha
    ).all():
        raise RuntimeError(
            "Nonfinite log-alpha values."
        )

    eo_count = int(
        np.sum(labels == 0)
    )

    ec_count = int(
        np.sum(labels == 1)
    )

    print(f"EO windows: {eo_count}")
    print(f"EC windows: {ec_count}")
    print(
        f"total: {eo_count + ec_count}"
    )

    if eo_count != 23_302:
        raise RuntimeError(
            f"Unexpected EO count: {eo_count}"
        )

    if ec_count != 24_533:
        raise RuntimeError(
            f"Unexpected EC count: {ec_count}"
        )

    # ========================================================
    # Step 5 — Subject × State aggregation
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — SUBJECT-LEVEL EO / EC ALPHA")
    print("=" * 80)

    # Dimensions:
    # subject × state × channel
    #
    # state:
    # 0 = EO
    # 1 = EC

    subject_state = np.full(
        (
            EXPECTED_SUBJECTS,
            2,
            19,
        ),
        np.nan,
        dtype=np.float64,
    )

    subject_state_counts = np.zeros(
        (
            EXPECTED_SUBJECTS,
            2,
        ),
        dtype=np.int64,
    )

    for subject_index in range(
        EXPECTED_SUBJECTS
    ):

        for label in (0, 1):

            mask = (
                (subject_indices == subject_index)
                & (labels == label)
            )

            count = int(
                np.sum(mask)
            )

            if count == 0:
                raise RuntimeError(
                    f"Subject {subject_index} "
                    f"has zero windows in state {label}."
                )

            subject_state_counts[
                subject_index,
                label,
            ] = count

            # Robust within-subject/state summary.
            subject_state[
                subject_index,
                label,
                :
            ] = np.median(
                log_alpha[mask],
                axis=0,
            )

    if not np.isfinite(
        subject_state
    ).all():
        raise RuntimeError(
            "Subject-state alpha contains NaN/Inf."
        )

    print(
        "All 198 subjects have "
        "both EO and EC Train alpha summaries: PASS"
    )

    # ========================================================
    # Step 6 — EC minus EO
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — EC MINUS EO ALPHA DIFFERENCE")
    print("=" * 80)

    # log10 power difference.
    log_difference = (
        subject_state[:, 1, :]
        - subject_state[:, 0, :]
    )

    # Convert to dB:
    #
    # 10 log10(P_EC / P_EO)
    delta_db = (
        10.0
        * log_difference
    )

    # --------------------------------------------------------
    # Whole-head descriptive summary
    # --------------------------------------------------------

    subject_global_delta_db = (
        np.mean(
            delta_db,
            axis=1,
        )
    )

    print(
        "\nWHOLE-HEAD MEAN ALPHA:"
    )

    print(
        f"median EC-EO: "
        f"{np.median(subject_global_delta_db):.3f} dB"
    )

    print(
        f"mean EC-EO:   "
        f"{np.mean(subject_global_delta_db):.3f} dB"
    )

    print(
        f"IQR: "
        f"["
        f"{np.percentile(subject_global_delta_db, 25):.3f}, "
        f"{np.percentile(subject_global_delta_db, 75):.3f}"
        f"] dB"
    )

    print(
        f"subjects with EC > EO: "
        f"{np.mean(subject_global_delta_db > 0):.3f}"
    )

    # ========================================================
    # Step 7 — Channel-wise spatial sanity
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — CHANNEL-WISE EC-EO ALPHA")
    print("=" * 80)

    channel_results = []

    for channel_index, channel in enumerate(
        channels
    ):

        values = delta_db[
            :,
            channel_index,
        ]

        channel_results.append(
            (
                channel,
                float(
                    np.median(values)
                ),
                float(
                    np.mean(values)
                ),
                float(
                    np.percentile(
                        values,
                        25,
                    )
                ),
                float(
                    np.percentile(
                        values,
                        75,
                    )
                ),
                float(
                    np.mean(
                        values > 0
                    )
                ),
            )
        )

    # Sort by median EC-EO effect.
    channel_results = sorted(
        channel_results,
        key=lambda row: row[1],
        reverse=True,
    )

    print(
        "\nChannel"
        "        Median_dB"
        "    Mean_dB"
        "      IQR_dB"
        "          P(EC>EO)"
    )

    print("-" * 72)

    for (
        channel,
        median_db,
        mean_db,
        q25,
        q75,
        positive_fraction,
    ) in channel_results:

        print(
            f"{channel:<8}"
            f"{median_db:>12.3f}"
            f"{mean_db:>12.3f}"
            f"    [{q25:>6.3f}, {q75:>6.3f}]"
            f"{positive_fraction:>12.3f}"
        )

    # ========================================================
    # Step 8 — Explicit O1 / O2 inspection
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — OCCIPITAL SANITY CHECK")
    print("=" * 80)

    for channel in (
        "O1",
        "O2",
    ):

        channel_index = channels.index(
            channel
        )

        values = delta_db[
            :,
            channel_index,
        ]

        # State absolute power in dB relative to 1 uV^2.
        eo_db = (
            10.0
            * subject_state[
                :,
                0,
                channel_index
            ]
        )

        ec_db = (
            10.0
            * subject_state[
                :,
                1,
                channel_index
            ]
        )

        print(f"\n{channel}")

        print(
            f"median EO alpha power: "
            f"{np.median(eo_db):.3f} dB re 1 uV^2"
        )

        print(
            f"median EC alpha power: "
            f"{np.median(ec_db):.3f} dB re 1 uV^2"
        )

        print(
            f"median EC-EO: "
            f"{np.median(values):.3f} dB"
        )

        print(
            f"subjects EC > EO: "
            f"{np.mean(values > 0):.3f}"
        )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAIN-ONLY ALPHA SANITY COMPLETE")
    print("=" * 80)

    print(
        f"subjects: {EXPECTED_SUBJECTS}"
    )

    print(
        f"Train windows analyzed: "
        f"{EXPECTED_WINDOWS}"
    )

    print(
        "frequency band: 8–13 Hz"
    )

    print(
        "input unit: physical uV"
    )

    print(
        "PSD method: Welch"
    )

    print(
        f"nperseg: {WELCH_NPERSEG}"
    )

    print(
        f"noverlap: {WELCH_NOVERLAP}"
    )

    print(
        "Validation EEG accessed: NO"
    )

    print(
        "Test EEG accessed: NO"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a Train-only descriptive sanity baseline."
    )

    print(
        "Do NOT use this result to change QC, "
        "cohort membership, or the locked split."
    )


if __name__ == "__main__":
    main()