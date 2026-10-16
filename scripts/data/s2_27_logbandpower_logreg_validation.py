from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scipy.signal import welch

from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# Project paths
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

TRAIN_FEATURES_FILE = (
    REPO_ROOT
    / "outputs"
    / "features"
    / "scenario2"
    / "real_train_logbandpower_v1.npz"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
)

VALIDATION_FEATURES_FILE = (
    REPO_ROOT
    / "outputs"
    / "features"
    / "scenario2"
    / "real_validation_logbandpower_v1.npz"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "real_logbandpower_logreg_validation_v1.json"
)


# ============================================================
# Predeclared baseline configuration
# ============================================================

SFREQ = 128.0
WINDOW_LENGTH = 512

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128

STATE_TO_LABEL = {
    "EO": 0,
    "EC": 1,
}

# Baseline classifier configuration.
#
# IMPORTANT:
# These values are fixed BEFORE seeing Validation results.
LOGREG_C = 1.0
LOGREG_PENALTY = "l2"
LOGREG_SOLVER = "lbfgs"
LOGREG_MAX_ITER = 2000

EXPECTED_SUBJECTS = 198
EXPECTED_VALIDATION_BLOCKS = 396
EXPECTED_VALIDATION_BLOCKS_PER_SUBJECT = 2

EPS = 1e-12


# ============================================================
# Validation-window record
# ============================================================

@dataclass(frozen=True)
class ValidationWindowRecord:
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


# ============================================================
# Step 1 — Build controlled Validation index
# ============================================================

def build_validation_window_index(
    split: pd.DataFrame,
) -> tuple[
    list[ValidationWindowRecord],
    dict[str, int],
]:
    print("\n" + "=" * 80)
    print("STEP 1 — BUILD CONTROLLED REAL VALIDATION INDEX")
    print("=" * 80)

    # --------------------------------------------------------
    # Deterministic subject mapping
    #
    # Same rule used previously for Train:
    # sorted subject IDs -> [0, 197]
    # --------------------------------------------------------

    final_subjects = sorted(
        split["subject_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(final_subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"found {len(final_subjects)}"
        )

    subject_to_index = {
        subject_id: index
        for index, subject_id
        in enumerate(final_subjects)
    }

    # --------------------------------------------------------
    # Validation ONLY
    # --------------------------------------------------------

    validation = split[
        split["split"].astype(str).eq(
            "Validation"
        )
    ].copy()

    if len(validation) != EXPECTED_VALIDATION_BLOCKS:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_BLOCKS} "
            f"Validation blocks, found {len(validation)}"
        )

    # All included Validation blocks were already required
    # to be eligible by final cohort lock.
    if not validation[
        "block_eligible"
    ].map(as_bool).all():
        raise RuntimeError(
            "At least one final Validation block "
            "is unexpectedly ineligible."
        )

    # --------------------------------------------------------
    # Every subject must have exactly:
    #
    # 1 EO Validation block
    # 1 EC Validation block
    # --------------------------------------------------------

    block_counts = (
        validation
        .groupby("subject_id")
        .size()
    )

    if not (
        block_counts
        == EXPECTED_VALIDATION_BLOCKS_PER_SUBJECT
    ).all():
        raise RuntimeError(
            "Validation block count per subject "
            "is not exactly two."
        )

    state_counts = (
        validation
        .groupby(
            ["subject_id", "eye_state"]
        )
        .size()
        .unstack(fill_value=0)
    )

    for state in ("EO", "EC"):
        if state not in state_counts.columns:
            raise RuntimeError(
                f"Missing Validation state: {state}"
            )

        if not (
            state_counts[state] == 1
        ).all():
            raise RuntimeError(
                f"Expected exactly one {state} "
                f"Validation block per subject."
            )

    validation = validation.sort_values(
        ["subject_id", "block_index"]
    )

    records: list[
        ValidationWindowRecord
    ] = []

    candidate_total = 0
    usable_total = 0

    state_window_counts = {
        "EO": 0,
        "EC": 0,
    }

    # --------------------------------------------------------
    # Build index without touching Test
    # --------------------------------------------------------

    for subject_id, rows in validation.groupby(
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

        with h5py.File(
            h5_path,
            "r",
        ) as h5:

            for _, row in rows.iterrows():

                block_uid = str(
                    row["block_uid"]
                )

                block_index = int(
                    row["block_index"]
                )

                eye_state = str(
                    row["eye_state"]
                )

                if eye_state not in STATE_TO_LABEL:
                    raise RuntimeError(
                        f"Unknown eye state: "
                        f"{eye_state}"
                    )

                block = h5[
                    f"blocks/{block_uid}"
                ]

                # --------------------------------------------
                # Fail-closed Validation identity checks
                # --------------------------------------------

                h5_split = as_text(
                    block.attrs[
                        "provisional_split"
                    ]
                )

                if h5_split != "Validation":
                    raise RuntimeError(
                        f"Refusing non-Validation block: "
                        f"{block_uid} -> {h5_split}"
                    )

                if as_text(
                    block.attrs["subject_id"]
                ) != subject_id:
                    raise RuntimeError(
                        "Subject mismatch."
                    )

                if as_text(
                    block.attrs["state"]
                ) != eye_state:
                    raise RuntimeError(
                        "State mismatch."
                    )

                if not as_bool(
                    block.attrs[
                        "block_eligible"
                    ]
                ):
                    raise RuntimeError(
                        f"Ineligible Validation block: "
                        f"{block_uid}"
                    )

                starts = block[
                    "window_start_samples"
                ][:]

                usable = block[
                    "window_usable_mask"
                ][:]

                if len(starts) != len(usable):
                    raise RuntimeError(
                        "Validation window metadata "
                        "length mismatch."
                    )

                manifest_candidate = int(
                    row[
                        "candidate_windows"
                    ]
                )

                manifest_usable = int(
                    row[
                        "usable_windows"
                    ]
                )

                if len(starts) != manifest_candidate:
                    raise RuntimeError(
                        f"Candidate window mismatch: "
                        f"{block_uid}"
                    )

                if int(
                    usable.sum()
                ) != manifest_usable:
                    raise RuntimeError(
                        f"Usable window mismatch: "
                        f"{block_uid}"
                    )

                candidate_total += len(starts)
                usable_total += int(
                    usable.sum()
                )

                eeg = block["eeg_uv"]

                if eeg.shape[0] != 19:
                    raise RuntimeError(
                        f"Unexpected EEG shape: "
                        f"{eeg.shape}"
                    )

                for candidate_index in np.flatnonzero(
                    usable
                ):

                    candidate_index = int(
                        candidate_index
                    )

                    start = int(
                        starts[
                            candidate_index
                        ]
                    )

                    stop = (
                        start
                        + WINDOW_LENGTH
                    )

                    if stop > eeg.shape[1]:
                        raise RuntimeError(
                            "Validation window "
                            "exceeds EEG block."
                        )

                    records.append(
                        ValidationWindowRecord(
                            subject_id=subject_id,

                            subject_index=(
                                subject_to_index[
                                    subject_id
                                ]
                            ),

                            block_uid=block_uid,
                            block_index=block_index,

                            eye_state=eye_state,
                            label=STATE_TO_LABEL[
                                eye_state
                            ],

                            candidate_window_index=(
                                candidate_index
                            ),

                            start_sample=start,

                            h5_path=h5_path,
                        )
                    )

                    state_window_counts[
                        eye_state
                    ] += 1

    print(
        f"Validation blocks: "
        f"{len(validation)}"
    )

    print(
        f"Validation candidate windows: "
        f"{candidate_total}"
    )

    print(
        f"Validation usable windows: "
        f"{usable_total}"
    )

    print(
        f"EO usable windows: "
        f"{state_window_counts['EO']}"
    )

    print(
        f"EC usable windows: "
        f"{state_window_counts['EC']}"
    )

    print(
        f"subjects: "
        f"{len(subject_to_index)}"
    )

    if candidate_total != (
        EXPECTED_VALIDATION_BLOCKS
        * 25
    ):
        raise RuntimeError(
            f"Unexpected Validation candidate count: "
            f"{candidate_total}"
        )

    if len(records) != usable_total:
        raise RuntimeError(
            "Validation index size mismatch."
        )

    print(
        "\nVALIDATION INDEX PASS"
    )

    print(
        "Test EEG accessed while indexing: NO"
    )

    return (
        records,
        subject_to_index,
    )


# ============================================================
# Step 2 — Extract Validation log-bandpower features
# ============================================================

def extract_validation_features(
    records: list[
        ValidationWindowRecord
    ],
    feature_names_train: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    print("\n" + "=" * 80)
    print("STEP 2 — EXTRACT REAL VALIDATION LOG-BANDPOWER")
    print("=" * 80)

    channels = [
        name.split(
            "_",
            maxsplit=1,
        )[1]
        for name in feature_names_train[
            :19
        ]
    ]

    expected_feature_names = []

    for band_name in BANDS:

        for channel in channels:

            expected_feature_names.append(
                f"{band_name}_{channel}"
            )

    expected_feature_names = np.asarray(
        expected_feature_names,
        dtype="U64",
    )

    if not np.array_equal(
        feature_names_train.astype("U64"),
        expected_feature_names,
    ):
        raise RuntimeError(
            "Train feature definition "
            "does not match expected baseline."
        )

    n_windows = len(records)
    n_features = len(
        expected_feature_names
    )

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

    band_items = list(
        BANDS.items()
    )

    current_h5_path = None
    current_h5 = None

    i = 0
    block_counter = 0

    start_time = time.perf_counter()

    try:

        while i < n_windows:

            first = records[i]

            # --------------------------------------------
            # Contiguous records from same block
            # --------------------------------------------

            j = i + 1

            while (
                j < n_windows
                and records[j].subject_id
                == first.subject_id
                and records[j].block_uid
                == first.block_uid
            ):
                j += 1

            block_records = records[
                i:j
            ]

            # --------------------------------------------
            # Open subject HDF5
            # --------------------------------------------

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
                    f"Unexpected Validation "
                    f"window shape: "
                    f"{windows.shape}"
                )

            if not np.isfinite(
                windows
            ).all():
                raise RuntimeError(
                    "Nonfinite Validation EEG."
                )

            # --------------------------------------------
            # SAME Welch definition used for Train
            # --------------------------------------------

            freqs, psd = welch(
                windows,

                fs=SFREQ,

                window="hann",

                nperseg=(
                    WELCH_NPERSEG
                ),

                noverlap=(
                    WELCH_NOVERLAP
                ),

                detrend="constant",

                scaling="density",

                axis=-1,
            )

            feature_parts = []

            for (
                band_name,
                (low_hz, high_hz),
            ) in band_items:

                mask = (
                    (freqs >= low_hz)
                    & (freqs < high_hz)
                )

                if band_name == "beta":
                    mask = (
                        (freqs >= low_hz)
                        & (freqs <= high_hz)
                    )

                band_freqs = freqs[
                    mask
                ]

                power = np.trapezoid(
                    psd[
                        ...,
                        mask
                    ],
                    x=band_freqs,
                    axis=-1,
                )

                if np.any(
                    power <= 0
                ):
                    raise RuntimeError(
                        f"Non-positive "
                        f"{band_name} power."
                    )

                feature_parts.append(
                    np.log10(
                        np.maximum(
                            power,
                            EPS,
                        )
                    )
                )

            block_X = np.concatenate(
                feature_parts,
                axis=1,
            )

            if block_X.shape != (
                len(block_records),
                76,
            ):
                raise RuntimeError(
                    f"Unexpected Validation "
                    f"feature shape: "
                    f"{block_X.shape}"
                )

            X[
                i:j
            ] = block_X.astype(
                np.float32
            )

            y[
                i:j
            ] = first.label

            subject_index[
                i:j
            ] = first.subject_index

            block_index[
                i:j
            ] = first.block_index

            block_counter += 1

            if (
                block_counter == 1
                or block_counter % 100 == 0
                or j == n_windows
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"blocks={block_counter:>3}"
                    f" | windows={j:>5}/{n_windows}"
                    f" | elapsed={elapsed:>6.1f}s"
                )

            i = j

    finally:

        if current_h5 is not None:
            current_h5.close()

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Validation features contain NaN/Inf."
        )

    print(
        f"\nValidation X shape: "
        f"{X.shape}"
    )

    print(
        f"EO windows: "
        f"{int(np.sum(y == 0))}"
    )

    print(
        f"EC windows: "
        f"{int(np.sum(y == 1))}"
    )

    print(
        "Test EEG accessed during "
        "feature extraction: NO"
    )

    return (
        X,
        y,
        subject_index,
        block_index,
    )


# ============================================================
# Step 3 — Subject-macro Balanced Accuracy
# ============================================================

def subject_macro_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_index: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
]:

    subject_scores = []

    subjects = np.unique(
        subject_index
    )

    if len(subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} "
            f"Validation subjects, "
            f"found {len(subjects)}"
        )

    for subject in subjects:

        mask = (
            subject_index
            == subject
        )

        subject_y = y_true[
            mask
        ]

        subject_pred = y_pred[
            mask
        ]

        # Every subject must contain both EO and EC.
        if len(
            np.unique(
                subject_y
            )
        ) != 2:
            raise RuntimeError(
                f"Subject {subject} "
                f"does not contain both states."
            )

        score = balanced_accuracy_score(
            subject_y,
            subject_pred,
        )

        subject_scores.append(
            float(score)
        )

    subject_scores = np.asarray(
        subject_scores,
        dtype=np.float64,
    )

    return (
        float(
            subject_scores.mean()
        ),
        subject_scores,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — LOG-BANDPOWER + LOGISTIC REGRESSION")
    print("=" * 80)

    print(
        "\nTHIS IS THE FIRST CONTROLLED "
        "SCIENTIFIC VALIDATION ACCESS."
    )

    print(
        "\nPredeclared baseline:"
        f"\nC = {LOGREG_C}"
        f"\npenalty = {LOGREG_PENALTY}"
        f"\nsolver = {LOGREG_SOLVER}"
        f"\nmax_iter = {LOGREG_MAX_ITER}"
        "\nfeature scaler fit = TRAIN ONLY"
        "\nclassifier fit = TRAIN ONLY"
        "\nevaluation = REAL VALIDATION"
        "\nFinal Test = SEALED"
    )

    print(
        f"\nscikit-learn: "
        f"{sklearn_version}"
    )

    # ========================================================
    # Step 1 — Load Train feature package
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — LOAD REAL TRAIN FEATURES")
    print("=" * 80)

    train_package = np.load(
        TRAIN_FEATURES_FILE,
        allow_pickle=False,
    )

    X_train = np.asarray(
        train_package["X"],
        dtype=np.float64,
    )

    y_train = np.asarray(
        train_package["y"],
        dtype=np.int64,
    )

    train_subject_index = np.asarray(
        train_package[
            "subject_index"
        ],
        dtype=np.int64,
    )

    feature_names = np.asarray(
        train_package[
            "feature_names"
        ],
    )

    if X_train.shape != (
        47_835,
        76,
    ):
        raise RuntimeError(
            f"Unexpected Train shape: "
            f"{X_train.shape}"
        )

    if not np.isfinite(
        X_train
    ).all():
        raise RuntimeError(
            "Train features contain NaN/Inf."
        )

    print(
        f"Train X: "
        f"{X_train.shape}"
    )

    print(
        f"Train EO: "
        f"{int(np.sum(y_train == 0))}"
    )

    print(
        f"Train EC: "
        f"{int(np.sum(y_train == 1))}"
    )

    # ========================================================
    # Step 2 — Build controlled Validation
    # ========================================================

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    validation_records, subject_mapping = (
        build_validation_window_index(
            split
        )
    )

    X_val, y_val, val_subject_index, val_block_index = (
        extract_validation_features(
            validation_records,
            feature_names,
        )
    )

    # ========================================================
    # Step 3 — Save Validation features
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — SAVE VALIDATION FEATURE PACKAGE")
    print("=" * 80)

    VALIDATION_FEATURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        VALIDATION_FEATURES_FILE,

        X=X_val.astype(
            np.float32
        ),

        y=y_val,

        subject_index=(
            val_subject_index
        ),

        block_index=(
            val_block_index
        ),

        feature_names=(
            feature_names.astype(
                "U64"
            )
        ),
    )

    print(
        f"saved: "
        f"{VALIDATION_FEATURES_FILE}"
    )

    print(
        f"size MiB: "
        f"{VALIDATION_FEATURES_FILE.stat().st_size / 1024**2:.2f}"
    )

    # ========================================================
    # Step 4 — TRAIN-ONLY feature standardization
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — FIT FEATURE SCALER ON TRAIN ONLY")
    print("=" * 80)

    feature_scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    X_train_scaled = (
        feature_scaler.fit_transform(
            X_train
        )
    )

    X_val_scaled = (
        feature_scaler.transform(
            X_val
        )
    )

    if not np.isfinite(
        X_train_scaled
    ).all():
        raise RuntimeError(
            "Nonfinite scaled Train features."
        )

    if not np.isfinite(
        X_val_scaled
    ).all():
        raise RuntimeError(
            "Nonfinite scaled Validation features."
        )

    print(
        f"max |Train feature mean|: "
        f"{np.abs(X_train_scaled.mean(axis=0)).max():.3e}"
    )

    print(
        f"max |Train feature std-1|: "
        f"{np.abs(X_train_scaled.std(axis=0) - 1).max():.3e}"
    )

    print(
        "Validation statistics used for fitting: NO"
    )

    # ========================================================
    # Step 5 — Fit Logistic Regression
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — FIT LOGISTIC REGRESSION ON TRAIN")
    print("=" * 80)

    model = LogisticRegression(
        C=LOGREG_C,

        penalty=LOGREG_PENALTY,

        solver=LOGREG_SOLVER,

        max_iter=LOGREG_MAX_ITER,

        class_weight=None,

        fit_intercept=True,

        random_state=20260907,
    )

    fit_start = time.perf_counter()

    model.fit(
        X_train_scaled,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    print(
        f"fit seconds: "
        f"{fit_seconds:.3f}"
    )

    print(
        f"iterations: "
        f"{model.n_iter_.tolist()}"
    )

    # ========================================================
    # Step 6 — Real Validation predictions
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6 — EVALUATE REAL VALIDATION")
    print("=" * 80)

    y_pred = model.predict(
        X_val_scaled
    )

    y_prob = model.predict_proba(
        X_val_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Window-level secondary metrics
    # --------------------------------------------------------

    window_accuracy = accuracy_score(
        y_val,
        y_pred,
    )

    window_balanced_accuracy = (
        balanced_accuracy_score(
            y_val,
            y_pred,
        )
    )

    window_macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
    )

    window_roc_auc = roc_auc_score(
        y_val,
        y_prob,
    )

    window_mcc = matthews_corrcoef(
        y_val,
        y_pred,
    )

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=[0, 1],
    )

    # --------------------------------------------------------
    # PRIMARY metric:
    # subject-macro Balanced Accuracy
    # --------------------------------------------------------

    (
        subject_macro_ba,
        subject_scores,
    ) = subject_macro_balanced_accuracy(
        y_val,
        y_pred,
        val_subject_index,
    )

    print(
        "\nPRIMARY METRIC"
    )

    print(
        f"Subject-macro Balanced Accuracy: "
        f"{subject_macro_ba:.6f}"
    )

    print(
        "\nSubject BA distribution:"
    )

    print(
        f"  min    = "
        f"{subject_scores.min():.6f}"
    )

    print(
        f"  Q1     = "
        f"{np.percentile(subject_scores, 25):.6f}"
    )

    print(
        f"  median = "
        f"{np.median(subject_scores):.6f}"
    )

    print(
        f"  Q3     = "
        f"{np.percentile(subject_scores, 75):.6f}"
    )

    print(
        f"  max    = "
        f"{subject_scores.max():.6f}"
    )

    print(
        "\nWINDOW-LEVEL SECONDARY METRICS"
    )

    print(
        f"Accuracy:          "
        f"{window_accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{window_balanced_accuracy:.6f}"
    )

    print(
        f"Macro-F1:          "
        f"{window_macro_f1:.6f}"
    )

    print(
        f"ROC-AUC:           "
        f"{window_roc_auc:.6f}"
    )

    print(
        f"MCC:               "
        f"{window_mcc:.6f}"
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        "rows = true [EO, EC]"
    )

    print(
        "cols = predicted [EO, EC]"
    )

    print(cm)

    # ========================================================
    # Step 7 — Inspect strongest coefficients
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 — COEFFICIENT SANITY")
    print("=" * 80)

    coefficients = (
        model.coef_[0]
    )

    abs_order = np.argsort(
        np.abs(coefficients)
    )[::-1]

    print(
        "\nTop 15 standardized coefficients:"
    )

    for index in abs_order[:15]:

        print(
            f"{feature_names[index]:<20}"
            f"{coefficients[index]:>12.6f}"
        )

    # ========================================================
    # Step 8 — Save result
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 8 — SAVE VALIDATION BASELINE RESULT")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "status": (
            "REAL_VALIDATION_BASELINE_COMPLETE"
        ),

        "model": (
            "logbandpower_logistic_regression"
        ),

        "feature_definition": {
            "bands_hz": {
                name: [
                    float(low),
                    float(high),
                ]
                for (
                    name,
                    (low, high),
                ) in BANDS.items()
            },

            "channels": 19,
            "feature_count": 76,

            "welch_nperseg": (
                WELCH_NPERSEG
            ),

            "welch_noverlap": (
                WELCH_NOVERLAP
            ),

            "physical_input_unit": "uV",
        },

        "classifier": {
            "C": LOGREG_C,
            "penalty": (
                LOGREG_PENALTY
            ),
            "solver": (
                LOGREG_SOLVER
            ),
            "max_iter": (
                LOGREG_MAX_ITER
            ),
            "feature_standardization": (
                "StandardScaler fit Train only"
            ),
        },

        "train": {
            "windows": int(
                len(y_train)
            ),

            "subjects": int(
                len(
                    np.unique(
                        train_subject_index
                    )
                )
            ),
        },

        "validation": {
            "windows": int(
                len(y_val)
            ),

            "subjects": int(
                len(
                    np.unique(
                        val_subject_index
                    )
                )
            ),

            "eo_windows": int(
                np.sum(
                    y_val == 0
                )
            ),

            "ec_windows": int(
                np.sum(
                    y_val == 1
                )
            ),
        },

        "primary_metric": {
            "name": (
                "subject_macro_balanced_accuracy"
            ),

            "value": float(
                subject_macro_ba
            ),
        },

        "subject_balanced_accuracy": {
            "minimum": float(
                subject_scores.min()
            ),

            "q1": float(
                np.percentile(
                    subject_scores,
                    25,
                )
            ),

            "median": float(
                np.median(
                    subject_scores
                )
            ),

            "q3": float(
                np.percentile(
                    subject_scores,
                    75,
                )
            ),

            "maximum": float(
                subject_scores.max()
            ),
        },

        "window_metrics": {
            "accuracy": float(
                window_accuracy
            ),

            "balanced_accuracy": float(
                window_balanced_accuracy
            ),

            "macro_f1": float(
                window_macro_f1
            ),

            "roc_auc": float(
                window_roc_auc
            ),

            "mcc": float(
                window_mcc
            ),
        },

        "confusion_matrix": (
            cm.astype(int).tolist()
        ),

        "final_test_accessed": False,

        "scientific_validation_opened": True,

        "sklearn_version": (
            sklearn_version
        ),
    }

    with RESULT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print(
        f"saved: "
        f"{RESULT_FILE}"
    )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 80)
    print("LOG-BANDPOWER LOGISTIC REGRESSION BASELINE COMPLETE")
    print("=" * 80)

    print(
        f"PRIMARY subject-macro BA: "
        f"{subject_macro_ba:.6f}"
    )

    print(
        f"Validation windows: "
        f"{len(y_val)}"
    )

    print(
        "Feature scaler fit on Validation: NO"
    )

    print(
        "Classifier fit on Validation: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "INTERPRET BASELINE RESULTS "
        "BEFORE ANY TUNING"
    )


if __name__ == "__main__":
    main()