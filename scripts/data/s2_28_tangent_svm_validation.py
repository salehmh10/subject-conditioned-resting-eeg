from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    WINDOW_LENGTH,
    build_train_window_index,
)

from s2_27_logbandpower_logreg_validation import (
    build_validation_window_index,
    subject_macro_balanced_accuracy,
)


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "real_tangent_linear_svm_validation_v1.json"
)


# ============================================================
# Predeclared baseline
# ============================================================

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_SUBJECTS = 198

SVM_C = 1.0
SVM_MAX_ITER = 10_000

COVARIANCE_ESTIMATOR = "oas"
TANGENT_METRIC = "riemann"


# ============================================================
# Utility
# ============================================================

def load_locked_scaler() -> dict:

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError(
            "Normalization is not locked."
        )

    return scaler


# ============================================================
# Window -> covariance
# ============================================================

def compute_covariances(
    records,
    scaler: dict,
    label: str,
):

    print("\n" + "=" * 80)
    print(f"COMPUTE {label} COVARIANCE MATRICES")
    print("=" * 80)

    means = np.asarray(
        scaler["means_uv"],
        dtype=np.float64,
    )

    stds = np.asarray(
        scaler["stds_uv"],
        dtype=np.float64,
    )

    n = len(records)

    covariances = np.empty(
        (n, 19, 19),
        dtype=np.float64,
    )

    y = np.empty(
        n,
        dtype=np.int64,
    )

    subject_index = np.empty(
        n,
        dtype=np.int64,
    )

    estimator = Covariances(
        estimator=COVARIANCE_ESTIMATOR
    )

    current_h5_path = None
    current_h5 = None

    i = 0
    block_counter = 0

    start_time = time.perf_counter()

    try:

        while i < n:

            first = records[i]

            j = i + 1

            while (
                j < n
                and records[j].subject_id
                == first.subject_id
                and records[j].block_uid
                == first.block_uid
            ):
                j += 1

            block_records = records[i:j]

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

            windows = np.stack(
                [
                    np.asarray(
                        eeg[
                            :,
                            r.start_sample:
                            r.start_sample + WINDOW_LENGTH
                        ],
                        dtype=np.float64,
                    )
                    for r in block_records
                ],
                axis=0,
            )

            if windows.shape[1:] != (19, 512):
                raise RuntimeError(
                    f"Unexpected EEG shape: "
                    f"{windows.shape}"
                )

            if not np.isfinite(windows).all():
                raise RuntimeError(
                    "Nonfinite EEG."
                )

            # ------------------------------------------------
            # Apply the ONE frozen Train-only scaler.
            #
            # Important:
            # no per-window normalization.
            # ------------------------------------------------

            windows = (
                (
                    windows
                    - means[
                        None,
                        :,
                        None,
                    ]
                )
                / stds[
                    None,
                    :,
                    None,
                ]
            )

            if not np.isfinite(windows).all():
                raise RuntimeError(
                    "Nonfinite normalized EEG."
                )

            block_cov = estimator.transform(
                windows
            )

            if block_cov.shape != (
                len(block_records),
                19,
                19,
            ):
                raise RuntimeError(
                    f"Unexpected covariance shape: "
                    f"{block_cov.shape}"
                )

            if not np.isfinite(block_cov).all():
                raise RuntimeError(
                    "Nonfinite covariance matrix."
                )

            covariances[i:j] = block_cov

            y[i:j] = first.label

            subject_index[i:j] = (
                first.subject_index
            )

            block_counter += 1

            if (
                block_counter == 1
                or block_counter % 200 == 0
                or j == n
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"blocks={block_counter:>4}"
                    f" | windows={j:>6}/{n}"
                    f" | elapsed={elapsed:>7.1f}s"
                )

            i = j

    finally:

        if current_h5 is not None:
            current_h5.close()

    print(
        f"\n{label} covariance shape: "
        f"{covariances.shape}"
    )

    return (
        covariances,
        y,
        subject_index,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print(
        "SCENARIO 2 — "
        "TANGENT SPACE + LINEAR SVM"
    )
    print("=" * 80)

    print(
        "\nPredeclared baseline:"
        f"\nCovariance estimator = {COVARIANCE_ESTIMATOR}"
        f"\nTangent metric = {TANGENT_METRIC}"
        f"\nLinear SVM C = {SVM_C}"
        "\nNormalization = locked global Train-only"
        "\nTangent reference fit = Train only"
        "\nClassifier fit = Train only"
        "\nEvaluation = Real Validation"
        "\nFinal Test = SEALED"
    )

    # ========================================================
    # Step 1 — Frozen inputs
    # ========================================================

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    scaler = load_locked_scaler()

    train_records, train_subject_map = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    if len(train_records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            f"Unexpected Train count: "
            f"{len(train_records)}"
        )

    validation_records, val_subject_map = (
        build_validation_window_index(
            split
        )
    )

    if train_subject_map != val_subject_map:
        raise RuntimeError(
            "Train/Validation subject mapping mismatch."
        )

    # ========================================================
    # Step 2 — Covariance matrices
    # ========================================================

    train_cov, y_train, train_subject = (
        compute_covariances(
            train_records,
            scaler,
            "TRAIN",
        )
    )

    val_cov, y_val, val_subject = (
        compute_covariances(
            validation_records,
            scaler,
            "VALIDATION",
        )
    )

    print(
        "\nTrain covariance matrices:",
        train_cov.shape,
    )

    print(
        "Validation covariance matrices:",
        val_cov.shape,
    )

    # ========================================================
    # Step 3 — Tangent Space
    # ========================================================

    print("\n" + "=" * 80)
    print("FIT TANGENT SPACE ON TRAIN ONLY")
    print("=" * 80)

    tangent = TangentSpace(
        metric=TANGENT_METRIC
    )

    tangent_start = (
        time.perf_counter()
    )

    X_train = tangent.fit_transform(
        train_cov
    )

    tangent_fit_seconds = (
        time.perf_counter()
        - tangent_start
    )

    X_val = tangent.transform(
        val_cov
    )

    print(
        f"Train tangent shape: "
        f"{X_train.shape}"
    )

    print(
        f"Validation tangent shape: "
        f"{X_val.shape}"
    )

    print(
        f"Tangent fit/Train transform seconds: "
        f"{tangent_fit_seconds:.2f}"
    )

    if X_train.shape[1] != 190:
        raise RuntimeError(
            f"Expected 190 tangent features, "
            f"got {X_train.shape[1]}"
        )

    if not np.isfinite(X_train).all():
        raise RuntimeError(
            "Nonfinite Train tangent features."
        )

    if not np.isfinite(X_val).all():
        raise RuntimeError(
            "Nonfinite Validation tangent features."
        )

    print(
        "Validation used to fit tangent reference: NO"
    )

    # ========================================================
    # Step 4 — Linear SVM
    # ========================================================

    print("\n" + "=" * 80)
    print("FIT LINEAR SVM ON TRAIN")
    print("=" * 80)

    model = LinearSVC(
        C=SVM_C,
        max_iter=SVM_MAX_ITER,
        class_weight=None,
        random_state=20260907,
    )

    svm_start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    svm_seconds = (
        time.perf_counter()
        - svm_start
    )

    print(
        f"SVM fit seconds: "
        f"{svm_seconds:.3f}"
    )

    print(
        f"iterations: "
        f"{model.n_iter_}"
    )

    # ========================================================
    # Step 5 — Validation
    # ========================================================

    print("\n" + "=" * 80)
    print("EVALUATE REAL VALIDATION")
    print("=" * 80)

    y_pred = model.predict(
        X_val
    )

    decision = model.decision_function(
        X_val
    )

    subject_macro_ba, subject_scores = (
        subject_macro_balanced_accuracy(
            y_val,
            y_pred,
            val_subject,
        )
    )

    accuracy = accuracy_score(
        y_val,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_val,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
    )

    roc_auc = roc_auc_score(
        y_val,
        decision,
    )

    mcc = matthews_corrcoef(
        y_val,
        y_pred,
    )

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=[0, 1],
    )

    # ========================================================
    # Results
    # ========================================================

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
        "\nWINDOW METRICS"
    )

    print(
        f"Accuracy:          {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.6f}"
    )

    print(
        f"Macro-F1:          {macro_f1:.6f}"
    )

    print(
        f"ROC-AUC:           {roc_auc:.6f}"
    )

    print(
        f"MCC:               {mcc:.6f}"
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(cm)

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "status": (
            "REAL_VALIDATION_BASELINE_COMPLETE"
        ),

        "model": (
            "oas_covariance_riemann_tangent_linear_svm"
        ),

        "covariance_estimator": (
            COVARIANCE_ESTIMATOR
        ),

        "tangent_metric": (
            TANGENT_METRIC
        ),

        "classifier": {
            "type": "LinearSVC",
            "C": SVM_C,
            "max_iter": SVM_MAX_ITER,
        },

        "train_windows": int(
            len(y_train)
        ),

        "validation_windows": int(
            len(y_val)
        ),

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
                accuracy
            ),

            "balanced_accuracy": float(
                balanced_accuracy
            ),

            "macro_f1": float(
                macro_f1
            ),

            "roc_auc": float(
                roc_auc
            ),

            "mcc": float(
                mcc
            ),
        },

        "confusion_matrix": (
            cm.astype(int).tolist()
        ),

        "normalization": (
            "locked_global_train_only"
        ),

        "tangent_reference_fit": (
            "Train_only"
        ),

        "classifier_fit": (
            "Train_only"
        ),

        "final_test_accessed": False,
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
        f"\nsaved: {RESULT_FILE}"
    )

    print("\n" + "=" * 80)
    print(
        "TANGENT SPACE + LINEAR SVM "
        "BASELINE COMPLETE"
    )
    print("=" * 80)

    print(
        f"PRIMARY subject-macro BA: "
        f"{subject_macro_ba:.6f}"
    )

    print(
        "Final Test accessed: NO"
    )


if __name__ == "__main__":
    main()