from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

METRIC_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
)

LOGREG_FILE = (
    METRIC_DIR
    / "real_logbandpower_logreg_validation_v1.json"
)

TANGENT_FILE = (
    METRIC_DIR
    / "real_tangent_linear_svm_validation_v1.json"
)

EEGNET_FILE = (
    METRIC_DIR
    / "real_eegnetv4_validation_v1.json"
)

OUTPUT_FILE = (
    METRIC_DIR
    / "real_classifier_baseline_summary_v1.json"
)


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def extract_metrics(
    name: str,
    result: dict,
) -> dict:

    if result.get(
        "final_test_accessed"
    ) is not False:
        raise RuntimeError(
            f"{name}: Final Test access invariant failed."
        )

    primary = result[
        "primary_metric"
    ]

    if primary[
        "name"
    ] != "subject_macro_balanced_accuracy":
        raise RuntimeError(
            f"{name}: unexpected primary metric."
        )

    window = result[
        "window_metrics"
    ]

    subject_dist = result[
        "subject_balanced_accuracy"
    ]

    return {
        "name": name,

        "subject_macro_balanced_accuracy": float(
            primary["value"]
        ),

        "window_accuracy": float(
            window["accuracy"]
        ),

        "window_balanced_accuracy": float(
            window["balanced_accuracy"]
        ),

        "window_macro_f1": float(
            window["macro_f1"]
        ),

        "window_roc_auc": float(
            window["roc_auc"]
        ),

        "window_mcc": float(
            window["mcc"]
        ),

        "subject_ba_min": float(
            subject_dist["minimum"]
        ),

        "subject_ba_q1": float(
            subject_dist["q1"]
        ),

        "subject_ba_median": float(
            subject_dist["median"]
        ),

        "subject_ba_q3": float(
            subject_dist["q3"]
        ),

        "subject_ba_max": float(
            subject_dist["maximum"]
        ),

        "confusion_matrix": result[
            "confusion_matrix"
        ],
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — REAL CLASSIFIER BASELINE COMPARISON")
    print("=" * 80)

    # --------------------------------------------------------
    # Step 1 — Read frozen outputs
    # --------------------------------------------------------

    logreg_raw = load_json(
        LOGREG_FILE
    )

    tangent_raw = load_json(
        TANGENT_FILE
    )

    eegnet_raw = load_json(
        EEGNET_FILE
    )

    logreg = extract_metrics(
        "Log-bandpower + Logistic Regression",
        logreg_raw,
    )

    tangent = extract_metrics(
        "Tangent Space + Linear SVM",
        tangent_raw,
    )

    eegnet = extract_metrics(
        "EEGNet baseline",
        eegnet_raw,
    )

    results = [
        logreg,
        tangent,
        eegnet,
    ]

    # --------------------------------------------------------
    # Step 2 — Print comparison
    # --------------------------------------------------------

    print("\nPRIMARY METRIC")
    print(
        "Subject-macro Balanced Accuracy"
    )

    print("-" * 80)

    ranked = sorted(
        results,
        key=lambda x:
        x[
            "subject_macro_balanced_accuracy"
        ],
        reverse=True,
    )

    for rank, result in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result['name']:<42}"
            f"{result['subject_macro_balanced_accuracy']:.6f}"
        )

    print("\n" + "=" * 80)
    print("SECONDARY WINDOW METRICS")
    print("=" * 80)

    header = (
        f"{'Model':<38}"
        f"{'BA':>9}"
        f"{'Acc':>9}"
        f"{'F1':>9}"
        f"{'AUC':>9}"
        f"{'MCC':>9}"
    )

    print(header)
    print("-" * len(header))

    for result in ranked:

        print(
            f"{result['name']:<38}"
            f"{result['window_balanced_accuracy']:>9.4f}"
            f"{result['window_accuracy']:>9.4f}"
            f"{result['window_macro_f1']:>9.4f}"
            f"{result['window_roc_auc']:>9.4f}"
            f"{result['window_mcc']:>9.4f}"
        )

    # --------------------------------------------------------
    # Step 3 — Subject-level distribution
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUBJECT BALANCED-ACCURACY DISTRIBUTION")
    print("=" * 80)

    header = (
        f"{'Model':<38}"
        f"{'Min':>8}"
        f"{'Q1':>8}"
        f"{'Median':>10}"
        f"{'Q3':>8}"
        f"{'Max':>8}"
    )

    print(header)
    print("-" * len(header))

    for result in ranked:

        print(
            f"{result['name']:<38}"
            f"{result['subject_ba_min']:>8.3f}"
            f"{result['subject_ba_q1']:>8.3f}"
            f"{result['subject_ba_median']:>10.3f}"
            f"{result['subject_ba_q3']:>8.3f}"
            f"{result['subject_ba_max']:>8.3f}"
        )

    # --------------------------------------------------------
    # Step 4 — EEGNet training metadata
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EEGNET TRAINING SUMMARY")
    print("=" * 80)

    training = eegnet_raw[
        "training"
    ]

    print(
        f"best epoch: "
        f"{training['best_epoch']}"
    )

    print(
        f"training seconds: "
        f"{training['training_seconds']:.2f}"
    )

    print(
        f"training minutes: "
        f"{training['training_seconds'] / 60:.2f}"
    )

    print(
        f"batch size: "
        f"{training['batch_size']}"
    )

    print(
        f"learning rate: "
        f"{training['learning_rate']}"
    )

    # --------------------------------------------------------
    # Step 5 — Verify common study boundary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STUDY-BOUNDARY CHECK")
    print("=" * 80)

    for (
        name,
        raw,
    ) in [
        ("LogReg", logreg_raw),
        ("Tangent-SVM", tangent_raw),
        ("EEGNet", eegnet_raw),
    ]:

        if raw.get(
            "final_test_accessed"
        ) is not False:
            raise RuntimeError(
                f"{name}: Final Test was accessed."
            )

        print(
            f"{name:<14} Final Test accessed: NO"
        )

    # --------------------------------------------------------
    # Step 6 — Save compact summary
    # --------------------------------------------------------

    output = {
        "status": (
            "REAL_CLASSIFIER_BASELINES_COMPLETE"
        ),

        "evaluation_partition": (
            "Real Validation"
        ),

        "primary_metric": (
            "subject_macro_balanced_accuracy"
        ),

        "ranking": [
            result["name"]
            for result in ranked
        ],

        "models": results,

        "best_validation_baseline": {
            "name": ranked[0]["name"],

            "subject_macro_balanced_accuracy": (
                ranked[0][
                    "subject_macro_balanced_accuracy"
                ]
            ),
        },

        "all_baselines_untuned": True,

        "final_test_accessed": False,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print("\n" + "=" * 80)
    print("REAL CLASSIFIER BASELINES COMPLETE")
    print("=" * 80)

    print(
        f"best current baseline: "
        f"{ranked[0]['name']}"
    )

    print(
        f"subject-macro BA: "
        f"{ranked[0]['subject_macro_balanced_accuracy']:.6f}"
    )

    print(
        f"\nsaved: {OUTPUT_FILE}"
    )

    print(
        "\nAll three baselines remain UNTUNED."
    )

    print(
        "Final Test remains SEALED."
    )

    print(
        "\nNEXT:"
    )

    print(
        "PREPARE B0 — STATE-ONLY DDPM"
    )


if __name__ == "__main__":
    main()