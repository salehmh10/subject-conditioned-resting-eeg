"""Strict presentation-data validation; never runs experiments or hypothesis tests."""
from __future__ import annotations

from collections import Counter
import math
import statistics

from .sources import EvidenceError

CONDITIONS = ["REAL_ONLY", "REAL_DUPLICATE_1X_CONTROL", "TSTR_B0_1X", "TSTR_B1_1X",
              "TSTR_B2_1X", "REAL_PLUS_B0_0P5X", "REAL_PLUS_B1_0P5X", "REAL_PLUS_B2_0P5X",
              "REAL_PLUS_B0_1X", "REAL_PLUS_B1_1X", "REAL_PLUS_B2_1X"]
FAMILIES = ["LOGBANDPOWER_LOGREG", "OAS_RIEMANN_LINEAR_SVM", "EEGNETV4"]


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def assert_unique(rows, keys, label="records"):
    seen = set()
    for row in rows:
        require(all(key in row and row[key] is not None for key in keys),
                f"Missing identity in {label}: {keys}")
        identity = tuple(row[key] for key in keys)
        require(identity not in seen, f"Duplicate {label} identity: {identity}")
        seen.add(identity)


def fraction_to_percent(value):
    return None if value is None else value * 100.0


def normalize_contrast(delta, low, high, source_direction):
    require(low <= high, "Confidence interval endpoints are reversed in source")
    if source_direction == "left_minus_right":
        return delta, low, high
    if source_direction == "right_minus_left":
        return -delta, -high, -low
    raise EvidenceError(f"Unknown contrast direction: {source_direction}")


def validate_bundle(bundle):
    require(bundle["schema_version"] == 1, "Unsupported presentation bundle version")
    rows = bundle["utility_runs"]
    assert_unique(rows, ("split", "family", "condition", "seed"), "utility fits")
    summary = bundle["utility_summary"]
    assert_unique(summary, ("split", "family", "condition"), "utility summaries")
    for split in ("validation", "final_test"):
        subset = [r for r in rows if r["split"] == split]
        require(len(subset) == 55, f"{split}: expected all 55 final v2 fits")
        sums = [r for r in summary if r["split"] == split]
        require(len(sums) == 33, f"{split}: expected 33 classifier-condition summaries")
        for family in FAMILIES:
            family_rows = [r for r in subset if r["family"] == family]
            counts = Counter(r["condition"] for r in family_rows)
            expected = 3 if family == "EEGNETV4" else 1
            require(set(counts) == set(CONDITIONS), f"Incomplete condition coverage: {split}/{family}")
            require(set(counts.values()) == {expected}, f"Wrong seed count: {split}/{family}")
        for row in sums:
            fits = [r for r in subset if r["family"] == row["family"] and r["condition"] == row["condition"]]
            values = [r["subject_macro_balanced_accuracy"] for r in fits]
            require(all(0 <= value <= 1 for value in values), "Utility fractions must lie in [0, 1]")
            require(math.isclose(statistics.mean(values), row["mean"], abs_tol=1e-12), "Saved mean conflicts with fits")
            require(math.isclose(statistics.pstdev(values), row["population_sd"], abs_tol=1e-12),
                    "Saved population SD conflicts with ddof=0 seed values")
    for row in bundle["contrasts"]:
        means = {r["condition"]: r["mean"] for r in summary if r["split"] == row["split"] and r["family"] == row["family"]}
        require(math.isclose(row["delta"], means[row["left"]] - means[row["right"]], abs_tol=1e-12),
                "Contrast direction conflicts with saved condition means")
        require(row["ci_low"] <= row["ci_high"], "Invalid contrast confidence interval")
    assert_unique(bundle["histories"], ("condition", "seed"), "learning histories")
    require(len(bundle["histories"]) == 33, "Expected all 33 saved EEGNet learning histories")
    for row in bundle["histories"]:
        epochs = row["epochs"]
        require([e["epoch"] for e in epochs] == list(range(1, row["stopped_epoch"] + 1)),
                "Learning history does not cover actual stopping epoch")
        require(1 <= row["best_epoch"] <= row["stopped_epoch"] <= 120, "Invalid selected/stopping epoch")
    assert_unique(bundle["fidelity_metrics"], ("metric", "model"), "fidelity summaries")
    require(len(bundle["fidelity_metrics"]) == 45 * 3, "Expected 45 metrics for all three generators")
    assert_unique(bundle["fidelity_subjects"], ("subject_id", "state", "model"), "participant Fidelity")
    assert_unique(bundle["fingerprint_subjects"], ("subject_id", "state", "model"), "fingerprint queries")
    n = bundle["metadata"]["n_subjects"]
    require(len(bundle["psd"]["participants"]) == n, "PSD participant coverage mismatch")
    require(len(bundle["connectivity"]["participants"]) == n, "Connectivity participant coverage mismatch")
    require(len(bundle["fidelity_subjects"]) == n * 2 * 3, "Incomplete participant Fidelity records")
    for record in bundle["provenance"]["sources"].values():
        path = record["path"]
        require(not path.startswith(("/", "\\")) and ":" not in path, "Personal/absolute source path in bundle")
        if "/utility/" in path and record["role"] != "study metadata; no utility performance":
            require("eleven_condition_s2_75_v2_120epoch_es/" in path, "Old utility evidence is forbidden")
    return {"status": "passed", "fits_per_split": 55, "summaries_per_split": 33,
            "fidelity_metrics_per_generator": 45, "histories": 33, "participants": n,
            "mean_sd_convention": "saved population SD, ddof=0"}


def acceptance_anchors(bundle):
    """User-provided anchors are checks only; they never supply displayed evidence."""
    checks = []
    def check(name, actual, expected, tolerance):
        checks.append({"name": name, "actual": actual, "expected": expected,
                       "tolerance": tolerance, "passed": abs(actual - expected) <= tolerance})
    for model, expected in zip(("B0", "B1", "B2"), (3.5732, 1.1560, 0.9389)):
        value = next(r["mean"] for r in bundle["fidelity_metrics"] if r["metric"] == "log_psd_mae_db" and r["model"] == model)
        check(f"Log-PSD MAE {model}", value, expected, 0.00005)
    for condition, expected in (("REAL_ONLY", 88.417), ("REAL_PLUS_B1_1X", 88.983), ("REAL_PLUS_B2_1X", 89.135)):
        value = next(r["mean"] for r in bundle["utility_summary"] if r["split"] == "final_test" and r["family"] == "EEGNETV4" and r["condition"] == condition)
        check(f"EEGNet Final Test {condition} (%)", 100 * value, expected, 0.0005)
    for right, anchor in (("REAL_ONLY", (0.718, 0.250, 1.201)), ("REAL_PLUS_B1_1X", (0.152, -0.276, 0.570))):
        row = next(r for r in bundle["contrasts"] if r["split"] == "final_test" and r["family"] == "EEGNETV4" and r["left"] == "REAL_PLUS_B2_1X" and r["right"] == right)
        for field, expected in zip(("delta", "ci_low", "ci_high"), anchor):
            check(f"EEGNet B2 minus {right} {field} (pp)", row[field] * 100, expected, 0.0006)
    for state, values in (("EO", (0.51, 75.25, 71.72)), ("EC", (1.01, 84.34, 95.96))):
        for model, expected in zip(("B0", "B1", "B2"), values):
            value = next(r["top1"] for r in bundle["fingerprint"] if r["state"] == state and r["model"] == model)
            check(f"{state} {model} fingerprint Top-1 (%)", value * 100, expected, 0.0051)
    require(all(c["passed"] for c in checks), "Numerical acceptance anchor mismatch; investigate saved evidence")
    return checks
