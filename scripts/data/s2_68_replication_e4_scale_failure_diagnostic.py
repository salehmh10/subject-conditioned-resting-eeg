from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from s2_23_check_real_train_dataset import NORMALIZATION_FILE
from s2_50b_b1_raw_candidate_subject_aware_fidelity_sweep_v2 import compute_candidate_metrics
from s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep import (
    evaluate_eligibility,
    full_log_psd_mae,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

S2_66_ROOT = (
    REPO_ROOT / "outputs" / "protocols" / "scenario2" / "b2"
    / "validation_checkpoint_selector_v1"
)

S2_67_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "fresh_seed_validation_checkpoint_selection_v1"
)

VAL_DEV_REAL_NPZ = S2_66_ROOT / "references" / "s2_66_val_dev_real_reference_v1.npz"
VAL_DEV_B1_NPZ = S2_66_ROOT / "generated" / "s2_66_val_dev_b1_synthetic_v1.npz"
VAL_CONF_REAL_NPZ = S2_66_ROOT / "references" / "s2_66_val_conf_real_reference_v1.npz"
VAL_CONF_B1_NPZ = S2_66_ROOT / "generated" / "s2_66_val_conf_b1_synthetic_v1.npz"

S2_67_SUMMARY = S2_67_ROOT / "metrics" / "s2_67_fresh_seed_selector_summary_v1.json"
PRIMARY_E1_DEV_NPZ = (
    S2_67_ROOT / "generated" / "val_dev"
    / "s2_67_primary_epoch_001_val_dev_synthetic_v1.npz"
)
REPLICATION_E4_DEV_NPZ = (
    S2_67_ROOT / "generated" / "val_dev"
    / "s2_67_replication_epoch_004_val_dev_synthetic_v1.npz"
)
PRIMARY_E1_CONF_NPZ = (
    S2_67_ROOT / "generated" / "val_conf"
    / "s2_67_primary_epoch_001_val_conf_synthetic_v1.npz"
)

RUN_ROOT = (
    REPO_ROOT / "outputs" / "runs" / "scenario2" / "b2"
    / "replication_e4_scale_failure_diagnostic_v1"
)
METRIC_DIR = RUN_ROOT / "metrics"

PROTOCOL_JSON = METRIC_DIR / "s2_68_replication_e4_scale_failure_diagnostic_protocol_v1.json"
GUARDRAIL_CSV = METRIC_DIR / "s2_68_guardrail_slack_comparison_v1.csv"
BOOTSTRAP_CSV = METRIC_DIR / "s2_68_scale_subject_bootstrap_v1.csv"
SUMMARY_JSON = METRIC_DIR / "s2_68_replication_e4_scale_failure_summary_v1.json"

LOWER_SCALE_GUARD = 0.90
UPPER_SCALE_GUARD = 1.10

PSD_IMPROVEMENT_FRACTION = 0.05
BANDPOWER_TOLERANCE_DB = 0.25
ALPHA_EFFECT_TOLERANCE_DB = 0.50
MIN_MEAN_SPATIAL_R = 0.98
TOP1_RETENTION_FRACTION = 0.90
FINGERPRINT_R_MAX_DROP = 0.05
MARGIN_RETENTION_FRACTION = 0.85
MIN_FIXED_WRONG_WIN_FRACTION = 0.90
MAX_MEAN_WPLI_DROP = 0.15

BOOTSTRAP_SEED = 20261801
BOOTSTRAP_REPS = 5000
N_SUBJECTS = 64


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError("Cannot save empty CSV.")
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_pair(real_path: Path, b1_path: Path):
    real_pkg = np.load(real_path, allow_pickle=False)
    b1_pkg = np.load(b1_path, allow_pickle=False)

    real_z = real_pkg["real_z"].astype(np.float32, copy=False)
    b1_z = b1_pkg["synth_z"].astype(np.float32, copy=False)

    real_idx = real_pkg["selected_subject_indices"].astype(np.int64).tolist()
    b1_idx = b1_pkg["selected_subject_indices"].astype(np.int64).tolist()
    real_ids = real_pkg["selected_subject_ids"].astype(str).tolist()
    b1_ids = b1_pkg["selected_subject_ids"].astype(str).tolist()

    if real_idx != b1_idx or real_ids != b1_ids:
        raise RuntimeError("Real/B1 cohort mismatch.")
    if real_z.shape[0] != N_SUBJECTS:
        raise RuntimeError(f"Unexpected subject count: {real_z.shape}")

    return real_z, b1_z, real_idx, real_ids


def load_candidate(path: Path, expected_idx, expected_ids):
    pkg = np.load(path, allow_pickle=False)
    idx = pkg["selected_subject_indices"].astype(np.int64).tolist()
    ids = pkg["selected_subject_ids"].astype(str).tolist()

    if idx != expected_idx or ids != expected_ids:
        raise RuntimeError(f"Candidate cohort mismatch: {path}")

    z = pkg["synth_z"].astype(np.float32, copy=False)

    if z.shape[0] != N_SUBJECTS:
        raise RuntimeError(f"Unexpected candidate shape: {z.shape}")

    return z


def evaluate_bundle(real_z, synth_z, scaler):
    generic, _, _ = compute_candidate_metrics(
        real_z=real_z,
        synth_z=synth_z,
        scaler=scaler,
    )
    full_psd = full_log_psd_mae(
        real_z=real_z,
        synth_z=synth_z,
        scaler=scaler,
    )
    return {"generic": generic, "full_log_psd": full_psd}


def scalar_metrics(bundle):
    g = bundle["generic"]
    s = g["subject_identity"]

    return {
        "full_psd": float(bundle["full_log_psd"]["mean_mae_db"]),
        "scale_EO": float(g["scale"]["EO"]["normalized_std_ratio"]),
        "scale_EC": float(g["scale"]["EC"]["normalized_std_ratio"]),
        "band": float(g["mean_bandpower_mae_db"]),
        "alpha": float(g["alpha_effect"]["channel_mae_db"]),
        "mean_spatial_r": float(np.mean([
            g["spatial"]["EO"]["correlation_r"],
            g["spatial"]["EC"]["correlation_r"],
        ])),
        "top1": float(s["subject_retrieval_top1_accuracy"]),
        "fingerprint_r": float(s["subject_fingerprint_pearson_r_median"]),
        "margin": float(s["subject_correct_vs_median_wrong_margin_mean"]),
        "fixed_wrong": float(s["subject_correct_better_than_fixed_wrong_fraction"]),
        "mean_wpli": float(np.mean([
            g["spatial"]["EO"]["alpha_dwpli2_r"],
            g["spatial"]["EC"]["alpha_dwpli2_r"],
        ])),
    }


def scale_slack(ratio: float) -> float:
    if ratio < LOWER_SCALE_GUARD:
        return ratio - LOWER_SCALE_GUARD
    if ratio > UPPER_SCALE_GUARD:
        return UPPER_SCALE_GUARD - ratio
    return min(ratio - LOWER_SCALE_GUARD, UPPER_SCALE_GUARD - ratio)


def reconstruct_guardrails(label, candidate_bundle, b1_bundle):
    c = scalar_metrics(candidate_bundle)
    b = scalar_metrics(b1_bundle)

    rows = []

    def add(name, value, threshold, relation, slack, passed):
        rows.append({
            "candidate": label,
            "criterion": name,
            "candidate_value": float(value),
            "threshold_value": float(threshold),
            "relation": relation,
            "slack_positive_is_pass": float(slack),
            "passed": bool(passed),
        })

    psd_thr = b["full_psd"] * (1.0 - PSD_IMPROVEMENT_FRACTION)
    add(
        "full_psd_improvement",
        c["full_psd"],
        psd_thr,
        "<=",
        psd_thr - c["full_psd"],
        c["full_psd"] <= psd_thr,
    )

    for state in ("EO", "EC"):
        ratio = c[f"scale_{state}"] / b[f"scale_{state}"]
        add(
            f"{state.lower()}_scale_parent_relative",
            ratio,
            1.0,
            "0.90 <= ratio <= 1.10",
            scale_slack(ratio),
            LOWER_SCALE_GUARD <= ratio <= UPPER_SCALE_GUARD,
        )

    band_thr = b["band"] + BANDPOWER_TOLERANCE_DB
    add("bandpower_preserved", c["band"], band_thr, "<=", band_thr - c["band"], c["band"] <= band_thr)

    alpha_thr = b["alpha"] + ALPHA_EFFECT_TOLERANCE_DB
    add("alpha_effect_preserved", c["alpha"], alpha_thr, "<=", alpha_thr - c["alpha"], c["alpha"] <= alpha_thr)

    add(
        "mean_spatial_r",
        c["mean_spatial_r"],
        MIN_MEAN_SPATIAL_R,
        ">=",
        c["mean_spatial_r"] - MIN_MEAN_SPATIAL_R,
        c["mean_spatial_r"] >= MIN_MEAN_SPATIAL_R,
    )

    top1_thr = TOP1_RETENTION_FRACTION * b["top1"]
    add("subject_top1_preserved", c["top1"], top1_thr, ">=", c["top1"] - top1_thr, c["top1"] >= top1_thr)

    fp_thr = b["fingerprint_r"] - FINGERPRINT_R_MAX_DROP
    add(
        "fingerprint_r_preserved",
        c["fingerprint_r"],
        fp_thr,
        ">=",
        c["fingerprint_r"] - fp_thr,
        c["fingerprint_r"] >= fp_thr,
    )

    margin_thr = MARGIN_RETENTION_FRACTION * b["margin"]
    add("subject_margin_preserved", c["margin"], margin_thr, ">=", c["margin"] - margin_thr, c["margin"] >= margin_thr)

    add(
        "fixed_wrong_win_preserved",
        c["fixed_wrong"],
        MIN_FIXED_WRONG_WIN_FRACTION,
        ">=",
        c["fixed_wrong"] - MIN_FIXED_WRONG_WIN_FRACTION,
        c["fixed_wrong"] >= MIN_FIXED_WRONG_WIN_FRACTION,
    )

    wpli_thr = max(0.0, b["mean_wpli"] - MAX_MEAN_WPLI_DROP)
    add(
        "wpli_not_catastrophically_degraded",
        c["mean_wpli"],
        wpli_thr,
        ">=",
        c["mean_wpli"] - wpli_thr,
        c["mean_wpli"] >= wpli_thr,
    )

    helper = evaluate_eligibility(
        candidate=candidate_bundle,
        b1_baseline=b1_bundle,
    )

    manual = all(r["passed"] for r in rows)

    if manual != bool(helper["eligible"]):
        raise RuntimeError(
            f"{label}: manual guard reconstruction mismatch "
            f"(manual={manual}, helper={helper['eligible']})."
        )

    return rows, helper


def pooled_state_std(z, state_index, subject_sample=None):
    values = z[:, state_index, ...] if subject_sample is None else z[subject_sample, state_index, ...]
    return float(np.std(values.astype(np.float64, copy=False), ddof=0))


def validate_scale_formula(real_z, synth_z, bundle, label):
    for state_index, state_name in enumerate(("EO", "EC")):
        pooled_ratio = (
            pooled_state_std(synth_z, state_index)
            / pooled_state_std(real_z, state_index)
        )
        helper_ratio = float(
            bundle["generic"]["scale"][state_name]["normalized_std_ratio"]
        )
        if not np.isclose(pooled_ratio, helper_ratio, rtol=0.0, atol=1e-7):
            raise RuntimeError(
                f"{label}/{state_name}: scale formula mismatch "
                f"{pooled_ratio} vs {helper_ratio}"
            )


def bootstrap_scale(real_z, b1_z, candidate_z, label):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    for state_index, state_name in enumerate(("EO", "EC")):
        draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)

        for rep in range(BOOTSTRAP_REPS):
            subject_sample = rng.integers(
                0,
                N_SUBJECTS,
                size=N_SUBJECTS,
                endpoint=False,
            )

            real_std = pooled_state_std(real_z, state_index, subject_sample)
            b1_std = pooled_state_std(b1_z, state_index, subject_sample)
            cand_std = pooled_state_std(candidate_z, state_index, subject_sample)

            b1_scale = b1_std / real_std
            cand_scale = cand_std / real_std

            draws[rep] = cand_scale / b1_scale

        real_std = pooled_state_std(real_z, state_index)
        point = (
            (pooled_state_std(candidate_z, state_index) / real_std)
            / (pooled_state_std(b1_z, state_index) / real_std)
        )

        ci_low, ci_high = np.percentile(draws, [2.5, 97.5])

        rows.append({
            "candidate": label,
            "state": state_name,
            "point_parent_relative_scale": float(point),
            "guard_lower": LOWER_SCALE_GUARD,
            "guard_upper": UPPER_SCALE_GUARD,
            "point_guard_slack": float(scale_slack(point)),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_mean": float(np.mean(draws)),
            "bootstrap_std": float(np.std(draws, ddof=0)),
            "bootstrap_ci95_low": float(ci_low),
            "bootstrap_ci95_high": float(ci_high),
            "bootstrap_fraction_inside_guard": float(
                np.mean((draws >= LOWER_SCALE_GUARD) & (draws <= UPPER_SCALE_GUARD))
            ),
        })

    return rows


def main():
    print("=" * 80)
    print("SCENARIO 2 — s2_68 REPLICATION E4 SCALE FAILURE DIAGNOSTIC")
    print("=" * 80)

    print("\nScientific question:")
    print("  Was REPLICATION E4 a broad failure or a near-boundary EO-scale miss?")

    print("\nGovernance:")
    print("  New training: NO")
    print("  New sampling: NO")
    print("  Threshold changes: NO")
    print("  Retroactive PASS: NO")
    print("  Final Test: NO")

    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    for p in (
        NORMALIZATION_FILE,
        S2_67_SUMMARY,
        VAL_DEV_REAL_NPZ,
        VAL_DEV_B1_NPZ,
        VAL_CONF_REAL_NPZ,
        VAL_CONF_B1_NPZ,
        PRIMARY_E1_DEV_NPZ,
        REPLICATION_E4_DEV_NPZ,
        PRIMARY_E1_CONF_NPZ,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    with S2_67_SUMMARY.open("r", encoding="utf-8") as f:
        s2_67 = json.load(f)

    if s2_67["status"] != "S2_67_FRESH_SEED_VALIDATION_SELECTOR_COMPLETE":
        raise RuntimeError("s2_67 is not complete.")

    if s2_67["selector_mechanism_replicated"] is not False:
        raise RuntimeError("s2_68 expects the observed selector failure.")

    protocol = {
        "status": "S2_68_REPLICATION_E4_SCALE_FAILURE_DIAGNOSTIC_PROTOCOL_FROZEN",
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "reps": BOOTSTRAP_REPS,
            "unit": "subject",
            "scale_guard": [LOWER_SCALE_GUARD, UPPER_SCALE_GUARD],
        },
        "important_limit": (
            "Diagnostic uncertainty cannot retroactively change the s2_67 decision."
        ),
        "new_training": False,
        "new_sampling": False,
        "Validation_EEG_accessed": True,
        "Final_Test_accessed": False,
        "thresholds_changed": False,
        "formal_model_freeze_decision_made": False,
    }

    if PROTOCOL_JSON.is_file():
        with PROTOCOL_JSON.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != protocol:
            raise RuntimeError("Conflicting s2_68 protocol already exists.")
        print("identical s2_68 protocol already exists.")
    else:
        atomic_json_save(protocol, PROTOCOL_JSON)
        print(f"protocol: {PROTOCOL_JSON}")

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if scaler["status"] != "TRAIN_NORMALIZATION_LOCKED":
        raise RuntimeError("Normalization is not locked.")

    dev_real, dev_b1, dev_idx, dev_ids = load_pair(
        VAL_DEV_REAL_NPZ,
        VAL_DEV_B1_NPZ,
    )
    conf_real, conf_b1, conf_idx, conf_ids = load_pair(
        VAL_CONF_REAL_NPZ,
        VAL_CONF_B1_NPZ,
    )

    primary_dev = load_candidate(PRIMARY_E1_DEV_NPZ, dev_idx, dev_ids)
    repl_e4_dev = load_candidate(REPLICATION_E4_DEV_NPZ, dev_idx, dev_ids)
    primary_conf = load_candidate(PRIMARY_E1_CONF_NPZ, conf_idx, conf_ids)

    print("\nSTEP 1 — RECOMPUTE EXACT METRICS")

    dev_b1_bundle = evaluate_bundle(dev_real, dev_b1, scaler)
    conf_b1_bundle = evaluate_bundle(conf_real, conf_b1, scaler)
    primary_dev_bundle = evaluate_bundle(dev_real, primary_dev, scaler)
    repl_e4_bundle = evaluate_bundle(dev_real, repl_e4_dev, scaler)
    primary_conf_bundle = evaluate_bundle(conf_real, primary_conf, scaler)

    validate_scale_formula(dev_real, dev_b1, dev_b1_bundle, "VAL_DEV_B1")
    validate_scale_formula(dev_real, primary_dev, primary_dev_bundle, "PRIMARY_E1_VAL_DEV")
    validate_scale_formula(dev_real, repl_e4_dev, repl_e4_bundle, "REPLICATION_E4_VAL_DEV")

    print("exact scale formula consistency: PASS")

    print("\nSTEP 2 — EXACT GUARDRAIL SLACK")

    guard_rows = []
    helper_details = {}

    for label, candidate_bundle, b1_bundle in (
        ("PRIMARY_E1_VAL_DEV", primary_dev_bundle, dev_b1_bundle),
        ("REPLICATION_E4_VAL_DEV", repl_e4_bundle, dev_b1_bundle),
        ("PRIMARY_E1_VAL_CONF", primary_conf_bundle, conf_b1_bundle),
    ):
        rows, helper = reconstruct_guardrails(
            label,
            candidate_bundle,
            b1_bundle,
        )
        guard_rows.extend(rows)
        helper_details[label] = helper

        print(f"\n{label}: helper eligible={helper['eligible']}")
        for row in rows:
            if (
                not row["passed"]
                or "scale_parent_relative" in row["criterion"]
            ):
                print(
                    f"  {row['criterion']}: "
                    f"value={row['candidate_value']:.8f} "
                    f"| slack={row['slack_positive_is_pass']:+.8f} "
                    f"| pass={row['passed']}"
                )

    save_csv(guard_rows, GUARDRAIL_CSV)

    print("\nSTEP 3 — SUBJECT BOOTSTRAP OF PARENT-RELATIVE SCALE")

    bootstrap_rows = []
    bootstrap_rows += bootstrap_scale(
        dev_real,
        dev_b1,
        primary_dev,
        "PRIMARY_E1_VAL_DEV",
    )
    bootstrap_rows += bootstrap_scale(
        dev_real,
        dev_b1,
        repl_e4_dev,
        "REPLICATION_E4_VAL_DEV",
    )

    save_csv(bootstrap_rows, BOOTSTRAP_CSV)

    for row in bootstrap_rows:
        print(
            f"{row['candidate']} {row['state']}: "
            f"point={row['point_parent_relative_scale']:.6f} "
            f"| slack={row['point_guard_slack']:+.6f} "
            f"| CI95=[{row['bootstrap_ci95_low']:.6f}, "
            f"{row['bootstrap_ci95_high']:.6f}] "
            f"| inside={row['bootstrap_fraction_inside_guard']:.3f}"
        )

    repl_eo = next(
        r for r in bootstrap_rows
        if r["candidate"] == "REPLICATION_E4_VAL_DEV" and r["state"] == "EO"
    )

    failed = [
        r["criterion"]
        for r in guard_rows
        if (
            r["candidate"] == "REPLICATION_E4_VAL_DEV"
            and not r["passed"]
        )
    ]

    summary = {
        "status": "S2_68_REPLICATION_E4_SCALE_FAILURE_DIAGNOSTIC_COMPLETE",
        "replication_E4_failed_criteria": failed,
        "replication_E4_failed_count": len(failed),
        "replication_E4_EO_parent_relative_scale": float(
            repl_eo["point_parent_relative_scale"]
        ),
        "replication_E4_EO_scale_guard_slack": float(
            repl_eo["point_guard_slack"]
        ),
        "replication_E4_EO_bootstrap_CI95": [
            float(repl_eo["bootstrap_ci95_low"]),
            float(repl_eo["bootstrap_ci95_high"]),
        ],
        "replication_E4_EO_bootstrap_fraction_inside_guard": float(
            repl_eo["bootstrap_fraction_inside_guard"]
        ),
        "s2_67_failure_still_binding": True,
        "retroactive_pass_allowed": False,
        "artifacts": {
            "protocol": str(PROTOCOL_JSON),
            "guardrail_csv": str(GUARDRAIL_CSV),
            "bootstrap_csv": str(BOOTSTRAP_CSV),
        },
        "governance": {
            "new_training": False,
            "new_sampling": False,
            "Validation_EEG_accessed": True,
            "Final_Test_accessed": False,
            "thresholds_changed": False,
            "formal_model_freeze_decision_made": False,
        },
        "next": (
            "Use the exact failure magnitude and bootstrap uncertainty to decide "
            "between one final predeclared multi-sampling-seed selector family "
            "on fresh training seeds versus terminating continuation-B2."
        ),
    }

    atomic_json_save(summary, SUMMARY_JSON)

    print("\n" + "=" * 80)
    print("s2_68 REPLICATION E4 SCALE FAILURE DIAGNOSTIC COMPLETE")
    print("=" * 80)

    print(f"REPLICATION E4 failed criteria: {failed}")
    print(
        f"EO parent-relative scale: "
        f"{repl_eo['point_parent_relative_scale']:.8f}"
    )
    print(
        f"EO guard slack: "
        f"{repl_eo['point_guard_slack']:+.8f}"
    )
    print(
        f"EO bootstrap CI95: "
        f"[{repl_eo['bootstrap_ci95_low']:.8f}, "
        f"{repl_eo['bootstrap_ci95_high']:.8f}]"
    )

    print("\nImportant:")
    print("  s2_67 failure remains binding.")
    print("  No threshold changed.")
    print("  No B2 freeze decision made.")
    print("  Final Test remains sealed.")


if __name__ == "__main__":
    main()
