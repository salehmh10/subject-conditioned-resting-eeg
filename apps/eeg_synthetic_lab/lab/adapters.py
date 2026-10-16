"""Translate saved reports into an explicitly sourced, presentation-only schema."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
from pathlib import Path

from .sources import EvidenceReader, EvidenceError
from .validation import CONDITIONS, FAMILIES, assert_unique, normalize_contrast, require

MODELS = ("Real", "B0", "B1", "B2")
STATES = ("EO", "EC")
BANDS = ("theta", "alpha", "beta")


def provenance(source_ids, transformation, units, aggregation, split="real_train"):
    return {"source_ids": [source_ids] if isinstance(source_ids, str) else list(source_ids),
            "transformation": transformation, "units": units,
            "aggregation": aggregation, "analysis_split": split}


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def array_mean(values):
    """Equal-weight finite mean; undefined remains undefined when no values exist."""
    first = values[0]
    if isinstance(first, list):
        require(all(isinstance(x, list) and len(x) == len(first) for x in values), "Mismatched numeric array shapes")
        return [array_mean([v[index] for v in values]) for index in range(len(first))]
    kept = [finite(x) for x in values if x is not None]
    kept = [x for x in kept if x is not None]
    return statistics.mean(kept) if kept else None


def metric_definition(metric):
    if metric.startswith(("log_psd", "bandpower", "relative_bandpower", "alpha_")):
        group = "Spectrum and Alpha"
    elif metric.startswith(("covariance", "correlation", "connectivity")):
        group = "Spatial Statistics"
    elif any(word in metric for word in ("diversity", "copy", "duplicate", "nearest_real", "sliced_wasserstein")):
        group = "Distribution and Diversity"
    else:
        group = "Time and Complexity"
    if "real_sd_units" in metric:
        unit = "Real Train SD units"
    elif "_db" in metric:
        unit = "dB"
    elif "uv2" in metric:
        unit = "uV²"
    elif "uv" in metric:
        unit = "uV"
    elif metric.endswith("_r"):
        unit = "correlation r"
    elif "fraction" in metric or "relative_bandpower" in metric:
        unit = "fraction"
    elif "relative_frobenius" in metric:
        unit = "dimensionless"
    elif metric.endswith("_z") or "rmse_z" in metric:
        unit = "Train-z units"
    elif "per_sample" in metric:
        unit = "per sample"
    else:
        unit = "dimensionless"
    direction = "higher" if metric.endswith("_r") else "lower"
    if "diversity_ratio" in metric:
        direction = "parity"
    elif "nearest_real" in metric:
        direction = "descriptive"
    return group, unit, direction


def flatten_metrics(report, key, source, models=("B0", "B1", "B2")):
    records = []
    for metric, by_model in report[key].items():
        group, unit, direction = metric_definition(metric)
        for model in models:
            if model not in by_model:
                continue
            value = by_model[model]
            records.append({"metric": metric, "model": model, "group": group,
                            "mean": value["mean"], "median": value["median"],
                            "ci_low": value["ci95"][0], "ci_high": value["ci95"][1],
                            "n_subjects": value["n_subjects"], "unit": unit, "direction": direction,
                            "provenance": provenance(source, f"Copy saved {key}.{metric}.{model}; no recomputation of confidence intervals",
                                unit, "equal participant weight; equal EO/EC within participant")})
    return records


def prepare_fidelity(reader: EvidenceReader, paths: dict):
    b0, base = paths["fidelity_b0"], paths["fidelity_b1_b2"]
    report_path = f"{b0}/metrics/s2_74e_b0_complete_report_v1.json"
    baseline_path = f"{base}/metrics/s2_74b_complete_report_v1.json"
    report = reader.json(report_path)
    baseline = reader.json(baseline_path)
    require(reader.manifest[baseline_path]["sha256"] == report["baseline_s2_74B_complete_sha256"], "B0 extension references a different baseline report")
    conflicts = []
    def compare_saved(original, combined, identity):
        if isinstance(original, dict):
            for key, value in original.items():
                if key not in combined:
                    conflicts.append(f"{identity}/{key}: baseline field missing from extension")
                else:
                    compare_saved(value, combined[key], f"{identity}/{key}")
        elif original != combined:
            conflicts.append(f"{identity}: baseline and extension disagree")
    for section in ("full_pool_metrics", "fingerprint", "alpha_EC_minus_EO", "connectivity",
                    "connectivity_EC_minus_EO", "matched_train_block_calibration_summary",
                    "matched_train_block_connectivity_summary"):
        compare_saved(baseline[section], report[section], section)
    require(not conflicts, "Conflicting Fidelity reports: " + "; ".join(conflicts))
    plan_path = f"{base}/protocol/s2_74b_analysis_plan_v1.json"
    plan = reader.json(plan_path, "frozen Fidelity method")
    b0plan_path = f"{b0}/protocol/s2_74e_b0_analysis_plan_v1.json"
    reader.json(b0plan_path, "B0 evaluation-stratum definition")
    script_path = "scripts/data/s2_74b_final_pool_fidelity.py"
    script = reader.read_bytes(script_path, "read-only aggregation formula inspection").decode("utf-8-sig")
    # Parse literals without executing or importing any study module.
    channels = None
    for node in ast.parse(script).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CHANNELS" for t in node.targets):
            channels = ast.literal_eval(node.value)
    require(channels is not None and len(channels) == 19, "Missing verified channel order")
    settings = plan["settings"]
    fs = settings["welch"]["fs"]
    nperseg = settings["welch"]["nperseg"]
    frequencies = [i * fs / nperseg for i in range(nperseg // 2 + 1)]
    fidelity = flatten_metrics(report, "full_pool_metrics", report_path)
    calibration = flatten_metrics(report, "matched_train_block_calibration_summary", report_path,
                                  ("Real_B", "B0", "B1", "B2"))
    fingerprint = []
    for identity, item in report["fingerprint"].items():
        state, model = identity.split("_")
        fingerprint.append({"state": state, "model": model, **{k: item[k] for k in ("top1", "top5", "n_candidates", "n_valid_queries", "margin")},
                            "provenance": provenance(report_path, f"Copy saved fingerprint.{identity}", "fraction; correlation margin", "198 candidate participants, in-sample 76 log-bandpower features")})
    fingerprint_subjects = []
    for root, filename in ((b0, "subject_B0_fingerprint.csv"), (base, "subject_fingerprint.csv")):
        source = f"{root}/tables/{filename}"
        for row in reader.csv(source):
            fingerprint_subjects.append({"subject_id": row["subject_id"], "state": row["state"], "model": row["generator"],
                "rank": int(row["rank"]), "margin": finite(row["correct_vs_best_wrong_margin"]),
                "correct_subject_r": finite(row["correct_subject_r"]), "rmse_db": finite(row["fingerprint_rmse_db"]),
                "provenance": provenance(source, "Copy saved keyed participant fingerprint diagnostics", "rank; r; dB", "participant/state")})
    subject_files = sorted((reader.repository / base / "subjects").glob("*.json"))
    b0_files = sorted((reader.repository / b0 / "subjects").glob("*.json"))
    require({p.stem for p in subject_files} == {p.stem for p in b0_files}, "B0 and baseline participant identities differ")
    psd_participants, conn_participants, alpha_participants = {}, {}, {}
    conn_valid_bins, source_by_subject = {}, {}
    fidelity_subjects, connectivity_metrics = [], []
    for file in subject_files:
        sid = file.stem
        base_file, b0_file = f"{base}/subjects/{sid}.json", f"{b0}/subjects/{sid}.json"
        original, added = reader.json(base_file), reader.json(b0_file)
        require(original["subject_id"] == added["subject_id"] == sid, "Participant ID differs from saved filename")
        require(original["subject_index"] == added["subject_index"], "Participant index disagreement")
        require(added["baseline_subject_sha256"] == reader.manifest[base_file]["sha256"], "B0 participant result refers to changed baseline source")
        source_by_subject[sid] = [base_file, b0_file]
        psd_participants[sid], conn_participants[sid], conn_valid_bins[sid], alpha_participants[sid] = {}, {}, {}, {}
        for state in STATES:
            require(state in original["states"] and state in added["states"], "Missing eye-state participant record")
            psd_participants[sid][state] = {}
            conn_participants[sid][state] = {band: {} for band in BANDS}
            conn_valid_bins[sid][state] = {band: {} for band in BANDS}
            for model in MODELS:
                src = added if model == "B0" else original
                source = b0_file if model == "B0" else base_file
                data = src["states"][state]
                summary = data["source_summaries"][model]
                powers = summary["mean_psd_uv2_hz"]
                require(len(powers) == len(channels) and all(len(x) == len(frequencies) for x in powers), "Unexpected PSD shape")
                psd_participants[sid][state][model] = array_mean(powers)
                if model != "Real":
                    fidelity_subjects.append({"subject_id": sid, "state": state, "model": model,
                        "metrics": data["full_pool_metrics"][model],
                        "provenance": provenance(source, "Copy saved full-pool metrics", "metric-specific; see metric definitions", "participant/state")})
                for band in BANDS:
                    mat = data["nonoverlap_connectivity_matrices"][band][model]
                    matrix = [[None if i == j else finite(v) for j, v in enumerate(row)] for i, row in enumerate(mat["matrix"])]
                    require(len(matrix) == 19 and all(len(row) == 19 for row in matrix), "Unexpected connectivity matrix shape")
                    conn_participants[sid][state][band][model] = matrix
                    conn_valid_bins[sid][state][band][model] = mat["valid_frequency_bins"]
                    if model != "Real":
                        connectivity_metrics.append({"subject_id": sid, "state": state, "band": band, "model": model,
                            **data["nonoverlap_connectivity"][band][model],
                            "provenance": provenance(source, "Copy saved edge/node/strength diagnostics", "dwPLI²; r; dimensionless relative error", "participant/state/band")})
        for model in MODELS:
            src = added if model == "B0" else original
            eo = src["states"]["EO"]["source_summaries"][model]["median_bandpower_db"][2]
            ec = src["states"]["EC"]["source_summaries"][model]["median_bandpower_db"][2]
            alpha_participants[sid][model] = [y - x for x, y in zip(eo, ec)]
    psd_group = {state: {model: array_mean([x[state][model] for x in psd_participants.values()]) for model in MODELS} for state in STATES}
    conn_group = {state: {band: {model: array_mean([x[state][band][model] for x in conn_participants.values()]) for model in MODELS} for band in BANDS} for state in STATES}
    alpha_group = {model: array_mean([x[model] for x in alpha_participants.values()]) for model in MODELS}
    all_subject_sources = [source for pair in source_by_subject.values() for source in pair]
    array_prov = provenance(all_subject_sources + [script_path, plan_path],
        "Join explicit subject_id/state/model; equal participant weight; no bootstrap recomputation", "section-specific", "group and saved participant summaries")
    return {"fidelity_metrics": fidelity, "calibration": calibration, "fingerprint": fingerprint,
        "fingerprint_subjects": fingerprint_subjects, "fidelity_subjects": fidelity_subjects,
        "connectivity_metrics": connectivity_metrics,
        "psd": {"frequencies_hz": frequencies, "group": psd_group, "participants": psd_participants,
                "unit": "uV^2/Hz", "aggregation": "Equal participant and channel mean in linear power before conversion to dB; display 1–40 Hz",
                "participant_aggregation": "Equal channel mean of saved participant mean-window PSD",
                "provenance": array_prov, "sources_by_subject": source_by_subject},
        "connectivity": {"channels": channels, "bands": list(BANDS), "group": conn_group, "participants": conn_participants,
                "valid_frequency_bins": conn_valid_bins, "unit": "dwPLI²",
                "aggregation": "Equal participant mean per finite edge; diagonal masked; undefined edges remain null; negative estimates retained",
                "provenance": array_prov, "sources_by_subject": source_by_subject},
        "alpha_effect": {"channels": channels, "group": alpha_group, "participants": alpha_participants,
                "unit": "dB EC minus EO", "aggregation": "Within-participant EC minus EO median-window alpha bandpower, then equal participant mean",
                "provenance": array_prov},
        "fidelity_group_connectivity": report["connectivity"],
        "fidelity_group_alpha": report["alpha_EC_minus_EO"],
        "method_definitions": {k: v for k, v in settings.items() if k not in ("excluded", "cpu_threads_requested", "version")},
        "conflicts": conflicts, "channels": channels, "n_subjects": report["subjects"]}


def numeric_csv(row):
    result = dict(row)
    for key in ("seed", "best_epoch", "stopped_epoch", "optimizer_updates_done", "subject_index"):
        if key in row:
            result[key] = int(row[key]) if row[key] else None
    for key in ("subject_macro_balanced_accuracy", "subject_macro_roc_auc", "global_balanced_accuracy", "global_roc_auc", "balanced_accuracy", "roc_auc"):
        if key in row:
            result[key] = finite(row[key]) if row[key] else None
    if "confusion_matrix_json" in result:
        import json
        result["confusion_matrix"] = json.loads(result.pop("confusion_matrix_json"))
        result["confusion_matrix_definition"] = "Saved global window counts; rows=true EO/EC, columns=predicted EO/EC; no seed pooling"
    return result


def prepare_utility(reader: EvidenceReader, paths: dict):
    root = paths["utility_v2"]
    vp = f"{root}/final_validation/s2_75b_v2_complete_validation_report.json"
    tp = f"{root}/final_test_s2_77_v1/s2_77_v2_complete_final_test_report.json"
    vr, tr = reader.json(vp), reader.json(tp)
    require(vr["EEGNet_max_epochs"] == 120 and tr["v1_24_epoch_models_used"] is False, "Wrong utility experiment policy")
    source_script = "scripts/data/s2_77_v2_single_final_test_evaluation.py"
    reader.read_bytes(source_script, "read-only saved contrast direction inspection")
    reader.read_bytes("scripts/data/s2_75b_v2_utility_120ep_earlystop.py", "read-only epoch/selection policy inspection")
    runs, summary, contrasts, histories, utility_subjects = [], [], [], [], []
    split_inputs = (("validation", vp, vr["metric_subject_macro_ba"], f"{root}/final_validation/all_55_classifier_validation_runs.csv"),
                    ("final_test", tp, tr["results"], f"{root}/final_test_s2_77_v1/tables/all_55_final_test_fits.csv"))
    for split, source, report, table_path in split_inputs:
        for row in reader.csv(table_path):
            runs.append({"split": split, **numeric_csv(row),
                "provenance": provenance(table_path, "Copy saved final v2 per-fit metrics; retain exact seed ID", "fraction", "classifier/condition/seed", split)})
        for family, conditions in report.items():
            for condition, result in conditions.items():
                if split == "validation":
                    metric = result["subject_macro_balanced_accuracy"]
                    mean, sd, n, interval, auc = metric["mean"], metric["population_sd"], metric["n_seeds"], [None, None], result["subject_macro_roc_auc"]["mean"]
                else:
                    mean, sd, n, interval, auc = (result["subject_macro_balanced_accuracy_mean"], result["subject_macro_balanced_accuracy_population_sd"],
                        result["n_seeds"], result["paired_subject_bootstrap_ci95"], result["subject_macro_roc_auc_mean"])
                summary.append({"split": split, "family": family, "condition": condition, "mean": mean,
                    "population_sd": sd, "n_seeds": n, "ci_low": interval[0], "ci_high": interval[1], "auc": auc,
                    "sd_definition": "population SD across training seeds (ddof=0); single-fit families have no seed uncertainty",
                    "ci_definition": "Saved participant-bootstrap percentile 95%; not multiplicity adjusted" if split == "final_test" else "No summary interval saved here",
                    "provenance": provenance(source, "Copy saved classifier-condition mean, population SD and available interval", "fraction", "participant-macro; mean over available training seeds", split)})
    for family, values in vr["paired_subject_bootstrap_comparisons"].items():
        for identity, row in values.items():
            contrasts.append({"split": "validation", "family": family, "id": identity, "left": row["left"], "right": row["right"],
                "delta": row["point_subject_macro_ba_difference"], "ci_low": row["paired_95pct_subject_bootstrap_ci"][0],
                "ci_high": row["paired_95pct_subject_bootstrap_ci"][1], "n_subjects": row["n_subjects"],
                "direction": "left_minus_right", "source_direction": "left_minus_right",
                "provenance": provenance(vp, "Copy saved left-minus-right paired contrast and existing interval", "fraction difference; multiply by 100 for percentage points", "paired participant difference; seed mean", "validation")})
    for row in tr["paired_contrasts"]:
        delta, low, high = normalize_contrast(row["delta_right_minus_left"], row["ci95_low"], row["ci95_high"], "right_minus_left")
        contrasts.append({"split": "final_test", "family": row["family"], "id": row["contrast_id"], "left": row["left"], "right": row["right"],
            "delta": delta, "ci_low": low, "ci_high": high, "n_subjects": row["n_subjects"],
            "direction": "left_minus_right", "source_direction": "right_minus_left",
            "provenance": provenance([tp, source_script], "Negate saved right-minus-left point and reverse/negate CI endpoints to display left-minus-right", "fraction difference; multiply by 100 for percentage points", "paired participant difference; seed mean", "final_test")})
    for family in FAMILIES:
        for file in sorted((reader.repository / root / "runs" / family).glob("*/*_receipt.json")):
            relative = file.relative_to(reader.repository).as_posix()
            row = reader.json(relative, "final v2 saved fit receipt and history")
            desc = row["descriptor"]
            require(desc["family"] == family and desc["version"] == "s2_75b_11condition_120epoch_earlystop_v2", "Old/mismatched fit receipt")
            condition, seed = desc["condition"], desc["seed"]
            result = row["result"]
            fit = next(r for r in runs if r["split"] == "validation" and r["family"] == family and r["condition"] == condition and r["seed"] == seed)
            require(math.isclose(fit["subject_macro_balanced_accuracy"], result["subject_macro_balanced_accuracy"], abs_tol=1e-12), "Saved receipt and Validation table disagree")
            fit["confusion_matrix"] = result["confusion_matrix"]
            fit["confusion_matrix_definition"] = "Saved global window counts; rows=true EO/EC, columns=predicted EO/EC; no seed pooling"
            fit["provenance"]["source_ids"].append(relative)
            for index, (ba, auc) in enumerate(zip(result["subject_ba"], result["subject_auc"])):
                utility_subjects.append({"split": "validation", "family": family, "condition": condition, "seed": seed,
                    "subject_index": index, "balanced_accuracy": ba, "roc_auc": auc,
                    "provenance": provenance(relative, "Copy explicitly indexed saved participant metrics", "fraction", "participant/fit", "validation")})
            if family == "EEGNETV4":
                method = row["method"]
                histories.append({"condition": condition, "seed": seed, "best_epoch": method["best_epoch"],
                    "stopped_epoch": method["stopped_epoch"], "stop_reason": method["stop_reason"], "train_rows": method["train_rows"],
                    "epochs": [{"epoch": e["epoch"], "train_loss": e["train_loss"], "validation_loss": e["validation_loss"],
                                "validation_ba": e["validation_subject_macro_ba"], "optimizer_updates": e["optimizer_updates"]} for e in method["training_history"]],
                    "provenance": provenance(relative, "Copy actual saved epoch history; selected/stopping epoch retained; no interpolation", "cross entropy; BA fraction; epoch", "condition/training seed", "train_and_validation")})
    subject_table = f"{root}/final_test_s2_77_v1/tables/per_subject_55_fits.csv"
    for row in reader.csv(subject_table):
        utility_subjects.append({"split": "final_test", **numeric_csv(row),
            "provenance": provenance(subject_table, "Copy saved explicitly indexed participant metrics", "fraction", "participant/fit", "final_test")})
    return {"utility_runs": runs, "utility_summary": summary, "contrasts": contrasts,
            "histories": histories, "utility_subjects": utility_subjects,
            "test_window_count": tr["n_test_windows"],
            "utility_policy": {"max_epochs": vr["EEGNet_max_epochs"], "min_epochs": vr["EEGNet_min_epochs"],
                               "patience": vr["EEGNet_patience"], "selection": "Best real Validation subject-macro balanced accuracy; lower Validation cross-entropy tie-breaker"}}


def build_bundle(reader, paths):
    fidelity = prepare_fidelity(reader, paths)
    utility = prepare_utility(reader, paths)
    plan_path = "outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1/protocol/s2_73_generation_plan_v1.json"
    plan = reader.json(plan_path, "study metadata; no utility performance")
    core, contract = plan["core"], plan["core"]["data_contract"]
    scaler_path = "outputs/metrics/scenario2/train_normalization.json"
    scaler = reader.json(scaler_path, "locked normalization metadata only")
    n_train = contract["Train"]["usable_windows"]
    require(scaler["fit_split"] == "Train" and scaler["fitting_train_window_count"] == n_train, "Train metadata conflict")
    require(core["channel_order"] == fidelity["channels"] == scaler["channel_order"], "Channel order conflict")
    half_counts = {h["train_rows"] - n_train for h in utility["histories"] if h["condition"].endswith("0P5X")}
    require(len(half_counts) == 1, "Matched half-subset count differs among conditions")
    subject_ids = core["subject_ids_in_index_order"]
    require(len(set(subject_ids)) == len(subject_ids) == fidelity["n_subjects"], "Invalid participant identity mapping")
    require(set(subject_ids) == set(fidelity["psd"]["participants"]), "Utility/Fidelity participant identities differ")
    for row in utility["utility_subjects"]:
        row["subject_id"] = subject_ids[row["subject_index"]]
        row["provenance"]["source_ids"].append(plan_path)
    metadata = {"dataset": "MPI-LEMON resting-state EEG", "n_subjects": contract["subjects"],
        "channels": fidelity.pop("channels"), "fs_hz": contract["window"]["sampling_rate_hz"],
        "window_seconds": contract["window"]["seconds"], "window_samples": contract["window"]["samples"],
        "n_train": n_train, "n_validation": contract["Validation"]["usable_windows"],
        "n_test": utility.pop("test_window_count"), "n_synthetic_per_generator": core["expected_windows_per_generator"],
        "n_half_subset": next(iter(half_counts)), "condition_order": CONDITIONS, "families": FAMILIES,
        "utility_policy": utility.pop("utility_policy"), "definitions": fidelity.pop("method_definitions"),
        "limitations": [
            "Train, Validation and Final Test contain different blocks from the same participants; unseen-participant and new-session generalization are not demonstrated.",
            "Fidelity is descriptive and referenced to Real Train. Matched-count Real–Real calibration uses disjoint Train blocks and differs from full-pool analysis.",
            "B0 receives eye state only; its participant ID is an evaluation-stratum label. B1 receives participant identity and eye state. B2 is continued training and checkpoint selection in that conditioned family.",
            "B0 versus B1/B2 does not isolate a causal effect of participant conditioning because training histories differ.",
            "TSTR uses synthetic classifier-training data; EEGNet checkpoint selection still uses real Validation.",
            "Equal epoch ceilings do not imply equal optimizer-update budgets; Real Duplicate remains a control.",
            "Participant-bootstrap confidence intervals and between-seed population SD quantify different uncertainty; intervals are not multiplicity adjusted.",
            "Fingerprints use 76 log-bandpower features. In-sample identification and absence of detected exact copies are not privacy guarantees.",
            "dwPLI² is sensor-space phase-lag connectivity; negative debiased estimates are retained. It is neither anatomical nor causal connectivity."],
        "provenance": provenance([plan_path, scaler_path], "Copy verified data-contract counts; half-subset count from saved training rows minus Real Train rows", "participants; windows; samples; Hz", "study metadata", "all")}
    fidelity.pop("n_subjects")
    bundle = {"schema_version": 1, "name": "EEG Synthetic Lab", "metadata": metadata,
        "availability": {"group_fidelity": True, "participant_fidelity": True, "group_psd": True,
            "participant_psd": True, "channel_psd": False, "group_connectivity": True,
            "participant_connectivity": True, "fingerprint": True, "participant_fingerprint": True,
            "utility": True, "learning_histories": True, "waveform_previews": False,
            "waveform_reason": "Prepare the separate bounded Train/synthetic preview bundle with export_train_previews.py.",
            "external_preview_bundle": "previews.json", "missing_capabilities": ["Per-channel official PSD controls are unavailable in this compact bundle; participant curves are equal-channel averages."]},
        **fidelity, **utility,
        "provenance": {"sources": dict(reader.manifest), "policy": "Read-only saved numeric evidence; no experiment execution, no raw Validation/Test waveforms", "conflicts": fidelity["conflicts"]}}
    preview_path = "apps/eeg_synthetic_lab/data/presentation_bundle/previews.json"
    if (reader.repository / preview_path).is_file():
        preview = reader.json(preview_path, "bounded Train preview availability")
        if preview.get("available"):
            bundle["availability"]["waveform_previews"] = True
            bundle["availability"]["waveform_reason"] = "Verified bounded Real Train and precomputed synthetic examples are available in the separate preview bundle."
    bundle["provenance"]["sources"] = dict(reader.manifest)
    # Many rows share the exact same transformation. Intern them losslessly to keep
    # the local evidence bundle compact without rounding any scientific number.
    transforms = {}
    for section in ("fidelity_metrics", "calibration", "fidelity_subjects", "fingerprint", "fingerprint_subjects",
                    "connectivity_metrics", "utility_runs", "utility_summary", "contrasts", "histories", "utility_subjects"):
        for row in bundle[section]:
            prov = row["provenance"]
            key = hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()[:20]
            if key in transforms:
                require(transforms[key] == prov, "Unexpected provenance identifier collision")
            transforms[key] = prov
            row["provenance"] = {"transform_id": key}
    bundle["provenance"]["transforms"] = transforms
    return clean_json(bundle)
