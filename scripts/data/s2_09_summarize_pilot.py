"""Verify completed pilot receipts and aggregate persisted Train-only QC scalars.

This command never opens raw EEG through a waveform loader. It reads small raw
sidecars, checks raw file sizes, and verifies already-derived Train HDF5 files.
It does not select thresholds, delete raw data, or change the frozen method.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from resting_eeg.common import canonical_hash, sha256_file, write_json
from resting_eeg.data.access import TrainAccessGate, sample_interval
from resting_eeg.data.integrity import PILOT_SUBJECTS, audit_pilot_inventory
from resting_eeg.preprocessing.pipeline import code_digest, load_config, verify_output
from resting_eeg.preprocessing.signal import expected_samples


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_frame(path):
    return pd.read_csv(path, keep_default_na=False)


def truth(series):
    values = series.astype(str).str.lower()
    require(values.isin(["true", "false"]).all(), "Invalid boolean in persisted QC report")
    return values == "true"


def write_csv(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.8g")
    temporary.replace(path)


def rate_summary(subjects, blocks, windows, sanity):
    rows = []
    for rate, subset in subjects.groupby("native_sfreq", sort=True):
        ids = set(subset.subject_id)
        b = blocks.loc[blocks.subject_id.isin(ids)]
        w = windows.loc[windows.subject_id.isin(ids)]
        s = sanity.loc[sanity.subject_id.isin(ids)]
        rows.append(dict(
            native_sfreq=float(rate), n_subjects=len(subset), n_qc_pass=int(subset.qc_pass.sum()),
            n_blocks=len(b), n_usable_blocks=int(b.block_eligible.sum()),
            candidate_windows=len(w), usable_windows=int(w.usable.sum()),
            retained_fraction=float(w.usable.mean()),
            minimum_native_finite_fraction=float(s.finite_fraction.min()),
            native_nan_count=int(s.nan_count.sum()), native_inf_count=int(s.inf_count.sum()),
            finite_compact_window_fraction=float(w.finite.mean()),
            n_bad_eeg=int(subset.n_bad_eeg.sum()), n_bad_target=int(subset.n_bad_target.sum()),
        ))
    return pd.DataFrame(rows)


def verify_registry(registry, gate, integrity, expected_by_subject):
    require(len(registry) > 0, "Missing Train waveform access evidence")
    require(set(registry.subject_id) == set(PILOT_SUBJECTS), "Access registry contains missing/non-pilot subjects")
    require(truth(registry.access_allowed).all(), "Access registry contains non-permitted access")
    require((registry.split == "Train").all(), "Held-out access recorded")
    require((registry.split_manifest_sha256 == gate.split_hash).all(), "Registry split hash mismatch")
    require((registry.block_inventory_sha256 == gate.inventory_hash).all(), "Registry inventory hash mismatch")
    for row in registry.itertuples(index=False):
        block = gate.authorize(row.subject_id, row.block_index)
        rate = float(integrity.loc[row.subject_id, "native_sfreq"])
        start, stop = sample_interval(block["start_sec"], block["end_estimate_capped_sec"], rate)
        require(row.block_uid == block["block_uid"] and row.state == block["state"], "Registry block identity mismatch")
        require(row.start_sec == block["start_sec"] and row.end_sec == block["end_estimate_capped_sec"], "Registry seconds differ from authoritative interval")
        require(row.native_sfreq == rate and row.start_sample == start and row.stop_sample_exclusive == stop,
                "Registry native sample interval mismatch")
        require(bool(row.purpose) and bool(row.script_name) and len(row.git_head) == 40, "Incomplete access provenance")
    actual = set(registry.block_uid)
    expected = {uid for values in expected_by_subject.values() for uid in values}
    require(actual == expected and len(expected) == 150, "Access registry does not cover exactly the 150 provisional Train blocks")
    pd.to_datetime(registry.timestamp_utc, utc=True, errors="raise")
    return dict(
        access_registry_pass=True, n_access_events=len(registry), n_unique_train_blocks=len(actual),
        n_repeated_authorized_events=len(registry) - len(actual), n_subjects=registry.subject_id.nunique(),
        n_validation_accesses=0, n_test_accesses=0, n_interrupted_fragment_accesses=0,
        complete_Train_only=True, exact_authoritative_interval_checks_pass=True,
        split_manifest_sha256=gate.split_hash, block_inventory_sha256=gate.inventory_hash,
        git_heads=sorted(registry.git_head.unique().tolist()),
        future_split_constraint="Every development-accessed block must remain Train at final split lock, or its subject must be excluded.",
        registry_semantics="Durable authorization is written before raw value reads; events are permitted read attempts.",
    )


def validate_subject_reports(subject, tables, summary, expected_blocks):
    expected = set(expected_blocks)
    for name, frame in tables.items():
        require(set(frame.subject_id) == {subject}, f"Wrong subject in {name}: {subject}")
        if "block_uid" in frame:
            require(set(frame.block_uid) == expected, f"Unexpected/missing Train blocks in {name}: {subject}")
    sanity, metrics, decisions = tables["signal_sanity"], tables["bad_channel_metrics"], tables["bad_channel_decisions"]
    windows, blocks, candidates = tables["window_qc"], tables["block_qc"], tables["qc_candidates"]
    require(not sanity.duplicated(["block_uid", "channel"]).any() and len(sanity) == 620, f"Native sanity completeness failed: {subject}")
    require(not metrics.duplicated(["block_uid", "channel"]).any() and len(metrics) == 610, f"Bad-channel metrics completeness failed: {subject}")
    require(not decisions.channel.duplicated().any() and len(decisions) == 61, f"Bad-channel decisions incomplete: {subject}")
    require(not blocks.block_uid.duplicated().any() and len(blocks) == 10, f"Block QC completeness failed: {subject}")
    require(not windows.duplicated(["block_uid", "window_start_sample"]).any(), f"Duplicate window QC: {subject}")
    require((windows.usable == (windows.reason == "usable")).all(), f"Window QC reason mismatch: {subject}")
    require(len(windows) == summary["candidate_windows"] and int(windows.usable.sum()) == summary["usable_windows"], f"Window/subject counts disagree: {subject}")
    require(int(blocks.block_eligible.sum()) == summary["usable_blocks"], f"Block/subject counts disagree: {subject}")
    require(set(decisions.loc[decisions.bad, "channel"]) == set(summary["bad_channels"]), f"Bad-channel summary mismatch: {subject}")
    require(len(candidates) == 6 and not candidates.duplicated(["multiplier", "state"]).any(), f"Candidate comparison incomplete: {subject}")
    for row in blocks.itertuples():
        w = windows.loc[windows.block_uid == row.block_uid]
        require(len(w) == row.candidate_windows and int(w.usable.sum()) == row.usable_windows, f"Window/block counts disagree: {row.block_uid}")
        require(set(w.state) == {row.state}, f"Window/block state disagreement: {row.block_uid}")


def verify_compact_metadata(path, receipt, blocks, tables, config, config_hash, source_hashes, gate):
    verified = verify_output(path, config_hash, [block["block_uid"] for block in blocks])
    for key in ["output_sha256", "output_bytes", "config_sha256", "expected_blocks"]:
        require(verified[key] == receipt[key], f"Receipt/output mismatch for {key}: {path.name}")
    total_samples = 0
    with h5py.File(path, "r") as handle:
        require(handle.attrs["input_signature"] == receipt["input_signature"], "Output input-signature mismatch")
        require(handle.attrs["code_content_sha256"] == receipt["code_content_sha256"], "Output code digest mismatch")
        require(handle.attrs["code_git_head"] == receipt["code_git_head"], "Output code commit mismatch")
        require(handle.attrs["split_manifest_sha256"] == gate.split_hash and handle.attrs["block_inventory_sha256"] == gate.inventory_hash, "Output authoritative metadata hashes mismatch")
        require(json.loads(handle.attrs["target_channels"]) == config["target_channels"], "Output differs from frozen target order")
        require(bool(handle.attrs["qc_pass"]) == receipt["qc_pass"], "Output subject QC disagreement")
        for block in blocks:
            group = handle["blocks"][block["block_uid"]]
            rate = float(group.attrs["native_sfreq"])
            require(rate == receipt["qc_summary"]["native_sfreq"], "Stored native sampling rate differs from verified subject rate")
            start, stop = sample_interval(block["start_sec"], block["end_estimate_capped_sec"], rate)
            n = expected_samples(stop - start, rate) - 2 * config["edge"]["guard_samples_at_128"]
            require(group["eeg_uv"].shape == (19, n), f"Stored sample count mismatch: {block['block_uid']}")
            require(group.attrs["first_processed_sample_sec"] == start / rate + config["edge"]["guard_sec"], "Stored native-time origin mismatch")
            require(group.attrs["state"] == block["state"] and group.attrs["subject_id"] == block["subject_id"], "Output block identity mismatch")
            require(group.attrs["source_eeg_sha256"] == source_hashes[".eeg"] and group.attrs["source_vhdr_sha256"] == source_hashes[".vhdr"], "Output source hash mismatch")
            windows = tables["window_qc"].loc[tables["window_qc"].block_uid == block["block_uid"]]
            require(np.array_equal(group["window_start_samples"][:], windows.window_start_sample.to_numpy()), "Output/scalar window starts mismatch")
            require(np.array_equal(group["window_usable_mask"][:], windows.usable.to_numpy()), "Output/scalar window masks mismatch")
            require(np.array_equal(group["window_qc_reason"][:].astype(str), windows.reason.to_numpy()), "Output/scalar window reasons mismatch")
            row = tables["block_qc"].loc[tables["block_qc"].block_uid == block["block_uid"]].iloc[0]
            require(bool(group.attrs["block_eligible"]) == bool(row.block_eligible), "Output/scalar block eligibility mismatch")
            total_samples += n
    return verified, total_samples


def create_figure(path, subjects, blocks, rates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(subjects))
    labels = subjects.subject_id.str.replace("sub-", "", regex=False)
    colors = ["#267997" if value else "#b44843" for value in subjects.qc_pass]
    axes[0, 0].bar(x, subjects.retained_fraction * 100, color=colors)
    axes[0, 0].set(title="Provisional Train window retention", ylabel="Usable candidates (%)", ylim=(0, 105))
    axes[0, 0].legend(handles=[Patch(color="#267997", label="Subject QC pass"),
                               Patch(color="#b44843", label="Subject QC fail")], frameon=False, fontsize=8)
    axes[0, 1].bar(x, subjects.n_bad_eeg, color="#267997", label="All EEG")
    axes[0, 1].bar(x, subjects.n_bad_target, color="#dda15e", label="Target subset")
    axes[0, 1].set(title="Subject bad-channel decisions", ylabel="Channels",
                   ylim=(0, max(1, int(subjects.n_bad_eeg.max()) + 1)))
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    if not subjects.n_bad_eeg.any():
        axes[0, 1].text(.5, .5, "No channels met the frozen bad-channel rule",
                        transform=axes[0, 1].transAxes, ha="center", va="center", fontsize=10)
    axes[0, 1].legend(frameon=False)
    for ax in axes[0]:
        ax.set_xticks(x, labels, rotation=70, ha="right")
    state = blocks.groupby(["subject_id", "state"])[["candidate_windows", "usable_windows"]].sum()
    for name, offset, color in [("EO", -.18, "#267997"), ("EC", .18, "#dda15e")]:
        grouped = state.xs(name, level="state").reindex(subjects.subject_id)
        rejection = 100 * (1 - grouped.usable_windows / grouped.candidate_windows)
        axes[1, 0].bar(x + offset, rejection, width=.36, color=color, label=name)
    axes[1, 0].set(title="EO/EC rejection balance (pooled rule)", ylabel="Rejected candidates (%)", ylim=(0, 105))
    axes[1, 0].set_xticks(x, labels, rotation=70, ha="right")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(np.arange(len(rates)), 100 * rates.retained_fraction, color="#267997")
    axes[1, 1].set_xticks(np.arange(len(rates)), [f"{int(row.native_sfreq)} Hz\nn={row.n_subjects}" for row in rates.itertuples()])
    axes[1, 1].set(title="Retention by native rate (descriptive)", ylabel="Usable candidates (%)", ylim=(0, 105))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    figure.suptitle("MPI-LEMON: outcome-blind engineering QC on provisional Train only", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, metadata={"Software": "matplotlib", "Description": "Persisted Train-only QC scalar summaries"})
    plt.close(figure)


from resting_eeg.data.paths import load_paths, relocated_path


def summarize(repo_root, raw_root, processed_root, metrics_root, config_path, metadata_root, figure_path):
    config, config_hash = load_config(config_path)
    current_code_hash = code_digest(repo_root)
    pilot_manifest = csv_frame(metrics_root / "pilot_download_manifest.csv")
    full_manifest = csv_frame(metrics_root / "full_candidate_download_manifest.csv")
    raw_inventory = audit_pilot_inventory(raw_root, pilot_manifest)
    require(raw_inventory["inventory_pass"], "Current raw pilot inventory is inconsistent")
    integrity = csv_frame(metrics_root / "pilot_loader_integrity.csv")
    require(set(integrity.subject_id) == set(PILOT_SUBJECTS) and len(integrity) == 15, "Loader integrity subject inventory differs")
    require(truth(integrity.integrity_pass).all(), "Pilot loader integrity did not pass")
    integrity = integrity.set_index("subject_id")
    gate = TrainAccessGate(metrics_root / "provisional_block_split_manifest.csv", metrics_root / "lemon_block_inventory.csv",
                           metrics_root / "pilot_waveform_access_registry.csv", repo_root)
    expected_by_subject, accumulators, subject_summaries, processed_rows, cleanup_rows = {}, {}, [], [], []
    names = ["signal_sanity", "bad_channel_metrics", "bad_channel_decisions", "window_qc", "block_qc", "qc_candidates"]
    for subject in PILOT_SUBJECTS:
        output = processed_root / f"{subject}_preprocessed.h5"
        receipt_path = output.with_suffix(".provenance.json")
        require(output.is_file() and receipt_path.is_file(), f"Pilot subject output/receipt not complete: {subject}")
        require(not output.with_suffix(".lock").exists(), f"Pilot subject is still being processed: {subject}")
        receipt = read_json(receipt_path)
        require(receipt["subject_id"] == subject and receipt["source_hash_verified"] is True and receipt["processed_verified"] is True,
                f"Invalid source/output verification receipt: {subject}")
        require(relocated_path(receipt["output_path"], repo_root) == output.resolve(), f"Receipt points to a different output: {subject}")
        require(receipt["config_sha256"] == config_hash and receipt["code_content_sha256"] == current_code_hash,
                f"Receipt differs from frozen configuration/source: {subject}")
        blocks = gate.train_blocks(subject)
        require(len(blocks) == 10 and Counter(block["state"] for block in blocks) == {"EO": 5, "EC": 5}, "Unexpected provisional Train design")
        expected_by_subject[subject] = [block["block_uid"] for block in blocks]
        sources = full_manifest.loc[full_manifest.subject_id == subject].to_dict("records")
        require(canonical_hash(sources) == canonical_hash(receipt["source_manifest_rows"]), f"Receipt/source manifest mismatch: {subject}")
        source_hashes = {}
        for source in sources:
            path = raw_root / source["relative_path"]
            require(path.is_file() and path.stat().st_size == source["payload_size_bytes"], f"Current raw payload size mismatch: {subject}")
            source_hashes[path.suffix] = source["payload_sha256"]
            prefix = "payload" if path.suffix == ".eeg" else "header"
            require(integrity.loc[subject, prefix + "_sha256"] == source["payload_sha256"], f"Loader/source SHA256 mismatch: {subject}")
        directory = raw_root / subject / "eeg"
        sidecars = {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file() and p.suffix not in (".eeg", ".vhdr")}
        require(sidecars == receipt["source_sidecar_sha256"] and len(sidecars) == 5, f"Metadata sidecar hash mismatch: {subject}")
        signature = canonical_hash(dict(config=config_hash, sources=sources, sidecars=sidecars, split=gate.split_hash,
                                        inventory=gate.inventory_hash, code=current_code_hash))
        require(signature == receipt["input_signature"], f"Current source signature mismatch: {subject}")
        subject_dir = metrics_root / "subjects" / subject
        tables = {name: csv_frame(subject_dir / f"{name}.csv") for name in names}
        bool_columns = {
            "bad_channel_decisions": ["catastrophic", "bad"],
            "window_qc": ["finite", "flat", "native_integrity_failure", "qc_reference_failed", "usable"],
            "block_qc": ["block_eligible"], "qc_candidates": ["selected"],
        }
        for name, columns in bool_columns.items():
            for column in columns:
                tables[name][column] = truth(tables[name][column])
        summary = read_json(subject_dir / "subject_qc.json")
        require(summary == receipt["qc_summary"], f"Scalar QC summary/receipt mismatch: {subject}")
        require(summary["native_sfreq"] == float(integrity.loc[subject, "native_sfreq"]), f"Subject native rate differs from verified header: {subject}")
        validate_subject_reports(subject, tables, summary, expected_by_subject[subject])
        native_names = set(json.loads(integrity.loc[subject, "native_channel_order"]))
        for _, group in tables["signal_sanity"].groupby("block_uid"):
            require(set(group.channel) == native_names, f"Native sanity channel inventory mismatch: {subject}")
        for _, group in tables["bad_channel_metrics"].groupby("block_uid"):
            require(set(group.channel) == native_names - {"VEOG"}, f"EEG metrics channel inventory mismatch: {subject}")
        require(set(tables["bad_channel_decisions"].channel) == native_names - {"VEOG"}, f"EEG decision channel inventory mismatch: {subject}")
        candidate_table = tables["qc_candidates"]
        require(set(candidate_table.multiplier) == set(config["qc"]["multiplier_candidates"]), "QC candidate values differ from frozen method")
        require((candidate_table.selected == (candidate_table.multiplier == config["qc"]["multiplier"])).all(), "QC selected multiplier differs from frozen method")
        for row in tables["block_qc"].itertuples():
            eligible = (row.usable_windows >= config["qc"]["minimum_usable_windows_per_block"]
                        and row.usable_windows / row.candidate_windows >= config["qc"]["minimum_usable_fraction_per_block"])
            require(row.block_eligible == eligible, f"Block eligibility differs from frozen rule: {row.block_uid}")
        expected_qc = (summary["threshold_fit_available"] and summary["bad_channel_limits_pass"]
                       and all(summary["usable_blocks_by_state"][state] >= config["qc"]["minimum_usable_Train_blocks_per_state"] for state in ("EO", "EC")))
        require(summary["qc_pass"] == expected_qc, f"Subject QC differs from frozen rule: {subject}")
        for name, table in tables.items():
            if "block_uid" in table:
                state_lookup = {block["block_uid"]: block["state"] for block in blocks}
                require((table.state == table.block_uid.map(state_lookup)).all(), f"Non-authoritative state labels: {subject}/{name}")
            if "native_sfreq" not in table:
                table["native_sfreq"] = summary["native_sfreq"]
            accumulators.setdefault(name, []).append(table)
        checked, stored_samples = verify_compact_metadata(output, receipt, blocks, tables, config, config_hash, source_hashes, gate)
        subject_summaries.append(summary)
        raw_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        cleanup_blockers = []
        if not metadata_root.is_dir():
            cleanup_blockers.append("read_only_metadata_clone_unavailable")
        if not summary["qc_pass"]:
            cleanup_blockers.append("subject_QC_failed_preserve_for_investigation")
        cleanup_rows.append(dict(
            subject_id=subject, raw_directory=str((raw_root / subject).resolve()), raw_bytes=raw_bytes,
            source_hash_verified=True, current_raw_sizes_match=True,
            compact_output_path=str(output.resolve()), compact_output_hash=checked["output_sha256"],
            processed_verified=True, qc_pass=summary["qc_pass"],
            eligible_for_raw_cleanup=not cleanup_blockers, cleanup_blocker=";".join(cleanup_blockers),
            cleanup_executed=False,
        ))
        processed_rows.append(dict(
            subject_id=subject, output_path=str(output.resolve()), output_bytes=checked["output_bytes"],
            output_sha256=checked["output_sha256"], config_sha256=config_hash, code_git_head=receipt["code_git_head"],
            code_content_sha256=current_code_hash, input_signature=signature,
            source_eeg_sha256=source_hashes[".eeg"], source_vhdr_sha256=source_hashes[".vhdr"],
            source_manifest="outputs/metrics/scenario2/full_candidate_download_manifest.csv",
            source_manifest_row_indices=json.dumps(full_manifest.index[full_manifest.subject_id == subject].tolist()),
            created_utc=receipt["created_utc"], native_sfreq=summary["native_sfreq"], processed_sfreq=128,
            n_blocks=len(blocks), n_stored_time_samples=stored_samples,
            uncompressed_eeg_bytes=19 * stored_samples * 4,
            physical_unit="uV", dtype="float32", format="HDF5", compression="gzip-4+shuffle",
            candidate_windows=summary["candidate_windows"], usable_windows=summary["usable_windows"],
            usable_blocks=summary["usable_blocks"], qc_pass=summary["qc_pass"], processed_verified=True,
            normalization_fitted=False, scope="provisional_Train_only",
        ))
        print(f"Verified completed pilot output and scalar reports: {subject}", flush=True)

    registry = csv_frame(gate.registry_path)
    access_summary = verify_registry(registry, gate, integrity, expected_by_subject)
    merged = {name: pd.concat(frames, ignore_index=True) for name, frames in accumulators.items()}
    subjects = pd.DataFrame(subject_summaries)
    blocks, windows, sanity, decisions = merged["block_qc"], merged["window_qc"], merged["signal_sanity"], merged["bad_channel_decisions"]
    processed, cleanup = pd.DataFrame(processed_rows), pd.DataFrame(cleanup_rows)
    rates = rate_summary(subjects, blocks, windows, sanity)
    target_sanity = sanity.loc[sanity.channel.isin(config["target_channels"] + ["VEOG"])].copy()
    require(len(target_sanity) == 3000, "Target+VEOG sanity report must contain 15x10x20 rows")
    for name, column in [("signal_sanity", "channel"), ("bad_channel_metrics", "channel"), ("bad_channel_decisions", "channel"),
                         ("window_qc", "window_start_sample"), ("block_qc", "block_uid"), ("qc_candidates", "multiplier")]:
        merged[name] = merged[name].sort_values(["subject_id"] + (["block_uid"] if "block_uid" in merged[name] and column != "block_uid" else []) + [column], kind="stable")
    scalar_subjects = subjects.copy()
    for name in ["bad_channels", "usable_blocks_by_state", "thresholds"]:
        scalar_subjects[name] = scalar_subjects[name].map(lambda value: json.dumps(value, sort_keys=True))
    csv_outputs = {
        "pilot_train_signal_sanity.csv": target_sanity,
        "pilot_bad_channel_metrics.csv": merged["bad_channel_metrics"],
        "pilot_bad_channel_decisions.csv": merged["bad_channel_decisions"],
        "pilot_qc_candidate_comparison.csv": merged["qc_candidates"],
        "pilot_train_window_qc.csv": merged["window_qc"], "pilot_train_block_qc.csv": merged["block_qc"],
        "pilot_train_subject_qc.csv": scalar_subjects,
        "pilot_processed_manifest.csv": processed, "pilot_raw_cleanup_manifest.csv": cleanup,
        "pilot_native_rate_qc_summary.csv": rates,
    }
    for filename, frame in csv_outputs.items():
        write_csv(metrics_root / filename, frame)

    state_summary = []
    for state, group in blocks.groupby("state"):
        candidates, retained = int(group.candidate_windows.sum()), int(group.usable_windows.sum())
        state_summary.append(dict(state=state, candidate_windows=candidates, usable_windows=retained,
                                  rejected_windows=candidates-retained, retained_fraction=retained/candidates,
                                  eligible_blocks=int(group.block_eligible.sum()), total_blocks=len(group)))
    reasons = Counter(reason for cell in windows.loc[~windows.usable, "reason"] for reason in str(cell).split("|"))
    eligible_block_ids = set(blocks.loc[blocks.block_eligible, "block_uid"])
    passing_subject_ids = set(subjects.loc[subjects.qc_pass, "subject_id"])
    failed_details = []
    for subject in subject_summaries:
        if not subject["qc_pass"]:
            failed_details.append(dict(
                subject_id=subject["subject_id"], failure_reason=subject["failure_reason"],
                eligible_Train_blocks_by_state=subject["usable_blocks_by_state"],
                required_eligible_Train_blocks_per_state=config["qc"]["minimum_usable_Train_blocks_per_state"],
                bad_channel_limits_pass=subject["bad_channel_limits_pass"],
                threshold_fit_available=subject["threshold_fit_available"],
                failing_states=[state for state in ("EO", "EC") if subject["usable_blocks_by_state"][state] < config["qc"]["minimum_usable_Train_blocks_per_state"]],
            ))
    nonfinite_rows = sanity.loc[sanity.finite_fraction < 1, ["subject_id", "block_uid", "channel", "finite_fraction", "nan_count", "inf_count"]]
    sanity_summary = dict(
        scope="provisional_Train_only", physical_unit="uV", mne_unit="V", volts_to_microvolts=1e6,
        n_subjects=15, n_unique_Train_blocks=150, n_all_channel_block_observations=len(sanity),
        n_target_plus_VEOG_block_observations=len(target_sanity), published_channels=config["target_channels"]+["VEOG"],
        all_62_channels_audited=True, minimum_finite_fraction=float(sanity.finite_fraction.min()),
        total_nan_count=int(sanity.nan_count.sum()), total_inf_count=int(sanity.inf_count.sum()),
        nonfinite_channel_blocks=nonfinite_rows.to_dict("records"),
        minimum_observed_uv=float(sanity.minimum.min()), maximum_observed_uv=float(sanity.maximum.max()),
        maximum_exact_zero_fraction=float(sanity.exact_zero_fraction.max()),
        maximum_flat_difference_fraction=float(sanity.flat_fraction.max()),
        maximum_extrema_repeat_fraction=float(sanity.extrema_repeat_fraction.max()),
        native_rate_groups=rates.to_dict("records"),
        interpretation="Broadband engineering sanity only; outlying amplitude alone is not an exclusion or physiological outcome.",
    )
    bad_summary = dict(
        n_subjects=15, n_decisions=len(decisions), n_bad_eeg=int(subjects.n_bad_eeg.sum()),
        n_bad_target=int(subjects.n_bad_target.sum()), n_subjects_with_bad_channels=int((subjects.n_bad_eeg > 0).sum()),
        max_bad_eeg_observed=int(subjects.n_bad_eeg.max()), max_bad_target_observed=int(subjects.n_bad_target.max()),
        n_subjects_within_interpolation_limits=int(subjects.bad_channel_limits_pass.sum()),
        fit_pool="subject_pooled_EO_EC_provisional_Train", state_specific_rules_used=False,
        bad_channels_by_subject={s["subject_id"]: s["bad_channels"] for s in subject_summaries},
        interpolation_limit_eeg=config["bad_channels"]["max_bad_eeg"],
        interpolation_limit_target=config["bad_channels"]["max_bad_target"],
    )
    qc_summary = dict(
        n_subjects=15, n_qc_pass=int(subjects.qc_pass.sum()), n_qc_fail=int((~subjects.qc_pass).sum()),
        candidate_windows=len(windows), usable_windows=int(windows.usable.sum()), rejected_windows=int((~windows.usable).sum()),
        usable_windows_in_eligible_blocks=int(windows.loc[windows.block_uid.isin(eligible_block_ids), "usable"].sum()),
        usable_windows_in_eligible_blocks_of_QC_passing_subjects=int(windows.loc[windows.block_uid.isin(eligible_block_ids) & windows.subject_id.isin(passing_subject_ids), "usable"].sum()),
        retained_fraction=float(windows.usable.mean()), n_Train_blocks=len(blocks), n_eligible_blocks=int(blocks.block_eligible.sum()),
        rejection_reason_counts=dict(sorted(reasons.items())), rejection_reasons_may_overlap=True,
        state_balance=state_summary, native_rate_groups=rates.to_dict("records"),
        selected_multiplier=config["qc"]["multiplier"], candidate_multipliers=config["qc"]["multiplier_candidates"],
        thresholds_fitted_on="subject_pooled_EO_EC_Train", thresholds_selected_using_heldout_data=False,
        failed_subjects=failed_details,
        scope="Engineering rejection balance only; no Alpha, Connectivity, classifier, or model outcomes computed.",
    )
    total_bytes = int(processed.output_bytes.sum())
    estimate_train = total_bytes * 211 / 15
    estimate_all = estimate_train * 16 / 10
    storage_summary = dict(
        pilot_subjects=15, pilot_Train_blocks=150, pilot_compact_bytes=total_bytes,
        pilot_compact_mib=total_bytes/1024**2, pilot_compact_gib=total_bytes/1024**3,
        pilot_raw_bytes=raw_inventory["raw_bytes"], pilot_raw_gib=raw_inventory["raw_gib"],
        pilot_uncompressed_target_eeg_bytes=int(processed.uncompressed_eeg_bytes.sum()),
        mean_compact_bytes_per_subject=total_bytes/15,
        estimated_211_Train_only_bytes=round(estimate_train), estimated_211_Train_only_gib=estimate_train/1024**3,
        estimated_211_all_16_blocks_bytes=round(estimate_all), estimated_211_all_16_blocks_gib=estimate_all/1024**3,
        all_blocks_to_Train_storage_multiplier=1.6,
        estimation_method="Observed pilot mean subject HDF5 bytes x 211; all-16-block planning estimate x 16/10.",
        estimation_limitations="Planning estimate assumes similar block durations/compression; pilot sampling-rate mix and later cohort/QC may differ. No held-out data processed.",
        format="HDF5", dtype="float32", channels=19, physical_unit="uV", sampling_rate_hz=128,
        compression="gzip level 4 + shuffle", chunks=[19,1024],
        overlapping_windows_materialized=False, window_representation="block-continuous arrays plus start-index/usability/reason datasets",
        final_normalization_fitted=False, raw_cleanup_executed=False,
        n_eligible_for_later_raw_cleanup=int(cleanup.eligible_for_raw_cleanup.sum()),
        raw_payload_hash_evidence="Every source SHA256 verified in loader audit and processing receipt; aggregate rechecks exact current sizes and sidecar hashes.",
        output_hashes_reverified=True,
    )
    synthetic = csv_frame(metrics_root / "pilot_resampling_validation.csv")
    require(set(synthetic.native_sfreq) == {500,1000,2500} and truth(synthetic.pass_validation).all()
            and truth(synthetic.deterministic).all(), "Synthetic resampling validation does not pass all native rates")
    resampling_summary = dict(
        synthetic_validation_path="outputs/metrics/scenario2/pilot_resampling_validation.csv",
        all_synthetic_rates_pass=True, target_sfreq=128, window_samples=512, stride_samples=256,
        edge_guard_sec=config["edge"]["guard_sec"], edge_guard_samples=config["edge"]["guard_samples_at_128"],
        maximum_combined_half_support_sec=float(synthetic.combined_half_support_sec.max()),
        maximum_passband_gain_error=float(np.abs(synthetic[["gain_5hz","gain_20hz","gain_35hz"]].to_numpy(float)-1).max()),
        maximum_60hz_residual=float(synthetic.residual_60hz.max()), maximum_100hz_residual=float(synthetic.residual_100hz.max()),
        all_real_compact_shapes_and_finiteness_verified=True, real_Train_native_rate_groups=rates.to_dict("records"),
        engineering_tones_hz=[5,20,35,60,100], physiological_spectral_outcomes_computed=False,
    )
    json_outputs = {
        "pilot_train_signal_sanity_summary.json": sanity_summary,
        "pilot_bad_channel_summary.json": bad_summary,
        "pilot_train_qc_summary.json": qc_summary,
        "pilot_storage_summary.json": storage_summary,
        "pilot_resampling_summary.json": resampling_summary,
        "pilot_waveform_access_registry_summary.json": access_summary,
    }
    for filename, summary in json_outputs.items():
        write_json(metrics_root / filename, summary)
    create_figure(figure_path, subjects, blocks, rates)
    report = dict(pilot_subjects=15, integrity_pass=15, processed_verified=15, qc_pass=int(subjects.qc_pass.sum()),
                  unique_Train_blocks_accessed=access_summary["n_unique_train_blocks"], access_events=len(registry),
                  compact_bytes=total_bytes, estimated_211_all_16_blocks_gib=estimate_all/1024**3,
                  raw_cleanup_executed=False, raw_waveform_values_read_by_summary=False,
                  config_sha256=config_hash, code_content_sha256=current_code_hash,
                  output_csv_bytes={name:(metrics_root/name).stat().st_size for name in csv_outputs},
                  figure=str(figure_path.relative_to(repo_root)))
    write_json(metrics_root / "pilot_aggregation_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--raw-root", type=Path, default=load_paths(Path(__file__).resolve().parents[2])["raw_root"])
    parser.add_argument("--processed-root", type=Path, default=load_paths(Path(__file__).resolve().parents[2])["processed_root"]/"train-only-v1")
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/metrics/scenario2"))
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing/frozen_preprocessing_v1.yaml"))
    parser.add_argument("--metadata-root", type=Path, default=load_paths(Path(__file__).resolve().parents[2])["metadata_root"])
    parser.add_argument("--figure", type=Path, default=Path("outputs/figures/scenario2/pilot_train_qc_summary.png"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    local = lambda path: path.resolve() if path.is_absolute() else (root / path).resolve()
    result = summarize(root, local(args.raw_root), local(args.processed_root), local(args.metrics_root),
                       local(args.config), args.metadata_root.resolve(), local(args.figure))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
