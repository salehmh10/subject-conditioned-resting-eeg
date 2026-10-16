"""Copy an explicit, compact evidence allowlist without executing research code.

This maintainer tool reads saved text/metadata only. It never imports the research
package, reads EEG arrays or checkpoints, or recomputes a scientific result.
Original file bytes (including historical paths and line endings) are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


UTILITY = "outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es"
REPORT = "reports/supervisor_eeg_synthetic_report"
FROZEN = "outputs/frozen/scenario2"
PROTOCOL = "outputs/protocols/scenario2"


def inclusion_list() -> list[tuple[str, str, str]]:
    """Return (source-relative, publication-relative, kind), no recursive copy."""
    items: list[tuple[str, str, str]] = []

    def add(source: str, destination: str, kind: str) -> None:
        items.append((source, destination, kind))

    for name in (
        "fidelity_representative.csv", "utility_contrasts.csv",
        "utility_learning_curves.csv", "utility_stopping.csv",
        "utility_summary.csv", "utility_test_all_55_fits.csv",
        "utility_validation_all_55_fits.csv", "calibration_table.tex",
        "fidelity_table.tex", "final_test_table.tex",
    ):
        add(f"{REPORT}/tables/{name}", f"results/tables/{name}", "saved_paper_table")
    add(f"{REPORT}/source_map.csv", "evidence/provenance/paper_source_map.csv", "historical_paper_source_map")

    for name in ("all_55_classifier_validation_runs.csv", "s2_75b_v2_complete_validation_report.json"):
        add(f"{UTILITY}/final_validation/{name}", f"evidence/utility/validation/{name}", "saved_validation_result")
    for name in (
        "s2_77_v2_complete_final_test_report.json",
        "tables/all_55_final_test_fits.csv", "tables/all_11_condition_summary.csv",
        "tables/frozen_paired_subject_contrasts.csv",
        "manifests/s2_77_final_test_complete.json", "manifests/s2_77_single_test_access.json",
    ):
        add(f"{UTILITY}/final_test_s2_77_v1/{name}", f"evidence/utility/test/{name}", "saved_final_test_result")

    for folder, stem, target in (
        ("final_pool_fidelity_s2_74b_v1", "s2_74b", "b1_b2"),
        ("b0_fidelity_extension_s2_74e_v1", "s2_74e_b0", "b0_extension"),
    ):
        source = f"outputs/runs/scenario2/characterization/{folder}"
        report = f"{stem}_complete_report_v1.json"
        plan = f"{stem}_analysis_plan_v1.json"
        add(f"{source}/metrics/{report}", f"evidence/fidelity/{target}/{report}", "saved_aggregate_fidelity")
        add(f"{source}/protocol/{plan}", f"evidence/fidelity/{target}/{plan}", "frozen_fidelity_protocol")

    frozen_files = (
        "b0/b0_v_raw_epoch020_v1/metrics/b0_frozen_selection_v1.json",
        "b0/b0_v_raw_epoch020_v1/metrics/b0_frozen_selection_summary_v1.txt",
        "b0/b0_v_raw_epoch020_v1/handoff/b0_to_b1_handoff_v1.json",
        "b1/b1_subject_state_raw_epoch010_v1/metrics/b1_frozen_selection_v1.json",
        "b1/b1_subject_state_raw_epoch010_v1/metrics/b1_frozen_selection_summary_v1.txt",
        "b1/b1_subject_state_raw_epoch010_v1/handoff/b1_to_b2_handoff_v1.json",
        "b2/b2_validation_selected_replication_e1_v1/metrics/b2_frozen_selection_v1.json",
        "b2/b2_validation_selected_replication_e1_v1/metrics/b2_frozen_selection_summary_v1.txt",
        "b2/b2_validation_selected_replication_e1_v1/metrics/b2_freeze_manifest_v1.json",
        "b2/b2_validation_selected_replication_e1_v1/handoff/b2_to_utility_handoff_v1.json",
        "utility/b1_b2_pool_integrity_half_subset_v1/manifests/s2_74_freeze_manifest_v1.json",
        "ablation/b0_pool_integrity_half_subset_s2_74d_v1/manifests/s2_74d_b0_freeze_manifest_v1.json",
        "utility/s2_76_v2_final_test_preflight_v1/manifests/s2_76_v2_pre_final_test_freeze_manifest.json",
    )
    for name in frozen_files:
        add(f"{FROZEN}/{name}", f"evidence/frozen/scenario2/{name}", "frozen_model_or_utility_metadata")

    protocol_files = (
        "utility/b1_vs_b2_utility_protocol_v1/metrics/s2_72_b1_vs_b2_downstream_utility_protocol_v1.json",
        "utility/b1_vs_b2_utility_protocol_v1/metrics/s2_72_b1_vs_b2_downstream_utility_protocol_manifest_v1.json",
        "utility/b0_amended_11_condition_120epoch_es_v2/manifests/s2_75a_v2_120epoch_es_amendment.json",
        "b2/sampling_robust_checkpoint_selector_v2/metrics/s2_69_sampling_robust_selector_protocol_v2.json",
    )
    for name in protocol_files:
        add(f"{PROTOCOL}/{name}", f"evidence/protocols/scenario2/{name}", "historical_frozen_protocol")

    for name in ("train_normalization.json", "final_cohort_split_lock.json"):
        add(f"outputs/metrics/scenario2/{name}", f"evidence/preprocessing/{name}", "frozen_preprocessing_metadata")
    return sorted(items)


OMITTED = [
    {"category": "EEG arrays and checkpoints", "reason": "Raw/processed EEG, synthetic pools, model weights, fitted classifiers and window predictions are excluded. No such artifacts were opened or rehashed for this export. Their recorded references/hashes remain in historical frozen metadata."},
    {"category": "Participant result diagnostics", "reason": "Per-subject utility tables, fidelity subject JSON/CSV files, waveform previews and individual predictions are not part of this compact evidence export. Historical public-pseudonym cohort/QC metadata already tracked in Git is retained separately for provenance."},
    {"category": "Detailed cohort assignments are not duplicated", "source_paths": ["outputs/metrics/scenario2/final_block_split_manifest.csv", "outputs/metrics/scenario2/final_cohort_manifest.csv", "outputs/metrics/scenario2/final_cohort_split_summary.json"], "reason": "These original tracked paths are retained in the publication repository and covered by provenance/source_inventory.json. The included cohort lock also records their original sizes and SHA256s. This evidence subtree does not duplicate them."},
    {"category": "Synthetic subset row indices", "reason": "Per-subject/state nested-subset view and master files are omitted. The included s2_74/s2_74d freeze manifests retain subset counts and original file hashes."},
    {"category": "Training receipts and caches", "reason": "Individual run receipts, cache manifests and generation journals are omitted. Saved aggregate result tables and previously extracted all-seed learning/stopping tables are included unchanged; this export does not independently reconstruct them."},
    {"category": "Aggregate plot source details", "reason": "Participant-level PSD/connectivity arrays used by some historical figures are omitted. Existing figures may be included elsewhere as saved presentation assets, but cannot all be reconstructed from this compact evidence set."},
    {"category": "Historical source-map dependencies", "reason": "paper_source_map.csv is copied unchanged and describes original workspace paths, including intentionally omitted dependencies. Resolve included source paths through this inventory. It is not a claim that every mapped dependency is shipped."},
]


def export(source: Path, destination: Path) -> dict:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination or source.is_relative_to(destination):
        raise ValueError("Destination must be a separate publication directory, not the source or its ancestor")
    files = []
    for source_path, public_path, kind in inclusion_list():
        original = (source / source_path).resolve(strict=True)
        target = (destination / public_path).resolve()
        if not original.is_relative_to(source) or not target.is_relative_to(destination):
            raise ValueError("Resolved path leaves its declared root")
        data = original.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different publication bytes: {public_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        files.append({"path": public_path, "source_path": source_path,
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "size_bytes": len(data), "kind": kind})
    manifest = {
        "schema_version": "publication-evidence-inventory-v1",
        "scope": "Explicit copy of saved final-study aggregate evidence; no research execution or statistical recalculation",
        "source_path_base": "Original research repository root; source_path values are repository-relative",
        "byte_preservation": "Original content copied byte-for-byte, including historical paths and line endings",
        "historical_context": "Earlier frozen records describe their then-current state (for example Test still sealed). The s2_77 completion receipt records the later completed Final Test.",
        "deliberate_pseudonymous_metadata": [
            "The frozen B1 handoff preserves public dataset pseudonyms in subject embedding index order.",
            "The Train-only normalization JSON preserves public dataset pseudonyms with sample counts needed to describe the frozen scaler. No waveform arrays, participant clinical records or participant-level predictive performance are exported.",
        ],
        "validation_scope": "Only copy integrity (SHA256/size) is checked during publication preparation. Prior scientific values are not re-evaluated.",
        "files": sorted(files, key=lambda item: item["path"]),
        "omitted_dependencies": OMITTED,
    }
    inventory = destination / "evidence/provenance/evidence_inventory.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original research root (read-only)")
    parser.add_argument("--destination", type=Path, required=True, help="Separate publication root")
    args = parser.parse_args()
    manifest = export(args.source, args.destination)
    print(f"Copied {len(manifest['files'])} saved evidence files; no research execution.")


if __name__ == "__main__":
    main()
