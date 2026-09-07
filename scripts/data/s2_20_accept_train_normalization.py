"""Accept the single clean local normalization commit after s2_19 passes.

Read saved test/review/post-commit verifier evidence; write only the external
acceptance record. No EEG is accessed, no Git commit is made and nothing is pushed.
"""
from pathlib import Path
import argparse
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from resting_eeg.common import sha256_file, utc_now
from resting_eeg.data import final_cohort_split as frozen
from resting_eeg.data import train_normalization as norm
from resting_eeg.data import train_normalization_verify as check
from resting_eeg.data.paths import load_paths


def validate_reviews(reviews):
    norm.require(set(reviews) == {"data_leakage", "code_test"}, "Both independent reviews are required")
    for review in reviews.values():
        norm.require(review.get("status") == "passed" and review.get("unresolved_findings") == [], "Unresolved reviewer findings")
        norm.require(set(review.get("findings", [])) <= set(review.get("resolutions", [])), "Reviewer resolution missing")


def validate_postcommit_report(report, head):
    norm.require(report.get("verification") == "passed", "Verifier did not pass")
    repository = report.get("repository", {})
    norm.require(repository.get("current_head") == head and repository.get("worktree_clean") is True
                 and repository.get("worktree_clean_required") is True, "Verifier is not for this clean committed HEAD")
    norm.require(report.get("final_test_sealed") is True, "Verifier did not preserve Final Test seal")


def accept(root, evidence):
    # 1. Require exactly one non-merge commit from the accepted starting state.
    root, evidence = Path(root).resolve(), Path(evidence).resolve()
    canonical = load_paths(root)["runtime_root"] / "docs/train_normalization_lock_20260907"
    norm.require(evidence == canonical.resolve(), "Acceptance evidence directory must be canonical")
    repository = frozen.validate_repository_boundaries(root, require_clean=True)
    head = repository["current_head"]
    norm.require(frozen._git_output(root, "branch", "--show-current") == "main", "Acceptance branch is not main")
    norm.require(frozen._git_output(root, "rev-list", "--parents", "-n", "1", head).split()
                 == [head, norm.STARTING_HEAD], "Exactly one new non-merge local commit is required")
    inputs = norm.load_locked_inputs(root)
    output = root / norm.METRICS
    lock = check.verify_locked_artifact_hashes(output)
    for name, digest in lock["implementation_sha256"].items():
        frozen.validate_file_hash(root / name, digest, "implementation")

    # 2. Check full guarded tests, compilation and all saved verifier exit codes.
    exit_names = ["fit", "pytest", "compileall", "normalization_verifier_precommit",
                  "normalization_verifier_postcommit", "final_cohort_verifier_precommit",
                  "final_cohort_verifier_postcommit", "full_cohort_verifier_precommit",
                  "full_cohort_verifier_postcommit"]
    exits = {name: int((evidence / f"{name}_exit_code.txt").read_text(encoding="utf-8-sig").strip()) for name in exit_names}
    norm.require(all(value == 0 for value in exits.values()), "A required command failed")
    junit_path = evidence / "pytest.xml"
    norm.require(sha256_file(junit_path) == sha256_file(root / "outputs/logs/pytest.xml"), "Stale JUnit copy")
    junit = frozen.junit_summary(junit_path)
    frozen.validate_full_junit(junit, minimum_tests=422)
    new_tests = {case.attrib["name"].split("[")[0] for case in ET.parse(junit_path).iter("testcase")
                 if "test_train_normalization" in case.attrib.get("classname", "")}
    norm.require({"test_overlapping_windows_count_once", "test_disjoint_window_unions",
                  "test_partially_overlapping_chains", "test_ineligible_train_contributes_zero",
                  "test_heldout_block_rejected", "test_excluded_subject_rejected", "test_wrong_channel_order_fails",
                  "test_missing_hdf5_provenance_fails", "test_source_hash_mismatch_fails", "test_nonfinite_eeg_fails",
                  "test_tiny_std_fails", "test_duplicate_block_accounting_fails", "test_missing_train_assignment_fails",
                  "test_deterministic_rerun_canonical_artifacts", "test_locked_input_loader_integration_with_committed_metadata"}
                 <= new_tests, "Required normalization tests missing from full suite")

    # 3. Require independent reviews and clean post-commit preservation evidence.
    reviews = frozen.read_json(evidence / "review_results.json")
    validate_reviews(reviews)
    normalization = frozen.read_json(evidence / "normalization_verifier_postcommit.json")
    final_cohort = frozen.read_json(evidence / "final_cohort_verifier_postcommit.json")
    full_cohort = frozen.read_json(evidence / "full_cohort_verifier_postcommit.json")
    for report in (normalization, final_cohort):
        validate_postcommit_report(report, head)
    frozen.validate_full_cohort_verifier(full_cohort, head, junit["tests"])
    norm.require(normalization["deterministic_regeneration"] == "byte_identical", "Determinism not verified")
    hashes = {name: sha256_file(output / name) for name in (norm.SCALER_FILE, norm.BLOCK_FILE, norm.LOCK_FILE)}
    norm.require(normalization["artifact_sha256"] == hashes and final_cohort["artifact_sha256"] == norm.FINAL_HASHES,
                 "Post-commit artifact hashes changed")
    scaler = frozen.read_json(output / norm.SCALER_FILE)
    second = normalization["second_pass"]
    norm.require(second["verification"] == "passed"
                 and second["maximum_absolute_normalized_mean"] <= check.NORMALIZED_TOLERANCE
                 and second["maximum_absolute_normalized_std_deviation"] <= check.NORMALIZED_TOLERANCE,
                 "Second-pass numerical verification failed")

    # 4. Bind the exact committed outcome outside Git (avoids a self-reference).
    result = dict(status=norm.STATUS, accepted_utc=utc_now(), starting_head=norm.STARTING_HEAD,
                  final_committed_head=head, branch="main", one_new_local_nonmerge_commit=True,
                  normalization_artifact_hashes=hashes, final_cohort_split_input_hashes=norm.FINAL_HASHES,
                  accepted_final_cohort_record_sha256=inputs["accepted_record_sha256"],
                  counts_and_duration=normalization["counts_and_duration"], channel_order=norm.CHANNELS,
                  means_uv=scaler["means_uv"], stds_uv=scaler["stds_uv"], second_pass=second,
                  tests=junit, normalization_test_names=sorted(new_tests), exit_codes=exits,
                  compileall="passed", verifier_results={"normalization": normalization,
                  "final_cohort_split": final_cohort, "full_cohort_preservation": full_cohort}, reviewer_results=reviews,
                  worktree_clean=True, git_push_performed=False,
                  no_push_confirmation="No git push command executed; application-level confirmation.",
                  final_test_sealed=True, scientific_heldout_outcomes_accessed=False, batch_006_exists=False,
                  historical_verifier_note="Prior no_normalization flags describe unchanged preprocessing/HDF5 evidence; the new global scaler is separate.",
                  evidence_file_sha256={p.name: sha256_file(p) for p in sorted(evidence.iterdir()) if p.is_file()},
                  next_action="REAL-DATA BASELINES")
    destination = evidence / "final_acceptance.json"
    norm.require(not destination.exists(), "External acceptance already exists; preserve it")
    frozen.atomic_write(destination, frozen.json_bytes(result))
    return dict(status=result["status"], final_committed_head=head, acceptance_path=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(accept(ROOT, args.evidence_directory), indent=2))


if __name__ == "__main__":
    main()
