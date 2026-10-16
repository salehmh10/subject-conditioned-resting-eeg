"""Evidence parser/contract regression tests, using saved artifacts where available."""
import copy
import json
from pathlib import Path
import re
import statistics
import sys
import warnings
import zipfile

import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from lab.adapters import array_mean, metric_definition
from lab.sources import EvidenceError, EvidenceReader, discover_sources, safe_relative_path, safe_zip_json, unique_object
from lab.validation import acceptance_anchors, assert_unique, fraction_to_percent, normalize_contrast, validate_bundle


@pytest.fixture(scope="module")
def bundle():
    path = APP / "data" / "presentation_bundle" / "bundle.json"
    if not path.exists():
        pytest.skip("Run preparation before actual evidence integration tests")
    return json.loads(path.read_text(encoding="utf-8"))


def test_duplicate_json_keys_rejected():
    with pytest.raises(EvidenceError, match="Duplicate JSON key"):
        json.loads('{"state":"EO","state":"EC"}', object_pairs_hook=unique_object)


def test_duplicate_and_missing_record_identity():
    with pytest.raises(EvidenceError, match="Duplicate"):
        assert_unique([{"state": "EO"}, {"state": "EO"}], ("state",))
    with pytest.raises(EvidenceError, match="Missing identity"):
        assert_unique([{"state": None}], ("state",))


def test_missing_file_is_actionable(tmp_path):
    with pytest.raises(EvidenceError, match="Missing saved evidence"):
        EvidenceReader(tmp_path).json("missing.json")


def test_reader_does_not_open_raw_arrays(tmp_path):
    (tmp_path / "signal.h5").write_bytes(b"test fixture only")
    with pytest.raises(EvidenceError, match="Unsupported"):
        EvidenceReader(tmp_path).read_bytes("signal.h5")


def test_source_hash_changes_detected(tmp_path):
    source = tmp_path / "test.json"
    source.write_text('{"value":1}', encoding="utf-8")
    reader = EvidenceReader(tmp_path)
    reader.json("test.json")
    source.write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="hash changed"):
        reader.verify_unchanged()


@pytest.mark.parametrize("path", ["../x.json", "..\\x.json", "/x.json", "C:\\x.json", "//server/file.json", "a/../x.json", "a/./x.json"])
def test_unsafe_source_paths_rejected(path):
    with pytest.raises(EvidenceError):
        safe_relative_path(path)


def test_zip_windows_separators_without_extraction(tmp_path):
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("evidence\\reports\\fidelity.json", '{"mean":1}')
    assert safe_zip_json(archive, "evidence/reports/fidelity.json") == {"mean": 1}
    assert len(list(tmp_path.iterdir())) == 1


def test_zip_rejects_traversal_anywhere(tmp_path):
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("safe.json", "{}")
        handle.writestr("..\\escape.json", "{}")
    with pytest.raises(EvidenceError, match="Traversal"):
        safe_zip_json(archive, "safe.json")


def test_zip_rejects_normalized_duplicate(tmp_path):
    archive = tmp_path / "evidence.zip"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("a/b.json", "{}")
            handle.writestr("a\\b.json", "{}")
    with pytest.raises(EvidenceError, match="Duplicate normalized"):
        safe_zip_json(archive, "a/b.json")


def test_percent_and_percentage_point_conversion():
    assert fraction_to_percent(0.88417) == pytest.approx(88.417)
    assert fraction_to_percent(0.00718) == pytest.approx(0.718)
    assert fraction_to_percent(None) is None


def test_contrast_sign_and_ci_reversal():
    assert normalize_contrast(-0.00718, -0.01201, -0.00250, "right_minus_left") == pytest.approx((0.00718, 0.00250, 0.01201))
    assert normalize_contrast(0.002, -0.003, 0.008, "left_minus_right") == (0.002, -0.003, 0.008)
    with pytest.raises(EvidenceError, match="endpoints"):
        normalize_contrast(0, 1, -1, "left_minus_right")


def test_array_mean_preserves_negative_and_missing():
    assert array_mean([[None, -0.03, None], [None, -0.01, 0.2]]) == [None, -0.02, 0.2]
    with pytest.raises(EvidenceError, match="shapes"):
        array_mean([[1], [1, 2]])


def test_sd_md_is_microvolts_and_diversity_is_parity():
    assert metric_definition("sd_md_pooled_uv")[1:] == ("uV", "lower")
    assert metric_definition("covariance_relative_frobenius_z")[1] == "dimensionless"
    assert metric_definition("rms_pairwise_diversity_ratio")[2] == "parity"
    assert metric_definition("nearest_real_rmse_z_median")[2] == "descriptive"


def test_actual_bundle_contract_and_anchors(bundle):
    assert validate_bundle(bundle)["status"] == "passed"
    assert all(row["passed"] for row in acceptance_anchors(bundle))


def test_missing_condition_rejected(bundle):
    altered = dict(bundle)
    altered["utility_runs"] = bundle["utility_runs"][:-1]
    with pytest.raises(EvidenceError, match="55"):
        validate_bundle(altered)


def test_duplicate_real_fit_rejected(bundle):
    altered = dict(bundle)
    altered["utility_runs"] = bundle["utility_runs"] + [bundle["utility_runs"][0]]
    with pytest.raises(EvidenceError, match="Duplicate"):
        validate_bundle(altered)


def test_saved_population_sd_is_not_sample_sd(bundle):
    fit_values = [r["subject_macro_balanced_accuracy"] for r in bundle["utility_runs"]
                  if r["split"] == "final_test" and r["family"] == "EEGNETV4" and r["condition"] == "REAL_ONLY"]
    summary = next(r for r in bundle["utility_summary"] if r["split"] == "final_test" and r["family"] == "EEGNETV4" and r["condition"] == "REAL_ONLY")
    assert summary["population_sd"] == pytest.approx(statistics.pstdev(fit_values), abs=1e-14)
    assert summary["population_sd"] != pytest.approx(statistics.stdev(fit_values))


def test_no_old_utility_evidence(bundle):
    for source in bundle["provenance"]["sources"].values():
        if "/utility/" in source["path"] and source["role"] != "study metadata; no utility performance":
            assert "eleven_condition_s2_75_v2_120epoch_es/" in source["path"]
    assert len(bundle["histories"]) == 33
    assert all(max(e["epoch"] for e in r["epochs"]) == r["stopped_epoch"] for r in bundle["histories"])


def test_actual_matrices_keep_negative_values_and_mask_diagonal(bundle):
    matrix = bundle["connectivity"]["participants"]["sub-010002"]["EO"]["alpha"]["B1"]
    assert all(matrix[i][i] is None for i in range(19))
    assert any(value < 0 for row in matrix for value in row if value is not None)
    assert len(bundle["psd"]["frequencies_hz"]) == 129


def test_all_record_provenance_resolves(bundle):
    transforms = bundle["provenance"]["transforms"]
    sources = bundle["provenance"]["sources"]
    for section in ("fidelity_metrics", "calibration", "fidelity_subjects", "fingerprint", "fingerprint_subjects",
                    "connectivity_metrics", "utility_runs", "utility_summary", "contrasts", "histories", "utility_subjects"):
        for row in bundle[section]:
            transform = transforms[row["provenance"]["transform_id"]]
            assert all(source_id in sources for source_id in transform["source_ids"])
            assert all(key in transform for key in ("units", "aggregation", "analysis_split", "transformation"))


def test_shared_aggregate_bundle_has_explicit_reduced_capabilities(bundle):
    sys.path.insert(0, str(APP / "tools"))
    from build_distribution import aggregate_bundle
    shared = aggregate_bundle(bundle)
    for section in ("psd", "connectivity", "alpha_effect"):
        assert not any(key in shared[section] for key in ("participants", "valid_frequency_bins", "sources_by_subject"))
    for key in ("fidelity_subjects", "fingerprint_subjects", "utility_subjects"):
        assert shared[key] == []
    assert len(shared["connectivity_metrics"]) == 18
    for row in shared["connectivity_metrics"]:
        assert "subject_id" not in row and row["n_subjects"] == 198
        originals = [r for r in bundle["connectivity_metrics"] if all(r[k] == row[k] for k in ("state", "band", "model"))]
        assert row["upper_mae"] == pytest.approx(statistics.mean(r["upper_mae"] for r in originals), abs=1e-14)
        assert "Equal-participant" in row["aggregation"]
        assert row["provenance"]["transform_id"] in shared["provenance"]["transforms"]
    assert shared["availability"]["waveform_previews"] is False
    assert shared["availability"]["participant_connectivity"] is False
    assert shared["availability"]["group_connectivity"] is True
    assert shared["availability"]["external_preview_bundle"] is None
    assert "omits" in shared["availability"]["waveform_reason"]
    text = json.dumps(shared)
    assert re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text) is None
