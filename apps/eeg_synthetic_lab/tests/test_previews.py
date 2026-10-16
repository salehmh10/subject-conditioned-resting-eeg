"""Guarded preview tests use artificial fixtures only inside the test directory."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
spec = importlib.util.spec_from_file_location("bounded_previews", APP / "tools/export_train_previews.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def row():
    return {"split": "Train", "final_lock_status": "LOCKED_UNCHANGED_PROVISIONAL_ROLE", "block_eligible": "True",
            "subject_id": "fixture-participant", "block_uid": "fixture-block", "eye_state": "EO"}


class DatasetSpy:
    shape = (19, 1024)

    def __init__(self):
        self.slices = []

    def __getitem__(self, key):
        self.slices.append(key)
        return np.zeros((19, key[1].stop - key[1].start), dtype=np.float32)


class BlockSpy:
    def __init__(self, split="Train", unit="uV"):
        self.attrs = {"provisional_split": split, "block_eligible": True,
                      "subject_id": "fixture-participant", "block_uid": "fixture-block", "state": "EO",
                      "channel_names": json.dumps(preview.CHANNELS), "physical_unit": unit, "processed_sfreq": 128}
        self.lookups = []
        self.dataset = DatasetSpy()

    def __getitem__(self, key):
        self.lookups.append(key)
        return self.dataset


@pytest.mark.parametrize("split", ["Validation", "Test", "Final Test", "train", ""])
def test_manifest_guard_precedes_dataset_lookup(split):
    r, b = row(), BlockSpy()
    r["split"] = split
    with pytest.raises(preview.PreviewGuardError, match="non-Train"):
        preview.guarded_real_slice(r, b, 0)
    assert not b.lookups


@pytest.mark.parametrize("split", ["Validation", "Test"])
def test_hdf5_guard_precedes_dataset_lookup(split):
    b = BlockSpy(split=split)
    with pytest.raises(preview.PreviewGuardError, match="non-Train"):
        preview.guarded_real_slice(row(), b, 0)
    assert not b.lookups


@pytest.mark.parametrize("attribute,value", [("state", "EC"), ("subject_id", "different"), ("block_uid", "different"),
                                             ("block_eligible", False), ("physical_unit", "V"),
                                             ("processed_sfreq", 250), ("channel_names", '["O1"]')])
def test_identity_units_and_geometry_fail_before_dataset_lookup(attribute, value):
    b = BlockSpy()
    b.attrs[attribute] = value
    with pytest.raises(preview.PreviewGuardError):
        preview.guarded_real_slice(row(), b, 0)
    assert not b.lookups


@pytest.mark.parametrize("start", [-1, 513, 1024, 1.2])
def test_window_boundary_rejected_without_waveform_slice(start):
    b = BlockSpy()
    with pytest.raises(preview.PreviewGuardError, match="bounds"):
        preview.guarded_real_slice(row(), b, start)
    assert not b.dataset.slices


def test_exact_boundary_and_bounded_slice():
    b = BlockSpy()
    result = preview.guarded_real_slice(row(), b, 512)
    assert result.shape == (19, 512)
    assert b.dataset.slices == [(slice(None), slice(512, 1024))]


def test_selection_is_sorted_deterministic_even_and_bounded():
    subjects = [f"fixture-{i:03d}" for i in range(198)]
    selected = preview.select_subjects(subjects[::-1])
    assert selected == preview.select_subjects(subjects + subjects)
    assert len(selected) == 12 and selected[0] == subjects[0] and selected[-1] == subjects[-1]
    gaps = np.diff([subjects.index(x) for x in selected])
    assert gaps.max() - gaps.min() <= 1
    assert preview.evenly_spaced_indices(2, 3) == [0, 1]
    assert preview.evenly_spaced_indices(125, 3) == [0, 62, 124]


def test_missing_source_and_duplicate_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        preview.load_json(tmp_path / "missing.json")
    p = tmp_path / "duplicate.json"
    p.write_text('{"key": 1, "key": 2}', encoding="utf-8")
    with pytest.raises(preview.PreviewGuardError, match="Duplicate"):
        preview.load_json(p)


def synthetic_fixture(tmp_path, **updates):
    scaler = {"fit_split": "Train", "ddof": 0, "channel_order": preview.CHANNELS,
              "means_uv": [3.0] * 19, "stds_uv": [2.0] * 19}
    metadata = {"generator_name": "B1", "subject_id": "fixture-participant", "eye_state": "EO", "array_key": "x_z",
                "signal_domain": "frozen_train_global_z", "normalization_sha256": "fixture-scaler-hash",
                "channel_order": preview.CHANNELS, "sampling_rate_hz": 128, "n_channels": 19, "n_times": 512,
                "expected_count": 5}
    metadata.update(updates)
    z = np.stack([np.full((19, 512), i, dtype=np.float32) for i in range(5)])
    p = tmp_path / "fixture.npz"
    np.savez_compressed(p, x_z=z, metadata_json=json.dumps(metadata))
    return p, scaler


def test_standardized_synthetic_inverse_transform_once(tmp_path):
    p, scaler = synthetic_fixture(tmp_path)
    values = preview.read_synthetic_slices(p, "B1", "fixture-participant", "EO", [0, 2, 4], 5, scaler, "fixture-scaler-hash")
    np.testing.assert_array_equal(values[:, 0, 0], [3, 7, 11])
    assert values.shape == (3, 19, 512)


@pytest.mark.parametrize("indices", [[-1], [5], [0, 0], [0, 1, 2, 3]])
def test_synthetic_slice_bounds(tmp_path, indices):
    p, scaler = synthetic_fixture(tmp_path)
    with pytest.raises(preview.PreviewGuardError):
        preview.read_synthetic_slices(p, "B1", "fixture-participant", "EO", indices, 5, scaler, "fixture-scaler-hash")


@pytest.mark.parametrize("changes", [{"signal_domain": "uV"}, {"normalization_sha256": "wrong"}, {"subject_id": "wrong"}])
def test_synthetic_metadata_required(tmp_path, changes):
    p, scaler = synthetic_fixture(tmp_path, **changes)
    with pytest.raises(preview.PreviewGuardError):
        preview.read_synthetic_slices(p, "B1", "fixture-participant", "EO", [0], 5, scaler, "fixture-scaler-hash")


def test_npz_bytes_are_deterministic(tmp_path):
    one, two = tmp_path / "one.npz", tmp_path / "two.npz"
    values = {"data": np.arange(30).reshape(2, 15)}
    preview.deterministic_npz(one, values)
    preview.deterministic_npz(two, values)
    assert one.read_bytes() == two.read_bytes()
    with np.load(one, allow_pickle=False) as d:
        np.testing.assert_array_equal(d["data"], values["data"])


def test_actual_preview_bundle_schema_and_provenance():
    p = APP / "data/presentation_bundle/previews.json"
    if not p.exists():
        pytest.skip("Real previews have not been prepared")
    meta = preview.load_json(p)
    if not meta["available"]:
        assert meta["reason"] and meta["action"]
        return
    assert meta["schema_version"] == "eeg_synthetic_lab.previews.v1"
    assert meta["sources"] == ["Real", "B0", "B1", "B2"]
    assert meta["units"] == "uV" and meta["fs_hz"] == 128
    records = meta["records"]
    assert len(records) <= 72
    keys = [(r["participant_id"], r["state"], r["slot"]) for r in records]
    assert len(keys) == len(set(keys))
    assert all(r["analysis_split"] == "Train" for r in records)
    assert meta["waveform_reads"]["Validation"] == meta["waveform_reads"]["Final_Test"] == 0
    assert meta["source_hashes_unchanged_after_export"]
    assert not meta["pointwise_pairing"] and not meta["raw_HDF5_runtime_dependency"]
    path = p.parent / meta["file"]
    assert preview.sha256(path) == meta["npz_sha256"]
    with np.load(path, allow_pickle=False) as data:
        assert data["waveforms_uv"].shape == (len(records), 4, 19, 512)
        assert data["psd_uv2_hz"].shape == (len(records), 4, 19, 129)
        assert np.isfinite(data["waveforms_uv"]).all()
        assert np.isfinite(data["psd_uv2_hz"]).all() and np.all(data["psd_uv2_hz"] >= 0)
        np.testing.assert_array_equal(data["time_s"], np.arange(512) / 128)
        np.testing.assert_array_equal(data["frequency_hz"], np.arange(129) / 2)
    assert path.stat().st_size < 50_000_000


def test_independent_canonical_scientific_acceptance_anchors():
    """Read saved reports only; reference anchors are checks, never application data."""
    repo = APP.parents[1]
    char = repo / "outputs/runs/scenario2/characterization"
    fpath = char / "b0_fidelity_extension_s2_74e_v1/metrics/s2_74e_b0_complete_report_v1.json"
    if not fpath.exists():
        pytest.skip("Canonical research evidence is not shipped in a presentation-only distribution")
    fidelity = preview.load_json(fpath)
    original = preview.load_json(char / "final_pool_fidelity_s2_74b_v1/metrics/s2_74b_complete_report_v1.json")
    for metric in original["full_pool_metrics"]:
        for generator in ("B1", "B2"):
            assert fidelity["full_pool_metrics"][metric][generator] == original["full_pool_metrics"][metric][generator]
    for generator, expected in zip(("B0", "B1", "B2"), (3.5732, 1.1560, 0.9389)):
        assert fidelity["full_pool_metrics"]["log_psd_mae_db"][generator]["mean"] == pytest.approx(expected, abs=0.00005)
    for state, expected in (("EO", [0.51, 75.25, 71.72]), ("EC", [1.01, 84.34, 95.96])):
        for generator, value in zip(("B0", "B1", "B2"), expected):
            record = fidelity["fingerprint"][state + "_" + generator]
            assert 100 * record["top1"] == pytest.approx(value, abs=0.0051)
            assert record["n_candidates"] == 198
    utility = preview.load_json(repo / "outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es/final_test_s2_77_v1/s2_77_v2_complete_final_test_report.json")
    assert (utility["n_subjects"], utility["n_fits"], utility["n_test_windows"]) == (198, 55, 19131)
    assert utility["v1_24_epoch_models_used"] is False
    results = utility["results"]["EEGNETV4"]
    for condition, expected in (("REAL_ONLY", 88.417), ("REAL_PLUS_B1_1X", 88.983), ("REAL_PLUS_B2_1X", 89.135)):
        record = results[condition]
        assert 100 * record["subject_macro_balanced_accuracy_mean"] == pytest.approx(expected, abs=0.0005)
        assert record["subject_macro_balanced_accuracy_population_sd"] == pytest.approx(np.std(record["seed_values"], ddof=0))
    for id_, expected, interval in (("P2_AUG_B2_1X_VS_REAL", 0.718, (0.250, 1.201)), ("P3_AUG_B2_1X_VS_B1_1X", 0.152, (-0.276, 0.570))):
        contrast = next(r for r in utility["paired_contrasts"] if r["family"] == "EEGNETV4" and r["contrast_id"] == id_)
        direct = results[contrast["right"]]["subject_macro_balanced_accuracy_mean"] - results[contrast["left"]]["subject_macro_balanced_accuracy_mean"]
        assert contrast["delta_right_minus_left"] == pytest.approx(direct)
        assert -100 * direct == pytest.approx(expected, abs=0.0005)
        assert [-100 * contrast["ci95_high"], -100 * contrast["ci95_low"]] == pytest.approx(interval, abs=0.0005)


@pytest.fixture(scope="module")
def actual_presentation_bundle():
    path = APP / "data/presentation_bundle/bundle.json"
    if not path.exists():
        pytest.skip("Main presentation data has not been prepared")
    return preview.load_json(path)


def test_actual_fingerprint_figure_percentages_and_uniform_reference(actual_presentation_bundle):
    from lab.charts import fingerprint_chart
    rows = actual_presentation_bundle["fingerprint"]
    for metric, rank in (("top1", 1), ("top5", 5)):
        fig = fingerprint_chart(rows, metric)
        for trace in fig.data:
            selected = {r["state"]: r for r in rows if r["model"] == trace.name}
            np.testing.assert_allclose(trace.y, [100 * selected[s][metric] for s in ("EO", "EC")])
        assert fig.layout.shapes[0].y0 == pytest.approx(100 * rank / 198)
        assert "Uniform-random ranking reference" in fig.layout.annotations[0].text


def test_actual_connectivity_figure_preserves_negative_missing_and_signed_difference(actual_presentation_bundle):
    from lab.charts import connectivity_grid
    c = actual_presentation_bundle["connectivity"]
    # Participant estimates expose signed debiasing more clearly than group means.
    matrices = next(iter(c["participants"].values()))["EO"]["alpha"]
    direct = connectivity_grid(matrices, c["channels"], "EO", "alpha")
    zmin, zmax = direct.data[0].zmin, direct.data[0].zmax
    assert zmin < 0
    for model, trace in zip(("Real", "B0", "B1", "B2"), direct.data):
        expected = np.asarray(matrices[model], dtype=float)
        np.fill_diagonal(expected, np.nan)
        np.testing.assert_allclose(trace.z, expected, equal_nan=True)
        assert trace.zmin == zmin and trace.zmax == zmax
    difference = connectivity_grid(matrices, c["channels"], "EO", "alpha", difference=True)
    for model, trace in zip(("B0", "B1", "B2"), difference.data):
        expected = np.asarray(matrices[model], dtype=float) - np.asarray(matrices["Real"], dtype=float)
        np.fill_diagonal(expected, np.nan)
        np.testing.assert_allclose(trace.z, expected, equal_nan=True)
        assert trace.zmin == -trace.zmax


def test_actual_saved_psd_alpha_and_metric_units(actual_presentation_bundle):
    from lab.adapters import metric_definition
    b = actual_presentation_bundle
    assert metric_definition("sd_md_pooled_uv")[1] == "uV"
    assert metric_definition("covariance_relative_frobenius_z")[1] == "dimensionless"
    assert metric_definition("rms_pairwise_diversity_ratio")[2] == "parity"
    assert metric_definition("nearest_real_rmse_z_median")[2] == "descriptive"
    repo = APP.parents[1]
    sid = sorted(b["psd"]["participants"])[0]
    sources = b["psd"]["sources_by_subject"][sid]
    if not all((repo / source).exists() for source in sources):
        pytest.skip("Original participant summaries are not shipped in a presentation-only distribution")
    original, added = [preview.load_json(repo / source) for source in sources]
    for model in ("Real", "B0", "B1", "B2"):
        saved = added if model == "B0" else original
        for state in ("EO", "EC"):
            spectrum = saved["states"][state]["source_summaries"][model]["mean_psd_uv2_hz"]
            np.testing.assert_allclose(b["psd"]["participants"][sid][state][model], np.mean(spectrum, axis=0))
        means = saved["states"]
        expected_alpha = np.asarray(means["EC"]["source_summaries"][model]["median_bandpower_db"][2]) - np.asarray(means["EO"]["source_summaries"][model]["median_bandpower_db"][2])
        np.testing.assert_allclose(b["alpha_effect"]["participants"][sid][model], expected_alpha)
