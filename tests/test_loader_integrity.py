"""Synthetic BrainVision integrity regressions; no real EEG values are accessed."""

import json
from pathlib import Path

import mne
import pandas as pd
import pytest

from resting_eeg.data.integrity import (
    PILOT_SUBJECTS, TARGET_CHANNELS, audit_pilot_inventory, audit_subject,
    sha256_file, validate_header_links,
)


CHANNELS = (
    "Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 VEOG CP5 CP1 CP2 "
    "CP6 AFz P7 P3 Pz P4 P8 PO9 O1 Oz O2 PO10 AF7 AF3 AF4 AF8 F5 F1 F2 F6 "
    "FT7 FC3 FC4 FT8 C5 C1 C2 C6 TP7 CP3 CPz CP4 TP8 P5 P1 P2 P6 PO7 PO3 POz PO4 PO8"
).split()


def write_recording(tmp_path: Path, sfreq: int = 500):
    subject = "sub-synthetic"
    directory = tmp_path / subject / "eeg"
    directory.mkdir(parents=True)
    stem = f"{subject}_task-resting_eeg"
    eeg = directory / f"{stem}.eeg"
    eeg.write_bytes(bytes(62 * sfreq * 2 * 4))
    vmrk = directory / f"{stem}.vmrk"
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        f"[Common Infos]\nCodepage=UTF-8\nDataFile={eeg.name}\n"
        "[Marker Infos]\nMk1=Stimulus,S200,1,1,0\n", encoding="utf-8",
    )
    vhdr = directory / f"{stem}.vhdr"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        f"[Common Infos]\nCodepage=UTF-8\nDataFile={eeg.name}\nMarkerFile={vmrk.name}\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=62\n"
        f"SamplingInterval={1e6 / sfreq}\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
        "[Channel Infos]\n" + "\n".join(f"Ch{i}={name},,0.1,µV" for i, name in enumerate(CHANNELS, 1))
        + "\n", encoding="utf-8",
    )
    vhdr.with_suffix(".json").write_text(json.dumps({
        "SamplingFrequency": sfreq, "RecordingDuration": 2 - 1 / sfreq,
        "EEGChannelCount": 61, "EOGChannelCount": 1,
    }), encoding="utf-8")
    pd.DataFrame({
        "name": CHANNELS, "type": ["EOG" if name == "VEOG" else "EEG" for name in CHANNELS],
        "units": "µV", "sampling_frequency": sfreq,
    }).to_csv(directory / f"{subject}_task-resting_channels.tsv", sep="\t", index=False)
    manifest = pd.DataFrame([
        {"subject_id": subject, "file_role": role,
         "relative_path": path.relative_to(tmp_path).as_posix(),
         "payload_size_bytes": path.stat().st_size, "payload_sha256": sha256_file(path)}
        for role, path in [("waveform", eeg), ("brainvision_header", vhdr)]
    ])
    blocks = pd.DataFrame([{
        "subject_id": subject, "start_sec": 0.1, "end_estimate_capped_sec": 1.8,
        "is_study_block": True, "is_interruption_fragment": False,
    }])
    return vhdr, manifest, blocks


def test_header_links_reference_expected_local_triplet(tmp_path):
    vhdr, _, _ = write_recording(tmp_path)
    result = validate_header_links(vhdr)
    assert result["eeg_path"] == vhdr.with_suffix(".eeg")
    assert result["vmrk_path"] == vhdr.with_suffix(".vmrk")
    assert result["header_paths_safe"]
    assert result["vmrk_eeg_link_ok"]


@pytest.mark.parametrize("unsafe", [
    "../escape.eeg", "..\\escape.eeg", "/tmp/escape.eeg", "C:\\escape.eeg",
    "C:escape.eeg", "\\\\server\\share\\escape.eeg", "nested/escape.eeg",
])
def test_header_data_link_cannot_escape_eeg_directory(tmp_path, unsafe):
    vhdr, _, _ = write_recording(tmp_path)
    text = vhdr.read_text(encoding="utf-8")
    vhdr.write_text(text.replace(f"DataFile={vhdr.stem}.eeg", f"DataFile={unsafe}"), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe|directory"):
        validate_header_links(vhdr)


@pytest.mark.parametrize("link", ["DataFile", "MarkerFile"])
def test_header_missing_linked_file_is_rejected(tmp_path, link):
    vhdr, _, _ = write_recording(tmp_path)
    vhdr.with_suffix(".eeg" if link == "DataFile" else ".vmrk").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_header_links(vhdr)


def test_marker_link_is_validated_independently(tmp_path):
    vhdr, _, _ = write_recording(tmp_path)
    marker = vhdr.with_suffix(".vmrk")
    text = marker.read_text(encoding="utf-8")
    marker.write_text(text.replace(f"DataFile={vhdr.stem}.eeg", "DataFile=../escape.eeg"), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        validate_header_links(vhdr)


def test_marker_cannot_point_to_a_different_local_recording(tmp_path):
    vhdr, _, _ = write_recording(tmp_path)
    (vhdr.parent / "other.eeg").write_bytes(b"synthetic")
    marker = vhdr.with_suffix(".vmrk")
    text = marker.read_text(encoding="utf-8")
    marker.write_text(text.replace(f"DataFile={vhdr.stem}.eeg", "DataFile=other.eeg"), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        validate_header_links(vhdr)


@pytest.mark.parametrize("sfreq", [500, 1000, 2500])
def test_header_audit_native_shape_units_and_montage_without_value_reads(tmp_path, monkeypatch, sfreq):
    vhdr, manifest, blocks = write_recording(tmp_path, sfreq)

    def forbidden(*args, **kwargs):
        raise AssertionError("waveform values must never be requested by header integrity")

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", forbidden)
    monkeypatch.setattr(mne.io.BaseRaw, "load_data", forbidden)
    result = audit_subject("sub-synthetic", manifest, tmp_path, blocks, TARGET_CHANNELS)
    assert result["integrity_pass"], result["failure_reason"]
    assert not result["waveform_values_accessed"]
    assert result["n_times"] == sfreq * 2
    assert result["file_size_inferred_n_times"] == sfreq * 2
    assert result["native_sfreq"] == sfreq
    assert result["n_eeg_channels"] == 61
    assert result["n_eog_channels"] == 1
    assert json.loads(result["mne_calibration_to_volts"]) == [1e-7]
    assert result["unit_calibration_ok"] and result["montage_valid"]


def test_payload_corruption_refuses_mne_open(tmp_path, monkeypatch):
    vhdr, manifest, blocks = write_recording(tmp_path)
    eeg = vhdr.with_suffix(".eeg")
    with eeg.open("r+b") as handle:
        handle.write(b"bad!")

    def forbidden(*args, **kwargs):
        raise AssertionError("must refuse unverified payload before MNE")

    monkeypatch.setattr(mne.io, "read_raw_brainvision", forbidden)
    result = audit_subject("sub-synthetic", manifest, tmp_path, blocks)
    assert not result["integrity_pass"]
    assert result["payload_size_ok"] and not result["payload_hash_ok"]
    assert not result["mne_header_open_ok"]
    assert "SHA256 mismatch" in result["failure_reason"]


def test_metadata_sampling_disagreement_fails_integrity(tmp_path):
    vhdr, manifest, blocks = write_recording(tmp_path)
    metadata = json.loads(vhdr.with_suffix(".json").read_text())
    metadata["SamplingFrequency"] = 1000
    vhdr.with_suffix(".json").write_text(json.dumps(metadata))
    result = audit_subject("sub-synthetic", manifest, tmp_path, blocks)
    assert not result["integrity_pass"]
    assert not result["metadata_sfreq_ok"]
    assert result["native_shape_ok"]


def test_metadata_duration_disagreement_fails_integrity(tmp_path):
    vhdr, manifest, blocks = write_recording(tmp_path)
    metadata = json.loads(vhdr.with_suffix(".json").read_text())
    metadata["RecordingDuration"] = 10
    vhdr.with_suffix(".json").write_text(json.dumps(metadata))
    result = audit_subject("sub-synthetic", manifest, tmp_path, blocks)
    assert not result["integrity_pass"]
    assert not result["metadata_duration_ok"]


def test_retained_block_beyond_recording_fails_integrity(tmp_path):
    _, manifest, blocks = write_recording(tmp_path)
    blocks.loc[0, "end_estimate_capped_sec"] = 2.01
    result = audit_subject("sub-synthetic", manifest, tmp_path, blocks)
    assert not result["integrity_pass"]
    assert not result["block_bounds_covered"]


def test_non_frame_aligned_binary_size_fails_integrity(tmp_path):
    vhdr, manifest, blocks = write_recording(tmp_path)
    eeg = vhdr.with_suffix(".eeg")
    with eeg.open("ab") as handle:
        handle.write(b"\0")
    manifest.loc[0, "payload_size_bytes"] = eeg.stat().st_size
    manifest.loc[0, "payload_sha256"] = sha256_file(eeg)
    result = audit_subject("sub-synthetic", manifest, tmp_path, blocks)
    assert not result["integrity_pass"]
    assert not result["native_shape_ok"]


def test_manifest_path_cannot_reference_another_subject(tmp_path):
    _, manifest, blocks = write_recording(tmp_path)
    manifest.loc[0, "relative_path"] = "sub-other/eeg/waveform.eeg"
    result = audit_subject("sub-synthetic", manifest, tmp_path, blocks)
    assert not result["integrity_pass"]
    assert not result["mne_header_open_ok"]
    assert "outside subject" in result["failure_reason"]


def test_exact_pilot_inventory_and_no_nonpilot_files(tmp_path):
    rows = []
    for subject in PILOT_SUBJECTS:
        directory = tmp_path / subject / "eeg"
        directory.mkdir(parents=True)
        for suffix in ["eeg.eeg", "eeg.vhdr", "channels.tsv", "eeg.json", "eeg.vmrk", "events.json", "events.tsv"]:
            path = directory / f"{subject}_task-resting_{suffix}"
            path.write_bytes(b"x")
            if path.suffix in [".eeg", ".vhdr"]:
                rows.append({"subject_id": subject, "relative_path": path.relative_to(tmp_path).as_posix(), "payload_size_bytes": 1})
    manifest = pd.DataFrame(rows)
    result = audit_pilot_inventory(tmp_path, manifest)
    assert result["inventory_pass"]
    assert result["n_staged_files"] == 105 and result["n_sidecars"] == 75
    (tmp_path / "extra.part").write_bytes(b"x")
    result = audit_pilot_inventory(tmp_path, manifest)
    assert not result["inventory_pass"]
    assert result["n_part_files"] == 1
    assert result["unexpected_paths"] == ["extra.part"]
