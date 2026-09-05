"""BrainVision payload and metadata checks that never request waveform values.

SHA256 verification reads opaque payload bytes. MNE is opened with preload=False
and is used only for metadata; no get_data, load_data, filtering, or PSD is used.
"""

from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import warnings

import numpy as np
import pandas as pd


TARGET_CHANNELS = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8", "C3",
    "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2",
]
PILOT_SUBJECTS = [
    "sub-010023", "sub-010059", "sub-010071", "sub-010078", "sub-010081",
    "sub-010126", "sub-010134", "sub-010216", "sub-010250", "sub-010276",
    "sub-010277", "sub-010278", "sub-010279", "sub-010280", "sub-010282",
]


def sha256_file(path: Path) -> str:
    """Hash opaque bytes with bounded memory, without interpreting EEG values."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_brainvision_header(path: Path) -> configparser.ConfigParser:
    """Parse only the textual BrainVision header/marker sections."""
    content = Path(path).read_bytes()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1252")
    if "Brain Vision Data Exchange" not in text.splitlines()[0]:
        raise ValueError(f"not a BrainVision text file: {path}")
    # Free-form comments following [Comment] are not INI key/value entries.
    text = text.split("[Comment]", 1)[0]
    config = configparser.ConfigParser(interpolation=None, strict=True)
    config.read_string("\n".join(text.splitlines()[1:]))
    return config


def _safe_link(parent: Path, name: str) -> Path:
    """Reject POSIX/Windows absolute paths, traversal, and symlink escapes."""
    posix = PurePosixPath(name.replace("\\", "/"))
    windows = PureWindowsPath(name)
    if (
        not name or posix.is_absolute() or windows.is_absolute() or windows.drive
        or ".." in posix.parts or ":" in name or "\x00" in name
    ):
        raise ValueError(f"unsafe BrainVision link: {name!r}")
    resolved_parent = parent.resolve()
    target = resolved_parent.joinpath(*posix.parts).resolve()
    if target.parent != resolved_parent:
        raise ValueError(f"BrainVision link must remain in the subject EEG directory: {name!r}")
    if not target.is_file():
        raise ValueError(f"missing BrainVision linked file: {target.name}")
    return target


def validate_header_links(vhdr: Path) -> dict:
    """Return validated sibling links, or raise ValueError before MNE is opened.

    Both VHDR links and the marker's DataFile must point to the expected sibling
    basename. This avoids accepting a different recording in the same folder.
    """
    vhdr = Path(vhdr)
    if not vhdr.is_file() or vhdr.suffix.lower() != ".vhdr":
        raise ValueError(f"missing or invalid BrainVision header: {vhdr}")
    if vhdr.resolve().parent != vhdr.parent.resolve():
        raise ValueError("BrainVision header symlink escapes its EEG directory")
    config = read_brainvision_header(vhdr)
    common = config["Common Infos"]
    eeg = _safe_link(vhdr.parent, common["DataFile"])
    vmrk = _safe_link(vhdr.parent, common["MarkerFile"])
    if eeg != vhdr.with_suffix(".eeg").resolve():
        raise ValueError("header DataFile does not reference the expected local EEG")
    if vmrk != vhdr.with_suffix(".vmrk").resolve():
        raise ValueError("header MarkerFile does not reference the expected local marker file")
    marker = read_brainvision_header(vmrk)
    marker_eeg = _safe_link(vmrk.parent, marker["Common Infos"]["DataFile"])
    if marker_eeg != eeg:
        raise ValueError("marker DataFile and header DataFile disagree")
    return {
        "eeg_path": eeg, "vmrk_path": vmrk,
        "vhdr_eeg_link_ok": True, "vhdr_vmrk_link_ok": True,
        "vmrk_eeg_link_ok": True, "header_paths_safe": True,
    }


def _subject_manifest_paths(subject_id: str, manifest: pd.DataFrame, raw_root: Path) -> dict:
    if not subject_id.startswith("sub-") or any(c in subject_id for c in "/\\:"):
        raise ValueError("unsafe subject ID")
    rows = manifest.loc[manifest.subject_id == subject_id]
    if len(rows) != 2 or rows.file_role.duplicated().any():
        raise ValueError("subject manifest must contain exactly one waveform and one header")
    expected_roles = {"waveform": ".eeg", "brainvision_header": ".vhdr"}
    if set(rows.file_role) != set(expected_roles):
        raise ValueError("unexpected subject manifest roles")
    paths = {}
    expected_parent = (raw_root / subject_id / "eeg").resolve()
    if raw_root.resolve() not in expected_parent.parents:
        raise ValueError("subject directory escapes raw root")
    for index, row in rows.iterrows():
        name = str(row.relative_path)
        relative = PurePosixPath(name.replace("\\", "/"))
        if relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts:
            raise ValueError("unsafe relative manifest path")
        path = raw_root.joinpath(*relative.parts).resolve()
        if path.parent != expected_parent or path.suffix != expected_roles[row.file_role]:
            raise ValueError("manifest payload path is outside subject EEG directory or has wrong extension")
        digest = str(row.payload_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid expected SHA256")
        paths[row.file_role] = (index, row, path)
    return paths


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def audit_subject(
    subject_id: str,
    manifest: pd.DataFrame,
    raw_root: Path,
    blocks: pd.DataFrame,
    target_channels: list[str] | None = None,
) -> dict:
    """Audit one subject, returning a CSV-safe record; failures never read values."""
    import mne

    target_channels = TARGET_CHANNELS if target_channels is None else target_channels
    result = {"subject_id": subject_id, "waveform_values_accessed": False}
    check_names = [
        "payload_size_ok", "payload_hash_ok", "header_size_ok", "header_hash_ok",
        "vhdr_eeg_link_ok", "vhdr_vmrk_link_ok", "vmrk_eeg_link_ok", "header_paths_safe",
        "mne_header_open_ok", "header_sfreq_ok", "metadata_sfreq_ok", "channel_metadata_ok",
        "channel_composition_ok", "target_19_present", "native_shape_ok", "block_bounds_covered",
        "metadata_duration_ok", "unit_calibration_ok", "montage_valid",
    ]
    result.update(dict.fromkeys(check_names, False))
    result.update(integrity_pass=False, failure_reason="", warnings="")
    raw = None
    captured = []
    try:
        paths = _subject_manifest_paths(subject_id, manifest, Path(raw_root))
        for role, prefix in [("waveform", "payload"), ("brainvision_header", "header")]:
            index, row, path = paths[role]
            size = path.stat().st_size
            result[f"{prefix}_size_bytes"] = size
            result[f"{prefix}_manifest_row_index"] = int(index)
            result[f"{prefix}_relative_path"] = str(row.relative_path)
            result[f"{prefix}_expected_sha256"] = str(row.payload_sha256)
            result[f"{prefix}_size_ok"] = size == int(row.payload_size_bytes)
            result[f"{prefix}_sha256"] = sha256_file(path)
            result[f"{prefix}_hash_ok"] = result[f"{prefix}_sha256"].lower() == str(row.payload_sha256).lower()
            if not result[f"{prefix}_size_ok"] or not result[f"{prefix}_hash_ok"]:
                raise ValueError(f"{prefix} size/SHA256 mismatch")
        vhdr = paths["brainvision_header"][2]
        result['verified_header_path'] = str(vhdr.resolve())
        links = validate_header_links(vhdr)
        if links["eeg_path"] != paths["waveform"][2]:
            raise ValueError("linked EEG does not match manifest waveform path")
        result.update({key: value for key, value in links.items() if key.endswith("_ok") or key == "header_paths_safe"})
        header = read_brainvision_header(vhdr)
        common = header["Common Infos"]
        if common["DataFormat"] != "BINARY" or common["DataOrientation"] != "MULTIPLEXED":
            raise ValueError("unsupported BrainVision data format/orientation")
        if header["Binary Infos"]["BinaryFormat"] != "IEEE_FLOAT_32":
            raise ValueError("expected IEEE_FLOAT_32 binary samples")
        n_channels = int(common["NumberOfChannels"])
        sfreq = 1e6 / float(common["SamplingInterval"])
        channel_entries = [header["Channel Infos"][f"Ch{i}"].split(",") for i in range(1, n_channels + 1)]
        header_names = [entry[0].replace("\\1", ",") for entry in channel_entries]
        resolutions = np.asarray([float(entry[2]) for entry in channel_entries])
        header_units = [entry[3].strip() for entry in channel_entries]
        unit_to_v = {"V": 1.0, "mV": 1e-3, "µV": 1e-6, "μV": 1e-6, "uV": 1e-6, "nV": 1e-9}
        calibration_v = resolutions * np.asarray([unit_to_v[unit] for unit in header_units])
        metadata_path = vhdr.with_suffix(".json")
        channels_path = vhdr.with_name(vhdr.name.replace("_eeg.vhdr", "_channels.tsv"))
        for sidecar in [metadata_path, channels_path]:
            if sidecar.resolve().parent != vhdr.parent.resolve():
                raise ValueError("metadata sidecar escapes subject EEG directory")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        channels = pd.read_csv(channels_path, sep="\t")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, eog=["VEOG"], verbose="ERROR")
            if raw.preload:
                raise RuntimeError("MNE unexpectedly preloaded waveform data")
            result["mne_header_open_ok"] = True
            native_sfreq = float(raw.info["sfreq"])
            types = raw.get_channel_types()
            result.update(
                native_sfreq=native_sfreq,
                sampling_interval_microseconds=float(common["SamplingInterval"]),
                n_times=int(raw.n_times), duration_sec=float(raw.n_times / native_sfreq),
                last_sample_sec=float((raw.n_times - 1) / native_sfreq),
                metadata_duration_sec=float(metadata["RecordingDuration"]),
                n_total_channels=len(raw.ch_names), n_eeg_channels=types.count("eeg"),
                n_eog_channels=types.count("eog"),
                native_channel_order=json.dumps(raw.ch_names),
                header_units=json.dumps(sorted(set(header_units)), ensure_ascii=False),
                header_resolutions=json.dumps(sorted(set(resolutions.tolist()))),
                mne_calibration_to_volts=json.dumps(sorted(set(calibration_v.tolist()))),
                mne_output_unit="V", compact_conversion_volts_to_microvolts=1e6,
                binary_format="IEEE_FLOAT_32", data_orientation="MULTIPLEXED",
            )
            result["header_sfreq_ok"] = bool(np.isclose(sfreq, native_sfreq, rtol=0, atol=1e-8))
            result["metadata_sfreq_ok"] = bool(
                np.isclose(float(metadata["SamplingFrequency"]), native_sfreq, rtol=0, atol=1e-8)
                and np.allclose(channels.sampling_frequency.to_numpy(float), native_sfreq, rtol=0, atol=1e-8)
            )
            result["channel_composition_ok"] = bool(
                n_channels == len(raw.ch_names) == 62 and types.count("eeg") == 61
                and types.count("eog") == 1 and raw.ch_names[types.index("eog")] == "VEOG"
                and int(metadata["EEGChannelCount"]) == 61 and int(metadata["EOGChannelCount"]) == 1
            )
            result["channel_metadata_ok"] = bool(
                header_names == raw.ch_names == channels.name.tolist()
                and len(set(raw.ch_names)) == 62
                and channels.type.str.lower().tolist() == types
                and all(unit_to_v.get(unit, np.nan) == unit_to_v.get(str(metadata_unit), np.nan)
                        for unit, metadata_unit in zip(header_units, channels.units))
            )
            result["target_19_present"] = bool(
                len(target_channels) == len(set(target_channels)) == 19
                and all(raw.ch_names.count(name) == 1 for name in target_channels)
            )
            file_samples, remainder = divmod(result["payload_size_bytes"], n_channels * 4)
            result["file_size_inferred_n_times"] = file_samples
            result["native_shape_ok"] = bool(remainder == 0 and file_samples == raw.n_times)
            result["metadata_duration_ok"] = bool(
                abs(float(metadata["RecordingDuration"]) - result["duration_sec"]) <= 1 / native_sfreq + 1e-8
            )
            # MNE's public per-channel calibration/range fields define the same SI scaling
            # used by its lazy binary reader. _cals is cross-checked without reading samples.
            info_cals = np.asarray([channel["cal"] * channel["range"] for channel in raw.info["chs"]])
            result["unit_calibration_ok"] = bool(
                np.all(np.isfinite(calibration_v)) and np.all(calibration_v > 0)
                and np.allclose(info_cals, calibration_v, rtol=1e-12, atol=0)
                and np.allclose(raw._cals, calibration_v, rtol=1e-12, atol=0)
            )
            retained = blocks.loc[blocks.subject_id == subject_id].copy()
            if "is_study_block" in retained:
                retained = retained.loc[_truth(retained.is_study_block)]
            if "is_interruption_fragment" in retained:
                retained = retained.loc[~_truth(retained.is_interruption_fragment)]
            starts = retained.start_sec.to_numpy(float)
            ends = retained.end_estimate_capped_sec.to_numpy(float)
            result["retained_metadata_blocks"] = len(retained)
            result["maximum_retained_end_sec"] = float(ends.max()) if len(ends) else None
            result["block_bounds_covered"] = bool(
                len(retained) > 0 and np.all(np.isfinite(starts)) and np.all(np.isfinite(ends))
                and np.all(starts >= 0) and np.all(ends > starts)
                and np.all(ends <= result["duration_sec"] + 1e-8)
            )
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage, on_missing="raise", verbose="ERROR")
            eeg_indices = [i for i, kind in enumerate(types) if kind == "eeg"]
            coordinates = np.asarray([raw.info["chs"][i]["loc"][:3] for i in eeg_indices])
            radii = np.linalg.norm(coordinates, axis=1)
            result["montage_valid"] = bool(
                coordinates.shape == (61, 3) and np.all(np.isfinite(coordinates))
                and np.all((radii > 0.01) & (radii < 0.3))
                and len(np.unique(coordinates, axis=0)) == 61
            )
            result["montage"] = "standard_1005"
            result["montage_coordinate_source"] = "template; individual electrode coordinates unavailable"
        failed = [name for name in check_names if not result[name]]
        result["integrity_pass"] = not failed
        result["failure_reason"] = ";".join(failed)
    except Exception as error:
        result["failure_reason"] = f"{type(error).__name__}: {error}"
    finally:
        if raw is not None:
            raw.close()
        result["warnings"] = " | ".join(str(item.message) for item in captured)
    return result


def audit_pilot_inventory(raw_root: Path, manifest: pd.DataFrame) -> dict:
    """Check exact pilot membership and all staged filenames, without modifying data."""
    raw_root = Path(raw_root)
    subjects = sorted(path.name for path in raw_root.iterdir() if path.is_dir())
    files = sorted(path for path in raw_root.rglob("*") if path.is_file())
    eeg = [path for path in files if path.suffix.lower() == ".eeg"]
    vhdr = [path for path in files if path.suffix.lower() == ".vhdr"]
    parts = [path for path in files if path.name.endswith(".part")]
    expected_paths = set(manifest.relative_path.astype(str))
    sidecars = []
    for subject in PILOT_SUBJECTS:
        for suffix in ["channels.tsv", "eeg.json", "eeg.vmrk", "events.json", "events.tsv"]:
            sidecars.append(f"{subject}/eeg/{subject}_task-resting_{suffix}")
    expected_paths.update(sidecars)
    actual_paths = {path.relative_to(raw_root).as_posix() for path in files}
    inventory = {
        "subject_ids": subjects, "n_subjects": len(subjects), "n_staged_files": len(files),
        "n_eeg_files": len(eeg), "n_vhdr_files": len(vhdr), "n_backed_payloads": len(eeg) + len(vhdr),
        "n_sidecars": len(files) - len(eeg) - len(vhdr) - len(parts), "n_part_files": len(parts),
        "raw_bytes": sum(path.stat().st_size for path in files),
        "manifest_payload_bytes": int(manifest.payload_size_bytes.sum()),
        "actual_payload_bytes": sum(path.stat().st_size for path in eeg + vhdr),
        "missing_paths": sorted(expected_paths - actual_paths),
        "unexpected_paths": sorted(actual_paths - expected_paths),
        "waveform_values_accessed": False,
    }
    inventory["raw_gib"] = inventory["raw_bytes"] / 1024**3
    inventory["inventory_pass"] = bool(
        subjects == PILOT_SUBJECTS and sorted(manifest.subject_id.unique()) == PILOT_SUBJECTS
        and len(manifest) == 30 and len(files) == 105 and len(eeg) == len(vhdr) == 15
        and inventory["n_sidecars"] == 75 and not parts
        and inventory["actual_payload_bytes"] == inventory["manifest_payload_bytes"]
        and not inventory["missing_paths"] and not inventory["unexpected_paths"]
    )
    return inventory
