"""Bounded, read-only Train/synthetic preview preparation. Never imported by the UI.

No experiment module is imported. A locked Train row AND matching HDF5 Train
attributes are required before accessing an EEG dataset. Only selected 512-sample
slices are materialized. Synthetic NPZ members are read at selected byte offsets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
CHANNELS = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8", "C3", "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2"]
SOURCES = ["Real", "B0", "B1", "B2"]
SCHEMA = "eeg_synthetic_lab.previews.v1"
FS, NC, NT = 128, 19, 512


class PreviewGuardError(ValueError):
    """A provenance, selection, boundary, or unit check failed closed."""


def require(condition, message):
    if not condition:
        raise PreviewGuardError(message)


def load_json(path):
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, "Duplicate JSON key: " + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def true(value):
    return text(value).lower() in {"true", "1"}


def evenly_spaced_indices(size, limit):
    require(isinstance(size, int) and size >= 0, "Invalid population size")
    require(isinstance(limit, int) and 1 <= limit <= 12, "Selection limit must be 1 through 12")
    count = min(size, limit)
    if count < 2:
        return list(range(count))
    return [(i * (size - 1)) // (count - 1) for i in range(count)]


def select_subjects(subjects, limit=12):
    ordered = sorted(set(subjects))
    return [ordered[i] for i in evenly_spaced_indices(len(ordered), limit)]


def validate_scaler(scaler):
    require(scaler["fit_split"] == "Train", "Scaler was not fitted on Train")
    require(scaler["channel_order"] == CHANNELS, "Scaler channel order differs")
    require(scaler["ddof"] == 0, "Unexpected scaler SD convention")
    means = np.asarray(scaler["means_uv"], dtype=np.float64)
    stds = np.asarray(scaler["stds_uv"], dtype=np.float64)
    require(means.shape == stds.shape == (NC,), "Scaler shape differs")
    require(np.isfinite(means).all() and np.isfinite(stds).all() and (stds > 0).all(), "Invalid scaler values")
    return means, stds


def assert_train_block(row, block, channels=CHANNELS):
    # This must remain ahead of ANY block['eeg_uv'] access, including shape access.
    require(row["split"] == "Train", "Forbidden non-Train manifest row")
    require(row["final_lock_status"] == "LOCKED_UNCHANGED_PROVISIONAL_ROLE", "Split row is not locked")
    require(true(row["block_eligible"]), "Ineligible manifest block")
    a = block.attrs
    require(text(a["provisional_split"]) == "Train", "Forbidden non-Train HDF5 attribute")
    require(true(a["block_eligible"]), "Ineligible HDF5 block")
    for attribute, key in (("subject_id", "subject_id"), ("block_uid", "block_uid"), ("state", "eye_state")):
        require(text(a[attribute]) == row[key], "HDF5 identity differs: " + attribute)
    require(json.loads(text(a["channel_names"])) == channels, "HDF5 channel order differs")
    require(text(a["physical_unit"]) == "uV", "Real waveform is not stored in microvolts")
    require(float(a["processed_sfreq"]) == FS, "HDF5 sampling rate differs")


def guarded_real_slice(row, block, start):
    assert_train_block(row, block)
    eeg = block["eeg_uv"]
    require(len(eeg.shape) == 2 and eeg.shape[0] == NC, "Invalid Real EEG shape")
    require(isinstance(start, (int, np.integer)) and 0 <= start and start + NT <= eeg.shape[1], "Real window exceeds block bounds")
    result = np.asarray(eeg[:, int(start):int(start) + NT], dtype=np.float32)
    require(result.shape == (NC, NT) and np.isfinite(result).all(), "Invalid Real preview values")
    return result


def read_npy_header(stream):
    version = np.lib.format.read_magic(stream)
    require(version in {(1, 0), (2, 0)}, "Unsupported NPY version")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    require(not dtype.hasobject and not fortran, "Object or Fortran arrays are forbidden")
    return shape, dtype, stream.tell()


def read_synthetic_slices(path, generator, participant, state, indices, expected_count, scaler, scaler_sha):
    means, stds = validate_scaler(scaler)
    require(1 <= len(indices) <= 3 and len(set(indices)) == len(indices), "Synthetic preview count exceeds limit or repeats")
    require(all(isinstance(i, (int, np.integer)) and 0 <= i < expected_count for i in indices), "Synthetic sample index exceeds bounds")
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate NPZ member")
        # Metadata can be decoded independently without touching x_z.
        with archive.open("metadata_json.npy") as f:
            meta = json.loads(str(np.lib.format.read_array(f, allow_pickle=False)))
        require(meta["generator_name"] == generator and meta["subject_id"] == participant and meta["eye_state"] == state, "Synthetic stratum identity differs")
        require(meta["array_key"] == "x_z" and meta["signal_domain"] == "frozen_train_global_z", "Synthetic signal domain is not frozen standardized z")
        require(meta["normalization_sha256"] == scaler_sha, "Synthetic normalization identity differs")
        require(meta["channel_order"] == CHANNELS and meta["sampling_rate_hz"] == FS, "Synthetic channel/rate differs")
        require(meta["n_channels"] == NC and meta["n_times"] == NT and meta["expected_count"] == expected_count, "Synthetic dimensions differ")
        with archive.open("x_z.npy") as f:
            shape, dtype, data_start = read_npy_header(f)
            require(shape == (expected_count, NC, NT) and dtype.kind == "f" and dtype.itemsize == 4, "Synthetic array shape/dtype differs")
            window_bytes = NC * NT * dtype.itemsize
            result = []
            for index in indices:
                f.seek(data_start + int(index) * window_bytes)
                raw = f.read(window_bytes)
                require(len(raw) == window_bytes, "Truncated synthetic sample")
                x_z = np.frombuffer(raw, dtype=dtype).reshape(NC, NT)
                require(np.isfinite(x_z).all(), "Nonfinite synthetic sample")
                # x_z is already standardized: inverse transform exactly once.
                result.append((x_z.astype(np.float64) * stds[:, None] + means[:, None]).astype(np.float32))
    return np.stack(result)


def deterministic_npz(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key, array in sorted(arrays.items()):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buf.getvalue())


def export_previews(repo=REPO_ROOT, processed_root=None, output=None):
    # Optional preparation dependencies are deliberately not runtime imports.
    import h5py
    from scipy.signal import welch

    repo = Path(repo).resolve()
    out = Path(output or APP_ROOT / "data/presentation_bundle").resolve()
    require(out.is_relative_to(APP_ROOT.resolve()), "Preview output must remain inside the application")
    processed = Path(processed_root or repo.parent / "Datasets/MPI-LEMON-Scenario2/processed/all-blocks-v1").resolve()
    metrics = repo / "outputs/metrics/scenario2"
    split_path = metrics / "final_block_split_manifest.csv"
    norm_path = metrics / "train_normalization.json"
    norm_lock_path = metrics / "train_normalization_lock.json"
    split_lock_path = metrics / "final_cohort_split_lock.json"
    scaler, lock = load_json(norm_path), load_json(norm_lock_path)
    validate_scaler(scaler)
    tracked = {}

    def track(path, expected=None, display=None):
        actual = sha256(path)
        require(expected is None or actual == expected, "Locked source hash differs: " + Path(path).name)
        tracked[Path(path)] = {"path": display or Path(path).relative_to(repo).as_posix(), "sha256": actual}
        return actual

    scaler_sha = track(norm_path, lock["artifact_sha256"]["train_normalization.json"])
    track(split_path, lock["final_input_hashes"]["final_block_split_manifest.csv"])
    track(split_lock_path, lock["final_input_hashes"]["final_cohort_split_lock.json"])
    track(norm_lock_path)
    # Preserve which source definitions were inspected without importing them.
    for name in ("s2_23_check_real_train_dataset.py", "s2_74b_final_pool_fidelity.py"):
        track(repo / "scripts/data" / name)
    with split_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = [(r["subject_id"], r["block_uid"]) for r in rows]
    require(len(set(keys)) == len(keys), "Duplicate split block key")
    train = [r for r in rows if r["split"] == "Train" and true(r["block_eligible"])]
    require(len(train) == scaler["eligible_train_block_count"], "Train block total differs from lock")
    subjects = select_subjects(r["subject_id"] for r in train)
    require(len(set(r["subject_id"] for r in train)) == scaler["included_subject_count"], "Train cohort differs")
    roots = {"B0": repo / "outputs/runs/scenario2/ablation/b0_state_only_1x_s2_74c_v1",
             "B1": repo / "outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1",
             "B2": repo / "outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1"}
    manifests = {}
    for generator in SOURCES[1:]:
        name = "s2_74c_b0_pool_manifest_v1.json" if generator == "B0" else "s2_73_pool_manifest_v1.json"
        p = roots[generator] / "manifests" / name
        manifests[generator] = load_json(p)
        track(p)
    records, arrays = [], []
    for participant in subjects:
        p = processed / (participant + "_all_blocks.h5")
        track(p, lock["input_hdf5_receipt_bundle"][participant]["hdf5_sha256"], "external_processed/all-blocks-v1/" + p.name)
        with h5py.File(p, "r") as h5:
            for state in ("EO", "EC"):
                candidates = []
                stratum = sorted((r for r in train if r["subject_id"] == participant and r["eye_state"] == state), key=lambda r: int(r["block_index"]))
                for row in stratum:
                    block = h5["blocks/" + row["block_uid"]]
                    assert_train_block(row, block)
                    starts = np.asarray(block["window_start_samples"][:])
                    usable = np.asarray(block["window_usable_mask"][:])
                    require(starts.ndim == 1 and starts.shape == usable.shape and starts.dtype.kind in "iu", "Window metadata dimensions/dtype differ")
                    require(np.isin(usable, [0, 1]).all() and int(usable.sum()) == int(row["usable_windows"]), "Window mask/count differs")
                    # Masks are metadata, not waveform arrays. Their byte hash is frozen.
                    mask_bytes = json.dumps(usable.astype(bool).tolist(), separators=(",", ":"), allow_nan=False).encode()
                    require(hashlib.sha256(mask_bytes).hexdigest() == row["window_usable_mask_sha256"], "Window mask hash differs")
                    candidates.extend((row, int(starts[i]), int(i)) for i in np.flatnonzero(usable))
                indices = evenly_spaced_indices(len(candidates), 3)
                require(indices, "No eligible Train windows")
                real = [guarded_real_slice(candidates[i][0], h5["blocks/" + candidates[i][0]["block_uid"]], candidates[i][1]) for i in indices]
                synthetic, chosen = {}, {}
                for generator in SOURCES[1:]:
                    key = generator + "/" + participant + "/" + state
                    entry = manifests[generator]["committed"][key]
                    require(entry["expected_count"] == len(candidates), "Synthetic stratum count differs from locked Train")
                    syn_path = (roots[generator] / entry["file"].replace("\\", "/")).resolve()
                    require(syn_path.is_relative_to(roots[generator].resolve()), "Synthetic path escapes pool")
                    track(syn_path, entry["sha256"])
                    chosen[generator] = evenly_spaced_indices(entry["expected_count"], 3)
                    synthetic[generator] = read_synthetic_slices(syn_path, generator, participant, state, chosen[generator], entry["expected_count"], scaler, scaler_sha)
                for slot, i in enumerate(indices):
                    row, start, candidate = candidates[i]
                    records.append({"participant_id": participant, "state": state, "slot": slot + 1,
                                    "real_block_uid": row["block_uid"], "start_sample": start,
                                    "real_candidate_index": candidate, "real_stratum_index": i,
                                    "synthetic_indices": {g: chosen[g][slot] for g in SOURCES[1:]},
                                    "analysis_split": "Train", "aggregation_level": "selected_window"})
                    arrays.append(np.stack([real[slot]] + [synthetic[g][slot] for g in SOURCES[1:]]))
    waveforms = np.stack(arrays).astype(np.float32)
    frequencies, psd = welch(waveforms, fs=FS, window="hann", nperseg=256, noverlap=128,
                             nfft=256, detrend="constant", scaling="density", average="mean", axis=-1)
    require(waveforms.shape[0] <= 72 and waveforms.nbytes + psd.nbytes < 50_000_000, "Preview bundle exceeds bound")
    # Whole-file hashing is opaque integrity access; no held-out arrays are indexed.
    for path, record in tracked.items():
        require(sha256(path) == record["sha256"], "Source changed during preview export: " + path.name)
    out.mkdir(parents=True, exist_ok=True)
    payload = out / "previews.npz"
    deterministic_npz(payload, {"waveforms_uv": waveforms, "psd_uv2_hz": psd.astype(np.float32),
                                "frequency_hz": frequencies, "time_s": np.arange(NT) / FS})
    metadata = {"schema_version": SCHEMA, "available": True, "file": payload.name,
                "npz_sha256": sha256(payload), "sources": SOURCES, "channels": CHANNELS,
                "fs_hz": FS, "units": "uV", "psd_units": "uV^2/Hz", "records": records,
                "participants": subjects, "states": ["EO", "EC"], "slots": [1, 2, 3],
                "selection": "Up to 12 evenly spaced positions in sorted eligible participants; up to 3 evenly spaced valid windows per state. No outcome or fidelity scores used.",
                "synthetic_label": "Precomputed synthetic sample", "pointwise_pairing": False,
                "normalization": "Real stored microvolts unchanged; synthetic x_z * frozen Train std_uv + mean_uv, exactly once.",
                "psd_method": "Selected-window preview: scipy.signal.welch, periodic Hann, 256 samples, 128 overlap, nfft=256, constant detrend, mean density. Distinct from original group fidelity aggregation.",
                "array_shapes": {"waveforms_uv": list(waveforms.shape), "psd_uv2_hz": list(psd.shape)},
                "waveform_reads": {"Train": len(records), "Validation": 0, "Final_Test": 0, "synthetic": 3 * len(records)},
                "source_files": list(tracked.values()), "source_hashes_unchanged_after_export": True,
                "raw_HDF5_runtime_dependency": False}
    (out / "previews.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, help="Directory containing accepted all-blocks HDF5 files; read only")
    parser.add_argument("--strict", action="store_true", help="Return failure if safe Train previews are unavailable")
    args = parser.parse_args()
    try:
        meta = export_previews(processed_root=args.processed_root)
        print(f"Prepared {len(meta['records'])} Train examples with Real/B0/B1/B2; no Validation/Test waveform reads.")
    except (FileNotFoundError, PreviewGuardError, KeyError, ImportError) as error:
        out = APP_ROOT / "data/presentation_bundle"
        out.mkdir(parents=True, exist_ok=True)
        reason = "Safe Train preview extraction unavailable: " + type(error).__name__
        (out / "previews.json").write_text(json.dumps({"schema_version": SCHEMA, "available": False, "reason": reason,
                                                     "action": "Check the accepted processed root, locked split/scaler, pool manifests, and preparation dependencies; then rerun the exporter."}, indent=2) + "\n", encoding="utf-8")
        print(reason + ". Existing study artifacts were not modified.")
        print(str(error), file=sys.stderr)
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
