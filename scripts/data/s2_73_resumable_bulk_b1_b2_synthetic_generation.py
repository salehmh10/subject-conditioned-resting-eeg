"""s2_73: frozen B1/B2 bulk generation, default cumulative target 2 percent.

Save this file under scripts/data in the existing Scenario-2 repository.
Run with the existing repository Python:
    python -u scripts/data/s2_73_resumable_bulk_b1_b2_synthetic_generation.py --stop-at-percent 2
Re-run the same command after an interruption. A later --stop-at-percent 10
means reach 10 percent TOTAL, using the same verified outputs and seeds.

One complete NPZ per generator x subject x eye state is the storage unit.
A scheduled stop happens only after a complete B1/B2 pair. CUDA work is
sequential; one GPU model is loaded with the appropriate frozen weights.
The ancestral DDPM has 1000 steps, v-prediction, FP32, batch size 64 and a
continuous seeded RNG stream per shard. No data or model tuning is performed.

NPZ members: x_z (float32 normalized EEG, n x 19 x 512), sample_indices
(int64), metadata_json (Unicode scalar). x_z is already standardized using
the frozen global Train scaler; do NOT normalize it again or call it uV.

The runner verifies the actual source and model hashes supplied in the user's
source handoff. It reads eligible Train HDF5 metadata only, never real EEG
waveform arrays. Model checkpoint bytes are hash-verified before loading.
Final Test is not opened. Successful 2-percent completion is still PARTIAL.

Progress: outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1/progress/
Reports: the same output root under metrics/ and logs/. Existing committed
shards are verified before reuse. Incomplete uncommitted files are quarantined;
committed corruption stops the runner. The script never changes batch size,
precision, sampler, seeds, or frozen model versions to recover from errors.
"""
from __future__ import annotations

"""Atomic, resumable s2_73 storage. Mutations require output_lock(root)."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

import numpy as np


class IntegrityError(RuntimeError):
    """A committed artifact is inconsistent; stop instead of regenerating it."""


def _store_assert_unlinked(root, path):
    """Inspect lexical ancestors before following any path into the output tree."""
    root, path = Path(root), Path(path)
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise IntegrityError("Storage path escapes the output root: " + str(path)) from exc
    if ".." in parts:
        raise IntegrityError("Parent traversal is not allowed in storage paths.")
    current = root
    for component in (None,) + parts:
        if component is not None:
            current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise IntegrityError("Storage path contains a symlink/reparse point: " + str(current))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _store_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _store_invalid_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def _store_json_loads(text):
    return json.loads(text, object_pairs_hook=_store_json_pairs,
                      parse_constant=_store_invalid_constant)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return _store_json_loads(handle.read())


def _store_fsync_dir(path):
    if os.name != "nt":
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _store_temp_path(path):
    return path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")


def atomic_json_write(path, payload):
    path = Path(path)
    absolute = path.absolute()
    for locked_root, pid in _STORE_ACTIVE_LOCKS.items():
        if pid == os.getpid() and absolute.is_relative_to(Path(locked_root)):
            _store_assert_unlinked(Path(locked_root), absolute)
    encoded = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _store_temp_path(path)
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _store_fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


_STORE_ACTIVE_LOCKS = {}


@contextlib.contextmanager
def output_lock(root):
    """Take a nonblocking OS lock; process termination releases it automatically."""
    root = Path(root).absolute()
    _store_assert_unlinked(root, root)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = str(root)
    if _STORE_ACTIVE_LOCKS.get(key) == os.getpid():
        raise IntegrityError("Output directory is already locked in this process: " + key)
    lock_path = root / ".s2_73.lock"
    _store_assert_unlinked(root, lock_path)
    handle = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise IntegrityError("Another process holds the output lock: " + key) from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise IntegrityError("Another process holds the output lock: " + key) from exc
        locked = True
        _STORE_ACTIVE_LOCKS[key] = os.getpid()
        yield root
    finally:
        if locked:
            _STORE_ACTIVE_LOCKS.pop(key, None)
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class ShardStore:
    """Each committed NPZ is authenticated by a separately committed receipt."""

    def __init__(self, root):
        root = Path(root).absolute()
        _store_assert_unlinked(root, root)
        self.root = root.resolve()

    def _locked(self):
        return _STORE_ACTIVE_LOCKS.get(str(self.root)) == os.getpid()

    def _require_lock(self):
        if not self._locked():
            raise IntegrityError("Storage mutation requires output_lock(root).")

    def _paths(self, generator, subject, state):
        if (generator not in ("B1", "B2") or state not in ("EO", "EC")
                or not isinstance(subject, str)
                or re.fullmatch(r"sub-[0-9]+", subject) is None):
            raise IntegrityError("Invalid generator, subject ID, or eye state.")
        data = self.root / "generated" / generator / subject / (state + ".npz")
        receipt = self.root / "manifests" / "shards" / generator / subject / (state + ".json")
        _store_assert_unlinked(self.root, data)
        _store_assert_unlinked(self.root, receipt)
        return data, receipt

    def _metadata(self, metadata, generator, subject, state):
        if not isinstance(metadata, dict):
            raise IntegrityError("Shard metadata must be a JSON object.")
        for key, expected in (("generator_name", generator), ("subject_id", subject),
                              ("eye_state", state)):
            if metadata.get(key) != expected:
                raise IntegrityError("Shard metadata mismatch: " + key)
        count = metadata.get("expected_count")
        if type(count) is not int or count < 1:
            raise IntegrityError("expected_count must be a positive integer.")
        indices = metadata.get("sample_indices")
        if (not isinstance(indices, list) or len(indices) != count
                or any(type(value) is not int or value != index
                       for index, value in enumerate(indices))):
            raise IntegrityError("metadata.sample_indices must equal 0..expected_count-1.")
        try:
            canonical_json_bytes(metadata)
        except (TypeError, ValueError) as exc:
            raise IntegrityError("Shard metadata is not finite canonical JSON.") from exc
        return count

    def _array(self, values, count):
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float32)
                or values.shape != (count, 19, 512) or not np.isfinite(values).all()):
            raise IntegrityError("x_z must be finite float32 with shape (count, 19, 512).")

    def _read_npz(self, path, metadata, count):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if (len(archive.files) != 3
                        or set(archive.files) != {"x_z", "sample_indices", "metadata_json"}):
                    raise IntegrityError("Unexpected or duplicate NPZ members: " + str(path))
                self._array(archive["x_z"], count)
                indices = archive["sample_indices"]
                if (indices.dtype != np.dtype(np.int64) or indices.shape != (count,)
                        or not np.array_equal(indices, np.arange(count, dtype=np.int64))):
                    raise IntegrityError("NPZ sample_indices do not match metadata.")
                encoded = archive["metadata_json"]
                if (encoded.shape != () or encoded.dtype.kind != "U"
                        or encoded.item() != canonical_json_bytes(metadata).decode("utf-8")):
                    raise IntegrityError("NPZ metadata_json does not match its receipt.")
        except IntegrityError:
            raise
        except Exception as exc:
            raise IntegrityError("Cannot validate NPZ: " + str(path)) from exc

    def _quarantine(self, paths):
        for path in paths:
            _store_assert_unlinked(self.root, path)
        existing = [path for path in paths if path.exists() or path.is_symlink()]
        if not existing:
            return
        self._require_lock()
        destination = self.root / "quarantine" / uuid.uuid4().hex
        for path in existing:
            target = destination / path.relative_to(self.root)
            _store_assert_unlinked(self.root, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(path), str(target))
            _store_fsync_dir(path.parent)
            _store_fsync_dir(target.parent)

    def _partials(self, data, receipt):
        _store_assert_unlinked(self.root, data)
        _store_assert_unlinked(self.root, receipt)
        return [partial for path in (data, receipt)
                for partial in path.parent.glob("." + path.name + ".*.tmp")]

    def verify(self, generator, subject, state, expected_meta):
        data, receipt_path = self._paths(generator, subject, state)
        if not isinstance(expected_meta, dict):
            raise IntegrityError("expected_meta must be a dictionary.")
        if receipt_path.is_symlink() or data.is_symlink():
            raise IntegrityError("Shard paths must not be symbolic links.")
        if not receipt_path.exists():
            if data.exists():
                self._quarantine([data] + self._partials(data, receipt_path))
            elif self._locked():
                self._quarantine(self._partials(data, receipt_path))
            return None
        if not data.is_file() or not receipt_path.is_file():
            raise IntegrityError("Committed shard or receipt is missing/not a file: " + str(data))
        try:
            receipt = load_json(receipt_path)
            if (not isinstance(receipt, dict)
                    or set(receipt) != {"schema_version", "file", "sha256", "metadata"}
                    or type(receipt["schema_version"]) is not int
                    or receipt["schema_version"] != 1
                    or receipt["file"] != data.relative_to(self.root).as_posix()
                    or not isinstance(receipt["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is None):
                raise IntegrityError("Invalid shard receipt schema: " + str(receipt_path))
            metadata = receipt["metadata"]
            count = self._metadata(metadata, generator, subject, state)
            for key, expected in expected_meta.items():
                if key not in metadata or canonical_json_bytes(metadata[key]) != canonical_json_bytes(expected):
                    raise IntegrityError("Committed metadata differs from the run: " + str(key))
            if sha256_file(data) != receipt["sha256"]:
                raise IntegrityError("SHA256 mismatch for committed shard: " + str(data))
            self._read_npz(data, metadata, count)
        except IntegrityError:
            raise
        except Exception as exc:
            raise IntegrityError("Cannot validate committed shard: " + str(data)) from exc
        if self._locked():
            self._quarantine(self._partials(data, receipt_path))
        return receipt

    def commit(self, generator, subject, state, x_z, metadata):
        self._require_lock()
        data, receipt_path = self._paths(generator, subject, state)
        count = self._metadata(metadata, generator, subject, state)
        self._array(x_z, count)
        metadata = _store_json_loads(canonical_json_bytes(metadata).decode("utf-8"))
        if self.verify(generator, subject, state, metadata) is not None:
            raise IntegrityError("Refusing to overwrite an already committed shard: " + str(data))
        data.parent.mkdir(parents=True, exist_ok=True)
        temporary = _store_temp_path(data)
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, x_z=x_z,
                                sample_indices=np.arange(count, dtype=np.int64),
                                metadata_json=np.asarray(canonical_json_bytes(metadata).decode("utf-8")))
            handle.flush()
            os.fsync(handle.fileno())
        self._read_npz(temporary, metadata, count)
        os.replace(str(temporary), str(data))
        _store_fsync_dir(data.parent)
        receipt = {"schema_version": 1, "file": data.relative_to(self.root).as_posix(),
                   "sha256": sha256_file(data), "metadata": metadata}
        atomic_json_write(receipt_path, receipt)
        return receipt


"""Pinned-source and frozen-input preflight for the s2_73 runner.

Inlined with storage helpers sha256_file, canonical_json_bytes, load_json,
and IntegrityError supplied by the runner. No project main() is executed.
"""
import hashlib
import importlib
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath


PREFLIGHT_SOURCE_SHA256 = {
    "scripts/data/s2_23_check_real_train_dataset.py": "e880eebee4dfeb79f41da809bbb4f8d6c7d233df68d972702f093f8aef4cde34",
    "scripts/data/s2_24_full_train_loader_sanity.py": "d38ac23a01272f6227d978343f767b1ff771899471b9f8520254602487d95859",
    "scripts/data/s2_32_b0_unet_shape_test.py": "ff45bfe50ade25759c9ae681364cf351e94fd61aa995fda4ba799537b52cb242",
    "scripts/data/s2_33_b0_one_batch_diffusion_sanity.py": "ddb2fd497e4e7953bb757b52b9c55940b1b6b68459fe702bbd1303b0f6754256",
    "scripts/data/s2_39_b0_epsilon_to_v_conversion_sanity.py": "dec78a1fcbd88d77940b80444e365549695aae87ebb1e84c7c3de72c4d9754cb",
    "scripts/data/s2_42_b0_v_final_sampling_real_train_fidelity_audit.py": "b828ade3120e67be4f84c36e591272f05ee6be6f4bf8755ff6f7557896912d90",
    "scripts/data/s2_47_b1_architecture_warmstart_sanity.py": "c3f70738635f7207f9a9566dee0032b222ef5fe8af5a6d46fd0bb3c71a19ed97",
}
PREFLIGHT_REFERENCE_SHA256 = {
    "scripts/data/s2_48_b1_1000_step_subject_conditioned_pilot.py": "20dbacd50e45c7734fa14409c2b1d8864f7a58d039df310f0ef0ebed59dc39c6",
    "scripts/data/s2_56_b2_raw_candidate_subject_aware_spectral_fidelity_sweep.py": "eaef8ff682d9f61fc4bddcd03043d3370b430cfcae79ba0c49193403722bfd52",
}
PREFLIGHT_PROTOCOL_REL = (
    "outputs/protocols/scenario2/utility/b1_vs_b2_utility_protocol_v1/metrics/"
    "s2_72_b1_vs_b2_downstream_utility_protocol_v1.json"
)
PREFLIGHT_PROTOCOL_MANIFEST_REL = (
    "outputs/protocols/scenario2/utility/b1_vs_b2_utility_protocol_v1/metrics/"
    "s2_72_b1_vs_b2_downstream_utility_protocol_manifest_v1.json"
)
PREFLIGHT_B1_ROOT = "outputs/frozen/scenario2/b1/b1_subject_state_raw_epoch010_v1"
PREFLIGHT_B2_ROOT = "outputs/frozen/scenario2/b2/b2_validation_selected_replication_e1_v1"
PREFLIGHT_METADATA_SHA256 = {
    PREFLIGHT_PROTOCOL_REL: "5b98ff09d977006b96ea80e43d4f13f33742745b734d5dfa94bb1eaacb682a22",
    PREFLIGHT_PROTOCOL_MANIFEST_REL: "c33ab53dae4714674b195ca085e94647eb58b5612f57c3bdec3fd25cd449a975",
    PREFLIGHT_B1_ROOT + "/metrics/b1_frozen_selection_v1.json": "950b757151c8ab41df46909373d9da04b8ad06d19462b836072b5ba89530bef6",
    PREFLIGHT_B1_ROOT + "/handoff/b1_to_b2_handoff_v1.json": "80f5eeb679da0edd34573110e8b1d180f0593cf4d4a05bad72d5d524af269b47",
    PREFLIGHT_B2_ROOT + "/metrics/b2_frozen_selection_v1.json": "41a2eb11694463c7a0bd360f84e378150a9bb02775ea67371f95732aadab6df4",
    PREFLIGHT_B2_ROOT + "/metrics/b2_freeze_manifest_v1.json": "adbea8bebee5b5adbcf40dddb4e3a9e6c14664402956fe72330ef072eda27845",
    PREFLIGHT_B2_ROOT + "/handoff/b2_to_utility_handoff_v1.json": "75338107aa7ff41fd54a95c06d3bc677fa5311efd0ed24a3b77dbeea5e1ca8e4",
}
PREFLIGHT_CHANNELS = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8",
    "C3", "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2",
]


def _p73_require(condition, message):
    if not condition:
        raise IntegrityError(message)


def _p73_expect(payload, expected, label):
    for key, value in expected.items():
        _p73_require(payload.get(key) == value, f"{label}: unexpected {key!r}.")


def _p73_repo_output(repo: Path, recorded_path: str) -> Path:
    """Rebase only an unambiguous outputs/... suffix, never another location."""
    parts = PureWindowsPath(str(recorded_path)).parts
    _p73_require(parts.count("outputs") == 1, f"Invalid frozen output path: {recorded_path}")
    suffix = parts[parts.index("outputs"):]
    _p73_require(all(p not in ("", ".", "..") and ":" not in p for p in suffix),
                 f"Unsafe frozen output path: {recorded_path}")
    path = repo.joinpath(*suffix).resolve()
    _p73_require(path.is_relative_to(repo / "outputs"), f"Output path escapes repo: {path}")
    return path


def prepare_frozen_context(repo: Path, torch, np, pd, log=print) -> dict:
    repo = Path(repo).resolve()
    tracked = {}

    def verify(path, expected):
        path = Path(path).resolve()
        _p73_require(path.is_file(), f"Missing frozen input: {path}")
        actual = sha256_file(path)
        _p73_require(actual == expected,
                     f"SHA256 mismatch: {path}\nExpected: {expected}\nActual:   {actual}")
        tracked[str(path)] = actual
        return path

    def recorded(item, path_key="path", hash_key="sha256"):
        return verify(_p73_repo_output(repo, item[path_key]), item[hash_key])

    log("[preflight] Verify pinned project helpers before importing them.")
    source_hashes = dict(PREFLIGHT_SOURCE_SHA256)
    for rel, digest in PREFLIGHT_REFERENCE_SHA256.items():
        if (repo / rel).is_file():
            source_hashes[rel] = digest
    for rel, digest in source_hashes.items():
        path = verify(repo / rel, digest)
        cached = sys.modules.get(path.stem)
        if cached is not None:
            _p73_require(Path(cached.__file__).resolve() == path,
                         f"Project helper already imported from another repo: {path.stem}")
    scripts = repo / "scripts" / "data"
    sys.path.insert(0, str(scripts))
    train_api = importlib.import_module("s2_23_check_real_train_dataset")
    schedule_api = importlib.import_module("s2_33_b0_one_batch_diffusion_sanity")
    reverse_api = importlib.import_module("s2_42_b0_v_final_sampling_real_train_fidelity_audit")
    model_api = importlib.import_module("s2_47_b1_architecture_warmstart_sanity")
    for rel in PREFLIGHT_SOURCE_SHA256:
        module = sys.modules.get(Path(rel).stem)
        _p73_require(module is not None and Path(module.__file__).resolve() == (repo / rel).resolve(),
                     f"Unexpected project helper import origin: {rel}")

    log("[preflight] Verify frozen protocol, selections, and handoffs.")
    metadata = {}
    for rel, digest in PREFLIGHT_METADATA_SHA256.items():
        metadata[rel] = load_json(verify(repo / rel, digest))
    protocol = metadata[PREFLIGHT_PROTOCOL_REL]
    pmanifest = metadata[PREFLIGHT_PROTOCOL_MANIFEST_REL]
    b1sel = metadata[PREFLIGHT_B1_ROOT + "/metrics/b1_frozen_selection_v1.json"]
    b1handoff = metadata[PREFLIGHT_B1_ROOT + "/handoff/b1_to_b2_handoff_v1.json"]
    b2sel = metadata[PREFLIGHT_B2_ROOT + "/metrics/b2_frozen_selection_v1.json"]
    b2manifest = metadata[PREFLIGHT_B2_ROOT + "/metrics/b2_freeze_manifest_v1.json"]
    b2handoff = metadata[PREFLIGHT_B2_ROOT + "/handoff/b2_to_utility_handoff_v1.json"]
    for payload, status in [
        (protocol, "S2_72_B1_VS_B2_UTILITY_PROTOCOL_FROZEN"),
        (pmanifest, "S2_72_UTILITY_PROTOCOL_MANIFEST_COMPLETE"),
        (b1sel, "B1_SELECTION_FROZEN"), (b1handoff, "B1_TO_B2_HANDOFF_READY"),
        (b2sel, "B2_SELECTION_FROZEN"), (b2manifest, "B2_FREEZE_MANIFEST_COMPLETE"),
        (b2handoff, "B2_TO_UTILITY_HANDOFF_FROZEN"),
    ]:
        _p73_expect(payload, {"status": status}, status)
    protocol_path = recorded(pmanifest["protocol_json"])
    _p73_require(protocol_path == repo / PREFLIGHT_PROTOCOL_REL, "Protocol manifest path disagrees.")
    _p73_expect(pmanifest, {"protocol_status": protocol["status"]}, "Protocol manifest")
    _p73_expect(protocol, {"protocol_version": "b1_vs_b2_utility_protocol_v1"}, "Protocol")
    sg, dc, fm = protocol["synthetic_generation"], protocol["data_contract"], protocol["frozen_models"]
    _p73_expect(sg, {
        "batch_size": 64, "diffusion_steps": 1000, "precision": "FP32",
        "sampler": "ancestral DDPM", "prediction_target": "v",
        "generation_master_seed": 20262201, "expected_windows_per_generator": 47835,
        "pool_ratio_per_generator": 1.0, "common_random_numbers_across_B1_B2": True,
        "generation_seed_formula": "generation_seed = 20262201 + 2 * subject_index + eye_state_label",
        "common_rng_scope": "entire ancestral reverse-sampling RNG stream for each matched subject-state shard",
    }, "Generation contract")
    _p73_expect(sg["storage"], {"atomic_write": True, "dtype": "float32"}, "Storage contract")
    _p73_expect(dc, {"subjects": 198}, "Data contract")
    _p73_expect(dc["Train"], {"assigned_blocks": 1980, "eligible_blocks": 1970, "usable_windows": 47835}, "Train")
    _p73_expect(dc["window"], {"channels": 19, "samples": 512, "sampling_rate_hz": 128,
                             "seconds": 4.0, "stride_samples": 256}, "Window")
    _p73_expect(protocol["data_handling_rules"], {"EO_label": 0, "EC_label": 1,
                    "synthetic_training_only": True, "synthetic_samples_never_added_to_validation_or_test": True}, "Data handling")
    _p73_expect(protocol["final_test_policy"], {"current_status": "SEALED",
                    "may_be_opened_during_bulk_generation": False}, "Final Test policy")
    _p73_expect(b2handoff["utility_stage_rules"], {"Final_Test_currently_sealed": True}, "B2 handoff")

    log("[preflight] Hash frozen data metadata and model files (full B2 is hash-only).")
    split_path, scaler_path = recorded(dc["split_manifest"]), recorded(dc["normalization"])
    _p73_require(Path(train_api.FINAL_SPLIT_MANIFEST).resolve() == split_path,
                 "Train helper uses a different split manifest.")
    _p73_require(Path(train_api.NORMALIZATION_FILE).resolve() == scaler_path,
                 "Train helper uses a different normalization file.")
    generator_paths = {name: recorded(fm[name], "generator_path", "generator_file_sha256")
                       for name in ("B1", "B2")}
    full_path = recorded(fm["B2"], "full_checkpoint_path", "full_checkpoint_file_sha256")
    selection_path = recorded(fm["B2"], "selection_record_path", "selection_record_sha256")
    _p73_require(selection_path == repo / PREFLIGHT_B2_ROOT / "metrics/b2_frozen_selection_v1.json",
                 "B2 selection path disagrees.")
    for name, expected_path in [("B1_generator", generator_paths["B1"]),
                                ("B2_generator", generator_paths["B2"]),
                                ("B2_selection", selection_path),
                                ("split_manifest", split_path), ("normalization", scaler_path)]:
        item = pmanifest["frozen_inputs"][name]
        _p73_require(_p73_repo_output(repo, item["path"]) == expected_path and
                     item["sha256"] == tracked[str(expected_path)], f"Manifest disagrees: {name}")
    for name, expected_path in [("frozen_generator", generator_paths["B2"]),
                                ("frozen_full_checkpoint", full_path),
                                ("parent_B1_generator", generator_paths["B1"]),
                                ("selection_record", selection_path)]:
        item = b2manifest["artifacts"][name]
        _p73_require(_p73_repo_output(repo, item["path"]) == expected_path and
                     item["sha256"] == tracked[str(expected_path)], f"B2 freeze manifest disagrees: {name}")

    packages = {name: torch.load(path, map_location="cpu", weights_only=False)
                for name, path in generator_paths.items()}
    b1, b2 = packages["B1"], packages["B2"]
    _p73_expect(b1, {"status": "B1_SUBJECT_STATE_V_FROZEN_GENERATOR", "selected_epoch": 10,
                     "selected_weight_kind": "RAW", "prediction_target": "v", "EMA_enabled": False}, "B1 package")
    _p73_expect(b2, {"status": "B2_VALIDATION_SELECTED_GENERATOR_FROZEN", "model_name": "B2",
                     "model_version": "b2_validation_selected_replication_e1_v1"}, "B2 package")
    for name, package_key in (("B1", "selected_model_state_sha256"), ("B2", "model_state_sha256")):
        state_hash = model_api.state_dict_sha256(packages[name]["model_state_dict"])
        _p73_require(state_hash == fm[name]["model_state_sha256"] == packages[name][package_key],
                     f"Frozen {name} tensor state hash mismatch.")
    _p73_require(b1sel["frozen_artifacts"]["selected_model_state_sha256"] == fm["B1"]["model_state_sha256"],
                 "B1 selection state disagrees.")
    _p73_expect(b2sel["selected_source"], {"run": "REPLICATION", "epoch": 1, "stage": "s2_70",
                    "model_state_sha256": fm["B2"]["model_state_sha256"]}, "B2 selected source")
    _p73_expect(b2["source"], {"run": "REPLICATION", "epoch": 1, "stage": "s2_70",
                    "frozen_full_checkpoint_file_sha256": fm["B2"]["full_checkpoint_file_sha256"]}, "B2 source")
    _p73_expect(b2["lineage"], {"parent_generator_file_sha256": fm["B1"]["generator_file_sha256"],
                    "parent_model_state_sha256": fm["B1"]["model_state_sha256"]}, "B2 lineage")
    _p73_expect(b2sel["parent_B1"], {"generator_file_sha256": fm["B1"]["generator_file_sha256"],
                    "model_state_sha256": fm["B1"]["model_state_sha256"]}, "B2 selection parent")
    contract = b1["effective_contract"]
    _p73_expect(contract, {"prediction_target": "v", "input_channels": 19, "input_samples": 512,
        "sampling_rate_hz": 128, "subject_count": 198, "subject_embedding_dim": 128,
        "subject_projection_output_dim": 256, "eye_state_mapping": {"EO": 0, "EC": 1}}, "B1 contract")
    _p73_require(isinstance(contract["condition_injection_module"], str) and
                 bool(contract["condition_injection_module"]), "Missing conditioning injection target.")
    ids = contract["subject_ids_in_index_order"]
    _p73_require(isinstance(ids, list) and len(ids) == len(set(ids)) == 198 and ids == sorted(ids),
                 "Invalid frozen subject index order.")
    _p73_require(ids == b1handoff["subject_ids_in_index_order"], "B1 mapping / handoff disagreement.")

    scaler = load_json(scaler_path)
    _p73_expect(scaler, {"status": "TRAIN_NORMALIZATION_LOCKED", "fit_split": "Train", "scaler_count": 1,
        "fitting_train_window_count": 47835, "channel_order": PREFLIGHT_CHANNELS,
        "window_length_samples": 512, "window_stride_samples": 256, "sampling_rate_hz": 128}, "Frozen scaler")
    means = np.asarray(scaler["means_uv"], dtype=np.float64)
    stds = np.asarray(scaler["stds_uv"], dtype=np.float64)
    _p73_require(means.shape == stds.shape == (19,) and np.isfinite(means).all() and
                 np.isfinite(stds).all() and (stds > 0).all(), "Invalid frozen channel means/stds.")

    log("[preflight] Build eligible Train index from HDF5 metadata only; heldout waveforms are not read.")
    split = pd.read_csv(split_path)
    needed = {"split", "subject_id", "eye_state", "block_uid", "block_index", "block_eligible",
              "candidate_windows", "usable_windows", "window_usable_mask_sha256"}
    _p73_require(needed.issubset(split.columns), "Missing frozen split-manifest columns.")
    train = split[split["split"].astype(str).eq("Train")].copy()
    eligible = train[train["block_eligible"].map(train_api.as_bool)].copy()
    _p73_require(len(train) == 1980 and len(eligible) == 1970 and
                 train["block_uid"].is_unique, "Train assignment accounting mismatch.")
    records, mapping = train_api.build_train_window_index(split_manifest=split, scaler=scaler)
    _p73_require(mapping == {subject_id: i for i, subject_id in enumerate(ids)},
                 "Train index / frozen model subject mapping disagreement.")
    counts = Counter((int(r.subject_index), int(r.label)) for r in records)
    expected_keys = {(i, label) for i in range(198) for label in (0, 1)}
    _p73_require(set(counts) == expected_keys and sum(counts.values()) == len(records) == 47835 and
                 all(n > 0 for n in counts.values()), "Train subject/state coverage or total mismatch.")
    block_records = defaultdict(list)
    index_rows = []
    for r in records:
        _p73_require(r.subject_id == ids[int(r.subject_index)] and
                     r.eye_state == ("EO", "EC")[int(r.label)], "Window identity mismatch.")
        block_records[r.block_uid].append(r)
        index_rows.append([r.subject_id, int(r.subject_index), r.block_uid, int(r.block_index),
                           r.eye_state, int(r.label), int(r.candidate_window_index), int(r.start_sample)])
    _p73_require(set(block_records) == set(eligible["block_uid"].astype(str)), "Train block coverage mismatch.")
    csv_counts = Counter()
    for _, row in eligible.iterrows():
        uid, subject_id, state = str(row["block_uid"]), str(row["subject_id"]), str(row["eye_state"])
        _p73_require(subject_id in mapping and state in ("EO", "EC"), f"Unknown Train stratum: {uid}")
        n_candidates, n_usable = int(row["candidate_windows"]), int(row["usable_windows"])
        mask = [False] * n_candidates
        for record in block_records[uid]:
            index = int(record.candidate_window_index)
            _p73_require(0 <= index < n_candidates and not mask[index] and
                         int(record.start_sample) == index * 256, f"Window geometry changed: {uid}")
            mask[index] = True
        # Exact frozen canonical_hash(mask.tolist()) rule: compact sorted-key JSON.
        mask_sha = hashlib.sha256(canonical_json_bytes(mask)).hexdigest()
        _p73_require(sum(mask) == n_usable and mask_sha == str(row["window_usable_mask_sha256"]),
                     f"Frozen usable-window mask changed: {uid}")
        csv_counts[(mapping[subject_id], {"EO": 0, "EC": 1}[state])] += n_usable
    _p73_require(csv_counts == counts, "Per-stratum Train index / split counts disagree.")
    pairs = []
    for subject_index, subject_id in enumerate(ids):
        for label, state in enumerate(("EO", "EC")):
            n = counts[(subject_index, label)]
            pairs.append({"subject_id": subject_id, "subject_index": subject_index,
                "eye_state": state, "eye_state_label": label, "expected_count": n,
                "generation_seed": 20262201 + 2 * subject_index + label,
                "batch_sizes": [min(64, n - start) for start in range(0, n, 64)]})
    index_sha = hashlib.sha256(canonical_json_bytes(index_rows)).hexdigest()
    plan_core = {
        "schema_version": "s2_73_paired_1x_generation_plan_v1",
        "protocol_sha256": PREFLIGHT_METADATA_SHA256[PREFLIGHT_PROTOCOL_REL],
        "protocol_version": protocol["protocol_version"], "frozen_models": fm,
        "data_contract": dc, "synthetic_generation": sg,
        "source_sha256": source_hashes,
        "input_sha256": {Path(path).relative_to(repo).as_posix(): digest
                         for path, digest in sorted(tracked.items())},
        "subject_ids_in_index_order": ids, "channel_order": PREFLIGHT_CHANNELS,
        "train_window_index_sha256": index_sha,
        "train_window_index_columns": ["subject_id", "subject_index", "block_uid", "block_index",
                                       "eye_state", "label", "candidate_window_index", "start_sample"],
        "expected_windows_per_generator": 47835, "expected_pairs": 396, "expected_shards": 792,
        "signal_key": "x_z", "signal_units": "frozen_train_global_z",
        "normalization_applied_again_to_synthetic": False, "pairs": pairs,
        "validation_waveforms_accessed": False, "final_test_waveforms_accessed": False,
    }
    log("[preflight] PASS: 47,835 Train windows, 198 subjects, 396 subject/state pairs; both generators verified.")
    return {"protocol": protocol, "protocol_path": protocol_path,
        "protocol_sha256": PREFLIGHT_METADATA_SHA256[PREFLIGHT_PROTOCOL_REL],
        "packages": packages, "state_hash_fn": model_api.state_dict_sha256,
        "model_class": model_api.B1SubjectStateUNet1D,
        "schedule_fn": schedule_api.build_cosine_schedule, "reverse_fn": reverse_api.build_reverse_schedule,
        "scaler": scaler, "pairs": pairs, "plan_core": plan_core,
        "tracked_files": [{"path": path, "sha256": digest} for path, digest in sorted(tracked.items())]}


"""Source-verified s2_73 model and sampler helpers; imported libraries are injected."""

import hashlib
import time


def build_frozen_model(torch, B1SubjectStateUNet1D, frozen_b1: dict, state: dict, device):
    """Use the exact s2_56 constructor recipe for either frozen model state."""
    contract = frozen_b1["effective_contract"]
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in state.values()):
        raise ValueError("Frozen floating-point model state must be FP32.")
    model = B1SubjectStateUNet1D(
        b0_contract=contract["parent_B0_effective_contract"],
        injection_module_name=str(contract["condition_injection_module"]),
        num_subjects=int(contract["subject_count"]),
        subject_embedding_dim=int(contract["subject_embedding_dim"]),
        condition_dim=int(contract["subject_projection_output_dim"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def sample_frozen_shard(
    torch, np, model, subject_index: int, eye_state_label: int,
    n_samples: int, seed: int, reverse_schedule: dict, device, on_batch=None,
):
    """Return normalized FP32 EEG plus a stream trace, with one RNG per shard.

    Equations and draw order follow s2_48.sample_b1_subjects. Only generator
    ownership changes: batches continue one stream seeded once for the shard.
    Callback arguments: 1-based batch index, batch count, batch size, seconds.
    Timing is excluded from the trace so paired B1/B2 traces can match exactly.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("The frozen production sampler requires CUDA.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(subject_index, int) or not 0 <= subject_index < model.num_subjects:
        raise ValueError("Subject index is outside the frozen model mapping.")
    if eye_state_label not in (0, 1):
        raise ValueError("Eye-state label must be EO=0 or EC=1.")
    if torch.is_autocast_enabled():
        raise RuntimeError("Autocast must be disabled for frozen FP32 generation.")
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device != device:
            raise ValueError("Model and sampler CUDA devices differ.")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("Model floating-point tensors must be FP32.")
    for key in ("alpha_bar", "coef_x0", "coef_xt", "posterior_variance"):
        tensor = reverse_schedule[key]
        if tuple(tensor.shape) != (1000,) or tensor.dtype != torch.float32 or tensor.device != device:
            raise ValueError("Reverse schedule must contain 1000 FP32 values on the sampler device.")
        if not torch.isfinite(tensor).all():
            raise ValueError("Reverse schedule contains nonfinite values.")
    alpha_bar = reverse_schedule["alpha_bar"]
    coef_x0 = reverse_schedule["coef_x0"]
    coef_xt = reverse_schedule["coef_xt"]
    posterior_variance = reverse_schedule["posterior_variance"]
    if torch.any(alpha_bar <= 0) or torch.any(alpha_bar >= 1) or torch.any(posterior_variance < 0):
        raise ValueError("Reverse schedule has invalid alpha/variance bounds.")

    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def rng_hash():
        return hashlib.sha256(generator.get_state().cpu().numpy().tobytes(order="C")).hexdigest()

    initial_hash = rng_hash()
    trace = {
        "schema": "s2_73_continuous_subject_state_rng_v1", "seed": int(seed),
        "subject_index": int(subject_index), "eye_state_label": int(eye_state_label),
        "n_samples": n_samples, "batch_size": 64, "timesteps": 1000,
        "initial_rng_state_sha256": initial_hash, "chunks": [],
    }
    output = np.empty((n_samples, 19, 512), dtype=np.float32)
    n_batches = (n_samples + 63) // 64
    with torch.inference_mode():
        for index, start in enumerate(range(0, n_samples, 64)):
            count = min(64, n_samples - start)
            shape = (count, 19, 512)
            before_hash = rng_hash()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
            randn_calls = 1
            state = torch.full((count,), eye_state_label, device=device, dtype=torch.long)
            subjects = torch.full((count,), subject_index, device=device, dtype=torch.long)
            for t in reversed(range(1000)):
                timestep = torch.full((count,), t, device=device, dtype=torch.long)
                v_pred = model(x, timestep, state, subjects)
                if tuple(v_pred.shape) != shape or v_pred.dtype != torch.float32:
                    raise RuntimeError("Unexpected v-prediction shape or dtype.")
                if not torch.isfinite(v_pred).all():
                    raise RuntimeError(f"Nonfinite v prediction at timestep {t}.")
                sqrt_ab = torch.sqrt(alpha_bar[t])
                sqrt_omab = torch.sqrt(1.0 - alpha_bar[t])
                x0_pred = sqrt_ab * x - sqrt_omab * v_pred
                model_mean = coef_x0[t] * x0_pred + coef_xt[t] * x
                if t > 0:
                    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                    randn_calls += 1
                    x = model_mean + torch.sqrt(posterior_variance[t]) * noise
                else:
                    x = model_mean
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"Nonfinite reverse state at timestep {t}.")
            batch = x.detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(batch).all():
                raise RuntimeError("Nonfinite generated batch.")
            output[start:start + count] = batch
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            trace["chunks"].append({
                "batch_index": index + 1, "sample_start": start, "batch_size": count,
                "shape": list(shape), "randn_calls": randn_calls,
                "normal_values_requested": randn_calls * count * 19 * 512,
                "rng_state_before_sha256": before_hash, "rng_state_after_sha256": rng_hash(),
            })
            if on_batch is not None:
                on_batch(index + 1, n_batches, count, elapsed)
    trace["final_rng_state_sha256"] = rng_hash()
    if output.dtype != np.float32 or not np.isfinite(output).all():
        raise RuntimeError("Invalid final normalized EEG shard.")
    return output, trace


"""Execution scheduling, progress, and the command-line entry point."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import math
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback


P73_VERSION = "s2_73_paired_bulk_v1"
P73_TOTAL = 47835
P73_OUTPUT_REL = Path("outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1")


def p73_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def target_window_count(percent, total=P73_TOTAL):
    try:
        value = Decimal(str(percent))
    except InvalidOperation as exc:
        raise ValueError("--stop-at-percent must be a number in (0, 100].") from exc
    if not value.is_finite() or not Decimal(0) < value <= Decimal(100):
        raise ValueError("--stop-at-percent must be in (0, 100].")
    return int((value * total / 100).to_integral_value(rounding=ROUND_CEILING))


def p73_key(generator, pair):
    return f"{generator}/{pair['subject_id']}/{pair['eye_state']}"


def p73_counts(pairs, receipts):
    counts = {name: {"shards": 0, "windows": 0} for name in ("B1", "B2")}
    paired_windows = paired_shards = 0
    for pair in pairs:
        present = []
        for name in ("B1", "B2"):
            found = p73_key(name, pair) in receipts
            present.append(found)
            if found:
                counts[name]["shards"] += 1
                counts[name]["windows"] += pair["expected_count"]
        if all(present):
            paired_shards += 1
            paired_windows += pair["expected_count"]
    return {
        "generators": counts, "paired_subject_states": paired_shards,
        "paired_windows_per_generator": paired_windows,
        "paired_percent": 100.0 * paired_windows / P73_TOTAL,
        "committed_windows_total": sum(row["windows"] for row in counts.values()),
        "committed_shards_total": sum(row["shards"] for row in counts.values()),
    }


def p73_check_trace(metadata, pair):
    trace = metadata.get("rng_trace")
    expected = {
        "schema": "s2_73_continuous_subject_state_rng_v1",
        "seed": pair["generation_seed"], "subject_index": pair["subject_index"],
        "eye_state_label": pair["eye_state_label"], "n_samples": pair["expected_count"],
        "batch_size": 64, "timesteps": 1000,
    }
    if not isinstance(trace, dict):
        raise IntegrityError("Missing RNG trace in a completed shard.")
    for key, value in expected.items():
        if canonical_json_bytes(trace.get(key)) != canonical_json_bytes(value):
            raise IntegrityError("RNG trace mismatch: " + key)
    chunks = trace.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(pair["batch_sizes"]):
        raise IntegrityError("Incorrect RNG chunk count.")
    previous = trace.get("initial_rng_state_sha256")
    start = 0
    for index, (chunk, count) in enumerate(zip(chunks, pair["batch_sizes"]), 1):
        if not isinstance(chunk, dict):
            raise IntegrityError("Invalid RNG chunk.")
        required = {
            "batch_index": index, "sample_start": start, "batch_size": count,
            "shape": [count, 19, 512], "randn_calls": 1000,
            "normal_values_requested": 1000 * count * 19 * 512,
            "rng_state_before_sha256": previous,
        }
        for key, value in required.items():
            if canonical_json_bytes(chunk.get(key)) != canonical_json_bytes(value):
                raise IntegrityError("RNG chunk mismatch: " + key)
        for digest in (chunk.get("rng_state_before_sha256"), chunk.get("rng_state_after_sha256")):
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise IntegrityError("Invalid RNG-state SHA256.")
        previous = chunk["rng_state_after_sha256"]
        start += count
    if trace.get("final_rng_state_sha256") != previous:
        raise IntegrityError("Broken RNG stream at shard end.")
    timings = metadata.get("timing", {}).get("batches")
    if not isinstance(timings, list) or len(timings) != len(chunks):
        raise IntegrityError("Missing batch timing records.")
    for row, count in zip(timings, pair["batch_sizes"]):
        if (row.get("batch_size") != count or not isinstance(row.get("seconds"), (int, float))
                or not math.isfinite(row["seconds"]) or row["seconds"] <= 0):
            raise IntegrityError("Invalid batch timing record.")


def p73_check_pair(pair, receipts):
    keys = [p73_key(name, pair) for name in ("B1", "B2")]
    if all(key in receipts for key in keys):
        first, second = [receipts[key]["metadata"]["rng_trace"] for key in keys]
        if canonical_json_bytes(first) != canonical_json_bytes(second):
            raise IntegrityError("B1/B2 random streams differ for " + keys[0][3:])


def p73_check_prefix(pairs, receipts):
    """Our frozen scheduler creates a prefix, followed by at most one B1 shard."""
    pending_seen = False
    for pair in pairs:
        b1, b2 = [p73_key(name, pair) in receipts for name in ("B1", "B2")]
        if b2 and not b1:
            raise IntegrityError("B2 exists without its scheduled B1 partner.")
        if pending_seen and (b1 or b2):
            raise IntegrityError("Completed files do not follow the frozen generation order.")
        if not (b1 and b2):
            pending_seen = True


def execute_paired_plan(pairs, receipts, store, target, make_metadata, generate, on_commit):
    """The same loop is tested with a tiny injected generator, without CUDA."""
    for pair in pairs:
        if p73_counts(pairs, receipts)["paired_windows_per_generator"] >= target:
            break
        for name in ("B1", "B2"):
            key = p73_key(name, pair)
            if key in receipts:
                continue
            metadata = make_metadata(name, pair)
            values, extra = generate(name, pair)
            if set(metadata).intersection(extra):
                raise IntegrityError("Generator attempted to replace frozen metadata.")
            metadata.update(extra)
            p73_check_trace(metadata, pair)
            # Compare to a completed partner BEFORE committing a mismatched B2.
            partner = receipts.get(p73_key("B1", pair)) if name == "B2" else None
            if partner is not None and canonical_json_bytes(partner["metadata"]["rng_trace"]) != canonical_json_bytes(metadata["rng_trace"]):
                raise IntegrityError("Common-random-stream verification failed before B2 commit.")
            receipt = store.commit(name, pair["subject_id"], pair["eye_state"], values, metadata)
            receipts[key] = receipt
            del values
            p73_check_pair(pair, receipts)
            on_commit(name, pair, receipt)
    counts = p73_counts(pairs, receipts)
    if counts["paired_windows_per_generator"] < target:
        raise IntegrityError("Generation stopped before the requested cumulative target.")
    return counts


def p73_eta(pairs, receipts):
    """Estimate remaining SAMPLING time using observed batch sizes by generator."""
    exact, fallback = defaultdict(list), defaultdict(list)
    for receipt in receipts.values():
        meta = receipt["metadata"]
        for row in meta["timing"]["batches"]:
            exact[(meta["generator_name"], row["batch_size"])].append(row["seconds"])
            fallback[meta["generator_name"]].append(row["seconds"])
    total = 0.0
    for pair in pairs:
        for name in ("B1", "B2"):
            if p73_key(name, pair) in receipts:
                continue
            if not fallback[name]:
                return None
            for count in pair["batch_sizes"]:
                samples = exact.get((name, count)) or fallback[name]
                total += sum(samples) / len(samples)
    return total


def p73_duration(seconds):
    if seconds is None:
        return "pending paired timing data"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def p73_resources(output, torch=None, device=None, pagefiles=False):
    result = {"output_free_gib": shutil.disk_usage(output).free / 2**30}
    if os.name == "nt":
        for drive in ("C:\\", "D:\\"):
            if Path(drive).exists():
                result[drive[0] + "_free_gib"] = shutil.disk_usage(drive).free / 2**30
        if pagefiles:
            command = (
                "Get-CimInstance Win32_PageFileUsage | "
                "Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage | "
                "ConvertTo-Json -Compress"
            )
            try:
                proc = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                result["pagefile_usage_mb"] = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                result["pagefile_usage_mb"] = None
    if torch is not None and device is not None:
        free, total = torch.cuda.mem_get_info(device)
        result.update(gpu_free_gib=free / 2**30, gpu_total_gib=total / 2**30,
                      gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)
    return result


class SafeStop(RuntimeError):
    pass


def p73_disk_check(output, reserve_gib, system_reserve_gib, extra_bytes=0):
    free = shutil.disk_usage(output).free
    required = reserve_gib * 2**30 + extra_bytes
    if free < required:
        raise SafeStop(
            f"Insufficient output disk space: free={free / 2**30:.2f} GiB, "
            f"required={required / 2**30:.2f} GiB. Completed shards remain valid."
        )
    if os.name == "nt":
        drive = Path(os.environ.get("SystemDrive", "C:") + "\\")
        if drive.exists() and shutil.disk_usage(drive).free < system_reserve_gib * 2**30:
            raise SafeStop("Insufficient free space on the Windows system drive.")


class _P73Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file
    def write(self, value):
        self.console.write(value)
        self.file.write(value)
        self.file.flush()
        return len(value)
    def flush(self):
        self.console.flush()
        self.file.flush()
    def isatty(self):
        return False


def p73_runtime(torch, device):
    props = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "gpu_name": props.name, "gpu_memory_bytes": props.total_memory,
        "gpu_compute_capability": [props.major, props.minor],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def p73_main_locked(args, repo, output, run_id):
    started = time.perf_counter()
    receipts, pairs, context, runtime = {}, [], None, None
    target = target_window_count(args.stop_at_percent)
    session_new = {"B1": 0, "B2": 0}
    session_batches = []
    phase_seconds = {}
    current = None
    plan_sha = None
    run_metrics = output / "metrics" / (run_id + ".json")
    latest = output / "progress" / "s2_73_progress_v1.json"
    manifest_path = output / "manifests" / "s2_73_pool_manifest_v1.json"
    tracked_stats = {}
    torch = device = None

    def summary(status, error=None, resources=None):
        return {
            "stage": "s2_73", "implementation_version": P73_VERSION,
            "run_id": run_id, "updated_at_utc": p73_now(), "status": status,
            "requested_stop_at_percent": str(args.stop_at_percent),
            "requested_windows_per_generator": target,
            "full_pool_windows_per_generator": P73_TOTAL,
            "full_pool_expected_shards": 792,
            "generation_plan_sha256": plan_sha,
            "verified_progress": p73_counts(pairs, receipts) if pairs else None,
            "current_uncommitted_work": current,
            "new_committed_windows_this_run": session_new,
            "active_wall_seconds_this_run": time.perf_counter() - started,
            "phase_seconds": phase_seconds,
            "sampling_batch_seconds_this_run": sum(row["seconds"] for row in session_batches),
            "remaining_sampling_seconds_estimate": p73_eta(pairs, receipts) if pairs else None,
            "eta_method": "Per-generator observed batch time; exact-size mean when available. Excludes future startup, pauses, verification and I/O.",
            "runtime": runtime, "resources": resources, "error": error,
            "stored_signal": {"array_key": "x_z", "domain": "frozen_train_global_z", "dtype": "float32"},
            "Final_Test_waveform_accessed_by_this_run": False,
            "Validation_waveform_accessed_by_this_run": False,
            "real_waveform_arrays_read_by_this_run": False,
        }

    def save_progress(status, error=None, resources=None, final=False):
        payload = summary(status, error, resources)
        atomic_json_write(latest, payload)
        if final:
            atomic_json_write(run_metrics, payload)
        return payload

    def save_pool():
        atomic_json_write(manifest_path, {
            "schema": "s2_73_pool_manifest_v1", "generation_plan_sha256": plan_sha,
            "status": "S2_73_FULL_POOL_COMPLETE" if len(receipts) == 792 else "S2_73_POOL_PARTIAL",
            "progress": p73_counts(pairs, receipts), "updated_at_utc": p73_now(),
            "committed": {key: {"file": row["file"], "sha256": row["sha256"],
                                "expected_count": row["metadata"]["expected_count"]}
                          for key, row in sorted(receipts.items())},
            "Final_Test_accessed": False,
        })

    def check_sources(full=False):
        for row in context["tracked_files"]:
            path = Path(row["path"])
            stat = path.stat()
            value = (stat.st_size, stat.st_mtime_ns)
            if value != tracked_stats[str(path)]:
                raise IntegrityError("A frozen input changed during the run: " + str(path))
            if full and sha256_file(path) != row["sha256"]:
                raise IntegrityError("Frozen input hash changed: " + str(path))

    try:
        print(f"\nS2_73 — PAIRED B1/B2 GENERATION — cumulative target {args.stop_at_percent}%")
        print(f"Target: at least {target:,} windows/model, then finish the current pair.")
        print("GPU work is sequential; B1 and B2 progress together. Final Test remains unused.")
        print("Output:", output)
        p73_disk_check(output, args.min_free_gib, args.min_system_free_gib)
        phase_start = time.perf_counter()
        import torch
        import pandas as pd
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use the existing project .venv; no CPU fallback is permitted.")
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("The production sampler requires a CUDA device.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        # Same reproducibility settings used in the supplied sampling sources.
        random.seed(20262201)
        np.random.seed(20262201)
        torch.manual_seed(20262201)
        torch.cuda.manual_seed_all(20262201)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_default_dtype(torch.float32)
        runtime = p73_runtime(torch, device)
        print("GPU:", runtime["gpu_name"], f"({runtime['gpu_memory_bytes'] / 2**30:.2f} GiB)")
        phase_seconds["imports_and_device_setup"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        context = prepare_frozen_context(repo, torch, np, pd, log=lambda x: print(x, flush=True))
        pairs = context["pairs"]
        implementation_path = Path(__file__).resolve()
        script_sha = sha256_file(implementation_path)
        context["tracked_files"].append({"path": str(implementation_path), "sha256": script_sha})
        for row in context["tracked_files"]:
            info = Path(row["path"]).stat()
            tracked_stats[row["path"]] = (info.st_size, info.st_mtime_ns)
        plan = {
            "schema": P73_VERSION, "core": context["plan_core"],
            "implementation_sha256": script_sha, "runtime": runtime,
            "ordering": "subject_index ascending; EO then EC; B1 then B2",
            "array_key": "x_z", "signal_domain": "frozen_train_global_z",
            "pairs": pairs,
        }
        plan_sha = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
        plan_path = output / "protocol" / "s2_73_generation_plan_v1.json"
        if plan_path.exists():
            if canonical_json_bytes(load_json(plan_path)) != canonical_json_bytes(plan):
                raise IntegrityError("Frozen generation plan/runtime/source mismatch. Existing output was preserved.")
        else:
            if manifest_path.exists() or any((output / "generated").rglob("*.npz")) or any((output / "manifests" / "shards").rglob("*.json")):
                raise IntegrityError("Existing generation artifacts lack their frozen plan. Refusing to adopt them.")
            atomic_json_write(plan_path, plan)
        phase_seconds["input_and_train_metadata_preflight"] = time.perf_counter() - phase_start
        store = ShardStore(output)

        def make_metadata(name, pair):
            return {
                "schema_version": 1, "generator_name": name,
                "generator_model_state_sha256": context["protocol"]["frozen_models"][name]["model_state_sha256"],
                "generator_file_sha256": context["protocol"]["frozen_models"][name]["generator_file_sha256"],
                "subject_id": pair["subject_id"], "subject_index": pair["subject_index"],
                "eye_state": pair["eye_state"], "eye_state_label": pair["eye_state_label"],
                "expected_count": pair["expected_count"], "generation_seed": pair["generation_seed"],
                "sample_indices": list(range(pair["expected_count"])),
                "sampling_rate_hz": 128, "n_channels": 19, "n_times": 512,
                "channel_order": context["scaler"]["channel_order"],
                "normalization_sha256": context["protocol"]["data_contract"]["normalization"]["sha256"],
                "signal_domain": "frozen_train_global_z", "array_key": "x_z",
                "protocol_sha256": context["protocol_sha256"], "generation_plan_sha256": plan_sha,
                "batch_size": 64, "batch_sizes": pair["batch_sizes"],
                "diffusion_steps": 1000, "precision": "FP32", "prediction_target": "v",
                "sampler": "ancestral DDPM", "Final_Test_accessed": False,
            }

        phase_start = time.perf_counter()
        previous_manifest = load_json(manifest_path) if manifest_path.exists() else None
        expected_data = {
            f"generated/{name}/{pair['subject_id']}/{pair['eye_state']}.npz"
            for pair in pairs for name in ("B1", "B2")
        }
        expected_receipts = {
            f"manifests/shards/{name}/{pair['subject_id']}/{pair['eye_state']}.json"
            for pair in pairs for name in ("B1", "B2")
        }
        for directory, suffix, allowed in (
            (output / "generated", "*.npz", expected_data),
            (output / "manifests" / "shards", "*.json", expected_receipts),
        ):
            for path in directory.rglob(suffix):
                if path.relative_to(output).as_posix() not in allowed:
                    raise IntegrityError("Unexpected file outside the frozen shard plan: " + str(path))
        print("Validating all previously completed shards before reuse...", flush=True)
        for index, pair in enumerate(pairs, 1):
            for name in ("B1", "B2"):
                receipt = store.verify(name, pair["subject_id"], pair["eye_state"], make_metadata(name, pair))
                if receipt is not None:
                    p73_check_trace(receipt["metadata"], pair)
                    receipts[p73_key(name, pair)] = receipt
            p73_check_pair(pair, receipts)
            if index % 50 == 0:
                print(f"  resume audit: {index}/{len(pairs)} subject-state pairs", flush=True)
        p73_check_prefix(pairs, receipts)
        if previous_manifest is not None:
            if previous_manifest.get("generation_plan_sha256") != plan_sha:
                raise IntegrityError("Pool manifest belongs to a different generation plan.")
            if not isinstance(previous_manifest.get("committed"), dict):
                raise IntegrityError("Invalid pool manifest.")
            for key, record in previous_manifest["committed"].items():
                if (key not in receipts or receipts[key]["file"] != record["file"]
                        or receipts[key]["sha256"] != record["sha256"]):
                    raise IntegrityError("Previously committed work is missing or changed: " + key)
        phase_seconds["resume_integrity_audit"] = time.perf_counter() - phase_start
        save_pool()
        resources = p73_resources(output, torch, device, pagefiles=True)
        print("Initial resources:", json.dumps(resources, ensure_ascii=False), flush=True)
        progress = p73_counts(pairs, receipts)
        print(f"Verified start: B1={progress['generators']['B1']['windows']:,}, "
              f"B2={progress['generators']['B2']['windows']:,}; paired={progress['paired_percent']:.4f}%")
        save_progress("S2_73_PREFLIGHT_PASSED", resources=resources)
        if progress["paired_windows_per_generator"] >= target:
            check_sources(full=True)
            status = "S2_73_FULL_POOL_COMPLETE" if len(receipts) == 792 else "S2_73_TARGET_ALREADY_SATISFIED_PARTIAL"
            save_progress(status, resources=resources, final=True)
            print(status + " — no new samples were generated.")
            return 0

        phase_start = time.perf_counter()
        # One model occupies the GPU. CPU checkpoint dictionaries are reused for switches.
        model = build_frozen_model(torch, context["model_class"], context["packages"]["B1"],
                                   context["packages"]["B1"]["model_state_dict"], device)
        model.requires_grad_(False)
        loaded_name = "B1"
        schedule_cpu = context["schedule_fn"](timesteps=1000, cosine_s=0.008)
        reverse = context["reverse_fn"](schedule_cpu, device)
        phase_seconds["model_and_schedule_setup"] = time.perf_counter() - phase_start
        print("Generation starts. Only committed files count toward the percentage.", flush=True)

        def generate(name, pair):
            nonlocal current, loaded_name
            check_sources()
            count = pair["expected_count"]
            p73_disk_check(output, args.min_free_gib, args.min_system_free_gib, count * 19 * 512 * 4 * 2)
            if loaded_name != name:
                model.load_state_dict(context["packages"][name]["model_state_dict"], strict=True)
                model.eval()
                loaded_name = name
            current = {"generator": name, "subject": pair["subject_id"], "state": pair["eye_state"],
                       "expected_count": count, "generated_but_uncommitted": 0}
            batch_rows = []
            print(f"\n[{name}] {pair['subject_id']} {pair['eye_state']} n={count} "
                  f"seed={pair['generation_seed']}", flush=True)
            save_progress("S2_73_GENERATING")

            def on_batch(index, total, n, seconds):
                row = {"batch_index": index, "batch_size": n, "seconds": seconds}
                batch_rows.append(row)
                session_batches.append({"generator": name, **row})
                current["generated_but_uncommitted"] += n
                print(f"  batch {index}/{total}: {n} windows in {seconds:.2f}s "
                      f"({n / seconds:.3f} windows/s); shard still uncommitted", flush=True)
                save_progress("S2_73_GENERATING")
                p73_disk_check(output, args.min_free_gib, args.min_system_free_gib, count * 19 * 512 * 4 * 2)

            shard_start = time.perf_counter()
            values, trace = sample_frozen_shard(
                torch, np, model, pair["subject_index"], pair["eye_state_label"], count,
                pair["generation_seed"], reverse, device, on_batch=on_batch,
            )
            if context["state_hash_fn"](model.state_dict()) != context["protocol"]["frozen_models"][name]["model_state_sha256"]:
                raise IntegrityError("Model state changed during inference.")
            check_sources()
            return values, {
                "created_at_utc": p73_now(), "rng_trace": trace,
                "timing": {"batches": batch_rows,
                           "sampling_seconds": sum(row["seconds"] for row in batch_rows),
                           "generation_call_wall_seconds": time.perf_counter() - shard_start},
            }

        def on_commit(name, pair, receipt):
            nonlocal current
            session_new[name] += pair["expected_count"]
            current = None
            save_pool()
            resources = p73_resources(output, torch, device)
            save_progress("S2_73_GENERATING", resources=resources)
            counts = p73_counts(pairs, receipts)
            print(f"  COMMITTED {receipt['file']} SHA256={receipt['sha256']}")
            print(f"  B1={counts['generators']['B1']['windows']:,}/{P73_TOTAL:,}; "
                  f"B2={counts['generators']['B2']['windows']:,}/{P73_TOTAL:,}; "
                  f"paired progress={counts['paired_percent']:.4f}%; "
                  f"remaining sampling~{p73_duration(p73_eta(pairs, receipts))}; "
                  f"disk free={resources['output_free_gib']:.2f} GiB", flush=True)

        phase_start = time.perf_counter()
        progress = execute_paired_plan(pairs, receipts, store, target, make_metadata, generate, on_commit)
        phase_seconds["generation_and_commits"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        check_sources(full=True)
        p73_check_prefix(pairs, receipts)
        for pair in pairs:
            p73_check_pair(pair, receipts)
        phase_seconds["final_input_and_pair_audit"] = time.perf_counter() - phase_start
        status = "S2_73_FULL_POOL_COMPLETE" if progress["committed_shards_total"] == 792 else "S2_73_TARGET_REACHED_PARTIAL"
        resources = p73_resources(output, torch, device, pagefiles=True)
        save_pool()
        payload = save_progress(status, resources=resources, final=True)
        print("\n" + status)
        print(f"B1: {progress['generators']['B1']['windows']:,}; B2: {progress['generators']['B2']['windows']:,}")
        print(f"Paired progress: {progress['paired_percent']:.4f}%")
        print("Active time this run:", p73_duration(payload["active_wall_seconds_this_run"]))
        print("Observed sampling time:", p73_duration(payload["sampling_batch_seconds_this_run"]))
        print("Estimated remaining sampling time:", p73_duration(payload["remaining_sampling_seconds_estimate"]))
        print("Final Test accessed: NO")
        print("RUN REPORT:", run_metrics)
        print("LATEST PROGRESS:", latest)
        print("All committed samples are retained for the next cumulative target.", flush=True)
        return 0
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status, code = "S2_73_INTERRUPTED_RESUMABLE", 130
        elif isinstance(exc, SafeStop):
            status, code = "S2_73_STOPPED_INSUFFICIENT_DISK", 2
        else:
            status, code = "S2_73_STOPPED_ERROR", 1
        error = {"type": type(exc).__name__, "message": str(exc)}
        try:
            save_progress(status, error=error, final=True)
        except Exception as report_error:
            print("Could not write the latest report:", report_error, file=sys.stderr)
        print("\n" + status, file=sys.stderr)
        if not isinstance(exc, (KeyboardInterrupt, SafeStop)):
            traceback.print_exc()
        else:
            print(str(exc) or "Interrupted by the user.", file=sys.stderr)
        print("Completed shard receipts remain authoritative. Re-run the same command to resume after resolving the cause.", file=sys.stderr)
        print("RUN REPORT:", run_metrics, file=sys.stderr)
        return code


def main():
    parser = argparse.ArgumentParser(
        description="Frozen paired B1/B2 generation. Default stops near 2% cumulatively; safely resumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stop-at-percent", default="2", help="Cumulative target, not an additional percentage.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-free-gib", type=float, default=4.0, help="Output disk free-space reserve.")
    parser.add_argument("--min-system-free-gib", type=float, default=1.0, help="Windows system-drive reserve.")
    args = parser.parse_args()
    try:
        target_window_count(args.stop_at_percent)
        if any(not math.isfinite(value) or value < 0 for value in (args.min_free_gib, args.min_system_free_gib)):
            raise ValueError("Disk reserves must be finite and nonnegative.")
        repo = args.repo_root.resolve(strict=True)
        if not (repo / "scripts/data").is_dir():
            raise ValueError("--repo-root must contain scripts/data.")
        output = repo / P73_OUTPUT_REL
        # Reject an output path redirected into unrelated/frozen directories.
        cursor = repo
        for part in P73_OUTPUT_REL.parts:
            cursor /= part
            if cursor.is_symlink() or (cursor.exists() and getattr(cursor.lstat(), "st_file_attributes", 0) & 0x400):
                raise IntegrityError("Output paths must not contain symlinks/junctions.")
        if output.exists():
            for parent, directories, files in os.walk(output, followlinks=False):
                for name in directories + files:
                    path = Path(parent) / name
                    if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                        raise IntegrityError("Existing output contains a symlink/junction: " + str(path))
        with output_lock(output):
            run_id = "s2_73_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            log_path = output / "logs" / (run_id + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("x", encoding="utf-8", buffering=1) as handle:
                out, err = _P73Tee(sys.stdout, handle), _P73Tee(sys.stderr, handle)
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    print("LOG:", log_path, flush=True)
                    return p73_main_locked(args, repo, output, run_id)
    except Exception as exc:
        print(f"S2_73 could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
