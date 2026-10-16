"""s2_74C — resumable B0 1x generation for the subject-conditioning ablation.

Purpose
-------
Generate one frozen 1x synthetic pool from the already-frozen B0 state-only
model, using the EXACT same subject×eye-state counts, per-stratum generation
seeds, ancestral DDPM sampler, FP32 precision, 1000 reverse steps, batch
chunking, and random-number streams used by the completed B1/B2 s2_73 pools.

This is a SUPPLEMENTARY / DESCRIPTIVE ABLATION. It does NOT modify the frozen
s2_72 downstream-utility protocol, does NOT add B0 to the eight s2_75 utility
conditions, does NOT retrain any generator, and never reads Validation or Final
Test waveforms.

Scientific interpretation boundary
----------------------------------
B0 is p(EEG | eye_state), not p(EEG | subject, eye_state).  Therefore the
subject_id attached to each B0 shard is ONLY an evaluation stratum used to:
  * match the Real Train subject×state sample count,
  * reuse the same random stream as the corresponding B1/B2 shard, and
  * enable a fair subject-conditioning ablation downstream.
The subject_id is NEVER passed into B0.

Run
---
Save under scripts/data and run from the repository root:

    python -u scripts/data/s2_74c_resumable_b0_subject_conditioning_ablation_generation.py --stop-at-percent 2

A later --stop-at-percent 100 means reach 100% TOTAL; committed shards are
validated and reused.  A scheduled stop occurs only after a complete B0 shard.

Output
------
outputs/runs/scenario2/ablation/b0_state_only_1x_s2_74c_v1/

NPZ members are identical in shape/schema to s2_73:
  x_z             float32, (n, 19, 512), already in frozen Train-global z space
  sample_indices  int64, 0..n-1
  metadata_json   canonical JSON scalar

Do NOT normalize x_z again and do NOT call x_z microvolts.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import stat
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING

import numpy as np


VERSION = "s2_74c_b0_subject_conditioning_ablation_generation_v1"
TOTAL_WINDOWS = 47_835
EXPECTED_PAIRS = 396
OUTPUT_REL = Path("outputs/runs/scenario2/ablation/b0_state_only_1x_s2_74c_v1")

# Exact source that produced the already-completed B1/B2 pools.
S73_REL = Path("scripts/data/s2_73_resumable_bulk_b1_b2_synthetic_generation.py")
S73_SHA256 = "b82c964a19e85117fe883736374d5387d09f508ce7021bac98eeb62bd1a179c6"
S74_REL = Path("scripts/data/s2_74_audit_and_freeze_half_subsets.py")
S74_SHA256 = "749fb1a484deccff4eb41d3fbffd452e987348691fbf814ca304f55efd167fee"
S45_REL = Path("scripts/data/s2_45_freeze_b0_selection.py")
S45_SHA256 = "5330256d79d2efe414a6a6b4b264bd9aaa284eb40f1fe04c6592576d3e1dbbb5"

# s2_73/s2_74 artifacts already verified by the completed s2_74B analysis plan.
S73_PLAN_REL = Path(
    "outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1/"
    "protocol/s2_73_generation_plan_v1.json"
)
S73_PLAN_FILE_SHA256 = "483f59eed67f6303048fe131338c46f3438c60d47ec08fc698db7f78a55c4f12"
S73_PLAN_CANONICAL_SHA256 = "3833d824e31b350728893184114b639c6a2b4659be95e0d651b1c6eb69c10998"
S73_POOL_MANIFEST_REL = Path(
    "outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1/"
    "manifests/s2_73_pool_manifest_v1.json"
)
S73_POOL_MANIFEST_FILE_SHA256 = "befe649a3d2f6c7087329ad9eca9e7484d77b8d66d14d11c27a802e7a432a03f"
S74_FREEZE_REL = Path(
    "outputs/frozen/scenario2/utility/b1_b2_pool_integrity_half_subset_v1/"
    "manifests/s2_74_freeze_manifest_v1.json"
)
S74_FREEZE_FILE_SHA256 = "7928c9f7be94b369240ab2145f480a478dfaa14c65e9fc4799f36671bfa3b1c5"
S73_OUTPUT_REL = Path("outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1")

# Frozen B0 package created by s2_45.
B0_ROOT_REL = Path("outputs/frozen/scenario2/b0/b0_v_raw_epoch020_v1")
B0_GENERATOR_REL = B0_ROOT_REL / "checkpoints/b0_v_raw_epoch020_generator_v1.pt"
B0_ARCHIVE_REL = B0_ROOT_REL / "checkpoints/b0_v_raw_epoch020_full_checkpoint_v1.pt"
B0_SELECTION_REL = B0_ROOT_REL / "metrics/b0_frozen_selection_v1.json"
B0_HANDOFF_REL = B0_ROOT_REL / "handoff/b0_to_b1_handoff_v1.json"

CHANNELS = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8",
    "C3", "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2",
]


class IntegrityError(RuntimeError):
    pass


class SafeStop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_bytes(obj):
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _pairs_no_duplicates(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("Duplicate JSON key: " + key)
        out[key] = value
    return out


def _bad_constant(value):
    raise ValueError("Non-finite JSON value: " + value)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.loads(
            f.read(), object_pairs_hook=_pairs_no_duplicates, parse_constant=_bad_constant
        )


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state_dict):
    h = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(json.dumps(list(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()


def fsync_dir(path):
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def temp_path(path):
    return path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(path)
    try:
        with tmp.open("xb") as f:
            f.write(canonical_bytes(obj) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
        fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def assert_no_link(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise IntegrityError("Path escapes output root: " + str(path)) from exc
    current = root
    for part in (None,) + rel.parts:
        if part is not None:
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise IntegrityError("Symlink/reparse point in output path: " + str(current))


_ACTIVE_LOCKS = set()


@contextlib.contextmanager
def output_lock(root):
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    assert_no_link(root, root)
    root = root.resolve()
    key = str(root)
    if key in _ACTIVE_LOCKS:
        raise IntegrityError("Output root already locked by this process.")
    lock_path = root / ".s2_74c.lock"
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
                raise IntegrityError("Another process holds the s2_74C output lock.") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise IntegrityError("Another process holds the s2_74C output lock.") from exc
        locked = True
        _ACTIVE_LOCKS.add(key)
        yield root
    finally:
        if locked:
            _ACTIVE_LOCKS.discard(key)
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class B0Store:
    """Atomic B0 shard + authenticated JSON receipt."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def paths(self, subject, state):
        require(re.fullmatch(r"sub-[0-9]+", subject) is not None, "Invalid subject ID")
        require(state in ("EO", "EC"), "Invalid eye state")
        data = self.root / "generated/B0" / subject / f"{state}.npz"
        receipt = self.root / "manifests/shards/B0" / subject / f"{state}.json"
        assert_no_link(self.root, data)
        assert_no_link(self.root, receipt)
        return data, receipt

    @staticmethod
    def validate_metadata(meta, subject, state):
        require(isinstance(meta, dict), "Shard metadata is not an object")
        require(meta.get("generator_name") == "B0", "generator_name mismatch")
        require(meta.get("subject_id") == subject, "subject_id mismatch")
        require(meta.get("eye_state") == state, "eye_state mismatch")
        n = meta.get("expected_count")
        require(type(n) is int and n > 0, "Invalid expected_count")
        idx = meta.get("sample_indices")
        require(
            isinstance(idx, list) and len(idx) == n
            and all(type(v) is int and v == i for i, v in enumerate(idx)),
            "sample_indices must be exactly 0..n-1",
        )
        require(meta.get("subject_condition_used") is False, "B0 must not use subject conditioning")
        require(
            meta.get("subject_id_role") == "evaluation_stratum_only_not_model_condition",
            "B0 subject-id role is not explicit",
        )
        canonical_bytes(meta)
        return n

    @staticmethod
    def validate_array(x, n):
        require(isinstance(x, np.ndarray), "x_z is not ndarray")
        require(x.dtype == np.float32, "x_z must be float32")
        require(x.shape == (n, 19, 512), "x_z shape mismatch")
        require(np.isfinite(x).all(), "x_z contains nonfinite values")

    def validate_npz(self, path, meta, n):
        try:
            with np.load(path, allow_pickle=False) as z:
                require(set(z.files) == {"x_z", "sample_indices", "metadata_json"}, "Unexpected NPZ members")
                self.validate_array(z["x_z"], n)
                idx = z["sample_indices"]
                require(idx.dtype == np.int64 and idx.shape == (n,), "NPZ sample_indices schema mismatch")
                require(np.array_equal(idx, np.arange(n, dtype=np.int64)), "NPZ sample_indices mismatch")
                mj = z["metadata_json"]
                require(mj.shape == () and mj.dtype.kind == "U", "metadata_json schema mismatch")
                require(mj.item() == canonical_bytes(meta).decode("utf-8"), "Embedded metadata mismatch")
        except IntegrityError:
            raise
        except Exception as exc:
            raise IntegrityError("Cannot validate NPZ: " + str(path)) from exc

    def quarantine_orphans(self, data, receipt):
        candidates = []
        if data.exists() and not receipt.exists():
            candidates.append(data)
        for base in (data, receipt):
            if base.parent.exists():
                candidates.extend(base.parent.glob("." + base.name + ".*.tmp"))
        if not candidates:
            return
        qroot = self.root / "quarantine" / uuid.uuid4().hex
        for p in candidates:
            if not p.exists():
                continue
            target = qroot / p.relative_to(self.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(p), str(target))

    def verify(self, subject, state, expected_meta):
        data, rp = self.paths(subject, state)
        self.quarantine_orphans(data, rp)
        if not rp.exists():
            return None
        require(data.is_file() and rp.is_file(), "Committed shard/receipt missing")
        rec = load_json(rp)
        require(set(rec) == {"schema_version", "file", "sha256", "metadata"}, "Receipt schema mismatch")
        require(rec["schema_version"] == 1, "Receipt version mismatch")
        require(rec["file"] == data.relative_to(self.root).as_posix(), "Receipt file mismatch")
        require(re.fullmatch(r"[0-9a-f]{64}", rec["sha256"]) is not None, "Bad receipt SHA")
        meta = rec["metadata"]
        n = self.validate_metadata(meta, subject, state)
        for key, value in expected_meta.items():
            require(key in meta and canonical_bytes(meta[key]) == canonical_bytes(value),
                    "Committed metadata differs: " + key)
        require(sha256_file(data) == rec["sha256"], "Committed shard SHA mismatch")
        self.validate_npz(data, meta, n)
        return rec

    def commit(self, subject, state, x_z, meta):
        data, rp = self.paths(subject, state)
        n = self.validate_metadata(meta, subject, state)
        self.validate_array(x_z, n)
        require(self.verify(subject, state, meta) is None, "Refusing to overwrite committed B0 shard")
        data.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path(data)
        try:
            with tmp.open("xb") as f:
                np.savez_compressed(
                    f,
                    x_z=x_z,
                    sample_indices=np.arange(n, dtype=np.int64),
                    metadata_json=np.asarray(canonical_bytes(meta).decode("utf-8")),
                )
                f.flush()
                os.fsync(f.fileno())
            self.validate_npz(tmp, meta, n)
            os.replace(str(tmp), str(data))
            fsync_dir(data.parent)
        finally:
            if tmp.exists():
                tmp.unlink()
        rec = {
            "schema_version": 1,
            "file": data.relative_to(self.root).as_posix(),
            "sha256": sha256_file(data),
            "metadata": json.loads(canonical_bytes(meta).decode("utf-8")),
        }
        atomic_json(rp, rec)
        return rec


def import_exact(path, expected_sha, module_name):
    path = Path(path).resolve()
    require(path.is_file(), "Missing required source: " + str(path))
    actual = sha256_file(path)
    require(actual == expected_sha, f"Source SHA mismatch: {path}\nExpected {expected_sha}\nActual   {actual}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "Cannot import pinned source")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def path_matches_recorded(recorded, actual):
    """Compare only the unique outputs/... suffix; old drive letters are irrelevant."""
    parts = re.split(r"[\\/]", str(recorded))
    require(parts.count("outputs") == 1, "Recorded path has ambiguous outputs suffix")
    suffix = Path(*parts[parts.index("outputs"):])
    return suffix.as_posix() == Path(actual).as_posix()


def target_windows(percent):
    try:
        p = Decimal(str(percent))
    except InvalidOperation as exc:
        raise ValueError("--stop-at-percent must be numeric") from exc
    if not p.is_finite() or not Decimal(0) < p <= Decimal(100):
        raise ValueError("--stop-at-percent must be in (0,100]")
    return int((p * TOTAL_WINDOWS / Decimal(100)).to_integral_value(rounding=ROUND_CEILING))


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"


def pair_key(pair):
    return f"B0/{pair['subject_id']}/{pair['eye_state']}"


def validate_pair_contract(pairs):
    require(len(pairs) == EXPECTED_PAIRS, "Expected 396 subject-state pairs")
    subjects = []
    total = 0
    for i, pair in enumerate(pairs):
        si = i // 2
        label = i % 2
        state = ("EO", "EC")[label]
        require(pair["subject_index"] == si, "Pair ordering subject_index mismatch")
        require(pair["eye_state_label"] == label and pair["eye_state"] == state, "Pair ordering state mismatch")
        if label == 0:
            subjects.append(pair["subject_id"])
        else:
            require(pair["subject_id"] == subjects[-1], "EO/EC subject mismatch")
        n = int(pair["expected_count"])
        require(n > 0, "Empty subject-state stratum")
        require(pair["generation_seed"] == 20262201 + 2 * si + label, "Generation seed mismatch")
        expected_batches = [min(64, n - start) for start in range(0, n, 64)]
        require(pair["batch_sizes"] == expected_batches, "Frozen chunking mismatch")
        total += n
    require(len(subjects) == 198 and len(set(subjects)) == 198, "Subject mapping must contain 198 unique subjects")
    require(total == TOTAL_WINDOWS, f"Expected {TOTAL_WINDOWS} total windows, found {total}")
    return subjects


def load_reference_rng(repo, s73, pairs, pool_manifest, s73_plan_sha):
    """Validate B1/B2 receipts and return the exact frozen trace per stratum.

    This reads only JSON receipts; no B1/B2 waveform arrays are opened here.
    s2_74's pinned freeze manifest is the authority that those pools were fully
    hash/schema/finite audited before this supplementary ablation.
    """
    root = repo / S73_OUTPUT_REL
    traces = {}
    receipt_records = []
    committed = pool_manifest["committed"]
    for pair in pairs:
        found = []
        for g in ("B1", "B2"):
            key = f"{g}/{pair['subject_id']}/{pair['eye_state']}"
            rp = root / "manifests/shards" / g / pair["subject_id"] / f"{pair['eye_state']}.json"
            require(rp.is_file(), "Missing frozen RNG receipt: " + str(rp))
            rec = load_json(rp)
            require(rec.get("file") == committed[key]["file"], "Receipt/pool manifest file mismatch")
            require(rec.get("sha256") == committed[key]["sha256"], "Receipt/pool manifest shard SHA mismatch")
            meta = rec.get("metadata", {})
            require(meta.get("generator_name") == g, "Reference receipt generator mismatch")
            require(meta.get("subject_id") == pair["subject_id"], "Reference receipt subject mismatch")
            require(meta.get("eye_state") == pair["eye_state"], "Reference receipt state mismatch")
            require(meta.get("expected_count") == pair["expected_count"], "Reference receipt count mismatch")
            require(meta.get("generation_seed") == pair["generation_seed"], "Reference receipt seed mismatch")
            require(meta.get("generation_plan_sha256") == s73_plan_sha, "Reference receipt generation-plan mismatch")
            s73.p73_check_trace(meta, pair)
            found.append(meta["rng_trace"])
            receipt_records.append([rp.relative_to(repo).as_posix(), sha256_file(rp)])
        require(canonical_bytes(found[0]) == canonical_bytes(found[1]), "Frozen B1/B2 RNG traces differ")
        traces[pair["subject_id"] + "/" + pair["eye_state"]] = found[0]
    return traces, canonical_sha(sorted(receipt_records))


def prepare(repo, torch):
    repo = Path(repo).resolve()
    s73_path = repo / S73_REL
    s74_path = repo / S74_REL
    s45_path = repo / S45_REL
    require(sha256_file(s74_path) == S74_SHA256, "s2_74 source changed")
    require(sha256_file(s45_path) == S45_SHA256, "s2_45 B0 freeze source changed")
    s73 = import_exact(s73_path, S73_SHA256, "_s2_73_b0_ablation_source")

    # Re-run the original s2_73 preflight. It reads Train metadata only and
    # verifies the frozen B1/B2/data/protocol lineage used to define the pairs.
    import pandas as pd
    context = s73.prepare_frozen_context(repo, torch, np, pd, log=lambda x: print(x, flush=True))
    pairs = context["pairs"]
    subjects = validate_pair_contract(pairs)

    # Verify the exact completed/audited B1/B2 plan and pool from which the
    # common random streams are inherited.
    plan_path = repo / S73_PLAN_REL
    pool_path = repo / S73_POOL_MANIFEST_REL
    freeze_path = repo / S74_FREEZE_REL
    for path, digest, label in (
        (plan_path, S73_PLAN_FILE_SHA256, "s2_73 plan file"),
        (pool_path, S73_POOL_MANIFEST_FILE_SHA256, "s2_73 pool manifest"),
        (freeze_path, S74_FREEZE_FILE_SHA256, "s2_74 freeze manifest"),
    ):
        require(path.is_file(), "Missing " + label + ": " + str(path))
        require(sha256_file(path) == digest, label + " SHA256 changed")
    plan = load_json(plan_path)
    plan_sha = canonical_sha(plan)
    require(plan_sha == S73_PLAN_CANONICAL_SHA256, "Unexpected canonical s2_73 plan SHA")
    require(plan.get("implementation_sha256") == S73_SHA256, "s2_73 plan used another implementation")
    require(canonical_bytes(plan.get("pairs")) == canonical_bytes(pairs), "Current preflight pairs differ from frozen s2_73 plan")
    pool = load_json(pool_path)
    require(pool.get("status") == "S2_73_FULL_POOL_COMPLETE", "s2_73 full pool is not complete")
    require(pool.get("generation_plan_sha256") == plan_sha, "s2_73 pool plan mismatch")
    require(len(pool.get("committed", {})) == 792, "s2_73 pool does not contain 792 committed shards")
    freeze = load_json(freeze_path)
    require(freeze.get("status") == "S2_74_POOL_INTEGRITY_AND_0P5X_FROZEN", "s2_74 is not frozen")
    require(freeze.get("implementation_sha256") == S74_SHA256, "s2_74 implementation mismatch")
    vp = freeze["verified_pool"]
    require(vp["B1"]["shards"] == 396 and vp["B1"]["windows"] == TOTAL_WINDOWS, "s2_74 B1 count mismatch")
    require(vp["B2"]["shards"] == 396 and vp["B2"]["windows"] == TOTAL_WINDOWS, "s2_74 B2 count mismatch")
    require(vp["all_B1_B2_common_rng_streams_verified"] is True, "s2_74 did not verify common RNG")
    require(freeze["scientific_boundary"]["Final_Test_waveforms_accessed"] is False, "Final Test boundary violated")

    reference_traces, reference_receipt_aggregate = load_reference_rng(repo, s73, pairs, pool, plan_sha)

    # Frozen B0 identity.
    sel_path = repo / B0_SELECTION_REL
    handoff_path = repo / B0_HANDOFF_REL
    gen_path = repo / B0_GENERATOR_REL
    archive_path = repo / B0_ARCHIVE_REL
    for p in (sel_path, handoff_path, gen_path, archive_path):
        require(p.is_file(), "Missing frozen B0 artifact: " + str(p))
    selection = load_json(sel_path)
    handoff = load_json(handoff_path)
    require(selection.get("status") == "B0_SELECTION_FROZEN", "B0 selection is not frozen")
    require(selection.get("b0_frozen") is True, "B0 frozen flag false")
    require(selection.get("selected_model") == "B0_STATE_ONLY_DDPM_V", "Unexpected B0 model")
    require(selection.get("selected_epoch") == 20, "B0 epoch must be RAW20")
    require(selection.get("selected_weight_kind") == "raw", "B0 selected weight kind must be raw")
    require(selection.get("prediction_target") == "v", "B0 must use v prediction")
    require(handoff.get("status") == "B0_TO_B1_HANDOFF_READY", "B0 handoff status mismatch")
    require(handoff.get("selected_model_state_sha256") == selection["frozen_artifacts"]["selected_model_state_sha256"],
            "B0 selection/handoff model-state mismatch")
    require(handoff.get("current_conditioning") == {"eye_state": True, "subject": False, "EO": 0, "EC": 1},
            "B0 conditioning contract mismatch")
    require(handoff.get("prediction_target") == "v", "B0 handoff target mismatch")
    require(handoff.get("input_shape") == [19, 512] and handoff.get("sampling_rate_hz") == 128,
            "B0 signal contract mismatch")
    require(path_matches_recorded(selection["frozen_artifacts"]["lean_generator_checkpoint"], B0_GENERATOR_REL),
            "B0 selection points to another lean generator")
    require(path_matches_recorded(selection["frozen_artifacts"]["full_archival_checkpoint"], B0_ARCHIVE_REL),
            "B0 selection points to another archival checkpoint")
    require(sha256_file(archive_path) == selection["frozen_artifacts"]["full_archival_checkpoint_sha256"],
            "Frozen B0 archival checkpoint changed")

    package = torch.load(gen_path, map_location="cpu", weights_only=False)
    require(package.get("status") == "B0_V_FROZEN_GENERATOR", "Unexpected B0 generator status")
    require(package.get("model_name") == "B0_STATE_ONLY_DDPM_V_FROZEN", "Unexpected B0 generator model_name")
    require(package.get("selected_epoch") == 20 and package.get("selected_weight_kind") == "raw", "Unexpected B0 generator identity")
    require(package.get("prediction_target") == "v", "B0 generator is not v-prediction")
    require(package.get("selected_model_state_sha256") == handoff["selected_model_state_sha256"],
            "B0 generator/selection state hash mismatch")
    actual_state_sha = state_dict_sha256(package["model_state_dict"])
    require(actual_state_sha == package["selected_model_state_sha256"], "B0 generator tensor hash mismatch")
    require(sha256_file(archive_path) == package["selected_source_checkpoint_sha256"],
            "B0 frozen archive no longer matches selected source bytes")
    generator_file_sha = sha256_file(gen_path)

    # Import exact B0 constructor from the already-pinned source checked by s2_73.
    scripts = repo / "scripts/data"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    b0_api = importlib.import_module("s2_32_b0_unet_shape_test")
    require(Path(b0_api.__file__).resolve() == (scripts / "s2_32_b0_unet_shape_test.py").resolve(), "Unexpected B0 constructor origin")

    tracked = {Path(row["path"]).resolve(): row["sha256"] for row in context["tracked_files"]}
    tracked.update({
        s73_path.resolve(): S73_SHA256,
        s74_path.resolve(): S74_SHA256,
        s45_path.resolve(): S45_SHA256,
        plan_path.resolve(): S73_PLAN_FILE_SHA256,
        pool_path.resolve(): S73_POOL_MANIFEST_FILE_SHA256,
        freeze_path.resolve(): S74_FREEZE_FILE_SHA256,
        sel_path.resolve(): sha256_file(sel_path),
        handoff_path.resolve(): sha256_file(handoff_path),
        gen_path.resolve(): generator_file_sha,
        archive_path.resolve(): sha256_file(archive_path),
    })

    protocol = context["protocol"]
    require(protocol["final_test_policy"]["current_status"] == "SEALED", "Final Test is not sealed")
    require(protocol["synthetic_generation"]["batch_size"] == 64, "Frozen batch size changed")
    require(protocol["synthetic_generation"]["diffusion_steps"] == 1000, "Frozen diffusion steps changed")
    require(protocol["synthetic_generation"]["precision"] == "FP32", "Frozen precision changed")
    require(protocol["synthetic_generation"]["sampler"] == "ancestral DDPM", "Frozen sampler changed")

    state_hash_fn = context["state_hash_fn"]
    schedule_fn = context["schedule_fn"]
    reverse_fn = context["reverse_fn"]
    scaler = context["scaler"]
    # Release B1/B2 checkpoint tensors after lineage verification; B0 is the only
    # model used during the supplementary generation.
    context["packages"].clear()
    gc.collect()

    return {
        "s73": s73,
        "pairs": pairs,
        "subjects": subjects,
        "protocol": protocol,
        "s73_plan": plan,
        "s73_plan_sha": plan_sha,
        "b0_package": package,
        "b0_model_class": b0_api.B0StateOnlyUNet1D,
        "b0_generator_file_sha": generator_file_sha,
        "b0_model_state_sha": actual_state_sha,
        "b0_selection_sha": sha256_file(sel_path),
        "b0_handoff_sha": sha256_file(handoff_path),
        "b0_archive_sha": sha256_file(archive_path),
        "reference_traces": reference_traces,
        "reference_receipt_aggregate_sha": reference_receipt_aggregate,
        "state_hash_fn": state_hash_fn,
        "schedule_fn": schedule_fn,
        "reverse_fn": reverse_fn,
        "scaler": scaler,
        "tracked": tracked,
    }


def build_b0_model(torch, model_class, package, device):
    state = package["model_state_dict"]
    require(all((not t.is_floating_point()) or t.dtype == torch.float32 for t in state.values()),
            "Frozen B0 floating weights must be FP32")
    model = model_class(package["effective_contract"])
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval().requires_grad_(False)
    return model


def sample_b0_shard(torch, model, pair, reverse, device, on_batch=None):
    """Same RNG draw order as s2_73, but B0 receives eye state only.

    The trace schema is intentionally identical to the s2_73 trace so that it
    can be compared byte-for-byte with the frozen B1/B2 receipt trace.
    """
    n = int(pair["expected_count"])
    seed = int(pair["generation_seed"])
    label = int(pair["eye_state_label"])
    subject_index = int(pair["subject_index"])
    require(device.type == "cuda", "B0 production ablation generation requires CUDA")
    require(not torch.is_autocast_enabled(), "Autocast must be disabled")
    for t in list(model.parameters()) + list(model.buffers()):
        require(t.device == device, "Model/device mismatch")
        if t.is_floating_point():
            require(t.dtype == torch.float32, "B0 model must be FP32")
    for key in ("alpha_bar", "coef_x0", "coef_xt", "posterior_variance"):
        t = reverse[key]
        require(tuple(t.shape) == (1000,) and t.dtype == torch.float32 and t.device == device,
                "Reverse schedule contract mismatch")
        require(torch.isfinite(t).all(), "Reverse schedule nonfinite")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def rng_hash():
        return hashlib.sha256(generator.get_state().cpu().numpy().tobytes(order="C")).hexdigest()

    trace = {
        "schema": "s2_73_continuous_subject_state_rng_v1",
        "seed": seed,
        "subject_index": subject_index,
        "eye_state_label": label,
        "n_samples": n,
        "batch_size": 64,
        "timesteps": 1000,
        "initial_rng_state_sha256": rng_hash(),
        "chunks": [],
    }
    out = np.empty((n, 19, 512), dtype=np.float32)
    alpha_bar = reverse["alpha_bar"]
    coef_x0 = reverse["coef_x0"]
    coef_xt = reverse["coef_xt"]
    posterior_variance = reverse["posterior_variance"]
    n_batches = len(pair["batch_sizes"])

    with torch.inference_mode():
        for bi, start in enumerate(range(0, n, 64), 1):
            count = min(64, n - start)
            shape = (count, 19, 512)
            before = rng_hash()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
            randn_calls = 1
            state = torch.full((count,), label, device=device, dtype=torch.long)
            for step in reversed(range(1000)):
                timestep = torch.full((count,), step, device=device, dtype=torch.long)
                v = model(x, timestep, state)
                require(tuple(v.shape) == shape and v.dtype == torch.float32, "Unexpected B0 prediction schema")
                require(torch.isfinite(v).all(), f"Nonfinite B0 v prediction at timestep {step}")
                sqrt_ab = torch.sqrt(alpha_bar[step])
                sqrt_omab = torch.sqrt(1.0 - alpha_bar[step])
                x0 = sqrt_ab * x - sqrt_omab * v
                mean = coef_x0[step] * x0 + coef_xt[step] * x
                if step > 0:
                    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                    randn_calls += 1
                    x = mean + torch.sqrt(posterior_variance[step]) * noise
                else:
                    x = mean
                require(torch.isfinite(x).all(), f"Nonfinite B0 reverse state at timestep {step}")
            batch = x.detach().cpu().numpy().astype(np.float32, copy=False)
            require(np.isfinite(batch).all(), "Nonfinite B0 generated batch")
            out[start:start + count] = batch
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            trace["chunks"].append({
                "batch_index": bi,
                "sample_start": start,
                "batch_size": count,
                "shape": list(shape),
                "randn_calls": randn_calls,
                "normal_values_requested": randn_calls * count * 19 * 512,
                "rng_state_before_sha256": before,
                "rng_state_after_sha256": rng_hash(),
            })
            if on_batch is not None:
                on_batch(bi, n_batches, count, seconds)
    trace["final_rng_state_sha256"] = rng_hash()
    require(out.dtype == np.float32 and np.isfinite(out).all(), "Invalid final B0 shard")
    return out, trace


def counts(pairs, receipts):
    windows = 0
    shards = 0
    for pair in pairs:
        if pair_key(pair) in receipts:
            shards += 1
            windows += pair["expected_count"]
    return {"shards": shards, "windows": windows, "percent": 100.0 * windows / TOTAL_WINDOWS}


def eta_seconds(pairs, receipts):
    seconds = 0.0
    sampled = 0
    for rec in receipts.values():
        timing = rec["metadata"].get("timing", {})
        for row in timing.get("batches", []):
            s = row.get("seconds")
            n = row.get("batch_size")
            if isinstance(s, (int, float)) and s > 0 and isinstance(n, int) and n > 0:
                seconds += float(s)
                sampled += n
    c = counts(pairs, receipts)
    remaining = TOTAL_WINDOWS - c["windows"]
    if sampled <= 0 or seconds <= 0:
        return None
    return remaining / (sampled / seconds)


def make_metadata(ctx, plan_sha, pair):
    n = pair["expected_count"]
    return {
        "schema_version": 1,
        "generator_name": "B0",
        "generator_model_state_sha256": ctx["b0_model_state_sha"],
        "generator_file_sha256": ctx["b0_generator_file_sha"],
        "subject_id": pair["subject_id"],
        "subject_index": pair["subject_index"],
        "subject_condition_used": False,
        "subject_id_role": "evaluation_stratum_only_not_model_condition",
        "eye_state": pair["eye_state"],
        "eye_state_label": pair["eye_state_label"],
        "expected_count": n,
        "generation_seed": pair["generation_seed"],
        "sample_indices": list(range(n)),
        "sampling_rate_hz": 128,
        "n_channels": 19,
        "n_times": 512,
        "channel_order": CHANNELS,
        "normalization_sha256": ctx["protocol"]["data_contract"]["normalization"]["sha256"],
        "signal_domain": "frozen_train_global_z",
        "array_key": "x_z",
        "parent_s2_73_generation_plan_sha256": ctx["s73_plan_sha"],
        "ablation_generation_plan_sha256": plan_sha,
        "batch_size": 64,
        "batch_sizes": pair["batch_sizes"],
        "diffusion_steps": 1000,
        "precision": "FP32",
        "prediction_target": "v",
        "sampler": "ancestral DDPM",
        "same_rng_stream_as_frozen_B1_B2": True,
        "Validation_waveforms_accessed": False,
        "Final_Test_accessed": False,
    }


def audit_expected_files(output, pairs):
    allowed_npz = {f"generated/B0/{p['subject_id']}/{p['eye_state']}.npz" for p in pairs}
    allowed_rec = {f"manifests/shards/B0/{p['subject_id']}/{p['eye_state']}.json" for p in pairs}
    for root, pattern, allowed in (
        (output / "generated", "*.npz", allowed_npz),
        (output / "manifests/shards", "*.json", allowed_rec),
    ):
        if root.exists():
            for path in root.rglob(pattern):
                require(path.relative_to(output).as_posix() in allowed,
                        "Unexpected file outside frozen B0 shard plan: " + str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Resumable frozen B0 1x generation for supplementary subject-conditioning ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stop-at-percent", default="2", help="Cumulative target, not an additional percentage")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--min-system-free-gib", type=float, default=1.0)
    args = parser.parse_args()
    target = target_windows(args.stop_at_percent)
    repo = args.repo_root.resolve()
    output = repo / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_report = output / "metrics" / f"s2_74c_b0_run_{run_id}.json"
    progress_path = output / "progress/s2_74c_b0_progress_v1.json"
    manifest_path = output / "manifests/s2_74c_b0_pool_manifest_v1.json"
    plan_path = output / "protocol/s2_74c_b0_generation_plan_v1.json"
    started = time.perf_counter()
    current = None
    receipts = {}
    new_windows = 0
    new_batch_seconds = 0.0
    ctx = None
    tracked_stats = {}

    def report(status, error=None, final=False):
        c = counts(ctx["pairs"], receipts) if ctx is not None else {"shards": 0, "windows": 0, "percent": 0.0}
        payload = {
            "schema": "s2_74c_b0_progress_v1",
            "status": status,
            "updated_at_utc": now_utc(),
            "target_percent": float(Decimal(str(args.stop_at_percent))),
            "target_windows_minimum": target,
            "B0": c,
            "new_windows_this_run": new_windows,
            "sampling_batch_seconds_this_run": new_batch_seconds,
            "active_wall_seconds_this_run": time.perf_counter() - started,
            "remaining_sampling_seconds_estimate": eta_seconds(ctx["pairs"], receipts) if ctx is not None else None,
            "currently_uncommitted": current,
            "scientific_role": "supplementary_subject_conditioning_ablation_only",
            "s2_75_protocol_changed": False,
            "subject_condition_used_by_B0": False,
            "Validation_waveforms_accessed": False,
            "Final_Test_waveforms_accessed": False,
            "error": error,
        }
        atomic_json(progress_path, payload)
        if final:
            atomic_json(run_report, payload)
        return payload

    def save_manifest(plan_sha):
        c = counts(ctx["pairs"], receipts)
        payload = {
            "schema": "s2_74c_b0_pool_manifest_v1",
            "status": "S2_74C_B0_FULL_POOL_COMPLETE" if c["shards"] == EXPECTED_PAIRS else "S2_74C_B0_POOL_PARTIAL",
            "generation_plan_sha256": plan_sha,
            "B0": c,
            "committed": {
                key: {"file": rec["file"], "sha256": rec["sha256"], "expected_count": rec["metadata"]["expected_count"]}
                for key, rec in sorted(receipts.items())
            },
            "subject_condition_used": False,
            "subject_id_role": "evaluation_stratum_only_not_model_condition",
            "same_subject_state_counts_as_B1_B2": True,
            "same_generation_seeds_and_rng_streams_as_B1_B2": True,
            "Validation_waveforms_accessed": False,
            "Final_Test_waveforms_accessed": False,
            "updated_at_utc": now_utc(),
        }
        atomic_json(manifest_path, payload)

    def check_tracked(full=False):
        for path, digest in ctx["tracked"].items():
            stat_now = path.stat()
            current_stat = (stat_now.st_size, stat_now.st_mtime_ns)
            require(current_stat == tracked_stats[str(path)], "Frozen input changed during run: " + str(path))
            if full:
                require(sha256_file(path) == digest, "Frozen input hash changed during run: " + str(path))

    with output_lock(output):
        try:
            print("\n" + "=" * 80)
            print(f"S2_74C — B0 SUBJECT-CONDITIONING ABLATION GENERATION — cumulative target {args.stop_at_percent}%")
            print("=" * 80)
            print(f"Target: at least {target:,}/{TOTAL_WINDOWS:,} B0 windows, then finish current shard.")
            print("B0 receives EYE STATE ONLY. subject_id is evaluation-stratum metadata, never a model input.")
            print("Same counts/seeds/random streams as frozen B1/B2. Final Test remains sealed.")
            print("Output:", output)

            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable. Use the existing project .venv; no CPU generation fallback.")
            device = torch.device(args.device)
            if device.type != "cuda":
                raise ValueError("Generation requires CUDA")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
            random.seed(20262201)
            np.random.seed(20262201)
            torch.manual_seed(20262201)
            torch.cuda.manual_seed_all(20262201)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.set_default_dtype(torch.float32)

            # Import s2_73 first so we can reuse its disk/resource helpers too.
            require((repo / S73_REL).is_file(), "Missing s2_73 source")
            s73_for_disk = import_exact(repo / S73_REL, S73_SHA256, "_s2_73_b0_disk_helpers")
            s73_for_disk.p73_disk_check(output, args.min_free_gib, args.min_system_free_gib)
            runtime = s73_for_disk.p73_runtime(torch, device)
            print("GPU:", runtime["gpu_name"], f"({runtime['gpu_memory_bytes']/2**30:.2f} GiB)")

            # prepare() imports the same exact s2_73 under another isolated name;
            # this is intentional and side-effect free.
            ctx = prepare(repo, torch)
            ctx["tracked"][Path(__file__).resolve()] = sha256_file(Path(__file__).resolve())
            for path in ctx["tracked"]:
                st = path.stat()
                tracked_stats[str(path)] = (st.st_size, st.st_mtime_ns)

            plan = {
                "schema": VERSION,
                "implementation_sha256": sha256_file(Path(__file__).resolve()),
                "runtime": runtime,
                "scientific_role": "supplementary_subject_conditioning_ablation_only",
                "s2_75_protocol_changed": False,
                "model": {
                    "name": "B0",
                    "frozen_identity": "b0_v_raw_epoch020_v1",
                    "conditioning": {"eye_state": True, "subject": False, "EO": 0, "EC": 1},
                    "subject_id_role": "evaluation_stratum_only_not_model_condition",
                    "generator_file_sha256": ctx["b0_generator_file_sha"],
                    "model_state_sha256": ctx["b0_model_state_sha"],
                    "selection_manifest_sha256": ctx["b0_selection_sha"],
                    "handoff_sha256": ctx["b0_handoff_sha"],
                    "archive_checkpoint_sha256": ctx["b0_archive_sha"],
                },
                "parent_frozen_B1_B2_generation": {
                    "s2_73_plan_file_sha256": S73_PLAN_FILE_SHA256,
                    "s2_73_plan_canonical_sha256": ctx["s73_plan_sha"],
                    "s2_73_pool_manifest_sha256": S73_POOL_MANIFEST_FILE_SHA256,
                    "s2_74_freeze_manifest_sha256": S74_FREEZE_FILE_SHA256,
                    "B1_B2_reference_receipts_aggregate_sha256": ctx["reference_receipt_aggregate_sha"],
                },
                "sampling": {
                    "sampler": "ancestral DDPM",
                    "prediction_target": "v",
                    "diffusion_steps": 1000,
                    "precision": "FP32",
                    "batch_size": 64,
                    "generation_master_seed": 20262201,
                    "generation_seed_formula": "20262201 + 2 * subject_index + eye_state_label",
                    "same_rng_stream_as_B1_B2": True,
                },
                "signal": {
                    "array_key": "x_z",
                    "domain": "frozen_train_global_z",
                    "channels": 19,
                    "samples": 512,
                    "sampling_rate_hz": 128,
                    "channel_order": CHANNELS,
                    "normalization_sha256": ctx["protocol"]["data_contract"]["normalization"]["sha256"],
                },
                "expected_windows": TOTAL_WINDOWS,
                "expected_shards": EXPECTED_PAIRS,
                "ordering": "subject_index ascending; EO then EC",
                "pairs": ctx["pairs"],
                "Validation_waveforms_accessed": False,
                "Final_Test_waveforms_accessed": False,
            }
            plan_sha = canonical_sha(plan)
            if plan_path.exists():
                require(canonical_bytes(load_json(plan_path)) == canonical_bytes(plan),
                        "Existing B0 generation plan differs from current runtime/source. Do not overwrite it.")
            else:
                existing_npz = list((output / "generated").rglob("*.npz")) if (output / "generated").exists() else []
                existing_rec = list((output / "manifests/shards").rglob("*.json")) if (output / "manifests/shards").exists() else []
                require(not existing_npz and not existing_rec and not manifest_path.exists(),
                        "B0 generation artifacts exist without their frozen plan; refusing adoption")
                atomic_json(plan_path, plan)
            print("Generation plan frozen:", plan_path)
            print("Generation plan canonical SHA256:", plan_sha)

            store = B0Store(output)
            audit_expected_files(output, ctx["pairs"])
            print("Validating all previously committed B0 shards before reuse...", flush=True)
            previous_manifest = load_json(manifest_path) if manifest_path.exists() else None
            for i, pair in enumerate(ctx["pairs"], 1):
                expected = make_metadata(ctx, plan_sha, pair)
                rec = store.verify(pair["subject_id"], pair["eye_state"], expected)
                if rec is not None:
                    # B0 trace must still equal the already-frozen B1/B2 trace.
                    ref = ctx["reference_traces"][pair["subject_id"] + "/" + pair["eye_state"]]
                    require(canonical_bytes(rec["metadata"].get("rng_trace")) == canonical_bytes(ref),
                            "Committed B0 random stream differs from frozen B1/B2")
                    ctx["s73"].p73_check_trace(rec["metadata"], pair)
                    receipts[pair_key(pair)] = rec
                if i % 50 == 0:
                    print(f"  resume audit: {i}/{EXPECTED_PAIRS} subject-state strata", flush=True)
            if previous_manifest is not None:
                require(previous_manifest.get("generation_plan_sha256") == plan_sha, "Existing B0 pool manifest plan mismatch")
                for key, row in previous_manifest.get("committed", {}).items():
                    require(key in receipts, "Previously committed B0 shard disappeared: " + key)
                    require(receipts[key]["file"] == row["file"] and receipts[key]["sha256"] == row["sha256"],
                            "Previously committed B0 shard changed: " + key)
            save_manifest(plan_sha)
            c0 = counts(ctx["pairs"], receipts)
            print(f"Verified start: B0={c0['windows']:,}/{TOTAL_WINDOWS:,} ({c0['percent']:.4f}%), shards={c0['shards']}/396")
            report("S2_74C_B0_PREFLIGHT_PASSED")
            if c0["windows"] >= target:
                check_tracked(full=True)
                status = "S2_74C_B0_FULL_POOL_COMPLETE" if c0["shards"] == EXPECTED_PAIRS else "S2_74C_B0_TARGET_ALREADY_SATISFIED_PARTIAL"
                payload = report(status, final=True)
                print("\n" + status + " — no new samples generated.")
                print("RUN REPORT:", run_report)
                print("LATEST PROGRESS:", progress_path)
                return 0

            print("Loading frozen B0 RAW20 model...", flush=True)
            model = build_b0_model(torch, ctx["b0_model_class"], ctx["b0_package"], device)
            require(ctx["state_hash_fn"](model.state_dict()) == ctx["b0_model_state_sha"], "B0 model changed after load")
            schedule_cpu = ctx["schedule_fn"](timesteps=1000, cosine_s=0.008)
            reverse = ctx["reverse_fn"](schedule_cpu, device)
            print("Generation starts. Only committed B0 shards count toward progress.", flush=True)

            for pair in ctx["pairs"]:
                if pair_key(pair) in receipts:
                    continue
                check_tracked()
                n = pair["expected_count"]
                s73_for_disk.p73_disk_check(
                    output, args.min_free_gib, args.min_system_free_gib,
                    extra_bytes=n * 19 * 512 * 4 * 2,
                )
                current = {
                    "subject": pair["subject_id"], "state": pair["eye_state"],
                    "expected_count": n, "generated_but_uncommitted": 0,
                }
                batch_rows = []
                print(f"\n[B0] {pair['subject_id']} {pair['eye_state']} n={n} seed={pair['generation_seed']} ", flush=True)
                report("S2_74C_B0_GENERATING")

                def on_batch(index, total, count, seconds):
                    nonlocal new_batch_seconds
                    batch_rows.append({"batch_index": index, "batch_size": count, "seconds": seconds})
                    new_batch_seconds += seconds
                    current["generated_but_uncommitted"] += count
                    print(f"  batch {index}/{total}: {count} windows in {seconds:.2f}s "
                          f"({count/seconds:.3f} windows/s); shard still uncommitted", flush=True)
                    report("S2_74C_B0_GENERATING")

                shard_started = time.perf_counter()
                values, trace = sample_b0_shard(torch, model, pair, reverse, device, on_batch=on_batch)
                reference = ctx["reference_traces"][pair["subject_id"] + "/" + pair["eye_state"]]
                require(canonical_bytes(trace) == canonical_bytes(reference),
                        "B0 random stream does not exactly match frozen B1/B2 before commit")
                require(ctx["state_hash_fn"](model.state_dict()) == ctx["b0_model_state_sha"],
                        "B0 model state changed during inference")
                check_tracked()
                meta = make_metadata(ctx, plan_sha, pair)
                meta.update({
                    "created_at_utc": now_utc(),
                    "rng_trace": trace,
                    "timing": {
                        "batches": batch_rows,
                        "sampling_seconds": sum(r["seconds"] for r in batch_rows),
                        "generation_call_wall_seconds": time.perf_counter() - shard_started,
                    },
                })
                rec = store.commit(pair["subject_id"], pair["eye_state"], values, meta)
                receipts[pair_key(pair)] = rec
                new_windows += n
                current = None
                save_manifest(plan_sha)
                c = counts(ctx["pairs"], receipts)
                resources = s73_for_disk.p73_resources(output, torch, device)
                report("S2_74C_B0_GENERATING")
                print(f"  COMMITTED {rec['file']} SHA256={rec['sha256']}")
                print(f"  B0={c['windows']:,}/{TOTAL_WINDOWS:,}; progress={c['percent']:.4f}%; "
                      f"remaining sampling~{duration(eta_seconds(ctx['pairs'], receipts))}; "
                      f"disk free={resources['output_free_gib']:.2f} GiB", flush=True)
                if c["windows"] >= target:
                    break

            check_tracked(full=True)
            # Full replay of committed metadata/NPZ integrity before reporting status.
            for pair in ctx["pairs"]:
                if pair_key(pair) not in receipts:
                    continue
                expected = make_metadata(ctx, plan_sha, pair)
                rec = store.verify(pair["subject_id"], pair["eye_state"], expected)
                require(rec is not None, "Committed B0 shard disappeared during final audit")
                ref = ctx["reference_traces"][pair["subject_id"] + "/" + pair["eye_state"]]
                require(canonical_bytes(rec["metadata"]["rng_trace"]) == canonical_bytes(ref),
                        "B0/B1/B2 common RNG verification failed in final replay")
                ctx["s73"].p73_check_trace(rec["metadata"], pair)

            c = counts(ctx["pairs"], receipts)
            status = "S2_74C_B0_FULL_POOL_COMPLETE" if c["shards"] == EXPECTED_PAIRS else "S2_74C_B0_TARGET_REACHED_PARTIAL"
            save_manifest(plan_sha)
            payload = report(status, final=True)
            print("\n" + status)
            print(f"B0: {c['windows']:,}/{TOTAL_WINDOWS:,} windows; {c['shards']}/396 shards")
            print(f"Progress: {c['percent']:.4f}%")
            print("Active time this run:", duration(payload["active_wall_seconds_this_run"]))
            print("Observed sampling time:", duration(payload["sampling_batch_seconds_this_run"]))
            print("Estimated remaining sampling time:", duration(payload["remaining_sampling_seconds_estimate"]))
            print("Subject conditioning used by B0: NO")
            print("Validation waveform accessed: NO")
            print("Final Test accessed: NO")
            print("RUN REPORT:", run_report)
            print("LATEST PROGRESS:", progress_path)
            print("POOL MANIFEST:", manifest_path)
            print("All committed B0 shards are retained for the next cumulative target.", flush=True)
            return 0

        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                status, code = "S2_74C_B0_INTERRUPTED_RESUMABLE", 130
            elif isinstance(exc, SafeStop):
                status, code = "S2_74C_B0_STOPPED_SAFE", 2
            else:
                status, code = "S2_74C_B0_STOPPED_ERROR", 1
            error = {"type": type(exc).__name__, "message": str(exc)}
            try:
                if ctx is not None:
                    report(status, error=error, final=True)
            except Exception as report_exc:
                print("Could not write failure report:", report_exc, file=sys.stderr)
            print("\n" + status, file=sys.stderr)
            if not isinstance(exc, KeyboardInterrupt):
                traceback.print_exc()
            else:
                print("Interrupted by user; committed shards remain authoritative.", file=sys.stderr)
            print("Re-run the same command to resume. Never edit the Python file between resumptions.", file=sys.stderr)
            print("RUN REPORT:", run_report, file=sys.stderr)
            return code


if __name__ == "__main__":
    raise SystemExit(main())
