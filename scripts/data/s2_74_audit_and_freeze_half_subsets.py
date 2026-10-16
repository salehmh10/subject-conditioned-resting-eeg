from __future__ import annotations

"""
s2_74 — full synthetic-pool integrity audit + deterministic nested 0.5x freeze.

Purpose
-------
1) Read-only verification of the completed s2_73 B1/B2 1x pools.
2) Freeze the predeclared nested 0.5x subset indices. No EEG is generated.
3) Never read Validation or Final-Test waveforms.
4) Never tune, select, or modify B1/B2.

Save under:
    scripts/data/s2_74_audit_and_freeze_half_subsets.py

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -u .\\scripts\\data\\s2_74_audit_and_freeze_half_subsets.py

The script expects the s2_73 output root:
    outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1

The 0.5x rule is the frozen s2_72 rule:
- for every subject x eye-state stratum with n synthetic samples,
  retain floor(n/2) indices from a deterministic permutation;
- use exactly the same retained pool indices for B1 and B2;
- master seed = 20262251;
- no new diffusion generation.

Implementation detail frozen by this stage:
- one MT19937 RandomState stream seeded once with 20262251;
- strata are processed in the exact s2_73 generation-plan order;
- for each stratum, permutation(n) is drawn, first floor(n/2) are retained,
  and retained indices are stored sorted for efficient downstream loading.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STAGE = "s2_74"
VERSION = "s2_74_pool_integrity_half_subset_v1"
EXPECTED_PROTOCOL_SHA256 = "5b98ff09d977006b96ea80e43d4f13f33742745b734d5dfa94bb1eaacb682a22"
EXPECTED_WINDOWS_PER_GENERATOR = 47_835
EXPECTED_PAIRS = 396
EXPECTED_SHARDS_PER_GENERATOR = 396
EXPECTED_SHARDS_TOTAL = 792
HALF_MASTER_SEED = 20262251
CHANNELS = 19
TIMES = 512


class IntegrityError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_nonfinite_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(
            handle.read(),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_write(path: Path, payload) -> None:
    encoded = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def stable_write(path: Path, payload) -> None:
    """Frozen artifact: create once, or verify byte-equivalent payload on rerun."""
    if path.exists():
        existing = load_json(path)
        if canonical_json_bytes(existing) != canonical_json_bytes(payload):
            raise IntegrityError(f"Refusing to overwrite changed frozen artifact: {path}")
        return
    atomic_json_write(path, payload)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def resolve_repo(explicit: str | None) -> Path:
    if explicit:
        repo = Path(explicit).expanduser().resolve()
    else:
        # Expected location is repo/scripts/data/<this file>.
        repo = Path(__file__).resolve().parents[2]
    require((repo / "scripts" / "data").is_dir(), f"Not a Scenario-2 repo root: {repo}")
    return repo


def protocol_path(repo: Path) -> Path:
    return (
        repo
        / "outputs"
        / "protocols"
        / "scenario2"
        / "utility"
        / "b1_vs_b2_utility_protocol_v1"
        / "metrics"
        / "s2_72_b1_vs_b2_downstream_utility_protocol_v1.json"
    )


def generation_root(repo: Path) -> Path:
    return (
        repo
        / "outputs"
        / "runs"
        / "scenario2"
        / "utility"
        / "bulk_synthetic_b1_b2_1x_v1"
    )


def freeze_root(repo: Path) -> Path:
    return (
        repo
        / "outputs"
        / "frozen"
        / "scenario2"
        / "utility"
        / "b1_b2_pool_integrity_half_subset_v1"
    )


def expected_pair_shape(pair: dict) -> None:
    required = {
        "subject_id",
        "subject_index",
        "eye_state",
        "eye_state_label",
        "expected_count",
        "generation_seed",
        "batch_sizes",
    }
    require(required.issubset(pair), f"Generation-plan pair missing fields: {required - set(pair)}")
    require(re.fullmatch(r"sub-[0-9]+", str(pair["subject_id"])) is not None, "Invalid subject_id")
    require(type(pair["subject_index"]) is int and 0 <= pair["subject_index"] < 198, "Invalid subject_index")
    require(pair["eye_state"] in ("EO", "EC"), "Invalid eye_state")
    expected_label = 0 if pair["eye_state"] == "EO" else 1
    require(pair["eye_state_label"] == expected_label, "eye_state label mismatch")
    n = pair["expected_count"]
    require(type(n) is int and n > 0, "Invalid expected_count")
    expected_seed = 20262201 + 2 * pair["subject_index"] + expected_label
    require(pair["generation_seed"] == expected_seed, "Generation seed formula mismatch")
    expected_batches = [min(64, n - start) for start in range(0, n, 64)]
    require(pair["batch_sizes"] == expected_batches, "Batch-size schedule mismatch")


def verify_plan_and_protocol(repo: Path, gen_root: Path):
    p_path = protocol_path(repo)
    require(p_path.is_file(), f"Missing frozen s2_72 protocol: {p_path}")
    protocol_sha = sha256_file(p_path)
    require(protocol_sha == EXPECTED_PROTOCOL_SHA256, "Frozen s2_72 protocol file SHA256 mismatch")
    protocol = load_json(p_path)
    require(protocol.get("status") == "S2_72_B1_VS_B2_UTILITY_PROTOCOL_FROZEN", "s2_72 protocol is not frozen")

    sg = protocol.get("synthetic_generation", {})
    require(sg.get("expected_windows_per_generator") == EXPECTED_WINDOWS_PER_GENERATOR, "Protocol pool size changed")
    require(sg.get("half_ratio") == 0.5, "Protocol half ratio changed")
    require(sg.get("half_subset_master_seed") == HALF_MASTER_SEED, "Protocol half-subset seed changed")
    require(sg.get("common_random_numbers_across_B1_B2") is True, "Protocol common-RNG rule changed")
    require(sg.get("2x_pool_in_this_protocol") is False, "Unexpected 2x protocol")

    plan_path = gen_root / "protocol" / "s2_73_generation_plan_v1.json"
    require(plan_path.is_file(), f"Missing s2_73 generation plan: {plan_path}")
    plan = load_json(plan_path)
    plan_sha = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    require(plan.get("core", {}).get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "Generation plan protocol SHA mismatch")
    require(plan.get("array_key") == "x_z", "Unexpected synthetic array key")
    require(plan.get("signal_domain") == "frozen_train_global_z", "Unexpected synthetic signal domain")

    pairs = plan.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == EXPECTED_PAIRS, "Expected 396 subject-state pairs")
    for pair in pairs:
        expected_pair_shape(pair)

    # Exact pair ordering from s2_73: subject index ascending; EO then EC.
    observed = [(p["subject_index"], p["eye_state"]) for p in pairs]
    expected = [(s, st) for s in range(198) for st in ("EO", "EC")]
    require(observed == expected, "s2_73 pair ordering changed")
    require(sum(p["expected_count"] for p in pairs) == EXPECTED_WINDOWS_PER_GENERATOR, "Pair counts do not sum to 47,835")

    return protocol, p_path, protocol_sha, plan, plan_path, plan_sha, pairs


def validate_rng_trace(trace: dict, pair: dict) -> None:
    require(isinstance(trace, dict), "Missing rng_trace")
    expected = {
        "schema": "s2_73_continuous_subject_state_rng_v1",
        "seed": pair["generation_seed"],
        "subject_index": pair["subject_index"],
        "eye_state_label": pair["eye_state_label"],
        "n_samples": pair["expected_count"],
        "batch_size": 64,
        "timesteps": 1000,
    }
    for key, value in expected.items():
        require(trace.get(key) == value, f"RNG trace mismatch: {key}")
    chunks = trace.get("chunks")
    require(isinstance(chunks, list) and len(chunks) == len(pair["batch_sizes"]), "RNG chunk count mismatch")
    previous = trace.get("initial_rng_state_sha256")
    require(isinstance(previous, str) and re.fullmatch(r"[0-9a-f]{64}", previous), "Invalid initial RNG digest")
    start = 0
    for index, (chunk, count) in enumerate(zip(chunks, pair["batch_sizes"]), 1):
        require(isinstance(chunk, dict), "Invalid RNG chunk")
        checks = {
            "batch_index": index,
            "sample_start": start,
            "batch_size": count,
            "shape": [count, CHANNELS, TIMES],
            "randn_calls": 1000,
            "normal_values_requested": 1000 * count * CHANNELS * TIMES,
            "rng_state_before_sha256": previous,
        }
        for key, value in checks.items():
            require(chunk.get(key) == value, f"RNG chunk mismatch: {key}")
        after = chunk.get("rng_state_after_sha256")
        require(isinstance(after, str) and re.fullmatch(r"[0-9a-f]{64}", after), "Invalid RNG after-state digest")
        previous = after
        start += count
    require(trace.get("final_rng_state_sha256") == previous, "Broken RNG stream at shard end")


def validate_receipt_and_npz(gen_root: Path, generator: str, pair: dict):
    subject = pair["subject_id"]
    state = pair["eye_state"]
    n = pair["expected_count"]
    rel_data = Path("generated") / generator / subject / f"{state}.npz"
    rel_receipt = Path("manifests") / "shards" / generator / subject / f"{state}.json"
    data_path = gen_root / rel_data
    receipt_path = gen_root / rel_receipt
    require(data_path.is_file(), f"Missing shard: {rel_data.as_posix()}")
    require(receipt_path.is_file(), f"Missing receipt: {rel_receipt.as_posix()}")
    require(not data_path.is_symlink() and not receipt_path.is_symlink(), "Synthetic artifacts must not be symlinks")

    receipt = load_json(receipt_path)
    require(set(receipt) == {"schema_version", "file", "sha256", "metadata"}, f"Receipt schema mismatch: {rel_receipt}")
    require(receipt["schema_version"] == 1, "Receipt schema version mismatch")
    require(receipt["file"] == rel_data.as_posix(), "Receipt file path mismatch")
    require(re.fullmatch(r"[0-9a-f]{64}", str(receipt["sha256"])) is not None, "Invalid shard SHA256")
    observed_sha = sha256_file(data_path)
    require(observed_sha == receipt["sha256"], f"Shard SHA256 mismatch: {rel_data}")

    meta = receipt["metadata"]
    require(isinstance(meta, dict), "Receipt metadata must be an object")
    expected_meta = {
        "schema_version": 1,
        "generator_name": generator,
        "subject_id": subject,
        "subject_index": pair["subject_index"],
        "eye_state": state,
        "eye_state_label": pair["eye_state_label"],
        "expected_count": n,
        "generation_seed": pair["generation_seed"],
        "sample_indices": list(range(n)),
        "sampling_rate_hz": 128,
        "n_channels": CHANNELS,
        "n_times": TIMES,
        "signal_domain": "frozen_train_global_z",
        "array_key": "x_z",
        "batch_size": 64,
        "batch_sizes": pair["batch_sizes"],
        "diffusion_steps": 1000,
        "precision": "FP32",
        "prediction_target": "v",
        "sampler": "ancestral DDPM",
        "Final_Test_accessed": False,
    }
    for key, value in expected_meta.items():
        require(meta.get(key) == value, f"Metadata mismatch {generator}/{subject}/{state}: {key}")
    validate_rng_trace(meta.get("rng_trace"), pair)

    # Full data audit: decompress each committed shard, verify schema, dtype, shape,
    # finite values, exact sample indices, and embedded canonical metadata.
    try:
        with np.load(data_path, allow_pickle=False) as archive:
            require(set(archive.files) == {"x_z", "sample_indices", "metadata_json"}, f"NPZ members mismatch: {rel_data}")
            x = archive["x_z"]
            require(x.dtype == np.float32, f"x_z dtype mismatch: {rel_data}")
            require(x.shape == (n, CHANNELS, TIMES), f"x_z shape mismatch: {rel_data}")
            require(np.isfinite(x).all(), f"Non-finite x_z found: {rel_data}")
            indices = archive["sample_indices"]
            require(indices.dtype == np.int64 and indices.shape == (n,), f"sample_indices schema mismatch: {rel_data}")
            require(np.array_equal(indices, np.arange(n, dtype=np.int64)), f"sample_indices values mismatch: {rel_data}")
            embedded = archive["metadata_json"]
            require(embedded.shape == () and embedded.dtype.kind == "U", f"metadata_json schema mismatch: {rel_data}")
            require(embedded.item() == canonical_json_bytes(meta).decode("utf-8"), f"Embedded metadata mismatch: {rel_data}")
            # Read-only descriptive integrity quantities; NOT a scientific fidelity analysis.
            min_value = float(np.min(x))
            max_value = float(np.max(x))
            mean_value = float(np.mean(x, dtype=np.float64))
            std_value = float(np.std(x, dtype=np.float64))
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError(f"Cannot validate NPZ {rel_data}: {exc}") from exc

    return {
        "receipt": receipt,
        "file": rel_data.as_posix(),
        "receipt_file": rel_receipt.as_posix(),
        "sha256": observed_sha,
        "count": n,
        "integrity_summary_z": {
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
        },
    }


def verify_pool_manifest(gen_root: Path, plan_sha: str, pairs: list[dict]):
    path = gen_root / "manifests" / "s2_73_pool_manifest_v1.json"
    require(path.is_file(), f"Missing s2_73 pool manifest: {path}")
    manifest = load_json(path)
    require(manifest.get("schema") == "s2_73_pool_manifest_v1", "Pool manifest schema mismatch")
    require(manifest.get("generation_plan_sha256") == plan_sha, "Pool manifest plan SHA mismatch")
    require(manifest.get("status") == "S2_73_FULL_POOL_COMPLETE", "s2_73 pool is not marked FULL_POOL_COMPLETE")
    require(manifest.get("Final_Test_accessed") is False, "s2_73 reports Final Test access")
    progress = manifest.get("progress", {})
    require(progress.get("committed_shards_total") == EXPECTED_SHARDS_TOTAL, "Pool manifest does not contain 792 committed shards")
    require(progress.get("committed_windows_total") == 2 * EXPECTED_WINDOWS_PER_GENERATOR, "Pool manifest window total mismatch")
    generators = progress.get("generators", {})
    for name in ("B1", "B2"):
        require(generators.get(name, {}).get("shards") == EXPECTED_SHARDS_PER_GENERATOR, f"{name} shard count mismatch")
        require(generators.get(name, {}).get("windows") == EXPECTED_WINDOWS_PER_GENERATOR, f"{name} window count mismatch")
    committed = manifest.get("committed")
    require(isinstance(committed, dict) and len(committed) == EXPECTED_SHARDS_TOTAL, "Pool committed table mismatch")
    expected_keys = {
        f"{g}/{p['subject_id']}/{p['eye_state']}" for p in pairs for g in ("B1", "B2")
    }
    require(set(committed) == expected_keys, "Pool manifest committed-key set mismatch")
    return path, manifest


def check_no_unexpected_committed_files(gen_root: Path, pairs: list[dict]) -> None:
    expected_npz = {
        (Path("generated") / g / p["subject_id"] / f"{p['eye_state']}.npz").as_posix()
        for p in pairs for g in ("B1", "B2")
    }
    observed_npz = {
        path.relative_to(gen_root).as_posix()
        for path in (gen_root / "generated").rglob("*.npz")
    }
    require(observed_npz == expected_npz, "Unexpected/missing NPZ files relative to frozen plan")

    expected_receipts = {
        (Path("manifests") / "shards" / g / p["subject_id"] / f"{p['eye_state']}.json").as_posix()
        for p in pairs for g in ("B1", "B2")
    }
    observed_receipts = {
        path.relative_to(gen_root).as_posix()
        for path in (gen_root / "manifests" / "shards").rglob("*.json")
    }
    require(observed_receipts == expected_receipts, "Unexpected/missing shard receipts relative to frozen plan")


def build_half_subset(pairs: list[dict]) -> tuple[list[dict], int]:
    rng = np.random.RandomState(HALF_MASTER_SEED)
    rows = []
    total = 0
    for pair in pairs:
        n = pair["expected_count"]
        k = n // 2
        # RandomState/MT19937 is intentional and is frozen in this stage artifact.
        permutation = rng.permutation(n)
        retained = np.sort(permutation[:k]).astype(np.int64)
        retained_list = [int(v) for v in retained]
        require(len(retained_list) == k and len(set(retained_list)) == k, "Half-subset duplicate index")
        require(all(0 <= v < n for v in retained_list), "Half-subset index out of range")
        rows.append({
            "subject_id": pair["subject_id"],
            "subject_index": pair["subject_index"],
            "eye_state": pair["eye_state"],
            "eye_state_label": pair["eye_state_label"],
            "full_count": n,
            "retained_count": k,
            "retained_pool_indices": retained_list,
        })
        total += k
    return rows, total


def build_generator_view(generator: str, rows: list[dict], verified_sha: dict[str, str], master_sha: str):
    strata = []
    for row in rows:
        rel = Path("generated") / generator / row["subject_id"] / f"{row['eye_state']}.npz"
        strata.append({
            "subject_id": row["subject_id"],
            "subject_index": row["subject_index"],
            "eye_state": row["eye_state"],
            "eye_state_label": row["eye_state_label"],
            "source_shard": rel.as_posix(),
            "source_shard_sha256": verified_sha[rel.as_posix()],
            "full_count": row["full_count"],
            "retained_count": row["retained_count"],
            "retained_pool_indices": row["retained_pool_indices"],
        })
    return {
        "schema": "s2_74_generator_0p5x_view_v1",
        "stage": STAGE,
        "generator": generator,
        "source_pool": "s2_73 frozen 1x pool",
        "master_subset_manifest_sha256": master_sha,
        "array_key": "x_z",
        "signal_domain": "frozen_train_global_z",
        "ratio_label": "0.5x_nested",
        "total_retained_windows": sum(x["retained_count"] for x in strata),
        "strata": strata,
        "Final_Test_accessed": False,
        "Validation_waveforms_accessed": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description="s2_74: audit completed s2_73 pool and freeze deterministic nested 0.5x manifests")
    p.add_argument("--repo-root", default=None, help="Scenario-2 repository root; normally auto-detected")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    repo = resolve_repo(args.repo_root)
    gen_root = generation_root(repo)
    out_root = freeze_root(repo)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("\n" + "=" * 80)
    print("S2_74 — POOL INTEGRITY AUDIT + FREEZE NESTED 0.5x SUBSETS")
    print("=" * 80)
    print("Repository:", repo)
    print("s2_73 pool:", gen_root)
    print("s2_74 frozen output:", out_root)
    print("Final Test waveform access: FORBIDDEN / NOT NEEDED")
    print("Validation waveform access: FORBIDDEN / NOT NEEDED")

    try:
        require(gen_root.is_dir(), f"Missing s2_73 output root: {gen_root}")
        protocol, p_path, protocol_sha, plan, plan_path, plan_sha, pairs = verify_plan_and_protocol(repo, gen_root)
        pool_manifest_path, pool_manifest = verify_pool_manifest(gen_root, plan_sha, pairs)
        check_no_unexpected_committed_files(gen_root, pairs)

        print("[1/4] Frozen protocol, generation plan, and FULL_POOL_COMPLETE manifest: PASS")
        print(f"      protocol SHA256: {protocol_sha}")
        print(f"      generation plan SHA256: {plan_sha}")

        audit_rows = []
        verified_sha = {}
        by_generator = {"B1": {"shards": 0, "windows": 0}, "B2": {"shards": 0, "windows": 0}}
        print("[2/4] Full read/hash/schema/finite audit of all 792 committed shards...")
        for pair_index, pair in enumerate(pairs, 1):
            pair_results = {}
            for generator in ("B1", "B2"):
                result = validate_receipt_and_npz(gen_root, generator, pair)
                pair_results[generator] = result
                by_generator[generator]["shards"] += 1
                by_generator[generator]["windows"] += result["count"]
                verified_sha[result["file"]] = result["sha256"]
                audit_rows.append({
                    "generator": generator,
                    "subject_id": pair["subject_id"],
                    "subject_index": pair["subject_index"],
                    "eye_state": pair["eye_state"],
                    "expected_count": result["count"],
                    "file": result["file"],
                    "sha256": result["sha256"],
                    "integrity_summary_z": result["integrity_summary_z"],
                })

            # Strong paired check: identical random stream and immutable matched count.
            m1 = pair_results["B1"]["receipt"]["metadata"]
            m2 = pair_results["B2"]["receipt"]["metadata"]
            require(canonical_json_bytes(m1["rng_trace"]) == canonical_json_bytes(m2["rng_trace"]),
                    f"B1/B2 common RNG stream mismatch: {pair['subject_id']} {pair['eye_state']}")
            for field in ("subject_id", "subject_index", "eye_state", "eye_state_label", "expected_count",
                          "generation_seed", "sample_indices", "batch_size", "batch_sizes", "diffusion_steps",
                          "precision", "prediction_target", "sampler", "sampling_rate_hz", "n_channels", "n_times",
                          "channel_order", "normalization_sha256", "signal_domain", "array_key", "protocol_sha256",
                          "generation_plan_sha256"):
                require(canonical_json_bytes(m1.get(field)) == canonical_json_bytes(m2.get(field)),
                        f"B1/B2 paired metadata mismatch {field}: {pair['subject_id']} {pair['eye_state']}")

            if pair_index % 25 == 0 or pair_index == len(pairs):
                print(f"      audited {pair_index:3d}/{len(pairs)} subject-state pairs ({pair_index*2:3d}/792 shards)", flush=True)

        for generator in ("B1", "B2"):
            require(by_generator[generator]["shards"] == EXPECTED_SHARDS_PER_GENERATOR, f"{generator} final shard count mismatch")
            require(by_generator[generator]["windows"] == EXPECTED_WINDOWS_PER_GENERATOR, f"{generator} final window count mismatch")
        print("      PASS: B1=396 shards/47,835 windows; B2=396 shards/47,835 windows")
        print("      PASS: all arrays finite float32 (n,19,512), all file hashes valid, common RNG verified")

        print("[3/4] Build deterministic nested 0.5x retained-index manifest...")
        half_rows, half_total = build_half_subset(pairs)
        master = {
            "schema": "s2_74_nested_0p5x_master_manifest_v1",
            "stage": STAGE,
            "source_protocol_sha256": protocol_sha,
            "source_generation_plan_sha256": plan_sha,
            "source_pool_manifest_sha256": sha256_file(pool_manifest_path),
            "source_pool_root_relative_to_repo": gen_root.relative_to(repo).as_posix(),
            "ratio": 0.5,
            "selection_rule": "For each subject-state stratum with n samples, retain floor(n/2) indices from a deterministic permutation; identical retained indices are used for B1 and B2.",
            "implementation": {
                "master_seed": HALF_MASTER_SEED,
                "rng": "numpy.random.RandomState(MT19937)",
                "rng_seeded_once": True,
                "stratum_order": "exact s2_73 generation-plan pair order: subject_index ascending, EO then EC",
                "draw": "rng.permutation(n); retain first floor(n/2); store retained set sorted ascending",
            },
            "full_windows_per_generator": EXPECTED_WINDOWS_PER_GENERATOR,
            "retained_windows_per_generator": half_total,
            "strata_count": len(half_rows),
            "same_retained_pool_indices_for_B1_and_B2": True,
            "strata": half_rows,
            "Final_Test_accessed": False,
            "Validation_waveforms_accessed": False,
        }
        master_path = out_root / "manifests" / "s2_74_nested_0p5x_master_manifest_v1.json"
        stable_write(master_path, master)
        master_sha = sha256_file(master_path)

        b1_view = build_generator_view("B1", half_rows, verified_sha, master_sha)
        b2_view = build_generator_view("B2", half_rows, verified_sha, master_sha)
        b1_path = out_root / "manifests" / "s2_74_B1_nested_0p5x_view_v1.json"
        b2_path = out_root / "manifests" / "s2_74_B2_nested_0p5x_view_v1.json"
        stable_write(b1_path, b1_view)
        stable_write(b2_path, b2_view)
        b1_sha, b2_sha = sha256_file(b1_path), sha256_file(b2_path)
        require(b1_view["total_retained_windows"] == b2_view["total_retained_windows"] == half_total,
                "B1/B2 half-view retained totals differ")
        print(f"      PASS: 0.5x retained windows per generator = {half_total:,}")
        print("      PASS: B1 and B2 use the exact same retained pool indices in every stratum")

        print("[4/4] Freeze s2_74 artifacts and write audit report...")
        freeze_manifest = {
            "schema": "s2_74_freeze_manifest_v1",
            "stage": STAGE,
            "status": "S2_74_POOL_INTEGRITY_AND_0P5X_FROZEN",
            "implementation_version": VERSION,
            "implementation_sha256": script_sha,
            "source_artifacts": {
                "s2_72_protocol": {"path": p_path.relative_to(repo).as_posix(), "sha256": protocol_sha},
                "s2_73_generation_plan": {"path": plan_path.relative_to(repo).as_posix(), "canonical_sha256": plan_sha},
                "s2_73_pool_manifest": {"path": pool_manifest_path.relative_to(repo).as_posix(), "sha256": sha256_file(pool_manifest_path)},
            },
            "verified_pool": {
                "B1": by_generator["B1"],
                "B2": by_generator["B2"],
                "total_shards": EXPECTED_SHARDS_TOTAL,
                "all_npz_schema_valid": True,
                "all_float32": True,
                "all_finite": True,
                "all_subject_state_counts_exact": True,
                "all_file_sha256_verified": True,
                "all_embedded_metadata_verified": True,
                "all_generation_seeds_verified": True,
                "all_B1_B2_common_rng_streams_verified": True,
            },
            "nested_0p5x": {
                "master": {"path": master_path.relative_to(repo).as_posix(), "sha256": master_sha},
                "B1_view": {"path": b1_path.relative_to(repo).as_posix(), "sha256": b1_sha},
                "B2_view": {"path": b2_path.relative_to(repo).as_posix(), "sha256": b2_sha},
                "retained_windows_per_generator": half_total,
            },
            "scientific_boundary": {
                "this_stage_is_pool_integrity_not_fidelity_selection": True,
                "no_PSD_or_connectivity_model_selection_performed": True,
                "no_classifier_training_performed": True,
                "no_model_or_hyperparameter_changed": True,
                "Validation_waveforms_accessed": False,
                "Final_Test_waveforms_accessed": False,
            },
        }
        freeze_manifest_path = out_root / "manifests" / "s2_74_freeze_manifest_v1.json"
        stable_write(freeze_manifest_path, freeze_manifest)

        elapsed = time.perf_counter() - started
        report = {
            "schema": "s2_74_audit_report_v1",
            "stage": STAGE,
            "status": "S2_74_COMPLETE",
            "run_utc": now_utc(),
            "elapsed_seconds": elapsed,
            "repo": str(repo),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "implementation_sha256": script_sha,
            "pool": {"B1": by_generator["B1"], "B2": by_generator["B2"]},
            "half_retained_windows_per_generator": half_total,
            "frozen_artifacts": {
                "freeze_manifest": str(freeze_manifest_path),
                "half_master_manifest": str(master_path),
                "B1_half_view": str(b1_path),
                "B2_half_view": str(b2_path),
            },
            "audit_rows": audit_rows,
            "Validation_waveforms_accessed": False,
            "Final_Test_waveforms_accessed": False,
        }
        report_path = out_root / "metrics" / f"s2_74_audit_{run_stamp}.json"
        atomic_json_write(report_path, report)

        print("\n" + "=" * 80)
        print("S2_74_COMPLETE")
        print(f"B1 verified: {by_generator['B1']['shards']} shards, {by_generator['B1']['windows']:,} windows")
        print(f"B2 verified: {by_generator['B2']['shards']} shards, {by_generator['B2']['windows']:,} windows")
        print(f"Nested 0.5x retained per generator: {half_total:,} windows")
        print("Validation waveform accessed: NO")
        print("Final Test waveform accessed: NO")
        print(f"Elapsed: {elapsed/60:.2f} min")
        print("FREEZE MANIFEST:", freeze_manifest_path)
        print("AUDIT REPORT:", report_path)
        print("Next stage after review: s2_75 utility classifier experiments")
        print("=" * 80)
        return 0

    except Exception as exc:
        elapsed = time.perf_counter() - started
        print("\nS2_74_FAILED", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Elapsed before failure: {elapsed/60:.2f} min", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
