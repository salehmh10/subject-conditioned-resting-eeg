#!/usr/bin/env python3
"""Collect source context for s2_73. This script does not generate EEG.

Run with the existing project Python in isolated mode, for example:
    python -I collect_s2_73_source_context.py --repo-root path/to/repository

The ZIP path is printed when collection finishes. Attach that ZIP for source
review before implementing the actual 2-percent B1/B2 generation run.
"""

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid
import zipfile


MIB = 1024 * 1024
TOTAL_LIMIT = 50 * MIB
PROTOCOL_ROOT = Path(
    "outputs/protocols/scenario2/utility/b1_vs_b2_utility_protocol_v1"
)
PROTOCOL_FILE = PROTOCOL_ROOT / "metrics/s2_72_b1_vs_b2_downstream_utility_protocol_v1.json"
EXPECTED_PROTOCOL_SHA256 = "5b98ff09d977006b96ea80e43d4f13f33742745b734d5dfa94bb1eaacb682a22"
JSON_ROOTS = (
    PROTOCOL_ROOT,
    Path("outputs/frozen/scenario2/b1/b1_subject_state_raw_epoch010_v1"),
    Path("outputs/frozen/scenario2/b2/b2_validation_selected_replication_e1_v1"),
)
PRUNED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "outputs", "checkpoints",
    "__pycache__", "datasets", "raw", "project_sources", "site-packages",
}
SENSITIVE_NAME = re.compile(
    r"(?:^|[_.-])(?:credentials?|secrets?|tokens?|passwords?|private_keys?|api_keys?)(?:[_.-]|$)",
    re.IGNORECASE,
)
ANCHORS = ("s2_70*.py", "s2_71*.py", "s2_72*.py")


def safe_path(path, root):
    """Reject links/reparse points and paths outside the approved root."""
    try:
        relative = path.absolute().relative_to(root)
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            info = cursor.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                return False
        path.resolve(strict=True).relative_to(root)
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def walk_files(start, repo, suffix):
    if not safe_path(start, repo) or not start.is_dir():
        return
    for current, directories, files in os.walk(start, followlinks=False):
        parent = Path(current)
        directories[:] = sorted(
            name for name in directories
            if name.lower() not in PRUNED_DIRS
            and not name.startswith(".") and not SENSITIVE_NAME.search(name)
            and not (parent == repo and name.lower() == "data")
            and safe_path(parent / name, repo)
        )
        for name in sorted(files):
            path = parent / name
            if (path.suffix.lower() == suffix and not name.startswith(".")
                    and not SENSITIVE_NAME.search(name) and safe_path(path, repo)):
                yield path


def new_output_folder(repo):
    parent = repo
    for name in ("outputs", "source_handoff"):
        parent = parent / name
        if parent.exists() or parent.is_symlink():
            if not safe_path(parent, repo) or not parent.is_dir():
                raise ValueError("Output parent is unsafe or is not a directory.")
        else:
            parent.mkdir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / ("s2_73_context_" + stamp + "_" + uuid.uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="Repository directory; defaults to the current directory.")
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve(strict=True)
    if not repo.is_dir():
        parser.error("--repo-root must be a directory.")
    plausible = any(safe_path(repo / relative, repo) and (repo / relative).is_dir()
                    for relative in (Path("scripts/data"), Path("src")))
    if not plausible:
        parser.error("Expected scripts/data or src. Run from the project or pass --repo-root.")

    output = new_output_folder(repo)
    temporary = output / "s2_73_source_context.zip.tmp"
    final = output / "s2_73_source_context.zip"
    own_script = Path(__file__).resolve()
    files, skipped, seen = [], [], set()
    total_bytes = 0
    actual_protocol_sha256 = None
    # Prefer the exact protocol and metadata before the broader source inventory.
    candidates = [(repo / PROTOCOL_FILE, 5 * MIB, "protocol")]
    for root in JSON_ROOTS:
        candidates.extend((path, 5 * MIB, "metadata_json")
                          for path in walk_files(repo / root, repo, ".json"))
    candidates.extend((path, 2 * MIB, "python_source")
                      for path in walk_files(repo, repo, ".py"))

    with temporary.open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, limit, kind in candidates:
                relative = path.relative_to(repo).as_posix()
                if relative in seen:
                    continue
                seen.add(relative)
                if path == own_script:
                    continue
                try:
                    if not safe_path(path, repo) or not path.is_file():
                        skipped.append({"path": relative, "reason": "missing_or_unsafe"})
                        continue
                    if path.stat().st_size > limit:
                        skipped.append({"path": relative, "reason": "per_file_size_limit"})
                        continue
                    with path.open("rb") as source:
                        payload = source.read(limit + 1)
                    if len(payload) > limit or total_bytes + len(payload) > TOTAL_LIMIT:
                        skipped.append({"path": relative, "reason": "size_limit"})
                        continue
                except OSError as error:
                    skipped.append({"path": relative, "reason": type(error).__name__})
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                archive.writestr("project/" + relative, payload)
                files.append({"path": relative, "kind": kind,
                              "bytes": len(payload), "sha256": digest})
                total_bytes += len(payload)
                if path == repo / PROTOCOL_FILE:
                    actual_protocol_sha256 = digest

            matches = {pattern: sorted(item["path"] for item in files
                       if item["kind"] == "python_source"
                       and fnmatch.fnmatch(Path(item["path"]).name, pattern))
                       for pattern in ANCHORS}
            missing = [pattern for pattern, paths in matches.items() if not paths]
            if actual_protocol_sha256 is None:
                missing.append(PROTOCOL_FILE.as_posix())
            for root in JSON_ROOTS:
                if not safe_path(repo / root, repo) or not (repo / root).is_dir():
                    missing.append(root.as_posix())
            hash_matches = actual_protocol_sha256 == EXPECTED_PROTOCOL_SHA256
            manifest = {
                "purpose": "SOURCE COLLECTION ONLY; no EEG generation or model loading",
                "intended_next_run": "First approximately 2% of paired B1/B2 generation",
                "source_anchors": matches, "missing_anchors": sorted(set(missing)),
                "protocol_sha256": {"expected": EXPECTED_PROTOCOL_SHA256,
                                    "actual": actual_protocol_sha256, "matches": hash_matches},
                "total_bytes": total_bytes, "files": sorted(files, key=lambda item: item["path"]),
                "skipped": sorted(skipped, key=lambda item: item["path"]),
            }
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            archive.writestr("README.txt", (
                "SOURCE COLLECTION ONLY\n\n"
                "No EEG was generated. No project module, checkpoint or waveform was loaded.\n"
                "Project files were read without modification. Only Python sources and JSON\n"
                "from the three specified protocol/frozen roots are included.\n\n"
                "Next intended task: implement the actual frozen s2_73 generator, stopping\n"
                "after approximately 2% of paired B1/B2 windows at complete shard boundaries.\n"
                "This bundle does not implement or authorize changes to that protocol.\n"
                "Inspect manifest.json for missing anchors, skipped files and protocol hash\n"
                "status. A completed collection is not proof of readiness to generate.\n"
            ))
        raw.flush()
        os.fsync(raw.fileno())
    if final.exists():
        raise FileExistsError("Refusing to replace an existing output ZIP.")
    os.replace(temporary, final)
    print("SOURCE COLLECTION ONLY - no EEG generated.")
    print("ZIP:", final)
    print("Collected files:", len(files), "| Bytes:", total_bytes, "| Skipped:", len(skipped))
    print("Protocol hash matches handoff:", "YES" if hash_matches else "NO / MISSING")
    for pattern, paths in matches.items():
        print("Source anchor", pattern + ":", "PRESENT" if paths else "MISSING")
    print("Missing required anchors:", len(set(missing)), "(see manifest.json)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print("Source collection failed:", type(error).__name__, str(error), file=sys.stderr)
        sys.exit(1)
