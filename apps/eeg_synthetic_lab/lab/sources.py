"""Preparation-only, read-only readers for saved scientific evidence.

This module never imports study code and deliberately has no array/model reader.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


class EvidenceError(ValueError):
    """Saved evidence is missing, inconsistent, or unsafe to interpret."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def safe_relative_path(value: str) -> str:
    """Normalize either ZIP separator while rejecting absolute/traversal names."""
    value = value.replace("\\", "/")
    if not value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise EvidenceError(f"Absolute or empty source path is forbidden: {value}")
    parts = value.split("/")
    if any(part in ("..", ".") for part in parts):
        raise EvidenceError(f"Traversal source path is forbidden: {value}")
    return PurePosixPath(value).as_posix()


def safe_zip_json(archive: Path, member: str):
    """Read directly; never extract archive contents or deserialize executable data."""
    desired = safe_relative_path(member)
    with zipfile.ZipFile(archive) as handle:
        names = {}
        for info in handle.infolist():
            normalized = safe_relative_path(info.filename)
            if normalized in names:
                raise EvidenceError(f"Duplicate normalized ZIP member: {normalized}")
            names[normalized] = info
        if desired not in names:
            raise EvidenceError(f"Missing archive member: {desired}")
        info = names[desired]
        if info.file_size > 64 * 1024 * 1024:
            raise EvidenceError("Refusing unexpectedly large JSON archive member")
        return json.loads(handle.read(info).decode("utf-8-sig"), object_pairs_hook=unique_object)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EvidenceReader:
    """Resolve only relative repository paths and record consumed source signatures."""

    def __init__(self, repository: Path):
        self.repository = repository.resolve()
        self.manifest = {}

    def path(self, relative: str) -> Path:
        relative = safe_relative_path(relative)
        target = (self.repository / relative).resolve()
        if not target.is_relative_to(self.repository):
            raise EvidenceError("Source resolves outside the repository")
        if not target.is_file():
            raise EvidenceError(f"Missing saved evidence: {relative}")
        return target

    def read_bytes(self, relative: str, role: str = "scientific evidence") -> bytes:
        path = self.path(relative)
        if path.suffix.lower() not in {".json", ".csv", ".py", ".md", ".png"}:
            raise EvidenceError(f"Unsupported presentation evidence format: {path.suffix}")
        raw = path.read_bytes()
        signature = hashlib.sha256(raw).hexdigest()
        source_id = safe_relative_path(relative)
        previous = self.manifest.get(source_id)
        if previous and previous["sha256"] != signature:
            raise EvidenceError(f"Source changed during preparation: {source_id}")
        self.manifest[source_id] = {"path": source_id, "sha256": signature,
                                    "bytes": len(raw), "role": role}
        return raw

    def json(self, relative: str, role: str = "saved metrics"):
        return json.loads(self.read_bytes(relative, role).decode("utf-8-sig"),
                          object_pairs_hook=unique_object)

    def csv(self, relative: str, role: str = "saved table") -> list[dict]:
        text = self.read_bytes(relative, role).decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise EvidenceError(f"Missing or duplicate CSV columns: {relative}")
        return list(reader)

    def verify_unchanged(self) -> dict:
        changed = [key for key, item in self.manifest.items()
                   if sha256_file(self.path(item["path"])) != item["sha256"]]
        if changed:
            raise EvidenceError(f"Scientific source hash changed: {', '.join(changed)}")
        return {"status": "verified_unchanged", "file_count": len(self.manifest),
                "scope": "Every scientific source actually consumed; no raw recording scans"}


DEFAULT_SOURCES = {
    "fidelity_b0": "outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1",
    "fidelity_b1_b2": "outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1",
    "utility_v2": "outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es",
}


def discover_sources(repository: Path, config: Path | None = None) -> dict:
    """Check only the requested canonical directories, never recursive raw-data trees."""
    paths = dict(DEFAULT_SOURCES)
    if config:
        supplied = json.loads(config.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
        unknown = set(supplied) - set(paths)
        if unknown:
            raise EvidenceError(f"Unknown source configuration keys: {sorted(unknown)}")
        paths.update(supplied)
    for key, value in paths.items():
        paths[key] = safe_relative_path(value)
        if not (repository / value).is_dir():
            raise EvidenceError(f"Missing canonical evidence directory: {value}. Set {key} in a source configuration JSON.")
    if not paths["utility_v2"].endswith("eleven_condition_s2_75_v2_120epoch_es"):
        raise EvidenceError("Only the final s2_75_v2 120-epoch utility root is accepted")
    return paths
