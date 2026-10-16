"""Verify publication-copy hashes and sizes only, without scientific execution.

The default inventories use repository-relative paths and require no access to
the original research machine. This is packaging validation, not a scientific
test, result recalculation, or verification of omitted external dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root: Path, manifest_paths: list[str]) -> tuple[int, list[str]]:
    root = root.resolve(strict=True)
    errors: list[str] = []
    checked = 0
    for manifest_path in manifest_paths:
        manifest = (root / manifest_path).resolve()
        if not manifest.is_relative_to(root):
            errors.append(f"Manifest leaves repository: {manifest_path}")
            continue
        try:
            inventory = json.loads(manifest.read_text(encoding="utf-8-sig"))
            entries = inventory["files"]
            if not isinstance(entries, list):
                raise ValueError("files must be a list")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Cannot load {manifest_path}: {exc}")
            continue
        seen: set[str] = set()
        for entry in entries:
            try:
                relative = entry["path"]
                logical = PurePosixPath(relative)
                if logical.is_absolute() or ".." in logical.parts or "\\" in relative or ":" in relative:
                    raise ValueError("path must be a safe repository-relative POSIX path")
                if relative in seen:
                    raise ValueError("duplicate entry in manifest")
                seen.add(relative)
                path = (root / relative).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError("path leaves publication root")
                expected_hash = entry["sha256"]
                expected_size = entry["size_bytes"]
                if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
                    raise ValueError("invalid SHA256")
                data = path.read_bytes()
                if len(data) != expected_size:
                    raise ValueError(f"size differs: expected {expected_size}, got {len(data)}")
                if hashlib.sha256(data).hexdigest() != expected_hash:
                    raise ValueError("SHA256 differs")
                checked += 1
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"{entry.get('path', '<missing path>')}: {exc}")
    return checked, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", action="append", help="Repository-relative inventory; repeatable")
    args = parser.parse_args()
    manifests = args.manifest
    if manifests is None:
        manifests = ["evidence/provenance/evidence_inventory.json"]
        if (args.root / "provenance/source_inventory.json").is_file():
            manifests.append("provenance/source_inventory.json")
    checked, errors = verify(args.root, manifests)
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Copy integrity: {checked} files matched; {len(errors)} errors; {len(manifests)} inventories.")
    print("No scientific code, tests, inference, training, or statistical recalculation executed.")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
