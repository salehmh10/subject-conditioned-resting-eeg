"""Explicit, deterministic preparation of saved evidence (no study execution)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

APP = Path(__file__).resolve().parents[1]
REPOSITORY = APP.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(APP))

from lab.adapters import build_bundle
from lab.sources import EvidenceError, EvidenceReader, discover_sources, sha256_file
from lab.validation import acceptance_anchors, validate_bundle


def prepare(repository=REPOSITORY, config=None, force=False):
    output = APP / "data" / "presentation_bundle"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "bundle.json"
    sources = discover_sources(repository, config)
    code_files = [Path(__file__), APP / "lab" / "sources.py", APP / "lab" / "adapters.py", APP / "lab" / "validation.py"]
    code_signature = hashlib.sha256("".join(sha256_file(p) for p in code_files).encode()).hexdigest()
    if target.exists() and not force:
        current = json.loads(target.read_text(encoding="utf-8"))
        manifest = current["provenance"]["sources"]
        if current.get("preparation", {}).get("code_signature") == code_signature and current["preparation"].get("source_roots") == sources:
            unchanged = all((repository / row["path"]).is_file() and sha256_file(repository / row["path"]) == row["sha256"] for row in manifest.values())
            if unchanged:
                print(f"Presentation bundle is current; verified {len(manifest)} unchanged source hashes.")
                return target
    started = time.perf_counter()
    reader = EvidenceReader(repository)
    bundle = build_bundle(reader, sources)
    bundle["validation"] = validate_bundle(bundle)
    bundle["validation"]["acceptance_anchors"] = acceptance_anchors(bundle)
    bundle["provenance"]["integrity"] = reader.verify_unchanged()
    bundle["preparation"] = {"code_signature": code_signature, "source_roots": sources,
                             "format": "deterministic compact JSON; no generated timestamp"}
    payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    temporary = output / "bundle.json.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    manifest = {"bundle_sha256": hashlib.sha256(payload).hexdigest(), "bundle_bytes": len(payload),
                "schema_version": bundle["schema_version"], "sources": reader.manifest,
                "integrity": bundle["provenance"]["integrity"], "validation": bundle["validation"]}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Prepared {target.relative_to(APP)} ({len(payload) / 1024**2:.2f} MiB) in {time.perf_counter() - started:.2f} s.")
    print(f"Verified {len(reader.manifest)} source hashes unchanged; all numerical acceptance anchors passed.")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, help="Optional JSON paths relative to the repository")
    parser.add_argument("--force", action="store_true", help="Rebuild deterministic presentation data even when source signatures match")
    args = parser.parse_args()
    try:
        prepare(config=args.sources, force=args.force)
    except (EvidenceError, KeyError, ValueError) as error:
        parser.exit(1, f"Presentation preparation stopped: {error}\nNo experiment was executed. Inspect the source mapping and saved evidence.\n")


if __name__ == "__main__":
    main()
