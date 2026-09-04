"""Stage NEMAR payloads from an audited manifest (dry-run unless --execute).

This script is prepared for the next project phase. It was not executed with
``--execute`` during the pre-download phase.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path, PurePosixPath

import pandas as pd


REQUIRED_COLUMNS = {
    "subject_id",
    "relative_path",
    "source_url",
    "payload_size_bytes",
    "payload_sha256",
}


def safe_destination(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe manifest path: {relative_path!r}")
    root_resolved = root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise ValueError(f"manifest path escapes destination: {relative_path!r}")
    return target


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, digest: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"size mismatch for {path}: expected {size}, found {actual_size}")
    actual_digest = sha256_file(path)
    if actual_digest.lower() != digest.lower():
        raise ValueError(f"SHA256 mismatch for {path}: expected {digest}, found {actual_digest}")


def validate_manifest(frame: pd.DataFrame, destination: Path) -> list[tuple[pd.Series, Path]]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest lacks columns: {sorted(missing)}")
    if frame["relative_path"].duplicated().any():
        raise ValueError("manifest contains duplicate relative_path values")
    validated: list[tuple[pd.Series, Path]] = []
    for _, row in frame.iterrows():
        url = str(row["source_url"])
        if not url.startswith("https://nemar.s3."):
            raise ValueError(f"unexpected non-NEMAR HTTPS URL: {url}")
        digest = str(row["payload_sha256"])
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError(f"invalid SHA256 for {row['relative_path']}")
        if int(row["payload_size_bytes"]) <= 0:
            raise ValueError(f"invalid payload size for {row['relative_path']}")
        validated.append((row, safe_destination(destination, str(row["relative_path"]))))
    return validated


def fetch_one(row: pd.Series, target: Path) -> str:
    size = int(row["payload_size_bytes"])
    digest = str(row["payload_sha256"])
    if target.exists():
        verify_file(target, size, digest)
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(str(row["source_url"]), headers={"User-Agent": "scenario2-audited-downloader/1"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("xb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        verify_file(partial, size, digest)
        os.replace(partial, target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return "downloaded_and_verified"


def copy_sidecars(subject_ids: list[str], metadata_root: Path, destination: Path) -> int:
    copied = 0
    for subject_id in subject_ids:
        source_dir = metadata_root / subject_id / "eeg"
        target_dir = destination / subject_id / "eeg"
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.glob(f"{subject_id}_task-resting_*")):
            if source.suffix.lower() in {".eeg", ".vhdr"}:
                continue
            target = target_dir / source.name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f"refusing to overwrite differing sidecar: {target}")
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", default="data_raw/MPI-LEMON")
    parser.add_argument("--metadata-root", default="D:/SHARIF/PROJECT/Datasets/MPI-LEMON")
    parser.add_argument("--execute", action="store_true", help="perform network downloads; absent means validation-only")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    destination = Path(args.destination)
    rows = validate_manifest(manifest, destination)
    total_bytes = int(manifest["payload_size_bytes"].sum())
    print(f"Validated {len(rows)} assets for {manifest['subject_id'].nunique()} subjects ({total_bytes} bytes).")
    if not args.execute:
        print("DRY RUN ONLY: no network request or filesystem write was made. Add --execute in the next phase.")
        return

    for index, (row, target) in enumerate(rows, start=1):
        status = fetch_one(row, target)
        print(f"[{index}/{len(rows)}] {status}: {target}")
    copied = copy_sidecars(sorted(manifest["subject_id"].unique()), Path(args.metadata_root), destination)
    print(f"Complete: all payloads verified; copied {copied} local metadata sidecars.")


if __name__ == "__main__":
    main()
