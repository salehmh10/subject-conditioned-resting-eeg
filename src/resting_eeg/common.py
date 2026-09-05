"""Small durable provenance helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_head(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def exclusive_lock(path):
    """Cross-process fail-closed lock. Stale locks require explicit investigation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError(f'Active or stale lock; investigate before retry: {path}') from exc
    try:
        handle.write(f'{os.getpid()} {utc_now()}\n')
        handle.flush()
        yield
    finally:
        handle.close()
        path.unlink()


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(path.with_suffix('.csv.lock')):
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open(encoding='utf-8', newline='') as stream:
                if next(csv.reader(stream)) != list(row):
                    raise ValueError('Access registry schema mismatch')
        with path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
