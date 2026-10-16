#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""s2_75C: one-time, report-only recovery for the s2_75B-v2 CSV fieldnames bug.

NEVER trains a model, changes source code, rewrites task receipts, or reads EEG
Train/Validation/Final Test waveforms. Reuses the *unchanged* original aggregate()
and its hash/prediction checks. The only in-memory fix adds four EEGNet-specific
CSV headers so DictWriter accepts the mixed classical/EEGNet result rows.

From the Scenario-2 repository root:
    .\.venv\Scripts\python.exe -u .\scripts\data\s2_75c_finalize_v2_report_csv_fix.py

The original s2_75b_v2_utility_120ep_earlystop.py MUST remain byte-for-byte
identical to the version that generated the 55 completed task receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import sys
from contextlib import contextmanager
from pathlib import Path

ORIGINAL_FILENAME = "s2_75b_v2_utility_120ep_earlystop.py"
ORIGINAL_SHA256 = "4fac6df5363b465a60c98788f4aa968ec99a2339589455925c00f27753146043"
EXPECTED_AMENDMENT_SHA256 = "86688d27dcd7d961e2ac96e4e2bd5de1b4278b4d14e123a8efb1b47a2d5a0650"
EXTRA_FIELDS = ("best_epoch", "stopped_epoch", "stop_reason", "optimizer_updates_done")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def use_fixed_csv_fieldnames():
    """Temporary repair only for the CSV writer in original aggregate().

    Original aggregate takes fieldnames from rows[0] (a classical model), but
    EEGNet rows have four additional metadata keys. DictWriter then raises a
    ValueError. Fix: add the four explicit known columns; leave all row values
    intact, and restore the standard-library class in a finally block.
    """
    original_writer = csv.DictWriter

    class CompatibleV2CsvWriter(original_writer):
        def __init__(self, f, fieldnames, *args, **kwargs):
            columns = list(fieldnames)
            require(columns[:3] == ["family", "condition", "seed"],
                    "Unexpected CSV header layout; refusing recovery")
            for key in EXTRA_FIELDS:
                if key not in columns:
                    columns.append(key)
            super().__init__(f, columns, *args, **kwargs)

    csv.DictWriter = CompatibleV2CsvWriter
    try:
        yield
    finally:
        csv.DictWriter = original_writer


def self_test() -> None:
    """No access to the real repository or any real data."""
    saved = csv.DictWriter
    rows = [
        {"family": "LOGBANDPOWER_LOGREG", "condition": "REAL_ONLY", "seed": 20262301,
         "subject_macro_balanced_accuracy": 0.8},
        {"family": "EEGNETV4", "condition": "REAL_ONLY", "seed": 20262301,
         "subject_macro_balanced_accuracy": 0.9,
         "best_epoch": 19, "stopped_epoch": 25, "stop_reason": "early_stopping",
         "optimizer_updates_done": 9350},
    ]
    out = io.StringIO(newline="")
    with use_fixed_csv_fieldnames():
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    require(csv.DictWriter is saved, "CSV patch did not restore original writer")
    parsed = list(csv.DictReader(io.StringIO(out.getvalue())))
    require(len(parsed) == 2 and parsed[0]["best_epoch"] == "" and
            parsed[1]["best_epoch"] == "19" and
            parsed[1]["stop_reason"] == "early_stopping" and
            parsed[1]["optimizer_updates_done"] == "9350",
            "CSV fieldnames regression test failed")
    print("S2_75C_CSV_FIX_SELF_TEST_PASS")


def finalize(repo: Path) -> None:
    repo = repo.resolve()
    require((repo / "scripts" / "data").is_dir(), "Not a Scenario-2 repository root")
    original = repo / "scripts" / "data" / ORIGINAL_FILENAME
    require(original.is_file() and not original.is_symlink(),
            "Missing original s2_75B-v2 script")
    actual = sha256(original)
    require(actual == ORIGINAL_SHA256,
            f"Original script SHA256 differs; DO NOT edit it or use this recovery: {actual}")
    print("Original v2 script SHA256: VERIFIED UNCHANGED", flush=True)

    # Imported as a module: original main() is NOT called, so no run-all/training.
    spec = importlib.util.spec_from_file_location("_s2_75b_v2_original_for_report", original)
    require(spec is not None and spec.loader is not None, "Cannot import original script")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    ctx = mod.context(repo)
    require(ctx["am_sha"] == EXPECTED_AMENDMENT_SHA256,
            "Frozen v2 utility amendment SHA256 differs")
    require(len(mod.schedule(ctx)) == 55, "Expected exactly 55 v2 tasks")
    print("S2_75C_REPORT_ONLY: 55 receipts/models/predictions will be reverified; no training", flush=True)

    # Reuse the original exclusive lock and its full receipt/prediction/hash checks.
    with mod.writer_lock(ctx["out"]):
        with use_fixed_csv_fieldnames():
            report = mod.aggregate(ctx)

        require(report["status"] == "S2_75B_V2_VALIDATION_UTILITY_COMPLETE" and
                report["n_tasks"] == 55,
                "Final report status or number of tasks is not complete")
        final_dir = ctx["out"] / "final_validation"
        csv_path = final_dir / "all_55_classifier_validation_runs.csv"
        json_path = final_dir / "s2_75b_v2_complete_validation_report.json"
        require(csv_path.is_file() and json_path.is_file(), "Expected reports missing")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            table = list(reader)
        require(len(table) == 55 and all(key in headers for key in EXTRA_FIELDS),
                "CSV incomplete or EEGNet metadata columns missing")
        family_count = {family: sum(row["family"] == family for row in table)
                        for family in mod.FAMILIES}
        require(family_count == {"LOGBANDPOWER_LOGREG": 11,
                                 "OAS_RIEMANN_LINEAR_SVM": 11, "EEGNETV4": 33},
                "Unexpected family counts in CSV")
        for row in table:
            if row["family"] == "EEGNETV4":
                require(all(row[key] != "" for key in EXTRA_FIELDS),
                        "Missing EEGNet metadata in CSV")
            else:
                require(all(row[key] == "" for key in EXTRA_FIELDS),
                        "Unexpected EEGNet fields in classical classifier row")

        figs = report.get("figures", {})
        print("S2_75C_RECOVERY_COMPLETE: all 55 fit receipts/predictions verified; CSV fixed", flush=True)
        print("CSV:", csv_path, flush=True)
        print("REPORT:", json_path, flush=True)
        print("FIGURES:", figs.get("status", "UNKNOWN"), "count=", len(figs.get("files", [])), flush=True)
        print("Validation-based model selection: YES (development evidence)", flush=True)
        print("Final Test waveform accessed: NO", flush=True)
        if figs.get("status") != "CREATED" or len(figs.get("files", [])) != 14:
            print("WARNING: expected 14 figures; inspect figures status/error before reporting.", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--self-test", action="store_true", help="Offline self-test; reads no project data")
    args = p.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            finalize(args.repo_root)
        return 0
    except Exception as exc:
        print("S2_75C_RECOVERY_FAILED:", type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
