from pathlib import Path

import pandas as pd

from scripts.audit.s2_01_audit_metadata import (
    audit_subject,
    channel_type_counts,
    inspect_small_pointer,
)


TARGETS = ["Fp1", "Fp2"]


def test_pointer_parser_extracts_size_and_hash(tmp_path: Path) -> None:
    digest = "a" * 64
    pointer = tmp_path / "sample.eeg"
    pointer.write_text(f"/annex/objects/SHA256E-s123--{digest}.eeg\n", encoding="utf-8")
    parsed = inspect_small_pointer(pointer)
    assert parsed["is_pointer"] is True
    assert parsed["payload_size_bytes"] == 123
    assert parsed["payload_sha256"] == digest
    assert parsed["payload_present"] is False


def test_materialized_eeg_is_not_opened_as_text(tmp_path: Path) -> None:
    payload = tmp_path / "sample.eeg"
    payload.write_bytes(b"x" * 513)
    parsed = inspect_small_pointer(payload)
    assert parsed["is_pointer"] is False
    assert parsed["payload_present"] is True


def test_channel_types_are_separated() -> None:
    channels = pd.DataFrame({"type": ["EEG", "eeg", "EOG", "MISC", None]})
    assert channel_type_counts(channels) == {"EEG": 2, "EOG": 1, "MISC": 1, "MISSING": 1}


def test_duplicate_target_name_is_not_complete(tmp_path: Path) -> None:
    subject = tmp_path / "sub-test" / "eeg"
    subject.mkdir(parents=True)
    pd.DataFrame(
        {
            "name": ["Fp1", "Fp1", "Fp2", "VEOG"],
            "type": ["EEG", "EEG", "EEG", "EOG"],
            "status": ["good"] * 4,
        }
    ).to_csv(subject / "sub-test_task-resting_channels.tsv", sep="\t", index=False)
    inventory, _, _ = audit_subject(subject.parent, TARGETS)
    assert inventory["target_19_present"] is False
    assert inventory["duplicate_target_channels"] == "Fp1"
    assert inventory["n_eeg_channels"] == 3
    assert inventory["n_eog_channels"] == 1
