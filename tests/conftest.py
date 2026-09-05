"""Require the external pytest root before pytest creates any test directories."""
from pathlib import Path
import pytest
from resting_eeg.data.environment import configure_environment, system_drive_guard
from resting_eeg.data.paths import load_paths


def pytest_configure(config):
    root=Path(__file__).resolve().parents[1]
    configure_environment(root)
    expected=load_paths(root)['temp_root']/'pytest'
    supplied=config.option.basetemp
    if supplied is None or Path(supplied).resolve()!=expected:
        raise pytest.UsageError(f'Explicit --basetemp {expected} required; dot-source scripts/setup_scenario2_environment.ps1')
    system_drive_guard('full_pytest_suite')
