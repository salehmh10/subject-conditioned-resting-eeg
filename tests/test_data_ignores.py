from pathlib import Path
import subprocess
import pytest


@pytest.mark.parametrize('relative', ['data_raw/MPI-LEMON/test.eeg','data_processed/MPI-LEMON/test.h5',
                                     'data_normalized/test.npy','.venv/pyvenv.cfg'])
def test_scientific_payloads_and_local_environment_are_git_ignored(relative):
    repo=Path(__file__).resolve().parents[1]
    assert subprocess.run(['git','check-ignore','-q',relative],cwd=repo).returncode == 0
