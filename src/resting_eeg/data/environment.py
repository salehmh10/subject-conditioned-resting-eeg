"""Project-controlled cache/temp environment and Windows stability guard."""
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile

MINIMUM_SYSTEM_FREE_BYTES = 3 * 1073741824


class SystemDriveLowSpaceError(OSError):
    pass


def system_drive_guard(operation, *, windows=None, disk_usage=None):
    if windows is None:
        windows = sys.platform == 'win32'
    if not windows:
        return dict(operation=operation, applicable=False)
    free = (disk_usage or shutil.disk_usage)('C:/').free
    evidence = dict(operation=operation, applicable=True, system_drive='C:/',
                    free_bytes=free, minimum_free_bytes=MINIMUM_SYSTEM_FREE_BYTES)
    if free < MINIMUM_SYSTEM_FREE_BYTES:
        runtime=os.environ.get('SCENARIO2_RUNTIME_ROOT')
        if runtime:
            from resting_eeg.common import write_json,utc_now
            write_json(Path(runtime)/'c_drive_low_space_guard.json',dict(evidence,status='c_drive_low_space_guard',timestamp_utc=utc_now()))
        raise SystemDriveLowSpaceError(f'c_drive_low_space_guard: {operation}; C: free {free} bytes < {MINIMUM_SYSTEM_FREE_BYTES}')
    return evidence


def environment_values(storage):
    temp = storage['temp_root']
    return dict(TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp),
        PIP_CACHE_DIR=str(temp/'pip'), MPLCONFIGDIR=str(temp/'matplotlib'),
        JOBLIB_TEMP_FOLDER=str(temp/'joblib'), PYTHONPYCACHEPREFIX=str(temp/'python'),
        MNE_DONTWRITE_HOME='true', _MNE_FAKE_HOME_DIR=str(temp/'mne'),
        MNE_CACHE_DIR=str(temp/'mne'), SCENARIO2_RUNTIME_ROOT=str(storage['runtime_root']),
        XDG_CACHE_HOME=str(temp/'python'), XDG_CONFIG_HOME=str(temp/'python'),
        PYTEST_ADDOPTS=f'--basetemp="{temp / "pytest"}" -o cache_dir="{temp / "pytest-cache"}"')


def configure_environment(repo_root, *, create=True):
    from resting_eeg.data.paths import load_paths
    storage = load_paths(repo_root)
    values = environment_values(storage)
    os.environ.update(values)
    # tempfile may already have cached a C: path in this interpreter.
    tempfile.tempdir = None
    sys.pycache_prefix = values['PYTHONPYCACHEPREFIX']
    if create:
        for suffix in ('pytest','pip','matplotlib','joblib','python','mne','batch','pytest-cache'):
            (storage['temp_root']/suffix).mkdir(parents=True,exist_ok=True)
        storage['runtime_root'].mkdir(parents=True,exist_ok=True)
    actual = Path(tempfile.gettempdir()).resolve()
    if actual != storage['temp_root'] or (sys.platform == 'win32' and actual.drive.upper() != 'D:'):
        raise ValueError('Project tempfile must resolve to the configured external D: temp root')
    return dict(environment=values, tempfile_gettempdir=str(actual),
                resolved_paths={k:str(v) for k,v in storage.items()},
                system_drive=system_drive_guard('environment_preflight'))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--values-only',action='store_true')
    parser.add_argument('--for-tests',action='store_true')
    args=parser.parse_args()
    if args.values_only:
        from resting_eeg.data.paths import load_paths
        result=environment_values(load_paths(args.repo_root))
    else:
        result=configure_environment(args.repo_root)
        if args.for_tests:
            result['pytest_guard']=system_drive_guard('full_pytest_suite')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
