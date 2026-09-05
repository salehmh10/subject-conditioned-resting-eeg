"""Central external storage policy; no writable root overlaps repo or metadata."""
from pathlib import Path
import os
import yaml


def overlap(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


def validate_paths(repo_root, values):
    from resting_eeg.data.streaming import _assert_no_links
    repo = Path(repo_root).resolve()
    metadata = Path(values['metadata_root']).resolve()
    data = Path(values['scenario2_data_root']).resolve()
    if overlap(data, repo) or overlap(data, metadata):
        raise ValueError('External Scenario 2 root overlaps repository or metadata clone')
    for key in ('raw_root','processed_root','temp_root','runtime_root'):
        target = Path(values[key]).resolve()
        _assert_no_links(Path(values[key]).absolute())
        if target == data or data not in target.parents or overlap(target,repo) or overlap(target,metadata):
            raise ValueError(f'Unsafe external storage root: {key}')
    writable = [values[k] for k in ('raw_root','processed_root','temp_root','runtime_root')]
    for i, first in enumerate(writable):
        if any(overlap(first, second) for second in writable[i+1:]):
            raise ValueError('Writable storage roots overlap each other')
    return {k:Path(v).resolve() for k,v in values.items()}


def load_paths(repo_root, config_path=None):
    repo = Path(repo_root).resolve()
    policy = repo/'configs/paths/data_paths.yaml'
    if not policy.exists():
        policy = Path(__file__).resolve().parents[3]/'configs/paths/data_paths.yaml'
    spec = yaml.safe_load(policy.read_text())
    override = config_path or os.environ.get('SCENARIO2_PATH_CONFIG')
    local = repo/'configs/paths/local_data_paths.yaml'
    if override or local.exists():
        spec.update(yaml.safe_load(Path(override or local).read_text()))
    def resolve(base, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (base/path).resolve()
    data = resolve(repo,spec['scenario2_data_root'])
    values = dict(metadata_root=resolve(repo,spec['metadata_root']),scenario2_data_root=data)
    values.update({k:resolve(data,spec[k]) for k in ('raw_root','processed_root','temp_root','runtime_root')})
    return validate_paths(repo,values)


def relocated_path(path, repo_root=None):
    """Resolve historical receipt paths without rewriting scientific provenance."""
    repo = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    original = Path(path)
    original = original.resolve() if original.is_absolute() else (repo/original).resolve()
    policy = load_paths(repo)
    mappings = [(repo/'data_raw/MPI-LEMON',policy['raw_root']),
        (repo/'data_processed/MPI-LEMON',policy['processed_root']/'train-only-v1'),
        (repo/'data_processed/MPI-LEMON-all-blocks-v1',policy['processed_root']/'all-blocks-v1')]
    for old,new in mappings:
        if original == old or old in original.parents:
            return new/original.relative_to(old)
    return original
