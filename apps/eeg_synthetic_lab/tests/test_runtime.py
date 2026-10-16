"""Runtime and presentation safety contracts, independent of source preparation."""
import ast
import importlib.util
import json
from pathlib import Path
import socket

import pytest

from lab.formatting import percent, percentage_points, reverse_contrast

APP = Path(__file__).resolve().parents[1]


def test_fraction_percent_and_pp_are_distinct():
    assert percent(.88417, 3) == '88.417%'
    assert percentage_points(.00718) == '+0.718 pp'
    assert reverse_contrast(.00718, .00250, .01201) == (-.00718, -.01201, -.00250)
    assert percent(None) == 'Unavailable'


def test_runtime_has_no_scientific_execution_or_raw_read_dependencies():
    paths = [APP / 'app.py', APP / 'lab/config.py', APP / 'lab/formatting.py']
    paths += list((APP / 'lab/views').glob('*.py')) + list((APP / 'lab/charts').glob('*.py'))
    prohibited = {'torch', 'h5py', 'scipy', 'resting_eeg', 'scripts', 'subprocess'}
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [n.name.split('.')[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or '').split('.')[0]]
            else:
                continue
            assert not set(names) & prohibited, (path.name, names)
        assert 'allow_pickle=True' not in path.read_text(encoding='utf-8')


def test_available_port_does_not_take_an_occupied_socket():
    spec = importlib.util.spec_from_file_location('lab_launcher', APP / 'tools/launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with socket.socket() as busy:
        busy.bind(('127.0.0.1', 0))
        busy.listen()
        port = module.available_port(busy.getsockname()[1])
        assert port != busy.getsockname()[1]
    with pytest.raises(ValueError):
        module.available_port(80)


def test_local_only_configuration():
    import tomllib
    config = tomllib.loads((APP / '.streamlit/config.toml').read_text())
    assert config['browser']['gatherUsageStats'] is False
    assert config['server']['address'] == '127.0.0.1'
    assert config['server']['fileWatcherType'] == 'none'


def test_psd_display_range_ignores_hidden_stopband_without_dropping_data():
    from lab.charts import psd_chart
    frequencies=[0,1,8,13,40,64]
    values=[-100,5,10,7,-12,-200]
    figure=psd_chart(frequencies, {'Real':values})
    assert list(figure.data[0].y)==values
    assert figure.layout.yaxis.range[0]>-30
    assert figure.layout.yaxis.range[1]>10


def test_prepared_evidence_complete_and_only_final_utility():
    path = APP / 'data/presentation_bundle/bundle.json'
    if not path.exists():
        pytest.skip('Run prepare_data.ps1 to test real evidence.')
    data = json.loads(path.read_text(encoding='utf-8'))
    for split in ('validation', 'final_test'):
        rows = [r for r in data['utility_runs'] if r['split'] == split]
        assert len(rows) == 55
        assert len({(r['family'], r['condition'], str(r['seed'])) for r in rows}) == 55
        assert len([r for r in data['utility_summary'] if r['split'] == split]) == 33
    # A path outside the final utility root is not an acceptable utility source.
    for row in data.get('provenance', {}).get('sources', {}).values():
        path_text = str(row.get('path', row.get('source_path', '')))
        if '/utility/' in path_text and 'eleven_condition' in path_text:
            assert 'eleven_condition_s2_75_v2_120epoch_es' in path_text
