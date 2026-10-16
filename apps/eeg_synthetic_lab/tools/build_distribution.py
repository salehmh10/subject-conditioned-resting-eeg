"""Package the app with aggregate evidence, excluding environments and raw data."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

APP = Path(__file__).resolve().parents[1]


def aggregate_bundle(original):
    import math
    import statistics

    bundle = deepcopy(original)
    # Retain the same descriptive mean of saved participant diagnostics that the
    # local group view displays, without retaining any participant-level row.
    grouped = {}
    for row in original.get('connectivity_metrics', []):
        identity = (row['state'], row['band'], row['model'])
        grouped.setdefault(identity, []).append(row)
    aggregate_connectivity = []
    transformations = bundle.setdefault('provenance', {}).setdefault('transforms', {})
    for (state, band, model), rows in sorted(grouped.items()):
        record = {'state': state, 'band': band, 'model': model,
                  'aggregation': 'Equal-participant finite mean of saved connectivity diagnostics',
                  'n_subjects': len(rows)}
        numeric_fields = sorted({key for row in rows for key, value in row.items()
                                 if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))})
        for field in numeric_fields:
            values = [float(row[field]) for row in rows if row.get(field) is not None
                      and math.isfinite(float(row[field]))]
            record[field] = statistics.fmean(values) if values else None
        source_ids = set()
        for row in rows:
            lineage = row.get('provenance', {})
            if 'transform_id' in lineage:
                lineage = original['provenance']['transforms'][lineage['transform_id']]
            source_ids.update(lineage.get('source_ids', []))
        lineage = {'source_ids': sorted(source_ids),
                   'transformation': 'Equal-participant finite arithmetic mean of each saved per-participant connectivity diagnostic; no metric recalculation from group matrices and no new uncertainty interval',
                   'units': 'dwPLI² for strength/MAE; correlation r; dimensionless relative Frobenius error; fraction; valid-edge count',
                   'aggregation': 'group/state/band/model; equal participant weight separately for finite values of each diagnostic',
                   'analysis_split': 'real_train'}
        transform_id = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()[:20]
        transformations[transform_id] = lineage
        record['provenance'] = {'transform_id': transform_id}
        aggregate_connectivity.append(record)
    bundle['connectivity_metrics'] = aggregate_connectivity
    for key in ('fidelity_subjects', 'fingerprint_subjects', 'utility_subjects'):
        bundle[key] = []
    for key in ('psd', 'connectivity', 'alpha_effect'):
        for field in ('participants', 'sources_by_subject', 'valid_frequency_bins'):
            bundle.get(key, {}).pop(field, None)
    bundle['metadata']['shared_export'] = True
    bundle['metadata']['export_scope'] = 'Aggregate edition: participant diagnostic arrays and full preview pool omitted.'
    availability = bundle.setdefault('availability', {})
    for key, value in list(availability.items()):
        if isinstance(value, bool) and ('participant' in key or 'subject' in key or 'preview' in key):
            availability[key] = False
    availability['external_preview_bundle'] = None
    availability['waveform_reason'] = 'The aggregate shared edition omits the local waveform preview pool. The standalone HTML may retain one documented illustrative example.'
    availability['participant_reason'] = 'The aggregate shared edition omits participant diagnostic arrays; group numeric results remain available.'
    availability['missing_capabilities'] = list(availability.get('missing_capabilities', [])) + [
        'Participant-level diagnostic controls and the waveform preview pool are unavailable in this aggregate shared edition.'
    ]
    # Removed records must not leave a large, unused participant provenance table.
    used_transforms = set()
    def collect(value):
        if isinstance(value, dict):
            if 'transform_id' in value:
                used_transforms.add(value['transform_id'])
            for key, item in value.items():
                if key != 'transforms':
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(bundle)
    provenance = bundle.get('provenance', {})
    provenance['transforms'] = {key: value for key, value in provenance.get('transforms', {}).items() if key in used_transforms}
    bundle.setdefault('validation', {})['scope'] = 'Original local bundle passed complete participant validation; this distribution intentionally contains aggregate evidence only.'
    return bundle


def main():
    original = json.loads((APP / 'data/presentation_bundle/bundle.json').read_text(encoding='utf-8'))
    serialized = json.dumps(aggregate_bundle(original), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    output = APP / 'exports/EEG_Synthetic_Lab.zip'
    paths = []
    for dirname in ('lab', 'tools', 'assets', 'config', '.streamlit', 'tests'):
        paths += [p for p in (APP / dirname).rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.local.' not in p.name]
    paths += [APP / name for name in ('app.py', 'README.md', 'requirements.txt', 'requirements-previews.txt',
              'requirements-dev.txt', 'requirements-tested.txt', 'pytest.ini', '.gitignore', 'setup.ps1', 'prepare_data.ps1', 'run.ps1', 'run.cmd') if (APP/name).exists()]
    for name in ('DATA_SCHEMA.md', 'SOURCE_MAPPING.md', 'SCIENTIFIC_QA.md', 'OFFLINE_CHECK.md', 'TEST_RESULTS.md'):
        if (APP/'docs'/name).exists():
            paths.append(APP/'docs'/name)
    paths.append(APP/'exports/EEG_Synthetic_Lab_Presentation.html')
    # Intentionally exclude installation logs, original git status, local preview provenance,
    # environment/cache directories, and participant preview pools from shared distribution.
    forbidden = {'.venv', '.cache', '.tmp', '__pycache__', '.git'}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            rel = path.relative_to(APP)
            if set(rel.parts) & forbidden or path.suffix.lower() in {'.pth','.pt','.ckpt','.h5','.hdf5','.ttf','.woff','.woff2','.zip','.pyc'}:
                raise ValueError(f'Unsafe distribution member: {rel}')
            archive.write(path, 'eeg_synthetic_lab/'+rel.as_posix())
        archive.writestr('eeg_synthetic_lab/data/presentation_bundle/bundle.json', serialized)
        manifest = {'schema_version':1, 'scope':'aggregate shared edition',
                    'bundle_sha256':hashlib.sha256(serialized).hexdigest(),
                    'note':'Derived from the locally validated bundle. Local participant arrays and preview pool omitted. The HTML includes one tiny Train-only illustrative example when available.'}
        archive.writestr('eeg_synthetic_lab/data/presentation_bundle/distribution_manifest.json', json.dumps(manifest, indent=2))
    print(f'Created {output.name}: {output.stat().st_size:,} bytes; aggregate edition.')


if __name__ == '__main__':
    main()
