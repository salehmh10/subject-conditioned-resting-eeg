#!/usr/bin/env python3
"""Verify packaged saved aggregates only, using the Python standard library.

No analysis code is imported, no external paths are followed, and no features,
participant bootstraps, training, generation, or Test evaluation are run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

BASE = 'evidence/validation_reference/'
ORIGINAL_SHA256 = '418892ffff6e31f92f38112da3506369581b8e555935a6889c4397c3ae4f70b6'
PORTABILITY = 'provenance/validation_portability.json'
MODELS = ('B0', 'B1', 'B2')
METRICS = ('log_psd_mae_db', 'sd_md_pooled_uv', 'alpha_effect_mae_db', 'alpha_effect_pattern_r')


class VerificationError(ValueError):
    """Invalid, missing, inconsistent, or altered publication evidence."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and bool(relative), 'Invalid publication path')
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and '..' not in parts.parts and '\\' not in relative
            and ':' not in relative, f'Unsafe publication path: {relative!r}')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f'Path escapes checkout: {relative}')
    require(path.is_file(), f'Missing packaged input: {relative}')
    return path


def load_json(path):
    def invalid_constant(value):
        raise VerificationError(f'Nonfinite JSON constant in {path.name}: {value}')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key in {path.name}: {key}')
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid_constant,
                           object_pairs_hook=unique_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f'Invalid JSON input {path.name}: {exc}') from exc
    require(isinstance(value, dict), f'Expected JSON object: {path.name}')
    return value


def exact_keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), f'Invalid schema: {label}')


def finite(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f'{label}/{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f'{label}/{index}')
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f'Nonfinite number: {label}')


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f'Expected finite number: {label}')
    return value


def close(actual, expected, label, tolerance=1e-10):
    require(abs(number(actual, label) - number(expected, label)) <= tolerance,
            f'Arithmetic/correspondence mismatch: {label}')


def stats(record, label):
    exact_keys(record, ('n_participants', 'mean', 'median', 'q25', 'q75', 'iqr', 'ci95_mean'), label)
    require(type(record['n_participants']) is int and record['n_participants'] == 198,
            f'Incorrect participant count: {label}')
    for key in ('mean', 'median', 'q25', 'q75', 'iqr'):
        number(record[key], f'{label}/{key}')
    require(record['q25'] <= record['median'] <= record['q75'], f'Quantile ordering: {label}')
    require(record['iqr'] >= 0, f'Negative IQR: {label}')
    close(record['iqr'], record['q75'] - record['q25'], f'{label}/IQR')
    ci = record['ci95_mean']
    require(isinstance(ci, list) and len(ci) == 2, f'Interval schema: {label}')
    require(number(ci[0], label) <= number(ci[1], label), f'Interval ordering: {label}')


def comparisons(records, sources, label):
    contrasts = [(left, right) for left, right in
                 [('B1', 'B0'), ('B2', 'B1'), ('B2', 'B0'),
                  ('B0', 'Train'), ('B1', 'Train'), ('B2', 'Train')]
                 if left in sources and right in sources]
    exact_keys(records, list(sources) + [f'{a}_minus_{b}' for a, b in contrasts], label)
    for key, record in records.items():
        stats(record, f'{label}/{key}')
    for left, right in contrasts:
        close(records[f'{left}_minus_{right}']['mean'], records[left]['mean'] - records[right]['mean'],
              f'{label}/{left}_minus_{right}')


def validate_summary(summary):
    exact_keys(summary, ('status', 'interpretation', 'plan_sha256', 'n_participants',
                        'Test_waveforms_read', 'bootstrap', 'fidelity', 'fingerprint', 'nearest',
                        'original_table_dispersion', 'original_result_reproduction', 'counts', 'limitations'), 'summary')
    require(summary['status'] == 'COMPLETE' and summary['n_participants'] == 198,
            'Summary completion/cohort mismatch')
    require(summary['Test_waveforms_read'] is False, 'Summary reports Test waveform access')
    require('Exploratory post hoc' in summary['interpretation'] and
            'development/selection' in summary['interpretation'], 'Missing exploratory/selection scope')
    require(re.fullmatch('[0-9a-f]{64}', summary['plan_sha256']) is not None, 'Invalid plan hash')
    bootstrap = summary['bootstrap']
    exact_keys(bootstrap, ('seed', 'repetitions', 'unit', 'interval', 'seeds_resampled'), 'bootstrap')
    require(bootstrap['seed'] == 2026101703 and bootstrap['repetitions'] == 5000
            and bootstrap['unit'] == 'participant' and bootstrap['seeds_resampled'] is False
            and 'unadjusted' in bootstrap['interval'], 'Bootstrap settings mismatch')
    counts = {'Train_windows': 47835, 'Validation_windows': 9635,
              'matched_windows_per_source': 9635, 'matched_min': 15, 'matched_max': 25}
    require(summary['counts'] == counts, 'Cohort/window count mismatch')
    finite(summary, 'summary')
    exact_keys(summary['fidelity'], ('full', 'matched'), 'fidelity modes')
    exact_keys(summary['fingerprint'], ('full', 'matched'), 'fingerprint modes')
    for mode in ('full', 'matched'):
        exact_keys(summary['fidelity'][mode], ('versus_Validation', 'versus_Train'), f'{mode} references')
        fingerprint_keys = []
        for reference, other in [('Validation', 'Train'), ('Train', 'Validation')]:
            sources = (other,) + MODELS
            metrics = summary['fidelity'][mode]['versus_' + reference]
            exact_keys(metrics, METRICS, f'{mode}/{reference} metrics')
            for metric, values in metrics.items():
                comparisons(values, sources, f'{mode}/{reference}/{metric}')
            for state in ('EO', 'EC'):
                for source in sources:
                    key = f'{reference}_{state}_{source}'
                    fingerprint_keys.append(key)
                    record = summary['fingerprint'][mode].get(key)
                    exact_keys(record, ('n_queries', 'n_candidates', 'top1_count', 'top5_count',
                                       'top1_percent', 'top5_percent'), f'fingerprint/{mode}/{key}')
                    require(record['n_queries'] == record['n_candidates'] == 198, 'Fingerprint cohort mismatch')
                    require(type(record['top1_count']) is int and type(record['top5_count']) is int
                            and 0 <= record['top1_count'] <= record['top5_count'] <= 198,
                            f'Fingerprint count ordering: {mode}/{key}')
                    for k in (1, 5):
                        close(record[f'top{k}_percent'], 100 * record[f'top{k}_count'] / 198,
                              f'fingerprint/{mode}/{key}/top{k}')
        exact_keys(summary['fingerprint'][mode], fingerprint_keys, f'fingerprint/{mode}')
    exact_keys(summary['nearest'], ('summaries', 'differences_vs_Validation'), 'nearest')
    near = summary['nearest']
    names = [g + suffix for g in MODELS for suffix in ('_all_saved', '_matched_queries')]
    exact_keys(near['summaries'], ['Validation'] + names, 'nearest summaries')
    exact_keys(near['differences_vs_Validation'], names, 'nearest contrasts')
    for key, record in near['summaries'].items():
        stats(record, 'nearest/' + key)
        require(record['mean'] >= 0, 'Negative nearest-window distance')
    for key, record in near['differences_vs_Validation'].items():
        stats(record, 'nearest contrast/' + key)
        close(record['mean'], near['summaries'][key]['mean'] - near['summaries']['Validation']['mean'],
              'nearest contrast/' + key)
    exact_keys(summary['original_table_dispersion'], METRICS[:3], 'original table dispersion')
    for metric, values in summary['original_table_dispersion'].items():
        comparisons(values, MODELS, 'original dispersion/' + metric)
        for model in MODELS:
            close(values[model]['mean'], summary['fidelity']['full']['versus_Train'][metric][model]['mean'],
                  f'original reproduced mean/{metric}/{model}')
    reproduction = summary['original_result_reproduction']
    require(isinstance(reproduction, list) and len(reproduction) == 9, 'Missing reproduction assertions')
    require({(r['metric'], r['source']) for r in reproduction} ==
            {(metric, model) for metric in METRICS[:3] for model in MODELS}, 'Reproduction assertion coverage')
    for row in reproduction:
        exact_keys(row, ('metric', 'source', 'max_abs_error', 'passed'), 'reproduction assertion')
        require(row['passed'] is True and 0 <= number(row['max_abs_error'], 'reproduction error') < 1e-9,
                'Archived original reproduction assertion failed')
    require(isinstance(summary['limitations'], list) and len(summary['limitations']) >= 5, 'Missing limitations')


def verify_files(root, entries, label):
    require(isinstance(entries, list) and entries, f'Empty file inventory: {label}')
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict) and {'path', 'sha256', 'size_bytes'} <= set(entry),
                f'Invalid inventory entry: {label}')
        relative = entry['path']
        require(relative not in seen, f'Duplicate inventory path: {relative}')
        seen.add(relative)
        path = safe_path(root, relative)
        require(type(entry['size_bytes']) is int and path.stat().st_size == entry['size_bytes'],
                f'Size mismatch: {relative}')
        require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None, f'Invalid SHA256: {relative}')
        require(sha256(path) == entry['sha256'], f'SHA256 mismatch: {relative}')
    return seen


def pointer(document, parts):
    value = document
    require(isinstance(parts, list) and parts, 'Invalid reported-results pointer')
    try:
        for part in parts:
            value = value[part]
    except (KeyError, IndexError, TypeError) as exc:
        raise VerificationError(f'Unresolvable reported-results pointer: {parts}') from exc
    return value


def verify(root):
    root = Path(root).resolve()
    inventory = load_json(safe_path(root, BASE + 'publication_inventory.json'))
    require(inventory.get('schema') == 'validation_reference_publication_inventory_v1', 'Inventory schema')
    require(re.fullmatch('[0-9a-f]{40}', inventory.get('baseline_commit', '')) is not None, 'Baseline commit schema')
    packaged = verify_files(root, inventory['files'], 'publication')
    required = {BASE + name for name in ('summary.json', 'completion.json', 'original_source/analyze_round3.py',
                                        'public_metadata.json', 'reported_results.json')}
    required |= {PORTABILITY, 'results/fidelity_data.json', 'results/utility_data.json', 'results/methods_data.json'}
    require(required <= packaged, 'Required evidence is missing from publication inventory')
    summary = load_json(safe_path(root, BASE + 'summary.json'))
    completion = load_json(safe_path(root, BASE + 'completion.json'))
    metadata = load_json(safe_path(root, BASE + 'public_metadata.json'))
    report = load_json(safe_path(root, BASE + 'reported_results.json'))
    fidelity = load_json(safe_path(root, 'results/fidelity_data.json'))
    utility = load_json(safe_path(root, 'results/utility_data.json'))
    methods = load_json(safe_path(root, 'results/methods_data.json'))
    validate_summary(summary)
    exact_keys(completion, ('status', 'completed_utc', 'participants', 'plan_sha256', 'script_sha256',
                           'summary_sha256', 'all_original_reproduction_checks_passed', 'Test_waveforms_read',
                           'new_training', 'new_generation'), 'completion')
    require(completion['status'] == 'COMPLETE' and completion['participants'] == 198
            and completion['all_original_reproduction_checks_passed'] is True, 'Completion status/cohort')
    require(all(completion[k] is False for k in ('Test_waveforms_read', 'new_training', 'new_generation')),
            'Completion execution boundary')
    require(completion['summary_sha256'] == sha256(root / BASE / 'summary.json'), 'Completion summary hash')
    require(completion['script_sha256'] == ORIGINAL_SHA256 == sha256(root / BASE / 'original_source/analyze_round3.py'),
            'Original analysis source hash')
    require(completion['plan_sha256'] == summary['plan_sha256'], 'Summary/completion plan hash')
    require(metadata.get('schema') == 'validation_reference_public_metadata_v1', 'Metadata schema')
    finite(metadata, 'public metadata')
    text = json.dumps(metadata)
    require(not re.search(r'[A-Za-z]:[\\/]|sub-\d+', text), 'Private path or participant identifier in metadata')
    require(metadata['analysis_environment'] == {'python': '3.12.3', 'numpy': '2.5.2', 'scipy': '1.18.1', 'h5py': '3.16.0'},
            'Analysis environment mismatch')
    settings = metadata['analysis_settings']
    require(settings['allowed_real_splits'] == ['Train', 'Validation'] and settings['bootstrap'] == summary['bootstrap'],
            'Extracted frozen settings mismatch')
    require(all(settings[k] is False for k in ('new_training', 'new_generation', 'model_selection')),
            'Extracted analysis execution boundary')
    require(metadata['cohort'] == {'participants': 198, 'states': 2, **summary['counts']}, 'Metadata cohort counts')
    original_plan = [r for r in metadata['source_references'] if r['artifact'].endswith('/analysis_round3/plan.json')]
    require(len(original_plan) == 1 and original_plan[0]['sha256'] == summary['plan_sha256']
            and original_plan[0]['publication_path'] is None, 'External frozen-plan reference mismatch')
    for reference in metadata['source_references']:
        require(re.fullmatch('[0-9a-f]{64}', reference['sha256']) is not None, 'Invalid retained source hash')
        if reference['publication_path'] is not None:
            close_hash = sha256(safe_path(root, reference['publication_path']))
            require(close_hash == reference['sha256'], 'Packaged original source-reference hash mismatch')
    archived = metadata['archived_verification']
    require(archived['status'] == 'PASS' and archived['n_checks'] == len(archived['checks']) == 47
            and all(row['passed'] is True for row in archived['checks']), 'Archived verification assertion schema')
    require(methods['n_subjects'] == fidelity['n_subjects'] == 198, 'Original cohort mismatch')
    require(methods['split_eligible_window_counts'] == {'Train': 47835, 'Validation': 9635, 'Final Test': 19131},
            'Original window counts mismatch')
    require(methods['split_assigned_block_counts'] == {'Train': 1980, 'Validation': 396, 'Test': 792},
            'Original block counts mismatch')
    for metric in METRICS:
        for model in MODELS:
            old = (fidelity['alpha_EC_minus_EO']['channel_mae_db' if metric == 'alpha_effect_mae_db' else 'effect_topography_r'][model]
                   if metric.startswith('alpha_effect') else fidelity['full_pool_metrics'][metric][model])
            current = summary['fidelity']['full']['versus_Train'][metric][model]
            for key in ('mean', 'median'):
                close(current[key], old[key], f'unchanged original Train {metric}/{model}/{key}')
            require(old['n_subjects'] == 198, 'Original fidelity participant count')
    for model in MODELS:
        close(summary['nearest']['summaries'][model + '_all_saved']['mean'],
              fidelity['full_pool_metrics']['nearest_real_rmse_z_median'][model]['mean'],
              'unchanged original nearest/' + model)
        for state in ('EO', 'EC'):
            old = fidelity['fingerprint'][state + '_' + model]
            new = summary['fingerprint']['full']['Train_' + state + '_' + model]
            require(old['n_candidates'] == old['n_valid_queries'] == 198, 'Original fingerprint cohort')
            for k in (1, 5):
                close(new[f'top{k}_percent'], 100 * old[f'top{k}'], 'unchanged original identification')
    require(report.get('schema') == 'validation_reference_reported_results_v1', 'Reported results schema')
    finite(report, 'reported results')
    require(report['model_names'] == {'B0': 'SC', 'B1': 'SSC', 'B2': 'SSC-C'}, 'Model name correspondence')
    expected_categories = {'Table II': 30, 'Table II paired intervals': 9, 'Table III': 28,
                           'Nearest-window narrative': 14, 'Training identification narrative': 6,
                           'Matched identification narrative': 6, 'Validation fidelity narrative': 12}
    require(Counter(r['category'] for r in report['checks']) == expected_categories, 'Reported result coverage')
    require(len({(r['category'], r['label']) for r in report['checks']}) == len(report['checks']), 'Duplicate reported checks')
    documents = {'summary': summary, 'training_fidelity': fidelity}
    for row in report['checks']:
        require(row['document'] in documents and type(row['decimal_places']) is int
                and 0 <= row['decimal_places'] <= 6, 'Reported-check schema')
        value = number(pointer(documents[row['document']], row['pointer']), row['label']) * row['multiplier']
        require(f"{value:.{row['decimal_places']}f}" == row['reported'], 'Reported result differs: ' + row['label'])
    snapshot = report['unchanged_utility_aggregate_snapshot']
    require(snapshot == {k: utility[k] for k in ('summary', 'test_contrasts_left_minus_right')}, 'Original utility changed')
    finite(snapshot, 'utility')
    require(len(utility['summary']) == 33 and len(utility['conditions']) == 11 and len(utility['families']) == 3,
            'Utility condition/family coverage')
    require(sum(r['n_seeds'] for r in utility['summary']) == 55, 'Utility original fit count')
    for row in utility['summary']:
        require(0 <= row['test_mean_percent'] <= 100 and 0 <= row['validation_mean_percent'] <= 100
                and row['test_population_sd_pp'] >= 0 and row['validation_population_sd_pp'] >= 0,
                'Utility percentage/dispersion range')
    for row in utility['test_contrasts_left_minus_right']:
        require(row['ci95_low_pp'] <= row['ci95_high_pp'] and row['n_subjects'] == 198, 'Utility interval/cohort')
    duplication = report['manuscript_only_post_hoc_duplication']
    selected = {r['condition']: r['test_mean_percent'] for r in utility['summary'] if r['family'] == duplication['family']}
    delta = selected[duplication['left']] - selected[duplication['right']]
    require(f'{delta:.3f}' == duplication['reported_delta_pp'] == '0.453', 'Post hoc duplication mean')
    require(duplication['reported_ci95_pp'] == ['-0.071', '0.996'] and 'cannot' in duplication['verification_boundary'],
            'Manuscript-only interval/boundary changed')
    main_ref = next(r for r in metadata['source_references'] if r['artifact'].endswith('/main.tex'))
    require(report['manuscript_source_sha256'] == main_ref['sha256'], 'Reported paper source reference')
    utility_ref = next(r for r in metadata['source_references'] if r['artifact'].endswith('/tables/utility.tex'))
    require(report['utility_table_source_sha256'] == utility_ref['sha256'], 'Reported utility table source reference')
    portability = load_json(safe_path(root, PORTABILITY))
    require(portability.get('schema') == 'validation_portability_v1', 'Portability inventory schema')
    portability_files = verify_files(root, portability['files'], 'portability')
    lineage = portability['source_lineage']
    require(lineage['original_source_path'] == BASE + 'original_source/analyze_round3.py'
            and lineage['original_source_sha256'] == ORIGINAL_SHA256, 'Portable source lineage')
    require(lineage['derivative_path'] in portability_files and lineage['changes'], 'Missing derivative provenance')
    require(sha256(safe_path(root, lineage['derivative_path'])) == lineage['derivative_sha256'], 'Derivative lineage hash')
    return {'status': 'PASS', 'packaged_files': len(packaged), 'portable_files': len(portability_files),
            'participants': 198, 'Train_windows': 47835, 'Validation_windows': 9635,
            'reported_correspondence_checks': len(report['checks']),
            'boundary': 'Saved aggregates and packaged hashes only; no waveform access, independent feature/rank/paired-bootstrap recomputation, or Test evaluation. The post hoc duplication CI is manuscript-only.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Publication checkout (default: parent of this tools directory).')
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except (VerificationError, OSError, KeyError, TypeError, StopIteration) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
