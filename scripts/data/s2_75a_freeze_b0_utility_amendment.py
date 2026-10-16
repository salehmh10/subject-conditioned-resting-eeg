r"""s2_75A — freeze a prospective-to-UTILITY 11-condition amendment.

Adds three B0 conditions to the original immutable eight-condition s2_72
protocol.  It does not rewrite s2_72, s2_74, s2_74D, s2_74E, or any EEG.
The amendment is made AFTER descriptive B0 fidelity, BEFORE utility training.
No Validation/Test waveform is read.  Run from project root:

  .\.venv\Scripts\python.exe -u .\scripts\data\s2_75a_freeze_b0_utility_amendment.py

Use --verify-only to verify an existing frozen amendment without writing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

ORIG_PROTO_REL = Path('outputs/protocols/scenario2/utility/b1_vs_b2_utility_protocol_v1/metrics/s2_72_b1_vs_b2_downstream_utility_protocol_v1.json')
ORIG_PROTO_SHA = '5b98ff09d977006b96ea80e43d4f13f33742745b734d5dfa94bb1eaacb682a22'
B12_FREEZE_REL = Path('outputs/frozen/scenario2/utility/b1_b2_pool_integrity_half_subset_v1/manifests/s2_74_freeze_manifest_v1.json')
B0_FREEZE_REL = Path('outputs/frozen/scenario2/ablation/b0_pool_integrity_half_subset_s2_74d_v1/manifests/s2_74d_b0_freeze_manifest_v1.json')
B0_FIDELITY_REL = Path('outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1/metrics/s2_74e_b0_complete_report_v1.json')
B12_FIDELITY_REL = Path('outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1/metrics/s2_74b_complete_report_v1.json')
OUT_REL = Path('outputs/protocols/scenario2/utility/b0_amended_11_condition_v1/manifests/s2_75a_amendment_v1.json')
ORIGINAL_IDS = (
    'REAL_ONLY', 'REAL_DUPLICATE_1X_CONTROL',
    'TSTR_B1_1X', 'TSTR_B2_1X',
    'REAL_PLUS_B1_0P5X', 'REAL_PLUS_B1_1X',
    'REAL_PLUS_B2_0P5X', 'REAL_PLUS_B2_1X',
)
ADDED_IDS = ('TSTR_B0_1X', 'REAL_PLUS_B0_0P5X', 'REAL_PLUS_B0_1X')


class ContractError(RuntimeError):
    pass


def need(ok, why):
    if not ok:
        raise ContractError(why)


def canonical(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def load_json(p):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            need(k not in out, 'Duplicate JSON key: '+k)
            out[k] = v
        return out
    def nonfinite(v):
        raise ContractError('Nonfinite JSON: '+str(v))
    with Path(p).open(encoding='utf-8') as f:
        return json.load(f, object_pairs_hook=unique, parse_constant=nonfinite)


def artifact(repo, rel, expected_sha=None):
    path = (repo / rel).resolve()
    need(path.is_relative_to(repo) and path.is_file() and not path.is_symlink(),
         'Missing or unsafe artifact: '+str(rel))
    digest = sha_file(path)
    if expected_sha is not None:
        need(digest == expected_sha, 'SHA256 mismatch: '+str(rel))
    return path, digest, load_json(path)


def from_record(repo, record):
    need(isinstance(record, dict) and isinstance(record.get('path'), str),
         'Missing frozen source record')
    p, digest, _ = artifact(repo, Path(record['path']), record.get('sha256'))
    return {'path':p.relative_to(repo).as_posix(), 'sha256':digest}


def preflight(repo):
    p, ph, proto = artifact(repo, ORIG_PROTO_REL, ORIG_PROTO_SHA)
    need(proto.get('status') == 'S2_72_B1_VS_B2_UTILITY_PROTOCOL_FROZEN',
         'Original s2_72 not frozen')
    conditions = proto['utility_conditions']
    need([x['id'] for x in conditions] == list(ORIGINAL_IDS), 'Original eight conditions changed')
    need(proto['primary_classifier'] == 'EEGNETV4' and
         proto['primary_metric'] == 'subject-macro balanced accuracy', 'Primary outcome changed')
    need(proto['classifier_families']['EEGNETV4']['training_seeds'] == [20262301, 20262302, 20262303],
         'EEGNet seeds changed')
    need(proto['data_contract']['Train']['usable_windows'] == 47835 and
         proto['data_contract']['Validation']['usable_windows'] == 9635 and
         proto['data_contract']['subjects'] == 198, 'Frozen data counts changed')
    need(proto['final_test_policy']['current_status'] == 'SEALED' and
         not proto['final_test_policy']['may_be_opened_during_validation_utility_training'],
         'Final Test boundary changed')
    for c in proto['primary_comparisons']:
        need(c['left'] in ORIGINAL_IDS and c['right'] in ORIGINAL_IDS, 'Original comparison changed')

    b12p, b12h, b12 = artifact(repo, B12_FREEZE_REL)
    need(b12.get('status') == 'S2_74_POOL_INTEGRITY_AND_0P5X_FROZEN', 's2_74 incomplete')
    need(b12['verified_pool']['B1'] == {'shards':396, 'windows':47835} and
         b12['verified_pool']['B2'] == {'shards':396, 'windows':47835}, 'B1/B2 pool counts changed')
    need(b12['verified_pool']['all_B1_B2_common_rng_streams_verified'] is True,
         'B1/B2 RNG was not audited')
    need(b12['nested_0p5x']['retained_windows_per_generator'] == 23757,
         'B1/B2 nested half count changed')
    need(b12['scientific_boundary']['Final_Test_waveforms_accessed'] is False,
         'Original audit used Final Test')

    b0p, b0h, b0 = artifact(repo, B0_FREEZE_REL)
    need(b0.get('status') == 'S2_74D_B0_POOL_AUDITED_AND_0P5X_FROZEN', 'B0 audit incomplete')
    need(b0['verified_pool']['B0'] == {'shards':396, 'windows':47835}, 'B0 size changed')
    need(b0['verified_pool']['subject_condition_used'] is False, 'B0 must be state-only')
    need(b0['verified_pool']['all_B0_B1_B2_common_rng_streams_verified'] is True,
         'B0 RNG verification absent')
    need(b0['nested_0p5x']['retained_windows'] == 23757 and
         b0['nested_0p5x']['identical_indices_as_B1_B2'] is True,
         'B0 0.5x indices not shared')
    need(b0['scientific_boundary']['Final_Test_waveforms_accessed'] is False,
         'B0 audit used Final Test')

    # The original and B0 half-subset views are verified byte-for-byte against
    # their immutable freeze records.  This is a metadata-only check.
    source = {
        's2_72_original_protocol': {'path':p.relative_to(repo).as_posix(), 'sha256':ph},
        's2_74_original_B1_B2_freeze': {'path':b12p.relative_to(repo).as_posix(), 'sha256':b12h},
        's2_74d_B0_freeze': {'path':b0p.relative_to(repo).as_posix(), 'sha256':b0h},
    }
    for label, x in (
        ('original_half_master', b12['nested_0p5x']['master']),
        ('B1_half_view', b12['nested_0p5x']['B1_view']),
        ('B2_half_view', b12['nested_0p5x']['B2_view']),
        ('B0_half_view', b0['nested_0p5x']['B0_view']),
    ):
        source[label] = from_record(repo, x)
    master = artifact(repo, Path(source['original_half_master']['path']))[2]
    views = {g:artifact(repo, Path(source[f'{g}_half_view']['path']))[2]
             for g in ('B0','B1','B2')}
    need(len(master['strata']) == 396 and
         sum(row['retained_count'] for row in master['strata']) == 23757,
         'Frozen half master malformed')
    for g, view in views.items():
        need(view['total_retained_windows'] == 23757 and len(view['strata']) == 396,
             'Half-view malformed: '+g)
        for r, x in zip(master['strata'], view['strata']):
            need((r['subject_id'],r['eye_state']) == (x['subject_id'],x['eye_state']) and
                 r['retained_pool_indices'] == x['retained_pool_indices'],
                 'Half-view differs from original master: '+g)

    _, baseline_sha, old_fidelity = artifact(repo, B12_FIDELITY_REL)
    need(old_fidelity.get('status') == 'S2_74B_COMPLETE' and old_fidelity.get('subjects') == 198,
         'Original 198-subject fidelity incomplete')
    _, b0_fid_sha, b0_fidelity = artifact(repo, B0_FIDELITY_REL)
    need(b0_fidelity.get('status') == 'S2_74E_COMPLETE' and b0_fidelity.get('subjects') == 198,
         'B0 fidelity incomplete')
    need(b0_fidelity['figures']['status'] == 'CREATED' and len(b0_fidelity['figures']['files']) == 11,
         'Four-way fidelity charts incomplete')
    for report in (old_fidelity, b0_fidelity):
        need(report.get('Validation_waveforms_accessed') is False and
             report.get('Final_Test_waveforms_accessed') is False,
             'Fidelity boundaries violated')
    source['s2_74b_original_fidelity'] = {'path':B12_FIDELITY_REL.as_posix(), 'sha256':baseline_sha}
    source['s2_74e_B0_fidelity'] = {'path':B0_FIDELITY_REL.as_posix(), 'sha256':b0_fid_sha}
    return proto, source


def make_amendment(proto, sources):
    # New entries are appended, never interleaved into or substituted for the
    # original s2_72 condition list.
    additions = [
        {'id':'TSTR_B0_1X', 'short_name':'TSTR-B0',
         'role':'supplementary state-only synthetic-only utility',
         'train_sources':['B0 synthetic 1x'], 'synthetic_generator':'B0', 'synthetic_ratio':1.0},
        {'id':'REAL_PLUS_B0_0P5X', 'short_name':'Real+B0 0.5x',
         'role':'supplementary B0 augmentation dose',
         'train_sources':['Real Train', 'nested B0 synthetic 0.5x'],
         'synthetic_generator':'B0', 'synthetic_ratio':0.5},
        {'id':'REAL_PLUS_B0_1X', 'short_name':'Real+B0 1x',
         'role':'supplementary B0 full augmentation',
         'train_sources':['Real Train', 'B0 synthetic 1x'],
         'synthetic_generator':'B0', 'synthetic_ratio':1.0},
    ]
    ids = [x['id'] for x in proto['utility_conditions']] + [x['id'] for x in additions]
    need(len(ids) == len(set(ids)) == 11 and tuple(ids[8:]) == ADDED_IDS,
         'Condition matrix invalid')
    return {
        'schema':'s2_75a_utility_amendment_v1',
        'status':'S2_75A_11_CONDITION_AMENDMENT_FROZEN',
        'amendment_implementation_sha256':sha_file(Path(__file__)),
        'original_protocol_remains_immutable':True,
        'timing':'Supplementary B0 amendment AFTER descriptive fidelity, BEFORE utility/Validation utility training.',
        'primary_s2_72_comparisons_preserved_unmodified':proto['primary_comparisons'],
        'primary_classifier':proto['primary_classifier'],
        'primary_metric':proto['primary_metric'],
        'source_artifacts':sources,
        'original_conditions':proto['utility_conditions'],
        'new_conditions':additions,
        'utility_conditions':proto['utility_conditions'] + additions,
        'secondary_supplementary_comparisons':[
            {'id':'S1_TSTR_B0_VS_B1','left':'TSTR_B0_1X','right':'TSTR_B1_1X'},
            {'id':'S2_TSTR_B0_VS_B2','left':'TSTR_B0_1X','right':'TSTR_B2_1X'},
            {'id':'S3_AUG_B0_1X_VS_REAL','left':'REAL_PLUS_B0_1X','right':'REAL_ONLY'},
            {'id':'S4_AUG_B0_1X_VS_B1','left':'REAL_PLUS_B0_1X','right':'REAL_PLUS_B1_1X'},
            {'id':'S5_AUG_B0_1X_VS_B2','left':'REAL_PLUS_B0_1X','right':'REAL_PLUS_B2_1X'},
            {'id':'S6_AUG_B0_0P5X_VS_REAL','left':'REAL_PLUS_B0_0P5X','right':'REAL_ONLY'},
        ],
        'classifier_families':proto['classifier_families'],
        'normalization_and_splits':'Identical locked Real Train/Validation split and Train-only channel scaler.',
        'training':{'EEGNet_fixed_optimizer_updates':24*374,
                    'calculation':'24 * ceil(47835 / frozen_baseline_batch_size_128)',
                    'no_validation_early_stopping':True,
                    'no_validation_checkpoint_selection':True,
                    'no_hyperparameter_search':True,
                    'synthetic_training_only':True},
        'statistics':proto['statistics'],
        'Final_Test_access':'SEALED, no use in s2_75',
        'no_pruning_based_on_Validation':True,
        'no_clean_causal_subject_conditioning_isolation_claim':True,
        'B0_subject_id_use':'Matching/evaluation stratum ONLY; never a B0 generation input or classifier feature.',
        'Validation_waveforms_accessed':False,
        'Final_Test_waveforms_accessed':False,
    }


def freeze(repo, verify_only=False):
    proto, sources = preflight(repo)
    payload = make_amendment(proto, sources)
    path = repo / OUT_REL
    if path.exists():
        need(not path.is_symlink() and canonical(load_json(path)) == canonical(payload),
             'Existing amendment differs: refusing to overwrite '+str(path))
        print('S2_75A_REUSED_IDENTICAL_FROZEN_AMENDMENT')
    else:
        need(not verify_only, 'No frozen amendment exists; omit --verify-only to create it')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('xb') as f:
                f.write(canonical(payload)+b'\n'); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists(): tmp.unlink()
        print('S2_75A_NEW_AMENDMENT_FROZEN')
    print('S2_75A_11_CONDITION_AMENDMENT_FROZEN')
    print('Original conditions: 8; added B0: 3; total: 11')
    print('EEGNet fixed updates per fit: 8976; seeds: 20262301/02/03')
    print('No classifier trained. Validation waveform accessed: NO; Final Test accessed: NO')
    print('AMENDMENT:',path)
    print('SHA256:',sha_file(path))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    try:
        repo=args.repo_root.resolve()
        need((repo/'scripts/data').is_dir(), 'Not the Scenario-2 repo')
        freeze(repo,args.verify_only)
        return 0
    except Exception as e:
        print('S2_75A_FAILED:',type(e).__name__+': '+str(e),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())