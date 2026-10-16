"""s2_74D: supplementary B0 pool audit and frozen 0.5x view (read-only inputs).

Based on the original, SHA-pinned s2_74 and s2_74C protocols. Does NOT run the
B1/B2 s2_74 script, change any existing artifacts, or read Real/Validation/Test
waveforms. B0's subject_id is an EVALUATION STRATUM ONLY, not conditioning.

Run from repository root after B0 s2_74C FULL_POOL_COMPLETE:
  .\\.venv\\Scripts\\python.exe -u .\\scripts\\data\\s2_74d_b0_pool_integrity_and_freeze.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = 's2_74d_b0_pool_integrity_half_subset_v1'
B0C_SHA = 'db984a91540646e2f36aac6a8fdefe2cf171c21148c5a7bfacf29ca5f6fe831e'
S74_SHA = '749fb1a484deccff4eb41d3fbffd452e987348691fbf814ca304f55efd167fee'
B0_ROOT_REL = Path('outputs/runs/scenario2/ablation/b0_state_only_1x_s2_74c_v1')
FROZEN_REL = Path('outputs/frozen/scenario2/ablation/b0_pool_integrity_half_subset_s2_74d_v1')
RUN_REL = Path('outputs/runs/scenario2/ablation/b0_audit_s2_74d_v1')
EXPECTED_WINDOWS = 47835
EXPECTED_SHARDS = 396
EXPECTED_HALF = 23757


class AuditError(RuntimeError):
    pass


def require(ok, why):
    if not ok:
        raise AuditError(why)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    def no_duplicates(pairs):
        obj = {}
        for k, v in pairs:
            require(k not in obj, f'Duplicate JSON key {k}: {path}')
            obj[k] = v
        return obj
    def no_special(v):
        raise AuditError('Non-finite JSON value: ' + str(v))
    with Path(path).open('r', encoding='utf8') as f:
        return json.load(f, object_pairs_hook=no_duplicates, parse_constant=no_special)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as f:
            f.write(canonical(payload) + b'\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stable_json(path, payload):
    if path.exists():
        require(canonical(load_json(path)) == canonical(payload),
                'Refusing to overwrite changed frozen B0 artifact: ' + str(path))
    else:
        atomic_json(path, payload)


def import_pinned(path, sha, alias):
    require(path.is_file(), 'Missing script: ' + str(path))
    require(sha_file(path) == sha, 'Script SHA mismatch: ' + str(path))
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_existing_file(root, path):
    root = Path(root).resolve()
    path = Path(path)
    require(path.is_file(), 'Missing input file: ' + str(path))
    require(path.resolve().is_relative_to(root), 'Input path outside expected root: ' + str(path))
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink() and not (hasattr(os.path, 'isjunction') and os.path.isjunction(cursor)),
                'Input path includes reparse point/symlink: ' + str(cursor))
        cursor = cursor.parent
    return path


def check_sources(repo, a, c):
    gen = repo / c.S73_OUTPUT_REL
    proto, p_path, p_sha, parent_plan, gp, gp_sha, pairs = a.verify_plan_and_protocol(repo, gen)
    pm_path, pm = a.verify_pool_manifest(gen, gp_sha, pairs)
    fr_path = repo / c.S74_FREEZE_REL
    require(sha_file(fr_path) == c.S74_FREEZE_FILE_SHA256, 'Original s2_74 freeze SHA changed')
    fr = load_json(fr_path)
    require(fr.get('status') == 'S2_74_POOL_INTEGRITY_AND_0P5X_FROZEN', 'Original s2_74 not frozen')
    require(fr.get('implementation_sha256') == S74_SHA, 'Original s2_74 implementation differs')
    require(fr['verified_pool']['all_B1_B2_common_rng_streams_verified'] is True,
            'Original frozen B1/B2 common RNG absent')
    master_record = fr['nested_0p5x']['master']
    master_path = (repo / master_record['path']).resolve()
    require(master_path.is_relative_to(repo) and sha_file(master_path) == master_record['sha256'],
            'Changed s2_74 nested 0.5x master')
    master = load_json(master_path)
    expected_half, half_count = a.build_half_subset(pairs)
    require(half_count == EXPECTED_HALF, 'Unexpected 0.5x sample count')
    require(canonical(master['strata']) == canonical(expected_half), 'Original nested half-subset indices changed')
    for key in ('B1_view', 'B2_view'):
        item = fr['nested_0p5x'][key]
        p = (repo / item['path']).resolve()
        require(p.is_relative_to(repo) and sha_file(p) == item['sha256'], 'Changed frozen ' + key)
    require(sha_file(gp) == c.S73_PLAN_FILE_SHA256 and gp_sha == c.S73_PLAN_CANONICAL_SHA256,
            'Parent generation plan digest differs')
    require(sha_file(pm_path) == c.S73_POOL_MANIFEST_FILE_SHA256,
            'Parent pool manifest digest differs')
    return gen, proto, parent_plan, pairs, pm, master, master_record, expected_half


def audit(repo):
    scripts = repo / 'scripts/data'
    a = import_pinned(scripts / 's2_74_audit_and_freeze_half_subsets.py', S74_SHA, '_s74d_pinned_a')
    c = import_pinned(scripts / 's2_74c_resumable_b0_subject_conditioning_ablation_generation.py',
                      B0C_SHA, '_s74d_pinned_c')
    gen, protocol, parent_plan, pairs, parent_manifest, half_master, master_record, half_rows = check_sources(repo, a, c)
    b0root = repo / B0_ROOT_REL
    plan_path = safe_existing_file(b0root, b0root / 'protocol/s2_74c_b0_generation_plan_v1.json')
    pool_path = safe_existing_file(b0root, b0root / 'manifests/s2_74c_b0_pool_manifest_v1.json')
    plan, pool = load_json(plan_path), load_json(pool_path)
    plan_sha = hashlib.sha256(canonical(plan)).hexdigest()

    require(plan.get('schema') == c.VERSION and plan.get('implementation_sha256') == B0C_SHA,
            'B0 generation plan schema/implementation mismatch')
    require(plan.get('scientific_role') == 'supplementary_subject_conditioning_ablation_only' and
            plan.get('s2_75_protocol_changed') is False, 'Wrong B0 scientific role')
    require(plan.get('Validation_waveforms_accessed') is False and plan.get('Final_Test_waveforms_accessed') is False,
            'Forbidden waveform access in B0 generation plan')
    require(plan.get('expected_windows') == EXPECTED_WINDOWS and plan.get('expected_shards') == EXPECTED_SHARDS,
            'Unexpected B0 plan counts')
    require(canonical(plan.get('pairs')) == canonical(pairs), 'B0/parent subject-state pair plans differ')
    require(plan.get('ordering') == 'subject_index ascending; EO then EC', 'B0 pair ordering changed')
    model = plan['model']
    require(model['name'] == 'B0' and model['frozen_identity'] == 'b0_v_raw_epoch020_v1',
            'Wrong frozen B0 model')
    require(model['conditioning'] == {'eye_state': True, 'subject': False, 'EO': 0, 'EC': 1},
            'Wrong B0 conditioning')
    require(model['subject_id_role'] == 'evaluation_stratum_only_not_model_condition',
            'B0 evaluation-stratum meaning changed')
    sampling = plan['sampling']
    for key, value in {'sampler':'ancestral DDPM','prediction_target':'v','diffusion_steps':1000,
                       'precision':'FP32','batch_size':64,'generation_master_seed':20262201,
                       'generation_seed_formula':'20262201 + 2 * subject_index + eye_state_label',
                       'same_rng_stream_as_B1_B2':True}.items():
        require(sampling.get(key) == value, 'B0 sampling contract changed: ' + key)
    signal = plan['signal']
    for key, value in {'array_key':'x_z','domain':'frozen_train_global_z','channels':19,
                       'samples':512,'sampling_rate_hz':128,'channel_order':c.CHANNELS,
                       'normalization_sha256':protocol['data_contract']['normalization']['sha256']}.items():
        require(signal.get(key) == value, 'B0 signal contract changed: ' + key)
    for key, value in {'s2_73_plan_file_sha256':c.S73_PLAN_FILE_SHA256,
                       's2_73_plan_canonical_sha256':c.S73_PLAN_CANONICAL_SHA256,
                       's2_73_pool_manifest_sha256':c.S73_POOL_MANIFEST_FILE_SHA256,
                       's2_74_freeze_manifest_sha256':c.S74_FREEZE_FILE_SHA256}.items():
        require(plan['parent_frozen_B1_B2_generation'].get(key) == value,
                'B0 parent lineage changed: ' + key)
    for p, expected in ((c.B0_GENERATOR_REL, model['generator_file_sha256']),
                        (c.B0_ARCHIVE_REL, model['archive_checkpoint_sha256']),
                        (c.B0_SELECTION_REL, model['selection_manifest_sha256']),
                        (c.B0_HANDOFF_REL, model['handoff_sha256'])):
        f = safe_existing_file(repo, repo / p)
        require(sha_file(f) == expected, 'Changed frozen B0 artifact: ' + str(p))
    require(pool.get('schema') == 's2_74c_b0_pool_manifest_v1' and
            pool.get('status') == 'S2_74C_B0_FULL_POOL_COMPLETE', 'B0 pool not FULL_POOL_COMPLETE')
    require(pool.get('generation_plan_sha256') == plan_sha, 'B0 pool/plan SHA mismatch')
    require(pool.get('B0', {}).get('shards') == EXPECTED_SHARDS and
            pool.get('B0', {}).get('windows') == EXPECTED_WINDOWS, 'B0 pool counts wrong')
    require(pool.get('subject_condition_used') is False and
            pool.get('subject_id_role') == 'evaluation_stratum_only_not_model_condition',
            'B0 pool falsely implies subject conditioning')
    require(pool.get('same_subject_state_counts_as_B1_B2') is True and
            pool.get('same_generation_seeds_and_rng_streams_as_B1_B2') is True,
            'B0 pool common-RNG provenance changed')
    require(pool.get('Validation_waveforms_accessed') is False and
            pool.get('Final_Test_waveforms_accessed') is False, 'B0 pool violated scientific boundary')

    expected_keys = {f"B0/{p['subject_id']}/{p['eye_state']}" for p in pairs}
    committed = pool.get('committed')
    require(isinstance(committed, dict) and set(committed) == expected_keys,
            'B0 pool committed key set != frozen 396 strata')
    # Reject missing/extraneous or symlinked synthetic files, not just count checks.
    expected_npz = {f"generated/B0/{p['subject_id']}/{p['eye_state']}.npz" for p in pairs}
    expected_receipts = {f"manifests/shards/B0/{p['subject_id']}/{p['eye_state']}.json" for p in pairs}
    present_npz = {p.relative_to(b0root).as_posix() for p in (b0root / 'generated').rglob('*.npz')}
    present_receipts = {p.relative_to(b0root).as_posix() for p in (b0root / 'manifests/shards').rglob('*.json')}
    require(present_npz == expected_npz, 'B0 data files mismatch/missing/extra')
    require(present_receipts == expected_receipts, 'B0 receipt files mismatch/missing/extra')

    ctx = {'b0_model_state_sha':model['model_state_sha256'],
           'b0_generator_file_sha':model['generator_file_sha256'],
           's73_plan_sha':c.S73_PLAN_CANONICAL_SHA256,
           'protocol':protocol}
    shards = []; verified_shas = {}; rec_agg = []
    tracked_stats = {}
    for i, pair in enumerate(pairs, 1):
        sid, st, n = pair['subject_id'], pair['eye_state'], pair['expected_count']
        rel_data = f'generated/B0/{sid}/{st}.npz'
        rel_receipt = f'manifests/shards/B0/{sid}/{st}.json'
        data = safe_existing_file(b0root, b0root / rel_data)
        receipt_file = safe_existing_file(b0root, b0root / rel_receipt)
        entry = committed[f'B0/{sid}/{st}']
        require(entry['file'] == rel_data and entry['expected_count'] == n, 'B0 manifest shard mismatch')
        receipt = load_json(receipt_file)
        require(set(receipt) == {'schema_version','file','sha256','metadata'} and
                receipt['schema_version'] == 1 and receipt['file'] == rel_data,
                'B0 receipt schema/path mismatch: ' + rel_receipt)
        require(isinstance(receipt['sha256'], str) and re.fullmatch('[a-f0-9]{64}', receipt['sha256']),
                'Bad B0 receipt SHA format')
        digest = sha_file(data)
        require(digest == receipt['sha256'] == entry['sha256'], 'B0 shard SHA mismatch: ' + rel_data)
        meta = receipt['metadata']
        c.B0Store.validate_metadata(meta, sid, st)
        for key, value in c.make_metadata(ctx, plan_sha, pair).items():
            require(key in meta and canonical(meta[key]) == canonical(value),
                    'B0 metadata mismatch: ' + rel_data + ':' + key)
        a.validate_rng_trace(meta.get('rng_trace'), pair)
        for g in ('B1', 'B2'):
            refpath = safe_existing_file(gen, gen / f'manifests/shards/{g}/{sid}/{st}.json')
            ref = load_json(refpath)
            refentry = parent_manifest['committed'][f'{g}/{sid}/{st}']
            require(ref.get('file') == refentry['file'] and ref.get('sha256') == refentry['sha256'],
                    'B1/B2 reference receipt vs frozen pool mismatch')
            require(ref['metadata']['generation_seed'] == pair['generation_seed'],
                    'Reference seed changed')
            require(canonical(ref['metadata']['rng_trace']) == canonical(meta['rng_trace']),
                    'B0 vs ' + g + ' RNG trace differs: ' + sid + '/' + st)
            rec_agg.append([refpath.relative_to(repo).as_posix(), sha_file(refpath)])
        with np.load(data, allow_pickle=False) as z:
            require(set(z.files) == {'x_z','sample_indices','metadata_json'},
                    'B0 NPZ member set mismatch: ' + rel_data)
            x = z['x_z']
            require(x.dtype == np.float32 and x.shape == (n, 19, 512) and np.isfinite(x).all(),
                    'B0 NPZ data shape/dtype/finite invalid: ' + rel_data)
            ix = z['sample_indices']
            require(ix.dtype == np.int64 and ix.shape == (n,) and np.array_equal(ix, np.arange(n)),
                    'B0 NPZ indices invalid: ' + rel_data)
            embedded = z['metadata_json']
            require(embedded.shape == () and embedded.dtype.kind == 'U' and
                    embedded.item() == canonical(meta).decode('utf8'),
                    'B0 embedded NPZ metadata mismatch: ' + rel_data)
            stats = {'min':float(x.min()), 'max':float(x.max()),
                     'mean':float(x.mean(dtype=np.float64)), 'std':float(x.std(dtype=np.float64))}
        tracked_stats[data] = (data.stat().st_size, data.stat().st_mtime_ns)
        tracked_stats[receipt_file] = (receipt_file.stat().st_size, receipt_file.stat().st_mtime_ns)
        verified_shas[rel_data] = digest
        shards.append({'subject_id':sid,'state':st,'n':n,'file':rel_data,'sha256':digest,
                       'receipt_sha256':sha_file(receipt_file), 'stats_z':stats})
        if i % 25 == 0 or i == len(pairs):
            print(f'  audited {i:3d}/396 B0 shards', flush=True)
    require(hashlib.sha256(canonical(sorted(rec_agg))).hexdigest() ==
            plan['parent_frozen_B1_B2_generation']['B1_B2_reference_receipts_aggregate_sha256'],
            'B0 plan original reference receipt aggregate changed')
    for p, original in tracked_stats.items():
        require((p.stat().st_size, p.stat().st_mtime_ns) == original,
                'B0 input modified during audit: ' + str(p))
    require(sum(s['n'] for s in shards) == EXPECTED_WINDOWS, 'Audited B0 total count incorrect')

    rows = []
    for row in half_rows:
        rel = f"generated/B0/{row['subject_id']}/{row['eye_state']}.npz"
        rows.append({'subject_id':row['subject_id'],'subject_index':row['subject_index'],
                     'eye_state':row['eye_state'],'eye_state_label':row['eye_state_label'],
                     'source_shard':rel,'source_shard_sha256':verified_shas[rel],
                     'full_count':row['full_count'],'retained_count':row['retained_count'],
                     'retained_pool_indices':row['retained_pool_indices']})
    require(sum(row['retained_count'] for row in rows) == EXPECTED_HALF,
            'B0 nested 0.5x count differs from B1/B2')
    view = {'schema':'s2_74d_B0_nested_0p5x_view_v1','stage':'s2_74d',
            'generator':'B0','source_pool':'s2_74c frozen B0 1x supplementary pool',
            'master_subset_manifest_sha256':master_record['sha256'],
            'array_key':'x_z','signal_domain':'frozen_train_global_z',
            'ratio_label':'0.5x_nested', 'total_retained_windows':EXPECTED_HALF,
            'strata':rows,'Final_Test_accessed':False,'Validation_waveforms_accessed':False}
    freeze_dir = repo / FROZEN_REL
    view_path = freeze_dir / 'manifests/s2_74d_B0_nested_0p5x_view_v1.json'
    stable_json(view_path, view)
    freeze = {'schema':'s2_74d_b0_freeze_manifest_v1','stage':'s2_74d',
              'status':'S2_74D_B0_POOL_AUDITED_AND_0P5X_FROZEN',
              'version':VERSION,'implementation_sha256':sha_file(Path(__file__)),
              'source_artifacts':{
                  's2_74_original_script':{'path':'scripts/data/s2_74_audit_and_freeze_half_subsets.py','sha256':S74_SHA},
                  's2_74c_generator_script':{'path':'scripts/data/s2_74c_resumable_b0_subject_conditioning_ablation_generation.py','sha256':B0C_SHA},
                  'B0_generation_plan':{'path':plan_path.relative_to(repo).as_posix(),'sha256':sha_file(plan_path),'canonical_sha256':plan_sha},
                  'B0_pool_manifest':{'path':pool_path.relative_to(repo).as_posix(),'sha256':sha_file(pool_path)},
                  'B1_B2_0p5x_master':{'path':master_record['path'],'sha256':master_record['sha256']},
                  'B1_B2_s2_74_freeze':{'path':c.S74_FREEZE_REL.as_posix(),'sha256':c.S74_FREEZE_FILE_SHA256},
              },
              'verified_pool':{'B0':{'shards':EXPECTED_SHARDS,'windows':EXPECTED_WINDOWS},
                               'all_file_sha256_verified':True,'all_npz_schema_valid':True,
                               'all_float32':True,'all_finite':True,'all_subject_state_counts_exact':True,
                               'all_B0_B1_B2_common_rng_streams_verified':True,'subject_condition_used':False},
              'nested_0p5x':{'B0_view':{'path':view_path.relative_to(repo).as_posix(),'sha256':sha_file(view_path)},
                             'master_sha256':master_record['sha256'],
                             'retained_windows':EXPECTED_HALF,
                             'identical_indices_as_B1_B2':True},
              'scientific_boundary':{'supplementary_ablation_not_controlled_isolation_of_subject_conditioning':True,
                                     's2_72_original_protocol_unchanged':True,'s2_75_protocol_changed':False,
                                     'generator_changed':False,'no_fidelity_selection':True,
                                     'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False}}
    freeze_path = freeze_dir / 'manifests/s2_74d_b0_freeze_manifest_v1.json'
    stable_json(freeze_path, freeze)
    return freeze_path, view_path, shards


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = pa.parse_args()
    repo = args.repo_root.resolve()
    started = time.perf_counter()
    try:
        require((repo / 'scripts/data').is_dir(), 'Invalid Scenario-2 repo root')
        print('S2_74D — independent B0 full-pool audit + frozen shared 0.5x view', flush=True)
        freeze_path, view_path, shards = audit(repo)
        runid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        report_path = repo / RUN_REL / 'metrics' / f's2_74d_b0_audit_{runid}.json'
        atomic_json(report_path, {'schema':VERSION,'status':'S2_74D_COMPLETE',
                                  'duration_seconds':time.perf_counter()-started,
                                  'shards':len(shards),'windows':sum(r['n'] for r in shards),
                                  'full_shard_audit':shards,
                                  'freeze_manifest':str(freeze_path),'B0_half_view':str(view_path),
                                  'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False})
        print('\nS2_74D_COMPLETE')
        print('B0 audited: 396/396 shards; 47,835/47,835 windows; float32/finite/SHA PASS')
        print('B0 nested 0.5x: 23,757 windows, SAME frozen indices as B1/B2')
        print('B0 subject conditioning used: NO')
        print('Validation waveform accessed: NO; Final Test waveform accessed: NO')
        print('FREEZE MANIFEST:',freeze_path)
        print('B0 0.5X VIEW:',view_path)
        print('AUDIT REPORT:',report_path)
        return 0
    except Exception as exc:
        print('S2_74D_FAILED:',type(exc).__name__+': '+str(exc),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
