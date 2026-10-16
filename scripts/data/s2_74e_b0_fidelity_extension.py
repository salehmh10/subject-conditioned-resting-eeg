"""s2_74E: append read-only B0 descriptive fidelity to the frozen s2_74B study.

Reuses the EXACT SHA-pinned s2_74B feature, PSD/Welch, SD/MD, SWD, Hjorth,
nonoverlapping Vinck debiased squared-wPLI, Real-Real calibration, bootstrap,
and fingerprint implementations. The original s2_74B analysis and B1/B2
per-subject results are read-only and remain unchanged. B0 is evaluated at
exactly the same Real Train subject/state strata and index selections.

Preconditions:
- s2_74D complete, independent B0 audit and frozen 0.5x view.
- s2_74B completed 198-subject baseline output exists.
- Real Train is still readable at the ORIGINAL frozen paths.

Run from repo root (CPU-only; cumulative subject targets):
  .\\.venv\\Scripts\\python.exe -u .\\scripts\\data\\s2_74e_b0_fidelity_extension.py --stop-after-subjects 2
  .\\.venv\\Scripts\\python.exe -u .\\scripts\\data\\s2_74e_b0_fidelity_extension.py --stop-after-subjects 198

NOTE: B0 is state-only, subject IDs are EVALUATION STRATA, and differences from
B1/B2 are descriptive, not a clean controlled causal effect of conditioning.
No Validation or Final Test waveforms are read. No model is selected or altered.
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = 's2_74e_b0_descriptive_fidelity_extension_v1'
B_SHA = '09dcaa49c74f7ab055f0932054f0cc97ec346a71558dcdc8b036c6d52536e8ed'
D_REL = Path('scripts/data/s2_74d_b0_pool_integrity_and_freeze.py')
B0_C_REL = Path('outputs/runs/scenario2/ablation/b0_state_only_1x_s2_74c_v1')
D_FROZEN_REL = Path('outputs/frozen/scenario2/ablation/b0_pool_integrity_half_subset_s2_74d_v1')
B_BASELINE_REL = Path('outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1')
OUT_REL = Path('outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1')
FC_METRICS = ('upper_mae','upper_r','relative_frobenius','node_mean_mae')
GROUPS = ('B0','B1','B2')
STATES = ('EO','EC')


class FidelityError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise FidelityError(message)


def pinned_import(path, digest, alias):
    require(path.is_file(), 'Missing required script: ' + str(path))
    require(sha_file(path) == digest, 'Source script changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(x)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def safe_repo_file(repo, relative):
    p = (repo / relative).resolve()
    require(p.is_relative_to(repo) and p.is_file(), 'Missing/out-of-repo artifact: ' + str(relative))
    return p


def frozen_preflight(repo, b, d):
    a, gen, proto, plan, pairs, scaler, grouped, tracked, parent_pool = b.prepare(repo)
    baseline_root = repo / B_BASELINE_REL
    apath = baseline_root / 'protocol/s2_74b_analysis_plan_v1.json'
    cp = baseline_root / 'metrics/s2_74b_complete_report_v1.json'
    baseline_plan = b.load_json(apath)
    baseline_summary = b.load_json(cp)
    require(baseline_plan['implementation_sha256'] == B_SHA, 'Original s2_74B implementation differs')
    require(canonical(baseline_plan['settings']) == canonical(b.settings()),
            'Original s2_74B settings differ from pinned implementation')
    require(baseline_plan['generation_plan_canonical_sha256'] == b.sha_obj(plan),
            'Original s2_74B generation plan differs')
    require(baseline_plan['train_index_sha256'] == plan['core']['train_window_index_sha256'],
            'Original s2_74B Train index differs')
    require(baseline_summary.get('status') == 'S2_74B_COMPLETE' and baseline_summary.get('subjects') == 198,
            'Original 198-subject s2_74B baseline is not complete')
    require(baseline_summary.get('Validation_waveforms_accessed') is False and
            baseline_summary.get('Final_Test_waveforms_accessed') is False,
            'Forbidden original baseline boundary status')
    for rel, digest in baseline_plan['inputs'].items():
        p = safe_repo_file(repo, rel)
        require(sha_file(p) == digest, 'Original baseline pinned artifact differs: ' + rel)
    subjects = plan['core']['subject_ids_in_index_order']
    require(len(subjects) == 198, 'Expected 198 subjects')
    require(set(baseline_plan['selections']) == {sid+'/'+st for sid in subjects for st in STATES},
            'Original selection keys differ')
    for pair in pairs:
        sid, st = pair['subject_id'], pair['eye_state']
        sel = b.select_indices(grouped[(sid,st)], b.ANALYSIS_SEED+1000+2*pair['subject_index']+pair['eye_state_label'])
        require(canonical(sel) == canonical(baseline_plan['selections'][sid+'/'+st]),
                'Real/synthetic index selections no longer reproduce for ' + sid+'/'+st)

    freeze_path = repo / D_FROZEN_REL / 'manifests/s2_74d_b0_freeze_manifest_v1.json'
    freeze = b.load_json(freeze_path)
    require(freeze.get('status') == 'S2_74D_B0_POOL_AUDITED_AND_0P5X_FROZEN',
            's2_74D independent B0 audit/freeze missing')
    require(freeze.get('implementation_sha256') == sha_file(repo / D_REL),
            's2_74D frozen implementation digest differs')
    require(freeze['verified_pool']['B0'] == {'shards':396,'windows':47835} and
            freeze['verified_pool']['all_B0_B1_B2_common_rng_streams_verified'] is True and
            freeze['verified_pool']['subject_condition_used'] is False,
            's2_74D B0 audit contract incorrect')
    require(freeze['scientific_boundary']['Validation_waveforms_accessed'] is False and
            freeze['scientific_boundary']['Final_Test_waveforms_accessed'] is False,
            's2_74D boundary violated')
    sources = freeze['source_artifacts']
    for name, item in sources.items():
        src = safe_repo_file(repo, item['path'])
        require(sha_file(src) == item['sha256'], 'B0 frozen input changed: '+name)
    view_record = freeze['nested_0p5x']['B0_view']
    viewfile = safe_repo_file(repo, view_record['path'])
    require(sha_file(viewfile) == view_record['sha256'], 'B0 half-view changed')
    view = b.load_json(viewfile)
    require(view['total_retained_windows'] == 23757 and view['generator'] == 'B0',
            'B0 frozen half-view count wrong')
    pool_path = repo / B0_C_REL / 'manifests/s2_74c_b0_pool_manifest_v1.json'
    pool = b.load_json(pool_path)
    require(pool['status'] == 'S2_74C_B0_FULL_POOL_COMPLETE' and
            pool['B0']['shards'] == 396 and pool['B0']['windows'] == 47835,
            'B0 pool no longer complete')
    require(len(pool['committed']) == 396, 'B0 manifest has wrong shard count')

    # Pre-existing baseline subject outputs are authenticated once, before use.
    baselines = []
    baseline_receipts = {}
    for si, sid in enumerate(subjects):
        path = baseline_root / 'subjects' / f'{sid}.json'
        receipt = path.with_suffix('.sha256')
        require(path.is_file() and receipt.is_file(), 'Missing original s2_74B subject: ' + sid)
        digest = sha_file(path)
        require(receipt.read_text(encoding='utf8').strip() == digest, 'Original s2_74B cached subject corrupt: ' + sid)
        result = b.load_json(path)
        require(result['subject_id'] == sid and result['subject_index'] == si and
                result['analysis_plan_sha256'] == sha_file(apath), 'Original s2_74B cached subject lineage mismatch: ' + sid)
        require(set(result['states']) == {'EO','EC'}, 'Original subject state keys differ: ' + sid)
        baselines.append(result)
        baseline_receipts[sid] = digest

    tracked.update({apath:sha_file(apath),cp:sha_file(cp),freeze_path:sha_file(freeze_path),
                    viewfile:sha_file(viewfile),pool_path:sha_file(pool_path),
                    repo / D_REL:sha_file(repo / D_REL),
                    repo / 'scripts/data/s2_74b_final_pool_fidelity.py':B_SHA})
    return {'b':b,'a':a,'gen':gen,'proto':proto,'plan':plan,'pairs':pairs,
            'scaler':scaler,'grouped':grouped,'baseline_plan':baseline_plan,
            'baseline_summary':baseline_summary,'baselines':baselines,
            'baseline_receipts':baseline_receipts,'apath':apath,
            'freeze':freeze,'freeze_path':freeze_path,'view_path':viewfile,'pool':pool,
            'tracked':tracked,'subjects':subjects}


def b0_subject_state(ctx, sid, st, si, projections, baseline, do_analysis=True):
    b = ctx['b']
    pair = ctx['pairs'][2*si+(st == 'EC')]
    require(pair['subject_id'] == sid and pair['eye_state'] == st, 'Unexpected pair ordering')
    xr = b.read_real(ctx['grouped'][(sid,st)],ctx['scaler'])
    real_sha = b.sha_array(xr)
    require(real_sha == baseline['input_hashes'][st]['real_train_z_sha256'],
            'Real Train waveform hash differs from frozen s2_74B analysis: ' + sid+'/'+st)
    row = ctx['pool']['committed'][f'B0/{sid}/{st}']
    b0file = ctx['repo'] / B0_C_REL / row['file']
    require(b0file.is_file(), 'Missing B0 shard: ' + str(b0file))
    require(row['expected_count'] == len(xr), 'B0 vs Real count mismatch: ' + sid+'/'+st)
    digest = sha_file(b0file)
    require(digest == row['sha256'], 'B0 shard changed after s2_74D: ' + sid+'/'+st)
    receipt_path = ctx['repo'] / B0_C_REL / f'manifests/shards/B0/{sid}/{st}.json'
    receipt = b.load_json(receipt_path)
    require(receipt['sha256'] == digest and receipt['file'] == row['file'] and
            receipt['metadata']['subject_condition_used'] is False,
            'B0 receipt changed/conditioning incorrect: ' + sid+'/'+st)
    inputs = {'real_train_z_sha256':real_sha,
              'B0':{'file_sha256':digest,'receipt_sha256':sha_file(receipt_path)}}
    if not do_analysis:
        # A resumed run still verifies original Real Train and B0 input hashes,
        # but avoids recomputing all expensive features and connectivity.
        return None, inputs
    with np.load(b0file,allow_pickle=False) as z:
        x0 = z['x_z']
    require(x0.dtype == np.float32 and x0.shape == xr.shape and np.isfinite(x0).all(),
            'B0 array schema/finite differs: ' + sid+'/'+st)
    require(sha_file(b0file) == digest, 'B0 shard changed during analysis: ' + sid+'/'+st)
    sel = ctx['baseline_plan']['selections'][sid+'/'+st]
    fr = b.features(xr,ctx['scaler'],projections)
    f0 = b.features(x0,ctx['scaler'],projections)
    full, details = b.compare(fr,f0)
    fc, mats, calibrated_fc = {}, {}, {}
    r_ix, z_ix = sel['nonoverlap_real'],sel['nonoverlap_synthetic']
    ra = b.subfeat(fr,sel['rr_real_A'])
    b0_cal = b.subfeat(f0,sel['rr_synthetic'])
    calibration = b.compare(ra,b0_cal)[0]
    for band in b.FC_BANDS:
        rm, rc = b.dwpli2(fr['fft'][r_ix],band)
        sm, sc = b.dwpli2(f0['fft'][z_ix],band)
        mats[band] = {'B0': {'matrix':sm, 'valid_frequency_bins':sc}}
        fc[band] = {'B0':b.fc_compare(rm,sm)}
        rm_a = b.dwpli2(ra['fft'],band)[0]
        sm_a = b.dwpli2(b0_cal['fft'],band)[0]
        calibrated_fc[band] = {'B0':b.fc_compare(rm_a,sm_a)}
    entry = {'full_pool_metrics':{'B0':full},
             'descriptor_distances':{'B0':details},
             'source_summaries':{'B0':b.summary_feature(f0)},
             'nonoverlap_connectivity':fc,
             'nonoverlap_connectivity_matrices':mats,
             'matched_train_block_calibration':{
                 'n_per_group':sel['rr_n'],
                 'feature_metrics':{'B0':calibration},'connectivity':calibrated_fc},
             'sample_counts':{'Real':len(xr),'B0':len(x0)},
             'connectivity_windows_per_group':len(r_ix)}
    original = baseline['states'][st]
    require(original['sample_counts']['Real'] == len(xr) and
            original['connectivity_windows_per_group'] == len(r_ix) and
            original['matched_train_block_calibration']['n_per_group'] == sel['rr_n'],
            'Baseline vs B0 sampling/calibration mismatch: '+sid+'/'+st)
    return entry, inputs


def metric_array(b0results, baselines, extractor, generator):
    out = []
    for r0, rb in zip(b0results,baselines):
        vals = [extractor(r0, rb, st, generator) for st in STATES]
        out.append(np.mean([np.nan if v is None else float(v) for v in vals]))
    return np.asarray(out, dtype=float)


def paired_summary(b, b0results, baselines, extractor, existing=None, keys=('B0','B1','B2')):
    vals = {g:metric_array(b0results,baselines,extractor,g) for g in keys}
    summary = {g:(existing[g] if existing is not None and g in existing else b.boot_mean(vals[g]))
               for g in keys}
    for a,c in (('B0','B1'),('B0','B2')):
        summary[a+'_minus_'+c] = b.boot_mean(vals[a]-vals[c])
    if existing is None:
        summary['B2_minus_B1'] = b.boot_mean(vals['B2']-vals['B1'])
    elif 'B2_minus_B1' in existing:
        summary['B2_minus_B1'] = existing['B2_minus_B1']
    return summary


def access_state(r0, rb, st, name):
    return r0['states'][st] if name == 'B0' else rb['states'][st]


def merged_state(r0, rb, st, key, g, extra=None):
    src = access_state(r0,rb,st,g)
    if key == 'source_summaries':
        return src[key][g]
    if key == 'full_pool_metrics':
        return src[key][g]
    if key == 'nonoverlap_connectivity':
        return src[key][extra][g]
    if key == 'nonoverlap_connectivity_matrices':
        return src[key][extra][g]['matrix']
    if key == 'calibration':
        return src['matched_train_block_calibration']['feature_metrics'][g]
    if key == 'calibration_fc':
        return src['matched_train_block_calibration']['connectivity'][extra][g]
    raise KeyError(key)


def alpha_effect(b, r0, rb, g):
    def values(st, who):
        return np.asarray(merged_state(r0,rb,st,'source_summaries',who)['median_bandpower_db'],dtype=float)[2]
    real = values('EC','Real')-values('EO','Real')
    syn = values('EC',g)-values('EO',g)
    return {'channel_mae_db':float(np.abs(syn-real).mean()),
            'posterior_O1_O2_error_db':float(np.abs(syn[17:19].mean()-real[17:19].mean())),
            'real_EC_minus_EO_O1_O2_db':float(real[17:19].mean()),
            'synthetic_EC_minus_EO_O1_O2_db':float(syn[17:19].mean()),
            'effect_topography_r':b.correlation(real,syn)}


def fc_state_effect(b,r0,rb,g,band):
    real = np.asarray(merged_state(r0,rb,'EC','nonoverlap_connectivity_matrices','Real',band)) - \
           np.asarray(merged_state(r0,rb,'EO','nonoverlap_connectivity_matrices','Real',band))
    syn = np.asarray(merged_state(r0,rb,'EC','nonoverlap_connectivity_matrices',g,band)) - \
          np.asarray(merged_state(r0,rb,'EO','nonoverlap_connectivity_matrices',g,band))
    valid=np.isfinite(real[b.UT]) & np.isfinite(syn[b.UT])
    rr=real[b.UT][valid]; ss=syn[b.UT][valid]
    return {'EC_minus_EO_upper_mae':b.finite_mean(np.abs(ss-rr)),
            'EC_minus_EO_upper_r':b.correlation(ss,rr),
            'valid_edges':int(valid.sum()),
            'real_mean_EC_minus_EO':b.finite_mean(rr),
            'synthetic_mean_EC_minus_EO':b.finite_mean(ss)}


def fingerprint(b, r0s, rbs, st, g):
    real=np.stack([np.asarray(merged_state(r0,rb,st,'source_summaries','Real')['median_bandpower_db']).ravel()
                   for r0,rb in zip(r0s,rbs)])
    syn=np.stack([np.asarray(merged_state(r0,rb,st,'source_summaries',g)['median_bandpower_db']).ravel()
                  for r0,rb in zip(r0s,rbs)])
    sims=np.array([[b.correlation(a,ref) for ref in real] for a in syn])
    valid=np.isfinite(sims).all(axis=1)
    ranks=[]; margins=[]; rows=[]
    for i in range(len(r0s)):
        if not valid[i]: continue
        other=np.delete(sims[i],i)
        rank=1+int(np.sum(other>=sims[i,i]))
        margin=float(sims[i,i]-other.max()) if len(other) else np.nan
        ranks.append(rank); margins.append(margin)
        rows.append({'subject_id':r0s[i]['subject_id'],'state':st,'generator':g,
                     'correct_subject_r':sims[i,i],'correct_vs_best_wrong_margin':margin,
                     'rank':rank,'fingerprint_rmse_db':float(np.sqrt(np.mean((syn[i]-real[i])**2)))})
    return {'n_candidates':len(r0s),'n_valid_queries':len(ranks),
            'top1':float(np.mean(np.array(ranks)<=1)) if ranks else None,
            'top5':float(np.mean(np.array(ranks)<=min(5,len(r0s)))) if ranks else None,
            'margin':b.boot_mean(margins),
            'note':'EEG log-bandpower adaptation, in-sample; B0 subject IDs label evaluation strata ONLY; not privacy or unseen-subject generalization.'},rows


def aggregate(ctx, results, out):
    b = ctx['b']; base=ctx['baselines']; origin=ctx['baseline_summary']
    require(len(results)==len(base)==198,'Cannot aggregate without all 198 participants')
    output={'schema':VERSION,'status':'S2_74E_COMPLETE','subjects':198,
            'baseline_s2_74B_complete_sha256':sha_file(ctx['repo']/B_BASELINE_REL/'metrics/s2_74b_complete_report_v1.json'),
            'fidelity_method':'Exact SHA-pinned s2_74B computations, original frozen subject/state selections and Real Train hashes.',
            'scientific_interpretation':'Supplementary descriptive B0 state-only reference; B0 vs B1/B2 not a clean isolation of subject conditioning because training histories differ.',
            'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False,
            'generator_changed':False,'synthetic_pool_filtered':False,
            'original_s2_72_utility_protocol_changed':False}
    metrics = {}
    keys=list(origin['full_pool_metrics'])
    for key in keys:
        require(all(key in r['states']['EO']['full_pool_metrics']['B0'] for r in results),
                'New B0 metric key missing: '+key)
        extractor = lambda r0,rb,st,g,k=key: merged_state(r0,rb,st,'full_pool_metrics',g)[k]
        metrics[key]=paired_summary(b,results,base,extractor,origin['full_pool_metrics'][key])
    output['full_pool_metrics']=metrics

    # Equal-subject weight, within-subject EC-minus-EO spectral effect.
    alpha_rows=[]; alpha_by={g:[] for g in GROUPS}
    for r0,rb in zip(results,base):
        for g in GROUPS:
            met=alpha_effect(b,r0,rb,g); alpha_by[g].append(met)
            alpha_rows.append({'subject_id':r0['subject_id'],'generator':g,**met})
    alpha_sum={}
    for key in ('channel_mae_db','posterior_O1_O2_error_db','effect_topography_r'):
        arr={g:np.asarray([np.nan if x[key] is None else x[key] for x in alpha_by[g]],dtype=float)
             for g in GROUPS}
        old=origin['alpha_EC_minus_EO'][key]
        alpha_sum[key]={**old,'B0':b.boot_mean(arr['B0']),
                        'B0_minus_B1':b.boot_mean(arr['B0']-arr['B1']),
                        'B0_minus_B2':b.boot_mean(arr['B0']-arr['B2'])}
    output['alpha_EC_minus_EO']=alpha_sum
    b.csv_write(out/'tables/subject_alpha_effect.csv',alpha_rows)

    conn={}; conrows=[]
    for band in b.FC_BANDS:
        conn[band]={}
        for key in FC_METRICS:
            extractor=lambda r0,rb,st,g,bd=band,k=key: merged_state(r0,rb,st,'nonoverlap_connectivity',g,bd)[k]
            conn[band][key]=paired_summary(b,results,base,extractor,origin['connectivity'][band][key])
    for r0,rb in zip(results,base):
        for st in STATES:
            for band in b.FC_BANDS:
                for g in GROUPS:
                    conrows.append({'subject_id':r0['subject_id'],'state':st,'band':band,
                                    'generator':g,'n_windows_per_group':r0['states'][st]['connectivity_windows_per_group'],
                                    **merged_state(r0,rb,st,'nonoverlap_connectivity',g,band)})
    b.csv_write(out/'tables/subject_connectivity.csv',conrows)
    output['connectivity']=conn

    calibration={}; calibr_rows=[]
    for key in origin['matched_train_block_calibration_summary']:
        extractor = lambda r0,rb,st,g,k=key: merged_state(r0,rb,st,'calibration',g)[k]
        arr={g:metric_array(results,base,extractor,g) for g in GROUPS}
        rr=np.array([np.mean([b.scalar(rb['states'][st]['matched_train_block_calibration']['feature_metrics']['Real_B'][key]) for st in STATES])
                     for rb in base])
        calibration[key]={**origin['matched_train_block_calibration_summary'][key],
                          'B0':b.boot_mean(arr['B0']),
                          'B0_minus_RealReal':b.boot_mean(arr['B0']-rr)}
    for r0,rb in zip(results,base):
        for st in STATES:
            n=rb['states'][st]['matched_train_block_calibration']['n_per_group']
            for g in ('Real_B',)+GROUPS:
                met=(rb['states'][st]['matched_train_block_calibration']['feature_metrics'][g] if g=='Real_B'
                     else merged_state(r0,rb,st,'calibration',g))
                calibr_rows.append({'subject_id':r0['subject_id'],'state':st,
                                   'comparison':'Real_A_vs_'+g,'n_per_group':n,**met})
    b.csv_write(out/'tables/matched_real_real_calibration.csv',calibr_rows)
    output['matched_train_block_calibration_summary']=calibration

    calfc={}; calfc_rows=[]
    for band in b.FC_BANDS:
        calfc[band]={}
        for key in FC_METRICS:
            extractor=lambda r0,rb,st,g,bd=band,k=key: merged_state(r0,rb,st,'calibration_fc',g,bd)[k]
            arr=metric_array(results,base,extractor,'B0')
            rr=np.asarray([np.mean([b.scalar(rb['states'][st]['matched_train_block_calibration']['connectivity'][band]['Real_B'][key])
                                    for st in STATES]) for rb in base])
            calfc[band][key]={**origin['matched_train_block_connectivity_summary'][band][key],
                              'B0':b.boot_mean(arr),
                              'B0_minus_RealReal':b.boot_mean(arr-rr)}
    for r0,rb in zip(results,base):
        for st in STATES:
            for band in b.FC_BANDS:
                n=rb['states'][st]['matched_train_block_calibration']['n_per_group']
                for g in ('Real_B',)+GROUPS:
                    met=rb['states'][st]['matched_train_block_calibration']['connectivity'][band][g] if g=='Real_B' \
                        else merged_state(r0,rb,st,'calibration_fc',g,band)
                    calfc_rows.append({'subject_id':r0['subject_id'],'state':st,'band':band,
                                       'comparison':'Real_A_vs_'+g,'n_per_group':n,**met})
    b.csv_write(out/'tables/matched_real_real_connectivity.csv',calfc_rows)
    output['matched_train_block_connectivity_summary']=calfc

    fc_effect={}; fc_effect_rows=[]
    for band in b.FC_BANDS:
        mets={g:[] for g in GROUPS}
        for r0,rb in zip(results,base):
            for g in GROUPS:
                one=fc_state_effect(b,r0,rb,g,band)
                mets[g].append(one)
                fc_effect_rows.append({'subject_id':r0['subject_id'],'band':band,'generator':g,**one})
        fc_effect[band]={}
        for key in ('EC_minus_EO_upper_mae','EC_minus_EO_upper_r'):
            ar={g:np.asarray([x[key] for x in mets[g]],dtype=float) for g in GROUPS}
            fc_effect[band][key]={**origin['connectivity_EC_minus_EO'][band][key],
                                  'B0':b.boot_mean(ar['B0']),
                                  'B0_minus_B1':b.boot_mean(ar['B0']-ar['B1']),
                                  'B0_minus_B2':b.boot_mean(ar['B0']-ar['B2'])}
    b.csv_write(out/'tables/subject_connectivity_state_effect.csv',fc_effect_rows)
    output['connectivity_EC_minus_EO']=fc_effect

    fp=dict(origin['fingerprint']); fprows=[]
    for st in STATES:
        value,rows=fingerprint(b,results,base,st,'B0')
        fp[st+'_B0']=value
        fprows.extend(rows)
        # Existing original fingerprint rows are not changed or re-estimated.
    b.csv_write(out/'tables/subject_B0_fingerprint.csv',fprows)
    output['fingerprint']=fp
    output['bootstrap']=origin['bootstrap']
    output['reference']=origin['reference']
    output['reference_boundaries']={'in_sample':True,'subject_strata_frozen':True,
        'validation_or_test_waveforms':False,'subject_conditioning_causal_effect_identified':False}
    rows=[]
    for r0,rb in zip(results,base):
        for st in STATES:
            for g in GROUPS:
                rows.append({'subject_id':r0['subject_id'],'state':st,'generator':g,
                             **merged_state(r0,rb,st,'full_pool_metrics',g)})
    b.csv_write(out/'tables/subject_state_metrics.csv',rows)
    return output


def make_figures(ctx,results,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'status':'SKIPPED_MATPLOTLIB_UNAVAILABLE','files':[]}
    b=ctx['b']; base=ctx['baselines']
    where=out/'figures'; where.mkdir(parents=True,exist_ok=True)
    files=[]
    def save(fig,name):
        import io
        fig.tight_layout()
        buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=160)
        plt.close(fig)
        b.atomic_bytes(where/name,buf.getvalue())
        files.append((Path('figures')/name).as_posix())
    try:
        freq=np.fft.rfftfreq(256,1/b.FS)
        mask=(freq>=1)&(freq<=40)
        for st in STATES:
            fig,ax=plt.subplots(figsize=(8,4.8))
            for g in ('Real',)+GROUPS:
                arr=np.asarray([merged_state(r0,rb,st,'source_summaries',g)['mean_psd_uv2_hz']
                                 for r0,rb in zip(results,base)],dtype=float)
                power=arr.mean(axis=(0,1))
                ax.plot(freq[mask],10*np.log10(np.maximum(power[mask],b.TINY)),label=g)
            ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB re 1 uV^2/Hz)',
                   title=st+': equal-subject mean PSD; Real / B0 / B1 / B2 (descriptive)')
            ax.legend();ax.grid(alpha=.2);save(fig,'group_psd_'+st+'.png')
            mats={g:np.nanmean(np.asarray([merged_state(r0,rb,st,'nonoverlap_connectivity_matrices',g,'alpha')
                 for r0,rb in zip(results,base)],dtype=float),axis=0)
                  for g in ('Real',)+GROUPS}
            vals=np.concatenate([m[b.UT] for m in mats.values()]);vals=vals[np.isfinite(vals)]
            lo=float(vals.min()) if vals.size else 0.;hi=float(vals.max()) if vals.size else 1.
            if hi<=lo: hi=lo+1e-8
            for g,mat in mats.items():
                view=mat.copy();np.fill_diagonal(view,np.nan)
                fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(np.ma.masked_invalid(view),vmin=lo,vmax=hi)
                ax.set_xticks(range(b.NC),b.CHANNELS,rotation=90)
                ax.set_yticks(range(b.NC),b.CHANNELS)
                ax.set_title(f'{st}: alpha debiased wPLI squared, {g}\nEqual-subject mean; common scale')
                fig.colorbar(im,ax=ax);save(fig,f'group_alpha_connectivity_{st}_{g}.png')
        fig,ax=plt.subplots(figsize=(10,4.8))
        for g in ('Real',)+GROUPS:
            delta=np.asarray([
                np.asarray(merged_state(r0,rb,'EC','source_summaries',g)['median_bandpower_db'])[2]
                -np.asarray(merged_state(r0,rb,'EO','source_summaries',g)['median_bandpower_db'])[2]
                for r0,rb in zip(results,base)])
            ax.plot(range(b.NC),delta.mean(axis=0),marker='o',label=g)
        ax.set_xticks(range(b.NC),b.CHANNELS,rotation=60)
        ax.set(ylabel='EC minus EO alpha power (dB)',title='Within-subject alpha state effect; descriptive')
        ax.grid(alpha=.2);ax.legend();save(fig,'group_alpha_state_effect.png')
        return {'status':'CREATED','files':files}
    except Exception as e:
        plt.close('all')
        return {'status':'PLOT_ERROR_QUANTITATIVE_RESULTS_RETAINED','files':files,'error':str(e)}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--stop-after-subjects',type=int,default=2,
                        help='Cumulative target, not additional subjects; first test 2 then 198.')
    args=parser.parse_args()
    require(1<=args.stop_after_subjects<=198,'--stop-after-subjects must be 1..198')
    repo=args.repo_root.resolve()
    out=repo/OUT_REL
    begin=time.perf_counter()
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    limits=None
    try:
        from threadpoolctl import threadpool_limits
        limits=threadpool_limits(limits=2)
    except ImportError:
        pass
    try:
        b=pinned_import(repo/'scripts/data/s2_74b_final_pool_fidelity.py',B_SHA,'_s74e_pinned_b')
        d=pinned_import(repo/D_REL,sha_file(repo/D_REL),'_s74e_pinned_d')
        # Use the EXACT inherited s2_74B writer-lock implementation in a NEW directory.
        with b.writer_lock(out):
            print('S2_74E — B0 descriptive fidelity extension — CPU only; frozen s2_74B methods',flush=True)
            ctx=frozen_preflight(repo,b,d)
            ctx['repo']=repo
            baseline_ap=ctx['apath']; baseline_sha=sha_file(baseline_ap)
            frozen_sha=sha_file(ctx['freeze_path'])
            analysis_plan={
                'schema':'s2_74e_b0_analysis_plan_v1',
                'implementation_sha256':sha_file(Path(__file__)),
                'original_s2_74b_source_sha256':B_SHA,
                'original_s2_74b_plan_sha256':baseline_sha,
                'original_s2_74b_complete_report_sha256':sha_file(repo/B_BASELINE_REL/'metrics/s2_74b_complete_report_v1.json'),
                'B0_s2_74d_freeze_manifest_sha256':frozen_sha,
                'B0_s2_74d_view_sha256':sha_file(ctx['view_path']),
                'B0_s2_74c_pool_manifest_sha256':sha_file(repo/B0_C_REL/'manifests/s2_74c_b0_pool_manifest_v1.json'),
                'parent_generation_plan_canonical_sha256':b.sha_obj(ctx['plan']),
                'baseline_settings':b.settings(),
                'selections':ctx['baseline_plan']['selections'],
                'input_sha256':{str(p.relative_to(repo)):digest for p,digest in ctx['tracked'].items()},
                'analysis_scope':'B0 descriptive in-sample Real Train comparison, plus reuse of frozen s2_74B B1/B2 results; no model selection.',
                'B0_subject_id_role':'evaluation_stratum_only_not_model_condition',
                'no_causal_claim_from_B0_B1_B2':True,
                'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False}
            p=out/'protocol/s2_74e_b0_analysis_plan_v1.json'
            b.freeze_json(p,analysis_plan)
            plan_sha=sha_file(p)
            print('New B0 analysis plan frozen:',p,flush=True)
            rng=np.random.RandomState(b.ANALYSIS_SEED)
            projections=rng.normal(size=(b.NC*b.NT,b.N_PROJ))
            projections/=np.linalg.norm(projections,axis=0,keepdims=True)
            results=[]; fresh=0; elapsed_fresh=0.
            for si,sid in enumerate(ctx['subjects'][:args.stop_after_subjects]):
                tick=time.perf_counter(); org=ctx['baselines'][si]
                entries={};input_hashes={}
                subject_path=out/'subjects'/f'{sid}.json'
                receipt=subject_path.with_suffix('.sha256')
                cached=subject_path.exists() and receipt.exists()
                for st in STATES:
                    # Read one subject at a time; all h5py waveform reads are guarded as Train by s2_74B.
                    e,inp=b0_subject_state(ctx,sid,st,si,projections,org,do_analysis=not cached)
                    if not cached:
                        entries[st]=e
                    input_hashes[st]=inp
                if cached:
                    require(sha_file(subject_path)==receipt.read_text().strip(),
                            'Cached B0 subject output checksum mismatch: '+sid)
                    r=b.load_json(subject_path)
                    require(r['analysis_plan_sha256']==plan_sha and r['subject_id']==sid and
                            r['subject_index']==si and r['input_hashes']==input_hashes and
                            r['baseline_subject_sha256']==ctx['baseline_receipts'][sid],
                            'Cached B0 subject input/plan changed: '+sid)
                    print(f'  REUSED {si+1}/198 {sid}',flush=True)
                else:
                    # As in s2_74B, an incomplete (orphan) subject output is
                    # recomputed and atomically replaced; it is never trusted.
                    r={'subject_id':sid,'subject_index':si,
                       'analysis_plan_sha256':plan_sha,'baseline_subject_sha256':ctx['baseline_receipts'][sid],
                       'input_hashes':input_hashes,'states':entries,
                       'elapsed_seconds':time.perf_counter()-tick,
                       'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False}
                    b.write_json(subject_path,r)
                    b.atomic_bytes(receipt,(sha_file(subject_path)+'\n').encode())
                    r=b.load_json(subject_path)
                    fresh+=1;elapsed_fresh+=r['elapsed_seconds']
                    print(f'  COMMITTED {si+1}/198 {sid}: {r["elapsed_seconds"]:.1f}s',flush=True)
                results.append(r)
                b.write_json(out/'progress/s2_74e_b0_progress_v1.json',
                    {'status':'S2_74E_PARTIAL','verified_subjects':len(results),
                     'target_subjects':args.stop_after_subjects,'analysis_plan_sha256':plan_sha,
                     'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False})
            for path,digest in ctx['tracked'].items():
                require(sha_file(path)==digest, 'Source artifact changed during fidelity: '+str(path))
            if len(results)==198:
                report=aggregate(ctx,results,out)
                report['figures']=make_figures(ctx,results,out)
                status='S2_74E_COMPLETE'
                b.write_json(out/'metrics/s2_74e_b0_complete_report_v1.json',report)
            else:
                status='S2_74E_PARTIAL'
                report={'status':status,'subjects_verified':len(results),
                        'group_scientific_conclusions':'NOT produced from timing prefix'}
            report.update({'elapsed_seconds_this_run':time.perf_counter()-begin,
                           'fresh_subjects_this_run':fresh,
                           'mean_fresh_subject_seconds':elapsed_fresh/fresh if fresh else None,
                           'estimated_remaining_subject_seconds':(198-len(results))*elapsed_fresh/fresh if fresh else None,
                           'time_estimate_excludes_final_aggregation':True,
                           'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False})
            rpt=out/'metrics'/f's2_74e_b0_run_{run}.json'
            b.write_json(rpt,report)
            b.write_json(out/'progress/s2_74e_b0_progress_v1.json',
                         {'status':status,'verified_subjects':len(results),
                          'all_198_complete':len(results)==198,'analysis_plan_sha256':plan_sha,
                          'latest_report':str(rpt),
                          'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False})
            print('\n'+status)
            print(f'Verified B0 subjects: {len(results)}/198; elapsed {(time.perf_counter()-begin)/60:.2f} min')
            if fresh:
                print(f'Estimated remaining subject compute: {(198-len(results))*elapsed_fresh/fresh/60:.1f} min (preliminary)')
            print('Validation waveform accessed: NO\nFinal Test waveform accessed: NO')
            print('RUN REPORT:',rpt)
            print('ANALYSIS PLAN:',p)
            return 0
    except KeyboardInterrupt:
        print('S2_74E_INTERRUPTED — committed subject outputs retained.',file=sys.stderr)
        return 130
    except Exception as exc:
        print('S2_74E_FAILED: '+type(exc).__name__+': '+str(exc),file=sys.stderr)
        return 1
    finally:
        if limits is not None:
            limits.restore_original_limits()


if __name__=='__main__':
    raise SystemExit(main())
