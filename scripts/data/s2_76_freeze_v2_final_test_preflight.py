#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""s2_76: pre-Final-Test artifact freeze for the 11-condition s2_75B-v2 utility study.

READS: frozen split/scaler metadata, saved Validation summaries and predictions,
       receipts and binary hashes of all 55 frozen trained models.
NEVER READS: Test HDF5 groups, Test waveforms, or any EEG waveform in this stage.
NEVER DOES: model training, fitting, prediction on Test, rewriting source files.

Run from Scenario-2 repo root:
  .\.venv\Scripts\python.exe -u .\scripts\data\s2_76_freeze_v2_final_test_preflight.py
Later verify frozen manifest with the same command and --verify-only.

The historical s2_72 protocol and s2_75A-v1 amendment remain intact. This
freeze declares s2_75B-v2 (post-pilot 120-epoch/early-stop revision) as the
SOLE basis of the later single, all-55-condition s2_77 Final Test evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

# Source/report bytes are pinned to the files reviewed on 2026-10-08.
S75A_V2_SHA = 'f71e275af0d0221d5e75b6bd58de166b5a5e93fc8f2b0e3cf799111412b36947'
S75B_V2_SHA = '4fac6df5363b465a60c98788f4aa968ec99a2339589455925c00f27753146043'
S75C_SHA = '000ae00014b16de3b72ffaca5611cdb0300da1ebaf3436dfa5dbc961a15aa6aa'
AMENDMENT_SHA = '86688d27dcd7d961e2ac96e4e2bd5de1b4278b4d14e123a8efb1b47a2d5a0650'
VALIDATION_REPORT_SHA = 'c84c4ff6ffe3d6450529745766d97d9a75fbf60435f2a3294e1024ae77b714b8'
VALIDATION_CSV_SHA = 'ae5b568664b384515c4567a3cca75d01fc330cd1184d7ce848d01f58260de074'
ORIGINAL_PROTOCOL_SHA = '5b98ff09d977006b96ea80e43d4f13f33742745b734d5dfa94bb1eaacb682a22'

STAGE = 's2_76_v2_final_test_preflight_v1'
STATUS = 'S2_76_V2_FINAL_TEST_PREFLIGHT_FROZEN'
FREEZE_REL = Path('outputs/frozen/scenario2/utility/s2_76_v2_final_test_preflight_v1/manifests/s2_76_v2_pre_final_test_freeze_manifest.json')
REPORT_REL = Path('outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es/final_validation/s2_75b_v2_complete_validation_report.json')
CSV_REL = Path('outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es/final_validation/all_55_classifier_validation_runs.csv')

class FreezeError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise FreezeError(message)


def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=True,separators=(',', ':'),allow_nan=False).encode('utf-8')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(part)
    return digest.hexdigest()


def load_json(path):
    def unique(pairs):
        obj={}
        for k,v in pairs:
            require(k not in obj,'Duplicate JSON key '+k+' in '+str(path))
            obj[k]=v
        return obj
    def reject_bad(value):
        raise FreezeError('Invalid JSON constant '+str(value))
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f,object_pairs_hook=unique,parse_constant=reject_bad)


def safe_file(repo,relative,digest=None):
    path=(repo/relative).resolve()
    require(path.is_relative_to(repo) and path.is_file() and not path.is_symlink(),
            'Missing/unsafe repository file: '+str(relative))
    if digest is not None:
        require(sha256_file(path)==digest,'SHA256 mismatch: '+str(relative))
    return path


def import_v2(repo):
    relative=Path('scripts/data/s2_75b_v2_utility_120ep_earlystop.py')
    path=safe_file(repo,relative,S75B_V2_SHA)
    spec=importlib.util.spec_from_file_location('_s76_verified_s75b_v2',path)
    require(spec and spec.loader,'Cannot import original s2_75B-v2 source')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def read_test_split_metadata(ctx):
    """IMPORTANT: this function reads only the signed CSV; it never opens HDF5."""
    import pandas as pd
    split=pd.read_csv(ctx['split'])
    require({'split','subject_id','eye_state','block_eligible','candidate_windows',
             'usable_windows','block_uid','block_index'} <= set(split.columns),
            'Frozen split CSV lacks required metadata columns')
    test=split[split['split'].astype(str).eq('Test')].copy()
    require(len(test)==792,'Expected 792 assigned Test blocks from original s2_72')
    require(len(ctx['am']['utility_conditions'])==11,'Changed utility conditions')
    subjects=sorted(test['subject_id'].astype(str).unique().tolist())
    need_subjects=sorted({p['subject_id'] for p in ctx['pairs']})
    require(subjects==need_subjects and len(subjects)==198,
            'Test subjects differ from frozen Train cohort')
    combos=Counter(zip(test['subject_id'].astype(str),test['eye_state'].astype(str)))
    require(set(combos)=={(s,st) for s in subjects for st in ('EO','EC')} and
            all(v==2 for v in combos.values()),
            'Test must have exactly 2 EO and 2 EC blocks per subject')
    eligible=test['block_eligible'].astype(str).str.strip().str.lower()
    require(eligible.isin(['true','1']).all(),'Unexpected ineligible Test block')
    require(test['block_uid'].is_unique,'Duplicated Test block UID')
    candidate=pd.to_numeric(test['candidate_windows'],errors='raise')
    usable=pd.to_numeric(test['usable_windows'],errors='raise')
    require(candidate.notna().all() and usable.notna().all() and
            (candidate>=0).all() and (usable>0).all() and (usable<=candidate).all() and
            (candidate%1==0).all() and (usable%1==0).all(),
            'Invalid candidate/usable count metadata in Test split')
    return {'metadata_source':'frozen final_block_split_manifest.csv ONLY',
            'assigned_test_blocks':792,'subjects':198,'blocks_per_subject_state':2,
            'candidate_test_windows_from_manifest':int(candidate.sum()),
            'usable_test_windows_from_manifest':int(usable.sum()),
            'EO_usable_windows_from_manifest':int(usable[test['eye_state'].eq('EO')].sum()),
            'EC_usable_windows_from_manifest':int(usable[test['eye_state'].eq('EC')].sum()),
            'test_hdf5_opened':False,'test_waveforms_read':False}


def expected_parts_sources(condition,cond_definition):
    if condition in ('REAL_ONLY','REAL_DUPLICATE_1X_CONTROL'):
        names=['Real']
    elif condition.startswith('TSTR_'):
        names=[cond_definition['synthetic_generator']]
    elif condition.startswith('REAL_PLUS_'):
        names=['Real',cond_definition['synthetic_generator']]
    else:
        raise FreezeError('Unknown condition '+condition)
    return [(name,None) for name in names]


def check_validation_csv(rows_by_task,task,key):
    family,cond,seed=task
    row=rows_by_task[key]
    require(row['family']==family and row['condition']==cond and int(row['seed'])==seed,
            'CSV task mismatch')
    return row


def verify_55(ctx,mod,report,csv_rows):
    import numpy as np
    planned=mod.schedule(ctx)
    require(len(planned)==55 and len(set(planned))==55,'Incorrect 55-task schedule')
    require(set((r['family'],r['condition'],int(r['seed'])) for r in csv_rows)==set(planned)
            and len(csv_rows)==55,'CSV tasks differ from frozen 55-task plan')
    rows_by_task={(r['family'],r['condition'],int(r['seed'])):r for r in csv_rows}
    defs={entry['id']:entry for entry in ctx['am']['utility_conditions']}
    require(len(defs)==11,'Duplicated or missing utility conditions')
    task_list=[]
    label_sha=None;subject_sha=None
    grouped={}
    scalar_keys=('subject_macro_balanced_accuracy','subject_macro_roc_auc',
                 'global_balanced_accuracy','global_roc_auc')
    for i,(family,condition,seed) in enumerate(planned,1):
        cond=defs[condition]
        paths=mod.task_paths(ctx,family,condition,seed)
        desc=mod.descriptor(ctx,family,cond,seed,expected_parts_sources(condition,cond))
        require(desc['code_sha256']==S75B_V2_SHA and desc['s2_75a_amendment_sha256']==AMENDMENT_SHA,
                'Changed classifier training descriptor')
        rec=mod.existing_task(paths,desc)
        require(rec is not None,'Missing completed task '+str((family,condition,seed)))
        require(rec['Final_Test_waveforms_accessed'] is False and rec['Validation_waveforms_accessed'] is True,
                'Classifier receipt indicates wrong data-access policy')
        for artifact_key, path_key in (('model','model'),('pred','pred')):
            expected_path=paths[path_key].relative_to(ctx['repo']).as_posix()
            require(rec['files'][artifact_key]['path']==expected_path,
                    'Receipt artifact path differs from frozen schedule: '+expected_path)
        row=check_validation_csv(rows_by_task,(family,condition,seed),(family,condition,seed))
        with np.load(paths['pred'],allow_pickle=False) as z:
            require(set(z.files)=={'y_true','y_pred','y_score','subject_index'},
                    'Unexpected Validation prediction archive fields')
            y=np.asarray(z['y_true']);pred=np.asarray(z['y_pred']);score=np.asarray(z['y_score']);subjects=np.asarray(z['subject_index'])
            if label_sha is None:
                label_sha=hashlib.sha256(y.tobytes()).hexdigest()
                subject_sha=hashlib.sha256(subjects.tobytes()).hexdigest()
            require(hashlib.sha256(y.tobytes()).hexdigest()==label_sha and
                    hashlib.sha256(subjects.tobytes()).hexdigest()==subject_sha,
                    'Validation population differs among classifier predictions')
            scored=mod.perf_scores(y,pred,score,subjects)
        for metric in scalar_keys:
            expected=float(rec['result'][metric])
            require(abs(scored[metric]-expected)<1e-10 and abs(float(row[metric])-expected)<1e-10,
                    'Recomputed Validation metric differs: '+str((family,condition,seed,metric)))
        require(np.allclose(scored['subject_ba'],rec['result']['subject_ba'],atol=1e-10),
                'Per-subject BA changed relative to Validation receipt')
        method=rec['method']
        meta={'family':family,'condition':condition,'seed':seed,
              'receipt':{'path':paths['receipt'].relative_to(ctx['repo']).as_posix(),
                         'sha256':sha256_file(paths['receipt'])},
              'model':rec['files']['model'],
              'validation_predictions':rec['files']['pred'],
              'validation_subject_macro_ba':scored['subject_macro_balanced_accuracy']}
        if family=='EEGNETV4':
            best=int(method['best_epoch']);stopped=int(method['stopped_epoch'])
            require(1<=best<=stopped<=120 and len(method['training_history'])==stopped,
                    'Invalid EEGNet training history')
            require(method['stop_reason'] in ('early_stopping','max_epochs') and
                    method['Validation_early_stopping'] is True and
                    method['Validation_best_checkpoint_selection'] is True,
                    'Invalid v2 best-Validation checkpoint provenance')
            hist=method['training_history']
            require(abs(float(hist[best-1]['validation_subject_macro_ba'])-
                        rec['result']['subject_macro_balanced_accuracy'])<1e-10,
                    'Selected EEGNet checkpoint differs from best recorded epoch')
            require(row['best_epoch']==str(best) and row['stopped_epoch']==str(stopped) and
                    row['stop_reason']==method['stop_reason'] and
                    int(row['optimizer_updates_done'])==int(method['optimizer_updates_done']),
                    'EEGNet early-stop CSV data differs')
            meta['best_epoch']=best;meta['stopped_epoch']=stopped
            meta['stop_reason']=method['stop_reason']
        else:
            require(not any(row[x] for x in ('best_epoch','stopped_epoch','stop_reason','optimizer_updates_done')),
                    'Unexpected EEGNet-only metrics in classical classifier')
        task_list.append(meta)
        grouped.setdefault((family,condition),[]).append(scored)
        if i%11==0 or i==55:
            print(f'  verified {i:2d}/55 task receipts/models/predictions + Validation scores',flush=True)
    for (family,condition),values in grouped.items():
        entry=report['metric_subject_macro_ba'][family][condition]
        for metric in scalar_keys:
            means=[v[metric] for v in values]
            expected=entry[metric]
            require(abs(float(np.mean(means))-float(expected['mean']))<1e-10 and
                    abs(float(np.std(means,ddof=0))-float(expected['population_sd']))<1e-10 and
                    int(expected['n_seeds'])==len(values),
                    'Final Validation report summary differs from 55 verified tasks')
    return task_list,{'y_true_sha256':label_sha,'subject_index_sha256':subject_sha,
                      'validation_rows':9635,'subjects':198}


def build_freeze(repo):
    safe_file(repo,'scripts/data/s2_75a_v2_freeze_120ep_earlystop.py',S75A_V2_SHA)
    safe_file(repo,'scripts/data/s2_75c_finalize_v2_report_csv_fix.py',S75C_SHA)
    mod=import_v2(repo)
    ctx=mod.context(repo)  # Metadata-only. No EEG arrays/index/HDF5 opened.
    require(ctx['am_sha']==AMENDMENT_SHA,'Frozen v2 amendment changed')
    require(mod.FAMILIES==('LOGBANDPOWER_LOGREG','OAS_RIEMANN_LINEAR_SVM','EEGNETV4'),
            'Classifier families differ')
    require(mod.EEG_SEEDS==(20262301,20262302,20262303) and mod.MAX_EPOCHS==120 and
            mod.MIN_EPOCHS==8 and mod.PATIENCE==6,'EEGNet v2 budget changed')
    require(ctx['proto']['final_test_policy']['current_status']=='SEALED' and
            ctx['proto']['final_test_policy']['train_plus_validation_refit'] is False,
            'Original final-test policy violated')
    report_file=safe_file(repo,REPORT_REL,VALIDATION_REPORT_SHA)
    csv_file=safe_file(repo,CSV_REL,VALIDATION_CSV_SHA)
    report=load_json(report_file)
    require(report['status']=='S2_75B_V2_VALIDATION_UTILITY_COMPLETE' and
            report['n_tasks']==55 and report['n_conditions']==11 and
            report['source_amendment_sha256']==AMENDMENT_SHA and
            report['Final_Test_waveforms_accessed'] is False and
            report['EEGNet_selected_using_Real_Validation'] is True and
            report['post_pilot_method_revision_from_s2_72'] is True,
            'Validation report integrity/provenance differs')
    figs=report['figures']
    require(figs['status']=='CREATED' and len(figs['files'])==14 and
            len(set(figs['files']))==14,'Missing or unexpected 14 Validation figures')
    verified_figures=[]
    for filename in figs['files']:
        p=safe_file(repo,mod.RUN_REL/filename)
        verified_figures.append({'path':p.relative_to(repo).as_posix(),'sha256':sha256_file(p)})
    with csv_file.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        headers=reader.fieldnames or []
        required={'family','condition','seed','subject_macro_balanced_accuracy',
                  'subject_macro_roc_auc','global_balanced_accuracy','global_roc_auc',
                  'best_epoch','stopped_epoch','stop_reason','optimizer_updates_done'}
        require(set(headers)==required,'Missing/different CSV report columns')
        table=list(reader)
    print('S2_76_V2_PRETEST: verifies only saved artifacts and locked split metadata',flush=True)
    tasks,validation_population=verify_55(ctx,mod,report,table)
    test_meta=read_test_split_metadata(ctx)
    own=Path(__file__).resolve()
    require(own==(repo/'scripts/data'/own.name).resolve(), 'Freeze script must run from scripts/data')
    source={'s2_76_pretest_implementation':sha256_file(own),
            'original_s2_72_protocol':ORIGINAL_PROTOCOL_SHA,
            's2_75a_v2_source':S75A_V2_SHA,'s2_75b_v2_training_source':S75B_V2_SHA,
            's2_75c_report_recovery_source':S75C_SHA,
            's2_75a_v2_frozen_amendment':AMENDMENT_SHA,
            's2_75b_v2_validation_report':VALIDATION_REPORT_SHA,
            's2_75b_v2_validation_csv':VALIDATION_CSV_SHA,
            'split_manifest':sha256_file(ctx['split']),
            'train_normalization':sha256_file(ctx['normalization'])}
    require(source['original_s2_72_protocol']==sha256_file(repo/Path(
        'outputs/protocols/scenario2/utility/b1_vs_b2_utility_protocol_v1/metrics/s2_72_b1_vs_b2_downstream_utility_protocol_v1.json')),
        'Original s2_72 protocol changed')
    return {'schema':STAGE,'status':STATUS,
            'reference_study':'s2_75B-v2 11-condition 120-epoch Early Stopping ONLY',
            'post_pilot_training_policy_revision':True,
            'training':{'permitted':False,'refit_on_Train_plus_Validation':False,
                        'best_validation_checkpoint_frozen':True},
            'test_access':{'stage_s2_76_test_waveforms_read':False,
                           'stage_s2_77_requires_separate_explicit_open':True,
                           'opening_only_after_this_manifest_frozen':True,
                           'one_final_test_evaluation':True},
            'classifier_families':list(mod.FAMILIES),
            'all_11_conditions_in_original_amendment_order':[c['id'] for c in ctx['am']['utility_conditions']],
            'EEGNet_seeds':list(mod.EEG_SEEDS),
            'classifier_fits':55,'source_hashes':source,
            'validation_population':validation_population,
            'test_metadata_only':test_meta,
            'final_test_indexing_rule':('Use precisely the eligible Test blocks in the signed final split, '
                   'sorted by subject_id and block_index; use each block window_start_samples '
                   'at positions where window_usable_mask is true; no cross-block windowing.'),
            'test_input_contract':{'samples_per_window':512,'channels':19,'sample_rate_hz':128,
                    'labels':{'EO':0,'EC':1},
                    'normalization':'frozen Train-only 19-channel global mean/std for EEGNet and OAS; '
                                     'original locked uV Welch log10 bandpower for LogisticRegression'},
            'threshold_policy':{'LOGBANDPOWER_LOGREG':'original classifier.predict, no tuning',
                                'OAS_RIEMANN_LINEAR_SVM':'zero margin from original LinearSVC.predict',
                                'EEGNETV4':'fixed softmax argmax (= class-1 probability >= 0.5), no tuning'},
            'final_test_metrics':['subject_macro_balanced_accuracy','subject_macro_roc_auc',
                                  'global_balanced_accuracy','global_roc_auc',
                                  'confusion_matrix','per_subject_balanced_accuracy'],
            'condition_stats':{'primary_metric':'subject_macro_balanced_accuracy',
                       'EEGNet':'mean and population SD across all 3 fixed seeds; no best seed selection',
                       'paired_subject_bootstrap':{'replicates':5000,'seed':20262401,
                         'unit':'198 subjects, paired across conditions and matched seeds',
                         'confidence_interval':'95% percentile; not multiplicity-adjusted'},
                       'all_frozen_original_and_B0_supplementary_contrasts':[
                         {'id':x['id'],'left':x['left'],'right':x['right'],
                          'classification':'original_pairs_under_revised_v2_training'}
                         for x in ctx['am']['primary_s2_72_comparisons_preserved_unmodified']]+[
                         {'id':x['id'],'left':x['left'],'right':x['right'],
                          'classification':'supplementary_B0'}
                         for x in ctx['am']['secondary_supplementary_comparisons']]},
            'validation_figures_readonly_sha256':verified_figures,
            'tasks':tasks,
            'scientific_limits':['same 198 subjects in Train/Validation/Test, held-out complete blocks',
                  'v2 revised after examining a v1 Validation pilot',
                  'no independent unseen-subject generalization claim',
                  'Final Test must not be reused for selection/tuning']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--verify-only',action='store_true',help='Require existing frozen manifest; do not create/modify')
    args=ap.parse_args()
    repo=args.repo_root.expanduser().resolve()
    try:
        require((repo/'scripts/data').is_dir(),'Wrong Scenario-2 repository root')
        manifest=build_freeze(repo)
        target=repo/FREEZE_REL
        if target.is_file():
            require(not target.is_symlink() and canonical(load_json(target))==canonical(manifest),
                    'Existing s2_76 manifest differs. Never overwrite frozen pretest selection.')
            print('S2_76_V2_REUSED_IDENTICAL_FROZEN_MANIFEST',flush=True)
        else:
            require(not target.exists(),'Unexpected non-file final freeze path')
            require(not args.verify_only,'No frozen s2_76 manifest to verify')
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                with tmp.open('xb') as f:
                    f.write(canonical(manifest)+b'\n');f.flush();os.fsync(f.fileno())
                os.replace(tmp,target)
            finally:
                if tmp.exists(): tmp.unlink()
            print('S2_76_V2_NEW_PRETEST_MANIFEST_FROZEN',flush=True)
        print(STATUS,flush=True)
        print('All 55/55 frozen classifiers/models/Validation predictions: VERIFIED',flush=True)
        print('Validation figures:',len(manifest['validation_figures_readonly_sha256']))
        print('Test blocks from locked split metadata:',manifest['test_metadata_only']['assigned_test_blocks'])
        print('Test usable windows from locked CSV metadata:',manifest['test_metadata_only']['usable_test_windows_from_manifest'])
        print('Models retrained: NO; Test HDF5 opened: NO; Test waveforms accessed: NO')
        print('FREEZE MANIFEST:',target)
        print('SHA256:',sha256_file(target))
        print('Next stage only after inspection: s2_77 single Final-Test opening')
        return 0
    except Exception as exc:
        print('S2_76_V2_FAILED:',type(exc).__name__+': '+str(exc),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
