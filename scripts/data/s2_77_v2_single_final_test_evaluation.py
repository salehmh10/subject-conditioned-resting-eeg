#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""s2_77 — one declared Final-Test evaluation of all 55 frozen s2_75B-v2 fits.

Run from Scenario-2 repo root:

  .\.venv\Scripts\python.exe -u .\scripts\data\s2_77_v2_single_final_test_evaluation.py --preflight

ONLY after confirming the preflight and authorizing the one-time Final-Test opening:

  .\.venv\Scripts\python.exe -u .\scripts\data\s2_77_v2_single_final_test_evaluation.py --open-final-test

On accidental interruption of THAT SAME registered one-time evaluation:

  .\.venv\Scripts\python.exe -u .\scripts\data\s2_77_v2_single_final_test_evaluation.py --resume

Never trains/refits/tunes/selects a model on Test. Only s2_75B-v2 models are
accepted. Previous 24-epoch v1 models are never loaded or compared. Test data
may be opened only after explicit --open-final-test and a durable access marker.
The completed Test report cannot be regenerated or overwritten by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

STAGE = 's2_77_v2_single_final_test_v1'
S76_SOURCE_SHA = 'd4f8d054ae8c007479982941460a0ca9c01c715b2789eb34fe7fe33ba7134d98'
S76_MANIFEST_SHA = '68df0a06b54505afa419e1170a31184631afe45ed6d3f2e89ac9fc28f99b1046'
S75B_V2_SHA = '4fac6df5363b465a60c98788f4aa968ec99a2339589455925c00f27753146043'
EXPECTED_BLOCKS, EXPECTED_WINDOWS, NSUB, NC, NT = 792, 19131, 198, 19, 512
BS = 128
OUT_REL = Path('outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es/final_test_s2_77_v1')
ACCESS_NAME = 'manifests/s2_77_single_test_access.json'
INPUT_NAME = 'cache/test_input_v1/s2_77_test_input_commit.json'
COMPLETE_NAME = 'manifests/s2_77_final_test_complete.json'
REPORT_NAME = 's2_77_v2_complete_final_test_report.json'

class FinalTestError(RuntimeError):
    pass

def require(condition, why):
    if not condition:
        raise FinalTestError(why)

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()

def hash_array(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def load_json(path):
    def uniq(pairs):
        d = {}
        for key, val in pairs:
            require(key not in d, 'Duplicate JSON key in '+str(path)+': '+key)
            d[key] = val
        return d
    def reject(v):
        raise FinalTestError('Invalid JSON number '+str(v)+' in '+str(path))
    with Path(path).open(encoding='utf8') as f:
        return json.load(f, object_pairs_hook=uniq, parse_constant=reject)

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

def create_once(path, payload):
    """Immutable files: identical reruns are okay, modified content is refused."""
    path = Path(path)
    if path.exists():
        require(path.is_file() and not path.is_symlink() and
                canonical(load_json(path)) == canonical(payload),
                'Refusing to change an existing artifact: '+str(path))
    else:
        atomic_bytes(path, canonical(payload)+b'\n')

def import_exact(path, digest, name):
    require(path.is_file() and not path.is_symlink() and hash_file(path)==digest,
            'Frozen Python source differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load source '+str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def validate_freeze(repo):
    """ZERO HDF5 access, ZERO Test waveform reads. Re-verify s2_76 manifest."""
    s76 = import_exact(repo/'scripts/data/s2_76_freeze_v2_final_test_preflight.py',
                       S76_SOURCE_SHA, '_s2_77_pinned_s76')
    frozen_path = repo/s76.FREEZE_REL
    require(frozen_path.is_file() and hash_file(frozen_path)==S76_MANIFEST_SHA,
            'The explicitly approved s2_76 freeze manifest differs from the approved SHA256')
    frozen = load_json(frozen_path)
    require(frozen['status']==s76.STATUS and frozen['classifier_fits']==55,
            'Wrong pre-final-test freeze')
    require(frozen['reference_study']==
            's2_75B-v2 11-condition 120-epoch Early Stopping ONLY',
            'Wrong study version')
    require(frozen['test_access']['one_final_test_evaluation'] is True and
            frozen['training']['permitted'] is False and
            frozen['training']['refit_on_Train_plus_Validation'] is False,
            'Test/Training frozen policy differs')
    require(frozen['test_metadata_only']['assigned_test_blocks']==EXPECTED_BLOCKS and
            frozen['test_metadata_only']['usable_test_windows_from_manifest']==EXPECTED_WINDOWS and
            frozen['test_metadata_only']['subjects']==NSUB,
            'Unexpected frozen Final Test metadata')
    require(frozen['source_hashes']['s2_75b_v2_training_source']==S75B_V2_SHA,
            'Changed frozen v2 training script identity')
    # This full check rereads only signed metadata, saved models and VALIDATION
    # predictions; NEVER Test waveforms or Test HDF5.
    actual = s76.build_freeze(repo)
    require(canonical(actual)==canonical(frozen),
            's2_76 source/model/Validation freeze does not reconstruct identically')
    model_module = s76.import_v2(repo)
    ctx = model_module.context(repo)  # No HDF5, no signal loading.
    require(ctx['proto']['final_test_policy']['current_status']=='SEALED',
            'Underlying Test split is not sealed')
    assert model_module.NC==NC and model_module.NT==NT
    plan = model_module.schedule(ctx)
    require(len(plan)==55 and [(t['family'],t['condition'],int(t['seed']))
                               for t in frozen['tasks']]==list(plan),
            'Frozen model order differs from executable 55-task plan')
    return s76, model_module, ctx, frozen

def bool_value(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ('true','1','yes')

def str_value(v):
    if isinstance(v, bytes):
        return v.decode('utf8')
    return str(v)

def test_rows(ctx, frozen):
    """Signed CSV only; does not read/inspect any Test HDF5."""
    split = pd.read_csv(ctx['split'])
    test = split[split['split'].astype(str).eq('Test')].copy()
    require(len(test)==EXPECTED_BLOCKS and test['block_uid'].is_unique,
            'Final Test CSV population differs')
    test = test.sort_values(['subject_id','block_index']).reset_index(drop=True)
    mapping = {ctx['pairs'][2*i]['subject_id']:i for i in range(NSUB)}
    require(len(mapping)==NSUB and set(mapping)==set(test['subject_id'].astype(str)),
            'Frozen subject-ID mapping is inconsistent with Test split')
    combos = Counter((str(r.subject_id),str(r.eye_state)) for r in test.itertuples())
    require(set(combos)=={(sid,st) for sid in mapping for st in ('EO','EC')} and
            all(num==2 for num in combos.values()),
            'Wrong number of Test blocks per subject/eye state')
    require(all(bool_value(v) for v in test['block_eligible']),
            'Unexpected ineligible Test block')
    require(int(test['usable_windows'].sum())==EXPECTED_WINDOWS and
            int(test['usable_windows'].sum())==
              frozen['test_metadata_only']['usable_test_windows_from_manifest'],
            'Frozen Test window count differs')
    return test, mapping

def raw_features(uv):
    """Exact original Real log-bandpower convention: float64 uV Welch, no retraining."""
    from scipy.signal import welch
    from scipy.integrate import trapezoid
    uv = np.asarray(uv, dtype=np.float64)
    require(uv.ndim==3 and uv.shape[1:]==(NC,NT) and np.isfinite(uv).all(),
            'Invalid raw Test EEG input for LogBandpower')
    freqs, psd = welch(uv, fs=128.0, window='hann', nperseg=256,
                       noverlap=128, detrend='constant', scaling='density', axis=-1)
    parts = []
    for name,(lo,hi) in [('delta',(1.,4.)),('theta',(4.,8.)),
                         ('alpha',(8.,13.)),('beta',(13.,30.))]:
        mask = (freqs>=lo) & ((freqs<=hi) if name=='beta' else (freqs<hi))
        bp = trapezoid(psd[...,mask], x=freqs[mask],axis=-1)
        require(np.isfinite(bp).all() and np.all(bp>0),
                'Invalid Test bandpower for '+name)
        parts.append(np.log10(np.maximum(bp,1e-12)))
    arr = np.concatenate(parts,axis=1).astype(np.float32)
    require(arr.shape==(len(uv),76) and np.isfinite(arr).all(),
            'Wrong Test LogBandpower feature dimension')
    return arr

def read_and_cache_test(ctx, frozen, mod, out):
    """Called ONLY after durable one-time access marker exists."""
    import h5py
    from pathlib import Path
    test,subject_mapping=test_rows(ctx,frozen)
    src_mod=mod.import_baseline(ctx,'s2_27_logbandpower_logreg_validation.py')
    h5_root=Path(src_mod.ALL_BLOCKS_ROOT)
    require(h5_root.is_dir(), 'Missing signed preprocessed all-blocks-v1 HDF5 directory')
    scaler=load_json(ctx['normalization'])
    means=np.asarray(scaler['means_uv'],dtype=np.float64)
    stds=np.asarray(scaler['stds_uv'],dtype=np.float64)
    require(means.shape==stds.shape==(NC,) and np.isfinite(means).all() and
            np.isfinite(stds).all() and np.all(stds>0),
            'Frozen Train scaler invalid')
    cache_dir=out/'cache'
    cache_dir.mkdir(parents=True,exist_ok=True)
    final_cache_dir=cache_dir/'test_input_v1'
    z_dest=final_cache_dir/'test_z_float32.npy'
    feat_dest=final_cache_dir/'test_logbandpower_float32.npy'
    meta_dest=out/INPUT_NAME
    if meta_dest.exists():
        meta=load_json(meta_dest)
        require(meta['schema']==STAGE+'_test_inputs' and
                meta['s2_76_freeze_sha256']==S76_MANIFEST_SHA and
                meta['n_windows']==EXPECTED_WINDOWS and meta['n_subjects']==NSUB,
                'Modified frozen Test input manifest')
        for key,path,shape in [('test_z',z_dest,(EXPECTED_WINDOWS,NC,NT)),
                               ('logbandpower',feat_dest,(EXPECTED_WINDOWS,76))]:
            require(path.is_file() and hash_file(path)==meta['files'][key]['sha256'],
                    'Test input cache changed: '+key)
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            require(arr.shape==shape and arr.dtype==np.float32,
                    'Wrong Test input shape or dtype: '+key)
        require(len(meta['y_true'])==len(meta['subject_index'])==EXPECTED_WINDOWS and
                hash_array(np.asarray(meta['y_true'],dtype=np.int64))==meta['y_true_sha256'] and
                hash_array(np.asarray(meta['subject_index'],dtype=np.int64))==meta['subject_index_sha256'],
                'Changed Test label/subject population')
        return np.load(z_dest,mmap_mode='r',allow_pickle=False),np.load(feat_dest,mmap_mode='r',allow_pickle=False),meta
    require(not final_cache_dir.exists(),
            'Orphan Test cache without its complete atomic manifest; investigate')
    for old in cache_dir.glob('.s2_77_input_staging_*'):
        require(old.is_dir() and not old.is_symlink(),
                'Unexpected staged Test cache path; refusing cleanup')
        print('  Removing interrupted UNCOMMITTED Test input staging cache:',old.name,flush=True)
        shutil.rmtree(old)
    staging_dir=cache_dir/('.s2_77_input_staging_'+uuid.uuid4().hex)
    staging_dir.mkdir()
    temp_z=staging_dir/'test_z_float32.npy'
    temp_feat=staging_dir/'test_logbandpower_float32.npy'
    z=None;features=None
    y=np.empty(EXPECTED_WINDOWS,dtype=np.int64)
    sub=np.empty(EXPECTED_WINDOWS,dtype=np.int64)
    window_index=hashlib.sha256()
    position=0
    blocks_checked=0
    try:
        z=np.lib.format.open_memmap(temp_z, mode='w+',dtype=np.float32,
                                    shape=(EXPECTED_WINDOWS,NC,NT))
        features=np.lib.format.open_memmap(temp_feat,mode='w+',dtype=np.float32,
                                           shape=(EXPECTED_WINDOWS,76))
        for sid,group in test.groupby('subject_id',sort=True):
            sid=str(sid)
            h5_path=h5_root/(sid+'_all_blocks.h5')
            require(h5_path.is_file(), 'Missing Test subject HDF5: '+str(h5_path))
            with h5py.File(h5_path,'r') as f:
                for row in group.itertuples(index=False):
                    st=str(row.eye_state);label=0 if st=='EO' else 1
                    uid=str(row.block_uid)
                    require(st in ('EO','EC'), 'Invalid Test eye_state')
                    block=f['blocks/'+uid]
                    require(str_value(block.attrs['provisional_split'])=='Test' and
                            str_value(block.attrs['subject_id'])==sid and
                            str_value(block.attrs['state'])==st and
                            bool_value(block.attrs['block_eligible']),
                            'A supposedly Test block has mismatched HDF5 identity: '+uid)
                    starts=np.asarray(block['window_start_samples'][:],dtype=np.int64)
                    mask=np.asarray(block['window_usable_mask'][:],dtype=np.bool_)
                    require(len(starts)==len(mask)==int(row.candidate_windows) and
                            int(mask.sum())==int(row.usable_windows),
                            'Test HDF5 usable-window metadata differs from frozen CSV: '+uid)
                    eeg=block['eeg_uv']
                    require(eeg.ndim==2 and eeg.shape[0]==NC,
                            'Wrong number of Test EEG channels: '+uid)
                    good=np.flatnonzero(mask)
                    require(len(good)>0, 'Test block has no valid windows: '+uid)
                    uv=np.empty((len(good),NC,NT),dtype=np.float64)
                    for j,index in enumerate(good):
                        start=int(starts[index]);stop=start+NT
                        require(start>=0 and stop<=eeg.shape[1],
                                'Test EEG window outside original block: '+uid)
                        v=np.asarray(eeg[:,start:stop],dtype=np.float64)
                        require(v.shape==(NC,NT) and np.isfinite(v).all(),
                                'Non-finite or wrongly-shaped Test EEG: '+uid)
                        uv[j]=v
                        window_index.update(canonical([sid,int(subject_mapping[sid]),uid,
                            int(row.block_index),st,label,int(index),start])+b'\n')
                    end=position+len(good)
                    require(end<=EXPECTED_WINDOWS,'More Test windows than frozen metadata allows')
                    z[position:end]=((uv.astype(np.float32).astype(np.float64)-means[None,:,None])
                                     /stds[None,:,None]).astype(np.float32)
                    features[position:end]=raw_features(uv)
                    y[position:end]=label
                    sub[position:end]=subject_mapping[sid]
                    position=end
                    blocks_checked+=1
                    if blocks_checked%100==0 or blocks_checked==EXPECTED_BLOCKS:
                        print(f'  TEST READ: {blocks_checked}/{EXPECTED_BLOCKS} blocks; '
                              f'{position}/{EXPECTED_WINDOWS} usable windows',flush=True)
        require(blocks_checked==EXPECTED_BLOCKS and position==EXPECTED_WINDOWS,
                'Final Test windows/blocks do not agree with s2_76 signed metadata')
        require(set(np.unique(y))=={0,1} and len(np.unique(sub))==NSUB,
                'Wrong Final Test population')
        for i in range(NSUB):
            ss=y[sub==i]
            require(np.any(ss==0) and np.any(ss==1),
                    'Final Test subject missing EO or EC data')
        z.flush(); features.flush()
        del z,features
        z=None;features=None
        meta={'schema':STAGE+'_test_inputs','s2_76_freeze_sha256':S76_MANIFEST_SHA,
              'n_windows':EXPECTED_WINDOWS,'n_subjects':NSUB,'n_blocks':EXPECTED_BLOCKS,
              'EO_windows':int(np.sum(y==0)),'EC_windows':int(np.sum(y==1)),
              'y_true':y.tolist(),'subject_index':sub.tolist(),
              'y_true_sha256':hash_array(y),'subject_index_sha256':hash_array(sub),
              'test_window_index_sha256':window_index.hexdigest(),
              'files':{'test_z':{'sha256':hash_file(temp_z),'shape':[EXPECTED_WINDOWS,NC,NT]},
                       'logbandpower':{'sha256':hash_file(temp_feat),'shape':[EXPECTED_WINDOWS,76]}},
              'normalization_sha256':hash_file(ctx['normalization']),
              'split_manifest_sha256':hash_file(ctx['split']),
              'Final_Test_waveforms_accessed':True}
        create_once(staging_dir/'s2_77_test_input_commit.json',meta)
        # All three Test-input files become visible together or not at all.
        os.replace(staging_dir, final_cache_dir)
        print('S2_77_TEST_INPUTS_COMMITTED:',EXPECTED_WINDOWS,'windows',flush=True)
        return np.load(z_dest,mmap_mode='r',allow_pickle=False),np.load(feat_dest,mmap_mode='r',allow_pickle=False),meta
    finally:
        if z is not None:
            z.flush();del z
        if features is not None:
            features.flush();del features
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

def perf_scores_test(y,pred,score,sub):
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score,confusion_matrix
    y=np.asarray(y,dtype=np.int64);pred=np.asarray(pred,dtype=np.int64)
    score=np.asarray(score,dtype=np.float64);sub=np.asarray(sub,dtype=np.int64)
    require(y.shape==pred.shape==score.shape==sub.shape==(EXPECTED_WINDOWS,) and
            np.isfinite(score).all() and set(np.unique(pred)) <= {0,1} and
            set(np.unique(y))=={0,1},'Invalid prediction Test schema')
    subject_ba=[];subject_auc=[]
    for i in range(NSUB):
        sel=sub==i
        require((y[sel]==0).any() and (y[sel]==1).any(),
                'Test subject lacks both labels')
        subject_ba.append(float(balanced_accuracy_score(y[sel],pred[sel])))
        subject_auc.append(float(roc_auc_score(y[sel],score[sel])))
    return {'subject_macro_balanced_accuracy':float(np.mean(subject_ba)),
            'subject_macro_roc_auc':float(np.mean(subject_auc)),
            'global_balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'global_roc_auc':float(roc_auc_score(y,score)),
            'subject_ba':subject_ba,'subject_auc':subject_auc,
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def predict_one(ctx,mod,frozen_task,z,feats):
    import joblib
    family=frozen_task['family'];condition=frozen_task['condition'];seed=int(frozen_task['seed'])
    model_path=(ctx['repo']/frozen_task['model']['path']).resolve()
    require(model_path.is_relative_to(ctx['repo']) and model_path.is_file() and not model_path.is_symlink() and
            hash_file(model_path)==frozen_task['model']['sha256'],
            'Changed trained model: '+str(model_path))
    n=len(z)
    pred=np.empty(n,dtype=np.int64);score=np.empty(n,dtype=np.float64)
    if family=='LOGBANDPOWER_LOGREG':
        obj=joblib.load(model_path)
        require(set(obj)=={'scaler','classifier','feature_definition'},
                'Unexpected LogReg frozen model package')
        x=np.asarray(feats,dtype=np.float64)
        xs=obj['scaler'].transform(x)
        pred[:]=obj['classifier'].predict(xs)
        score[:]=obj['classifier'].predict_proba(xs)[:,1]
    elif family=='OAS_RIEMANN_LINEAR_SVM':
        from pyriemann.estimation import Covariances
        obj=joblib.load(model_path)
        require(set(obj)=={'tangent','classifier','covariance'},
                'Unexpected SVM frozen model package')
        cov=Covariances(estimator='oas')
        for k in range(0,n,128):
            test_cov=cov.transform(np.asarray(z[k:k+128],dtype=np.float64))
            tangent=obj['tangent'].transform(test_cov)
            pred[k:k+128]=obj['classifier'].predict(tangent)
            score[k:k+128]=obj['classifier'].decision_function(tangent)
    elif family=='EEGNETV4':
        import torch
        source=mod.import_baseline(ctx,'s2_29_eegnetv4_validation.py')
        package=torch.load(model_path,map_location='cpu',weights_only=True)
        require(package['schema']==mod.VERSION+'_EEGNet_selected_best_validation_state' and
                package['condition']==condition and package['seed']==seed and
                package['baseline_architecture_sha256']==mod.SOURCE_HASHES['s2_29_eegnetv4_validation.py'] and
                package['best_epoch']==frozen_task['best_epoch'],
                'Selected EEGNet package differs from v2 freeze')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net=source.EEGNetV4Baseline().to(device)
        net.load_state_dict(package['model_state_dict'],strict=True)
        net.eval().requires_grad_(False)
        with torch.inference_mode():
            for k in range(0,n,BS):
                v=torch.from_numpy(np.array(z[k:k+BS],copy=True)).to(device)
                logits=net(v)
                require(torch.isfinite(logits).all().item(),
                        'Nonfinite EEGNet Test logits')
                pred[k:k+BS]=torch.argmax(logits,dim=1).cpu().numpy()
                score[k:k+BS]=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
        del net
    else:
        raise FinalTestError('Unexpected classifier family: '+family)
    require(np.isfinite(score).all() and set(np.unique(pred))<={0,1},
            'Invalid classifier inference output')
    return pred,score

def save_or_verify_predictions(ctx,mod,task,z,features,meta,out):
    """Atomic *directory* commit: each Test model receipt and its predictions together."""
    name=f"{task['family']}__{task['condition']}__{task['seed']}"
    root=out/'predictions'
    dest=root/name
    pp=dest/'predictions.npz';rp=dest/'receipt.json'
    labels=np.asarray(meta['y_true'],dtype=np.int64)
    subs=np.asarray(meta['subject_index'],dtype=np.int64)
    if dest.exists():
        require(dest.is_dir() and not dest.is_symlink() and pp.is_file() and rp.is_file(),
                'Orphan or unsafe Final Test prediction directory: '+name)
        receipt=load_json(rp)
        require(receipt['schema']==STAGE+'_prediction_receipt' and
                receipt['family']==task['family'] and receipt['condition']==task['condition'] and
                receipt['seed']==task['seed'] and
                receipt['model_sha256']==task['model']['sha256'] and
                receipt['s2_76_freeze_sha256']==S76_MANIFEST_SHA and
                receipt['test_input_sha256']==hash_file(out/INPUT_NAME) and
                hash_file(pp)==receipt['prediction_sha256'],
                'Changed final-Test prediction receipt: '+name)
        with np.load(pp,allow_pickle=False) as data:
            require(set(data.files)=={'y_true','y_pred','y_score','subject_index'},
                    'Unexpected predictions archive keys')
            yy=data['y_true'];sb=data['subject_index'];pr=data['y_pred'];sc=data['y_score']
            require(np.array_equal(yy,labels) and np.array_equal(sb,subs),
                    'Changed Test label/subject population in '+name)
            met=perf_scores_test(yy,pr,sc,sb)
            require(canonical(met)==canonical(receipt['result']),
                    'Stored Test metric differs from predictions in '+name)
        return receipt
    root.mkdir(parents=True,exist_ok=True)
    prefix='.tmp_'+hashlib.sha256(name.encode('utf8')).hexdigest()[:12]+'_'
    for old in root.glob(prefix+'*'):
        require(old.is_dir() and not old.is_symlink(),
                'Unexpected staged Test prediction path; refusing cleanup')
        print('  Removing interrupted UNCOMMITTED Test prediction staging cache:',old.name,flush=True)
        shutil.rmtree(old)
    staging=root/(prefix+uuid.uuid4().hex[:12])
    staging.mkdir()
    try:
        pred,score=predict_one(ctx,mod,task,z,features)
        met=perf_scores_test(labels,pred,score,subs)
        buffer=io.BytesIO()
        np.savez_compressed(buffer,y_true=labels,y_pred=pred,y_score=score,subject_index=subs)
        atomic_bytes(staging/'predictions.npz',buffer.getvalue())
        receipt={'schema':STAGE+'_prediction_receipt','family':task['family'],
                 'condition':task['condition'],'seed':task['seed'],
                 's2_76_freeze_sha256':S76_MANIFEST_SHA,
                 'test_input_sha256':hash_file(out/INPUT_NAME),
                 'model_sha256':task['model']['sha256'],
                 'prediction_sha256':hash_file(staging/'predictions.npz'),'result':met,
                 'training_performed':False,'Final_Test_waveforms_accessed':True}
        create_once(staging/'receipt.json',receipt)
        os.replace(staging,dest)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)

def bootstrap_mean(values,draws):
    values=np.asarray(values,dtype=np.float64)
    require(values.shape==(NSUB,) and np.isfinite(values).all(),
            'Wrong per-subject bootstrap data')
    samples=values[draws].mean(axis=1)
    return {'mean':float(values.mean()),'ci95_percentile':np.quantile(samples,[0.025,0.975]).tolist()}

def write_csv_once(path,rows,columns):
    from io import StringIO
    s=StringIO(newline='')
    wr=csv.DictWriter(s,fieldnames=columns,extrasaction='raise')
    wr.writeheader();wr.writerows(rows)
    data=s.getvalue().encode('utf-8-sig')
    if path.exists():
        require(path.read_bytes()==data,'Refusing to overwrite changed Test CSV: '+str(path))
    else:
        atomic_bytes(path,data)

def summarize_all(frozen,receipts,out):
    """Descriptive, predeclared, paired subject resampling. No model selection."""
    from numpy.random import RandomState
    by={(r['family'],r['condition'],int(r['seed'])):r for r in receipts}
    conds=frozen['all_11_conditions_in_original_amendment_order']
    families=frozen['classifier_families']
    seeds=frozen['EEGNet_seeds']
    draws=RandomState(20262401).randint(0,NSUB,size=(5000,NSUB))
    summary={};table=[];per_subject=[];contrast_table=[]
    for family in families:
        summary[family]={}
        expected_seeds=seeds if family=='EEGNETV4' else [20262301]
        for cond in conds:
            members=[by[(family,cond,seed)] for seed in expected_seeds]
            b=np.asarray([r['result']['subject_ba'] for r in members],dtype=np.float64)
            a=np.asarray([r['result']['subject_auc'] for r in members],dtype=np.float64)
            require(b.shape==a.shape==(len(expected_seeds),NSUB),
                    'Malformed per-subject Test results')
            by_seed=np.asarray([r['result']['subject_macro_balanced_accuracy'] for r in members],dtype=np.float64)
            pooled=b.mean(axis=0)
            v={'n_seeds':len(expected_seeds),'seed_values':by_seed.tolist(),
               'subject_macro_balanced_accuracy_mean':float(by_seed.mean()),
               'subject_macro_balanced_accuracy_population_sd':float(by_seed.std(ddof=0)),
               'paired_subject_bootstrap_ci95':bootstrap_mean(pooled,draws)['ci95_percentile'],
               'subject_macro_roc_auc_mean':float(np.mean([r['result']['subject_macro_roc_auc'] for r in members])),
               'global_balanced_accuracy_mean':float(np.mean([r['result']['global_balanced_accuracy'] for r in members])),
               'global_roc_auc_mean':float(np.mean([r['result']['global_roc_auc'] for r in members]))}
            summary[family][cond]=v
            table.append({'family':family,'condition':cond,'n_seeds':len(expected_seeds),
                          'subject_macro_ba_mean':v['subject_macro_balanced_accuracy_mean'],
                          'subject_macro_ba_population_sd':v['subject_macro_balanced_accuracy_population_sd'],
                          'subject_macro_ba_ci_low':v['paired_subject_bootstrap_ci95'][0],
                          'subject_macro_ba_ci_high':v['paired_subject_bootstrap_ci95'][1],
                          'subject_macro_auc_mean':v['subject_macro_roc_auc_mean']})
            for seed,bb,aa in zip(expected_seeds,b,a):
                for sid in range(NSUB):
                    per_subject.append({'family':family,'condition':cond,'seed':seed,
                                        'subject_index':sid,'balanced_accuracy':float(bb[sid]),
                                        'roc_auc':float(aa[sid])})
        for contrast in frozen['condition_stats']['all_frozen_original_and_B0_supplementary_contrasts']:
            left,right=contrast['left'],contrast['right']
            require(left in summary[family] and right in summary[family],
                    'Invalid frozen Test contrast identity')
            diff=[]
            for seed in expected_seeds:
                l=np.asarray(by[(family,left,seed)]['result']['subject_ba'])
                r=np.asarray(by[(family,right,seed)]['result']['subject_ba'])
                diff.append(r-l)
            delta=np.mean(diff,axis=0)
            c=bootstrap_mean(delta,draws)
            contrast_table.append({'family':family,'contrast_id':contrast['id'],
                                   'contrast_group':contrast['classification'],
                                   'left':left,'right':right,'delta_right_minus_left':c['mean'],
                                   'ci95_low':c['ci95_percentile'][0],
                                   'ci95_high':c['ci95_percentile'][1],
                                   'n_subjects':NSUB,'n_seeds':len(expected_seeds)})
    return summary,table,per_subject,contrast_table

def create_figures(summary,out,frozen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures'
    figdir.mkdir(parents=True,exist_ok=True)
    made=[]
    for family in frozen['classifier_families']:
        conditions=frozen['all_11_conditions_in_original_amendment_order']
        vals=[100*summary[family][c]['subject_macro_balanced_accuracy_mean'] for c in conditions]
        fig,ax=plt.subplots(figsize=(11,5.8))
        ax.bar(range(len(conditions)),vals)
        ax.set_xticks(range(len(conditions)),conditions,rotation=55,ha='right')
        ax.set_ylabel('Final Test subject-macro balanced accuracy (%)')
        ax.set_title(f'{family} — 11 frozen conditions (NO Test-based selection)')
        ax.set_ylim(max(0,min(vals)-10),min(100,max(vals)+5))
        ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        buffer=io.BytesIO()
        fig.savefig(buffer,format='png',dpi=160)
        plt.close(fig)
        p=figdir/(family+'_all_11_test_BA.png')
        if p.exists():
            require(p.read_bytes()==buffer.getvalue(),
                    'Refusing to overwrite existing Final Test plot')
        else:
            atomic_bytes(p,buffer.getvalue())
        made.append({'path':p.relative_to(out).as_posix(),'sha256':hash_file(p)})
    return made

def finalize(ctx,mod,frozen,out):
    require(not (out/COMPLETE_NAME).exists(),
            'A Final Test report already exists; do NOT reaggregate or use Test to tune')
    receipts=[]
    for task in frozen['tasks']:
        name=f"{task['family']}__{task['condition']}__{task['seed']}"
        rp=out/'predictions'/name/'receipt.json'
        require(rp.is_file(),'Missing Final Test prediction receipt: '+name)
        rec=load_json(rp)
        require(rec['family']==task['family'] and rec['condition']==task['condition'] and
                rec['seed']==task['seed'] and rec['model_sha256']==task['model']['sha256'],
                'Wrong Final Test predicted model identity')
        receipts.append(rec)
    require(len(receipts)==55,'Not exactly 55 frozen Test prediction receipts')
    summary,rows,per_subject,contrasts=summarize_all(frozen,receipts,out)
    out_report=out/REPORT_NAME
    pred_rows=[]
    for r in receipts:
        obj=r['result']
        pred_rows.append({'family':r['family'],'condition':r['condition'],'seed':r['seed'],
                          'subject_macro_balanced_accuracy':obj['subject_macro_balanced_accuracy'],
                          'subject_macro_roc_auc':obj['subject_macro_roc_auc'],
                          'global_balanced_accuracy':obj['global_balanced_accuracy'],
                          'global_roc_auc':obj['global_roc_auc'],
                          'confusion_matrix_json':json.dumps(obj['confusion_matrix'],separators=(',',':'))})
    tab=out/'tables'
    tab.mkdir(parents=True,exist_ok=True)
    write_csv_once(tab/'all_55_final_test_fits.csv',pred_rows,
                   ['family','condition','seed','subject_macro_balanced_accuracy',
                    'subject_macro_roc_auc','global_balanced_accuracy','global_roc_auc','confusion_matrix_json'])
    write_csv_once(tab/'all_11_condition_summary.csv',rows,
                   ['family','condition','n_seeds','subject_macro_ba_mean','subject_macro_ba_population_sd',
                    'subject_macro_ba_ci_low','subject_macro_ba_ci_high','subject_macro_auc_mean'])
    write_csv_once(tab/'per_subject_55_fits.csv',per_subject,
                   ['family','condition','seed','subject_index','balanced_accuracy','roc_auc'])
    write_csv_once(tab/'frozen_paired_subject_contrasts.csv',contrasts,
                   ['family','contrast_id','contrast_group','left','right','delta_right_minus_left',
                    'ci95_low','ci95_high','n_subjects','n_seeds'])
    fig_status={'status':'CREATED','files':create_figures(summary,out,frozen)}
    input_meta=load_json(out/INPUT_NAME)
    report={'schema':STAGE+'_report','status':'S2_77_V2_FINAL_TEST_COMPLETE',
            'reference':'s2_75B-v2 ONLY; 120 max epochs and best-Validation early stopping',
            'final_test_opened':True,'n_fits':55,'n_conditions':11,'n_families':3,
            'n_subjects':NSUB,'n_test_blocks':EXPECTED_BLOCKS,'n_test_windows':EXPECTED_WINDOWS,
            'n_test_EO':input_meta['EO_windows'],'n_test_EC':input_meta['EC_windows'],
            's2_76_freeze_sha256':S76_MANIFEST_SHA,
            'source_v2_training_sha256':S75B_V2_SHA,
            'test_input_commit_sha256':hash_file(out/INPUT_NAME),
            'source_test_index_sha256':input_meta['test_window_index_sha256'],
            'results':summary,'figures':fig_status,
            'paired_contrasts':contrasts,
            'bootstrap':{'repetitions':5000,'seed':20262401,'unit':'subject',
                         'matched_seeds_for_EEGNet':True,'CI':'percentile 95%',
                         'multiplicity_adjusted':False,'formal_confirmatory_significance':False},
            'scientific_limits':frozen['scientific_limits'],
            'Train_or_Validation_refit':False,'test_based_model_selection':False,
            'test_based_hyperparameter_tuning':False,
            'v1_24_epoch_models_used':False}
    create_once(out_report,report)
    done={'schema':STAGE+'_completion_receipt','status':'S2_77_V2_FINAL_TEST_COMPLETE',
          's2_76_freeze_sha256':S76_MANIFEST_SHA,
          'test_input_sha256':hash_file(out/INPUT_NAME),
          'test_report_sha256':hash_file(out_report),
          'n_test_predictions':55,
          'test_waveforms_opened_as_one_declared_evaluation':True,
          'no_retraining_or_test_based_selection':True}
    create_once(out/COMPLETE_NAME,done)
    print('S2_77_V2_FINAL_TEST_COMPLETE: 55/55 frozen model evaluations',flush=True)
    print('Final Test:',EXPECTED_BLOCKS,'blocks;',EXPECTED_WINDOWS,'usable windows',flush=True)
    print('REPORT:',out_report,flush=True)
    print('REPORT SHA256:',hash_file(out_report),flush=True)
    print('TABLES:',tab,flush=True)
    print('FIGURES:',len(fig_status['files']),flush=True)
    print('COMPLETION RECEIPT:',out/COMPLETE_NAME,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight',action='store_true',help='ZERO Test HDF5 access; verify all prior 55 fits and manifest')
    g.add_argument('--open-final-test',action='store_true',help='Explicit first & only declared Final Test evaluation')
    g.add_argument('--resume',action='store_true',help='Continue the SAME registered evaluation after interruption')
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    repo=args.repo_root.expanduser().resolve()
    try:
        require((repo/'scripts/data').is_dir(),'Not a Scenario-2 repository root')
        out=repo/OUT_REL
        s76,mod,ctx,frozen=validate_freeze(repo)
        test,mapping=test_rows(ctx,frozen)
        require(len(test)==EXPECTED_BLOCKS and len(mapping)==NSUB,'Bad signed Test split')
        complete=out/COMPLETE_NAME
        if args.preflight:
            print('S2_77_V2_PREFLIGHT_PASS: 55 saved v2 fits; s2_76 SHA256 verified',flush=True)
            print('Test blocks from locked CSV:',len(test),'Test usable windows:',EXPECTED_WINDOWS,flush=True)
            print('Final Test HDF5 opened: NO; Test waveforms read: NO; Models retrained: NO',flush=True)
            print('Access already initiated:',(out/ACCESS_NAME).exists(),flush=True)
            return 0
        if complete.exists():
            fin=load_json(complete)
            require(fin['status']=='S2_77_V2_FINAL_TEST_COMPLETE' and
                    hash_file(out/REPORT_NAME)==fin['test_report_sha256'] and
                    hash_file(out/INPUT_NAME)==fin['test_input_sha256'],
                    'Completed Test receipt/report/input hashes differ')
            print('S2_77_V2_ALREADY_FINALIZED — no repeat Test reading or evaluation permitted',flush=True)
            print('REPORT:',out/REPORT_NAME,flush=True)
            return 0
        with mod.writer_lock(out):
            marker_path=out/ACCESS_NAME
            marker={'schema':STAGE+'_access_receipt','s2_76_freeze_sha256':S76_MANIFEST_SHA,
                    'script_sha256':hash_file(Path(__file__)),
                    's2_75b_v2_source_sha256':S75B_V2_SHA,
                    'approved_tasks':55,'test_input_count_from_locked_split':EXPECTED_WINDOWS,
                    'access_purpose':'One complete read-only evaluation of all 55 previously frozen models',
                    'Final_Test_access_explicit':True,'training_and_tuning_permitted':False}
            if args.open_final_test:
                require(not marker_path.exists(),
                        'Final Test already opened. Use --resume ONLY to continue the existing registered run.')
                require(not (out/'predictions').exists() and not (out/'cache').exists(),
                        'Unregistered existing Test artifacts in output folder')
                create_once(marker_path,marker)
                print('S2_77_V2_SINGLE_FINAL_TEST_OPEN_AUTHORIZED — durable access receipt created',flush=True)
            else:
                require(marker_path.is_file() and not marker_path.is_symlink(),
                        '--resume requires a prior explicitly registered Test opening')
                require(canonical(load_json(marker_path))==canonical(marker),
                        'Final Test resumed under a changed script/protocol; forbidden')
                print('S2_77_V2_RESUMING_SAME_REGISTERED_EVALUATION',flush=True)
            z,features,test_input=read_and_cache_test(ctx,frozen,mod,out)
            require(test_input['n_windows']==EXPECTED_WINDOWS,'Test input count differs')
            for i,task in enumerate(frozen['tasks'],1):
                save_or_verify_predictions(ctx,mod,task,z,features,test_input,out)
                if i%5==0 or i==55:
                    print(f'  S2_77_TEST_EVALUATED {i}/55 model fits (no model selection)',flush=True)
            finalize(ctx,mod,frozen,out)
        return 0
    except KeyboardInterrupt:
        print('S2_77_V2_INTERRUPTED: same frozen Test evaluation may continue ONLY with --resume',
              file=sys.stderr)
        return 130
    except Exception as exc:
        print('S2_77_V2_FAILED: '+type(exc).__name__+': '+str(exc),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
