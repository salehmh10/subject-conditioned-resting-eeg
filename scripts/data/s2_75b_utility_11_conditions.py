r"""s2_75B — Train/Validation utility, original eight + three B0 conditions.

Practical entrypoints (from Scenario-2 repository root):

  .\.venv\Scripts\python.exe -u .\scripts\data\s2_75b_utility_11_conditions.py --preflight
  .\.venv\Scripts\python.exe -u .\scripts\data\s2_75b_utility_11_conditions.py --run-one --family LOGBANDPOWER_LOGREG --condition REAL_ONLY
  .\.venv\Scripts\python.exe -u .\scripts\data\s2_75b_utility_11_conditions.py --run-all
  .\.venv\Scripts\python.exe -u .\scripts\data\s2_75b_utility_11_conditions.py --aggregate

For a short, RESUMABLE EEGNet timing pilot, use --stop-at-updates 100 with
--run-one, --family EEGNETV4, --condition REAL_ONLY, --seed 20262301.
Omit --stop-at-updates to continue the SAME run to 8976 optimizer updates.

The frozen s2_72 original protocol, s2_74/s2_74D pools, the new s2_75A
amendment and the original s2_26/s2_27/s2_28/s2_29 classifier recipes are
checked before training.  Only eligible Real Train and Real Validation are
ever read.  Final Test remains SEALED; no Validation-based selection.

Cache files and models are NEW and separated from original outputs.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name,'2')

import argparse
import hashlib
import importlib.util
import io
import json
import math
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

VERSION = 's2_75b_11condition_utility_v1'
RUN_REL = Path('outputs/runs/scenario2/utility/eleven_condition_s2_75_v1')
AMPL = Path('outputs/protocols/scenario2/utility/b0_amended_11_condition_v1/manifests/s2_75a_amendment_v1.json')
PARENT = Path('outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1')
B0ROOT = Path('outputs/runs/scenario2/ablation/b0_state_only_1x_s2_74c_v1')
S74 = Path('outputs/frozen/scenario2/utility/b1_b2_pool_integrity_half_subset_v1/manifests/s2_74_freeze_manifest_v1.json')
S74D = Path('outputs/frozen/scenario2/ablation/b0_pool_integrity_half_subset_s2_74d_v1/manifests/s2_74d_b0_freeze_manifest_v1.json')
TRAIN_FEATS = Path('outputs/features/scenario2/real_train_logbandpower_v1.npz')
VAL_FEATS = Path('outputs/features/scenario2/real_validation_logbandpower_v1.npz')
SOURCE_HASHES = {
    's2_23_check_real_train_dataset.py':'e880eebee4dfeb79f41da809bbb4f8d6c7d233df68d972702f093f8aef4cde34',
    's2_26_train_logbandpower_features.py':'7127d5d29b26eb7031cb46e5739eb074c685f06f7a6d14db8e85395e8667c13f',
    's2_27_logbandpower_logreg_validation.py':'0fc2bdcef431af66ba3a983e13b9f069b94f8bd471ef1a0d01ffe16869d78b40',
    's2_28_tangent_svm_validation.py':'988959cf1f00366bab58d823d3758db824016fe7e54176774a8c7f0a97773d38',
    's2_29_eegnetv4_validation.py':'8838c5447a1ea5c605a7130e9eaeea0940a2ca3f5f92a1d1148dd3f3a9610fe2',
}
FAMILIES = ('LOGBANDPOWER_LOGREG','OAS_RIEMANN_LINEAR_SVM','EEGNETV4')
EEG_SEEDS = (20262301,20262302,20262303)
CLASSICAL_SEED = 20262301
NTRAIN,NVAL,NSUB,NC,NT,BS,UPDATES = 47835,9635,198,19,512,128,8976
ORIG_CONDS = ('REAL_ONLY','REAL_DUPLICATE_1X_CONTROL','TSTR_B1_1X','TSTR_B2_1X',
              'REAL_PLUS_B1_0P5X','REAL_PLUS_B1_1X','REAL_PLUS_B2_0P5X','REAL_PLUS_B2_1X')
B0_CONDS = ('TSTR_B0_1X','REAL_PLUS_B0_0P5X','REAL_PLUS_B0_1X')
STATES = ('EO','EC')


class UtilityError(RuntimeError):
    pass


def need(ok, why):
    if not ok: raise UtilityError(why)


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(',',':'), allow_nan=False).encode('utf-8')


def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load_json(p):
    with Path(p).open('r',encoding='utf8') as f: return json.load(f)


def write_bytes_atomic(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with t.open('xb') as f: f.write(b);f.flush();os.fsync(f.fileno())
        os.replace(t,p)
    finally:
        if t.exists():t.unlink()


def write_json_atomic(p,obj):
    write_bytes_atomic(p,canonical(obj)+b'\n')


@contextmanager
def writer_lock(root):
    """One training/aggregation writer at a time, including Windows RDP sessions."""
    root.mkdir(parents=True,exist_ok=True)
    p=root/'.writer.lock';f=p.open('a+b')
    if p.stat().st_size==0:f.write(b'0');f.flush()
    f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        f.close();raise UtilityError('Another s2_75B process is active') from exc
    try:yield
    finally:
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        f.close()


def imported(path, sha, alias):
    need(path.is_file() and sha_file(path)==sha,'Frozen baseline source mismatch: '+str(path))
    spec=importlib.util.spec_from_file_location(alias,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[alias]=mod;spec.loader.exec_module(mod)
    return mod


def safe_source(repo,rel,digest=None):
    p=(repo/rel).resolve()
    need(p.is_relative_to(repo) and p.is_file() and not p.is_symlink(), 'Unsafe/missing source: '+str(rel))
    if digest is not None:need(sha_file(p)==digest,'Frozen artifact changed: '+str(rel))
    return p


def pool_manifest_meta(repo, am):
    orig=am['source_artifacts']
    p12=load_json(safe_source(repo,S74,orig['s2_74_original_B1_B2_freeze']['sha256']))
    p0=load_json(safe_source(repo,S74D,orig['s2_74d_B0_freeze']['sha256']))
    gen_plan_rel=Path(p12['source_artifacts']['s2_73_generation_plan']['path'])
    plan=load_json(safe_source(repo,gen_plan_rel))
    pairs=plan['pairs']
    need(len(pairs)==396 and sum(x['expected_count'] for x in pairs)==NTRAIN,
         'Original pair plan differs')
    for si in range(NSUB):
        for j,st in enumerate(STATES):
            p=pairs[2*si+j]
            need(p['subject_index']==si and p['eye_state']==st and
                 p['eye_state_label']==j, 'Pair order differs')
    pool12=load_json(safe_source(repo,Path(p12['source_artifacts']['s2_73_pool_manifest']['path']),
                                  p12['source_artifacts']['s2_73_pool_manifest']['sha256']))
    pool0=load_json(safe_source(repo,Path(p0['source_artifacts']['B0_pool_manifest']['path']),
                                 p0['source_artifacts']['B0_pool_manifest']['sha256']))
    need(pool12['status']=='S2_73_FULL_POOL_COMPLETE' and pool0['status']=='S2_74C_B0_FULL_POOL_COMPLETE',
         'Pools not complete')
    need(len(pool12['committed'])==792 and len(pool0['committed'])==396,'Pool entries incomplete')
    master=load_json(safe_source(repo,Path(am['source_artifacts']['original_half_master']['path']),
                                 am['source_artifacts']['original_half_master']['sha256']))
    need(len(master['strata'])==396 and sum(x['retained_count'] for x in master['strata'])==23757,
         'Half subset contract changed')
    for g in ('B0','B1','B2'):
        view=load_json(safe_source(repo,Path(am['source_artifacts'][g+'_half_view']['path']),
                                   am['source_artifacts'][g+'_half_view']['sha256']))
        need(view['total_retained_windows']==23757 and len(view['strata'])==396,
             'Half subset malformed: '+g)
        for a,b in zip(master['strata'],view['strata']):
            need(a['subject_id']==b['subject_id'] and a['eye_state']==b['eye_state'] and
                 a['retained_pool_indices']==b['retained_pool_indices'],
                 'Half indices differ: '+g)
    return pairs, pool12, pool0, master


def context(repo):
    sys.path.insert(0,str(repo/'scripts/data'))
    # Verify the amendment, not merely a local copy, and every frozen source.
    amend_py=repo/'scripts/data/s2_75a_freeze_b0_utility_amendment.py'
    need(amend_py.is_file(), 'Missing s2_75A script')
    a=imported(amend_py,sha_file(amend_py),'_s2_75a_pinned')
    a.freeze(repo,verify_only=True)
    amend_path=safe_source(repo,AMPL)
    am=load_json(amend_path)
    need(am['status']=='S2_75A_11_CONDITION_AMENDMENT_FROZEN', 'Amendment status wrong')
    need(am['amendment_implementation_sha256']==sha_file(amend_py),
         'Amendment script changed after freeze')
    conditions=am['utility_conditions']
    need(tuple(x['id'] for x in conditions)==ORIG_CONDS+B0_CONDS,'Utility condition set differs')
    need(am['training']['EEGNet_fixed_optimizer_updates']==UPDATES,
         'EEGNet fixed update budget differs')
    src=repo/'scripts/data'
    for name,sha in SOURCE_HASHES.items():
        safe_source(repo,Path('scripts/data')/name,sha)
    proto=load_json(safe_source(repo,a.ORIG_PROTO_REL,a.ORIG_PROTO_SHA))
    assert proto['primary_comparisons']==am['primary_s2_72_comparisons_preserved_unmodified']
    # Verify immutable local Train scaler & split; never read Final Test EEG.
    split=repo/'outputs/metrics/scenario2/final_block_split_manifest.csv'
    norm=repo/'outputs/metrics/scenario2/train_normalization.json'
    need(sha_file(split)==proto['data_contract']['split_manifest']['sha256'] and
         sha_file(norm)==proto['data_contract']['normalization']['sha256'],
         'Locked split or scaler changed')
    pairs,pool12,pool0,master=pool_manifest_meta(repo,am)
    return {'repo':repo,'out':repo/RUN_REL,'am':am,'am_sha':sha_file(amend_path),
            'proto':proto,'pairs':pairs,'pool12':pool12,'pool0':pool0,'master':master,
            'split':split,'normalization':norm,'source_hashes':SOURCE_HASHES}


def import_baseline(ctx,name):
    path=ctx['repo']/'scripts/data'/name
    return imported(path,SOURCE_HASHES[name],'_s75_'+name[:-3])


def build_indices(ctx):
    """Read ONLY Train and Validation eligible metadata, not signal samples."""
    helper=import_baseline(ctx,'s2_23_check_real_train_dataset.py')
    orig=import_baseline(ctx,'s2_27_logbandpower_logreg_validation.py')
    split=pd.read_csv(ctx['split'])
    scaler=load_json(ctx['normalization'])
    need(scaler['status']=='TRAIN_NORMALIZATION_LOCKED','Scaler not frozen')
    need(len(scaler['means_uv'])==len(scaler['stds_uv'])==NC,'Wrong scaler channels')
    tr, mapping=helper.build_train_window_index(split,scaler)
    val, valmap=orig.build_validation_window_index(split)
    need(mapping==valmap and len(tr)==NTRAIN and len(val)==NVAL and len(mapping)==NSUB,
         'Train/Validation index mapping/count differs')
    # Verify the frozen train-window-index signature.  This is metadata only.
    cols=load_json(safe_source(ctx['repo'],Path('outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1/protocol/s2_73_generation_plan_v1.json')))['core']['train_window_index_columns']
    idx=[[getattr(r,c) for c in cols] for r in tr]
    sha=hashlib.sha256(canonical(idx)).hexdigest()
    genplan=load_json(ctx['repo']/PARENT/'protocol/s2_73_generation_plan_v1.json')
    need(sha==genplan['core']['train_window_index_sha256'], 'Train index differs from original generation')
    for p in ctx['pairs']:
        n=sum(r.subject_id==p['subject_id'] and r.eye_state==p['eye_state'] for r in tr)
        need(n==p['expected_count'],'Subject-state Real Train count changed')
    return tr,val,scaler


def read_windows(records,scaler,required_split,dest,validator=None):
    """Atomic normalized float32 .npy cache. Reads only requested split slices."""
    means=np.asarray(scaler['means_uv'],dtype=np.float64)
    stds=np.asarray(scaler['stds_uv'],dtype=np.float64)
    need(np.all(np.isfinite(means)) and np.all(stds>0) and np.all(np.isfinite(stds)),
         'Frozen scaler invalid')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name('.'+dest.name+'.'+uuid.uuid4().hex+'.tmp')
    current_path=None;h=None
    try:
        x=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(records),NC,NT))
        for i,r in enumerate(records):
            need(str(r.eye_state) in STATES,'Unexpected state')
            if r.h5_path!=current_path:
                if h is not None:h.close()
                h=h5py.File(r.h5_path,'r');current_path=r.h5_path
            block=h['blocks/'+r.block_uid]
            splitval=block.attrs['provisional_split']
            if isinstance(splitval,bytes):splitval=splitval.decode('utf8')
            need(str(splitval)==required_split,'Forbidden non-'+required_split+' read: '+r.block_uid)
            # Match the frozen s2_24/s2_74B loader: read raw as float32,
            # THEN normalize in float64 and cast to float32.
            v=np.asarray(block['eeg_uv'][:,r.start_sample:r.start_sample+NT],dtype=np.float32)
            need(v.shape==(NC,NT) and np.isfinite(v).all(), 'Invalid real EEG window')
            z=(v.astype(np.float64)-means[:,None])/stds[:,None]
            x[i]=z.astype(np.float32)
            if (i+1)%10000==0: print(f'    {required_split} {i+1}/{len(records)}',flush=True)
        x.flush()
        if validator is not None: validator(x)
        del x
        os.replace(tmp,dest)
    finally:
        if h is not None:h.close()
        if tmp.exists(): tmp.unlink()


def cache_array(ctx,source,kind,indices=None,scaler=None,records=None):
    """A separate immutable cache per source; re-audit hashes when loaded."""
    cache_memory=ctx.setdefault('_cache_memory',{})
    if source in cache_memory:return cache_memory[source]
    cache=ctx['out']/'cache'
    p=cache/f'{source}_z.npy'
    meta_p=cache/f'{source}_cache.json'
    n=NVAL if source=='Real_Validation' else NTRAIN
    expected={'schema':VERSION+'_cache','source':source,'length':n,
              'amendment_sha256':ctx['am_sha'], 'normalization_sha256':sha_file(ctx['normalization']),
              'split_manifest_sha256':sha_file(ctx['split'])}
    if source.startswith('B'):
        pool=ctx['pool0'] if source=='B0' else ctx['pool12']
        # Canonical receipt identity covers all pinned original shard SHA records.
        expected['committed_table_sha256']=hashlib.sha256(canonical(
            {k:v for k,v in pool['committed'].items() if k.startswith(source+'/')})).hexdigest()
    if p.exists() and meta_p.exists():
        meta=load_json(meta_p)
        need(all(meta.get(k)==v for k,v in expected.items()),'Cached metadata changed: '+source)
        need(p.stat().st_size>0 and sha_file(p)==meta.get('npy_sha256'),
             'Cached binary changed: '+source)
        arr=np.load(p,mmap_mode='r',allow_pickle=False)
        need(arr.shape==(n,NC,NT) and arr.dtype==np.float32, 'Cached shape changed')
        cache_memory[source]=(arr,meta)
        return arr,meta
    need(not p.exists() and not meta_p.exists(),
         'Orphan cache file: inspect before continuing '+source)
    if source.startswith('Real'):
        need(records is not None and scaler is not None,'Missing real cache inputs')
        validator=(lambda x:verify_original_train_hashes(ctx,x,records)) if source=='Real' else None
        read_windows(records,scaler,'Validation' if source=='Real_Validation' else 'Train',p,
                     validator=validator)
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            a=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(n,NC,NT))
            position=0
            pool=ctx['pool0'] if source=='B0' else ctx['pool12']
            root=ctx['repo']/(B0ROOT if source=='B0' else PARENT)
            for i,pair in enumerate(ctx['pairs'],1):
                key=f"{source}/{pair['subject_id']}/{pair['eye_state']}"
                ent=pool['committed'][key]
                shard=root/'generated'/source/pair['subject_id']/(pair['eye_state']+'.npz')
                need(shard.is_file() and not shard.is_symlink(), 'Missing synthetic '+str(shard))
                need(sha_file(shard)==ent['sha256'],'Modified synthetic shard '+key)
                with np.load(shard,allow_pickle=False) as z:
                    v=z['x_z'];si=z['sample_indices']
                    need(v.shape==(pair['expected_count'],NC,NT) and v.dtype==np.float32 and
                         np.isfinite(v).all(), 'Bad synthetic x_z: '+key)
                    need(np.array_equal(si,np.arange(len(v),dtype=np.int64)),
                         'Synthetic sample indices changed: '+key)
                    a[position:position+len(v)]=v
                    position+=len(v)
                if i%40==0 or i==len(ctx['pairs']):
                    print(f'    {source} {i}/396 shards',flush=True)
            need(position==NTRAIN,'Wrong synthetic pool total')
            a.flush();del a
            os.replace(tmp,p)
        finally:
            if tmp.exists():tmp.unlink()
    meta={**expected,'npy_sha256':sha_file(p)}
    write_json_atomic(meta_p,meta)
    cache_memory[source]=(np.load(p,mmap_mode='r',allow_pickle=False),meta)
    return cache_memory[source]


def verify_original_train_hashes(ctx, x, records):
    """Reproduce 396 pinned Real-Train SHA values from s2_74B; no Test read."""
    by={(p['subject_id'],p['eye_state']):[] for p in ctx['pairs']}
    for i,r in enumerate(records):
        by[(r.subject_id,r.eye_state)].append(i)
    for si in range(NSUB):
        sid=ctx['pairs'][2*si]['subject_id']
        p=ctx['repo']/Path('outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1/subjects')/(sid+'.json')
        original=load_json(safe_source(ctx['repo'],p.relative_to(ctx['repo'])))
        for st in STATES:
            z=np.ascontiguousarray(x[np.asarray(by[(sid,st)],dtype=np.int64)])
            head=hashlib.sha256(canonical({'shape':list(z.shape),'dtype':str(z.dtype)}))
            head.update(z.tobytes())
            digest=head.hexdigest()
            need(digest==original['input_hashes'][st]['real_train_z_sha256'],
                 'Real Train changed relative to s2_74B, subject/state: '+sid+'/'+st)
    print('  Real Train: all 396 original s2_74B waveform SHA256 values match',flush=True)


def labels(ctx,real_train,real_val):
    real_y=np.array([r.label for r in real_train],dtype=np.int64)
    val_y=np.array([r.label for r in real_val],dtype=np.int64)
    val_sub=np.array([r.subject_index for r in real_val],dtype=np.int64)
    syn_y=np.concatenate([np.full(p['expected_count'],p['eye_state_label'],dtype=np.int64)
                          for p in ctx['pairs']])
    need(len(syn_y)==len(real_y)==NTRAIN and len(val_y)==NVAL and
         set(np.unique(real_y))==set(np.unique(syn_y))==set(np.unique(val_y))=={0,1} and
         len(np.unique(val_sub))==NSUB,'Label/index contract differs')
    return {'Real':real_y,'Real_Validation':val_y,'B0':syn_y,'B1':syn_y,'B2':syn_y,'val_sub':val_sub}


def half_indices(master,pairs):
    need(len(master['strata'])==len(pairs)==396,'Half index malformed')
    inds=[];offset=0
    for p,row in zip(pairs,master['strata']):
        n=p['expected_count'];kk=n//2
        sel=np.asarray(row['retained_pool_indices'],dtype=np.int64)
        need(row['subject_id']==p['subject_id'] and row['eye_state']==p['eye_state'] and
             row['full_count']==n and len(sel)==kk and
             (not kk or (sel[0]>=0 and sel[-1]<n)) and
             np.array_equal(sel,np.unique(sel)), 'Invalid shared half indices')
        inds.append(offset+sel);offset+=n
    out=np.concatenate(inds)
    need(len(out)==23757 and offset==NTRAIN,'Half population mismatch')
    return out


def training_parts(ctx,cond,lab):
    """Tuple (source, indices), synthetic 0.5x from identical frozen indices."""
    ident=cond['id'];g=cond.get('synthetic_generator')
    real=np.arange(NTRAIN,dtype=np.int64)
    if ident=='REAL_ONLY':parts=[('Real',real)]
    elif ident=='REAL_DUPLICATE_1X_CONTROL':parts=[('Real',real),('Real',real)]
    elif ident.startswith('TSTR_'):parts=[(g,real)]
    elif ident.startswith('REAL_PLUS_'):
        synidx=half_indices(ctx['master'],ctx['pairs']) if cond['synthetic_ratio']==0.5 else real
        parts=[('Real',real),(g,synidx)]
    else:raise UtilityError('Unexpected condition: '+ident)
    total=sum(len(ii) for _,ii in parts)
    need(total=={'REAL_ONLY':NTRAIN,'REAL_DUPLICATE_1X_CONTROL':2*NTRAIN}.get(
         ident, NTRAIN if ident.startswith('TSTR_') else (NTRAIN+23757 if cond['synthetic_ratio']==0.5 else 2*NTRAIN)),
         'Training set size mismatch')
    yy=np.concatenate([lab[source][ix] for source,ix in parts])
    need(set(np.unique(yy))=={0,1},'Training condition has one class')
    return parts,yy


def unique_sources(parts):
    return sorted({s for s,_ in parts})


def stacked_parts(source_arrays,parts,kind=None):
    arrays=[np.asarray(source_arrays[s][ix],dtype=np.float64 if kind=='float64' else np.float32)
            for s,ix in parts]
    return np.concatenate(arrays,axis=0)


def make_features(xz,scaler):
    from scipy.signal import welch
    eps=1e-12
    uv=np.asarray(xz,dtype=np.float64)*np.asarray(scaler['stds_uv'])[None,:,None]+\
       np.asarray(scaler['means_uv'])[None,:,None]
    f,psd=welch(uv,fs=128.0,window='hann',nperseg=256,noverlap=128,
                detrend='constant',scaling='density',axis=-1)
    vals=[]
    for band,(lo,hi) in [('delta',(1.,4.)),('theta',(4.,8.)),('alpha',(8.,13.)),('beta',(13.,30.))]:
        mask=(f>=lo)&((f<=hi) if band=='beta' else (f<hi))
        integrate=getattr(np,'trapezoid',None) or np.trapz
        bp=integrate(psd[...,mask],x=f[mask],axis=-1)
        need(np.isfinite(bp).all() and np.all(bp>0), 'Invalid feature band power')
        vals.append(np.log10(np.maximum(bp,eps)))
    out=np.concatenate(vals,axis=1).astype(np.float32)
    need(out.shape==(len(xz),76) and np.isfinite(out).all(),'Feature shape invalid')
    return out


def cached_features(ctx,src,x,scaler):
    feature_memory=ctx.setdefault('_feature_memory',{})
    if src in feature_memory:return feature_memory[src]
    root=ctx['out']/'features';p=root/(src+'_logbp.npy'); meta=root/(src+'_logbp.json')
    original=ctx['out']/'cache'/(src+'_cache.json')
    need(original.is_file(),'Missing source array cache receipt')
    expected={'source':src,'version':VERSION,'cache_meta_sha256':sha_file(original),
              'feature_recipe_sha256':SOURCE_HASHES['s2_26_train_logbandpower_features.py']}
    if src=='Real' or src=='Real_Validation':
        orig=ctx['repo']/(TRAIN_FEATS if src=='Real' else VAL_FEATS)
        need(orig.is_file(),'Missing original real LogBandpower feature package '+str(orig))
        expected['original_real_feature_package_sha256']=sha_file(orig)
    if p.exists() and meta.exists():
        m=load_json(meta)
        need(all(m.get(k)==v for k,v in expected.items()) and sha_file(p)==m.get('sha256'),
             'Changed feature cache '+src)
        feature_memory[src]=np.load(p,mmap_mode='r',allow_pickle=False)
        return feature_memory[src]
    need(not p.exists() and not meta.exists(),'Orphan feature cache '+src)
    root.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(x),76))
        if src in ('Real','Real_Validation'):
            # Reuse exactly the earlier s2_26/s2_27 REAL features instead of
            # reconstituting uV from rounded train-z. Preserve bitwise baseline.
            with np.load(orig,allow_pickle=False) as z:
                f=z['X'];y=z['y'];subjects=z['subject_index']
                expected_rec=ctx['_indices'][0 if src=='Real' else 1]
                need(f.shape==(len(x),76) and f.dtype==np.float32 and
                     len(y)==len(subjects)==len(x) and np.isfinite(f).all(),
                     'Original baseline feature schema invalid')
                need(np.array_equal(y,np.asarray([r.label for r in expected_rec])) and
                     np.array_equal(subjects,np.asarray([r.subject_index for r in expected_rec])),
                     'Original feature labels or row order do not match frozen index')
                need(len(np.unique(subjects))==NSUB,'Original feature subjects invalid')
                arr[:]=f
        else:
            for start in range(0,len(x),256):
                end=min(start+256,len(x));arr[start:end]=make_features(x[start:end],scaler)
        arr.flush();del arr;os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()
    write_json_atomic(meta,{**expected,'sha256':sha_file(p)})
    feature_memory[src]=np.load(p,mmap_mode='r',allow_pickle=False)
    return feature_memory[src]


def cached_covariance(ctx,src,x):
    cov_memory=ctx.setdefault('_cov_memory',{})
    if src in cov_memory:return cov_memory[src]
    from pyriemann.estimation import Covariances
    root=ctx['out']/'covariance';p=root/(src+'_oas.npy');meta=root/(src+'_oas.json')
    original=ctx['out']/'cache'/(src+'_cache.json')
    expected={'source':src,'version':VERSION,'cache_meta_sha256':sha_file(original),
              'covariance_recipe_sha256':SOURCE_HASHES['s2_28_tangent_svm_validation.py']}
    if p.exists() and meta.exists():
        m=load_json(meta)
        need(all(m.get(k)==v for k,v in expected.items()) and sha_file(p)==m.get('sha256'),
             'Changed covariance cache '+src)
        cov_memory[src]=np.load(p,mmap_mode='r',allow_pickle=False)
        return cov_memory[src]
    need(not p.exists() and not meta.exists(),'Orphan covariance cache '+src)
    root.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float64,shape=(len(x),NC,NC))
        est=Covariances(estimator='oas')
        for start in range(0,len(x),256):
            end=min(start+256,len(x));arr[start:end]=est.transform(np.asarray(x[start:end],dtype=np.float64))
        arr.flush();del arr;os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()
    write_json_atomic(meta,{**expected,'sha256':sha_file(p)})
    cov_memory[src]=np.load(p,mmap_mode='r',allow_pickle=False)
    return cov_memory[src]


def perf_scores(y,pred,score,subjects):
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score,confusion_matrix
    y=np.asarray(y,dtype=np.int64);pred=np.asarray(pred,dtype=np.int64)
    score=np.asarray(score,dtype=np.float64);subjects=np.asarray(subjects,dtype=np.int64)
    need(y.shape==pred.shape==score.shape==subjects.shape==(NVAL,), 'Prediction shape invalid')
    need(np.isfinite(score).all() and set(np.unique(y))=={0,1},'Invalid labels/scores')
    sba=[];sauc=[]
    for i in range(NSUB):
        mask=subjects==i
        need(np.any(y[mask]==0) and np.any(y[mask]==1),'Validation subject lacks both states')
        sba.append(float(balanced_accuracy_score(y[mask],pred[mask])))
        sauc.append(float(roc_auc_score(y[mask],score[mask])))
    return {'subject_macro_balanced_accuracy':float(np.mean(sba)),
            'subject_macro_roc_auc':float(np.mean(sauc)),
            'global_balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'global_roc_auc':float(roc_auc_score(y,score)),
            'subject_ba':sba,'subject_auc':sauc,
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def descriptor(ctx,family,cond,seed,parts):
    source_meta={s:sha_file(ctx['out']/'cache'/(s+'_cache.json')) for s in unique_sources(parts)+['Real_Validation']}
    return {'version':VERSION,'family':family,'condition':cond['id'],'seed':seed,
            's2_75a_amendment_sha256':ctx['am_sha'],
            'classifier_source_sha256':SOURCE_HASHES[{'LOGBANDPOWER_LOGREG':'s2_27_logbandpower_logreg_validation.py',
                                                    'OAS_RIEMANN_LINEAR_SVM':'s2_28_tangent_svm_validation.py',
                                                    'EEGNETV4':'s2_29_eegnetv4_validation.py'}[family]],
            'cache_manifest_sha256':source_meta,
            'code_sha256':sha_file(Path(__file__))}


def task_paths(ctx,family,cond,seed):
    folder=ctx['out']/'runs'/family/cond
    stem=str(seed)
    return {'folder':folder,'receipt':folder/(stem+'_receipt.json'),
            'model':folder/(stem+('.pt' if family=='EEGNETV4' else '.joblib')),
            'pred':folder/(stem+'_validation_predictions.npz'),
            'resume':folder/(stem+'_resume.pt')}


def existing_task(paths,desc):
    receipt=paths['receipt']
    if not receipt.exists():
        need(not paths['model'].exists() and not paths['pred'].exists(),
             'Orphan final artifact; investigate before retrying: '+str(paths['folder']))
        return None
    r=load_json(receipt)
    need(r['descriptor']==desc and r['status']=='S2_75B_TASK_COMPLETE',
         'Completed task differs from frozen specification')
    for k in ('model','pred'):
        need(paths[k].is_file() and sha_file(paths[k])==r['files'][k]['sha256'],
             'Frozen model/prediction changed: '+str(paths[k]))
    return r


def save_joblib_atomic(obj,path):
    import joblib
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        joblib.dump(obj,tmp,compress=0)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def save_torch_atomic(obj,path):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(obj,tmp);os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def save_task(ctx,family,cond,seed,desc,paths,model_obj,y,pred,score,subjects,extra):
    result=perf_scores(y,pred,score,subjects)
    # Enforce no existing final file is ever overwritten.
    need(not any(paths[k].exists() for k in ('model','pred','receipt')),
         'Refusing to overwrite an existing final task')
    paths['folder'].mkdir(parents=True,exist_ok=True)
    if family=='EEGNETV4':save_torch_atomic(model_obj,paths['model'])
    else:save_joblib_atomic(model_obj,paths['model'])
    out=io.BytesIO()
    np.savez_compressed(out,y_true=y.astype(np.int64),y_pred=pred.astype(np.int64),
                        y_score=score.astype(np.float64),subject_index=subjects.astype(np.int64))
    write_bytes_atomic(paths['pred'],out.getvalue())
    receipt={'schema':VERSION+'_task_receipt','status':'S2_75B_TASK_COMPLETE',
             'descriptor':desc,'result':result,
             'method':extra,'Validation_waveforms_accessed':True,
             'Final_Test_waveforms_accessed':False,
             'files':{k:{'path':paths[k].relative_to(ctx['repo']).as_posix(),
                         'sha256':sha_file(paths[k])} for k in ('model','pred')}}
    write_json_atomic(paths['receipt'],receipt)
    print(f"S2_75B_TASK_COMPLETE {family}/{cond['id']}/{seed} "+
          f"subject-macro BA={result['subject_macro_balanced_accuracy']:.6f}",flush=True)
    print('RECEIPT:',paths['receipt'],flush=True)
    return receipt


def classical_train(ctx,family,cond,seed,parts,yy,arrays,lab,scaler,desc,paths):
    from sklearn.preprocessing import StandardScaler
    if family=='LOGBANDPOWER_LOGREG':
        from sklearn.linear_model import LogisticRegression
        ff={s:cached_features(ctx,s,arrays[s],scaler) for s in unique_sources(parts)+['Real_Validation']}
        x=stacked_parts(ff,parts,'float64')
        xv=np.asarray(ff['Real_Validation'],dtype=np.float64)
        learned=StandardScaler(with_mean=True,with_std=True)
        xs=learned.fit_transform(x)
        xv=learned.transform(xv)
        model=LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',max_iter=2000,
                                 class_weight=None,fit_intercept=True,random_state=20260907)
        model.fit(xs,yy)
        score=model.predict_proba(xv)[:,1]
        pred=model.predict(xv)
        fitted={'scaler':learned,'classifier':model,'feature_definition':'original s2_26/27 (uV Welch log10 bandpower)'}
        extra={'train_rows':len(yy),'feature_scaler_fit':'condition Train only',
               'threshold':'original classifier predict; no tuning'}
    else:
        from pyriemann.tangentspace import TangentSpace
        from sklearn.svm import LinearSVC
        cc={s:cached_covariance(ctx,s,arrays[s]) for s in unique_sources(parts)+['Real_Validation']}
        x=stacked_parts(cc,parts,'float64')
        xv=np.asarray(cc['Real_Validation'],dtype=np.float64)
        tangent=TangentSpace(metric='riemann')
        xs=tangent.fit_transform(x)
        xvs=tangent.transform(xv)
        model=LinearSVC(C=1.0,max_iter=10000,class_weight=None,random_state=20260907)
        model.fit(xs,yy)
        score=model.decision_function(xvs)
        pred=model.predict(xvs)
        fitted={'tangent':tangent,'classifier':model,'covariance':'original OAS 19x19'}
        extra={'train_rows':len(yy),'tangent_reference_fit':'condition Train only',
               'decision_boundary':0.0,'new_feature_scaler':False}
    return save_task(ctx,family,cond,seed,desc,paths,fitted,lab['Real_Validation'],
                     pred,score,lab['val_sub'],extra)


def next_torch_shuffled_batch(active_perm,shuffle_rng,pos,batch):
    """Torch DataLoader-style full-cycle shuffle, resumable from saved state."""
    import torch
    n=len(active_perm)
    need(n>0 and 0<=pos<n and batch>0,'Invalid batch cursor')
    end=min(pos+batch,n)
    ix=active_perm[pos:end].numpy()
    if end==n:
        return ix,torch.randperm(n,generator=shuffle_rng),0,True
    return ix,active_perm,end,False


def gather_x(arrays,parts,idx):
    """Load a mixed batch from memory-mapped, preverified source caches."""
    n=len(idx);out=np.empty((n,NC,NT),dtype=np.float32)
    starts=[];ofs=0
    for s,ii in parts:starts.append((ofs,ofs+len(ii),s,ii));ofs+=len(ii)
    need(int(idx.min())>=0 and int(idx.max())<ofs,'Batch source index outside training pool')
    for lo,hi,s,ii in starts:
        which=np.flatnonzero((idx>=lo)&(idx<hi))
        if which.size:out[which]=arrays[s][ii[idx[which]-lo]]
    return out


def eegnet_train(ctx,cond,seed,parts,yy,arrays,lab,desc,paths,
                 stop_at_updates=None,checkpoint_every=400):
    import torch
    from torch import nn
    need(torch.cuda.is_available(),'GPU/CUDA required for frozen EEGNet baseline')
    original=import_baseline(ctx,'s2_29_eegnetv4_validation.py')
    need(original.BATCH_SIZE==BS and original.LEARNING_RATE==1e-3 and
         original.WEIGHT_DECAY==0 and original.DROPOUT==0.50 and
         original.EXPECTED_TRAIN_WINDOWS==NTRAIN,
         'Original EEGNet recipe changed')
    original.set_seed(seed)
    device=torch.device('cuda')
    model=original.EEGNetV4Baseline().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=original.LEARNING_RATE,
                               weight_decay=original.WEIGHT_DECAY)
    criterion=nn.CrossEntropyLoss()
    n=len(yy);cycle=0;pos=0;updates=0
    # The original EEGNet baseline shuffled with a torch.Generator. Keep a
    # dedicated, ONE-TIME-seeded generator for pool reshuffles. Save its state
    # and the active permutation for exact mid-cycle resume.
    shuffle_rng=torch.Generator(device='cpu')
    shuffle_rng.manual_seed(seed)
    perm=torch.randperm(n,generator=shuffle_rng)
    cp=paths['resume']
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        need(state['descriptor']==desc and state['total_updates']==UPDATES and
             state['seed']==seed and state['train_count']==n,
             'Existing mid-run checkpoint does not match frozen training contract')
        model.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        torch.set_rng_state(state['torch_cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng_states'])
        shuffle_rng.set_state(state['shuffle_rng_state'])
        perm=state['active_perm']
        updates=state['updates'];cycle=state['cycle'];pos=state['position']
        need(0<=updates<=UPDATES and cycle>=0 and 0<=pos<n,'Invalid checkpoint position')
        need(tuple(perm.shape)==(n,) and perm.dtype==torch.int64,
             'Checkpoint permutation does not match training pool')
        print(f'  EEGNet resume at update {updates}/{UPDATES}',flush=True)
    end=UPDATES if stop_at_updates is None else min(UPDATES,stop_at_updates)
    need(end>=updates,'--stop-at-updates is before existing checkpoint: raise target')
    model.train();started=time.perf_counter()
    while updates<end:
        ii,perm,pos,newcycle=next_torch_shuffled_batch(perm,shuffle_rng,pos,BS)
        if newcycle:cycle+=1
        bx=torch.from_numpy(gather_x(arrays,parts,ii)).to(device)
        by=torch.from_numpy(yy[ii]).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(bx);loss=criterion(logits,by)
        need(torch.isfinite(loss).item(),'Nonfinite EEGNet loss')
        loss.backward();optimizer.step();model.apply_max_norm()
        updates+=1
        if updates%100==0 or updates==end:
            print(f'  EEGNet {cond["id"]} seed={seed} updates={updates}/{UPDATES} '
                  f'loss={loss.item():.5f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        if updates%checkpoint_every==0 or updates==end:
            state={'schema':VERSION+'_resumable_optimization','descriptor':desc,
                   'train_count':n,'seed':seed,'total_updates':UPDATES,'updates':updates,
                   'cycle':cycle,'position':pos,'model_state_dict':model.state_dict(),
                   'optimizer_state_dict':optimizer.state_dict(),
                   'torch_cpu_rng':torch.get_rng_state(),
                   'cuda_rng_states':torch.cuda.get_rng_state_all(),
                   'shuffle_rng_state':shuffle_rng.get_state(),
                   'active_perm':perm}
            save_torch_atomic(state,cp)
    if updates<UPDATES:
        print(f'S2_75B_EEGNET_PARTIAL_SAVED updates={updates}/{UPDATES}; '
              'rerun SAME condition and seed without --stop-at-updates',flush=True)
        return None
    model.eval()
    val=arrays['Real_Validation'];pred=[];scores=[]
    with torch.no_grad():
        for i in range(0,NVAL,BS):
            x=torch.from_numpy(np.array(val[i:i+BS],copy=True)).to(device)
            z=model(x)
            prob=torch.softmax(z,dim=1)[:,1]
            scores.append(prob.cpu().numpy());pred.append(torch.argmax(z,dim=1).cpu().numpy())
    p=np.concatenate(pred);sc=np.concatenate(scores)
    frozen={'schema':VERSION+'_EEGNet_final_state','model_state_dict':{k:v.detach().cpu()
             for k,v in model.state_dict().items()},'condition':cond['id'],
            'seed':seed,'updates':updates,'baseline_architecture_sha256':SOURCE_HASHES['s2_29_eegnetv4_validation.py'],
            'descriptor':desc,'checkpoint_selection':'fixed final update; no Validation selection'}
    result=save_task(ctx,'EEGNETV4',cond,seed,desc,paths,frozen,lab['Real_Validation'],
                     p,sc,lab['val_sub'],{'train_rows':n,'batch_size':BS,'fixed_updates':updates,
                     'optimizer':'Adam(lr=1e-3,weight_decay=0)','max_norm_after_every_step':True,
                     'Validation_early_stopping':False,'Final_Test_accessed':False})
    # The complete/frozen final state has its own SHA receipt; the mid-run
    # checkpoint may now be discarded, but leaving it is safer until review.
    return result


def run_one(ctx,condition,family,seed,stop_at_updates=None,checkpoint_every=400):
    cond=next(x for x in ctx['am']['utility_conditions'] if x['id']==condition)
    need(family in FAMILIES,'Unknown classifier')
    need(seed in (EEG_SEEDS if family=='EEGNETV4' else (CLASSICAL_SEED,)),
         'Invalid frozen classifier seed')
    if stop_at_updates is not None:need(family=='EEGNETV4','Stop-update target is only for EEGNet')
    # Build indices and data only after zero-EEG preflight has passed.
    if '_indices' not in ctx:
        ctx['_indices']=build_indices(ctx)
    tr,val,scaler=ctx['_indices']
    if '_labels' not in ctx:
        ctx['_labels']=labels(ctx,tr,val)
    lab=ctx['_labels']
    parts,yy=training_parts(ctx,cond,lab)
    arr={}
    for s in unique_sources(parts)+['Real_Validation']:
        arr[s],_=cache_array(ctx,s,'windows',scaler=scaler,
                             records=(tr if s=='Real' else val if s=='Real_Validation' else None))
    desc=descriptor(ctx,family,cond,seed,parts)
    paths=task_paths(ctx,family,cond['id'],seed)
    old=existing_task(paths,desc)
    if old:
        print('S2_75B_REUSED_VERIFIED',family,condition,seed)
        return old
    print(f'RUN {family}/{condition}/{seed}: Train rows={len(yy):,} '
          f'Validation rows={NVAL:,}; Final Test SEALED',flush=True)
    if family=='EEGNETV4':
        return eegnet_train(ctx,cond,seed,parts,yy,arr,lab,desc,paths,
                            stop_at_updates,checkpoint_every)
    return classical_train(ctx,family,cond,seed,parts,yy,arr,lab,scaler,desc,paths)


def schedule(ctx):
    return [(family,c['id'],seed) for family in FAMILIES
            for c in ctx['am']['utility_conditions']
            for seed in (EEG_SEEDS if family=='EEGNETV4' else (CLASSICAL_SEED,))]


def aggregate(ctx):
    allruns={};rows=[]
    tasks=schedule(ctx)
    for family,cond,seed in tasks:
        paths=task_paths(ctx,family,cond,seed)
        need(paths['receipt'].is_file(),'Missing complete condition/seed: '+f'{family}/{cond}/{seed}')
        r=load_json(paths['receipt'])
        # Integrity-check all classifier/prediction receipts before statistics.
        need(r['status']=='S2_75B_TASK_COMPLETE' and r['descriptor']['s2_75a_amendment_sha256']==ctx['am_sha'] and
             r['descriptor']['code_sha256']==sha_file(Path(__file__)),
             'A classifier has a different frozen analysis contract')
        for f in ('model','pred'):
            need(sha_file(paths[f])==r['files'][f]['sha256'],'Classifier/prediction changed')
        with np.load(paths['pred'],allow_pickle=False) as pred:
            stats=perf_scores(pred['y_true'],pred['y_pred'],pred['y_score'],pred['subject_index'])
        need(all(np.allclose(stats[k],r['result'][k],atol=1e-12) for k in
                 ('subject_macro_balanced_accuracy','subject_macro_roc_auc',
                  'global_balanced_accuracy','global_roc_auc')),
             'Saved metrics mismatch recomputed predictions')
        key=(family,cond,seed);allruns[key]=r
        rows.append({'family':family,'condition':cond,'seed':seed,
                     **{k:v for k,v in stats.items() if isinstance(v,float)}})

    summary={};comparison={}
    rng=np.random.RandomState(ctx['proto']['statistics']['bootstrap_seed'])
    bidx=rng.randint(0,NSUB,size=(ctx['proto']['statistics']['bootstrap_repetitions'],NSUB))
    for family in FAMILIES:
        seeds=EEG_SEEDS if family=='EEGNETV4' else (CLASSICAL_SEED,)
        summary[family]={}
        for cond in ORIG_CONDS+B0_CONDS:
            rr=[allruns[(family,cond,seed)]['result'] for seed in seeds]
            keys=('subject_macro_balanced_accuracy','subject_macro_roc_auc','global_balanced_accuracy','global_roc_auc')
            summary[family][cond]={key:{'mean':float(np.mean([r[key] for r in rr])),
                                        'population_sd':float(np.std([r[key] for r in rr],ddof=0)),
                                        'n_seeds':len(seeds)} for key in keys}
        comparison[family]={}
        cdefs=ctx['am']['primary_s2_72_comparisons_preserved_unmodified']+ctx['am']['secondary_supplementary_comparisons']
        for c in cdefs:
            # Each subject bootstrap replicate uses THE SAME 198 sampled
            # subjects across matched conditions/seeds, as frozen by s2_72.
            da=np.mean([np.array(allruns[(family,c['left'],seed)]['result']['subject_ba'])
                        for seed in seeds],axis=0)
            db=np.mean([np.array(allruns[(family,c['right'],seed)]['result']['subject_ba'])
                        for seed in seeds],axis=0)
            delta=da-db
            boot=delta[bidx].mean(axis=1)
            comparison[family][c['id']]={'left':c['left'],'right':c['right'],
                    'status':'original_s2_72_primary' if c in ctx['am']['primary_s2_72_comparisons_preserved_unmodified']
                             else 'supplementary_pre_utility_amendment',
                    'point_subject_macro_ba_difference':float(delta.mean()),
                    'paired_95pct_subject_bootstrap_ci':np.quantile(boot,[0.025,0.975]).tolist(),
                    'bootstrap_repetitions':len(bidx),'n_subjects':NSUB,
                    'eegnet_seeds':list(seeds)}
    out=ctx['out']/'final_validation'
    out.mkdir(parents=True,exist_ok=True)
    import csv
    fd=io.StringIO(newline='');w=csv.DictWriter(fd,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_bytes_atomic(out/'all_55_classifier_validation_runs.csv',fd.getvalue().encode('utf-8-sig'))
    report={'schema':VERSION+'_complete_validation_report','status':'S2_75B_VALIDATION_UTILITY_COMPLETE',
            'original_primary_compared_separately_from_supplementary':True,
            'n_conditions':11,'n_tasks':len(tasks),'n_classifier_families':3,
            'metric_subject_macro_ba':summary,'paired_subject_bootstrap_comparisons':comparison,
            'source_amendment_sha256':ctx['am_sha'],
            'Validation_is_development_evidence_not_independent_test':True,
            'Final_Test_waveforms_accessed':False,
            'no_utility_model_chosen_based_on_validation':True}
    report['figures']=make_utility_figures(ctx,summary,comparison)
    write_json_atomic(out/'s2_75b_complete_validation_report_v1.json',report)
    print('S2_75B_VALIDATION_UTILITY_COMPLETE: 55 classifier fits verified; Final Test SEALED')
    print('REPORT:',out/'s2_75b_complete_validation_report_v1.json')
    return report


def make_utility_figures(ctx,summary,comparison):
    """Report-level visualizations; never use Validation to select conditions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'status':'SKIPPED_MATPLOTLIB_UNAVAILABLE','files':[]}
    dest=ctx['out']/'final_validation'/'figures';dest.mkdir(parents=True,exist_ok=True)
    files=[]
    def save(fig,name):
        fig.tight_layout()
        b=io.BytesIO();fig.savefig(b,format='png',dpi=170,bbox_inches='tight')
        plt.close(fig)
        p=dest/name;write_bytes_atomic(p,b.getvalue())
        files.append(p.relative_to(ctx['out']).as_posix())
    try:
        conditions=ORIG_CONDS+B0_CONDS
        fig,ax=plt.subplots(figsize=(13,9))
        yy=np.arange(len(conditions))
        for j,family in enumerate(FAMILIES):
            means=[summary[family][c]['subject_macro_balanced_accuracy']['mean'] for c in conditions]
            sd=[summary[family][c]['subject_macro_balanced_accuracy']['population_sd'] for c in conditions]
            ax.barh(yy+(j-1)*0.24,means,height=0.23,xerr=sd,capsize=2,label=family)
        ax.set_yticks(yy,conditions)
        ax.invert_yaxis();ax.set_xlim(0,1)
        ax.set(xlabel='Real Validation subject-macro balanced accuracy',
               title='All 11 frozen conditions, three classifiers (EEGNet three-seed population SD)')
        ax.legend(loc='lower right',fontsize=8)
        ax.grid(axis='x',alpha=.2)
        save(fig,'all_11_condition_subject_macro_BA.png')

        # Original primary + explicitly supplementary contrasts remain distinct.
        for section in ('original_s2_72_primary','supplementary_pre_utility_amendment'):
            rows=[(name,c) for name,c in comparison['EEGNETV4'].items() if c['status']==section]
            if not rows:continue
            labels=[name for name,_ in rows]
            vals=np.array([c['point_subject_macro_ba_difference'] for _,c in rows])
            cis=np.array([c['paired_95pct_subject_bootstrap_ci'] for _,c in rows])
            err=np.vstack([vals-cis[:,0],cis[:,1]-vals])
            fig,ax=plt.subplots(figsize=(10,max(3.8,0.65*len(rows)+1.9)))
            yy=np.arange(len(rows))
            ax.errorbar(vals,yy,xerr=err,fmt='o',capsize=3)
            ax.axvline(0,linestyle='--',linewidth=1)
            ax.set_yticks(yy,labels);ax.invert_yaxis()
            ax.set(xlabel='Paired subject-macro BA difference (mean across three matched EEGNet seeds)',
                   title=('Original predeclared s2_72 comparisons' if section.startswith('original')
                          else 'Supplementary B0 comparisons fixed before utility'))
            ax.grid(axis='x',alpha=.2)
            save(fig,section+'_EEGNet_subject_bootstrap.png')
        return {'status':'CREATED','files':files}
    except Exception as exc:
        plt.close('all')
        return {'status':'PLOT_ERROR_QUANT_RESULTS_RETAINED','files':files,'error':str(exc)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true',help='Metadata-only checks: NO EEG reading or training')
    group.add_argument('--run-one',action='store_true',help='One complete or resumable fit')
    group.add_argument('--run-all',action='store_true',help='All 55 predeclared fits; skips complete ones')
    group.add_argument('--aggregate',action='store_true',help='Verify 55 fits and make Validation report')
    p.add_argument('--family',choices=FAMILIES)
    p.add_argument('--condition',choices=ORIG_CONDS+B0_CONDS)
    p.add_argument('--seed',type=int)
    p.add_argument('--stop-at-updates',type=int,default=None,
                   help='Only for EEGNet; cumulative partial optimizer-update target')
    p.add_argument('--checkpoint-every',type=int,default=400)
    args=p.parse_args()
    try:
        need(args.checkpoint_every>0 and (args.stop_at_updates is None or 1<=args.stop_at_updates<=UPDATES),
             'Invalid optimizer checkpoint/partial target')
        repo=args.repo_root.resolve()
        need((repo/'scripts/data').is_dir(),'Wrong project root')
        ctx=context(repo)
        tasks=schedule(ctx)
        print('S2_75B_PREFLIGHT_PASS: 11 conditions; 3 classifiers; 55 fits',flush=True)
        print('Original 8 preserved; B0 adds 3. Final Test SEALED.',flush=True)
        print('Training update budget EEGNet:',UPDATES,flush=True)
        if args.preflight:return 0
        with writer_lock(ctx['out']):
            if args.aggregate:
                need(args.family is None and args.condition is None and args.seed is None,
                     'Aggregate is all predeclared fits only')
                aggregate(ctx);return 0
            if args.run_one:
                need(args.family and args.condition,'--run-one needs both --family and --condition')
                seed=args.seed if args.seed is not None else (EEG_SEEDS[0] if args.family=='EEGNETV4' else CLASSICAL_SEED)
                run_one(ctx,args.condition,args.family,seed,args.stop_at_updates,args.checkpoint_every)
            else:
                need(args.family is None and args.condition is None and args.seed is None and
                     args.stop_at_updates is None,'--run-all cannot filter or stop early')
                for fam,cond,seed in tasks:
                    run_one(ctx,cond,fam,seed,None,args.checkpoint_every)
                aggregate(ctx)
        return 0
    except KeyboardInterrupt:
        print('S2_75B_INTERRUPTED: completed tasks and EEGNet checkpoints retained.',file=sys.stderr)
        return 130
    except Exception as e:
        print('S2_75B_FAILED:',type(e).__name__+': '+str(e),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())