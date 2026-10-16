r"""s2_74B: literature-grounded, descriptive final-pool EEG characterization.

Read-only inputs: frozen s2_72, completed s2_73, frozen s2_74, eligible Real Train.
NO generator, training, Validation waveform, Test waveform, rejection, or selection.
Run from existing repo, after saving this file in scripts/data:
  .\.venv\Scripts\python.exe -u .\scripts\data\s2_74b_final_pool_fidelity.py --stop-after-subjects 2
Later use --stop-after-subjects 198. The subject target is cumulative, not additional.

All windows: SD-MD (pooled samples/channel adaptation), Welch PSD, bandpower,
Hjorth, skewness/kurtosis, permutation entropy, autocorrelation, covariance,
correlation, sliced W1, diversity and same-stratum nearest-neighbour diagnostics.
Connectivity: nonoverlapping Real Train windows, matched synthetic count/indices,
Vinck debiased squared wPLI (alpha primary; theta and beta secondary).
Matched Real-Real calibration: disjoint TRAIN blocks, equal count, nonoverlapping
windows. This is NOT held-out generalization or a privacy guarantee.

CPU only (two threads requested). Results resume per subject, after verifying
inputs and cached-output hashes. Every analysis setting is fixed before signals
are read and kept in protocol/s2_74b_analysis_plan_v1.json. Existing sources and
synthetic pools are never modified. Statistical CIs are descriptive, marginal,
subject-bootstrap CIs conditional on the fixed two pools, not selection gates.
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import platform
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.signal import welch
from scipy.stats import kurtosis, skew, wasserstein_distance

VERSION = 's2_74b_descriptive_fidelity_v1'
S74_SHA = '749fb1a484deccff4eb41d3fbffd452e987348691fbf814ca304f55efd167fee'
S23_SHA = 'e880eebee4dfeb79f41da809bbb4f8d6c7d233df68d972702f093f8aef4cde34'
PROTOCOL_SHA = '5b98ff09d977006b96ea80e43d4f13f33742745b734d5dfa94bb1eaacb682a22'
CHANNELS = ['Fp1','Fp2','F7','F8','F3','F4','Fz','T7','T8','C3','C4','Cz',
            'P7','P8','P3','P4','Pz','O1','O2']
BANDS = {'delta': (1.,4.), 'theta': (4.,8.), 'alpha': (8.,13.), 'beta': (13.,30.)}
FC_BANDS = ('theta','alpha','beta')
FS, NT, NC = 128, 512, 19
UT = np.triu_indices(NC, 1)
TINY = np.finfo(np.float64).tiny
ANALYSIS_SEED = 20262701
BOOT_SEED = 20262702
N_PROJ = 128
BOOT_REPS = 5000
OUT_REL = Path('outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1')

SOURCES = {
 'klein2024': {'title':'Synthesizing EEG Signals from Event-Related Potential Paradigms with Conditional Diffusion Models',
  'url':'https://arxiv.org/abs/2403.18486',
  'use':'SWD family, SD-MD, condition-wise evaluation, real-real reference; pooled SD and resting-state adaptations explicit.'},
 'hartmann2018': {'title':'EEG-GAN: Generative adversarial networks for electroencephalograhic (EEG) brain signals',
  'url':'https://arxiv.org/abs/1806.01875','status':'preprint',
  'use':'Sliced Wasserstein and nearest-real-sample diagnostics; not a formal privacy test.'},
 'vinck2011': {'title':'An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias',
  'doi':'10.1016/j.neuroimage.2011.01.055','use':'Debiased squared wPLI estimator.'},
 'hardmeier2014': {'title':'Reproducibility of Functional Connectivity and Graph Measures Based on the Phase Lag Index (PLI) and Weighted Phase Lag Index (wPLI) Derived from High Resolution EEG',
  'doi':'10.1371/journal.pone.0108648','use':'Motivates resting-EEG connectivity and within-real calibration; not reproducing long-term ICC.'},
 'barry2007': {'title':'EEG differences between eyes-closed and eyes-open resting conditions',
  'doi':'10.1016/j.clinph.2007.07.028','use':'Bandpower, EO/EC and spatial effects; numerical error metrics are project adaptations.'},
 'hjorth1970': {'title':'EEG analysis based on time domain properties',
  'doi':'10.1016/0013-4694(70)90143-4','use':'Activity, mobility and complexity; derivatives here are finite differences per sample.'},
 'bandt2002': {'title':'Permutation Entropy: A Natural Complexity Measure for Time Series',
  'doi':'10.1103/PhysRevLett.88.174102','use':'Ordinal permutation entropy, fixed order 3 and delay 1.'},
 'finn2015': {'title':'Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity',
  'doi':'10.1038/nn.4135','use':'Identification concept only; original modality fMRI. Our log-bandpower fingerprint is an EEG/project adaptation.'},
 'silveira2026': {'title':'Interpretable metrics for evaluating fidelity, diversity, and privacy in synthetic EEG data',
  'doi':'10.1186/s13064-026-00323-z','use':'Complementary descriptors, real-real calibration and nearest-neighbour caution. No composite score or borrowed thresholds.'},
}

class AuditError(RuntimeError):
    pass

def require(ok, message):
    if not ok:
        raise AuditError(message)

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(',',':'), allow_nan=False).encode()

def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def sha_obj(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def sha_array(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(canonical({'shape':list(x.shape),'dtype':str(x.dtype)}))
    h.update(x.tobytes())
    return h.hexdigest()

def load_json(p):
    def pairs(items):
        obj = {}
        for k,v in items:
            require(k not in obj, f'Duplicate JSON key: {k}')
            obj[k] = v
        return obj
    def reject(v):
        raise AuditError('Nonfinite JSON constant: '+v)
    with Path(p).open(encoding='utf-8') as f:
        return json.load(f, object_pairs_hook=pairs, parse_constant=reject)

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return x

def atomic_bytes(p, b):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('xb') as f:
            f.write(b); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if tmp.exists(): tmp.unlink()

def write_json(p,x):
    atomic_bytes(p, canonical(clean(x))+b'\n')

def freeze_json(p,x):
    if p.exists():
        require(canonical(load_json(p)) == canonical(clean(x)), 'Frozen analysis plan differs: '+str(p))
    else:
        write_json(p,x)

def csv_write(p,rows):
    rows = list(rows)
    if not rows: return
    s = io.StringIO(newline='')
    w = csv.DictWriter(s,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
    w.writeheader(); w.writerows(rows)
    atomic_bytes(p,s.getvalue().encode('utf-8-sig'))

@contextmanager
def writer_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    p = root/'.writer.lock'
    require(not p.is_symlink(),'Lock cannot be a symlink')
    f = p.open('a+b')
    if p.stat().st_size == 0: f.write(b'0'); f.flush()
    f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as e:
        f.close(); raise AuditError('Another s2_74B writer is active') from e
    try: yield
    finally:
        f.seek(0)
        if os.name == 'nt':
            msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        f.close()

def import_pinned(path, digest, name):
    require(path.is_file(), 'Missing required source: '+str(path))
    require(sha_file(path)==digest, 'Source changed; not importing: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def repo_path(repo, relative):
    p = (repo/relative).resolve()
    require(p.is_relative_to(repo),'Path outside repository: '+str(relative))
    return p

def prepare(repo):
    sdir = repo/'scripts/data'
    sys.path.insert(0,str(sdir))
    a = import_pinned(sdir/'s2_74_audit_and_freeze_half_subsets.py',S74_SHA,'_s74_readonly')
    helper = import_pinned(sdir/'s2_23_check_real_train_dataset.py',S23_SHA,'_s23_readonly')
    gen = a.generation_root(repo)
    proto,pp,psha,plan,gp,gsha,pairs = a.verify_plan_and_protocol(repo,gen)
    _,pool_manifest=a.verify_pool_manifest(gen,gsha,pairs)
    a.check_no_unexpected_committed_files(gen,pairs)
    fp = a.freeze_root(repo)/'manifests/s2_74_freeze_manifest_v1.json'
    fr = load_json(fp)
    require(fr.get('status')=='S2_74_POOL_INTEGRITY_AND_0P5X_FROZEN','s2_74 is not frozen')
    require(fr.get('implementation_sha256')==S74_SHA,'s2_74 implementation mismatch')
    sa = fr['source_artifacts']
    require(sa['s2_72_protocol']['sha256']==psha==PROTOCOL_SHA,'Protocol mismatch')
    require(sa['s2_73_generation_plan']['canonical_sha256']==gsha,'Plan mismatch')
    tracked = {pp:psha,fp:sha_file(fp),sdir/'s2_74_audit_and_freeze_half_subsets.py':S74_SHA,
               sdir/'s2_23_check_real_train_dataset.py':S23_SHA,gp:sha_file(gp)}
    for key in ('s2_73_pool_manifest',):
        item = sa[key]; p=repo_path(repo,item['path'])
        require(sha_file(p)==item['sha256'],'Changed frozen artifact: '+str(p)); tracked[p]=item['sha256']
    for key in ('master','B1_view','B2_view'):
        item = fr['nested_0p5x'][key]; p=repo_path(repo,item['path'])
        require(sha_file(p)==item['sha256'],'Changed 0.5x artifact: '+str(p)); tracked[p]=item['sha256']
    require(fr['nested_0p5x']['retained_windows_per_generator']==23757,'Unexpected half-subset total')
    for key,p in [('split_manifest',helper.FINAL_SPLIT_MANIFEST),('normalization',helper.NORMALIZATION_FILE)]:
        require(p.resolve().is_relative_to(repo),'Helper points to another repo')
        digest=proto['data_contract'][key]['sha256']
        require(sha_file(p)==digest, 'Frozen '+key+' changed'); tracked[p]=digest
    scaler=load_json(helper.NORMALIZATION_FILE)
    require(scaler['channel_order']==CHANNELS,'Channel order changed')
    split=pd.read_csv(helper.FINAL_SPLIT_MANIFEST)
    records,mapping=helper.build_train_window_index(split,scaler)
    cols=plan['core']['train_window_index_columns']
    idx=[[getattr(r,k) for k in cols] for r in records]
    require(sha_obj(idx)==plan['core']['train_window_index_sha256'],'Train index differs from generation')
    require(list(mapping)==plan['core']['subject_ids_in_index_order'],'Subject mapping changed')
    grouped=defaultdict(list)
    for r in records: grouped[(r.subject_id,r.eye_state)].append(r)
    for pair in pairs:
        require(len(grouped[(pair['subject_id'],pair['eye_state'])])==pair['expected_count'],'Stratum count changed')
    return a,gen,proto,plan,pairs,scaler,grouped,tracked,pool_manifest

def select_indices(records, seed):
    """Greedy non-overlap within each block; RR uses distinct whole Train blocks."""
    by=defaultdict(list)
    for i,r in enumerate(records): by[r.block_uid].append((r.start_sample,i))
    nonoverlap={}
    for b,items in sorted(by.items()):
        chosen=[]; end=-1
        for st,i in sorted(items):
            if st>=end: chosen.append(i); end=st+NT
        nonoverlap[b]=chosen
    rng=np.random.RandomState(seed)
    blocks=sorted(nonoverlap)
    require(len(blocks)>=2,'Need at least two eligible Train blocks for Real-Real calibration')
    perm=rng.permutation(len(blocks))
    cut=len(blocks)//2
    ab=[blocks[i] for i in perm[:cut]]; bb=[blocks[i] for i in perm[cut:]]
    ia=np.array([i for b in ab for i in nonoverlap[b]],dtype=np.int64)
    ib=np.array([i for b in bb for i in nonoverlap[b]],dtype=np.int64)
    m=min(len(ia),len(ib))
    require(m>=4,'Too few non-overlap windows for descriptive RR calibration')
    ra=np.sort(rng.choice(ia,m,replace=False)); rb=np.sort(rng.choice(ib,m,replace=False))
    rr=np.array(sorted(i for ids in nonoverlap.values() for i in ids),dtype=np.int64)
    syn=np.sort(rng.choice(len(records),len(rr),replace=False))
    sg=np.sort(rng.choice(len(records),m,replace=False))
    return {'nonoverlap_real':rr.tolist(),'nonoverlap_synthetic':syn.tolist(),
            'rr_real_A':ra.tolist(),'rr_real_B':rb.tolist(),'rr_synthetic':sg.tolist(),
            'rr_A_blocks':ab,'rr_B_blocks':bb,'rr_n':m,
            'meaning':'Disjoint TRAIN blocks for calibration; both were seen by generators.'}

def read_real(records,scaler):
    means=np.asarray(scaler['means_uv'],dtype=np.float64)
    stds=np.asarray(scaler['stds_uv'],dtype=np.float64)
    require(means.shape==stds.shape==(NC,) and np.all(stds>0),'Invalid scaler')
    x=np.empty((len(records),NC,NT),dtype=np.float32)
    # Only requested eligible Train block datasets are sliced. No held-out EEG.
    with h5py.File(records[0].h5_path,'r') as h:
        for i,r in enumerate(records):
            b=h[f'blocks/{r.block_uid}']
            split=b.attrs['provisional_split']
            if isinstance(split,bytes): split=split.decode()
            require(split=='Train','Forbidden non-Train waveform read')
            v=np.asarray(b['eeg_uv'][:,r.start_sample:r.start_sample+NT],dtype=np.float32)
            require(v.shape==(NC,NT) and np.isfinite(v).all(),'Bad Real Train window')
            x[i]=((v.astype(np.float64)-means[:,None])/stds[:,None]).astype(np.float32)
    require(np.isfinite(x).all(),'Nonfinite normalized Real Train')
    return x

def permutation_entropy(x):
    w=np.lib.stride_tricks.sliding_window_view(x,3,axis=-1)
    order=np.argsort(w,axis=-1,kind='stable')
    code=order[...,0]*9+order[...,1]*3+order[...,2]
    valid_codes=(5,7,11,15,19,21)
    counts=np.stack([(code==c).sum(axis=-1) for c in valid_codes],axis=-1)
    require(np.all(counts.sum(axis=-1)==x.shape[-1]-2),'Permutation encoding error')
    p=counts/(x.shape[-1]-2)
    ent=-(p*np.log(np.maximum(p,TINY))).sum(axis=-1)/np.log(6.)
    tie=np.mean((w[...,0]==w[...,1])|(w[...,0]==w[...,2])|(w[...,1]==w[...,2]),axis=-1)
    return ent,tie

def features(xz,scaler,projections):
    xz=np.asarray(xz,dtype=np.float64)
    uv=xz*np.asarray(scaler['stds_uv'])[None,:,None]+np.asarray(scaler['means_uv'])[None,:,None]
    cen=uv-uv.mean(axis=-1,keepdims=True)
    var=np.var(uv,axis=-1); v1=np.var(np.diff(uv,axis=-1),axis=-1)
    v2=np.var(np.diff(uv,n=2,axis=-1),axis=-1)
    mob=np.full_like(var,np.nan); comp=np.full_like(var,np.nan)
    good=var>0; mob[good]=np.sqrt(v1[good]/var[good])
    good=(v1>0)&(mob>0); comp[good]=np.sqrt(v2[good]/v1[good])/mob[good]
    pe,ties=permutation_entropy(uv)
    desc={'mean_uv':uv.mean(-1),'std_uv':np.sqrt(var),'rms_uv':np.sqrt(np.mean(uv*uv,axis=-1)),
          'hjorth_activity_uv2':var,'hjorth_mobility_per_sample':mob,'hjorth_complexity':comp,
          'skewness':skew(uv,axis=-1,bias=False),'excess_kurtosis':kurtosis(uv,axis=-1,bias=False,fisher=True),
          'permutation_entropy_m3_tau1':pe}
    f,p=welch(uv,fs=FS,window='hann',nperseg=256,noverlap=128,detrend='constant',scaling='density',axis=-1)
    trap=getattr(np,'trapezoid',None) or np.trapz
    bp=np.stack([trap(p[..., (f>=lo)&(f<=hi)],f[(f>=lo)&(f<=hi)],axis=-1) for lo,hi in BANDS.values()],axis=1)
    total=trap(p[...,(f>=1)&(f<=40)],f[(f>=1)&(f<=40)],axis=-1)
    rel=bp/np.maximum(total[:,None,:],TINY)
    zcen=xz-xz.mean(axis=-1,keepdims=True)
    cov=np.einsum('nct,ndt->ncd',zcen,zcen,optimize=True)/(NT-1)
    sd=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))
    denom=sd[:,:,None]*sd[:,None,:]
    corr=np.divide(cov,denom,out=np.full_like(cov,np.nan),where=denom>0)
    sp=np.fft.rfft(cen*np.hanning(NT)[None,None,:],axis=-1)
    a=np.fft.rfft(cen,n=2*NT,axis=-1)
    ac=np.fft.irfft(a*a.conj(),n=2*NT,axis=-1)[...,:129]
    ac=np.divide(ac,ac[...,:1],out=np.full_like(ac,np.nan),where=ac[...,:1]>0)
    return {'xz':xz,'uv':uv,'desc':desc,'ties':ties,'freq':f,'psd':p,'bp_db':10*np.log10(np.maximum(bp,TINY)),
            'bp_rel':rel,'cov':cov,'corr':corr,'fft':sp,'acf':ac,
            'proj':xz.reshape(len(xz),-1)@projections}

def subfeat(a,idx):
    idx=np.asarray(idx,dtype=int)
    return {k:({n:v[idx] for n,v in val.items()} if k=='desc' else val if k=='freq' else val[idx]) for k,val in a.items()}

def finite_mean(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    return float(x.mean()) if x.size else float('nan')

def correlation(a,b):
    a=np.asarray(a).ravel(); b=np.asarray(b).ravel()
    valid=np.isfinite(a)&np.isfinite(b)
    if valid.sum()<3: return float('nan')
    a=a[valid]-a[valid].mean(); b=b[valid]-b[valid].mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d>1e-15 else float('nan')

def norm_ratio(a,b):
    denom=np.linalg.norm(b)
    return float(np.linalg.norm(a)/denom) if denom>1e-15 else float('nan')

def nearest_rmse(a,b):
    a=a.reshape(len(a),-1); b=b.reshape(len(b),-1)
    d2=np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*(a@b.T),0)/a.shape[1]
    return np.sqrt(d2.min(axis=1))

def compare(r,s):
    mask=(r['freq']>=1)&(r['freq']<=40)
    rp=10*np.log10(np.maximum(r['psd'].mean(0)[:,mask],TINY))
    sp=10*np.log10(np.maximum(s['psd'].mean(0)[:,mask],TINY))
    rb=np.median(r['bp_db'],axis=0); sb=np.median(s['bp_db'],axis=0)
    rsd=np.std(r['uv'],axis=(0,2)); ssd=np.std(s['uv'],axis=(0,2))
    rc=r['cov'].mean(0); sc=s['cov'].mean(0)
    rr=r['corr'].mean(0); sr=s['corr'].mean(0)
    out={'log_psd_mae_db':float(np.mean(np.abs(sp-rp))),
         'bandpower_median_mae_db':float(np.mean(np.abs(sb-rb))),
         'relative_bandpower_mae':float(np.mean(np.abs(s['bp_rel'].mean(0)-r['bp_rel'].mean(0)))),
         'sd_md_pooled_uv':float(np.mean(np.abs(ssd-rsd))),
         'mean_amplitude_mae_uv':float(np.mean(np.abs(s['uv'].mean((0,2))-r['uv'].mean((0,2))))),
         'covariance_relative_frobenius_z':norm_ratio(sc-rc,rc),
         'correlation_upper_mae':finite_mean(np.abs(sr[UT]-rr[UT])),
         'correlation_upper_r':correlation(sr[UT],rr[UT]),
         'acf_mae_lag1_128':finite_mean(np.abs(s['acf'].mean(0)[:,1:]-r['acf'].mean(0)[:,1:])),
         'sliced_wasserstein1_z':float(np.mean([wasserstein_distance(r['proj'][:,i],s['proj'][:,i]) for i in range(N_PROJ)]))}
    for bi,band in enumerate(BANDS):
        out['bandpower_'+band+'_mae_db']=float(np.mean(np.abs(sb[bi]-rb[bi])))
        out['bandpower_'+band+'_w1_db']=float(np.mean([wasserstein_distance(r['bp_db'][:,bi,c],s['bp_db'][:,bi,c]) for c in range(NC)]))
        out['bandpower_'+band+'_topography_r']=correlation(rb[bi],sb[bi])
    details={}
    for name,rv in r['desc'].items():
        sv=s['desc'][name]; raw=[]; scaled=[]; lost=[]
        for c in range(NC):
            ra=rv[:,c]; sy=sv[:,c]; ra=ra[np.isfinite(ra)]; sy=sy[np.isfinite(sy)]
            lost.append([len(rv)-len(ra),len(sv)-len(sy)])
            if not len(ra) or not len(sy): raw.append(np.nan); scaled.append(np.nan); continue
            d=wasserstein_distance(ra,sy); den=np.std(ra,ddof=1) if len(ra)>1 else 0
            raw.append(d); scaled.append(d/den if den>1e-15 else np.nan)
        out[name+'_w1']=finite_mean(raw); out[name+'_w1_real_sd_units']=finite_mean(scaled)
        details[name]={'w1_by_channel':raw,'w1_real_sd_units_by_channel':scaled,'undefined_windows_real_synth':lost,
                       'real_mean':np.mean(rv,axis=0),'synth_mean':np.mean(sv,axis=0)}
    # Ratio of RMS pairwise distance, not mean Euclidean distance; 1 is parity.
    rvar=np.var(r['xz'].reshape(len(r['xz']),-1),axis=0,ddof=1).mean()
    svar=np.var(s['xz'].reshape(len(s['xz']),-1),axis=0,ddof=1).mean()
    out['rms_pairwise_diversity_ratio']=float(np.sqrt(svar/rvar)) if rvar>0 else np.nan
    nn=nearest_rmse(s['xz'],r['xz'])
    out['nearest_real_rmse_z_median']=float(np.median(nn))
    out['nearest_real_rmse_z_q05']=float(np.quantile(nn,.05))
    rh={sha_array(x.astype(np.float32)) for x in r['xz']}
    sh=[sha_array(x.astype(np.float32)) for x in s['xz']]
    out['exact_same_stratum_train_copy_fraction']=sum(h in rh for h in sh)/len(sh)
    out['within_stratum_exact_duplicate_fraction']=1-len(set(sh))/len(sh)
    return out,details

def dwpli2(sp,band):
    freqs=np.fft.rfftfreq(NT,1/FS); lo,hi=BANDS[band]
    bins=np.flatnonzero((freqs>=lo)&(freqs<=hi))
    acc=np.zeros((NC,NC)); count=np.zeros((NC,NC),dtype=int)
    for k in bins:
        z=sp[:,:,k]; im=(z[:,:,None]*z[:,None,:].conj()).imag
        si=im.sum(0); sa=np.abs(im).sum(0); sq=(im*im).sum(0)
        num=si*si-sq; den=sa*sa-sq
        # Degenerate denominators are undefined, not fabricated zero connections.
        valid=(den>np.maximum(sa*sa,sq)*1e-12)&(den>0)
        val=np.divide(num,den,out=np.zeros_like(den),where=valid)
        acc+=np.where(valid,val,0); count+=valid
    out=np.divide(acc,count,out=np.full_like(acc,np.nan),where=count>0)
    np.fill_diagonal(out,0.)
    np.fill_diagonal(count,0)
    return out,count

def fc_compare(r,s):
    d=s-r
    valid=np.isfinite(r[UT])&np.isfinite(s[UT])
    rv=r[UT][valid]; sv=s[UT][valid]
    nod_r=np.array([finite_mean(np.delete(row,i)) for i,row in enumerate(r)])
    nod_s=np.array([finite_mean(np.delete(row,i)) for i,row in enumerate(s)])
    return {'upper_mae':finite_mean(np.abs(sv-rv)), 'upper_r':correlation(rv,sv),
            'relative_frobenius':norm_ratio(sv-rv,rv), 'valid_edges':int(valid.sum()),
            'mean_connectivity_real':finite_mean(rv),'mean_connectivity_synth':finite_mean(sv),
            'node_mean_mae':finite_mean(np.abs(nod_s-nod_r)),
            'negative_edge_fraction_real':float((rv<0).mean()) if rv.size else np.nan,
            'negative_edge_fraction_synth':float((sv<0).mean()) if sv.size else np.nan}

def summary_feature(f):
    return {'pooled_mean_uv':f['uv'].mean(axis=(0,2)), 'pooled_std_uv':f['uv'].std(axis=(0,2)),
            'mean_psd_uv2_hz':f['psd'].mean(0), 'median_bandpower_db':np.median(f['bp_db'],axis=0),
            'mean_relative_bandpower':f['bp_rel'].mean(0),'mean_covariance_z':f['cov'].mean(0),
            'mean_correlation':f['corr'].mean(0),'mean_acf':f['acf'].mean(0),
            'permutation_tie_fraction':f['ties'].mean(0),
            'descriptor_quantiles':{n:np.nanquantile(v,[.05,.25,.5,.75,.95],axis=0) for n,v in f['desc'].items()}}

def analyze_state(xs,scaler,selection,projections):
    f={k:features(x,scaler,projections) for k,x in xs.items()}
    main={}; details={}
    for g in ('B1','B2'): main[g],details[g]=compare(f['Real'],f[g])
    fc={}; matrices={}
    ix={'Real':selection['nonoverlap_real'],'B1':selection['nonoverlap_synthetic'],'B2':selection['nonoverlap_synthetic']}
    for band in FC_BANDS:
        mats={}; valid={}
        for name in f: mats[name],valid[name]=dwpli2(f[name]['fft'][ix[name]],band)
        matrices[band]={n:{'matrix':m,'valid_frequency_bins':valid[n]} for n,m in mats.items()}
        fc[band]={g:fc_compare(mats['Real'],mats[g]) for g in ('B1','B2')}
    ra=subfeat(f['Real'],selection['rr_real_A']); rb=subfeat(f['Real'],selection['rr_real_B'])
    base={'Real_B':compare(ra,rb)[0]}
    bs={g:subfeat(f[g],selection['rr_synthetic']) for g in ('B1','B2')}
    for g in bs: base[g]=compare(ra,bs[g])[0]
    base_fc={}
    for band in FC_BANDS:
        rma=dwpli2(ra['fft'],band)[0]
        base_fc[band]={g:fc_compare(rma,dwpli2(v['fft'],band)[0]) for g,v in {'Real_B':rb,**bs}.items()}
    return {'full_pool_metrics':main,'descriptor_distances':details,'source_summaries':{n:summary_feature(v) for n,v in f.items()},
            'nonoverlap_connectivity':fc,'nonoverlap_connectivity_matrices':matrices,
            'matched_train_block_calibration':{'n_per_group':selection['rr_n'],'feature_metrics':base,'connectivity':base_fc},
            'sample_counts':{n:len(x) for n,x in xs.items()},'connectivity_windows_per_group':len(ix['Real'])}

def boot_mean(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if not x.size: return {'n_subjects':0,'mean':None,'ci95':None}
    rng=np.random.RandomState(BOOT_SEED)
    vals=np.empty(BOOT_REPS)
    for st in range(0,BOOT_REPS,100):
        sz=min(100,BOOT_REPS-st); ids=rng.randint(0,len(x),(sz,len(x)))
        vals[st:st+sz]=x[ids].mean(1)
    return {'n_subjects':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
            'ci95':np.quantile(vals,[.025,.975]).tolist()}

def scalar(v): return np.nan if v is None else float(v)

def aggregate(results,out):
    rows=[]
    for res in results:
        for state,entry in res['states'].items():
            sid=res['subject_id']
            for g,m in entry['full_pool_metrics'].items():
                rows.append({'subject_id':sid,'state':state,'generator':g,**m})
    csv_write(out/'tables/subject_state_metrics.csv',rows)
    # Every paired CI resamples subjects, retaining EO/EC together. No window p-values.
    paired={}
    keys=sorted(results[0]['states']['EO']['full_pool_metrics']['B1'])
    for key in keys:
        arr={g:np.array([np.mean([scalar(r['states'][st]['full_pool_metrics'][g][key]) for st in ('EO','EC')]) for r in results]) for g in ('B1','B2')}
        paired[key]={'B1':boot_mean(arr['B1']),'B2':boot_mean(arr['B2']),
                     'B2_minus_B1':boot_mean(arr['B2']-arr['B1'])}
    alpha=[]; fcrows=[]
    for r in results:
        effects={n:np.asarray(r['states']['EC']['source_summaries'][n]['median_bandpower_db'])[2]-np.asarray(r['states']['EO']['source_summaries'][n]['median_bandpower_db'])[2] for n in ('Real','B1','B2')}
        for g in ('B1','B2'):
            alpha.append({'subject_id':r['subject_id'],'generator':g,'channel_mae_db':float(np.abs(effects[g]-effects['Real']).mean()),
                          'posterior_O1_O2_error_db':float(np.abs(effects[g][17:19].mean()-effects['Real'][17:19].mean())),
                          'real_EC_minus_EO_O1_O2_db':float(effects['Real'][17:19].mean()),
                          'synthetic_EC_minus_EO_O1_O2_db':float(effects[g][17:19].mean()),
                          'effect_topography_r':correlation(effects['Real'],effects[g])})
        for st in ('EO','EC'):
            for band in FC_BANDS:
                for g in ('B1','B2'):
                    fcrows.append({'subject_id':r['subject_id'],'state':st,'band':band,'generator':g,
                                   'n_windows_per_group':r['states'][st]['connectivity_windows_per_group'],
                                   **r['states'][st]['nonoverlap_connectivity'][band][g]})
    csv_write(out/'tables/subject_alpha_effect.csv',alpha)
    csv_write(out/'tables/subject_connectivity.csv',fcrows)
    alpha_summary={}
    for key in ('channel_mae_db','posterior_O1_O2_error_db','effect_topography_r'):
        b1=np.array([scalar(v[key]) for v in alpha if v['generator']=='B1'])
        b2=np.array([scalar(v[key]) for v in alpha if v['generator']=='B2'])
        alpha_summary[key]={'B1':boot_mean(b1),'B2':boot_mean(b2),'B2_minus_B1':boot_mean(b2-b1)}
    fc_summary={}
    for band in FC_BANDS:
        fc_summary[band]={}
        for key in ('upper_mae','upper_r','relative_frobenius','node_mean_mae'):
            ar={g:np.array([np.mean([scalar(r['states'][st]['nonoverlap_connectivity'][band][g][key]) for st in ('EO','EC')]) for r in results]) for g in ('B1','B2')}
            fc_summary[band][key]={'B1':boot_mean(ar['B1']),'B2':boot_mean(ar['B2']),
                                   'B2_minus_B1':boot_mean(ar['B2']-ar['B1'])}
    # Calibration CSV preserves matched counts; never compares unmatched RR vs full n directly.
    rrrows=[]
    for r in results:
        for st in ('EO','EC'):
            b=r['states'][st]['matched_train_block_calibration']
            for name,m in b['feature_metrics'].items():
                rrrows.append({'subject_id':r['subject_id'],'state':st,'comparison':'Real_A_vs_'+name,'n_per_group':b['n_per_group'],**m})
    csv_write(out/'tables/matched_real_real_calibration.csv',rrrows)
    rrfc=[]
    for r in results:
        for st in ('EO','EC'):
            b=r['states'][st]['matched_train_block_calibration']
            for band in FC_BANDS:
                for name,m in b['connectivity'][band].items():
                    rrfc.append({'subject_id':r['subject_id'],'state':st,'band':band,
                                 'comparison':'Real_A_vs_'+name,'n_per_group':b['n_per_group'],**m})
    csv_write(out/'tables/matched_real_real_connectivity.csv',rrfc)
    rr_summary={}
    for key in keys:
        ar={name:np.asarray([np.mean([scalar(r['states'][st]['matched_train_block_calibration']['feature_metrics'][name][key])
                                      for st in ('EO','EC')]) for r in results]) for name in ('Real_B','B1','B2')}
        rr_summary[key]={name:boot_mean(v) for name,v in ar.items()}
        rr_summary[key].update({g+'_minus_RealReal':boot_mean(ar[g]-ar['Real_B']) for g in ('B1','B2')})
    rr_fc_summary={}
    for band in FC_BANDS:
        rr_fc_summary[band]={}
        for key in ('upper_mae','upper_r','relative_frobenius','node_mean_mae'):
            ar={name:np.asarray([np.mean([scalar(r['states'][st]['matched_train_block_calibration']['connectivity'][band][name][key])
                                          for st in ('EO','EC')]) for r in results]) for name in ('Real_B','B1','B2')}
            rr_fc_summary[band][key]={name:boot_mean(v) for name,v in ar.items()}
            rr_fc_summary[band][key].update({g+'_minus_RealReal':boot_mean(ar[g]-ar['Real_B']) for g in ('B1','B2')})
    # State effects use within-participant EC minus EO, not an across-subject pairing.
    fc_effect=[]
    for r in results:
        for band in FC_BANDS:
            delta={n:np.asarray(r['states']['EC']['nonoverlap_connectivity_matrices'][band][n]['matrix'],dtype=float)
                       -np.asarray(r['states']['EO']['nonoverlap_connectivity_matrices'][band][n]['matrix'],dtype=float)
                   for n in ('Real','B1','B2')}
            for g in ('B1','B2'):
                valid=np.isfinite(delta['Real'][UT])&np.isfinite(delta[g][UT])
                dr=delta['Real'][UT][valid]; ds=delta[g][UT][valid]
                fc_effect.append({'subject_id':r['subject_id'],'band':band,'generator':g,
                                  'EC_minus_EO_upper_mae':finite_mean(np.abs(ds-dr)),
                                  'EC_minus_EO_upper_r':correlation(ds,dr),'valid_edges':int(valid.sum()),
                                  'real_mean_EC_minus_EO':finite_mean(dr),'synthetic_mean_EC_minus_EO':finite_mean(ds)})
    csv_write(out/'tables/subject_connectivity_state_effect.csv',fc_effect)
    fc_effect_summary={}
    for band in FC_BANDS:
        fc_effect_summary[band]={}
        for key in ('EC_minus_EO_upper_mae','EC_minus_EO_upper_r'):
            ar={g:np.asarray([v[key] for v in fc_effect if v['band']==band and v['generator']==g],dtype=float) for g in ('B1','B2')}
            fc_effect_summary[band][key]={'B1':boot_mean(ar['B1']),'B2':boot_mean(ar['B2']),
                                         'B2_minus_B1':boot_mean(ar['B2']-ar['B1'])}
    fp={}; fprows=[]
    for st in ('EO','EC'):
        real=np.stack([np.asarray(r['states'][st]['source_summaries']['Real']['median_bandpower_db']).ravel() for r in results])
        for g in ('B1','B2'):
            syn=np.stack([np.asarray(r['states'][st]['source_summaries'][g]['median_bandpower_db']).ravel() for r in results])
            sims=np.array([[correlation(a,b) for b in real] for a in syn])
            valid=np.isfinite(sims).all(axis=1)
            ranks=[]; margins=[]
            for i in range(len(results)):
                if not valid[i]: continue
                # Pessimistic tie handling: all other scores >= own score precede own.
                other=np.delete(sims[i],i); rank=1+int(np.sum(other>=sims[i,i]))
                margin=float(sims[i,i]-other.max()) if len(other) else np.nan
                ranks.append(rank); margins.append(margin)
                fprows.append({'subject_id':results[i]['subject_id'],'state':st,'generator':g,
                               'correct_subject_r':sims[i,i],'correct_vs_best_wrong_margin':margin,'rank':rank,
                               'fingerprint_rmse_db':float(np.sqrt(np.mean((syn[i]-real[i])**2)))})
            fp[st+'_'+g]={'n_candidates':len(results),'n_valid_queries':len(ranks),
                         'top1':float(np.mean(np.array(ranks)<=1)) if ranks else None,
                         'top5':float(np.mean(np.array(ranks)<=min(5,len(results)))) if ranks else None,
                         'margin':boot_mean(margins),'note':'EEG log-bandpower adaptation, in-sample; not identity anonymity or unseen-subject generalization.'}
    csv_write(out/'tables/subject_fingerprint.csv',fprows)
    return {'schema':VERSION,'status':'S2_74B_COMPLETE','subjects':len(results),
            'full_pool_metrics':paired,'alpha_EC_minus_EO':alpha_summary,'connectivity':fc_summary,
            'matched_train_block_calibration_summary':rr_summary,'matched_train_block_connectivity_summary':rr_fc_summary,
            'connectivity_EC_minus_EO':fc_effect_summary,'fingerprint':fp,'bootstrap':{'seed':BOOT_SEED,'repetitions':BOOT_REPS,'unit':'subject','ci':'percentile marginal 95%',
                'multiplicity_adjusted':False,'formal_significance_or_equivalence_claim':False},
            'reference':'Eligible Real Train; in-sample descriptive fidelity, NOT independent validation',
            'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False,
            'generator_changed':False,'synthetic_pool_filtered':False,'s2_75_protocol_changed':False}

def make_figures(results,out):
    """One plot per file; participant-averaged visual summaries, not significance tests."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'status':'SKIPPED_MATPLOTLIB_UNAVAILABLE','note':'All quantitative tables remain available.'}
    dest=out/'figures'; dest.mkdir(parents=True,exist_ok=True)
    paths=[]
    def save(fig,name):
        fig.tight_layout()
        buffer=io.BytesIO(); fig.savefig(buffer,format='png',dpi=160)
        plt.close(fig); file=dest/name; atomic_bytes(file,buffer.getvalue())
        paths.append(file.relative_to(out).as_posix())
    try:
        freq=np.fft.rfftfreq(256,1/FS); mask=(freq>=1)&(freq<=40)
        for st in ('EO','EC'):
            fig,ax=plt.subplots(figsize=(8,4.8))
            for source in ('Real','B1','B2'):
                arr=np.asarray([r['states'][st]['source_summaries'][source]['mean_psd_uv2_hz'] for r in results],dtype=float)
                p=arr.mean(axis=(0,1))
                ax.plot(freq[mask],10*np.log10(np.maximum(p[mask],TINY)),label=source)
            ax.set(xlabel='Frequency (Hz)',ylabel='PSD (dB re 1 uV^2/Hz)',
                   title=f'{st}: equal-subject, equal-channel mean PSD (descriptive)')
            ax.legend(); ax.grid(alpha=.2); save(fig,f'group_psd_{st}.png')
            mats={source:np.nanmean(np.asarray([r['states'][st]['nonoverlap_connectivity_matrices']['alpha'][source]['matrix']
                           for r in results],dtype=float),axis=0) for source in ('Real','B1','B2')}
            values=np.concatenate([m[UT] for m in mats.values()]); values=values[np.isfinite(values)]
            lo=float(values.min()) if values.size else 0.; hi=float(values.max()) if values.size else 1.
            if hi<=lo: hi=lo+1e-8
            for source,m in mats.items():
                view=m.copy(); np.fill_diagonal(view,np.nan)
                fig,ax=plt.subplots(figsize=(8,7))
                im=ax.imshow(np.ma.masked_invalid(view),vmin=lo,vmax=hi)
                ax.set_xticks(range(NC),CHANNELS,rotation=90); ax.set_yticks(range(NC),CHANNELS)
                ax.set_title(f'{st}: alpha debiased wPLI squared, {source}\nEqual-subject mean; common scale for all three sources')
                fig.colorbar(im,ax=ax); save(fig,f'group_alpha_connectivity_{st}_{source}.png')
        fig,ax=plt.subplots(figsize=(10,4.8))
        for source in ('Real','B1','B2'):
            delta=np.asarray([np.asarray(r['states']['EC']['source_summaries'][source]['median_bandpower_db'])[2]
                              -np.asarray(r['states']['EO']['source_summaries'][source]['median_bandpower_db'])[2] for r in results])
            ax.plot(range(NC),delta.mean(axis=0),marker='o',label=source)
        ax.set_xticks(range(NC),CHANNELS,rotation=60)
        ax.set(ylabel='EC minus EO alpha power (dB)',title='Mean participant alpha effect: descriptive, not a significance test')
        ax.legend(); ax.grid(alpha=.2); save(fig,'group_alpha_state_effect.png')
        return {'status':'CREATED','files':paths}
    except Exception as e:
        plt.close('all')
        return {'status':'PLOT_ERROR_QUANTITATIVE_RESULTS_RETAINED','files':paths,'error':str(e)}


def settings():
    return {'version':VERSION,'sources':SOURCES,'reference':'Real Train only; all eligible usable windows for full-pool descriptors',
            'units':'Recover uV with frozen Train scaler; normalized z for SWD/covariance/nearest-neighbour',
            'welch':{'fs':128,'window':'periodic Hann','nperseg':256,'noverlap':128,'detrend':'constant','scaling':'density'},
            'spectral_comparison':'1-40 Hz; mean linear PSD then 10*log10; bandpower median of per-window dB',
            'bands_hz_inclusive':BANDS,'relative_bandpower_denominator_hz':[1,40],
            'sd_md':'mean channel absolute SD difference; SD pooled over windows and time (resting-state adaptation)',
            'hjorth':'activity=var(x); mobility=sqrt(var(diff(x))/var(x)); complexity=mobility(diff(x))/mobility(x); per-sample differences',
            'permutation_entropy':{'order':3,'delay':1,'normalize_log_factorial':True,'ties':'stable index order, tie fraction reported'},
            'acf':'mean per-window demeaned autocorrelation; zero padding 1024; normalize by lag zero; compare lags 1..128',
            'swd':{'p':1,'projections':N_PROJ,'projection_seed':ANALYSIS_SEED,'unit_norm_gaussian_directions':True,
                   'representation':'flattened 19x512 fixed Train-z; same projections for both generators and RR'},
            'connectivity':{'metric':'Vinck debiased squared wPLI; DO NOT square it again','fft_n':512,'window':'symmetric np.hanning(512)',
              'demean_each_channel_window':True,'observations':'non-overlap 4-second Train windows, count-matched synthetic indices',
              'primary_band':'alpha','secondary_bands':['theta','beta'],'frequency_average':'compute estimator per bin, then average',
              'negative_estimates':'retained; not negative anatomical connections','zero_denominator':'undefined, reported; never fill with zero',
              'graph_metrics':'edge MAE, edge r, relative Frobenius, mean connectivity, mean incident weight per node; no thresholding'},
            'rr_calibration':'fixed disjoint TRAIN block halves; nonoverlap windows; equal m for Real A, Real B, B1, B2',
            'copying':'same-subject/state nearest normalized RMSE + bitwise exact match; no privacy guarantee',
            'diversity':'ratio RMS pairwise distance (synthetic/real); parity at one; not larger-is-better',
            'aggregation':'equal subject weight, EO/EC equal inside subject; paired B2-B1 intervals',
            'connectivity_state_effect':'Within-subject EC minus EO matrices; upper-triangle MAE and r of synthetic vs real contrasts',
            'bootstrap_repetitions':BOOT_REPS,'bootstrap_seed':BOOT_SEED,'no_selection_threshold':True,
            'excluded':['ERP PAD/PLD','ImageNet FID','Inception Score','unpaired waveform reconstruction error','global composite fidelity score',
                        'new training','new generation','filtering synthetic samples','Validation/Test waveforms'],
            'fid_status':'Not included in this stage; requires a separately frozen, shared and suitable EEG feature extractor.',
            'cpu_threads_requested':2}

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--stop-after-subjects',type=int,default=2,help='Cumulative subject target, 1..198. Default 2 for timing.')
    args=parser.parse_args()
    require(1<=args.stop_after_subjects<=198,'Subject target must be 1..198')
    repo=args.repo_root.resolve(); require((repo/'scripts/data').is_dir(),'Not a project root')
    out=repo/OUT_REL
    start=time.perf_counter(); run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    limiter=None
    try:
        from threadpoolctl import threadpool_limits
        limiter=threadpool_limits(limits=2)
    except ImportError: pass
    with writer_lock(out):
        try:
            print('S2_74B — DESCRIPTIVE FINAL-POOL FIDELITY — CPU ONLY',flush=True)
            a,gen,proto,plan,pairs,scaler,grouped,tracked,pool_manifest=prepare(repo)
            selections={}
            for p in pairs:
                k=p['subject_id']+'/'+p['eye_state']
                selections[k]=select_indices(grouped[(p['subject_id'],p['eye_state'])],ANALYSIS_SEED+1000+2*p['subject_index']+p['eye_state_label'])
            ap={'settings':settings(),'implementation_sha256':sha_file(Path(__file__)),
                'inputs':{str(p.relative_to(repo)):h for p,h in tracked.items()},
                'generation_plan_canonical_sha256':sha_obj(plan),'train_index_sha256':plan['core']['train_window_index_sha256'],
                'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__,'pandas':pd.__version__},
                'selections':selections}
            apath=out/'protocol/s2_74b_analysis_plan_v1.json'; freeze_json(apath,ap)
            ah=sha_file(apath)
            print('Analysis plan frozen:',apath,flush=True)
            rng=np.random.RandomState(ANALYSIS_SEED)
            proj=rng.normal(size=(NC*NT,N_PROJ)); proj/=np.linalg.norm(proj,axis=0,keepdims=True)
            subjects=plan['core']['subject_ids_in_index_order']
            results=[]; fresh=0; fresh_seconds=0
            for si,sid in enumerate(subjects[:args.stop_after_subjects]):
                tick=time.perf_counter(); inputs={}; arrays={}
                for st in ('EO','EC'):
                    pair=pairs[2*si+(st=='EC')]
                    xr=read_real(grouped[(sid,st)],scaler)
                    inputs[st]={'real_train_z_sha256':sha_array(xr)}; arrays[st]={'Real':xr}
                    metas={}
                    for g in ('B1','B2'):
                        v=a.validate_receipt_and_npz(gen,g,pair)
                        frozen_shard=pool_manifest['committed'][g+'/'+sid+'/'+st]
                        require(v['sha256']==frozen_shard['sha256'] and v['file']==frozen_shard['file']
                                and v['count']==frozen_shard['expected_count'], 'Shard differs from frozen full-pool manifest')
                        meta=v['receipt']['metadata']; metas[g]=meta
                        require(meta['normalization_sha256']==proto['data_contract']['normalization']['sha256'],'Synthetic scaler mismatch')
                        require(meta['generation_plan_sha256']==sha_obj(plan),'Synthetic plan mismatch')
                        require(meta['channel_order']==CHANNELS,'Synthetic channel order mismatch')
                        inputs[st][g]={'file_sha256':v['sha256'],'receipt_sha256':sha_file(gen/v['receipt_file'])}
                        with np.load(gen/v['file'],allow_pickle=False) as z: arrays[st][g]=z['x_z']
                        require(sha_file(gen/v['file'])==frozen_shard['sha256'], 'Shard changed during read')
                    require(canonical(metas['B1']['rng_trace'])==canonical(metas['B2']['rng_trace']),'Paired RNG mismatch')
                p=out/'subjects'/f'{sid}.json'; receipt=p.with_suffix('.sha256')
                if p.exists() and receipt.exists():
                    require(sha_file(p)==receipt.read_text().strip(),'Cached output changed: '+sid)
                    res=load_json(p)
                    require(res['analysis_plan_sha256']==ah and res['input_hashes']==inputs,'Cached input/plan mismatch: '+sid)
                    print(f'  REUSED {si+1}/198 {sid}',flush=True)
                else:
                    res={'subject_id':sid,'subject_index':si,'analysis_plan_sha256':ah,'input_hashes':inputs,'states':{}}
                    for st in ('EO','EC'):
                        print(f'  ANALYZE {si+1}/198 {sid} {st} n={len(arrays[st]["Real"])}',flush=True)
                        res['states'][st]=analyze_state(arrays[st],scaler,selections[sid+'/'+st],proj)
                    res['elapsed_seconds']=time.perf_counter()-tick
                    # An orphan result (crash before receipt) is recomputed, never trusted.
                    write_json(p,res); atomic_bytes(receipt,(sha_file(p)+'\n').encode())
                    res=load_json(p); fresh+=1; fresh_seconds+=res['elapsed_seconds']
                    print(f'  COMMITTED {si+1}/198 {sid}: {res["elapsed_seconds"]:.1f}s',flush=True)
                results.append(res)
                del arrays
                write_json(out/'progress/s2_74b_progress_v1.json',{'verified_subjects_this_run':len(results),'target_subjects':args.stop_after_subjects,
                  'analysis_plan_sha256':ah,'all_198_complete':False,'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False})
            for p,digest in tracked.items(): require(sha_file(p)==digest,'Frozen input changed during analysis: '+str(p))
            if len(results)==198:
                report=aggregate(results,out)
                report['figures']=make_figures(results,out)
                status='S2_74B_COMPLETE'
                write_json(out/'metrics/s2_74b_complete_report_v1.json',report)
            else:
                status='S2_74B_PARTIAL'; report={'status':status,'subjects_verified':len(results),'group_scientific_conclusions':'NOT produced from timing prefix'}
            report.update({'elapsed_seconds_this_run':time.perf_counter()-start,'fresh_subjects_this_run':fresh,
                'mean_fresh_subject_seconds':fresh_seconds/fresh if fresh else None,
                'estimated_remaining_analysis_seconds':(198-len(results))*fresh_seconds/fresh if fresh else None,
                'time_estimate_excludes_final_aggregation':True,'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False})
            rpt=out/'metrics'/f's2_74b_run_{run}.json'; write_json(rpt,report)
            print('\n'+status,flush=True)
            print(f'Verified subjects: {len(results)}/198; elapsed: {(time.perf_counter()-start)/60:.2f} min')
            if fresh: print(f'Estimated remaining subject analysis: {(198-len(results))*fresh_seconds/fresh/60:.1f} min (preliminary)')
            print('Validation waveform accessed: NO\nFinal Test waveform accessed: NO')
            print('RUN REPORT:',rpt)
            print('ANALYSIS PLAN:',apath)
            write_json(out/'progress/s2_74b_progress_v1.json',{'status':status,'verified_subjects':len(results),'all_198_complete':len(results)==198,
                'analysis_plan_sha256':ah,'latest_report':str(rpt),'Validation_waveforms_accessed':False,'Final_Test_waveforms_accessed':False})
            return 0
        except KeyboardInterrupt:
            print('\nS2_74B_INTERRUPTED — completed subject files retained; rerun same command.',file=sys.stderr)
            return 130
        except Exception as exc:
            err={'status':'S2_74B_FAILED','error':type(exc).__name__+': '+str(exc),'elapsed_seconds':time.perf_counter()-start}
            write_json(out/'metrics'/f's2_74b_failed_{run}.json',err)
            print('\n'+err['error'],file=sys.stderr)
            print('No input was changed. Send this error and failure report.',file=sys.stderr)
            return 1
        finally:
            if limiter is not None: limiter.restore_original_limits()

if __name__=='__main__':
    raise SystemExit(main())
