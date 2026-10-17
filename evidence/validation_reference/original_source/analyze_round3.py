"""Exploratory post hoc Validation-reference EEG analyses; original inputs read-only.

Prepare the immutable settings/selection plan before waveform access:
  .venv/Scripts/python.exe -B reports/icspis_2026_paper/tools/analyze_round3.py --prepare
Then run (CPU only, two BLAS threads, resumable per participant):
  .venv/Scripts/python.exe -B reports/icspis_2026_paper/tools/analyze_round3.py --run
Only analysis_round3/ is written. No Test datasets, model checkpoints or training.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np
import scipy
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/icspis_2026_paper/analysis_round3"
DATA = Path(r"D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\processed\all-blocks-v1")
SPLIT = ROOT / "outputs/metrics/scenario2/final_block_split_manifest.csv"
NORM = ROOT / "outputs/metrics/scenario2/train_normalization.json"
ORIGINAL = ROOT / "outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1/tables"
POOL_DIRS = {
    "B0": ROOT / "outputs/runs/scenario2/ablation/b0_state_only_1x_s2_74c_v1",
    "B1": ROOT / "outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1",
    "B2": ROOT / "outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1",
}
MANIFESTS = {"B0": POOL_DIRS["B0"] / "manifests/s2_74c_b0_pool_manifest_v1.json",
             "B1": POOL_DIRS["B1"] / "manifests/s2_73_pool_manifest_v1.json"}
FS, NC, NT = 128, 19, 512
STATES, GENERATORS = ("EO", "EC"), ("B0", "B1", "B2")
BANDS = ((1.,4.),(4.,8.),(8.,13.),(13.,30.))
TINY = np.finfo(np.float64).tiny

def require(ok, text):
    if not ok:
        raise RuntimeError(text)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)

def write_csv(path, data):
    require(bool(data), "Empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def truth(v):
    return str(v).lower() in ("true", "1", "yes")

def text(v):
    return v.decode() if isinstance(v,bytes) else str(v)

def prepare():
    require(not (OUT/"plan.json").exists(), "Plan already exists; do not overwrite an analysis plan")
    split=rows(SPLIT); norm=load(NORM)
    subjects=sorted({r["subject_id"] for r in split})
    require(len(subjects)==198, "Expected 198 participants")
    eligible=[r for r in split if r["split"] in ("Train","Validation") and truth(r["block_eligible"])]
    manifests={g:load(MANIFESTS["B0" if g=="B0" else "B1"]) for g in GENERATORS}
    selections={}; file_inventory=[]
    for si,sid in enumerate(subjects):
        h5=DATA/(sid+"_all_blocks.h5")
        require(h5.is_file(), "Missing "+str(h5))
        src={r["source_output_sha256"] for r in split if r["subject_id"]==sid}
        require(len(src)==1,"Inconsistent source hash")
        file_inventory.append({"subject_id":sid,"path":str(h5),"bytes":h5.stat().st_size,"expected_sha256":next(iter(src))})
        for ti,st in enumerate(STATES):
            ns={sp:sum(int(r["usable_windows"]) for r in eligible if r["subject_id"]==sid and r["eye_state"]==st and r["split"]==sp) for sp in ("Train","Validation")}
            require(min(ns.values())>0,"Empty stratum")
            n=min(ns.values());rng=np.random.RandomState(2026101702+2*si+ti)
            ent={"counts":ns,"matched_n":n,"Train_indices":sorted(rng.choice(ns["Train"],n,replace=False).tolist()),"Validation_indices":sorted(rng.choice(ns["Validation"],n,replace=False).tolist()),"synthetic_indices":sorted(rng.choice(ns["Train"],n,replace=False).tolist()),"synthetic":{}}
            for g in GENERATORS:
                row=manifests[g]["committed"][g+"/"+sid+"/"+st]
                require(row["expected_count"]==ns["Train"],"Synthetic count mismatch")
                path=POOL_DIRS[g]/row["file"]
                require(path.is_file(),"Missing "+str(path))
                ent["synthetic"][g]={"path":str(path),"bytes":path.stat().st_size,"sha256":row["sha256"]}
            selections[sid+"/"+st]=ent
    require(sum(x["counts"]["Train"] for x in selections.values())==47835,"Train count")
    require(sum(x["counts"]["Validation"] for x in selections.values())==9635,"Validation count")
    rng=np.random.RandomState(20261017);sid=subjects[int(rng.randint(198))];st=STATES[int(rng.randint(2))];ch=int(rng.randint(19))
    illustration={"seed":20261017,"subject_id":sid,"state":st,"channel_index":ch,"channel":norm["channel_order"][ch],"window_indices":{}}
    for g in ("Validation",)+GENERATORS:
        n=selections[sid+"/"+st]["counts"]["Validation" if g=="Validation" else "Train"]
        illustration["window_indices"][g]=int(rng.randint(n))
    paths=[SPLIT,NORM,Path(__file__),*set(MANIFESTS.values()),ORIGINAL/"subject_state_metrics.csv",ORIGINAL/"subject_alpha_effect.csv",ROOT/"reports/icspis_2026_paper/figure_sources/fidelity_data.json",ROOT/"scripts/data/s2_74b_final_pool_fidelity.py",ROOT/"scripts/data/s2_74e_b0_fidelity_extension.py"]
    plan={"schema":"round3_exploratory_validation_v1","status":"SETTINGS_FROZEN_BEFORE_WAVEFORM_READS","created_utc":now(),"interpretation":"Exploratory post hoc analysis, not preregistered confirmatory evidence. Validation blocks were held out from gradient training, but used in model development/selection. Test waveforms will not be accessed.","allowed_real_splits":["Train","Validation"],"new_training":False,"new_generation":False,"model_selection":False,"threads":2,"subjects":subjects,"processing_order":[sid]+[s for s in subjects if s!=sid],"channels":norm["channel_order"],"window":{"fs":128,"samples":512,"channels":19},"features":{"welch":"Hann, nperseg=256, noverlap=128, constant detrend, density, one-sided; mean linear PSD then 10log10","log_psd_mae":"absolute dB difference averaged over 19 channels and 1..40Hz inclusive","bands_hz_inclusive":BANDS,"alpha":"median of window-wise 10log10 integrated8..13Hz power; EC-minus-EO 19-channel vector, channel MAE and Pearson r","fingerprint":"median 10log10 bandpowers, band-major76-vector; Pearson retrieval over198same-state real references; pessimistic ties","nearest":"min waveform RMSE over19x512globallyTrain-standardized coordinates against all same-stratumTrain windows; median of querydistances, then meanEO/EC and participants","precision":"Stored float32 standardized windows promoted to float64 for computations"},"comparisons":{"full":"All fixed synthetic windows versus all eligible Validation windows; Train-vs-Validation reference and exact Train-reference reproduction also computed","matched":"One frozen without-replacement subset per source/stratum, n=min(nTrain,nValidation); same indices acrosssyntheticmodels; same realValidationreference; no repeated-selection search","nearest":"Validationqueries vs allTrain; originalfullsynthetic-stratum medians fromsavedtables; also matchedsyntheticquery subset vs allTrain; no privacythreshold","dispersion":"OriginalTableII first3rows: participantmeansofstates orparticipantalphaerror, median/Q1/Q3/IQR, pairedSSC-minus-SC bootstrapCI"},"bootstrap":{"seed":2026101703,"repetitions":5000,"unit":"participant","interval":"percentile95%, unadjusted, conditional on fixed data/model/pool/subset","seeds_resampled":False},"selections":selections,"real_file_inventory":file_inventory,"illustration":illustration,"source_hashes":[{"path":str(p),"sha256":sha(p)} for p in paths],"runtime":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"h5py":h5py.__version__}}
    write(OUT/"plan.json",plan)
    print("PLAN FROZEN",OUT/"plan.json","illustration",illustration,flush=True)

def corr(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    require(np.isfinite(a).all() and np.isfinite(b).all(),"Nonfinite correlation input")
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    require(den>1e-15,"Degenerate correlation")
    return float(a@b/den)

def signal_features(z,norm):
    z=np.asarray(z,dtype=np.float64)
    uv=z*np.asarray(norm["stds_uv"])[None,:,None]+np.asarray(norm["means_uv"])[None,:,None]
    f,p=welch(uv,fs=128,window="hann",nperseg=256,noverlap=128,detrend="constant",scaling="density",axis=-1)
    bp=np.stack([np.trapezoid(p[...,(f>=lo)&(f<=hi)],f[(f>=lo)&(f<=hi)],axis=-1) for lo,hi in BANDS],axis=1)
    return {"psd":p,"bp_db":10*np.log10(np.maximum(bp,TINY)),"uv":uv,"freq":f}

def small_features(f,idx=None):
    p=f["psd"] if idx is None else f["psd"][idx]
    b=f["bp_db"] if idx is None else f["bp_db"][idx]
    u=f["uv"] if idx is None else f["uv"][idx]
    mask=(f["freq"]>=1)&(f["freq"]<=40)
    return {"log_psd":10*np.log10(np.maximum(p.mean(0)[:,mask],TINY)),"median_bp":np.median(b,axis=0),"pooled_sd":u.transpose(1,0,2).reshape(NC,-1).std(axis=1)}

def distance_metrics(a,b):
    return {"log_psd_mae_db":float(np.abs(a["log_psd"]-b["log_psd"]).mean()),"sd_md_pooled_uv":float(np.abs(a["pooled_sd"]-b["pooled_sd"]).mean())}

def nearest(query,reference):
    a=np.asarray(query,dtype=np.float64).reshape(len(query),-1);b=np.asarray(reference,dtype=np.float64).reshape(len(reference),-1)
    d2=np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*(a@b.T),0)/a.shape[1]
    d=np.sqrt(d2.min(1));q=np.quantile(d,[.05,.25,.5,.75,.95])
    return {"n_queries":len(a),"n_reference":len(b),"median":float(q[2]),"q05":float(q[0]),"q25":float(q[1]),"q75":float(q[3]),"q95":float(q[4]),"mean":float(d.mean())}

def read_real(sid,sp,st,split,norm,h5):
    require(sp in ("Train","Validation"),"Forbidden split")
    chosen=sorted([r for r in split if r["subject_id"]==sid and r["split"]==sp and r["eye_state"]==st and truth(r["block_eligible"])],key=lambda r:int(r["block_index"]))
    values=[];records=[]
    for row in chosen:
        b=h5["blocks/"+row["block_uid"]]
        require(text(b.attrs["provisional_split"])==sp and text(b.attrs["state"])==st,"Split/state mismatch")
        require(json.loads(text(b.attrs["channel_names"]))==norm["channel_order"] and text(b.attrs["physical_unit"])=="uV" and float(b.attrs["processed_sfreq"])==128,"Signal schema")
        starts=b["window_start_samples"][:];mask=b["window_usable_mask"][:].astype(bool)
        require(int(mask.sum())==int(row["usable_windows"]),"Window count mismatch")
        for i in np.flatnonzero(mask):
            start=int(starts[i]);v=np.asarray(b["eeg_uv"][:,start:start+NT],dtype=np.float32)
            require(v.shape==(NC,NT) and np.isfinite(v).all(),"Real input invalid")
            z=((v.astype(np.float64)-np.asarray(norm["means_uv"])[:,None])/np.asarray(norm["stds_uv"])[:,None]).astype(np.float32)
            values.append(z);records.append({"block_uid":row["block_uid"],"candidate_window_index":int(i),"start_sample":start})
    return np.stack(values),records

def analyze_subject(sid,plan,split,norm,plan_sha):
    started=time.perf_counter();inventory=next(x for x in plan["real_file_inventory"] if x["subject_id"]==sid);hpath=Path(inventory["path"])
    require(sha(hpath)==inventory["expected_sha256"],"Real input SHA changed: "+sid)
    result={"subject_id":sid,"plan_sha256":plan_sha,"states":{},"source_hdf5_sha256":inventory["expected_sha256"],"Test_waveforms_read":False}
    with h5py.File(hpath,"r") as h5:
        for st in STATES:
            sel=plan["selections"][sid+"/"+st];arrays={};records={}
            for sp in ("Train","Validation"):
                arrays[sp],records[sp]=read_real(sid,sp,st,split,norm,h5)
                require(len(arrays[sp])==sel["counts"][sp],"Frozen count mismatch")
            sources={}
            for g in GENERATORS:
                meta=sel["synthetic"][g];path=Path(meta["path"])
                require(sha(path)==meta["sha256"],"Synthetic source hash changed")
                with np.load(path,allow_pickle=False) as z:arrays[g]=z["x_z"]
                require(arrays[g].shape==arrays["Train"].shape and arrays[g].dtype==np.float32 and np.isfinite(arrays[g]).all(),"Synthetic schema")
                sources[g]=meta["sha256"]
            feats={g:signal_features(z,norm) for g,z in arrays.items()}
            state={"counts":{g:len(z) for g,z in arrays.items()},"matched_n":sel["matched_n"],"source_synthetic_sha256":sources,"full":{},"matched":{}}
            for mode in ("full","matched"):
                sm={g:small_features(f,None if mode=="full" else sel[(g if g in ("Train","Validation") else "synthetic")+"_indices"]) for g,f in feats.items()}
                state[mode]["metrics_vs_Validation"]={g:distance_metrics(sm[g],sm["Validation"]) for g in ("Train",)+GENERATORS}
                state[mode]["metrics_vs_Train"]={g:distance_metrics(sm[g],sm["Train"]) for g in ("Validation",)+GENERATORS}
                state[mode]["fingerprint"]={g:f["median_bp"].ravel().tolist() for g,f in sm.items()}
                state[mode]["alpha_median_db"]={g:f["median_bp"][2].tolist() for g,f in sm.items()}
            state["nearest_to_Train"]={"Validation_all":nearest(arrays["Validation"],arrays["Train"])}
            for g in GENERATORS:state["nearest_to_Train"][g+"_matched_queries"]=nearest(arrays[g][sel["synthetic_indices"]],arrays["Train"])
            ill=plan["illustration"]
            if sid==ill["subject_id"] and st==ill["state"]:
                ci=ill["channel_index"];indices=ill["window_indices"]
                illustration={"selection_frozen_in_plan":ill,"plan_sha256":plan_sha,"scope":"Independently selected illustrative windows; not paired reconstructions or participant-level performance evidence.","units":"microvolts","sampling_rate_hz":128,"time_seconds":(np.arange(NT)/FS).tolist(),"signals_uv":{g:feats[g]["uv"][idx,ci].tolist() for g,idx in indices.items()},"real_source":{"path":str(hpath),"sha256":inventory["expected_sha256"],"split":"Validation",**records["Validation"][indices["Validation"]]},"synthetic_sources":{g:{**sel["synthetic"][g],"window_index":indices[g]} for g in GENERATORS}}
                write(OUT/"illustrative_waveforms.json",illustration)
                print("ILLUSTRATION_EXPORTED",sid,st,ill["channel"],flush=True)
            result["states"][st]=state
    require(sha(hpath)==inventory["expected_sha256"],"Real file changed during read")
    result["elapsed_seconds"]=time.perf_counter()-started
    write(OUT/"participants"/(sid+".json"),result)
    return result

def aggregate(results,plan):
    subjects=plan["subjects"];results=sorted(results,key=lambda r:r["subject_id"])
    require([r["subject_id"] for r in results]==subjects,"Missing participant")
    rng=np.random.RandomState(plan["bootstrap"]["seed"]);ix=rng.randint(0,198,size=(5000,198))
    def stats(v):
        a=np.asarray(v,dtype=np.float64);require(a.shape==(198,) and np.isfinite(a).all(),"Invalid participant vector")
        q=np.quantile(a,[.25,.5,.75]);ci=np.quantile(a[ix].mean(1),[.025,.975])
        return {"n_participants":198,"mean":float(a.mean()),"median":float(q[1]),"q25":float(q[0]),"q75":float(q[2]),"iqr":float(q[2]-q[0]),"ci95_mean":ci.tolist()}
    def comparisons(values):
        out={g:stats(v) for g,v in values.items()}
        for left,right in (("B1","B0"),("B2","B1"),("B2","B0"),("B0","Train"),("B1","Train"),("B2","Train")):
            if left in values and right in values:out[left+"_minus_"+right]=stats(np.array(values[left])-np.array(values[right]))
        return out
    summary={"status":"COMPLETE","interpretation":plan["interpretation"],"plan_sha256":sha(OUT/"plan.json"),"n_participants":198,"Test_waveforms_read":False,"bootstrap":plan["bootstrap"],"fidelity":{},"fingerprint":{},"nearest":{},"original_table_dispersion":{}}
    participant_rows=[];state_rows=[];finger_rows=[]
    for mode in ("full","matched"):
        summary["fidelity"][mode]={};summary["fingerprint"][mode]={}
        for ref in ("Validation","Train"):
            sources=(("Train",) if ref=="Validation" else ("Validation",))+GENERATORS
            md={}
            for metric in ("log_psd_mae_db","sd_md_pooled_uv"):
                vals={g:[np.mean([r["states"][st][mode]["metrics_vs_"+ref][g][metric] for st in STATES]) for r in results] for g in sources}
                md[metric]=comparisons(vals)
                for g,v in vals.items():
                    for sid,val in zip(subjects,v):participant_rows.append({"subject_id":sid,"mode":mode,"reference":ref,"source":g,"metric":metric,"value":float(val)})
            for metric in ("alpha_effect_mae_db","alpha_effect_pattern_r"):
                vals={g:[] for g in sources}
                for r in results:
                    effects={g:np.asarray(r["states"]["EC"][mode]["alpha_median_db"][g])-np.asarray(r["states"]["EO"][mode]["alpha_median_db"][g]) for g in ("Train","Validation")+GENERATORS}
                    for g in sources:vals[g].append(float(np.abs(effects[g]-effects[ref]).mean()) if metric.endswith("mae_db") else corr(effects[g],effects[ref]))
                md[metric]=comparisons(vals)
                for g,v in vals.items():
                    for sid,val in zip(subjects,v):participant_rows.append({"subject_id":sid,"mode":mode,"reference":ref,"source":g,"metric":metric,"value":float(val)})
            summary["fidelity"][mode]["versus_"+ref]=md
            for st in STATES:
                real=np.array([r["states"][st][mode]["fingerprint"][ref] for r in results])
                for g in sources:
                    syn=np.array([r["states"][st][mode]["fingerprint"][g] for r in results]);sims=np.array([[corr(a,b) for b in real] for a in syn]);ranks=[]
                    for i,sid in enumerate(subjects):
                        others=np.delete(sims[i],i);rank=1+int((others>=sims[i,i]).sum());ranks.append(rank)
                        finger_rows.append({"subject_id":sid,"mode":mode,"reference":ref,"state":st,"source":g,"rank":rank,"correct_subject_r":float(sims[i,i]),"correct_vs_best_wrong_margin":float(sims[i,i]-others.max())})
                    summary["fingerprint"][mode][ref+"_"+st+"_"+g]={"n_queries":198,"n_candidates":198,"top1_count":int((np.array(ranks)<=1).sum()),"top5_count":int((np.array(ranks)<=5).sum()),"top1_percent":float((np.array(ranks)<=1).mean()*100),"top5_percent":float((np.array(ranks)<=5).mean()*100)}
    original=rows(ORIGINAL/"subject_state_metrics.csv");alpha=rows(ORIGINAL/"subject_alpha_effect.csv")
    om={(r["subject_id"],r["state"],r["generator"]):r for r in original};oa={(r["subject_id"],r["generator"]):r for r in alpha}
    reproduction=[]
    for metric in ("log_psd_mae_db","sd_md_pooled_uv","alpha_effect_mae_db"):
        values={g:[float(oa[sid,g]["channel_mae_db"]) if metric=="alpha_effect_mae_db" else np.mean([float(om[sid,st,g][metric]) for st in STATES]) for sid in subjects] for g in GENERATORS}
        summary["original_table_dispersion"][metric]=comparisons(values)
        for g in GENERATORS:
            found=[r["value"] for r in participant_rows if r["mode"]=="full" and r["reference"]=="Train" and r["source"]==g and r["metric"]==metric]
            err=float(np.max(np.abs(np.asarray(found)-values[g])));reproduction.append({"metric":metric,"source":g,"max_abs_error":err,"passed":err<1e-9})
    vals={"Validation":[np.mean([r["states"][st]["nearest_to_Train"]["Validation_all"]["median"] for st in STATES]) for r in results]}
    for g in GENERATORS:
        vals[g+"_all_saved"]=[np.mean([float(om[sid,st,g]["nearest_real_rmse_z_median"]) for st in STATES]) for sid in subjects]
        vals[g+"_matched_queries"]=[np.mean([r["states"][st]["nearest_to_Train"][g+"_matched_queries"]["median"] for st in STATES]) for r in results]
    summary["nearest"]["summaries"]={g:stats(v) for g,v in vals.items()}
    summary["nearest"]["differences_vs_Validation"]={g:stats(np.array(v)-np.array(vals["Validation"])) for g,v in vals.items() if g!="Validation"}
    for g,v in vals.items():
        for sid,val in zip(subjects,v):participant_rows.append({"subject_id":sid,"mode":"full_or_matched_as_named","reference":"all_same_stratum_Train","source":g,"metric":"nearest_rmse_stratum_median","value":float(val)})
    for r in results:
        for st,s in r["states"].items():
            for g,d in s["nearest_to_Train"].items():state_rows.append({"subject_id":r["subject_id"],"state":st,"source":g,**d})
    summary["original_result_reproduction"]=reproduction
    require(all(x["passed"] for x in reproduction),"Original method reproduction discrepancy; inspect before using results")
    summary["counts"]={"Train_windows":sum(r["states"][st]["counts"]["Train"] for r in results for st in STATES),"Validation_windows":sum(r["states"][st]["counts"]["Validation"] for r in results for st in STATES),"matched_windows_per_source":sum(r["states"][st]["matched_n"] for r in results for st in STATES),"matched_min":min(r["states"][st]["matched_n"] for r in results for st in STATES),"matched_max":max(r["states"][st]["matched_n"] for r in results for st in STATES)}
    summary["limitations"]=["Exploratory post hoc, not independent confirmation; Validation contributed to model development/checkpoint selection.","Same participants acrossTrain/Validation; oneValidationblock perstate, overlapping4swindows with2sstride.","Equal window count does not equalize block diversity or temporal dependence: Train spansmultipleblocks; Validation oneblockperstate.","Intervals unadjusted, participantbootstrap only; generator retraining/pool sampling/subset uncertainty not included.","Nearest distances compare sharedTrain reference; no validated copying/privacy threshold, temporalalignment or membershipinference test."]
    write_csv(OUT/"per_participant.csv",participant_rows);write_csv(OUT/"nearest_per_stratum.csv",state_rows);write_csv(OUT/"fingerprint_per_participant.csv",finger_rows)
    write(OUT/"summary.json",summary)
    methods="""# Exploratory post hoc Validation-reference analysis\n\nSettings and all random selections were frozen in plan.json before waveform access. Only Train and Validation blocks were read; Test waveform datasets were never indexed. Validation was excluded from generator gradient training but participated in development and checkpoint selection, so these analyses are exploratory, not independent confirmatory validation. No model was trained, generated, selected or changed.\n\nWelch PSD and bandpower definitions match the original fidelity analysis: Hann256, overlap128, constant detrending,128Hz; mean linear PSD before10log10; inclusive1–40Hz log-PSD MAE; inclusive bands1–4/4–8/8–13/13–30Hz. Alpha effects are EC-minus-EO differences of per-channel window-median alpha dB power. Fingerprints are76-dimensional median bandpower vectors, ranked by Pearson correlation against198 same-state real references, with pessimistic tie handling. All existing synthetic windows were compared with all eligible Validation windows; Train-to-Validation comparisons use the same metric definitions. A separate frozen count-matched variant uses one without-replacement subset per source/stratum and equal window counts; it does not equalize number of blocks or temporal dependence.\n\nNearest-window distances are minimum RMSE over19x512 globallyTrain-standardized coordinates against all same-stratumTrain windows. Query distances are summarized by their stratum median, followed by equal EO/EC and participant averages. Validation queries are all eligibleValidationwindows. Both savedfullsynthetic medians and newlycomputedcountmatchedsynthetic-query medians are reported; the latter use the frozen matched subsets. Waveform distances do not establish a privacy guarantee or a near-copy threshold.\n\nOriginalTableII dispersion uses saved participant rows: Q1/median/Q3/IQR and pairedSSC-minus-SC percentile95% intervals from5000participantbootstrapresamples. No windows, model seeds or syntheticpools are resampled. All intervals are unadjusted. ExactTrain-reference reproduction checks for the first3TableIIrows accompany summary.json.\n\nThe waveform illustration was selected entirely from metadata using frozen seed20261017 before waveform viewing: participant,state,channel and independent windowindices. The curves are unpaired examples in microvolts, not reconstructions. Sourcefilehashes and windowindices accompany the portableJSON.\n\nCode definitions checked against scripts/data/s2_74b_final_pool_fidelity.py:309–338,351–378,405–406,499–502,585–605, and s2_74e_b0_fidelity_extension.py:183–214,245–253.\n"""
    (OUT/"methods_and_evidence.md").write_text(methods,encoding="utf-8")
    write(OUT/"completion.json",{"status":"COMPLETE","completed_utc":now(),"participants":198,"plan_sha256":sha(OUT/"plan.json"),"script_sha256":sha(Path(__file__)),"summary_sha256":sha(OUT/"summary.json"),"all_original_reproduction_checks_passed":True,"Test_waveforms_read":False,"new_training":False,"new_generation":False})
    print("COMPLETE",OUT/"summary.json",flush=True)

def run(limit):
    plan=load(OUT/"plan.json");plan_sha=sha(OUT/"plan.json")
    for item in plan["source_hashes"]:require(sha(item["path"])==item["sha256"],"Frozen source changed: "+item["path"])
    split=rows(SPLIT);norm=load(NORM);results=[]
    for i,sid in enumerate(plan["processing_order"]):
        path=OUT/"participants"/(sid+".json")
        if path.exists():
            result=load(path);require(result["plan_sha256"]==plan_sha,"Stale cache")
        else:
            result=analyze_subject(sid,plan,split,norm,plan_sha)
            print(f"PARTICIPANT {i+1}/198 {sid} {result['elapsed_seconds']:.2f}s",flush=True)
        results.append(result)
        if limit and len(results)>=limit:
            print("STOPPED_AFTER",limit,"participants; resumable",flush=True);return
    aggregate(results,plan)

if __name__=="__main__":
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--prepare",action="store_true");group.add_argument("--run",action="store_true");parser.add_argument("--stop-after",type=int,default=0);args=parser.parse_args()
    if args.prepare:prepare()
    else:run(args.stop_after)
