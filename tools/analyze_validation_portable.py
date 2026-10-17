"""Portable derivative of the archived exploratory Validation-reference analysis.

The original scientific source is preserved separately, SHA-256:
418892ffff6e31f92f38112da3506369581b8e555935a6889c4397c3ae4f70b6.
See docs/PORTABLE_VALIDATION_ANALYSIS.md for lineage and execution boundaries.
--help and --check-config require only Python's standard library and never open
waveform bytes. --prepare/--run are explicit external-data research operations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FS, NC, NT = 128, 19, 512
STATES, GENERATORS = ("EO", "EC"), ("B0", "B1", "B2")
BANDS = ((1.,4.),(4.,8.),(8.,13.),(13.,30.))
ORIGINAL_SOURCE_SHA256 = "418892ffff6e31f92f38112da3506369581b8e555935a6889c4397c3ae4f70b6"
# Canonical JSON (sorted keys, compact separators) of the original frozen plan's
# features/comparisons objects, retaining their exact metric-definition labels.
DEFINITIONS_SHA256 = "49945dd4f5a5445d76ac028c6ffc392216a62752431ea16a38d5c8924073e302"
CONFIG_SCHEMA = "portable_validation_analysis_config_v1"
REPOSITORY = Path(__file__).resolve().parents[1]
_CONFIG = {}
_RESOLVED = {}

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
                path=_pool_path(g,row)
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
    paths=list(_metadata_paths().values())
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
    started=time.perf_counter();inventory=next(x for x in plan["real_file_inventory"] if x["subject_id"]==sid);hpath=_resolved_path(inventory["path"])
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
                meta=sel["synthetic"][g];path=_resolved_path(meta["path"])
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


# Portability layer. The feature, distance, retrieval and aggregation functions
# above retain the archived implementation; this layer controls paths and state.
def _digest(value, description):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "Invalid SHA-256 for " + description)
    return value


def _inside(child, parent):
    return child == parent or parent in child.parents


def _path(value, base):
    require(isinstance(value, str) and bool(value.strip()), "Expected a nonempty path")
    require(not any(c in value for c in "<>"), "Replace example path placeholders")
    p = Path(value).expanduser()
    # A Windows absolute path is not a portable relative path on POSIX.
    if os.name != "nt":
        require(not re.match(r"^[A-Za-z]:[\\/]", value),
                "Replace the Windows path with this machine's local path: " + value)
    return (p if p.is_absolute() else base / p).resolve()


def _metadata_paths():
    return {
        "split_manifest": SPLIT,
        "normalization": NORM,
        "original_script": _CONFIG["original_script"],
        "portable_script": Path(__file__).resolve(),
        "B0_pool_manifest": MANIFESTS["B0"],
        "B1_B2_pool_manifest": MANIFESTS["B1"],
        "subject_state_metrics": ORIGINAL / "subject_state_metrics.csv",
        "subject_alpha_effect": ORIGINAL / "subject_alpha_effect.csv",
        "original_aggregate_fidelity": _CONFIG["original_aggregate_fidelity"],
        "original_B1_B2_fidelity_script": _CONFIG["original_fidelity_scripts"]["B1_B2"],
        "original_B0_fidelity_script": _CONFIG["original_fidelity_scripts"]["B0"],
    }


def _configure(config_path):
    global _CONFIG, OUT, DATA, SPLIT, NORM, ORIGINAL, POOL_DIRS, MANIFESTS
    config_path = Path(config_path).resolve()
    require(config_path.is_file(), "Missing configuration: " + str(config_path))
    raw = load(config_path)
    require(isinstance(raw, dict), "Configuration must be a JSON object")
    required = {"schema", "processed_hdf5_dir", "split_manifest", "normalization",
                "pool_dirs", "pool_manifests", "original_fidelity_tables",
                "original_aggregate_fidelity", "original_fidelity_scripts",
                "original_script", "output_dir"}
    require(required <= raw.keys(), "Missing configuration keys: " + ", ".join(sorted(required - raw.keys())))
    require(raw.keys() <= required | {"frozen_plan", "frozen_plan_sha256"},
            "Unknown configuration keys: " + ", ".join(sorted(raw.keys() - required - {"frozen_plan", "frozen_plan_sha256"})))
    require(raw["schema"] == CONFIG_SCHEMA, "Unsupported configuration schema")
    base = config_path.parent
    cfg = {"configuration_path": config_path, "configuration_sha256": sha(config_path)}
    for name in ("processed_hdf5_dir", "split_manifest", "normalization",
                 "original_fidelity_tables", "original_aggregate_fidelity",
                 "original_script", "output_dir"):
        cfg[name] = _path(raw[name], base)
    for name, keys in (("pool_dirs", set(GENERATORS)), ("pool_manifests", {"B0", "B1"}),
                       ("original_fidelity_scripts", {"B0", "B1_B2"})):
        require(isinstance(raw[name], dict) and set(raw[name]) == keys, "Wrong keys in " + name)
        cfg[name] = {k: _path(v, base) for k, v in raw[name].items()}
    if raw.get("frozen_plan") is not None:
        cfg["frozen_plan"] = _path(raw["frozen_plan"], base)
        cfg["frozen_plan_sha256"] = _digest(raw.get("frozen_plan_sha256"), "frozen plan")
    else:
        require(raw.get("frozen_plan_sha256") is None, "A frozen-plan digest requires frozen_plan")
    _CONFIG = cfg
    OUT, DATA = cfg["output_dir"], cfg["processed_hdf5_dir"]
    SPLIT, NORM = cfg["split_manifest"], cfg["normalization"]
    ORIGINAL, POOL_DIRS, MANIFESTS = cfg["original_fidelity_tables"], cfg["pool_dirs"], cfg["pool_manifests"]
    # Scientific outputs include private participant records. Keep them outside
    # the publication checkout, even if a caller supplies a permissive .gitignore.
    require(not _inside(OUT, REPOSITORY), "Output must be outside the publication repository")
    input_dirs = {DATA, ORIGINAL, *POOL_DIRS.values()}
    for directory in input_dirs:
        require(not _inside(OUT, directory) and not _inside(directory, OUT),
                "Output must not overlap an input directory: " + str(directory))
    inputs = set(_metadata_paths().values()) | {config_path}
    if "frozen_plan" in cfg:
        inputs.add(cfg["frozen_plan"])
    require(not any(_inside(p, OUT) for p in inputs), "Output must not contain any input file")
    return cfg


def _metadata_check():
    """Inspect declared metadata only. Do not hash/open HDF5 or NPZ files."""
    for path in {DATA, ORIGINAL, *POOL_DIRS.values()}:
        require(path.is_dir(), "Missing input directory: " + str(path))
    for name, path in _metadata_paths().items():
        expected_suffix = ".py" if "script" in name else ".csv" if name in {
            "split_manifest", "subject_state_metrics", "subject_alpha_effect"} else ".json"
        require(path.suffix.lower() == expected_suffix, "Unexpected metadata file extension: " + name)
        require(path.is_file(), "Missing metadata file: " + str(path))
    require(sha(_CONFIG["original_script"]) == ORIGINAL_SOURCE_SHA256,
            "Archived original analysis source hash mismatch")
    split = rows(SPLIT)
    columns = {"subject_id", "split", "block_eligible", "eye_state", "usable_windows",
               "source_output_sha256", "block_index", "block_uid"}
    require(bool(split) and columns <= split[0].keys(), "Invalid split-manifest schema")
    subjects = sorted({r["subject_id"] for r in split})
    require(len(subjects) == 198, "Expected 198 participants in split metadata")
    require(all(re.fullmatch(r"[A-Za-z0-9_-]+", sid) for sid in subjects), "Unsafe subject identifier")
    for r in split:
        require(r["split"] in ("Train", "Validation", "Test"), "Unknown split label")
        require(r["eye_state"] in STATES, "Unknown eye-state label")
        require(int(r["usable_windows"]) >= 0, "Negative window count")
        _digest(r["source_output_sha256"], "split HDF5 reference")
    norm = load(NORM)
    require(isinstance(norm, dict), "Normalization must be a JSON object")
    require(len(norm.get("channel_order", [])) == NC and
            len(set(norm["channel_order"])) == NC, "Expected 19 distinct channels")
    for key in ("means_uv", "stds_uv"):
        values = norm.get(key)
        require(isinstance(values, list) and len(values) == NC and
                all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values),
                "Invalid normalization " + key)
    require(all(v > 0 for v in norm["stds_uv"]), "Normalization standard deviations must be positive")
    manifests = {g: load(path) for g, path in MANIFESTS.items()}
    for g, manifest in manifests.items():
        require(isinstance(manifest, dict) and isinstance(manifest.get("committed"), dict),
                "Invalid pool manifest: " + g)
    aggregate_input = load(_CONFIG["original_aggregate_fidelity"])
    require(isinstance(aggregate_input, dict) and bool(aggregate_input), "Invalid original aggregate fidelity JSON")
    for filename, required in (
        ("subject_state_metrics.csv", {"subject_id", "state", "generator", "log_psd_mae_db",
                                       "sd_md_pooled_uv", "nearest_real_rmse_z_median"}),
        ("subject_alpha_effect.csv", {"subject_id", "generator", "channel_mae_db"}),
    ):
        data = rows(ORIGINAL / filename)
        require(bool(data) and required <= data[0].keys(), "Invalid original fidelity table: " + filename)
    return split, norm, manifests


def _register_path(mapping, recorded, resolved):
    require(isinstance(recorded, str) and bool(recorded), "Missing frozen path label")
    resolved = Path(resolved).resolve()
    require(recorded not in mapping or mapping[recorded] == resolved,
            "Conflicting resolutions for one frozen path label")
    mapping[recorded] = resolved


def _bind_metadata(plan, mapping):
    """Match frozen source records by content identity, never by old roots."""
    known = _metadata_paths()
    by_hash = {}
    for role, path in known.items():
        by_hash.setdefault(sha(path), []).append((role, path))
    entries = plan.get("source_hashes")
    require(isinstance(entries, list) and bool(entries), "Missing frozen source hashes")
    bound_roles = set()
    for entry in entries:
        require(isinstance(entry, dict), "Invalid frozen source hash record")
        expected = _digest(entry.get("sha256"), "frozen metadata source")
        candidates = by_hash.get(expected, [])
        require(bool(candidates), "Frozen metadata hash has no configured match: " + expected)
        bound_roles.update(role for role, _ in candidates)
        # Identical copies are interchangeable for the sole source-hash check.
        _register_path(mapping, entry.get("path"), candidates[0][1])
    require(set(known) - {"portable_script"} <= bound_roles,
            "The frozen plan does not bind all required configured metadata")


def _pool_path(generator, entry):
    value = entry.get("file")
    require(isinstance(value, str) and bool(value), "Missing pool-manifest file")
    # Accept either separator from the original Windows manifests; reject all
    # absolute paths and traversal. The configured pool root is authoritative.
    value = value.replace("\\", "/")
    require(not value.startswith("/") and not re.match(r"^[A-Za-z]:", value),
            "Pool-manifest file must be relative")
    require(all(part not in ("", ".", "..") for part in value.split("/")),
            "Unsafe pool-manifest file")
    root = POOL_DIRS[generator]
    result = (root / value).resolve()
    require(_inside(result, root), "Pool file escapes its configured directory")
    require(result.suffix.lower() == ".npz", "Expected a synthetic NPZ file")
    return result


def _file_stat(path, size):
    require(type(size) is int and size >= 0, "Invalid frozen byte count")
    require(path.is_file(), "Missing external waveform file: " + str(path))
    require(path.stat().st_size == size, "External file byte count changed: " + str(path))


def _bind_wave_paths(plan, manifests, mapping):
    """Bind private inventories without reading any waveform bytes."""
    for entry in plan["real_file_inventory"]:
        sid = entry["subject_id"]
        require(isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", sid), "Unsafe subject identifier")
        path = (DATA / (sid + "_all_blocks.h5")).resolve()
        require(_inside(path, DATA), "HDF5 path escapes configured storage")
        _digest(entry.get("expected_sha256"), "real inventory")
        _file_stat(path, entry.get("bytes"))
        _register_path(mapping, entry.get("path"), path)
    for key, selection in plan["selections"].items():
        for generator in GENERATORS:
            entry = manifests["B0" if generator == "B0" else "B1"]["committed"].get(generator + "/" + key)
            require(isinstance(entry, dict), "Missing synthetic manifest entry: " + generator + "/" + key)
            path = _pool_path(generator, entry)
            frozen = selection["synthetic"][generator]
            expected = _digest(frozen.get("sha256"), "synthetic selection")
            require(entry.get("sha256") == expected, "Frozen selection / pool manifest hash mismatch")
            require(entry.get("expected_count") == selection["counts"]["Train"], "Pool count mismatch")
            _file_stat(path, frozen.get("bytes"))
            _register_path(mapping, frozen.get("path"), path)


def _indices(value, count, selected):
    return (isinstance(value, list) and len(value) == selected and
            all(type(v) is int and 0 <= v < count for v in value) and
            value == sorted(set(value)))


def _validate_plan(plan, split, norm):
    require(isinstance(plan, dict), "Frozen plan must be a JSON object")
    require(plan.get("schema") == "round3_exploratory_validation_v1", "Unsupported frozen plan schema")
    require(plan.get("status") == "SETTINGS_FROZEN_BEFORE_WAVEFORM_READS", "Plan is not frozen")
    require(plan.get("allowed_real_splits") == ["Train", "Validation"], "Only Train/Validation waveform indexing is permitted")
    require(all(plan.get(k) is False for k in ("new_training", "new_generation", "model_selection")),
            "Plan requests a forbidden operation")
    require(plan.get("threads") == 2, "Expected two analysis threads")
    require(plan.get("window") == {"fs": FS, "samples": NT, "channels": NC}, "Window geometry changed")
    require(plan.get("channels") == norm["channel_order"], "Plan / normalization channel mismatch")
    definitions = json.dumps({k: plan.get(k) for k in ("features", "comparisons")},
                             sort_keys=True, separators=(",", ":"))
    require(hashlib.sha256(definitions.encode("utf-8")).hexdigest() == DEFINITIONS_SHA256,
            "Frozen metric definitions or comparison descriptions changed")
    subjects = sorted({r["subject_id"] for r in split})
    require(len(subjects) == 198 and plan.get("subjects") == subjects, "Frozen cohort mismatch")
    require(sorted(plan.get("processing_order", [])) == subjects, "Invalid frozen processing order")
    bootstrap = plan.get("bootstrap", {})
    require(bootstrap.get("seed") == 2026101703 and bootstrap.get("repetitions") == 5000 and
            bootstrap.get("unit") == "participant" and bootstrap.get("seeds_resampled") is False and
            bootstrap.get("interval") == "percentile95%, unadjusted, conditional on fixed data/model/pool/subset",
            "Frozen bootstrap settings changed")
    inventory = plan.get("real_file_inventory", [])
    require(isinstance(inventory, list) and sorted(r.get("subject_id", "") for r in inventory) == subjects,
            "Frozen real inventory cohort mismatch")
    for record in inventory:
        expected = {r["source_output_sha256"] for r in split if r["subject_id"] == record["subject_id"]}
        require(expected == {record.get("expected_sha256")}, "Frozen inventory / split source hash mismatch")
    selections = plan.get("selections", {})
    require(isinstance(selections, dict) and set(selections) == {s + "/" + t for s in subjects for t in STATES},
            "Frozen subject/state selections mismatch")
    totals = {"Train": 0, "Validation": 0}
    for key, selection in selections.items():
        sid, state = key.split("/")
        counts = {sp: sum(int(r["usable_windows"]) for r in split if r["subject_id"] == sid and
                         r["eye_state"] == state and r["split"] == sp and truth(r["block_eligible"]))
                  for sp in ("Train", "Validation")}
        require(selection.get("counts") == counts and min(counts.values()) > 0, "Frozen stratum counts mismatch")
        n = min(counts.values())
        require(selection.get("matched_n") == n, "Frozen matched count mismatch")
        for source in ("Train", "Validation", "synthetic"):
            require(_indices(selection.get(source + "_indices"), counts["Train" if source == "synthetic" else source], n),
                    "Invalid frozen selected indices")
        require(set(selection.get("synthetic", {})) == set(GENERATORS), "Missing frozen model selections")
        for sp in totals:
            totals[sp] += counts[sp]
    require(totals == {"Train": 47835, "Validation": 9635}, "Cohort window totals changed")
    illustration = plan.get("illustration", {})
    require(illustration.get("seed") == 20261017 and illustration.get("subject_id") in subjects and
            illustration.get("state") in STATES and type(illustration.get("channel_index")) is int and
            0 <= illustration["channel_index"] < NC, "Invalid frozen illustration metadata")
    require(illustration.get("channel") == norm["channel_order"][illustration["channel_index"]], "Illustration channel mismatch")
    choices = illustration.get("window_indices", {})
    require(set(choices) == {"Validation", *GENERATORS}, "Invalid illustration source keys")
    counts = selections[illustration["subject_id"] + "/" + illustration["state"]]["counts"]
    for source, index in choices.items():
        require(type(index) is int and 0 <= index < counts["Validation" if source == "Validation" else "Train"],
                "Illustration index outside source")


def _resolve_plan(plan, split, norm, manifests):
    global _RESOLVED
    _validate_plan(plan, split, norm)
    mapping = {}
    _bind_metadata(plan, mapping)
    _bind_wave_paths(plan, manifests, mapping)
    _RESOLVED = mapping
    return mapping


def _resolved_path(recorded):
    require(recorded in _RESOLVED, "Unresolved frozen input path")
    return _RESOLVED[recorded]


def _input_plan():
    path = _CONFIG["frozen_plan"]
    require(path.suffix.lower() == ".json" and path.is_file(), "Missing frozen plan JSON: " + str(path))
    require(sha(path) == _CONFIG["frozen_plan_sha256"], "Frozen plan SHA-256 mismatch")
    plan = load(path)
    require(isinstance(plan, dict), "Frozen plan must be a JSON object")
    return plan


def _science_imports():
    global np, h5py, scipy, welch, TINY
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    import h5py
    import numpy as np
    import scipy
    from scipy.signal import welch
    TINY = np.finfo(np.float64).tiny
    expected = {"Python": (3, 12, 3), "NumPy": "2.5.2", "SciPy": "1.18.1", "h5py": "3.16.0"}
    actual = {"Python": sys.version_info[:3], "NumPy": np.__version__, "SciPy": scipy.__version__, "h5py": h5py.__version__}
    require(actual == expected, "Scientific execution requires the recorded environment: " + str(expected) + "; found " + str(actual))


def _state_path():
    return OUT / "portable_run.json"


def _prepare_portable(split, norm, manifests):
    require(not OUT.exists(), "Output already exists; choose a fresh external directory")
    if "frozen_plan" in _CONFIG:
        plan = _input_plan()
        _resolve_plan(plan, split, norm, manifests)
        OUT.mkdir(parents=True, exist_ok=False)
        # Copy exact bytes. Do not serialize or replace historical path strings.
        (OUT / "plan.json").write_bytes(_CONFIG["frozen_plan"].read_bytes())
        require(sha(OUT / "plan.json") == _CONFIG["frozen_plan_sha256"], "Frozen plan copy mismatch")
    else:
        _science_imports()
        OUT.mkdir(parents=True, exist_ok=False)
        prepare()
        plan = load(OUT / "plan.json")
        _resolve_plan(plan, split, norm, manifests)
    write(_state_path(), {
        "schema": "portable_validation_run_v1", "state": "PREPARED",
        "created_utc": now(), "configuration_sha256": _CONFIG["configuration_sha256"],
        "portable_script_sha256": sha(Path(__file__)), "original_script_sha256": ORIGINAL_SOURCE_SHA256,
        "plan_sha256": sha(OUT / "plan.json"),
        "path_resolution": [{"frozen_path_label": label, "resolved_path": str(path)}
                            for label, path in sorted(_RESOLVED.items())],
        "waveform_bytes_read": False,
        "cache_policy": "No participant cache reuse; a failed or partial run requires a new output directory.",
    })
    print("PREPARED", OUT, "(no waveform bytes read)", flush=True)


def _prepared_state():
    require(OUT.is_dir() and _state_path().is_file(), "Run --prepare first in a fresh external output directory")
    require({p.name for p in OUT.iterdir()} == {"plan.json", "portable_run.json"},
            "Output contains results, caches, or unexpected files; refusing to overwrite or reuse them")
    state = load(_state_path())
    require(state.get("schema") == "portable_validation_run_v1" and state.get("state") == "PREPARED", "Output is not a fresh prepared run")
    require(state.get("configuration_sha256") == _CONFIG["configuration_sha256"], "Configuration changed after preparation")
    require(state.get("portable_script_sha256") == sha(Path(__file__)), "Portable script changed after preparation")
    require(state.get("original_script_sha256") == ORIGINAL_SOURCE_SHA256, "Original source identity changed")
    require(state.get("plan_sha256") == sha(OUT / "plan.json"), "Prepared plan hash changed")
    if "frozen_plan" in _CONFIG:
        _input_plan()
        require(state["plan_sha256"] == _CONFIG["frozen_plan_sha256"], "Prepared plan differs from supplied original")
    return state


def _run_portable(split, norm, manifests):
    state = _prepared_state()
    plan = load(OUT / "plan.json")
    _resolve_plan(plan, split, norm, manifests)
    _science_imports()
    # Mark before waveform access. Interrupted outputs cannot be resumed or
    # confused with caches from the archived implementation or another config.
    state.update(state="RUNNING", started_utc=now(), waveform_bytes_read=None)
    write(_state_path(), state)
    results = []
    for i, sid in enumerate(plan["processing_order"]):
        result = analyze_subject(sid, plan, split, norm, state["plan_sha256"])
        print(f"PARTICIPANT {i+1}/198 {sid} {result['elapsed_seconds']:.2f}s", flush=True)
        results.append(result)
    aggregate(results, plan)
    state.update(state="COMPLETE", completed_utc=now(), waveform_bytes_read=True,
                 summary_sha256=sha(OUT / "summary.json"), completion_sha256=sha(OUT / "completion.json"))
    write(_state_path(), state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-config", action="store_true", help="Validate metadata, mappings and file sizes only; never open waveform bytes")
    group.add_argument("--prepare", action="store_true", help="Copy an unchanged private frozen plan, or freeze a new metadata-only plan, in a fresh output")
    group.add_argument("--run", action="store_true", help="Explicit full external-data analysis; requires a fresh --prepare output and recorded scientific environment")
    parser.add_argument("--config", required=True, type=Path, help="JSON config; relative paths resolve beside this file")
    args = parser.parse_args(argv)
    try:
        _configure(args.config)
        split, norm, manifests = _metadata_check()
        if args.check_config:
            if OUT.exists():
                _prepared_state()
                _resolve_plan(load(OUT / "plan.json"), split, norm, manifests)
            elif "frozen_plan" in _CONFIG:
                _resolve_plan(_input_plan(), split, norm, manifests)
            print("CONFIG_VALID (metadata only; no waveform bytes read or scientific imports; full execution is not verified)")
        elif args.prepare:
            _prepare_portable(split, norm, manifests)
        else:
            _run_portable(split, norm, manifests)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
