import numpy as np
import pandas as pd
import streamlit as st
from ..charts import fidelity_bars,psd_chart,alpha_chart
from ..formatting import readable_metric
from .common import heading,chart,select,table,message


def interpretation(rows):
    valid=[r for r in rows if r.get("mean") is not None]
    if not valid:
        return "No saved estimates are available for this metric."
    unit=valid[0].get("unit","")
    text="; ".join(f"{r['model']}: {r['mean']:.3g} {unit}" for r in valid)+". "
    direction=valid[0].get("direction")
    if any(s in valid[0]["metric"] for s in ("copy","duplicate")):
        text+="A zero exact-match count does not rule out memorization or establish privacy protection."
    elif len({r['mean'] for r in valid})==1:
        text+="The displayed means are equal; this metric does not distinguish the generators."
    elif direction=="lower":
        text+=f"{min(valid,key=lambda r:r['mean'])['model']} has the smallest displayed deviation on this metric; this is not an overall quality score."
    elif direction=="higher":
        text+=f"{max(valid,key=lambda r:r['mean'])['model']} has the largest displayed value on this metric; interpretation remains metric-specific."
    elif direction=="parity":
        text+="Values near 1 indicate parity with the reference; larger is not automatically better."
    else:
        text+="These values are descriptive and do not define a universal ranking."
    return text


def psd_figure(bundle,state="EC",subject=None):
    p=bundle["psd"]
    source=p["participants"][subject] if subject else p["group"]
    spectra={m:(10*np.log10(np.maximum(np.asarray(v,dtype=float),1e-30))).tolist() for m,v in source[state].items()}
    aggregation=f"Participant {subject} · channel mean" if subject else p.get("aggregation","Group mean")
    return psd_chart(p["frequencies_hz"],spectra,state,"All-channel mean",aggregation)


def spectrum(bundle,prefix="fidelity",presentation=False):
    p=bundle.get("psd",{})
    if not p.get("group"):
        st.info("Numerical group PSD is unavailable in this bundle.")
        return
    c=st.columns(2)
    with c[0]: state=select("Eye state",["EC","EO"],prefix+":state",format_func=lambda s:"Eyes closed (EC)" if s=="EC" else "Eyes open (EO)")
    with c[1]:
        subjects=sorted(p.get("participants",{}))
        choice=select("Aggregation",["Group"]+subjects,prefix+":aggregation",help="Participant curves use the saved participant arrays. Channel-averaged curves do not support a channel selector.")
    fig=psd_figure(bundle,state,None if choice=="Group" else choice)
    chart(fig,"Power is averaged in linear units before conversion to dB. Curves describe Real Train and frozen synthetic pools; the shaded region is alpha (8–13 Hz).",prefix+"_psd")
    a=bundle.get("alpha_effect",{})
    if a.get("group"):
        values=a["group"] if choice=="Group" else a.get("participants",{}).get(choice)
        if values:
            chart(alpha_chart(a["channels"],values,"Group mean across participants" if choice=="Group" else f"Participant {choice}"),"Positive EC-minus-EO values indicate greater alpha power with eyes closed. The channel profile is a state effect, not a reconstruction score.",prefix+"_alpha")


def render(bundle):
    heading("SIGNAL FIDELITY","Similarity has more than one dimension","Descriptive comparisons against Real Train, preserving each metric's units and interpretation.")
    tabs=st.tabs(["Spectrum & alpha","Metric families","Participant diagnostics","Matched-count Real–Real calibration"])
    with tabs[0]: spectrum(bundle)
    with tabs[1]:
        rows=bundle.get("fidelity_metrics",[])
        groups=[g for g in ["Spectrum and Alpha","Time and Complexity","Spatial Statistics","Distribution and Diversity"] if any(r["group"]==g for r in rows)]
        if not groups:
            st.info("The prepared bundle contains no group fidelity metrics.")
        else:
            group=select("Metric family",groups,"fidelity:group")
            available=[r for r in rows if r["group"]==group]
            metrics=list(dict.fromkeys(r["metric"] for r in available))
            metric=select("Metric",metrics,"fidelity:metric",index=metrics.index("log_psd_mae_db") if "log_psd_mae_db" in metrics else 0,format_func=readable_metric)
            chosen=[r for r in available if r["metric"]==metric]
            chart(fidelity_bars(chosen),interpretation(chosen),"fidelity_metric",chosen)
            st.caption("Intervals are the saved participant-summary intervals where present. No composite score is formed from unlike quantities.")
            with st.expander("All saved metrics in this family"):
                table(available,"fidelity_family")
    with tabs[2]:
        records=bundle.get("fidelity_subjects",[])
        if not records:
            st.info("Participant-level fidelity diagnostics are unavailable.")
        else:
            cols=st.columns(2)
            with cols[0]: subject=select("Participant",sorted({r["subject_id"] for r in records}),"fidelity:subject")
            with cols[1]: state=select("Participant eye state",["EC","EO"],"fidelity:subject_state")
            rows=[]
            for r in records:
                if r["subject_id"]==subject and r["state"]==state:
                    for metric,value in r["metrics"].items():
                        rows.append({"Participant":subject,"Eye state":state,"Generator":r["model"],"Metric":readable_metric(metric),"Value":value})
            table(rows,"participant_fidelity","Saved participant × state metrics; these are individual diagnostics, separate from the participant-aggregated summaries.")
    with tabs[3]:
        message("Real–Real calibration compares disjoint Train blocks at matched counts. Its reference and sample sizes differ from the full-pool comparison above.")
        rows=bundle.get("calibration",[])
        if rows:
            metric=select("Calibration metric",list(dict.fromkeys(r["metric"] for r in rows)),"calibration:metric",format_func=readable_metric)
            chosen=[r for r in rows if r["metric"]==metric]
            fig=fidelity_bars(chosen)
            fig.update_layout(title_text=readable_metric(metric)+"<br><sup>Matched-count calibration · disjoint Real Train blocks</sup>")
            chart(fig,"Real_B is the saved Real–Real reference. Compare only within this matched-count calibration; it is not independent Final-Test validation.","calibration_metric",chosen)
        else:
            st.info("Matched-count calibration estimates are unavailable in this bundle.")
