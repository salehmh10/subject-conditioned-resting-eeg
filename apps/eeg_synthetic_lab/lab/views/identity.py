import streamlit as st
from ..charts import fingerprint_chart,fidelity_bars
from ..formatting import readable_metric
from .common import heading,chart,select,table,message
from .fidelity import interpretation


def render(bundle):
    heading("PARTICIPANT IDENTITY & DIVERSITY","What remains distinctive about a participant?","Fingerprint retrieval uses 76 log-bandpower features; diversity and copy checks provide separate diagnostics.")
    metric=select("Retrieval measure",["top1","top5"],"identity:metric",format_func=lambda x:"Top-1 identification" if x=="top1" else "Top-5 identification")
    rows=bundle.get("fingerprint",[])
    if rows:
        chosen=[r for r in rows if r["model"]=="B2"]
        desc="; ".join(f"{r['state']} {100*r[metric]:.2f}%" for r in chosen)
        chart(fingerprint_chart(rows,metric),f"B2 {metric.replace('top','Top-')} retrieval: {desc}. These are in-sample comparisons to known Real Train participant profiles; high retrieval is not a privacy guarantee.","fingerprint",rows)
    else:
        st.info("Fingerprint summaries are unavailable in the prepared bundle.")
    message("B0 does not receive a participant identity. Its nominal participant groups are evaluation strata, so low individual-reference retrieval does not by itself invalidate a population-level state-only generator.")
    tabs=st.tabs(["Participant diagnostics","Diversity & copy checks"])
    with tabs[0]:
        data=bundle.get("fingerprint_subjects",[])
        if data:
            cols=st.columns(2)
            with cols[0]: subject=select("Participant",sorted({r["subject_id"] for r in data}),"identity:subject")
            with cols[1]: state=select("Eye state",["EC","EO"],"identity:state")
            table([r for r in data if r["subject_id"]==subject and r["state"]==state],"fingerprint_subject","Saved rank, own-participant correlation, retrieval margin and feature error. Ranks refer to the source candidate set.")
        else:
            st.info("Individual retrieval ranks are not included in this bundle edition.")
    with tabs[1]:
        data=[r for r in bundle.get("fidelity_metrics",[]) if any(s in r["metric"].lower() for s in ("diversity","copy","duplicate","pairwise","nearest","coverage","memor"))]
        if data:
            metrics=list(dict.fromkeys(r["metric"] for r in data))
            metric_id=select("Diagnostic",metrics,"identity:diagnostic",format_func=readable_metric)
            selected=[r for r in data if r["metric"]==metric_id]
            chart(fidelity_bars(selected),interpretation(selected),"diversity",selected)
        else:
            st.info("No separately named diversity/copy metrics are present in this bundle.")
        st.caption("A diversity ratio near one indicates parity, not maximization. No exact-copy detections do not establish the absence of memorization or prove privacy protection.")
