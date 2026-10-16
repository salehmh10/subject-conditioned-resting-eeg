import numpy as np
import pandas as pd
import streamlit as st
from ..charts import connectivity_grid
from .common import heading,chart,select,table


def render_content(bundle,prefix="connectivity",presentation=False):
    c=bundle.get("connectivity",{})
    if not c.get("group"):
        st.info("Numerical connectivity matrices are unavailable. Participant matrices cannot be inferred from saved group images.")
        return
    cols=st.columns(3)
    with cols[0]: state=select("Eye state",["EC","EO"],prefix+":state",format_func=lambda s:"Eyes closed (EC)" if s=="EC" else "Eyes open (EO)")
    with cols[1]: band=select("Frequency band",c.get("bands",["theta","alpha","beta"]),prefix+":band",index=1,format_func=str.capitalize)
    with cols[2]: subject=select("Aggregation",["Group"]+sorted(c.get("participants",{})),prefix+":participant")
    data=c["group"] if subject=="Group" else c["participants"][subject]
    matrices=data[state][band]
    cross_state=st.checkbox("Use a common color range across EO and EC",key=prefix+":common_range",help="Otherwise the range is shared across Real/B0/B1/B2 within this state and band; changing state can change the range.")
    limits=None
    if cross_state:
        values=[]
        for s in ("EO","EC"):
            for mat in data[s][band].values():
                a=np.asarray(mat,dtype=float); values.extend(a[np.isfinite(a)].tolist())
        if values:
            limits=(min(values),max(values))
    aggregation="Equal-participant group mean" if subject=="Group" else f"Participant {subject}"
    fig=connectivity_grid(matrices,c["channels"],state,band,limits=limits,aggregation=aggregation)
    finite={m:np.asarray(mat,dtype=float) for m,mat in matrices.items()}
    means={m:float(np.nanmean(a[np.triu_indices(len(a),1)])) for m,a in finite.items()}
    interpretation="Mean upper-triangle strength: "+"; ".join(f"{m} {v:.4f}" for m,v in means.items())+". Edge strength and the spatial edge pattern are different summaries."
    chart(fig,interpretation,prefix+"_matrices")
    st.caption("dwPLI² is sensor-space phase-lag connectivity. Negative debiased estimates and undefined entries are preserved; blank diagonal cells are excluded. This is not anatomical or causal connectivity.")
    if not presentation:
        chart(connectivity_grid(matrices,c["channels"],state,band,difference=True,aggregation=aggregation),"Signed differences show where each synthetic edge estimate is larger or smaller than Real Train. Zero is centered on a symmetric scale.",prefix+"_differences")
        records=[r for r in bundle.get("connectivity_metrics",[]) if r["state"]==state and r["band"]==band and (subject=="Group" or r["subject_id"]==subject)]
        if records:
            fields={"upper_r":"Edge-pattern correlation","upper_mae":"Edge MAE","relative_frobenius":"Relative Frobenius error","node_mean_mae":"Node-mean MAE","mean_connectivity_real":"Real mean strength","mean_connectivity_synth":"Synthetic mean strength","valid_edges":"Valid edges"}
            df=pd.DataFrame(records)
            if subject=="Group":
                cols=[k for k in fields if k in df]
                df=df.groupby("model",sort=False)[cols].mean().reset_index()
            else:
                df=df[[k for k in ["model",*fields] if k in df]]
            df=df.rename(columns={"model":"Generator",**fields})
            table(df,prefix+"_metrics","Separate diagnostics, not interchangeable scores. Group rows are equal-participant means of saved participant diagnostics; missing values are excluded per metric.")
            st.caption("Correlation measures edge-pattern agreement; MAE measures absolute edge error; Frobenius error scales the matrix error by reference magnitude; node-mean error summarizes channel-level mean connectivity.")


def render(bundle):
    heading("SENSOR-SPACE CONNECTIVITY","Do the relationships between channels survive?","Compare saved Real/B0/B1/B2 dwPLI² matrices with shared scales and explicit signed differences.")
    render_content(bundle)
