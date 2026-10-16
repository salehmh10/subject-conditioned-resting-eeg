import streamlit as st
from ..charts import workflow_chart, split_chart
from ..config import APP_ROOT
from ..formatting import percent
from .common import heading, message, chart, headline, eegnet_family


def render(bundle):
    m=bundle.get("metadata",{})
    heading("RESEARCH AT A GLANCE","How close is synthetic EEG — and how useful?","A completed study of signal fidelity and eye-state classification, presented from saved research evidence.")
    cols=st.columns(4)
    values=[("Participants",m.get("n_subjects"),"Same participants across splits"),("EEG channels",len(m.get("channels",[])),f"{m.get('fs_hz','—')} Hz sampling"),("Train windows",f"{m.get('n_train',0):,}","Real Train fidelity reference"),("Final-Test windows",f"{m.get('n_test',0):,}","Previously held-out blocks")]
    for col,(label,value,detail) in zip(cols,values):
        with col:
            st.metric(label,value)
            st.caption(detail)
    message(headline(bundle)+" This comparison does not establish a universal classifier benefit or conclusive superiority over B1.")
    chart(workflow_chart(),"The two evaluation branches remain separate: a closer signal-level match does not, by itself, establish a classification benefit.","workflow")
    st.subheader("Three generators, one shared backbone")
    cols=st.columns(3)
    for col,title,body,foot in zip(cols,["B0 · Eye state","B1 · Participant + state","B2 · Continued training"],
            ["Conditions on eyes open or eyes closed. Attached participant labels are evaluation strata, not model inputs.","Adds a participant embedding to the temporal 1D U-Net's state conditioning.","Uses the same participant-and-state-conditioned architecture as B1, with continued training and a selected checkpoint."],
            ["State-only reference","Participant-aware generator","Same architecture as B1"]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(body)
                st.caption(foot)
    st.caption("All use the temporal 1D U-Net with 19 × 512 input/output, a diffusion-time embedding, and final v-prediction. Unequal training histories prevent a pure causal attribution to participant conditioning.")
    with st.expander("Shared temporal architecture",expanded=False):
        st.image(str(APP_ROOT/"assets"/"architecture.svg"),width="stretch")
        st.caption("Verified model structure. B1 adds a participant branch; B2 preserves that architecture. The diagram does not represent live training or inference.")
    with st.expander("Generator versus training seed"):
        st.write("B0, B1 and B2 identify generative models. The three EEGNet training seeds identify independent classifier fits that reuse each frozen synthetic pool. They are not three generators or three regenerated EEG pools.")
    with st.expander("Study data and evaluation target"):
        chart(split_chart(m),"Complete blocks are separated across splits; this evaluates new blocks of known participants, not unseen participants or new sessions.","overview_splits")
        st.write(f"{m.get('dataset','MPI-LEMON')} resting-state EEG, eyes open (EO) and eyes closed (EC). Four-second windows contain 512 samples; the Train, Validation and Final-Test sets share participant identities.")
