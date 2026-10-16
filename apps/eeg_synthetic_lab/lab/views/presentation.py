import plotly.graph_objects as go
import streamlit as st
from ..config import STEPS,COLORS,APP_ROOT
from ..charts import workflow_chart,split_chart,connectivity_grid,fingerprint_chart,utility_comparison,contrast_forest
from ..charts.builders import theme
from .common import heading,chart,headline,eegnet_family
from .fidelity import psd_figure
from .signal import previews,preview_figure


MESSAGES=[
    "Two questions guide the study: do the signals resemble real EEG, and do they support useful eye-state classification?",
    "The evaluation holds out complete blocks from known participants; it does not test new people or new sessions.",
    "B0 conditions on eye state. B1 and B2 also condition on participant identity, with a shared architecture and different training histories.",
    "These separate, precomputed windows illustrate signal appearance; selected examples are not evidence of group performance.",
    "Spectrum and alpha describe a specific aspect of Fidelity relative to Real Train, without defining an overall quality winner.",
    "Shared-scale sensor-space connectivity maps expose both agreement and residual differences across generators.",
    "Participant-aware generators can retain a known participant's spectral fingerprint, which is distinct from a privacy guarantee.",
    "The complete condition comparison retains synthetic-only training, both augmentation doses, and the Real Duplicate control.",
    "The EEGNet augmentation benefit and the comparison with B1 answer different questions; their saved intervals need separate interpretation.",
    "The study supports a bounded conclusion: useful synthetic augmentation in a specific comparison, with participant, model-selection and privacy limits.",
]

NOTES=[
    "Separate descriptive generative characterization from classifier utility. A good score in one branch does not logically guarantee success in the other.",
    "Train, Validation and Final Test contain different blocks from the same 198 participants. Four-second windows overlap only within their original blocks; splits are block-based.",
    "B2 is continued training of the participant-conditioned family. It is not a new architecture or a proven spectral-loss enhancement. B0 participant IDs label evaluation strata only.",
    "The preview exporter selected participants evenly from the sorted eligible list and fixed sample slots, independently of outcomes. Each synthetic sample is not a pointwise reconstruction of the shown Real Train window.",
    "The group curve averages power in linear units before conversion to dB. Alpha effects compare EC with EO; channel-specific effects are available in Explore mode.",
    "dwPLI² is debiased sensor-space phase-lag connectivity, not anatomical or causal connectivity. Negative estimates and missing edges remain in the saved numerical data. Group averaging may hide participant variation.",
    "Profiles use 76 log-bandpower features over 19 channels and four bands. The random-ranking reference is a simple uniform reference, not a fitted statistical null. B0 does not receive identity as input.",
    "EEGNet points summarize three training seeds with population SD. Classical families each have one fit per condition. TSTR still uses real Validation for EEGNet checkpoint selection. Equal epoch ceilings do not equalize updates.",
    "Saved bootstrap intervals are paired over participants, with EEGNet seeds averaged within participant. They are descriptive and not adjusted for multiple comparisons. An interval crossing zero does not establish superiority.",
    "The reference for Fidelity is Real Train. Unequal training histories limit causal conclusions. The final result does not establish unseen-participant generalization, universal classifier benefit, no memorization or privacy protection.",
]


def conclusion_chart():
    fig=go.Figure()
    rows=[("What is measured","Signal fidelity against Real Train<br>Eye-state classification on held-out blocks",COLORS["Real"]),
          ("What is supported","A saved EEGNet augmentation comparison<br>Metric-specific signal and identity findings",COLORS["B1"]),
          ("What remains unestablished","New-participant / new-session generalization<br>Universal utility · causal conditioning benefit · privacy",COLORS["B2"])]
    for i,(title,body,color) in enumerate(rows):
        y=.83-i*.3
        fig.add_shape(type="rect",x0=.02,x1=.98,y0=y-.11,y1=y+.11,fillcolor="#F4F8FA",line=dict(color="#D6E3EA",width=1))
        fig.add_annotation(x=.045,y=y+.055,text=f"<b>{title}</b>",showarrow=False,xanchor="left",font=dict(size=20,color=color))
        fig.add_annotation(x=.045,y=y-.037,text=body,showarrow=False,xanchor="left",align="left",font=dict(size=16,color="#233C58"))
    theme(fig,"A bounded scientific conclusion","Each claim stays within the evaluation that produced it",490)
    fig.update_xaxes(visible=False,range=[0,1])
    fig.update_yaxes(visible=False,range=[0,1])
    return fig


def step_figure(bundle,index):
    """Export-friendly representative aggregate figures; preview handled separately."""
    if index in (0,2): return workflow_chart()
    if index==1: return split_chart(bundle.get("metadata",{}))
    if index==4 and bundle.get("psd",{}).get("group"): return psd_figure(bundle,"EC")
    if index==5 and bundle.get("connectivity",{}).get("group"):
        c=bundle["connectivity"]
        return connectivity_grid(c["group"]["EC"]["alpha"],c["channels"],"EC","alpha")
    if index==6: return fingerprint_chart(bundle.get("fingerprint",[]))
    if index==7: return utility_comparison(bundle.get("utility_summary",[]),bundle.get("utility_runs",[]))
    if index==8:
        family=eegnet_family(bundle)
        selected=[r for r in bundle.get("contrasts",[]) if r["family"]==family and r["split"]=="final_test" and {r["left"],r["right"]} in ({"REAL_PLUS_B2_1X","REAL_ONLY"},{"REAL_PLUS_B2_1X","REAL_PLUS_B1_1X"})]
        return contrast_forest(selected,family) if selected else conclusion_chart()
    return conclusion_chart()


def render(bundle):
    if "presentation_step" not in st.session_state: st.session_state.presentation_step=0
    index=st.session_state.presentation_step
    cols=st.columns([1,5,1])
    with cols[0]:
        if st.button("← Previous",disabled=index==0,width="stretch",key="presentation_previous"):
            st.session_state.presentation_step-=1; st.rerun()
    with cols[1]: st.markdown(f'<div class="step-counter">{index+1:02d} / {len(STEPS):02d} &nbsp; {STEPS[index]}</div>',unsafe_allow_html=True)
    with cols[2]:
        if st.button("Next →",disabled=index==len(STEPS)-1,width="stretch",type="primary",key="presentation_next"):
            st.session_state.presentation_step+=1; st.rerun()
    st.progress((index+1)/len(STEPS))
    heading("GUIDED PRESENTATION",STEPS[index],MESSAGES[index])
    if index==2:
        st.image(str(APP_ROOT/"assets"/"architecture.svg"),width="stretch")
        st.caption("Verified shared temporal denoiser · 19 × 512 input/output · participant embedding only in B1/B2 · final v-prediction for all generators.")
    elif index==3:
        meta,arrays=previews()
        if meta is not None:
            fig=preview_figure(meta,arrays,0,"O1")
            chart(fig,"Deterministically selected research example · Real Train versus precomputed synthetic samples · separate windows, not aligned reconstructions.","presentation_waveform")
        else:
            st.info("Waveform examples are omitted in this bundle edition. The numerical aggregate evidence remains available in the following steps.")
            chart(workflow_chart(),"The evaluation branches use saved evidence; no new signal is generated during the presentation.","presentation_waveform_scope")
    else:
        caption=headline(bundle) if index in (7,8) else NOTES[index].split(". ")[0]+"."
        chart(step_figure(bundle,index),caption,f"presentation_{index}")
    with st.expander("Speaker notes"):
        st.write(NOTES[index])
    st.caption("Presentation preset · Explore mode provides metric, participant, condition and seed controls.")
