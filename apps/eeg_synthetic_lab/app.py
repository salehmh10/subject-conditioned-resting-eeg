"""Local entry point. Reads only the prepared presentation bundle and app assets."""
from pathlib import Path
import sys
import streamlit as st

APP_DIRECTORY=Path(__file__).resolve().parent
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0,str(APP_DIRECTORY))

from lab.config import APP_ROOT,PAGES
from lab.views import VIEWS,presentation
from lab.views.common import read_bundle


st.set_page_config(page_title="EEG Synthetic Lab",page_icon="◉",layout="wide",initial_sidebar_state="expanded",menu_items={"Get Help":None,"Report a bug":None,"About":"EEG Synthetic Lab · Local presentation of completed research evidence."})
st.markdown("<style>"+(APP_ROOT/"assets"/"style.css").read_text(encoding="utf-8")+"</style>",unsafe_allow_html=True)


def reset_view():
    for key in list(st.session_state):
        if key not in ("navigation","mode"):
            del st.session_state[key]


bundle=read_bundle()
with st.sidebar:
    st.markdown('<div class="brand">EEG Synthetic Lab</div><div class="brand-sub">Resting-state EEG<br>Signal Fidelity × Classification Utility</div>',unsafe_allow_html=True)
    mode=st.radio("Mode",["Explore","Guided Presentation"],key="mode")
    if mode=="Explore":
        page=st.radio("Explore the evidence",PAGES,key="navigation")
    else:
        st.caption("Ten focused steps from the research question to a bounded conclusion.")
    st.divider()
    st.button("Reset view",on_click=reset_view,width="stretch")
    st.caption("Prepared evidence · local runtime\n\nNo training or live generation")
    if bundle.get("metadata",{}).get("shared_export"):
        st.caption("Aggregate edition · participant diagnostics and waveform previews omitted")
    with st.expander("About this presentation"):
        st.write("The app presents saved Fidelity and final utility evidence. Validation is used for model selection; Final Test uses held-out blocks of known participants.")
        st.caption("Interactive chart downloads embed their data and Plotly for offline viewing.")

if mode=="Guided Presentation":
    presentation.render(bundle)
else:
    VIEWS[page](bundle)
