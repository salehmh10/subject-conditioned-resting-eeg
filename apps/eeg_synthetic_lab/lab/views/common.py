"""Shared view components and persistent, view-scoped controls."""
import html
import json
from pathlib import Path
import pandas as pd
import plotly.io as pio
import streamlit as st

from ..config import BUNDLE_ROOT, PLOT_CONFIG


@st.cache_resource(show_spinner=False)
def load_bundle(path, signature):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_bundle():
    path = BUNDLE_ROOT / "bundle.json"
    if not path.is_file():
        st.error("Presentation data is not prepared yet. In the application folder, run .\\prepare_data.ps1, then refresh this page.")
        st.stop()
    return load_bundle(str(path), path.stat().st_mtime_ns)


def select(label, options, key, index=0, format_func=str, help=None):
    """Preserve semantic selections even when Streamlit removes hidden widgets."""
    options = list(options)
    saved = st.session_state.get("pref:"+key)
    if saved not in options:
        saved = options[min(index, len(options)-1)]
    widget_key="control:"+key
    if widget_key not in st.session_state or st.session_state[widget_key] not in options:
        st.session_state[widget_key] = saved
    def remember():
        st.session_state["pref:"+key] = st.session_state[widget_key]
    return st.selectbox(label, options, key=widget_key, format_func=format_func, on_change=remember, help=help)


def heading(kicker, title, intro):
    st.markdown(f'<div class="kicker">{html.escape(kicker)}</div><h1>{html.escape(title)}</h1><p class="page-intro">{html.escape(intro)}</p>', unsafe_allow_html=True)


def message(text):
    st.markdown(f'<div class="finding"><span>READING THE EVIDENCE</span>{html.escape(text)}</div>',unsafe_allow_html=True)


def frame(rows):
    return pd.DataFrame([{k:v for k,v in r.items() if k != "provenance" and not isinstance(v,(list,dict))} for r in rows])


@st.cache_data(show_spinner=False,max_entries=32)
def chart_html(json_figure):
    fig=pio.from_json(json_figure)
    return pio.to_html(fig,include_plotlyjs=True,full_html=True,config=PLOT_CONFIG)


def chart(fig, interpretation, key, table=None):
    st.plotly_chart(fig,use_container_width=True,config=PLOT_CONFIG,key="plot:"+key)
    if interpretation:
        st.caption(interpretation)
    with st.expander("Download this chart" if table is None else "Download this chart and displayed data"):
        if table is not None:
            df=table if isinstance(table,pd.DataFrame) else frame(table)
            st.download_button("Download CSV",df.to_csv(index=False).encode("utf-8"),file_name=f"eeg_{key}.csv",mime="text/csv",key="csv:"+key)
        if st.button("Prepare interactive HTML",key="prepare:"+key):
            st.session_state["ready:"+key]=True
        if st.session_state.get("ready:"+key):
            st.download_button("Download interactive HTML",chart_html(fig.to_json()),file_name=f"eeg_{key}.html",mime="text/html",key="html:"+key)
        st.caption("HTML embeds Plotly and the displayed data for offline use.")


def table(rows, key, caption=None):
    df=rows if isinstance(rows,pd.DataFrame) else frame(rows)
    if caption:
        st.caption(caption)
    st.dataframe(df,width="stretch",hide_index=True)
    st.download_button("Download table CSV",df.to_csv(index=False).encode("utf-8"),file_name=f"eeg_{key}.csv",mime="text/csv",key="table:"+key)


def capability(bundle,key,label):
    value=bundle.get("availability",{}).get(key)
    available=value if isinstance(value,bool) else value.get("available",False) if isinstance(value,dict) else False
    if not available:
        reason=value.get("reason", "This data is not present in the prepared bundle.") if isinstance(value,dict) else "This data is not present in the prepared bundle."
        st.info(f"{label} unavailable. {reason}")
    return available


def eegnet_family(bundle):
    return next((r["family"] for r in bundle.get("utility_summary",[]) if "EEGNET" in r["family"].upper()),None)


def headline(bundle):
    family=eegnet_family(bundle)
    rows={r["condition"]:r for r in bundle.get("utility_summary",[]) if r["family"]==family and r["split"]=="final_test"}
    b2,real=rows.get("REAL_PLUS_B2_1X"),rows.get("REAL_ONLY")
    if b2 and real:
        delta=100*(b2["mean"]-real["mean"])
        return f"B2 augmentation changes EEGNet Final-Test subject-macro balanced accuracy by {delta:+.3f} percentage points versus Real only."
    return "Fidelity and classification utility address complementary parts of the research question."
