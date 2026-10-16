import json
from pathlib import Path
import numpy as np
import streamlit as st
from ..config import BUNDLE_ROOT
from ..charts import waveform_chart,psd_chart
from .common import heading,chart,select


@st.cache_resource(show_spinner=False)
def load_previews(signature):
    meta=json.loads((BUNDLE_ROOT/"previews.json").read_text(encoding="utf-8"))
    with np.load(BUNDLE_ROOT/"previews.npz",allow_pickle=False) as source:
        arrays={k:source[k] for k in source.files}
    return meta,arrays


def previews():
    meta_path=BUNDLE_ROOT/"previews.json"
    data_path=BUNDLE_ROOT/"previews.npz"
    if not meta_path.is_file() or not data_path.is_file():
        return None,None
    return load_previews((meta_path.stat().st_mtime_ns,data_path.stat().st_mtime_ns))


def preview_figure(meta,arrays,idx=0,channel="O1",stacked=False):
    waves={m:arrays["waveforms_uv"][idx,i] for i,m in enumerate(meta["sources"])}
    return waveform_chart(arrays["time_s"],waves,meta["channels"],channel,"µV",stacked)


def render(bundle):
    heading("SIGNAL EXPLORER","Look at the signal, without generating new data","Bounded, precomputed Real Train and synthetic examples selected independently of fidelity outcomes.")
    meta,arrays=previews()
    if meta is None or not meta.get("available",True):
        st.info("Waveform previews are not included in this bundle. Run the documented Train-only preview preparation when the verified source recordings are available. Other numerical views remain available.")
        return
    records=meta["records"]
    cols=st.columns(4)
    with cols[0]: participant=select("Participant",sorted({r["participant_id"] for r in records}),"signal:participant")
    with cols[1]: state=select("Eye state",sorted({r["state"] for r in records if r["participant_id"]==participant}),"signal:state",format_func=lambda s:"Eyes closed (EC)" if s=="EC" else "Eyes open (EO)")
    with cols[2]: channel=select("EEG channel",meta["channels"],"signal:channel",index=meta["channels"].index("O1") if "O1" in meta["channels"] else 0)
    with cols[3]: slot=select("Precomputed sample slot",sorted({r["slot"] for r in records if r["participant_id"]==participant and r["state"]==state}),"signal:slot",help="The same slot number labels separate precomputed windows. It does not create a pointwise Real–synthetic pair.")
    idx=next(i for i,r in enumerate(records) if r["participant_id"]==participant and r["state"]==state and r["slot"]==slot)
    stacked=st.checkbox("Stack all channels",key="signal:stacked")
    chart(preview_figure(meta,arrays,idx,channel,stacked),f"{participant} · {state} · slot {slot}. B0/B1/B2 each show a precomputed synthetic sample, not a pointwise reconstruction of the Real Train window. All panels share time and amplitude axes.","waveforms")
    st.caption("Zoom using the plot toolbar or drag a region. Hover reports the original amplitude in µV, including in the vertically offset stacked view.")
    if st.checkbox("Show selected-window PSD preview",key="signal:psd"):
        ci=meta["channels"].index(channel)
        power={m:(10*np.log10(np.maximum(arrays["psd_uv2_hz"][idx,i,ci],1e-30))).tolist() for i,m in enumerate(meta["sources"])}
        fig=psd_chart(arrays["frequency_hz"],power,state,channel,"Selected-window preview · one precomputed window per source")
        chart(fig,"Selected-window Welch PSD (Hann 256 samples, overlap 128), derived during preview preparation. This is separate from the official group Fidelity calculation and cannot establish group-level performance.","preview_psd")
    with st.expander("Preview selection and units"):
        st.write(f"{len(meta.get('participants',[])) or len({r['participant_id'] for r in records})} participants were selected evenly from the sorted eligible participant list, with up to three windows per eye state. Selection does not use fidelity scores or classification outcomes.")
        st.write("Real examples are verified Train-only slices. Stored standardized synthetic samples are converted once using the locked Train normalization metadata. The app reads only this compact preview bundle, never raw recordings or model weights.")
