import streamlit as st
from ..charts import learning_chart
from ..config import CONDITIONS
from ..formatting import condition_label
from .common import heading,chart,select,table


METHODS = [
    ("Data and preprocessing", "MPI-LEMON resting-state EEG uses 19 channels at 128 Hz. Four-second windows contain 512 samples, with a two-second stride. The pipeline applies 1–40 Hz filtering, resampling, Train-derived channel/QC decisions and global per-channel Train normalization. Complete blocks separate Train, Validation and Final Test."),
    ("Shared generator architecture", "The temporal 1D U-Net has widths 64/128/256/256, two residual blocks per level, and attention at the deeper levels and bottleneck. Final models use v-prediction and ancestral DDPM sampling. B0 conditions on eye state; B1/B2 also use a participant embedding. B2 retains the B1 architecture and has no selected spectral-loss improvement."),
    ("Signal Fidelity", "The reference is Real Train, so Fidelity is descriptive characterization. Group PSD averages linear power before dB conversion. Matched-count Real–Real calibration uses disjoint Train blocks and must remain separate from full-pool comparisons."),
    ("Classifiers", "LogBandpower + Logistic Regression uses 76 log-bandpower features. OAS/Riemannian + Linear SVM uses covariance matrices and a training-fitted tangent-space reference. EEGNetv4 is a compact temporal/spatial convolutional classifier. There is one fit per classical condition and three EEGNet training seeds: 55 fits across 11 conditions."),
    ("EEGNet checkpoint selection", "The maximum is 120 actual epochs, minimum stopping epoch 8 and patience 6. Highest real Validation subject-macro balanced accuracy selects the checkpoint; ties favor lower Validation cross-entropy, then the earlier checkpoint. The selected epoch may be earlier than epoch 8. TSTR uses synthetic training data but still uses real Validation for checkpoint selection."),
    ("Uncertainty and comparisons", "Subject-macro balanced accuracy equally averages participants after averaging EO and EC recall within each participant. EEGNet summaries use the mean and population SD (ddof=0) over three classifier seeds. Saved paired subject-bootstrap intervals quantify different uncertainty from seed SD, and are not multiplicity-adjusted."),
]

LIMITATIONS = [
    "Different blocks from the same participants do not demonstrate unseen-participant or new-session generalization.",
    "Unequal generator training histories prevent a pure causal claim about participant conditioning. Development-based generator selection also limits independent confirmation.",
    "Better individual-reference Fidelity does not establish universal classifier utility. B2 augmentation improves the reported EEGNet comparison; superiority over B1 is not conclusive.",
    "Equal epoch ceilings do not equalize optimizer-update budgets. The Real Duplicate condition remains an important control.",
    "Fingerprint retrieval uses 76 log-bandpower features, not alpha alone or a proven connectivity biometric. In-sample identification and no exact-copy detections do not establish privacy protection.",
    "dwPLI² reflects sensor-space phase-lag connectivity. Negative debiased estimates and undefined values are meaningful and are retained; this estimator is not squared again.",
]

GLOSSARY = [
    {"Metric":"Log-PSD MAE","Meaning":"Absolute difference between log-power spectra","Interpretation":"Lower is closer; dB"},
    {"Metric":"Alpha EC minus EO","Meaning":"Eyes-closed minus eyes-open alpha bandpower","Interpretation":"Signed effect; dB"},
    {"Metric":"SD-MD","Meaning":"Mean absolute difference in channel SD, pooled over windows and samples","Interpretation":"Lower is closer; µV"},
    {"Metric":"Sliced Wasserstein distance","Meaning":"Distribution distance from projected features","Interpretation":"Lower is closer; source-defined normalization"},
    {"Metric":"Hjorth descriptors","Meaning":"Activity, mobility and complexity of time-domain signals","Interpretation":"Distinct units; examine each descriptor's discrepancy"},
    {"Metric":"Permutation entropy / ACF","Meaning":"Ordinal signal complexity / lagged temporal dependence","Interpretation":"Complementary time-structure diagnostics"},
    {"Metric":"dwPLI²","Meaning":"Debiased squared weighted phase-lag index","Interpretation":"Dimensionless; negative estimates can occur"},
    {"Metric":"Fingerprint Top-1 / Top-5","Meaning":"Known participant is first / among first five retrieved profiles","Interpretation":"Candidate count matters; not a privacy guarantee"},
    {"Metric":"Subject-macro balanced accuracy","Meaning":"Mean participant-wise average of EO and EC recall","Interpretation":"Higher is better; %"},
    {"Metric":"Percentage-point difference","Meaning":"100 × (left BA fraction − right BA fraction)","Interpretation":"A difference of 0.007 is 0.7 pp"},
]


def render(bundle):
    heading("LEARNING BEHAVIOR & METHODS","See how checkpoints were selected","Saved classifier histories end at each seed's actual stopping epoch. Final Test plays no role in these curves.")
    tabs=st.tabs(["Learning behavior","Methods","Metric glossary","Limitations & references"])
    with tabs[0]:
        histories=bundle.get("histories",[])
        if histories:
            conditions=[c for c in CONDITIONS if any(h["condition"]==c for h in histories)]
            cols=st.columns(2)
            with cols[0]: condition=select("Training condition",conditions,"learning:condition",format_func=condition_label)
            seeds=sorted({h["seed"] for h in histories if h["condition"]==condition})
            with cols[1]: seed=select("Training seed",["All three seeds"]+seeds,"learning:seed")
            selected=[h for h in histories if h["condition"]==condition and (seed=="All three seeds" or h["seed"]==seed)]
            chart(learning_chart(histories,condition,None if seed=="All three seeds" else [seed]),"The star marks the Validation-selected checkpoint and the square the actual stopping epoch. Curves are never extended to the 120-epoch ceiling after stopping.","learning")
            table([{"Condition":condition_label(h["condition"]),"Training seed":h["seed"],"Selected epoch":h["best_epoch"],"Stopping epoch":h["stopped_epoch"],"Saved epoch rows":len(h["epochs"])} for h in selected],"learning_stops")
            st.caption("Minimum epoch 8 constrains stopping, not the earliest selectable checkpoint. The specified tie-breaker is lower real Validation cross-entropy.")
        else:
            st.info("Numerical training histories are unavailable in this bundle. Best-epoch summaries cannot reconstruct a learning curve.")
    with tabs[1]:
        for title,body in METHODS:
            st.markdown(f"#### {title}")
            st.write(body)
        with st.expander("Prepared bundle availability and source provenance"):
            st.json(bundle.get("availability",{}),expanded=False)
            sources=bundle.get("provenance",{}).get("sources",{})
            if isinstance(sources,dict): sources=list(sources.values())
            if isinstance(sources,list):
                table(sources,"provenance")
            else:
                st.json(sources,expanded=False)
            st.caption("Source identifiers and hashes document lineage. The runtime reads the prepared presentation bundle and app-owned assets only.")
    with tabs[2]: table(GLOSSARY,"glossary")
    with tabs[3]:
        for limitation in LIMITATIONS: st.markdown("- "+limitation)
        st.markdown("#### Selected primary references")
        st.markdown("[MPI-LEMON dataset — Babayan et al.](https://www.nature.com/articles/sdata2018308)  \n[Conditional EEG diffusion — Klein et al. (2024)](https://arxiv.org/abs/2403.18486)  \n[DDPM — Ho et al.](https://proceedings.neurips.cc/paper_files/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)  \n[EEGNet — Lawhern et al.](https://doi.org/10.1088/1741-2552/aace8c)  \n[Debiased weighted phase-lag index — Vinck et al.](https://doi.org/10.1016/j.neuroimage.2011.01.055)  \n[Permutation entropy — Bandt and Pompe](https://doi.org/10.1103/PhysRevLett.88.174102)")
        st.caption("Links are optional reading. The presentation itself does not fetch these sites. The conditional diffusion reference concerns event-related potentials; this study uses resting-state EO/EC and does not inherit session conditioning or classifier-free guidance.")
