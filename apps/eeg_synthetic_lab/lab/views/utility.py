import pandas as pd
import streamlit as st
from ..charts import utility_comparison,contrast_forest,utility_dose,utility_validation_test
from ..formatting import classifier_label,condition_label
from .common import heading,chart,select,table,message,headline,eegnet_family,frame


def comparison_message(rows,split,families):
    descriptions=[]
    for family in families:
        by_condition={r["condition"]:r for r in rows if r["family"]==family and r["split"]==split}
        real,b2,duplicate=(by_condition.get(c) for c in ("REAL_ONLY","REAL_PLUS_B2_1X","REAL_DUPLICATE_1X_CONTROL"))
        if real and b2:
            text=f"{classifier_label(family)}: B2 at 1× changes BA by {100*(b2['mean']-real['mean']):+.3f} pp versus Real only"
            if len(families)==1 and duplicate:
                text+=f"; Real Duplicate changes it by {100*(duplicate['mean']-real['mean']):+.3f} pp"
            descriptions.append(text)
    suffix=" These effects depend on the classifier." if len(families)>1 else " This is a condition-specific descriptive comparison."
    if split=="validation": suffix+=" Validation is model-selection evidence."
    return ". ".join(descriptions)+"."+suffix


def render(bundle):
    heading("CLASSIFICATION UTILITY","Does synthetic EEG help eye-state prediction?","All 11 conditions and all three classifier families, using the completed 120-epoch / early-stopping study.")
    rows=bundle.get("utility_summary",[])
    if not rows:
        st.info("Prepared utility summaries are unavailable.")
        return
    cols=st.columns([2,2,1])
    families=list(dict.fromkeys(r["family"] for r in rows))
    with cols[0]: split=select("Evaluation split",["final_test","validation"],"utility:split",format_func=lambda s:"Final Test — held-out blocks" if s=="final_test" else "Validation — model selection")
    with cols[1]: chosen=select("Classifier comparison",["All three classifiers"]+families,"utility:comparison",format_func=lambda f:f if f=="All three classifiers" else classifier_label(f))
    with cols[2]: seeds=st.checkbox("Show EEGNet seeds",value=True,key="utility:seeds")
    tabs=st.tabs(["Complete comparison","Paired contrasts","Augmentation dose","Validation vs Test","Full results table"])
    with tabs[0]:
        fig=utility_comparison(rows,bundle.get("utility_runs",[]),split,None if chosen=="All three classifiers" else [chosen],seeds)
        selected=[r for r in rows if r["split"]==split and (chosen=="All three classifiers" or r["family"]==chosen)]
        caption="Classical classifiers have one fit per condition. EEGNet diamonds summarize three training seeds; population SD across seeds is distinct from a participant-bootstrap confidence interval."
        chart(fig,caption,"utility_comparison",selected)
        message(comparison_message(rows,split,families if chosen=="All three classifiers" else [chosen]))
        st.caption("Real Duplicate is a control for repeated data and greater optimizer-update opportunity. Equal epoch ceilings do not imply equal numbers of parameter updates.")
    with tabs[1]:
        family=select("Classifier for paired contrasts",families,"utility:contrast_family",index=families.index(eegnet_family(bundle)),format_func=classifier_label)
        contrasts=[r for r in bundle.get("contrasts",[]) if r["split"]==split and r["family"]==family]
        if contrasts:
            chart(contrast_forest(contrasts,family,split),"These are saved paired participant contrasts. Descriptive intervals are not multiplicity-adjusted and must not be confused with between-seed SD.","utility_contrasts",contrasts)
        else:
            st.info("No saved paired contrasts are available for this classifier and split. No interval is inferred from the summary values.")
    with tabs[2]:
        family=select("Classifier for dose comparison",families,"utility:dose_family",index=families.index(eegnet_family(bundle)),format_func=classifier_label)
        chart(utility_dose(rows,family,split),"Each synthetic dose is compared with the same Real-only starting point. The dotted line retains the Real Duplicate control; connected points describe the saved conditions, not an estimated dose-response model.","utility_dose")
        st.caption("The matched 0.5× subset contains 23,757 synthetic windows; 1× contains 47,835. Each synthetic pool is fixed across classifier seeds.")
    with tabs[3]:
        family=select("Classifier for split comparison",families,"utility:split_family",index=families.index(eegnet_family(bundle)),format_func=classifier_label)
        chart(utility_validation_test(rows,family),"This descriptive comparison uses different blocks from the same participants. Final Test did not provide the learning curves or choose the classifier checkpoints.","utility_split")
    with tabs[4]:
        level=select("Table detail",["33 classifier × condition summaries","55 individual fits"],"utility:table_detail")
        data=[r for r in (rows if level.startswith("33") else bundle.get("utility_runs",[])) if r["split"]==split]
        df=frame(data)
        if "family" in df: df["family"]=df["family"].map(classifier_label)
        if "condition" in df: df["condition"]=df["condition"].map(condition_label)
        percent_fields={"mean":"Subject-macro BA (%)","population_sd":"Between-seed population SD (pp)","ci_low":"Subject-bootstrap CI lower (%)","ci_high":"Subject-bootstrap CI upper (%)","subject_macro_balanced_accuracy":"Subject-macro BA (%)","global_balanced_accuracy":"Global balanced accuracy (%)"}
        for key,label in percent_fields.items():
            if key in df: df[label]=pd.to_numeric(df.pop(key),errors="coerce")*100
        df=df.rename(columns={"family":"Classifier","condition":"Training condition","n_seeds":"Number of fits","seed":"Training seed","auc":"Subject-macro AUC","subject_macro_roc_auc":"Subject-macro AUC","global_roc_auc":"Global AUC"})
        table(df,"utility_full","Click a column header to sort. BA percentages and percentage-point SD are displayed explicitly; AUC remains on its original 0–1 scale.")
    with st.expander("How to read the 11 training conditions"):
        st.write("Real only uses Real Train. Real duplicate repeats every Real Train row once. TSTR trains the classifier on a frozen synthetic pool only. Real + B0/B1/B2 adds a matched half or full synthetic pool to Real Train. EEGNet still uses real Validation data to select its checkpoint in TSTR conditions.")
        st.write("Subject-macro balanced accuracy is computed by averaging EO and EC recall within each participant, then averaging participants equally. It is not pooled window accuracy.")
