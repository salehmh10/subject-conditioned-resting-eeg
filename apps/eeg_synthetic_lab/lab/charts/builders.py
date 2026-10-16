"""Display saved observations without fitting models or inventing uncertainty."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import textwrap
from plotly.subplots import make_subplots

from ..config import COLORS, DASHES, CONDITIONS, CLASSIFIER_COLORS
from ..formatting import classifier_label, condition_label, condition_model, readable_metric


def theme(fig, title, subtitle="", height=530):
    subtitle="<br>".join(textwrap.wrap(subtitle,width=105,break_long_words=False,break_on_hyphens=False))
    fig.update_layout(
        title=dict(text=title + (f"<br><span style='font-size:12px;color:#536B7E'>{subtitle}</span>" if subtitle else ""), x=0, xanchor="left", font=dict(size=21)),
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", size=13, color="#233C58"), height=height,
        margin=dict(l=55, r=28, t=105, b=100),
        legend=dict(orientation="h", yanchor="top", y=-.19, x=0, title_text="",font=dict(size=12)),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=13),
    )
    fig.update_xaxes(gridcolor="#EAF0F4", zerolinecolor="#CAD7E1", linecolor="#D9E3EB",automargin=True)
    fig.update_yaxes(gridcolor="#EAF0F4", zerolinecolor="#CAD7E1", linecolor="#D9E3EB",automargin=True)
    return fig


def split_label(split):
    return "Final Test — held-out blocks" if split == "final_test" else "Validation — model selection"


def utility_comparison(summary, runs=None, split="final_test", families=None, show_seeds=True):
    rows = [r for r in summary if r["split"] == split and (not families or r["family"] in families)]
    family_keys = list(dict.fromkeys(r["family"] for r in rows))
    fig = make_subplots(rows=1, cols=max(1, len(family_keys)), shared_yaxes=True,
                        horizontal_spacing=0.055, subplot_titles=[classifier_label(f).replace(" + ", " +<br>") for f in family_keys])
    ymin, ymax = [], []
    for col, family in enumerate(family_keys, 1):
        family_rows = {r["condition"]: r for r in rows if r["family"] == family}
        ordered = [family_rows[c] for c in CONDITIONS if c in family_rows]
        y = [CONDITIONS.index(r["condition"]) for r in ordered]
        means = [r["mean"] * 100 for r in ordered]
        sd = [r.get("population_sd", 0) * 100 if r.get("n_seeds", 1) > 1 else 0 for r in ordered]
        colors = [COLORS[condition_model(r["condition"])] for r in ordered]
        hover = [condition_label(r["condition"]) for r in ordered]
        seed_error=dict(type="data", array=sd, color="#536B7E", thickness=1.4, width=4) if any(r.get("n_seeds",1)>1 for r in ordered) else None
        fig.add_trace(go.Scatter(x=means, y=y, mode="markers", name=classifier_label(family), showlegend=False,
                                marker=dict(color=colors, size=12, symbol="diamond", line=dict(width=1, color="white")),
                                error_x=seed_error,
                                customdata=hover, hovertemplate="%{customdata}<br>Mean subject-macro BA: %{x:.3f}%<extra></extra>"), row=1, col=col)
        if show_seeds and runs:
            for j, r in enumerate(runs):
                if r["split"] == split and r["family"] == family and r["condition"] in CONDITIONS and "EEGNET" in family.upper():
                    val = r.get("subject_macro_balanced_accuracy")
                    if val is not None:
                        fig.add_trace(go.Scatter(x=[100 * val], y=[CONDITIONS.index(r["condition"])], mode="markers",
                                                marker=dict(size=6, symbol="circle-open", color=COLORS[condition_model(r["condition"])]), showlegend=False,
                                                hovertemplate=f"{condition_label(r['condition'])}<br>Training seed {r['seed']}<br>Subject-macro BA: %{{x:.3f}}%<extra></extra>"), row=1, col=col)
        if means:
            fig.update_xaxes(range=[min(means) - 3, max(means) + 2], title_text="Subject-macro BA (%)", row=1, col=col)
        fig.update_yaxes(tickvals=list(range(11)), ticktext=[condition_label(c) for c in CONDITIONS], autorange="reversed", row=1, col=col)
    theme(fig, "How training data change eye-state classification", f"{split_label(split)} · 198 participants · diamond = mean; EEGNet whiskers = population SD across 3 seeds", 670)
    fig.update_layout(margin=dict(l=195, r=22, t=140, b=60))
    for a in fig.layout.annotations:
        a.font.size = 13
    return fig


def utility_dose(summary, family, split="final_test"):
    rows = {r["condition"]: r for r in summary if r["family"] == family and r["split"] == split}
    fig = go.Figure()
    base = rows.get("REAL_ONLY")
    for model in ("B0", "B1", "B2"):
        selected = [base, rows.get(f"REAL_PLUS_{model}_0P5X"), rows.get(f"REAL_PLUS_{model}_1X")]
        fig.add_trace(go.Scatter(x=[0, .5, 1], y=[100*r["mean"] if r else None for r in selected], mode="lines+markers", name=model,
                                line=dict(color=COLORS[model], dash=DASHES[model], width=3), marker=dict(size=10),
                                hovertemplate=model + "<br>Synthetic dose: %{x}×<br>Subject-macro BA: %{y:.3f}%<extra></extra>"))
    duplicate = rows.get("REAL_DUPLICATE_1X_CONTROL")
    if duplicate:
        fig.add_hline(y=100*duplicate["mean"], line_dash="dot", line_color=COLORS["Real"], annotation_text="Real duplicate control", annotation_position="bottom right")
    theme(fig, "Augmentation dose and the duplicate control", f"{classifier_label(family)} · {split_label(split)} · condition means")
    fig.update_xaxes(title="Synthetic windows / Real Train windows", tickvals=[0,.5,1], ticktext=["0× · Real only", "0.5× · 23,757", "1× · 47,835"])
    fig.update_yaxes(title="Subject-macro balanced accuracy (%)")
    return fig


def utility_validation_test(summary, family):
    val = {r["condition"]: r for r in summary if r["split"] == "validation" and r["family"] == family}
    test = {r["condition"]: r for r in summary if r["split"] == "final_test" and r["family"] == family}
    fig = go.Figure()
    for c in CONDITIONS:
        if c in val and c in test:
            fig.add_trace(go.Scatter(x=[100*val[c]["mean"],100*test[c]["mean"]], y=[condition_label(c)]*2, mode="lines+markers", name=condition_label(c), showlegend=False,
                                    line=dict(color=COLORS[condition_model(c)], width=2), marker=dict(size=11,symbol=["circle-open","diamond"]),
                                    customdata=["Validation — model selection", "Final Test — held-out blocks"],
                                    hovertemplate=f"{condition_label(c)}<br>%{{customdata}}<br>%{{x:.3f}}%<extra></extra>"))
    theme(fig, "Validation and Final Test, shown descriptively", f"{classifier_label(family)} · open circle = Validation; diamond = Final Test · no new hypothesis test", 630)
    fig.update_xaxes(title="Subject-macro balanced accuracy (%)")
    fig.update_yaxes(autorange="reversed")
    return fig


def contrast_forest(contrasts, family=None, split="final_test"):
    rows = [r for r in contrasts if r["split"] == split and (family is None or r["family"] == family)]
    fig = go.Figure()
    for i, r in enumerate(rows):
        label = f"{condition_label(r['left'])}<br>minus {condition_label(r['right'])}"
        d,lo,hi = [100*r[k] if r.get(k) is not None else None for k in ("delta","ci_low","ci_high")]
        err = dict(type="data", array=[hi-d], arrayminus=[d-lo], symmetric=False, thickness=2, width=5) if lo is not None and hi is not None else None
        fig.add_trace(go.Scatter(x=[d],y=[label],mode="markers",showlegend=False,error_x=err,
                                marker=dict(color=COLORS[condition_model(r["left"])],size=11,symbol="diamond"),
                                hovertemplate=f"{label}<br>Difference: %{{x:+.3f}} pp<br>Saved 95% interval: {lo:+.3f}, {hi:+.3f} pp<extra></extra>" if lo is not None else label))
    fig.add_vline(x=0,line_dash="dot",line_color="#8D9BAA")
    theme(fig, "Saved paired participant contrasts", f"{split_label(split)} · left condition minus right condition · existing descriptive 95% intervals", max(450, 90 + len(rows)*58))
    fig.update_xaxes(title="Difference in subject-macro BA (percentage points)")
    fig.update_yaxes(autorange="reversed")
    return fig


def fidelity_bars(rows, metric=None):
    selected = [r for r in rows if metric is None or r["metric"] == metric]
    fig = go.Figure()
    for r in selected:
        value = r.get("mean")
        lo, hi = r.get("ci_low"), r.get("ci_high")
        err = dict(type="data", array=[hi-value], arrayminus=[value-lo], symmetric=False) if lo is not None and hi is not None and value is not None else None
        fig.add_trace(go.Bar(x=[r["model"]], y=[value], name=r["model"], marker_color=COLORS.get(r["model"], "#486581"), error_y=err, showlegend=False,
                            text=[f"{value:.3f}" if value is not None else "Unavailable"], textposition="outside",
                            hovertemplate=f"{r['model']}<br>Mean: %{{y:.5g}} {r.get('unit','')}<extra></extra>"))
    if selected:
        r = selected[0]
        direction = {"lower":"lower is closer", "higher":"higher is better", "parity":"parity at 1", "descriptive":"descriptive; no preferred direction"}.get(r.get("direction"), "descriptive")
        theme(fig, r.get("label", readable_metric(r["metric"])), f"Real Train reference · participant-level summary · {direction}", 430)
        fig.update_yaxes(title=r.get("unit", "Source units"), rangemode="tozero")
        if "parity" in direction:
            fig.add_hline(y=1,line_dash="dot",annotation_text="Parity at 1")
    return fig


def psd_chart(frequencies, spectra, state="EC", channel="Channel mean", aggregation="Group · linear power averaged before dB"):
    fig = go.Figure()
    for model in ("Real","B0","B1","B2"):
        if model in spectra:
            fig.add_trace(go.Scatter(x=frequencies,y=spectra[model],name=model,mode="lines",
                                    line=dict(color=COLORS[model],width=2.6,dash=DASHES[model]),
                                    hovertemplate=model + "<br>%{x:.1f} Hz<br>%{y:.2f} dB (µV²/Hz)<extra></extra>"))
    fig.add_vrect(x0=8,x1=13,fillcolor="#087F8C",opacity=.06,line_width=0)
    theme(fig, f"{state} power spectral density · {channel}", f"{aggregation} · Real Train reference")
    fig.update_xaxes(title="Frequency (Hz)",range=[1,40])
    fig.update_yaxes(title="Power spectral density [dB (µV²/Hz)]")
    # Plotly otherwise autoscales y using the hidden 40–64 Hz stopband too.
    # Keep every saved point, but derive initial limits from the displayed band.
    visible=(np.asarray(frequencies,dtype=float)>=1)&(np.asarray(frequencies,dtype=float)<=40)
    values=np.concatenate([np.asarray(v,dtype=float)[visible] for v in spectra.values()]) if spectra else np.array([])
    values=values[np.isfinite(values)]
    if len(values):
        low,high=float(values.min()),float(values.max())
        padding=max(1.0,(high-low)*.08)
        fig.update_yaxes(range=[low-padding,high+padding])
    return fig


def alpha_chart(channels, values, aggregation="Group mean across participants"):
    fig = go.Figure()
    for model, data in values.items():
        fig.add_trace(go.Scatter(x=channels,y=data,mode="lines+markers",name=model,line=dict(color=COLORS.get(model),dash=DASHES.get(model),width=2),
                                hovertemplate=model + "<br>%{x}<br>EC − EO: %{y:.3f} dB<extra></extra>"))
    theme(fig,"Alpha eye-closure effect",f"{aggregation} · EC minus EO · Real Train reference")
    fig.update_yaxes(title="Alpha bandpower difference (dB)")
    fig.update_xaxes(title="EEG channel")
    return fig


def connectivity_grid(matrices, channels, state="EC", band="alpha", difference=False, limits=None, aggregation="Group mean across participants"):
    models = [m for m in ("Real","B0","B1","B2") if m in matrices]
    if difference:
        models = [m for m in models if m != "Real"]
    cols = 3 if difference else 2
    fig = make_subplots(rows=1 if difference else 2,cols=cols,subplot_titles=[f"{m} − Real" if difference else m for m in models],vertical_spacing=.16,horizontal_spacing=.12)
    vals={}
    for m in models:
        mat=np.asarray(matrices[m],dtype=float).copy()
        if difference:
            mat=mat-np.asarray(matrices["Real"],dtype=float)
        np.fill_diagonal(mat,np.nan)
        vals[m]=mat
    finite=np.concatenate([a[np.isfinite(a)] for a in vals.values()]) if vals else np.array([0.])
    if limits is None:
        zmin,zmax=(float(np.min(finite)),float(np.max(finite))) if len(finite) else (-1,1)
        if difference:
            zmax=max(abs(zmin),abs(zmax)); zmin=-zmax
    else:
        zmin,zmax=limits
    for i,(m,mat) in enumerate(vals.items()):
        fig.add_trace(go.Heatmap(z=mat,x=channels,y=channels,zmin=zmin,zmax=zmax,
                                colorscale="RdBu_r" if difference else [[0,"#F3F7FA"],[.45,"#70B6C3"],[1,"#183D5D"]],
                                showscale=i==len(vals)-1,colorbar=dict(title="Δ dwPLI²" if difference else "dwPLI²",thickness=12,len=.78),
                                hoverongaps=False,hovertemplate=f"{m}{' − Real' if difference else ''}<br>%{{y}} ↔ %{{x}}<br>dwPLI²: %{{z:.4f}}<extra></extra>"), row=i//cols+1,col=i%cols+1)
    theme(fig,f"{state} · {band.capitalize()} sensor-space connectivity" + (" differences" if difference else ""),f"{aggregation} · same color scale across displayed matrices · diagonal masked", 490 if difference else 850)
    fig.update_xaxes(tickangle=-60,tickfont_size=10,showgrid=False)
    fig.update_yaxes(autorange="reversed",tickfont_size=10,showgrid=False)
    fig.update_layout(margin=dict(l=38,r=85,t=112,b=75))
    return fig


def fingerprint_chart(rows, metric="top1"):
    fig=go.Figure()
    for model in ("B0","B1","B2"):
        selected={r["state"]:r for r in rows if r["model"]==model}
        fig.add_trace(go.Bar(x=["EO","EC"],y=[100*selected[s][metric] if s in selected else None for s in ("EO","EC")],name=model,marker_color=COLORS[model],
                            text=[f"{100*selected[s][metric]:.2f}%" if s in selected else "" for s in ("EO","EC")],textposition="outside",
                            hovertemplate=model+"<br>%{x}<br>Identification: %{y:.2f}%<extra></extra>"))
    count=next((r.get("n_candidates") for r in rows if r.get("n_candidates")),None)
    if count:
        rank=1 if metric=="top1" else 5
        fig.add_hline(y=100*rank/count,line_dash="dot",line_color="#536B7E",annotation_text=f"Uniform-random ranking reference: {rank}/{count}",annotation_position="top left")
    theme(fig,"Participant fingerprint · " + ("Top-1" if metric=="top1" else "Top-5"),f"76 log-bandpower features · {count or 'source-reported'} candidate participants · Real Train reference",480)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Correct participant retrieval (%)",range=[0,112])
    fig.update_xaxes(title="Eye state")
    return fig


def learning_chart(histories, condition, seeds=None):
    selected=[h for h in histories if h["condition"]==condition and (not seeds or h["seed"] in seeds)]
    fig=make_subplots(rows=1,cols=2,subplot_titles=["Training and Validation loss","Validation — model selection"],horizontal_spacing=.1)
    for i,h in enumerate(selected):
        epochs=h["epochs"]; color=CLASSIFIER_COLORS[i%len(CLASSIFIER_COLORS)]
        x=[e["epoch"] for e in epochs]
        for field,dash,label in (("train_loss","solid","Training"),("validation_loss","dash","Validation")):
            fig.add_trace(go.Scatter(x=x,y=[e.get(field) for e in epochs],mode="lines",name=f"Seed {h['seed']} · {label}",line=dict(color=color,dash=dash,width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=x,y=[100*e["validation_ba"] if e.get("validation_ba") is not None else None for e in epochs],mode="lines",name=f"Seed {h['seed']} · BA",line=dict(color=color,width=2),showlegend=False),row=1,col=2)
        best=next((e for e in epochs if e["epoch"]==h.get("best_epoch")),None)
        stop=next((e for e in epochs if e["epoch"]==h.get("stopped_epoch")),None)
        for e,symbol,label in ((best,"star","Selected epoch"),(stop,"square-open","Stopping epoch")):
            if e and e.get("validation_ba") is not None:
                fig.add_trace(go.Scatter(x=[e["epoch"]],y=[100*e["validation_ba"]],mode="markers",marker=dict(color=color,symbol=symbol,size=13),showlegend=False,
                                        hovertemplate=f"Seed {h['seed']} · {label}<br>Epoch %{{x}}<br>Validation BA: %{{y:.3f}}%<extra></extra>"),row=1,col=2)
    theme(fig,f"Learning behavior · {condition_label(condition)}","Actual saved epochs only · star = selected checkpoint · square = actual stopping epoch",560)
    fig.update_xaxes(title="Actual epoch")
    fig.update_yaxes(title="Cross-entropy loss",row=1,col=1)
    fig.update_yaxes(title="Subject-macro BA (%)",row=1,col=2)
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=-.21),margin=dict(b=140))
    return fig


def workflow_chart():
    fig=go.Figure()
    boxes=[(.12,.53,"MPI-LEMON EEG","Known participants · held-out blocks",COLORS["Real"]),(.43,.78,"B0 · eye state","Shared temporal 1D U-Net",COLORS["B0"]),(.43,.51,"B1 · participant + state","Participant embedding added",COLORS["B1"]),(.43,.24,"B2 · participant + state","Continued training · same architecture",COLORS["B2"]),(.81,.69,"Signal Fidelity","Compare with Real Train",COLORS["Real"]),(.81,.31,"Classification Utility","Evaluate saved held-out-block results",COLORS["Real"])]
    for x,y,title,detail,c in boxes:
        fig.add_shape(type="rect",x0=x-.145,x1=x+.145,y0=y-.10,y1=y+.10,line=dict(color=c,width=1.5),fillcolor="#F4F8FA")
        fig.add_annotation(x=x,y=y+.025,text=f"<b>{title}</b>",showarrow=False,font=dict(size=15,color=c))
        fig.add_annotation(x=x,y=y-.035,text=detail,showarrow=False,font=dict(size=10,color="#536B7E"))
    for y in (.78,.51,.24):
        fig.add_annotation(x=.28,y=y,ax=.267,ay=.53,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowcolor="#A8BBC9")
        for end in (.69,.31):
            fig.add_annotation(x=.65,y=end,ax=.58,ay=y,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowcolor="#A8BBC9")
    theme(fig,"One study, two complementary questions","Fidelity describes signals; Utility evaluates how classifier training data affect eye-state prediction",410)
    fig.update_xaxes(visible=False,range=[-.035,1])
    fig.update_yaxes(visible=False,range=[.04,.99])
    fig.update_layout(margin=dict(l=10,r=10,t=100,b=10))
    return fig


def split_chart(metadata):
    counts=metadata.get("window_counts",{})
    x=["Train", "Validation", "Final Test"]
    y=[metadata.get("n_train",counts.get("train")),metadata.get("n_validation",counts.get("validation")),metadata.get("n_test",counts.get("final_test",counts.get("test")))]
    fig=go.Figure(go.Bar(x=x,y=y,marker_color=["#233C58","#087F8C","#625AB4"],text=[f"{n:,}" if n is not None else "Unavailable" for n in y],textposition="outside",hovertemplate="%{x}<br>%{y:,} windows<extra></extra>"))
    theme(fig,"Different blocks from the same participants","Four-second windows · each participant contributes to all three splits",440)
    fig.update_yaxes(title="Number of EEG windows",rangemode="tozero")
    return fig


def waveform_chart(time_seconds, waves, channels, selected_channel=None, unit="µV", stacked=False):
    fig=make_subplots(rows=2,cols=2,subplot_titles=list(waves),shared_xaxes=True,shared_yaxes=True,vertical_spacing=.14)
    arrays={m:np.asarray(v,dtype=float) for m,v in waves.items()}
    max_abs=max(float(np.nanmax(np.abs(v))) for v in arrays.values())
    offset=max_abs*2.2
    for k,(model,arr) in enumerate(arrays.items()):
        indices=range(len(channels)) if stacked else [channels.index(selected_channel) if selected_channel in channels else 0]
        for j in indices:
            y=arr[j]+(j*offset if stacked else 0)
            fig.add_trace(go.Scatter(x=time_seconds,y=y,mode="lines",line=dict(color=COLORS[model],width=1),showlegend=False,customdata=arr[j],
                                    hovertemplate=f"{model} · {channels[j]}<br>%{{x:.3f}} s<br>%{{customdata:.2f}} {unit}<extra></extra>"),row=k//2+1,col=k%2+1)
    theme(fig,"Precomputed Train and synthetic examples","Separate precomputed windows · synthetic samples are not pointwise reconstructions",850 if stacked else 550)
    fig.update_xaxes(title="Time (s)")
    if stacked:
        fig.update_yaxes(tickvals=[j*offset for j in range(len(channels))],ticktext=channels,title="Channel (shared vertical offset)")
    else:
        fig.update_yaxes(title=f"Amplitude ({unit})",range=[-max_abs*1.05,max_abs*1.05])
    return fig
