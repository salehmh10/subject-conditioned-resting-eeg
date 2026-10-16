"""Redraw saved Fidelity summaries; never imports evaluation code or reads signals.

Portable: python tools/build_fidelity_figures.py
Refresh from this repository's saved reports only: add --extract.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]
B0 = Path("outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1")
B12 = Path("outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1")
REPORT0 = B0 / "metrics/s2_74e_b0_complete_report_v1.json"
REPORT12 = B12 / "metrics/s2_74b_complete_report_v1.json"
CHANNELS = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "T7", "T8", "C3", "C4", "Cz", "P7", "P8", "P3", "P4", "Pz", "O1", "O2"]
SOURCES = ("Real", "B0", "B1", "B2")
COLORS = {"Real": "#222222", "B0": "#D98932", "B1": "#277DA8", "B2": "#8C4F9F"}


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract():
    r, original = read(REPORT0), read(REPORT12)
    # The extension must preserve the original final B1/B2 estimates exactly.
    for key in r["full_pool_metrics"]:
        for g in ("B1", "B2"):
            assert r["full_pool_metrics"][key][g] == original["full_pool_metrics"][key][g]
    subs = sorted((ROOT / B0 / "subjects").glob("*.json"))
    assert len(subs) == 198
    records = [(json.loads(p.read_text()), read(B12 / "subjects" / p.name)) for p in subs]
    for a, b in records:
        assert a["subject_id"] == b["subject_id"]

    def source_state(a, b, state, source):
        return (a if source == "B0" else b)["states"][state]

    psd, alpha, connectivity = {}, {}, {}
    for state in ("EO", "EC"):
        psd[state], connectivity[state] = {}, {}
        for source in SOURCES:
            powers = np.asarray([source_state(a, b, state, source)["source_summaries"][source]["mean_psd_uv2_hz"] for a, b in records])
            psd[state][source] = (10 * np.log10(powers.mean(axis=(0, 1)))).tolist()
            matrices = np.asarray([source_state(a, b, state, source)["nonoverlap_connectivity_matrices"]["alpha"][source]["matrix"] for a, b in records], dtype=float)
            connectivity[state][source] = np.nanmean(matrices, axis=0).tolist()
    for source in SOURCES:
        differences = []
        for a, b in records:
            ec = np.asarray(source_state(a, b, "EC", source)["source_summaries"][source]["median_bandpower_db"])[2]
            eo = np.asarray(source_state(a, b, "EO", source)["source_summaries"][source]["median_bandpower_db"])[2]
            differences.append(ec - eo)
        alpha[source] = np.mean(differences, axis=0).tolist()

    data = {
        "reference": "Real Train; descriptive in-sample characterization",
        "n_subjects": 198,
        "aggregation": "Equal weight for subjects; EO and EC averaged within each subject for aggregate table estimates.",
        "bootstrap": r["bootstrap"],
        "channels": CHANNELS,
        "frequency_hz": np.fft.rfftfreq(256, 1 / 128).tolist(),
        "psd_db": psd,
        "alpha_ec_minus_eo_db": alpha,
        "alpha_connectivity": connectivity,
        "full_pool_metrics": r["full_pool_metrics"],
        "alpha_EC_minus_EO": r["alpha_EC_minus_EO"],
        "connectivity": r["connectivity"],
        "connectivity_EC_minus_EO": r["connectivity_EC_minus_EO"],
        "fingerprint": r["fingerprint"],
        "matched_train_block_calibration_summary": r["matched_train_block_calibration_summary"],
        "matched_train_block_connectivity_summary": r["matched_train_block_connectivity_summary"],
        "source_reports": [REPORT0.as_posix(), REPORT12.as_posix()],
        "verification": {"198_paired_saved_subject_reports": True, "B1_B2_full_pool_summaries_match_original_exactly": True, "no_signal_file_access": True},
    }
    write_json(REPORT / "fidelity_data.json", data)
    make_tables(data)
    return data


def make_tables(data):
    # Fourteen representative rows. Fingerprint and calibration are displayed separately.
    specs = [
        ("Amplitude", "SD-MD", "full_pool_metrics", "sd_md_pooled_uv", "uV", "lower", "Mean absolute channel SD difference; each SD pools windows and time."),
        ("Spectrum", "Log-PSD MAE", "full_pool_metrics", "log_psd_mae_db", "dB", "lower", "MAE over channels and 1-40 Hz; per-channel mean linear PSD converted to dB."),
        ("Spectrum", "Bandpower MAE", "full_pool_metrics", "bandpower_median_mae_db", "dB", "lower", "Mean absolute error in channel-wise median per-window log-bandpower across delta/theta/alpha/beta."),
        ("State effect", "Alpha EC-EO MAE", "alpha_EC_minus_EO", "channel_mae_db", "dB", "lower", "Within-subject EC-EO median alpha bandpower contrast; MAE across channels."),
        ("Temporal", "Hjorth mobility W1", "full_pool_metrics", "hjorth_mobility_per_sample_w1", "per sample", "lower", "Mean across channels of 1-Wasserstein distance between per-window mobility distributions; mobility=sqrt(var(diff(x))/var(x))."),
        ("Temporal", "ACF MAE", "full_pool_metrics", "acf_mae_lag1_128", "unitless", "lower", "Mean absolute error of per-channel mean normalized ACF at lags 1..128."),
        ("Temporal", "Permutation entropy W1", "full_pool_metrics", "permutation_entropy_m3_tau1_w1", "unitless", "lower", "Mean channel W1 of normalized permutation entropy, order 3 delay 1; stable-index ties."),
        ("Distribution", "Sliced Wasserstein-1", "full_pool_metrics", "sliced_wasserstein1_z", "Train-z", "lower", "Mean W1 across 128 fixed unit-normal Gaussian projections of flattened 19x512 windows, normalized by the fixed Train scaler."),
        ("Diversity", "RMS pairwise ratio", "full_pool_metrics", "rms_pairwise_diversity_ratio", "ratio", "closer to 1", "Synthetic/real ratio of RMS pairwise distance in fixed Train-z coordinates; not larger-is-better."),
        ("Spatial", "Covariance relative error", "full_pool_metrics", "covariance_relative_frobenius_z", "unitless", "lower", "Frobenius norm of difference between mean window covariance matrices divided by norm of the real matrix, Train-z units."),
        ("Spatial", "Channel correlation r", "full_pool_metrics", "correlation_upper_r", "r", "higher", "Pearson r between upper triangles of mean within-window channel correlation matrices; distinct from lagged phase connectivity."),
        ("Connectivity", "Alpha dwPLI2 edge MAE", "connectivity.alpha", "upper_mae", "unitless", "lower", "MAE across 171 off-diagonal edges of alpha debiased squared wPLI matrices, count-matched nonoverlap windows."),
        ("Connectivity", "Alpha dwPLI2 edge r", "connectivity.alpha", "upper_r", "r", "higher", "Pearson r across 171 off-diagonal alpha dwPLI2 edges; measures pattern, not magnitude agreement."),
        ("Connectivity", "Alpha node-mean MAE", "connectivity.alpha", "node_mean_mae", "unitless", "lower", "MAE across channels in the mean incident alpha dwPLI2 edge weight; no graph thresholding."),
    ]
    rows, sources = [], []
    for family, name, section, key, unit, direction, definition in specs:
        branch = data
        for node in section.split("."):
            branch = branch[node]
        row = dict(family=family, metric=name, key=key, unit=unit, direction=direction,
                   reference="Real Train within same subject and eye state", definition=definition)
        for source in ("B0", "B1", "B2"):
            row[source] = branch[key][source]["mean"]
        rows.append(row)
        sources.append(dict(output="tables/fidelity_representative.csv", item=key, source_file=REPORT0.as_posix(), source_location=section+"."+key+".{B0,B1,B2}.mean", transformation="None: exact saved subject-macro means; displayed rounding only"))
    with (REPORT / "tables/fidelity_representative.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for item, transform in [
        ("psd", "Mean saved linear per-channel PSD across 198 subjects and 19 channels, then 10log10; 1-40 Hz display; no intervals."),
        ("alpha_effect", "For each subject/source/channel, saved median alpha bandpower EC minus EO; equal-subject mean; no intervals."),
        ("connectivity_EO", "Equal-subject mean of saved alpha dwPLI2 matrices; diagonal masked; common range across Real/B0/B1/B2 within EO."),
        ("connectivity_EC", "Equal-subject mean of saved alpha dwPLI2 matrices; diagonal masked; common range across Real/B0/B1/B2 within EC."),
    ]:
        sources.append(dict(output="figures/fidelity_"+item+".pdf", item=item, source_file=f"{B0.as_posix()}/subjects/*.json; {B12.as_posix()}/subjects/*.json", source_location="states.EO/EC.source_summaries or states.EO/EC.nonoverlap_connectivity_matrices.alpha", transformation=transform))
    sources.append(dict(output="figures/fidelity_fingerprint.pdf", item="fingerprint", source_file=REPORT0.as_posix(), source_location="fingerprint.{EO,EC}_{B0,B1,B2}.{top1,top5}", transformation="Multiply saved fractions by 100; no intervals or re-evaluation."))
    sources.append(dict(output="fidelity_data.json", item="all numerical fidelity findings", source_file=REPORT0.as_posix()+"; "+REPORT12.as_posix(), source_location="full_pool_metrics; alpha_EC_minus_EO; connectivity; fingerprint; matched_train_block_*", transformation="Saved reports copied as evidence; no new statistical tests."))
    with (REPORT / "source_map_fidelity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sources[0])); writer.writeheader(); writer.writerows(sources)


def save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(REPORT / "figures" / f"fidelity_{name}.{suffix}", dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "legend.frameon": False, "axes.titleweight": "semibold", "figure.constrained_layout.use": True})
    freq = np.asarray(data["frequency_hz"]); mask = (freq >= 1) & (freq <= 40)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
    for ax, state in zip(axes, ("EO", "EC")):
        ax.axvspan(8, 13, color="#777777", alpha=.08, lw=0)
        for g in SOURCES:
            ax.plot(freq[mask], np.asarray(data["psd_db"][state][g])[mask], label=g, c=COLORS[g], lw=1.75)
        ax.set(title=state, xlabel="Frequency (Hz)", xlim=(1, 40)); ax.grid(alpha=.15)
        ax.set_xticks([1, 10, 20, 30, 40])
    axes[0].set_ylabel(r"PSD (dB re 1 $\mu$V$^2$/Hz)")
    axes[1].legend(ncol=4, loc="upper right", fontsize=8, handlelength=1.4, columnspacing=0.8)
    save(fig, "psd")

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axhline(0, lw=.8, color=".65")
    for g in SOURCES:
        ax.plot(range(19), data["alpha_ec_minus_eo_db"][g], label=g, c=COLORS[g], lw=1.6, marker="o", ms=3.2)
    ax.set_xticks(range(19), CHANNELS, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Alpha EC − EO (dB)"); ax.grid(axis="y", alpha=.15)
    ax.legend(ncol=4, loc="upper left", fontsize=9)
    save(fig, "alpha_effect")

    for state in ("EO", "EC"):
        matrices = {g: np.asarray(data["alpha_connectivity"][state][g]) for g in SOURCES}
        edges = np.concatenate([m[np.triu_indices(19, 1)] for m in matrices.values()])
        lo, hi = np.nanmin(edges), np.nanmax(edges)
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.1))
        cmap = plt.colormaps["viridis"].copy(); cmap.set_bad("#EEEEEE")
        for ax, g in zip(axes.ravel(), SOURCES):
            mat = matrices[g].copy(); np.fill_diagonal(mat, np.nan)
            im = ax.imshow(np.ma.masked_invalid(mat), vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
            ax.set_title(g, color=COLORS[g], fontsize=12, pad=5)
            ax.set_xticks(range(19), CHANNELS, rotation=90, fontsize=7)
            ax.set_yticks(range(19), CHANNELS, fontsize=7)
            ax.tick_params(length=0, pad=2)
        bar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.83, fraction=.045, pad=.025)
        bar.set_label(r"Alpha dwPLI$^2$"); bar.ax.tick_params(labelsize=8)
        save(fig, "connectivity_" + state)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    x = np.arange(3); width = .34
    for ax, state in zip(axes, ("EO", "EC")):
        for i, g in enumerate(("B0", "B1", "B2")):
            for j, metric in enumerate(("top1", "top5")):
                value = 100 * data["fingerprint"][state+"_"+g][metric]
                ax.bar(i+(j-.5)*width, value, width, color=COLORS[g], alpha=1 if j == 0 else .35,
                       edgecolor=COLORS[g], linewidth=.8, hatch=None if j == 0 else "//")
                ax.text(i+(j-.5)*width, value+1.5, f"{value:.1f}", ha="center", fontsize=7.5, rotation=0)
        ax.set(title=state, ylim=(0, 110), xticks=x, xticklabels=["B0", "B1", "B2"])
        ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    axes[0].set_ylabel("Subject identification (%)")
    axes[0].legend(handles=[Patch(facecolor=".35", label="Top-1"), Patch(facecolor=".8", hatch="//", label="Top-5")], loc="upper left", fontsize=8)
    save(fig, "fingerprint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--extract", action="store_true"); args = parser.parse_args()
    data = extract() if args.extract else json.loads((REPORT / "fidelity_data.json").read_text(encoding="utf-8"))
    plot(data)
    print("Fidelity evidence and five figure families written under", REPORT)
