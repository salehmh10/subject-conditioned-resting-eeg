"""Generate native LaTeX tables from the delivered saved-evidence snapshot."""
import csv
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((REPORT / "tables/fidelity_representative.csv").open(encoding="utf-8")))
groups = {"Amplitude": "دامنه", "Spectrum": "طیف", "State effect": "اثر حالت", "Temporal": "زمان", "Distribution": "توزیع", "Diversity": "تنوع", "Spatial": "فضایی", "Connectivity": "اتصال آلفا"}
names = {
    "SD-MD": r"SD-MD", "Log-PSD MAE": r"Log-PSD MAE", "Bandpower MAE": r"Bandpower MAE",
    "Alpha EC-EO MAE": r"Alpha EC--EO MAE", "Hjorth mobility W1": r"Hjorth mobility $W_1$",
    "ACF MAE": r"ACF MAE", "Permutation entropy W1": r"Permutation entropy $W_1$",
    "Sliced Wasserstein-1": r"Sliced Wasserstein-1", "RMS pairwise ratio": r"RMS pairwise ratio",
    "Covariance relative error": r"Covariance relative error", "Channel correlation r": r"Channel correlation $r$",
    "Alpha dwPLI2 edge MAE": r"dwPLI$^2$ edge MAE", "Alpha dwPLI2 edge r": r"dwPLI$^2$ edge $r$",
    "Alpha node-mean MAE": r"dwPLI$^2$ node-mean MAE",
}
units = {"uV": r"$\mu$V", "dB": "dB", "per sample": "sample$^{-1}$", "unitless": "---", "Train-z": "Train-z", "ratio": "---", "r": "---"}
lines = [r"% Generated from fidelity_representative.csv; numbers rounded to four decimals.",
         r"{\fontsize{10}{12}\selectfont\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.13}",
         r"\begin{tabular}{@{}p{1.50cm}p{4.60cm}ccccc@{}}", r"\toprule",
         r"خانواده & معیار & واحد & \lr{B0} & \lr{B1} & \lr{B2} & جهت مطلوب \\", r"\midrule"]
previous = None
for row in rows:
    family = row["family"]
    if previous is not None and family != previous:
        lines.append(r"\addlinespace[2pt]")
    direction = "نزدیک‌تر به ۱" if row["direction"] == "closer to 1" else "بیشتر" if row["direction"] == "higher" else "کمتر"
    cells = [groups[family] if family != previous else "", r"\lr{"+names[row["metric"]]+"}", r"\lr{"+units[row["unit"]]+"}"]
    cells += [r"\lr{"+f"{float(row[g]):.4f}"+"}" for g in ("B0", "B1", "B2")]
    cells.append(direction)
    lines.append(" & ".join(cells) + r" \\")
    previous = family
lines += [r"\bottomrule", r"\end{tabular}", r"}"]
(REPORT / "tables/fidelity_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")

data = json.loads((REPORT / "fidelity_data.json").read_text(encoding="utf-8"))
specs = [
    (r"Log-PSD MAE (dB)", data["matched_train_block_calibration_summary"]["log_psd_mae_db"], "کمتر"),
    (r"Sliced Wasserstein-1 (Train-z)", data["matched_train_block_calibration_summary"]["sliced_wasserstein1_z"], "کمتر"),
    (r"Alpha dwPLI$^2$ edge MAE", data["matched_train_block_connectivity_summary"]["alpha"]["upper_mae"], "کمتر"),
    (r"Alpha dwPLI$^2$ edge $r$", data["matched_train_block_connectivity_summary"]["alpha"]["upper_r"], "بیشتر"),
]
lines = [r"% Matched disjoint Train block calibration; all compared against Real A.",
         r"{\fontsize{10}{12}\selectfont\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.14}",
         r"\begin{tabular}{@{}p{5.4cm}ccccc@{}}", r"\toprule",
         r"معیار & \lr{Real--Real} & \lr{B0} & \lr{B1} & \lr{B2} & جهت مطلوب \\", r"\midrule"]
for name, values, direction in specs:
    cells = [r"\lr{"+name+"}"]+[r"\lr{"+f"{values[g]['mean']:.4f}"+"}" for g in ("Real_B", "B0", "B1", "B2")]+[direction]
    lines.append(" & ".join(cells)+r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"}"]
(REPORT / "tables/calibration_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
mapping_path = REPORT / "source_map_fidelity.csv"
mapping = list(csv.DictReader(mapping_path.open(encoding="utf-8")))
outputs = {row["output"] for row in mapping}
for output, item, source, location, transformation in [
    ("tables/fidelity_table.tex", "14 representative fidelity means", "tables/fidelity_representative.csv", "all 14 rows", "Four-decimal display; native LaTeX; exact saved means preserved in CSV"),
    ("tables/calibration_table.tex", "4 matched Real-Real calibration metrics", "fidelity_data.json", "matched_train_block_calibration_summary; matched_train_block_connectivity_summary.alpha", "Four-decimal display; Real_B means Real B versus A, synthetic means source versus Real A"),
]:
    if output not in outputs:
        mapping.append(dict(output=output, item=item, source_file=source, source_location=location, transformation=transformation))
with mapping_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(mapping[0])); writer.writeheader(); writer.writerows(mapping)
print("Wrote native Fidelity and Real-Real calibration tables")
