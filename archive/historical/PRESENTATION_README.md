# EEG Synthetic Lab

A local, English research presentation for a completed MPI-LEMON synthetic resting-state EEG study. The question is: **How closely do synthetic EEG signals reproduce real EEG, and does that similarity translate into useful eye-state classification?**

## Launch the prepared application

From the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\apps\eeg_synthetic_lab\run.ps1"
```

Or double-click `run.cmd` inside this folder. The expected address is **http://127.0.0.1:8501**. If that port is occupied, the launcher chooses another available loopback port and prints the address. It never terminates another process. Press Ctrl+C in the launching terminal to stop the app. Normal launch neither installs packages nor prepares evidence.

## First setup or reinstall

Python **3.12** was tested; a compatible installed Python 3.10–3.13 is required. Installation may need the internet. No GPU, PyTorch, API key, external server, or existing experiment environment is required.

```powershell
Set-Location ".\apps\eeg_synthetic_lab"
.\setup.ps1
.\prepare_data.ps1
.\run.ps1
```

When PowerShell execution policy prevents direct invocation, use `powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup.ps1"` (and the same form for the other scripts). `setup.ps1 -PythonExecutable "C:\Path With Spaces\Python\python.exe"` selects an explicit installed Python. The dedicated `.venv`, package caches and temporary files stay within this app directory. Repeating setup or preparation is supported. Preparation checks source signatures and reuses unchanged results.

The prepared bundle supplied in this workspace supports offline presentation after installation. The distributable ZIP contains an **aggregate edition** and can run without original scientific artifacts; do not rerun preparation on that copy unless the original repository evidence is available. A fresh installation is not an offline installation.

## What you can explore

- Overview, generator definitions and a vector architecture/workflow diagram.
- Genuine precomputed Real Train and synthetic signal previews: participant, EO/EC, channel and sample slot; single-channel or stacked channels; transparent selected-window PSD.
- Forty-five full-pool Fidelity metrics, official group and participant PSD, alpha effects, and a distinct matched-count Real–Real calibration.
- Real/B0/B1/B2 dwPLI² heatmaps, signed difference maps and saved diagnostics by eye state, band and participant where available.
- Top-1/Top-5 participant fingerprints, candidate count, rank/margin diagnostics, diversity and exact-copy summaries.
- All eleven utility conditions for three classifier families; 55 saved fits in each split, EEGNet seeds, saved paired intervals, dose comparisons and Validation-versus-Test displays.
- Thirty-three saved EEGNet learning histories with actual selected and stopping epochs; methods, glossary, limitations and collapsed source lineage.
- A ten-step Guided Presentation, with previous/next controls and speaker notes. Every chart can be downloaded as self-contained interactive HTML, with CSV where a numeric table is displayed.

The **Final Test — held-out blocks** view is the utility default. Validation remains explicitly labeled **model selection**. No experiment code is run by the interface.

## Offline snapshot and distribution

- `exports/EEG_Synthetic_Lab_Presentation.html`: a standalone ten-step snapshot. Open it in Edge, Chrome or Firefox without Python or internet. It embeds Plotly and numerical data, includes aggregate results plus one small Train-only single-channel example, and contains no full datasets or checkpoint weights.
- `exports/EEG_Synthetic_Lab.zip`: application source and an aggregate prepared bundle. It excludes `.venv`, caches, local signal preview pools, participant diagnostic arrays, full raw data, checkpoints and fonts. The local workspace edition retains participant exploration.

Rebuild these artifacts explicitly:

```powershell
. .\tools\app_env.ps1
& $LabPython .\tools\export_presentation_html.py
& $LabPython .\tools\build_distribution.py
```

The HTML is a **presentation snapshot**, not the full application. No upload or public deployment is performed.

## Scientific scope

The study evaluates different blocks of the **same 198 participants**, not unseen people or new sessions. Fidelity is descriptive relative to Real Train. Matched-count Real–Real calibration uses disjoint Train blocks and is presented separately from full-pool results.

B0 receives eye state only; its nominal participant groups are evaluation strata. B1 and B2 receive identity and state. B2 is a selected continuation checkpoint with the same architecture, not a new spectral-loss model. Training histories differ. TSTR uses synthetic classifier-training windows while EEGNet checkpoint selection still uses real Validation. Equal epoch limits do not imply equal optimizer-update budgets; Real Duplicate remains visible.

Performance is subject-macro balanced accuracy. Saved EEGNet SD is population SD over three seeds (`ddof=0`); paired subject-bootstrap intervals are a distinct, unadjusted descriptive uncertainty summary. The data support a small B2 augmentation gain for this EEGNet comparison, not universal classifier benefit or conclusive superiority over B1.

Fingerprints use 76 log-bandpower features. In-sample identification and no exact-copy detections do not establish privacy or absence of memorization. Sensor-space dwPLI² is neither anatomical nor causal connectivity; negative estimates and undefined entries remain intact.

## Data isolation and source mapping

Runtime reads only `data/presentation_bundle/` and app-owned assets. Read-only discovery/preparation lives in `lab/sources.py`, `lab/adapters.py` and `tools/prepare_presentation_data.py`. The separate `tools/export_train_previews.py` verifies locked Train provenance before bounded slices; raw Validation and Final-Test waveform arrays are not read. The app has no raw HDF5 dependency.

See `docs/DATA_SCHEMA.md`, `docs/SOURCE_MAPPING.md`, `docs/SCIENTIFIC_QA.md` and `docs/PREVIEW_PROVENANCE.md`. Source paths, SHA256 values, transformations, units, aggregation and split accompany derived evidence. Preparation errors disable affected availability rather than substituting invented observations.

## Verification and troubleshooting

Actual verification results are recorded in `docs/TEST_RESULTS.md`, `docs/OFFLINE_CHECK.md`, and `docs/screenshots/`. Exact installed package versions are in `requirements-tested.txt`; direct requirements are pinned in `requirements*.txt`. Streamlit and Plotly assets are served locally; usage telemetry is disabled.

To run development checks:

```powershell
.\setup.ps1 -Dev
. .\tools\app_env.ps1
& $LabPython -m pytest -q
```

- **Missing environment/packages:** rerun `setup.ps1`; it modifies only this app's environment.
- **Missing bundle:** run `prepare_data.ps1` from the original repository with saved evidence available. The aggregate ZIP already includes a prepared bundle.
- **Preview unavailable:** inspect `docs/PREVIEW_PROVENANCE.md`; the remaining numerical pages still work. Use `prepare_data.ps1 -SkipPreviews` to prepare only numerical evidence and the snapshot.
- **Port occupied:** the launcher picks another port automatically; use the printed address. You can request a starting port with `run.ps1 -Port 8510`.
- **Download requires preparation:** expand the chart download area and press **Prepare interactive HTML**, then download. Plotly is embedded for offline viewing.
- **No participant controls in the ZIP:** this is intentional for the aggregate shared edition. Use the original local app for participant diagnostics and previews.

All generated files belong under this application directory. There is no automatic training, sampling, classifier inference, checkpoint selection, external upload, telemetry, or deletion of scientific artifacts.
