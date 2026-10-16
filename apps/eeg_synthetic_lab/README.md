# EEG Synthetic Lab — aggregate publication edition

An English interactive presentation of saved final results. The included bundle
is the pre-existing aggregate distribution: no participant diagnostic arrays,
waveform previews, raw EEG, checkpoints or live scientific computation.

From the repository root, use a dedicated environment:

```console
python -m venv .venv-app
# Activate .venv-app using your platform's activation command.
python -m pip install -e ".[app]"
python -m streamlit run apps/eeg_synthetic_lab/app.py
```

Original Windows `setup.ps1` and `run.ps1` launchers remain available within
this directory and use an app-local environment. Normal launch reads the
prepared aggregate bundle. **Do not run prepare_data.ps1 in the publication
copy**: it expects the original external research artifacts and is not needed
to display the included results.

Explore group PSD, alpha effects, connectivity, fingerprint summaries, all 11
utility conditions and 55 saved fits, and learning histories. Participant and
waveform controls are unavailable. Final Test means held-out blocks of the
same 198 participants. B2 is a selected continuation with zero spectral weight.

Original source, tests, schema and scientific notes are preserved. Historical
test reports describe earlier app development; they were not rerun for release.
The standalone HTML is omitted because it contains a real waveform example.
See the [historical local-edition README](../../archive/historical/PRESENTATION_README.md)
and the study [limitations](../../docs/LIMITATIONS.md).
