# Attribution and third-party materials

No license for the original project code has been selected or granted in this
initial public snapshot. The maintainer has deferred that decision. The EEG
dataset, dependencies and cited publications retain their own licenses.

## MPI-LEMON / NEMAR dataset

This study uses **LEMON: MPI Leipzig Mind-Brain-Body EEG (Resting State)**,
NEMAR dataset **nm000179**, version 1.0.0 in the research metadata snapshot.
The [upstream dataset description](https://github.com/nemarDatasets/nm000179/blob/main/dataset_description.json)
declares **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
See the [license](https://creativecommons.org/licenses/by/4.0/) and
[dataset DOI](https://doi.org/10.82901/nemar.nm000179).

Please cite Babayan, A. et al. (2019), *A mind-brain-body dataset of MRI, EEG,
cognition, emotion, and peripheral physiology in young and old adults*,
Scientific Data 6, 180308. [DOI: 10.1038/sdata.2018.308](https://doi.org/10.1038/sdata.2018.308).

Original EEG recordings are not redistributed here. Research manifests retain
the dataset's public pseudonymous participant identifiers and the project's
derived split/QC decisions. These are research-derived records, not upstream
measurements or the dataset authors' conclusions. Cite the dataset as well as
the software when using these derived records.

## Dependencies and references

Python, NumPy, SciPy, pandas, MNE-Python, PyTorch, scikit-learn, pyRiemann,
Matplotlib, Streamlit, Plotly and other dependencies are installed separately
and retain their respective licenses. No dependency installation, proprietary
font, third-party paper PDF, LaTeX distribution, or private runtime is bundled.

The presentation bundle comes from the existing aggregate distribution. No
waveform preview or standalone HTML with embedded recordings is included.
Scientific references are in [references.bib](results/provenance/references.bib).
