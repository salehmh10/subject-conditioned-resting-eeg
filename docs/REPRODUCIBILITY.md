# Reproducibility and preserved evidence

This publication package separates inspection of completed results from execution of the original research pipeline. Publication packaging did not rerun training, sampling, evaluation, bootstrap resampling, or the historical test suite.

## Inspect the completed results

No environment setup is needed to read [CSV tables](../results/tables/), [prepared figures](../results/figures/), the [results narrative](RESULTS.md), or the [saved reports](../evidence/). The [evidence inventory](../evidence/provenance/evidence_inventory.json) records the original and publication locations with hashes.

The authoritative final utility experiment is `s2_75B-v2`, followed by the frozen pretest record `s2_76` and single Final Test evaluation `s2_77`. Earlier utility runs are development history. Final reports describe 55 fits across 11 conditions and three classifier families; the EEGNet classifier has three seeds per condition.

Useful entry points:

| Artifact | Publication path |
|---|---|
| Full Test report | `evidence/utility/test/s2_77_v2_complete_final_test_report.json` |
| All 55 Test fits | `results/tables/utility_test_all_55_fits.csv` |
| All 55 Validation fits | `results/tables/utility_validation_all_55_fits.csv` |
| Original paired Test contrasts | `evidence/utility/test/tables/frozen_paired_subject_contrasts.csv` |
| B0/B1/B2 fidelity report | `evidence/fidelity/b0_extension/s2_74e_b0_complete_report_v1.json` |
| Original B1/B2 fidelity report | `evidence/fidelity/b1_b2/s2_74b_complete_report_v1.json` |
| Frozen records | `evidence/frozen/scenario2/` |

The presentation app is another view of precomputed evidence. Follow its [README](../apps/eeg_synthetic_lab/README.md) for installation and launch. The public aggregate edition does not need the original EEG datasets to present its prepared bundle. Rebuilding that bundle from original research outputs is a separate operation.

## Redraw saved aggregate results

The publication copy includes the original aggregate plotting tools and their saved inputs. From the repository root, in a suitable Python environment:

```text
python -m pip install ".[results]"
python results/tools/plot_utility.py
python results/tools/build_fidelity_figures.py
```

These commands redraw figures from `results/tables/` and `results/fidelity_data.json`; they do not train a model or access EEG waveforms. They write generated figures and, where implemented by the original plotting tool, display tables under `results/`. The supplied figures are already available, so running these commands is optional. They were not rerun during publication preparation.

Do not add `--extract` to the exported fidelity tool: that mode expects the original research output layout. This public copy uses its portable saved data by default. Original figure/table source maps are retained under `results/provenance/`; the separate evidence inventory maps curated scientific records.

## Environment groups

The root `pyproject.toml` separates optional dependencies for `results`, `app`, `research`, and `dev`. Select the group appropriate to the intended work; reading CSV files does not require any group. For example, in a chosen virtual environment:

```text
python -m pip install ".[app]"
```

An installable dependency group is not a guarantee that historical experiment commands run unchanged. Research execution also depends on external artifacts, original runtime details, and frozen identity checks. Retained dependency records describe the research history; a new installation is not evidence of numerical equivalence.

The aggregate app can be launched after installing the `app` group with:

```text
python -m streamlit run apps/eeg_synthetic_lab/app.py
```

## Original full-study environment

The original workflow was developed on Windows/PowerShell with a repository-local virtual environment and external dataset storage. `configs/paths/data_paths.yaml` resolves its default metadata and data roots relative to the repository: `../Datasets/MPI-LEMON` and `../Datasets/MPI-LEMON-Scenario2`. A different clone location therefore changes the resolved defaults. Configure an ignored `configs/paths/local_data_paths.yaml` or `SCENARIO2_PATH_CONFIG` for the actual storage layout.

In a restored original-style environment, the historical shell initialization is:

```powershell
. .\scripts\setup_scenario2_environment.ps1
```

This initializer expects `.venv\Scripts\python.exe` and sets project cache/temp paths according to the path policy. It is environment setup, not a command that reproduces results. Some later scripts and frozen records additionally contain original absolute paths or require exact source/artifact hashes. Changing a path override alone does not make every historical stage portable.

A full reproduction requires the original metadata, eligible processed blocks and masks, normalization, model lineage, checkpoints, synthetic pools, and stage-specific artifacts. These are not all distributed here. Consult [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and [PIPELINE.md](PIPELINE.md) before preparing a separate research run. No single uninterrupted clone-to-paper execution path is claimed by this release.

## Preserve scientific lineage

Frozen manifests bind code and artifact bytes. Retain their original hashes and provenance when relocating files. Publication paths are mapped by a separate inventory; historical absolute paths inside evidence are not silently rewritten. Any portable adaptation belongs in an explicit derivative with its own provenance and an explanation of how it differs.

Do not overwrite the preserved Final Test outputs or use them to choose new models. New experiments, prediction regeneration, bootstrap analyses, or portability validation require their own clearly labeled artifacts and protocol. Their results must not be presented as if they were the original frozen experiment.

Historical test files are included for source completeness. Their presence, or previously recorded pass counts in historical notes, does not mean they were rerun for this publication release. Distinguish packaging integrity from scientific replication in any future validation record.
