# Reproducibility and preserved evidence

This publication package separates inspection of completed results from execution of the original research pipeline. Publication packaging did not rerun training, sampling, Test evaluation, waveform analysis, bootstrap resampling, or the historical test suite. The supplementary Validation analysis had already completed; this update archives its aggregate evidence and supplies a separately named portable derivative.

## Verify saved outputs in a fresh clone

Python 3.12 with its standard library is sufficient. No package installation, EEG, checkpoints, synthetic pools, external paths, or network connection is needed after cloning:

```console
python -B tools/verify_copy_integrity.py --root .
python -B tools/verify_validation_evidence.py --root .
```

The first command checks the previously published source/evidence inventories. The second checks the new packaged hashes and schemas, cohort/window counts, top-1/top-5 count-to-percentage arithmetic, finite values and interval ordering, and correspondence with recorded Table II/III and nearest-window results. It also checks that original Train-reference and utility outcomes remain unchanged. Preserved reports of the original participant-level reproduction checks are labeled as archived assertions.

The Validation verifier locates the checkout relative to its script by default; an explicit root can also be used from another working directory or a path with spaces:

```console
python -B "/path with spaces/repository/tools/verify_validation_evidence.py" --root "/path with spaces/repository"
```

Aggregate-only verification cannot independently recompute participant-paired bootstrap intervals, signal features, frozen selections, or nearest-window distances. A passing result establishes consistency of the packaged evidence, not full scientific replication. The post hoc utility contrast versus duplication is labeled separately where its interval is available only as a manuscript-reported value.

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
| Exploratory Validation aggregate summary and receipt | `evidence/validation_reference/summary.json`, `completion.json` |
| Public Validation metadata and paper-table mapping | `evidence/validation_reference/public_metadata.json`, `reported_results.json` |
| Preserved original supplementary analysis | `evidence/validation_reference/original_source/analyze_round3.py` |
| Validation publication inventory and lineage | `evidence/validation_reference/publication_inventory.json` |
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

## Portable supplementary analysis and environment

The [portable derivative guide](PORTABLE_VALIDATION_ANALYSIS.md) documents the explicit configuration for processed HDF5 storage, split and normalization files, SC and SSC/SSC-C pool directories and manifests, original fidelity tables and aggregate input, and a separate output directory. Frozen-plan paths require an explicit documented mapping, including nested inventories and selections. Original hashes retain their meaning after path resolution. The derivative must not silently rewrite the frozen plan, reuse incompatible participant caches, or overwrite preserved results. Only Train and Validation waveform indexing is permitted.

The original supplementary analysis ran with **Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, and h5py 3.16.0**. Its separate dependency list is [requirements-validation-analysis.txt](../requirements-validation-analysis.txt); PyTorch, MNE, and GPU access are unnecessary for this analysis. Help/configuration and saved-output checks use separate/lazy imports and do not need those scientific dependencies. Full external-data execution of the portable derivative remains **untested in this update**; it needs the external artifacts and frozen plan that are deliberately absent from the public aggregate subset.

```console
python -B tools/analyze_validation_portable.py --help
```

See the derivative guide for configuration-only validation and future external-data execution commands. Such execution is a separate reproduction activity; no fresh analysis or full bootstrap rerun was performed during this publication update.

## Environment groups

The root `pyproject.toml` separates optional dependencies for `results`, `app`, `validation-analysis`, `research`, and `dev`. Select the group appropriate to the intended work; reading CSV files or running the saved-output verifiers does not require any group. For example, in a chosen virtual environment:

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

The original supplementary script SHA-256 is `418892ffff6e31f92f38112da3506369581b8e555935a6889c4397c3ae4f70b6`. Its summary and completion receipt are also preserved byte-for-byte, with hashes and destination paths in the [Validation publication inventory](../evidence/validation_reference/publication_inventory.json). That inventory distinguishes the derivative's own hash and changes from the original executed source. Original external-artifact hash references remain recorded even when the underlying files are not distributed. Public extracts do not contain the original plan, participant caches/rows, window indices, absolute-path inventories, or waveform examples.

The archived original script intentionally retains its original absolute path constants as source provenance; use the separately named derivative for portable configuration. Retaining that source does not publish the environment-specific inventories or external files to which those constants refer.

Do not overwrite the preserved Final Test outputs or use them to choose new models. New experiments, prediction regeneration, bootstrap analyses, or portability validation require their own clearly labeled artifacts and protocol. Their results must not be presented as if they were the original frozen experiment.

Historical test files are included for source completeness. Their presence, or previously recorded pass counts in historical notes, does not mean they were rerun for this publication release. Distinguish packaging integrity from scientific replication in any future validation record.
