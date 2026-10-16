# Installation and verification scope

This repository contains preserved research source, selected saved evidence, and publication utilities. Packaging does not change the scientific implementation. The source environments recorded Python **3.12.3**; the new package metadata requires Python **3.12 or later**. Availability and compatibility of the recorded scientific dependencies on other Python versions or operating systems have not been validated for this publication export.

**No dependency installation, scientific execution, experiment, or test suite was run when preparing this export.** The release workflow checks copied file hashes and publication metadata using the Python standard library. A passing workflow establishes copy integrity, not numerical reproducibility or scientific correctness.

## Verify preserved copies without installing dependencies

From the repository root, with Python 3.12 available:

```console
python tools/verify_copy_integrity.py --root .
```

The verifier checks the publication copies against `provenance/source_inventory.json` and `evidence/provenance/evidence_inventory.json`. It does not need the original private working directory, load participant waveforms, import the research package, or rerun any analysis. Frozen originals retain their recorded bytes and line endings.

## Choose a separate environment for each use

Create a virtual environment, activate it, and install only the group needed. For example, on Windows PowerShell:

```powershell
py -3.12 -m venv .venv-results
.\.venv-results\Scripts\Activate.ps1
python -m pip install -e ".[results]"
```

On Linux or macOS:

```sh
python3.12 -m venv .venv-results
. .venv-results/bin/activate
python -m pip install -e '.[results]'
```

Run these commands from the repository root. Use `.venv-app` for the `app` group and `.venv-research` for the `research` group. Do not combine `app` and `research`: their NumPy versions deliberately reflect different original environments. `dev` adds pytest only; it does not supply scientific or app dependencies.

| Group | Purpose | Scope |
| --- | --- | --- |
| Base editable install | Make the existing `resting_eeg` package importable | NumPy, Pandas and PyYAML; not the full research environment |
| `results` | Work with saved aggregate tables and plots | Base dependencies and Matplotlib |
| `app` | Run the presentation application from its prepared bundle | Original app pins: Streamlit 1.50.0, Plotly 6.3.1, NumPy 2.3.3, Pandas 2.3.3 |
| `research` | Install the recorded direct scientific dependencies | Includes MNE, SciPy, H5Py, scikit-learn, pyRiemann and PyTorch; external data and artifacts are still required |
| `dev` | Make pytest available for separately planned development | Existing research tests are retained unchanged and are not run by release CI |

The installation commands in this document are provided for users. They have not been executed as part of the publication packaging work.

## Saved results and presentation

In the results environment, the preserved plotting utility is available at `results/tools/plot_utility.py` with its accompanying saved tables. Consult its source and the results documentation for its inputs and output paths before execution. Plotting saved values is distinct from rerunning training or evaluating Final Test.

In a separate app environment, install and launch from the repository root:

```console
python -m pip install -e ".[app]"
python -m streamlit run apps/eeg_synthetic_lab/app.py
```

The app uses the included prepared presentation bundle. Its source documentation describes the original local workspace as well as the shared aggregate edition; participant-level controls and waveform previews depend on the actual bundle supplied. Original preparation tools may require external research evidence. Merely launching the prepared app does not require rebuilding that evidence or running scientific models.

## Research dependency snapshot and PyTorch build

The original files `requirements.txt`, `requirements-lock.txt`, and `requirements_current_exact.txt` are preserved. The first two describe an earlier, smaller preprocessing environment. The full recorded snapshot includes `torch==2.11.0+cu128`; it is an environment record, not a platform-independent lock file. The `research` extra lists the recorded direct dependency versions without replacing those snapshots.

In a dedicated research environment, select the PyTorch build first. For a CPU installation:

```console
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[research]"
```

For the recorded CUDA 12.8 family, on a supported NVIDIA system:

```console
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[research]"
```

The installed PyTorch local build suffix can satisfy the `torch==2.11.0` dependency. Wheel availability, driver compatibility, and CPU execution of the historical experiments have not been verified here. Some preserved entrypoints explicitly require CUDA. The complete original dependency snapshot remains the reference for the recorded environment; the new extras are convenience metadata and do not establish an identical transitive installation.

## Checkout and data-path requirements

Use an editable checkout when working with the original code. The distributable Python package includes only `src/resting_eeg`; it excludes datasets, saved outputs, checkpoints, the app, and chronological experiment scripts. Some package functions import `scripts.data` or locate configuration relative to the repository. Run them from the checkout root; a standalone wheel is not a complete experiment distribution.

`configs/paths/publication_data_paths.example.yaml` shows the schema supported by `resting_eeg.data.paths.load_paths`. To configure functions that use that resolver, copy the example to `configs/paths/local_data_paths.yaml` and adjust it, or select an absolute override file:

```powershell
$env:SCENARIO2_PATH_CONFIG = 'D:\path\to\local_data_paths.yaml'
```

```sh
export SCENARIO2_PATH_CONFIG=/absolute/path/to/local_data_paths.yaml
```

Writable raw, processed, temporary, and runtime roots must be distinct directories within the configured external data root and must not overlap the repository or metadata clone. The example is not applied automatically.

This override does not retrofit the frozen scripts. Several historical baseline scripts hardcode the original `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2` path, and the original Windows environment helper requires a D: temporary root. Those limitations remain visible because scientific source bytes have been preserved. Full execution requires the original expected layout or a separately documented, validated adaptation; this publication export does not claim general portable execution.

## Original tests and frozen Final Test

The original `tests/`, `pytest.ini`, and app tests are retained as source evidence. The root test configuration requires an explicit external `--basetemp`, configures external runtime directories, and contains D:-specific checks. Some tests also require historical output manifests. Installing the `dev` extra does not remove these requirements.

Release CI deliberately runs neither the root tests nor the app tests. It does not train a model, sample a generator, rebuild synthetic pools, select a checkpoint, download the dataset, or reopen the study's frozen Final Test. Any future full reproduction must be planned separately with its required artifacts and resource budget, and must preserve the distinction between original frozen results and a new reproduction run.
