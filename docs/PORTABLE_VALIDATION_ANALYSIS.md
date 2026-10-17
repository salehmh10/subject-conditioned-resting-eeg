# Portable exploratory Validation-reference analysis

The [portable derivative](../tools/analyze_validation_portable.py) is a separate
implementation adapted from the [unchanged archived source](../evidence/validation_reference/original_source/analyze_round3.py).
The archived SHA-256 is
`418892ffff6e31f92f38112da3506369581b8e555935a6889c4397c3ae4f70b6`.
The publication inventory records a distinct hash for the derivative. The original
plan, original summary and original completion hashes continue to identify their
original bytes; the derivative never represents its outputs as those artifacts.

This analysis is exploratory and post hoc. Validation blocks were excluded from
gradient training but used in development and checkpoint selection. Its 198
participants also occur in Train; these results do not demonstrate generalization
to new participants or independent confirmatory validation. Historical identifiers
remain B0=SC, B1=SSC and B2=SSC-C. SSC-C continues SSC with the same architecture
and v-prediction objective, with spectral-loss weight zero.

## Inspect the public evidence without external data

Use the saved-output verifier, which has only standard-library dependencies:

```text
python -B -S tools/verify_validation_evidence.py --root .
```

This validates the packaged aggregates and their reported correspondence. It does
not independently compute signal features, participant vectors, paired bootstraps,
or selected waveform indices. The original private plan and participant records
are deliberately not included. A fresh clone cannot perform full scientific
replication from the public aggregate subset.

## Explicit local configuration

Copy [the example configuration](../configs/validation_analysis.example.json) to
a private location and update its paths. Every relative path is resolved against
the configuration file's directory, independently of the shell working directory.
Moving the configuration file requires updating relative paths. The example
assumes it remains under `configs/`; all its external inputs and its new output
are outside the publication checkout. It is an example, not a runnable bundled
dataset. Never add a filled private configuration, plan, generated participant
records, or waveforms to Git.

| Configuration field | Required artifact |
|---|---|
| `processed_hdf5_dir` | Original processed HDF5 files, named `<subject_id>_all_blocks.h5` |
| `split_manifest` | Original block split CSV, including eligible Train/Validation counts and HDF5 hashes |
| `normalization` | Original Train-only normalization JSON and 19-channel order |
| `pool_dirs.B0`, `.B1`, `.B2` | Complete original SC, SSC and SSC-C synthetic NPZ pools |
| `pool_manifests.B0` | Original SC pool manifest |
| `pool_manifests.B1` | Original shared SSC/SSC-C manifest; its `committed` entries include both `B1/…` and `B2/…` |
| `original_fidelity_tables` | Original private `subject_state_metrics.csv` and `subject_alpha_effect.csv` |
| `original_aggregate_fidelity` | Original aggregate fidelity JSON; the byte-identical public copy is `results/fidelity_data.json` |
| `original_fidelity_scripts.B0`, `.B1_B2` | Byte-identical original fidelity-analysis source files |
| `original_script` | Byte-identical archived round3 source, not the derivative |
| `frozen_plan`, `frozen_plan_sha256` | Optional private original plan and its independently recorded digest |
| `output_dir` | New external directory that does not already exist or overlap inputs |

The aggregate fidelity JSON is an explicit provenance-bound input. The original
algorithm reads participant fidelity CSVs for numerical comparisons; it recorded
the aggregate JSON's hash as evidence provenance. The derivative retains that
distinction rather than substituting rounded aggregate values for participant
data. The example private plan digest is the digest referenced by the preserved
original completion record:
`0be340b11198078bdbcb0d5db71acef4042a8c2a6f7c9a92e64648fa2a2e5b43`.

Check configuration without NumPy, SciPy, h5py, waveform access or output writes:

```text
python -B -S tools/analyze_validation_portable.py --help
python -B -S tools/analyze_validation_portable.py --config /path/to/private/config.json --check-config
```

`--check-config` validates required metadata schemas and original source identity.
With a frozen plan it also verifies the plan digest, immutable settings, cohort,
counts, selected-index bounds, metadata content hashes, and relocated external
file existence/byte sizes. It does not open HDF5/NPZ bytes or validate waveform
content hashes. Those hashes are checked immediately before their inputs are read
by an explicit full analysis. Passing configuration checks is not a claim that
full execution has been validated.

## Resolve frozen paths without changing the plan

Historical path strings are retained only as labels, never opened as filesystem
locations. No search of old drives or automatic root substitution occurs. The
mapping is determined by configured inputs and scientific identity:

| Frozen location | Resolution and validation |
|---|---|
| `source_hashes[*]` | SHA-256 identity matches a declared metadata/source input; every required metadata role must be bound. Unknown hashes fail. |
| `real_file_inventory[*]` | `subject_id` maps to `<processed_hdf5_dir>/<subject_id>_all_blocks.h5`; expected SHA-256 must match split metadata, and byte counts must match. |
| `selections[subject/state].synthetic[model]` | Model and subject/state map to the corresponding configured manifest entry and pool directory; frozen/manifest hashes and expected counts must agree, and byte counts must match. |

Absolute roots and filenames embedded in the old plan need not exist. Pool
manifest file entries must remain relative and cannot traverse outside their
configured pool directories. Conflicting mappings fail. Metadata files are
hashed during configuration validation; HDF5/NPZ bytes are hashed only during
explicit scientific execution. Only `Train` and `Validation` are permitted for
HDF5 waveform indexing; the split manifest may contain `Test` metadata.

`--prepare` copies a supplied frozen plan byte-for-byte to a new output directory.
It does not rewrite nested paths or change the original digest. A separate local
`portable_run.json` records its resolved mapping, original/derivative identities,
configuration digest and run state. This mapping contains private environment and
participant information and is not a public artifact. If both optional frozen
plan fields are omitted, `--prepare` creates a separately identified new plan
using the original metadata-only selections and seeds; it does not claim the new
plan is byte-identical to the archived plan.

## Full external-data execution: not tested in this publication update

The recorded analysis environment is **Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
and h5py 3.16.0**. No PyTorch, MNE or GPU is needed. The derivative checks these
versions before scientific execution and sets the original four BLAS-related
thread environment variables to two before importing scientific libraries.
Install the dedicated requirements in an isolated environment with Python 3.12.3:

```text
python -m pip install -r requirements-validation-analysis.txt
```

Only after obtaining all external artifacts, checking their original provenance,
and choosing a fresh external output directory, a future authorized execution is:

```text
python -B tools/analyze_validation_portable.py --config /path/to/private/config.json --prepare
python -B tools/analyze_validation_portable.py --config /path/to/private/config.json --run
```

No scientific execution, waveform read, new analysis, training, generation, Test
evaluation or bootstrap rerun was performed for this publication update. Tests
exercise only metadata, path binding, command-line behavior and failure handling.
The full external-data path remains untested as a relocated execution.

The output must be outside the publication repository and outside every input
directory. `--prepare` refuses existing outputs. `--run` accepts only a directory
containing its unchanged `plan.json` and derivative-owned `PREPARED` state record,
with the same configuration and derivative hashes. Existing participant caches,
results, unexpected files, interrupted runs and completed runs are refused. There
is deliberately no resume or overwrite option: use a different output directory
after a failure. A full run creates private participant tables, cache-shaped
records and illustrative waveform JSON locally; none belongs in the public
aggregate release, and the script does not modify the manuscript or its figures.

## Changes from the original source

1. Replace machine-specific constants with explicit configuration and content-
   verified mapping of all nested frozen locations. Include the original aggregate
   fidelity input and both reference source scripts as explicit provenance inputs.
2. Move scientific imports behind explicit analysis operations. Help,
   configuration checks and the separate aggregate verifier work with the Python
   standard library alone.
3. Add strict configuration/plan checks and a separate run-state record; preserve
   frozen plan bytes. Refuse input/output overlap and all reuse of participant
   caches, replacing the original script's automatic participant resumption.
4. Resolve the two waveform-path lookups in `analyze_subject` through the validated
   map. All subsequent hash checks and Train/Validation-only indexing are retained.
5. For a newly prepared derivative plan, record both original and derivative
   source identities among its inputs; completed derivative outputs identify the
   derivative script. The original published summary/completion are not changed.

The functions for correlations, Welch/feature extraction, distance metrics,
nearest-window RMSE, real-window reading and final aggregation are retained from
the original source. Seeds 2026101702 (stratum selections), 20261017 (illustration)
and 2026101703 (bootstrap), frozen indices, float32 storage/float64 computation,
inclusive frequency bands, pessimistic fingerprint ties, equal participant/state
weighting and 5,000 participant bootstrap resamples are preserved. The scientific
limitations described in [LIMITATIONS.md](LIMITATIONS.md) remain applicable.
