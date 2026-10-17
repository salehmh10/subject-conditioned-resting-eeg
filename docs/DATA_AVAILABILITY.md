# Data and artifact availability

The study uses [MPI-LEMON, Babayan et al. (2019)](https://www.nature.com/articles/sdata2018308), resting-state EEG. The official [NEMAR distribution metadata](https://github.com/nemarDatasets/nm000179/blob/main/dataset_description.json) lists CC BY 4.0 and dataset DOI [10.82901/nemar.nm000179](https://doi.org/10.82901/nemar.nm000179). Obtain recordings through the official distribution and retain its required attribution. The dataset's CC BY 4.0 terms apply to the dataset and do not grant a license to this repository's code.

## Included

- Original scientific source, configuration, and historical research tests.
- Retained cohort, block, and QC metadata, including public pseudonymous study identifiers under `outputs/metrics/`.
- Curated frozen selection/protocol records and saved final reports.
- Aggregate and fit-level utility tables, saved learning histories, and representative fidelity tables.
- Prepared figures and an aggregate presentation bundle.
- A provenance inventory linking publication copies to original artifact paths and hashes.
- The completed exploratory Validation-reference analysis's original source, aggregate summary and completion receipt, plus explicitly labeled public metadata/table extracts and a separate portable derivative.

Some preserved reports can contain pseudonymous study identifiers and participant-level numerical descriptors. They are research evidence, not raw EEG, and should not be represented as anonymous data with a proven reidentification guarantee.

That description concerns the pre-existing historical package. The newly added `evidence/validation_reference/` subset is **aggregate-only**: it does not add participant result rows or signal examples. Its [publication inventory](../evidence/validation_reference/publication_inventory.json) records the source lineage, publication destinations, and hashes. Hash references to undistributed original inputs are retained as provenance, not advertised as downloadable artifacts.

## Not distributed as study datasets

Raw EEG, processed HDF5 datasets, normalized waveform caches, native participant-level signal preview pools, model checkpoints, and complete synthetic EEG pools are not included in this publication repository. The aggregate presentation is not a replacement for those research artifacts.

Saved paths in frozen records can reference files outside the public package. Such a reference records provenance; it is not a download endpoint or a promise that the artifact is bundled. The [evidence inventory](../evidence/provenance/evidence_inventory.json) identifies the curated publication copies.

The supplementary analysis's `participants/`, `per_participant.csv`, `fingerprint_per_participant.csv`, `nearest_per_stratum.csv`, original `plan.json`, waveform JSONs, window indices, and absolute-path inventories are excluded. The final manuscript and its waveform illustrations are not copied or changed by this update. No checkpoint or complete synthetic-pool publication is implied.

## Access and reconstruction

Use the original [path configuration](../configs/paths/data_paths.yaml) and [reproducibility notes](REPRODUCIBILITY.md) to understand external storage. The original workflow stores large Scenario 2 data outside Git. A metadata clone, original acquisition files, frozen preprocessing and cohort evidence, and stage-specific model artifacts are needed for full-study execution.

Automatic downloads, training, or evaluation are not part of browsing the results or launching the aggregate app. Additional artifact distribution would require an explicit release with documented rights, provenance, and any applicable access conditions. Checkpoint and full synthetic-pool download endpoints are not provided in this package.

The [saved-output verification commands](REPRODUCIBILITY.md#verify-saved-outputs-in-a-fresh-clone) run offline in a fresh clone with Python's standard library. They do not need EEG, pools, checkpoints, or original absolute paths. They validate aggregate consistency and retained hashes, but cannot independently recalculate signal features or paired bootstrap uncertainty.

Full execution of the [portable Validation derivative](PORTABLE_VALIDATION_ANALYSIS.md) additionally requires the original processed HDF5 files, frozen split and normalization, all three frozen synthetic pools and manifests, original fidelity inputs, and the original frozen plan with its selections. The explicit configuration maps those external inputs to their current locations and validates their original content hashes. These inputs are not reconstructed from public aggregate summaries. Full external-data execution remains untested in this publication update; the original code and completed results remain unchanged.
