# Data and artifact availability

The study uses [MPI-LEMON, Babayan et al. (2019)](https://www.nature.com/articles/sdata2018308), resting-state EEG. The official [NEMAR distribution metadata](https://github.com/nemarDatasets/nm000179/blob/main/dataset_description.json) lists CC BY 4.0 and dataset DOI [10.82901/nemar.nm000179](https://doi.org/10.82901/nemar.nm000179). Obtain recordings through the official distribution and retain its required attribution. Dataset licensing is separate from the license of this repository's code.

## Included

- Original scientific source, configuration, and historical research tests.
- Retained cohort, block, and QC metadata, including public pseudonymous study identifiers under `outputs/metrics/`.
- Curated frozen selection/protocol records and saved final reports.
- Aggregate and fit-level utility tables, saved learning histories, and representative fidelity tables.
- Prepared figures and an aggregate presentation bundle.
- A provenance inventory linking publication copies to original artifact paths and hashes.

Some preserved reports can contain pseudonymous study identifiers and participant-level numerical descriptors. They are research evidence, not raw EEG, and should not be represented as anonymous data with a proven reidentification guarantee.

## Not distributed as study datasets

Raw EEG, processed HDF5 datasets, normalized waveform caches, native participant-level signal preview pools, model checkpoints, and complete synthetic EEG pools are not included in this publication repository. The aggregate presentation is not a replacement for those research artifacts.

Saved paths in frozen records can reference files outside the public package. Such a reference records provenance; it is not a download endpoint or a promise that the artifact is bundled. The [evidence inventory](../evidence/provenance/evidence_inventory.json) identifies the curated publication copies.

## Access and reconstruction

Use the original [path configuration](../configs/paths/data_paths.yaml) and [reproducibility notes](REPRODUCIBILITY.md) to understand external storage. The original workflow stores large Scenario 2 data outside Git. A metadata clone, original acquisition files, frozen preprocessing and cohort evidence, and stage-specific model artifacts are needed for full-study execution.

Automatic downloads, training, or evaluation are not part of browsing the results or launching the aggregate app. Additional artifact distribution would require an explicit release with documented rights, provenance, and any applicable access conditions. Checkpoint and full synthetic-pool download endpoints are not provided in this package.
