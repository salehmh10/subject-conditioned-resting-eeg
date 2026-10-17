# Initial public snapshot — release notes draft

Initial curated public snapshot of the subject-conditioned resting-state EEG
research code and preserved final results, based on MPI-LEMON/NEMAR nm000179.

- Original scientific source, staged experiment scripts and research history.
- Final v2 utility evidence: 11 conditions, three classifier families, 55 fits.
- Selected B0/B1/B2 provenance and aggregate fidelity evidence.
- All-condition result tables, saved figures and an aggregate local presentation.
- Source/evidence SHA256 inventories, installation metadata and study documentation.
- Copy-integrity CI that does not execute research code or tests.

The subsequent final-manuscript synchronization adds the completed exploratory
Validation-reference analysis as an aggregate-only evidence subset. Its original
script, summary and completion receipt retain their bytes and hashes. A separately
named portable derivative, explicit input mapping, independent provenance,
recorded analysis dependencies and offline saved-output verifier accompany it.
Participant records, the original plan, window selections and waveform examples
are excluded. The paper and its final figures remain unchanged.

The study evaluates new recording blocks of the same 198 participants. Historical
B0/B1/B2 identifiers correspond to SC/SSC/SSC-C in the paper. Selected B2 has the
same architecture and v-prediction objective as B1 and zero spectral-loss weight. Its full-dose
EEGNet improvement is small and classifier-dependent; B2 superiority over B1 is
not established. Validation-reference spectral errors remain lower for the
subject-conditioned models, while identification remains below real block-to-block
retrieval and does not improve uniformly with continuation. Validation was held
out from gradient training but used in development and selection, so the
supplementary comparison is post hoc and selection-dependent, not independent
confirmation. The full limitations remain part of the released documentation.

Raw/processed EEG, synthetic pools, checkpoints, participant-level presentation
diagnostics, dependency installations and duplicate build products are excluded.
The data distribution and original research dependency snapshot are documented.

Per the author's instruction, publication preparation did not rerun scientific
analysis, historical research tests, training, sampling, inference or bootstrap
estimation. The supplementary update adds checks of help/configuration and
saved aggregate evidence, including hashes, schemas, arithmetic and reported
paper values. These do not independently recompute features or paired bootstrap
intervals. Full execution of the new derivative with external data remains
untested, and full-study execution retains its external artifact requirements.

The original project code is now covered by the [MIT License](../LICENSE).
Citation author metadata, CITATION.cff, a formal versioned release and DOI
archiving remain deferred at the maintainer's request. The package's proposed
1.0.0 metadata does not imply a published GitHub release. For now, reference the
repository URL and exact commit. The dataset DOI is a separate citation and
must not be presented as this software's DOI.
