# Publication provenance

This release is curated from research commit
`e797bb533c08498271d3fd4ac56fe3c12b6ef44e` plus previously untracked completed
modelling, evaluation and presentation sources. The original research history
is retained. The publication commit records the later files for the first time.

`source_inventory.json` maps each preserved copy to its original relative path,
size and SHA256. A source containing `!` denotes a member of an already-existing
ZIP distribution; the source archive itself is not bundled.
`../evidence/provenance/evidence_inventory.json` records the selected compact
scientific evidence. New publication documentation and packaging are identified
by Git rather than represented as original research files.

Original code, frozen manifests, documents and source maps can contain the
original machine's absolute paths. These are historical labels and checksum-bound
evidence, not portable locations. See the installation and reproducibility guides.

The publisher requested that existing research code and tests not be rerun.
This release therefore makes no claim of a new training run, Final-Test
evaluation, installation test or repeated scientific test suite. Publication
checks cover file-copy integrity, release contents and metadata. Historical
verification records retain their original scope and dates.

The optional standard-library copy check reads file bytes without importing
research code, opening EEG/checkpoints or recalculating results:

```console
python tools/verify_copy_integrity.py --root .
```
