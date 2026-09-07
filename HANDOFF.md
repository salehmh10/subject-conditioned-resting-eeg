# Scenario 2 Train-only normalization handoff

One global 19-channel scaler is fitted and frozen from the 198 included subjects: 1,980 assigned Train blocks, 1,970 eligible and ten frozen-ineligible. Only unique samples covered by usable windows of eligible Train blocks contribute; shared overlaps count once. The ten ineligible assignments remain auditable with zero sample contribution.

Measured volume: **12,881,152 unique samples = 100,634 seconds = 27.95388888888889 hours**. EO contributes 49,460 seconds and EC 51,174. Of 49,500 candidate windows, 47,898 are mask-usable and 47,835 are in eligible fitting blocks. The remaining 63 mask-usable windows contribute zero.

The independent Train-only pass confirms per-channel normalized mean/std within 1e-12: maximum absolute mean 1.8302699640687466e-17 and maximum std deviation 1.9984014443252818e-15. Deterministic refitting reproduces all three artifact bytes exactly.

Read [docs/TRAIN_NORMALIZATION_HANDOFF.md](docs/TRAIN_NORMALIZATION_HANDOFF.md) for channel values, exact accounting/hashes, tests/reviews, limitations and commands. The numbered workflow is s2_18 fit, s2_19 independent verification and s2_20 post-commit acceptance. The exact clean committed HEAD and final acceptance are recorded outside Git at:

`D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/docs/train_normalization_lock_20260907/final_acceptance.json`

All cohort/split, QC, masks, preprocessing, physical-unit HDF5, receipts and prior evidence remain unchanged. No scientific held-out analysis, training, synthetic EEG, Batch 006 or push occurred; Scenario 1 remains read-only and Final Test remains sealed. Stored historical normalization flags remain false because no EEG file was rewritten; the global scaler is separate.

NEXT ACTION — REAL-DATA BASELINES
