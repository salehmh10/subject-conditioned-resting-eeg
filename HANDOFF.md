# Scenario 2 final cohort and split lock handoff

The final scientific cohort and its unchanged block-level split are locked from frozen, outcome-blind evidence: **198 included subjects**, **13 post-download exclusions**, and **4 earlier structural exclusions**, accounting for all 215 dataset subjects exactly once.

The final assignment contains **1,980 Train, 396 Validation, and 792 Test blocks** (3,168 total), exactly 5/1/2 per eye state and included subject. Every role matches `provisional_block_split_manifest.csv`. Ten ineligible assigned Train blocks remain assigned and explicitly marked ineligible; no block was moved, promoted, rescued, or replaced.

Pilot exposure passes exactly: all 150 exposed blocks are either one of 130 retained Train blocks for included subjects or one of 20 blocks belonging to excluded `sub-010250` and `sub-010280`. No exposed block is Validation or Test.

Read [docs/FINAL_COHORT_SPLIT_HANDOFF.md](docs/FINAL_COHORT_SPLIT_HANDOFF.md) for exclusions, source/artifact hashes, verifier commands, reviewer resolutions, evidence paths, and scientific boundaries. The exact final local commit and clean post-commit acceptance are recorded outside Git at:

`D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/docs/final_cohort_split_lock_20260907/final_acceptance.json`

The accepted full-cohort streaming evidence and all historical raw/HDF5/receipt/runtime records remain unchanged. No Batch 006 exists, no signal payload is tracked in Git, no push occurred, normalization remains unfitted, and Final Test remains sealed.

NEXT ACTION â€” TRAIN-ONLY NORMALIZATION
