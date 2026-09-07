# Target
Fit, independently verify, freeze, locally commit, and externally accept one global 19-channel scaler from eligible final Real Train unique usable samples.

# Context
Starting main HEAD `9fa49dba1fda8fd0e37852ccdce642372aecca70`, clean and externally accepted. Final cohort/split: 198 subjects; 1,980 Train / 396 Validation / 792 Test. Ten Train blocks are frozen-ineligible; 1,970 are eligible.

# Assumptions
Frozen cohort, split, preprocessing, QC, masks, HDF5 and receipts remain immutable. Only eligible Train EEG is read; opaque container hashing and existing metadata preservation verification remain permitted. No training, held-out scientific analysis, Batch 006, or push.

# Done Criteria
One finite global channelwise float64 mean/population std (ddof=0; std >= 1e-6 uV), auditable 1,980 blocks, deduplicated samples, exact volume accounting, independent normalized second pass, deterministic artifacts, focused/full tests, compileall, two independent reviews, all preservation verifiers, current docs, one local non-merge commit, clean post-commit verification and external acceptance.

# Current Plan
The tracked package is complete. Resolve the terminal Git/acceptance state through the canonical external final_acceptance.json; commit once and run clean post-commit verifiers before s2_20 if acceptance is not yet present.

# Completed Steps
All starting invariants and clean final-lock verifier passed. Scenario 1 organization reviewed read-only. Implemented s2_18 fitting, s2_19 independent verification, s2_20 external acceptance, reusable numerical/identity functions and 28 focused tests. Generated all three artifacts and verified independent coverage/statistics plus byte-identical refit. Documentation and exact volume/scaler tables written.

Scenario 1 organizational pattern: numbered responsibilities, visible high-level stages, explicit paths/parameters, intermediate artifacts and concise checks. Adopted in s2_18/s2_19/s2_20; no dependency or scientific code imported. Its README reproduction section is a placeholder; Scenario 2 explicitly documents execution order.

# Current State
Scaler fitted and independently verified. Exactly 12,881,152 unique samples = 100,634 seconds (27.95388888888889 hours), EO/EC 49,460/51,174 seconds. 49,500 candidate, 47,898 mask-usable and 47,835 fitting windows; 63 usable windows in ineligible blocks contribute zero. No source files or held-out signals changed/read by fitting. Both independent reviews and all pre-commit preservation verifiers pass. The exact final committed HEAD and clean post-commit state are intentionally recorded externally at runtime/docs/train_normalization_lock_20260907/final_acceptance.json.

# Open Questions
None.

# Blockers
None.

# Verification Results
Full guarded suite: 426 passed in 42.98 seconds (28 normalization/acceptance cases). Previous run: 421 passed. Review found a tuple-return API integration issue, repaired and covered by a committed-metadata-only loader test; production metadata smoke passed. Verification script now exposes its stages directly. Independent max absolute z mean = 1.8302699640687466e-17; max population z std deviation = 1.9984014443252818e-15, tolerance 1e-12. All 1,980 block audit rows match independent bitmap coverage and deterministic artifacts match byte-for-byte. Compileall, previous full-cohort preservation verifier, final-cohort verifier and both reviewer cycles pass with no unresolved findings. C: guard and exact D: pytest base remain unchanged.

# Next Step
Read the canonical external acceptance record and compare its final_committed_head with clean HEAD. If it matches and status is TRAIN_NORMALIZATION_LOCKED, this phase is complete; the next separately authorized phase is REAL-DATA BASELINES. If absent, create the single local commit only while HEAD is the starting commit, then rerun all three verifiers with --require-clean and run s2_20; do not recommit or push. Limits: max_iterations=10, max_retries_per_error=2, max_reviewer_cycles=2; implementation iteration=3 and both review cycles completed.
