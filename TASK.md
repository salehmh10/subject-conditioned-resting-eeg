# Target
Deterministically derive, verify, commit, and externally accept the final Scenario 2 cohort and unchanged block-level Train/Validation/Test lock.

# Context
Starting branch `main` at accepted HEAD `77a6c3745408acc7c40260794a731123dc43c93b`; full-cohort streaming is complete for 211 structural candidates and 3,376 blocks.

# Assumptions
Frozen QC, provisional assignments, access rules, and source evidence are immutable. No EEG signal arrays, normalization fitting, model training, scientific held-out analysis, Batch 006, or push is authorized.

# Done Criteria
Required deterministic artifacts, tests, independent reviews, guarded full verification, one local commit, clean worktree, and external acceptance record all pass with 198 included subjects and unchanged 5/1/2 assignments.

# Current Plan
Create the single authorized local commit, reverify its clean state, and write the durable external acceptance record.

# Completed Steps
Read required instructions/evidence; verified every start invariant; implemented the deterministic generator, verifier, acceptance gate, and 15 focused tests; generated the four lock artifacts twice byte-identically; archived the prior report; and completed three independent reviews.

# Current State
The final-lock candidate contains 215 subject rows and 3,168 unchanged assignments for 198 included subjects. All pre-commit artifacts, documentation, reviews, tests, compilation, and verification pass; no normalization, training, scientific held-out analysis, Batch 006, or push occurred.

# Open Questions
None.

# Blockers
None.

# Verification Results
Start invariants pass; full guarded suite passes 398/398, including 15/15 final-lock tests; compileall and the pre-commit verifier pass. Three independent reviews pass with every valid finding resolved. Regeneration is byte-identical; counts are 1,980/396/792, pilot accounting is 130 included Train plus 20 excluded, 10 ineligible Train blocks remain unchanged, all 211 normalization flags are false, and no `eeg_uv` samples were read for the lock.

# Next Step
Commit once, run both clean-state verifiers, and write `final_acceptance.json` outside Git.
