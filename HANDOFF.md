# Scenario 2 final streaming handoff

Batch 005 is successfully terminal: 65/65 technically complete subjects, 1,040 blocks, worker exit 0, empty stderr, ledger complete, no unresolved subjects/downloads/production partials. Its supervisor/worker and launchers have exited and both execution locks are free. **Do not launch or restart any worker/supervisor, rerun preprocessing, or create Batch 006.**

The cumulative engineering audit verifies **211/211 subjects, 3,376 blocks, and all 770 protected prior files**. Train QC passes 205 subjects and fails six; provisional feasibility passes 198 and fails 13. Six prior Train-QC-failure raw subjects remain preserved. The final scientific cohort and split are still unlocked, and normalization is unfitted.

Read [docs/FULL_COHORT_STREAMING_HANDOFF.md](docs/FULL_COHORT_STREAMING_HANDOFF.md) for exact counts, exclusions, identities, byte accounting, validation commands, evidence paths, and phase boundaries. Final offline tests passed 383/383; compileall, the Batch 005 read-only verifier, cumulative audit, and compact result writing passed.

The authorized execution HEAD was `3a4965644edbfb16db007275fa9b1a0e232c3900`. The exact subsequent local documentation/results commit and final accepted status are resolved by:

`D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/docs/final_acceptance_20260907_resume/final_acceptance.json`

Confirm `final_committed_head` matches repository HEAD and the recorded post-commit `--require-clean` verifier passed before relying on `FULL_COHORT_STREAMING_COMPLETE`. The compact audit intentionally remains `FULL_COHORT_STREAMING_AWAITING_FINAL_ACCEPTANCE`, and the preserved worker heartbeat remains `BATCH_005_AWAITING_FINAL_ACCEPTANCE`; neither historical artifact should be rewritten to advertise final acceptance.

This root handoff was materialized only after successful terminal rediscovery. The previous live-run handoff at external `runtime/docs/HANDOFF.md`, its snapshots, all runtime journals/logs/locks, prior output evidence, and the earlier pytest path-mismatch failure JSON remain preserved. The exact required pytest command passed after existing fixtures/report were archived; no tests or guards were changed.

Final Cohort and Split Lock is a separate phase and has not begun. Preserve the frozen preprocessing/QC/cohort/split/access rules and provisional block assignments. Scientific held-out analysis, normalization fitting, worker execution, Batch 006, and Git push were not authorized by this acceptance task. Do not amend previous commits or push.

NEXT ACTION — FINAL COHORT AND SPLIT LOCK
