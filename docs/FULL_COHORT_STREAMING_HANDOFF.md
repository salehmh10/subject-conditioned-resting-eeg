# Full-cohort streaming final verification and acceptance

Date: 2026-09-07 (local). Scope: terminal engineering verification and a local documentation/results commit only.

## State and commit resolution

All 211 structural candidates are technically verified with 3,376 stored blocks. Batch 005 completed 65/65 subjects and 1,040 blocks, with zero technical failures and ledger `complete`. The preserved terminal heartbeat reports `batch_complete`, project state `BATCH_005_AWAITING_FINAL_ACCEPTANCE`, worker exit 0, and zero unresolved subjects. Worker stderr is empty. The supervisor/worker and their launchers have exited; nonblocking probes acquired and immediately released both existing execution locks without changing their contents.

Authorized execution HEAD, independently confirmed before documentation work:
`3a4965644edbfb16db007275fa9b1a0e232c3900`.

The final documentation/results commit is a new local child of that HEAD. Its exact full hash and final `FULL_COHORT_STREAMING_COMPLETE` status are recorded only after the post-commit clean-worktree verifier passes, in:

`D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/docs/final_acceptance_20260907_resume/final_acceptance.json`

Require that record's `final_committed_head` to equal repository HEAD and its post-commit verification to pass before relying on final accepted status. This document cannot contain its own commit hash. The generated `full_cohort_streaming_completion_audit.json` intentionally retains `FULL_COHORT_STREAMING_AWAITING_FINAL_ACCEPTANCE`; the terminal heartbeat also remains unchanged as processing history.

## Verified technical completion and frozen QC

| Group | Subjects | Blocks | Train QC pass / fail | Provisional feasibility pass / fail |
|---|---:|---:|---:|---:|
| Pilot | 15 | 240 | 14 / 1 | 13 / 2 |
| Batch 001 | 8 | 128 | 7 / 1 | 7 / 1 |
| Batch 002 | 16 | 256 | 15 / 1 | 15 / 1 |
| Batch 003 | 33 | 528 | 33 / 0 | 33 / 0 |
| Batch 004 | 74 | 1,184 | 71 / 3 | 68 / 6 |
| Batch 005 | 65 | 1,040 | 65 / 0 | 62 / 3 |
| Cumulative | 211 | 3,376 | 205 / 6 | 198 / 13 |

All candidates occur exactly once; no missing, duplicate, unfinished, unassigned, or replacement subject exists. No Batch 006 exists. Four structural exclusions remain unchanged: sub-010015, sub-010100, sub-010203, and sub-010285.

The six Train-QC failures, whose verified raw directories remain, are sub-010012, sub-010030, sub-010146, sub-010163, sub-010234, and sub-010250. The 13 provisional-feasibility failures are sub-010012, sub-010030, sub-010146, sub-010157, sub-010163, sub-010200, sub-010226, sub-010234, sub-010250, sub-010260, sub-010280, sub-010291, and sub-010316. Exact reasons and every subject's output/receipt/Train-reference hashes are in the committed cumulative subject manifest. These outcomes do not lock a final scientific cohort.

All 65 Batch 005 subjects pass Train QC, so their raw directories are absent after verified cleanup, including its three provisionally infeasible subjects. All 205 cumulatively deleted raw directories are absent; only the six required retained subjects remain. No unresolved production `.part` files or downloads exist. Synthetic pytest partial fixtures are test evidence and remain outside production staging; previous fixtures were archived before the exact-path test rerun.

## Network and storage accounting

Batch 005 contains 130 source assets in frozen authoritative order from sub-010242 through sub-010321. Declared payload, exact cap, expected new bytes, and measured response-body bytes are all **42,976,220,335**; unused capacity is zero. Validated resumed prefixes total **2,254,055,363 bytes**. There were **7 recorded network-wait episodes**, totaling **195.829 seconds** through recorded recoveries. These are transport events, not technical subject failures. Measurement excludes HTTP/TLS/ISP overhead.

| Storage metric | Bytes |
|---|---:|
| Batch 005 all-block HDF5 addition | 474,963,578 |
| Batch 005 raw bytes freed, including sidecars | 42,979,046,146 |
| Cumulative raw bytes freed | 130,919,680,737 |
| Cumulative retained raw | 3,902,615,061 |
| Cumulative all-block HDF5 | 1,541,179,379 |
| Cumulative Train-reference HDF5 | 962,901,478 |
| Cumulative provenance receipts | 6,448,433 |
| Total processed storage | 2,510,529,290 |

Signal, runtime, temporary, and cache data remain under `D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/`. The metadata clone at `D:/SHARIF/PROJECT/Datasets/MPI-LEMON` remains clean at `16702890507291b5a09fa7390245ec60bff6128e`.

## Acceptance checks and evidence

The exact required offline pytest command passed **383 tests, zero failures/errors/skips, in 47.11 seconds** (JUnit duration 47.087 seconds). Compileall exited 0. The Batch 005 read-only verifier passed with all 65 outputs/receipts, 1,040 blocks, durable terminal/cleanup ordering, measured traffic, access registries, frozen hashes, and prior evidence checked. The cumulative read-only audit and subsequent `s2_15 --write-results` both exited 0, verifying 211 all-block containers/receipts, source/sidecar identities, Train-reference provenance, all 770 protected prior files, retained raw hashes, deleted-directory absence, coverage, and metadata preservation. No signal payloads are tracked in Git.

The evidence directory above contains terminal rediscovery, environment reports, stdout/stderr and exit-code files for pytest, compileall, the Batch 005 verifier, the cumulative audit, result writing, and post-commit verification. `pytest.xml` is preserved externally and also available locally at `outputs/logs/pytest.xml`; SHA256 is `fc45e56c7261783f39f88b091034d6419af02fe8b4d4c8b3187ef618f579384b`. `preserved_execution_evidence.json` records 37 execution/handoff/history artifacts for final byte-preservation comparison.

The earlier invocation-path error is preserved at `runtime/docs/final_acceptance_20260907/acceptance_failure.json` under the external data root. It occurred before test collection, was not a processing failure, and was resolved by the user's instruction to run the required exact path. No tests or guards were modified. The previous pytest directory and report are archived as `prior_pytest/`, `prior_pytest_manifest.json`, and `prior_pytest.xml` in the resumed evidence directory.

The final clean verifier uses:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 005 --require-clean --test-report outputs/logs/pytest.xml
```

This is read-only engineering verification; never invoke a supervisor or worker to test completion. No scientific held-out analysis or native preprocessing was run during acceptance.

## Frozen identities

Configuration semantic hashes and exact file-byte hashes are distinct:

| Binding | SHA256 |
|---|---|
| Preprocessing semantic configuration | `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b` |
| Mechanical-access semantic configuration | `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4` |
| Preprocessing YAML file bytes | `da57489746e66fa819da16217e585cdd0ca3b4e5cbe4491603e831be98889110` |
| Mechanical-access YAML file bytes | `4fbf4106c7f42957325cd12ba66294fd2d1a43e435e5f7282a318e8102976053` |
| Variable-batching YAML file bytes | `41d53f288786880d9a92ea202a558abe0fa5030fb8a3a872c275e959e6048ced` |
| Provisional split manifest file | `6adc10b108a7e7b587757cbda630a3c4909c97f8cc2faf24320b0a6e89af0b26` |
| Block inventory file | `d5deff471d02baa55ac60539986595b82594d5137fcf1bffac6a5ec982b28742` |
| Full candidate download manifest file | `dd66de86255dda86ac96bcdd6fc2a52bdfcaaee26afc5bd7ecb3e55893b8cc05` |
| Batch 005 manifest file | `0eb72ed7be098c63abf35838efa16618751a0348bfda97a8e98e21cfce81f002` |
| Batch 005 plan file | `7ae47d1db50e84985af604e677265a690a245cc3d8cb986cdd7eb247a6c772dd` |
| Batch 005 execution authorization file | `0c0d9f46e8a2ac216d5f7e651182ce01de3a8375796ea1424879be9f2376595a` |

The committed `full_cohort_streaming_subject_manifest.csv`, `full_cohort_streaming_qc_summary.json`, `full_cohort_streaming_storage_summary.json`, and `full_cohort_streaming_completion_audit.json` are the compact cumulative results. Frozen Batch 005 plan, coverage, authorization, summary, cleanup, and output evidence remain alongside them in `outputs/metrics/scenario2/batches/`.

## Preserved boundaries and next phase

The historical external `runtime/docs/HANDOFF.md` and all earlier handoffs remain unchanged. Repository-level [HANDOFF.md](../HANDOFF.md) is now safe to materialize because Batch 005 is successfully terminal and no continuation is pending. Earlier no-successor assertions describe their own historical assignments; use the Batch 005 verifier for current cumulative verification.

No preprocessing, QC, cohort, split, or access rule changed. Fitting remains Train-only, held-out application remains mechanical, all 16 blocks and masks/reasons remain stored, the 150 previously exposed pilot blocks remain Train unless their subject is later excluded, and the split remains `PROVISIONAL_METADATA_ONLY`. Normalization is unfitted. No scientific held-out analysis, waveform plotting, final cohort/split lock, Batch 006, previous-commit amendment, or Git push occurred in this acceptance phase.

NEXT ACTION — FINAL COHORT AND SPLIT LOCK
