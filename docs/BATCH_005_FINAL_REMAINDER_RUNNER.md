# Batch 005 final remainder streaming

The user explicitly authorizes all 65 still-unassigned structurally eligible subjects as one final remainder batch. This supersedes the previous assignment's Batch 005 prohibition. No Batch 006 is permitted. The clean starting HEAD was `db42b774d70d2ad058a7c02c03e9a99b0fc1f9cf`. Before changes, the existing Batch 004 read-only verifier passed with 74/74 complete subjects and 1,184 blocks, and all 146 cumulative HDF5 outputs and receipts were reverified without native waveform reprocessing. Train QC passes 140, and provisional feasibility passes 136. A 770-file baseline protects earlier evidence and all six intentionally retained raw subjects.

The planner derives every remaining complete `.eeg`/`.vhdr` pair from the authoritative full candidate manifest, excluding verified completions and previous assignments. It reconciles 65 subjects / 130 assets, sub-010242 through the final authoritative candidate sub-010321, with exactly 42,976,220,335 bytes. The cap equals that computed total; unused capacity is zero. No rounded GiB cutoff, bin packing, skipped subject, replacement, or future batch is allowed. Coverage reconciles all 211 candidates exactly once. Structural exclusions and previously retained complete subjects cannot enter this batch.

The plan includes its manifest hash. Exact canonical plan and plan-file hashes are bound in the immutable ledger and coverage audit, then in the authorization: a file cannot contain its own whole-file SHA256. The plan identifies this binding explicitly. Frozen membership, order, cap, inventory, and hashes remain immutable. The variable batching configuration and both scientific hashes remain unchanged.

The exact payload is a network budget. Dynamic disk checks reserve only the current subject's remaining payload, 512 MiB preprocessing, 32 MiB compact output, and 1 GiB safety margin. Retained raw is already deducted from actual free disk. Only the current subject and technically completed QC-failure raw directories may exist in staging. C: must have at least 3 GiB before expensive operations. All signal/temp/cache/runtime paths use external D:, with the environment bootstrap before Python commands. No pagefile changes are made.

The existing scientific streaming pipeline fits subject parameters from provisional Train only, applies the frozen method mechanically to held-out blocks, and stores all 16 blocks with masks/reasons. Source identity, complete hashes, HDF5 reopen, receipts, Train reference provenance, durable completion and Train QC gate immediate raw cleanup. Train-QC failures retain raw; provisional held-out infeasibility alone does not block valid cleanup. Earlier retained subjects sub-010012, sub-010030, sub-010146, sub-010163, sub-010234, and sub-010250 remain untouched.

The generalized supervisor retains the tested Batch 004 transport policy. DNS, timeout/stall/reset/disconnect, temporary TLS, and transient HTTP failures wait indefinitely on the same subject and asset. Fsynced partials survive. Recovery uses two tiny exact-object checks five seconds apart; resumption requires HTTP 206 and exact Content-Range. Ignored Range quarantines the prefix and performs a controlled full restart; final full-file size/SHA256 remains mandatory. Backoff is 15/30/60/120/300 seconds with small jitter, capped at 300 except a larger Retry-After. Retry-After is honored for every transient HTTP response. Persistent identity, unsafe path, disk, filesystem or provenance failures preserve evidence and report `FINAL_REMAINDER_HARD_BLOCKED`.

The supervisor uses the OS single-instance lock and a per-run token to recognize the actual Windows interpreter behind the venv launcher. It runs one scientific worker, independently of the launch shell, with external PID/lock/stdout/stderr and 30-second heartbeat files prefixed `batch_005_autoresume`. No service or scheduled task is installed. Progress is durable after every subject, archived every ten terminals and at completion. It includes membership, completed/unresolved IDs, QC/feasibility, measured response bytes, reused prefixes, outages, storage and disk space.

After offline tests and the clean local implementation commit, durable authorization is written before the first real request:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_14_autonomous_batch_supervisor.py --batch-id 005 --authorize-batch
.\scripts\run_batch_005_autoresume.ps1
```

To request a graceful pause, the user may create `D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/batches/STOP_BATCH_005`. The active chunk is finished/fsynced, its response closes safely, and the worker exits `manually_paused` with partials and journals intact. The runner never creates this file automatically during ordinary operation. Ctrl+C uses the same safe pause semantics. Resume uses the same frozen batch and authorization; transient recovery requires no new prompt.

Processing completion is `BATCH_005_AWAITING_FINAL_ACCEPTANCE`. Only after 65 new / 211 cumulative verified outputs, 3,376 blocks, zero partials/unassigned candidates, final offline tests, cumulative preservation and registry verification, final documentation, local results commit and clean worktree may the task report `FULL_COHORT_STREAMING_COMPLETE`.

Scientific held-out analysis, waveform plots, numerical normalization, final cohort/split lock, threshold changes and Git push remain prohibited. The next phase after acceptance is final cohort and split lock under the already frozen outcome-blind QC rules, requiring a separate task.

Pre-network validation: 383 offline tests passed in 71.37 seconds; compileall passed. Six legacy Retry-After assertions were updated to match the authorized behavior before the passing rerun. No real EEG request occurred during tests. Exact report hash and test coverage are in batch_005_pre_network_validation.json.
