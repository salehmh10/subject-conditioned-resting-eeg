# 1. Executive status

Completed phase: frozen mechanical held-out pilot processing, all-block QC, and subject-streaming preparation.

`MECHANICAL_HELDOUT_PROCESSING_READY_FOR_FULL_STREAMING` means the existing 15-subject pilot has 15 verified all-block outputs, exact 150/90 Train/held-out accounting, frozen apply-only QC, assessed provisional feasibility, tested streaming and deletion guards, committed evidence, and a passing clean-worktree acceptance check. Run section 18 verification against the final local HEAD to certify that terminal state.

The final 211-subject cohort, final allocation/Test lock, numerical normalization, physiological analyses, and model choices remain unfrozen. No non-pilot execution is part of this completed phase.

# 2. Git state

- Starting branch: `main`; starting HEAD: `d951dca79d93d40935007ed3da09aeac49a75349`; initially clean.
- Authorization, implementation, tests and preflight: `887634f16a9637fe2ba91b27dd02f37850d5fd30` (`docs: authorize frozen mechanical held-out processing`).
- Pilot results and small audit evidence: `711c4e537a9466396fa11a7d2181cc31087eda1a` (`data: record pilot all-block QC and feasibility`).
- Final documentation commit: the commit containing this handoff (`docs: hand off all-block streaming phase`). Resolve its exact hash with `git rev-parse HEAD`; a document cannot embed the hash of its own containing commit.
- Final acceptance requires a clean worktree; no existing commit was amended or rewritten, and no Git push was performed.

# 3. Authorization timing

Authorization/code commit `887634f16a9637fe2ba91b27dd02f37850d5fd30` was created at 2026-09-05T11:09:43Z. The first held-out authorization was durably logged at `2026-09-05T11:09:58.426899+00:00`, before its corresponding value I/O. All 90 records reference that same committed code HEAD. The worktree was explicitly verified clean immediately before real execution. No real execution software correction or failed mechanical transaction occurred.

# 4. Frozen preprocessing integrity

Canonical preprocessing SHA256: `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b`. Both `configs/preprocessing/frozen_preprocessing_v1.yaml` and the existing `frozen_preprocessing_v1.sha256` remain byte-identical to the preflight baseline. The checksum file uses `.sha256`, not `.yaml.sha256`.

Filtering, resampling, the 433-sample edge guard, full-montage CAR, ordered channels, artifact thresholds, bad-channel rules, VEOG rule, block eligibility and Train subject QC are unchanged. The fitting functions gained split-provenance refusal checks only; their numerical methods did not change. No normalization scaler was fitted.

# 5. Pilot subject interpretation

This is within-subject block holdout, not disjoint subject splits. Each of 15 subjects has 8 EO and 8 EC blocks: 5 Train / 1 Validation / 2 Test per state. The 150 previously exposed blocks remain Train; every future retained subject must preserve those assignments or be excluded. The 90 remaining blocks received only frozen mechanical processing. No interrupted fragment was read.

# 6. Mechanical access control

Access specification: `configs/access/mechanical_heldout_v1.yaml`; canonical SHA256 `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4`. It pins the split and inventory hashes and the required frozen preprocessing digest, without duplicating preprocessing parameters.

`MechanicalHeldoutAccessGate` joins authoritative identities, checks completeness/study status, refuses interruption fragments, verifies manifests/config, and requires Validation or Test. `MechanicalHeldoutBlockLoader` uses the existing private MNE loader, a one-subject process lock, and exact half-open native sample bounds with a 65-second maximum block. No public raw MNE object is exposed. Authorization is fsynced to CSV before `get_data`.

Allowed purposes: `frozen_preprocessing`, `fixed_waveform_qc`, `compact_storage`, `output_verification`. Unknown purposes and plotting, visualization, PSD, Alpha, IAF, connectivity, wPLI, classifier, generator, TSTR, augmentation, hyperparameter selection, threshold fitting and normalization fitting are denied.

`pilot_mechanical_heldout_access_registry.csv` contains exactly 90 unique records: 30 Validation and 60 Test, all with purpose `frozen_preprocessing`. `pilot_waveform_access_registry.csv` is unchanged at exactly 150 unique Train records. Compact output reopening is mechanical verification; it adds no native waveform read records.

# 7. All-block processing

Ignored output root: `data_processed/MPI-LEMON-all-blocks-v1/`. Fifteen `<subject>_all_blocks.h5` files and adjacent `.provenance.json` receipts contain 240 complete study blocks. All 150 Train datasets were copied without raw Train rereads; arrays, masks, reasons and original block attributes were independently compared against preserved Train HDF5. Ninety held-out blocks were newly processed.

Train thresholds, bad-channel decisions, fit availability, channel order and source provenance were cross-checked between HDF5, receipts, per-subject JSON/CSV and committed pilot aggregates before application. Root provenance records exact Train fit block IDs and a canonical parameter-source hash. All 16 groups remain present even when ineligible.

Files were verified while temporary, fsynced, paired with a durable pending receipt, and atomically promoted. A partial two-file transaction fails closed instead of being silently overwritten. `pilot_all_block_processed_manifest.csv` records every output SHA256/size/config/parameter hash; the verifier reopened all 15 outputs successfully.

# 8. QC results

Engineering masks and block eligibility only. A usable-window mask does not make an ineligible block or subject eligible for scientific training.

| Split | State | Eligible / assigned blocks | Usable / candidate windows | Retention |
| --- | --- | ---: | ---: | ---: |
| Train | EO | 68 / 75 | 1632 / 1875 | 87.04% |
| Train | EC | 75 / 75 | 1858 / 1875 | 99.09% |
| Validation | EO | 13 / 15 | 312 / 375 | 83.20% |
| Validation | EC | 15 / 15 | 371 / 375 | 98.93% |
| Test | EO | 28 / 30 | 653 / 750 | 87.07% |
| Test | EC | 30 / 30 | 743 / 750 | 99.07% |

Train QC passes for 14/15 subjects. Validation has 28/30 eligible blocks and 683/750 usable windows; Test has 58/60 eligible blocks and 1396/1500 usable windows. EO retention is lower than EC under the unchanged pooled Train method. These are engineering limitations to report, not a basis for tuning or reassignment. No PSD, Alpha, topology, Connectivity, classifier result or waveform plot was produced.

# 9. Provisional 5/1/2 feasibility

13/15 subjects are `pilot_all_block_qc_feasible`; two are `pilot_all_block_qc_not_feasible`. All source/header/output checks pass. Counts below are eligible assigned blocks. Validation requires 1/1 per state and Test 2/2 per state; Train also requires the frozen subject criterion (including at least three eligible blocks per state).

| Subject | Train criterion | EO Train | EC Train | EO Val | EC Val | EO Test | EC Test | Feasible | Reason if not feasible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| sub-010023 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010059 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010071 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010078 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010081 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010126 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010134 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010216 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010250 | fail | 1 | 5 | 0 | 1 | 0 | 2 | no | frozen_Train_QC_failed; EO_Validation_assigned_block_ineligible; EO_Test_assigned_block_ineligible |
| sub-010276 | pass | 4 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010277 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010278 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010279 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |
| sub-010280 | pass | 3 | 5 | 0 | 1 | 2 | 2 | no | EO_Validation_assigned_block_ineligible |
| sub-010282 | pass | 5 | 5 | 1 | 1 | 2 | 2 | yes | - |

No held-out spare exists: all three non-Train blocks per state are assigned. No block identity or allocation changed. These results do not select or lock the final 211-subject cohort.

# 10. sub-010250

Its Train failure is preserved: only 1/5 EO Train blocks is eligible, versus 5/5 EC. The assigned EO Validation block and both EO Test blocks are also ineligible; EC Validation and both EC Test blocks pass. All 16 blocks remain in its verified compact output. The original pooled Train thresholds and channel decisions were used without relaxation, refitting or rescue. Its raw data remains retained and cleanup is blocked by the existing Train subject-QC requirement.

# 11. All-block storage

HDF5 contains continuous float32 microvolts, 19 ordered channels, 128 Hz, gzip level 4 + shuffle, channel/sample chunks (19,1024), and window starts/masks/reasons. Windows are 512 samples with stride 256 and never cross blocks; overlapping windows are not materialized. Measured pilot size: **109,736,191 bytes (104.652587 MiB)**. Measured mean times 211 estimates **1,543,622,420.07 bytes (1.437610 GiB)**. This is a planning estimate, excluding receipts, preserved Train copies, temporary caches and filesystem overhead.

The old 68,563,819-byte Train-only outputs remain unchanged as the reference for exposed data and fitted parameters. Their separate preservation is intentional. One subject and one native block are processed at a time, avoiding 125 GiB of simultaneous raw staging.

# 12. Updated streaming workflow

`--processing-scope train-only` preserves the legacy processing option. `--processing-scope all-blocks-mechanical` requires the frozen config, mechanical access config, and `Train_only` fitting. For each future subject: download/resume one source pair; verify payload hashes; copy/verify sidecars; audit BrainVision headers; read Train first; fit subject parameters from Train only; mechanically apply them to held-out; retain all 16 blocks; verify output/provenance; assess provisional feasibility; journal completion; optionally perform guarded cleanup; continue sequentially.

New subjects fit Train through `TrainAccessGate`; the all-block path writes new Train reads to `streaming_train_access_registry.csv`, preserving the original pilot Train registry. Existing pilot Train parameters and data are reused. The held-out registry retains provisional roles and apply-only provenance. The strict pilot verifier is intentionally no longer applicable once future non-pilot waveform processing begins.

Reruns reopen and hash existing all-block outputs and compare them to durable receipts, with no new raw value reads. Source pairs are reverified by the orchestrator. Missing raw after a completed guarded cleanup is skipped only after matching source fingerprints and reverifying output/receipt/journal. Download `.part` files resume through the existing strict HTTP Range implementation. An incomplete processing output, changed source/sidecar/parameter evidence or corrupt receipt stops the subject and preserves raw.

If a software bug is discovered after held-out access, stop execution, document it, invalidate only affected derived outputs, fix and commit code without changing scientific thresholds, then restart with the new recorded HEAD. Do not delete access history or blindly remove lock files. No such correction was needed in this pilot; the Windows fsync correction was caught in synthetic tests before authorization/execution.

# 13. Cleanup status

`pilot_all_block_raw_cleanup_manifest.csv` marks 14 subjects technically eligible for later cleanup. This is storage-completeness eligibility under the existing Train-QC safeguard, not 5/1/2 scientific feasibility: sub-010280 has all blocks preserved despite its failed Validation block. Sub-010250 remains blocked by Train-QC failure. **No pilot raw EEG or raw subject directory was deleted.** All 105 staged files remain byte-identical, totaling 7,013,640,445 bytes.

Deletion requires explicit `--delete-raw-after-success`, a verified 16-block all-block output (Train-only is refused), Train-derived apply-only provenance, passing Train subject QC, source `.eeg`/`.vhdr` hashes, reopened dataset/attribute checks, output hash, durable adjacent receipt, matching complete journal, no unresolved `.part` or processing failure, available metadata sidecars, and a direct confined raw subject directory with no links/reparse points. Raw retention remains the default. Synthetic temporary-directory tests exercised successful guarded deletion and failures; no real pilot cleanup was executed.

# 14. Tests

Final full suite: **208 passed in 43.08 seconds**, with JUnit at `outputs/logs/pytest.xml`. `python -m compileall scripts src tests` passed. The focused mechanical + streaming suite has 74 cases, included in the full result. Tests use synthetic or mocked data, with network requests denied. They cover allowed/forbidden access, exact bounds and log-before-I/O, held-out fitting refusal, unchanged parameter application, Train dataset copy, 16-block output/split/config/geometry verification, corruption detection, dry-run/local-existing streaming, guarded deletion and raw preservation.

Real verification passed for 15 source/header/output sets, 240 block groups, 150 copied Train datasets, and exactly 90 held-out intervals. The idempotent 15-subject rerun completed in 15.875 seconds; all output hashes and both access histories stayed unchanged, with zero new value reads/downloads/deletions. The full dry run planned 211 subjects, found 15 staged pairs and passed all disk checks.

# 15. Leakage audit

- No Train parameter used held-out data; parameter hashes and fit block IDs agree with preserved Train evidence.
- No held-out threshold, bad-channel or normalization fit occurred.
- No scientific held-out computation, signal plot, classifier/generator run, physiology or augmentation analysis occurred.
- No non-pilot EEG was downloaded and no non-pilot subject appears in a real access registry.
- No interruption fragment was accessed and no pilot raw file was deleted.
- Original Train registry, old Train HDF5/receipts, frozen YAML and frozen checksum file are unchanged.
- No metadata-clone write, Git push, amend or history rewrite occurred.

`s2_11_verify_mechanical_pilot.py` checks exact authoritative intervals, source/raw fingerprints, authorization commit timing, output arrays/masks/provenance, restricted purpose/caller strings, fit provenance and a mechanical-path call-graph scan. This is an application-level audit of the executed workflow, not an OS-wide surveillance or security-sandbox claim.

# 16. Repository files

- `configs/access/mechanical_heldout_v1.yaml` and `.sha256`: separate frozen-mechanical access contract.
- `src/resting_eeg/data/mechanical_access.py`: held-out gate and bounded loader.
- `src/resting_eeg/preprocessing/all_blocks.py`: Train evidence verification, unchanged held-out application, combined storage/verification and feasibility.
- `pipeline.py` and `signal.py`: Train split provenance and refusal checks; numerical method unchanged.
- `streaming.py` and `s2_07_stream_preprocess_subjects.py`: explicit processing modes, all-block receipts, sequential lifecycle and stricter cleanup.
- `tests/test_mechanical.py` and `tests/test_streaming.py`: synthetic access/storage/streaming/deletion regressions.
- `scripts/data/s2_11_verify_mechanical_pilot.py`: automated real pilot acceptance and compact summary regeneration.
- `outputs/metrics/scenario2/mechanical_preflight.json`: starting Git state and raw/Train/config preservation hashes.
- `pilot_mechanical_heldout_access_registry.csv`, `pilot_mechanical_access_summary.json`: exact 90-read audit.
- `pilot_all_block_qc.csv`, subject feasibility, processed manifest, QC/storage summaries and cleanup manifest: requested engineering evidence.
- `all_block_streaming_dry_run_summary.json`, `pilot_all_block_idempotence_summary.json`, `pilot_all_block_journal_summary.json`: dry-run/reuse/journal evidence.
- Runtime `streaming_all_block_status.jsonl` remains local and ignored because it contains repeated detailed receipts. Its compact committed summary records its hash, size and per-subject output hashes. Original historical Train journal is preserved.
- README, PROTOCOL, DECISIONS and this handoff: authorization, current results and continuation boundary.

# 17. Open questions

1. What proportion of all 211 structural candidates passes frozen Train QC and assigned held-out mechanical feasibility?
2. Which subjects satisfy the final deterministic cohort/split lock while keeping all 150 exposed pilot blocks Train?
3. What are measured full-cohort storage/runtime distributions and the final unique-sample Train normalization scaler?
4. In separately authorized later phases, what physiology and model/augmentation outcomes emerge under the final lock and analysis plan?

# 18. Exact next action

## NEXT ACTION — FULL SUBJECT-BY-SUBJECT ALL-BLOCK STREAMING

Run from the repository root with the existing `.venv`. Non-pilot and deletion commands below are prepared for the next phase only; none was executed here. All commands use the frozen config and mechanical access config defaults shown explicitly in the dry run. Raw is retained unless deletion is explicitly selected.

1. Full read-only dry run:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --processing-scope all-blocks-mechanical --config configs/preprocessing/frozen_preprocessing_v1.yaml --mechanical-access-config configs/access/mechanical_heldout_v1.yaml --fit-parameters-from Train_only --dry-run --keep-raw
```

2. One non-pilot subject, retaining raw:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --subject-id sub-010002 --execute --keep-raw --retries 3 --timeout 60
```

3. The alternative with guarded raw deletion:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --subject-id sub-010002 --execute --delete-raw-after-success --retries 3 --timeout 60
```

4. Bounded sequential processing:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --max-subjects 5 --execute --continue-on-error --keep-raw --retries 3 --timeout 60
```

5. Full resumable sequential processing with explicit guarded cleanup:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --execute --continue-on-error --delete-raw-after-success --retries 3 --timeout 60
```

6. Verification and restart:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml outputs/logs/pytest.xml
.\.venv\Scripts\python.exe -m compileall scripts src tests
.\.venv\Scripts\python.exe scripts/data/s2_11_verify_mechanical_pilot.py --require-clean --test-report outputs/logs/pytest.xml
git status --short
git rev-parse HEAD
```

The strict acceptance command above is for the current exact-15 pilot state, before any non-pilot staging. To verify pilot reuse without changing the committed journals/access evidence:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --manifest outputs/metrics/scenario2/pilot_download_manifest.csv --execute --local-existing-only --keep-raw --journal outputs/logs/pilot_all_block_restart_verify.jsonl
```

After future full streaming begins, verify each subject through the built-in source/header/all-block/receipt checks. Restart command 5 unchanged for the full sequence; interrupted payloads resume and verified completed outputs are reused. A selected retry can use:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --subject-id sub-010002 --execute --keep-raw --retries 3 --timeout 60
```

Use repeated `--subject-id` or `--subjects-file path/to/subjects.txt` for a selected retry list. If a processing transaction is partial, investigate the journal and preserved `.h5.tmp`/pending receipt first; follow section 12 for any software correction. Never bypass output/provenance verification or remove raw to clear a failure.
