# 1. Executive status

Batch 001 execution is complete and stopped. `BATCH_001_STREAMING_COMPLETE` means the frozen 5-GiB batch has terminal records for every member, verified 16-block outputs, authorized Train-only fitting and held-out apply-only processing, guarded cleanup/retention, measured network accounting, passing tests and verifier, a complete handoff, and clean locally committed Git state. Run the section 15 clean-worktree verifier against the final handoff commit to certify the terminal state.

All eight subjects completed technically. Seven pass frozen Train QC and provisional 5/1/2 feasibility; sub-010012 fails both and retains raw. There are no unresolved technical failures. No replacement subject or Batch 002 was planned or executed. Scientific held-out outcomes remain sealed; normalization remains unfitted; the final cohort and splits remain unlocked.

# 2. Git state

- Starting `main` HEAD: `dacf69f1caa259a8cf9c39e248a4311b12ca9edb`; starting worktree clean.
- Variable batching implementation and external migration commit: `13cd91df96357d42dfc2a7284ac9382050878a13`.
- D-only environment and C: safety-guard hardening commit: `5a9210ea2cbf85f205c518d5abb39ed208d7d575`.
- Results/documentation commit: `dd04e8bb753d91f77b7c2a11fa68d27b98071eeb`.
- Final local HEAD: the commit containing this handoff, resolved by `git rev-parse HEAD`. Its exact hash and final clean-worktree result are recorded after commit in external `runtime/batches/batch_001_final_acceptance.json`; a committed file cannot embed its own containing commit hash.
- All intended source/config/test/small-evidence/documentation changes are local commits. No push, amend, history rewrite, signal-file commit, or credentials change occurred.

# 3. Variable batch policy

Each future batch requires a separately selected explicit user cap before freeze. Caps may differ, including 5, 7 or 10 GiB. One GiB is exactly 1,073,741,824 bytes; Batch 001 cap is exactly **5,368,709,120 bytes**. Frozen membership, order, cap and manifest hash cannot change.

Membership accounts for each complete declared EEG/header pair, independent of local files or partial bytes. Verified all-block completions and subjects already assigned to earlier frozen batches are excluded. The planner preserves authoritative candidate order and stops at the first subject that does not fit; it performs no bin-packing and does not pull replacements after failure. No future membership was pre-generated. Batch 002 cap remains deliberately **UNDECIDED**.

# 4. Pilot cleanup

After migration and complete reverification, deleted: sub-010023, sub-010059, sub-010071, sub-010078, sub-010081, sub-010126, sub-010134, sub-010216, sub-010276, sub-010277, sub-010278, sub-010279, sub-010280, sub-010282.

Retained: **sub-010250**, due to its unchanged frozen Train-QC failure. Exactly **6,362,309,563 bytes (5.925362523 GiB)** were freed from 14 directories. All 15 Pilot all-block outputs and Train reference outputs remain intact. Sub-010280 was technically safe to delete despite assigned Validation infeasibility: all 16 blocks/masks/reasons, Train QC, receipts and durable completion evidence passed. Cleanup did not equate held-out feasibility with storage safety.

Evidence: `pilot_cleanup_execution.csv`, `pilot_cleanup_summary.json`, and external `pilot_cleanup_journal.jsonl`. Source hashes, BrainVision metadata, all-block output/receipt, Train provenance, journal persistence, path confinement and absence of unresolved partials were rechecked before each deletion.

# 5. Batch 001 membership

| Order | Subject | Declared bytes | Assets |
| --- | --- | ---: | ---: |
| 1 | sub-010002 | 633,641,759 | 2 |
| 2 | sub-010003 | 624,924,559 | 2 |
| 3 | sub-010004 | 633,666,559 | 2 |
| 4 | sub-010005 | 679,868,959 | 2 |
| 5 | sub-010006 | 625,358,559 | 2 |
| 6 | sub-010007 | 634,063,359 | 2 |
| 7 | sub-010010 | 624,974,159 | 2 |
| 8 | sub-010012 | 648,385,359 | 2 |

Total: **5,104,883,272 bytes (4.754293031990528 GiB)**, 16 assets, eight subjects. Unused capacity: **263,825,848 bytes (0.2457069680094719 GiB)**.

The first excluded next subject is **sub-010016**, with 625,866,959 declared bytes; reason: `next_subject_would_exceed_batch_cap`. It remains unassigned, not a planned Batch 002 member.

Frozen manifest SHA256: `21a5676eb5fcefbe5f776bf467d9eace8e3d63ee0ede9d102c6f5bcdf43fb865`. Canonical plan SHA256: `bc875d3808a29aa7733b28cd0fb70c25b7b21ec65e70648b3ca75ab910579166`. Manifest, plan and original authorization retained their exact hashes through interruption and resume; ledger status/timestamps advanced to complete without changing frozen fields.

# 6. Batch 001 authorization

The implementation/migration was committed before the first network request. Membership froze at **2026-09-05T13:36:43.761301+00:00**. Original authorization at **2026-09-05T13:37:04.903750+00:00** references implementation HEAD `13cd91d...`; first request began afterward. Original authorization file SHA256: `b842c7245d2a376f6cf10467980b0362e5f49d4ee597cdfffc6228680470018e`.

The separate resume authorization at **2026-09-05T14:36:39.315025+00:00** references committed hardening HEAD `5a9210e...` and the unchanged original authorization hash. Resume authorization SHA256: `3bc80539703d078c1f2f71140275263038395f8e4db067f937288c012cda7907`. It reverified sub-010002 and authorized the same remaining members only. It recorded 4,051,811,113 expected new bytes after subtracting the completed canary and preserved partial from the remaining payload. Both authorizations prohibit future batch planning/execution within this assignment and record resolved storage paths.

Frozen preprocessing canonical SHA256: `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b`.

Frozen mechanical access canonical SHA256: `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4`.

Variable batching YAML byte SHA256: `41d53f288786880d9a92ea202a558abe0fa5030fb8a3a872c275e959e6048ced`. The scientific configuration byte hashes also match their preflight baseline. The original authorization, resume authorization, network preflight and ledger remain small repository evidence.

# 7. Execution

| Subject | Terminal processing status | Train QC / cleanup | 5/1/2 feasible |
| --- | --- | --- | --- |
| sub-010002 | complete, 16 verified blocks | pass; raw deleted | yes |
| sub-010003 | complete, 16 verified blocks | pass; raw deleted | yes |
| sub-010004 | complete, 16 verified blocks | pass; raw deleted | yes |
| sub-010005 | complete, 16 verified blocks | pass; raw deleted | yes |
| sub-010006 | complete, 16 verified blocks | pass; raw deleted | yes |
| sub-010007 | complete, 16 verified blocks | pass; raw deleted | yes |
| sub-010010 | complete, 16 verified blocks | pass; raw deleted | yes |
| sub-010012 | complete, 16 verified blocks | fail; retained | no |

For each subject: verify/resume one source pair, verify sidecars/header/channels/rate, fit from ten provisional Train blocks only, mechanically apply to six held-out blocks, retain every block including ineligible masks/reasons, reopen/hash the compact output, persist receipt and completion journal, then perform guarded cleanup or record retention before the next subject starts.

Sub-010002 passed the first-subject canary checks, including 16 blocks, Train-only fitting, apply-only provenance, output hash, durable journal, raw deletion and unchanged frozen hashes. The hardened rerun reverified and skipped it without redownload, raw recreation or regeneration; its original canary record and output hashes remain unchanged.

# 8. Internet accounting

- Initial expected new response payload: **5,104,883,272 bytes / 4.754293031990528 GiB**.
- Actual measured response-body payload across both runs: **5,104,883,272 bytes / 4.754293031990528 GiB**.
- Initially verified existing raw payload / initial partial prefix: **0 / 0 bytes**. These are initial preflight quantities, not resume-time quantities.
- Resume-time local prefix reused: **419,430,400 bytes / 400 MiB**. HTTP 206 `Content-Range: bytes 419430400-624922799/624922800` was validated.
- Newly measured payload after that preserved prefix and completed canary: **4,051,811,113 bytes**.
- Network retries: **0**. User-initiated interruptions: **1**. No server-ignored Range restart occurred.

Measurement counts bytes returned by the application's response reader, including retry bodies where present. HTTP/TLS/ISP overhead and bytes not returned to the reader during interruption are not measured; expected values are never labelled as actual traffic. Prefix identity, length and local hash were recorded before resume; the full source SHA256 was verified after completion.

# 9. Preprocessing

The frozen numerical method is unchanged: filters, resampling, 433-sample edge guard, ordered 19 channels, bad-channel metrics/thresholds, interpolation, full-montage CAR, artifact and VEOG rules, block eligibility, Train subject-QC criterion, HDF5 representation and mechanical access restrictions. Outputs remain continuous float32 microvolts at 128 Hz with the frozen compression/chunking and masks/reasons. Provisional block identities and assignments were not changed. No normalization parameters were fitted.

# 10. QC

Batch Train QC passes for 7/8, and provisional 5/1/2 feasibility passes for 7/8. Sub-010012 has eligible EO Train **1/5**, EO Validation **0/1**, EO Test **1/2**; EC Train **5/5**, EC Validation **1/1**, EC Test **2/2**. Its exact reasons are `frozen_Train_QC_failed;EO_Validation_assigned_block_ineligible;EO_Test_assigned_block_ineligible`. No threshold relaxation, rescue or reassignment occurred. These are engineering masks/eligibility only, not scientific held-out outcomes.

# 11. Cleanup

Batch raw deleted: sub-010002, sub-010003, sub-010004, sub-010005, sub-010006, sub-010007, sub-010010. Total freed: **4,456,805,158 bytes** including sidecars.

Batch raw retained: **sub-010012**, **648,429,341 bytes**, blocker `frozen_Train_QC_failed`. Its technically complete output remains preserved. Pilot sub-010250 also remains retained. Both retained raw source pairs were independently rehashed after execution. No unresolved `.part` remains in real raw staging. Every deleted directory is absent, and HDF5 verification and durable completion records precede each deletion.

# 12. Compact outputs

Eight added all-block HDF5 files, **128 block groups**, **58,508,245 bytes (55.797810555 MiB)**. All reopen and match their adjacent durable receipts; all expected datasets, attributes, channel order, 128-Hz rate and Train/apply-only provenance pass.

| Subject | HDF5 bytes | HDF5 SHA256 |
| --- | ---: | --- |
| sub-010002 | 7,382,961 | `adbd9ac7ba876b028803de01dfd38f58c8062ab1e6e8137d06e7855a7b383da6` |
| sub-010003 | 7,268,893 | `17833ac308dc61e893c9dae4a3927f94b6ea1cf49c1bcbc962438e35d04f3da9` |
| sub-010004 | 7,310,733 | `9f43a71db5a12d0142e153016f29348051afad4a2892085de4712a3434354bf9` |
| sub-010005 | 7,214,475 | `2ae037ce8ed09afbfdc2fb08c09d7253c6e0471f382b2748ca9f0d8ff02ed3b8` |
| sub-010006 | 7,347,949 | `ce325e7929dad1ccbf8bc4bb0433b91ca9e104ad9382ccbdd7eebd4388a55669` |
| sub-010007 | 7,342,738 | `2a93668fd5809c616c71fb009626f0abeaf94e7f1c3e00d9d4481bb6beda05b2` |
| sub-010010 | 7,307,849 | `8d6a132044e98032fb4c31543a94e8fd0c2bedbf2deb193da3a02c3d1c304e8e` |
| sub-010012 | 7,332,647 | `0f95225c0d55c449376232fc3bbf2f1a31dc3a46dc7ef1ca005e08c79118f463` |

`batch_001_output_manifest.csv` also records full external paths and receipt hashes. Runtime download journal: **337,951 bytes**, SHA256 `6f30d1e2a35c58ecba9c75bdc70f18ca830aae49641fb116dde9f0b48753050e`. Runtime streaming journal: **588,429 bytes**, SHA256 `8e5c833511f3c25748709be1176848fb8654f0988aa11a5cc68838ba0b1c4b98`. Detailed journals stay external; compact summaries/hashes are committed.

# 13. Cumulative project progress

Pilot + Batch 001: **23 verified all-block subjects / 368 groups**, comprising 230 Train and 138 mechanically processed held-out blocks. Cumulative Train-QC pass: **21/23**; provisional 5/1/2 feasible: **20/23**. These counts do not lock a final cohort.

All-block HDF5: **168,244,436 bytes**. Preserved Train HDF5 references: **105,122,424 bytes**. Total processed directory storage including adjacent receipts: **274,069,895 bytes**. Remaining local raw: **1,299,760,223 bytes**, only sub-010012 and sub-010250. Remaining unassigned structural candidates: **188**.

D: free-space snapshots before/after batch execution: **50,063,872,000 / 49,019,097,088 bytes**. These volume snapshots also reflect temp/cache and unrelated activity; raw bytes freed are accounted independently. Completed synthetic pytest fixture directories were removed after checking no pytest process remained; raw, processed outputs and runtime journals were preserved.

# 14. Leakage audit

The original Pilot Train access registry is unchanged at 150 records. The streaming Train registry has 80 unique authorized Train records. The held-out registry has 138 unique records total (90 Pilot + 48 Batch), with authorized purposes and exact provisional/native intervals. Per-record code HEADs and authorization timing match the original or superseding resume authorization. Train parameters match Train reference HDF5, receipts, per-subject QC and source provenance.

No PSD, alpha power/topology, IAF, connectivity, wPLI, classifier/generator performance, TSTR, augmentation performance, held-out waveform plots, scientific held-out analysis, normalization fitting, interrupted-fragment read, out-of-batch download, final cohort/split lock, metadata-clone write or Git push occurred. The verifier checks executed workflow evidence and relevant call graphs; it is not OS-wide surveillance.

# 15. Tests

Final full suite: **275 passed in 31.39 seconds**, JUnit at `outputs/logs/pytest.xml`; SHA256 `59063608a0d4e196f9024dffc8315f06f9996d38d43ae3a85abbb44e77a5c1a3`. `compileall scripts src tests` passed. Read-only Batch 001 verifier passed for all eight outputs and access/cleanup/accounting evidence. Synthetic tests prohibit real EEG requests and cover variable caps, immutable freeze, contiguous order, no replacements/next-batch execution, strict resume, cleanup guards, idempotent reuse, environment isolation and the C: guard.

An initial baseline run had one environmental failure because C:-based fixtures fell below the existing disk guard; all 208 baseline tests passed using D:. Implementation tests passed 267 cases before real I/O; hardening and final suites passed 275. Scientific rules and production disk safeguards were not relaxed.

Final clean-worktree acceptance (no network or preprocessing):

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 001 --require-clean --test-report outputs/logs/pytest.xml
git status --short
git rev-parse HEAD
```

For a new full test run, first ensure another pytest is not using the shared base, dot-source the bootstrap with `-ForTests`, and explicitly pass `--basetemp "D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/temp/pytest"`.

# 16. Failures/restarts and interruption/resume

The user manually stopped the initial process because C: had approximately 600 MB free, then manually cleared approximately 7.7 GB of pip cache and reported recovery to approximately 8.8 GiB. No active streaming/pytest process remained when checked. The agent preserved and hashed the 400-MiB sub-010003 partial and reverified completed sub-010002 before making engineering changes. No Windows pagefile configuration was changed.

Hardening was committed before any resumed request. The exact continuation commands executed were:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 001 --batch-manifest outputs/metrics/scenario2/batches/batch_001_manifest.csv --authorize-resume
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 001 --batch-manifest outputs/metrics/scenario2/batches/batch_001_manifest.csv --execute-batch --delete-raw-after-success --continue-on-error --retries 3 --timeout 60
```

The execution command reused sub-010002 and resumed sub-010003 from byte 419,430,400. It processed only the original remaining members. These commands document the completed restart; **no retry is needed now**. Sub-010012 is a legitimate frozen QC failure, not an unresolved processing failure, and must not be reprocessed to rescue it. If a future technical restart is separately required after a code change, preserve the same manifest and create a numbered resume authorization before I/O. Original authorizations must not be overwritten.

# External data storage layout

- `metadata_root`: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON`
- `scenario2_data_root`: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2`
- `raw_root`: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\raw`
- `processed_root`: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\processed`
- `temp_root`: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp`
- `runtime_root`: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\runtime`

All-block files live in `processed/all-blocks-v1`; preserved Train references in `processed/train-only-v1`. The committed relative policy is `configs/paths/data_paths.yaml`; ignored machine-local `configs/paths/local_data_paths.yaml` or `SCENARIO2_PATH_CONFIG` can override it. Authorizations freeze resolved execution paths. No generated data is written into the original metadata clone.

Migration inventory: **166 files / 7,193,811,222 bytes**. All 30 raw payload hashes and 30 HDF5 outputs were verified; HDF5 and receipts remained byte-identical. Old copies were removed only after destination verification. No large raw/processed payload remains inside the Git repository. The metadata clone is clean and unchanged. Small migration summaries and the full file/hash inventory are in `outputs/metrics/scenario2/storage_migration_*.json`.

The reusable bootstrap sets TEMP/TMP/TMPDIR to `temp/`, PIP_CACHE_DIR to `temp/pip`, MPLCONFIGDIR to `temp/matplotlib`, JOBLIB_TEMP_FOLDER to `temp/joblib`, PYTHONPYCACHEPREFIX to `temp/python`, MNE cache/fake-home to `temp/mne`, and pytest base/cache to `temp/pytest` / `temp/pytest-cache`. `temp/batch` also exists. XDG cache/config paths are external. Exact environment and Python tempfile resolution are recorded in interruption, resume authorization, and latest environment preflight JSON files. These are process-local settings, not assumptions about a user's interactive PowerShell.

The Windows guard checks C: before download/resume, each asset request, preprocessing and full tests. Below **3 GiB / 3,221,225,472 bytes**, it records `c_drive_low_space_guard` and stops before the expensive operation, preserving artifacts. Project-controlled writes remain on D:; the policy does not control unrelated applications or OS-managed storage. Final environment/free-space evidence is also written to external `runtime/batches/batch_001_final_acceptance.json` after the final Git commit.

# 17. Exact next step

**Batch 002 was NOT planned or executed. Its size remains user-selectable.** The user must choose a maximum GiB value before any Batch 002 manifest is generated. It will start from the first remaining unassigned subject in authoritative order under its newly selected cap. Existing Batch 001 cap and membership remain immutable.

Examples only; none was executed:

```powershell
. .\scripts\setup_scenario2_environment.ps1
# 5 GiB example:
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --batch-id 002 --batch-max-gib 5 --plan-next-batch
# 7 GiB example:
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --batch-id 002 --batch-max-gib 7 --plan-next-batch
# 10 GiB example:
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --batch-id 002 --batch-max-gib 10 --plan-next-batch
```

NEXT ACTION — CHOOSE BATCH 002 SIZE
