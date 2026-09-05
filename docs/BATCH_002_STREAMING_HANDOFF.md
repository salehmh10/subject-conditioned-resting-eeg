# 1. Executive state

**BATCH_002_STREAMING_COMPLETE.** Execution stopped after all 16 immutable Batch 002 members completed technically. Every output has all 16 assigned blocks and a verified receipt. Fifteen pass frozen Train QC and provisional 5/1/2 feasibility. Sub-010030 fails both and retains raw. There are no unresolved technical failures, retries, or interruptions.

The final cohort and split remain unlocked. Normalization remains unfitted; scientific held-out outcomes remain sealed. Batch 003 has no selected cap, membership, or execution. The final clean-worktree acceptance command below certifies the final local handoff commit.

# 2. Git state and commits

- Starting clean `main` HEAD: `bb51533b9a5019a8006235e30c1fe3d00684526c`.
- Tested implementation, frozen 10-GiB plan, prior-state checks, and pre-I/O documentation: `2124c7e39ebf6032649c823cbf678d7215c13d72`.
- Compact results, final validation evidence, registries, and documentation: `53ab2bc9b16b10cb3eff57ee9f41ef34a4191347`.
- Final HEAD is the commit containing this handoff; resolve it with `git rev-parse HEAD`. Its exact hash and clean-worktree result are saved after commit in external `runtime/batches/batch_002_final_acceptance.json`. A committed file cannot include its own containing commit hash.

No existing commit was amended, no history was rewritten, and no Git push occurred. All raw/processed signal files remain outside Git. `docs/BATCH_001_STREAMING_HANDOFF.md` is unchanged historical evidence.

# 3. Exact 10-GiB policy

The user separately selected **10 GiB = 10,737,418,240 bytes** for Batch 002 only. One GiB is 1,073,741,824 bytes. Membership counts complete declared `.eeg` plus `.vhdr` pairs, independent of local payload/partial bytes. Verified Pilot and Batch 001 completions, all existing verified outputs, and earlier assigned subjects are excluded.

The planner follows `full_candidate_download_manifest.csv` in authoritative subject order, beginning with the verified first remaining subject, sub-010016. It stops when the next complete subject would exceed the cap. There is no bin packing, skipping to smaller later subjects, or replacement after failure.


# 4. Frozen membership

| Order | Subject | Declared bytes |
| --- | --- | ---: |
| 1 | sub-010016 | 625,866,959 |
| 2 | sub-010017 | 626,263,759 |
| 3 | sub-010019 | 650,319,759 |
| 4 | sub-010020 | 624,763,359 |
| 5 | sub-010021 | 626,858,959 |
| 6 | sub-010022 | 621,303,759 |
| 7 | sub-010024 | 630,566,559 |
| 8 | sub-010026 | 644,516,559 |
| 9 | sub-010027 | 633,976,559 |
| 10 | sub-010028 | 636,555,759 |
| 11 | sub-010029 | 636,456,559 |
| 12 | sub-010030 | 645,359,759 |
| 13 | sub-010031 | 646,004,559 |
| 14 | sub-010032 | 620,820,159 |
| 15 | sub-010033 | 625,767,759 |
| 16 | sub-010034 | 646,016,959 |

**16 subjects / 32 assets; 10,141,417,744 bytes (9.444931283593 GiB).** Unused: 596,000,496 bytes (0.555068716407 GiB).

The first unassigned subject afterward is **sub-010035**, with **633,988,959 bytes**. Adding it would exceed the cap. It is not assigned to Batch 003.

Manifest SHA256: `0cd751ec43982097fc4cc396fccb8341d4690c4178e35f3e9ea11feddb4c31c1`.
Canonical plan SHA256: `b71998c721d05e019b2346d50a57aee59027974b9fbafd66f4b39795601e10a6`.
Plan-file byte SHA256: `abd98c78058f2c8a9a38e5a030200be1a7465ae03982abc36e2d7ef89f1c4292`.

# 5. Starting state, authorization, and immutability

Before changes, the clean Batch 001 verifier passed and all **23 prior outputs** were reverified against receipts and durable completion journals. `batch_002_starting_state.json` captures 132 artifact hashes/sizes, the exact Batch 001 ledger entry, prior registry row counts/hashes, completed IDs, frozen hashes, and the D-only environment. Its SHA256 is `8a1042e6a40ee8837cfca172f94e596b666f4a2530c635916c9bf08b243f9b94`. This protects prior outputs/receipts, retained raw, original Batch 001 files/journals, the split/block inventory, and unchanged scientific source/configurations. Full hashes were rechecked before authorization and at final acceptance; small prior records and registry preservation were checked at every subject boundary.

Membership froze at **2026-09-05T16:03:17.089599+00:00**. Durable atomic authorization was created at **2026-09-05T16:07:17.842102+00:00**, before the first request, referencing committed execution HEAD **2124c7e39ebf6032649c823cbf678d7215c13d72**. Authorization SHA256: `1ed616e0e814e69735ebb148964aa7ff1744601776420c7036053d882d73c2cd`. It records exact membership/cap, manifest/plan/config hashes, expected bytes, resolved paths, and explicitly prohibits Batch 003. The plan's `authorization_head` is the freeze-time starting HEAD; actual execution HEAD is in `execution_authorization.json`.

Batch 001's ledger entry and all recorded artifacts remain unchanged. None of the completed Pilot/Batch 001 subjects was downloaded or reprocessed; this includes sub-010250 and sub-010012. Their compact outputs and intentionally retained raw remain intact.

# 6. Network preflight and accounting

- Cap: **10,737,418,240 bytes**.
- Declared, expected new, and actual measured payload: **10,141,417,744 bytes / 9.444931283593178 GiB** each.
- Verified existing bytes and valid partial-prefix bytes at preflight: **0 / 0**.
- Resumed-prefix bytes validated: **0**; no Range request was needed.
- Retries / interruptions / manual restarts: **0 / 0 / 0**.
- Largest subject payload: **650,319,759 bytes** (sub-010019).
- Conservative compact-output allowance: **536,870,912 bytes / 512 MiB**, including temporary allowance; actual all-block addition is below.
- Preflight free space: C: **14,811,791,360 bytes**; D: **49,516,765,184 bytes**.
- End-of-execution free space: C: **14,031,372,288 bytes**; D: **48,674,717,696 bytes**.

Application accounting sums actual response-body bytes, including retries if any. HTTP/TLS/ISP overhead and bytes never returned to the application reader are not measured. Expected transfer is not substituted for measured traffic. Only the 32 assets in the frozen manifest appear in the Batch 002 download journal. Network execution has stopped.


# 7. Per-subject terminal results

| Subject | Status | Train QC | 5/1/2 feasible | Raw | Bytes freed / retained |
| --- | --- | --- | --- | --- | ---: |
| sub-010016 | complete | pass | yes | deleted | 625,910,833 |
| sub-010017 | complete | pass | yes | deleted | 626,307,634 |
| sub-010019 | complete | pass | yes | deleted | 650,363,674 |
| sub-010020 | complete | pass | yes | deleted | 624,807,101 |
| sub-010021 | complete | pass | yes | deleted | 626,902,970 |
| sub-010022 | complete | pass | yes | deleted | 621,347,714 |
| sub-010024 | complete | pass | yes | deleted | 630,610,464 |
| sub-010026 | complete | pass | yes | deleted | 644,560,707 |
| sub-010027 | complete | pass | yes | deleted | 634,020,484 |
| sub-010028 | complete | pass | yes | deleted | 636,599,706 |
| sub-010029 | complete | pass | yes | deleted | 636,500,467 |
| sub-010030 | complete | fail | no | retained | 645,403,807 |
| sub-010031 | complete | pass | yes | deleted | 646,048,535 |
| sub-010032 | complete | pass | yes | deleted | 620,864,063 |
| sub-010033 | complete | pass | yes | deleted | 625,811,729 |
| sub-010034 | complete | pass | yes | deleted | 646,060,850 |

Each subject completed source size/SHA256 verification, sidecar/link/header/sample-count/channel/block checks, Train-first fitting, unchanged held-out mechanical application, output reopen/hash/receipt verification, durable completion/QC journals, and cleanup or retention before the next subject. The first-subject canary passed. All 16 blocks remain stored, including ineligible blocks with their masks and reasons.

# 8. QC and provisional feasibility

Train QC passes **15/16**; provisional 5/1/2 feasibility passes **15/16**. Sub-010030 has eligible EO Train **2/5**, EO Validation **1/1**, EO Test **0/2**; EC Train **5/5**, EC Validation **1/1**, EC Test **2/2**. Exact reason: `frozen_Train_QC_failed;EO_Test_assigned_block_ineligible`. Its cleanup blocker is `frozen_Train_QC_failed`.

No threshold relaxation, state-specific fitting, rescue, or block reassignment occurred. These counts are mechanical QC eligibility, not scientific held-out outcomes. Technical completion and cleanup eligibility are separate from provisional feasibility. There are no technical failures and no replacement subjects.

# 9. Raw cleanup and retention

Deleted: sub-010016, sub-010017, sub-010019, sub-010020, sub-010021, sub-010022, sub-010024, sub-010026, sub-010027, sub-010028, sub-010029, sub-010031, sub-010032, sub-010033, sub-010034. All 15 directories are absent after verified cleanup; source/output/provenance checks and durable completion preceded deletion. Freed **9,496,716,931 bytes (8.844506862573 GiB)** including sidecars.

Batch 002 retained: **sub-010030**, **645,403,807 bytes**, because frozen Train QC failed. Cumulative retained raw also includes unchanged sub-010012 (**648,429,341 bytes**) and sub-010250 (**651,330,882 bytes**). All three source pairs were freshly rehashed after execution. No real `.part` remains. No retained subject needs reprocessing merely because QC failed.

# 10. Compact output size and hashes

Batch all-block HDF5 addition: **116,775,031 bytes (111.365347862 MiB)** across 16 outputs and **256 groups**, comprising 160 Train and 96 held-out blocks. HDF5 and adjacent receipts reopen and verify the ordered 19 channels, 128 Hz, float32 microvolts, 512-sample windows/256-sample stride, masks/reasons, exact assigned blocks/splits, frozen and source hashes, Train-only fitting provenance, and held-out apply-only provenance.


| Subject | HDF5 bytes | HDF5 SHA256 |
| --- | ---: | --- |
| sub-010016 | 7,319,381 | `f862fdac9950330cdd51eb1212dc5c655a9d6685f2813a045ff22b31b54202cc` |
| sub-010017 | 7,264,667 | `d7b3c4e179a7e934b0cedc41b98f1ab25391d3817eaf5dc48a5e4ff2ea2345b7` |
| sub-010019 | 7,322,840 | `d671f883f61c130d2aee70af697a582e268c58a5617f9fe569afd3d8617c0203` |
| sub-010020 | 7,318,158 | `e068569a8306eb0909a1a5cd31f3336fd5e78c8676c2569e953254c72001c61f` |
| sub-010021 | 7,290,234 | `52078cee0309f156575e8aa6b9c08761ea5a35aacf563749b068e9953901149c` |
| sub-010022 | 7,283,067 | `be13d75db2f7e8c64288e3e9abb313b476159bf38c7138ca9db9192b9f5e0df8` |
| sub-010024 | 7,227,416 | `388cea946dce185de9f3a130640159aa04917b08b65d33acd80574c624740545` |
| sub-010026 | 7,330,463 | `029bce4a6bbb5bd194e23c14ad6e4374b95b2591b304a403f17ab074d0c27183` |
| sub-010027 | 7,278,112 | `dfed2ad57ff595a437de2f5941eb04b4bf83588c05616a0440bdda1489df230c` |
| sub-010028 | 7,305,090 | `7eaa4a1c52bfa30f271a3725b04ed4e378e8db3c327b06dd4f86a286ae6cf1b6` |
| sub-010029 | 7,275,501 | `82e3c1360f2f3cc715e000dd754ca87322661981c2edf7584846348d1488e880` |
| sub-010030 | 7,248,476 | `a8d3de7471a6ab4084b4b4500fe394acb770d22c4bc33dd6d6aaf6c34e158d08` |
| sub-010031 | 7,316,723 | `812b7615c8560ec858b1f9cec77fc2662ff6a2340a7de46721344253056cf0b7` |
| sub-010032 | 7,344,471 | `de45c14dc3fe647b99c304ad5e7d9a0bbec105011669be4b9c421381468e7005` |
| sub-010033 | 7,295,973 | `8382aeba4c22e7f5a524366a61bf4d3021c20bb08937921f9bd2069d7f74e184` |
| sub-010034 | 7,354,459 | `e4684868ad3bf486265e73759a43613bbdd5ce4b020138e594e5f26365624c79` |

`batch_002_output_manifest.csv` additionally contains resolved paths and receipt hashes. Preserved Train-reference HDF5 addition is **72,959,828 bytes**. Total processed addition including receipts is **190,223,465 bytes**.

# 11. Cumulative progress and disk usage

- **39/211 verified all-block subjects**, **624 blocks**: 390 Train and 234 mechanically processed held-out blocks.
- Cumulative Train-QC pass: **36/39**; provisional 5/1/2 feasible: **35/39**. No final cohort/split is declared.
- Remaining unassigned structural candidates: **172**.
- All-block HDF5: **285,019,467 bytes**.
- Preserved Train HDF5 references: **178,082,252 bytes**.
- Total processed directory including receipts: **464,293,360 bytes (0.432406887412 GiB)**.
- Retained raw: **1,945,164,030 bytes (1.811575172469 GiB)** in exactly three directories listed above.

Volume free-space snapshots also reflect caches and unrelated activity. They are not substitutes for separately measured raw bytes freed. Completed synthetic pytest fixtures were removed only after confirming no pytest process remained, using the verified external `temp/pytest` path. All real raw, processed outputs, and runtime journals were preserved.

# 12. D-only environment and storage

Read-only metadata: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON`.
External Scenario 2 root: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2`.

Raw uses `raw/<subject>/eeg`; all-block output uses `processed/all-blocks-v1`; Train references use `processed/train-only-v1`; temporary/cache work uses `temp`; detailed journals use `runtime/batches`. The committed relative policy is `configs/paths/data_paths.yaml`. No signal payload is under the repository or original metadata clone.

Dot-source `scripts/setup_scenario2_environment.ps1` before every Python command. TEMP/TMP/TMPDIR and Python `tempfile.gettempdir()` resolve to external `temp`; pip, matplotlib, joblib, Python bytecode, MNE, XDG, and pytest use its configured subdirectories. Exact values are in `batch_002_latest_environment_preflight.json`. The **3-GiB C: guard** remains active before requests, subject download/resume, preprocessing, and full tests; D: capacity checks remain active. No Windows pagefile setting was changed.

# 13. Leakage and scientific restriction audit

Frozen preprocessing canonical SHA256: `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b`.
Frozen mechanical-access canonical SHA256: `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4`.
Variable batching YAML byte SHA256: `41d53f288786880d9a92ea202a558abe0fa5030fb8a3a872c275e959e6048ced`.

The frozen numerical method and QC are unchanged, including filters/resampling, 433-sample edge guard, channel order, bad-channel metrics/limits, interpolation, CAR, artifact/VEOG rules, masks, HDF5 representation, and provisional allocation. Prior scientific source/configuration hashes remain intact.

The original Pilot Train registry remains unchanged at 150 records. The cumulative streaming Train registry has 240 records, including exactly 160 Batch 002 records. The held-out registry has 234 records, including exactly 96 Batch 002 records. The verifier checks exact allowed subjects/blocks/native intervals, authorized purposes, code HEADs/timestamps, uniqueness, and matching Train-derived parameters. Existing prior records are unchanged.

No PSD, Alpha, IAF, topography, connectivity, wPLI, classifier/generator performance, TSTR, augmentation result, held-out waveform plot, or scientific held-out analysis was calculated or inspected. No normalization was fitted, no interrupted fragment was read, and no final cohort or split was locked. This audit covers application code, provenance, and registries, not OS-wide surveillance.

# 14. Tests and verification

Pre-network full suite: **280 passed in 31.73 seconds**. Final full suite: **280 passed in 33.43 seconds**. `compileall scripts src tests` passed both times. Final JUnit: `outputs/logs/pytest.xml`, SHA256 `7c277c729facf4e689eefda8f2f69fb81f74c603ca4e01a12f48b475b4e6d840`.

Offline tests include exact 10-GiB contiguous Batch 002 membership, exclusion/no omissions or duplication, immutable Batch 001 files/ledger, registry/artifact tampering, no future planning, sequential execution, resume behavior, D-only paths, C: guard, guarded cleanup, QC-failed retention, frozen apply-only processing, and scientific restrictions. Tests use synthetic/mock payloads and do not request real EEG.

The read-only Batch 002 verifier passed all 16 outputs/256 blocks, receipt and Train-reference provenance, prior artifacts, network whitelist/accounting, sequential durable completion/deletion, cumulative registries, unchanged frozen science, no Batch 003, no signal files in Git, and a clean metadata clone. Final clean-worktree verification is saved externally after the handoff commit.

Detailed journal evidence remains external:


- `batch_002_download_journal.jsonl`: 671,426 bytes; SHA256 `6639e5ecdebeac87eaf0616b11e38087edb7b63cb782e342c9708e29784e2aa7`.

- `batch_002_streaming_journal.jsonl`: 1,124,068 bytes; SHA256 `4ba20b922caa0fa29472191ed99db8ceb53a35cd218e0bba96471db1ea340e0a`.

# 15. Exact verification and restart commands

Current state needs **verification only**, not re-execution. From the repository root:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 002 --require-clean --test-report outputs/logs/pytest.xml
git status --short
git rev-parse HEAD
```

For a new full test run, first ensure no other pytest uses the shared base:

```powershell
. .\scripts\setup_scenario2_environment.ps1 -ForTests
.\.venv\Scripts\python.exe -m pytest -q --basetemp "D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/temp/pytest" --junitxml outputs/logs/pytest.xml
.\.venv\Scripts\python.exe -m compileall scripts src tests
```

The exact completed execution commands were:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 002 --batch-max-gib 10 --batch-manifest outputs/metrics/scenario2/batches/batch_002_manifest.csv --authorize-batch
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 002 --batch-max-gib 10 --batch-manifest outputs/metrics/scenario2/batches/batch_002_manifest.csv --execute-batch --delete-raw-after-success --continue-on-error --retries 3 --timeout 60
```

No restart is needed. Sub-010030 is a completed QC failure, not an unresolved technical failure. Do not rerun the authorization/execution commands at the later handoff HEAD. If a separately authorized technical continuation ever becomes necessary with unfinished members after a code change, preserve the original frozen manifest/authorization and use the existing numbered resume mechanism before I/O:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 002 --batch-manifest outputs/metrics/scenario2/batches/batch_002_manifest.csv --authorize-resume
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 002 --batch-max-gib 10 --batch-manifest outputs/metrics/scenario2/batches/batch_002_manifest.csv --execute-batch --delete-raw-after-success --continue-on-error --retries 3 --timeout 60
```

Those are conditional interrupted-run commands, not instructions to reprocess this completed batch. Never overwrite frozen records or substitute an unbounded full-manifest execution.

# NEXT ACTION — CHOOSE BATCH 003 SIZE

**Batch 003 was not planned or executed. Its cap remains UNDECIDED.** No future manifest, request, or ledger assignment exists. The user may separately choose a future cap. This handoff does not select one and does not authorize further network execution.

