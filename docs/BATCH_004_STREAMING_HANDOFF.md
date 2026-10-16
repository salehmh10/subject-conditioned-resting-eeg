# Batch 004 streaming handoff

## 1. Executive status

**BATCH_004_FAILED_OR_PARTIAL.** Network execution is stopped after **74/74 frozen members reached durable terminal states**. **12 completed technically (192 blocks); 62 remain unresolved download failures**, with QC and provisional feasibility unassessed. The ledger is `failed_or_partial`. This does not claim all 74 were processed.

The immutable **45-GiB cap is 48,318,382,080 bytes**. Frozen membership contains **74 subjects / 148 assets** in contiguous authoritative order, beginning with sub-010075 and ending with sub-010241. Declared and initially expected payload: **48,277,992,566 bytes (44.962384334 GiB)**. Unused membership capacity: **40,389,514 bytes (0.037615666 GiB)**. The first unassigned candidate remains sub-010242 (662,012,959 bytes); no replacement or Batch 005 membership was generated.

Measured response-body transfer is **8,318,270,212 bytes (7.746992830 GiB)**: 7,789,787,908 bytes for the 12 completed source pairs plus a preserved **528,482,304-byte (504-MiB / 0.4921875-GiB)** sub-010232 prefix. There were **186 retries and 248 interruption events** (247 DNS failures and one read timeout), **zero manual restarts**, and **zero validated resumed-prefix bytes**. Connectivity briefly recovered for sub-010232, then timed out; its final Range retry failed DNS before any HTTP 206 response. No new request followed the final terminal sequence.

All 12 completed subjects pass frozen Train QC and provisional 5/1/2 feasibility. Their guarded raw deletions freed **7,790,315,121 bytes (7.255296336 GiB)**. The 62 unresolved raw directories remain: sub-010232 contains the partial and 61 are empty. No download failure is described as a QC failure. The three prior retained QC-failure subjects sub-010012, sub-010030, and sub-010250 are unchanged.

All-block HDF5 addition is **87,664,489 bytes (83.603372 MiB)**. Cumulative progress is **84/211 complete subjects / 1,344 blocks**, with **81 Train-QC passes**, **80 provisional-feasibility passes**, **65 unassigned candidates**, and **62 assigned unresolved Batch 004 members**. Total retained raw is **2,473,646,334 bytes (2.303762672 GiB)**; processed storage including Train references and receipts is **999,601,328 bytes (0.930951282 GiB)**.

The one-subject disk guard remains active; the 45-GiB network budget never became a simultaneous disk requirement. All 381 protected prior artifacts, original access records, scientific source/configuration, and the metadata clone remain unchanged. Each terminal subject has durable progress; snapshots exist after 10/20/30/40/50/60/70 terminals and final stop. Final tests: **299 passed in 38.11 seconds**; compileall and the read-only terminal-state verifier passed. Scientific held-out outcomes remain sealed, normalization remains unfitted, and the final cohort/split remain unlocked. **Batch 005 is unplanned. No Git push occurred.**

## 2. Git state and local commits

Starting clean HEAD: `157f4ba1fdda3375d426f8fc87a7f66f5db29941`. Tested pre-network implementation/docs/plan commit and authorized execution HEAD: `e9eb5fd014ea86d68f6da3935df10e384caf96eb`. Final local HEAD is the commit containing this handoff; resolve it with `git rev-parse HEAD`. Its exact hash and final clean-worktree verification are recorded after commit in external `runtime/batches/batch_004_final_acceptance.json`. A committed file cannot contain its own containing commit hash.

No amend, history rewrite, push, credential change, or signal-file commit occurred. Final changes contain compact results, authorization, validation, registries, and documentation only. Earlier Batch 001/002/003 files, handoffs, ledger entries, output/receipt hashes, and raw evidence match the 381-artifact starting baseline.

## 3. Exact 45-GiB policy

One GiB is exactly 1,073,741,824 bytes. Batch 004 cap is 48,318,382,080 bytes of complete declared EEG/header membership, independent of existing or partial data. The planner excludes all 72 verified completions and earlier assignments, preserves authoritative order, and stops at the next subject that does not fit. Membership, order, cap, manifest hash, and plan remain immutable. No bin packing, replacements, or future membership were used.

Declared: 48,277,992,566 bytes / 44.962384333834 GiB. Unused: 40,389,514 bytes / 0.037615666166 GiB. First unassigned: sub-010242, 662,012,959 bytes, reason `next_subject_would_exceed_batch_cap`.

## 4. Frozen membership and per-subject terminal results

| Order | Subject | Declared bytes | Terminal status | Train QC | Provisional 5/1/2 | Raw result |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | sub-010075 | 665,373,359 | complete | pass | pass | deleted |
| 2 | sub-010076 | 644,950,559 | complete | pass | pass | deleted |
| 3 | sub-010077 | 645,136,559 | complete | pass | pass | deleted |
| 4 | sub-010079 | 633,616,959 | complete | pass | pass | deleted |
| 5 | sub-010080 | 641,069,359 | complete | pass | pass | deleted |
| 6 | sub-010083 | 622,109,759 | complete | pass | pass | deleted |
| 7 | sub-010084 | 693,422,159 | complete | pass | pass | deleted |
| 8 | sub-010085 | 651,931,759 | complete | pass | pass | deleted |
| 9 | sub-010086 | 630,690,559 | complete | pass | pass | deleted |
| 10 | sub-010087 | 655,639,359 | complete | pass | pass | deleted |
| 11 | sub-010088 | 670,246,559 | complete | pass | pass | deleted |
| 12 | sub-010089 | 635,600,959 | complete | pass | pass | deleted |
| 13 | sub-010090 | 647,690,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 14 | sub-010091 | 660,648,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 15 | sub-010092 | 639,841,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 16 | sub-010093 | 628,371,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 17 | sub-010094 | 648,769,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 18 | sub-010104 | 645,074,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 19 | sub-010136 | 648,087,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 20 | sub-010137 | 707,508,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 21 | sub-010138 | 668,696,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 22 | sub-010141 | 661,665,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 23 | sub-010142 | 637,014,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 24 | sub-010146 | 676,979,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 25 | sub-010148 | 643,040,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 26 | sub-010150 | 649,922,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 27 | sub-010152 | 668,064,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 28 | sub-010155 | 672,292,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 29 | sub-010157 | 656,792,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 30 | sub-010162 | 627,255,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 31 | sub-010163 | 659,929,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 32 | sub-010164 | 650,691,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 33 | sub-010165 | 636,295,359 | download_interrupted | not assessed | not assessed | retained empty directory |
| 34 | sub-010166 | 633,096,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 35 | sub-010168 | 636,816,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 36 | sub-010170 | 631,794,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 37 | sub-010176 | 619,443,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 38 | sub-010183 | 631,459,359 | download_interrupted | not assessed | not assessed | retained empty directory |
| 39 | sub-010191 | 704,148,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 40 | sub-010192 | 654,324,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 41 | sub-010193 | 655,068,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 42 | sub-010194 | 673,011,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 43 | sub-010195 | 658,354,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 44 | sub-010196 | 662,136,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 45 | sub-010197 | 621,638,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 46 | sub-010199 | 630,876,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 47 | sub-010200 | 681,952,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 48 | sub-010201 | 649,687,359 | download_interrupted | not assessed | not assessed | retained empty directory |
| 49 | sub-010202 | 629,450,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 50 | sub-010204 | 678,467,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 51 | sub-010207 | 626,499,359 | download_interrupted | not assessed | not assessed | retained empty directory |
| 52 | sub-010210 | 660,748,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 53 | sub-010213 | 657,164,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 54 | sub-010214 | 660,475,359 | download_interrupted | not assessed | not assessed | retained empty directory |
| 55 | sub-010215 | 672,850,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 56 | sub-010218 | 641,664,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 57 | sub-010219 | 625,494,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 58 | sub-010220 | 647,703,359 | download_interrupted | not assessed | not assessed | retained empty directory |
| 59 | sub-010222 | 633,195,359 | download_interrupted | not assessed | not assessed | retained empty directory |
| 60 | sub-010223 | 634,522,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 61 | sub-010224 | 661,231,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 62 | sub-010226 | 647,963,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 63 | sub-010227 | 662,124,559 | download_interrupted | not assessed | not assessed | retained empty directory |
| 64 | sub-010228 | 662,236,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 65 | sub-010230 | 688,896,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 66 | sub-010231 | 679,434,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 67 | sub-010232 | 676,533,359 | download_interrupted | not assessed | not assessed | retained 504-MiB partial |
| 68 | sub-010233 | 675,119,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 69 | sub-010234 | 620,410,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 70 | sub-010236 | 647,690,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 71 | sub-010238 | 675,119,759 | download_interrupted | not assessed | not assessed | retained empty directory |
| 72 | sub-010239 | 650,852,959 | download_interrupted | not assessed | not assessed | retained empty directory |
| 73 | sub-010240 | 661,306,159 | download_interrupted | not assessed | not assessed | retained empty directory |
| 74 | sub-010241 | 632,600,159 | download_interrupted | not assessed | not assessed | retained empty directory |

## 5. Frozen hashes and durable authorization

| Artifact | SHA256 |
| --- | --- |
| Manifest | `0be1d41165aa55e30dc86e72724a752270de4a8bec7a50fbad5595a158e37fb5` |
| Canonical plan | `1c0b6356326029f27b95d016b500564b293769e698df1da26fc571b74d28acaa` |
| Plan file bytes | `4e21f8371494a61b665fe4e868bc82794ffed31e21fe4d1c9a61e7b2fdd7cb48` |
| Execution authorization | `b8f4c21caf1b05856a4c4727dd813e19c3fe7ed66e6ad1b150000bd8195a38f8` |
| Preprocessing | `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b` |
| Mechanical access | `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4` |
| Variable batching config | `41d53f288786880d9a92ea202a558abe0fa5030fb8a3a872c275e959e6048ced` |

Freeze: `2026-09-05T21:05:18.305273+00:00`. Atomic durable authorization: `2026-09-05T21:10:50.872850+00:00`, before first request `2026-09-05T21:11:17.747565+00:00`. It references committed execution HEAD `e9eb5fd014ea86d68f6da3935df10e384caf96eb`, exact membership/cap/order, manifest and plan hashes, scientific/configuration hashes, resolved metadata/raw/processed/temp/runtime roots, local payload/partial counts, expected response bytes, C:/D: snapshot, and explicit Batch 005 prohibition. The original authorization has not been overwritten. There is no numbered resume authorization.

The pre-network committed preflight was refreshed immediately before authorization without network access. The authorization pins the refreshed preflight hash. The plan field `authorization_head` denotes the freeze-time starting HEAD; the actual execution HEAD is in the execution authorization.

## 6. Network preflight, actual accounting, and failures

| Measure | Bytes | GiB |
| --- | ---: | ---: |
| Cap | 48,318,382,080 | 45.000000000000 |
| Declared / initial expected transfer | 48,277,992,566 | 44.962384333834 |
| Unused membership capacity | 40,389,514 | 0.037615666166 |
| Initially verified existing payload | 0 | 0.000000000000 |
| Initial partial prefix | 0 | 0.000000000000 |
| Measured response body | 8,318,270,212 | 7.746992830187 |
| Completed source pair payload | 7,789,787,908 | 7.254805330187 |
| Preserved incomplete prefix | 528,482,304 | 0.492187500000 |
| Validated resumed prefix | 0 | 0.000000000000 |
| Untransferred declared payload | 39,959,722,354 | 37.215391503647 |

There were 186 scheduled retries and 248 download interruption events: 247 DNS `getaddrinfo` failures and one read timeout. No manual interruption or restart occurred. All 74 frozen subjects were attempted in order. Network ended at `2026-09-05T22:07:16.761630+00:00`; summary/stop checkpoint at `2026-09-05T22:07:28.092096+00:00`. Process exit code: 1. Application bytes include received partial/retry bodies but exclude HTTP/TLS/ISP overhead. Expected transfer is never labelled as measured traffic.

The first 12 subjects completed. DNS failures began for sub-010090 at 2026-09-05T21:54:56.418475+00:00. Bounded retries and continue-on-error recorded terminal failures without replacements. For sub-010232, attempts 1/2 failed DNS; attempt 3 received HTTP 200 at 22:02:46.295203 UTC and transferred 528,482,304 bytes before a read timeout at 22:05:52.688377 UTC. Attempt 4 proposed Range offset 528,482,304, but failed DNS at 22:05:56.696850 before an HTTP response. Thus the prefix was preserved but no resumed prefix was validated. Remaining members then exhausted bounded retries. All failed members remain assigned and unassessed.

## 7. First-subject canary and sequential processing

sub-010075 passed automatic canary verification at 2026-09-05T21:15:19.262005+00:00, before subject 2 started at 21:15:20.083869 UTC. Independent read-only verification confirmed source fingerprints, all 16 assigned block IDs, Train-derived parameter identity against the Train reference, held-out apply-only provenance, HDF5/receipt hashes, durable completion, and absent raw. Evidence: `batch_004_canary_verification.json` and `batch_004_canary_independent_verification.json`.

One new subject was staged at a time. Each completed output has verified native/source/header/channel/sample/block coverage, 19 ordered target channels, 128 Hz, float32 microvolts, 512-sample windows and 256-sample stride, masks/reasons, source/science hashes, and durable receipts. All 16 blocks remain stored, including any ineligible windows/blocks. Subject parameters were fitted from provisional Train first, then applied unchanged to Validation/Test. Existing 72 subjects were verified without raw waveform reprocessing.

## 8. Train QC and provisional feasibility

12/12 technically completed Batch 004 subjects pass frozen Train QC and provisional 5/1/2 feasibility. There are zero assessed failures for either measure. The 62 unresolved download subjects have no processing output, Train access, held-out access, or QC/feasibility assessment. False CSV booleans on those rows must be interpreted with the explicit assessment columns. No rescue, threshold relaxation, held-out refit, block reassignment, or replacement occurred.

## 9. Guarded raw cleanup and preserved failures

Twelve verified deletions freed 7,790,315,121 bytes (7.255296335556 GiB), including sidecars; their IDs are the first 12 table rows. Exact bytes were journaled before/after deletion and all directories are absent. Source/HDF5/provenance checks, passing Train QC, all 16 blocks, durable completion, safe direct-child paths, and absence of partials preceded deletion. Held-out feasibility alone would not block otherwise valid cleanup; no such failure arose.

Batch 004 retains 62 directories: sub-010232 contains 528,482,304 bytes and 61 directories are empty. All three prior raw subjects remain unchanged: sub-010012, sub-010030, sub-010250, totalling 1,945,164,030 bytes. Total current raw: 2,473,646,334 bytes in 65 directories. Retained files and empty directories are inventoried in `batch_004_storage_verification.json`.

Partial path: `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\raw\sub-010232\eeg\sub-010232_task-resting_eeg.eeg.part`. Prefix SHA256: `105dca88b62df36d3b88d0b66d3dfab76120b2b60c5672db904f7551180f86a1`. This identifies an incomplete prefix, not a whole-file integrity pass. The complete sub-010232 EEG must pass its original manifest size/SHA256 before promotion or any preprocessing.

## 10. Compact outputs and hashes

All-block HDF5 addition: 87,664,489 bytes / 83.603371620 MiB. Total processed addition including Train references and receipts: 142,800,383 bytes. Outputs live under external `processed/all-blocks-v1`; Train references under `processed/train-only-v1`.

| Subject | HDF5 bytes | Output SHA256 | Receipt SHA256 |
| --- | ---: | --- | --- |
| sub-010075 | 7,268,010 | `2cfda12212370d675d573b0ac3409bc6d1f10428e078303f4e4543b8195892fe` | `e123ca401896baff903a944cde9e06a8657aa98ab52e286d63ad3e89841952d0` |
| sub-010076 | 7,226,503 | `b9e6a10e56d3b96cc32cdf54df29d7cc23847207b5e0b3dfef6e6f75b1c9d318` | `30c81959a98879f50e666bfb99347359803c05d51a761c8e99144035d421fcb8` |
| sub-010077 | 7,295,531 | `ec9b17672d9baeb03c6ddcdcf1f3d62bfe374b11da833340f678d2f7dbabffe6` | `466c8fce6842249a7a4fb3a3bb42bf95478bf78a02031414e8ee162d60ab1bcd` |
| sub-010079 | 7,350,542 | `fe9f2785bb60df1b23142e21f91e507800cc2ff760f059a27ea6639227b3787e` | `655c12cb8fafdebdc1a6d012d4f6a5abf0d5d1826eac8154f039c18f82781ed8` |
| sub-010080 | 7,288,829 | `e4b3012b017e7a5aa3b713138abd1e37847b4ec13560f7a1b20ab1d00b011813` | `bd5b40f5359fc533247054e3234060a0a135b8dcf5cdecc54a6edab50eca7388` |
| sub-010083 | 7,311,213 | `6a172b73ea67932cf2af9bc72c239c5ab5938180210d8667b11b343017525ebf` | `46039da07c913aa532154067de71a850fabfbf49eb41cd8ff5da4c9ad9b17a83` |
| sub-010084 | 7,229,464 | `942443b09aaf7374a1d86044e2bb10e0cdf670221dac81f01a4a5825133b5bf1` | `8249fc1138fd970591f773253a06a14741241ca5e4cd7ce8953781c9c35d9b27` |
| sub-010085 | 7,327,739 | `bd4185f662f56c4b33dfb8b612447b89917d658ff77fe22346d9ce0ff0f03d5e` | `808672cb096047d5bd287e147052e39dde04eae1dceb204d0066de0e7aeca40c` |
| sub-010086 | 7,256,806 | `4fb10b7aab00df958d4f8bbf615f0606bc8ea43c2414028a171e9a6fe159f721` | `8965ec9749638f43eda2f02842fc2c212f4a55af2968e82458eed291c790bdd1` |
| sub-010087 | 7,362,649 | `582cd1e5f411d90ec1acc437311a37801d9f8185042f8c214fd9f064732a36df` | `6e6b240c345193a5e46b7e6867a6aa86101fff0238c3d4ab9182bae379398570` |
| sub-010088 | 7,376,851 | `ffcd2c421277823f74b5dbe73feeb73ab75f0fe598b5b8e1c7ea12fd1f413ade` | `03e94b89ac0b8d34a158e39bb4538584aa795d2ebc5c90a817731008e92481d6` |
| sub-010089 | 7,370,352 | `cc660165c1c78a5a6530b1ff1fa92d9f50815309302e353e54739a196f38c676` | `7f5ccbcf01fbb5677c3b19153869a1ef22d41ed7218f41070e5dfb4850188574` |

## 11. Long-batch checkpointing

Download and streaming journals were fsynced at durable records. Each of 74 terminal subjects has a progress entry containing outcome/counts, measured bytes, exact raw bytes freed/retained, compact output, C:/D: free space, journal byte positions, and manifest hash. The progress journal contains 75 records including final closure. Latest snapshot is external `runtime/batches/batch_004_progress.json`; numbered archives and the final snapshot are below.

| Snapshot | Terminals | Complete | SHA256 |
| --- | ---: | ---: | --- |
| batch_004_progress_010.json | 10 | 10 | `eb237b0d91d537cca324d17760f645a6782e44fda39a43eebf5000aae719ef94` |
| batch_004_progress_020.json | 20 | 12 | `f1b34b4e9020417182f99aa2ad0f82676abe4b103f1d5f06456e519cfcb79c36` |
| batch_004_progress_030.json | 30 | 12 | `a69235c2e5ae35a1242cfc3e29dec9378f4ac6bc29f8e8da4ae1b7427c8e3fdd` |
| batch_004_progress_040.json | 40 | 12 | `d55e6db54756a80d7b061b0dd78d976abfb45afaaaa24dbaba2b382e66526408` |
| batch_004_progress_050.json | 50 | 12 | `073ee716ade99e4a130acd2fca83e52c6b6ecab1fbcff0fc32a1830121910a0a` |
| batch_004_progress_060.json | 60 | 12 | `49fba8e00f25a98b071b58988716ec5a3b096ef1adfc2796b71009bb182fb563` |
| batch_004_progress_070.json | 70 | 12 | `101776f72c080f34a5a92f73dee58ae99370f5e5b9f838391251db130b517131` |
| batch_004_progress_final.json | 74 | 12 | `594d1cebc10109affc594a12ce48f6ecf31599f5e42a9746d541bec273547cb9` |

Final streaming journal: `c2adabea66aebca7415e9fa4fd04dd5dd3f137ba3c06da912bdae9a9cf04716e`; download journal: `cdc5fa8fd29a92efba641776fcf655a9ab5f76e7708bd7b8868aabd364ee3a14`; progress journal: `7ca2000643e8991ddfcf8fe3d137db2ceaa6824d19ef133df87795c2b035bdb7`. Detailed journals remain external and are not committed. Checkpointing did not alter membership or start a future batch.

## 12. Cumulative progress and current storage

84/211 complete all-block subjects / 1,344 blocks: 840 Train and 504 mechanical held-out blocks. Cumulative Train-QC pass: 81/84; provisional feasibility: 80/84. There are 65 unassigned candidates beginning at sub-010242 plus 62 assigned unresolved Batch 004 members. These are engineering labels; no final cohort or split is locked.

All-block HDF5: 613,636,564 bytes. Train-reference HDF5: 383,399,021 bytes. Total processed including receipts: 999,601,328 bytes (0.930951282382 GiB). Total retained raw: 2,473,646,334 bytes (2.303762672469 GiB). Storage inventory UTC: `2026-09-05T22:15:59.981352+00:00`.

## 13. D-only environment and disk guards

Read-only metadata: `D:/SHARIF/PROJECT/Datasets/MPI-LEMON`. Scenario 2 root: `D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2`, with `raw`, `processed`, `temp`, and `runtime`. Dot-source the environment bootstrap before every Python command. TEMP/TMP/TMPDIR, pip, matplotlib, joblib, Python bytecode, MNE, XDG, and pytest caches resolve beneath external D: temp. Python `tempfile.gettempdir()` is verified there. No pagefile setting was changed.

The network cap was not reserved as simultaneous disk. Largest subject: 707,508,559 bytes. Maximum additional one-subject requirement: 2,351,675,727 bytes (2.190168693 GiB), comprising remaining raw plus 512-MiB processing, 32-MiB compact output, and 1-GiB safety margin. Including already occupied prior raw, peak footprint is 4,296,839,757 bytes. Retained raw is already deducted from free disk and is not charged twice. The dynamic per-subject D: guard remained active as failures accumulated.

Authorization free C:/D:: 14,338,703,360 / 48,267,194,368 bytes. Post-stop storage snapshot free C:/D:: 13,268,246,528 / 47,589,797,888 bytes. Final current free-space values are recorded externally after the local commit. The 3-GiB C: guard remained active and did not fire. Volume free-space changes include unrelated activity; exact raw bytes freed come from inventories.

## 14. Leakage and scientific restriction audit

The verifier confirms 120 new Train access rows and 72 new mechanical held-out rows for exactly the 12 successful subjects. Cumulative streaming Train rows: 690, plus the original unchanged 150 Pilot Train rows. Cumulative mechanical held-out rows: 504. Previous registry rows are unchanged; new intervals, purposes, code HEADs, and authorization timestamps match the frozen allocation and permissions. Failed downloads have no waveform-access rows.

No PSD, Alpha power/topology, IAF, connectivity, wPLI, classifier/generator performance, TSTR, augmentation results, held-out waveform visualization, or scientific held-out analysis was computed or inspected. No numerical normalization was fitted, no final cohort was declared, and no final split was locked. Frozen preprocessing/QC/assignment and access hashes are unchanged. The metadata clone remains clean at its starting HEAD. No signal payload is tracked by Git. The audit covers application code, provenance, and registries, not OS-wide surveillance.

## 15. Tests and read-only verifier

Before network: 299 tests passed in 38.50 seconds; compileall passed before execution commit and authorization. One initial new checkpoint test fixture omitted its runtime directory; the fixture was corrected before the passing rerun. No real EEG request was made by tests. Final: 299 passed in 38.11 seconds (JUnit suite 38.067 seconds), zero failures/errors/skips; compileall passed. Final JUnit SHA256: `12e232a6038fa446ca2574b1e156cc4735dbe96fe302447f0a5f742153c6b12d`.

Regressions cover exact 45-GiB conversion, contiguous 74-subject selection and exclusions, three prior immutable batches, no duplicates/bin packing/future batch, network budget versus one-subject disk space, retained raw/partial accounting, durable checkpoints, sequential execution, Range resume, preserved failures, all-block cleanup, Train-only fitting, held-out apply-only restrictions, no normalization/scientific analysis, D-only paths, C: guard, and idempotent completion reuse.

The Batch 004 read-only verifier passed all 74 terminal records, 12 new outputs/192 blocks, prior 72 outputs and 381 protected artifacts, raw retention/deletion, checkpoints, response-byte reconciliation, access registries, frozen hashes, no Batch 005, and the clean metadata clone. `batch_004_validation.json` records the expected pre-results-commit dirty worktree. Final clean acceptance is recorded externally after the handoff commit. Passing terminal-evidence verification does not change the 62 unresolved subjects into technical successes.

## 16. Interruption and conditional continuation

Remain stopped. Do not rerun the planner, change the 45-GiB cap/membership/order, discard the partial, pull replacements, reprocess the 84 verified completions, or plan Batch 005. A later separately authorized continuation must first restore connectivity, verify clean tested Git state and all frozen authorization/plan/manifest hashes, reverify completed outputs and retained raw, and hash the existing sub-010232 prefix. Create a numbered resume authorization; do not overwrite the original. Its remaining-transfer preflight must identify exactly the 62 unresolved members and the valid local prefix.

Current untransferred declared payload estimate is 39,959,722,354 bytes (37.215391503647 GiB) if the prefix is reused. This is not a new authorization or a changed membership cap. Proper HTTP 206/Content-Range and final full-file SHA256 remain mandatory. Prefix reuse is not guaranteed before the server responds. The following read-only verification is safe now:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 004 --require-clean --test-report outputs/logs/pytest.xml
```

Only after a later authorized continuation and fresh checks, the existing `--authorize-resume` operation creates a numbered authorization under this unchanged freeze. Execution must reuse completed outputs idempotently and operate only on unresolved Batch 004 IDs from `batch_004_subject_results.csv`. No continuation command was executed after terminal closure in this assignment.

## 17. Stop boundary and compact evidence

Batch 005 has no selected cap, subject assignment, manifest, plan, authorization, runtime journal, or payload request. Batch 004 network execution is stopped. Compact repository evidence includes the manifest/plan/ledger, starting baseline, preflight/authorization, canary checks, subject/cleanup/output CSVs, summary, storage verification, execution audit, test/verifier evidence, and this handoff. Detailed journals, progress snapshots, raw, partials, HDF5 files, and caches remain external.

## NEXT ACTION — CHOOSE BATCH 005 SIZE

Batch 005 remains unplanned and its size is undecided. Do not choose its size or generate membership automatically. The 62 unresolved subjects remain assigned to Batch 004; address them only through a separately authorized continuation under the unchanged Batch 004 freeze.
