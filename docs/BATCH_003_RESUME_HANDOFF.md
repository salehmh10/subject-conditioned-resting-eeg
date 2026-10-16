# Batch 003 resume handoff

**BATCH_003_RESUME_COMPLETE.** Both previously unresolved members, **sub-010073 and sub-010074**, completed technically, passed frozen Train QC and provisional 5/1/2 feasibility, and had raw cleanup verified. Batch 003 now has **33/33 complete outputs and 528 blocks**; its ledger is `complete`, with no unresolved technical failure or assessed QC/feasibility failure.

The continuation reused the verified **402,653,184-byte (384-MiB)** sub-010073 prefix through HTTP 206 with exact Content-Range, then verified the complete source SHA256. New response-body transfer was **889,641,134 bytes (0.828542871 GiB)**, with **0 new retries and 0 new interruptions**. Sub-010074 downloaded fresh. Only these two subjects received new requests or preprocessing; all 280 pre-existing output/receipt artifacts and old journal prefixes remain unchanged.

The frozen Batch 003 cap remains **20 GiB = 21,474,836,480 bytes**. Membership, order, manifest, plan, original authorization, and immutable ledger fields are unchanged. Total Batch 003 declared and measured response payload now both equal **21,299,575,247 bytes (19.836775257 GiB)**, including the prefix received during the prior attempt. Unused membership capacity remains **175,261,233 bytes (0.163224743 GiB)**. Earlier six retries/eight interruption events remain historical evidence; the numbered continuation did not add any.

The two new guarded deletions freed **1,292,382,143 bytes (1.203624665 GiB)**, including sidecars. New all-block HDF5 addition is **14,544,040 bytes (13.870277 MiB)**. Batch 003 has no retained raw or partials. Prior intentionally retained raw for sub-010012, sub-010030, and sub-010250 remains unchanged.

Cumulative progress is **72/211 complete subjects / 1,152 blocks**, **69 Train-QC passes**, **68 provisional-feasibility passes**, and **139 unassigned candidates** beginning with sub-010075. Current retained raw is **1,945,164,030 bytes (1.811575172 GiB)**; total processed storage including Train references and receipts is **856,800,945 bytes (0.797958062 GiB)**. No final cohort or split is locked.

Final verification passed for all 33 Batch 003 subjects; **291 tests passed in 33.03 seconds**, and compileall passed. Signal/temp/cache/runtime storage remains external on D:, with the 3-GiB C: guard active. Frozen preprocessing and mechanical-access hashes are unchanged. **Batch 004 was not planned. No scientific held-out analysis, normalization fit, or Git push occurred.**

## Git state and immutable evidence

- Starting clean HEAD: `ac58dba2c717af8877a2c4e85b9743592ebb2dab`.
- Tested execution HEAD: `14a1b10df5a226901f347e8c0cfca2f577449ed0`.
- Final local HEAD is the commit containing this completed handoff. Resolve it with `git rev-parse HEAD`; its exact hash and clean-worktree verification are recorded after commit in external `runtime/batches/batch_003_resume_final_acceptance.json`. A file cannot record its own containing commit hash.
- No existing commit was amended, no history was rewritten, and no Git push occurred.

The manifest, plan, original execution authorization, complete membership/order, and 20-GiB cap are byte-identical to the starting state. Earlier Batch 001/002 ledger entries and artifacts remain unchanged. Only Batch 003 operational ledger status and execution timestamps changed during continuation, ending in `complete`; the old ledger snapshot is retained in the external starting-state record. All 31 prior Batch 003 outputs and all earlier outputs were reverified without native waveform reprocessing.

| Frozen evidence | SHA256 |
| --- | --- |
| Manifest | `d8299bdde9d13186970d9b717f7f0a8fbcb252f2115a8b7d591e2e561e61c723` |
| Canonical plan | `14884fe2ef6aceabd6167f156c887bf1bd5343ab6099cd76ee8d9c6d5531c56a` |
| Original execution authorization file | `2626af44ec928b1598b5bfa1c97b8b3c0b5ced1997e290e39a47f345554befc5` |
| Frozen preprocessing | `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b` |
| Frozen mechanical access | `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4` |
| Resume starting-state record | `db6466a43ddc1e6df3ed877fb6a4787b8fbb7a4e06074486143ed655bd9ff051` |

## Connectivity and numbered authorization

DNS resolution for `nemar.s3.us-east-2.amazonaws.com` passed at `2026-09-05T20:34:58.924886+00:00`. A non-destructive HTTP HEAD request for the exact sub-010073 EEG source returned **200**, `Content-Length: 625542800`, and `Accept-Ranges: bytes`. **Zero response-body bytes** were read by this check. The request used the configured host without DNS overrides or payload probes. Evidence: `batch_003_resume_connectivity_preflight.json`.

Numbered record: **`batch_003_resume_authorization_001.json`**, created at **`2026-09-05T20:37:45.402712+00:00`**, SHA256 **`7bb658917ddeff1b61218b9333d06676da8c79062565149375d25eebc3da90e3`**. It references the unchanged original authorization, current committed code, all 31 completed members, exactly the two unresolved IDs, partial length/hash, remaining transfer estimate, frozen hashes, D-only paths, and explicit prohibition of Batch 004. The original authorization was not overwritten.

The top-level inherited `expected_new_network_bytes` remains the original full-batch expectation; `resume_network_preflight.expected_new_network_bytes` records the actual continuation expectation of 889,641,134 bytes. Its subject list/count explicitly identifies the two remaining members. The remaining-subset preflight is not a new membership allocation; authoritative unused batch capacity is still 175,261,233 bytes.

## Response accounting and terminal results

| Continuation measure | Value |
| --- | ---: |
| Original partial reused | 402,653,184 bytes / 0.375 GiB |
| New response-body bytes | 889,641,134 / 0.828542871 GiB |
| Payload retry / interruption events | 0 / 0 |
| Numbered resume authorizations | 1 |
| Manual interruptions during continuation | 0 |
| Exit code | 0 |

The initial partial SHA256 was verified as `72df8eb5246c8bff83eb2c576427faaf850e05870a374bb529b203e87cf3f233` before use. The server accepted `Range: bytes=402653184-` and returned **206** with **`Content-Range: bytes 402653184-625542799/625542800`** before appending. The whole file then passed exact 625,542,800-byte size and SHA256 `9024d86996460bd58d11bb5dcfa5d61fa104523f549b8869bf00f0df69abcd40`. No byte-zero restart was used. Sub-010074 had no source/partial before continuation and used a fresh 200 response, then passed full source verification.

Application accounting measures response-body bytes; HTTP/TLS/ISP overhead is excluded. The initial 384 MiB was counted during the earlier attempt and was not counted a second time as new transfer. Historical six retries and eight interruption events remain in the cumulative journal. The existing summary field `manually_interrupted_runs` counts numbered resume authorizations (now one); it does not mean this continuation was manually interrupted.

| Subject | Technical result | Blocks | Train QC | Provisional 5/1/2 | Raw outcome / bytes freed |
| --- | --- | ---: | --- | --- | --- |
| sub-010073 | complete | 16 | pass | pass | deleted / 625,588,452 |
| sub-010074 | complete | 16 | pass | pass | deleted / 666,793,691 |

Sub-010073 finished with guarded cleanup at `2026-09-05T20:39:48.964007+00:00`; sub-010074 finished at `2026-09-05T20:43:30.218147+00:00`. Batch closure was recorded at `2026-09-05T20:43:40.142752+00:00`. Network execution is stopped. Each subject has durable completion before and after raw cleanup; both final states are successful.

## Compact outputs and storage

| Subject | HDF5 bytes | Output SHA256 | Receipt SHA256 |
| --- | ---: | --- | --- |
| sub-010073 | 7,285,207 | `24edc57526ef3c0255d6ef641b43bcd5ba66a4d97b5d806d3d59ff8b0b510ba0` | `b3fc7f9f312170a7344641cb28163ad7558c3a6c9b0928110413a67aee979676` |
| sub-010074 | 7,258,833 | `791fd35685538c984ce343034d6aa7ab09f890486131c59255e201ec70246dae` | `65365ceb3296b257ae7d612484cd3497a0e788f9e7663acf723d02f007f0cd21` |

Outputs are external under `D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/processed/all-blocks-v1`. Exact paths and hashes are in the updated `batch_003_output_manifest.csv`. Both outputs were reopened to verify source/provenance, all 16 block IDs and splits, 19 ordered channels, 128 Hz, float32 microvolts, 512/256 window geometry, masks/reasons, Train-only fitted parameters, and unchanged held-out apply-only processing.

Total Batch 003 all-block HDF5 is **240,952,608 bytes** and total Batch 003 raw cleanup freed **21,301,024,388 bytes** across all 33 members. Current all-block HDF5 across 72 subjects is **525,972,075 bytes**; Train-reference HDF5 is **328,629,530 bytes**. Total processed-directory bytes include both sets and receipts.

Only prior Train-QC-failed raw subjects sub-010012 (648,429,341 bytes), sub-010030 (645,403,807 bytes), and sub-010250 (651,330,882 bytes) remain. Every retained file was rehashed and matched the starting inventory. Both resumed subjects have no raw directories or leftover partials; deletion occurred only through the existing guarded cleanup.

Storage snapshot `2026-09-05T20:46:56.556064+00:00`: **C: 14,393,200,640 free bytes (13.404712677 GiB)**; **D: 48,268,034,048 free bytes (44.953109741 GiB)**. Final current snapshots are additionally recorded in external acceptance. Free-space changes are volume-wide; exact raw bytes freed come from verified inventories.

## D-only environment and scientific restrictions

The environment bootstrap preceded each Python invocation. `TEMP`, `TMP`, `TMPDIR`, `PIP_CACHE_DIR`, `MPLCONFIGDIR`, `JOBLIB_TEMP_FOLDER`, `PYTHONPYCACHEPREFIX`, and `tempfile.gettempdir()` resolve below `D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/temp`; runtime journals are below the same external root. The original metadata clone remains unchanged. C: stayed above the 3-GiB guard; pagefile settings were not changed.

The frozen science remains unchanged: Train-only parameter fitting, held-out apply-only processing, all 16 assigned blocks/masks/reasons, no QC threshold changes or allocation changes. No PSD, Alpha, IAF, topography, connectivity, wPLI, classifier/generator results, TSTR, augmentation, or waveform plots were computed or inspected. No normalization was fitted; no final cohort or split was locked. Verification uses engineering provenance and registries, not scientific held-out outcomes.

## Tests, preservation audit, and final verification

**Before resume:** 291 tests passed in 51.39 seconds; compileall passed before the execution commit and numbered authorization. **After resume:** 291 tests passed in 33.03 seconds (JUnit suite time 33.018 seconds), zero failures/errors/skips; compileall passed.

Final JUnit SHA256: `b25471d78e4b50cd2f36f05fc600015584b2a92942b2b1349d03670d4fe06a96`. The read-only verifier passed for **33 subjects / 528 blocks**, zero technical failures, unchanged prior batches, unchanged scientific hashes, and absent Batch 004. The pre-results-commit verification has an expected dirty worktree; authoritative final clean acceptance is saved externally after commit.

The continuation audit rehashed all **280 pre-existing all-block/Train output and receipt files**, proving no regeneration. Original byte prefixes of both detailed journals remain unchanged. New download and streaming entries reference exactly sub-010073 and sub-010074; no earlier completed subject was requested or reprocessed. Batch 003 adds 20 new Train and 12 new held-out access rows. Cumulative all-block storage now has 720 Train and 432 held-out blocks. Compact summary and storage records distinguish cumulative values from continuation-only deltas.

| Detailed external journal | Final SHA256 |
| --- | --- |
| batch_003_download_journal.jsonl | `8c899c0c84352a68e9e9f195de418b9bfa20f28929fd318349ea734ab2d60a66` |
| batch_003_streaming_journal.jsonl | `2ec9e8a8140252f5dc495801611c5156264196504ee4597078ca730f70579905` |

The earlier `batch_003_validation.json` and `BATCH_003_STREAMING_HANDOFF.md` remain historical partial-state evidence. Current completion evidence is `batch_003_resume_summary.json`, `batch_003_resume_validation.json`, the updated main summary/output manifest/cleanup results/storage inventory, and external `batch_003_resume_final_acceptance.json`. Synthetic pytest fixtures were removed only after tests exited; real evidence remains intact.

## Verification and next action

No Batch 003 retry is needed. For read-only verification:

```powershell
Set-Location "D:/SHARIF/PROJECT/subject-conditioned-resting-eeg"
git status --short
git rev-parse HEAD
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 003 --require-clean --test-report outputs/logs/pytest.xml
```

Remain stopped. Batch 004 has no cap, membership, manifest, authorization, or request. Wait for a separate user decision before planning any next batch. Do not rerun the historical two-subject resume commands, introduce replacement subjects, perform scientific held-out analysis, fit normalization, or push automatically.

BATCH_003_RESUME_COMPLETE
