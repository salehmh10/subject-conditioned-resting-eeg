# Batch 004 autonomous resume handoff

**BATCH_004_AUTONOMOUS_RESUME_COMPLETE.** The existing frozen Batch 004 is **74/74 technically complete, with 1,184 verified block groups, zero unresolved subjects, and no remaining Batch 004 `.part` files**. Its ledger status is `complete`. All 62 continuation subjects ran in their original manifest order; the 12 previous Batch 004 completions and 72 earlier completions were reused unchanged. No Batch 005 exists or is authorized.

The scientific worker finished at **2026-09-06 02:42:39 UTC** (06:12:39 Tehran). All 74 Train-QC and provisional-feasibility outcomes were assessed: **71 Train-QC passes, 68 provisional-feasibility passes**. Technical completion does not imply QC or provisional inclusion. Cumulative progress is **146/211 subjects and 2,336 blocks**, with **140 Train-QC passes, 136 provisional-feasibility passes**, and **65 unassigned candidates**, beginning with sub-010242.

## Git and authorization

- Verified clean starting HEAD: `cd06b3671569346deb1945fb208fb659a1ad6373`.
- Committed implementation used for every newly processed subject: `0682fa0c065916cdb01bfa5115333ba5cf0e0084` (`data: add persistent network-wait and automatic Batch resume`).
- Final implementation HEAD, including the tested Windows supervisor reporting correction: `427e92ed06afb6e4eb993209ad7e356b903db9b9`.
- Numbered authorization: [batch_004_resume_authorization_001.json](../outputs/metrics/scenario2/batches/batch_004_resume_authorization_001.json), UTC `2026-09-05T22:54:03.889923+00:00`, SHA256 `9c7f8ac4712a0080c615c9dd76aa80451eba3a0ca532b6a4a38c5328d0a39df2`. It pins the processing implementation, exactly 12 completed and 62 unresolved members, D-only paths, preserved prefix, original authorization, and network policy. It was written after the clean implementation commit and before real continuation. No additional authorization was generated for the post-completion verification checks.
- Original authorization remains byte-identical: SHA256 `b8f4c21caf1b05856a4c4727dd813e19c3fe7ed66e6ad1b150000bd8195a38f8`.
- The exact **final results/documentation Git HEAD** and clean-worktree verification are recorded after the local commit in [`batch_004_autonomous_final_acceptance.json`](D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/batches/batch_004_autonomous_final_acceptance.json). The commit containing this handoff is also resolved by `git log -1 --format=%H -- docs/BATCH_004_AUTONOMOUS_RESUME_HANDOFF.md`. A tracked document cannot contain its own commit hash without changing that hash. The final response reports the exact final HEAD. No amend or push was performed.

## Frozen identity and preservation

| Item | Verified value |
|---|---|
| Batch cap | 45 GiB = 48,318,382,080 bytes |
| Membership | 74 subjects, sub-010075 through sub-010241, unchanged manifest order |
| Declared EEG/header payload | 48,277,992,566 bytes; unused cap 40,389,514 bytes |
| Manifest SHA256 | `0be1d41165aa55e30dc86e72724a752270de4a8bec7a50fbad5595a158e37fb5` |
| Plan canonical SHA256 | `1c0b6356326029f27b95d016b500564b293769e698df1da26fc571b74d28acaa` |
| Plan file SHA256 | `4e21f8371494a61b665fe4e868bc82794ffed31e21fe4d1c9a61e7b2fdd7cb48` |
| Preprocessing canonical SHA256 | `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b` |
| Mechanical-access canonical SHA256 | `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4` |
| Batch configuration file SHA256 | `41d53f288786880d9a92ea202a558abe0fa5030fb8a3a872c275e959e6048ced` |
| Autonomous preservation baseline SHA256 | `78e921a957d11d270efbc523c4e698218d331d28a075e492ad74839c51b8a6bc` |

The final audit rehashed all **357 protected pre-continuation output/receipt/raw files**, all **8 original progress archives**, and the original byte prefixes of download, streaming, and progress journals. The previous-batch verifier also rechecked its 381-artifact baseline and the 72 prior completions. Existing Train and mechanical held-out registry rows, including the 12 previously complete Batch 004 subjects, matched the clean starting commit exactly. All source pair sizes/SHA256 verification events, metadata sidecar hashes, HDF5/receipt integrity, Train-only parameter provenance, 16 authoritative blocks, and durable cleanup ordering passed. Earlier raw for sub-010012, sub-010030, and sub-010250 is unchanged. The metadata clone is clean.

## Network operation and accounting

The committed policy uses unlimited nonterminal `waiting_for_network` for transient DNS, timeout, reset/abort, temporary TLS, disconnect/incomplete body, and HTTP 408/425/429/500/502/503/504. It keeps the current asset and subject active. Backoff is 15/30/60/120/300 seconds with capped jitter; Retry-After is honored. Two exact-object tiny probes, five seconds apart, confirm recovery. Connect/read/no-byte-stall limits are 45/120/180 seconds. Partials are fsynced periodically and on request termination. Append requires exact HTTP 206 and Content-Range; complete size/SHA256 is mandatory before promotion. Ignored Range quarantines the prefix before a guarded full restart. Hard blockers preserve evidence and stop without replacements. See [operating guide](BATCH_004_AUTONOMOUS_RUNNER.md).

The real continuation needed **zero new network-wait episodes, zero outage-wait seconds, zero reconnections, zero probe-body bytes, and zero extra retry-body bytes**. There are therefore no observed outage durations to list. Slow but advancing transfers did not trigger outages. Offline tests exercise recovery; no real outage was induced. Historical **186 retries and 248 interruptions** remain unchanged in their original journal prefixes.

| Response-body accounting | Bytes |
|---|---:|
| Received before autonomous continuation | 8,318,270,212 |
| Received during the 62-subject continuation | 39,959,722,354 |
| Full Batch 004 measured body total | 48,277,992,566 |
| Declared Batch 004 payload | 48,277,992,566 |
| Validated pre-existing prefix reused | 528,482,304 |
| Extra retry or probe bodies | 0 |

Measurements are application response-body bytes; HTTP/TLS/ISP overhead is not measured. There was **one continuous scientific continuation and zero processing restarts**. The legacy summary field `manually_interrupted_runs: 1` counts the numbered resume authorization; it does not indicate an interruption of this autonomous scientific run. Its legacy preflight prefix-local field refers to the original Batch 004 preflight, while `resumed_prefix_bytes_validated` and this audit record the actual reused prefix.

### sub-010232

The original 504-MiB prefix was preserved until its frozen turn, after sub-010231. Immediately before that turn its 528,482,304 bytes still hashed to `105dca88b62df36d3b88d0b66d3dfab76120b2b60c5672db904f7551180f86a1`.

The request used `Range: bytes=528482304-`; the server returned **206** with exact `Content-Range: bytes 528482304-676531599/676531600`. The complete EEG was **676,531,600 bytes**, SHA256 `b716f66c6fd877b4b265cbf0d354a6f71edee2ed7b184219a2eed93e40b577bc`. The header was 1,759 bytes. Only **148,051,055 new pair-body bytes** were required. Full verification passed, all 16 blocks were stored, Train QC and provisional feasibility passed, and raw cleanup completed. The output SHA256 is `b461634b966e5dc3c37aa723a62a694905d730267de038a08b7529ef76c868db`; the receipt SHA256 is `a552b4020a4ab4e4bbc031f4a78dd83f52dbfd309c28fb342cf39e807cf3fdb8`. [Independent verification](../outputs/metrics/scenario2/batches/batch_004_autonomous_sub010232_verification.json).

## All 62 continuation outcomes

Every row completed in the displayed frozen order. All rows had zero new outage episodes and zero reconnections. Only sub-010232 needed an automatic Range resume (one). Exact timestamps, per-subject byte accounting, cleanup and feasibility reasons are in the [final audit](../outputs/metrics/scenario2/batches/batch_004_autonomous_final_audit.json).

| Subject | Technical blocks | Train QC | Provisional 5/1/2 | Raw | Range prefix reused (bytes) | New body bytes |
|---|---:|---|---|---|---:|---:|
| sub-010090 | complete / 16 | pass | pass | deleted | 0 | 647,690,959 |
| sub-010091 | complete / 16 | pass | pass | deleted | 0 | 660,648,959 |
| sub-010092 | complete / 16 | pass | pass | deleted | 0 | 639,841,759 |
| sub-010093 | complete / 16 | pass | pass | deleted | 0 | 628,371,759 |
| sub-010094 | complete / 16 | pass | pass | deleted | 0 | 648,769,759 |
| sub-010104 | complete / 16 | pass | pass | deleted | 0 | 645,074,559 |
| sub-010136 | complete / 16 | pass | pass | deleted | 0 | 648,087,759 |
| sub-010137 | complete / 16 | pass | pass | deleted | 0 | 707,508,559 |
| sub-010138 | complete / 16 | pass | pass | deleted | 0 | 668,696,559 |
| sub-010141 | complete / 16 | pass | pass | deleted | 0 | 661,665,759 |
| sub-010142 | complete / 16 | pass | pass | deleted | 0 | 637,014,559 |
| sub-010146 | complete / 16 | fail | fail | retained: Train QC | 0 | 676,979,759 |
| sub-010148 | complete / 16 | pass | pass | deleted | 0 | 643,040,959 |
| sub-010150 | complete / 16 | pass | pass | deleted | 0 | 649,922,959 |
| sub-010152 | complete / 16 | pass | pass | deleted | 0 | 668,064,159 |
| sub-010155 | complete / 16 | pass | pass | deleted | 0 | 672,292,559 |
| sub-010157 | complete / 16 | pass | fail | deleted | 0 | 656,792,559 |
| sub-010162 | complete / 16 | pass | pass | deleted | 0 | 627,255,759 |
| sub-010163 | complete / 16 | fail | fail | retained: Train QC | 0 | 659,929,759 |
| sub-010164 | complete / 16 | pass | pass | deleted | 0 | 650,691,759 |
| sub-010165 | complete / 16 | pass | pass | deleted | 0 | 636,295,359 |
| sub-010166 | complete / 16 | pass | pass | deleted | 0 | 633,096,159 |
| sub-010168 | complete / 16 | pass | pass | deleted | 0 | 636,816,159 |
| sub-010170 | complete / 16 | pass | pass | deleted | 0 | 631,794,159 |
| sub-010176 | complete / 16 | pass | pass | deleted | 0 | 619,443,759 |
| sub-010183 | complete / 16 | pass | pass | deleted | 0 | 631,459,359 |
| sub-010191 | complete / 16 | pass | pass | deleted | 0 | 704,148,159 |
| sub-010192 | complete / 16 | pass | pass | deleted | 0 | 654,324,959 |
| sub-010193 | complete / 16 | pass | pass | deleted | 0 | 655,068,959 |
| sub-010194 | complete / 16 | pass | pass | deleted | 0 | 673,011,759 |
| sub-010195 | complete / 16 | pass | pass | deleted | 0 | 658,354,959 |
| sub-010196 | complete / 16 | pass | pass | deleted | 0 | 662,136,959 |
| sub-010197 | complete / 16 | pass | pass | deleted | 0 | 621,638,559 |
| sub-010199 | complete / 16 | pass | pass | deleted | 0 | 630,876,559 |
| sub-010200 | complete / 16 | pass | fail | deleted | 0 | 681,952,159 |
| sub-010201 | complete / 16 | pass | pass | deleted | 0 | 649,687,359 |
| sub-010202 | complete / 16 | pass | pass | deleted | 0 | 629,450,559 |
| sub-010204 | complete / 16 | pass | pass | deleted | 0 | 678,467,759 |
| sub-010207 | complete / 16 | pass | pass | deleted | 0 | 626,499,359 |
| sub-010210 | complete / 16 | pass | pass | deleted | 0 | 660,748,159 |
| sub-010213 | complete / 16 | pass | pass | deleted | 0 | 657,164,559 |
| sub-010214 | complete / 16 | pass | pass | deleted | 0 | 660,475,359 |
| sub-010215 | complete / 16 | pass | pass | deleted | 0 | 672,850,559 |
| sub-010218 | complete / 16 | pass | pass | deleted | 0 | 641,664,559 |
| sub-010219 | complete / 16 | pass | pass | deleted | 0 | 625,494,959 |
| sub-010220 | complete / 16 | pass | pass | deleted | 0 | 647,703,359 |
| sub-010222 | complete / 16 | pass | pass | deleted | 0 | 633,195,359 |
| sub-010223 | complete / 16 | pass | pass | deleted | 0 | 634,522,159 |
| sub-010224 | complete / 16 | pass | pass | deleted | 0 | 661,231,759 |
| sub-010226 | complete / 16 | pass | fail | deleted | 0 | 647,963,759 |
| sub-010227 | complete / 16 | pass | pass | deleted | 0 | 662,124,559 |
| sub-010228 | complete / 16 | pass | pass | deleted | 0 | 662,236,159 |
| sub-010230 | complete / 16 | pass | pass | deleted | 0 | 688,896,159 |
| sub-010231 | complete / 16 | pass | pass | deleted | 0 | 679,434,959 |
| sub-010232 | complete / 16 | pass | pass | deleted | 528,482,304 | 148,051,055 |
| sub-010233 | complete / 16 | pass | pass | deleted | 0 | 675,119,759 |
| sub-010234 | complete / 16 | fail | fail | retained: Train QC | 0 | 620,410,959 |
| sub-010236 | complete / 16 | pass | pass | deleted | 0 | 647,690,959 |
| sub-010238 | complete / 16 | pass | pass | deleted | 0 | 675,119,759 |
| sub-010239 | complete / 16 | pass | pass | deleted | 0 | 650,852,959 |
| sub-010240 | complete / 16 | pass | pass | deleted | 0 | 661,306,159 |
| sub-010241 | complete / 16 | pass | pass | deleted | 0 | 632,600,159 |

sub-010146, sub-010163, and sub-010234 failed frozen Train QC and retain raw. sub-010157, sub-010200, and sub-010226 passed Train QC but have ineligible assigned provisional blocks; their raw deletion followed the existing Train-QC cleanup rules. These are mechanical eligibility outcomes, not scientific held-out analyses. No replacement or final cohort decision was made.

## Cleanup and compact storage

All 71 Batch 004 Train-QC-passing subjects have durable verified raw deletion. Of those, **59 deletions occurred during this continuation**, freeing **38,533,463,430 bytes** including sidecars. Whole-Batch 004 cleanup freed **46,323,778,551 bytes**. The three new retained raw subjects total **1,957,451,031 bytes**. Raw deletion followed verified output/receipt, durable completion, and recorded cleanup intent; no failed-QC raw was deleted.

| Retained subject | Bytes | Reason |
|---|---:|---|
| sub-010012 | 648,429,341 | prior preserved |
| sub-010030 | 645,403,807 | prior preserved |
| sub-010146 | 677,023,722 | Batch 004 frozen Train-QC failure |
| sub-010163 | 659,972,502 | Batch 004 frozen Train-QC failure |
| sub-010234 | 620,454,807 | Batch 004 frozen Train-QC failure |
| sub-010250 | 651,330,882 | prior preserved |

| Storage metric | Bytes |
|---|---:|
| All-block HDF5 added by the 62-subject continuation | 452,579,237 |
| All-block HDF5 added by all 74 Batch 004 subjects | 540,243,726 |
| Total processed storage added during continuation, including Train references and receipts | 737,239,223 |
| Cumulative all-block HDF5 | 1,066,215,801 |
| Cumulative processed storage including Train references and receipts | 1,736,840,551 |
| Cumulative retained raw | 3,902,615,061 |

Signal, temporary, cache, and runtime files remain on external D:. Disk guards remained active; no pagefile changes were made. At final worker verification, C: had approximately 13.41 GiB free and D: 42.28 GiB free. These are snapshots, not claimed minimums over the entire run.

## Supervisor verification and validation

The detached scientific worker remained independent of the original launch shell throughout all 62 subjects. The OS supervisor and execution locks were verified held. Worker heartbeats updated during downloads and processing; stdout/stderr stayed on external D:. No service, scheduled task, or STOP sentinel was created.

The original parent compared its Windows venv child-launcher PID (12276) with the actual interpreter PID (30080), rejected valid worker snapshots, and reported an incorrect hard-blocked status after exit 0. This affected parent reporting only; the detailed worker heartbeat and scientific run were correct. Both original terminal snapshots are preserved. After processing ended, the correction was tested and committed; it matches a fresh per-run supervision token, reports both PIDs, validates state against exit code, and clears completed asset sizes.

Two post-completion verification workers derived an empty unresolved list and completed without EEG requests or processing. During the first check, an unsupported `os.kill(pid, 0)` liveness probe terminated the parent on Windows; its child still completed. This verification incident is preserved in the audit. The repeated check used read-only PowerShell process inspection and passed: detached parent PID 22568 was alive after launcher return, the OS lock rejected another holder, its heartbeat advanced, and it accepted actual worker PID 21624 / launcher PID 17632 with `batch_complete`, exit 0, and 74/0 completed/unresolved counts. **588 journal, registry, HDF5, and receipt files remained byte-identical across both checks**. All runner processes have exited after completion.

Pre-network validation passed **344 offline tests**, including 45 added recovery cases, plus compileall. Final validation passed **355 tests in 38.01 seconds**, including 56 recovery/supervision cases, plus compileall. Tests used D-only temporary storage and mocked/local responses, with no real EEG requests. The full read-only Batch 004 verifier and autonomous preservation/accounting audit passed. The final clean-commit verifier result, exact final Git HEAD, hashes of this handoff and evidence, and terminal acceptance state are stored in the external final acceptance record linked above.

The scientific preprocessing, filters, resampling, edge guard, channel order, bad-channel/QC/interpolation/CAR/VEOG rules, block eligibility, provisional split, and HDF5 representation remain frozen. Subject parameter fitting used Train only; held-out application stayed mechanical. No PSD, Alpha, IAF, topology, connectivity, wPLI, classifier, generator, TSTR, augmentation, or waveform plots were computed or inspected. **No normalization was fitted, no final cohort or split was locked, no Batch 005 was planned, and nothing was pushed.**

The earlier [Batch 004 stopped-state handoff](BATCH_004_STREAMING_HANDOFF.md) remains unchanged as historical evidence. This handoff records the completed user-authorized continuation.
