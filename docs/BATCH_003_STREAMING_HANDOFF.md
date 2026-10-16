# Batch 003 terminal handoff

**BATCH_003_STREAMING_COMPLETE**: all 33 frozen subjects have durable terminal attempt states. **31 completed technically; 2 remain unresolved download failures.** The batch ledger remains `failed_or_partial`; this handoff does not claim all 33 were processed. Network execution stopped after the last terminal state and exited with code 1. No further request, replacement, or Batch 004 planning occurred.

## Git state and durable authority

- Starting clean HEAD: `28815a4a3245aacd36aea241a17cf6458f459ef4`.
- Tested pre-network implementation and frozen plan: `3e1d63b7557393c0a4f82a628b39c27c261f4f3b`.
- Tested terminal verifier, compact results, registries, and final README/protocol/decisions: `2e4466c587a64021477362575ddf603d9c339bb2`.
- Final HEAD is the commit containing this handoff. Resolve with `git rev-parse HEAD`; the exact hash and clean-worktree acceptance are recorded after commit in external `runtime/batches/batch_003_final_acceptance.json`. A committed file cannot contain its own commit hash.

No amend, history rewrite, or Git push occurred. The earlier Batch 001/002 handoffs remain unchanged. All 39 prior all-block outputs and receipts, prior Train references, access rows, both earlier ledger entries, and three retained raw subjects were verified before I/O and again afterward against the 222-artifact starting-state baseline.

Plan freeze: `2026-09-05T17:19:21.571085+00:00`. Durable execution authorization: `2026-09-05T17:23:56.778190+00:00` under code HEAD `3e1d63b7557393c0a4f82a628b39c27c261f4f3b`. The legacy `authorization_head` field in the plan/summary denotes the starting freeze HEAD, not the execution HEAD. The committed preflight was refreshed at authorization before any request; the authorization pins the refreshed preflight hash.

## Exact capacity and frozen membership

The user selected exactly **20 GiB = 21,474,836,480 bytes** for Batch 003 only. One GiB is 1,073,741,824 bytes. Membership counts each complete declared `.eeg` + `.vhdr` pair, independently of any existing bytes. Verified Pilot/Batch 001/002 outputs and previously assigned subjects were excluded. The planner preserved authoritative contiguous subject order and stopped at the first complete subject that did not fit; it did not bin-pack or add replacement subjects.

| Order | Subject | Declared bytes | Assets | Terminal result | Train QC | Provisional 5/1/2 | Raw |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 1 | sub-010035 | 633,988,959 | 2 | complete | pass | pass | deleted |
| 2 | sub-010036 | 644,789,359 | 2 | complete | pass | pass | deleted |
| 3 | sub-010037 | 623,002,559 | 2 | complete | pass | pass | deleted |
| 4 | sub-010038 | 667,568,159 | 2 | complete | pass | pass | deleted |
| 5 | sub-010039 | 624,614,559 | 2 | complete | pass | pass | deleted |
| 6 | sub-010040 | 622,952,959 | 2 | complete | pass | pass | deleted |
| 7 | sub-010041 | 629,239,759 | 2 | complete | pass | pass | deleted |
| 8 | sub-010042 | 639,841,759 | 2 | complete | pass | pass | deleted |
| 9 | sub-010044 | 641,342,159 | 2 | complete | pass | pass | deleted |
| 10 | sub-010045 | 630,355,759 | 2 | complete | pass | pass | deleted |
| 11 | sub-010046 | 624,775,759 | 2 | complete | pass | pass | deleted |
| 12 | sub-010047 | 623,597,759 | 2 | complete | pass | pass | deleted |
| 13 | sub-010048 | 626,586,159 | 2 | complete | pass | pass | deleted |
| 14 | sub-010049 | 629,016,559 | 2 | complete | pass | pass | deleted |
| 15 | sub-010050 | 644,776,959 | 2 | complete | pass | pass | deleted |
| 16 | sub-010051 | 630,727,759 | 2 | complete | pass | pass | deleted |
| 17 | sub-010052 | 695,939,359 | 2 | complete | pass | pass | deleted |
| 18 | sub-010053 | 633,567,359 | 2 | complete | pass | pass | deleted |
| 19 | sub-010056 | 647,033,759 | 2 | complete | pass | pass | deleted |
| 20 | sub-010060 | 651,510,159 | 2 | complete | pass | pass | deleted |
| 21 | sub-010061 | 699,994,159 | 2 | complete | pass | pass | deleted |
| 22 | sub-010062 | 640,846,159 | 2 | complete | pass | pass | deleted |
| 23 | sub-010063 | 664,678,959 | 2 | complete | pass | pass | deleted |
| 24 | sub-010064 | 671,734,559 | 2 | complete | pass | pass | deleted |
| 25 | sub-010065 | 663,451,359 | 2 | complete | pass | pass | deleted |
| 26 | sub-010066 | 637,374,159 | 2 | complete | pass | pass | deleted |
| 27 | sub-010067 | 662,608,159 | 2 | complete | pass | pass | deleted |
| 28 | sub-010068 | 656,742,959 | 2 | complete | pass | pass | deleted |
| 29 | sub-010069 | 641,069,359 | 2 | complete | pass | pass | deleted |
| 30 | sub-010070 | 663,860,559 | 2 | complete | pass | pass | deleted |
| 31 | sub-010072 | 639,692,959 | 2 | complete | pass | pass | deleted |
| 32 | sub-010073 | 625,544,559 | 2 | download_interrupted | not assessed | not assessed | retained partial |
| 33 | sub-010074 | 666,749,759 | 2 | download_interrupted | not assessed | not assessed | retained empty directory |

**33 subjects / 66 declared assets; 21,299,575,247 bytes (19.836775257 GiB).** Unused capacity: **175,261,233 bytes (0.163224743 GiB)**. First unassigned candidate: **sub-010075**, whose 665,373,359 bytes do not fit. It has no Batch 004 assignment.

## Frozen hashes

| Artifact / hash semantics | SHA256 |
| --- | --- |
| Batch 003 manifest file | `d8299bdde9d13186970d9b717f7f0a8fbcb252f2115a8b7d591e2e561e61c723` |
| Batch 003 canonical plan | `14884fe2ef6aceabd6167f156c887bf1bd5343ab6099cd76ee8d9c6d5531c56a` |
| Batch 003 plan file bytes | `75ae3ff8874f03284e6ff492b7e9a720f20592010d5876fff3c6fc9eebaf60ac` |
| Execution authorization file | `2626af44ec928b1598b5bfa1c97b8b3c0b5ced1997e290e39a47f345554befc5` |
| Authorization network preflight | `e8dfddcc1b8f7280560fc553dfea08fbb55fa58b0da055ce96647f2dcf8d1b10` |
| Starting-state baseline | `bda9d82e81f5c67d9853d8d7b5ac8941fe6dc6ba8445caa93078817b6066d7ad` |
| Frozen preprocessing canonical configuration | `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b` |
| Frozen mechanical-access canonical configuration | `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4` |
| Variable batching configuration | `41d53f288786880d9a92ea202a558abe0fa5030fb8a3a872c275e959e6048ced` |

Scientific source/configuration, QC thresholds, block allocation, and signal-processing rules are unchanged. Prior evidence preservation is enforced by `batch_003_starting_state.json` and the generalized prior-state verifier.

## Network accounting and stopped failure sequence

| Measure | Bytes | GiB |
| --- | ---: | ---: |
| Declared complete pairs | 21,299,575,247 | 19.836775257 |
| Initially verified local payload | 0 | 0.000000000 |
| Initial partial prefixes | 0 | 0.000000000 |
| Initially expected new response payload | 21,299,575,247 | 19.836775257 |
| Actually measured response-body payload | 20,409,934,113 | 19.008232386 |
| Successfully completed pair payload | 20,007,280,929 | 18.633232386 |
| Preserved incomplete prefix | 402,653,184 | 0.375000000 |
| Validated reused resume-prefix bytes | 0 | 0.000000000 |
| Untransferred declared payload | 889,641,134 | 0.828542871 |

Measured bytes count application response-body bytes, including bytes received on interrupted attempts. HTTP/TLS/ISP overhead and bytes not returned to the response reader are not measured. The 384-MiB partial is included in actual transfer; it is not a completed source asset or successfully resumed prefix.

First download journal event: `2026-09-05T17:24:29.276312+00:00`; last download journal event: `2026-09-05T19:36:12.409601+00:00`. Detailed request and progress records remain external. There were **6 scheduled retries, 8 download interruption events, 2 terminal subject failures, and 0 manual restarts / numbered resume authorizations**.

- **sub-010073:** received 402,653,184 bytes by `2026-09-05T19:34:55.635746+00:00`; initial read timeout at `19:35:57.834187` UTC. Its three retry requests proposed offset 402,653,184, but all failed DNS resolution before a response could validate HTTP 206 / Content-Range. Final subject terminal state: `2026-09-05T19:36:04.856293+00:00`.
- **sub-010074:** initial request and three retries failed DNS resolution before any response bytes. Final subject terminal state: `2026-09-05T19:36:12.410601+00:00`.
- The eight interruption events comprise one read timeout and seven DNS failures (`<urlopen error [Errno 11001] getaddrinfo failed>`). The process completed its attempt sequence and wrote the initial summary at `2026-09-05T19:36:22.695279+00:00`. The post-stop summary adds explicit assessed/terminal accounting; the initial summary is preserved externally as `batch_003_initial_terminal_summary.json`.

The two failures remain assigned to Batch 003 and are unassessed for Train QC and feasibility. Their CSV booleans must be read with `train_qc_assessed` and `provisional_feasibility_assessed`; false on an unassessed row is not a scientific QC failure. The verifier accepts these durable preserved download failures while continuing to reject unverified processing failures.

## Canary, processing, and leakage audit

The first subject, sub-010035, passed the canary at `2026-09-05T17:27:56.067452+00:00` before subject 2 began. Source hashes, sidecars and BrainVision linkage, native channels/rate/sample count/block coverage, all 16 assigned blocks, Train-only fitting, held-out apply-only behavior, receipt, completion journal, and raw cleanup verified.

The stream staged one active subject at a time. Each of the 31 completed outputs contains all 16 assigned blocks, including masks and reasons for ineligible blocks, with fixed 19-channel order, 128 Hz, float32 microvolts, and 512-sample windows / 256-sample stride. Frozen 1-40 Hz FIR and 433-sample edge guard remain unchanged. Subject-specific bad-channel/artifact parameters were fitted from provisional Train only, then applied unchanged to Validation/Test. Source and output hashes, split identities, provenance, and durable receipts were reopened and verified before cleanup.

Batch 003 adds 310 Train access rows and 186 mechanical held-out rows. Cumulatively there are 550 streaming Train rows, 150 unchanged Pilot Train rows, and 420 held-out rows. The two failed subjects have no preprocessing outputs or access records. All 31 technically completed subjects pass frozen Train QC and provisional 5/1/2 feasibility. No assessed Batch 003 subject fails either; no thresholds or allocations were changed to rescue subjects.

No PSD, Alpha, IAF, topography, connectivity, wPLI, classifier/generator evaluation, TSTR, augmentation analysis, or held-out waveform plots were performed. No held-out refitting, numerical normalization fit, final cohort declaration, or final split lock occurred. The audit checks application source, output provenance, and access registries; it is not OS-wide surveillance.

## Cleanup, retained raw, and disk usage

The 31 successful subjects in the membership table had their raw directories deleted immediately after full technical verification, durable completion, and Train-QC pass. Verified absent directories freed **20,008,642,245 bytes (18.634500210 GiB)**, comprising 20,007,280,929 payload bytes and 1,361,316 sidecar bytes. Held-out infeasibility alone would not have blocked otherwise eligible cleanup; it did not arise among these 31 subjects.

| Retained subject | Bytes | Reason |
| --- | ---: | --- |
| sub-010012 | 648,429,341 | Unchanged prior Train-QC failure; complete compact output |
| sub-010030 | 645,403,807 | Unchanged prior Train-QC failure; complete compact output |
| sub-010073 | 402,653,184 | Unresolved download; 384-MiB partial |
| sub-010074 | 0 | Unresolved download; empty directory |
| sub-010250 | 651,330,882 | Unchanged prior Train-QC failure; complete compact output |

Batch 003 retained raw is **402,653,184 bytes (0.375 GiB)** in two directories, including one empty directory. Total retained raw is **2,347,817,214 bytes (2.186575172 GiB)**. `batch_003_storage_verification.json` enumerates exact sizes and SHA256 for every retained file. The three old complete source pairs were rehashed; every retained file and the current directory set are checked by the final verifier.

Retained prefix: `raw/sub-010073/eeg/sub-010073_task-resting_eeg.eeg.part`.
SHA256: `72df8eb5246c8bff83eb2c576427faaf850e05870a374bb529b203e87cf3f233`.
This hash identifies the incomplete prefix only. The expected full EEG is 625,542,800 bytes with SHA256 `9024d86996460bd58d11bb5dcfa5d61fa104523f549b8869bf00f0df69abcd40`; future continuation must verify the complete file before promotion.

Batch all-block HDF5 addition: **226,408,568 bytes (215.920036 MiB)**. Cumulative all-block HDF5: **511,428,035 bytes**. Preserved Train reference HDF5: **319,542,416 bytes**. Total processed directory including receipts: **833,108,692 bytes (0.775892932 GiB)**.

| Snapshot | C: free bytes (GiB) | D: free bytes (GiB) |
| --- | ---: | ---: |
| Authorization preflight | 14,483,337,216 (13.488658905) | 48,674,209,792 (45.331390381) |
| Storage verification 2026-09-05T20:05:07.549351+00:00 | 13,969,948,672 (13.010528564) | 47,890,272,256 (44.601291656) |

Largest declared subject: 699,994,159 bytes (sub-010061). Conservative preflight compact allowance: 1,107,296,256 bytes (32 MiB per subject, including temporary allowance). Free-space snapshots are volume-wide and can vary independently of this task; raw bytes freed come from verified directory inventories, not free-space subtraction. Final current snapshots are also recorded in external acceptance.

## Compact output hashes

All outputs live under external `processed/all-blocks-v1/<subject>_all_blocks.h5`; every output has 16 blocks. `batch_003_output_manifest.csv` records full paths, byte sizes, output SHA256 and receipt SHA256. No output exists for either download failure.

| Subject | HDF5 bytes | HDF5 SHA256 | Receipt SHA256 |
| --- | ---: | --- | --- |
| sub-010035 | 7,245,980 | `acef24e2581c653bf62685e74bc001911babe5894c92249fb283e625458cebe6` | `0ca422fcab6152c64a24c4aecc2c552caab308b80c69bc45cf8447f82f50db5e` |
| sub-010036 | 7,343,833 | `b91619cfc6f1f2f47cbd88177c112b7be32d5afc893e6b0bbbd681a641f1fdf1` | `42aaaa746a4df0a0c2c069fa8452a1724d5e5810969bfb8aef959b6d14d0962f` |
| sub-010037 | 7,365,246 | `b2c5f3486807b1c1d050abb783c16218905c9f7096fe609ea983444c227d8e47` | `aa1fd14b67b4abf9a8bab22e12f32539613e8fe1901543c6b6aaaf79d294625d` |
| sub-010038 | 7,288,963 | `eb4388c829096d5b4d0a870e9c60cd0a96568809b0a98482c9d4b6298021e440` | `8073cfe354190a5c90e37b47963bb7eac63d1bb4147b4eaa9c12d398c6c1ebcb` |
| sub-010039 | 7,371,322 | `0daf9595511593345ca5a9613963c88b0e61fdf65ceeb6e7a55d74009dae360c` | `4cd5dacedccaeee0f2da209ee25fda03d910c314bbe84a10b2ac41336eb919c3` |
| sub-010040 | 7,235,619 | `9269fd3046ef4a88ad174cf402775fba58cf3b1c301417f59063085f7492b459` | `97f1fa7a932e51eeaf6046b0fe9cefd8513fc200639a410bf4c75feee8d011af` |
| sub-010041 | 7,296,601 | `353d502928611e31228b7314969600d6b18791833d01578051da6b2b36c2e9c1` | `fcb262fc370d1d060b480c1935e1f03e25554cc5aad4970ac6a6ed8bbc04c510` |
| sub-010042 | 7,320,015 | `538c5ee16abc53bc807c148747e00add3df74b9b527b84e96765c2459a26bbb3` | `9a9affa67b7ee2e14219f702b3e65bdfd872ecbf0880ab6a7e55555dc0ad8cc8` |
| sub-010044 | 7,376,879 | `14a085dded5ddb0f632c511e5f78ae10ec8b0f099025f557a3d52c489e41ada9` | `861de6c05cfde21c398db1c9df8b5b5a7b79bebe9ec030b90c22d2d2972f7517` |
| sub-010045 | 7,374,768 | `541b9dc9403b175c7ea5a33c5ea66cf3831bb37f49ec011f1e17c515e61ededd` | `184e1fea67c5210ad2fa0a4d83bf9c23411a90f2d8afac2c9cabc0c9ecb92db8` |
| sub-010046 | 7,345,091 | `962acb42d10269ac1d35c0785c403c131bffe9740a008544add1edd5c8bdadbf` | `00ddaa56e7f0dcc67722b67a4b0a04cf5be5f1a4ec516130671acbb68d8334dd` |
| sub-010047 | 7,334,949 | `7437843d5abc50c3ce42a9129a62877d48aaa2b0890c04743f8a60b963a4eed6` | `5fdd41c016e27f506e36c80afdf0c5fd070e3a973f0f1bb7e6cfbd2cc5e5cbaa` |
| sub-010048 | 7,375,422 | `b04e55e7066e7981992634dace5a5e64049dfe2c9ad78943cc7041ceba1ebac9` | `b47d9d405a6c44cb3f68cdc1ecd8e05d3b4a9ab477bf287eef301c967840e548` |
| sub-010049 | 7,357,964 | `fb436156c351eb0be8f542121e663d189383e5824f16782a4b1caac4f1419cb4` | `d7f3c9dcde891ad9f7f667ce394afa69327793086e26beea68969c027300407a` |
| sub-010050 | 7,337,963 | `cfc99b65199cd0ec66eb6afd1e015be42084f0511e4fd320b2b08a1bd38e4b3f` | `4f62b11e05def1bbcfd37fd63ccf999a4a7baa4c21dfda300ece939a0629da59` |
| sub-010051 | 7,291,163 | `24f0a551cd7e3279b6ca49bcac5c7ddfb3fe7064e75c6efdea33383330fcdb3c` | `0fd17604952c06f56b56b4f319d49cc4a677bcd1cae8ef6a55032555c56f438d` |
| sub-010052 | 7,279,000 | `30685339b039602d48e788234c5d17e322679c40835a626cac61c685b83d61e7` | `936846884d808b873c13e4c4db468781358221c15ec17bc1fe1fb0e8a45d8eb9` |
| sub-010053 | 7,308,929 | `c24ee1bd758d6807085d031341877ae62599121552b0f6f86dab3f8298d9ca6e` | `c486a163032e2f8fc5329989841237dab4c9ce088edc41279117ff807521bc97` |
| sub-010056 | 7,266,486 | `12b71eacddda4d52758289a66cd630c8e81c56ca67847861476885b2559bc7b2` | `41d9dd6dab0c43f7e6b05968ddb3e10ef4731814cfcd91868c48d42b18d51ae2` |
| sub-010060 | 7,229,740 | `7949aa8e791e0f5e1ec38ae082cf52862a9a425b8a95608766f72bc9c8c7f206` | `9fe6439a883dd997087592e4f6b2caad221c6e7f0c451790bd859c3e0149ed08` |
| sub-010061 | 7,329,451 | `0f319ec9114a5ad4961744bf3312d3348945f1bf0e0c34014b4b1f2fc4aa2a5c` | `f655113aa135f207b3235c5c301741a3174356de6ba0899f0ee9d051c928ecad` |
| sub-010062 | 7,228,060 | `4470a1d51aaa1580c9e08ea8b11dbbf55831e6f67585da2191e53d5427de37e1` | `ef0bff2cd7d3590b3e3db9700a57acab4a1bc25f47a934c9725662e8679af780` |
| sub-010063 | 7,257,902 | `24d8aac93fc6a5909642b46d779dbf2396422329fe229754feafd1c434d0b203` | `afed4608d9e4c4bee19edd23283c7acb7d37caf807f821fb5a3531493e9194b6` |
| sub-010064 | 7,332,819 | `27ea06dc6db279a9843b079d00c08cc75b13e882b3a3571f2dd5380b0a7d1451` | `1f9bc77a84e5aecc9ca6a07ad8a29dcded082ebab21185d18c032680a276979f` |
| sub-010065 | 7,236,746 | `f4ae9dd97477dd2175a3a7cb7a4faa214ea4b8810f34142754d9a4045f8ebefa` | `a92e1f221091b8cc44303ee35b603950a95d45cb6a5ff0f6e9f1b0124f77b81f` |
| sub-010066 | 7,309,814 | `1132299d80db571a32b662965f640eea0d8c90ab1413ee6da4949130921c35b2` | `2d28143da76cc79b64b523add36e30daa44602d3fbd7271ed51be4d55b0dca45` |
| sub-010067 | 7,255,178 | `276b51c271c44d78adcfa7953188ea5decead39cbad4b5956bf28f1e71d9bbee` | `6b5fa92acef745bd323e21305bed27484d74e3d9b98206e133fe455308ce8082` |
| sub-010068 | 7,310,983 | `cade7826262c9360551e318969130eaf5723406f1aae783e54c6d2f749d4ca04` | `1a109b2825c71c962f3a293b310bcd3634a5c5457a9c0a15f4afe164da89fa18` |
| sub-010069 | 7,301,847 | `65a57d9811795790c17d46614a57d7d57d66cb230920e23b19e9c1024cc6ba66` | `0c7a0d56cb8237f61bbc676cbb30870f53909f04ecd6288152c0b0b523a775b5` |
| sub-010070 | 7,216,852 | `2fe5eca53a5959eda6342ce3a397ecfeb33c9a51912d31b5b389bb73e4b44ecd` | `97b827d2854e856b0ae2a9d4e435ac7827a7faf421e54fb9b6032b76648e1507` |
| sub-010072 | 7,282,983 | `519723fd10b4949d81d691006327400ccf5b26cebbe3403f9a68e7bd3961739f` | `a05f04983a43c01638586888377e0ed1b84fe9f23120ab307515dea9f5428102` |

## Cumulative progress

**70/211 technically complete subjects**, storing **1,120 blocks** (700 Train and 420 held-out). Cumulative frozen Train-QC pass: **67/70**. Cumulative provisional-feasibility pass: **66/70**. These are engineering labels, not a final cohort. There are **139 unassigned candidates**, starting with sub-010075, and **2 assigned incomplete Batch 003 subjects**, so 141 structural candidates lack completed all-block outputs.

## D-only environment and evidence locations

External root: `D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2`. Raw, processed, temp, caches, and detailed runtime journals stay beneath this root. Metadata remains the unchanged read-only clone `D:/SHARIF/PROJECT/Datasets/MPI-LEMON`. Small versioned code/configuration/metrics/docs remain in the repository. No raw/processed signal files are tracked by Git.

Before every Python invocation, dot-source `scripts/setup_scenario2_environment.ps1`; use `-ForTests` for pytest. Stop before download, preprocessing, or tests if C: free space is below **3 GiB = 3,221,225,472 bytes**. The guard remained active; no pagefile settings changed.

| Environment entry | Recorded value |
| --- | --- |
| TEMP | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp` |
| TMP | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp` |
| TMPDIR | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp` |
| PIP_CACHE_DIR | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp\pip` |
| MPLCONFIGDIR | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp\matplotlib` |
| JOBLIB_TEMP_FOLDER | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp\joblib` |
| PYTHONPYCACHEPREFIX | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp\python` |
| tempfile.gettempdir() | `D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp` |

Detailed runtime files remain under external `runtime/batches`:

| Journal | Bytes | SHA256 |
| --- | ---: | --- |
| batch_003_download_journal.jsonl | 1,358,840 | `a2120c9aeace4afc9845b58041a12058990664f255349368de4c9da1b8877977` |
| batch_003_streaming_journal.jsonl | 2,182,164 | `faa1ba8210ed23344116f3d342bdebb49c8c6af31b66271591978866d80bea4b` |

Compact evidence under `outputs/metrics/scenario2/batches/` includes the starting-state baseline, manifest, plan, preflight, execution authorization, canary verification, subject results, cleanup results, output manifest, summary, storage verification, validation, and immutable-ledger membership. Detailed journals and original terminal summary remain external. There is no Batch 004 manifest, plan, authorization, journal, request, or selected cap. Synthetic test fixtures were removed after pytest exited; no real data was removed by that fixture cleanup.

## Tests and final acceptance

**Pre-network:** 284 tests passed in 64.42 seconds; compileall passed before local implementation commit and durable authorization. **Final:** 291 tests passed in 49.89 seconds (JUnit suite time 49.856 seconds), zero failures/errors/skips; compileall passed. Final JUnit SHA256: `f09d3b9406e26d01552cd82f77f7ba0007bcb5ea7792b22a294f80fe3193adfb`.

Tests cover exact 20-GiB conversion and contiguous membership, two-batch immutability, duplicate/future assignment rejection, D-only storage and C: guard, Range resume and bounded retries, sequential processing, cleanup/retention, Train-only fitting and held-out apply-only behavior. Seven added terminal-evidence cases distinguish completion from preserved failure and reject missing/mismatched/unsafe terminal evidence. The final verifier rehashes retained files and verifies all successful HDF5/receipts, source provenance, all blocks, frozen hashes, prior outputs, registries, request membership, absent successful raws, and no Batch 004.

The compact validation records the pre-results-commit verifier pass with an expected dirty worktree. The authoritative final clean-worktree pass is recorded externally after the handoff commit, with the exact final HEAD. Both explicitly report 33 terminal subjects, 31 successful subjects, and two technical failures.

## Exact verification and conditional restart instructions

Remain stopped now. To inspect the completed attempt sequence without network:

```powershell
Set-Location "D:/SHARIF/PROJECT/subject-conditioned-resting-eeg"
git status --short
git rev-parse HEAD
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 003 --require-clean --test-report outputs/logs/pytest.xml
```

If tests need rerunning, first confirm no other pytest process is using the shared base:

```powershell
. .\scripts\setup_scenario2_environment.ps1 -ForTests
.\.venv\Scripts\python.exe -m pytest -q --basetemp "D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/temp/pytest" --junitxml outputs/logs/pytest.xml
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe -m compileall scripts src tests
```

Only after a later explicit user continuation and restored connectivity, retry the two assigned failures under the same frozen Batch 003 membership/cap. Do not regenerate the plan or use the planner for a replacement. Reverify preserved evidence and require a clean tested commit first. Create a numbered resume authorization because the post-stop committed verifier code differs from the original execution HEAD. The commands below were **not executed**:

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 003 --batch-manifest outputs/metrics/scenario2/batches/batch_003_manifest.csv --authorize-resume
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --processing-scope all-blocks-mechanical --batch-id 003 --batch-max-gib 20 --batch-manifest outputs/metrics/scenario2/batches/batch_003_manifest.csv --execute-batch --delete-raw-after-success --continue-on-error --retries 3 --timeout 60 --subject-id sub-010073 --subject-id sub-010074
```

The new authorization must preserve the original authorization and freeze hashes. The 31 current and 39 earlier verified completions must not be downloaded or reprocessed. Recheck the partial and disk guards; Range reuse requires correct HTTP 206 / Content-Range and final whole-file size/SHA256. If the server ignores Range, the downloader must use its existing safe restart behavior. Current untransferred payload is 889,641,134 bytes if the prefix can be reused, but a fresh preflight must determine expected transfer at that time. Keep the two failures assigned even if continuation fails again. Do not fit normalization or perform scientific held-out analysis.

## NEXT ACTION — CHOOSE BATCH 004 SIZE

Batch 004 remains unplanned and its size is undecided. Wait for the user to select and authorize the next step. The two Batch 003 download failures remain assigned and unresolved; address them only under a separate continuation. Do not choose a Batch 004 cap, generate membership, or make further network requests automatically.
