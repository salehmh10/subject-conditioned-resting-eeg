# Train-only normalization handoff

Date: 2026-09-07. Starting branch `main`, clean accepted HEAD `9fa49dba1fda8fd0e37852ccdce642372aecca70`.

## What is frozen

Exactly one 19-channel global scaler pools all included subjects, EO and EC, and all eligible final Real Train blocks. Means and population standard deviations use float64 centered streaming moments, ddof=0, in physical microvolts. All standard deviations exceed the frozen 1e-6 uV threshold. The single scaler must later be applied unchanged to Real Train, Real Validation, Real Test and Synthetic EEG. No separate subject/state/block/window scaler is permitted.

The processed HDF5 files remain unmodified physical-unit data; their historical `normalization_fitted=false` attributes describe their stored values. The new scaler is a separate locked artifact. Historical preservation-verifier `no_normalization`/`normalization_unfitted` fields continue to describe those preserved preprocessing artifacts, not the existence of this new global scaler.

## Exact fitting population and sample rule

Only final included subjects and final Train assignments enter the reader. Every source HDF5 and receipt is checked against the accepted hash bundle before and after reading. Metadata validates subject/block identity, Train role, eligibility, 19-channel order, units, sampling rate, window geometry and frozen mask digest. No QC, masks, eligibility, cohort or assignment is recalculated or changed by the normalization fitter.

At 128 Hz, each usable window defines `[start, start+512)`, with 256-sample stride. Fit from the union within each eligible block; never bridge blocks. Shared overlap contributes once. The ten ineligible assignments remain in the CSV and contribute zero samples and zero fitting windows. Unique sample count means time samples per channel, not the sum over 19 channels.

No Validation/Test, excluded-subject or ineligible-block EEG dataset is indexed by fitting or numerical verification. Whole-file hashing is opaque integrity access. The older preservation verifiers inspect frozen metadata/provenance under their established rules. Raw EEG, preprocessing, scientific held-out outcomes and model training are outside this phase.

## Measured Train volume

| Measure | Total | EO | EC |
|---|---:|---:|---:|
| Included subjects | 198 | 198 | 198 |
| Assigned Train blocks | 1,980 | 990 | 990 |
| Eligible Train blocks | 1,970 | 981 | 989 |
| Ineligible Train blocks | 10 | 9 | 1 |
| Candidate windows, all assignments | 49,500 | 24,750 | 24,750 |
| Mask-usable windows, all assignments | 47,898 | 23,365 | 24,533 |
| Fitting windows, eligible blocks only | 47,835 | 23,302 | 24,533 |
| Unique samples used | 12,881,152 | 6,330,880 | 6,550,272 |
| Unique usable seconds | 100,634 | 49,460 | 51,174 |
| Unique usable hours | 27.95388888888889 | 13.738888888888889 | 14.215 |
| Mask-usable / candidate fraction | 0.9676363636363636 | 0.944040404040404 | 0.9912323232323232 |

The 63 mask-usable windows in frozen-ineligible blocks (all EO) are retained as bookkeeping but excluded from fitting. Summing 47,835 overlapping window lengths would incorrectly count 24,491,520 samples; the union actually contains 12,881,152.

Per-subject usable Train duration in seconds: **minimum 380; median 516; mean 508.25252525252523; maximum 520**. Individual exact sample totals are recorded in the scaler JSON. These are descriptive accounting values and did not change QC or membership.

## Channel scaler

Values below are in uV; standard deviations are population values (ddof=0). JSON retains the complete serialized float64 values.

| Channel | Mean uV | Std uV |
|---|---:|---:|
| Fp1 | -0.005891625469900178 | 15.125589302061588 |
| Fp2 | -0.005820589596257181 | 12.263085007088163 |
| F7 | -0.0017497464536473809 | 8.233109267184638 |
| F8 | -0.0018215679994339705 | 7.54598447773888 |
| F3 | -0.0012126294594818248 | 6.395330247206566 |
| F4 | -0.0009279142734871015 | 6.179091880765566 |
| Fz | -0.0005139864181169895 | 6.001615724855165 |
| T7 | 0.0004514947991348889 | 6.441616013997513 |
| T8 | 0.0010175431366221603 | 6.318878036341845 |
| C3 | 0.0006501051713423887 | 4.581889271034589 |
| C4 | 0.0010903541446220515 | 4.788202116718519 |
| Cz | 0.0009464463635519125 | 5.107573166694493 |
| P7 | 0.0007974860019397658 | 7.071187152017298 |
| P8 | 0.0009956420268900324 | 7.999193229815309 |
| P3 | 0.0007670007489048218 | 6.051619201868359 |
| P4 | 0.0012381375464522238 | 6.28236006533592 |
| Pz | 0.0008099245119524646 | 5.975436383723319 |
| O1 | 0.0008653813276213706 | 8.708097125047082 |
| O2 | 0.0011687317901135747 | 8.678107139437047 |

## Independent verification

The first pass combines centered chunk moments with the parallel variance formula, processing sorted subject IDs then numeric block indices and ascending merged intervals, in at most 4,096-sample chunks. It never loads the whole dataset.

A fresh Train-only second pass reconstructs coverage with a Boolean bitmap per block, independently of interval merging. It reads only covered indices in 2,048-sample partitions and uses compensated channel sums of x, z and z squared. This independently verifies the mean and population standard deviation and reconstructs every block audit row.

Maximum absolute normalized channel mean: **1.8302699640687466e-17**. Maximum absolute normalized population std deviation from 1: **1.9984014443252818e-15**. Both pass absolute float64 tolerance **1e-12**. The second pass reads exactly 1,970 eligible Train blocks and 12,881,152 unique samples, with zero forbidden contributions. A further deterministic refit reproduces scaler, block manifest and lock bytes exactly.

Tests cover all 14 requested categories, real committed-metadata loader integration without EEG fixtures, forbidden dataset access instrumentation, independent coverage/statistics tampering, numerical stability and fail-closed external acceptance. The full guarded suite passed 426/426 tests in 42.98 seconds, including 28 normalization/acceptance cases. Compileall, the full-cohort preservation verifier and final-cohort verifier passed. Both independent reviewers passed after two cycles with no unresolved findings. The source-validator tuple-return integration defect was repaired and covered by a metadata-only regression; the verification script was made explicitly sequential. Clean post-commit results are bound in the external acceptance evidence below.

## Workflow and commands

The read-only Scenario 1 review preserved numbered steps, explicit input/output paths, visible stages, intermediate artifacts and concise summaries. Its scientific normalization/window logic was not copied. Scenario 2 continues its own numbering:

1. `s2_18_fit_train_normalization.py`: validate final sources, pool unique eligible Train samples and write the three deterministic artifacts. Existing conflicting artifacts fail closed; identical reruns do not overwrite them.
2. `s2_19_verify_train_normalization.py`: reconstruct permitted inputs, independently verify bitmap coverage and z statistics, refit and compare bytes. `--require-clean` is mandatory for post-commit acceptance.
3. `s2_20_accept_train_normalization.py`: read final test/review/verifier evidence, require exactly one clean non-merge commit from the starting HEAD, and write external acceptance. This follows the existing separate cohort-acceptance script pattern.

```powershell
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe scripts/data/s2_18_fit_train_normalization.py
.\.venv\Scripts\python.exe scripts/data/s2_19_verify_train_normalization.py

. .\scripts\setup_scenario2_environment.ps1 -ForTests
.\.venv\Scripts\python.exe -m pytest -q --basetemp "D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp\pytest" --junitxml outputs/logs/pytest.xml
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe -m compileall scripts src tests
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 005 --test-report outputs/logs/pytest.xml
.\.venv\Scripts\python.exe scripts/data/s2_16_final_cohort_split.py
```

In shells that disable local scripts, a process-scoped `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` allows the existing bootstrap without changing user/machine policy. The configured external D: temp/cache directories need write permission; the C: free-space and exact pytest-path guards remain unchanged. Never run simultaneous pytest processes against the guarded base.

After the single local commit, rerun s2_12 and s2_16 with `--require-clean`, saving their JSON output and zero exit codes. Run s2_19 with `--require-clean --report <evidence>/normalization_verifier_postcommit.json`, then s2_20 with `--evidence-directory <evidence>`. The evidence directory is exactly:

`D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/docs/train_normalization_lock_20260907/`

Its `final_acceptance.json` resolves the final committed HEAD and final accepted status, avoiding a self-referential tracked commit hash. Pre-commit test reports, both reviewer cycles and verifier results are preserved there. The previous cohort phase's JUnit report is preserved separately before the required base/report reuse.

## Artifact identities

| Normalization artifact | SHA256 |
|---|---|
| `train_normalization.json` | `82af5c196145756f09658ba5fd887a9953962349c23af922eca20df6b86d7379` |
| `train_normalization_block_manifest.csv` | `38bdeae3cc8cdea3a14d866ab04ae2d25a66d20e99e99a91d48afcbcbdbf668a` |
| `train_normalization_lock.json` | `9d69dab30fa8c55a2e34c98090671649d425634f10c03f1d286e49db35aa2ee1` |

| Final cohort/split input | SHA256 |
|---|---|
| `final_block_split_manifest.csv` | `a58605010f7371fe1805a7834c205f2da7be816515779e15f2ba082bfd201f8b` |
| `final_cohort_manifest.csv` | `e9405aacd8109700ed55ef3df49b8b5ec0187a65b1c3d79049597fa27b490bc9` |
| `final_cohort_split_lock.json` | `c591a46aafaa11cda0cb5b8fd3f39d1a5df61ac3fa8973fae96f804857edfe0f` |
| `final_cohort_split_summary.json` | `fe90564149a5a7740d66952134473ec45122dd1c5acf51f7d7b5e6a0c0551adb` |

Preprocessing semantic SHA256: `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b`.

The normalization lock also binds the final accepted record, all frozen source hashes, the 198 included HDF5/receipt identities, the accepted 211-subject bundles, and the fitter/verifier implementation hashes. Per-block audit records include source identity, mask/start hashes, exact merged intervals, eligibility and frozen exclusion reasons.

## Limitations and next phase

This is a sample-weighted global scaler; unequal usable duration is not rebalanced across subjects or states. It describes stored float32 processed EEG accumulated in float64, not native raw precision. QC selection and the existing EO/EC retention difference remain frozen limitations; no outcome-driven cleaning or cohort adjustment occurred. Deterministic bytes are verified in the pinned project environment; another numerical software environment must be checked before reuse.

No normalization application to held-out EEG or synthetic generation was performed. Final cohort and split are unchanged; Final Test remains sealed, Batch 006 is absent, no raw/processed files were deleted or modified, Scenario 1 remains read-only, and no Git push was performed.

NEXT ACTION — REAL-DATA BASELINES
