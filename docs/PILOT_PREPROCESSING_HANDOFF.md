# Scenario 2 Pilot Preprocessing Handoff

## 1. Executive status

Project phase: **PREPROCESSING_FROZEN_READY_FOR_STREAMING**. The verified 15-subject pilot has been used only through 150 complete provisional Train blocks to validate a frozen, outcome-blind preprocessing/QC method. All 15 compact pilot Train outputs and provenance pass verification; 14 subjects pass pilot QC, while `sub-010250` is retained with an explicit QC failure. Resumable downloading and subject streaming pass local tests. The final acceptance command in section 20 verifies the locally committed state and clean worktree.

The structural cohort remains 211 subjects. Final waveform eligibility, final cohort membership, final Train/Validation/Test allocation, and the numerical global normalization scaler remain unfrozen. The current executable pipeline processes **provisional Train only**, including future streamed subjects. Mechanically applying the frozen rules to provisional Validation/Test for final QC requires a separately authorized future extension. There is no held-out split override in the current loader or CLI.

This phase does not include physiological Alpha/Connectivity analysis, generator/classifier training, TSTR, augmentation evaluation, new non-pilot EEG downloads, or pilot raw deletion. The earlier `PRE_DOWNLOAD_HANDOFF.md` describes a historical phase; its simultaneous full-download command is superseded by the streaming commands in this document.

## 2. Git state

Repository: `D:/SHARIF/PROJECT/subject-conditioned-resting-eeg`; branch: `main`.

The starting HEAD was `504670270775813795876d4ba20833117b0022d6`, with a clean worktree. The preflight inspected status, recent history, HEAD, unstaged/staged diffs, and ignored files. Evidence is in `outputs/metrics/scenario2/pilot_preflight.json`.

| Local commit | Purpose |
| --- | --- |
| `3931e811e5f6585a7732d006c161ea98299bb435` | `data: add verified HTTP Range resume and network-free regression tests` |
| `1322bad5cd136d22446af76efcc8c427d8a507b2` | `preprocess: specify Train-only pilot QC with verified subject streaming` |
| `82e79d5bca706549bc0c96ba7399e4611913f48c` | `data: verify pilot Train outputs and record frozen QC evidence` |
| Final handoff commit, resolve with `git rev-parse HEAD` | `docs: freeze Scenario 2 preprocessing and hand off streaming` |

Existing commits were not amended or rewritten. All commits are local; nothing was pushed. Resolve the final content-derived HEAD with `git rev-parse HEAD`: a commit cannot embed its own hash. The final completion message supplies the exact final HEAD. The read-only acceptance command with `--require-clean` must pass after the final local commit. All real raw access records identify processing HEAD `1322bad5cd136d22446af76efcc8c427d8a507b2`; no package source/config changes occurred during the real pilot run.

List every commit created in this phase with `git log --reverse --format="%H %s" 504670270775813795876d4ba20833117b0022d6..HEAD`.

The metadata clone remains clean at `16702890507291b5a09fa7390245ec60bff6128e`. Its read-only Git verification used a command-local `safe.directory` option; no global Git configuration was changed.

## 3. Environment

The project uses its own ignored `.venv` at `D:/SHARIF/PROJECT/subject-conditioned-resting-eeg/.venv`, created for this phase. No global Python package installation or Scenario 1 environment was modified or implicitly reused.

Python is **3.12.3**, 64-bit, on Windows 11. The executable is `.venv/Scripts/python.exe`.

| Package | Frozen version |
| --- | --- |
| numpy | 2.5.2 |
| pandas | 2.3.3 |
| scipy | 1.18.1 |
| mne | 1.12.1 |
| h5py | 3.16.0 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| matplotlib | 3.11.1 |

`requirements.txt` states direct dependencies, `requirements-lock.txt` pins the installed dependency graph, and `outputs/metrics/scenario2/pilot_environment.json` records the executable/platform and exact versions. The pipeline checks the eight configured package versions before processing.

Reproduce the environment from the repository root, using Python 3.12.3 when creating a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall scripts src tests
```

An existing functional `.venv` does not need recreation. Use its explicit interpreter in every command below.

## 4. Pilot inventory

The staged raw root is `data_raw/MPI-LEMON`; the read-only metadata clone remains `D:/SHARIF/PROJECT/Datasets/MPI-LEMON`.

The exact pilot IDs are:

```text
sub-010023 sub-010059 sub-010071 sub-010078 sub-010081
sub-010126 sub-010134 sub-010216 sub-010250 sub-010276
sub-010277 sub-010278 sub-010279 sub-010280 sub-010282
```

The reverified inventory contains **15 subjects, 105 staged files, 15 `.eeg`, 15 `.vhdr`, and 75 local metadata sidecars**. All 30 backed payloads match their manifest sizes and SHA256 values. There are **zero `.part` files**, zero missing paths, and zero unexpected paths.

Raw size including sidecars is **7,013,640,445 bytes = 6.5319616767 GiB**. Backed payloads alone are **7,012,992,712 bytes**; the sidecars contribute 647,733 bytes. The authoritative inventory is `outputs/metrics/scenario2/pilot_download_manifest.csv`; the integrity summary records both payload and total staged sizes.

The composition deliberately includes eight 2500 Hz subjects, six 1000 Hz subjects, and the 500 Hz subject. It covers both subject-specific S208 overrides, both acquisition-restart cases, and four ordinary 2500 Hz controls. No non-pilot subject was downloaded in this phase. The raw tree remains Git-ignored and retained in place.

## 5. Loader integrity results

**15/15 subjects pass integrity; 0 fail.** Every subject passes `.eeg`/`.vhdr` exact size and SHA256 verification, local BrainVision linkage, MNE `preload=False` header opening, native sampling-rate/sample-count agreement, 61 EEG + 1 VEOG composition, unique presence of all 19 targets, and coverage of retained study-block boundaries.

| Subject | Native Hz | Native samples | Exclusive-end duration, seconds | Integrity |
| --- | ---: | ---: | ---: | --- |
| sub-010023 | 2500 | 2,510,900 | 1004.36 | pass |
| sub-010059 | 2500 | 2,773,850 | 1109.54 | pass |
| sub-010071 | 2500 | 2,694,050 | 1077.62 | pass |
| sub-010078 | 2500 | 2,685,600 | 1074.24 | pass |
| sub-010081 | 2500 | 2,793,600 | 1117.44 | pass |
| sub-010126 | 2500 | 2,640,100 | 1056.04 | pass |
| sub-010134 | 2500 | 2,528,400 | 1011.36 | pass |
| sub-010216 | 500 | 523,980 | 1047.96 | pass |
| sub-010250 | 2500 | 2,626,150 | 1050.46 | pass |
| sub-010276 | 1000 | 1,072,840 | 1072.84 | pass |
| sub-010277 | 1000 | 1,088,220 | 1088.22 | pass |
| sub-010278 | 1000 | 1,081,740 | 1081.74 | pass |
| sub-010279 | 1000 | 1,115,540 | 1115.54 | pass |
| sub-010280 | 1000 | 1,049,540 | 1049.54 | pass |
| sub-010282 | 1000 | 1,093,580 | 1093.58 | pass |

The raw format is multiplexed IEEE float32 with 62 channels. The `.vhdr` links are confined to the staged subject directory and point to the expected local `.eeg` and `.vmrk`. The header resolution is 0.1 microvolts per binary value; MNE's conversion is `1e-7 V` per binary value. MNE data are in volts, and compact outputs multiply by `1e6` to store microvolts (`uV`). This was verified from header/MNE calibration, rather than inferred from signal appearance.

Duration here is `n_times / sfreq`, the exclusive end. The metadata's last-sample-time convention differs by one native sample and is handled explicitly. MNE header warnings: none recorded. The montage is a template, `standard_1005`; no individual measured electrode coordinates are staged.

Evidence: `outputs/metrics/scenario2/pilot_loader_integrity.csv` and `pilot_loader_integrity_summary.json`. Their zero-waveform-access fields describe the **header/hash audit itself**, not the later permitted Train preprocessing run.

## 6. Waveform access control

`TrainAccessGate` in `src/resting_eeg/data/access.py` joins the authoritative `provisional_block_split_manifest.csv` and `lemon_block_inventory.csv` by subject and block index. Authorization requires a unique complete study block, `is_interruption_fragment=False`, split exactly `Train`, status `PROVISIONAL_METADATA_ONLY`, and agreement of state, boundaries, and acquisition segment between the two manifests. Their hashes are checked again at authorization. Unknown, duplicate, incomplete, interrupted, Validation, and Test requests fail explicitly before waveform I/O.

`TrainBlockLoader` opens the verified subject header with `preload=False`, preserves native channel order, and reads exactly the approved half-open sample interval. Decimal arithmetic computes `start=ceil(start_sec*native_rate)` and `stop=ceil(end_sec*native_rate)`. The read is bounded to 65 seconds and one subject loader under a cross-process lock. No public loader method exposes its underlying MNE Raw object. This is an enforceable application boundary, not an operating-system security sandbox.

Every permitted native value read is durably appended **before I/O** to `outputs/metrics/scenario2/pilot_waveform_access_registry.csv`, with subject/state/block identity, provisional split, time/sample interval, purpose, code HEAD, source-manifest hashes, and authorization reason. Subject preprocessing reads each of its ten Train blocks once and uses a temporary disk cache for subsequent passes. Verification of already derived compact Train blocks does not read new raw intervals.

The final registry contains **150 unique Train blocks, 150 permitted read-attempt records, 15 subjects, and zero repeated events**. The exact interval/manifest audit passes, with **zero Validation, zero Test, and zero interrupted-fragment accesses**. An all-15-subject idempotent rerun left the registry unchanged. Evidence is in `pilot_waveform_access_registry_summary.json` and `pilot_reproducibility_summary.json`.

The authoritative split SHA256 is `6adc10b108a7e7b587757cbda630a3c4909c97f8cc2faf24320b0a6e89af0b26`; the block inventory SHA256 is `d5deff471d02baa55ac60539986595b82594d5137fcf1bffac6a5ec982b28742`. The package source-content hash recorded by aggregation is `652ed2428aae5732fcd4080da938dfedff13ba55ee625323dadb498ffce4bed0`.

**Future split constraint:** every pilot block whose waveform was accessed during development must remain Train in the final split, or its subject must be excluded. Reallocation cannot erase development exposure. Whole-file SHA256 hashing is opaque byte-integrity verification and does not authorize or perform physiological waveform analysis.

## 7. Frozen preprocessing pipeline

Canonical method: `scenario2-preprocessing-v1`, protocol 0.3, in `configs/preprocessing/frozen_preprocessing_v1.yaml`. The canonical parsed-config SHA256, enforced through the adjacent `.sha256` file, is:

```text
5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b
```

The complete operation order is:

1. Reverify the subject source payloads, BrainVision links/header, and authoritative Train block assignments. Read one complete approved block, retaining all 61 EEG channels and VEOG in original order. Record native signal sanity in microvolts before replacing any nonfinite values.
2. Use `standard_1005` template montage coordinates. Any native nonfinite sample is preserved as a failure in metrics; zero filling only permits subsequent filtering/interpolation computations and cannot make a failed unrepaired channel eligible.
3. Filter each complete block independently in volts with `mne.filter.filter_data`, `method='fir'`, `fir_design='firwin'`, Hamming window, `phase='zero'`, passband 1-40 Hz, low transition width 0.5 Hz, high transition width 10 Hz, `filter_length='auto'`, `pad='reflect_limited'`, and `n_jobs=1`. Transitions are 0.5-1 Hz and 40-50 Hz. FIR lengths are 3301/6601/16501 samples at 500/1000/2500 Hz.
4. Resample the filtered block to 128 Hz with `scipy.signal.resample_poly`. Reduce the integer ratio by GCD; design explicit FIR taps with `firwin(20*max(up,down)+1, 1/max(up,down), window=('kaiser',5.0))`; use `padtype='line'`. Output count is `ceil(native_samples*128/native_rate)`. The 62-channel float64 resampled block is stored in a temporary disk-backed cache.
5. Fit subject bad-channel decisions from pooled EO and EC Train blocks, using only the interior beyond the filter/resampling guard. The broadband metrics are channel robust scale `1.4826*MAD`, robust first-difference scale, and median absolute correlation with other EEG channels. Median-across-block values are compared across EEG channels. Log10 scale and derivative use robust z-scores with denominator floor 0.1; absolute z > 6 and median absolute correlation < 0.1 are outlier votes. Ordinary failure needs low correlation plus at least one amplitude/derivative vote. Amplitude and derivative alone cannot designate a bad channel. Any native nonfinite sample, or native robust scale < `1e-6 uV` in at least half of Train blocks, is catastrophic.
6. Allow at most six bad EEG channels and at most two bad target channels. Interpolate eligible bad EEG using MNE accurate spherical-spline interpolation with origin `(0,0,0.04)` meters on the full montage. If these limits fail, preserve a labeled diagnostic compact output with every window rejected; do not silently accept it.
7. Compute Common Average Reference across all 61 EEG channels after interpolation. VEOG is excluded. Then select the target channels in this exact order: `Fp1, Fp2, F7, F8, F3, F4, Fz, T7, T8, C3, C4, Cz, P7, P8, P3, P4, Pz, O1, O2`.
8. Convert volts to microvolts by `1e6`. VEOG remains unreferenced and has undergone the same filter/resampling; retain it transiently for QC only.
9. Exclude 433 output samples, **3.3828125 seconds**, from each block edge. This equals rounding `(3.3 + 0.078125)` seconds upward to the 128 Hz grid. The first retained sample timestamp is the exact native start-sample time plus the guard. No neighboring block or unlabeled interval supplies filter context.
10. Define candidate windows of **512 samples (4 seconds)** at **256-sample (2-second) stride**, wholly inside each trimmed block. Keep continuous data and window starts; do not materialize overlapping windows.
11. Fit one pooled EO+EC Train threshold vector per subject. For each feature, use `min(cap,max(floor,median + 8*1.4826*MAD))` over finite, non-flat Train candidates. Features, floors, and caps are in the table below. The predeclared multiplier is 8; 6 and 10 are reported for engineering sensitivity only.
12. Mark each candidate's usability and reasons. Reject any exceeded threshold, nonfinite values, target-channel robust scale < `0.01 uV`, VEOG robust scale < `0.01 uV`, or target repeated-first-difference fraction > 0.99. A native nonfinite VEOG sample rejects its entire block. Any native nonfinite, robust scale < `1e-6 uV`, or repeated-difference fraction > 0.99 on a channel not repaired by interpolation also rejects that block. If there are no finite non-flat Train fitting candidates, use fixed caps for diagnostic bookkeeping, reject all windows, and fail subject QC.
13. A block needs at least **10 usable windows** and **50% usable fraction**. A pilot subject needs at least **three usable Train blocks per state**, a valid threshold reference, and passing channel limits. Store ineligible blocks and their masks without reassigning the split. These are pilot engineering requirements; they do not establish final 5/1/2 feasibility.
14. Write one HDF5 containing continuous trimmed 19-channel float32 `uV` blocks, masks/reasons, and source/config/code provenance. Verify it before atomic final promotion and durable receipt creation. Remove the temporary cache. Numerical normalization remains unfitted.

| Pooled Train QC feature | Floor, uV | Cap, uV |
| --- | ---: | ---: |
| Maximum target peak-to-peak | 150 | 1000 |
| Maximum target robust scale | 30 | 200 |
| Maximum absolute target first difference | 50 | 300 |
| VEOG peak-to-peak | 250 | 1500 |

There is no ICA, ICLabel, ASR, source localization, task-specific spectral enhancement, or outcome-based selection. A source block is processed independently and only one native subject/block is materialized at a time. The remaining subject passes read one compact cached block at a time, not all subjects or overlapping windows.

## 8. Rationale for every frozen decision

| Decision | Engineering basis and practical limitation |
| --- | --- |
| Exact Train gate and independent blocks | Prevents development exposure of held-out signals and prevents time-dependent operations from bridging allocation boundaries. Tests explicitly exercise denied held-out and interrupted requests. |
| Explicit physical units | Header/MNE calibration agrees at `1e-7 V` per native binary value; storing `uV` prevents an undocumented million-fold scale error. |
| 1-40 Hz zero-phase FIR followed by 128 Hz polyphase resampling | The requested broadband representation is validated by synthetic 5/20/35 Hz preservation and 60/100 Hz suppression, independent of physiological Alpha. |
| 433-sample guard | Derived from actual finite impulse support, not chosen from EO/EC appearance. It excludes dependence on boundary padding; it does not remove ringing around artifacts inside the retained block. |
| Template full montage, bounded interpolation, full 61-channel CAR | All channels and template coordinates exist. Correlation corroboration avoids treating related amplitude/derivative votes as independent evidence. The interpolation limits bound repair burden; template geometry is not individual digitization. |
| Pooled robust Train thresholds with fixed floors/caps | One rule is applied across states; median/MAD limits sensitivity to extreme artifacts, floors prevent collapsing thresholds, and caps prevent excessively permissive references. These are engineering thresholds, not clinically validated artifact labels. |
| VEOG-only artifact proxy | VEOG supplies an outcome-blind ocular artifact indicator without adding an ICA model or generated target. It cannot characterize every artifact source. |
| Native integrity checks survive CAR | An unrepaired constant channel can acquire apparent variation after CAR; native failures therefore remain explicit block rejection evidence. |
| 4-second windows/2-second stride and continuous storage | Exact 512/256 geometry supports later indexing while avoiding duplicated overlapping sample storage. |
| At least 10 windows, 50% retention, three Train blocks/state | Ensures a minimum pilot engineering representation of both states. The final 5/1/2 allocation remains a future feasibility question. |
| Global normalization delayed | The fitting population is defined only after cohort/split lock; overlapping usable samples must not be counted twice. |
| Subject streaming and explicit cleanup | Bounds disk use to one source pair plus temporary computation/output allowances; verified compact provenance and a durable success journal precede any later raw deletion. |

MNE's documented FIR phase, length, transition-band, and padding semantics support the chosen implementation; exact behavior was also checked against the locally installed pinned source and synthetic tests. The versioned public documentation used as a semantic reference is [MNE `filter_data`](https://mne.tools/1.6/generated/mne.filter.filter_data.html); the actual environment is MNE 1.12.1, not documentation version 1.6.1.

The configuration and rule rationale were committed before any pilot value access, before generator/classifier development, before Validation waveform inspection, and before Final Test. The Train-only pilot confirms finite/unit-consistent data, valid resampling at all three native rates, no bad channels under the prespecified corroborated rule, 93.0667% usable candidates, and 14/15 subjects meeting pilot QC. Interpolation remains covered by synthetic tests because the real pilot required no repairs.

The 12.0533-percentage-point EO/EC rejection-rate difference and `sub-010250` failure are preserved as limitations, not removed by fitting state-specific thresholds or relaxing the method. All 15 subjects still provide at least one eligible Train block per state; one does not meet the required three EO blocks. The multiplier remains the predeclared 8 despite higher retention under candidate 10. No outcome-driven threshold optimization was used.

## 9. Pilot QC results

Final evidence is in `outputs/metrics/scenario2/pilot_train_qc_summary.json`, `pilot_train_subject_qc.csv`, `pilot_train_block_qc.csv`, `pilot_train_window_qc.csv`, and the native signal/bad-channel summaries.

| Quantity | Final result |
| --- | --- |
| Integrity pass | 15/15 |
| Subjects with verified compact Train outputs | 15/15 |
| Subjects passing pilot QC | 14/15; `sub-010250` has only 1 eligible EO Train block, below the required 3 |
| Unique processed Train blocks / eligible Train blocks | 150 / 143 |
| Candidate windows / usable windows / retained fraction | 3750 / 3490 / 93.0667% |
| Usable windows within eligible blocks | 3415; 3276 when also requiring a QC-passing subject |
| Bad EEG / target channel decisions | 0 / 0; no subject requires interpolation |
| Native signal finiteness | Minimum finite fraction 1.0; 0 NaNs, 0 Infs across 9300 channel-block observations |
| Native broad amplitude range | -1313.2 to 1900.3 uV across EEG/VEOG; amplitude alone is not a subject exclusion |
| Native repeated/clipping indicators | Maximum flat-difference fraction 0.0348333, exact-zero fraction 0.0239663, extrema-repeat fraction 0.000133333 |
| Rejected windows | 260; overlapping reasons: VEOG peak-to-peak 220, target peak-to-peak 71, target first difference 40, target robust scale 5 |
| EO retention / EC retention | 1632/1875 = 87.04% / 1858/1875 = 99.0933% |
| Absolute EO/EC rejection-rate difference | 12.0533 percentage points |
| Eligible EO / EC blocks | 68/75 / 75/75 |
| Candidate multiplier 6 / 8 / 10 usable windows | 3409 / 3490 / 3577 of 3750; predeclared 8 retained |

| Native rate | Subjects passing QC | Eligible blocks | Usable/candidate windows | Retention |
| --- | ---: | ---: | ---: | ---: |
| 500 Hz | 1/1 | 10/10 | 242/250 | 96.8% |
| 1000 Hz | 6/6 | 57/60 | 1409/1500 | 93.9333% |
| 2500 Hz | 7/8 | 76/80 | 1839/2000 | 91.95% |

Every native-rate group has finite compact candidates and zero native NaN/Inf samples. The 500 Hz result represents one subject and cannot establish population-wide rate equivalence. Candidate multipliers 6/8/10 retain EO windows 1560/1632/1714 and EC windows 1849/1858/1863, respectively; this is engineering sensitivity, not a selection optimization.

`sub-010250` has valid sources, no bad channels, and an available pooled threshold reference. Its fixed-rule VEOG threshold is **259.9425956041876 uV**. All **66 rejected windows** are EO candidates rejected by `veog_ptp_uv`; EO retains **59/125 = 47.2%** and EC retains **125/125 = 100%**. Only **1/5 EO** versus **5/5 EC** blocks meet block eligibility. Its verified diagnostic compact output and raw files are retained, and it is **ineligible for cleanup**. This explicit subject-QC failure does not invalidate the completed engineering implementation or make the subject part of a locked final cohort.

Per-subject evidence is written to the ignored local cache `outputs/metrics/scenario2/subjects/<subject_id>/`: `signal_sanity.csv`, `bad_channel_metrics.csv`, `bad_channel_decisions.csv`, `window_qc.csv`, `block_qc.csv`, `qc_candidates.csv`, and `subject_qc.json`. The committed aggregate signal/QC/output manifests are listed in section 18. Native sanity covers all 61 EEG channels and VEOG separately; no desired physiological pattern is an acceptance criterion.

EO-versus-EC rejection balance is an engineering check for structural loss of a state. It is not a physiology analysis or permission to fit per-state thresholds. Compact output verification and pilot QC are distinct: a diagnostic output can be structurally valid while subject QC fails, in which case raw cleanup is prohibited. A `True` window mask alone does not license use of an ineligible block or failed subject: downstream selection must also require block and subject eligibility. Thus 3490 mask-usable windows, 3415 within eligible blocks, and 3276 within eligible blocks of QC-passing subjects are deliberately separate counts.

## 10. Special-subject results

All special subjects pass loader integrity and pilot Train QC, with zero bad EEG/target channels. Each has 250 candidate windows; the table reports usable windows and eligible blocks.

| Subject | Preserved metadata semantics / native rate | Train QC result |
| --- | --- | --- |
| sub-010059 | 2500 Hz; two aborted EC fragments excluded; retained post-restart study sequence | pass; 214/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010078 | 2500 Hz; six initial S208 markers mapped to EC only for this subject | pass; 244/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010081 | 2500 Hz; one aborted EC fragment excluded; retained post-restart study sequence | pass; 250/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010126 | 2500 Hz; S208 is this subject's documented EC replacement | pass; 250/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010216 | 500 Hz; included explicitly for rare-rate validation | pass; 242/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010276 | 1000 Hz | pass; 220/250 windows, 9/10 blocks (4 EO, 5 EC) |
| sub-010277 | 1000 Hz | pass; 248/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010278 | 1000 Hz | pass; 244/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010279 | 1000 Hz | pass; 246/250 windows, 10/10 blocks (5 EO, 5 EC) |
| sub-010280 | 1000 Hz | pass; 201/250 windows, 8/10 blocks (3 EO, 5 EC) |
| sub-010282 | 1000 Hz | pass; 250/250 windows, 10/10 blocks (5 EO, 5 EC) |

The existing authoritative block inventory and `configs/audit/event_code_overrides.yaml` supply these semantics. No independent waveform-based state reconstruction, gap labeling, or global S208 mapping was introduced. No interrupted fragment was inspected.

## 11. Resampling validation

Synthetic 60-second signals at each native rate test 5, 20, and 35 Hz passband preservation and separate 60 and 100 Hz stopband/alias suppression. These frequencies are engineering probes; physiological Alpha is not a test input or tuning criterion. Repeated outputs are deterministic. Every case produces exactly 7680 samples at 128 Hz before edge trimming.

| Native Hz | Input samples | FIR samples | Output samples | Gain 5 Hz | Gain 20 Hz | Gain 35 Hz | Residual 60 Hz | Residual 100 Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 30,000 | 3,301 | 7,680 | 1.000429 | 0.999746 | 0.999570 | 0.000906679 | 6.28504e-8 |
| 1000 | 60,000 | 6,601 | 7,680 | 1.000405 | 0.999675 | 0.999609 | 0.000861161 | 5.00569e-8 |
| 2500 | 150,000 | 16,501 | 7,680 | 1.000438 | 0.999761 | 0.999644 | 0.000904948 | 6.76625e-8 |

All three pass. Passband gain differs from unity by less than 0.0005; residual 60 Hz amplitude is below 0.001 and residual 100 Hz below `7e-8`. The combined FIR/polyphase half-support is 3.378125 seconds at each rate. The frozen output-grid guard is 3.3828125 seconds. Evidence: `outputs/metrics/scenario2/pilot_resampling_validation.csv` and synthetic tests in `tests/test_preprocessing.py`.

Real block duration may differ from exactly 60 seconds by native sample-grid details. The pipeline verifies `ceil(native_samples*128/native_rate)` for every actual block and then removes 866 samples in total. All 150 real compact block shapes and finite values pass verification; native-rate retention is reported in section 9 and `pilot_native_rate_qc_summary.csv`. No native-rate group receives a different filter/QC rule. `pilot_resampling_summary.json` joins the synthetic results with this real Train-only engineering evidence.

## 12. Compact storage

Each subject produces `data_processed/MPI-LEMON/<subject_id>_preprocessed.h5` and an adjacent `<subject_id>_preprocessed.provenance.json` receipt. Raw, processed, normalized, checkpoints, and temporary caches remain Git-ignored; only small evidence/config/source files are committed.

```text
<subject_id>_preprocessed.h5
  root attributes: subject/source/config/code/split/inventory provenance,
                   original channel order/types, bad channels, thresholds,
                   scope=provisional_Train_only, normalization_fitted=False
  blocks/<subject_id>_block-XX/
    eeg_uv                 float32 [19, continuous_trimmed_samples]
    window_start_samples   int64 [candidate_windows]
    window_usable_mask     bool [candidate_windows]
    window_qc_reason       fixed-width byte strings [candidate_windows]
    attributes: state/block identity, original time bounds, native/processed
                sampling rates, channels/unit, trim/first-sample timing,
                Train/study flags, block eligibility, source/config/code hashes
```

EEG compression is gzip level 4 with shuffle and chunks `(19,min(1024,n_samples))`; the normal chunk is `(19,1024)`. Continuous values remain physical `uV`. VEOG is absent from the generated target container; its QC features/reasons remain in small metrics. Candidate windows are indexed views for later use, not overlapping stored arrays. The output retains all ten authorized Train blocks and records eligibility instead of silently removing blocks or changing the split.

`verify_output` reopens the container and checks expected block IDs, config hash, 19-channel float32 shape/order, physical unit, 128 Hz Train/study attributes, exact candidate geometry, mask/reason consistency, finite compact values, and mandatory provenance, then computes a byte SHA256. The durable receipt records source hashes, input signature, config/split/inventory/code hashes, expected blocks, output path/size/SHA256, and subject QC.

| Storage quantity | Result |
| --- | --- |
| Verified pilot HDF5 count / total bytes / MiB | 15 / 68,563,819 bytes / 65.387553 MiB (0.063855 GiB) |
| Adjacent provenance receipts, counted separately | 15 / 60,412 bytes |
| Uncompressed continuous target EEG payload | 77,690,012 bytes |
| Mean observed compact HDF5 per subject | 4,570,921.267 bytes |
| Estimated 211-subject **Train-only**, ten blocks/subject | 964,464,387 bytes / 0.898227 GiB |
| Estimated 211-subject **all-16-block** representation | 1,543,143,020 bytes / 1.437164 GiB |

The Train-only estimate scales measured pilot compact bytes by `211/15`. An all-16-block estimate additionally scales by `16/10` and is a planning projection; the current code does not produce held-out blocks. Compression varies with signal/content and metadata overhead, and final QC/cohort membership may change storage. These estimates are not observed full-cohort sizes. An uncompressed continuous payload estimate is `19 * 4 * retained_samples`; no factor for overlapping windows is added. Exact output paths, byte hashes, sizes, config, and source provenance are recorded in `pilot_processed_manifest.csv`; estimates are in `pilot_storage_summary.json`.

## 13. Normalization

The method is frozen: global channelwise mean and population standard deviation (`ddof=0`), fit from **unique continuous samples covered by usable windows in eligible final Real Train blocks**, only after final cohort/split lock. Each time sample is counted once even if covered by overlapping windows. Any channel standard deviation below `1e-6 uV` fails fitting. Apply that same scaler to Real Validation, Real Test, and Synthetic data.

The numerical scaler is **not fitted**. There is no per-window, per-state, Validation, Test, or synthetic fitting. Current HDF5 data remain float32 physical microvolts, with `normalization_fitted=False`. Full-cohort eligibility and final Train membership are prerequisites to the future numerical fit.

## 14. Resumable downloader

`scripts/data/s2_06_download_manifest.py` retains its manifest API and defaults to a validation-only dry run. Existing final files are checked by exact size/SHA256 and return `verified_existing`; corrupt final files are refused and preserved. A valid complete `.part` is verified and atomically promoted without a request. Oversize or complete corrupt partials are quarantined beside the original path with an `.invalid-<uuid>` suffix before restart.

An incomplete partial, including an empty partial, requests `Range: bytes=<current_size>-` and `Accept-Encoding: identity`. A 206 response must provide the exact requested start, final byte, and expected total in `Content-Range`; any supplied `Content-Length` must match. Invalid range/encoding/length headers are refused before appending. If a server ignores Range and returns valid HTTP 200, the full response replaces the partial from byte zero; a full response is never appended to an old prefix.

Completed bodies must match exact size and SHA256 before same-directory `os.replace` promotion. A checksum/oversize-body failure quarantines the partial and raises. Interrupted/short responses preserve received partial bytes, including bytes carried in an `IncompleteRead` exception. If all bytes arrived immediately before interruption, the next attempt can verify/promote the completed partial without issuing an invalid `bytes=N-` request.

Default timeout is **60 seconds per socket operation**, not a whole-asset deadline. Default retries are **3 additional attempts** (at most four attempts). Backoff starts at 1 second, doubles, and is capped at 60 seconds. Retryable HTTP codes are 408, 429, 500, 502, 503, and 504; permanent errors and malformed responses fail explicitly. `--continue-on-error` continues selected assets while preserving a failing final exit code.

The CLI supports repeated `--subject-id`, `--subjects-file` (newline IDs or CSV `subject_id`), `--max-subjects`, `--timeout`, `--retries`, `--backoff`, `--continue-on-error`, `--dry-run`, and explicit `--execute`. Unknown subjects fail validation; selection retains manifest order. Exit codes are 0 for success/dry run, 1 for execution failure, and 2 for invalid configuration/manifest. The standalone durable append-only journal defaults to `outputs/metrics/scenario2/download_journal.jsonl`; streaming uses `streaming_download_status.jsonl`.

The downloader's 60 new regression cases cover verified existing/complete partial files, Range resume, ignored Range, malformed headers, corrupt/oversize partial/final bodies, interruptions, retry bounds/backoff/status, atomic promotion, path/source checks, filtering, dry-run write/network absence, failure exits/continue, and sidecar idempotence. With the three pre-existing cohort/download cases, the focused run passed **63 tests**. All HTTP responses were mocked; **no new NEMAR EEG request was executed**. An actual read-only dry run selecting `sub-010023` planned two assets totaling 622,704,959 bytes.

Exact standalone validation command:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_06_download_manifest.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --subject-id sub-010002 --dry-run
```

Future downloader-only execution, after this phase boundary, is available with the same selection plus `--execute --retries 3 --timeout 60`. Prefer the streaming workflow so downloading is coupled to source/output verification and bounded subject processing. The old simultaneous full-manifest download plan is no longer the execution strategy.

## 15. Subject-streaming pipeline

The CLI is `scripts/data/s2_07_stream_preprocess_subjects.py`; the orchestration logic is `src/resting_eeg/data/streaming.py`. It validates one `.eeg`/`.vhdr` pair per selected subject from the authoritative full inventory, checks confined raw/output/metadata paths and disk availability, resumes/verifies payloads, copies/validates five required sidecars, audits headers, processes only complete provisional Train blocks, verifies output and durable provenance, records completion, and then proceeds to the next subject.

The default is a read-only dry run: no downloads, waveform reads, journal writes, or deletions. `--execute` enables processing; `--local-existing-only` refuses missing complete local source pairs instead of requesting them. Raw retention is the default and is explicit with `--keep-raw`. Selection uses repeated `--subject-id`, `--subjects-file`, and `--max-subjects`. `--continue-on-error` logs failures and continues while returning failure if any selected subject failed.

The disk guard estimates **remaining unmaterialized payload bytes + 512 MiB processing overhead + 32 MiB compact output allowance + 1 GiB safety margin**. It credits the sizes of local completed/partial payloads for space planning; integrity verification still follows. The output filesystem independently needs at least 32 MiB + 1 GiB free. For a fresh subject, remaining bytes equal the full source pair. A low-space plan fails before staging/processing.

The durable append-only state journal is `outputs/metrics/scenario2/streaming_subject_status.jsonl`, protected by an OS file lock. Records include run/time/subject/scope, source hashes, plans, output receipt, and statuses `planned`, `downloading`, `download_verified`, `preprocessing`, `processed_verified`, and `complete`; failures record `planning_failed`, `download_interrupted`, `preprocessing_failed`, or `cleanup_failed`. Cleanup records `raw_deleted` followed by a final completion record. An incomplete final JSON write can be recovered; interior corruption fails explicitly.

Idempotence uses an input signature over config, authoritative source rows, staged sidecars, split/block manifests, and all package source-file hashes. Existing outputs are reopened and checked against their durable SHA256 receipt; a changed signature refuses reuse and requires an explicit new output directory. Ambiguous orphan output/receipt files are preserved and regenerated safely. A successful later raw deletion can be resumed through journal/output verification without redownloading an already completed subject. An idempotent local rerun should not add new native waveform-access registry rows.

`--delete-raw-after-success` is a separate explicit request. Deletion requires passing subject QC, fresh source size/hash checks, valid compact structure/hash/config and persisted receipt, a durably written matching completion record, available metadata clone/sidecars, and paths confined to the direct raw subject directory with no symlinks/reparse points or unresolved partials. Output/config/journal locations cannot be inside the deletion target. Any preprocessing/verification failure preserves raw and does not claim successful cleanup. Pilot cleanup was **not executed**.

The full-inventory dry run planned **211 subjects**, found **15 local payload pairs**, and passed every disk check; the maximum required free space was **2,427,576,127 bytes**. It made zero download requests, raw value reads, or raw deletions. The completed pilot execution produced 15 verified outputs. An all-15-subject local-existing-only rerun completed with exit 0 in **26.172 seconds**, left every output hash and the access registry unchanged, and requested no network or deletion. Evidence: `streaming_dry_run_summary.json`, `streaming_subject_status.jsonl`, and `pilot_reproducibility_summary.json`. Exact executable commands are in section 20.

## 16. Pilot raw cleanup

The prepared report is `outputs/metrics/scenario2/pilot_raw_cleanup_manifest.csv`. It records subject, raw directory/bytes, source-hash verification, compact path/hash, processed verification, eligibility for raw cleanup, and a concrete blocker if ineligible. **14 subjects are eligible for later cleanup; `sub-010250` is blocked by failed subject QC. Zero cleanups were executed.** Eligibility is a report, not authorization already exercised.

**No pilot raw EEG or raw subject directory was deleted.** All 105 staged pilot files remain available in this phase. Compact Train outputs omit held-out blocks; deleting raw later trades local availability of those source intervals for the ability to redownload the verified source pair when separately authorized held-out processing is implemented.

The exact later cleanup command, documented only and **NOT RUN**, is:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/pilot_download_manifest.csv --execute --local-existing-only --delete-raw-after-success --continue-on-error
```

This uses the same processing/verification path and explicit cleanup safeguards; it is not a standalone directory deletion command. Subjects failing source/output/QC checks retain raw data. To retain pilot raw, use the local verification command in section 20 with `--keep-raw` instead.

## 17. Tests and validation

| Check | Result |
| --- | --- |
| Downloader plus pre-existing cohort/download focused suite | 63 passed; all HTTP mocked |
| Full project `.venv` pytest suite | 166 passed in 17.36 seconds; JUnit report `outputs/logs/pytest.xml` |
| `compileall scripts src tests` | Passed |
| Config digest/version enforcement | Passed; canonical config/code hashes agree with output/aggregate provenance |
| Native-rate synthetic validation | 3/3 deterministic passes |
| 15-subject loader/hash integrity | 15/15 pass |
| Authoritative Train-access registry audit | Passed; 150 unique Train blocks/150 records; 0 held-out/interrupted accesses |
| Full-manifest streaming dry run | Passed; 211 plans, 15 local pairs, all disk checks pass |
| Local-existing-only pilot execution | 15 verified compact outputs; 14 pass subject QC, 1 explicit QC failure |
| Idempotent local rerun / no new raw value access | Passed for all 15 subjects; every output hash and access registry unchanged |
| Independent scientific-array/mask reproducibility | Synthetic HDF5 independent rebuild test: all datasets identical |
| Raw/processed Git ignore and forbidden tracked-payload guard | Enforced by `s2_10_verify_preprocessing_freeze.py` |
| Final staged diff checks and clean Git worktree | Require the section 20 acceptance command after the final local commit |

Tests cover header links/escapes, native sample conversion, Train authorization and held-out/interrupted refusal, window boundaries and 512/256 geometry, bad-channel robust logic and state pooling, full-montage CAR and deterministic channel order, physical units, HDF5 structure/provenance/idempotence, downloader resume/failure cases, streaming dry-run/local-only operation, deletion safeguards, and failure preservation.

Reproducibility means equal scientific arrays, candidate indices, usability masks, and QC decisions for identical sources/config/code. Container creation timestamps or code HEAD provenance can make independently rebuilt HDF5 byte hashes differ; each container's own hash is recorded and verified. The independent rebuild evidence is **synthetic**; the real-data repeat is an idempotent verification/reuse of all 15 outputs, not a second raw waveform rebuild. Raw SHA256 checks can read full files as opaque bytes; all value-level tests involving real raw data remain behind the Train gate.

Warnings/limitations: template electrode coordinates; broad engineering artifact rules rather than validated artifact labels; missing participant-linked demographics; small rare-rate sample counts; unknown full-cohort eligibility; held-out mechanical processing still requires a future authorized extension. No Alpha, Connectivity, model, or augmentation outcome has been computed here.

## 18. Repository files added/modified

| File or path | Role |
| --- | --- |
| `requirements.txt`, `requirements-lock.txt` | Direct dependencies and exact environment reproduction |
| `configs/preprocessing/frozen_preprocessing_v1.yaml`, `.sha256` | Canonical full method specification and enforced config hash |
| `src/resting_eeg/common.py` | Hashing, durable writes, and process locks |
| `src/resting_eeg/data/access.py` | Train-only authorization, sample bounds, access registry, bounded loader |
| `src/resting_eeg/data/integrity.py` | Source/header/link/channel/rate/unit checks and exact pilot inventory guard |
| `src/resting_eeg/preprocessing/signal.py` | FIR/polyphase processing, native sanity, pooled bad channels/QC, CAR, window geometry |
| `src/resting_eeg/preprocessing/pipeline.py` | Sequential subject/cache/HDF5 processing, verification, provenance, idempotence |
| `src/resting_eeg/data/streaming.py` | Subject planning, disk guard, download/process journals, safe cleanup |
| `scripts/data/s2_06_download_manifest.py` | Hardened Range-resumable audited downloader |
| `scripts/data/s2_07_stream_preprocess_subjects.py` | Streaming CLI, dry run, local-only mode, filters and explicit raw retention/deletion |
| `scripts/data/s2_08_audit_pilot_integrity.py` | Reproducible exact-15 pilot header/hash audit without waveform-value reads |
| `scripts/data/s2_09_summarize_pilot.py` | Aggregate existing Train metrics, verify output/source-sidecar evidence, create storage/cleanup reports and QC figure; no new raw waveform reads |
| `scripts/data/s2_10_verify_preprocessing_freeze.py` | Read-only final acceptance over pilot inventory, access intervals, config/software, compact hashes, required evidence and Git cleanliness |
| `tests/test_download_resume.py` | 60 new network-free downloader cases |
| `tests/test_loader_integrity.py`, `tests/test_preprocessing.py`, `tests/test_streaming.py` | Scientific-access, processing, storage, and orchestration regressions |
| `outputs/metrics/scenario2/pilot_preflight.json`, `pilot_environment.json` | Starting repository/environment evidence |
| `outputs/metrics/scenario2/pilot_loader_integrity.csv`, `pilot_loader_integrity_summary.json` | All 15 source/header integrity results and raw inventory |
| `outputs/metrics/scenario2/pilot_resampling_validation.csv` | Synthetic numeric resampling evidence |
| `outputs/metrics/scenario2/pilot_waveform_access_registry.csv` | Durable permitted raw interval registry |
| `outputs/metrics/scenario2/subjects/` | Small per-subject native sanity, channel/window/block QC, thresholds and eligibility |
| `outputs/metrics/scenario2/streaming_subject_status.jsonl` | Durable subject workflow/provenance state |
| `outputs/metrics/scenario2/pilot_raw_cleanup_manifest.csv` | Prepared raw-retention/cleanup eligibility report; deletion not run |
| `pilot_train_signal_sanity.csv`, `pilot_train_signal_sanity_summary.json` | Published 19 targets + VEOG native sanity; summary also accounts for all 9300 full-montage channel-block observations |
| `pilot_bad_channel_metrics.csv`, `pilot_bad_channel_decisions.csv`, `pilot_bad_channel_summary.json` | Pooled Train channel metrics and all 915 subject/channel decisions |
| `pilot_train_window_qc.csv`, `pilot_train_block_qc.csv`, `pilot_train_subject_qc.csv`, `pilot_train_qc_summary.json` | Window/block/subject eligibility, rejection reasons and state balance |
| `pilot_qc_candidate_comparison.csv`, `pilot_native_rate_qc_summary.csv`, `pilot_resampling_summary.json` | Prespecified multiplier sensitivity and native-rate engineering summaries |
| `pilot_processed_manifest.csv`, `pilot_storage_summary.json`, `pilot_aggregation_verification.json` | Verified compact output hashes/sizes, measured/estimated storage and aggregation provenance |
| `pilot_waveform_access_registry_summary.json`, `pilot_reproducibility_summary.json`, `streaming_dry_run_summary.json` | Exact Train-only interval audit, all-15 idempotence, and full-inventory dry-run evidence |
| `outputs/figures/scenario2/pilot_train_qc_summary.png` | Shareable engineering QC figure without Alpha/Connectivity outcomes |
| `README.md`, `docs/PROTOCOL.md`, `docs/DECISIONS.md`, this handoff | Current phase, method rationale, amendments, limits, and exact continuation commands |

Raw/processed EEG containers and arrays, environments, caches, checkpoints, and large temporary artifacts are not committed. Existing metadata/cohort/split/full-inventory outputs remain authoritative and were not replaced by a new block-semantics reconstruction.

## 19. Open questions

1. Across the full 211 structural candidates, which subjects and blocks pass the frozen outcome-blind rules, and what are the final storage/runtime distributions?
2. After separately authorizing mechanical held-out QC without tuning, does the eligible block set support the intended final 5 Train / 1 Validation / 2 Test allocation per state while preserving every exposed pilot block as Train?
3. Which final subjects must be excluded or otherwise handled under a documented deterministic final-cohort/split lock because they cannot meet that allocation?
4. Once that lock exists, what numerical global scaler results from the unique usable final Real Train samples?
5. Only in later explicitly authorized scientific/model phases: which generator/classifier choices are selected on permitted Train/Validation data, and what are the untouched Final Test physiology/prediction/augmentation results?

Demographic distortion cannot currently be evaluated because all participant-linked age/sex/handedness/weight/height values are unavailable. No substitute demographic inference has been made. This handoff does not claim final physiological eligibility or universal generalization from the 15-subject pilot.

## 20. Exact next action

The commands below start from `D:/SHARIF/PROJECT/subject-conditioned-resting-eeg` with the pinned `.venv`. Network execution on `sub-010002` or other non-pilot subjects belongs to the **next phase only**. None of those network/deletion commands was run during this assignment. The current command scope remains provisional Train only.

## NEXT ACTION — SUBJECT-BY-SUBJECT STREAMING PREPROCESSING

1. Inspect the full 211-subject plan without downloading, processing values, writing journals, or deleting raw:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --dry-run --keep-raw
```

2. In the next phase, stream one subject with raw retained:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --subject-id sub-010002 --execute --keep-raw --retries 3 --timeout 60
```

3. If raw deletion is explicitly desired, use the guarded one-subject alternative:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --subject-id sub-010002 --execute --delete-raw-after-success --retries 3 --timeout 60
```

4. Resume sequential multi-subject processing, continuing logged failures and deleting a source directory only after all explicit cleanup safeguards pass:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --execute --continue-on-error --delete-raw-after-success --retries 3 --timeout 60
```

The same command is restartable. Add `--subjects-file path/to/subjects.txt` to restrict it to newline-listed IDs, or repeated `--subject-id` / `--max-subjects` for a bounded run. Use `--keep-raw` in place of the deletion flag when retaining all sources is required. These alternatives are mutually exclusive.

5. Verify the current phase, before adding non-pilot raw subjects, and exercise local idempotence without downloading or deleting:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml outputs/logs/pytest.xml
.\.venv\Scripts\python.exe -m compileall scripts src tests
.\.venv\Scripts\python.exe scripts/data/s2_08_audit_pilot_integrity.py
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/pilot_download_manifest.csv --subject-id sub-010023 --execute --local-existing-only --keep-raw --journal outputs/logs/pilot_idempotence_verify.jsonl
.\.venv\Scripts\python.exe scripts/data/s2_10_verify_preprocessing_freeze.py --require-clean --test-report outputs/logs/pytest.xml
git status --short
git diff --check
git rev-parse HEAD
```

The pilot inventory audit deliberately requires the exact 15 staged subjects; after beginning future non-pilot staging, use the streaming subject's built-in source/header/output verification rather than treating that strict pilot guard as a general-cohort audit. Each completed future subject records its output hash/size/QC/provenance in `streaming_subject_status.jsonl` and its adjacent `.provenance.json` receipt. Keep failed subjects' raw data and partial downloads for diagnosis/resume, and preserve the Train-only boundary until the later mechanical held-out extension is separately authorized.
