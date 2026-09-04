# Scenario 2 Pre-Download Handoff

## 1. Executive status

Project phase: **PRE_DOWNLOAD_READY**.

PRE_DOWNLOAD_READY means every scientifically justified metadata-only audit, structural decision, provisional split feasibility check, exact payload manifest, regression test, and protocol amendment has been completed and committed locally. It does **not** mean that the final physiological cohort or final Test split is locked. No real EEG waveform has been downloaded or analyzed, and no generator, classifier, or preprocessing pipeline has been trained or fitted.

The next phase starts with the explicitly controlled pilot download at the end of this document.

## 2. Git state

- Repository: `D:/SHARIF/PROJECT/subject-conditioned-resting-eeg`
- Branch: `main`
- Starting commit: `7dcb1e2` (`docs: initialize standalone Scenario 2 protocol`)
- Commits created before this handoff:
  - `a8f16a4` — `data: harden MPI-LEMON metadata and block audits`
  - `a623d0b` — `data: define structural cohort and pre-download plans`
  - `d0221b0` — `docs: record pre-download cohort and safeguards`
  - `ad7cbf1` — `data: synchronize metadata inventory output`
- Final handoff/acceptance commit: the current commit containing this file. A Git commit cannot embed its own content-derived identity; use `git rev-parse HEAD` to resolve it. The exact final HEAD is also reported in the completion chat.
- Final working-tree requirement: clean after the handoff commit and verification.
- Push status: nothing was pushed. No push command was run.

Scenario 1 remains untouched in `D:/SHARIF/PROJECT/eye-state-diffusion`, frozen commit `2731f3d`, tag `scenario1-final`.

## 3. Repository files added or modified

Core configuration:

- `configs/audit/metadata_audit.yaml`: dataset root, output root, and 19 target channels.
- `configs/audit/event_code_overrides.yaml`: standard state map, the only two S208 overrides, their evidence, and structural timing thresholds.
- `configs/audit/cohort_eligibility.yaml`: explicit pre-download structural criteria.
- `configs/audit/provisional_split.yaml`: deterministic 5/1/2 algorithm, pilot policy, and NEMAR object base URL.

Reproducible scripts:

- `scripts/audit/s2_01_audit_metadata.py`: safe metadata/type/status/schema/pointer inventory; never opens a materialized EEG payload.
- `scripts/audit/s2_02_audit_blocks.py`: event-context-aware block parser, acquisition-segment handling, override enforcement, and anomaly evidence.
- `scripts/audit/s2_03_build_structural_cohort.py`: machine-readable eligibility decisions.
- `scripts/audit/s2_04_audit_demographics.py`: full-versus-eligible participant metadata comparison.
- `scripts/audit/s2_05_plan_splits_and_downloads.py`: provisional allocation and exact pilot/full payload manifests.
- `scripts/data/s2_06_download_manifest.py`: next-phase staging tool; validates only unless `--execute` is explicitly supplied, writes outside the metadata clone, and checks exact size/SHA256.

Tests:

- `tests/test_metadata_audit.py`: pointer safety, type counts, and duplicate target detection.
- `tests/test_block_audit.py`: normal blocks, subject-only S208, unoverridden S208, missing/no-state events, truncation, gaps, alternation, repeated markers, and explicit acquisition restarts.
- `tests/test_cohort_split_download.py`: eligibility reasons, deterministic/disjoint/order-balanced allocation, and network-free manifest/path validation.

Outputs under `outputs/metrics/scenario2/`:

- metadata inventory, schema/value tables, and summary JSON;
- block inventory, subject summary, and event anomaly evidence;
- structural cohort manifest and summary;
- demographic cohort CSV/JSON;
- provisional block split and split/download summary;
- pilot and full candidate download manifests.

Documentation:

- `README.md`, `docs/PROTOCOL.md`, and `docs/DECISIONS.md` now reflect the verified pre-download evidence and safeguards.

## 4. MPI-LEMON metadata findings

- Dataset metadata clone: `D:/SHARIF/PROJECT/Datasets/MPI-LEMON`
- Clone commit: `16702890507291b5a09fa7390245ec60bff6128e`
- Subject directories: 215
- Unique `participants.tsv` IDs: 215; ID sets match exactly.
- Readable `channels.tsv`: 215/215
- Readable `events.tsv`: 214/215
- EEG JSON, `.eeg` entry, `.vhdr`, and `.vmrk`: 215/215
- True per-subject channel composition: 62 rows = 61 EEG + 1 EOG (`VEOG`) + 0 miscellaneous.
- All 13,330 channel rows are marked `good`.
- All 19 target scalp channels are present exactly once and marked good for 215/215 subjects.
- Sampling rates:
  - 2500 Hz: 208 subjects
  - 1000 Hz: 6 subjects (`sub-010276`, `sub-010277`, `sub-010278`, `sub-010279`, `sub-010280`, `sub-010282`)
  - 500 Hz: 1 subject (`sub-010216`)
- Sidecars declare reference FCz and ground AFz.
- Existing event tables use the columns `onset`, `duration`, `trial_type`, `value`, and `sample`.
- State semantics must use `trial_type`. The numeric `value` is subject-local and not scientifically authoritative.
- For every readable event table, `.vmrk` has the same number of `Mk*` entries and `.vmrk` position equals BIDS `sample + 1`.

The distributed participant columns are `age`, `sex`, `hand`, `weight`, and `height`, but all 215 values in each are `n/a`. Structural-exclusion demographic distortion therefore cannot be assessed from participant-level local metadata. The source README's age-group counts sum to 227, not 215, and are not subject-linked, so they were not substituted.

## 5. Event-code interpretation

- `Stimulus/S200` -> EO (standard)
- `Stimulus/S210` -> EC (standard)
- `Stimulus/S208` -> no global meaning; ignored and flagged unless a subject override exists.

Two subject-only overrides are frozen:

- `sub-010078`: six S208 markers at 9.204--19.204 s continue without a cue, comment, segment boundary, or cadence break into 24 S210 markers at 21.204--67.2052 s. The 30-marker uninterrupted run is EC.
- `sub-010126`: 240 S208 and 240 S200 markers, zero S210, and eight alternating approximately 60 s runs per code support S208 as the subject-wide EC replacement.

No future subject inherits either override automatically. `Stimulus/S  1`, `New Segment/`, and comments never receive EO/EC meaning.

## 6. Full anomaly analysis

### `sub-010015`

Six alternating study runs: 3 EC and 3 EO. The first five contain 30 markers; the final EO has 27 markers from 336.9548 to 388.9552 s. The recording ends at 388.9872 s, so at most 52.0324 s exist after final-EO onset; the conventional 54.0004 s cadence extension would exceed the recording. Structurally excluded for insufficient blocks and a genuinely truncated final EO.

### `sub-010059`

The old 134.723 s first-EC result incorrectly bridged three explicit acquisition segments. The parser now preserves:

- four-marker EC fragment at 9.4744--15.4744 s;
- `New Segment/` plus USB comment at 20.76 s;
- two-marker EC fragment at 29.0436--31.0436 s;
- further `New Segment/` markers at 34.54 and 74.08 s;
- a complete retained main sequence beginning with EC at 84.1964 s.

The 13.5692 and 53.1528 s same-state gaps cross explicit restarts and remain unlabeled. The main sequence has exactly 8 complete EC and 8 complete EO blocks. Retained structurally, with two interruption fragments excluded from study blocks and waveform QC required.

### `sub-010078`

Retained with the documented six-marker S208-to-EC substitution described above. The override yields 8 complete EC and 8 complete EO blocks. Long cross-state intervals remain unlabeled.

### `sub-010081`

The old 92.458 s first-EC result bridged an explicit restart. A four-marker EC fragment at 6.6788--12.6788 s ends before `New Segment/` plus USB comment at 16.5 s. The retained main sequence begins with EC at 39.1352 s and has exactly 8 complete blocks per state. The 26.4564 s same-state interval crosses the restart and is not assigned a state. Retained structurally, with one interruption fragment excluded and waveform QC required.

### `sub-010100`

No state markers occur during the first 170.64 s; a new segment then contains five complete EC/EO pairs followed by a 15-marker EC at 803.8968--831.8972 s. Recording duration is 832.0952 s, providing only a 28.1984 s upper bound from final-EC onset. There is no sixth EO. Structurally excluded for insufficient complete blocks and truncation.

### `sub-010126`

Retained with the subject-wide S208-to-EC override described above. All 16 runs contain 30 markers. A 33.6984 s cross-state transition is flagged and left unlabeled; it is not evidence of a missing block.

### `sub-010203`

No events TSV or events JSON. The 527-byte `.vmrk` is a generated BrainVision header with zero `Mk*` entries. The recording-length sidecar alone cannot reconstruct variable state starts or gaps. Structurally excluded because EO/EC timing is unavailable.

### `sub-010285`

The event table contains only `Comment/no USB Connection to actiCAP` at time 0; no S200, S210, S208, cue, or segment marker is present. Recording duration is only 80.0996 s. Structurally excluded because no state annotation exists.

### Non-2500 Hz subjects

The 500 Hz and six 1000 Hz subjects listed in section 4 each have complete structural metadata and blocks. None is excluded merely for native rate because all exceed the future 128 Hz target. They are intentionally included in the pilot to validate anti-aliasing and resampling.

## 7. Block structure findings

State markers normally occur every approximately 2 s. Repeated same-state markers are collapsed into a run, not treated as separate blocks. Explicit acquisition segments split runs; arbitrary time gaps do not.

Study-run distribution after interrupted fragments are removed:

- 211 subjects: 8 EO / 8 EC
- `sub-010015`: 3 EO / 3 EC
- `sub-010100`: 5 EO / 6 EC
- 2 subjects: 0 EO / 0 EC

Complete-block distribution:

- 211 subjects: 8 EO / 8 EC
- `sub-010015`: 2 complete EO / 3 complete EC
- `sub-010100`: 5 complete EO / 5 complete EC
- 2 subjects: 0 / 0

The inventory retains 2 interrupted EC fragments for `sub-010059` and 1 for `sub-010081`, but marks them `is_study_block=False`. Normal blocks have 30 markers and approximately 60 s cadence-extended span. The parser records marker span, cadence, largest within-run gap, nominal coverage, recording-capped end, segment boundaries, mapping source, completeness, and evidence status.

Across all raw mapped sequences, the only three same-state gaps above 2.1 s are the explicit-restart gaps in subjects 059/081. Cross-state gaps are common (3,180 transitions; median 4.404 s; 95th percentile 19.267 s; maximum 87.637 s), so transition length alone is not an exclusion rule. Intervals over the conservative 30 s audit warning threshold are recorded as unlabeled evidence, not filled or split into invented states.

## 8. Structural cohort

- Full metadata cohort: 215
- Pre-download structurally eligible: 211
- Structurally excluded: 4
- Final physiological cohort: not yet known

Exact exclusions and primary machine-readable reasons:

- `sub-010015`: `insufficient_complete_blocks_for_5_1_2`
- `sub-010100`: `insufficient_complete_blocks_for_5_1_2`
- `sub-010203`: `state_annotations_unavailable_missing_events`
- `sub-010285`: `state_annotations_unavailable_no_state_markers`

Every retained subject has `requires_waveform_qc=True`. Required future checks include payload integrity, BrainVision linkage, target-channel signal validity, bad channels, flat/noisy channels, artifacts, usable duration, and usable window counts under frozen outcome-blind rules. Subjects 059/081 additionally carry acquisition-restart evidence; 078/126 carry event-code overrides; the seven rare-rate subjects require resampling validation.

## 9. Split feasibility

The 5 Train / 1 Validation / 2 Test design is structurally feasible for every one of the 211 retained subjects and both states. The provisional manifest contains 3,376 assignments (`211 * 2 * 8`). Each subject/state has exactly 5/1/2 disjoint blocks.

Algorithm `scenario2-provisional-split-v1` orders subjects by SHA256 of a fixed seed plus subject ID, applies rank-modulo-eight cyclic rotations to `[Test, Train, Train, Validation, Train, Test, Train, Train]`, and offsets EC by four positions. Across each state and block ordinal, Train has 131--132 subjects, Validation 26--27, and Test 52--53.

The manifest is `PROVISIONAL_METADATA_ONLY`, not final or locked. Post-download outcome-blind QC may change eligibility. The final manifest must then be deterministically generated, checksummed, committed, and frozen. Final Test signals/outcomes cannot be inspected until preprocessing, models, hyperparameters, seeds, metrics, and statistics are frozen.

Pilot method development may access provisional Train blocks only. Because a downloaded `.eeg` contains a whole recording, any complete pilot block actually inspected during development must remain Train in the final manifest, or the subject must be excluded. Pilot provisional Validation/Test blocks must not guide methods.

## 10. Pilot download plan

- Pilot subjects: 15
- Assets to fetch: 30 (one `.eeg` waveform and one `.vhdr` header per subject)
- Pointer-declared payload: 7,012,992,712 bytes (6.531 GiB)
- Existing `.vmrk`, JSON, TSV, and scans metadata remain local and will be copied into the staging tree by the prepared tool.

Subjects and reasons:

- `sub-010216`: 500 Hz resampling case
- `sub-010276`, `sub-010277`, `sub-010278`, `sub-010279`, `sub-010280`, `sub-010282`: all 1000 Hz resampling cases
- `sub-010078`, `sub-010126`: both S208 override cases
- `sub-010059`, `sub-010081`: both resolved acquisition-restart cases
- `sub-010023`, `sub-010071`, `sub-010134`, `sub-010250`: deterministic ordinary 2500 Hz baselines

Exact paths, URLs, sizes, SHA256 digests, and selection reasons are in `outputs/metrics/scenario2/pilot_download_manifest.csv`.

## 11. Full candidate download plan

- Subjects: 211
- Assets: 422 = 211 `.eeg` waveforms + 211 `.vhdr` headers
- Pointer-declared payload: 134,813,081,876 bytes (125.554 GiB)
- Manifest: `outputs/metrics/scenario2/full_candidate_download_manifest.csv`

The metadata export stores ordinary small text pointers of the form `/annex/objects/SHA256E-s<SIZE>--<SHA256>.<ext>`. There is no local `.git/annex`, DataLad configuration, Git LFS configuration, or installed git-annex command. The source repository's archive workflow maps each key to `https://nemar.s3.us-east-2.amazonaws.com/nm000179/objects/<KEY>`.

Across all 215 metadata subjects, 430 backed `.eeg`/`.vhdr` pointers declare 136,275,336,096 bytes. The local `.nemar/availability-report.json` differs slightly from the per-key sum, and `.nemar/metadata.json` says 431 files rather than the 430 pointers found. Therefore every staged asset must be verified against its own pointer size and SHA256; aggregate metadata is not used as an integrity check.

The intended commands are:

```powershell
# Pilot first
python scripts/data/s2_06_download_manifest.py --manifest outputs/metrics/scenario2/pilot_download_manifest.csv --destination data_raw/MPI-LEMON --metadata-root D:/SHARIF/PROJECT/Datasets/MPI-LEMON --execute

# Full candidate only after the Train-only pilot validates the loader/resampling
# and outcome-blind QC rules have been frozen
python scripts/data/s2_06_download_manifest.py --manifest outputs/metrics/scenario2/full_candidate_download_manifest.csv --destination data_raw/MPI-LEMON --metadata-root D:/SHARIF/PROJECT/Datasets/MPI-LEMON --execute
```

**Neither command was run in the pre-download phase.** The script stages under this project's ignored `data_raw/` tree and never replaces pointers in the read-only metadata clone.

## 12. Tests and validation

- Environment: Python 3.12.3, pandas 2.2.2, PyYAML 6.0.1, pytest 9.1.1 on Windows 11.
- `python -m pytest -q`: **17 passed**.
- Python compile check: passed for `scripts/` and `tests/`.
- Full pipeline regeneration into an isolated temporary directory: **15/15 generated CSV/JSON outputs were byte-identical by SHA256** to the repository versions.
- Pilot manifest validation-only run: 30 assets for 15 subjects and 7,012,992,712 bytes validated; it explicitly made no network request and no filesystem write.
- Materialized `.eeg` payload guard: 0 files above the pointer-size threshold; all 215 remain pointer stubs.
- Generated anomaly evidence rows: 83.

Warnings/limitations:

- Participant-level demographics are unavailable.
- The local dataset README contains inaccurate universal 2500 Hz and 62-EEG statements.
- NEMAR aggregate availability/file-count metadata differs slightly from the per-pointer inventory.
- The downloader's actual network behavior cannot be exercised without crossing the deliberate phase boundary; size/SHA256 verification is mandatory during the next phase.

## 13. Open questions requiring waveform data

1. Do every `.vhdr`, `.vmrk`, and `.eeg` link and sample count agree after staging?
2. Are the 19 target signals present, finite, non-flat, correctly scaled, and usable in every retained block?
3. What outcome-blind bad-channel, artifact, and minimum-usable-duration thresholds should be frozen using provisional Train-only pilot data?
4. Does anti-aliasing/downsampling from 500, 1000, and 2500 Hz to 128 Hz behave consistently?
5. Which blocks pass the frozen waveform-QC rules, and does 5/1/2 remain feasible afterward?
6. How should incomplete usable windows be handled without crossing block boundaries?
7. Only after the final cohort/split is locked: which generator/classifier settings are selected on Train/Validation, and what are the untouched Final Test results?

## 14. Exact next step

## NEXT ACTION — REAL EEG DOWNLOAD

In the **next phase only**, from the Scenario 2 repository root, execute the audited 15-subject pilot download:

```powershell
python scripts/data/s2_06_download_manifest.py --manifest outputs/metrics/scenario2/pilot_download_manifest.csv --destination data_raw/MPI-LEMON --metadata-root D:/SHARIF/PROJECT/Datasets/MPI-LEMON --execute
```

Then verify all 30 assets and enforce the Train-only analytical-access rule before any waveform inspection. Do not start with the full 211-subject manifest.
