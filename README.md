# Subject-Conditioned Resting-State EEG

Scenario 2 of the synthetic EEG project studies subject- and eye-state-conditioned generation with MPI-LEMON resting-state EEG. Scenario 1 is frozen separately at `D:/SHARIF/PROJECT/eye-state-diffusion`, commit `2731f3d`, tag `scenario1-final`.

## Current status

The project is **PREPROCESSING_FROZEN_READY_FOR_STREAMING**. The 15-subject pilot has passed payload/header integrity and supports the frozen Train-only preprocessing and QC method. Compact pilot Train data are stored locally in ignored HDF5 files. No model or classifier has been trained, and nothing has been pushed.

The pre-download structural cohort contains 211 of 215 subjects. It is not the final physiological cohort: every retained subject and block still requires prespecified, outcome-blind waveform QC.

See [`docs/PILOT_PREPROCESSING_HANDOFF.md`](docs/PILOT_PREPROCESSING_HANDOFF.md) for results, limitations, and exact continuation commands. [`docs/PRE_DOWNLOAD_HANDOFF.md`](docs/PRE_DOWNLOAD_HANDOFF.md) is the historical metadata-phase record; its simultaneous full-download strategy is superseded by subject streaming.

The frozen method uses blockwise 1–40 Hz zero-phase filtering, anti-aliased 128 Hz resampling, pooled Train bad-channel/QC references, interpolation within fixed limits, full 61-channel CAR, and ordered 19-channel float32 µV storage. The filter-support edge guard is 3.3828125 seconds; 4-second candidate windows advance by 2 seconds. HDF5 stores continuous blocks and window masks, without duplicating overlapping windows. VEOG guides artifact QC but is not a generated target. Final normalization is specified but remains unfitted.

Only complete provisional Train blocks can pass the waveform access gate; Validation, Test, and interrupted fragments are refused and tested. Every exposed pilot block must remain Train in the future final split, or its subject must be excluded. No held-out waveform has been inspected, no non-pilot EEG has been downloaded, and all pilot raw files remain staged. The final cohort and split remain unfrozen.

## Reproduce and continue

Use the isolated project environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q scripts src tests
.\.venv\Scripts\python.exe scripts/data/s2_10_verify_preprocessing_freeze.py --require-clean
```

Plan the full cohort without network requests or waveform access:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --dry-run
```

Verify/reuse the already staged pilot with raw retained:

```powershell
.\.venv\Scripts\python.exe scripts/data/s2_07_stream_preprocess_subjects.py --manifest outputs/metrics/scenario2/pilot_download_manifest.csv --execute --local-existing-only --keep-raw --journal outputs/logs/pilot_idempotence_verify.jsonl
.\.venv\Scripts\python.exe scripts/data/s2_09_summarize_pilot.py
```

Execution uses one subject at a time and preserves valid partial downloads across connection failures. HTTP Range responses, final size, and SHA256 are checked before atomic promotion. Output/provenance receipts and a durable journal make reruns verifiable. Raw deletion is disabled by default; `--delete-raw-after-success` requires all safeguards to pass. The prepared pilot cleanup command has not been executed.

The current streaming implementation produces provisional Train blocks only. Future mechanical held-out QC requires a separately authorized extension after the method freeze; scientific held-out outcomes remain sealed. See the handoff before any non-pilot execution.

## Verified dataset facts

- 215 subject directories and 215 participant rows.
- Every subject has 62 channel rows: 61 EEG and 1 EOG (`VEOG`), not 62 EEG.
- All 19 target scalp channels are present and marked good for all subjects.
- Native sampling rates are 2500 Hz for 208 subjects, 1000 Hz for 6, and 500 Hz for 1.
- `Stimulus/S200` means EO and `Stimulus/S210` means EC.
- `Stimulus/S208` is mapped to EC only for `sub-010078` and `sub-010126`, using documented subject-level evidence.
- Distributed participant-level age, sex, handedness, weight, and height values are all `n/a`; demographic distortion cannot be tested from the local table.

## Structural cohort and split status

The four structural exclusions are:

- `sub-010015`: only 3 EO/3 EC runs; final EO is truncated.
- `sub-010100`: only 5 EO/6 EC runs; final EC is truncated and the sixth EO is absent.
- `sub-010203`: no events table and zero marker entries in its header-only `.vmrk`.
- `sub-010285`: no EO/EC state markers.

The remaining 211 subjects each have eight complete EO and eight complete EC blocks after explicit acquisition-interruption fragments are separated for `sub-010059` and `sub-010081`. Therefore 5 Train / 1 Validation / 2 Test blocks per subject and state is structurally feasible.

[`outputs/metrics/scenario2/provisional_block_split_manifest.csv`](outputs/metrics/scenario2/provisional_block_split_manifest.csv) is explicitly `PROVISIONAL_METADATA_ONLY`; it is not final or locked. Final membership requires waveform QC, a checksum, and a new commit.

## Reproduce the pre-download package

From the repository root:

```powershell
python scripts/audit/s2_01_audit_metadata.py
python scripts/audit/s2_02_audit_blocks.py
python scripts/audit/s2_03_build_structural_cohort.py
python scripts/audit/s2_04_audit_demographics.py
python scripts/audit/s2_05_plan_splits_and_downloads.py
python -m pytest -q
```

These commands read only local metadata. The prepared downloader is validation-only unless its explicit `--execute` flag is supplied.

## Planned study

The study asks whether subject-conditioned synthetic EEG preserves individual EO-to-EC alpha physiology, spatial alpha structure, and functional connectivity on held-out recording blocks, and whether synthetic augmentation improves held-out real classification across at least three classifiers. Validation and Test remain real-only; complete recording blocks are the atomic split unit.


## 2026-09-05: frozen mechanical held-out authorization

This amendment is written and committed before the first held-out value read.
The user separately authorizes mechanical access to the 90 provisional Validation/Test
blocks of the existing 15 pilot subjects through `MechanicalHeldoutAccessGate`.
The frozen preprocessing YAML and SHA256 remain unchanged:
`5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b`.
The separate `configs/access/mechanical_heldout_v1.yaml` defines access purposes,
manifest hashes and apply-only restrictions; it contains no preprocessing parameters.

Validation/Test may apply verified Train-derived thresholds and bad-channel decisions
but cannot fit any parameter, including normalization. Allowed operations are frozen
preprocessing, fixed waveform QC, compact storage and output verification. Scientific
outcomes remain sealed: no PSD, Alpha, IAF, topology, Connectivity, wPLI, classifier,
generator, TSTR, augmentation, plots, tuning or scientific evaluation is authorized.
All 150 previously exposed pilot blocks must remain Train or their subject must be
excluded. Reuse their existing verified compact datasets without native Train rereads.
Process sub-010250 identically and preserve its frozen Train-QC failure.

Store all 16 complete blocks, including ineligible blocks with masks and reasons, in
separate all-block HDF5 files. Keep the old Train files and original access registry.
Provisional feasibility requires frozen Train QC (at least three eligible blocks per
state), both assigned Validation blocks and all four assigned Test blocks eligible,
plus source/header/output verification. No spare held-out blocks, reassignment,
threshold changes, or subject rescue is permitted. This does not lock a final cohort,
final split or final Test, and normalization remains unfitted.

All-block storage and complete provenance/journal verification are now mandatory
before any future explicitly requested raw deletion, superseding earlier Train-only
cleanup eligibility. No pilot raw deletion, non-pilot download, metadata clone change,
or Git push is authorized in this phase. Future streaming supports `--processing-scope
all-blocks-mechanical`: fit on Train first, then mechanically apply to held-out, verify
all 16 blocks, durably journal, and delete only with the explicit guarded flag.
