# Subject-Conditioned Resting-State EEG

Scenario 2 of the synthetic EEG project studies subject- and eye-state-conditioned generation with MPI-LEMON resting-state EEG. Scenario 1 is frozen separately at `D:/SHARIF/PROJECT/eye-state-diffusion`, commit `2731f3d`, tag `scenario1-final`.

## Current status

The project is **PRE-DOWNLOAD READY**. All scientifically justified metadata-only work is complete. No real EEG waveform has been downloaded, no model or classifier has been trained, and no remote Git push has been performed.

The pre-download structural cohort contains 211 of 215 subjects. It is not the final physiological cohort: every retained subject and block still requires prespecified, outcome-blind waveform QC.

See [`docs/PRE_DOWNLOAD_HANDOFF.md`](docs/PRE_DOWNLOAD_HANDOFF.md) for the complete continuation record and the exact next-phase command. Do not run the download command during the pre-download phase.

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
