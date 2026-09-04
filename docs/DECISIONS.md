# Scenario 2 Decision Log

## 2026-09-04  Standalone repository

Scenario 2 is maintained in a separate repository:

`subject-conditioned-resting-eeg`

The frozen Scenario 1 repository will not be modified.

---

## 2026-09-04  Primary dataset

MPI-LEMON is the primary dataset for Scenario 2.

---

## 2026-09-04  Block-level splitting

Train, Validation, and Test splitting must occur at complete recording-block level.

Window-level splitting is prohibited.

---

## 2026-09-04  Distribution-preserving split

The split must preserve the original dataset distribution as closely as possible using non-outcome metadata and QC measures.

Physiological and predictive outcomes must not be used to optimize the split.

---

## 2026-09-04  Main generator ablation

The primary models are:

- B0: State-only DDPM
- B1: Subject + State DDPM
- B2: Subject + State DDPM + Spectral Regularization

---

## 2026-09-04  Connectivity

Functional Connectivity is a major evaluation domain but is not initially included in the training loss.

Primary candidate:

debiased wPLI in the Alpha band.

---

## 2026-09-04  Three classifiers

The augmentation study will use:

- Logistic Regression
- Tangent-Space Linear SVM
- EEGNetv4

---

## 2026-09-04  Augmentation ratios

Planned ratios:

- 0.5x
- 1.0x
- 2.0x

Validation and Test remain real-only.

---

## 2026-09-04  IAAFT control

IAAFT spectral-surrogate augmentation is included as a control experiment.

---

## Pending after MPI-LEMON audit

The following are intentionally not yet locked:

- usable subject cohort
- exact number of EO/EC blocks
- exact Train/Validation/Test block allocation
- final channel availability
- bad-channel eligibility rule
- artifact-QC thresholds
- block-duration eligibility rule

---

## 2026-09-05  Pre-download audit amendments

Timing: all decisions in this section were made before any real EEG waveform was downloaded or analyzed, before examining any Validation outcome, and before Final Test.

### Channel composition and sampling rates

The 62 rows in every `channels.tsv` are classified by BIDS type as 61 EEG and 1 EOG (`VEOG`), with 0 miscellaneous channels. They must not be reported as 62 EEG channels. All 19 target scalp channels are present and marked good in all 215 subjects.

Native sampling is non-uniform: 208 subjects are 2500 Hz; `sub-010216` is 500 Hz; `sub-010276`, `sub-010277`, `sub-010278`, `sub-010279`, `sub-010280`, and `sub-010282` are 1000 Hz. These seven are retained structurally because every rate supports the planned 128 Hz target. Resampling behavior remains a waveform-pilot question.

### State-code semantics

State meaning comes only from `trial_type`: S200 is EO and S210 is EC. S208 is not a global alias. It maps to EC only for `sub-010078` and `sub-010126`, through `configs/audit/event_code_overrides.yaml`, with the observed cadence and alternating-block evidence recorded there.

### Acquisition interruptions in sub-010059 and sub-010081

Explicit `New Segment/` markers prove that the apparent long first EC spans are merged acquisition fragments, not continuous state blocks. The parser separates and preserves two aborted EC fragments for `sub-010059` and one for `sub-010081`. Their post-restart sequences each contain eight complete EO and eight complete EC blocks, so both subjects remain structurally eligible and waveform-QC flagged. No EO/EC label is imputed inside any interruption or transition gap.

### Pre-download structural cohort

The cohort is defined by readable structural metadata, complete target-channel coverage, usable EO/EC annotations, and eight complete alternating blocks per state. It contains 211 subjects. It is explicitly not the final physiological cohort.

Structural exclusions are:

- `sub-010015`: insufficient blocks (3 EO/3 EC) and a truncated final EO.
- `sub-010100`: insufficient blocks (5 EO/6 EC), a truncated final EC, and no sixth EO.
- `sub-010203`: unavailable EO/EC annotations; events are absent and `.vmrk` has no marker entries.
- `sub-010285`: no state markers.

Every retained subject requires uniform, prespecified waveform QC. No alpha, IAF, connectivity, classifier, generator, TSTR, or augmentation outcome contributed to these decisions.

### Participant metadata limitation

All 215 values for age, sex, handedness, weight, and height in the distributed `participants.tsv` are `n/a`. Full-versus-retained demographic distortion therefore cannot be evaluated locally. Dataset README age-group aggregates are not substituted: they are not subject-linked and sum to 227 rather than the 215 EEG subjects.

### Provisional 5/1/2 feasibility

Five Train, one Validation, and two Test blocks per subject and state is structurally feasible for all 211 retained subjects. The metadata-only candidate uses a fixed SHA256 subject ordering, balanced cyclic block-order rotations, and an EO/EC offset. It is labeled `PROVISIONAL_METADATA_ONLY`, not final or locked.

Final membership waits for frozen outcome-blind waveform QC. After QC, the final manifest must be deterministically generated, checksummed, committed, and frozen. QC can enforce prespecified eligibility but cannot be optimized to make held-out sets cleaner or easier.

### Pilot access and leakage prevention

Because one `.eeg` file contains a whole recording, pilot download does not authorize inspection of every contained block. Pipeline development may access provisional Train blocks only, plus already-excluded fragments for loader diagnostics. Exposed complete blocks must remain Train in the final manifest or their subject must be excluded. Pilot Validation/Test signals cannot guide preprocessing or QC choices.

### Download planning

The metadata clone stores 215 `.eeg` and 215 `.vhdr` NEMAR annex-key pointer texts; it is not a functional git-annex/DataLad checkout. The full structural candidate manifest contains 422 assets for 211 subjects (134,813,081,876 declared bytes). The deterministic pilot contains 30 assets for 15 subjects (7,012,992,712 declared bytes) and covers all rare sampling rates, both S208 overrides, both resolved acquisition-interruption cases, and four ordinary 2500 Hz controls.

The future downloader stages outside the read-only metadata clone, verifies per-file size and SHA256, and is dry-run unless `--execute` is supplied. No download command was executed during this amendment.
