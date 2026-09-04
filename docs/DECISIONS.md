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
