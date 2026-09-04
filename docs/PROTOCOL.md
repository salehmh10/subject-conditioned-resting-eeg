# Scenario 2 Study Protocol

Version: 0.1
Status: Pre-audit protocol

## 1. Main objective

The study investigates subject-conditioned synthetic resting-state EEG generation under Eyes Open (EO) and Eyes Closed (EC) conditions.

The main question is whether synthetic EEG preserves individual EO-to-EC physiological changes on recording blocks excluded from generator training.

## 2. Dataset

Primary dataset:

MPI-LEMON

Target configuration:

- resting-state EEG
- EO and EC conditions
- approximately 19 scalp channels
- target sampling rate: 128 Hz
- target window length: 4 s
- target training stride: 2 s

Exact usable subjects, blocks, and channels will be determined during dataset audit.

## 3. Split rule

The split unit is a complete recording block.

Window-level random splitting is prohibited.

All windows belonging to a block must belong exclusively to:

- Train
- Validation
- Test

## 4. Distribution-preserving principle

Train, Validation, and Test must remain representative of the original dataset distribution as closely as reasonably possible.

The split may consider non-outcome information such as:

- subject identity
- eye state
- block order
- usable duration
- number of usable windows
- artifact rejection rate
- bad-channel burden
- general QC measures

The following must NOT be used to optimize the split:

- Alpha effect
- Individual Alpha Frequency
- Functional Connectivity
- Classifier performance
- Generator performance
- TSTR
- Augmentation performance

## 5. Provisional split

If the dataset audit confirms eight usable blocks per state per subject:

- 5 blocks Train
- 1 block Validation
- 2 blocks Test

This allocation is provisional and will not be locked until the dataset audit is complete.

## 6. Test policy

Final Test must not be used for tuning.

Before opening Test, the following must be locked:

- preprocessing
- split manifest
- normalization
- generator architecture
- spectral-loss weight
- checkpoint rule
- CFG
- final generator
- classifier hyperparameters
- augmentation ratio
- IAAFT settings
- random seeds
- metrics
- statistical analysis

## 7. Generative models

### B0
Eye-State-only DDPM.

### B1
Subject + Eye-State conditioned DDPM.

### B2
Subject + Eye-State DDPM with multi-resolution spectral regularization.

## 8. Connectivity

Functional Connectivity is an evaluation target, not a training loss in the main study.

Primary candidate:

- debiased wPLI
- Alpha band

## 9. Classification

Three classifiers are planned:

- Log-bandpower + Logistic Regression
- Tangent-Space + Linear SVM
- EEGNetv4

## 10. Synthetic augmentation

Planned conditions:

- Real only
- Synthetic only
- Real + 0.5x Synthetic
- Real + 1.0x Synthetic
- Real + 2.0x Synthetic

Validation and Test remain real-only.

## 11. IAAFT control

IAAFT spectral-surrogate augmentation will be used to determine whether augmentation benefit is explainable mainly by spectral-statistical replication or requires richer synthetic structure.

## 12. Primary classification metric

Subject-macro Balanced Accuracy.

Secondary metrics include:

- Accuracy
- Macro-F1
- ROC-AUC
- MCC
- Recall EO
- Recall EC
- Confusion Matrix

## 13. Statistics

Primary comparisons will use paired subject-level analysis with subject-level bootstrap confidence intervals.

## 14. Protocol amendments

Any scientifically meaningful protocol change must:

1. be documented in `docs/DECISIONS.md`
2. be committed to Git
3. include its rationale
4. state whether it occurred before or after examining Validation outcomes

No protocol change may be made in response to Final Test results.
