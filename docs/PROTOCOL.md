# Scenario 2 Study Protocol

Version: 0.2
Status: Pre-download protocol, amended before waveform analysis and before Final Test

## 1. Objective

The study investigates `p(EEG | subject, eye_state)` in MPI-LEMON resting-state EEG. The primary generative question is whether synthetic EEG preserves individual EO-to-EC changes in alpha physiology, spatial alpha structure, and functional connectivity on recording blocks excluded from generator training. The augmentation question is whether adding synthetic EEG improves real held-out eye-state classification across at least three classifiers.

## 2. Pre-download dataset audit

The metadata clone contains 215 subjects. Each `channels.tsv` has 62 rows comprising 61 EEG and 1 EOG (`VEOG`); all 19 target scalp channels are present and marked good. Native sampling frequencies are 2500 Hz (208 subjects), 1000 Hz (6), and 500 Hz (1). All are structurally compatible with a future 128 Hz target, subject to resampling validation.

The distributed `participants.tsv` contains 215 IDs, but age, sex, handedness, weight, and height are all `n/a`. Those fields cannot be used for subject-level balance or distortion analysis. Dataset README aggregates are not substituted because they are not subject-linked and their stated age-group counts sum to 227 rather than 215.

## 3. Event semantics and block reconstruction

State semantics use `trial_type`, never the numeric `value` column:

- `Stimulus/S200` -> EO
- `Stimulus/S210` -> EC

`Stimulus/S208` is not globally defined. Two evidence-backed subject overrides are frozen before waveform analysis:

- `sub-010078`: six initial S208 markers continue seamlessly at approximately 2 s cadence into 24 S210 markers, making one 30-marker EC run.
- `sub-010126`: 240 S208 and 240 S200 markers form eight alternating approximately 60 s runs per code, with no S210; S208 is the subject-specific EC replacement.

Repeated same-state markers form one state run. A numeric time gap alone never creates a block and never licenses an imputed state. An explicit `New Segment/` marker starts a new acquisition segment and prevents a run from bridging that boundary. `Stimulus/S  1` and comments provide context only; they have no EO/EC semantics.

`sub-010059` and `sub-010081` contain explicit acquisition restarts. Their aborted initial EC fragments are preserved as evidence but excluded from the primary block sequence. Each post-restart main sequence contains eight complete EO and eight complete EC blocks. Unlabeled inter-block and interruption intervals remain unlabeled.

## 4. Pre-download structural cohort

This cohort uses no physiological, predictive, or generative outcome. Eligibility requires:

1. readable channel metadata and all 19 target channels;
2. available EO/EC `trial_type` annotations;
3. eight alternating study blocks per state;
4. eight complete 30-marker, approximately 60 s blocks per state after explicit interruption fragments are separated.

The resulting pre-download structural cohort contains 211 subjects. It is not the final physiological cohort.

Four subjects are excluded structurally:

- `sub-010015`: 3 EO/3 EC runs, with a truncated 27-marker final EO.
- `sub-010100`: 5 EO/6 EC runs, with a truncated 15-marker final EC and no sixth EO.
- `sub-010203`: EO/EC annotations unavailable; no events table and no `Mk*` entries in `.vmrk`.
- `sub-010285`: events contain no state marker.

All 211 retained subjects require waveform integrity, bad-channel, signal-quality, artifact, duration, and usable-window QC after download.

## 5. Block-level split rule

Complete recording blocks are the atomic unit. Random window-level splitting is prohibited. Windows cannot cross block boundaries; filtering and other time-dependent preprocessing must not bridge blocks. Fitted normalization, transforms, generator training, and classifier training use Train data only. Hyperparameter selection uses real Validation only. Final Test remains real-only.

The study is a within-subject block holdout: subject identities occur in all partitions. It is not a held-out-subject generalization design.

## 6. Provisional metadata-only allocation

The 211-subject structural cohort supports 5 Train / 1 Validation / 2 Test blocks per subject and state. A deterministic candidate is stored as `provisional_block_split_manifest.csv` with status `PROVISIONAL_METADATA_ONLY`.

Subjects are ordered by SHA256 of a fixed version seed and subject ID. A fixed 5/1/2 base schedule is rotated by subject rank modulo eight, with a four-position EC offset. This makes every block order appear in every role and balances each state/order cell within one subject of perfect balance. No waveform or outcome enters the allocation.

This candidate is not final or locked. After outcome-blind QC rules are frozen and applied uniformly, the final eligible block set must be generated deterministically, checksummed, committed, and then frozen.

## 7. Pilot analytical-access rule

Downloading a BrainVision `.eeg` provides a subject's whole recording, including provisional Validation/Test blocks. During pilot pipeline development, only provisional Train blocks may be read or inspected, plus already-excluded interruption fragments strictly for loader diagnostics. Any complete pilot block exposed during method development must remain Train in the final manifest, or its subject must be excluded. Pilot Validation/Test signals must not influence filters, resampling, QC rules, artifact thresholds, or implementation choices.

Possession on disk is not permission for analytical access. Access should be logged in the post-download phase.

## 8. Final-Test lock sequence

Before the final manifest is created:

1. validate BrainVision links and resampling in the Train-only pilot;
2. freeze outcome-blind waveform-QC and block-eligibility rules;
3. apply those rules uniformly to the full candidate cohort;
4. generate, checksum, commit, and lock the final cohort/split manifest.

Before Final Test is opened, additionally freeze preprocessing, normalization, model architectures, spectral-loss weight, checkpoint rule, CFG, classifiers and hyperparameters, augmentation ratio, IAAFT settings, random seeds, metrics, and the statistical analysis plan. Tuning uses Train and real Validation only. Final Test is evaluated once under the prespecified plan.

QC quantities may determine pass/fail eligibility and feasibility under frozen rules. They must not be optimized to construct an unusually clean or easy Validation/Test set. Alpha power, IAF, connectivity, classifier accuracy, generator loss, TSTR, augmentation utility, and any final scientific outcome may never drive inclusion or allocation.

## 9. Planned models and evaluation

Generators:

- B0: eye-state-only DDPM
- B1: subject + eye-state DDPM
- B2: subject + eye-state DDPM with multi-resolution spectral regularization

Classifiers:

- log-bandpower + logistic regression
- covariance/tangent-space + linear SVM
- EEGNetv4

Augmentation ratios are 0.5x, 1.0x, and 2.0x, with an IAAFT spectral-surrogate control. The primary classification metric is subject-macro balanced accuracy. Secondary metrics include accuracy, macro-F1, ROC-AUC, MCC, state recalls, and confusion matrices. Primary comparisons use paired subject-level analysis with subject-level bootstrap confidence intervals. Debiased alpha-band wPLI is the primary connectivity candidate and an evaluation target, not a main-model training loss.

## 10. Amendments

Every scientifically meaningful amendment must be recorded in `docs/DECISIONS.md`, committed, and marked relative to waveform analysis, Validation outcomes, and Final Test. No protocol change may respond to Final Test results.
