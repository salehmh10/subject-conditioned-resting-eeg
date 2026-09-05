# Scenario 2 Study Protocol

Version: 0.3
Status: Pilot preprocessing protocol; amended before held-out waveform access and model development

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

Downloading a BrainVision `.eeg` provides a subject's whole recording, including provisional Validation/Test blocks. During pilot pipeline development, only complete provisional Train study blocks may be read or inspected. Interrupted fragments are prohibited even for diagnostics. Any complete pilot block exposed during method development must remain Train in the final manifest, or its subject must be excluded. Pilot Validation/Test signals must not influence filters, resampling, QC rules, artifact thresholds, or implementation choices.

Possession on disk is not permission for analytical access. `TrainAccessGate` joins the authoritative split and block inventory, refuses held-out/incomplete/interrupted blocks, and durably logs permitted sample intervals before reading. A cross-process lock allows one subject loader at a time; individual reads are bounded to 65 seconds. Opaque SHA256 byte reads and header metadata inspection are permitted over the whole recording.

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

## 11. Pilot preprocessing specification

The canonical configuration is `configs/preprocessing/frozen_preprocessing_v1.yaml`; its canonical JSON SHA256 is stored beside it. Software versions are pinned in `requirements-lock.txt` and enforced against the config. Values remain in physical units; numerical normalization is not fitted here.

Read a complete Train block using MNE `preload=False`, preserving native order, 61 EEG and VEOG. Assign `standard_1005` template coordinates (individual digitizations are unavailable). Filter each block independently with MNE Hamming/firwin zero-phase FIR, passband 1–40 Hz, transitions 0.5–1 Hz and 40–50 Hz, `reflect_limited` padding. FIR lengths at 500/1000/2500 Hz are 3301/6601/16501 samples. Resample with SciPy polyphase, explicit Kaiser beta 5 FIR of length `20*max(up,down)+1`, line padding, to 128 Hz; output length is the ceiling of native samples times 128/native rate.

The FIR half-support is 3.3 seconds and polyphase half-support is 0.078125 seconds. Round their sum upward to the output grid: trim 433 samples = 3.3828125 seconds from each edge. All subject-level metric calculations use the interior. The same finite-support rule applies to every subject/state. This excludes dependence on boundary padding, not interior artifact ringing. Synthetic 5/20/35 Hz tones test preservation; 60/100 Hz tones test stopband/alias suppression.

After resampling, fit bad channels over pooled EO and EC Train blocks. Per channel, median-across-block broadband robust scale, derivative robust scale, and median absolute correlation with the other EEG channels are used. Scale is `1.4826*MAD`. Log10 scale and derivative have cross-channel robust z-scores with denominator floor 0.1; absolute z > 6 and correlation < 0.1 are votes. Ordinary failure requires low correlation plus an amplitude or derivative outlier; amplitude and derivative alone cannot supply independent corroboration. Any native nonfinite sample, or native robust scale < 1e-6 µV in at least half the Train blocks, is catastrophic. Allow at most six bad EEG and two bad targets. Eligible channels use MNE accurate spherical-spline interpolation at fixed head origin (0,0,0.04 m) before CAR. Nonfinite samples are zero-filled solely to permit this computation; native sanity records preserve the failure. If channel limits fail, retain a labeled diagnostic compact output with all windows rejected; never silently accept it.

Apply CAR over all 61 EEG after interpolation, excluding VEOG. Select the ordered 19 channels from the config and convert volts to µV by 1e6. VEOG receives the same filter/resampling and edge handling, remains unreferenced, and is used only for QC. No ICA, ICLabel, ASR, spectral enhancement, or outcome analysis is included.

After trimming, define 512-sample windows with stride 256 (4/2 seconds), within each block. Features are maximum target peak-to-peak, maximum target robust scale, maximum absolute target sample difference, and VEOG peak-to-peak. Fit one threshold vector from the subject's pooled finite, non-flat Train windows: `min(cap, max(floor, median + 8*1.4826*MAD))`. Floors in µV are [150,30,50,250]; caps [1000,200,300,1500]. A window fails any exceeded threshold, any nonfinite value, any target scale < 0.01 µV, or any target repeated-difference fraction > 0.99. Any native nonfinite VEOG sample rejects every window in that block. Native nonfinite, scale <1e-6 ?V, or repeated-difference fraction >0.99 on an unrepaired channel also rejects the block, so CAR cannot hide local flatness. A block requires at least 10 usable windows and 50% usable fraction. A pilot subject requires at least three usable Train blocks per state and valid channel limits. This is a pilot engineering eligibility rule, not a declaration that the final 5/1/2 allocation is feasible after QC.

If no finite non-flat Train fitting windows exist, use the fixed caps only for diagnostic bookkeeping, mark every window rejected, and fail subject QC; this is not a fitted reference.

Candidate multipliers 6/8/10 are reported for engineering sensitivity; 8 is predeclared, not optimized for physiology or group separation. If the pilot reveals an engineering failure, changes require a documented amendment and regeneration before a freeze is claimed.

Store one HDF5 per subject with continuous trimmed 19-channel float32 µV blocks, gzip level 4, shuffle, chunks (19,1024), candidate starts, usability masks/reasons, and provenance. VEOG and overlapping windows are not stored. A temporary 62-channel float64 cache is disk-backed and removed after processing; only one native block is in memory. Byte hashes verify containers; independent rebuilds must have equal scientific arrays/masks, while creation timestamps can differ.

Final normalization is global channelwise mean and population standard deviation (ddof 0), fitted only after final cohort/split lock from unique continuous time samples covered by usable windows in eligible final Real Train blocks. Count overlapping samples once. Fail fitting if any channel standard deviation is below 1e-6 µV. Apply that same scaler to Real Validation, Real Test, and Synthetic data; no per-state/per-window fit.

## 12. Subject streaming and raw retention

The 211-subject manifest remains the authoritative inventory, but simultaneous full download is superseded by one-subject streaming. Download/resume and verify one source pair, copy validated sidecars, check headers, derive subject parameters from Train only, produce/verify compact output, durably journal completion, then proceed. The current implementation processes provisional Train only. Processing held-out blocks mechanically for final QC requires a separately authorized extension after the preprocessing freeze and before final split lock; held-out scientific outcomes remain sealed.

The downloader supports strict HTTP Range resume, safe full-response restart, exact size/SHA256 verification and atomic promotion, bounded retries, and a durable journal. Network tests use mocked responses only. Raw retention is the default. Explicit `--delete-raw-after-success` additionally requires fresh source/output/provenance verification, passing subject QC, a durable complete journal, accessible metadata, and confined paths with no unresolved partials. Pilot raw cleanup is prepared but is not executed in this phase.


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
