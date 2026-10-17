# Methods of the completed study

This document describes the final selected models, final utility experiment, and completed exploratory Validation-reference analysis. Historical protocols remain available as evidence of earlier stages; a candidate's name or an early contract does not override the final frozen selection. Historical identifiers B0/B1/B2 correspond to paper names SC/SSC/SSC-C, respectively.

## Cohort and split

The study uses MPI-LEMON resting EEG from 198 included participants. The original 215 participants are accounted for by 198 included participants, four structural exclusions, and 13 subsequent exclusions based on frozen engineering and feasibility evidence.

For each participant and eye state, five complete recording blocks are assigned to Train, one to Validation, and two to Test. There are 1,980 Train, 396 Validation, and 792 Test assignments. Ten Train blocks remain ineligible without replacement, leaving 1,970 eligible Train blocks. The same 198 participants appear in every split; the unit of generalization is a held-out recording block of a known participant.

Eligible windows number 47,835 / 9,635 / 19,131 for Train / Validation / Test. Final Test contains 9,353 EO and 9,778 EC windows. EO is encoded as 0 and EC as 1. Four-second windows contain 512 samples and advance by 256 samples; windows never cross block boundaries.

## Preprocessing and normalization

The frozen pipeline applies a zero-phase Hamming-window FIR bandpass from 1–40 Hz, with transition widths of 0.5 and 10 Hz, followed by polyphase resampling to 128 Hz. Each block edge has a 433-sample guard. Bad-channel decisions and artifact/QC thresholds are fitted using the participant's pooled EO/EC Train data and applied unchanged to held-out blocks. Bad channels are interpolated with spherical splines. Common-average reference uses all 61 EEG channels after interpolation, excluding VEOG; the 19 target scalp channels are then selected. No ICA is used.

Target channel order is Fp1, Fp2, F7, F8, F3, F4, Fz, T7, T8, C3, C4, Cz, P7, P8, P3, P4, Pz, O1, O2. Stored signals are in microvolts.

One global per-channel mean and population standard deviation is fitted from the union of usable sample intervals in eligible Train blocks. Overlap is counted once: 12,881,152 unique time samples correspond to 100,634 seconds. The scaler is shared across participants and states and applied unchanged. The historical `usable_train_window_count=47898` includes 63 mask-usable windows from ineligible blocks; the scientific fitting count is `fitting_train_window_count=47835`.

## Generators

All final generators use a temporal 1D U-Net with input/output shape 19 × 512, widths 64/128/256/256, two residual blocks per level, GroupNorm, SiLU, and four-head attention at the deeper levels and bottleneck. All final selected models predict v under a cosine diffusion schedule. Sampling uses ancestral DDPM with 1,000 reverse steps in FP32, batch size 64, and no waveform clipping in the sampling loop.

| Model | Conditioning | Selected state | Parameters |
|---|---|---|---:|
| B0 = SC | Eye state | Raw epoch 20 of v fine-tuning | 8,402,771 |
| B1 = SSC | Eye state and participant | Raw epoch 10 of B0-initialized fine-tuning | 8,526,931 |
| B2 = SSC-C | Eye state and participant | Epoch 1 of the selected continuation run | 8,526,931 |

The B1/B2 subject branch contains a 198 × 128 embedding table and a 128→256→256 projection, added to the state-conditioning representation. B2 uses the same architecture as B1 and **zero spectral-loss weight**. Earlier spectral candidates are part of development history.

B0 inherits epsilon pretraining with Validation-based selection before v fine-tuning. B1 inherits B0; Validation loss was monitored during training. B2 was selected after inspecting development Validation, including both cohorts labeled VAL_DEV and VAL_CONF. Their different histories prevent a controlled causal attribution to the subject condition alone. The frozen B2 record explicitly does not establish replication of the continuation recipe across training seeds.

Each final synthetic pool contains 47,835 windows matched to Real Train participant × state counts. B0's participant labels identify comparison strata; no subject identity is supplied to B0's network.

## Original Train-reference fidelity

Saved characterization covers spectral, amplitude, temporal, spatial, diversity, and copy-related descriptors. Full-pool fidelity is relative to Real Train, aggregated equally over participants and states where applicable. Alpha effects compare EC minus EO within participant. Separate Real–Real calibration uses disjoint Train-block halves and matched nonoverlapping window counts; both halves were available during generator training.

Connectivity uses debiased squared wPLI, estimated per FFT bin and averaged within bands. Finite negative estimates are retained. These are sensor-space descriptors. Fingerprints use 76 median log-bandpower features and rank a synthetic participant against 198 Real Train references.

## Exploratory Validation-reference analysis

The completed supplementary analysis uses all 47,835 eligible Train windows, all 9,635 Validation windows, and each generator's existing 47,835-window pool for the same 198 participants. It preserves the original metric definitions and contrasts each source with real Validation profiles. Validation was excluded from generator gradient training but used in development and checkpoint selection; spectral fidelity informed SSC-C selection with a different aggregation and additional eligibility criteria. This is exploratory post hoc evidence, not independent confirmation. No generator was retrained, reselected, or resampled. Only Train and Validation waveform datasets were indexed in the archived analysis; its whole-file integrity checks also covered HDF5 files containing unused Test datasets.

The archived plan fixed settings, source hashes, and random selections before supplementary waveform access. The count-matched sensitivity analysis uses one subset without replacement of 15–25 windows per participant/state, totaling 9,635 per source. All Validation windows are used; Train and synthetic sources are subsampled, with the same synthetic indices across B0/B1/B2. Matching counts does not match temporal dependence or recording-block diversity: Validation supplies one block per state, whereas Train spans several.

Processed values in microvolts are standardized with the original global Train-derived scaler to float32 and promoted to float64 for analysis; physical-unit metrics invert the same scaler. Welch PSD uses a 256-sample Hann window, 128-sample overlap, constant detrending, one-sided density scaling, and 128-Hz sampling. Linear spectra are averaged over windows before conversion to dB, with spectral MAE over 19 channels and inclusive 1–40 Hz bins. Channel-SD MAE compares pooled channel population standard deviations in microvolts. Band powers use trapezoidal integration over inclusive 1–4, 4–8, 8–13, and 13–30 Hz intervals. Alpha reactivity is EC minus EO of median windowwise alpha dB power; its 19-channel effect vectors supply MAE and Pearson correlation.

Fingerprints concatenate the four bands' median log powers into 76 features in band-major order. Each query is ranked against 198 same-state real references using Pearson correlation. Ties are pessimistic: every competing correlation at least as high as the correct participant's correlation ranks ahead of it. Both Train- and Validation-reference directions are saved. For example, `Validation_EO_B1` means SSC queries ranked against Validation references. Top-1 and top-5 rates correspond to integer counts out of 198; their chance levels are approximately 0.505% and 2.53%.

Nearest-window distance is minimum RMSE across all 19 × 512 globally Train-standardized coordinates, without realignment, to all Train candidates in the same participant/state. Distances are summarized by the median within each stratum, then averaged equally over states and participants. Full-pool synthetic medians come from original saved characterization tables; Validation and count-matched synthetic query medians were computed by the completed supplementary analysis. No copying threshold or equivalence margin was defined.

Metrics average the two states equally within each participant, then give participants equal weight; alpha effects already combine states within participant. The supplementary uncertainty estimates use 5,000 paired participant bootstrap resamples, seed **2026101703**, and unadjusted percentile 95% intervals. They condition on fixed data, models, pools, and selected windows; training seeds, pools, and subset selections are not resampled. The published summary also retains the first three original Table II metrics' participant dispersion and paired SSC-minus-SC contrasts.

The original execution environment was **Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, and h5py 3.16.0**. This analysis requires neither PyTorch nor MNE nor a GPU. The [archived script](../evidence/validation_reference/original_source/analyze_round3.py), [summary](../evidence/validation_reference/summary.json), and [completion receipt](../evidence/validation_reference/completion.json) are preserved byte-for-byte. A [separate portable derivative](PORTABLE_VALIDATION_ANALYSIS.md) provides configurable input paths and explicit path mapping; it has its own provenance and is not the original executed script. Full external-data execution of that derivative was not performed during this update.

## Final utility experiment

The final experiment is `s2_75B-v2`: Real only; exact Real Train duplication; TSTR for B0/B1/B2; and Real plus each generator at 0.5× and 1×. The half pool retains floor(n/2) within each participant × state stratum, totaling 23,757 synthetic windows. Training totals are 47,835, 71,592, or 95,670 rows.

- **LogBP + LR:** 76 Welch log-bandpower features; training-only StandardScaler; L2 logistic regression with C=1.
- **Riemann + SVM:** OAS covariance; training-fitted Riemannian tangent-space reference; LinearSVC with C=1.
- **EEGNet:** EEGNetv4-style network with F1=8, D=2, F2=16, temporal/separable kernels 64/16, dropout 0.5; Adam at 0.001, batch 128. Maximum epochs 120, minimum stopping epoch 8, patience 6.

EEGNet checkpoints maximize real Validation subject-macro balanced accuracy; ties favor lower Validation cross-entropy, then the earlier checkpoint. TSTR gradient updates use synthetic windows, but EEGNet checkpoint selection uses real Validation. Three classifier seeds are 20262301, 20262302, and 20262303; they reuse the same frozen synthetic pools. Classical conditions have one fit each, totaling 55 fits.

Primary utility is the mean of participant-level balanced accuracy, weighting the 198 participants equally. EEGNet summaries report mean and population SD (`ddof=0`) over classifier seeds. Saved paired 95% percentile bootstrap intervals use 5,000 subject resamples, seed 20262401; matched EEGNet seed differences are averaged within participant before resampling people. Intervals are not multiplicity-adjusted.

Final Test evaluated the already frozen classifiers without retraining or Test-based selection. Its preserved outcomes must not be reused for tuning or silently replaced by a new evaluation.

## Evidence

See the [original evidence inventory](../evidence/provenance/evidence_inventory.json), [Validation publication inventory](../evidence/validation_reference/publication_inventory.json), [frozen model records](../evidence/frozen/scenario2/), [final utility report](../evidence/utility/test/s2_77_v2_complete_final_test_report.json), and [pipeline map](PIPELINE.md). Publication copies retain their original provenance; source paths inside historical records may reference the original research workspace. The new public subset excludes participant rows and frozen selections. Its offline verifier checks aggregate consistency and hashes, not independent signal-feature calculation or bootstrap replication.
