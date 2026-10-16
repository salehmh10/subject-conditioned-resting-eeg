# Methods of the completed study

This document describes the final selected models and final utility experiment. Historical protocols remain available as evidence of earlier stages; a candidate's name or an early contract does not override the final frozen selection.

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
| B0 | Eye state | Raw epoch 20 of v fine-tuning | 8,402,771 |
| B1 | Eye state and participant | Raw epoch 10 of B0-initialized fine-tuning | 8,526,931 |
| B2 | Eye state and participant | Epoch 1 of the selected continuation run | 8,526,931 |

The B1/B2 subject branch contains a 198 × 128 embedding table and a 128→256→256 projection, added to the state-conditioning representation. B2 uses the same architecture as B1 and **zero spectral-loss weight**. Earlier spectral candidates are part of development history.

B0 inherits epsilon pretraining with Validation-based selection before v fine-tuning. B1 inherits B0; Validation loss was monitored during training. B2 was selected after inspecting development Validation, including both cohorts labeled VAL_DEV and VAL_CONF. Their different histories prevent a controlled causal attribution to the subject condition alone. The frozen B2 record explicitly does not establish replication of the continuation recipe across training seeds.

Each final synthetic pool contains 47,835 windows matched to Real Train participant × state counts. B0's participant labels identify comparison strata; no subject identity is supplied to B0's network.

## Fidelity

Saved characterization covers spectral, amplitude, temporal, spatial, diversity, and copy-related descriptors. Full-pool fidelity is relative to Real Train, aggregated equally over participants and states where applicable. Alpha effects compare EC minus EO within participant. Separate Real–Real calibration uses disjoint Train-block halves and matched nonoverlapping window counts; both halves were available during generator training.

Connectivity uses debiased squared wPLI, estimated per FFT bin and averaged within bands. Finite negative estimates are retained. These are sensor-space descriptors. Fingerprints use 76 median log-bandpower features and rank a synthetic participant against 198 Real Train references.

## Final utility experiment

The final experiment is `s2_75B-v2`: Real only; exact Real Train duplication; TSTR for B0/B1/B2; and Real plus each generator at 0.5× and 1×. The half pool retains floor(n/2) within each participant × state stratum, totaling 23,757 synthetic windows. Training totals are 47,835, 71,592, or 95,670 rows.

- **LogBP + LR:** 76 Welch log-bandpower features; training-only StandardScaler; L2 logistic regression with C=1.
- **Riemann + SVM:** OAS covariance; training-fitted Riemannian tangent-space reference; LinearSVC with C=1.
- **EEGNet:** EEGNetv4-style network with F1=8, D=2, F2=16, temporal/separable kernels 64/16, dropout 0.5; Adam at 0.001, batch 128. Maximum epochs 120, minimum stopping epoch 8, patience 6.

EEGNet checkpoints maximize real Validation subject-macro balanced accuracy; ties favor lower Validation cross-entropy, then the earlier checkpoint. TSTR gradient updates use synthetic windows, but EEGNet checkpoint selection uses real Validation. Three classifier seeds are 20262301, 20262302, and 20262303; they reuse the same frozen synthetic pools. Classical conditions have one fit each, totaling 55 fits.

Primary utility is the mean of participant-level balanced accuracy, weighting the 198 participants equally. EEGNet summaries report mean and population SD (`ddof=0`) over classifier seeds. Saved paired 95% percentile bootstrap intervals use 5,000 subject resamples, seed 20262401; matched EEGNet seed differences are averaged within participant before resampling people. Intervals are not multiplicity-adjusted.

Final Test evaluated the already frozen classifiers without retraining or Test-based selection. Its preserved outcomes must not be reused for tuning or silently replaced by a new evaluation.

## Evidence

See the [evidence inventory](../evidence/provenance/evidence_inventory.json), [frozen model records](../evidence/frozen/scenario2/), [final utility report](../evidence/utility/test/s2_77_v2_complete_final_test_report.json), and [pipeline map](PIPELINE.md). Publication copies retain their original provenance; source paths inside historical records may reference the original research workspace.
