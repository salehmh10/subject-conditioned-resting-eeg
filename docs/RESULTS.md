# Results of the completed study

These values come from saved final results. Repository publication did not rerun models, generate predictions, or compute new confidence intervals.

The paper names SC / SSC / SSC-C correspond to B0 / B1 / B2 in historical outputs. Original Train-reference fidelity and final utility outcomes remain unchanged; the additional Validation-reference results below are the completed exploratory analysis used in the final manuscript.

## Final Test utility

Subject-macro balanced accuracy, percent. EEGNet shows mean ± population SD over three classifier seeds. Each classical cell represents one fit. Test comprises 19,131 eligible real windows from 198 participants in held-out complete blocks.

| Training condition | LogBP + LR | Riemann + SVM | EEGNet |
|---|---:|---:|---:|
| Real only | 84.246 | 81.878 | 88.417 ± 0.242 |
| Real duplicate | 84.230 | 81.868 | 88.682 ± 0.286 |
| TSTR B0 | 82.200 | 78.639 | 85.257 ± 0.504 |
| TSTR B1 | 83.373 | 80.864 | 86.336 ± 0.358 |
| TSTR B2 | 82.987 | 78.867 | 87.253 ± 0.193 |
| Real + B0, 0.5× | 84.114 | 81.818 | 88.665 ± 0.409 |
| Real + B0, 1× | 83.978 | 81.504 | 87.619 ± 0.536 |
| Real + B1, 0.5× | 84.329 | 81.900 | 88.190 ± 0.267 |
| Real + B1, 1× | 84.181 | 81.706 | 88.983 ± 0.468 |
| Real + B2, 0.5× | 83.948 | 81.495 | 88.444 ± 0.657 |
| Real + B2, 1× | 83.809 | 81.016 | 89.135 ± 0.280 |

Sources: [all 55 Test fits](../results/tables/utility_test_all_55_fits.csv), [original condition summaries](../evidence/utility/test/tables/all_11_condition_summary.csv), and [saved Test report](../evidence/utility/test/s2_77_v2_complete_final_test_report.json).

## Preserved paired comparisons

Differences below are the first named condition minus the second, in **percentage points**. Intervals are saved paired percentile 95% subject-bootstrap CIs, unadjusted for multiple comparisons.

| Comparison | Difference | 95% CI |
|---|---:|---:|
| EEGNet: Real+B2 1× − Real only | +0.718 | [+0.250, +1.201] |
| EEGNet: Real+B2 1× − Real+B1 1× | +0.152 | [−0.276, +0.570] |
| EEGNet: Real+B0 0.5× − Real only | +0.248 | [−0.193, +0.698] |
| EEGNet: Real+B0 1× − Real only | −0.798 | [−1.358, −0.224] |
| EEGNet: TSTR B2 − TSTR B1 | +0.917 | [+0.076, +1.731] |
| LogBP+LR: Real+B2 1× − Real only | −0.437 | [−0.808, −0.082] |
| Riemann+SVM: Real+B2 1× − Real only | −0.862 | [−1.403, −0.333] |

The original [frozen contrast table](../evidence/utility/test/tables/frozen_paired_subject_contrasts.csv) stores `delta_right_minus_left`. The display table reverses the relevant comparisons by transforming `(d, low, high)` to `(-d, -high, -low)` before conversion to percentage points. Validation uses a different stored sign convention; column names must be read before combining the tables.

B2 full-dose augmentation has the highest EEGNet mean, but its interval versus B1 includes zero. The classical classifiers show reduced performance with full-dose B2 augmentation. Every TSTR result remains below its real-only baseline. Improved signal fidelity therefore does not imply improved utility for every classifier.

Real duplication improves EEGNet by a descriptive 0.265 points over real-only. The completed manuscript reports a **post hoc** B2 full-augmentation-minus-duplication difference of 0.453 points, with interval **[−0.071, +0.996]**; superiority over duplication is not established. This contrast is absent from the original frozen comparison table. The [reported-results extract](../evidence/validation_reference/reported_results.json) labels its manuscript provenance separately: the difference in means can be checked against public utility summaries, but this interval cannot be independently reconstructed or verified from the aggregate subset. No new paired analysis was run for this update.

## Original Train-reference fidelity (Table II)

The reference is **Real Train**, with participant-level aggregation. These are descriptive generation results, separate from classifier performance on Test.

| Representative metric | B0 | B1 | B2 |
|---|---:|---:|---:|
| Log-PSD MAE, dB | 3.573 | 1.156 | 0.939 |
| SD-MD, µV | 2.519 | 0.822 | 0.785 |
| Alpha EC−EO channel MAE, dB | 2.565 | 1.221 | 1.215 |
| Alpha EC−EO topography correlation | 0.636 | 0.828 | 0.830 |
| Autocorrelation MAE | 0.0534 | 0.0336 | 0.0323 |
| RMS diversity ratio | 1.283 | 1.028 | 1.003 |
| Channel-correlation pattern correlation | 0.948 | 0.985 | 0.984 |
| Alpha dwPLI² edge correlation | 0.105 | 0.334 | 0.369 |
| Alpha dwPLI² edge MAE | 0.0607 | 0.0412 | 0.0405 |
| Alpha dwPLI² node MAE | 0.0422 | 0.0274 | 0.0303 |

B1/B2 improve many subject-wise descriptors relative to B0. B2 is not uniformly better than B1: alpha-bandpower MAE and alpha node-mean error favor B1, and the saved B2−B1 alpha-effect MAE interval includes zero. B2 fingerprint Top-1 identification is higher in EC (95.96% versus 84.34%) but lower in EO (71.72% versus 75.25%). Both states must be retained when describing this finding.

No exact Real Train copies were detected by the specified same-participant/state bitwise checks. That result and the fingerprint scores do not demonstrate privacy or absence of memorization. Sensor-space phase-connectivity matching remains modest despite strong ordinary channel-correlation similarity.

Sources: [representative fidelity table](../results/tables/fidelity_representative.csv), [B0 extension report](../evidence/fidelity/b0_extension/s2_74e_b0_complete_report_v1.json), and [original B1/B2 report](../evidence/fidelity/b1_b2/s2_74b_complete_report_v1.json).

The supplementary analysis adds dispersion for the first three original Table II metrics while preserving their original means. Entries below are participant median [Q1, Q3]; paired differences are SSC minus SC means with saved, unadjusted 95% intervals.

| Metric | SC | SSC | SSC-C | Paired SSC − SC |
|---|---:|---:|---:|---:|
| Log-PSD MAE, dB | 3.465 [2.690, 4.298] | 1.103 [0.927, 1.349] | 0.915 [0.790, 1.028] | −2.417 [−2.572, −2.272] |
| Channel-SD MAE, µV | 2.330 [1.913, 2.958] | 0.727 [0.555, 0.950] | 0.718 [0.508, 0.922] | −1.697 [−1.828, −1.572] |
| Alpha-effect MAE, dB | 2.273 [1.357, 3.538] | 0.990 [0.686, 1.377] | 0.990 [0.714, 1.439] | −1.345 [−1.568, −1.132] |

The archived analysis records nine successful participant-level reproductions of these original Train-reference metrics and 12 successful Train-reference top-1/top-5 rate comparisons. Those are preserved assertions from the completed analysis, not participant-level checks repeated by the public aggregate verifier.

## Exploratory Validation-reference results (Table III)

Validation blocks were held out from generator gradient training but used in development and model selection. In particular, Validation spectral fidelity helped select SSC-C. These post hoc results are selection-dependent and do not provide independent confirmation or evidence of generalization to new participants. All 198 study participants contribute.

The following rows reproduce the completed manuscript's Table III from the [original aggregate summary](../evidence/validation_reference/summary.json). Errors are equally weighted participant means. Identification ranks each query source against 198 same-state real Validation profiles. All available windows are used except in the final row, which uses the single frozen count-matched subset.

| Measure | Real Train | SC | SSC | SSC-C |
|---|---:|---:|---:|---:|
| Log-PSD MAE, dB | 1.112 | 3.766 | 1.462 | 1.349 |
| Alpha-effect MAE, dB | 1.562 | 2.765 | 1.799 | 1.822 |
| Top-1 EO, % | 94.44 | 1.01 | 59.60 | 62.12 |
| Top-1 EC, % | 90.40 | 0.51 | 64.14 | 74.75 |
| Top-5 EO, % | 96.46 | 3.03 | 77.27 | 71.21 |
| Top-5 EC, % | 90.91 | 3.03 | 76.26 | 83.33 |
| Count-matched log-PSD MAE, dB | 1.214 | 3.771 | 1.541 | 1.435 |

SSC-C minus SSC spectral MAE is −0.113 dB [−0.141, −0.083] for full pools and −0.106 dB [−0.134, −0.077] after count matching. Count-matched SSC-C error remains 0.222 dB [0.193, 0.252] above the real Train-to-Validation reference. Their full-pool alpha-effect MAE difference is 0.024 dB [−0.065, 0.110]. The summary also retains channel-SD errors and alpha-pattern correlations omitted from the compact paper table.

Subject-conditioned identification remains below real Train-to-Validation retrieval. Full-pool SSC-C increases top-1 in both states relative to SSC, but EO top-5 decreases from 77.27% to 71.21%. After count matching, EO top-1 is 57.07% for SSC and 56.57% for SSC-C (a difference of one participant), with top-5 of 74.75% and 66.67%; EC top-1 is 61.62% and 73.23%. Continuation therefore does not establish uniform improvement in identity preservation.

## Nearest-Train-window comparison

RMSE uses all 19 × 512 globally Train-standardized coordinates. Every query is compared with all Train candidates of the same participant/state. The median query distance is taken per stratum, then the two states and 198 participants receive equal weight. Full-pool synthetic medians are retained from original saved characterization tables.

| Query source | Mean stratum-median RMSE | Difference from Validation [95% CI] |
|---|---:|---:|
| Real Validation | 1.070822 | Reference |
| SC, full pool | 1.069678 | −0.001144 [−0.032019, 0.029367] |
| SSC, full pool | 1.072907 | 0.002085 [−0.009984, 0.014256] |
| SSC-C, full pool | 1.053461 | −0.017361 [−0.027767, −0.006891] |
| SC, matched queries | 1.083635 | 0.012813 [−0.018373, 0.042892] |
| SSC, matched queries | 1.073721 | 0.002899 [−0.009191, 0.015281] |
| SSC-C, matched queries | 1.053585 | −0.017237 [−0.028204, −0.006747] |

SSC-C is modestly closer to Train than the real Validation reference. These distances cannot separate distributional similarity, between-block variation, and approximate memorization; they provide no calibrated copying threshold or privacy guarantee. Count matching uses 15–25 windows per participant/state, totaling 9,635 per source. It matches window counts, not temporal dependence or the number of recording blocks.

The supplementary intervals use 5,000 paired participant resamples, seed 2026101703, and are conditional on fixed models, pools, and subsets, without multiplicity adjustment. The [public metadata](../evidence/validation_reference/public_metadata.json) and [reported-results extract](../evidence/validation_reference/reported_results.json) identify the preserved evidence and table mappings. Offline verification checks packaged hashes, aggregate values, interval ordering, and count-to-percentage arithmetic; it cannot independently recompute signal features or paired bootstrap intervals. Participant rows, window indices, and waveform records are not included.

## Development results and learning histories

Validation remains development evidence. EEGNet real-only / B1 1× / B2 1× Validation means are 87.526% / 88.615% / 88.727%. A Test–Validation difference is not itself an augmentation effect.

The saved 33 EEGNet runs all stopped by early stopping. Selected epochs span 12–57; stopping epochs span 18–63. The preserved [learning curves](../results/tables/utility_learning_curves.csv), [stopping table](../results/tables/utility_stopping.csv), and [55 Validation fits](../results/tables/utility_validation_all_55_fits.csv) retain this development history.
