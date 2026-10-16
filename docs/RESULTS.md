# Results of the completed study

These values come from saved final results. Repository publication did not rerun models, generate predictions, or compute new confidence intervals.

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

Real duplication improves EEGNet by a descriptive 0.265 points over real-only. B2 full augmentation exceeds duplication by a descriptive 0.453 points. No paired duplicate contrast was supplied by the frozen comparison table; neither difference is assigned an invented CI or significance claim.

## Fidelity

The reference is **Real Train**, with participant-level aggregation. These are descriptive generation results, separate from classifier performance on Test.

| Representative metric | B0 | B1 | B2 |
|---|---:|---:|---:|
| Log-PSD MAE, dB | 3.573 | 1.156 | 0.939 |
| SD-MD, µV | 2.519 | 0.822 | 0.785 |
| Alpha EC−EO channel MAE, dB | 2.565 | 1.221 | 1.215 |
| Alpha EC−EO topography correlation | 0.636 | 0.828 | 0.830 |
| Alpha dwPLI² edge correlation | 0.105 | 0.334 | 0.369 |

B1/B2 improve many subject-wise descriptors relative to B0. B2 is not uniformly better than B1: alpha-bandpower MAE and alpha node-mean error favor B1, and the saved B2−B1 alpha-effect MAE interval includes zero. B2 fingerprint Top-1 identification is higher in EC (95.96% versus 84.34%) but lower in EO (71.72% versus 75.25%). Both states must be retained when describing this finding.

No exact Real Train copies were detected by the specified same-participant/state bitwise checks. That result and the fingerprint scores do not demonstrate privacy or absence of memorization. Sensor-space phase-connectivity matching remains modest despite strong ordinary channel-correlation similarity.

Sources: [representative fidelity table](../results/tables/fidelity_representative.csv), [B0 extension report](../evidence/fidelity/b0_extension/s2_74e_b0_complete_report_v1.json), and [original B1/B2 report](../evidence/fidelity/b1_b2/s2_74b_complete_report_v1.json).

## Development results and learning histories

Validation remains development evidence. EEGNet real-only / B1 1× / B2 1× Validation means are 87.526% / 88.615% / 88.727%. A Test–Validation difference is not itself an augmentation effect.

The saved 33 EEGNet runs all stopped by early stopping. Selected epochs span 12–57; stopping epochs span 18–63. The preserved [learning curves](../results/tables/utility_learning_curves.csv), [stopping table](../results/tables/utility_stopping.csv), and [55 Validation fits](../results/tables/utility_validation_all_55_fits.csv) retain this development history.
