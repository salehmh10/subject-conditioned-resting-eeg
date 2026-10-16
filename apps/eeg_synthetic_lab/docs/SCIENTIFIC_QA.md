# Independent scientific QA

Audit date: 2026-10-10. The audit read saved numeric evidence and method source code without importing or executing experiment modules. It did not train, sample, run classifier inference, rerun Final Test, or choose a checkpoint. Only the separate guarded preview exporter read approved Train and existing synthetic windows, as described in `PREVIEW_PROVENANCE.md`.

## Acceptance anchors checked against actual reports

These are audit results, not constants used to populate the application.

| Quantity | Canonical value | Requested precision | Result |
| --- | ---: | ---: | --- |
| B0 log-PSD MAE | 3.573220617 dB | 3.5732 | Pass |
| B1 log-PSD MAE | 1.156027888 dB | 1.1560 | Pass |
| B2 log-PSD MAE | 0.938874518 dB | 0.9389 | Pass |
| EEGNet Real Only Final Test | 88.417179169% | 88.417% | Pass |
| EEGNet Real+B1 1× Final Test | 88.982930737% | 88.983% | Pass |
| EEGNet Real+B2 1× Final Test | 89.135356702% | 89.135% | Pass |
| B2 1× augmentation minus Real Only | +0.718177533 pp | +0.718 pp | Pass |
| Same paired contrast 95% CI | [+0.249854213, +1.201024986] pp | [+0.250, +1.201] | Pass |
| B2 1× augmentation minus B1 1× augmentation | +0.152425965 pp | +0.152 pp | Pass |
| Same paired contrast 95% CI | [-0.276348825, +0.570037488] pp | [-0.276, +0.570] | Pass |
| EO fingerprint Top-1 B0/B1/B2 | 0.505050505 / 75.252525253 / 71.717171717% | 0.51 / 75.25 / 71.72% | Pass |
| EC fingerprint Top-1 B0/B1/B2 | 1.010101010 / 84.343434343 / 95.959595960% | 1.01 / 84.34 / 95.96% | Pass |

Canonical sources are `outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1/metrics/s2_74e_b0_complete_report_v1.json`, the B1/B2 complete report in `final_pool_fidelity_s2_74b_v1/metrics/`, and `outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es/final_test_s2_77_v1/s2_77_v2_complete_final_test_report.json`.

The entire B1 and B2 summary objects for all 45 full-pool metrics agree exactly between the original and B0-extension reports. No conflict was found for these purported equivalent numeric records.

The Final-Test contrast field is explicitly `delta_right_minus_left`. The code computes each matched seed's participant vector as `right - left`, then averages seeds and resamples subjects. Showing B2-minus-Real or B2-minus-B1 requires negating the estimate and reversing/negating CI endpoints. The source contrast IDs alone do not establish direction. The audit checks the saved contrast numerically against independently keyed condition summaries.

## Result interpretation

The central conclusion is limited and supported: B2 augmentation improves the specified EEGNet Final-Test comparison over Real Only, while the B2-versus-B1 interval includes zero. The two classical families do not share this benefit:

| Final-Test subject-macro BA | Real Only | Real Duplicate | Real+B1 1× | Real+B2 1× |
| --- | ---: | ---: | ---: | ---: |
| EEGNetv4 | 88.417% | 88.682% | 88.983% | 89.135% |
| LogBandpower + Logistic Regression | 84.246% | 84.230% | 84.181% | 83.809% |
| OAS/Riemannian + Linear SVM | 81.878% | 81.868% | 81.706% | 81.016% |

B0's lower similarity to a specific participant's reference does not show that a state-only population generator is invalid. B0's participant labels define evaluation strata and are not a participant conditioning input. Unequal generator training histories prevent a pure causal interpretation of the conditioning comparison.

## Dataset, model, and protocol checks

- The locked data contract records 198 participants, 19 channels, 128 Hz, four-second/512-sample windows, and a two-second stride. Train has 47,835 usable windows; Validation has 9,635. The Final-Test report records 19,131 windows (9,353 EO and 9,778 EC), 792 held-out blocks, 198 participants, 11 conditions, three families, and 55 fits.
- The frozen generation plan records 47,835 samples per generator, matched to Train strata. The frozen subset protocol records 23,757 samples for the nested half-size subset; it is not exactly 47,835 divided by two because stratum counts are rounded independently.
- Every participant has distinct Train/Validation/Test blocks. This supports held-out-block evaluation within the observed participants; it does not establish new-session or unseen-participant generalization.
- The preprocessing YAML specifies 1–40 Hz zero-phase FIR filtering, polyphase resampling to 128 Hz, Train-derived bad-channel and QC decisions, average reference over the full 61 EEG channels after interpolation and before selection of the final 19 channels, and physical HDF5 units of microvolts. No new preprocessing was performed.
- The generator contract is a temporal 1D U-Net, base width 64, multipliers [1,2,4,4], two residual blocks per level, four-head attention at levels 2 and 3, plus bottleneck attention. The B1 participant branch has 198 embeddings of dimension 128, projected into the 256-dimensional conditioning vector. B0 receives state only; B1/B2 receive state and participant. Frozen final generation metadata records v-prediction and ancestral DDPM sampling. B2 is a continued-training/selected-checkpoint member of this same family; the selected continuation recipe has spectral-loss coefficient zero.
- Only the final v2 classifier study is used. The Final-Test report explicitly states `v1_24_epoch_models_used=false`. Classical families have one fit each per condition; EEGNet has three classifier training seeds. Generator identities and classifier training seeds are different concepts.
- EEGNet uses a 120-epoch maximum, earliest stopping epoch 8, patience 6, and highest real Validation subject-macro BA with lower Validation cross-entropy as the tie-breaker. Numerically tied BA uses tolerance 1e-12; a fully tied checkpoint retains the earlier one. The selected epoch may precede epoch 8. Test does not enter checkpoint selection.
- The final report uses population seed SD (`ddof=0`). Subject-bootstrap intervals are conditional on the fixed trained models/pools, descriptive, and unadjusted for multiplicity. A seed SD and a subject-bootstrap interval are different uncertainties. Classical single-fit summaries should not acquire invented seed error bars.

## Metric semantics reviewed

- Official per-participant PSD averages linear Welch power across windows before converting to dB. The saved group PSD is formed from numeric participant summaries; a selected-window PSD is a separate preview. The per-participant/state log-PSD error averages channels and frequency bins from 1 to 40 Hz, then the report equally averages EO/EC within each participant before participant aggregation.
- SD-MD here is the mean absolute difference of Real/synthetic channel standard deviations, with each SD pooled across samples and windows (`ddof=0`). Its units are microvolts. It is not a standardized discrepancy of means. An initial UI glossary wording was corrected after independent review.
- Bandpower bands are delta 1–4, theta 4–8, alpha 8–13, beta 13–30 Hz. Fingerprint profiles concatenate four bandpower values for each of 19 channels: 76 log-bandpower features. They are not alpha-only or connectivity-biometric features. Every saved ranking has 198 candidates. Uniform-random references 1/198 and 5/198 are descriptive ranking references, not fitted statistical nulls.
- Hjorth mobility is defined using finite differences per sample, not per second. Complexity and permutation entropy are dimensionless. Permutation entropy uses order 3 and delay 1. The autocorrelation discrepancy covers lags 1–128.
- Connectivity uses Vinck's debiased squared weighted phase-lag index, already a squared estimator. Negative debiased values must remain negative; undefined denominators remain undefined. Comparison metrics use valid upper-triangle edges and exclude the diagonal. Heatmap diagonals can be masked for display. These are sensor-space phase-lag patterns, not anatomical or causal connectivity.
- RMS pairwise diversity is a ratio with parity at one; larger is not always better. Exact-copy fractions and nearest-real distances are limited diagnostics. Zero exact-copy detections do not prove absence of memorization or privacy protection.
- Matched-count Real–Real calibration uses disjoint Train blocks. Both sets belong to the generator's training domain, and their matched-count comparisons must remain separate from the full-pool analysis.

## Primary references verified

The following primary paper pages were checked through web search on 2026-10-10. They provide context for the methods, not external replacement values for this study:

- [Babayan et al., the MPI-LEMON dataset (Scientific Data, 2019)](https://pmc.ncbi.nlm.nih.gov/articles/6371893/), DOI 10.1038/sdata.2018.308.
- [Lawhern et al., EEGNet (2018)](https://pubmed.ncbi.nlm.nih.gov/29932424/), DOI 10.1088/1741-2552/aace8c.
- [Vinck et al., debiased weighted phase-lag index (NeuroImage, 2011)](https://www.sciencedirect.com/science/article/pii/S1053811911000917), DOI 10.1016/j.neuroimage.2011.01.055.

Reference links are optional reading and do not need to load for offline presentation. No study data was uploaded during the reference check.

## Tests and boundaries

The dedicated app environment executed 35 preview/scientific tests successfully in 3.29 seconds after the final bundle became available. These include all stated numerical anchors, B1/B2 report equality, saved-versus-direct contrast direction, CI reversal, population seed SD, and actual prepared preview shape/hash verification, as well as access-guard and deterministic-selection tests. Two complete bounded exports generated identical NPZ bytes and checked consumed source hashes before and after.

An independent review of the implemented chart calculations verified fingerprint percentage conversion and the 100 × k/198 uniform reference, selected-participant alpha EC-minus-EO effects against the saved source arrays, channel-mean linear PSD aggregation, shared connectivity scales, preservation of negative and undefined edges, and Synthetic-minus-Real differences with symmetric scales. Three executable tests cover these implemented transformations using the actual prepared bundle. The relative Frobenius covariance error is dimensionless; a discovered unit-label issue was corrected in the adapter before the final check. Preview captions were changed from statistical independence language to separate precomputed examples because the pools reuse generation seeds and RNG streams across generators.

The standalone HTML implementation was reviewed for fraction-to-percentage-point conversion, directed saved confidence intervals, complete classifier comparison, and limited-preview inclusion. It includes only one deterministic participant/state/channel record across four sources, with no participant identifier, rather than the full preview pool. Root integration testing separately verifies its rendering and offline behavior.

This document records scientific evidence and preview tests. Full application browser tests, export checks, screenshots, runtime timings, and packaging checks are recorded separately by the integration workstream. No browser pass is inferred from these numerical checks.

## Read-only scientific source signatures

These SHA256 values identify the report, metadata, and method files inspected for this audit. All paths are repository relative. Consumed preview waveform-file digests and before/after preservation checks are recorded separately in `previews.json`; the main data-adapter source manifest records the full prepared-bundle source set. No source file was modified.

| Source | SHA256 |
| --- | --- |
| `outputs/metrics/scenario2/final_block_split_manifest.csv` | `a58605010f7371fe1805a7834c205f2da7be816515779e15f2ba082bfd201f8b` |
| `outputs/metrics/scenario2/train_normalization.json` | `82af5c196145756f09658ba5fd887a9953962349c23af922eca20df6b86d7379` |
| `outputs/metrics/scenario2/train_normalization_lock.json` | `9d69dab30fa8c55a2e34c98090671649d425634f10c03f1d286e49db35aa2ee1` |
| `outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1/metrics/s2_74e_b0_complete_report_v1.json` | `e09514b6c776d8c4106b6c9f54ea072abc89b62b7938827cc87ba282ce2ab735` |
| `outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1/metrics/s2_74b_complete_report_v1.json` | `96526a457f51dca014f1c4733acfe94fa006d9d5cc95e60f9e6dd097f368518c` |
| `outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es/final_test_s2_77_v1/s2_77_v2_complete_final_test_report.json` | `24f565c60880f6ab889c08a528b3f34edfe1ddf455842283a7f4f6f66ac6face` |
| `outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1/protocol/s2_73_generation_plan_v1.json` | `483f59eed67f6303048fe131338c46f3438c60d47ec08fc698db7f78a55c4f12` |
| `outputs/frozen/scenario2/b2/b2_validation_selected_replication_e1_v1/metrics/b2_frozen_selection_v1.json` | `41a2eb11694463c7a0bd360f84e378150a9bb02775ea67371f95732aadab6df4` |
| `configs/preprocessing/frozen_preprocessing_v1.yaml` | `da57489746e66fa819da16217e585cdd0ca3b4e5cbe4491603e831be98889110` |
| `scripts/data/s2_23_check_real_train_dataset.py` | `e880eebee4dfeb79f41da809bbb4f8d6c7d233df68d972702f093f8aef4cde34` |
| `scripts/data/s2_31_freeze_b0_contract.py` | `26e47453ab45c71ac8af1da1e3426f46d295fb9f40bee4923dd1adc90a63937e` |
| `scripts/data/s2_32_b0_unet_shape_test.py` | `ff45bfe50ade25759c9ae681364cf351e94fd61aa995fda4ba799537b52cb242` |
| `scripts/data/s2_47_b1_architecture_warmstart_sanity.py` | `c3f70738635f7207f9a9566dee0032b222ef5fe8af5a6d46fd0bb3c71a19ed97` |
| `scripts/data/s2_65_early_epoch_training_seed_checkpoint_landscape.py` | `422225565528909b8be03b8e4b03991950d9e3a95958b9e1457a132f1f63a5fb` |
| `scripts/data/s2_67_fresh_seed_validation_checkpoint_selection.py` | `4efcb0a446b5d2b00e0ee01676848f574c0af3ccb9d2c8dc0b23a2d62a93effb` |
| `scripts/data/s2_70_final_sampling_robust_fresh_seed_selector.py` | `d5a9ca09970b82a974daf6c3115940ac103a7a0b30cf9615617be88ab0a71fdc` |
| `scripts/data/s2_74b_final_pool_fidelity.py` | `09dcaa49c74f7ab055f0932054f0cc97ec346a71558dcdc8b036c6d52536e8ed` |
| `scripts/data/s2_75b_v2_utility_120ep_earlystop.py` | `4fac6df5363b465a60c98788f4aa968ec99a2339589455925c00f27753146043` |
| `scripts/data/s2_77_v2_single_final_test_evaluation.py` | `5b0a706b90ba576a6e7748f870059d555e4042cc4566939b2228445608c606e6` |
