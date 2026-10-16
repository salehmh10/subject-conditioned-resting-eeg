# Scientific source mapping

Canonical extracted evidence is available. No handoff archive extraction is needed. All source artifacts remain read-only; preparation imports no experiment module.

| Presentation evidence | Canonical source and transformation |
| --- | --- |
| Complete 45 × 3 Fidelity summaries | `outputs/runs/scenario2/characterization/b0_fidelity_extension_s2_74e_v1/metrics/s2_74e_b0_complete_report_v1.json`; saved summaries and CIs copied; embedded B1/B2 values checked against the original baseline complete report |
| Real/B1/B2 participant numeric arrays | `outputs/runs/scenario2/characterization/final_pool_fidelity_s2_74b_v1/subjects/*.json`; explicit subject ID joins |
| B0 participant numeric arrays | B0 extension `subjects/*.json`; require the referenced baseline participant SHA to match the original source |
| PSD, alpha, connectivity aggregation | Read-only `scripts/data/s2_74b_final_pool_fidelity.py` and frozen `protocol/s2_74b_analysis_plan_v1.json`; equal-participant, equal-channel linear power mean before dB; within-participant alpha EC-minus-EO; finite per-edge equal-participant dwPLI² mean |
| B0 fingerprint diagnostics | B0 extension `tables/subject_B0_fingerprint.csv` |
| B1/B2 fingerprint diagnostics | Baseline `tables/subject_fingerprint.csv`; do not assume a B0-only CSV contains the other generators |
| Real–Real calibration | Complete extension report `matched_train_block_calibration_summary`; kept separate from full-pool records |
| Validation Utility | Only `outputs/runs/scenario2/utility/eleven_condition_s2_75_v2_120epoch_es/final_validation/`: complete report and `all_55_classifier_validation_runs.csv` |
| Final-Test Utility | Same v2 root, `final_test_s2_77_v1/`: complete report, `all_55_final_test_fits.csv`, and `per_subject_55_fits.csv` |
| Training curves and Validation participant diagnostics | Same v2 root, `runs/<family>/<condition>/*_receipt.json`; no checkpoint deserialization; actual `method.training_history` retained without extending stopping epochs |
| Utility condition summaries | Saved complete reports; preserve population SD, not sample SD; classical families have one seed |
| Saved contrast direction | Validation is left-minus-right. Final Test is right-minus-left, verified in `scripts/data/s2_77_v2_single_final_test_evaluation.py`, where `diff.append(r-l)` supplies the saved report. Display transform is point negation and endpoint reversal/negation |
| Shape/count metadata and participant index mapping | `outputs/runs/scenario2/utility/bulk_synthetic_b1_b2_1x_v1/protocol/s2_73_generation_plan_v1.json`, selected data-contract fields only; no absolute filesystem paths copied |
| Normalization metadata | `outputs/metrics/scenario2/train_normalization.json`, locked Train-only fit metadata and channel order; no signal arrays read |
| Matched half-subset count | Saved EEGNet 0.5x-condition `train_rows` minus verified Real Train count; all corresponding histories must agree |

Every consumed source appears in the generated source manifest with SHA256. Dataset/protocol metadata outside the v2 utility root is used only for counts, channel/subject identity and methods; it contributes no earlier utility result. Old 24-epoch fits are never scanned or displayed.

The adapter detects duplicate JSON keys, duplicate keyed rows, missing columns/identities, mismatched subject IDs, conflicting baseline report values, changed source hashes, and incomplete utility condition/seed coverage. Baseline-versus-extension agreement is checked recursively for every reused Fidelity metric, fingerprint, alpha effect, connectivity, state effect and matched calibration summary; no conflicts were found. Missing core canonical evidence stops preparation with an actionable English message rather than substituting synthetic values. Optional preview absence disables only waveforms.

The safe ZIP helper, tested with both separators and traversal/absolute/duplicate members, reads JSON directly without extraction. It is unused on this repository because complete canonical extracted evidence exists.

Preparation on 2026-10-10 produced a 53.38 MiB bundle, verified 468 consumed source signatures unchanged, and passed all 18 independent reporting-precision anchors. A repeated signature-cache preparation took 1.372 seconds and left bundle SHA256 `57bbe5c3c6ffb80e2b89fb4c8af83334616a7f240b08ad9efda464aa316521a4` unchanged. The full preparation after unit review took 13.60 seconds. These are observed timings on this machine, not performance guarantees.

The aggregate distribution adapter was independently checked: participant arrays/records and waveform pools are omitted; explicit capability messages describe that reduced scope; group scientific values remain numerical; no personal absolute filesystem paths occur in the serialized evidence. Unused per-record participant provenance transformations are pruned while aggregate source lineage remains.
