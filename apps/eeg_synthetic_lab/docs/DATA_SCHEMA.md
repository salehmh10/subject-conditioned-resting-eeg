# Presentation bundle schema 1

Runtime reads only `data/presentation_bundle/bundle.json` and optional app-owned preview files. The preparation modules read saved JSON/CSV and method source text, never import study modules, model checkpoints, or raw waveform arrays. All paths in the source manifest are repository relative. JSON is deterministic, UTF-8, finite, and contains null for undefined values.

## Core records

| Key | Shape / important fields |
| --- | --- |
| `schema_version` | Integer `1` |
| `metadata` | `dataset`, `n_subjects`, `channels`, `fs_hz`, `window_seconds`, `window_samples`, `n_train`, `n_validation`, `n_test`, `n_synthetic_per_generator`, `n_half_subset`, `condition_order`, `families`, `utility_policy`, `definitions`, `limitations` |
| `availability` | Explicit boolean capabilities and English reasons; external previews are separate |
| `fidelity_metrics` | 135 records, 45 metrics × B0/B1/B2: `metric`, `model`, `group`, `mean`, `median`, `ci_low`, `ci_high`, `n_subjects`, `unit`, `direction`, `provenance` |
| `calibration` | Same record shape, saved matched-count disjoint Train-block calibration; includes `Real_B` as reference. Never merge into full-pool scores |
| `fidelity_subjects` | 1,188 records: `subject_id`, `state`, `model`, `metrics` dictionary, `provenance` |
| `fingerprint` | Six model/state records: `top1`, `top5` fractions, `n_candidates`, `n_valid_queries`, saved `margin` summary, `provenance` |
| `fingerprint_subjects` | `subject_id`, `state`, `model`, `rank`, `margin`, `correct_subject_r`, `rmse_db`, `provenance` |
| `utility_runs` | 110 saved fits: `split`, `family`, `condition`, `seed`, `subject_macro_balanced_accuracy`, `subject_macro_roc_auc`, `global_balanced_accuracy`, `global_roc_auc`; saved global `confusion_matrix` and definition; Validation additionally includes best/stopping epoch |
| `utility_summary` | 66 records (33 per split): `split`, `family`, `condition`, `mean`, `population_sd`, `n_seeds`, `ci_low`, `ci_high`, `auc`, definitions and provenance |
| `contrasts` | 54 saved contrasts: `split`, `family`, `id`, `left`, `right`, `delta`, `ci_low`, `ci_high`, `direction`, `source_direction`, `n_subjects`, `provenance` |
| `histories` | 33 records: `condition`, `seed`, `best_epoch`, `stopped_epoch`, `stop_reason`, `train_rows`, `epochs`, `provenance`; epoch has `epoch`, `train_loss`, `validation_loss`, `validation_ba`, `optimizer_updates` |
| `utility_subjects` | Saved keyed per-participant metrics for all fits/splits; explicit source-index-to-ID mapping |
| `provenance` | `sources` dictionary keyed by repository-relative path; each entry has `path`, `sha256`, `bytes`, `role`. `integrity` records post-read unchanged hash verification |

All Utility BA, AUC, SD and intervals are fractions. Multiply by 100 for percent or percentage-point differences. Saved SD is population SD (`ddof=0`). Classical classifiers have one fit, no between-seed error bar. A saved participant-bootstrap interval is a different uncertainty measure. `split` is `validation` or `final_test`. `direction` for normalized contrasts is always `left_minus_right`; the Final-Test source is `right_minus_left`, so preparation negates the point and reverses/negates both interval endpoints.

Fidelity `group` is one of Spectrum and Alpha, Time and Complexity, Spatial Statistics, Distribution and Diversity. `direction` is `lower`, `higher`, `parity`, or `descriptive`. Nearest-real diagnostics are descriptive rather than a quality or privacy score.

Every derived record/array section references source IDs in the manifest and describes its transformation, units, aggregation level and analysis split. Repeated record provenance is losslessly interned: resolve `record.provenance.transform_id` in `bundle.provenance.transforms`, then resolve its `source_ids` in `bundle.provenance.sources` for paths and hashes. Array-section provenance is inline. No scientific numbers are rounded to reduce bundle size. No source is inferred by table row order.

## Numeric array sections

`psd.frequencies_hz` is the saved Welch grid (129 bins, 0–64 Hz). `psd.group[state][model]` is a 129-element linear-power array, in uV²/Hz. `psd.participants[subject_id][state][model]` contains equal-channel averages of the saved participant PSD. Groups average linear power equally over participants/channels before any display conversion `10 log10(power)`. Show the original comparison range 1–40 Hz. The compact bundle does not support per-channel official PSD selection.

`connectivity.channels` preserves the original 19-channel order. `connectivity.group[state][band][model]` is a 19×19 matrix. `connectivity.participants[subject_id][state][band][model]` contains genuine saved participant matrices. Models are `Real`, `B0`, `B1`, `B2`; states are EO/EC; bands theta/alpha/beta. Unit is dwPLI². Diagonals are null, undefined edges remain null, and negative estimates are retained. No estimator is squared again. `valid_frequency_bins` has the corresponding participant/state/band/model numeric count matrices. Group aggregation is a finite equal-participant mean per edge. `sources_by_subject` provides exact file lineage.

`connectivity_metrics` holds saved participant/state/band/model diagnostics: `upper_r`, `upper_mae`, `relative_frobenius`, `node_mean_mae`, `mean_connectivity_real`, `mean_connectivity_synth`, `negative_edge_fraction_real`, `negative_edge_fraction_synth`, `valid_edges`. These diagnostic means must not be replaced by differences calculated from the group-mean matrix. Group-mean difference maps are descriptive visual transforms.

`alpha_effect.group[model]` is a 19-channel vector. `alpha_effect.participants[subject_id][model]` contains the within-participant EC-minus-EO difference of saved median-window alpha bandpower in dB. Group means are equal-participant averages. `fidelity_group_alpha` and `fidelity_group_connectivity` retain original saved metric summaries/intervals.

## Optional waveforms and shared exports

`previews.json` and `previews.npz` are created by the separate bounded Train-only exporter. Main preparation never deletes or overwrites these files. They are illustrative samples, not paired reconstructions or official group PSD results.

Standalone HTML should include aggregate records and group arrays only. Strip `participants`, participant records, valid-frequency participant matrices and source-by-subject mappings for aggregate shared snapshots. Deidentified per-participant summary arrays in the local bundle remain numeric evidence, not raw EEG windows. No export needs model weights or full datasets.

## Integrity and restartability

`source_manifest.json` accompanies the bundle and includes its SHA256, size, consumed source signatures and validation results. Preparation validates complete 55-fit/33-summary coverage in each split, exact keyed identity, saved means and population SD, contrast direction, actual stopping epochs, all 45 Fidelity metrics, and user-supplied numerical anchors as independent checks. Sources are hashed before consumption and after preparation. A matching code signature plus unchanged consumed source hashes reuses the bundle. `--force` rebuilds it deterministically.
