# Pipeline and artifact map

This is a map of the completed research workflow. Numbered scripts include development branches, diagnostics, and superseded candidates; they are not a list to run blindly in numeric order. Many stages expect original external artifacts and exact provenance bindings.

| Stage | Representative source | Preserved outcome |
|---|---|---|
| Metadata and block audit | `scripts/audit/s2_01_audit_metadata.py` through `s2_05_plan_splits_and_downloads.py` | Structural cohort and block assignments |
| Download, preprocessing, cohort lock | Early `scripts/data/s2_*` stages and `src/resting_eeg/data/` | Frozen eligible blocks, masks, cohort and split |
| Train normalization | `s2_18_fit_train_normalization.py`, `s2_19_verify_train_normalization.py`, `s2_20_accept_train_normalization.py` | Train-only global channel scaler |
| Real-data development baselines | `s2_21`–`s2_30` | Development-era classifier and feature workflows |
| B0 construction and selection | `s2_31`–`s2_45` | Final raw epoch-20 v-prediction B0 |
| B1 conditioning and selection | `s2_46`–`s2_51` | Final raw epoch-10 subject/state B1 |
| B2 development and selection | `s2_52`–`s2_71` | Selected epoch-1 continuation, zero spectral weight |
| Utility protocol and pools | `s2_72`, `s2_73`, `s2_74`, `s2_74c`, `s2_74d` | Frozen full pools and nested half subsets |
| Final-pool characterization | `s2_74b_final_pool_fidelity.py`, `s2_74e_b0_fidelity_extension.py` | Train-referenced B0/B1/B2 fidelity |
| Final utility training | `s2_75a_v2_freeze_120ep_earlystop.py`, `s2_75b_v2_utility_120ep_earlystop.py` | 55 frozen classifier fits and Validation results |
| Saved report finalization | `s2_75c_finalize_v2_report_csv_fix.py` | Final v2 Validation tables/report |
| Pretest freeze | `s2_76_freeze_v2_final_test_preflight.py` | Frozen model/evaluation bindings |
| Final Test | `s2_77_v2_single_final_test_evaluation.py` | Completed one declared evaluation of 55 fits |
| Publication presentation | `results/`, `evidence/`, `apps/eeg_synthetic_lab/` | Views of saved results, without new model evaluation |

Script basenames in the table are under `scripts/data/` unless another path is given. Source preservation does not imply all intermediate artifacts are distributed.

## Read the final scientific branch correctly

B0's original contract describes epsilon prediction; the selected B0 later uses v-prediction, as do B1 and B2. Early B2 work includes spectral-loss variants, but the selected final continuation has zero spectral weight. Likewise, the earlier utility runner's budget is not the final 120/8/6 early-stopping budget.

Use frozen model selection records, final production scripts, and completed reports together. The [methods document](PUBLICATION_METHODS.md) resolves these stage differences. Historical notes saying Test is sealed describe their stage; the final `s2_77` completion receipt records that the declared evaluation subsequently finished.

## Publication path mapping

Original `outputs/frozen/scenario2/` records are curated under `evidence/frozen/scenario2/`. Final utility reports are under `evidence/utility/`, and fidelity reports under `evidence/fidelity/`. Portable display extracts live under `results/tables/`.

The [inventory](../evidence/provenance/evidence_inventory.json) is the mapping authority. Frozen source/artifact hashes retain their original meaning; relocating a publication copy does not change the recorded scientific execution path. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for execution prerequisites and limits.
