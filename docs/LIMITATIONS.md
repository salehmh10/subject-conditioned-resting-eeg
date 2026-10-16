# Scope and limitations

The results support a bounded comparison of synthetic resting EEG and eye-state classification in the completed MPI-LEMON experiment.

1. **Known-participant evaluation.** The same 198 participants occur in all splits. Complete blocks are held out; unseen-person, new-session, new-device, and cross-dataset generalization were not established.
2. **Development-informed selection.** Generator choices and the final utility training policy used Validation feedback. Both B2 cohorts labeled VAL_DEV and VAL_CONF were development evidence. The final study should not be described as fully preregistered or as independently confirmed by VAL_CONF.
3. **Unequal model histories.** B0, B1, and B2 have different training and selection histories. Their comparison does not isolate the causal effect of adding participant identity. The selected B2 checkpoint does not demonstrate reproducibility of its continuation recipe across training seeds.
4. **Train-referenced fidelity.** Generated-signal fidelity and fingerprints are relative to Real Train. The separate matched-count Real–Real reference also uses Train data available during generator fitting. These are not held-out generative-fidelity tests.
5. **Classifier-dependent utility.** Full-dose B2 augmentation improves this EEGNet comparison but reduces both classical classifiers' accuracy. Its advantage over B1 is not decisive. Half-dose and B0 results prevent a simple claim that more synthetic data always helps.
6. **Training exposure differs.** A full epoch contains 374, 560, or 748 optimizer steps depending on the training condition. Equal epoch ceilings do not equalize update budgets. The Real duplicate control adds exposure without adding signal diversity.
7. **Limited uncertainty scope.** EEGNet population SD summarizes three classifier-training seeds sharing fixed synthetic pools. Subject-bootstrap CIs summarize participant variability, with matched seeds averaged within participant. They are unadjusted for multiple comparisons and do not capture all generator-training or pool-generation uncertainty.
8. **TSTR includes real-data selection.** TSTR classifier gradient updates use synthetic signals, while EEGNet checkpoint selection uses real Validation. It is not a wholly synthetic end-to-end selection pipeline.
9. **Privacy is unestablished.** No exact-copy detections within the specified strata do not rule out memorization, membership inference, or reidentification. Fingerprint recognition is an in-sample physiological descriptor, not a privacy guarantee.
10. **Connectivity is sensor-space.** Debiased squared wPLI describes channel-level phase relationships, not anatomical or causal connectivity. Finite negative estimates are retained; high ordinary channel-correlation similarity does not establish strong phase-connectivity preservation.
11. **Demographic analyses are limited.** The local distributed participant table used by the project has missing demographic values. The study does not establish demographic representativeness or subgroup fairness.
12. **Public reproduction has boundaries.** The release preserves code and evidence but excludes raw signals, processed datasets, model checkpoints, and full synthetic pools. Historical scripts retain platform, path, artifact, and hash assumptions. No full-pipeline portability or fresh experiment replication was established during publication packaging.

Future work should distinguish a new research protocol from reproduction of the frozen study. Existing Final Test outcomes must not become a development set for further tuning.
