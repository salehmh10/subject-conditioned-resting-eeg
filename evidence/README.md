# Saved scientific evidence

This directory contains an explicit selection of **existing, frozen study
records**. Publication preparation copied their bytes and checked copy integrity;
it did not execute the research pipeline, rerun scientific tests, open EEG arrays,
load checkpoints, or recalculate statistics.

| Location | Contents |
| --- | --- |
| `utility/validation/` | Final v2 Validation report and all 55 saved classifier runs |
| `utility/test/` | Completed Final Test report, all 55 fit results, 33 family/condition summaries, saved paired contrasts and completion/access receipts |
| `fidelity/b1_b2/` | Original saved B1/B2 aggregate fidelity report and analysis plan |
| `fidelity/b0_extension/` | Saved B0 extension report containing B0/B1/B2 aggregate comparisons and its analysis plan |
| `frozen/scenario2/` | Selected B0/B1/B2 metadata, pool integrity summaries and the pre-Test freeze |
| `protocols/scenario2/` | Utility protocol, final v2 budget amendment and B2 selector protocol |
| `preprocessing/` | Frozen Train-only normalization metadata and cohort split lock |
| `provenance/evidence_inventory.json` | Publication path, original repository-relative source path, byte size and SHA256 for each copied file |
| `provenance/paper_source_map.csv` | Unchanged historical source-to-paper mapping |

The paper-facing CSV and LaTeX tables are in `../results/tables/`. They are also
unchanged copies, including the all-seed learning curves and stopping table.
The original Final Test contrast CSV records **right minus left**. The
paper-facing `utility_contrasts.csv` has the explicitly named
`delta_left_minus_right_pp` field. Respect each file's documented direction and
unit; these two files are different saved representations.

Historical records describe the state when each was frozen. Statements such as
“Final Test still sealed” in earlier metadata remain unchanged. The later
`utility/test/manifests/s2_77_final_test_complete.json` records completion.
Historical paths embedded in frozen records are evidence, not portable runtime
configuration. Resolve included copies through the inventory's `source_path` to
`path` mapping; never edit a frozen record just to make an old path look current.

Public dataset pseudonyms in the original B1 subject-index mapping and Train
normalization sample counts are deliberately retained. Original tracked
cohort/split/QC metadata remains at `../outputs/metrics/` and is covered by
`../provenance/source_inventory.json`. This export adds no per-participant
predictive-performance tables, physiological diagnostic arrays, clinical records,
raw waveforms, synthetic pools, checkpoint tensors or prediction arrays.

The inventory records omitted dependencies explicitly. It does not assert that
the complete historical dependency graph is distributed. In particular,
participant-level inputs behind some aggregate figures and bootstrap analyses
are excluded. Saved statistical intervals are reported unchanged; no new
bootstrap estimates were generated for this release.

From the repository root, check copied bytes with:

```sh
python tools/verify_copy_integrity.py --root .
```

This standard-library command checks declared file sizes and SHA256 values.
It is a packaging check and does not validate scientific conclusions or execute
research code. Maintainers with the original workspace can recreate this
allowlist using `tools/export_publication_evidence.py --source ORIGINAL_ROOT
--destination SEPARATE_PUBLICATION_ROOT`; this is copying, not a study rerun.
