# Final cohort and block-split lock

Date: 2026-09-07 (local). Scope: deterministic cohort and split selection from frozen engineering evidence only.

## Final state

The final Scenario 2 cohort contains **198 subjects**. Thirteen structurally eligible subjects are excluded after frozen Train QC and provisional 5/1/2 feasibility; the four earlier structural exclusions remain excluded. All 215 dataset subjects occur exactly once in `final_cohort_manifest.csv`.

Inclusion was derived—not copied from a prompt—as:

`technical_complete AND train_qc_pass AND provisional_5_1_2_feasible`

No physiological, spectral, connectivity, predictive, generative, augmentation, Validation, or Test outcome entered membership or assignment.

## Exclusions

The four pre-download structural exclusions are:

- `sub-010015`: `insufficient_complete_blocks_for_5_1_2`; 3 EO/3 EC study runs, only 2 complete EO and 3 complete EC.
- `sub-010100`: `insufficient_complete_blocks_for_5_1_2`; 5 EO/6 EC study runs, only 5 complete blocks per state.
- `sub-010203`: `state_annotations_unavailable_missing_events`; no events table and no marker entries.
- `sub-010285`: `state_annotations_unavailable_no_state_markers`; no mapped EO/EC marker.

The 13 post-download exclusions and complete frozen reasons are:

- `sub-010012`: frozen Train QC failed; assigned EO Validation and EO Test blocks are ineligible.
- `sub-010030`: frozen Train QC failed; an assigned EO Test block is ineligible.
- `sub-010146`: frozen Train QC failed; assigned EC Validation, EO Test, and EC Test blocks are ineligible.
- `sub-010157`: assigned EC Validation and EO Test blocks are ineligible.
- `sub-010163`: frozen Train QC failed; an assigned EC Test block is ineligible.
- `sub-010200`: assigned EO Test and EC Test blocks are ineligible.
- `sub-010226`: the assigned EO Validation block is ineligible.
- `sub-010234`: frozen Train QC failed; assigned EO Validation and EO Test blocks are ineligible.
- `sub-010250`: frozen Train QC failed; assigned EO Validation and EO Test blocks are ineligible.
- `sub-010260`: assigned EO Validation and EO Test blocks are ineligible.
- `sub-010280`: the assigned EO Validation block is ineligible.
- `sub-010291`: assigned EO Validation and EC Test blocks are ineligible.
- `sub-010316`: an assigned EO Test block is ineligible.

The CSV and JSON summary retain the exact machine-readable reason strings. Six of these subjects fail Train QC; seven pass Train QC but fail the already-frozen assigned held-out feasibility requirement.

## Locked block assignments

Filtering the immutable provisional manifest to the 198 included subjects produces:

| Split | EO | EC | Total |
|---|---:|---:|---:|
| Train | 990 | 990 | 1,980 |
| Validation | 198 | 198 | 396 |
| Test | 396 | 396 | 792 |
| Total | 1,584 | 1,584 | 3,168 |

Every included subject has 5 Train, 1 Validation, and 2 Test blocks in each state. Complete recording blocks remain atomic. Every final `(subject_id, block_index)` role equals its provisional role.

Ten included Train blocks are frozen-ineligible and remain Train without replacement:

- `sub-010020_block-16`
- `sub-010024_block-14`
- `sub-010066_block-16`
- `sub-010196_block-05`
- `sub-010232_block-02`
- `sub-010263_block-12`
- `sub-010263_block-16`
- `sub-010276_block-02`
- `sub-010305_block-02`
- `sub-010315_block-14`

All 1,188 included Validation/Test blocks are eligible. The final block manifest binds each frozen mask and reason vector by SHA256 and records deterministic rejection-reason counts. These QC datasets were read without dereferencing `eeg_uv`.

## Pilot exposure and access protection

The pilot registry contains 150 unique exposed Train blocks. Of these, 130 belong to 13 included pilot subjects and remain Train; 20 belong to excluded `sub-010250` and `sub-010280`. No pilot-exposed block becomes Validation or Test.

The 150-row pilot Train registry and disjoint 1,960-row streaming Train registry cover all 2,110 provisional Train assignments. The 1,266-row mechanical registry covers every provisional Validation/Test assignment. All access rows match block identity, state, interval, role, and frozen hashes. Mechanical access uses only the pre-authorized apply-only preprocessing purpose; no scientific held-out purpose appears.

## Artifact identities

| Artifact | SHA256 |
|---|---|
| `final_cohort_manifest.csv` | `e9405aacd8109700ed55ef3df49b8b5ec0187a65b1c3d79049597fa27b490bc9` |
| `final_block_split_manifest.csv` | `a58605010f7371fe1805a7834c205f2da7be816515779e15f2ba082bfd201f8b` |
| `final_cohort_split_summary.json` | `fe90564149a5a7740d66952134473ec45122dd1c5acf51f7d7b5e6a0c0551adb` |
| `final_cohort_split_lock.json` | `c591a46aafaa11cda0cb5b8fd3f39d1a5df61ac3fa8973fae96f804857edfe0f` |

The lock uses deterministic UTF-8/LF serialization, fixed CSV schema/order, sorted JSON keys, and no timestamps. Two successive generations from identical inputs are byte-identical.

## Frozen source bindings

| Source | SHA256 |
|---|---|
| Metadata inventory | `bf7b1f27ed9f1d7c0dcd137c4e52009a758803fec72d2fb702ab59963912457c` |
| Structural cohort manifest | `85bf4786b2edf06b0c6653c94bc5c3b71e5fa116c0c95d0d826e71cfabac43a3` |
| Full-cohort subject manifest | `81705601b558d5429de50cb9d26084de530d99708dbec0867c1f6f6c3fd52be6` |
| Full-cohort QC summary | `c7bdbc41171e85cbf5c4a63008ea5978f1145d4e6709842124b832b58dcdae05` |
| Full-cohort completion audit | `76af2a13a3a592e2b1035d9f36cb2c3c7578733ca17e2707cbba91f4bc03d738` |
| Provisional split manifest | `6adc10b108a7e7b587757cbda630a3c4909c97f8cc2faf24320b0a6e89af0b26` |
| Block inventory | `d5deff471d02baa55ac60539986595b82594d5137fcf1bffac6a5ec982b28742` |
| Pilot Train registry | `cc0ff1542f7cb026e491c36d1211ee119095d6a9a6334b81002dea6aec489fed` |
| Streaming Train registry | `60bc00030f6f5e910776b8d7ad90e6f5fbbe12b57f2eede7b5e984a1dc811da9` |
| Mechanical held-out registry | `50ffc3a04886ae7211a51162e664d8c34ac4a6342a93f3de9ff54312e85b389b` |
| Preprocessing YAML bytes | `da57489746e66fa819da16217e585cdd0ca3b4e5cbe4491603e831be98889110` |
| Mechanical-access YAML bytes | `4fbf4106c7f42957325cd12ba66294fd2d1a43e435e5f7282a318e8102976053` |

The preprocessing semantic hash is `5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b`; the mechanical-access semantic hash is `d04b43f1ebe32bf41293d14e7c2f034e03c567ed0063350630aa81410e2500d4`. The lock also binds the accepted streaming record and canonical bundles for all 211 HDF5 output and receipt hashes.

## Verification and reviews

The generator/verifier rejects duplicate or missing subjects, unexpected cohort counts, altered roles, pilot exposure violations, source-hash mismatches, non-idempotent output, registry gaps, ineligible held-out assignments, fitted normalization, Batch 006, and tracked signal payloads. Three harmless inventory/provisional floating-point text differences are compared numerically at `1e-9` absolute tolerance.

Independent Data Reconciliation and Leakage/Split reviews reproduced the subject, role, eligibility, registry, and pilot counts with no scientific contradiction. The Artifact/Test review identified fail-closed acceptance hardening issues: stale/tiny JUnit evidence, unparsed command/reviewer claims, merge-commit acceptance, incomplete full-cohort assertions, unresolved-review handling, alternate-path labeling, and an absolute tracked external path. All were resolved with expanded negative tests, testcase-level JUnit validation, parsed verifier/reviewer evidence, repeated compile/determinism checks, exact full-cohort assertions, merge/multiple-commit rejection, and a canonical runtime-relative binding.

The focused final-lock tests pass 15/15, and the guarded full suite passes 398/398 with no failures, errors, or skips. `compileall` and the pre-commit deterministic verifier pass. The external acceptance record binds the clean post-commit final-lock and Batch 005 preservation verifier results to the committed HEAD.

The required final command sequence is:

```powershell
. .\scripts\setup_scenario2_environment.ps1 -ForTests
.\.venv\Scripts\python.exe -m pytest -q --basetemp "D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp\pytest" --junitxml outputs/logs/pytest.xml
. .\scripts\setup_scenario2_environment.ps1
.\.venv\Scripts\python.exe -m compileall scripts src tests
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 005 --test-report outputs/logs/pytest.xml
.\.venv\Scripts\python.exe scripts/data/s2_16_final_cohort_split.py
```

After the single local commit, rerun both verifiers with `--require-clean`. Exact stdout/stderr, exit codes, JUnit evidence, reviewer records, final committed HEAD, and artifact hashes are preserved at:

`D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/runtime/docs/final_cohort_split_lock_20260907/`

The tracked handoff intentionally does not embed its own commit hash; `final_acceptance.json` in that external directory resolves the committed clean state.

## Preserved boundaries and next action

No raw, HDF5, receipt, access registry, historical journal, or frozen source was modified. No EEG samples were loaded to decide membership or assignments. No alpha/IAF/PSD/connectivity/wPLI, classifier, generator, TSTR/TRTR, augmentation, plot, or scientific Validation/Test result was computed. No Batch 006 exists; no push occurred.

Normalization is still unfitted and Final Test remains sealed. The next separate phase is to fit the one global channelwise mean and population standard deviation from unique continuous samples covered by usable windows in eligible final Real Train blocks only, counting overlaps once, then freeze that scaler before model work.

NEXT ACTION â€” TRAIN-ONLY NORMALIZATION
