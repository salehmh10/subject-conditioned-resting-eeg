# Scenario 2 Decision Log

## 2026-09-04  Standalone repository

Scenario 2 is maintained in a separate repository:

`subject-conditioned-resting-eeg`

The frozen Scenario 1 repository will not be modified.

---

## 2026-09-04  Primary dataset

MPI-LEMON is the primary dataset for Scenario 2.

---

## 2026-09-04  Block-level splitting

Train, Validation, and Test splitting must occur at complete recording-block level.

Window-level splitting is prohibited.

---

## 2026-09-04  Distribution-preserving split

The split must preserve the original dataset distribution as closely as possible using non-outcome metadata and QC measures.

Physiological and predictive outcomes must not be used to optimize the split.

---

## 2026-09-04  Main generator ablation

The primary models are:

- B0: State-only DDPM
- B1: Subject + State DDPM
- B2: Subject + State DDPM + Spectral Regularization

---

## 2026-09-04  Connectivity

Functional Connectivity is a major evaluation domain but is not initially included in the training loss.

Primary candidate:

debiased wPLI in the Alpha band.

---

## 2026-09-04  Three classifiers

The augmentation study will use:

- Logistic Regression
- Tangent-Space Linear SVM
- EEGNetv4

---

## 2026-09-04  Augmentation ratios

Planned ratios:

- 0.5x
- 1.0x
- 2.0x

Validation and Test remain real-only.

---

## 2026-09-04  IAAFT control

IAAFT spectral-surrogate augmentation is included as a control experiment.

---

## Pending after MPI-LEMON audit

The following are intentionally not yet locked:

- usable subject cohort
- exact number of EO/EC blocks
- exact Train/Validation/Test block allocation
- final channel availability
- final waveform-QC eligibility across the full cohort (methods specified below)

---

## 2026-09-05  Pre-download audit amendments

Timing: all decisions in this section were made before any real EEG waveform was downloaded or analyzed, before examining any Validation outcome, and before Final Test.

### Channel composition and sampling rates

The 62 rows in every `channels.tsv` are classified by BIDS type as 61 EEG and 1 EOG (`VEOG`), with 0 miscellaneous channels. They must not be reported as 62 EEG channels. All 19 target scalp channels are present and marked good in all 215 subjects.

Native sampling is non-uniform: 208 subjects are 2500 Hz; `sub-010216` is 500 Hz; `sub-010276`, `sub-010277`, `sub-010278`, `sub-010279`, `sub-010280`, and `sub-010282` are 1000 Hz. These seven are retained structurally because every rate supports the planned 128 Hz target. Resampling behavior remains a waveform-pilot question.

### State-code semantics

State meaning comes only from `trial_type`: S200 is EO and S210 is EC. S208 is not a global alias. It maps to EC only for `sub-010078` and `sub-010126`, through `configs/audit/event_code_overrides.yaml`, with the observed cadence and alternating-block evidence recorded there.

### Acquisition interruptions in sub-010059 and sub-010081

Explicit `New Segment/` markers prove that the apparent long first EC spans are merged acquisition fragments, not continuous state blocks. The parser separates and preserves two aborted EC fragments for `sub-010059` and one for `sub-010081`. Their post-restart sequences each contain eight complete EO and eight complete EC blocks, so both subjects remain structurally eligible and waveform-QC flagged. No EO/EC label is imputed inside any interruption or transition gap.

### Pre-download structural cohort

The cohort is defined by readable structural metadata, complete target-channel coverage, usable EO/EC annotations, and eight complete alternating blocks per state. It contains 211 subjects. It is explicitly not the final physiological cohort.

Structural exclusions are:

- `sub-010015`: insufficient blocks (3 EO/3 EC) and a truncated final EO.
- `sub-010100`: insufficient blocks (5 EO/6 EC), a truncated final EC, and no sixth EO.
- `sub-010203`: unavailable EO/EC annotations; events are absent and `.vmrk` has no marker entries.
- `sub-010285`: no state markers.

Every retained subject requires uniform, prespecified waveform QC. No alpha, IAF, connectivity, classifier, generator, TSTR, or augmentation outcome contributed to these decisions.

### Participant metadata limitation

All 215 values for age, sex, handedness, weight, and height in the distributed `participants.tsv` are `n/a`. Full-versus-retained demographic distortion therefore cannot be evaluated locally. Dataset README age-group aggregates are not substituted: they are not subject-linked and sum to 227 rather than the 215 EEG subjects.

### Provisional 5/1/2 feasibility

Five Train, one Validation, and two Test blocks per subject and state is structurally feasible for all 211 retained subjects. The metadata-only candidate uses a fixed SHA256 subject ordering, balanced cyclic block-order rotations, and an EO/EC offset. It is labeled `PROVISIONAL_METADATA_ONLY`, not final or locked.

Final membership waits for frozen outcome-blind waveform QC. After QC, the final manifest must be deterministically generated, checksummed, committed, and frozen. QC can enforce prespecified eligibility but cannot be optimized to make held-out sets cleaner or easier.

### Pilot access and leakage prevention

Because one `.eeg` file contains a whole recording, pilot download does not authorize inspection of every contained block. The original pre-download allowance for excluded-fragment diagnostics is superseded by the stricter 2026-09-05 amendment below. Exposed complete blocks must remain Train in the final manifest or their subject must be excluded. Pilot Validation/Test signals cannot guide preprocessing or QC choices.

## 2026-09-05  Pilot engineering specification before value access

Timing: specified after header/hash validation and synthetic tests, before any pilot waveform values, Validation/Test waveform inspection, generator/classifier development, or Final Test. The user explicitly disallows interruption-fragment access; the earlier exception is removed.

The 15 subjects and all 30 backed payloads pass size/SHA256, MNE header, linkage, sample-count, channel, unit and template-montage checks. Native header resolution 0.1 µV corresponds to 1e-7 V per binary unit in MNE. Metadata duration uses the last-sample-time convention. Use a new isolated `.venv`; no Scenario 1 or global installation was modified.

Protocol 0.3 section 11 and the canonical YAML specify every parameter. The 1–40 Hz FIR and polyphase resampler have finite combined half-support 3.378125 s; rounding upward gives the 433-sample edge guard. Synthetic passband gains are within 0.0005 of unity, 60 Hz residual amplitude <0.001, and 100 Hz residual <7e-8 at all three native rates. These are synthetic engineering tests, not physiological measurements. The implementation follows [MNE filter semantics](https://mne.tools/1.6/generated/mne.filter.filter_data.html) and locally inspected pinned source.

Use conservative robust broadband metrics and at least two ordinary bad-channel votes, bounded interpolation (six of 61 EEG, two of 19 targets), and full 61-channel CAR. Template coordinates are used because no individual digitization is staged. VEOG is an artifact proxy, not a generated channel. The thresholds are engineering safeguards, not clinically validated artifact labels. Channel-amplitude and derivative features are related, so ordinary failure requires low correlation plus an amplitude or derivative outlier. Amplitude and derivative alone cannot designate a bad channel.

Predeclare pooled Train window multiplier 8 with floors/caps and sensitivity candidates 6/8/10; no state-specific fitting or outcome optimization. Require ten windows and 50% retention per block, and three usable Train blocks per state for pilot engineering acceptance. The rule avoids claiming final 5/1/2 feasibility before held-out mechanical QC. Pilot evidence will confirm or challenge these fixed engineering choices and will be recorded below.

HDF5 stores continuous physical-unit blocks plus window indices/masks, avoiding overlap duplication. Final normalization uses unique usable final Real Train samples once each and remains unfitted. Streaming replaces simultaneous 125.554 GiB staging. Raw deletion requires explicit later execution plus successful fresh verification; all pilot raw remains staged in this phase.

### Download planning

The metadata clone stores 215 `.eeg` and 215 `.vhdr` NEMAR annex-key pointer texts; it is not a functional git-annex/DataLad checkout. The full structural candidate manifest contains 422 assets for 211 subjects (134,813,081,876 declared bytes). The deterministic pilot contains 30 assets for 15 subjects (7,012,992,712 declared bytes) and covers all rare sampling rates, both S208 overrides, both resolved acquisition-interruption cases, and four ordinary 2500 Hz controls.

The future downloader stages outside the read-only metadata clone, verifies per-file size and SHA256, and is dry-run unless `--execute` is supplied. No download command was executed during this amendment.

## 2026-09-05  Train-only pilot evidence and preprocessing freeze

Timing: after accessing exactly 150 complete provisional Train blocks (ten per pilot subject), before any Validation/Test waveform access, model/classifier development, or Final Test. All intervals and source metadata digests are in `pilot_waveform_access_registry.csv`. No interruption fragment was read. Those 150 blocks must remain Train if their subjects enter the final study.

All 15 raw recordings pass integrity, and all 15 compact Train outputs pass structural/provenance/hash verification. Every native target/VEOG sample audited is finite; no NaN/Inf or catastrophic channel flatness was detected. The fixed subject channel rule identifies zero bad EEG and zero bad targets. Real interpolation was therefore unnecessary; a synthetic missing-target test verifies interpolation before full-montage CAR. The review also added native block checks so CAR cannot conceal a local flat signal, and nonfinite VEOG cannot be silently repaired into apparently usable QC evidence.

Under the predeclared multiplier 8, 3,490/3,750 candidate window masks are usable (93.0667%); 143/150 blocks meet eligibility. Of those windows, 3,415 are in eligible blocks and 3,276 are in eligible blocks of the 14 QC-passing subjects. A usable mask alone does not authorize an ineligible block or subject for training. Sensitivity candidates 6 and 10 retain 3,409 and 3,577 windows respectively. Multiplier 8 is retained unchanged; these candidates do not establish an optimal artifact detector. Subject-median retention is 97.6%, and the minimum is 73.6%. Native-rate groups retain 96.8% (500 Hz; n=1), 93.9333% (1000 Hz; n=6), and 91.95% (2500 Hz; n=8). There is no rate-group QC collapse. Synthetic mechanical gain tests establish resampling stability; unequal real subject groups cannot establish causal amplitude equivalence across acquisition rates.

Fourteen subjects satisfy the pilot subject-QC rule. `sub-010250` fails because only one of its five EO Train blocks reaches the block minimum; its five EC Train blocks pass. Its 66 rejected EO windows all exceed the pooled VEOG peak-to-peak threshold of 259.9425956 µV. Preserve this failure, compact data, and raw evidence; do not relax the method to rescue the subject or reassign exposed blocks. This is provisional engineering ineligibility, not a final cohort/split decision. It blocks automated raw cleanup for this subject.

Across the pilot, EO retention is 1,632/1,875 (87.04%) and EC retention is 1,858/1,875 (99.0933%). VEOG contributes to 220 of 260 rejected windows; reasons can overlap. The method does not erase either state globally, but the 12.0533 percentage-point retention difference and subject-250 failure are material limitations. Keep state-pooled fitting and document this imbalance; do not introduce state-specific thresholds. No physiological spectrum, alpha topology, connectivity, classifier, generator, TSTR, or augmentation outcome informed the decision.

The pilot occupies 68,563,819 compact bytes (65.3876 MiB). A 211-subject planning estimate is 0.89823 GiB for ten Train blocks each, or 1.43716 GiB for all 16 blocks if later mechanical held-out processing is authorized. The second figure is extrapolation, not a generated dataset. Full-cohort QC, final allocation, and numerical normalization remain unfrozen.

The actual orchestrator processed only staged pilot subjects with `--local-existing-only --keep-raw`. A second full pilot run reverified sources and reused all 15 byte-identical outputs without adding any waveform access record. Independent synthetic rebuilds produce identical arrays/masks/reasons; HDF5 creation timestamps may differ. All 105 raw files remain, totaling 7,013,640,445 bytes. Fourteen subjects satisfy the prepared cleanup checks; none was cleaned. No non-pilot EEG request, raw deletion, metadata-clone mutation, or Git push occurred.

The current orchestrator deliberately stops at provisional Train. Later final-cohort QC needs an explicitly authorized mechanical held-out extension under this frozen method. If raw is cleaned before that extension, the verified source will have to be staged again to obtain held-out blocks. Final Test scientific outcomes remain protected by the separate study lock sequence.


## 2026-09-05: frozen mechanical held-out authorization

This amendment is written and committed before the first held-out value read.
The user separately authorizes mechanical access to the 90 provisional Validation/Test
blocks of the existing 15 pilot subjects through `MechanicalHeldoutAccessGate`.
The frozen preprocessing YAML and SHA256 remain unchanged:
`5ed46e77f318e2f6162cbd34f600951980117cef979bbdf12640094721d7947b`.
The separate `configs/access/mechanical_heldout_v1.yaml` defines access purposes,
manifest hashes and apply-only restrictions; it contains no preprocessing parameters.

Validation/Test may apply verified Train-derived thresholds and bad-channel decisions
but cannot fit any parameter, including normalization. Allowed operations are frozen
preprocessing, fixed waveform QC, compact storage and output verification. Scientific
outcomes remain sealed: no PSD, Alpha, IAF, topology, Connectivity, wPLI, classifier,
generator, TSTR, augmentation, plots, tuning or scientific evaluation is authorized.
All 150 previously exposed pilot blocks must remain Train or their subject must be
excluded. Reuse their existing verified compact datasets without native Train rereads.
Process sub-010250 identically and preserve its frozen Train-QC failure.

Store all 16 complete blocks, including ineligible blocks with masks and reasons, in
separate all-block HDF5 files. Keep the old Train files and original access registry.
Provisional feasibility requires frozen Train QC (at least three eligible blocks per
state), both assigned Validation blocks and all four assigned Test blocks eligible,
plus source/header/output verification. No spare held-out blocks, reassignment,
threshold changes, or subject rescue is permitted. This does not lock a final cohort,
final split or final Test, and normalization remains unfitted.

All-block storage and complete provenance/journal verification are now mandatory
before any future explicitly requested raw deletion, superseding earlier Train-only
cleanup eligibility. No pilot raw deletion, non-pilot download, metadata clone change,
or Git push is authorized in this phase. Future streaming supports `--processing-scope
all-blocks-mechanical`: fit on Train first, then mechanically apply to held-out, verify
all 16 blocks, durably journal, and delete only with the explicit guarded flag.


## 2026-09-05: completed frozen mechanical pilot

Authorization and execution code were committed at `887634f16a9637fe2ba91b27dd02f37850d5fd30` before the first held-out read. All 15 subjects completed without a real processing failure. The 15 separate all-block outputs contain 240 blocks: 150 byte-identical copied Train blocks and 90 newly mechanically processed held-out blocks (30 Validation, 60 Test). The original Train registry and frozen preprocessing files are unchanged.

Frozen Train QC passes 14 subjects; provisional 5/1/2 feasibility passes 13. `sub-010250` remains a Train-QC failure, with its EO Validation and two EO Test blocks also ineligible. `sub-010280` has an ineligible assigned EO Validation block. No reassignment, threshold change, bad-channel refit, normalization fit or rescue occurred. These are provisional engineering labels, not final inclusion/exclusion decisions.

Train EO/EC usable windows are 1632/1875 and 1858/1875; Validation EO/EC are 312/375 and 371/375; Test EO/EC are 653/750 and 743/750. Every assigned block remains stored with its mask and reason. State retention differences are reported without tuning.

All-block storage is 109,736,191 bytes (104.652587 MiB); mean-pilot extrapolation to 211 subjects is 1.437610 GiB. The idempotent pilot rerun reused every output with unchanged hashes and access registries. Full synthetic/mock pytest passed 208 tests; compileall and the full 211-subject read-only dry run passed.

Fourteen subjects are technically eligible for later cleanup under the preserved Train-QC safeguard and new mandatory all-block completeness checks; this differs from 13-subject scientific allocation feasibility. No cleanup was performed. All 105 raw files and old Train HDF5/receipts remain unchanged. No non-pilot download, scientific held-out computation, waveform plot, model training/evaluation, metadata-clone modification or Git push occurred. See `MECHANICAL_HELDOUT_HANDOFF.md` for detailed evidence and next-phase commands.


## Variable streaming batches and external storage (Batch 001 authorization)

Batch 001 has a maximum **5 GiB = 5,368,709,120 bytes** of declared `.eeg` plus `.vhdr` payload. Future batch caps are chosen separately by the user; no future membership is generated now. Frozen caps, membership, manifest hashes, and subject order are immutable. The planner excludes verified completed outputs and previously assigned subjects, follows the authoritative candidate order, and stops at the first subject that does not fit. It does not pack smaller later subjects into a gap or pull replacements after failure.

Execution uses only the frozen manifest and a durable authorization referencing committed code. It processes one new raw subject at a time: verified download/resume, Train-only parameter fitting, mechanical apply-only held-out processing, verified 16-block HDF5 and receipt, durable completion journal, then guarded raw cleanup before the next subject. Passing Train QC is required for deletion; legitimate held-out 5/1/2 infeasibility alone does not block technical cleanup. Download/preprocessing failures, partial files, and Train-QC failures retain raw. The first subject is checked as a canary. Infrastructure/software failure stops for investigation and a committed correction under the same membership and cap.

Expected network payload is reported separately from measured response-body bytes. Measurement includes received retry bodies; HTTP/TLS/ISP overhead is not measured. Existing verified files reduce expected transfer. Partial lengths are resumable prefixes, not cryptographically verified incomplete data; final whole-file SHA256 remains mandatory and ignored HTTP Range can increase transfer.

Large Scenario 2 data is deliberately external to Git, under `D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/`: `raw/`, `processed/all-blocks-v1/`, preserved `processed/train-only-v1/`, `temp/`, and `runtime/`. The original `D:/SHARIF/PROJECT/Datasets/MPI-LEMON` metadata clone remains read-only. `configs/paths/data_paths.yaml` commits relative defaults; an ignored `configs/paths/local_data_paths.yaml` or `SCENARIO2_PATH_CONFIG` may override local paths. Authorizations record resolved absolute paths. Batch commands refuse path substitutions that differ from this centralized policy.

Pilot migration inventories and verifies destination bytes and both sets of compact outputs before removing old copies. Original HDF5 files and receipts remain byte-identical; the relocation resolver translates their historical output paths. Small manifests, ledger, authorization, metrics and summaries remain in the repository; detailed batch runtime journals and processing caches are external. The migration summary records counts, hashes, roots and verification.

This is a storage and engineering change only. Frozen preprocessing/QC and provisional split rules are unchanged. Scientific held-out outcomes remain sealed, no waveform plots are produced, normalization remains unfitted, and no final cohort is locked. All commits are local; no push is authorized. **Batch 002 is neither planned nor executed, and its cap remains undecided.**

## Batch 001 interruption and durable D: temp/cache policy

The user manually stopped Batch 001 after C: fell to approximately 600 MB free, then manually cleared about 7.7 GB of pip cache; C: recovered to approximately 8.8 GiB. The agent did not change Windows pagefile configuration. `sub-010002` had already passed source/output/provenance verification and Train QC, and its raw directory was safely deleted. `sub-010003` retained a 419,430,400-byte waveform partial for strict HTTP Range resume.

Dot-source `scripts/setup_scenario2_environment.ps1` in every new shell before running Python, streaming, verification, compileall, or pip. It resolves the central path policy and sets TEMP/TMP/TMPDIR, pip, matplotlib, joblib, Python bytecode, MNE and XDG cache/config paths under the external D: Scenario 2 root. It verifies Python `tempfile.gettempdir()` in the execution environment. `-ForTests` also checks the system-drive guard. All remaining pytest runs explicitly use `--basetemp D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/temp/pytest`; pytest refuses another base and keeps its own cache external. Check for another running pytest process before reusing that base.

The Windows stability guard stops before subject download/resume, asset requests, preprocessing, and full tests if C: free space is below 3 GiB (3,221,225,472 bytes), recording `c_drive_low_space_guard` while preserving files and journals. Project-controlled data/temp/cache writes remain on D:. Reading Python installed on C: is permitted. This policy covers project-controlled writes, not unrelated applications or OS-managed storage.

The original Batch 001 authorization, plan, manifest hash, cap and membership are preserved. A separate numbered resume authorization references the original authorization hash, the committed hardening code, reverified completed subjects, remaining-transfer preflight, and exact environment. `sub-010002` is reused without raw recreation or processing. `sub-010003` resumes from its preserved prefix; validated HTTP Content-Range and actual new response-body bytes are recorded. Batch 002 remains unplanned, and all scientific/QC rules remain unchanged.


## 2026-09-05 — Batch 001 results and stop boundary

The immutable 5-GiB batch completed the ordered subjects sub-010002, sub-010003, sub-010004, sub-010005, sub-010006, sub-010007, sub-010010, sub-010012. Declared, initially expected, and measured application response payload each equal 5,104,883,272 bytes (4.754293031990528 GiB); unused capacity is 263,825,848 bytes. One user-initiated low-C-space stop preserved a 419,430,400-byte sub-010003 prefix, later reused under validated HTTP 206 Content-Range. No network retry occurred. HTTP/TLS/ISP overhead and bytes not returned by the response reader are not measured by application accounting.

All eight outputs have verified 16-block storage and durable receipts. Seven pass Train QC and provisional 5/1/2 feasibility. Sub-010012 fails frozen Train QC and assigned EO Validation/Test eligibility; its 648,429,341-byte raw directory is retained. The seven other Batch raw directories were deleted immediately after guarded success, freeing 4,456,805,158 bytes. Pilot cleanup deleted 14 directories and freed 6,362,309,563 bytes, retaining sub-010250. Held-out infeasibility alone did not block the technically valid Pilot sub-010280 cleanup.

Batch compact HDF5 addition is 58,508,245 bytes. Cumulative all-block storage is 168,244,436 bytes for 23 subjects / 368 blocks; 188 structural candidates remain unassigned. Preserved Train reference outputs remain external. Remaining raw storage is 1,299,760,223 bytes for sub-010012 and sub-010250. Migration preserved 166 files / 7,193,811,222 bytes, including byte-identical HDF5 and receipts, before removing old copies. The original metadata clone remains unchanged, and no large raw/processed signal file remains under the repository.

The full final suite passed 275 tests; compileall and the read-only batch verifier passed. Scientific preprocessing and QC hashes are unchanged. No scientific held-out analysis, waveform plot, normalization fit, final cohort/split lock, replacement subject, or Git push occurred. The first canary used implementation/migration commit 13cd91df96357d42dfc2a7284ac9382050878a13; the resumed subjects used D-only environment/guard commit 5a9210ea2cbf85f205c518d5abb39ed208d7d575 with a separate resume authorization preserving original frozen artifacts.

Batch 002 was NOT planned or executed. Its cap remains UNDECIDED and user-selectable. See BATCH_001_STREAMING_HANDOFF.md for final acceptance and examples that were not executed.


## 2026-09-05: Batch 002 authorization before network I/O

The user now explicitly authorizes Batch 002 only with exactly 10 GiB (10,737,418,240 bytes) of complete declared EEG/header payload. Batch 001 was verified complete at clean HEAD bb51533b9a5019a8006235e30c1fe3d00684526c before changes. Its original records, ledger entry, 23 existing outputs/receipts, retained raw, scientific source/configurations, and prior access records are protected by batch_002_starting_state.json. Historical Batch 002 prohibitions described the earlier assignment; this separate authorization supersedes that stop boundary for Batch 002 only. Batch 003 remains prohibited and unplanned.

The frozen contiguous membership is: sub-010016, sub-010017, sub-010019, sub-010020, sub-010021, sub-010022, sub-010024, sub-010026, sub-010027, sub-010028, sub-010029, sub-010030, sub-010031, sub-010032, sub-010033, sub-010034. Total 10,141,417,744 bytes (9.444931283593 GiB), 16 subjects / 32 assets, unused 596,000,496 bytes. The first unassigned subject afterward is sub-010035; it is not assigned to a future batch. No bin packing, replacements, or reprocessing of completed subjects is permitted.

All runtime/cache/temp/signal storage remains external on D:. The 3-GiB C: guard, sequential processing, strict source/Range verification, Train-only fitting, mechanical held-out apply-only access, all 16 blocks/masks/reasons, durable receipt/journal and immediate guarded cleanup remain mandatory. Train-QC failures retain raw. Scientific preprocessing, QC, provisional allocation, and normalization status are unchanged. No scientific held-out analysis, model evaluation, plots, final cohort lock, normalization fitting, or push is authorized.

The verifier now supports cumulative access registries and checks prior-batch preservation. Offline regressions cover the exact 10-GiB contiguous membership, exclusion, immutable prior ledger/artifacts, registry tampering and the Batch 003 boundary. Required tests and implementation/documentation commit precede durable execution authorization and first network I/O. Detailed execution journals remain external; only compact evidence is committed.


## 2026-09-05: Batch 002 results and stop boundary

Batch 002 execution is complete: **16 verified all-block outputs (256 blocks)** under an immutable **10-GiB cap (10,737,418,240 bytes)**. Declared, expected, and measured response-body payload are each **10,141,417,744 bytes (9.444931284 GiB)**; unused capacity is **596,000,496 bytes**. There were no retries, resumptions, interruptions, or technical failures.

Fifteen subjects pass frozen Train QC and provisional 5/1/2 feasibility. **sub-010030** fails both and retains its **645,403,807-byte** raw directory. The other fifteen raw directories were deleted after verification, freeing **9,496,716,931 bytes (8.844506863 GiB)**. Batch all-block HDF5 addition is **116,775,031 bytes (111.365348 MiB)**.

Cumulative progress is **39/211 complete all-block subjects / 624 blocks**, with **172 unassigned candidates**. Cumulative Train-QC pass is 36/39 and provisional feasibility is 35/39. Raw remains only for sub-010012, sub-010030, and sub-010250: **1,945,164,030 bytes (1.811575172 GiB)**. Processed storage including Train references and receipts is **464,293,360 bytes (0.432406887 GiB)**. These engineering labels do not lock the final cohort or split.

Batch 001 records, its ledger entry, all 23 prior outputs/receipts, retained raw, and prior access records remain unchanged. All signal/temp/cache/runtime data stays external on D:, and the original metadata clone remains unchanged. Final tests passed **280 in 33.43 seconds**; compileall and the read-only Batch 002 verifier passed. No scientific held-out analysis, waveform plot, normalization fit, final cohort/split lock, or Git push occurred. **Batch 003 was not planned; its size remains undecided.**

See `BATCH_002_STREAMING_HANDOFF.md` for exact membership, hashes, network accounting, cleanup, verification, and restart commands.


## 2026-09-05: Batch 003 authorization before network I/O

The user now explicitly authorizes Batch 003 only with exactly 20 GiB (21,474,836,480 bytes) of complete declared EEG/header payload. Starting clean HEAD is 28815a4a3245aacd36aea241a17cf6458f459ef4. The read-only Batch 002 verifier passed, and all 39 existing all-block outputs/receipts and prior completion journals were reverified. Batch 001/002 artifacts, both ledger entries, existing output/receipt hashes, retained raw, scientific source/configurations, split/block inventory, and prior access records are protected by batch_003_starting_state.json (222 artifacts). Historical Batch 003 prohibitions applied to the earlier assignment; this separate user authorization supersedes that boundary for Batch 003 only. Batch 004 remains prohibited and unplanned.

Frozen contiguous membership: sub-010035, sub-010036, sub-010037, sub-010038, sub-010039, sub-010040, sub-010041, sub-010042, sub-010044, sub-010045, sub-010046, sub-010047, sub-010048, sub-010049, sub-010050, sub-010051, sub-010052, sub-010053, sub-010056, sub-010060, sub-010061, sub-010062, sub-010063, sub-010064, sub-010065, sub-010066, sub-010067, sub-010068, sub-010069, sub-010070, sub-010072, sub-010073, sub-010074. Total 21,299,575,247 bytes (19.836775257252 GiB), 33 subjects / 66 assets, unused 175,261,233 bytes. Verified first remaining subject is sub-010035; the first unassigned afterward is sub-010075. The latter does not fit and is not assigned to a future batch. No bin packing, replacements, or reprocessing of completed subjects is permitted. Sub-010012, sub-010030, and sub-010250 remain intentionally retained and unchanged.

Execution remains strictly sequential with a first-subject canary. Only one subject is active staging work at a time; terminal QC/technical failures retain raw and partials. Train-only fitting, mechanical held-out apply-only processing, all 16 blocks/masks/reasons, source/sidecar/BrainVision/output/receipt verification, durable journals, and guarded immediate raw cleanup remain unchanged. Held-out infeasibility alone does not block cleanup when Train QC and technical completeness pass. A software/infrastructure failure stops for investigation and committed correction under the same immutable membership.

All signal/temp/cache/runtime storage remains external on D:. Dot-source the environment bootstrap before every Python command, and stop before expensive steps below 3 GiB C: free space. No pagefile change, scientific held-out analysis, waveform plot, numerical normalization fitting, final cohort/split lock, or push is authorized. The generalized prior-state guard and Batch 003 verifier enforce the two prior ledger entries and Batch 004 boundary; cumulative summary accounting now includes Train-QC and provisional-feasibility counts and total processed bytes. Required offline tests and local code/docs commit precede durable authorization and first network I/O.


## 2026-09-06: Batch 003 terminal accounting and stop boundary

**BATCH_003_STREAMING_COMPLETE** means the frozen attempt sequence has stopped: **33 terminal subjects, 31 technically complete, 2 unresolved download failures**. The ledger correctly remains `failed_or_partial`. The immutable cap is **20 GiB = 21,474,836,480 bytes**, covering 33 subjects / 66 assets and **21,299,575,247 declared bytes (19.836775257 GiB)**. Unused capacity is **175,261,233 bytes (0.163224743 GiB)**.

All 31 completed subjects pass frozen Train QC and provisional 5/1/2 feasibility; their 496 stored blocks, hashes, receipts, journals, and guarded raw deletion verify. No assessed Batch 003 subject fails QC or feasibility. **sub-010073 and sub-010074** failed downloading after retry exhaustion and have no QC/feasibility assessment or processed output. Sub-010073 retains a **402,653,184-byte (384-MiB)** partial; sub-010074 retains its empty raw directory. The initial timeout was followed by DNS failures. There were **6 retries, 8 download interruption events, 0 manual restarts, and 0 validated resumed-prefix bytes**. Network execution ended with exit code 1; no further download or replacement was attempted after all terminal states.

Initially expected transfer was **19.836775257 GiB**; measured response-body transfer was **20,409,934,113 bytes (19.008232386 GiB)**. HTTP/TLS/ISP overhead is not measured. Guarded cleanup deleted 31 raw directories and freed **20,008,642,245 bytes (18.634500210 GiB)** including sidecars. Batch all-block HDF5 addition is **226,408,568 bytes (215.920036 MiB)**.

Cumulative progress is **70/211 technically complete subjects / 1,120 blocks**, **67 Train-QC passes**, and **66 provisional-feasibility passes**. There are **139 unassigned candidates**, beginning with sub-010075, plus the two incomplete subjects still assigned to Batch 003. Current raw storage is **2,347,817,214 bytes (2.186575172 GiB)**; processed storage including Train references and receipts is **833,108,692 bytes (0.775892932 GiB)**. Prior Batch 001/002 evidence, all 39 prior outputs/receipts, and retained raw sub-010012, sub-010030, and sub-010250 remain unchanged.

Final tests passed **291 in 49.89 seconds**; compileall and the read-only terminal-state verifier passed. Post-stop verifier changes distinguish download failure from technical completion and require durable matching failure evidence and intact retained files; they do not change preprocessing or authorize further network I/O. All signal/temp/cache/runtime storage remains external on D:, with the 3-GiB C: guard active. Frozen preprocessing and access rules remain unchanged. No scientific held-out analysis, waveform plot, normalization fitting, final cohort/split lock, or Git push occurred. **Batch 004 was not planned; its size remains undecided.**

See `BATCH_003_STREAMING_HANDOFF.md` for exact membership, hashes, network accounting, cleanup, and conditional restart commands.


## 2026-09-06: Batch 003 continuation complete

**BATCH_003_RESUME_COMPLETE.** Both previously unresolved members, **sub-010073 and sub-010074**, completed technically, passed frozen Train QC and provisional 5/1/2 feasibility, and had raw cleanup verified. Batch 003 now has **33/33 complete outputs and 528 blocks**; its ledger is `complete`, with no unresolved technical failure or assessed QC/feasibility failure.

The continuation reused the verified **402,653,184-byte (384-MiB)** sub-010073 prefix through HTTP 206 with exact Content-Range, then verified the complete source SHA256. New response-body transfer was **889,641,134 bytes (0.828542871 GiB)**, with **0 new retries and 0 new interruptions**. Sub-010074 downloaded fresh. Only these two subjects received new requests or preprocessing; all 280 pre-existing output/receipt artifacts and old journal prefixes remain unchanged.

The frozen Batch 003 cap remains **20 GiB = 21,474,836,480 bytes**. Membership, order, manifest, plan, original authorization, and immutable ledger fields are unchanged. Total Batch 003 declared and measured response payload now both equal **21,299,575,247 bytes (19.836775257 GiB)**, including the prefix received during the prior attempt. Unused membership capacity remains **175,261,233 bytes (0.163224743 GiB)**. Earlier six retries/eight interruption events remain historical evidence; the numbered continuation did not add any.

The two new guarded deletions freed **1,292,382,143 bytes (1.203624665 GiB)**, including sidecars. New all-block HDF5 addition is **14,544,040 bytes (13.870277 MiB)**. Batch 003 has no retained raw or partials. Prior intentionally retained raw for sub-010012, sub-010030, and sub-010250 remains unchanged.

Cumulative progress is **72/211 complete subjects / 1,152 blocks**, **69 Train-QC passes**, **68 provisional-feasibility passes**, and **139 unassigned candidates** beginning with sub-010075. Current retained raw is **1,945,164,030 bytes (1.811575172 GiB)**; total processed storage including Train references and receipts is **856,800,945 bytes (0.797958062 GiB)**. No final cohort or split is locked.

Final verification passed for all 33 Batch 003 subjects; **291 tests passed in 33.03 seconds**, and compileall passed. Signal/temp/cache/runtime storage remains external on D:, with the 3-GiB C: guard active. Frozen preprocessing and mechanical-access hashes are unchanged. **Batch 004 was not planned. No scientific held-out analysis, normalization fit, or Git push occurred.**

See `BATCH_003_RESUME_HANDOFF.md` for final continuation evidence. Earlier partial-state entries are historical and are superseded by this completed-state entry.


## 2026-09-06: Batch 004 authorization before network I/O

The user authorizes **Batch 004 only**, with an immutable maximum **45 GiB = 48,318,382,080 bytes** of complete declared `.eeg` plus `.vhdr` payload. Batch 003 resumed successfully and is 33/33 complete. Before changes, the clean starting HEAD `157f4ba1fdda3375d426f8fc87a7f66f5db29941` and all **72 all-block subjects / 1,152 blocks** were reverified; 69 pass frozen Train QC and 68 pass provisional 5/1/2 feasibility. The 381-artifact starting baseline protects all prior batches, outputs/receipts, retained raw, scientific source/configuration, and prior access rows.

Batch 004 freezes **74 subjects / 148 assets**, starting at the verified first remaining candidate **sub-010075** and ending at **sub-010241**, in contiguous authoritative manifest order. Declared payload is **48,277,992,566 bytes (44.962384334 GiB)**, leaving **40,389,514 bytes (0.037615666 GiB)** unused. The next candidate, **sub-010242**, requires 662,012,959 bytes and does not fit. It remains unassigned. Exact ordered membership and per-subject bytes are in `outputs/metrics/scenario2/batches/batch_004_plan.json`. No bin packing, replacement, duplicate assignment, or earlier-subject reprocessing is permitted.

The 45-GiB cap is a **network membership budget**, not simultaneous disk usage. Preflight reserves the maximum remaining payload for one subject plus 512 MiB preprocessing, 32 MiB compact output, and 1 GiB safety margin. Already retained raw is reported in the total peak footprint and is already deducted from free disk; it is not charged twice. The dynamic per-subject D: guard remains active as retained failures accumulate. No full-batch staging is allowed.

Execution proceeds strictly one subject at a time, with a verified first-subject canary: download/Range resume, source and sidecar/header checks, Train-only fitting, unchanged mechanical Validation/Test application, all 16 blocks/masks/reasons, reopened HDF5 and durable receipt/journal, then immediate guarded raw cleanup. Frozen Train-QC failure retains raw; held-out feasibility failure alone does not block technically safe cleanup. Prior retained subjects sub-010012, sub-010030, and sub-010250 remain untouched. Infrastructure defects stop for tested, committed correction and numbered resume authorization under unchanged membership.

Detailed journals remain external on D:. A durable progress record is saved after every subject, with numbered snapshots after each ten terminal subjects and at final stop. It records measured response bytes, exact raw bytes freed/retained, compact output, counts, journal positions, free C:/D: space, and manifest hash. TEMP/TMP/TMPDIR and every cache remain beneath the external Scenario 2 D: root. The 3-GiB C: guard remains mandatory; pagefile settings are unchanged.

Offline synthetic/mocked tests, compileall, and the local code/docs/plan commit precede durable execution authorization and the first request. Scientific preprocessing and QC hashes remain unchanged. Scientific held-out outcomes remain sealed; normalization remains unfitted; the final cohort and split remain unlocked. **Batch 005 is prohibited and unplanned; its size remains a future user decision. No Git push is authorized.**


## 2026-09-06: Batch 004 terminal accounting and stop boundary

**BATCH_004_FAILED_OR_PARTIAL.** Network execution is stopped after **74/74 frozen members reached durable terminal states**. **12 completed technically (192 blocks); 62 remain unresolved download failures**, with QC and provisional feasibility unassessed. The ledger is `failed_or_partial`. This does not claim all 74 were processed.

The immutable **45-GiB cap is 48,318,382,080 bytes**. Frozen membership contains **74 subjects / 148 assets** in contiguous authoritative order, beginning with sub-010075 and ending with sub-010241. Declared and initially expected payload: **48,277,992,566 bytes (44.962384334 GiB)**. Unused membership capacity: **40,389,514 bytes (0.037615666 GiB)**. The first unassigned candidate remains sub-010242 (662,012,959 bytes); no replacement or Batch 005 membership was generated.

Measured response-body transfer is **8,318,270,212 bytes (7.746992830 GiB)**: 7,789,787,908 bytes for the 12 completed source pairs plus a preserved **528,482,304-byte (504-MiB / 0.4921875-GiB)** sub-010232 prefix. There were **186 retries and 248 interruption events** (247 DNS failures and one read timeout), **zero manual restarts**, and **zero validated resumed-prefix bytes**. Connectivity briefly recovered for sub-010232, then timed out; its final Range retry failed DNS before any HTTP 206 response. No new request followed the final terminal sequence.

All 12 completed subjects pass frozen Train QC and provisional 5/1/2 feasibility. Their guarded raw deletions freed **7,790,315,121 bytes (7.255296336 GiB)**. The 62 unresolved raw directories remain: sub-010232 contains the partial and 61 are empty. No download failure is described as a QC failure. The three prior retained QC-failure subjects sub-010012, sub-010030, and sub-010250 are unchanged.

All-block HDF5 addition is **87,664,489 bytes (83.603372 MiB)**. Cumulative progress is **84/211 complete subjects / 1,344 blocks**, with **81 Train-QC passes**, **80 provisional-feasibility passes**, **65 unassigned candidates**, and **62 assigned unresolved Batch 004 members**. Total retained raw is **2,473,646,334 bytes (2.303762672 GiB)**; processed storage including Train references and receipts is **999,601,328 bytes (0.930951282 GiB)**.

The one-subject disk guard remains active; the 45-GiB network budget never became a simultaneous disk requirement. All 381 protected prior artifacts, original access records, scientific source/configuration, and the metadata clone remain unchanged. Each terminal subject has durable progress; snapshots exist after 10/20/30/40/50/60/70 terminals and final stop. Final tests: **299 passed in 38.11 seconds**; compileall and the read-only terminal-state verifier passed. Scientific held-out outcomes remain sealed, normalization remains unfitted, and the final cohort/split remain unlocked. **Batch 005 is unplanned. No Git push occurred.**

See `BATCH_004_STREAMING_HANDOFF.md` for exact membership, results, preservation hashes, and conditional continuation instructions.

## 2026-09-06: Persistent Batch 004 network recovery authorized

The user explicitly authorizes continuation of exactly the 62 unresolved members of the existing Batch 004, starting at sub-010090 in frozen manifest order, without another prompt after temporary connectivity loss. This supersedes the earlier stopped-state instruction; historical failures remain audit evidence. All 84 existing completions, the three prior retained raw subjects, and the 528,482,304-byte sub-010232 partial were reverified at clean HEAD cd06b3671569346deb1945fb208fb659a1ad6373 before changes.

The engineering-only network policy waits indefinitely on transient DNS/TCP/TLS/HTTP conditions, keeps the same subject/asset active, preserves partials, confirms recovery with two tiny probes, and resumes strict validated Range requests automatically. A detached single-instance supervisor, external D: heartbeat/logs, progress watchdog, and explicit graceful stop support operation independent of another conversational prompt. Hard identity/filesystem/disk/provenance/invariant failures stop without replacements. Tests and a local commit precede the numbered resume authorization and real requests. See BATCH_004_AUTONOMOUS_RUNNER.md for implementation and operating details.

The exact 45-GiB cap, 74-member manifest/plan, original authorization, preprocessing/access hashes, Train-only fitting and held-out apply-only rules are unchanged. Original progress archives and journal prefixes are preserved. No Batch 005, scientific held-out analysis, normalization, final cohort/split lock, service, scheduled task, or Git push is authorized. Technical success requires 74 complete outputs and 1,184 verified blocks; final acceptance also requires tests, verifier, preservation checks, handoff, and clean Git.

## 2026-09-06: Batch 004 autonomous continuation complete

All 62 previously unresolved Batch 004 members completed in their original frozen order under implementation 0682fa0c065916cdb01bfa5115333ba5cf0e0084 and numbered authorization 001. Batch 004 is now 74/74 technically complete with 1,184 verified block groups, zero unresolved subjects/partials, and a complete ledger. The 12 existing Batch 004 completions and 72 earlier completions, prior raw, original authorizations, journal prefixes, and archived progress remain unchanged. Batch membership/cap and scientific hashes remain frozen.

The 528,482,304-byte sub-010232 prefix resumed via exact HTTP 206/Content-Range and full original size/SHA256 verification. The continuation received 39,959,722,354 body bytes, with no new outage episodes, reconnections, or extra retry bodies. Historical retries/interruptions are preserved. There was one continuous scientific continuation and no processing restart. A Windows venv PID-reporting correction was tested and committed only after processing; two empty-unresolved verification checks added no download or processing work. The first check's liveness-probe incident and the successful repeated supervision check are documented in the final handoff.

Frozen outcomes: 71 Batch 004 Train-QC passes, 68 provisional-feasibility passes; raw deleted for 71 subjects and retained for sub-010146, sub-010163, and sub-010234. Cumulative totals are 146 subjects/2,336 blocks, 140 Train-QC passes, 136 provisional-feasibility passes, and 65 unassigned candidates. Final 355 offline tests, compileall, full read-only verification, and preservation/accounting checks passed. No scientific held-out analysis, normalization fit, final cohort/split lock, Batch 005, or push occurred. Exact evidence, final commit resolution, byte accounting, and all 62 outcomes are in BATCH_004_AUTONOMOUS_RESUME_HANDOFF.md and its external final acceptance record.

## 2026-09-06: Final remainder Batch 005 authorization before network I/O

The user explicitly authorizes planning, freezing and executing all 65 remaining structurally eligible subjects as final Batch 005. The clean starting HEAD db42b774d70d2ad058a7c02c03e9a99b0fc1f9cf and full Batch 004 verifier passed before changes. All 146 existing outputs/receipts, prior evidence and access registries were reverified; the six intentionally retained QC-failure raw subjects remain untouched. The 770-file starting baseline and coverage audit bind preservation to this assignment.

The authoritative remainder reconciles exactly to 65 subjects / 130 assets, beginning with sub-010242 and ending with the final candidate sub-010321, with 42,976,220,335 declared bytes. The cap equals the exact computed sum and unused capacity is zero. All 211 candidates occur exactly once across previous completions and final membership, leaving zero unassigned candidates. Batch 006 is prohibited. This separate user authorization supersedes the earlier Batch 005 stop boundary without changing historical records.

The persistent supervisor is generalized to Batch 005 while preserving the scientific pipeline, subject-at-a-time disk usage, D-only environment, 3-GiB C: guard, strict Range/source verification, unlimited same-asset network waiting and graceful pause. The new Retry-After handling honors every transient HTTP response. A dynamic staging check rejects another unresolved raw subject. Tests and a clean local implementation commit precede durable Batch-005-only authorization and network I/O. Completion requires 211 verified outputs and 3,376 blocks, final tests/verifiers, preservation audit, handoff and clean local results commit.

Frozen preprocessing and mechanical access, Train-only fitting, held-out apply-only processing, all 16 blocks/masks/reasons, and guarded cleanup remain unchanged. Train-QC failures retain raw; held-out provisional infeasibility alone does not block cleanup. No scientific held-out analysis, normalization fit, final cohort/split lock, pagefile change, service, scheduled task, or Git push is authorized. See BATCH_005_FINAL_REMAINDER_RUNNER.md.

## 2026-09-07: Accept final streaming evidence in a separate terminal phase

Batch 005 processing completed successfully at execution HEAD `3a4965644edbfb16db007275fa9b1a0e232c3900`: 65/65 terminal and technically complete subjects, worker exit 0, empty stderr, zero technical failures, and ledger complete. Independent rediscovery confirmed exited supervisor/worker processes and free execution locks before any documentation change. No worker was launched or restarted for acceptance.

The initial acceptance invocation used a fresh D: pytest directory to preserve prior fixtures. The exact-path guard rejected it before collection; the failure JSON remains unchanged as history. The user directed resumption with the documented exact `temp/pytest` path. Existing fixtures and the old report were archived externally before reuse. The required command passed 383 tests in 47.11 seconds; compileall and both read-only verification stages passed without test, guard, implementation, or scientific-rule changes.

Accept technical completion separately from scientific eligibility: all 211 structural candidates have verified outputs and 3,376 blocks; 205 pass Train QC and 198 pass provisional feasibility. Six Train-QC failures retain 3,902,615,061 raw bytes. The 13 provisional-feasibility failures remain recorded under unchanged rules. Batch 005's three feasibility failures are sub-010260, sub-010291, and sub-010316, all Train-QC passing and technically complete. No subject replacement, block reassignment, threshold adjustment, or final membership selection occurred.

Commit only compact execution/results evidence and final documentation locally, without amend or push. Preserve the 770 prior protected files and external execution journals, snapshots, locks, logs, raw data, and prior handoffs. Materialize repository `HANDOFF.md` now because successful terminal state and exited processes make it safe; keep the external live-run handoff unchanged. The compact completion audit retains `FULL_COHORT_STREAMING_AWAITING_FINAL_ACCEPTANCE` by design. The post-commit `--require-clean` verifier and external `final_acceptance.json` establish the exact accepted HEAD and `FULL_COHORT_STREAMING_COMPLETE`, avoiding a self-referential commit hash in tracked documentation.

See [FULL_COHORT_STREAMING_HANDOFF.md](FULL_COHORT_STREAMING_HANDOFF.md) for counts, bytes, hashes, validation evidence, and the separate Final Cohort and Split Lock boundary. Batch 006 remains prohibited; scientific held-out analysis, normalization fitting, final cohort/split lock, and remote push have not occurred in this phase.
