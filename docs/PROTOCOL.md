# Scenario 2 Study Protocol

Version: 0.4
Status: Batch 003 complete after authorized two-subject continuation; scientific rules remain frozen

## 1. Objective

The study investigates `p(EEG | subject, eye_state)` in MPI-LEMON resting-state EEG. The primary generative question is whether synthetic EEG preserves individual EO-to-EC changes in alpha physiology, spatial alpha structure, and functional connectivity on recording blocks excluded from generator training. The augmentation question is whether adding synthetic EEG improves real held-out eye-state classification across at least three classifiers.

## 2. Pre-download dataset audit

The metadata clone contains 215 subjects. Each `channels.tsv` has 62 rows comprising 61 EEG and 1 EOG (`VEOG`); all 19 target scalp channels are present and marked good. Native sampling frequencies are 2500 Hz (208 subjects), 1000 Hz (6), and 500 Hz (1). All are structurally compatible with a future 128 Hz target, subject to resampling validation.

The distributed `participants.tsv` contains 215 IDs, but age, sex, handedness, weight, and height are all `n/a`. Those fields cannot be used for subject-level balance or distortion analysis. Dataset README aggregates are not substituted because they are not subject-linked and their stated age-group counts sum to 227 rather than 215.

## 3. Event semantics and block reconstruction

State semantics use `trial_type`, never the numeric `value` column:

- `Stimulus/S200` -> EO
- `Stimulus/S210` -> EC

`Stimulus/S208` is not globally defined. Two evidence-backed subject overrides are frozen before waveform analysis:

- `sub-010078`: six initial S208 markers continue seamlessly at approximately 2 s cadence into 24 S210 markers, making one 30-marker EC run.
- `sub-010126`: 240 S208 and 240 S200 markers form eight alternating approximately 60 s runs per code, with no S210; S208 is the subject-specific EC replacement.

Repeated same-state markers form one state run. A numeric time gap alone never creates a block and never licenses an imputed state. An explicit `New Segment/` marker starts a new acquisition segment and prevents a run from bridging that boundary. `Stimulus/S  1` and comments provide context only; they have no EO/EC semantics.

`sub-010059` and `sub-010081` contain explicit acquisition restarts. Their aborted initial EC fragments are preserved as evidence but excluded from the primary block sequence. Each post-restart main sequence contains eight complete EO and eight complete EC blocks. Unlabeled inter-block and interruption intervals remain unlabeled.

## 4. Pre-download structural cohort

This cohort uses no physiological, predictive, or generative outcome. Eligibility requires:

1. readable channel metadata and all 19 target channels;
2. available EO/EC `trial_type` annotations;
3. eight alternating study blocks per state;
4. eight complete 30-marker, approximately 60 s blocks per state after explicit interruption fragments are separated.

The resulting pre-download structural cohort contains 211 subjects. It is not the final physiological cohort.

Four subjects are excluded structurally:

- `sub-010015`: 3 EO/3 EC runs, with a truncated 27-marker final EO.
- `sub-010100`: 5 EO/6 EC runs, with a truncated 15-marker final EC and no sixth EO.
- `sub-010203`: EO/EC annotations unavailable; no events table and no `Mk*` entries in `.vmrk`.
- `sub-010285`: events contain no state marker.

All 211 retained subjects require waveform integrity, bad-channel, signal-quality, artifact, duration, and usable-window QC after download.

## 5. Block-level split rule

Complete recording blocks are the atomic unit. Random window-level splitting is prohibited. Windows cannot cross block boundaries; filtering and other time-dependent preprocessing must not bridge blocks. Fitted normalization, transforms, generator training, and classifier training use Train data only. Hyperparameter selection uses real Validation only. Final Test remains real-only.

The study is a within-subject block holdout: subject identities occur in all partitions. It is not a held-out-subject generalization design.

## 6. Provisional metadata-only allocation

The 211-subject structural cohort supports 5 Train / 1 Validation / 2 Test blocks per subject and state. A deterministic candidate is stored as `provisional_block_split_manifest.csv` with status `PROVISIONAL_METADATA_ONLY`.

Subjects are ordered by SHA256 of a fixed version seed and subject ID. A fixed 5/1/2 base schedule is rotated by subject rank modulo eight, with a four-position EC offset. This makes every block order appear in every role and balances each state/order cell within one subject of perfect balance. No waveform or outcome enters the allocation.

This candidate is not final or locked. After outcome-blind QC rules are frozen and applied uniformly, the final eligible block set must be generated deterministically, checksummed, committed, and then frozen.

## 7. Pilot analytical-access rule

Downloading a BrainVision `.eeg` provides a subject's whole recording, including provisional Validation/Test blocks. During pilot pipeline development, only complete provisional Train study blocks may be read or inspected. Interrupted fragments are prohibited even for diagnostics. Any complete pilot block exposed during method development must remain Train in the final manifest, or its subject must be excluded. Pilot Validation/Test signals must not influence filters, resampling, QC rules, artifact thresholds, or implementation choices.

Possession on disk is not permission for analytical access. `TrainAccessGate` joins the authoritative split and block inventory, refuses held-out/incomplete/interrupted blocks, and durably logs permitted sample intervals before reading. A cross-process lock allows one subject loader at a time; individual reads are bounded to 65 seconds. Opaque SHA256 byte reads and header metadata inspection are permitted over the whole recording.

## 8. Final-Test lock sequence

Before the final manifest is created:

1. validate BrainVision links and resampling in the Train-only pilot;
2. freeze outcome-blind waveform-QC and block-eligibility rules;
3. apply those rules uniformly to the full candidate cohort;
4. generate, checksum, commit, and lock the final cohort/split manifest.

Before Final Test is opened, additionally freeze preprocessing, normalization, model architectures, spectral-loss weight, checkpoint rule, CFG, classifiers and hyperparameters, augmentation ratio, IAAFT settings, random seeds, metrics, and the statistical analysis plan. Tuning uses Train and real Validation only. Final Test is evaluated once under the prespecified plan.

QC quantities may determine pass/fail eligibility and feasibility under frozen rules. They must not be optimized to construct an unusually clean or easy Validation/Test set. Alpha power, IAF, connectivity, classifier accuracy, generator loss, TSTR, augmentation utility, and any final scientific outcome may never drive inclusion or allocation.

## 9. Planned models and evaluation

Generators:

- B0: eye-state-only DDPM
- B1: subject + eye-state DDPM
- B2: subject + eye-state DDPM with multi-resolution spectral regularization

Classifiers:

- log-bandpower + logistic regression
- covariance/tangent-space + linear SVM
- EEGNetv4

Augmentation ratios are 0.5x, 1.0x, and 2.0x, with an IAAFT spectral-surrogate control. The primary classification metric is subject-macro balanced accuracy. Secondary metrics include accuracy, macro-F1, ROC-AUC, MCC, state recalls, and confusion matrices. Primary comparisons use paired subject-level analysis with subject-level bootstrap confidence intervals. Debiased alpha-band wPLI is the primary connectivity candidate and an evaluation target, not a main-model training loss.

## 10. Amendments

Every scientifically meaningful amendment must be recorded in `docs/DECISIONS.md`, committed, and marked relative to waveform analysis, Validation outcomes, and Final Test. No protocol change may respond to Final Test results.

## 11. Pilot preprocessing specification

The canonical configuration is `configs/preprocessing/frozen_preprocessing_v1.yaml`; its canonical JSON SHA256 is stored beside it. Software versions are pinned in `requirements-lock.txt` and enforced against the config. Values remain in physical units; numerical normalization is not fitted here.

Read a complete Train block using MNE `preload=False`, preserving native order, 61 EEG and VEOG. Assign `standard_1005` template coordinates (individual digitizations are unavailable). Filter each block independently with MNE Hamming/firwin zero-phase FIR, passband 1–40 Hz, transitions 0.5–1 Hz and 40–50 Hz, `reflect_limited` padding. FIR lengths at 500/1000/2500 Hz are 3301/6601/16501 samples. Resample with SciPy polyphase, explicit Kaiser beta 5 FIR of length `20*max(up,down)+1`, line padding, to 128 Hz; output length is the ceiling of native samples times 128/native rate.

The FIR half-support is 3.3 seconds and polyphase half-support is 0.078125 seconds. Round their sum upward to the output grid: trim 433 samples = 3.3828125 seconds from each edge. All subject-level metric calculations use the interior. The same finite-support rule applies to every subject/state. This excludes dependence on boundary padding, not interior artifact ringing. Synthetic 5/20/35 Hz tones test preservation; 60/100 Hz tones test stopband/alias suppression.

After resampling, fit bad channels over pooled EO and EC Train blocks. Per channel, median-across-block broadband robust scale, derivative robust scale, and median absolute correlation with the other EEG channels are used. Scale is `1.4826*MAD`. Log10 scale and derivative have cross-channel robust z-scores with denominator floor 0.1; absolute z > 6 and correlation < 0.1 are votes. Ordinary failure requires low correlation plus an amplitude or derivative outlier; amplitude and derivative alone cannot supply independent corroboration. Any native nonfinite sample, or native robust scale < 1e-6 µV in at least half the Train blocks, is catastrophic. Allow at most six bad EEG and two bad targets. Eligible channels use MNE accurate spherical-spline interpolation at fixed head origin (0,0,0.04 m) before CAR. Nonfinite samples are zero-filled solely to permit this computation; native sanity records preserve the failure. If channel limits fail, retain a labeled diagnostic compact output with all windows rejected; never silently accept it.

Apply CAR over all 61 EEG after interpolation, excluding VEOG. Select the ordered 19 channels from the config and convert volts to µV by 1e6. VEOG receives the same filter/resampling and edge handling, remains unreferenced, and is used only for QC. No ICA, ICLabel, ASR, spectral enhancement, or outcome analysis is included.

After trimming, define 512-sample windows with stride 256 (4/2 seconds), within each block. Features are maximum target peak-to-peak, maximum target robust scale, maximum absolute target sample difference, and VEOG peak-to-peak. Fit one threshold vector from the subject's pooled finite, non-flat Train windows: `min(cap, max(floor, median + 8*1.4826*MAD))`. Floors in µV are [150,30,50,250]; caps [1000,200,300,1500]. A window fails any exceeded threshold, any nonfinite value, any target scale < 0.01 µV, or any target repeated-difference fraction > 0.99. Any native nonfinite VEOG sample rejects every window in that block. Native nonfinite, scale <1e-6 ?V, or repeated-difference fraction >0.99 on an unrepaired channel also rejects the block, so CAR cannot hide local flatness. A block requires at least 10 usable windows and 50% usable fraction. A pilot subject requires at least three usable Train blocks per state and valid channel limits. This is a pilot engineering eligibility rule, not a declaration that the final 5/1/2 allocation is feasible after QC.

If no finite non-flat Train fitting windows exist, use the fixed caps only for diagnostic bookkeeping, mark every window rejected, and fail subject QC; this is not a fitted reference.

Candidate multipliers 6/8/10 are reported for engineering sensitivity; 8 is predeclared, not optimized for physiology or group separation. If the pilot reveals an engineering failure, changes require a documented amendment and regeneration before a freeze is claimed.

Store one HDF5 per subject with continuous trimmed 19-channel float32 µV blocks, gzip level 4, shuffle, chunks (19,1024), candidate starts, usability masks/reasons, and provenance. VEOG and overlapping windows are not stored. A temporary 62-channel float64 cache is disk-backed and removed after processing; only one native block is in memory. Byte hashes verify containers; independent rebuilds must have equal scientific arrays/masks, while creation timestamps can differ.

Final normalization is global channelwise mean and population standard deviation (ddof 0), fitted only after final cohort/split lock from unique continuous time samples covered by usable windows in eligible final Real Train blocks. Count overlapping samples once. Fail fitting if any channel standard deviation is below 1e-6 µV. Apply that same scaler to Real Validation, Real Test, and Synthetic data; no per-state/per-window fit.

## 12. Subject streaming and raw retention

The 211-subject manifest remains the authoritative inventory, but simultaneous full download is superseded by one-subject streaming. Download/resume and verify one source pair, copy validated sidecars, check headers, derive subject parameters from Train only, produce/verify compact output, durably journal completion, then proceed. The all-block extension below now fits from provisional Train only and applies the frozen method mechanically to Validation/Test before final split lock; held-out scientific outcomes remain sealed.

The downloader supports strict HTTP Range resume, safe full-response restart, exact size/SHA256 verification and atomic promotion, bounded retries, and a durable journal. Network tests use mocked responses only. Raw retention is the default. Explicit `--delete-raw-after-success` additionally requires fresh source/output/provenance verification, passing subject QC, a durable complete journal, accessible metadata, and confined paths with no unresolved partials. Pilot raw cleanup is prepared but is not executed in this phase. Train-only outputs are insufficient: all 16 assigned study blocks and apply-only Train parameter provenance must be verified before future deletion.


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
