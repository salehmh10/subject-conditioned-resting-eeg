# Subject-Conditioned Resting-State EEG

Scenario 2 studies subject- and eye-state-conditioned generation with MPI-LEMON resting EEG. Scenario 1 remains frozen separately.

## Current status

Batch 002 execution is complete: **16 verified all-block outputs (256 blocks)** under an immutable **10-GiB cap (10,737,418,240 bytes)**. Declared, expected, and measured response-body payload are each **10,141,417,744 bytes (9.444931284 GiB)**; unused capacity is **596,000,496 bytes**. There were no retries, resumptions, interruptions, or technical failures.

Fifteen subjects pass frozen Train QC and provisional 5/1/2 feasibility. **sub-010030** fails both and retains its **645,403,807-byte** raw directory. The other fifteen raw directories were deleted after verification, freeing **9,496,716,931 bytes (8.844506863 GiB)**. Batch all-block HDF5 addition is **116,775,031 bytes (111.365348 MiB)**.

Cumulative progress is **39/211 complete all-block subjects / 624 blocks**, with **172 unassigned candidates**. Cumulative Train-QC pass is 36/39 and provisional feasibility is 35/39. Raw remains only for sub-010012, sub-010030, and sub-010250: **1,945,164,030 bytes (1.811575172 GiB)**. Processed storage including Train references and receipts is **464,293,360 bytes (0.432406887 GiB)**. These engineering labels do not lock the final cohort or split.

Batch 001 records, its ledger entry, all 23 prior outputs/receipts, retained raw, and prior access records remain unchanged. All signal/temp/cache/runtime data stays external on D:, and the original metadata clone remains unchanged. Final tests passed **280 in 33.43 seconds**; compileall and the read-only Batch 002 verifier passed. No scientific held-out analysis, waveform plot, normalization fit, final cohort/split lock, or Git push occurred. **Batch 003 was not planned; its size remains undecided.**

See [BATCH_002_STREAMING_HANDOFF.md](docs/BATCH_002_STREAMING_HANDOFF.md) for the complete handoff. [Batch 001 handoff](docs/BATCH_001_STREAMING_HANDOFF.md) remains historical immutable evidence.

## Verify and next action

Check that no other pytest instance is using the shared external base, then run:

```powershell
. .\scripts\setup_scenario2_environment.ps1 -ForTests
.\.venv\Scripts\python.exe -m pytest -q --basetemp "D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/temp/pytest" --junitxml outputs/logs/pytest.xml
.\.venv\Scripts\python.exe -m compileall scripts src tests
.\.venv\Scripts\python.exe scripts/data/s2_12_verify_streaming_batch.py --batch-id 002 --require-clean --test-report outputs/logs/pytest.xml
```

**NEXT ACTION — CHOOSE BATCH 003 SIZE.** Batch 003 has no chosen cap or planned membership. Do not regenerate completed subjects or run an unbounded full-manifest stream. No further download is authorized by this completed assignment.

## Verified dataset facts

- 215 subject directories and 215 participant rows.
- Every subject has 62 channel rows: 61 EEG and 1 EOG (`VEOG`), not 62 EEG.
- All 19 target scalp channels are present and marked good for all subjects.
- Native sampling rates are 2500 Hz for 208 subjects, 1000 Hz for 6, and 500 Hz for 1.
- `Stimulus/S200` means EO and `Stimulus/S210` means EC.
- `Stimulus/S208` is mapped to EC only for `sub-010078` and `sub-010126`, using documented subject-level evidence.
- Distributed participant-level age, sex, handedness, weight, and height values are all `n/a`; demographic distortion cannot be tested from the local table.

## Structural cohort and split status

The four structural exclusions are:

- `sub-010015`: only 3 EO/3 EC runs; final EO is truncated.
- `sub-010100`: only 5 EO/6 EC runs; final EC is truncated and the sixth EO is absent.
- `sub-010203`: no events table and zero marker entries in its header-only `.vmrk`.
- `sub-010285`: no EO/EC state markers.

The remaining 211 subjects each have eight complete EO and eight complete EC blocks after explicit acquisition-interruption fragments are separated for `sub-010059` and `sub-010081`. Therefore 5 Train / 1 Validation / 2 Test blocks per subject and state is structurally feasible.

[`outputs/metrics/scenario2/provisional_block_split_manifest.csv`](outputs/metrics/scenario2/provisional_block_split_manifest.csv) is explicitly `PROVISIONAL_METADATA_ONLY`; it is not final or locked. Final membership requires waveform QC, a checksum, and a new commit.

## Reproduce the pre-download package

From the repository root:

```powershell
. .\scripts\setup_scenario2_environment.ps1 -ForTests
python scripts/audit/s2_01_audit_metadata.py
python scripts/audit/s2_02_audit_blocks.py
python scripts/audit/s2_03_build_structural_cohort.py
python scripts/audit/s2_04_audit_demographics.py
python scripts/audit/s2_05_plan_splits_and_downloads.py
python -m pytest -q --basetemp "D:/SHARIF/PROJECT/Datasets/MPI-LEMON-Scenario2/temp/pytest"
```

These commands read only local metadata. The prepared downloader is validation-only unless its explicit `--execute` flag is supplied.

## Planned study

The study asks whether subject-conditioned synthetic EEG preserves individual EO-to-EC alpha physiology, spatial alpha structure, and functional connectivity on held-out recording blocks, and whether synthetic augmentation improves held-out real classification across at least three classifiers. Validation and Test remain real-only; complete recording blocks are the atomic split unit.


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
