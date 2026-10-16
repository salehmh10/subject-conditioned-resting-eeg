# Verification record

Executed on Windows with Python 3.12.3 on 2026-10-10. Exact installed packages are recorded in `requirements-tested.txt`. Test output is retained in this directory; these are executed checks, not planned checks.

## Automated checks

- `python -m pytest -q`: **68 passed in 6.78 seconds**. Coverage includes actual source schemas, missing files/duplicate keys and rows, coverage of all 55 fits and 33 summaries per split, exclusion of historical utility runs, saved population SD, fraction/percent/percentage-point conversion, contrast direction and CI reversal, source-conflict detection, safe ZIP paths, all 18 numerical anchors, Train-only preview guards, deterministic selection, bounds/units, prepared preview hashes, saved PSD/alpha transforms, signed/missing connectivity, uniform fingerprint references, loopback port selection, runtime import isolation, and PSD display limits.
- `tools/check_ui.py`: **73 checks, zero exceptions**. All seven pages, representative changes to every available selector/checkbox, view reset, every chart HTML-preparation button, ten presentation steps and Previous navigation. A separate interface workstream also tested selection persistence across navigation.
- `tools/check_browser.py`: **29 checks, zero failures** in installed Microsoft Edge through Playwright. Seven rendered pages, ten guided steps, Previous navigation, ten offline HTML slides and the downloaded chart opened offline. Actual CSV and interactive HTML downloads were saved and inspected. No horizontal document overflow, page exceptions or JavaScript errors were reported.
- The aggregate ZIP was extracted into an app-owned temporary directory and exercised in a fresh Python process: **seven pages and ten guided steps passed**. Archive validation checks member paths, forbidden data/model/font/environment files, personal filesystem paths, removal of participant arrays, and preservation of 18 aggregate connectivity diagnostic rows. A path containing spaces is used; launching without its own environment produces the expected actionable prerequisite message.

## Performance measurements

The latest complete AppTest sweep measured 2.829 seconds for initial load, a median of 0.081 seconds for subsequent checks, and a maximum of 0.235 seconds for those cached checks. Browser navigation settles in approximately 0.4–0.9 seconds on this laptop, including a deliberate 0.35-second rendering allowance in the test harness. These are local observations, not speed guarantees. Detailed timings are in `ui-checks.json` and `browser-checks.json`.

The final cached preparation reverified all 468 scientific source hashes. A previously measured cached run took 1.372 seconds; the deterministic rebuild took 13.60 seconds. The compact JSON bundle is 53.38 MiB; optional precomputed previews add 13.08 MB. Runtime caches immutable prepared objects rather than discovering and rereading participant sources per interaction.

## Visual inspection and repair

Fourteen real browser screenshots are saved in `screenshots/`, at 1366×768 and 1920×1080, plus one full-page connectivity capture. Manual inspection covered the overview, PSD, connectivity, utility and offline snapshot. Inspection found and repaired a PSD legend/subtitle overlap, content under the sticky header, and a PSD axis initially influenced by hidden stopband values. All saved scientific points remain unchanged; the initial 1–40 Hz display now scales using values within that band.

The browser harness initially needed fixes to target Streamlit's visible radio labels and visible tab controls, and to wait for the new page title. Those failed harness runs are not counted as passing tests. Final results and screenshots came from the corrected harness and a fresh server after source changes.

## Integrity

`source_manifest.json` records 468 consumed numeric/method sources with SHA256 values. All were checked again after implementation and remained unchanged. The bounded exporter independently verified its frozen source signatures before and after waveform slicing; repeated exports produced identical NPZ bytes. `SCIENTIFIC_QA.md` records the additional method-source audit.

The final repository status outside `apps/` exactly matches `git-status-before.txt`; no tracked repository edits occurred. Existing untracked scientific artifacts were preserved. Environment, pip cache, Python cache settings, temporary directories, tests, screenshots, exports and documentation were directed inside `apps/eeg_synthetic_lab/`. No experiment module was imported or executed, and no training, classifier inference, new EEG generation or held-out waveform extraction was performed.

## Remaining scope limits

No required numerical page uses invented or placeholder research data. Official PSD controls reflect the prepared all-channel averages; channel-specific PSD is available for the bounded preview only. Shared ZIP exports intentionally omit participant arrays and the full waveform preview pool. The standalone HTML is a representative snapshot, with one documented single-channel Train example, not a replacement for every application control. Scientific limitations remain visible in the interface and the methods audit.
