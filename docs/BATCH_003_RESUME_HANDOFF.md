# Batch 003 continuation

The user explicitly authorizes continuation of the existing partial Batch 003 for **sub-010073 and sub-010074 only**. Starting clean HEAD is `ac58dba2c717af8877a2c4e85b9743592ebb2dab`. The original partial-state handoff remains historical evidence in `BATCH_003_STREAMING_HANDOFF.md`.

Before payload I/O, the read-only verifier passed for all 31 completed Batch 003 subjects and all prior completions. The manifest, plan, original authorization, scientific hashes, and earlier ledger entries remain unchanged. The external `runtime/batches/batch_003_resume_starting_state.json` records 280 existing output/receipt hashes, original journals and ledger, the verified partial, and the D-only environment.

The sub-010073 partial is exactly 402,653,184 bytes with SHA256 `72df8eb5246c8bff83eb2c576427faaf850e05870a374bb529b203e87cf3f233`. Sub-010074 has only an empty staging directory. Expected remaining response payload is 889,641,134 bytes if the server accepts Range. Whole-source size/SHA256 verification remains mandatory before promotion.

DNS and a zero-body HTTP HEAD check passed at `2026-09-05T20:34:58.924886+00:00`; the configured NEMAR host returned HTTP 200, the expected 625,542,800-byte full EEG length, and `Accept-Ranges: bytes`. Compact evidence is `outputs/metrics/scenario2/batches/batch_003_resume_connectivity_preflight.json`.

The resume authorization now explicitly names unresolved members and reports their count and IDs in its remaining-transfer preflight. This is an authorization-reporting correction only; frozen membership and scientific processing are unchanged. Full tests and compileall, followed by a local commit, precede numbered authorization and payload I/O. Current state: pre-resume verification; technical closure is not yet claimed.

Execution must select only the two unresolved subjects, preserve the existing prefix, require strict HTTP 206/Content-Range before appending, process Train first and mechanically apply to held-out, verify all 16 blocks and receipts, then use guarded cleanup only if Train QC and technical safeguards pass. Any new failure must preserve raw/partials and retain `failed_or_partial`.

No Batch 004 planning, replacement, scientific held-out analysis, normalization fit, final cohort/split lock, amend, or push is authorized.
