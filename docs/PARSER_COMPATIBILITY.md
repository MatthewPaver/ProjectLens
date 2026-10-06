# Browser parser compatibility contract

This contract describes the checked-in browser parsers, not certification of Primavera P6 export compatibility. No real client XER export was supplied or used for the audit fix. Northstar and Riverside are explicitly synthetic, share-safe examples. Adding anonymized real variants remains an external-evidence task requiring permission and provenance; do not relabel synthetic fixtures as real exports.

## Try a supported input

Run `python3 -m http.server 8000 --bind 127.0.0.1 --directory docs` and open `/change-assurance.html`. Select **Review my change pack**, supply `docs/demo/riverside-previous.xer` and `docs/demo/riverside-current.xer`, and use the accompanying narrative. For the detailed workflow, open `/schedule-review.html`, supply the Northstar previous/current files and optionally the baseline, risks CSV, decisions CSV and schedule-basis Markdown. No account, API, package installation or upload is required.

## Format and scope

- Text, tab-delimited `%T` table names, `%F` field headers and `%R` records. UTF-8 BOM and LF, CRLF or CR line endings are covered. Columns are mapped by field name; column reordering is covered. Unknown tables/fields are ignored unless referenced by a check.
- Both workflows require readable PROJECT and TASK rows. Empty text and missing required tables are rejected. Change assurance requires exactly one project; multiple projects produce an explicit corrective message. Detailed review selects the first PROJECT, filters its TASK rows by `proj_id`, preserves relationships touching its activities, and reports multiple-project scope.
- Task identity: `task_code`, with `task_id` fallback. Project name: `proj_short_name` then `proj_name`. Dates: ISO-style date prefix preferred, with browser Date parsing fallback. Ambiguous locale dates are not a portable supported contract.
- Finish: `early_end_date`, `target_end_date`, then `act_end_date`. Float: `total_float_hr_cnt`. Constraint: `constraint_type`/`cstr_type`; date `cstr_date`/`constraint_date`. TASKPRED compares predecessor/successor/type/lag signatures. These are observed export changes, not schedule recalculation.
- Detailed review additionally uses WBS, calendars, durations, statuses and the explicit risk/decision evidence files. Its CSV evidence reader is a lightweight line/code matcher, not a general RFC-4180 tabular importer; use the published simple one-line-per-record templates. Activity links require explicit codes such as `NS-900`. Missing/empty supplementary evidence stays missing; no owner, outcome or mitigation is inferred.
- Unsupported: MSP, PDF schedules, P6 scheduling-engine reproduction, arbitrary legacy encodings, proof of causation/entitlement, and reliable extraction of embedded baselines. Supply a separate baseline.

## Verification

`make browser-test` now installs only `requirements-browser.txt` and Chromium; it does not install forecasting or RAG packages. The suite exercises first-use inputs, demo blockers, CSV/Markdown supplementary evidence, exports, desktop/mobile layouts, and the synthetic parser variants above. Full Python/data/RAG tests remain a separate `make test` path.

Unconfigured sidecar mode is deliberately static and sends **zero** requests to port 8787. The browser regression fails on every failed request or console error; it no longer excludes localhost sidecar failures. Explicit sidecar opt-in is separate, and live model/network quality is not established by offline tests.
