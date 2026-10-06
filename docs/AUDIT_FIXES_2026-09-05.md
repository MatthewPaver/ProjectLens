# Audit fixes — 5 September 2026

Scope: finish and verify existing local audit improvements without commits, pushes, deployment, private inputs or live model calls. Pre-existing dirty RAG, workflow, README and artwork changes were preserved. Browser screenshots now go to `/tmp/projectlens-browser` rather than overwriting those assets.

## Completed

- Verified the existing explicit opt-in sidecar default. Strengthened browser coverage to require zero requests to port 8787; removed the previous request-failure exception. Offline/static cards remain labelled and the human Use/Ignore gate remains mandatory.
- Separated `requirements-browser.txt` and `install-browser`; `make browser-test` no longer installs pandas, forecasting or RAG packages. Basic local use remains Python's standard-library HTTP server only.
- Added `PARSER_COMPATIBILITY.md` and browser format checks for both XER parsers: UTF-8 BOM, LF/CRLF/CR, reordered fields, empty/invalid/missing-table input, plus basic/empty/unrelated supplementary CSV evidence.
- Fixed a scope error: simplified change assurance previously combined TASK rows from multi-project exports while using only the first project heading. It now rejects these files with a single-project export instruction. Detailed review's first-project scope remains explicitly documented.

## Exact verification

- `.venv/bin/python -m pytest Processing/tests -q`: **36 passed**, four existing dependency/dtype deprecation warnings; no failures.
- `.venv/bin/python scripts/run_browser_tests.py`: **all five suites passed** — public explorer smoke; board readiness; change assurance; detailed XER including real browser file inputs for synthetic XER/CSV/Markdown; new parser compatibility. Each parser read 22 synthetic tasks across four BOM/line-ending variants; reordered columns and three invalid-input cases passed. The detailed CSV cases and simplified multi-project rejection passed.
- `make -n browser-test`: dependency path contains venv → `requirements-browser.txt` → Chromium → browser runner only.
- Independent clean browser-only verification by the coordinating agent: new Python 3.11 environment `/tmp/projectlens-browser-clean-20260905`, wheel-only `requirements-browser.txt` installation (Playwright 1.62.0, pyee 13, greenlet 3.5.5; no optional application extras), then the full browser runner: **all five suites passed, exit 0**. Chromium binaries were reused from the existing browser cache; this verifies clean Python dependency setup, not a fresh browser-binary download.
- `git diff --check`: passed.
- RED/GREEN: the multi-project regression initially failed with “Change assurance must reject multi-project scope explicitly”; it passed after the parser guard. Test-fixture escaping and omitted trailing-field padding were corrected before interpreting results as product regressions.

## Remaining limits

No real, anonymized P6 export was supplied. Synthetic format variants are not independent evidence of broad real-export compatibility; obtaining permitted real fixtures remains blocked on source data/permission. No live Gemini/LangSmith sidecar, hosted service, credentials, or multi-user behavior was verified. No public release was made, so local fixes must not be described as deployed functionality.
