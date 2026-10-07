# ProjectLens

Browser-local change-assurance tool for project controls reviewers: compares a change pack's narrative with its Primavera P6 XER schedules and evidence, surfaces conflicts as at most three blockers, and records the human board decision. Live on GitHub Pages from `docs/`. See README.md.

## Layout
- `docs/` is the product (static HTML/JS, deployed as-is). `change-assurance.*` (primary workflow), `schedule-review.*` + `xer-review.js`, `board-readiness.*`, `index.html` + `app.js` (GMPP explorer), `demo/` (synthetic XER fixtures).
- `Processing/gmpp_pipeline.py` builds `docs/data/gmpp.json` from `Data/public/raw/` (`make public-data`).
- `Processing/precedent_rag/` optional local FastAPI + LangGraph sidecar (Gemini, LangSmith). Offline eval: `make precedent-eval-offline` (hit@5 = 7/8, pinned by `test_offline_eval_baseline_is_stable`).
- `Processing/analysis/`, `core/`, `output/`, `ingestion/` earlier CSV schedule pipeline (`make pipeline`).
- `Processing/tests/` pytest suite; `browser_*.py` Playwright journeys run by `scripts/run_browser_tests.py`.

## Constraints
- Never fabricate numbers. No random or synthetic values in outputs; missing inputs give NA or an empty result. `test_no_fabricated_outputs.py` AST-guards `Processing/` (only the seeded `decision_support.py` simulation is allowed).
- XER files, decisions and conditions stay in the browser. The static site must not probe localhost or hold API keys; the sidecar receives narrative and filters only.
- Schedule findings are deterministic code, not LLM output. Generated briefs fail closed (`summarize.py: evaluate_summary`) and every precedent needs a human Use/Ignore.
- Every README number must be reproducible by the command shown next to it.
- Internal notes, agent prompts/logs and market research stay out of the tree (see `.gitignore`). UK English in public docs.

## Commands
- `make test` (pytest), `make browser-test` (Playwright), `make public-data`, `make precedent-eval-offline`.
- Local site: `python3 -m http.server 8000 --bind 127.0.0.1 --directory docs`.
- Env (sidecar only, `.env`, never committed): `GEMINI_API_KEY`, `LANGSMITH_API_KEY`.
- CI: `.github/workflows/deploy-pages.yml` runs pytest + browser tests, then deploys `docs/` to Pages.
