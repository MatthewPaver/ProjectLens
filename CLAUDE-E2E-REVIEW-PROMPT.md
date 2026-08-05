# Claude prompt — ProjectLens end-to-end review

Copy everything below the line into Claude Code (or Claude) from the ProjectLens repo root.

---

You are doing a full end-to-end product + design + RAG quality review of **ProjectLens change assurance**. Do not redesign from scratch. Critique what ships, then propose a tight P0/P1/P2 list.

## Mandatory skills (read and follow before judging UI)

1. **Impeccable** — `/Users/mattpaver/Desktop/Repos/newco-vault/Method/skills/impeccable/SKILL.md`  
   - Prefer `audit` posture against the absolute bans and AI-slop test.  
   - Design context is already in `/Users/mattpaver/Desktop/Repos/ProjectLens/.impeccable.md` — read it; do **not** re-run teach unless context is wrong.

2. **Taste** — there is no separate “taste” skill on disk. Use **frontend-design** as Taste:  
   `/Users/mattpaver/.claude/skills/frontend-design/SKILL.md`  
   Judge distinctiveness, hierarchy, contrast, motion restraint, and whether the interface would pass the “AI made this” test.  
   **Constraint:** frontend-design is a *generator* skill. Use its criteria as an audit checklist only — do not propose replacement designs, and do not re-file a finding Impeccable already raised.

3. **LLM evaluation** — `/Users/mattpaver/Desktop/Repos/newco-vault/Method/skills/llm-evaluation/SKILL.md`  
   Judge retrieval + brief quality: faithfulness to cited cases, context precision, failure modes when matches are weak, and why LangSmith traces ≠ evals.

Also skim: `docs/precedent-rag.md`, prior critique at `/tmp/pl-review/CLAUDE-CRITIQUE.md` (verify claimed P0/P1/P2 fixes still hold on the live build).

**Before section A, record the local commit:** `git rev-parse --short HEAD`. GitHub Pages can lag local HEAD, so a fix committed but undeployed will read as unfixed. Either confirm the deployed page carries the same marker, or state the deploy lag explicitly as a caveat on every section-A finding.

## Surfaces to exercise (in order)

### A — Public GitHub Pages (static path)
1. Open https://matthewpaver.github.io/ProjectLens/change-assurance.html?v=<any-timestamp> — the `?v=` cache-bust matters, the Pages CDN holds ~10 min and a hard-refresh alone does not beat it.  
2. Run **Northstar** demo end to end: intake → readiness → precedents → Use/Ignore → decision save.  
3. Confirm the UI is honest that Pages has **no Gemini sidecar** (static fallback cards + clear status copy).  
4. Confirm XER math still feels local / deterministic (no fake “AI approved” language).

### B — Local full RAG path (required for LLM critique)

**Precondition — check this first.** `grep GEMINI_API_KEY .env` must show a non-empty value. As of this writing it is blank. If it is still blank: **STOP section B, say so in the critique, and do not review the offline fallback as if it were live retrieval.** The fallback path is a hardcoded card set (`docs/change-assurance.js:18`) — judging it as Gemini output is the single easiest way to produce a wrong review.

From `/Users/mattpaver/Desktop/Repos/ProjectLens`:
```bash
make install-rag
make precedent-rag          # :8787
# separate terminal: serve docs/ on :8765 (python -m http.server 8765 --directory docs)
curl -s http://127.0.0.1:8787/health    # must return ok before you trust anything below
```
Port 8765 is not arbitrary — it is in the sidecar CORS allowlist (`Processing/precedent_rag/server.py`). Serving docs/ on any other port will fail cross-origin.

Then open http://127.0.0.1:8765/change-assurance.html and repeat the Northstar flow with **live** retrieve + Gemini brief.

**An eval harness already exists — critique it, do not report it missing.**
```bash
make precedent-eval
# gold set: Processing/precedent_rag/data/eval_queries.json
```
The real questions are whether that gold set is large enough to mean anything, whether it measures faithfulness rather than just retrieval hits, and whether the judge is non-Gemini (self-judging is not evaluation).

Screenshots: use `make browser-test` / `scripts/run_browser_tests.py` rather than improvising a capture path.

### C — Portfolio framing
Open https://matthewpaver.github.io/preview.html?app=projectlens and check whether the store copy matches what the product actually proves (RAG stack, Pages vs local, human gate).

## What to write back

Save a new critique to `/tmp/pl-review/CLAUDE-CRITIQUE-E2E.md` with:

1. **Verdict** (2–4 sentences) — concept vs execution.  
2. **Product / workflow** — gate clarity, Use/Ignore consequence, BYO XER path, Pages honesty.  
3. **RAG / LLM quality** — citations-only brief, score labelling, weak-match honesty, eval gaps (gold set / faithfulness / non-Gemini judge).  
4. **Impeccable + Taste audit** — BAN violations, type contrast, sticky/clip bugs, token rot, what to keep.  
5. **Priority fixes** — P0 / P1 / P2 only. No drive-by redesign. Use these definitions, and respect the caps — the cap forces you to rank rather than list:
   - **P0** (max 3) — ships broken, or the interface claims something that is not true.
   - **P1** (max 5) — damages credibility with a technically literate reader.
   - **P2** (max 5) — polish.
6. **Interview signal** — what this project actually showcases about Matthew (RAG/LangGraph/LangSmith vs thin AI veneer). State the bar plainly: would a staff-level AI engineer read this as evidence of real retrieval work, or as a wrapper?

Take screenshots of the live local RAG run. Be blunt. Prefer evidence over taste opinions — cite a file path and line number, a screenshot, or a command's output for every claim.
