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

3. **LLM evaluation** — `/Users/mattpaver/Desktop/Repos/newco-vault/Method/skills/llm-evaluation/SKILL.md`  
   Judge retrieval + brief quality: faithfulness to cited cases, context precision, failure modes when matches are weak, and why LangSmith traces ≠ evals.

Also skim: `docs/precedent-rag.md`, prior critique at `/tmp/pl-review/CLAUDE-CRITIQUE.md` (verify claimed P0/P1/P2 fixes still hold on the live build).

## Surfaces to exercise (in order)

### A — Public GitHub Pages (static path)
1. Open https://matthewpaver.github.io/ProjectLens/change-assurance.html  
2. Hard-refresh. Run **Northstar** demo end to end: intake → readiness → precedents → Use/Ignore → decision save.  
3. Confirm the UI is honest that Pages has **no Gemini sidecar** (static fallback cards + clear status copy).  
4. Confirm XER math still feels local / deterministic (no fake “AI approved” language).

### B — Local full RAG path (required for LLM critique)
From `/Users/mattpaver/Desktop/Repos/ProjectLens`:
```bash
make install-rag
# ensure .env has GEMINI_API_KEY (+ optional LangSmith)
make precedent-rag          # :8787
# separate terminal: serve docs/ on :8765 (python -m http.server 8765 --directory docs)
```
Then open http://127.0.0.1:8765/change-assurance.html and repeat the Northstar flow with **live** retrieve + Gemini brief.

### C — Portfolio framing
Open https://matthewpaver.github.io/preview.html?app=projectlens and check whether the store copy matches what the product actually proves (RAG stack, Pages vs local, human gate).

## What to write back

Save a new critique to `/tmp/pl-review/CLAUDE-CRITIQUE-E2E.md` with:

1. **Verdict** (2–4 sentences) — concept vs execution.  
2. **Product / workflow** — gate clarity, Use/Ignore consequence, BYO XER path, Pages honesty.  
3. **RAG / LLM quality** — citations-only brief, score labelling, weak-match honesty, eval gaps (gold set / faithfulness / non-Gemini judge).  
4. **Impeccable + Taste audit** — BAN violations, type contrast, sticky/clip bugs, token rot, what to keep.  
5. **Priority fixes** — P0 / P1 / P2 only. No drive-by redesign.  
6. **Interview signal** — what this project actually showcases about Matthew (RAG/LangGraph/LangSmith vs thin AI veneer).

Take screenshots of the live local RAG run if you can. Be blunt. Prefer evidence over taste opinions.
