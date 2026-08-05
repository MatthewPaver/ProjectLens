# Cited precedent RAG

## What problem this solves

A reviewer preparing a change decision needs **comparable past decisions with sources**, not a chatbot that invents programme advice. ProjectLens keeps Primavera XER checks deterministic in the browser. The RAG sidecar only receives narrative text, blocker titles and soft metadata filters (sector / phase / change type).

## Architecture

```text
change-assurance.js (browser)
  ├─ XER parse + blockers          ← deterministic, local
  └─ POST /precedents/query        ← narrative + filters only
        │
        ▼
Processing/precedent_rag (local FastAPI)
  LangGraph: retrieve → summarise
  ├─ metadata soft-filter
  ├─ Gemini embeddings (hybrid rank + reasons[])
  ├─ Gemini brief (citation-only; invented DG-* ids stripped)
  └─ LangSmith traces (LANGSMITH_PROJECT=projectlens-precedent-rag)
        │
        ▼
Human Use / Ignore on each card → then decision register
```

**GitHub Pages** hosts the static demo only (`docs/` → Pages). There is no Gemini key on that host — the UI falls back to static precedent cards and says so. Live hybrid retrieve needs `make precedent-rag` locally, or set `window.PROJECTLENS_PRECEDENT_RAG_URL` to a CORS-enabled API you control (never bake secrets into the static site).

## Why Gemini + LangSmith (not keyword-only)

Token overlap (DecisionGraph) is fine for a static demo. It does not solve semantic “same failure mode, different words.” Embeddings do. LangSmith is the proof trail for retrieve → summarise runs so you can see whether citations stayed inside the shortlist.

## Eval harness

Offline unit tests (`Processing/tests/test_precedent_rag.py`) cover corpus shape, citation stripping and graph wiring without API keys.

Live retrieval quality:

```bash
make precedent-eval   # gold queries → top-k hit rate (needs GEMINI_API_KEY)
```

Judge criteria for the Gemini brief (manual or future LLM-as-judge, non-Gemini):

1. Every substantive claim cites a retrieved `DG-*` id
2. Brief does not treat the live narrative as established fact
3. Includes one concrete reviewer question grounded in the pack
4. Does not approve/reject the live decision

LangSmith project `projectlens-precedent-rag` is for traces, not a substitute for the gold-query hit rate.

## Non-goals

- Does not replace XER finish / float / constraint checks
- Does not auto-write to the decision register
- Does not claim contractual entitlement or causation
- Does not upload schedule files
