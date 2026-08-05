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

## Why Gemini + LangSmith (not keyword-only)

Token overlap (DecisionGraph) is fine for a static demo. It does not solve semantic “same failure mode, different words.” Embeddings do. LangSmith is the proof trail for retrieve → summarise runs so you can see whether citations stayed inside the shortlist.

## Eval first

`Processing/precedent_rag/data/eval_queries.json` is the measurement set. Run `make precedent-eval` before trusting rankings. Offline unit tests use a hashing embedder so CI does not need API keys; the live path always prefers Gemini.

## Non-goals

- Does not replace XER finish / float / constraint checks
- Does not auto-write to the decision register
- Does not claim contractual entitlement or causation
- Does not upload schedule files
