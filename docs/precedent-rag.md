# Cited public-programme evidence retrieval

## What problem this solves

A reviewer preparing a change decision may benefit from **comparable published programme evidence with sources**, not a chatbot that invents programme advice. ProjectLens keeps Primavera XER checks deterministic in the browser. The RAG sidecar only receives narrative text, blocker titles and soft metadata filters.

The former default corpus was 25 synthetic “decision cases” with invented evidence references and confidence values. It is retained only as a legacy fixture. The default corpus is now generated from the 189 current, source-linked GMPP records already used by ProjectLens. These records are status snapshots: they can support analogy and a reviewer question, but they do not prove what intervention worked or what caused an outcome.

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
  ├─ Gemini brief (citation-only; invented record ids stripped)
  └─ LangSmith traces (LANGSMITH_PROJECT=projectlens-precedent-rag)
        │
        ▼
Human Use / Ignore on each card → then decision register
```

**GitHub Pages** hosts the static demo only (`docs/` → Pages). There is no Gemini key on that host. With no sidecar URL configured, the UI uses static precedent cards without attempting a network request. Live hybrid retrieve needs `make precedent-rag` locally and an explicit `window.PROJECTLENS_PRECEDENT_RAG_URL`, or a CORS-enabled API you control. Never bake secrets into the static site.

## Why Gemini + LangSmith (not keyword-only)

The original DecisionGraph token-overlap experiment did not solve semantic “same failure mode, different words.” ProjectLens keeps only the reusable retrieval and review pattern. `make public-precedents` regenerates the corpus from `docs/data/gmpp.json`, records its SHA-256 and rejects source-less default cases. There is no fabricated confidence field; the relevance score is only a ranking signal. LangSmith is the optional proof trail for retrieve → summarise runs.

## Eval harness

Offline unit tests (`Processing/tests/test_precedent_rag.py`) cover corpus shape, citation stripping and graph wiring without API keys.

Live retrieval quality:

```bash
make precedent-eval   # gold queries → top-k hit rate (needs GEMINI_API_KEY)
```

Deterministic keyless baseline (hashing embedder, same ranking code, no network):

```bash
make precedent-eval-offline   # recorded 2026-10-06: hit@5 = 7/8 on the 189-record public corpus
```

Judge criteria for the Gemini brief (manual or future LLM-as-judge, non-Gemini):

1. Every substantive claim cites a retrieved `GMPP-*` record id
2. Brief does not treat the live narrative as established fact
3. Includes one concrete reviewer question grounded in the pack
4. Does not approve/reject the live decision

LangSmith project `projectlens-precedent-rag` is for traces, not a substitute for the gold-query hit rate.

## Non-goals

- Does not replace XER finish / float / constraint checks
- Does not auto-write to the decision register
- Does not claim contractual entitlement or causation
- Does not upload schedule files
- Does not call a public status record a successful precedent
