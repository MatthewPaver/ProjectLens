"""LangGraph flow: retrieve → cite → optional Gemini summary.

Traced in LangSmith when LANGSMITH_API_KEY / LANGCHAIN_TRACING_V2 are set.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from langsmith import traceable

from .retrieve import Embedder, hybrid_retrieve
from .summarize import summarise_precedents


class PrecedentState(TypedDict, total=False):
    query: dict[str, Any]
    limit: int
    # bool flag cannot share a name with a graph node — LangGraph treats keys as reserved
    want_summary: bool
    cases: list[dict[str, Any]]
    summary: dict[str, Any]
    embedder_name: str


def _langsmith_key() -> str | None:
    key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    return key or None


def _configure_langsmith() -> None:
    """Trace only when a real key is present — empty .env placeholders must not 401."""
    if _langsmith_key():
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "projectlens-precedent-rag")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"


def build_graph(embedder: Embedder | None = None):
    """Compile the small graph. Embedder injection keeps unit tests offline."""
    from langgraph.graph import END, StateGraph

    def retrieve_cases(state: PrecedentState) -> PrecedentState:
        hits = hybrid_retrieve(
            state["query"],
            limit=int(state.get("limit") or 5),
            embedder=embedder,
        )
        return {
            "cases": hits,
            "embedder_name": type(embedder).__name__ if embedder else "GeminiEmbedder",
        }

    def write_brief(state: PrecedentState) -> PrecedentState:
        if not state.get("want_summary", True):
            return {"summary": {"text": "", "cited_ids": [], "model": None, "error": "skipped"}}
        brief = summarise_precedents(state["query"], state.get("cases") or [])
        return {"summary": brief}

    graph = StateGraph(PrecedentState)
    graph.add_node("retrieve_cases", retrieve_cases)
    graph.add_node("write_brief", write_brief)
    graph.set_entry_point("retrieve_cases")
    graph.add_edge("retrieve_cases", "write_brief")
    graph.add_edge("write_brief", END)
    return graph.compile()


def _run_precedent_query_impl(
    query: dict[str, Any],
    *,
    limit: int = 5,
    summarise: bool = True,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    app = build_graph(embedder=embedder)
    final = app.invoke(
        {
            "query": query,
            "limit": limit,
            "want_summary": summarise,
        }
    )
    cases = final.get("cases") or []
    summary = final.get("summary") or {}
    return {
        "cases": [
            {
                "id": case.get("id"),
                "title": case.get("title"),
                "project": case.get("project"),
                "sector": case.get("sector"),
                "phase": case.get("phase"),
                "type": case.get("type"),
                "decision": case.get("decision"),
                "outcome": case.get("outcome"),
                "score": case.get("score"),
                "semantic": case.get("semantic"),
                "reasons": case.get("reasons"),
                "evidence": case.get("evidence"),
                "citation": case.get("citation"),
                "confidence": case.get("confidence"),
            }
            for case in cases
        ],
        "summary": summary,
        "mode": "gemini-hybrid" if embedder is None else f"hybrid:{type(embedder).__name__}",
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "projectlens-precedent-rag"),
        "human_gate": "Mark each precedent Use or Ignore before it touches the decision register.",
    }


# only wrap with LangSmith when a key exists — avoids 401 spam from empty placeholders
_traced_run = traceable(name="projectlens_precedent_query", run_type="chain")(_run_precedent_query_impl)


def run_precedent_query(
    query: dict[str, Any],
    *,
    limit: int = 5,
    summarise: bool = True,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Public entry used by the FastAPI sidecar and the eval CLI."""
    _configure_langsmith()
    runner = _traced_run if _langsmith_key() else _run_precedent_query_impl
    return runner(query, limit=limit, summarise=summarise, embedder=embedder)
