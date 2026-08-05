"""Cited precedent retrieval for ProjectLens change assurance.

XER math stays in the browser. This package answers "what happened last time?"
with inspectable sources, an optional Gemini summary, and LangSmith traces.
Nothing here is treated as advice until a human marks use / ignore.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_precedent_query"]


def run_precedent_query(*args: Any, **kwargs: Any):
    # lazy import so `import Processing.precedent_rag.cases` does not need LangSmith
    from .graph import run_precedent_query as _run

    return _run(*args, **kwargs)
