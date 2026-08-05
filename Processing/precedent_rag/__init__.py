"""Cited precedent retrieval for ProjectLens change assurance.

XER math stays in the browser. This package answers "what happened last time?"
with inspectable sources, an optional Gemini summary, and LangSmith traces.
Nothing here is treated as advice until a human marks use / ignore.
"""

from .graph import run_precedent_query

__all__ = ["run_precedent_query"]
