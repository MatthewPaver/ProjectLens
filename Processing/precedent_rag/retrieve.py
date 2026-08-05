"""Hybrid retrieve: metadata filters first, then Gemini embeddings for meaning.

We deliberately do not rely on keyword overlap alone — that is the DecisionGraph
demo path. Here embeddings do the semantic work; metadata keeps the shortlist
honest (sector / phase / change type).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any, Protocol

from .cases import case_text, load_cases


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    # bog-standard cosine — keep it local so we are not married to a vector DB yet
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class GeminiEmbedder:
    """Real Gemini embeddings — this is the production retrieval path."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        # langchain wrapper keeps LangSmith able to see embed calls when tracing
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to ProjectLens/.env "
                "(copy from another local project) — never commit the key."
            )
        self._model = GoogleGenerativeAIEmbeddings(
            # text-embedding-004 retired on newer AI Studio projects — gemini-embedding-001 is current
            model=model or os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
            google_api_key=key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


class HashingEmbedder:
    """Tiny local embedder for unit tests only — not used by the live sidecar."""

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in text.lower().split():
            vec[hash(token) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embedder(prefer_gemini: bool = True) -> Embedder:
    """Prefer Gemini; fall back to hashing only when explicitly allowed for tests."""
    if prefer_gemini:
        return GeminiEmbedder()
    return HashingEmbedder()


@lru_cache(maxsize=1)
def _case_corpus() -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    cases = tuple(load_cases())
    texts = tuple(case_text(case) for case in cases)
    return cases, texts


def metadata_pass(
    cases: list[dict[str, Any]],
    *,
    sector: str | None,
    phase: str | None,
    change_type: str | None,
) -> list[dict[str, Any]]:
    """Soft filter: keep exact metadata matches preferred, but do not empty the pool.

    Pure hard filters are brittle on a 24-case corpus; we score metadata later
    and only drop rows when the pool stays large enough.
    """
    if not any([sector, phase, change_type]):
        return list(cases)

    scored: list[tuple[int, dict[str, Any]]] = []
    for case in cases:
        hits = 0
        if sector and case.get("sector") == sector:
            hits += 1
        if phase and case.get("phase") == phase:
            hits += 1
        if change_type and case.get("type") == change_type:
            hits += 1
        scored.append((hits, case))

    # keep anything with at least one metadata hit if that still leaves ≥8 cases
    matched = [case for hits, case in scored if hits > 0]
    if len(matched) >= 8:
        return matched
    return list(cases)


def hybrid_retrieve(
    query: dict[str, Any],
    *,
    limit: int = 5,
    embedder: Embedder | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Rank precedents with metadata boosts + embedding similarity.

    Each hit carries `reasons` so the UI can show *why* it was retrieved —
    citations are useless if the reviewer cannot inspect the match basis.
    """
    corpus = list(cases) if cases is not None else list(_case_corpus()[0])
    sector = (query.get("sector") or "").strip() or None
    phase = (query.get("phase") or "").strip() or None
    change_type = (query.get("type") or query.get("change_type") or "").strip() or None
    problem = " ".join(
        filter(
            None,
            [
                query.get("problem"),
                query.get("narrative"),
                " ".join(query.get("blockers") or []),
            ],
        )
    ).strip()
    if not problem:
        raise ValueError("Query needs a problem / narrative to retrieve against")

    shortlist = metadata_pass(corpus, sector=sector, phase=phase, change_type=change_type)
    engine = embedder or get_embedder(prefer_gemini=True)

    query_vec = engine.embed_query(problem)
    doc_vecs = engine.embed_documents([case_text(case) for case in shortlist])

    ranked: list[dict[str, Any]] = []
    for case, doc_vec in zip(shortlist, doc_vecs):
        semantic = _cosine(query_vec, doc_vec)
        sector_hit = bool(sector and case.get("sector") == sector)
        phase_hit = bool(phase and case.get("phase") == phase)
        type_hit = bool(change_type and case.get("type") == change_type)
        # embeddings carry most of the weight — metadata is a steer, not a veto
        raw = semantic * 0.72 + (0.1 if sector_hit else 0) + (0.1 if phase_hit else 0) + (0.08 if type_hit else 0)
        score = round(min(0.99, max(0.0, raw)) * 100)
        reasons = [
            f"Semantic match · {semantic:.2f}",
            sector_hit and f"Same sector · {case.get('sector')}",
            phase_hit and f"Same phase · {case.get('phase')}",
            type_hit and f"Same change · {case.get('type')}",
        ]
        ranked.append(
            {
                **case,
                "score": score,
                "semantic": round(semantic, 4),
                "reasons": [r for r in reasons if r],
                "citation": {
                    "case_id": case.get("id"),
                    "evidence": list(case.get("evidence") or []),
                    "project": case.get("project"),
                    "year": case.get("year"),
                },
            }
        )

    ranked.sort(key=lambda row: (-row["score"], -float(row.get("confidence") or 0)))
    return ranked[:limit]
