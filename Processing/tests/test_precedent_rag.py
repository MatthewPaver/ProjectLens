"""Offline tests for hybrid retrieve + citation-safe summary."""

from __future__ import annotations

from Processing.precedent_rag.cases import load_cases, load_eval_queries
from Processing.precedent_rag.graph import run_precedent_query
from Processing.precedent_rag.retrieve import HashingEmbedder, hybrid_retrieve, metadata_pass
from Processing.precedent_rag.summarize import _strip_uncited_inventions, summarise_precedents


def test_corpus_size_in_viable_band():
    cases = load_cases()
    assert 16 <= len(cases) <= 50


def test_metadata_prefers_matching_sector():
    cases = load_cases()
    shortlist = metadata_pass(cases, sector="Rail", phase="Integration", change_type="Scope change")
    # soft filter keeps any metadata hit; every survivor should share ≥1 requested field
    assert len(shortlist) < len(cases)
    assert all(
        c.get("sector") == "Rail"
        or c.get("phase") == "Integration"
        or c.get("type") == "Scope change"
        for c in shortlist
    )


def test_hybrid_retrieve_returns_reasons_and_citations():
    hits = hybrid_retrieve(
        {
            "problem": "late signalling interface change during systems integration",
            "sector": "Rail",
            "phase": "Integration",
            "type": "Scope change",
        },
        limit=3,
        embedder=HashingEmbedder(),
    )
    assert len(hits) == 3
    assert hits[0]["reasons"]
    assert hits[0]["citation"]["case_id"] == hits[0]["id"]
    assert hits[0]["evidence"]


def test_summary_strips_invented_case_ids():
    text = "Follow [DG-024]. Ignore [DG-999] entirely. Keep going."
    cleaned = _strip_uncited_inventions(text, {"DG-024"})
    assert "DG-024" in cleaned
    assert "DG-999" not in cleaned


class _FakeLlm:
    def invoke(self, _messages):
        class Resp:
            content = (
                "Similar late interface work was handled with a design freeze [DG-024]. "
                "Compressed testing without protection failed in [DG-012]. "
                "Human gate: mark each precedent Use or Ignore before the decision register."
            )

        return Resp()


def test_summarise_only_cites_retrieved_ids():
    cases = [
        {
            "id": "DG-024",
            "title": "Late signalling",
            "decision": "freeze",
            "outcome": "+8 days",
            "evidence": ["CR-184"],
            "reasons": ["Same sector"],
            "score": 90,
        },
        {
            "id": "DG-012",
            "title": "Screening route",
            "decision": "compress test",
            "outcome": "+17 days",
            "evidence": ["SC-09"],
            "reasons": ["Same phase"],
            "score": 80,
        },
    ]
    brief = summarise_precedents(
        {"problem": "late interface change", "sector": "Rail"},
        cases,
        llm=_FakeLlm(),
    )
    assert brief["error"] is None
    assert set(brief["cited_ids"]) <= {"DG-024", "DG-012"}
    assert "DG-024" in brief["text"]


def test_graph_runs_offline_with_hash_embedder():
    result = run_precedent_query(
        {
            "problem": "Approve a late signalling interface change before system integration",
            "sector": "Rail",
            "phase": "Integration",
            "type": "Scope change",
        },
        limit=3,
        summarise=False,
        embedder=HashingEmbedder(),
    )
    assert result["human_gate"]
    assert len(result["cases"]) == 3
    assert result["summary"]["error"] == "skipped"


def test_eval_fixture_shape():
    queries = load_eval_queries()
    assert len(queries) >= 5
    for item in queries:
        assert item["must_include_any"]
        assert item["problem"]
