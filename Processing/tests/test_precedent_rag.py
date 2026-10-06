"""Offline tests for hybrid retrieve + citation-safe summary."""

from __future__ import annotations

from Processing.precedent_rag.cases import load_cases, load_eval_queries
from Processing.precedent_rag.graph import run_precedent_query
from Processing.precedent_rag.retrieve import HashingEmbedder, hybrid_retrieve, metadata_pass
from Processing.precedent_rag.summarize import (
    _strip_uncited_inventions,
    evaluate_summary,
    summarise_precedents,
)


def test_corpus_size_in_viable_band():
    cases = load_cases()
    assert 100 <= len(cases) <= 250
    assert all(case.get("sourceUrl") for case in cases)
    assert all(case.get("evidenceLevel") == "sourced-status-no-outcome" for case in cases)
    assert all("confidence" not in case for case in cases)


def test_metadata_prefers_matching_sector():
    cases = load_cases()
    shortlist = metadata_pass(cases, sector="Infrastructure and Construction", phase=None, change_type="Red")
    # soft filter keeps any metadata hit; every survivor should share ≥1 requested field
    assert len(shortlist) < len(cases)
    assert all(
        c.get("sector") == "Infrastructure and Construction"
        or c.get("type") == "Red"
        for c in shortlist
    )


def test_hybrid_retrieve_returns_reasons_and_citations():
    hits = hybrid_retrieve(
        {
            "problem": "rail infrastructure cost schedule approvals and scope risk",
            "sector": "Infrastructure and Construction",
            "phase": "Published portfolio status",
            "type": "Red",
        },
        limit=3,
        embedder=HashingEmbedder(),
    )
    assert len(hits) == 3
    assert hits[0]["reasons"]
    assert hits[0]["citation"]["case_id"] == hits[0]["id"]
    assert hits[0]["evidence"]
    assert hits[0]["citation"]["source_url"].startswith("https://")


def test_summary_strips_invented_case_ids():
    text = "Review [GMPP-DFT_0033_1819-Q1]. Ignore [GMPP-NOT_REAL] entirely. Keep going."
    cleaned = _strip_uncited_inventions(text, {"GMPP-DFT_0033_1819-Q1"})
    assert "GMPP-DFT_0033_1819-Q1" in cleaned
    assert "GMPP-NOT_REAL" not in cleaned


class _FakeLlm:
    def invoke(self, _messages):
        class Resp:
            content = (
                "A public rail record reports schedule and scope themes [GMPP-DFT_0033_1819-Q1]. "
                "The annual snapshot does not establish a causal outcome [GMPP-DFT_0033_1819-Q1]. "
                "Human gate: mark each precedent Use or Ignore before the decision register."
            )

        return Resp()


def test_summarise_only_cites_retrieved_ids():
    cases = [
        {
            "id": "GMPP-DFT_0033_1819-Q1",
            "title": "East Coast Mainline Programme",
            "decision": "Published delivery confidence: Red",
            "outcome": "Outcome not established",
            "evidence": ["NISTA annual release"],
            "reasons": ["Same sector"],
            "score": 90,
        },
    ]
    brief = summarise_precedents(
        {"problem": "late interface change", "sector": "Rail"},
        cases,
        llm=_FakeLlm(),
    )
    assert brief["error"] is None
    assert set(brief["cited_ids"]) == {"GMPP-DFT_0033_1819-Q1"}
    assert "GMPP-DFT_0033_1819-Q1" in brief["text"]
    assert brief["evaluation"]["passed"]


def test_summary_gate_rejects_uncited_claim_and_decision_overreach():
    cases = [{"id": "GMPP-DFT_0033_1819-Q1"}]
    brief = (
        "The annual release reports Red [GMPP-DFT_0033_1819-Q1]. The programme avoided further delay. "
        "Approve this change. Human gate: mark each precedent Use or Ignore before the decision register."
    )
    result = evaluate_summary(brief, cases)
    assert not result["passed"]
    assert not result["checks"]["claims_cited"]
    assert not result["checks"]["no_decision_overreach"]


def test_hashing_embedder_is_stable():
    first = HashingEmbedder().embed_query("late interface change")
    second = HashingEmbedder().embed_query("late interface change")
    assert first == second


def test_graph_runs_offline_with_hash_embedder():
    result = run_precedent_query(
        {
            "problem": "rail programme schedule cost and scope risk",
            "sector": "Infrastructure and Construction",
            "phase": "Published portfolio status",
            "type": "Red",
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
