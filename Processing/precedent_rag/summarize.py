"""Optional Gemini summary that may only cite retrieved cases.

If the model invents a case id, we strip that sentence — fail closed on citations,
fail open on "no summary" so the UI still shows the ranked precedents.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


SUMMARY_SYSTEM = """You help a UK project-controls reviewer prepare a board change decision.
You are NOT the decision authority and you must not approve or reject the live pack.

The live narrative is a CLAIM under review, not established fact. If it asserts
"no change" / "on track" while also mentioning finish movement, blockers, or
contradictions, treat that as a credibility problem — do not say the pack
"aligns with successful management."

Write a short decision-support brief (max 140 words) with this structure:

1) Pattern match — one sentence: which retrieved public programme record(s) have
   comparable delivery themes and why (cite the supplied record id).
2) Published evidence — one sentence stating what the annual release actually reports (cite).
3) Uncertainty — one sentence stating what outcome, intervention or causal link the source does not establish (cite).
4) Reviewer prompt — label this line exactly "Reviewer prompt:" then one concrete
   question that forces reconciliation of narrative vs evidence.

Rules:
- Only discuss precedents supplied in the user message.
- Every substantive claim must cite a supplied case id like [GMPP-DFT_0033_1819-Q1].
- Do not invent case ids, evidence refs, dates, costs, or outcomes.
- Never describe the records as proven precedents or claim that an intervention worked.
- Do not paraphrase every case in order — select; contrast; be useful.
- UK English.
- End with: "Human gate: mark each precedent Use or Ignore before the decision register."
"""


def _allowed_ids(cases: list[dict[str, Any]]) -> set[str]:
    return {str(case.get("id")) for case in cases if case.get("id")}


def _citation_ids(text: str) -> set[str]:
    return set(re.findall(r"\[([A-Z][A-Z0-9_-]{2,})\]", text))


def _strip_uncited_inventions(text: str, allowed: set[str]) -> str:
    """Drop sentences that cite an id outside the retrieved set."""
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        ids = _citation_ids(sentence)
        if ids and not ids.issubset(allowed):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def evaluate_summary(text: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail-closed contract check for the optional model-written brief."""
    value = str(text or "").strip()
    allowed = _allowed_ids(cases)
    cited = _citation_ids(value)
    invalid = sorted(cited - allowed)
    substantive = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", value)
        if sentence.strip()
        and not sentence.strip().startswith("Reviewer prompt:")
        and not sentence.strip().startswith("Human gate:")
    ]
    uncited = [sentence for sentence in substantive if not _citation_ids(sentence)]
    decision_overreach = bool(
        re.search(
            r"\b(?:approve|reject|return) (?:this|the live|the current) (?:change|pack|submission)\b",
            value,
            re.IGNORECASE,
        )
    )
    checks = {
        "non_empty": bool(value),
        "citations_present": bool(cited),
        "citations_valid": not invalid,
        "claims_cited": not uncited,
        "human_gate_preserved": value.endswith(
            "Human gate: mark each precedent Use or Ignore before the decision register."
        ),
        "no_decision_overreach": not decision_overreach,
        "concise": len(value.split()) <= 170,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cited_ids": sorted(cited & allowed),
        "invalid_ids": invalid,
        "uncited_claims": uncited,
    }


def summarise_precedents(
    query: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Ask Gemini for a short cited brief. Returns empty text on failure."""
    if not cases:
        return {"text": "", "cited_ids": [], "model": None, "error": "no_cases"}

    allowed = _allowed_ids(cases)
    payload = {
        "live_problem": query.get("problem") or query.get("narrative") or "",
        "filters": {
            "sector": query.get("sector"),
            "phase": query.get("phase"),
            "type": query.get("type") or query.get("change_type"),
        },
        "precedents": [
            {
                "id": case.get("id"),
                "title": case.get("title"),
                "decision": case.get("decision"),
                "outcome": case.get("outcome"),
                "evidence": case.get("evidence"),
                "source_url": case.get("sourceUrl") or (case.get("citation") or {}).get("source_url"),
                "claim_boundary": case.get("claimBoundary"),
                "reasons": case.get("reasons"),
                "score": case.get("score"),
            }
            for case in cases
        ],
    }

    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    try:
        if llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                return {"text": "", "cited_ids": [], "model": None, "error": "missing_gemini_key"}
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=0.2)

        prompt = "Write the cited precedent brief for this retrieval payload:\n" + json.dumps(payload, indent=2)
        try:
            # LangChain messages keep real LangSmith traces readable.
            from langchain_core.messages import HumanMessage, SystemMessage

            messages: list[Any] = [SystemMessage(content=SUMMARY_SYSTEM), HumanMessage(content=prompt)]
        except ImportError:
            # Injected test/model adapters do not need LangChain installed.
            messages = [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        response = llm.invoke(messages)
        raw = getattr(response, "content", str(response))
        if isinstance(raw, list):
            # some gemini wrappers return content blocks
            raw = " ".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in raw
            )
        text = _strip_uncited_inventions(str(raw), allowed)
        evaluation = evaluate_summary(text, cases)
        if not evaluation["passed"]:
            failed = ",".join(name for name, passed in evaluation["checks"].items() if not passed)
            return {
                "text": "",
                "cited_ids": evaluation["cited_ids"],
                "model": model_name,
                "error": f"grounding_gate_failed:{failed}",
                "evaluation": evaluation,
            }
        return {
            "text": text,
            "cited_ids": evaluation["cited_ids"],
            "model": model_name,
            "error": None,
            "evaluation": evaluation,
        }
    except Exception as exc:  # noqa: BLE001 — fail open; retrieval still useful
        return {"text": "", "cited_ids": [], "model": model_name, "error": str(exc)}
