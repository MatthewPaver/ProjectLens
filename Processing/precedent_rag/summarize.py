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

1) Pattern match — one sentence: which retrieved precedent(s) are closest to the
   live failure mode (late interface/scope during integration, compressed testing,
   narrative vs schedule mismatch) and why (cite [DG-xxx]).
2) What worked — one sentence naming an intervention that limited damage (cite).
3) Caution — one sentence naming a retrieved case where a similar change went badly
   (cite). Prefer compressed-test / unconditional-approval failures when present.
4) Reviewer prompt — label this line exactly "Reviewer prompt:" then one concrete
   question that forces reconciliation of narrative vs evidence.

Rules:
- Only discuss precedents supplied in the user message.
- Every substantive claim must cite a case id like [DG-024].
- Do not invent case ids, evidence refs, dates, costs, or outcomes.
- Do not paraphrase every case in order — select; contrast; be useful.
- UK English.
- End with: "Human gate: mark each precedent Use or Ignore before the decision register."
"""


def _allowed_ids(cases: list[dict[str, Any]]) -> set[str]:
    return {str(case.get("id")) for case in cases if case.get("id")}


def _strip_uncited_inventions(text: str, allowed: set[str]) -> str:
    """Drop sentences that name a DG-* id outside the retrieved set."""
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        ids = set(re.findall(r"DG-\d+", sentence))
        if ids and not ids.issubset(allowed):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


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

        # langchain message objects keep LangSmith traces readable
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(
                    content=(
                        "Write the cited precedent brief for this retrieval payload:\n"
                        + json.dumps(payload, indent=2)
                    )
                ),
            ]
        )
        raw = getattr(response, "content", str(response))
        if isinstance(raw, list):
            # some gemini wrappers return content blocks
            raw = " ".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in raw
            )
        text = _strip_uncited_inventions(str(raw), allowed)
        cited = sorted(set(re.findall(r"DG-\d+", text)) & allowed)
        return {"text": text, "cited_ids": cited, "model": model_name, "error": None}
    except Exception as exc:  # noqa: BLE001 — fail open; retrieval still useful
        return {"text": "", "cited_ids": [], "model": model_name, "error": str(exc)}
