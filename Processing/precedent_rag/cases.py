"""Load the traced public programme-evidence corpus from JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Cases live next to this package so the sidecar does not need a database yet.
DATA_DIR = Path(__file__).resolve().parent / "data"
CASES_PATH = DATA_DIR / "public_cases.json"
EVAL_PATH = DATA_DIR / "eval_queries.json"


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Return the full case list. Fail loud if the fixture file is missing."""
    target = path or CASES_PATH
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Precedent corpus at {target} is empty or not a list")
    missing_sources = [case.get("id") for case in cases if not case.get("sourceUrl")]
    if path is None and missing_sources:
        raise ValueError(f"Default public corpus has source-less cases: {missing_sources[:3]}")
    return cases


def load_eval_queries(path: Path | None = None) -> list[dict[str, Any]]:
    """Gold queries for retrieval quality — build these before trusting rankings."""
    target = path or EVAL_PATH
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def case_text(case: dict[str, Any]) -> str:
    """Flatten a case into the string we embed / score against."""
    risks = " ".join(case.get("risks") or [])
    evidence = " ".join(case.get("evidence") or [])
    return " ".join(
        [
            case.get("title", ""),
            case.get("problem", ""),
            case.get("context", ""),
            case.get("decision", ""),
            case.get("intervention", ""),
            case.get("outcome", ""),
            case.get("claimBoundary", ""),
            risks,
            evidence,
        ]
    ).strip()
