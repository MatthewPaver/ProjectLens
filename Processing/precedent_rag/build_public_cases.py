"""Derive the default retrieval corpus from ProjectLens' traced GMPP data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "data" / "gmpp.json"
OUTPUT = Path(__file__).resolve().parent / "data" / "public_cases.json"


def build(source: dict) -> list[dict]:
    cases = []
    for project in source["projects"]:
        if not project.get("sourceUrl") or not project.get("sourceLabel"):
            continue
        source_id = str(project["id"])
        cases.append(
            {
                "id": f"GMPP-{source_id}",
                "sourceRecordId": source_id,
                "project": project.get("name"),
                "sector": project.get("category"),
                "phase": "Published portfolio status",
                "type": project.get("deliveryConfidence") or "Not reported",
                "year": project.get("year"),
                "title": project.get("name"),
                "problem": project.get("description") or "",
                "context": " ".join(
                    value
                    for value in [
                        project.get("commentary"),
                        project.get("scheduleNarrative"),
                        project.get("costNarrative"),
                    ]
                    if value
                ),
                "risks": project.get("themes") or [],
                "decision": (
                    f"Published delivery confidence: {project.get('deliveryConfidence') or 'not reported'} "
                    f"({project.get('deliveryConfidenceSource') or 'source not specified'})."
                ),
                "intervention": "Not established by the annual portfolio release.",
                "outcome": "Outcome and causal effect are not established by this status snapshot.",
                "evidence": [
                    project.get("sourceLabel"),
                    f"Source record: {source_id}",
                    (project.get("evidenceExcerpt") or "")[:500],
                ],
                "sourceUrl": project.get("sourceUrl"),
                "sourceLabel": project.get("sourceLabel"),
                "evidenceLevel": "sourced-status-no-outcome",
                "claimBoundary": "Comparable public status evidence; not a proven precedent, intervention or outcome.",
            }
        )
    return sorted(cases, key=lambda item: item["id"])


def main() -> None:
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    cases = build(source)
    payload = {
        "schemaVersion": 1,
        "source": {
            "path": "docs/data/gmpp.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "generatedBy": "Processing/precedent_rag/build_public_cases.py",
        },
        "claimBoundary": "Retrieval returns comparable published status evidence. It does not establish causation, a decision precedent or a successful intervention.",
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} traced public evidence cases to {OUTPUT}")


if __name__ == "__main__":
    main()
