"""Static contract tests for the browser-local XER review demonstrator."""

from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def read_xer_tables(path: Path) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    table_name = ""
    fields: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        cells = line.split("\t")
        marker = cells[0].strip() if cells else ""
        if marker == "%T":
            table_name = cells[1].strip()
            tables.setdefault(table_name, [])
            fields = []
        elif marker == "%F" and table_name:
            fields = [cell.strip() for cell in cells[1:]]
        elif marker == "%R" and table_name and fields:
            tables[table_name].append(dict(zip(fields, cells[1:])))
    return tables


def test_demo_pair_contains_comparable_p6_evidence():
    previous = read_xer_tables(DOCS / "demo" / "northstar-previous.xer")
    current = read_xer_tables(DOCS / "demo" / "northstar-current.xer")

    assert {"PROJECT", "TASK", "TASKPRED", "PROJWBS"} <= previous.keys()
    assert {"PROJECT", "TASK", "TASKPRED", "PROJWBS"} <= current.keys()
    assert len(previous["TASK"]) >= 8
    assert len(current["TASK"]) >= 8

    previous_finish = datetime.fromisoformat(previous["PROJECT"][0]["scd_end_date"])
    current_finish = datetime.fromisoformat(current["PROJECT"][0]["scd_end_date"])
    assert (current_finish - previous_finish).days == 73

    previous_codes = {row["task_code"] for row in previous["TASK"]}
    current_codes = {row["task_code"] for row in current["TASK"]}
    assert previous_codes != current_codes


def test_demo_narrative_exercises_contradiction_review():
    narrative = (DOCS / "demo" / "northstar-narrative.txt").read_text(encoding="utf-8").lower()
    assert "unchanged" in narrative or "on track" in narrative
    assert "logic" not in narrative
    assert "constraint" not in narrative


def test_demo_includes_bounded_assurance_evidence():
    baseline = read_xer_tables(DOCS / "demo" / "northstar-baseline.xer")
    risks = (DOCS / "demo" / "northstar-risks.csv").read_text(encoding="utf-8")
    basis = (DOCS / "demo" / "northstar-schedule-basis.md").read_text(encoding="utf-8")
    decisions = (DOCS / "demo" / "northstar-decisions.csv").read_text(encoding="utf-8")

    assert len(baseline["TASK"]) >= 8
    assert len(baseline["CALENDAR"]) == 2
    assert risks.count("\n") >= 4
    assert "NS-900" not in decisions
    assert "Approved" in decisions and "Deferred" in decisions
    assert "contractual milestone" in basis


def test_review_page_states_privacy_and_model_boundary():
    page = (DOCS / "schedule-review.html").read_text(encoding="utf-8")
    script = (DOCS / "xer-review.js").read_text(encoding="utf-8")

    assert "Nothing uploaded" in page
    assert "run in your browser" in page
    assert "not a forecast" in page
    assert "window.ProjectLensXer" in script
    assert "localStorage" in script
    assert "Evidence completeness" in page
    assert "Intervention follow-through" in page
    assert "decisionReconciliation" in script
    assert "baselineMovements" in script
