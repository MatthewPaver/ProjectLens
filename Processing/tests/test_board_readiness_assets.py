"""Static contracts for the board-readiness product surface."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def test_board_readiness_has_a_complete_human_workflow():
    page = (DOCS / "board-readiness.html").read_text(encoding="utf-8")
    script = (DOCS / "board-readiness.js").read_text(encoding="utf-8")
    stylesheet = (DOCS / "board-readiness.css").read_text(encoding="utf-8")

    assert "Know what the pack can prove." in page
    assert "Set the review" in page
    assert "Add evidence" in page
    assert "Check findings" in page
    assert "Prepare the board" in page
    assert "Northstar is a clearly labelled synthetic" in page
    assert "official UK Government Major Projects Portfolio" in page
    assert page.count('data-file-kind=') == 5
    assert "window.ProjectEvidenceDesk" in script
    assert "A board or named authority remains responsible" in script
    assert "prefers-reduced-motion" in stylesheet
    assert "backdrop-filter" not in stylesheet
    assert "linear-gradient" not in stylesheet


def test_board_readiness_keeps_claims_traceable():
    script = (DOCS / "board-readiness.js").read_text(encoding="utf-8")

    assert script.count('id: "F-') == 5
    assert script.count("source:") >= 5
    assert script.count("reference:") >= 5
    assert "reviewedFindingIds" in script
    assert "humanReviewed" in script
