from pathlib import Path

from Processing.gmpp_pipeline import SOURCES, build_payload, classify_themes, load_year


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "public" / "raw"


def public_payload():
    records = []
    for year, source in SOURCES.items():
        records.extend(load_year(year, RAW / source["file"]))
    return build_payload(records)


def test_public_dataset_has_expected_longitudinal_coverage():
    payload = public_payload()
    assert payload["summary"]["latestProjects"] == 189
    assert payload["summary"]["historyRecords"] == 873
    assert payload["summary"]["matchedLatest"] == 170
    assert payload["summary"]["allFourYears"] == 129


def test_rating_transitions_are_comparable_and_bounded():
    payload = public_payload()
    transitions = payload["summary"]["transitions"]
    assert transitions["Worsened"] == 6
    assert transitions["Improved"] == 12
    assert transitions["Unchanged"] == 38
    assert all(0 <= project["attentionScore"] <= 100 for project in payload["projects"])


def test_theme_taxonomy_is_transparent_and_repeatable():
    narrative = "Supplier capacity caused a delay in testing and required a revised approval decision."
    assert classify_themes(narrative) == [
        "Approvals & decisions",
        "Commercial & supply chain",
        "Delivery & schedule",
        "Technical & integration",
    ]


def test_costs_are_context_not_cross_year_claims():
    payload = public_payload()
    assert "Whole-life cost values are not compared across years" in payload["meta"]["limitations"][2]
    assert all("wholeLifeCostDelta" not in project for project in payload["projects"])
