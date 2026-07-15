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
    assert payload["summary"]["historyRecords"] == 1417
    assert payload["summary"]["matchedLatest"] == 170
    assert payload["summary"]["allFourYears"] == 129
    assert payload["summary"]["allReleaseYears"] == 35


def test_transposed_2019_20_release_retains_stable_ids():
    records = load_year(2020, RAW / SOURCES[2020]["file"])
    assert len(records) == 125
    assert records[0]["id"] == "BEIS_0004_1920-Q2"
    assert records[0]["name"] == "Future Shared Services Programme"
    assert all(not record["id"].startswith("UNPUBLISHED-") for record in records)


def test_rating_transitions_are_comparable_and_bounded():
    payload = public_payload()
    transitions = payload["summary"]["transitions"]
    assert transitions["Worsened"] == 18
    assert transitions["Improved"] == 30
    assert transitions["Unchanged"] == 103
    # 19 projects are new/unmatched and 19 matched projects lack a comparable rating.
    assert transitions["Not comparable"] == 38
    assert payload["summary"]["ratingCounts"] == {
        "Amber": 109,
        "Red": 34,
        "Green": 29,
        "Not published": 17,
    }
    assert payload["summary"]["ratingSourceCounts"] == {
        "Independent assurance": 81,
        "SRO Q4 assessment": 91,
        "Not published": 17,
    }
    assert all(0 <= project["attentionScore"] <= 100 for project in payload["projects"])


def test_official_delivery_confidence_uses_sro_fallback_without_losing_raw_values():
    payload = public_payload()
    sro_fallbacks = [
        project for project in payload["projects"]
        if project["ipaDca"] is None and project["sroDca"] is not None
    ]
    assert len(sro_fallbacks) == 91
    assert all(project["deliveryConfidence"] == project["sroDca"] for project in sro_fallbacks)
    assert all(project["deliveryConfidenceSource"] == "SRO Q4 assessment" for project in sro_fallbacks)


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


def test_latest_leavers_are_visible_without_invented_outcomes():
    payload = public_payload()
    assert payload["summary"]["latestLeavers"] == 43
    assert len(payload["leavers"]) == 43
    assert payload["summary"]["reportedLatestLeavers"] == 42
    assert payload["summary"]["leaverCountDifference"] == 1
    assert all(leaver["year"] == 2025 for leaver in payload["leavers"])
    assert all(leaver["outcomeStatus"] == "Outcome not established in annual CSV" for leaver in payload["leavers"])
