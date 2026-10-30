"""GMPP narrative claim flag: unchanged wording while the published end date moved."""

from pathlib import Path

from Processing.gmpp_pipeline import SOURCES, annotate_history_claims, build_payload, detect_schedule_claims, load_claim_rules, load_year


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "public" / "raw"


def test_capability_targets_do_not_claim_programme_closure_is_unchanged():
    for capability in [
        "initial operating capability", "initial operational capability",
        "full operating capability", "full operational capability",
    ]:
        result = detect_schedule_claims(
            f"The programme remains on track to deliver its {capability} in October 2028.",
            "2030-04-30", "2032-04-30",
        )
        assert result["stable"] == []
        assert result["contradiction"] is None
        assert result["publishedDateCheck"] is None


def public_payload():
    records = []
    for year, source in SOURCES.items():
        records.extend(load_year(year, RAW / source["file"]))
    return build_payload(records)


def test_wedgetail_and_istar_2025_26_are_flagged_but_2021_22_is_not():
    payload = public_payload()
    by_id = {project["id"]: project for project in payload["projects"]}

    wedgetail = next(p for p in payload["projects"] if "WEDGETAIL" in p["name"].upper())
    istar = next(p for p in payload["projects"] if "ISTAR" in p["name"].upper())
    assert wedgetail["narrativeSaysUnchangedButMoved"] is True
    assert istar["narrativeSaysUnchangedButMoved"] is True
    assert wedgetail["narrativeExcerpt"]
    assert istar["narrativeExcerpt"]

    # 2021→2022 Wedgetail stated the move ("increased from… Correction of published position").
    history = wedgetail["history"]
    pair_2022 = next(item for item in history if item["year"] == 2022)
    assert pair_2022["narrativeSaysUnchangedButMoved"] is False


def test_detect_schedule_claims_matches_tuning_cases():
    rules = load_claim_rules()
    istar = detect_schedule_claims(
        "Compared to financial year 24/25-Q4, the programme's end-date at 25/26-Q4 remained the same at 31/12/2027.",
        "2033-05-02",
        "2027-12-31",
        rules,
    )
    assert istar["contradiction"] is not None

    correction = detect_schedule_claims(
        "Compared to 2021-Q4, the project end-date increased from 30 June 26 to 1 June 27 This is primarily due to the following factors: Correction of published position in March 21.",
        "2027-06-01",
        "2026-06-30",
        rules,
    )
    assert correction["contradiction"] is None


def test_stated_move_must_land_on_published_finish():
    # Round 2 N3: same answers as the JS engine for a mistargeted move and a direction-only one.
    rules = load_claim_rules()
    sentence = "the end date changed from 01/04/2027 to 30/09/2027 due to supplier delays"
    miss = detect_schedule_claims(sentence, "2028-03-31", "2027-04-01", rules)
    assert miss["contradiction"] is not None
    assert miss["contradiction"]["kind"] == "date-mismatch"

    near = detect_schedule_claims(sentence, "2027-10-05", "2027-04-01", rules)
    assert near["contradiction"] is None

    extended = "the end date was extended following re-planning"
    later = detect_schedule_claims(extended, "2028-01-01", "2027-10-03", rules)
    assert later["contradiction"] is None

    earlier = detect_schedule_claims(extended, "2027-07-05", "2027-10-03", rules)
    assert earlier["contradiction"] is not None
    assert earlier["contradiction"]["kind"] == "date-mismatch"


def test_end_date_scope_for_moves_and_stability():
    # Round 3 D2: contract-start moves and "unaffected" end-date claims mirror the JS engine.
    rules = load_claim_rules()
    narrative = (
        "The contract start date has moved back by four weeks. "
        "The overall programme end date is unaffected."
    )
    stable = detect_schedule_claims(narrative, "2028-06-30", "2028-06-30", rules)
    assert stable["contradiction"] is None

    moved = detect_schedule_claims(narrative, "2028-10-28", "2028-06-30", rules)
    assert moved["contradiction"] is not None
    assert moved["contradiction"]["kind"] == "says-unchanged"

    lengthened = detect_schedule_claims(
        "The end date has lengthened by 200 days.",
        "2027-07-19",
        "2026-12-31",
        rules,
    )
    assert lengthened["contradiction"] is None


def test_round4_programme_stability():
    for sentence in [
        "The programme remains on track.", "Compared to last year, the programme remains on track.",
        "The project delivery remains on schedule.", "Delivery is on schedule.",
        "The project is progressing to plan.", "The programme remains on course.",
        "The project is on target to complete.",
    ]:
        result = detect_schedule_claims(sentence, "2026-12-27", "2026-06-30")
        assert result["contradiction"]["kind"] == "says-unchanged"
    for sentence in [
        "Phase 2 remains on track; the programme end date has moved to 31/03/2027.",
        "The programme is not on track.", "The procurement is on schedule.",
        "The programme's Phase 2 remains on track.",
        "Milestones remain on track: Phase 1 March 2027, Phase 2 June 2027.",
    ]:
        assert detect_schedule_claims(sentence, "2027-03-31", "2026-06-30")["contradiction"] is None
    assert detect_schedule_claims("Delivery is on schedule.", "2026-06-30", "2026-06-30")["contradiction"] is None


def test_stability_keeps_the_nearest_subject():
    for sentence in [
        "Phase 2 completion remains on track.",
        "The programme's procurement completion is on schedule.",
        "The contract end date is unchanged.",
    ]:
        assert detect_schedule_claims(sentence, "2027-03-31", "2026-06-30")["contradiction"] is None
