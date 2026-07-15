#!/usr/bin/env python3
"""Build the public ProjectLens longitudinal dataset from annual GMPP releases.

The pipeline deliberately keeps exact calculations separate from narrative
classification. Dates, rating transitions and variances are deterministic.
Narrative themes use a visible keyword taxonomy and are presented as signals,
not causal proof or model predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Data" / "public" / "raw"
OUTPUT_PATH = ROOT / "docs" / "data" / "gmpp.json"

SOURCES = {
    2020: {
        "file": "gmpp_2019_20.csv",
        "label": "IPA Annual Report 2019-20",
        "url": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2020",
    },
    2021: {
        "file": "gmpp_2020_21.csv",
        "label": "IPA Annual Report 2020-21",
        "url": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2021",
    },
    2022: {
        "file": "gmpp_2021_22.csv",
        "label": "IPA Annual Report 2021-22",
        "url": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022",
    },
    2023: {
        "file": "gmpp_2022_23.csv",
        "label": "IPA Annual Report 2022-23",
        "url": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022-23",
    },
    2024: {
        "file": "gmpp_2023_24.csv",
        "label": "IPA Annual Report 2023-24",
        "url": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2023-24",
    },
    2025: {
        "file": "gmpp_2024_25.csv",
        "label": "NISTA Annual Report 2024-25",
        "url": "https://www.gov.uk/government/publications/nista-annual-report-2024-2025",
    },
    2026: {
        "file": "gmpp_2025_26.csv",
        "label": "NISTA Major Projects Annual Report 2025-26",
        "url": "https://www.gov.uk/government/publications/nista-major-projects-annual-report-2025-26",
    },
}

EXTERNAL_SOURCES = [
    {
        "label": "Major projects data archive",
        "type": "Historic disclosure",
        "url": "https://www.gov.uk/government/collections/major-projects-data",
        "use": "Earlier annual and departmental releases back to 2013",
    },
    {
        "label": "UK Infrastructure Pipeline",
        "type": "Forward look",
        "url": "https://www.gov.uk/government/publications/uk-infrastructure-pipeline",
        "use": "Status, expected completion and anticipated investment",
    },
    {
        "label": "NAO major-project value review",
        "type": "Independent scrutiny",
        "url": "https://www.nao.org.uk/insights/delivering-value-from-government-investment-in-major-projects/",
        "use": "Benefits, outcomes and recurring delivery lessons",
    },
    {
        "label": "Public Accounts Committee value inquiry",
        "type": "Parliamentary scrutiny",
        "url": "https://publications.parliament.uk/pa/cm5804/cmselect/cmpubacc/456/report.html",
        "use": "Evidence on whether outputs translated into public value",
    },
    {
        "label": "Contracts Finder",
        "type": "Commercial evidence",
        "url": "https://www.contractsfinder.service.gov.uk/Search",
        "use": "Procurement notices, awards and supplier opportunities",
    },
    {
        "label": "Evaluation Registry",
        "type": "Outcome evidence",
        "url": "https://evaluation-registry.cabinetoffice.gov.uk/",
        "use": "Registered government evaluations and published findings",
    },
]

THEMES = {
    "Approvals & decisions": [
        "approval", "business case", "planning decision", "consent", "decision", "review note",
        "spending review", "authorisation", "authorization",
    ],
    "Workforce & capability": [
        "workforce", "resource", "resourcing", "capacity", "skills", "recruit", "crew", "staff",
        "capability", "training",
    ],
    "Commercial & supply chain": [
        "supplier", "supply chain", "commercial", "contract", "procurement", "tender", "market",
        "inflation", "affordability",
    ],
    "Scope & requirements": [
        "scope", "requirement", "design", "specification", "strategic direction", "change control",
        "rebaseline", "re-baseline", "reset",
    ],
    "Governance & assurance": [
        "governance", "assurance", "accountability", "oversight", "review", "sro", "programme board",
    ],
    "Technical & integration": [
        "technical", "technology", "integration", "engineering", "testing", "digital", "data",
        "migration", "interoperability",
    ],
    "Delivery & schedule": [
        "schedule", "delay", "milestone", "delivery", "timetable", "completion", "progress", "pace",
        "critical path", "slippage",
    ],
    "Cost & funding": [
        "cost", "budget", "funding", "finance", "financial", "whole life", "spend", "expenditure",
    ],
}

RATING_SCORE = {"GREEN": 0, "AMBER": 1, "RED": 2}


@dataclass(frozen=True)
class Columns:
    project_id: str
    project_name: str
    department: str
    category: str
    description: str
    ipa_dca: str
    sro_dca: str
    commentary: str
    start_date: str
    end_date: str
    schedule_narrative: str
    fy_baseline: str
    fy_forecast: str
    fy_variance: str
    variance_narrative: str
    whole_life_cost: str
    cost_narrative: str
    benefits: str
    benefits_narrative: str


def find_column(frame: pd.DataFrame, *needles: str, exclude: tuple[str, ...] = ()) -> str:
    for column in frame.columns:
        cleaned = re.sub(r"\s+", " ", str(column)).lower()
        if all(needle.lower() in cleaned for needle in needles) and not any(
            item.lower() in cleaned for item in exclude
        ):
            return str(column)
    return ""


def columns_for(frame: pd.DataFrame) -> Columns:
    return Columns(
        project_id=find_column(frame, "major projects id")
        or find_column(frame, "gmpp id"),
        project_name=find_column(frame, "project name"),
        department=find_column(frame, "department"),
        category=find_column(frame, "annual report category"),
        description=find_column(frame, "description") or find_column(frame, "aims"),
        ipa_dca=find_column(frame, "ipa delivery confidence"),
        sro_dca=find_column(frame, "sro delivery confidence"),
        commentary=find_column(frame, "commentary", "delivery confidence")
        or find_column(frame, "commentary", "rag rating"),
        start_date=find_column(frame, "start date"),
        end_date=find_column(frame, "end date"),
        schedule_narrative=find_column(frame, "narrative", "schedule"),
        fy_baseline=find_column(frame, "financial year baseline"),
        fy_forecast=find_column(frame, "financial year forecast"),
        fy_variance=find_column(frame, "financial year variance"),
        variance_narrative=find_column(frame, "narrative", "variance", exclude=("schedule",)),
        whole_life_cost=find_column(frame, "whole life cost"),
        cost_narrative=find_column(frame, "narrative", "whole life cost")
        or find_column(frame, "costs narrative"),
        benefits=find_column(frame, "benefits", exclude=("narrative",)),
        benefits_narrative=find_column(frame, "benefits narrative")
        or find_column(frame, "narrative", "benefits"),
    )


def value(row: pd.Series, column: str) -> Any:
    if not column or column not in row or pd.isna(row[column]):
        return None
    result = row[column]
    return result.item() if hasattr(result, "item") else result


def text_value(row: pd.Series, column: str) -> str:
    result = value(row, column)
    if result is None:
        return ""
    return re.sub(r"\s+", " ", str(result)).strip()


def numeric_value(row: pd.Series, column: str) -> float | None:
    result = value(row, column)
    if result is None:
        return None
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return None if math.isnan(float(result)) else round(float(result), 3)
    cleaned = re.sub(r"[^0-9.()\-]", "", str(result).replace(",", ""))
    if not cleaned or cleaned in {"-", "."}:
        return None
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    try:
        return round(float(cleaned), 3)
    except ValueError:
        return None


def date_value(row: pd.Series, column: str) -> str | None:
    result = value(row, column)
    if result is None:
        return None
    raw = str(result).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        parsed = pd.to_datetime(raw, errors="coerce", yearfirst=True)
    else:
        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date().isoformat()


def rating_value(raw: str) -> str | None:
    upper = raw.strip().upper()
    for rating in ("RED", "AMBER", "GREEN"):
        if upper == rating:
            return rating.title()
    return None


def classify_themes(*parts: str) -> list[str]:
    corpus = " ".join(part for part in parts if part).lower()
    hits = []
    for theme, keywords in THEMES.items():
        score = sum(len(re.findall(rf"\b{re.escape(keyword)}\b", corpus)) for keyword in keywords)
        if score:
            hits.append((theme, score))
    return [theme for theme, _ in sorted(hits, key=lambda item: (-item[1], item[0]))[:4]]


def compact_excerpt(*parts: str, limit: int = 360) -> str:
    combined = " ".join(part for part in parts if part).strip()
    if len(combined) <= limit:
        return combined
    cut = combined[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def load_year(year: int, path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, encoding_errors="replace")
    # The 2019-20 consolidated CSV is published transposed, with one project per
    # column. Restore the same row-oriented contract used by later releases.
    if not find_column(frame, "project name") and find_column(frame, "gmpp id"):
        identifier_column = frame.columns[0]
        frame = (
            frame.set_index(identifier_column)
            .transpose()
            .reset_index(names=identifier_column)
        )
    cols = columns_for(frame)
    required = [cols.project_id, cols.project_name, cols.department]
    if not all(required):
        raise ValueError(f"Could not identify required columns in {path.name}: {cols}")
    records = []
    for _, row in frame.iterrows():
        name = text_value(row, cols.project_name)
        if not name:
            continue
        project_id = text_value(row, cols.project_id).upper()
        if not project_id:
            digest = hashlib.sha1(name.lower().encode("utf-8")).hexdigest()[:10].upper()
            project_id = f"UNPUBLISHED-{year}-{digest}"
        commentary = text_value(row, cols.commentary)
        schedule = text_value(row, cols.schedule_narrative)
        variance = text_value(row, cols.variance_narrative)
        cost = text_value(row, cols.cost_narrative)
        description = text_value(row, cols.description)
        ipa_dca = rating_value(text_value(row, cols.ipa_dca))
        sro_dca = rating_value(text_value(row, cols.sro_dca))
        delivery_confidence = ipa_dca or sro_dca
        delivery_confidence_source = (
            "Independent assurance" if ipa_dca else "SRO Q4 assessment" if sro_dca else None
        )
        records.append(
            {
                "id": project_id,
                "year": year,
                "name": name,
                "department": text_value(row, cols.department),
                "category": text_value(row, cols.category),
                "description": description,
                "ipaDca": ipa_dca,
                "sroDca": sro_dca,
                "deliveryConfidence": delivery_confidence,
                "deliveryConfidenceSource": delivery_confidence_source,
                "commentary": commentary,
                "startDate": date_value(row, cols.start_date),
                "endDate": date_value(row, cols.end_date),
                "scheduleNarrative": schedule,
                "fyBaseline": numeric_value(row, cols.fy_baseline),
                "fyForecast": numeric_value(row, cols.fy_forecast),
                "fyVariance": numeric_value(row, cols.fy_variance),
                "varianceNarrative": variance,
                "wholeLifeCost": numeric_value(row, cols.whole_life_cost),
                "costNarrative": cost,
                "benefits": numeric_value(row, cols.benefits),
                "benefitsNarrative": text_value(row, cols.benefits_narrative),
                "themes": classify_themes(commentary, schedule, variance, cost),
                "evidenceExcerpt": compact_excerpt(commentary, schedule),
                "sourceUrl": SOURCES[year]["url"],
                "sourceLabel": SOURCES[year]["label"],
            }
        )
    return records


def day_delta(current: str | None, previous: str | None) -> int | None:
    if not current or not previous:
        return None
    return (datetime.fromisoformat(current) - datetime.fromisoformat(previous)).days


def rating_delta(current: str | None, previous: str | None) -> int | None:
    current_key = current.upper() if current else None
    previous_key = previous.upper() if previous else None
    if current_key not in RATING_SCORE or previous_key not in RATING_SCORE:
        return None
    return RATING_SCORE[current_key] - RATING_SCORE[previous_key]


def transition_label(delta: int | None) -> str:
    if delta is None:
        return "Not comparable"
    if delta > 0:
        return "Worsened"
    if delta < 0:
        return "Improved"
    return "Unchanged"


def attention_score(current: dict[str, Any], previous: dict[str, Any] | None) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if current["deliveryConfidence"] == "Red":
        score += 35
        reasons.append("Current published delivery confidence is Red")
    elif current["deliveryConfidence"] == "Amber":
        score += 18
        reasons.append("Current published delivery confidence is Amber")
    if previous:
        movement = rating_delta(current["deliveryConfidence"], previous["deliveryConfidence"])
        if movement and movement > 0:
            score += 25
            reasons.append(
                "Published rating worsened from "
                f"{previous['deliveryConfidence']} to {current['deliveryConfidence']}"
            )
        finish_delta = day_delta(current["endDate"], previous["endDate"])
        if finish_delta and finish_delta > 0:
            score += min(20, round(finish_delta / 18))
            reasons.append(f"Published end date moved {finish_delta:,} days later")
    variance = current.get("fyVariance")
    if variance is not None and abs(variance) >= 5:
        score += min(15, round(abs(variance) / 3))
        reasons.append(f"In-year forecast variance is {variance:+.1f}%")
    if not current["evidenceExcerpt"]:
        score = max(0, score - 8)
    return min(100, score), reasons[:4]


def build_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_id[record["id"]].append(record)
    for history in by_id.values():
        history.sort(key=lambda item: item["year"])

    projects = []
    transitions = Counter()
    theme_counts = Counter()
    for project_id, history in by_id.items():
        current = history[-1]
        if current["year"] != 2026:
            continue
        previous = next((item for item in reversed(history[:-1]) if item["year"] == 2025), None)
        delta = rating_delta(
            current["deliveryConfidence"],
            previous["deliveryConfidence"] if previous else None,
        )
        status = transition_label(delta)
        transitions[status] += 1
        theme_counts.update(current["themes"])
        score, reasons = attention_score(current, previous)
        projects.append(
            {
                **current,
                "history": history,
                "previous": previous,
                "ratingDelta": delta,
                "transition": status,
                "endDateDeltaDays": day_delta(
                    current["endDate"], previous["endDate"] if previous else None
                ),
                "attentionScore": score,
                "attentionReasons": reasons,
            }
        )

    projects.sort(key=lambda item: (-item["attentionScore"], item["name"]))
    comparable_counts = {}
    year_sets = {
        year: {record["id"] for record in records if record["year"] == year}
        for year in SOURCES
    }
    ordered_years = sorted(SOURCES)
    for previous, current in zip(ordered_years, ordered_years[1:]):
        comparable_counts[f"{previous}-{current}"] = len(year_sets[previous] & year_sets[current])

    latest_leavers = []
    for project_id in sorted(year_sets[2025] - year_sets[2026]):
        final_record = next(
            (record for record in reversed(by_id[project_id]) if record["year"] == 2025),
            None,
        )
        if not final_record:
            continue
        latest_leavers.append(
            {
                **final_record,
                "outcomeStatus": "Outcome not established in annual CSV",
                "outcomeEvidence": (
                    "The project has no record in the 2025-26 release. The annual CSV does not "
                    "identify whether it delivered, closed early, was replaced or ceased to meet "
                    "the portfolio criteria."
                ),
            }
        )

    return {
        "meta": {
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "latestSnapshot": "2026-03-31",
            "method": "Deterministic year-on-year comparison with transparent keyword theme signals",
            "limitations": [
                "Delivery Confidence Assessments are point-in-time judgements, not final outcomes.",
                "Narrative themes are keyword signals and do not establish causation.",
                "Whole-life cost values are not compared across years because reporting price bases can differ.",
                "Missing and exempt values are excluded from like-for-like transitions.",
                "Absence from the latest release does not establish delivery success or failure.",
            ],
        },
        "summary": {
            "latestProjects": len(projects),
            "historyRecords": len(records),
            "matchedLatest": comparable_counts["2025-2026"],
            "allFourYears": len(set.intersection(*(year_sets[year] for year in (2023, 2024, 2025, 2026)))),
            "allReleaseYears": len(set.intersection(*year_sets.values())),
            "transitions": dict(transitions),
            "ratingCounts": dict(
                Counter(project["deliveryConfidence"] or "Not published" for project in projects)
            ),
            "ratingSourceCounts": dict(
                Counter(project["deliveryConfidenceSource"] or "Not published" for project in projects)
            ),
            "topThemes": [
                {"theme": theme, "count": count} for theme, count in theme_counts.most_common()
            ],
            "comparability": comparable_counts,
            "latestLeavers": len(latest_leavers),
            "reportedLatestLeavers": 42,
            "leaverCountDifference": len(latest_leavers) - 42,
        },
        "taxonomy": THEMES,
        "sources": [{"year": year, **source} for year, source in SOURCES.items()],
        "externalSources": EXTERNAL_SOURCES,
        "projects": projects,
        "leavers": latest_leavers,
    }


def validate(payload: dict[str, Any]) -> None:
    assert payload["summary"]["latestProjects"] >= 180
    assert payload["summary"]["matchedLatest"] >= 150
    assert payload["summary"]["allFourYears"] >= 120
    assert all(project["year"] == 2026 for project in payload["projects"])
    assert len({project["id"] for project in payload["projects"]}) == len(payload["projects"])
    for project in payload["projects"]:
        assert 0 <= project["attentionScore"] <= 100
        if project["ratingDelta"] is not None:
            assert project["transition"] in {"Improved", "Unchanged", "Worsened"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    records = []
    for year, source in SOURCES.items():
        path = args.raw_dir / source["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. See Data/public/README.md for source links.")
        records.extend(load_year(year, path))
    payload = build_payload(records)
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"Wrote {len(payload['projects'])} current projects and "
        f"{payload['summary']['historyRecords']} longitudinal records to {args.output}"
    )


if __name__ == "__main__":
    main()
