# ProjectLens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Public data](https://img.shields.io/badge/Data-UK_GMPP-d7ff4f?style=flat-square)
![License](https://img.shields.io/badge/Code-MIT-blue?style=flat-square)
[![Validate](https://github.com/MatthewPaver/ProjectLens/actions/workflows/validate.yml/badge.svg)](https://github.com/MatthewPaver/ProjectLens/actions/workflows/validate.yml)

**UK major-project early warning, built from published evidence**

ProjectLens shows what changed across the UK’s largest public projects, why it deserves scrutiny and what happened to comparable projects next.

[Open the live product](https://matthewpaver.github.io/ProjectLens/) · [Portfolio](https://matthewpaver.github.io/MatthewPaver/store/) · [Market scan](competitor-profiles/_summary.md)

</div>

## Why it exists

Major-project reports contain delivery-confidence assessments, dates, costs and narrative explanations. The information is public, but the releases are usually read as separate annual snapshots.

ProjectLens joins four annual Government Major Projects Portfolio releases into an inspectable history. It helps a reviewer answer:

- Which published delivery-confidence ratings worsened?
- Which end dates changed, and by how much?
- What explanations are departments publishing?
- Where do apparently positive and negative signals conflict?
- Which comparable projects later improved?
- What source evidence should a decision-maker read next?

## Current evidence base

| Measure | Current result |
| --- | ---: |
| Current projects | 189 |
| Records across four releases | 873 |
| Projects matched from 2024-25 to 2025-26 | 170 |
| Projects present in all four releases | 129 |
| Comparable IPA ratings that worsened | 6 |
| Comparable IPA ratings that improved | 12 |

The latest snapshot was reported by departments at 31 March 2026. Source links and licensing notes are in [`Data/public/README.md`](Data/public/README.md).

## Product workflow

1. **Briefing** presents a small, transparent attention queue.
2. **Explorer** searches and filters all 189 current projects.
3. **Project detail** shows the scoring reasons, source narrative and four-year trail.
4. **Case match** retrieves historical improvements and explains the similarity basis.
5. **Method** publishes the exact score, data boundaries and original sources.

The public interface is a static GitHub Pages application in [`docs/`](docs). No server or account is required.

## Method

Exact calculations are deterministic:

- stable GMPP identifiers join annual records
- exact Green, Amber and Red assessments produce year-on-year movement
- published end dates produce day differences
- annual forecast variance is taken from the source release
- missing and exempt values remain not comparable

The attention score is a review queue, not a probability of failure:

```text
+35 current Red IPA rating
+25 rating worsened
+0 to +20 later published end date
+0 to +15 material in-year variance
-8 no public evidence excerpt
```

Narrative themes use the visible keyword taxonomy in [`Processing/gmpp_pipeline.py`](Processing/gmpp_pipeline.py). They organise evidence but do not establish causation.

Whole-life cost values are not compared across years because published price bases can differ.

## Rebuild the public dataset

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Processing/gmpp_pipeline.py
```

This reads the unmodified annual CSV files under `Data/public/raw/`, validates the longitudinal output and writes `docs/data/gmpp.json`.

Serve the static product locally:

```bash
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Tests

```bash
make test
```

Tests cover the original schedule-processing modules plus public-data counts, matching, rating transitions, theme classification and score boundaries.

## Competitive position

nPlan, SmartPM, Nodes & Links and InEight already provide strong private schedule analytics, forecasting, quantitative risk and enterprise controls. ProjectLens deliberately does not imitate them.

Its narrower gap is **open, source-linked, longitudinal public delivery intelligence**. The dated market scan and source notes are in [`competitor-profiles/`](competitor-profiles).

## Boundaries

- Delivery Confidence Assessments are point-in-time judgements, not outcomes.
- Annual portfolio data cannot support activity-level critical-path forecasting.
- Theme matching does not prove root cause.
- Comparable cases prompt investigation and do not prescribe an intervention.
- A production internal version would require secure schedule connectors, permissions, audit logs and organisation-specific validation.

## Legacy schedule engine

The repository still contains the earlier schedule-processing demonstrator under [`Processing/analysis/`](Processing/analysis) and the local Flask interface under [`Website/`](Website). Its bundled Alpha, Beta and other project records are synthetic test fixtures. They are retained for engineering tests, not used as evidence in the public ProjectLens product.

## Repository layout

```text
Data/public/             official annual GMPP source files
Processing/gmpp_pipeline.py
                         longitudinal preparation and validation
docs/                    GitHub Pages product and generated JSON
competitor-profiles/     dated market scan and source notes
Processing/analysis/     legacy schedule-analysis modules
Processing/tests/        deterministic and integration tests
```

## License

ProjectLens code is MIT licensed. Government source data is used under the Open Government Licence. See each official publication for source terms.
