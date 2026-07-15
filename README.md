# ProjectLens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Public data](https://img.shields.io/badge/Data-UK_GMPP-d7ff4f?style=flat-square)
![License](https://img.shields.io/badge/Code-MIT-blue?style=flat-square)
[![Validate](https://github.com/MatthewPaver/ProjectLens/actions/workflows/validate.yml/badge.svg)](https://github.com/MatthewPaver/ProjectLens/actions/workflows/validate.yml)

**The open evidence layer for UK major-project delivery**

ProjectLens shows what changed across the UK’s largest public projects, where published signals conflict and which evidence deserves investigation. It also provides a private, browser-local workflow for comparing Primavera P6 XER submissions.

[Open the live product](https://matthewpaver.github.io/ProjectLens/) · [Portfolio](https://matthewpaver.github.io/MatthewPaver/store/) · [Market scan](competitor-profiles/_summary.md)

</div>

## Why it exists

Major-project reports contain delivery-confidence assessments, dates, costs and narrative explanations. The information is public, but the releases are usually read as separate annual snapshots.

ProjectLens joins seven annual Government Major Projects Portfolio releases into an inspectable history. It helps a reviewer answer:

- Which published delivery-confidence ratings worsened?
- Which end dates changed, and by how much?
- What explanations are departments publishing?
- Where do apparently positive and negative signals conflict?
- Which comparable projects later improved?
- What source evidence should a decision-maker read next?
- Which projects have left the portfolio without a confirmed outcome in the annual CSV?

The XER evidence review adds a second, deliberately separate workflow for private schedule submissions:

- What changed between two P6 exports?
- Which milestones, dates, logic, constraints and float movements are material?
- Does the status narrative agree with the schedule evidence?
- Which assurance questions and actions should be recorded before approval?

## Current evidence base

| Measure | Current result |
| --- | ---: |
| Current projects | 189 |
| Records across seven releases | 1,417 |
| Projects matched from 2024-25 to 2025-26 | 170 |
| Current projects present in all seven releases | 35 |
| Current projects present in the latest four releases | 129 |
| Comparable published DCA ratings that worsened | 18 |
| Comparable published DCA ratings that improved | 30 |
| Current Red published DCA ratings | 34 |
| Records absent from the latest release | 43 |

The latest snapshot was reported by departments at 31 March 2026. Source links and licensing notes are in [`Data/public/README.md`](Data/public/README.md).

## Product workflow

1. **Briefing** presents a small, transparent attention queue and outcome follow-up register.
2. **Explorer** searches, filters and saves a browser-local watchlist across all 189 current projects.
3. **Project detail** shows the scoring reasons, source narrative and up to seven years of history.
4. **Case match** retrieves historical improvements and explains the similarity basis.
5. **XER review** compares two schedule submissions, tests narrative claims and creates an evidence-linked action log.
6. **Method** publishes the exact score, data boundaries, original releases and authoritative expansion routes.

The interface is a static GitHub Pages application in [`docs/`](docs). No server or account is required. XER files are parsed locally in the browser and are not uploaded.

## Method

Exact calculations are deterministic:

- stable GMPP identifiers join annual records
- the independent IPA or NISTA assessment is used when published, otherwise the SRO Q4 assessment is used
- exact Green, Amber and Red assessments produce year-on-year movement
- published end dates produce day differences
- annual forecast variance is taken from the source release
- missing and exempt values remain not comparable

The attention score is a review queue, not a probability of failure:

```text
+35 current Red published DCA
+25 rating worsened
+0 to +20 later published end date
+0 to +15 material in-year variance
-8 no public evidence excerpt
```

Narrative themes use the visible keyword taxonomy in [`Processing/gmpp_pipeline.py`](Processing/gmpp_pipeline.py). They organise evidence but do not establish causation.

Whole-life cost values are not compared across years because published price bases can differ.

The XER materiality score is also deterministic. It ranks milestone movement, date movement, float erosion, relationship logic, constraints and duration changes. It is a review priority, not a forecast or a substitute for a validated critical-path or quantitative risk analysis.

## Try the XER workflow

Open [the live schedule evidence review](https://matthewpaver.github.io/ProjectLens/schedule-review.html) and select **Run the Northstar demo**. The synthetic pair is safe to share and intentionally contains:

- a 73-day project finish movement
- changed logic and constraints
- float erosion and integrity findings
- a status narrative that understates the schedule movement

You can then switch between executive and analyst views, inspect each material change, save assurance actions in the browser and download the complete review pack as JSON.

The repository also includes a [paced, silent product walkthrough](docs/assets/projectlens-evidence-demo.webm) for the LinkedIn launch.

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

Tests cover the original schedule-processing modules, public-data counts, DCA precedence, matching, transitions, theme classification, score boundaries and the synthetic XER evidence contract. Playwright browser checks cover the public workflow and the XER demo on desktop and mobile.

## Competitive position

nPlan, SmartPM, Nodes & Links and InEight already provide strong private schedule analytics, forecasting, quantitative risk and enterprise controls. ProjectLens deliberately does not imitate them.

Its narrower gap is **open, source-linked, longitudinal public delivery intelligence**, plus an evidence-linked assurance workflow that can complement existing planning tools. The dated market scan and source notes are in [`competitor-profiles/`](competitor-profiles).

## Boundaries

- Delivery Confidence Assessments are point-in-time judgements, not outcomes.
- Annual portfolio data cannot support activity-level critical-path forecasting.
- Theme matching does not prove root cause.
- Comparable cases prompt investigation and do not prescribe an intervention.
- Absence from a later annual release does not establish whether a project delivered, closed, changed identifier or left the portfolio for another reason.
- XER comparison identifies observable changes and integrity questions. It does not reproduce Primavera scheduling calculations or prove delay attribution.
- A production internal version would require secure schedule connectors, permissions, audit logs and organisation-specific validation.

## Legacy schedule engine

The repository still contains the earlier schedule-processing demonstrator under [`Processing/analysis/`](Processing/analysis) and the local Flask interface under [`Website/`](Website). Its bundled Alpha, Beta and other project records are synthetic test fixtures. They are retained for engineering tests, not used as evidence in the public ProjectLens product.

## Repository layout

```text
Data/public/             official annual GMPP source files
Processing/gmpp_pipeline.py
                         longitudinal preparation and validation
docs/                    GitHub Pages product and generated JSON
docs/schedule-review.*   browser-local XER evidence review
docs/demo/               synthetic, share-safe XER comparison pair
competitor-profiles/     dated market scan and source notes
Processing/analysis/     legacy schedule-analysis modules
Processing/tests/        deterministic and integration tests
```

## License

ProjectLens code is MIT licensed. Government source data is used under the Open Government Licence. See each official publication for source terms.
