# ProjectLens

ProjectLens is built for the reviewer who has to tell a project board whether a change pack can be trusted — before the meeting, using only the pack's own evidence.

## The problem

A project board usually decides on the polish of the pack, because nobody has time to reconcile the narrative against the schedules, risks, actions and prior conditions submitted alongside it. That is how a green status narrative claiming "no change to the finish date" gets approved while the current schedule has actually moved it by 73 days — exactly the failure the bundled Northstar example reproduces, alongside a high risk with no accountable owner, an overdue action and a prior approval condition still open. When the evidence disagrees with the story and no one checks, the board approves the story.

**Who it's for:** the project controls reviewer or PMO lead preparing a board decision on a change pack.

**What you get:**

- **Source-linked conflicts and gaps** — each finding names the evidence item that produced it (current pack, previous pack, RAID, commitments, schedule), so the board question writes itself.
- **A prepared decision, not a dashboard** — one readiness verdict, at most three blockers, and the specific questions to send for answers before the meeting.
- **A durable record** — the human decision, its owner, rationale and approval conditions are preserved, and each condition stays open until it is closed or formally waived.

<div align="center">

![ProjectLens change assurance workspace comparing a change pack narrative against its schedule evidence](docs/assets/change-assurance-overview.png)

[**2-minute walkthrough**](docs/assets/projectlens-evidence-demo.mp4) — a paced, silent MP4 product tour.

![Python](https://img.shields.io/badge/Python-3.11-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Public data](https://img.shields.io/badge/Data-UK_GMPP-d7ff4f?style=flat-square)
![License](https://img.shields.io/badge/Code-MIT-blue?style=flat-square)
[![Validate](https://github.com/MatthewPaver/ProjectLens/actions/workflows/validate.yml/badge.svg)](https://github.com/MatthewPaver/ProjectLens/actions/workflows/validate.yml)

[Prepare a board review](https://matthewpaver.github.io/ProjectLens/board-readiness.html) · [Review a change pack](https://matthewpaver.github.io/ProjectLens/change-assurance.html) · [Explore public evidence](https://matthewpaver.github.io/ProjectLens/) · [Open the detailed XER review](https://matthewpaver.github.io/ProjectLens/schedule-review.html) · [Portfolio](https://matthewpaver.github.io/MatthewPaver/store/)

</div>

## Non-goals

- It does not establish contractual entitlement, delay causation or a probability of failure — findings are prompts for human verification, not claims.
- It does not replace the board. The workflow ends by recording a human decision with its rationale and conditions; the system never makes the decision.
- It does not read Microsoft Project (MSP) or PDF schedules. The browser workflows parse Primavera P6 XER exports and CSV evidence only.

## Why this design

The central trade-off is browser-local XER comparison instead of a hosted platform. Schedule submissions are commercially sensitive, so the comparison runs entirely in the browser: XER files, decisions and conditions never leave the machine, which also means a reviewer can evaluate the product against a real pack with no install, no account and no data-sharing approval. The cost is deliberate: records persist only in one browser's local storage, there is no multi-user collaboration, and there are no server-side schedule connectors, permissions or organisational audit logs — a production internal version would need all of those (see Boundaries). That trade is acceptable because the target user is a single reviewer preparing one board meeting, and the fastest route to trust is letting them test it on private data without asking anyone's permission.

## Why it exists

A project board often receives a polished narrative alongside schedules, risks, actions and prior conditions that do not fully agree. The primary workflow reduces that job to four steps:

1. set the board date and review owner;
2. register the current pack, previous pack, RAID, commitments and schedule;
3. check each conflict or gap against its named source;
4. prepare the board questions, record the human decision and track conditions.

The public Northstar example contains a green narrative that says the finish is unchanged while the current XER moves it by 73 days, a high risk has no accountable owner, an action is overdue and a prior approval condition remains open. The fixture is explicitly synthetic and safe to share. Users can also register their own evidence locally in the browser; the detailed schedule module parses XER and CSV evidence without uploading it.

ProjectLens and DecisionGraph remain inspectable supporting tools. Users do not have to choose between them to complete a review.

## Supporting evidence products

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

- Is the submission complete enough to support the claimed conclusion?
- Which five to ten changes matter after thousands of raw movements are reduced by materiality?
- Which contractual milestones and exported critical or near-critical activities moved?
- Do the schedule, baseline, risk register, decision log and status narrative agree?
- Did the intervention recorded last month work in the latest submission?

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

1. **Board readiness** checks five familiar evidence groups and presents source-linked findings and decisions required.
2. **MeetingProof** captures commitments before they become untraceable project drift.
3. **DecisionGraph** retrieves comparable decisions and measured outcomes without claiming they prescribe the answer.
4. **Change assurance** presents one readiness decision, no more than three blockers and the next human action.
5. **Decision and follow-up registers** preserve authority, rationale, unresolved warnings and approval conditions.
6. **Detailed XER assurance** checks evidence completeness, compares submissions and a separate baseline, reconciles decisions and risks, and follows interventions into later updates.
7. **Public evidence** provides the longitudinal GMPP briefing, explorer, case match and outcome follow-up register.
8. **Method** publishes the exact score, data boundaries, original releases and authoritative expansion routes.

The interfaces are static GitHub Pages applications in [`docs/`](docs). No server or account is required. XER files, decisions and conditions remain in the browser for the public demonstrator.

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

The XER materiality score is also deterministic. It ranks milestone movement, date movement, float erosion, relationship logic, constraints, duration changes and business-critical milestone language. It first separates raw differences from material changes, then limits the executive queue to the highest-priority items while retaining the full analyst view.

Evidence coverage is reported separately from schedule quality. A missing baseline, risk register, schedule basis or decision log is visible rather than silently treated as evidence. The critical and near-critical view uses total float exported by P6. It does not recalculate a CPM network. The score is a review priority, not a forecast or a substitute for validated schedule logic or quantitative risk analysis.

## Try the XER workflow

Open [the live schedule evidence review](https://matthewpaver.github.io/ProjectLens/schedule-review.html) and select **Run the Northstar demo**. The synthetic pair is safe to share and intentionally contains:

- a 73-day project finish movement
- 22 raw changes reduced to 9 material changes and 8 executive priorities
- a separate baseline, risk register, schedule basis and decision log
- changed logic, constraints, float erosion and integrity findings
- an approved change, an unlinked change and a deferred decision
- a status narrative that understates the schedule movement
- a previous intervention that can be assessed against the later submission

You can switch between executive and analyst views, inspect why each change was prioritised, save assurance actions in the browser and download the complete evidence-linked review pack as JSON.

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
make browser-test
```

Tests cover the board-readiness human gate, original schedule-processing modules, public-data counts, DCA precedence, matching, transitions, theme classification, score boundaries and the synthetic XER evidence contract. Playwright checks exercise the board review, one-click change and XER demos, real browser file inputs, exported assurance packs, and desktop and mobile layouts.

The legacy Python pipeline reads optional local settings from `config.json`.
Copy [`config.example.json`](config.example.json) when exercising that path; the
public browser products require no configuration or credentials.

## Competitive position

nPlan, SmartPM, Nodes & Links and InEight already provide strong private schedule analytics, forecasting, quantitative risk and enterprise controls. ProjectLens deliberately does not imitate them.

Its narrower gap is **open, source-linked, longitudinal public delivery intelligence**, plus an evidence-linked assurance workflow that asks what an XER does not contain, reconciles the submission with other project evidence and tracks whether responses worked. It is designed to complement existing planning tools. The dated market scan and source notes are in [`competitor-profiles/`](competitor-profiles).

## Boundaries

- Delivery Confidence Assessments are point-in-time judgements, not outcomes.
- Annual portfolio data cannot support activity-level critical-path forecasting.
- Theme matching does not prove root cause.
- Comparable cases prompt investigation and do not prescribe an intervention.
- Absence from a later annual release does not establish whether a project delivered, closed, changed identifier or left the portfolio for another reason.
- XER comparison identifies observable changes and integrity questions. It does not reproduce Primavera scheduling calculations or prove delay attribution.
- Baseline assurance requires a separately supplied baseline because XER does not reliably carry baseline project data. Risk and decision links use explicit activity-code matching and remain prompts for human verification.
- If an XER contains multiple projects, the browser demonstrator analyses the first project and reports that scope limitation.
- A production internal version would require secure schedule connectors, permissions, audit logs and organisation-specific validation.

Oracle describes XER as a proprietary exchange format, notes that baseline project data is not supported in XER export, and documents differences in risk and financial-period transfer. Those format constraints are why the workflow begins with an evidence-completeness report. See Oracle's [supported file formats](https://docs.oracle.com/cd/F51303_01/English/admin/p6_pro_importing_exporting/import_export_file_formats.htm), [XER export notes](https://docs.oracle.com/cd/E75426_01/English/User_Guides/p6_pro_user/export_projects_to_an_xer_file.htm) and [risk import guidance](https://docs.oracle.com/cd/G48897_01/p6help/en/101760.htm).

## Legacy schedule engine

The repository still contains the earlier schedule-processing demonstrator under [`Processing/analysis/`](Processing/analysis) and the local Flask interface under [`Website/`](Website). Its bundled Alpha, Beta and other project records are synthetic test fixtures. They are retained for engineering tests, not used as evidence in the public ProjectLens product.

## Repository layout

```text
Data/public/             official annual GMPP source files
Processing/gmpp_pipeline.py
                         longitudinal preparation and validation
docs/                    GitHub Pages product and generated JSON
docs/schedule-review.*   browser-local XER evidence review
docs/demo/               synthetic, share-safe XER and evidence kit
competitor-profiles/     dated market scan and source notes
Processing/analysis/     legacy schedule-analysis modules
Processing/tests/        deterministic and integration tests
```

## Related work

Supporting products in this repository and alongside it provide meeting follow-up (MeetingProof), decision memory ([DecisionGraph](https://github.com/MatthewPaver/DecisionGraph)), longitudinal UK major-project evidence (the GMPP explorer in [`docs/`](docs)) and a detailed browser-local Primavera P6 XER assurance workflow ([schedule review](https://matthewpaver.github.io/ProjectLens/schedule-review.html)). None of them is required to complete a change review.

## History

The arc, from the repository's own log:

- **2025-04 to 2025-05** — repository created; initial schedule-processing pipeline and data uploads.
- **2025-11 to 2026-03** — housekeeping: README restructure, MIT licence, `.gitignore`, `requirements.txt`, standardised setup.
- **2026-05** — validation workflow, CI badge, reviewer notes and packaging; portfolio quick read.
- **2026-07-14 to 2026-07-15** — rebuilt as an evidence-linked assurance product: risk command centre, self-serve evidence room, XER comparison turned into evidence assurance, change assurance workflow.
- **2026-07-20** — project evidence desk built.
- **2026-07-27** — [v1.0.0 release](https://github.com/MatthewPaver/ProjectLens/releases/tag/v1.0.0).
- **2026-07-28** — browser workflows verified end-to-end and generated data removed from the tree.

## License

ProjectLens code is MIT licensed. Government source data is used under the Open Government Licence. See each official publication for source terms.
