# Method, evidence base and market position

Detail moved out of the main [README](../README.md) so the front page stays short. Nothing here is required to use the product.

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

Narrative themes use the visible keyword taxonomy in [`Processing/gmpp_pipeline.py`](../Processing/gmpp_pipeline.py). They organise evidence but do not establish causation.

Whole-life cost values are not compared across years because published price bases can differ.

The XER materiality score is also deterministic. It ranks milestone movement, date movement, float erosion, relationship logic, constraints, duration changes and business-critical milestone language. It first separates raw differences from material changes, then limits the executive queue to the highest-priority items while retaining the full analyst view.

Evidence coverage is reported separately from schedule quality. A missing baseline, risk register, schedule basis or decision log is visible rather than silently treated as evidence. The critical and near-critical view uses total float exported by P6. It does not recalculate a CPM network. The score is a review priority, not a forecast or a substitute for validated schedule logic or quantitative risk analysis.

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

The latest snapshot was reported by departments at 31 March 2026. Source links and licensing notes are in [`Data/public/README.md`](../Data/public/README.md).

## Competitive position

nPlan, SmartPM, Nodes & Links and InEight already provide strong private schedule analytics, forecasting, quantitative risk and enterprise controls. ProjectLens deliberately does not imitate them.

Its narrower gap is open, source-linked, longitudinal public delivery intelligence, plus an evidence-linked assurance workflow that asks what an XER does not contain, reconciles the submission with other project evidence and tracks whether responses worked. It is designed to complement existing planning tools. The dated market scan and source notes are in [`competitor-profiles/`](../competitor-profiles).

## History

The arc, from the repository's own log:

- **2025-04 to 2025-05** — repository created; initial schedule-processing pipeline and data uploads.
- **2025-11 to 2026-03** — housekeeping: README restructure, MIT licence, `.gitignore`, `requirements.txt`, standardised setup.
- **2026-05** — validation workflow, CI badge, reviewer notes and packaging; portfolio quick read.
- **2026-07-14 to 2026-07-15** — rebuilt as an evidence-linked assurance product: risk command centre, self-serve evidence room, XER comparison turned into evidence assurance, change assurance workflow.
- **2026-07-20** — project evidence desk built.
- **2026-07-27** — [v1.0.0 release](https://github.com/MatthewPaver/ProjectLens/releases/tag/v1.0.0).
- **2026-07-28** — browser workflows verified end-to-end and generated data removed from the tree.
