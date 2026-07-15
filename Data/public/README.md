# Government Major Projects Portfolio data

ProjectLens uses seven annual UK Government Major Projects Portfolio releases under the Open Government Licence.

| Local file | Official publication |
| --- | --- |
| `raw/gmpp_2019_20.csv` | [IPA Annual Report 2019-20](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2020) |
| `raw/gmpp_2020_21.csv` | [IPA Annual Report 2020-21](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2021) |
| `raw/gmpp_2021_22.csv` | [IPA Annual Report 2021-22](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022) |
| `raw/gmpp_2022_23.csv` | [IPA Annual Report 2022-23](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022-23) |
| `raw/gmpp_2023_24.csv` | [IPA Annual Report 2023-24](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2023-24) |
| `raw/gmpp_2024_25.csv` | [NISTA Annual Report 2024-25](https://www.gov.uk/government/publications/nista-annual-report-2024-2025) |
| `raw/gmpp_2025_26.csv` | [NISTA Major Projects Annual Report 2025-26](https://www.gov.uk/government/publications/nista-major-projects-annual-report-2025-26) |

Run `python Processing/gmpp_pipeline.py` to create `docs/data/gmpp.json`.

Exact dates, rating transitions and annual variances are calculated deterministically. Narrative themes use the transparent keyword taxonomy in the pipeline. They are signals for review, not causal findings. Whole-life costs are shown for context but are not compared between years because published price bases can differ.

For each annual record, ProjectLens follows the official reporting convention used in the source release:

1. use the independent IPA or NISTA Delivery Confidence Assessment when it is published
2. otherwise use the SRO's Q4 Delivery Confidence Assessment
3. retain missing or exempt assessments as not published

Both original fields and the selected source are retained in the generated JSON so the choice is inspectable.

The current longitudinal release contains 1,417 records and 189 current projects. A stable-ID comparison finds 43 records from 2024-25 that are absent from 2025-26, while the NISTA annual report describes 42 portfolio leavers. ProjectLens displays that difference instead of silently reconciling it. Absence alone does not establish an eventual project outcome.

The 2019-20 consolidated CSV is published with projects in columns. The pipeline transposes that release in memory, then applies the same stable-ID and field-selection contract used for later releases. The earlier 2018-19 consolidated CSV does not publish the same stable GMPP identifier, so it is intentionally excluded from the longitudinal join rather than matched by project name.

The product's external evidence register also links to the official major-project archive, UK Infrastructure Pipeline, NAO, Public Accounts Committee, Contracts Finder and Evaluation Registry. These are discovery routes, not project-specific evidence joins. ProjectLens only presents a source as evidence for a project when that relationship has been established.
