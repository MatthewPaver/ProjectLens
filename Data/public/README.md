# Government Major Projects Portfolio data

ProjectLens uses four annual UK Government Major Projects Portfolio releases under the Open Government Licence.

| Local file | Official publication |
| --- | --- |
| `raw/gmpp_2022_23.csv` | [IPA Annual Report 2022-23](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022-23) |
| `raw/gmpp_2023_24.csv` | [IPA Annual Report 2023-24](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2023-24) |
| `raw/gmpp_2024_25.csv` | [NISTA Annual Report 2024-25](https://www.gov.uk/government/publications/nista-annual-report-2024-2025) |
| `raw/gmpp_2025_26.csv` | [NISTA Major Projects Annual Report 2025-26](https://www.gov.uk/government/publications/nista-major-projects-annual-report-2025-26) |

Run `python Processing/gmpp_pipeline.py` to create `docs/data/gmpp.json`.

Exact dates, rating transitions and annual variances are calculated deterministically. Narrative themes use the transparent keyword taxonomy in the pipeline. They are signals for review, not causal findings. Whole-life costs are shown for context but are not compared between years because published price bases can differ.
