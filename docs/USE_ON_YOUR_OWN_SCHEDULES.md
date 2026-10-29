# Use ProjectLens on your own schedules

Three routes. Files stay on your machine. No account and no upload.

## 1. Browser

1. Open [change assurance](change-assurance.html) (or serve `docs/` locally).
2. Export two schedules from Primavera P6 as **XER** or **XML**, or from MS Project as **XML** (File → Save As → XML). Do not load `.mpp` directly.
3. Choose **Use your own schedules**, pick previous and current files, paste the narrative (or load `.txt` / `.md` / `.docx` on schedule review).
4. Select **Check decision readiness**. If activity codes overlap but names or IDs differ, confirm the match when asked. Multi-project files ask which project to use.

Local server:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
# open http://127.0.0.1:8000/change-assurance.html
```

## 2. Command line

```bash
node bin/projectlens.mjs compare \
  --previous path/to/previous.xer \
  --current path/to/current.xer \
  --narrative path/to/narrative.txt \
  --format json
```

Formats: `text`, `json`, `csv`, `md`. Optional `--project <id>` for multi-project XERs.

## 3. Monthly batch

Put successive exports for one or more projects in a folder (plus an optional `narrative.txt`):

```bash
node bin/projectlens.mjs batch --dir examples/monthly --format csv --out history.csv
```

Files are grouped by activity-code overlap, ordered by data date, and each consecutive pair writes one history row:

`project,previous_date,current_date,finish_previous,finish_current,days_moved,narrative_claim,contradiction`

## Export tips

| Tool | Format | How |
|---|---|---|
| Primavera P6 | XER or XML | File → Export → Primavera XER, or XML |
| MS Project | XML (MSPDI) | File → Save As → XML |
| MS Project `.mpp` | Not accepted | Export as XML first |
| Narrative | `.txt`, `.md`, `.docx` | Paste text, or load the file in schedule review |

## What it checks

- Scheduled finish movement against stability / move wording in the narrative
- Activity finish, float, constraint and relationship differences
- Ownership gaps when mitigation language has no owner or due date

## What it does not check

- Cause, entitlement or contractual approval
- A recalculated critical path (it uses exported float only)
- Whether a published government commentary is “wrong” — mismatches are stated as commentary and date disagreeing

## Privacy

Browser and CLI keep files local. The static site does not probe localhost for APIs. The optional precedent sidecar receives narrative text only, never the schedule file.

## Exit codes (CLI)

| Code | Meaning |
|---|---|
| 0 | Comparison finished; no narrative/date contradiction |
| 1 | Comparison finished; at least one contradiction |
| 3 | Input error (missing file, unsupported format, parse failure) |
