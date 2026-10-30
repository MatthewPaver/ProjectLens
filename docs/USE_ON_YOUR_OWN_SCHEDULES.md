# Use ProjectLens on your own schedules

Give ProjectLens two exports of the same schedule, one previous and one current, plus the written progress update. Use Primavera P6 XER/XML or MS Project XML.

You get the finish movement, the exact sentence and highlighted phrase behind a finding, and both dates. An explained move stays a review of the schedule facts. A mismatch without a finish move asks you to check the published date. These are review prompts, not proof of cause or approval.

Files stay on your machine. No account or upload is needed.

## 1. Quickest route: command line

From the repository root, with Node.js 18 or later, paste your update into a text file and supply two exports:

```bash
cat > /tmp/projectlens-update.txt <<'TEXT'
Paste your written progress update here.
TEXT
node bin/projectlens.mjs compare \
  --previous path/to/previous.xer \
  --current path/to/current.xer \
  --narrative /tmp/projectlens-update.txt \
  --format text
```

For a working synthetic example, use `examples/monthly/northstar-previous.xer` and `examples/monthly/northstar-current.xer`, and paste `The programme remains on track.` The same command reports a 73-day finish movement, quotes the sentence, marks `**remains on track**`, and exits 1.

Formats: `text`, `json`, `csv`, `md`. Text and Markdown show the quote; JSON includes structured `claims.evidence`. CSV keeps its history columns. Use `--project <id>` for multi-project XERs. CLI narrative files must be `.txt` or `.md`; convert Word text first.

**Accuracy:** set 4 is **28/30, seen**, reproduced with `npm run bench`. Set 5's independent score is **pending** because no set 5 file is available here. [Cases, labels and sources](../bench/real-cases/) are public. Sets 1–4 informed the rules; do not treat them as unseen accuracy.

## 2. Browser

1. Open [change assurance](change-assurance.html) (or serve `docs/` locally).
2. Export two schedules from Primavera P6 as **XER** or **XML**, or from MS Project as **XML** (File → Save As → XML). Do not load `.mpp` directly.
3. Choose **Use your own schedules**, pick previous and current files, paste the narrative (or load `.txt` / `.md` / `.docx` on schedule review).
4. Select **Check decision readiness**. If activity codes overlap but names or IDs differ, confirm the match when asked. Multi-project files ask which project to use.

Local server:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
# open http://127.0.0.1:8000/change-assurance.html
```

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
- Whether published government commentary is “wrong”; a mismatch says the published commentary and date disagree

## Privacy

Browser and CLI keep files local. The static site does not probe localhost for APIs. The optional precedent sidecar receives narrative text only, never the schedule file.

## Exit codes (CLI)

| Code | Meaning |
|---|---|
| 0 | Comparison finished; no narrative/date contradiction |
| 1 | Comparison finished; at least one contradiction |
| 3 | Input error (missing file, unsupported format, parse failure) |
