"""Browser parser format contracts using synthetic variants, never real client XERs."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
fixture = (ROOT / "docs/demo/northstar-previous.xer").read_text()

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    for page_name, api in [("change-assurance.html", "ProjectLensChangeAssurance"), ("schedule-review.html", "ProjectLensXer")]:
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:8765/{page_name}")
        page.wait_for_load_state("networkidle")
        results = page.evaluate(r"""({text, api}) => {
          const parse = window[api].parseXer;
          const original = parse(text, 'synthetic.xer');
          const variants = [text, '\ufeff' + text, text.replace(/\n/g, '\r\n'), text.replace(/\n/g, '\r')];
          for (const variant of variants) {
            const result = parse(variant, 'synthetic-format-variant.xer');
            if (result.tasks.length !== original.tasks.length || result.project.name !== original.project.name) throw Error('Line ending/BOM regression');
          }
          let fieldCount = 0;
          const reordered = text.split('\n').map(line => {
            const cells = line.split('\t');
            if (cells[0] === '%F') fieldCount = cells.length - 1;
            if (cells[0] === '%R') while (cells.length < fieldCount + 1) cells.push('');
            return ['%F', '%R'].includes(cells[0]) ? [cells[0], ...cells.slice(1).reverse()].join('\t') : line;
          }).join('\n');
          if (parse(reordered, 'synthetic-reordered.xer').tasks.length !== original.tasks.length) throw Error('Column order regression');
          for (const bad of ['', 'not an XER', '%T\tPROJECT\n%F\tproj_id\n%R\t1']) {
            let rejected = false;
            try { parse(bad, 'invalid.xer'); } catch { rejected = true; }
            if (!rejected) throw Error('Invalid input accepted');
          }
          if (api === 'ProjectLensChangeAssurance') {
            const multi = text.replace(/(%T\tPROJECT\n%F[^\n]+\n)(%R[^\n]+\n)/, '$1$2$2');
            let message = '';
            try { parse(multi, 'synthetic-multiple-projects.xer'); } catch (error) { message = error.message; }
            if (!/single project/i.test(message)) throw Error('Change assurance must reject multi-project scope explicitly');
          } else {
            const empty = window.ProjectLensXer.parseEvidenceText('', 'decision');
            if (empty.decisions.length || empty.activityCodes.length) throw Error('Empty CSV fabricated evidence');
            const invalid = window.ProjectLensXer.parseEvidenceText('unrelated,values\nfoo,bar', 'decision');
            if (invalid.decisions.length || invalid.activityCodes.length) throw Error('Unrelated CSV fabricated a linked decision');
            const csv = window.ProjectLensXer.parseEvidenceText('decision_id,status,activity_code\nD-1,Approved,NS-900', 'decision');
            if (csv.decisions.length !== 1 || csv.decisions[0].codes[0] !== 'NS-900') throw Error('Simple CSV decision contract failed');
          }
          return {tasks: original.tasks.length, variants: variants.length, invalidInputs: 3};
        }""", {"text": fixture, "api": api})
        print(page_name, results)
        page.close()
    browser.close()
