"""End-to-end browser checks for the local-only XER evidence review."""

from pathlib import Path
import json

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:8765/schedule-review.html"


def run_desktop(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page.add_init_script("""
      localStorage.setItem('projectlens:schedule-actions:v1', JSON.stringify([{
        id: 'prior-northstar-action',
        projectId: '1000',
        projectName: 'NORTHSTAR GRID CONNECTION',
        title: 'Recover ready-for-service commitment',
        evidence: 'NS-900 milestone movement',
        owner: 'Project Director',
        due: '2026-02-20',
        done: false,
        changeCode: 'NS-900',
        sourceScore: 55,
        sourceDataDate: '2026-01-31',
        outcomeOverride: '',
        createdAt: '2026-01-31T12:00:00Z'
      }]));
    """)
    errors = []
    failures = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("requestfailed", lambda request: failures.append(f"{request.method} {request.url}"))
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    assert page.locator("#runReview").is_disabled()

    page.get_by_role("button", name="Run the Northstar demo").click()
    page.locator("#reviewResults").wait_for(state="visible")
    assert page.locator("#reviewProjectName").inner_text() == "NORTHSTAR GRID CONNECTION"
    assert page.locator("#finishMovement").inner_text() == "+73d"
    assert int(page.locator("#materialChangeCount").inner_text()) >= 5
    assert int(page.locator("#contradictionCount").inner_text()) >= 2
    assert page.locator("#questionList .question-card").count() >= 3
    assert page.locator("#evidenceCoverage").inner_text() == "9/11"
    assert page.locator("#completenessGrid .completeness-card").count() == 11
    assert page.locator("#completenessGrid .completeness-card.missing").count() == 2
    assert page.locator("#pathCandidateList article").count() >= 5
    assert page.locator("#decisionReconciliation article.linked").count() >= 1
    assert page.locator("#decisionReconciliation article.unlinked").count() >= 1
    assert "NS-900" in page.locator("#baselineMovementList").inner_text()
    assert "REPEATED OR WORSENED" in page.locator("#followThroughList").inner_text().upper()
    assert "raw changes" in page.locator("#materialFunnelCopy").inner_text().lower()
    executive_change_count = page.locator("#changeList .change-row").count()
    assert 5 <= executive_change_count <= 10

    page.get_by_role("button", name="Analyst", exact=True).click()
    assert page.locator("#fitnessGrid .fitness-item").count() == 7
    assert page.locator("#integrityList .integrity-item").count() >= 3
    assert page.locator("#changeList .change-row").count() >= executive_change_count

    page.locator("#changeList .change-row").first.click()
    page.locator("#changeDialog[open]").wait_for()
    assert page.locator("#changeDialogEvidence > div").count() >= 1
    page.get_by_role("button", name="Add assurance action").click()
    assert page.locator("#actionList .action-row").count() >= 1

    with page.expect_download() as download_info:
        page.get_by_role("button", name="Download review pack").click()
    assert download_info.value.suggested_filename.endswith("-schedule-review.json")
    pack = json.loads(Path(download_info.value.path()).read_text())
    assert pack["completeness"]["presentCount"] == 9
    assert pack["rawChangeCount"] > len(pack["materialChanges"])
    assert len(pack["executivePriorities"]) <= 10
    assert pack["executivePriorities"]
    assert pack["pathCandidates"]
    assert pack["decisionReconciliation"]
    assert pack["baselineMovements"]
    assert pack["interventionOutcomes"]

    page.screenshot(path=str(ROOT / "docs" / "assets" / "schedule-review-overview.png"), full_page=True)
    assert not errors, errors
    assert not failures, failures
    page.close()


def run_mobile(browser):
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
    page.get_by_role("button", name="Run the Northstar demo").click()
    page.locator("#reviewResults").wait_for(state="visible")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
    assert page.locator("#changeList .change-row").count() >= 1
    page.screenshot(path=str(ROOT / "docs" / "assets" / "schedule-review-mobile.png"), full_page=False)
    page.close()


def run_uploaded_evidence(browser):
    page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    demo = ROOT / "docs" / "demo"
    page.locator("#previousFile").set_input_files(str(demo / "northstar-previous.xer"))
    page.locator("#currentFile").set_input_files(str(demo / "northstar-current.xer"))
    page.locator("#baselineFile").set_input_files(str(demo / "northstar-baseline.xer"))
    page.locator("#riskFile").set_input_files(str(demo / "northstar-risks.csv"))
    page.locator("#basisFile").set_input_files(str(demo / "northstar-schedule-basis.md"))
    page.locator("#decisionFile").set_input_files(str(demo / "northstar-decisions.csv"))
    page.locator("#narrativeText").fill((demo / "northstar-narrative.txt").read_text())
    assert page.locator("#runReview").is_enabled()
    page.locator("#runReview").click()
    page.locator("#reviewResults").wait_for(state="visible")

    assert page.locator("#baselineFileName").inner_text() == "northstar-baseline.xer"
    assert page.locator("#riskFileName").inner_text() == "northstar-risks.csv"
    assert page.locator("#evidenceCoverage").inner_text() == "9/11"
    assert page.locator("#finishMovement").inner_text() == "+73d"
    assert not errors, errors
    page.close()


with sync_playwright() as playwright:
    chromium = playwright.chromium.launch(headless=True)
    run_desktop(chromium)
    run_uploaded_evidence(chromium)
    run_mobile(chromium)
    chromium.close()
    print("XER review browser test passed: demo, analysis, actions, export and mobile")
