"""End-to-end browser checks for the simplified project change assurance workflow."""

from pathlib import Path
import os

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = Path(os.environ.get("PROJECTLENS_SCREENSHOT_DIR", "/tmp/projectlens-browser"))
SCREENSHOTS.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://127.0.0.1:8765/change-assurance.html"


def run_desktop(browser):
    page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1, reduced_motion="reduce")
    errors = []
    failures = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    sidecar_requests = []
    page.on("requestfailed", lambda request: failures.append(f"{request.method} {request.url}"))
    page.on("request", lambda request: sidecar_requests.append(request.url) if ":8787" in request.url else None)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    assert page.locator("[data-view-link]").count() == 3
    assert page.get_by_role("heading", name="The schedule changed. Did the update explain it?").is_visible()
    page.get_by_role("button", name="Try the Northstar example").click()
    page.locator("#readinessWorkspace").wait_for(state="visible")
    readiness_status = page.locator("#readinessStatus").inner_text()
    assert readiness_status.casefold() == "needs evidence before decision", readiness_status
    assert page.locator("#blockerList .blocker").count() == 3
    assert "3 findings" in page.locator("#readinessHeadline").inner_text()
    assert "26 Feb 2028" in page.locator("#findingEvidence").inner_text()
    assert page.locator("#showDecisionForm").bounding_box()["y"] < 720
    assert page.locator("#supportingEvidence").get_attribute("open") is None
    assert page.locator("#evidenceSummary .evidence-fact").count() == 6
    assert not sidecar_requests, sidecar_requests
    page.screenshot(path=str(SCREENSHOTS / "change-assurance-overview.png"), full_page=False)
    page.locator("[data-finding='1']").click()
    assert "constraint" in page.locator("#findingEvidence").inner_text().lower()

    page.get_by_role("button", name="Draft questions for the team").click()
    page.locator("#requestPanel").wait_for(state="visible")
    page.locator("#requestOwner").fill("Planning Manager")
    page.locator("#requestDue").fill("2026-07-22")
    page.get_by_role("button", name="Save evidence request").click()
    assert page.evaluate("JSON.parse(localStorage.getItem('projectlens:evidence-requests:v1')).length") == 1

    page.get_by_role("button", name="Record human decision").click()
    page.locator("#decisionPanel").wait_for(state="visible")
    # Supplemental context must never force the reviewer to dismiss unrelated examples.
    assert page.locator("#decisionForm button[type='submit']").is_enabled()
    assert "not reviewed" in page.locator("#precedentGateBanner").inner_text()
    page.get_by_label("Approve with conditions").check(force=True)
    page.locator("#decisionOwner").fill("Change Authority Chair")
    page.locator("#decisionRationale").fill("Proceed only after the named controls are confirmed and owned.")
    page.locator("#decisionCondition").fill("Protect the regression-test window")
    page.locator("#conditionOwner").fill("Test Lead")
    page.locator("#conditionDue").fill("2026-07-29")
    page.get_by_role("button", name="Save draft on this device").click()
    page.reload()
    page.get_by_role("button", name="Resume saved draft").click()
    assert page.locator("#decisionOwner").input_value() == "Change Authority Chair"
    assert page.locator("#conditionOwner").input_value() == "Test Lead"
    page.get_by_role("button", name="Save decision record").click()
    page.wait_for_url("**#decisions")
    page.locator("#decisionRegister .register-row").wait_for(state="visible")
    assert page.locator("#decisionRegister .register-row").count() == 1
    stored = page.evaluate("JSON.parse(localStorage.getItem('projectlens:change-decisions:v1'))[0]")
    assert stored.get("precedentsNotReviewed") and len(stored["precedentsNotReviewed"]) == 3, stored
    assert not stored["precedentsUsed"], stored
    assert any(finding["type"] == "Ownership gap" for finding in stored["findings"])
    assert page.evaluate("localStorage.getItem('projectlens:review-draft:v1')") is None
    assert len(stored["findings"]) >= 3, stored

    page.evaluate(
        """() => {
            window.__copied = null;
            Object.defineProperty(navigator, "clipboard", {
                configurable: true,
                value: { writeText: text => { window.__copied = text; return Promise.resolve(); } },
            });
        }"""
    )
    page.get_by_role("button", name="Export record").click()
    page.wait_for_function("() => window.__copied !== null")
    copied = page.evaluate("() => window.__copied")
    assert "Verdict: Approve with conditions" in copied, copied
    assert "Public-record context not reviewed:" in copied, copied
    assert "northstar-current.xer" in copied, copied
    assert "relationship:" in copied and "hours" in copied, copied
    assert "Decision owner: Change Authority Chair" in copied, copied
    assert "Protect the regression-test window" in copied, copied
    assert page.locator("#exportFallback").is_hidden()

    page.evaluate(
        'Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined })'
    )
    page.get_by_role("button", name="Export record").click()
    page.locator("#exportFallback").wait_for(state="visible")
    fallback_text = page.locator("#exportFallbackText").input_value()
    assert "Rationale: Proceed only after the named controls are confirmed and owned." in fallback_text

    page.locator('a[href="#follow-up"]').click()
    assert page.locator("#conditionRegister .register-row").count() == 1
    assert page.locator("#conditionNavCount").inner_text() == "1"
    assert page.locator("#conditionNavCount").is_visible()
    page.get_by_role("button", name="Mark closed").click()
    assert page.locator("#conditionNavCount").is_hidden()

    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Try the Northstar example").click()
    page.locator("#readinessWorkspace").wait_for(state="visible")
    page.evaluate("document.querySelector('#readinessWorkspace').scrollIntoView()")
    page.screenshot(path=str(SCREENSHOTS / "change-assurance-overview.png"), full_page=False)
    assert not errors, errors
    assert not failures, failures
    page.close()


def run_mobile(browser):
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1, reduced_motion="reduce")
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
    assert page.locator("[data-view-link]").count() == 3
    page.get_by_role("button", name="Try the Northstar example").click()
    page.locator("#readinessWorkspace").wait_for(state="visible")
    assert page.locator("#blockerList .blocker").count() == 3
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
    page.locator("[data-finding='2']").scroll_into_view_if_needed()
    finding_scroll = page.evaluate("window.scrollY")
    page.locator("[data-finding='2']").click()
    assert page.locator("#findingEvidence").is_visible()
    page.get_by_role("button", name="Back to findings").click()
    assert page.locator("[data-finding='2']").evaluate("el => el === document.activeElement")
    assert abs(page.evaluate("window.scrollY") - finding_scroll) <= 1
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(SCREENSHOTS / "change-assurance-mobile.png"), full_page=False)
    page.close()


def run_recovery(browser):
    page = browser.new_page(viewport={"width": 1280, "height": 720}, reduced_motion="reduce")
    page.goto(BASE_URL)
    rejected = page.evaluate("""([beforeText, afterText]) => {
        const api = window.ProjectLensChangeAssurance;
        const results = {};
        for (const kind of ['identical', 'reversed', 'identity']) {
            const before = api.parseXer(beforeText, 'previous.xer');
            const after = api.parseXer(afterText, 'current.xer');
            if (kind === 'identity') after.project.id = 'different';
            try { api.analyseReview(kind === 'reversed' ? after : before, kind === 'identical' ? before : kind === 'reversed' ? before : after, 'On track and unchanged.'); results[kind] = false; }
            catch { results[kind] = true; }
        }
        const before = api.parseXer(beforeText, 'previous.xer');
        const after = api.parseXer(afterText, 'current.xer');
        after.tasks.forEach(task => { task.float = null; });
        results.coverage = api.analyseReview(before, after, 'On track.').blockers.some(blocker => blocker.title.includes('could not be compared'));
        const relationshipReview = api.analyseReview(api.parseXer(beforeText, 'previous.xer'), api.parseXer(afterText, 'current.xer'), 'Unchanged.');
        results.relationships = relationshipReview.changes.relationEvidence.length === 5 && relationshipReview.changes.relationEvidence.every(item => item.predecessor && item.successor && item.type);
        return results;
    }""", [(ROOT / "docs/demo/northstar-previous.xer").read_text(), (ROOT / "docs/demo/northstar-current.xer").read_text()])
    assert all(rejected.values()), rejected
    page.get_by_role("button", name="Use your own XER files").click()
    page.locator("#assurancePrevious").set_input_files(ROOT / "docs/demo/northstar-previous.xer")
    page.locator("#assuranceCurrent").set_input_files(ROOT / "docs/demo/riverside-current.xer")
    page.get_by_role("button", name="Check decision readiness").click()
    page.locator("#workspaceError").wait_for(state="visible")
    assert "project names differ" in page.locator("#workspaceError").inner_text()
    assert page.locator("#packIntake").is_visible()
    page.locator("#assuranceCurrent").set_input_files(ROOT / "docs/demo/northstar-current.xer")
    page.locator("#assuranceNarrative").fill("Unchanged.")
    page.get_by_role("button", name="Check decision readiness").click()
    page.locator("#readinessWorkspace").wait_for(state="visible")
    page.locator("[data-finding='2']").click()
    assert "Relationship field evidence (5)" in page.locator("#findingEvidence").inner_text()
    assert page.locator("#findingEvidence li").count() == 5
    assert "Type:" in page.locator("#findingEvidence").inner_text()
    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Start another review").click()
    page.locator("#assurancePrevious").set_input_files(ROOT / "docs/demo/riverside-previous.xer")
    page.locator("#assuranceCurrent").set_input_files(ROOT / "docs/demo/riverside-current.xer")
    page.locator("#assuranceNarrative").fill((ROOT / "docs/demo/riverside-narrative.txt").read_text())
    page.get_by_role("button", name="Check decision readiness").click()
    page.locator("#readinessWorkspace").wait_for(state="visible")
    page.get_by_role("button", name="Record human decision").click()
    page.get_by_label("Approve with conditions").check()
    page.locator("#decisionOwner").fill("Review Chair")
    page.locator("#decisionRationale").fill("Accept only with owned follow-up.")
    page.locator("#decisionCondition").fill("Confirm schedule basis")
    page.get_by_role("button", name="Save decision record").click()
    assert "owner and a due date" in page.locator("#workspaceError").inner_text()
    page.locator("#conditionOwner").fill("Planner")
    page.locator("#conditionDue").fill("2026-11-01")
    page.evaluate("() => { window.__originalSetItem = Storage.prototype.setItem; Storage.prototype.setItem = function() { throw new DOMException('Quota exceeded', 'QuotaExceededError'); }; }")
    page.get_by_role("button", name="Save decision record").click()
    assert "could not be saved" in page.locator("#workspaceError").inner_text()
    assert page.locator("#decisionOwner").input_value() == "Review Chair"
    assert page.evaluate("localStorage.getItem('projectlens:change-decisions:v1')") is None
    page.evaluate("() => { Storage.prototype.setItem = window.__originalSetItem; }")
    page.get_by_role("button", name="Save decision record").click()
    page.wait_for_url("**#decisions")
    page.locator("#decisionRegister .register-row").wait_for(state="visible")
    assert page.locator("#decisionRegister .register-row").count() == 1
    page.close()


with sync_playwright() as playwright:
    chromium = playwright.chromium.launch(headless=True)
    run_desktop(chromium)
    run_mobile(chromium)
    run_recovery(chromium)
    chromium.close()
    print("Change assurance browser test passed: blockers, requests, decisions, conditions and mobile")
