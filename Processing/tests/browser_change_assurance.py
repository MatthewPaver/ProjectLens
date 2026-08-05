"""End-to-end browser checks for the simplified project change assurance workflow."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:8765/change-assurance.html"


def run_desktop(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    errors = []
    failures = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("requestfailed", lambda request: failures.append(f"{request.method} {request.url}")
            if "127.0.0.1:8787" not in request.url and "localhost:8787" not in request.url else None)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    assert page.locator("[data-view-link]").count() == 3
    assert page.get_by_role("heading", name="Is this change ready to decide?").is_visible()
    page.get_by_role("button", name="Try the Northstar example").click()
    page.locator("#readinessWorkspace").wait_for(state="visible")
    readiness_status = page.locator("#readinessStatus").inner_text()
    assert readiness_status.casefold() == "needs evidence before decision", readiness_status
    assert page.locator("#blockerList .blocker").count() == 3
    assert "73 days" in page.locator("#readinessHeadline").inner_text().casefold()
    # precedents surface by default — the RAG differentiator must not hide in a closed disclosure
    assert page.locator("#supportingEvidence").get_attribute("open") is not None
    assert page.get_by_role("button", name="Review cited precedents").is_visible()

    assert page.locator("#evidenceSummary .evidence-fact").count() == 6
    # wait for offline fallback or live RAG cards (either is fine in CI)
    page.wait_for_function("() => document.querySelectorAll('#comparableCases .comparable-case:not(.is-skeleton)').length >= 3")
    assert page.locator("#comparableCases .comparable-case").count() >= 3
    # gate label is section-level, not repeated on every card
    assert page.locator("#precedentSectionGate").is_visible()
    assert page.locator("#comparableCases .precedent-gate i").first.inner_text().casefold() != "awaiting human gate"

    page.get_by_role("button", name="Draft questions for the team").click()
    page.locator("#requestPanel").wait_for(state="visible")
    page.locator("#requestOwner").fill("Planning Manager")
    page.locator("#requestDue").fill("2026-07-22")
    page.get_by_role("button", name="Save evidence request").click()
    assert page.evaluate("JSON.parse(localStorage.getItem('projectlens:evidence-requests:v1')).length") == 1

    page.get_by_role("button", name="Record human decision").click()
    page.locator("#decisionPanel").wait_for(state="visible")
    # gate still locked until Use/Ignore
    assert page.locator("#decisionForm button[type='submit']").is_disabled()
    assert page.locator("#precedentGateBanner").is_visible()
    # human gate first — form fields stay locked until every card is marked
    # supporting evidence was already opened above
    for button in page.locator("#comparableCases button[data-gate='use']").all():
        button.click()
    assert page.locator("#decisionForm button[type='submit']").is_enabled()
    assert page.locator("#precedentGateBanner").is_hidden()
    page.locator("#decisionPanel").scroll_into_view_if_needed()
    page.get_by_label("Approve with conditions").check(force=True)
    page.locator("#decisionOwner").fill("Change Authority Chair")
    page.locator("#decisionRationale").fill("Proceed only after the named controls are confirmed and owned.")
    page.locator("#decisionCondition").fill("Protect the regression-test window")
    page.locator("#conditionOwner").fill("Test Lead")
    page.locator("#conditionDue").fill("2026-07-29")
    page.get_by_role("button", name="Save decision record").click()
    page.wait_for_url("**#decisions")
    page.locator("#decisionRegister .register-row").wait_for(state="visible")
    assert page.locator("#decisionRegister .register-row").count() == 1
    stored = page.evaluate("JSON.parse(localStorage.getItem('projectlens:change-decisions:v1'))[0]")
    assert stored.get("precedentsUsed"), stored

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
    page.screenshot(path=str(ROOT / "docs" / "assets" / "change-assurance-overview.png"), full_page=False)
    assert not errors, errors
    assert not failures, failures
    page.close()


def run_mobile(browser):
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
    assert page.locator("[data-view-link]").count() == 3
    page.get_by_role("button", name="Try the Northstar example").click()
    page.locator("#readinessWorkspace").wait_for(state="visible")
    assert page.locator("#blockerList .blocker").count() == 3
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
    page.evaluate("document.querySelector('#readinessWorkspace').scrollIntoView()")
    page.screenshot(path=str(ROOT / "docs" / "assets" / "change-assurance-mobile.png"), full_page=False)
    page.close()


with sync_playwright() as playwright:
    chromium = playwright.chromium.launch(headless=True)
    run_desktop(chromium)
    run_mobile(chromium)
    chromium.close()
    print("Change assurance browser test passed: blockers, requests, decisions, conditions and mobile")
