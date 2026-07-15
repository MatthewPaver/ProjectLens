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
    page.on("requestfailed", lambda request: failures.append(f"{request.method} {request.url}"))
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
    assert page.locator(".supporting-evidence").get_attribute("open") is None

    page.locator(".supporting-evidence summary").click()
    assert page.locator("#evidenceSummary .evidence-fact").count() == 6
    assert page.locator("#comparableCases .comparable-case").count() == 3

    page.get_by_role("button", name="Request answers").click()
    page.locator("#requestPanel").wait_for(state="visible")
    page.locator("#requestOwner").fill("Planning Manager")
    page.locator("#requestDue").fill("2026-07-22")
    page.get_by_role("button", name="Save evidence request").click()
    assert page.evaluate("JSON.parse(localStorage.getItem('projectlens:evidence-requests:v1')).length") == 1

    page.get_by_role("button", name="Record human decision").click()
    page.locator("#decisionPanel").wait_for(state="visible")
    page.get_by_label("Approve with conditions").check()
    page.locator("#decisionOwner").fill("Change Authority Chair")
    page.locator("#decisionRationale").fill("Proceed only after the named controls are confirmed and owned.")
    page.locator("#decisionCondition").fill("Protect the regression-test window")
    page.locator("#conditionOwner").fill("Test Lead")
    page.locator("#conditionDue").fill("2026-07-29")
    page.get_by_role("button", name="Save decision record").click()
    page.wait_for_url("**#decisions")
    page.locator("#decisionRegister .register-row").wait_for(state="visible")
    assert page.locator("#decisionRegister .register-row").count() == 1

    page.locator('a[href="#follow-up"]').click()
    assert page.locator("#conditionRegister .register-row").count() == 1
    assert page.locator("#conditionNavCount").inner_text() == "1"
    page.get_by_role("button", name="Mark closed").click()
    assert page.locator("#conditionNavCount").inner_text() == "0"

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
