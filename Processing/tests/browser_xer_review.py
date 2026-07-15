"""End-to-end browser checks for the local-only XER evidence review."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:8765/schedule-review.html"


def run_desktop(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
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

    page.get_by_role("button", name="Analyst", exact=True).click()
    assert page.locator("#fitnessGrid .fitness-item").count() == 7
    assert page.locator("#integrityList .integrity-item").count() >= 3

    page.locator("#changeList .change-row").first.click()
    page.locator("#changeDialog[open]").wait_for()
    assert page.locator("#changeDialogEvidence > div").count() >= 1
    page.get_by_role("button", name="Add assurance action").click()
    assert page.locator("#actionList .action-row").count() >= 1

    with page.expect_download() as download_info:
        page.get_by_role("button", name="Download review pack").click()
    assert download_info.value.suggested_filename.endswith("-schedule-review.json")

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


with sync_playwright() as playwright:
    chromium = playwright.chromium.launch(headless=True)
    run_desktop(chromium)
    run_mobile(chromium)
    chromium.close()
    print("XER review browser test passed: demo, analysis, actions, export and mobile")
