"""Desktop and mobile browser checks for Project Evidence Desk."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:8765/board-readiness.html"


def assert_no_overflow(page):
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"
    )


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    failures = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("requestfailed", lambda request: failures.append(request.url))
    page.goto(BASE_URL, wait_until="networkidle")

    assert page.get_by_role("heading", name="Know what the pack can prove.").is_visible()
    assert page.get_by_role("button", name="Check board readiness").is_disabled()
    page.get_by_role("button", name="Review the Northstar example").click()
    page.locator("#results").wait_for(state="visible")
    assert page.locator("[data-finding]").count() == 5
    assert page.locator("#decisionList li").count() == 3
    assert "not support the green status" in page.locator("#readiness-title").inner_text().lower()
    assert_no_overflow(page)

    for checkbox in page.locator("[data-review-finding]").all():
        checkbox.check()
    page.locator("#finalCheck").check()
    assert page.get_by_role("button", name="Mark brief reviewed").is_enabled()
    page.get_by_role("button", name="Mark brief reviewed").click()
    assert page.get_by_role("button", name="Download board brief").is_enabled()

    with page.expect_download() as download_info:
        page.get_by_role("button", name="Download board brief").click()
    assert download_info.value.suggested_filename.endswith(".md")
    page.screenshot(path=str(ROOT / "docs" / "assets" / "board-readiness-overview.png"), full_page=True)

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.goto(BASE_URL, wait_until="networkidle")
    mobile.get_by_role("button", name="Review the Northstar example").click()
    mobile.locator("#results").wait_for(state="visible")
    assert mobile.locator("[data-finding]").count() == 5
    assert_no_overflow(mobile)
    mobile.screenshot(path=str(ROOT / "docs" / "assets" / "board-readiness-mobile.png"), full_page=True)

    assert not errors, errors
    assert not failures, failures
    browser.close()

print("Board readiness browser QA passed: workflow, review gate, export and mobile")
