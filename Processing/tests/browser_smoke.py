"""Browser smoke test for the static ProjectLens product.

Run with a local server, for example:
python3 scripts/with_server.py --server "python3 -m http.server 8000 --directory docs" --port 8000 -- python3 Processing/tests/browser_smoke.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:8765"


def run_desktop(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    errors = []
    failures = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("requestfailed", lambda request: failures.append(f"{request.method} {request.url}"))
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.locator("#loading").wait_for(state="hidden")
    assert page.locator("#heroCount").inner_text() == "189"
    assert page.locator("#fourYearCount").inner_text() == "35"
    assert page.locator("#recordCount").inner_text() == "1,417"
    assert page.locator("#worsenedCount").inner_text() == "18"
    assert page.locator("#redCount").inner_text() == "34"
    page.locator(".hero h1").wait_for(state="visible")
    page.wait_for_timeout(900)
    page.screenshot(path=str(ROOT / "docs" / "assets" / "projectlens-overview.png"))

    page.get_by_role("button", name="Explorer", exact=True).click()
    page.locator("#movementFilter").select_option("Worsened")
    result_count = page.locator("#resultCount").inner_text()
    assert result_count.upper() == "18 PROJECTS", result_count
    page.locator("#projectTable .watch-button").first.click()
    assert page.locator("#watchCount").inner_text() == "1"
    page.locator("#watchlistFilter").click()
    assert page.locator("#resultCount").inner_text().upper() == "1 PROJECT"
    page.locator("#watchlistFilter").click()
    page.locator("#projectTable tr").first.click()
    page.locator("#projectDialog[open]").wait_for()
    assert page.locator("#detailTimeline article").count() >= 2
    assert "gov.uk" in page.locator("#detailSource").get_attribute("href")
    page.locator("#detailFindCases").click()
    page.locator('[data-view-panel="cases"]:not([hidden])').wait_for()
    assert page.locator("#caseGrid .case-card").count() >= 1

    page.get_by_role("button", name="Method", exact=True).click()
    assert page.locator("#limitationsList li").count() == 5
    assert page.locator("#sourceList a").count() == 7
    assert page.locator("#externalSourceList a").count() == 6
    assert page.locator('a[href="schedule-review.html"]').count() >= 1

    page.get_by_role("button", name="Briefing", exact=True).click()
    page.locator(".hero h1").wait_for(state="visible")
    assert not errors, errors
    assert not failures, failures
    page.close()


def run_mobile(browser):
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.locator("#loading").wait_for(state="hidden")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
    nav_buttons = page.locator(".nav-button")
    assert nav_buttons.count() == 4
    for index in range(nav_buttons.count()):
        button = nav_buttons.nth(index)
        box = button.bounding_box()
        assert button.is_visible() and box and box["x"] + box["width"] <= 391
    page.get_by_role("button", name="Explorer", exact=True).click()
    assert page.locator("#projectTable tr").count() > 0
    page.wait_for_timeout(600)
    page.screenshot(path=str(ROOT / "docs" / "assets" / "projectlens-mobile.png"), full_page=True)
    page.close()


with sync_playwright() as playwright:
    chromium = playwright.chromium.launch(headless=True)
    run_desktop(chromium)
    run_mobile(chromium)
    chromium.close()
    print("Browser smoke test passed: desktop, interactions, sources and mobile layout")
