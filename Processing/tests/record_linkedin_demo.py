"""Record a deliberately paced, silent ProjectLens product walkthrough."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "work" / "video"
BASE_URL = "http://127.0.0.1:8765"


def pause(page, milliseconds):
    page.wait_for_timeout(milliseconds)


VIDEO_DIR.mkdir(parents=True, exist_ok=True)
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        record_video_dir=str(VIDEO_DIR),
        record_video_size={"width": 1440, "height": 900},
        device_scale_factor=1,
    )
    page = context.new_page()
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.locator("#loading").wait_for(state="hidden")
    pause(page, 2400)

    page.locator("#openPriority").click()
    page.locator("#projectDialog[open]").wait_for()
    pause(page, 3200)
    page.locator("#detailEvidence").scroll_into_view_if_needed()
    pause(page, 2600)
    page.locator("#detailTimeline").scroll_into_view_if_needed()
    pause(page, 3000)
    page.locator("#projectDialog .dialog-close").click()
    pause(page, 900)

    page.get_by_role("button", name="Explorer", exact=True).click()
    pause(page, 1700)
    page.locator("#movementFilter").select_option("Worsened")
    pause(page, 2300)
    page.locator("#searchInput").fill("Unity Programme")
    pause(page, 1700)
    page.locator("#projectTable tr").first.click()
    page.locator("#projectDialog[open]").wait_for()
    pause(page, 2600)

    page.locator("#detailFindCases").click()
    page.locator('[data-view-panel="cases"]:not([hidden])').wait_for()
    pause(page, 3200)
    page.locator("#caseGrid .case-card").first.scroll_into_view_if_needed()
    pause(page, 2500)

    page.get_by_role("button", name="Method", exact=True).click()
    pause(page, 2600)
    page.locator(".score-explainer").scroll_into_view_if_needed()
    pause(page, 2700)
    page.locator(".limitations").scroll_into_view_if_needed()
    pause(page, 2600)

    video = page.video
    page.close()
    context.close()
    browser.close()
    output = VIDEO_DIR / "projectlens-linkedin-demo.webm"
    video.save_as(str(output))
    print(output)
