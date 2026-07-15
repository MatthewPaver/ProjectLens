"""Record a deliberately paced, silent ProjectLens evidence walkthrough."""

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
    pause(page, 3000)

    page.locator("#openPriority").click()
    page.locator("#projectDialog[open]").wait_for()
    pause(page, 3500)
    page.locator("#detailEvidence").scroll_into_view_if_needed()
    pause(page, 3200)
    page.locator("#detailTimeline").scroll_into_view_if_needed()
    pause(page, 3500)
    page.locator("#projectDialog .dialog-close").click()
    pause(page, 1200)

    page.get_by_role("button", name="Explorer", exact=True).click()
    pause(page, 2200)
    page.locator("#movementFilter").select_option("Worsened")
    pause(page, 3500)

    page.goto(f"{BASE_URL}/schedule-review.html")
    page.wait_for_load_state("networkidle")
    pause(page, 3200)
    page.get_by_role("button", name="Run the Northstar demo").click()
    page.locator("#reviewResults").wait_for(state="visible")
    pause(page, 4200)
    page.locator("#contradictionListXer").scroll_into_view_if_needed()
    pause(page, 4000)
    page.locator("#changeList").scroll_into_view_if_needed()
    pause(page, 3500)
    page.locator("#changeList .change-row").first.click()
    page.locator("#changeDialog[open]").wait_for()
    pause(page, 3500)
    page.get_by_role("button", name="Add assurance action").click()
    pause(page, 2500)
    page.locator("#actionList").scroll_into_view_if_needed()
    pause(page, 3500)

    video = page.video
    page.close()
    output = ROOT / "docs" / "assets" / "projectlens-evidence-demo.webm"
    video.save_as(str(output))
    context.close()
    browser.close()
    print(output)
