"""Record a deliberately paced, silent XER assurance walkthrough for LinkedIn."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "work" / "video"
BASE_URL = "http://127.0.0.1:8765"


def pause(page, milliseconds):
    page.wait_for_timeout(milliseconds)


def focus(page, selector, dwell=3600, block="center"):
    page.locator(selector).evaluate(
        "(node, block) => node.scrollIntoView({ behavior: 'smooth', block })",
        block,
    )
    pause(page, 1200)
    pause(page, dwell)


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
    page.goto(f"{BASE_URL}/schedule-review.html")
    page.wait_for_load_state("networkidle")
    pause(page, 4200)
    focus(page, "#intakeSection", dwell=3200, block="start")
    page.get_by_role("button", name="Run the Northstar demo").click()
    page.locator("#reviewResults").wait_for(state="visible")
    focus(page, ".review-pulse", dwell=4200)
    focus(page, ".completeness-section", dwell=4800, block="start")
    focus(page, ".executive-grid", dwell=4400, block="start")
    focus(page, ".evidence-focus-grid", dwell=5200, block="start")
    focus(page, ".material-section", dwell=4600, block="start")
    page.locator("#changeList .change-row").first.click()
    page.locator("#changeDialog[open]").wait_for()
    pause(page, 4000)
    page.locator("#changeDialog .dialog-close").click()
    pause(page, 1000)
    focus(page, ".follow-through-section", dwell=4400, block="start")
    focus(page, ".action-section", dwell=3600, block="start")
    focus(page, ".boundary-panel", dwell=4200)

    video = page.video
    page.close()
    output = ROOT / "docs" / "assets" / "projectlens-evidence-demo.webm"
    video.save_as(str(output))
    context.close()
    browser.close()
    print(output)
