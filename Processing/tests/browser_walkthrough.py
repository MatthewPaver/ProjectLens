"""Exercise the guide against the real review/record engine, with isolated examples."""
from pathlib import Path
import json
import os
from playwright.sync_api import sync_playwright, expect

BASE = os.environ.get("PROJECTLENS_BROWSER_BASE_URL", "http://127.0.0.1:8765/change-assurance.html")
CAPTURES = Path(os.environ.get("PROJECTLENS_SCREENSHOT_DIR", "/tmp/projectlens-browser"))
CAPTURES.mkdir(parents=True, exist_ok=True)


def next_step(page, label):
    page.get_by_role("button", name=f"Next: {label}", exact=True).click()


def snapshot(page):
    return page.evaluate("window.ProjectLensWalkthrough.snapshot()")


def ready(page):
    page.goto(f"{BASE}?walkthrough=1")
    expect(page.locator("#tourMovement")).to_contain_text("73 days")


def run_journey(browser):
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
    page = context.new_page()
    page.goto(BASE)
    # Existing work must survive the demonstration, its saves and its restart.
    preserved = page.evaluate("""() => {
      const record = {id:'existing-work',project:'Real work sentinel',owner:'Keep me',rationale:'Preserve this record',decision:'Return for revision',createdAt:'2026-10-01T00:00:00Z',evidenceVersion:'existing-source'};
      const values = {'projectlens:change-decisions:v1':JSON.stringify([record]),'projectlens:change-conditions:v1':'[]','projectlens:evidence-requests:v1':'[]','projectlens:review-draft:v1':JSON.stringify({marker:'existing draft'})};
      Object.entries(values).forEach(([key,value])=>localStorage.setItem(key,value)); return values;
    }""")
    requests, errors = [], []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    page.on("pageerror", lambda error: errors.append(str(error)))
    ready(page)
    assert snapshot(page)["decisions"] == []
    expect(page.locator("#tourPlay")).to_be_disabled()
    assert page.evaluate("getComputedStyle(document.querySelector('.tour-bar.current')).animationName") == "none"
    # The expected findings are recomputed by the publicly exposed product engine.
    assert page.evaluate("""() => {
      const {review,narrative}=ProjectLensWalkthrough.snapshot(), api=ProjectLensChangeAssurance;
      const actual=api.analyseReview(api.parseXer(review.previous.sourceText,review.previous.fileName),api.parseXer(review.current.sourceText,review.current.fileName),narrative);
      return JSON.stringify(actual.blockers)===JSON.stringify(review.blockers)&&actual.finishMovement===review.finishMovement;
    }""")
    page.screenshot(path=str(CAPTURES / "walkthrough-desktop.png"), full_page=True)
    next_step(page, "inspect the conflict")
    expect(page.locator("#tourTitle")).to_be_focused()
    expect(page.locator("#blockerList .blocker")).to_have_count(3)
    page.locator('[data-finding="1"]').click()
    expect(page.locator("#findingEvidence")).to_contain_text("constraint")
    next_step(page, "change the claim")
    original = snapshot(page)
    page.locator("#tourNarrative").fill("A new narrative awaiting checks.")
    expect(page.locator("#tourNext")).to_be_disabled()
    page.get_by_role("button", name="Try an acknowledged finish change", exact=True).click()
    expect(page.locator("#tourCheckStatus")).to_contain_text("5 findings in the original pack; 5 now")
    changed = snapshot(page)
    assert changed["review"]["inputIdentity"] != original["review"]["inputIdentity"]
    assert changed["review"]["previous"] == original["review"]["previous"]
    assert changed["review"]["current"] == original["review"]["current"]
    assert not any(finding["type"] == "Contradiction" for finding in changed["review"]["blockers"])
    next_step(page, "make a decision")
    expect(page.locator("#tourNext")).to_be_disabled()
    assert snapshot(page)["decisions"] == []
    page.get_by_label("Approve with conditions", exact=True).check()
    page.locator("#decisionCondition").fill("Confirm the schedule basis")
    page.get_by_role("button", name="Save decision record", exact=False).click()
    expect(page.locator("#workspaceError")).to_contain_text("owner and a due date")
    expect(page.locator("#tourPosition")).to_contain_text("Step 4")
    page.locator("#conditionOwner").fill("Example planning lead")
    page.locator("#conditionDue").fill("2027-12-01")
    page.get_by_role("button", name="Save decision record", exact=False).click()
    expect(page.locator("#tourPosition")).to_contain_text("Step 5")
    with page.expect_download() as event:
        page.get_by_role("button", name="Download walkthrough record", exact=True).click()
    record = json.loads(Path(event.value.path()).read_text())
    assert record["demonstration"] is True and record["storage"] == "memory-only"
    assert record["decisions"][0]["findings"] == changed["review"]["blockers"]
    assert record["decisions"][0]["narrative"] == changed["narrative"]
    assert record["decisions"][0]["inputIdentity"] == changed["review"]["inputIdentity"]
    assert len(record["decisions"][0]["precedentsNotReviewed"]) == 3
    assert record["conditions"][0]["decisionId"] == record["decisions"][0]["id"]
    next_step(page, "track the follow-up")
    expect(page.locator("#tourNext")).to_have_text("Follow-up still open")
    page.get_by_role("button", name="Mark closed", exact=True).click()
    expect(page.locator("#tourComplete")).to_contain_text("Condition closed")
    expect(page.locator("#tourNext")).to_have_text("Walkthrough complete")
    with page.expect_download() as event:
        page.get_by_role("button", name="Download walkthrough record", exact=True).click()
    assert json.loads(Path(event.value.path()).read_text())["conditions"][0]["closed"] is True
    assert page.evaluate("values => Object.entries(values).every(([key,value])=>localStorage.getItem(key)===value)", preserved)
    page.get_by_role("button", name="Restart", exact=True).click()
    expect(page.locator("#tourMovement")).to_contain_text("73 days")
    assert snapshot(page)["decisions"] == []
    assert snapshot(page)["conditions"] == []
    assert page.evaluate("values => Object.entries(values).every(([key,value])=>localStorage.getItem(key)===value)", preserved)
    assert all(method == "GET" for method, _ in requests), requests
    assert not any(":8787" in url for _, url in requests)
    assert not errors, errors
    context.close()


def run_phone(browser):
    context = browser.new_context(viewport={"width":390,"height":844}, reduced_motion="reduce")
    page = context.new_page()
    ready(page)
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
    page.screenshot(path=str(CAPTURES / "walkthrough-mobile.png"), full_page=True)
    for label in ["inspect the conflict", "change the claim", "make a decision"]:
        next_step(page, label)
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
    page.get_by_label("Return for revision", exact=True).check()
    page.get_by_role("button", name="Save decision record", exact=False).click()
    next_step(page, "track the follow-up")
    expect(page.locator("#tourComplete")).to_contain_text("without conditions")
    assert snapshot(page)["conditions"] == []
    context.close()


def run_playback_and_recovery(browser):
    context = browser.new_context(viewport={"width":1280,"height":900})
    page = context.new_page()
    page.clock.install()
    ready(page)
    page.get_by_role("button", name="Play walkthrough", exact=True).click()
    page.clock.fast_forward(10000)
    expect(page.locator("#tourPosition")).to_contain_text("Step 2")
    page.get_by_role("button", name="Pause walkthrough", exact=True).click()
    page.clock.fast_forward(20000)
    expect(page.locator("#tourPosition")).to_contain_text("Step 2")
    page.get_by_role("button", name="Play walkthrough", exact=True).click()
    page.clock.fast_forward(10000)
    expect(page.locator("#tourPosition")).to_contain_text("Step 3")
    page.locator("#tourNarrative").focus()
    page.clock.fast_forward(20000)
    expect(page.locator("#tourPosition")).to_contain_text("Step 3")
    page.get_by_role("button", name="Play walkthrough", exact=True).click()
    page.clock.fast_forward(10000)
    expect(page.locator("#tourPosition")).to_contain_text("Step 4")
    expect(page.locator("#tourPlay")).to_be_disabled()
    page.clock.fast_forward(30000)
    assert snapshot(page)["decisions"] == []
    context.close()
    context = browser.new_context()
    page = context.new_page()
    page.route("**/demo/northstar-previous.xer", lambda route: route.abort())
    page.goto(f"{BASE}?walkthrough=1")
    expect(page.locator("#tourError")).to_be_visible()
    expect(page.locator("#tourNext")).to_be_disabled()
    page.unroute("**/demo/northstar-previous.xer")
    page.get_by_role("button", name="Retry loading the example", exact=True).click()
    expect(page.locator("#tourMovement")).to_contain_text("73 days")
    expect(page.locator("#tourError")).to_be_hidden()
    expect(page.locator("#workspaceError")).to_be_hidden()
    context.close()


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        run_journey(browser)
        run_phone(browser)
        run_playback_and_recovery(browser)
        browser.close()
    print("Walkthrough passed: real checks, changed inputs, decision validation, export, closure, isolation, phone, playback and retry.")
