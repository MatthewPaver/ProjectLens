"""Product-led landing, adversarial narrative checks and complete own-file journey."""

import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ.get("PROJECTLENS_BROWSER_BASE_URL", "http://127.0.0.1:8765/change-assurance.html")
CAPTURES = Path(os.environ.get("PROJECTLENS_SCREENSHOT_DIR", "/tmp/projectlens-product"))
CAPTURES.mkdir(parents=True, exist_ok=True)


def loaded(page):
    page.goto(BASE)
    expect(page.locator("#example-finish")).to_be_enabled()
    page.evaluate("document.fonts.ready")


def landing(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append((request.method, request.url)))
    loaded(page)
    expect(page.locator("#productTitle")).to_have_text("The schedule changed.Did the update explain it?")
    expect(page.locator("#exampleTitle")).to_have_text("The finish moves by 73 days. The update says unchanged.")
    expect(page.locator("#exampleComparison")).to_contain_text("15 Dec 2027")
    expect(page.locator("#exampleComparison")).to_contain_text("26 Feb 2028")
    assert page.evaluate("localStorage.getItem('projectlens:change-decisions:v1')") is None
    page.locator("#exampleReplay").click()
    assert page.locator(".example-line").evaluate("el => el.getAnimations().length") == 1
    page.locator("#exampleReplay").click()
    assert page.locator(".example-line").evaluate("el => el.getAnimations().length") == 1
    page.locator("#example-finish").focus()
    page.keyboard.press("ArrowDown")
    expect(page.locator("#example-constraints")).to_be_focused()
    expect(page.locator("#exampleTitle")).to_have_text("3 activities have changed constraints.")
    page.locator("#exampleInspect").click()
    expect(page.locator("#exampleSource tbody tr")).to_have_count(3)
    expect(page.locator("#exampleSource")).to_contain_text("TASK")
    page.locator("#example-relationships").click()
    expect(page.locator("#exampleSource tbody tr")).to_have_count(5)
    expect(page.locator("#exampleQuestion")).to_have_text("What changed in the delivery sequence, and why?")
    page.locator("#exampleInspect").click()
    page.locator("#example-finish").click()
    page.locator("#exampleInspect").click()
    expect(page.locator("#exampleSource")).to_contain_text("PROJECT.scd_end_date")
    page.locator("#exampleInspect").click()
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(CAPTURES / "projectlens-redesign-after-landing.png"))
    page.screenshot(path=".impeccable/review/desktop.png", full_page=True)
    assert all(method == "GET" for method, _ in requests), requests
    assert not errors, errors
    page.close()


def semantic_regressions(browser):
    page = browser.new_page()
    loaded(page)
    result = page.evaluate("""async () => {
      const api = ProjectLensChangeAssurance;
      const [a,b] = await Promise.all(['previous','current'].map(async kind => api.parseXer(await (await fetch(`demo/northstar-${kind}.xer`)).text(), `${kind}.xer`)));
      return [
        'The finish has slipped by 73 days. Constraints remain unexplained. Logic is unverified. Governance action has no owner and no due date.',
        'This project is not on track. The finish moves 73 days. All constraint and logic changes are acknowledged. The action owner is the Planning Manager, due 1 December 2027.',
        'Everything is explained. Constraints and logic are covered. Owner and due date supplied.'
      ].map(text => {
        const review = api.analyseReview(a,b,text);
        return { movement:review.finishMovement, findings:review.blockers, constraints:review.changes.constraintChanges, relationships:review.changes.relationChanges, float:review.changes.floatErosion };
      });
    }""")
    for review in result:
        assert (review["movement"], review["constraints"], review["relationships"], review["float"]) == (73, 3, 5, 9)
        assert len(review["findings"]) >= 4
        assert any("constraint changes" in row["title"] for row in review["findings"])
        assert any("relationship signatures" in row["title"] for row in review["findings"])
        assert any(row["type"] == "Float loss" for row in review["findings"])
    assert any(row["type"] == "Ownership gap" for row in result[0]["findings"])
    assert not any(row["type"] == "Contradiction" for row in result[1]["findings"])
    page.close()


def own_files(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
    loaded(page)
    page.locator("#showOwnPack").click()
    expect(page.locator("#productLanding")).to_be_hidden()
    expect(page.locator("#intakeTitle")).to_be_focused()
    for kind in ("Previous", "Current"):
        page.locator(f"#assurance{kind}").set_input_files(ROOT / "docs" / "demo" / f"northstar-{kind.lower()}.xer")
    original = (ROOT / "docs/demo/northstar-narrative.txt").read_text()
    page.locator("#assuranceNarrative").fill(original)
    page.locator("#runAssuranceReview").click()
    expect(page.locator("#readinessWorkspace")).to_be_visible()
    expect(page.locator("#reviewFacts")).to_contain_text("Constraint changes")
    page.screenshot(path=str(CAPTURES / "projectlens-redesign-after-review.png"))
    revision = "This project is not on track. The finish moves 73 days. All constraint and logic changes are acknowledged. The action owner is the Planning Manager, due 1 December 2027."
    page.locator("#narrativeRevision summary").click()
    page.locator("#revisedNarrative").fill(revision)
    page.locator("#recheckNarrative").click()
    expect(page.locator("#narrativeRevision")).not_to_have_attribute("open", "")
    expect(page.locator(".original-narrative")).to_contain_text(original)
    expect(page.locator("#reviewFacts")).to_contain_text("Constraint changes")
    page.locator("#showDecisionForm").click()
    page.get_by_label("Approve with conditions", exact=True).check()
    page.locator("#decisionOwner").fill("Review chair")
    page.locator("#decisionRationale").fill("Proceed only after the stated evidence is supplied.")
    page.locator("#decisionForm button[type=submit]").click()
    expect(page.locator("#workspaceError")).to_contain_text("Conditional approval needs")
    page.locator("#decisionCondition").fill("Supply the approved instruction for changed constraints")
    page.locator("#conditionOwner").fill("Planning manager")
    page.locator("#conditionDue").fill("2027-12-01")
    page.locator("#saveDraft").click()
    page.reload()
    page.locator("#resumeDraft").click()
    expect(page.locator("#decisionOwner")).to_have_value("Review chair")
    page.locator("#decisionForm button[type=submit]").click()
    page.wait_for_url("**#decisions")
    record = page.evaluate("JSON.parse(localStorage.getItem('projectlens:change-decisions:v1'))[0]")
    assert record["originalNarrative"] == original
    assert record["narrative"] == revision
    assert record["unresolved"] == 4
    page.evaluate("Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:text=>{window.__export=text;return Promise.resolve()}}})")
    page.locator("#exportDecisions").click()
    page.wait_for_function("typeof window.__export === 'string'")
    exported = page.evaluate("window.__export")
    assert original in exported and revision in exported
    page.locator('[data-view-link="follow-up"]').click()
    page.locator("#conditionRegister button").click()
    expect(page.locator("#conditionRegister")).to_contain_text("Closed")
    assert page.evaluate("JSON.parse(localStorage.getItem('projectlens:change-conditions:v1'))[0].closed") is True
    page.close()


def responsive_and_recovery(browser):
    for width in (320, 390, 768, 1280, 1920):
        page = browser.new_page(viewport={"width": width, "height": 844}, reduced_motion="reduce")
        loaded(page)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), width
        expect(page.locator('#assuranceToast')).to_be_hidden()
        header = page.locator('.assurance-header').bounding_box()
        navigation = page.locator('.product-navigation').bounding_box()
        assert navigation['y'] >= header['y'] and navigation['y'] + navigation['height'] <= header['y'] + header['height'] + 1, (width, header, navigation)
        assert page.locator("#loadAssuranceDemo").bounding_box()["y"] < 800, width
        expect(page.locator("#exampleReplay")).to_be_hidden()
        page.locator("#example-relationships").click()
        page.locator("#exampleInspect").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), width
        if width == 390:
            page.locator("#exampleInspect").click()
            page.locator("#example-finish").click()
            page.screenshot(path=".impeccable/review/mobile.png", full_page=True)
            page.screenshot(path=str(CAPTURES / "projectlens-redesign-after-mobile.png"), full_page=True)
            page.locator('#loadAssuranceDemo').click()
            expect(page.locator('#readinessWorkspace')).to_be_visible()
            expect(page.locator('#assuranceToast')).not_to_have_class('assurance-toast show')
            expect(page.locator('#assuranceToast')).to_be_hidden()
            header = page.locator('.assurance-header').bounding_box()
            navigation = page.locator('.review-navigation').bounding_box()
            assert navigation['y'] + navigation['height'] <= header['y'] + header['height'] + 1
        page.close()
    page = browser.new_page()
    page.route("**/demo/northstar-current.xer", lambda route: route.fulfill(status=503, body="Unavailable"))
    page.goto(BASE)
    expect(page.locator("#exampleRetry")).to_be_visible()
    expect(page.locator("#exampleInspect")).to_be_disabled()
    page.unroute("**/demo/northstar-current.xer")
    page.locator("#exampleRetry").click()
    expect(page.locator("#exampleInspect")).to_be_enabled()
    page.close()


if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch()
        landing(browser)
        semantic_regressions(browser)
        own_files(browser)
        responsive_and_recovery(browser)
        browser.close()
    print("Product journeys passed: real evidence, keyboard, animation, negation, own files, draft, decision, condition, five widths and retry")
