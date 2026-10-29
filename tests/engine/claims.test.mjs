// Narrative claim and UK date detection for detectScheduleClaims.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "../..");
const RESEARCH = path.resolve(REPO, "../research_notes/Real case tests/projectlens-inputs");
const require = createRequire(import.meta.url);
const engine = require(path.join(REPO, "docs/engine/projectlens-engine.js"));

const pairs = JSON.parse(fs.readFileSync(path.join(RESEARCH, "gmpp_pairs.json"), "utf8"));

function pair(nameRe, prevYear, year) {
  const hit = pairs.find(p => nameRe.test(p.name) && p.prevYear === prevYear && p.year === year);
  assert.ok(hit, `missing pair ${nameRe} ${prevYear}->${year}`);
  return hit;
}

describe("detectScheduleClaims", () => {
  it("exports detectScheduleClaims and claim rules", () => {
    assert.equal(typeof engine.detectScheduleClaims, "function");
    const rules = JSON.parse(fs.readFileSync(path.join(REPO, "docs/engine/claim-rules.json"), "utf8"));
    assert.ok(rules.stablePhrases.includes("remained the same"));
    assert.ok(rules.movePhrases.includes("correction of published position"));
  });

  it("Land ISTAR 2025→2026: date in narrative mismatches current finish", () => {
    const p = pair(/ISTAR/i, 2025, 2026);
    const result = engine.detectScheduleClaims(p.narrative, {
      currentFinish: p.curEnd,
      previousFinish: p.prevEnd,
    });
    assert.equal(result.contradiction?.kind, "date-mismatch");
  });

  it("E7 Wedgetail 2025→2026: contradiction while finish moved", () => {
    const p = pair(/Wedgetail/i, 2025, 2026);
    const result = engine.detectScheduleClaims(p.narrative, {
      currentFinish: p.curEnd,
      previousFinish: p.prevEnd,
    });
    assert.ok(result.contradiction, "expected a contradiction");
    assert.ok(["says-unchanged", "date-mismatch"].includes(result.contradiction.kind));
  });

  it("E7 Wedgetail 2021→2022: stated move suppresses contradiction", () => {
    const p = pair(/Wedgetail/i, 2021, 2022);
    const result = engine.detectScheduleClaims(p.narrative, {
      currentFinish: p.curEnd,
      previousFinish: p.prevEnd,
    });
    assert.equal(result.contradiction, null);
  });

  it("does not treat 'not on track' as a stable claim", () => {
    const result = engine.detectScheduleClaims("The programme is not on track for the original finish.", {
      currentFinish: "2030-01-01",
      previousFinish: "2027-01-01",
    });
    assert.equal(result.stable.length, 0);
    assert.equal(result.contradiction, null);
  });

  it("parses every Review Focus 1 UK date form to the right ISO date", () => {
    const cases = [
      ["31/12/2027", "2027-12-31"],
      ["31 December 2027", "2027-12-31"],
      ["31 Dec 27", "2027-12-31"],
      ["30 June 26", "2026-06-30"],
      ["December 2027", "2027-12-31"],
      ["2027-12-31", "2027-12-31"],
      ["Q4 2027-28", "2028-03-31"],
      ["25/26-Q4", "2026-03-31"],
    ];
    for (const [raw, iso] of cases) {
      const result = engine.detectScheduleClaims(`Finish is ${raw}.`, {
        currentFinish: iso,
        previousFinish: iso,
      });
      assert.ok(result.dates.some(d => d.iso === iso && d.raw.includes(raw.split(" ")[0])), `failed for ${raw}: ${JSON.stringify(result.dates)}`);
      const hit = result.dates.find(d => d.iso === iso);
      assert.equal(hit.iso, iso, raw);
    }
  });

  it("gives no contradiction when a stated move lands on the published finish", () => {
    // Round 2 N3: a move only suppresses when it lands on currentFinish (within 14 days).
    const candidates = pairs.filter(p => Math.abs(p.deltaDays) >= 7 && (p.explainsCorrection || (p.mentionsMove && !p.claimsSame)));
    assert.ok(candidates.length > 0);
    let checked = 0;
    for (const p of candidates) {
      const result = engine.detectScheduleClaims(p.narrative, {
        currentFinish: p.curEnd,
        previousFinish: p.prevEnd,
      });
      const lands = result.movesStated.some(move => {
        if (!move.to) return false;
        const delta = engine.dayDifference(
          engine.parseDate(move.to),
          engine.parseDate(p.curEnd),
        );
        return delta !== null && Math.abs(delta) <= 14;
      });
      if (!lands) continue;
      checked += 1;
      assert.equal(
        result.contradiction,
        null,
        `${p.name} ${p.prevYear}->${p.year}: ${JSON.stringify(result.contradiction)} :: ${p.narrative.slice(0, 120)}`,
      );
    }
    assert.ok(checked > 0, "expected at least one landing move case");
  });

  // Round 2 N1: more "unchanged" wording and whitespace / NBSP normalisation.
  it("flags 'has remained static' when the finish moved", () => {
    const result = engine.detectScheduleClaims(
      "The completion date has remained static since the last report.",
      { currentFinish: "2028-01-01", previousFinish: "2027-06-15" },
    );
    assert.equal(result.contradiction?.kind, "says-unchanged");
  });

  it("flags 'remains constant' when the named date mismatches", () => {
    const result = engine.detectScheduleClaims(
      "Delivery remains constant at 30/09/2028.",
      { currentFinish: "2029-06-30", previousFinish: "2029-06-30" },
    );
    assert.equal(result.contradiction?.kind, "date-mismatch");
  });

  it("accepts a true unchanged claim with matching date", () => {
    const result = engine.detectScheduleClaims(
      "The baseline end date remains 31/08/26 and has not changed.",
      { currentFinish: "2026-08-31", previousFinish: "2026-08-31" },
    );
    assert.equal(result.contradiction, null);
  });

  it("parses a date that contains a non-breaking space", () => {
    // U+00A0 sits between August and the year in real GMPP paste-outs.
    const narrative = "Finish is 31 August\u00A02026.";
    const result = engine.detectScheduleClaims(narrative, {
      currentFinish: "2026-08-31",
      previousFinish: "2026-08-31",
    });
    assert.ok(
      result.dates.some(d => d.iso === "2026-08-31"),
      `expected 2026-08-31 in ${JSON.stringify(result.dates)}`,
    );
  });

  // Round 2 N2: reporting periods are not finish dates; quarter finishes are.
  it("ignores financial-year reporting periods after 'compared to'", () => {
    const result = engine.detectScheduleClaims(
      "Compared to financial year 21/22-Q4, the end date remained 31/03/2027.",
      { currentFinish: "2027-03-31", previousFinish: "2027-03-31" },
    );
    assert.equal(result.contradiction, null);
    assert.deepEqual(
      result.dates.map(d => d.iso),
      ["2027-03-31"],
    );
  });

  it("treats an end-date FY quarter as a finish claim when current is inside it", () => {
    // Plan wrote Q2 25/26 with 2026-06-15; under UK FY that quarter is Jul–Sep 2025.
    // Q1 26/27 is Apr–Jun 2026, which is the range those dates actually test.
    const sentence = "The current forecast end date is Q1 26/27 following re-planning.";
    const inside = engine.detectScheduleClaims(sentence, {
      currentFinish: "2026-06-15",
      previousFinish: "2026-06-15",
    });
    assert.equal(inside.contradiction, null);
    assert.ok(inside.dates.some(d => d.precision === "quarter"));

    const outside = engine.detectScheduleClaims(sentence, {
      currentFinish: "2026-12-20",
      previousFinish: "2026-12-20",
    });
    assert.equal(outside.contradiction?.kind, "date-mismatch");
  });

  it("also accepts the plan's Q2 25/26 wording against a date inside that quarter", () => {
    const sentence = "The current forecast end date is Q2 25/26 following re-planning.";
    const inside = engine.detectScheduleClaims(sentence, {
      currentFinish: "2025-08-15",
      previousFinish: "2025-08-15",
    });
    assert.equal(inside.contradiction, null);

    const outside = engine.detectScheduleClaims(sentence, {
      currentFinish: "2026-12-20",
      previousFinish: "2026-12-20",
    });
    assert.equal(outside.contradiction?.kind, "date-mismatch");
  });

  // Round 2 N3: a stated move only explains the published date it lands on.
  it("flags a stated move whose 'to' date misses the published finish", () => {
    const sentence = "the end date changed from 01/04/2027 to 30/09/2027 due to supplier delays";
    const miss = engine.detectScheduleClaims(sentence, {
      currentFinish: "2028-03-31",
      previousFinish: "2027-04-01",
    });
    assert.equal(miss.contradiction?.kind, "date-mismatch");
    assert.match(
      miss.contradiction.detail,
      /moved to 2027-09-30.*published end date is 2028-03-31/i,
    );

    const near = engine.detectScheduleClaims(sentence, {
      currentFinish: "2027-10-05",
      previousFinish: "2027-04-01",
    });
    assert.equal(near.contradiction, null);
  });

  it("accepts a direction-only extension only when the finish moved later", () => {
    const sentence = "the end date was extended following re-planning";
    const later = engine.detectScheduleClaims(sentence, {
      currentFinish: "2028-01-01",
      previousFinish: "2027-10-03",
    });
    assert.equal(later.contradiction, null);

    const earlier = engine.detectScheduleClaims(sentence, {
      currentFinish: "2027-07-05",
      previousFinish: "2027-10-03",
    });
    assert.equal(earlier.contradiction?.kind, "date-mismatch");
  });
});
