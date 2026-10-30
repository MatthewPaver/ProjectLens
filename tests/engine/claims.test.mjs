// Narrative claim and UK date detection for detectScheduleClaims.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "../..");
const RESEARCH = path.resolve(REPO, "../research_notes/Real case tests/projectlens-inputs");
const require = createRequire(import.meta.url);
const engine = require(path.join(REPO, "docs/engine/projectlens-engine.js"));

const pairs = JSON.parse(fs.readFileSync(path.join(RESEARCH, "gmpp_pairs.json"), "utf8"));

describe("explained moves retain historical dates and capability scope", () => {
  for (const predicate of ["was previously scheduled", "was originally expected"]) {
    it(`keeps a finish that ${predicate} in the past`, () => {
      const result = engine.detectScheduleClaims(
        `The project ${predicate} to complete by 18 September 2028. Closure now follows the final audit.`,
        { previousFinish: "2028-09-18", currentFinish: "2028-12-18" },
      );
      assert.equal(result.dates[0].role, "previous-finish");
      assert.equal(result.contradiction, null);
      assert.equal(result.publishedDateCheck, null);
    });
  }

  for (const capability of [
    "initial operating capability", "initial operational capability",
    "full operating capability", "full operational capability",
  ]) {
    it(`keeps ${capability} separate from programme closure`, () => {
      const result = engine.detectScheduleClaims(
        `The programme remains on track to deliver its ${capability} in October 2028.`,
        { previousFinish: "2032-04-30", currentFinish: "2030-04-30" },
      );
      assert.deepEqual(result.stable, []);
      assert.equal(result.contradiction, null);
      assert.equal(result.publishedDateCheck, null);
    });
  }
});

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
    // Finish did not move: Round 3 D3 turns this into a published-date check, not a contradiction.
    const result = engine.detectScheduleClaims(
      "Delivery remains constant at 30/09/2028.",
      { currentFinish: "2029-06-30", previousFinish: "2029-06-30" },
    );
    assert.equal(result.contradiction, null);
    assert.equal(result.publishedDateCheck?.narrativeDate, "2028-09-30");
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
    // Reporting FY labels may sit in dates with role reporting; only current-finish counts.
    const finishDates = result.dates.filter(d => d.role === "current-finish");
    assert.deepEqual(
      finishDates.map(d => d.iso),
      ["2027-03-31"],
    );
    assert.ok(
      result.dates.every(d => d.role !== "current-finish" || d.iso === "2027-03-31"),
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

    // No move + outside the quarter → publishedDateCheck (Round 3 D3).
    const outside = engine.detectScheduleClaims(sentence, {
      currentFinish: "2026-12-20",
      previousFinish: "2026-12-20",
    });
    assert.equal(outside.contradiction, null);
    assert.ok(outside.publishedDateCheck);

    // Finish moved + outside the quarter → still a date-mismatch contradiction.
    const movedOutside = engine.detectScheduleClaims(sentence, {
      currentFinish: "2026-12-20",
      previousFinish: "2026-06-15",
    });
    assert.equal(movedOutside.contradiction?.kind, "date-mismatch");
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
    assert.equal(outside.contradiction, null);
    assert.ok(outside.publishedDateCheck);

    const movedOutside = engine.detectScheduleClaims(sentence, {
      currentFinish: "2026-12-20",
      previousFinish: "2025-08-15",
    });
    assert.equal(movedOutside.contradiction?.kind, "date-mismatch");
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

  // Round 3 D1: each date gets a role; only current-finish drives date-mismatch.
  it("classifies the standard GMPP template dates and stays quiet when they match", () => {
    const narrative =
      "The scheduled baseline project end date at Q2 2021 (30th September 2020) is 31/12/22, " +
      "has not changed since last year's Q2 1920 date of 31/12/22, due to supplier phasing.";
    const result = engine.detectScheduleClaims(narrative, {
      currentFinish: "2022-12-31",
      previousFinish: "2022-12-31",
    });
    assert.equal(result.contradiction, null);
    const roles = result.dates.map(d => d.role);
    assert.ok(roles.includes("reporting"), `roles=${JSON.stringify(roles)}`);
    assert.ok(roles.includes("current-finish"), `roles=${JSON.stringify(roles)}`);
    assert.ok(roles.includes("previous-finish"), `roles=${JSON.stringify(roles)}`);
  });

  it("uses only the current-finish date for date-mismatch on a lengthened-end sentence", () => {
    const narrative =
      "The scheduled baseline project end date at Q2 2021 (30th September 2020) is 30/06/25, " +
      "has lengthened by 365 days since last year's Q2 1920 date of 30/06/24 due to scope.";
    const match = engine.detectScheduleClaims(narrative, {
      currentFinish: "2025-06-30",
      previousFinish: "2024-06-30",
    });
    assert.equal(match.contradiction, null);

    const miss = engine.detectScheduleClaims(narrative, {
      currentFinish: "2026-03-31",
      previousFinish: "2024-06-30",
    });
    assert.equal(miss.contradiction?.kind, "date-mismatch");
    assert.match(miss.contradiction.detail, /30\/06\/25|2025-06-30/);
  });

  it("treats milestone dates as other and does not flag an on-track list", () => {
    const result = engine.detectScheduleClaims(
      "Milestones remain on track: Phase 1 go-live Apr 2021, Phase 2 Mar 2023.",
      { currentFinish: "2025-12-31", previousFinish: "2025-12-31" },
    );
    assert.equal(result.contradiction, null);
    const milestoneDates = result.dates.filter(d =>
      /apr\s*2021|mar\s*2023/i.test(d.raw),
    );
    assert.equal(milestoneDates.length, 2);
    assert.ok(milestoneDates.every(d => d.role === "other"));
  });

  it("ignores a review-month date when delivery is on track and the finish did not move", () => {
    const result = engine.detectScheduleClaims(
      "The March 2022 review confirmed delivery was on track.",
      { currentFinish: "2027-06-30", previousFinish: "2027-06-30" },
    );
    assert.equal(result.contradiction, null);
  });

  // Round 3 D2: move / stable phrases only count when about the end date.
  it("ignores a contract-start move when the programme end is unaffected", () => {
    const narrative =
      "The contract start date has moved back by four weeks. The overall programme end date is unaffected.";
    const stable = engine.detectScheduleClaims(narrative, {
      currentFinish: "2028-06-30",
      previousFinish: "2028-06-30",
    });
    assert.equal(stable.contradiction, null);

    const moved = engine.detectScheduleClaims(narrative, {
      currentFinish: "2028-10-28",
      previousFinish: "2028-06-30",
    });
    assert.equal(moved.contradiction?.kind, "says-unchanged");
  });

  it("accepts 'has lengthened by' when the finish moved by that amount", () => {
    const result = engine.detectScheduleClaims(
      "The end date has lengthened by 200 days.",
      { currentFinish: "2027-07-19", previousFinish: "2026-12-31" },
    );
    assert.equal(result.contradiction, null);
    assert.ok(result.movesStated.some(m => /lengthened by/i.test(m.phrase)));
  });

  // Round 3 D3: no-move + mismatched current-finish → check, not contradiction.
  it("returns publishedDateCheck when the finish did not move but the narrative date differs", () => {
    const result = engine.detectScheduleClaims(
      "The end date has changed since last year and is now 31/03/2027.",
      { currentFinish: "2026-03-31", previousFinish: "2026-03-31" },
    );
    assert.equal(result.contradiction, null);
    assert.equal(result.publishedDateCheck?.narrativeDate, "2027-03-31");
    assert.equal(result.publishedDateCheck?.publishedDate, "2026-03-31");
    assert.match(
      result.publishedDateCheck.detail,
      /narrative gives the end date as .*published end date is/i,
    );
  });
});

// Round 4 narrative-only examples are synthetic regression inputs, not project records.
describe("Round 4 programme stability", () => {
  for (const sentence of ["The programme remains on track.", "Compared to last year, the programme remains on track.", "The project delivery remains on schedule.", "Delivery is on schedule.", "The project is progressing to plan.", "The programme remains on course.", "The project is on target to complete."]) {
    it(sentence, () => {
      const result = engine.detectScheduleClaims(sentence, { previousFinish: "2026-06-30", currentFinish: "2026-12-27" });
      assert.equal(result.contradiction?.kind, "says-unchanged");
    });
  }
  for (const sentence of ["Phase 2 remains on track; the programme end date has moved to 31/03/2027.", "The programme is not on track.", "The procurement is on schedule.", "The programme's Phase 2 remains on track.", "Milestones remain on track: Phase 1 March 2027, Phase 2 June 2027."]) {
    it(sentence, () => assert.equal(engine.detectScheduleClaims(sentence, { previousFinish: "2026-06-30", currentFinish: "2027-03-31" }).contradiction, null));
  }
  it("delivery on schedule without a move stays quiet", () => {
    assert.equal(engine.detectScheduleClaims("Delivery is on schedule.", { previousFinish: "2026-06-30", currentFinish: "2026-06-30" }).contradiction, null);
  });
});

describe("Round 4 published-date checks", () => {
  for (const sentence of [
    "The project is now expected to complete in March 2027.",
    "Completion is now expected in March 2027.",
    "The end date is Q4 2026/27.",
    "The scheduled completion date is March 2027.",
    "The project will complete by 31 March 2027.",
  ]) {
    it(sentence, () => {
      const result = engine.detectScheduleClaims(sentence, { previousFinish: "2026-06-30", currentFinish: "2026-06-30" });
      assert.equal(result.contradiction, null);
      assert.equal(result.publishedDateCheck?.narrativeDate, "2027-03-31");
    });
  }
  it("matching end date needs no check", () => {
    const result = engine.detectScheduleClaims("The end date remains 30/06/2026.", { previousFinish: "2026-06-30", currentFinish: "2026-06-30" });
    assert.equal(result.publishedDateCheck, null);
  });
  it("does not inherit finish context across sentences", () => {
    const result = engine.detectScheduleClaims("The end date is unchanged. The review is in March 2027.", { previousFinish: "2026-06-30", currentFinish: "2026-06-30" });
    assert.equal(result.publishedDateCheck, null);
  });
});

it("keeps end-date subject across a parenthesised reporting date", () => {
  const result = engine.detectScheduleClaims(
    "The scheduled baseline project end date at Q2 2025 (30th September 2024), as recorded in the annual return, is 31/03/2027.",
    { previousFinish: "2026-06-30", currentFinish: "2026-06-30" },
  );
  assert.equal(result.publishedDateCheck?.narrativeDate, "2027-03-31");
});

it("an on-track claim for the explicitly stated new finish explains that date", () => {
  const result = engine.detectScheduleClaims("The programme is on track to complete by 31 March 2027.", {
    previousFinish: "2026-06-30", currentFinish: "2027-03-31",
  });
  assert.equal(result.contradiction, null);
});
it("accepts qualified whole-programme subjects", () => {
  const result = engine.detectScheduleClaims("The programme overall remains firmly on track.", {
    previousFinish: "2026-06-30", currentFinish: "2027-03-31",
  });
  assert.equal(result.contradiction?.kind, "says-unchanged");
});

it("a later phase reference does not change the programme subject", () => {
  const result = engine.detectScheduleClaims("The programme remains on track, with phase reviews continuing.", {
    previousFinish: "2026-06-30", currentFinish: "2027-03-31",
  });
  assert.equal(result.contradiction?.kind, "says-unchanged");
});
it("a named current finish elsewhere qualifies an on-track assessment", () => {
  const result = engine.detectScheduleClaims("The programme remains on track. The completion date is now 31 March 2027.", {
    previousFinish: "2026-06-30", currentFinish: "2027-03-31",
  });
  assert.equal(result.contradiction, null);
});

it("recognises an end-date extension without an auxiliary verb", () => {
  const result = engine.detectScheduleClaims("The programme remains on track. The end date extended to 31 March 2027.", { previousFinish: "2026-06-30", currentFinish: "2027-03-31" });
  assert.equal(result.contradiction, null);
});

describe("forecast finish assertions", () => {
  for (const sentence of ["The project completion date will be 31 March 2027.", "The project end date has been set for 31 March 2027.", "The programme end date: 31 March 2027."]) {
    it(sentence, () => {
      const r = engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30", currentFinish:"2026-06-30"});
      assert.equal(r.publishedDateCheck?.narrativeDate, "2027-03-31");
    });
  }
});

describe("whole-programme assertions and stated delay", () => {
  for (const sentence of ["The programme remains in line with plan.", "The project shows no deviation."]) {
    it(sentence, () => assert.equal(engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2027-03-31"}).contradiction?.kind, "says-unchanged"));
  }
  it("programme delay explains a later finish", () => {
    const r = engine.detectScheduleClaims("The programme remains on track. Delivery has been delayed by nine months.", {previousFinish:"2026-06-30",currentFinish:"2027-03-31"});
    assert.equal(r.contradiction, null);
  });
  it("a bare completion label names the current finish", () => {
    const r = engine.detectScheduleClaims("Anticipated completion: March 2027.", {previousFinish:"2026-06-30",currentFinish:"2026-06-30"});
    assert.equal(r.publishedDateCheck?.narrativeDate, "2027-03-31");
  });
});

it("programme-level delay wording explains a later finish", () => {
  for (const sentence of ["The programme remains on track. The programme has encountered delays.", "The programme remains on track. The completion date has slipped."]) {
    assert.equal(engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2027-03-31"}).contradiction, null);
  }
});

it("reads a programme end assertion without the word date", () => {
  const r = engine.detectScheduleClaims("The project end is March 2027.", {previousFinish:"2026-06-30",currentFinish:"2026-06-30"});
  assert.equal(r.publishedDateCheck?.narrativeDate, "2027-03-31");
});
it("recognises earlier and later programme finish explanations", () => {
  for (const [text, current] of [["The programme remains on track. The project end date is earlier following replanning.", "2026-03-31"], ["The programme remains on track. The end date is later following replanning.", "2027-03-31"]]) {
    assert.equal(engine.detectScheduleClaims(text, {previousFinish:"2026-06-30",currentFinish:current}).contradiction, null);
  }
});

it("does not interpret an intermediate delivery target as whole-programme stability", () => {
  for (const sentence of ["The programme remains on track to achieve its procurement milestones.", "The project remains on schedule for phase 2 delivery."]) {
    assert.equal(engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2027-03-31"}).contradiction, null);
  }
});

it("intermediate capability targets are not whole-programme end claims", () => {
  for (const sentence of ["The programme remains on schedule for initial operating capability.", "The programme remains on track for the in-service date."]) {
    assert.equal(engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2027-03-31"}).contradiction, null);
  }
});

describe("shared finding sentences", () => {
  it("formats a published-date check consistently in UK English", () => {
    const options = { previousFinish: "2026-06-30", currentFinish: "2026-06-30" };
    const claims = engine.detectScheduleClaims("Completion is now expected in March 2027.", options);
    const sentence = "The narrative gives the end date as 31 Mar 2027; the published end date is 30 Jun 2026. Check which is current.";
    assert.equal(engine.findingSentence(claims, options), sentence);
    assert.equal(claims.publishedDateCheck.detail, sentence);
  });
  it("reports an explained move", () => {
    const options = { previousFinish: "2026-06-30", currentFinish: "2027-03-31" };
    const claims = engine.detectScheduleClaims("The end date moved from 30 June 2026 to 31 March 2027.", options);
    assert.equal(engine.findingSentence(claims, options), "The narrative explains the move.");
  });
});

it("ships the same claim rules in the browser and Node", () => {
  const context = vm.createContext({});
  vm.runInContext(fs.readFileSync(path.join(REPO, "docs/engine/projectlens-engine.js"), "utf8"), context);
  const browserRules = JSON.parse(JSON.stringify(context.ProjectLensEngine.loadClaimRules()));
  assert.deepEqual(browserRules, engine.loadClaimRules());
  const options = { previousFinish: "2026-06-30", currentFinish: "2027-03-31" };
  const result = context.ProjectLensEngine.detectScheduleClaims("The project is progressing to plan.", options);
  assert.equal(result.contradiction?.kind, "says-unchanged");
});

it("keeps stability attached to the nearest subject, including explicit completion wording", () => {
  for (const sentence of ["Phase 2 completion remains on track.", "The programme's procurement completion is on schedule.", "The contract end date is unchanged."]) {
    assert.equal(engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2027-03-31"}).contradiction, null);
  }
});

describe("finish clauses without an is cue", () => {
  for (const sentence of [
    "The programme will finish on 31 March 2027.",
    "The planned completion by March 2027 remains achievable.",
    "The current completion date: March 2027.",
    "The expected project closure is March 2027.",
  ]) {
    it(sentence, () => {
      const r = engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2026-06-30"});
      assert.equal(r.publishedDateCheck?.narrativeDate, "2027-03-31");
    });
  }
});

describe("current completion predicates", () => {
  for (const sentence of [
    "The programme will now be completed by March 2027.",
    "The project will complete in March 2027.",
    "The programme will be delivered by March 2027.",
    "Forecast completion March 2027.",
    "The project is due to end in March 2027.",
  ]) {
    it(sentence, () => {
      const r = engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2026-06-30"});
      assert.equal(r.publishedDateCheck?.narrativeDate, "2027-03-31");
    });
  }
});

describe("whole-programme stability predicates", () => {
  for (const sentence of [
    "The programme continues to be on track.",
    "The project schedule remains on track.",
    "The programme is still running on schedule.",
    "The overall schedule remains on track.",
  ]) {
    it(sentence, () => assert.equal(engine.detectScheduleClaims(sentence, {previousFinish:"2026-06-30",currentFinish:"2027-03-31"}).contradiction?.kind, "says-unchanged"));
  }
});

// Authored narrative sentences for miss categories in the seen public-record sets.
describe("Round 5 schedule claims and finish formats", () => {
  const moved = { previousFinish: "2028-09-18", currentFinish: "2029-02-12" };
  for (const sentence of [
    "On schedule.",
    "No deviation from the agreed schedule.",
    "The delivery schedule has remained stable during this review.",
    "The programme is on-track with no changes to the dates of key milestones.",
    "The project ran to agreed schedule.",
    "The programme continued as scheduled.",
    "The construction phase was extended. This will not impact on the project end date.",
  ]) {
    it(`flags stability: ${sentence}`, () => {
      assert.equal(engine.detectScheduleClaims(sentence, moved).contradiction?.kind, "says-unchanged");
    });
  }
  for (const sentence of [
    "The project will complete in 2028.",
    "The programme will start migration in July 2028 and complete in November 2028.",
    "The end date is estimated as 2028-Q4.",
    "An extension was approved to extend the end date to 18/09/2028.",
    "Closed as planned in November 2028.",
    "Final programme closure took place in November 2028.",
    "The programme will not close down until 18 September 2028.",
  ]) {
    it(`checks a named finish: ${sentence}`, () => {
      const result = engine.detectScheduleClaims(sentence, {
        previousFinish: "2027-04-01", currentFinish: "2027-04-01",
      });
      assert.equal(result.contradiction, null);
      assert.ok(result.publishedDateCheck, JSON.stringify(result.dates));
    });
  }
  for (const sentence of [
    "The project will complete in 2029.",
    "The end date remains 2029-Q1.",
  ]) {
    it(`respects date precision: ${sentence}`, () => {
      const result = engine.detectScheduleClaims(sentence, moved);
      assert.equal(result.contradiction, null);
      assert.equal(result.publishedDateCheck, null);
    });
  }
  for (const sentence of [
    "The phase is on schedule.",
    "The programme remains on track for its procurement milestones.",
    "The programme is not on schedule.",
    "The review took place in November 2028.",
    "The contract closed as planned in November 2028.",
    "The programme remains on track. The end date moved to 12 February 2029.",
  ]) {
    it(`keeps scope or explanation: ${sentence}`, () => {
      const result = engine.detectScheduleClaims(sentence, moved);
      assert.equal(result.contradiction, null);
      assert.equal(result.publishedDateCheck, null);
    });
  }
  it("keeps an actual closure separate from a forecast finish after a stated overrun", () => {
    const result = engine.detectScheduleClaims(
      "The project closed on 25 March 2029. Handover continued beyond the original plan.", moved,
    );
    assert.equal(result.dates[0].role, "actual-finish");
    assert.equal(result.contradiction, null);
  });
  it("an actual closure date does not qualify an earlier on-schedule claim", () => {
    const result = engine.detectScheduleClaims(
      "The programme remained on schedule and formally closed on 12 February 2029.", moved,
    );
    assert.equal(result.contradiction?.kind, "says-unchanged");
  });
});

describe("Round 5 finding evidence", () => {
  const dates = { previousFinish: "2027-12-15", currentFinish: "2028-02-26" };
  it("returns the exact sentence and phrase despite normalised spacing", () => {
    const sentence = "The programme REMAINS\u00a0 on\ntrack; all checks continue.";
    const result = engine.detectScheduleClaims(`An earlier review took place. ${sentence} More information follows.`, dates);
    assert.deepEqual(result.evidence, {
      sentence, phrase: "REMAINS\u00a0 on\ntrack", ...dates,
    });
    assert.equal(engine.findingSentence(result, dates),
      "The update says 'REMAINS on track'. The schedule's finish moved from 15 Dec 2027 to 26 Feb 2028 (73 days).");
  });
  it("returns evidence for a published-date check", () => {
    const sentence = "The end date remains 18/09/2028.";
    const result = engine.detectScheduleClaims(`The review finished. ${sentence}`, {
      previousFinish: "2028-04-01", currentFinish: "2028-04-01",
    });
    assert.deepEqual(result.evidence, {
      sentence, phrase: "18/09/2028", narrativeDate: "2028-09-18",
      previousFinish: "2028-04-01", currentFinish: "2028-04-01",
    });
  });
  it("keeps the whole sentence rather than a bounded clause", () => {
    const sentence = `The programme remains on schedule, ${"with review work continuing, ".repeat(8)}and the review is complete.`;
    assert.equal(engine.detectScheduleClaims(sentence, dates).evidence?.sentence, sentence);
  });
  it("returns the mismatched date as evidence", () => {
    const sentence = "The project will complete by 15 December 2027.";
    const evidence = engine.detectScheduleClaims(sentence, dates).evidence;
    assert.equal(evidence?.sentence, sentence);
    assert.equal(evidence?.phrase, "15 December 2027");
    assert.equal(evidence?.narrativeDate, dates.previousFinish);
  });
  it("returns direction-only evidence when the stated direction disagrees", () => {
    const sentence = "The end date was extended following review.";
    const result = engine.detectScheduleClaims(sentence, {
      previousFinish: "2028-02-26", currentFinish: "2027-12-15",
    });
    assert.equal(result.evidence?.sentence, sentence);
    assert.equal(result.evidence?.phrase, "was extended");
  });
  it("returns no finding evidence for an explained move", () => {
    assert.equal(engine.detectScheduleClaims("The end date moved to 26 February 2028.", dates).evidence, null);
  });
  for (const date of ["2029", "2029-Q1"]) {
    it(`keeps the stated precision in the finding for ${date}`, () => {
      const options = { previousFinish: "2027-12-15", currentFinish: "2027-12-15" };
      const result = engine.detectScheduleClaims(`The end date remains ${date}.`, options);
      assert.equal(engine.findingSentence(result, options),
        `The narrative gives the end date as ${date}; the published end date is 15 Dec 2027. Check which is current.`);
      assert.equal(result.publishedDateCheck.detail, engine.findingSentence(result, options));
    });
  }
});
