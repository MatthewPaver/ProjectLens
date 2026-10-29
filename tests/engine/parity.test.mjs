// Parity: shared engine must match origin/main page engines on Northstar + real XERs.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "../..");
const RESEARCH = path.resolve(REPO, "../research_notes/Real case tests/projectlens-inputs");
const require = createRequire(import.meta.url);

const DEFAULT_RULES = {
  maxDurationHours: 20 * 8,
  materialSlipDays: 10,
  floatThresholdHours: 0,
  nearCriticalHours: 10 * 8,
  executiveChangeLimit: 10,
  allowPositiveLag: false,
};

const REAL_XER_SEVEN = [
  "xer-splitter-airport/airport.xer",
  "xer-splitter-rehab/rehab-2.xer",
  "ashspace-sample/sample.xer",
  "p6-analyzer-example/example_xer_file.xer",
  "mpxj-calendar/ProjectDefaultCalendar.xer",
  "mpxj-percent/percent-complete-8.4.xer",
  "mpxj-rates/resource-rates-test.xer",
];

function gitShow(relPath) {
  return execFileSync("git", ["show", `origin/main:${relPath}`], {
    cwd: REPO,
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  });
}

function slice(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  if (start < 0 || end < 0) throw new Error(`Markers not found: ${startMarker} / ${endMarker}`);
  return source.slice(start, end);
}

function loadLegacyEngines() {
  const ca = gitShow("docs/change-assurance.js");
  const xr = gitShow("docs/xer-review.js");
  const caCode = slice(ca, "function escapeHtml(", "function showToast(");
  const xrCode = slice(xr, "function numberValue(", "function updateFileCard(");
  const caCtx = vm.createContext({ Intl, Date, Math, Number, String, JSON, Map, Set, Error, console });
  vm.runInContext(`${caCode}\n;globalThis.__e = { parseXer, analyseReview, taskChanges };`, caCtx);
  const xrCtx = vm.createContext({ Intl, Date, Math, Number, String, JSON, Map, Set, Error, console });
  vm.runInContext(`${xrCode}\n;globalThis.__e = { parseXer, buildReview, compareSchedules, parseEvidenceText };`, xrCtx);
  return { assurance: caCtx.__e, review: xrCtx.__e };
}

function readUtf8(file) {
  return new TextDecoder("utf-8").decode(fs.readFileSync(file));
}

function isMapLike(value) {
  return Boolean(value) && typeof value.get === "function" && typeof value.keys === "function" && value[Symbol.iterator];
}

function isSetLike(value) {
  return Boolean(value) && typeof value.has === "function" && typeof value.add === "function" && value[Symbol.iterator] && value.size !== undefined && !isMapLike(value);
}

function stripVolatile(value) {
  if (value instanceof Date) return value.toISOString();
  // instanceof fails across node:vm realms, so duck-type Map/Set.
  if (isMapLike(value)) {
    return { __map: [...value.entries()].map(([k, v]) => [k, stripVolatile(v)]) };
  }
  if (isSetLike(value)) {
    return { __set: [...value].map(stripVolatile).sort() };
  }
  // Copy into this realm: legacy engines run in node:vm, and vm arrays fail deepEqual.
  if (Array.isArray(value)) return Array.from(value, stripVolatile);
  if (value && typeof value === "object") {
    const out = {};
    for (const [key, child] of Object.entries(value)) {
      // Skip timestamps, raw XER rows, and nested schedule graphs (they embed the same rows).
      if (
        key === "generatedAt"
        || key === "raw"
        || key === "tables"
        || key === "previous"
        || key === "current"
        || key === "before"
        || key === "after"
        || key === "baseline"
        || key === "taskById"
        || key === "taskByCode"
        || key === "sourceText"
      ) continue;
      out[key] = stripVolatile(child);
    }
    return out;
  }
  return value;
}

function loadEngine() {
  return require(path.join(REPO, "docs/engine/projectlens-engine.js"));
}

describe("ProjectLensEngine parity with origin/main", () => {
  it("exports the shared surface", () => {
    const engine = loadEngine();
    for (const name of [
      "parseXer",
      "parseXerAssurance",
      "taskChanges",
      "analyseReview",
      "compareSchedules",
      "buildReview",
      "parseEvidenceText",
    ]) {
      assert.equal(typeof engine[name], "function", `missing ${name}`);
    }
  });

  it("matches Northstar assurance and schedule-review outputs", () => {
    const legacy = loadLegacyEngines();
    const engine = loadEngine();
    const prevPath = path.join(REPO, "docs/demo/northstar-previous.xer");
    const curPath = path.join(REPO, "docs/demo/northstar-current.xer");
    const narrative = fs.readFileSync(path.join(REPO, "docs/demo/northstar-narrative.txt"), "utf8");
    const prevText = readUtf8(prevPath);
    const curText = readUtf8(curPath);

    const legacyPrevA = legacy.assurance.parseXer(prevText, "northstar-previous.xer");
    const legacyCurA = legacy.assurance.parseXer(curText, "northstar-current.xer");
    const legacyReviewA = legacy.assurance.analyseReview(legacyPrevA, legacyCurA, narrative.trim());

    const engPrevA = engine.parseXerAssurance(prevText, "northstar-previous.xer");
    const engCurA = engine.parseXerAssurance(curText, "northstar-current.xer");
    const engReviewA = engine.analyseReview(engPrevA, engCurA, narrative.trim());
    // Schedule deltas must stay byte-identical to origin/main; narrative claim wording may evolve.
    assert.equal(engReviewA.finishMovement, legacyReviewA.finishMovement);
    assert.deepEqual(stripVolatile(engReviewA.changes), stripVolatile(legacyReviewA.changes));
    assert.ok(engReviewA.blockers.some(b => b.type === "Contradiction"));

    const legacyPrevR = legacy.review.parseXer(prevText, "northstar-previous.xer");
    const legacyCurR = legacy.review.parseXer(curText, "northstar-current.xer");
    const legacyReviewR = legacy.review.buildReview(legacyPrevR, legacyCurR, narrative.trim(), DEFAULT_RULES, {}, null);

    const engPrevR = engine.parseXer(prevText, "northstar-previous.xer");
    const engCurR = engine.parseXer(curText, "northstar-current.xer");
    const engReviewR = engine.buildReview(engPrevR, engCurR, narrative.trim(), DEFAULT_RULES, {}, null);
    assert.equal(engReviewR.finishMovement.days, legacyReviewR.finishMovement.days);
    assert.deepEqual(stripVolatile(engReviewR.changes), stripVolatile(legacyReviewR.changes));
    assert.ok(engReviewR.conflicts.some(c => /contradiction/i.test(c.type)));
  });

  it("parses the seven real public XER samples like origin/main", () => {
    const legacy = loadLegacyEngines();
    const engine = loadEngine();
    for (const rel of REAL_XER_SEVEN) {
      const file = path.join(RESEARCH, "real-xer", rel);
      assert.ok(fs.existsSync(file), `missing fixture ${rel}`);
      const text = readUtf8(file);
      const name = path.basename(file);

      let legacyReviewResult;
      let legacyReviewError;
      try {
        legacyReviewResult = stripVolatile(legacy.review.parseXer(text, name));
      } catch (error) {
        legacyReviewError = error.message;
      }

      let engReviewResult;
      let engReviewError;
      try {
        engReviewResult = stripVolatile(engine.parseXer(text, name));
      } catch (error) {
        engReviewError = error.message;
      }
      assert.equal(engReviewError, legacyReviewError, `review parse error mismatch for ${rel}`);
      if (!legacyReviewError) assert.deepEqual(engReviewResult, legacyReviewResult);

      let legacyAssuranceResult;
      let legacyAssuranceError;
      try {
        legacyAssuranceResult = stripVolatile(legacy.assurance.parseXer(text, name));
      } catch (error) {
        legacyAssuranceError = error.message;
      }

      let engAssuranceResult;
      let engAssuranceError;
      try {
        engAssuranceResult = stripVolatile(engine.parseXerAssurance(text, name));
      } catch (error) {
        engAssuranceError = error.message;
      }
      assert.equal(engAssuranceError, legacyAssuranceError, `assurance parse error mismatch for ${rel}`);
      if (!legacyAssuranceError) assert.deepEqual(engAssuranceResult, legacyAssuranceResult);
    }
  });
});
