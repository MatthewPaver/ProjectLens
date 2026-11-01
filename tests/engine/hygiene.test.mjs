// Real-export hygiene: encodings, multi-project listing, completed-work float, speed.
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

const DEFAULT_RULES = {
  maxDurationHours: 20 * 8,
  materialSlipDays: 10,
  floatThresholdHours: 0,
  nearCriticalHours: 10 * 8,
  executiveChangeLimit: 50,
  allowPositiveLag: false,
};

describe("real-export hygiene", () => {
  it("decodeXerBytes reads Windows-1252 Café", () => {
    assert.equal(typeof engine.decodeXerBytes, "function");
    // Cafés: C a f é as Windows-1252 (é = 0xE9)
    const bytes = Uint8Array.from([0x43, 0x61, 0x66, 0xE9]);
    const decoded = engine.decodeXerBytes(bytes);
    assert.equal(decoded.text, "Café");
    assert.equal(decoded.encoding, "windows-1252");
  });

  it("listProjects finds 2+ projects in the multi-project variant", () => {
    assert.equal(typeof engine.listProjects, "function");
    const text = fs.readFileSync(path.join(RESEARCH, "variants/R4-multiproject/two-projects.xer"), "utf8");
    const projects = engine.listProjects(text);
    assert.ok(projects.length >= 2, `got ${projects.length}`);
    assert.ok(projects.every(p => p.projId && p.activityCount >= 0));
  });

  it("completed-work-only update does not raise Evidence gap for blank float", () => {
    // R1 current has blank float on completed activities — that must not be an evidence gap.
    const previous = engine.parseXerAssurance(
      fs.readFileSync(path.join(RESEARCH, "variants/R1-airport-terminal2/previous.xer"), "utf8"),
      "previous.xer",
    );
    const current = engine.parseXerAssurance(
      fs.readFileSync(path.join(RESEARCH, "variants/R1-airport-terminal2/current.xer"), "utf8"),
      "current.xer",
    );
    const review = engine.analyseReview(
      previous,
      current,
      "The finish moved later as planned. Owner: Planner. Due 1 Jan 2013.",
    );
    assert.ok(
      !review.blockers.some(b => b.type === "Evidence gap" && /float values could not be compared/i.test(b.title)),
      JSON.stringify(review.blockers.map(b => b.title)),
    );
  });

  it("6,977-activity file compares in under 2000 ms", () => {
    const prevPath = path.join(RESEARCH, "variants/R2-rehab-large/previous.xer");
    const curPath = path.join(RESEARCH, "variants/R2-rehab-large/current.xer");
    const prevText = fs.readFileSync(prevPath);
    const curText = fs.readFileSync(curPath);
    const t0 = Date.now();
    const previous = engine.parseXer(engine.decodeXerBytes(prevText).text, "previous.xer");
    const current = engine.parseXer(engine.decodeXerBytes(curText).text, "current.xer");
    engine.compareSchedules(previous, current, DEFAULT_RULES);
    const ms = Date.now() - t0;
    assert.ok(ms < 2000, `took ${ms} ms`);
  });
});
