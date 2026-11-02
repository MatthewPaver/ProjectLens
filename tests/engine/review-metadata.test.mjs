import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { it } from "node:test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const require = createRequire(import.meta.url);
const engine = require(path.join(root, "docs/engine/projectlens-engine.js"));
const source = fs.readFileSync(path.join(root, "docs/change-assurance.js"), "utf8");
const previous = engine.parseXerAssurance(fs.readFileSync(path.join(root, "docs/demo/northstar-previous.xer"), "utf8"), "previous.xer");
const current = engine.parseXerAssurance(fs.readFileSync(path.join(root, "docs/demo/northstar-current.xer"), "utf8"), "current.xer");

function renderedMetadata(before, after) {
  const nodes = new Map();
  const select = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {});
    return nodes.get(selector);
  };
  const helperStart = source.indexOf("function evidenceVersion(");
  const helperEnd = source.indexOf("async function inputIdentity(", helperStart);
  const renderStart = source.indexOf("function renderReview(");
  const renderEnd = source.indexOf('  $("#readinessCard")', renderStart);
  assert.ok(helperStart >= 0 && helperEnd > helperStart && renderEnd > renderStart);
  // Execute the production metadata binding, before unrelated workspace rendering.
  const render = vm.runInNewContext(source.slice(helperStart, helperEnd) + source.slice(renderStart, renderEnd) + "}\nrenderReview;", {
    $: select, assuranceState: {}, formatDate: engine.formatDate, displayProjectName: name => name,
  });
  render({ previous: before, current: after, blockers: [], inputIdentity: "0123456789abcdef", generatedAt: "2026-10-08T12:00:00Z" });
  return nodes.get("#reviewEvidenceVersion").textContent;
}

it("shows both parsed data dates, separately from finish dates and check time", () => {
  const text = renderedMetadata(previous, current);
  assert.match(text, /Previous data date: 31 Jan 2026/);
  assert.match(text, /current data date: 28 Feb 2026/);
  assert.match(text, /previous\.xer → current\.xer/);
  assert.match(text, /SHA-256 0123456789ab/);
  assert.match(text, /checked/);
});

it("retains data dates after a saved review is restored from JSON", () => {
  const restored = JSON.parse(JSON.stringify({ previous, current }));
  assert.match(renderedMetadata(restored.previous, restored.current), /Previous data date: 31 Jan 2026; current data date: 28 Feb 2026/);
});

it("labels missing or invalid source dates without inventing a date", () => {
  const before = { ...previous, project: { ...previous.project, dataDate: null } };
  const after = { ...current, project: { ...current.project, dataDate: "not-a-date" } };
  assert.match(renderedMetadata(before, after), /Previous data date: Not supplied; current data date: Not supplied/);
});
