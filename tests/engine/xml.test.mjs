// Same mini schedule in XER, P6 XML and MSPDI must yield the same taskChanges.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "../..");
const FIX = path.join(REPO, "tests/fixtures/schedules");
const require = createRequire(import.meta.url);
const engine = require(path.join(REPO, "docs/engine/projectlens-engine.js"));
require(path.join(REPO, "docs/engine/xml-schedules.js")); // attaches parseP6Xml / parseMspdiXml

function summarise(changes) {
  return changes.changes
    .map(c => `${c.code}:${c.summary}`)
    .sort();
}

describe("XML schedule parsers", () => {
  it("exports parseP6Xml and parseMspdiXml", () => {
    assert.equal(typeof engine.parseP6Xml, "function");
    assert.equal(typeof engine.parseMspdiXml, "function");
  });

  it("XER, P6 XML and MSPDI give the same taskChanges against the edited copy", () => {
    const xerPrev = engine.parseXerAssurance(fs.readFileSync(path.join(FIX, "mini-previous.xer"), "utf8"), "mini-previous.xer");
    const xerCur = engine.parseXerAssurance(fs.readFileSync(path.join(FIX, "mini-current.xer"), "utf8"), "mini-current.xer");
    const xerChanges = engine.taskChanges(xerPrev, xerCur);

    const p6Prev = engine.parseP6Xml(fs.readFileSync(path.join(FIX, "mini-previous.p6.xml"), "utf8"), "mini-previous.p6.xml");
    const p6Cur = engine.parseP6Xml(fs.readFileSync(path.join(FIX, "mini-current.p6.xml"), "utf8"), "mini-current.p6.xml");
    // Convert review-shaped XML parses into assurance shape for taskChanges.
    const toAssurance = parsed => ({
      ...parsed,
      sourceText: parsed.sourceText || JSON.stringify(parsed.project),
      taskByCode: new Map(parsed.tasks.map(t => [t.code, t])),
      relations: parsed.relations instanceof Set
        ? parsed.relations
        : new Set((parsed.relations || []).map(r => JSON.stringify([r.predecessorCode, r.taskCode, r.type || "PR_FS", r.lag || 0]))),
    });
    const p6Changes = engine.taskChanges(toAssurance(p6Prev), toAssurance(p6Cur));

    const mspPrev = engine.parseMspdiXml(fs.readFileSync(path.join(FIX, "mini-previous.mspdi.xml"), "utf8"), "mini-previous.mspdi.xml");
    const mspCur = engine.parseMspdiXml(fs.readFileSync(path.join(FIX, "mini-current.mspdi.xml"), "utf8"), "mini-current.mspdi.xml");
    const mspChanges = engine.taskChanges(toAssurance(mspPrev), toAssurance(mspCur));

    assert.deepEqual(summarise(p6Changes), summarise(xerChanges));
    assert.deepEqual(summarise(mspChanges), summarise(xerChanges));
    assert.ok(xerChanges.changes.length > 0);
  });
});
