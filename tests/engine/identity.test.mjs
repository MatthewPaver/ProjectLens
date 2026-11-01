// Project identity via activity-code overlap (renamed / re-numbered schedules).
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

function read(rel) {
  return fs.readFileSync(path.join(RESEARCH, "real-xer", rel), "utf8");
}

function rewriteIdentity(text, { projId, shortName }) {
  // Rewrite PROJECT identity fields only; keep every task_code so overlap stays high.
  return text.split(/\r?\n/).map(line => {
    const cells = line.split("\t");
    if (cells[0] === "%F" && cells.includes("proj_id")) {
      rewriteIdentity._fields = cells.slice(1);
      return line;
    }
    if (cells[0] === "%R" && rewriteIdentity._fields) {
      const fields = rewriteIdentity._fields;
      const row = Object.fromEntries(fields.map((f, i) => [f, cells[i + 1] ?? ""]));
      if ("proj_id" in row) row.proj_id = projId;
      if ("proj_short_name" in row) row.proj_short_name = shortName;
      return ["%R", ...fields.map(f => row[f] ?? "")].join("\t");
    }
    if (cells[0] === "%T") rewriteIdentity._fields = null;
    return line;
  }).join("\n");
}

describe("matchProjects", () => {
  it("exports matchProjects", () => {
    assert.equal(typeof engine.matchProjects, "function");
  });

  it("same real-xer with rewritten proj_id and name scores same", () => {
    const original = read("xer-splitter-airport/airport.xer");
    const rewritten = rewriteIdentity(original, { projId: "99999", shortName: "TERMINAL-2-UPD01" });
    const previous = engine.parseXerAssurance(original, "airport.xer");
    // Assurance parser rejects multi-project; airport is single-project.
    // Use review parser for the rewritten copy then build a minimal assurance-like view via parseXerAssurance.
    const current = engine.parseXerAssurance(rewritten, "airport-renamed.xer");
    const match = engine.matchProjects(previous, current);
    assert.ok(match.score >= 0.8, `score ${match.score}`);
    assert.equal(match.decision, "same");
    assert.ok(match.shared > 0);
  });

  it("two unrelated real-xer files score different", () => {
    const a = engine.parseXerAssurance(read("ashspace-sample/sample.xer"), "ashspace.xer");
    const b = engine.parseXerAssurance(read("p6-analyzer-example/example_xer_file.xer"), "example.xer");
    const match = engine.matchProjects(a, b);
    assert.ok(match.score < 0.5, `score ${match.score}`);
    assert.equal(match.decision, "different");
  });
});
