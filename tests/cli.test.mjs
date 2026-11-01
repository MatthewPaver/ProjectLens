// CLI: compare, batch and input-error exit codes.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..");
const CLI = path.join(REPO, "bin/projectlens.mjs");

function run(args, opts = {}) {
  return spawnSync(process.execPath, [CLI, ...args], {
    encoding: "utf8",
    cwd: REPO,
    ...opts,
  });
}

describe("projectlens CLI", () => {
  it("compare on Northstar exits 1 with a 73-day contradiction in JSON", () => {
    const result = run([
      "compare",
      "--previous", "docs/demo/northstar-previous.xer",
      "--current", "docs/demo/northstar-current.xer",
      "--narrative", "docs/demo/northstar-narrative.txt",
      "--format", "json",
    ]);
    assert.equal(result.status, 1, result.stderr || result.stdout);
    const payload = JSON.parse(result.stdout);
    assert.equal(payload.finishMovement, 73);
    assert.ok(payload.contradiction);
  });

  it("batch on examples/monthly writes the history CSV header", () => {
    const out = path.join(os.tmpdir(), `projectlens-batch-${Date.now()}.csv`);
    const result = run([
      "batch",
      "--dir", "examples/monthly",
      "--format", "csv",
      "--out", out,
    ]);
    assert.ok(result.status === 0 || result.status === 1, result.stderr || result.stdout);
    const csv = fs.readFileSync(out, "utf8");
    assert.match(
      csv.split(/\r?\n/)[0],
      /^project,previous_date,current_date,finish_previous,finish_current,days_moved,narrative_claim,contradiction$/,
    );
    fs.unlinkSync(out);
  });

  it("refuses .mpp with exit 3", () => {
    const mpp = path.join(os.tmpdir(), `fake-${Date.now()}.mpp`);
    fs.writeFileSync(mpp, "not a real mpp");
    const result = run([
      "compare",
      "--previous", mpp,
      "--current", "docs/demo/northstar-current.xer",
    ]);
    fs.unlinkSync(mpp);
    assert.equal(result.status, 3, result.stderr || result.stdout);
    assert.match(result.stderr || result.stdout, /Export this schedule from MS Project as XML/);
  });
});

describe("Round 4 plain findings", () => {
  const args = ["compare", "--previous", "examples/monthly/northstar-previous.xer",
    "--current", "examples/monthly/northstar-current.xer",
    "--narrative", "examples/monthly/northstar-narrative.txt"];
  const sentence = "The update says 'remains on track'. The schedule's finish moved from 15 Dec 2027 to 26 Feb 2028 (73 days).";
  it("prints the shared contradiction sentence", () => {
    const result = run([...args, "--format", "text"]);
    assert.equal(result.status, 1);
    assert.ok(result.stdout.includes(sentence), result.stdout);
  });
  it("includes claim, contradiction and publishedDateCheck in JSON", () => {
    const result = run([...args, "--format", "json"]);
    const payload = JSON.parse(result.stdout);
    assert.equal(payload.claim, sentence);
    assert.equal(payload.contradiction, true);
    assert.equal(payload.publishedDateCheck, null);
    assert.equal(payload.claims.evidence.phrase, "remains on track");
    assert.equal(payload.claims.evidence.previousFinish, "2027-12-15");
    assert.equal(payload.claims.evidence.currentFinish, "2028-02-26");
  });
  for (const format of ["text", "md"]) {
    it(`quotes the exact sentence and marks the matched phrase in ${format}`, () => {
      const result = run([...args, "--format", format]);
      assert.equal(result.status, 1);
      assert.match(result.stdout, /Northstar Grid Connection \*\*remains on track\*\* and the key completion date is unchanged\./);
      assert.ok(result.stdout.includes(sentence), result.stdout);
    });
  }
});
