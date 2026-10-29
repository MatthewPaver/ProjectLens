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
