// Real-case accuracy benchmark for the narrative check. Every pair is real published
// UK Government Major Projects Portfolio data (Open Government Licence); sources in SOURCES.md.
// Labels: contradiction | explained | no-move | published-date-check.
// Exits 1 if any set scores below its floor in floors.json (the score when this was added).
// Usage: node bench/real-cases/run.mjs [--update-floors]
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const dir = path.dirname(fileURLToPath(import.meta.url));
const E = require('../../docs/engine/projectlens-engine.js');
const floors = JSON.parse(fs.readFileSync(path.join(dir, 'floors.json'), 'utf8'));
const sets = fs.readdirSync(dir).filter((f) => /^projectlens-heldout\d*\.json$/.test(f)).sort();
const scores = {};
let failed = false;
for (const file of sets) {
  const raw = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
  const pairs = Array.isArray(raw) ? raw : (raw.pairs || raw.cases);
  const byLabel = {};
  const misses = [];
  let right = 0;
  for (const p of pairs) {
    const r = E.detectScheduleClaims(p.narrative || '', { previousFinish: p.prevEnd, currentFinish: p.curEnd });
    const got = r.contradiction ? 'contradiction' : (r.publishedDateCheck ? 'published-date-check' : 'none');
    const want = p.label === 'contradiction' || p.label === 'published-date-check' ? p.label : 'none';
    const ok = got === want;
    byLabel[p.label] ||= [0, 0];
    byLabel[p.label][1] += 1;
    if (ok) { right += 1; byLabel[p.label][0] += 1; } else misses.push(`  ${p.id} ${p.label}, got ${got} (moved ${p.daysMoved} days): ${p.project.slice(0, 60)}`);
  }
  scores[file] = right;
  const floor = floors[file] ?? 0;
  if (right < floor) failed = true;
  const parts = Object.entries(byLabel).map(([k, [a, b]]) => `${k} ${a}/${b}`).join(', ');
  console.log(`${file}: ${right}/${pairs.length} (floor ${floor}) ${right < floor ? 'BELOW FLOOR' : 'ok'} | ${parts}`);
  if (misses.length) console.log(misses.join('\n'));
}
if (process.argv.includes('--update-floors')) {
  fs.writeFileSync(path.join(dir, 'floors.json'), `${JSON.stringify(scores, null, 2)}\n`);
  console.log('floors.json updated');
}
process.exit(failed ? 1 : 0);
