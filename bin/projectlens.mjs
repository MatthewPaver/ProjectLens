#!/usr/bin/env node
/**
 * ProjectLens local CLI — compare two schedules, or batch a folder by project.
 * No network calls. Exit 0 = no contradiction, 1 = contradiction found, 3 = input error.
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..");
const require = createRequire(import.meta.url);
const engine = require(path.join(REPO, "docs/engine/projectlens-engine.js"));
require(path.join(REPO, "docs/engine/xml-schedules.js"));

function fail(message, code = 3) {
  process.stderr.write(`${message}\n`);
  process.exit(code);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token.startsWith("--")) {
      const key = token.slice(2);
      const next = argv[i + 1];
      if (!next || next.startsWith("--")) args[key] = true;
      else {
        args[key] = next;
        i += 1;
      }
    } else args._.push(token);
  }
  return args;
}

function readSchedule(filePath, projectId) {
  if (!filePath) fail("Missing schedule file path.");
  if (/\.mpp$/i.test(filePath)) {
    fail("Export this schedule from MS Project as XML (File > Save As > XML) and load that");
  }
  if (/\.pdf$/i.test(filePath)) fail("Paste the narrative text");
  if (!fs.existsSync(filePath)) fail(`File not found: ${filePath}`);
  const bytes = fs.readFileSync(filePath);
  const name = path.basename(filePath);
  if (/\.xml$/i.test(filePath)) {
    const text = bytes.toString("utf8");
    if (/<APIBusinessObjects|<Project[\s\S]*<Activity/i.test(text)) {
      return engine.parseP6Xml(text, name);
    }
    return engine.parseMspdiXml(text, name);
  }
  if (/\.xer$/i.test(filePath)) {
    const { text } = engine.decodeXerBytes(bytes);
    return engine.parseXerAssurance(text, name, projectId ? { projectId } : {});
  }
  fail(`Unsupported schedule format: ${name}`);
}

function readNarrative(filePath) {
  if (!filePath) return "";
  if (/\.pdf$/i.test(filePath)) fail("Paste the narrative text");
  if (/\.docx$/i.test(filePath)) fail("CLI narrative input accepts .txt or .md; paste Word text or convert first.");
  return fs.readFileSync(filePath, "utf8");
}

function iso(value) {
  return engine.isoDate(value) || "";
}

function comparePair(previous, current, narrative, { confirmedMatch = true } = {}) {
  const review = engine.analyseReview(previous, current, narrative, { confirmedMatch });
  const claims = engine.detectScheduleClaims(narrative, {
    currentFinish: current.project.scheduledFinish,
    previousFinish: previous.project.scheduledFinish,
  });
  return {
    project: current.project.name,
    finishMovement: review.finishMovement,
    finishPrevious: iso(previous.project.scheduledFinish),
    finishCurrent: iso(current.project.scheduledFinish),
    dataDatePrevious: iso(previous.project.dataDate),
    dataDateCurrent: iso(current.project.dataDate),
    contradiction: Boolean(claims.contradiction) || review.blockers.some(b => b.type === "Contradiction"),
    narrativeClaim: claims.stable[0]?.phrase || claims.movesStated[0] && "move stated" || "",
    claim: engine.findingSentence(claims, {
      currentFinish: current.project.scheduledFinish,
      previousFinish: previous.project.scheduledFinish,
    }),
    publishedDateCheck: claims.publishedDateCheck,
    claims,
    blockers: review.blockers,
  };
}

function formatResult(result, format) {
  if (format === "json") return `${JSON.stringify(result, null, 2)}\n`;
  if (format === "csv") {
    const header = "project,previous_date,current_date,finish_previous,finish_current,days_moved,narrative_claim,contradiction";
    const row = [
      result.project,
      result.dataDatePrevious,
      result.dataDateCurrent,
      result.finishPrevious,
      result.finishCurrent,
      result.finishMovement,
      JSON.stringify(result.narrativeClaim || ""),
      result.contradiction,
    ].join(",");
    return `${header}\n${row}\n`;
  }
  const evidence = result.claims.evidence;
  const quotation = evidence ? `Update: ${evidence.sentence.replace(evidence.phrase, phrase => `**${phrase}**`)}` : null;
  if (format === "md") {
    const lines = [
      `# ${result.project}`,
      "",
      `- Finish movement: ${result.finishMovement} days (${result.finishPrevious} → ${result.finishCurrent})`,
      `- Contradiction: ${result.contradiction ? "yes" : "no"}`,
      `- Narrative claim: ${result.narrativeClaim || "none"}`,
    ];
    if (result.claim) lines.push(`- ${result.claim}`);
    if (quotation) lines.push("", quotation);
    lines.push("");
    return lines.join("\n");
  }
  // text
  const lines = [
    `Project: ${result.project}`,
    `Finish: ${result.finishPrevious} → ${result.finishCurrent} (${result.finishMovement} days)`,
    `Contradiction: ${result.contradiction ? "yes" : "no"}`,
  ];
  if (result.narrativeClaim) lines.push(`Narrative claim: ${result.narrativeClaim}`);
  if (result.claim) lines.push(result.claim);
  if (quotation) lines.push(quotation);
  for (const blocker of (result.blockers || []).filter(item => item.title !== result.claim)) lines.push(`- ${blocker.type}: ${blocker.title}`);
  return `${lines.join("\n")}\n`;
}

function cmdCompare(args) {
  const format = args.format || "text";
  const previous = readSchedule(args.previous, args.project);
  const current = readSchedule(args.current, args.project);
  const narrative = readNarrative(args.narrative);
  let result;
  try {
    result = comparePair(previous, current, narrative);
  } catch (error) {
    fail(error.message || String(error));
  }
  process.stdout.write(formatResult(result, format));
  process.exit(result.contradiction ? 1 : 0);
}

function listScheduleFiles(dir) {
  return fs.readdirSync(dir)
    .filter(name => /\.(xer|xml)$/i.test(name))
    .map(name => path.join(dir, name))
    .sort();
}

function cmdBatch(args) {
  const dir = args.dir;
  if (!dir || !fs.existsSync(dir)) fail("batch requires --dir <folder>");
  const format = args.format || "csv";
  if (!["csv", "json"].includes(format)) fail("batch --format must be csv or json");
  const files = listScheduleFiles(dir);
  if (files.length < 2) fail("batch needs at least two schedule files in the folder");

  const parsed = files.map(filePath => {
    const schedule = readSchedule(filePath);
    return { filePath, schedule, dataDate: schedule.project.dataDate?.getTime() || 0 };
  });

  // Group by matchProjects clusters (greedy).
  const groups = [];
  for (const item of parsed) {
    let placed = false;
    for (const group of groups) {
      const probe = engine.matchProjects(group[0].schedule, item.schedule);
      if (probe.decision !== "different") {
        group.push(item);
        placed = true;
        break;
      }
    }
    if (!placed) groups.push([item]);
  }

  const narrativePath = ["narrative.txt", "northstar-narrative.txt", "narrative.md"]
    .map(name => path.join(dir, name))
    .find(p => fs.existsSync(p));
  const narrative = narrativePath ? fs.readFileSync(narrativePath, "utf8") : "";

  const rows = [];
  let anyContradiction = false;
  for (const group of groups) {
    group.sort((a, b) => a.dataDate - b.dataDate);
    for (let i = 1; i < group.length; i += 1) {
      const previous = group[i - 1].schedule;
      const current = group[i].schedule;
      try {
        const result = comparePair(previous, current, narrative);
        anyContradiction = anyContradiction || result.contradiction;
        rows.push({
          project: result.project,
          previous_date: result.dataDatePrevious,
          current_date: result.dataDateCurrent,
          finish_previous: result.finishPrevious,
          finish_current: result.finishCurrent,
          days_moved: result.finishMovement,
          narrative_claim: result.narrativeClaim,
          contradiction: result.contradiction,
        });
      } catch (error) {
        fail(error.message || String(error));
      }
    }
  }

  const header = "project,previous_date,current_date,finish_previous,finish_current,days_moved,narrative_claim,contradiction";
  let output;
  if (format === "json") output = `${JSON.stringify(rows, null, 2)}\n`;
  else {
    const lines = [header, ...rows.map(row => [
      row.project,
      row.previous_date,
      row.current_date,
      row.finish_previous,
      row.finish_current,
      row.days_moved,
      JSON.stringify(row.narrative_claim || ""),
      row.contradiction,
    ].join(","))];
    output = `${lines.join("\n")}\n`;
  }

  if (args.out) fs.writeFileSync(args.out, output);
  else process.stdout.write(output);
  process.exit(anyContradiction ? 1 : 0);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const command = args._[0];
  if (command === "compare") return cmdCompare(args);
  if (command === "batch") return cmdBatch(args);
  fail("Usage: projectlens compare|batch ...");
}

main();
