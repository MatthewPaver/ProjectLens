/**
 * ProjectLens shared schedule engine.
 * Works as window.ProjectLensEngine in the browser and as module.exports in Node.
 * Two parse entry points exist because change-assurance and schedule-review historically
 * returned different shapes; Task 1 is a move, not a behaviour change.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(typeof require === "function" ? require : null);
  } else {
    root.ProjectLensEngine = factory(null);
  }
}(typeof self !== "undefined" ? self : this, function (nodeRequire) {
"use strict";

// Keep in sync with claim-rules.json (Node loads the JSON file; browser uses this copy).
const EMBEDDED_CLAIM_RULES = {
  stablePhrases: [
    "unchanged", "remained the same", "remains the same", "remained at", "remains at",
    "remained scheduled", "remains on track", "on track", "on-track", "on target", "on schedule",
    "no slippage", "no change", "not moved", "still forecast", "still expected",
    "as planned", "no material change", "no impact",
    "remained static", "remains static", "has not changed", "have not changed", "not changed",
    "no movement", "unaltered", "remained constant", "remains constant",
  ],
  movePhrases: [
    "increased from", "increased to", "decreased from", "changed from", "moved from", "moved to",
    "has moved to", "has been moved", "has been extended", "extended to", "slipped from",
    "revised from", "revised to", "re-baselined", "correction of published position", "delayed to",
    "now forecast",
  ],
  negations: ["not", "never", "no longer"],
};

function loadClaimRules() {
  if (nodeRequire) {
    try {
      return nodeRequire("./claim-rules.json");
    } catch (error) {
      // Fall through to the embedded copy when the JSON path is unavailable.
    }
  }
  return EMBEDDED_CLAIM_RULES;
}

function numberValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(String(value).replace(/,/g, ""));
  return Number.isFinite(number) ? number : null;
}

function decodeXerBytes(bytes) {
  // Try strict UTF-8 first (what File.text() often does), then honour ERMHDR hints.
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(arr);
    return { text, encoding: "utf-8" };
  } catch (error) {
    // Not valid UTF-8 — fall through to a Windows code page.
  }
  const head = new TextDecoder("windows-1252").decode(arr.subarray(0, Math.min(arr.length, 240)));
  if (/1256|arabic/i.test(head)) {
    return { text: new TextDecoder("windows-1256").decode(arr), encoding: "windows-1256" };
  }
  return { text: new TextDecoder("windows-1252").decode(arr), encoding: "windows-1252" };
}

function parseXerTables(text) {
  const tables = {};
  let tableName = null;
  let fields = [];
  const lines = String(text || "").replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n").split("\n");
  for (const line of lines) {
    if (!line) continue;
    const cells = line.split("\t");
    const marker = cells[0].trim();
    if (marker === "%T") {
      tableName = (cells[1] || "").trim().toUpperCase();
      if (tableName) tables[tableName] ||= [];
      fields = [];
    } else if (marker === "%F" && tableName) {
      fields = cells.slice(1).map(field => field.trim());
    } else if (marker === "%R" && tableName && fields.length) {
      const record = {};
      fields.forEach((field, index) => { record[field] = cells[index + 1] ?? ""; });
      tables[tableName].push(record);
    }
  }
  return tables;
}

function listProjects(text) {
  const tables = parseXerTables(text);
  const projects = tables.PROJECT || [];
  const tasks = tables.TASK || [];
  return projects.map(project => {
    const projId = project.proj_id || "";
    const activityCount = tasks.filter(row => !projId || row.proj_id === projId).length;
    return {
      projId,
      shortName: project.proj_short_name || project.proj_name || "Unnamed project",
      name: project.proj_name || project.proj_short_name || "Unnamed project",
      activityCount,
    };
  });
}

function parseDate(value) {
  if (!value) return null;
  const clean = String(value).trim();
  const isoMatch = clean.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (isoMatch) return new Date(`${isoMatch[1]}-${isoMatch[2]}-${isoMatch[3]}T12:00:00Z`);
  const timestamp = Date.parse(clean);
  return Number.isNaN(timestamp) ? null : new Date(timestamp);
}

function isoDate(value) {
  const parsed = value instanceof Date ? value : parseDate(value);
  return parsed ? parsed.toISOString().slice(0, 10) : null;
}

function formatDate(value) {
  const parsed = value instanceof Date ? value : parseDate(value);
  if (!parsed) return "Not supplied";
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(parsed);
}

function dayDifference(current, previous) {
  if (!current || !previous) return null;
  return Math.round((current.getTime() - previous.getTime()) / 86400000);
}

function signed(value, suffix = "") {
  if (value === null || value === undefined) return "Not comparable";
  if (value === 0) return `0${suffix}`;
  return `${value > 0 ? "+" : ""}${Math.round(value * 10) / 10}${suffix}`;
}

function humanFileSize(bytes) {
  if (!bytes) return "synthetic demo";
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function nonEmptyLines(text) {
  return String(text || "").split(/\r?\n/).map(line => line.trim()).filter(Boolean);
}

function parseEvidenceText(text, kind) {
  const lines = nonEmptyLines(text);
  const header = (lines[0] || "").toLowerCase();
  const body = lines.slice(1);
  const activityCodes = [...new Set((String(text).match(/\b[A-Z]{1,8}-\d{2,6}\b/g) || []))];
  const decisionLines = /status|decision_id|activity_code/.test(header) ? body : lines;
  const decisions = kind === "decision" ? decisionLines.filter(line => /\b(approved|rejected|deferred|authorised|authorized|decision|change)\b/i.test(line)).map(line => {
    const status = /\bapproved|authorised|authorized\b/i.test(line) ? "Approved" : /\brejected\b/i.test(line) ? "Rejected" : /\bdeferred\b/i.test(line) ? "Deferred" : "Recorded";
    const codes = [...new Set(line.match(/\b[A-Z]{1,8}-\d{2,6}\b/g) || [])];
    return { status, codes, text: line };
  }) : [];
  const riskRows = kind === "risk" ? Math.max(0, body.length || (header ? lines.length : 0)) : 0;
  return { lines, header, activityCodes, decisions, riskRows, characterCount: String(text || "").length };
}

function parseXerReview(text, fileName = "schedule.xer", options = {}) {
  if (!text || !text.trim()) throw new Error(`${fileName} is empty.`);
  const tables = parseXerTables(text);

  if (!tables.TASK?.length) throw new Error(`${fileName} does not contain a readable TASK table.`);
  if (!tables.PROJECT?.length) throw new Error(`${fileName} does not contain a readable PROJECT table.`);

  const wbsById = new Map((tables.PROJWBS || []).map(row => [row.wbs_id, row]));
  const projectRow = options.projectId
    ? (tables.PROJECT.find(row => row.proj_id === options.projectId) || tables.PROJECT[0])
    : tables.PROJECT[0];
  const taskRows = tables.TASK.filter(row => !projectRow.proj_id || row.proj_id === projectRow.proj_id);
  const tasks = taskRows.map(row => {
    const duration = numberValue(row.target_drtn_hr_cnt ?? row.orig_drtn_hr_cnt);
    const remaining = numberValue(row.remain_drtn_hr_cnt);
    const start = parseDate(row.early_start_date || row.target_start_date || row.act_start_date);
    const finish = parseDate(row.early_end_date || row.target_end_date || row.act_end_date);
    const taskType = row.task_type || "";
    return {
      id: row.task_id,
      projectId: row.proj_id,
      code: row.task_code || row.task_id,
      name: row.task_name || row.task_code || row.task_id,
      wbsId: row.wbs_id,
      wbsName: wbsById.get(row.wbs_id)?.wbs_name || wbsById.get(row.wbs_id)?.wbs_short_name || "Unassigned WBS",
      calendarId: row.clndr_id || null,
      type: taskType,
      status: row.status_code || "",
      duration,
      remaining,
      float: numberValue(row.total_float_hr_cnt),
      start,
      finish,
      actualStart: parseDate(row.act_start_date),
      actualFinish: parseDate(row.act_end_date),
      constraint: row.constraint_type || row.cstr_type || "",
      constraintDate: parseDate(row.cstr_date || row.constraint_date),
      raw: row,
      isMilestone: /mile/i.test(taskType) || duration === 0,
    };
  });
  const taskById = new Map(tasks.map(task => [task.id, task]));
  const calendars = (tables.CALENDAR || []).map(row => ({
    id: row.clndr_id,
    name: row.clndr_name || row.clndr_id || "Unnamed calendar",
    type: row.clndr_type || "Unknown",
  }));
  const relations = (tables.TASKPRED || []).filter(row => taskById.has(row.task_id) || taskById.has(row.pred_task_id)).map(row => ({
    id: row.task_pred_id || `${row.pred_task_id}->${row.task_id}`,
    taskId: row.task_id,
    predecessorId: row.pred_task_id,
    type: row.pred_type || "PR_FS",
    lag: numberValue(row.lag_hr_cnt) || 0,
    taskCode: taskById.get(row.task_id)?.code || row.task_id,
    predecessorCode: taskById.get(row.pred_task_id)?.code || row.pred_task_id,
    external: !taskById.has(row.task_id) || !taskById.has(row.pred_task_id),
  }));

  return {
    fileName,
    tables,
    tasks,
    taskById,
    relations,
    calendars,
    projectCount: tables.PROJECT.length,
    project: {
      id: projectRow.proj_id || "unknown-project",
      name: projectRow.proj_short_name || projectRow.proj_name || "Unnamed P6 project",
      dataDate: parseDate(projectRow.last_recalc_date || projectRow.data_date),
      plannedStart: parseDate(projectRow.plan_start_date || projectRow.scd_start_date),
      scheduledFinish: parseDate(projectRow.scd_end_date || projectRow.plan_end_date),
    },
    tableCounts: Object.fromEntries(Object.entries(tables).map(([name, rows]) => [name, rows.length])),
    evidenceObjects: {
      riskRecords: ["RISK", "PROJRISK", "RISKTYPE"].reduce((sum, name) => sum + (tables[name]?.length || 0), 0),
      financialPeriodRecords: ["FINPER", "FINPERIOD", "TASKFIN", "TRSRCFIN"].reduce((sum, name) => sum + (tables[name]?.length || 0), 0),
      activityCodeAssignments: ["TASKACTV", "ACTVCODE", "ACTVTYPE"].reduce((sum, name) => sum + (tables[name]?.length || 0), 0),
      resourceAssignments: (tables.TASKRSRC || []).length,
      globalObjectRecords: ["UDFTYPE", "UDFVALUE", "RSRC", "ROLERATE", "CURRTYPE"].reduce((sum, name) => sum + (tables[name]?.length || 0), 0),
    },
  };
}

function relationSignatures(schedule) {
  const signatures = new Map(schedule.tasks.map(task => [task.code, []]));
  schedule.relations.forEach(relation => {
    if (!signatures.has(relation.taskCode)) signatures.set(relation.taskCode, []);
    signatures.get(relation.taskCode).push(`${relation.predecessorCode}|${relation.type}|${relation.lag}`);
  });
  signatures.forEach(values => values.sort());
  return signatures;
}

function relationshipMaps(schedule) {
  const predecessors = new Map(schedule.tasks.map(task => [task.id, []]));
  const successors = new Map(schedule.tasks.map(task => [task.id, []]));
  schedule.relations.filter(relation => !relation.external).forEach(relation => {
    predecessors.get(relation.taskId)?.push(relation);
    successors.get(relation.predecessorId)?.push(relation);
  });
  return { predecessors, successors };
}

function isBusinessCritical(task) {
  return /\b(contract|completion|handover|takeover|acceptance|ready|readiness|operational|benefit|commission|key date|sectional|possession)\b/i.test(`${task?.code || ""} ${task?.name || ""}`);
}

function compareSchedules(previous, current, rules) {
  const previousByCode = new Map(previous.tasks.map(task => [task.code, task]));
  const currentByCode = new Map(current.tasks.map(task => [task.code, task]));
  const previousRelations = relationSignatures(previous);
  const currentRelations = relationSignatures(current);
  const allCodes = [...new Set([...previousByCode.keys(), ...currentByCode.keys()])];
  const changes = [];

  for (const code of allCodes) {
    const before = previousByCode.get(code) || null;
    const after = currentByCode.get(code) || null;
    const types = [];
    const evidence = [];
    let score = 0;
    let finishDelta = null;
    let floatDelta = null;

    if (!before || !after) {
      const added = Boolean(after);
      types.push(added ? "added" : "removed");
      evidence.push({ label: added ? "Current XER" : "Previous XER", value: `${added ? "Added" : "Removed"} activity ${code}` });
      score += 22;
      if ((after || before)?.isMilestone) score += 20;
    } else {
      finishDelta = dayDifference(after.finish, before.finish);
      const startDelta = dayDifference(after.start, before.start);
      const durationDelta = after.duration !== null && before.duration !== null ? after.duration - before.duration : null;
      floatDelta = after.float !== null && before.float !== null ? after.float - before.float : null;
      const constraintChanged = `${before.constraint}|${isoDate(before.constraintDate)}` !== `${after.constraint}|${isoDate(after.constraintDate)}`;
      const beforeLogic = (previousRelations.get(code) || []).join(";");
      const afterLogic = (currentRelations.get(code) || []).join(";");
      const logicChanged = beforeLogic !== afterLogic;

      if (finishDelta !== null && finishDelta !== 0) {
        types.push("dates");
        score += Math.abs(finishDelta) >= rules.materialSlipDays ? Math.min(36, 8 + Math.abs(finishDelta)) : 3;
        if (finishDelta > 0) score += 5;
        evidence.push({ label: "Finish movement", value: `${formatDate(before.finish)} → ${formatDate(after.finish)} (${signed(finishDelta, "d")})` });
      }
      if (startDelta !== null && startDelta !== 0) evidence.push({ label: "Start movement", value: `${formatDate(before.start)} → ${formatDate(after.start)} (${signed(startDelta, "d")})` });
      if (durationDelta !== null && Math.abs(durationDelta) >= 8) {
        types.push("duration");
        score += Math.min(15, 5 + Math.abs(durationDelta) / 16);
        evidence.push({ label: "Original duration", value: `${before.duration}h → ${after.duration}h (${signed(durationDelta, "h")})` });
      }
      if (floatDelta !== null && floatDelta !== 0) {
        types.push("float");
        if (floatDelta < 0) score += Math.min(24, 7 + Math.abs(floatDelta) / 8);
        evidence.push({ label: "Total float", value: `${before.float}h → ${after.float}h (${signed(floatDelta, "h")})` });
      }
      if (constraintChanged) {
        types.push("constraint");
        score += 18;
        evidence.push({ label: "Constraint", value: `${before.constraint || "None"} ${formatDate(before.constraintDate)} → ${after.constraint || "None"} ${formatDate(after.constraintDate)}` });
      }
      if (logicChanged) {
        types.push("logic");
        score += 16;
        evidence.push({ label: "Predecessor logic", value: `${previousRelations.get(code)?.length || 0} → ${currentRelations.get(code)?.length || 0} relationships` });
      }
      if (after.float !== null && after.float < rules.floatThresholdHours) score += 10;
      if (after.isMilestone && types.length) score += 18;
    }

    if (!types.length) continue;
    const task = after || before;
    const currentFloat = after?.float ?? before?.float ?? null;
    const pathClass = currentFloat === null ? "Not supplied" : currentFloat <= 0 ? "Critical or negative float" : currentFloat <= rules.nearCriticalHours ? "Near-critical" : "Outside threshold";
    const businessCritical = isBusinessCritical(task);
    if (pathClass === "Critical or negative float") score += 16;
    else if (pathClass === "Near-critical") score += 8;
    if (businessCritical) score += 16;
    if (pathClass !== "Outside threshold") evidence.push({ label: "Path sensitivity", value: `${pathClass} · exported total float ${currentFloat === null ? "not supplied" : `${currentFloat}h`}` });
    if (businessCritical) evidence.push({ label: "Business milestone signal", value: "Name or code indicates completion, acceptance, readiness or commissioning significance" });
    changes.push({
      id: `${code}-${changes.length}`,
      code,
      name: task.name,
      wbsName: task.wbsName,
      isMilestone: task.isMilestone,
      types: [...new Set(types)],
      finishDelta,
      floatDelta,
      currentFloat,
      pathClass,
      businessCritical,
      score: Math.min(100, Math.round(score)),
      severity: score >= 50 ? "high" : score >= 27 ? "medium" : "low",
      before,
      after,
      evidence,
      summary: evidence[0]?.value || `${types.join(", ")} change`,
    });
  }
  return changes.sort((a, b) => b.score - a.score || a.code.localeCompare(b.code));
}

function assessIntegrity(schedule, rules) {
  const { predecessors, successors } = relationshipMaps(schedule);
  const activeTasks = schedule.tasks.filter(task => !/complete/i.test(task.status));
  const missingPredecessors = activeTasks.filter(task => !task.isMilestone && (predecessors.get(task.id)?.length || 0) === 0);
  const missingSuccessors = activeTasks.filter(task => !task.isMilestone && (successors.get(task.id)?.length || 0) === 0);
  const longActivities = activeTasks.filter(task => task.duration !== null && task.duration > rules.maxDurationHours);
  const negativeFloat = activeTasks.filter(task => task.float !== null && task.float < rules.floatThresholdHours);
  const constrained = activeTasks.filter(task => task.constraint);
  const positiveLag = schedule.relations.filter(relation => relation.lag > 0);
  const external = schedule.relations.filter(relation => relation.external);
  const issue = (type, label, items, detail, severity = "warn") => ({ type, label, count: items.length, examples: items.slice(0, 4).map(item => item.code || item.taskCode || item.id), detail, severity });
  return [
    issue("missing-predecessor", "Activities without predecessors", missingPredecessors, "Active non-milestone activities should normally be connected to the logic network."),
    issue("missing-successor", "Activities without successors", missingSuccessors, "Open-ended work can weaken the reliability of downstream dates."),
    issue("long-duration", "Long activities", longActivities, `Original duration exceeds ${rules.maxDurationHours / 8} eight-hour equivalent days.`),
    issue("negative-float", "Float below threshold", negativeFloat, `Total float is below ${rules.floatThresholdHours} hours.`, "high"),
    issue("constraints", "Dated constraints", constrained, "Constraints require a documented basis because they can influence float and driving paths."),
    issue("positive-lag", "Positive relationship lag", rules.allowPositiveLag ? [] : positiveLag, "Positive lag is disabled by the selected acceptance rules."),
    issue("external-logic", "External or unresolved relationships", external, "One side of the relationship is not present in this XER submission."),
  ].filter(item => item.count > 0);
}

function assessObjectDrift(previous, current) {
  const previousCalendars = new Map(previous.calendars.map(item => [item.id, item.name]));
  const currentCalendars = new Map(current.calendars.map(item => [item.id, item.name]));
  const calendarDefinitions = [...new Set([...previousCalendars.keys(), ...currentCalendars.keys()])].filter(id => previousCalendars.get(id) !== currentCalendars.get(id));
  const previousTasks = new Map(previous.tasks.map(task => [task.code, task]));
  const calendarAssignments = current.tasks.filter(task => previousTasks.has(task.code) && previousTasks.get(task.code).calendarId !== task.calendarId);
  const objectCountChanges = [
    ["Activity-code assignments", previous.evidenceObjects.activityCodeAssignments, current.evidenceObjects.activityCodeAssignments],
    ["Financial-period records", previous.evidenceObjects.financialPeriodRecords, current.evidenceObjects.financialPeriodRecords],
    ["Global-object records", previous.evidenceObjects.globalObjectRecords, current.evidenceObjects.globalObjectRecords],
  ].filter(([, before, after]) => before !== after);
  const findings = [];
  if (calendarDefinitions.length) findings.push({ type: "calendar-definition", label: "Changed calendar definitions", count: calendarDefinitions.length, examples: calendarDefinitions.slice(0, 4), detail: "Calendar IDs or names differ between submissions. Working-time definitions require review in P6.", severity: "high" });
  if (calendarAssignments.length) findings.push({ type: "calendar-assignment", label: "Changed activity calendars", count: calendarAssignments.length, examples: calendarAssignments.slice(0, 4).map(task => task.code), detail: "Activity calendar assignments changed and may affect dates, float and financial-period interpretation.", severity: "high" });
  if (objectCountChanges.length) findings.push({ type: "global-object-drift", label: "Code or global-object count changes", count: objectCountChanges.length, examples: objectCountChanges.map(([label]) => label), detail: objectCountChanges.map(([label, before, after]) => `${label}: ${before} → ${after}`).join("; "), severity: "warn" });
  if (current.projectCount > 1) findings.push({ type: "multiple-projects", label: "Multiple projects in current XER", count: current.projectCount, examples: [], detail: "The XER contains multiple projects. This browser review currently uses the first PROJECT record while preserving external relationships.", severity: "high" });
  return findings;
}

function assessCompleteness(previous, current, evidence, narrative, baseline) {
  const embeddedRiskCount = current.evidenceObjects.riskRecords;
  const suppliedRiskCount = evidence.risk?.riskRows || 0;
  const entries = [
    { key: "activities", label: "Activities", present: current.tasks.length > 0, status: current.tasks.length.toLocaleString(), detail: "Readable current TASK records" },
    { key: "relationships", label: "Relationships", present: current.relations.length > 0, status: current.relations.length.toLocaleString(), detail: "Readable current TASKPRED records" },
    { key: "calendars", label: "Calendars", present: current.calendars.length > 0, status: current.calendars.length ? current.calendars.length.toLocaleString() : "Missing", detail: current.calendars.length ? `${new Set(current.tasks.map(task => task.calendarId).filter(Boolean)).size} calendars assigned to activities` : "Calendar assumptions cannot be checked" },
    { key: "baseline", label: "Separate baseline", present: Boolean(baseline), status: baseline ? "Supplied" : "Missing", detail: baseline ? `${baseline.tasks.length.toLocaleString()} baseline activities available` : "XER does not reliably export baselines; current dates cannot be tested against commitment" },
    { key: "risk", label: "Risk evidence", present: embeddedRiskCount + suppliedRiskCount > 0, status: embeddedRiskCount + suppliedRiskCount > 0 ? `${embeddedRiskCount + suppliedRiskCount} records` : "Missing", detail: embeddedRiskCount ? "Risk-like records found in the XER" : suppliedRiskCount ? "Separate risk register supplied" : "Schedule movement cannot be reconciled to risk ownership" },
    { key: "basis", label: "Schedule basis", present: Boolean(evidence.basis?.characterCount), status: evidence.basis?.characterCount ? "Supplied" : "Missing", detail: evidence.basis?.characterCount ? `${evidence.basis.characterCount.toLocaleString()} characters of assumptions and basis` : "Calendar, productivity and sequencing assumptions are not evidenced" },
    { key: "narrative", label: "Update narrative", present: Boolean(narrative.trim()), status: narrative.trim() ? "Supplied" : "Missing", detail: narrative.trim() ? `${narrative.trim().length.toLocaleString()} characters available for claim testing` : "Contradiction testing is limited" },
    { key: "decisions", label: "Decision log", present: Boolean(evidence.decision?.decisions.length), status: evidence.decision?.decisions.length ? `${evidence.decision.decisions.length} decisions` : "Missing", detail: evidence.decision?.decisions.length ? `${evidence.decision.activityCodes.length} referenced activity codes` : "Material changes cannot be linked to an approval record" },
    { key: "float-contingency", label: "Float and contingency basis", present: Boolean(evidence.basis?.lines.join(" ").match(/\bfloat\b/i) && evidence.basis?.lines.join(" ").match(/\bcontingency\b/i)), status: evidence.basis?.lines.join(" ").match(/\bfloat\b/i) && evidence.basis?.lines.join(" ").match(/\bcontingency\b/i) ? "Addressed" : "Unclear", detail: "The schedule basis should state float ownership and where schedule contingency is held" },
    { key: "financial-periods", label: "Financial periods", present: current.evidenceObjects.financialPeriodRecords > 0, status: current.evidenceObjects.financialPeriodRecords ? `${current.evidenceObjects.financialPeriodRecords} records` : "Not present", detail: current.evidenceObjects.financialPeriodRecords ? "Financial-period records are present but still require calendar compatibility" : "Period performance cannot be assured from this submission" },
    { key: "codes", label: "Codes and global objects", present: current.evidenceObjects.activityCodeAssignments + current.evidenceObjects.globalObjectRecords > 0, status: current.evidenceObjects.activityCodeAssignments + current.evidenceObjects.globalObjectRecords > 0 ? "Present" : "Not present", detail: `${current.evidenceObjects.activityCodeAssignments} code assignments and ${current.evidenceObjects.globalObjectRecords} global-object records` },
  ];
  const presentCount = entries.filter(item => item.present).length;
  const coverageLabel = presentCount >= 9 ? "Strong evidence set" : presentCount >= 6 ? "Bounded evidence set" : "Limited evidence set";
  const missing = entries.filter(item => !item.present).map(item => item.label.toLowerCase());
  return {
    entries,
    presentCount,
    totalCount: entries.length,
    coverageLabel,
    missing,
    boundary: missing.length ? `Conclusions are bounded because ${missing.join(", ")} ${missing.length === 1 ? "is" : "are"} missing or not present.` : "All defined evidence classes are present. Their quality and contractual status still require human review.",
  };
}

function assessFitness(previous, current, narrative, completeness) {
  const tableCheck = (name, label, required = true) => {
    const previousCount = previous.tableCounts[name] || 0;
    const currentCount = current.tableCounts[name] || 0;
    const good = previousCount > 0 && currentCount > 0;
    return { label, status: good ? "Present" : required ? "Missing" : "Not supplied", tone: good ? "good" : required ? "warn" : "info", detail: `${previousCount.toLocaleString()} previous / ${currentCount.toLocaleString()} current records` };
  };
  const sequenceGood = previous.project.dataDate && current.project.dataDate && current.project.dataDate > previous.project.dataDate;
  return [
    tableCheck("TASK", "Activity table"),
    tableCheck("TASKPRED", "Relationship table"),
    tableCheck("PROJWBS", "Work breakdown structure"),
    tableCheck("CALENDAR", "Calendars", false),
    { label: "Submission sequence", status: sequenceGood ? "Valid" : "Check dates", tone: sequenceGood ? "good" : "warn", detail: `${formatDate(previous.project.dataDate)} → ${formatDate(current.project.dataDate)}` },
    { label: "Evidence coverage", status: `${completeness.presentCount}/${completeness.totalCount}`, tone: completeness.presentCount >= 9 ? "good" : "warn", detail: completeness.coverageLabel },
    { label: "Narrative", status: narrative.trim() ? "Supplied" : "Missing", tone: narrative.trim() ? "good" : "warn", detail: narrative.trim() ? `${narrative.trim().length.toLocaleString()} characters available for claim testing` : "Contradiction testing is limited without the update narrative." },
  ];
}

function pathCandidates(schedule, rules) {
  const { predecessors, successors } = relationshipMaps(schedule);
  return schedule.tasks
    .filter(task => !/complete/i.test(task.status) && task.float !== null && task.float <= rules.nearCriticalHours)
    .map(task => ({
      code: task.code,
      name: task.name,
      float: task.float,
      classification: task.float <= 0 ? "Critical or negative float" : "Near-critical",
      predecessors: predecessors.get(task.id)?.length || 0,
      successors: successors.get(task.id)?.length || 0,
      finish: isoDate(task.finish),
      isMilestone: task.isMilestone,
      businessCritical: isBusinessCritical(task),
    }))
    .sort((a, b) => a.float - b.float || Number(b.isMilestone) - Number(a.isMilestone) || a.code.localeCompare(b.code));
}

function reconcileDecisions(materialChanges, decisionEvidence) {
  const decisions = decisionEvidence?.decisions || [];
  return materialChanges.map(change => {
    const matches = decisions.filter(decision => decision.codes.includes(change.code));
    return {
      changeId: change.id,
      code: change.code,
      name: change.name,
      linked: matches.length > 0,
      decisions: matches,
      status: matches.length ? matches.map(item => item.status).join(" / ") : "No linked decision",
    };
  });
}

function compareBaseline(baseline, current) {
  if (!baseline) return [];
  const baselineByCode = new Map(baseline.tasks.map(task => [task.code, task]));
  return current.tasks
    .filter(task => task.isMilestone && task.finish && baselineByCode.get(task.code)?.finish)
    .map(task => {
      const committed = baselineByCode.get(task.code);
      return { code: task.code, name: task.name, baseline: isoDate(committed.finish), current: isoDate(task.finish), movementDays: dayDifference(task.finish, committed.finish) };
    })
    .sort((a, b) => Math.abs(b.movementDays) - Math.abs(a.movementDays));
}

function selectPriorityChanges(materialChanges, limit) {
  const selected = [];
  const perWbs = new Map();
  const wbsNames = new Set(materialChanges.map(change => change.wbsName));
  // Flat WBS schedules used to shrink the executive queue to 2 items; skip that cap then.
  const applyWbsCap = wbsNames.size > 1;
  const mustSee = materialChanges.filter(change => change.isMilestone || change.businessCritical);
  for (const change of [...mustSee, ...materialChanges]) {
    if (selected.some(item => item.id === change.id)) continue;
    const count = perWbs.get(change.wbsName) || 0;
    if (applyWbsCap && count >= 2 && !change.isMilestone && !change.businessCritical) continue;
    selected.push(change);
    perWbs.set(change.wbsName, count + 1);
    if (selected.length >= limit) break;
  }
  return selected;
}

function projectFinishMovement(previous, current) {
  let before = previous.project.scheduledFinish;
  let after = current.project.scheduledFinish;
  let label = "Project scheduled finish";
  if (!before || !after) {
    const currentMilestones = current.tasks.filter(task => task.isMilestone && task.finish).sort((a, b) => b.finish - a.finish);
    const milestone = currentMilestones.find(item => previous.tasks.some(task => task.code === item.code && task.finish));
    if (milestone) {
      const previousMilestone = previous.tasks.find(task => task.code === milestone.code);
      before = previousMilestone.finish;
      after = milestone.finish;
      label = milestone.name;
    }
  }
  return { days: dayDifference(after, before), before, after, label };
}

function narrativeAssessment(narrative, changes, finishMovement) {
  const text = narrative.toLowerCase();
  const conflicts = [];
  const relationshipChanges = changes.filter(change => change.types.includes("logic"));
  const constraintChanges = changes.filter(change => change.types.includes("constraint"));
  const floatErosion = changes.filter(change => change.evidence.some(item => item.label === "Total float" && /\(-/.test(item.value)));
  const claims = detectScheduleClaims(narrative, {
    currentFinish: finishMovement.after,
    previousFinish: finishMovement.before,
  });
  const statedStable = claims.stable.length > 0 && !claims.movesStated.some(move => move.to === toIsoDay(finishMovement.after));
  const finishSlip = finishMovement.days || 0;

  if (!narrative.trim()) {
    conflicts.push({ type: "Evidence gap", title: "No schedule narrative was supplied", detail: "The XER can show what changed, but not the planner's stated reason or approval basis." });
    return conflicts;
  }
  if (claims.contradiction && Math.abs(finishSlip) > 0) {
    conflicts.push({
      type: "Direct contradiction",
      title: `The narrative presents stability while the scheduled finish moved ${finishSlip} days`,
      detail: `${finishMovement.label}: ${formatDate(finishMovement.before)} → ${formatDate(finishMovement.after)}. ${claims.contradiction.detail}`,
    });
  }
  if (statedStable && floatErosion.some(change => change.score >= 30)) conflicts.push({ type: "Hidden pressure", title: "The narrative presents stability while material float was consumed", detail: `${floatErosion.length} changed activities show float erosion that may reduce schedule resilience.` });
  if (relationshipChanges.length && !/\b(logic|relationship|predecessor|successor|resequence|sequence)\b/.test(text)) conflicts.push({ type: "Unexplained change", title: `${relationshipChanges.length} logic changes are not addressed in the narrative`, detail: "Changed predecessor logic can alter the driving path even when headline dates appear stable." });
  if (constraintChanges.length && !/\bconstraint|mandatory|must finish|must start|fixed date\b/.test(text)) conflicts.push({ type: "Unexplained change", title: `${constraintChanges.length} constraint changes are not addressed in the narrative`, detail: "A dated constraint can influence float and the apparent critical path." });
  if (finishSlip >= 7 && !claims.contradiction && !claims.movesStated.length && !/\b(delay|slip|later|movement|forecast|completion|finish)\b/.test(text)) {
    conflicts.push({ type: "Narrative omission", title: "Material finish movement is not described", detail: `${finishMovement.label} moved ${finishSlip} days later between submissions.` });
  }
  return conflicts;
}

function buildQuestions(changes, conflicts, integrity, completeness, decisionReconciliation, riskEvidence) {
  const questions = [];
  changes.filter(change => change.severity === "high").slice(0, 4).forEach(change => {
    questions.push({ type: "Material change", question: `What approved change or delivery event explains the movement in ${change.code} ${change.name}?`, evidence: change.summary, changeId: change.id });
  });
  conflicts.slice(0, 3).forEach(conflict => questions.push({ type: conflict.type, question: `How should the submission narrative be corrected or evidenced to address: ${conflict.title.toLowerCase()}?`, evidence: conflict.detail }));
  completeness.entries.filter(item => !item.present).slice(0, 3).forEach(item => questions.push({ type: "Evidence gap", question: `What evidence or limitation statement is required because ${item.label.toLowerCase()} is not present?`, evidence: item.detail }));
  decisionReconciliation.filter(item => !item.linked).slice(0, 3).forEach(item => questions.push({ type: "Approval trace", question: `Which approved decision authorises the material change to ${item.code} ${item.name}?`, evidence: "No matching activity code was found in the supplied decision log.", changeId: item.changeId }));
  if (riskEvidence) changes.filter(change => change.severity === "high" && !riskEvidence.activityCodes.includes(change.code)).slice(0, 2).forEach(change => questions.push({ type: "Risk trace", question: `Which owned risk explains the high-priority movement in ${change.code} ${change.name}?`, evidence: "No matching activity code was found in the supplied risk evidence.", changeId: change.id }));
  integrity.filter(item => item.severity === "high" || item.count >= 3).slice(0, 3).forEach(item => questions.push({ type: "Schedule integrity", question: `What is the acceptance or remediation decision for ${item.count} ${item.label.toLowerCase()}?`, evidence: `${item.detail} Examples: ${item.examples.join(", ") || "none"}.` }));
  return questions.slice(0, 10);
}

function buildReview(previous, current, narrative, rules, evidence = {}, baseline = null) {
  const changes = compareSchedules(previous, current, rules);
  const integrity = [...assessIntegrity(current, rules), ...assessObjectDrift(previous, current)];
  const completeness = assessCompleteness(previous, current, evidence, narrative, baseline);
  const fitness = assessFitness(previous, current, narrative, completeness);
  const finishMovement = projectFinishMovement(previous, current);
  const conflicts = narrativeAssessment(narrative, changes, finishMovement);
  const materialChanges = changes.filter(change => change.score >= 27 || change.isMilestone || change.businessCritical || change.types.includes("constraint") || Math.abs(change.finishDelta || 0) >= rules.materialSlipDays);
  materialChanges.forEach(change => { change.riskLinked = Boolean(evidence.risk?.activityCodes.includes(change.code)); });
  const priorityChanges = selectPriorityChanges(materialChanges, rules.executiveChangeLimit);
  const paths = pathCandidates(current, rules);
  const decisionReconciliation = reconcileDecisions(materialChanges, evidence.decision);
  const baselineMovements = compareBaseline(baseline, current);
  const questions = buildQuestions(priorityChanges, conflicts, integrity, completeness, decisionReconciliation, evidence.risk);
  return {
    generatedAt: new Date().toISOString(),
    previous,
    current,
    narrative,
    rules,
    evidence,
    baseline,
    changes,
    materialChanges,
    priorityChanges,
    integrity,
    fitness,
    completeness,
    paths,
    decisionReconciliation,
    baselineMovements,
    finishMovement,
    conflicts,
    questions,
  };
}


function parseXerAssurance(text, fileName, options = {}) {
  if (!text?.trim()) throw new Error(`${fileName} is empty.`);
  const tables = parseXerTables(text);

  if (!tables.PROJECT?.length || !tables.TASK?.length) throw new Error(`${fileName} is not a readable Primavera P6 schedule file (XER).`);
  if (tables.PROJECT.length !== 1 && !options.projectId) {
    throw new Error(`${fileName} contains multiple projects. Export a single project for change assurance, or use the detailed schedule review to inspect first-project scope.`);
  }

  const project = options.projectId
    ? (tables.PROJECT.find(row => row.proj_id === options.projectId) || null)
    : tables.PROJECT[0];
  if (!project) throw new Error(`${fileName} has no project matching the selected project ID.`);
  if (!project.proj_id?.trim()) throw new Error(`${fileName} has no project ID. Export a single-project schedule file (XER) and retry.`);
  const taskRows = tables.TASK.filter(row => !project.proj_id || row.proj_id === project.proj_id);
  const tasks = taskRows.map(row => ({
    code: row.task_code || row.task_id,
    name: row.task_name || row.task_code || row.task_id,
    finish: parseDate(row.early_end_date || row.target_end_date || row.act_end_date),
    float: numberValue(row.total_float_hr_cnt),
    constraint: row.constraint_type || row.cstr_type || "",
    constraintDate: parseDate(row.cstr_date || row.constraint_date),
    status: row.status_code || "",
    type: row.task_type || "",
  }));
  if (tasks.some(task => !task.code) || new Set(tasks.map(task => task.code)).size !== tasks.length) throw new Error(`${fileName} needs unique, non-empty activity codes for comparison.`);
  if (!parseDate(project.scd_end_date || project.plan_end_date)) throw new Error(`${fileName} has no supported project scheduled finish. Export a calculated single-project schedule and retry.`);
  const codeById = new Map(taskRows.map(row => [row.task_id, row.task_code || row.task_id]));
  const relations = (tables.TASKPRED || []).map(row => JSON.stringify([codeById.get(row.pred_task_id) || row.pred_task_id, codeById.get(row.task_id) || row.task_id, row.pred_type || "Not supplied", numberValue(row.lag_hr_cnt)]));

  return {
    fileName,
    sourceText: text.replace(/\r\n?/g, "\n").trim(),
    project: {
      id: project.proj_id || "unassigned",
      name: project.proj_short_name || project.proj_name || "Unnamed project",
      dataDate: parseDate(project.last_recalc_date || project.data_date),
      scheduledFinish: parseDate(project.scd_end_date || project.plan_end_date),
    },
    tasks,
    taskByCode: new Map(tasks.map(task => [task.code, task])),
    relations: new Set(relations),
  };
}

function taskChanges(previous, current) {
  const codes = new Set([...previous.taskByCode.keys(), ...current.taskByCode.keys()]);
  const changes = [];
  const activityEvidence = [];
  const snapshot = task => task ? { finish: formatDate(task.finish), totalFloatHours: task.float, constraintType: task.constraint || "None", constraintDate: formatDate(task.constraintDate) } : null;
  let constraintChanges = 0;
  let floatErosion = 0;

  codes.forEach(code => {
    const before = previous.taskByCode.get(code);
    const after = current.taskByCode.get(code);
    if (!before || !after) {
      activityEvidence.push({ code, name: (after || before).name, before: snapshot(before), after: snapshot(after) });
      changes.push({ code, name: (after || before).name, summary: after ? "Activity added" : "Activity removed" });
      return;
    }
    const finishDelta = dayDifference(after.finish, before.finish);
    const constraintChanged = `${before.constraint}|${formatDate(before.constraintDate)}` !== `${after.constraint}|${formatDate(after.constraintDate)}`;
    const floatDelta = before.float !== null && after.float !== null ? after.float - before.float : null;
    if (finishDelta || constraintChanged || (floatDelta !== null && floatDelta !== 0)) activityEvidence.push({ code, name: after.name, before: snapshot(before), after: snapshot(after) });
    if (constraintChanged) constraintChanges += 1;
    if (floatDelta !== null && floatDelta < 0) floatErosion += 1;
    if ((finishDelta && Math.abs(finishDelta) >= 7) || constraintChanged || (floatDelta !== null && floatDelta <= -8)) {
      const evidence = [];
      if (finishDelta) evidence.push(`finish ${finishDelta > 0 ? "+" : ""}${finishDelta}d`);
      if (constraintChanged) evidence.push("constraint changed");
      if (floatDelta !== null && floatDelta < 0) evidence.push(`float ${floatDelta}h`);
      changes.push({ code, name: after.name, summary: evidence.join(" · ") });
    }
  });

  const relationEvidence = [...new Set([...previous.relations, ...current.relations])].filter(signature => previous.relations.has(signature) !== current.relations.has(signature)).map(signature => {
    const [predecessor, successor, type, lagHours] = JSON.parse(signature);
    return { change: current.relations.has(signature) ? "Added" : "Removed", predecessor, successor, type, lagHours };
  });
  const relationChanges = relationEvidence.length;
  return { changes, activityEvidence, relationEvidence, constraintChanges, floatErosion, relationChanges };
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function toIsoDay(value) {
  if (!value) return null;
  if (value instanceof Date) return isoDate(value);
  const clean = String(value).trim();
  if (/^\d{4}-\d{2}-\d{2}/.test(clean)) return clean.slice(0, 10);
  const parsed = parseDate(clean);
  return parsed ? isoDate(parsed) : null;
}

function dayDeltaIso(a, b) {
  const left = toIsoDay(a);
  const right = toIsoDay(b);
  if (!left || !right) return null;
  return dayDifference(parseDate(left), parseDate(right));
}

function expandTwoDigitYear(year) {
  const n = Number(year);
  if (!Number.isFinite(n)) return null;
  return n < 100 ? 2000 + n : n;
}

function lastDayOfMonth(year, month) {
  // month is 1-12; Date.UTC day 0 of next month is the last day of this one
  return new Date(Date.UTC(year, month, 0, 12, 0, 0));
}

function financialYearQuarterEnd(startYearTwoDigitOrFour, quarter) {
  // UK financial year: Apr (Y) → Mar (Y+1). Q4 ends 31 March of the second calendar year.
  const start = expandTwoDigitYear(startYearTwoDigitOrFour);
  if (!start || quarter < 1 || quarter > 4) return null;
  const endYear = start + 1;
  if (quarter === 1) return new Date(Date.UTC(start, 5, 30, 12)); // 30 Jun
  if (quarter === 2) return new Date(Date.UTC(start, 8, 30, 12)); // 30 Sep
  if (quarter === 3) return new Date(Date.UTC(start, 11, 31, 12)); // 31 Dec
  return new Date(Date.UTC(endYear, 2, 31, 12)); // 31 Mar
}

const MONTH_INDEX = {
  jan: 1, january: 1, feb: 2, february: 2, mar: 3, march: 3, apr: 4, april: 4,
  may: 5, jun: 6, june: 6, jul: 7, july: 7, aug: 8, august: 8, sep: 9, sept: 9, september: 9,
  oct: 10, october: 10, nov: 11, november: 11, dec: 12, december: 12,
};

function normaliseNarrativeText(text) {
  // Paste from Word / PDFs often leaves NBSP and double spaces inside dates.
  return String(text || "").replace(/\u00A0/g, " ").replace(/\s+/g, " ");
}

function extractNarrativeDates(text) {
  const dates = [];
  const push = (raw, date, index) => {
    if (!date || Number.isNaN(date.getTime())) return;
    dates.push({ iso: isoDate(date), raw, index });
  };

  // ISO first so later patterns do not re-hit the same span awkwardly.
  for (const match of text.matchAll(/\b(\d{4}-\d{2}-\d{2})\b/g)) {
    push(match[1], parseDate(match[1]), match.index);
  }

  // 31/12/2027 or 31/12/27
  for (const match of text.matchAll(/\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})\b/g)) {
    const day = Number(match[1]);
    const month = Number(match[2]);
    const year = expandTwoDigitYear(match[3]);
    push(match[0], new Date(Date.UTC(year, month - 1, day, 12)), match.index);
  }

  // 31 December 2027 / 31 Dec 27 / 30 June 26
  for (const match of text.matchAll(/\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{2,4})\b/gi)) {
    const day = Number(match[1]);
    const month = MONTH_INDEX[match[2].toLowerCase()];
    const year = expandTwoDigitYear(match[3]);
    push(match[0], new Date(Date.UTC(year, month - 1, day, 12)), match.index);
  }

  // December 2027 (month + year only → last day of that month)
  for (const match of text.matchAll(/\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b/gi)) {
    const month = MONTH_INDEX[match[1].toLowerCase()];
    const year = Number(match[2]);
    push(match[0], lastDayOfMonth(year, month), match.index);
  }

  // Q4 2027-28 (financial-year quarter). Mark as fyQuarter so mismatch logic can ignore them.
  for (const match of text.matchAll(/\bQ([1-4])\s+(\d{4})-(\d{2})\b/gi)) {
    const quarter = Number(match[1]);
    const startYear = Number(match[2]);
    const date = financialYearQuarterEnd(startYear, quarter);
    if (date) dates.push({ iso: isoDate(date), raw: match[0], index: match.index, fyQuarter: true });
  }

  // 25/26-Q4
  for (const match of text.matchAll(/\b(\d{2})\/(\d{2})-Q([1-4])\b/gi)) {
    const startYear = expandTwoDigitYear(match[1]);
    const quarter = Number(match[3]);
    const date = financialYearQuarterEnd(startYear, quarter);
    if (date) dates.push({ iso: isoDate(date), raw: match[0], index: match.index, fyQuarter: true });
  }

  // 23-Jan-2031 (P6 / GMPP style)
  for (const match of text.matchAll(/\b(\d{1,2})-(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)-(\d{4})\b/gi)) {
    const day = Number(match[1]);
    const month = MONTH_INDEX[match[2].toLowerCase()];
    const year = Number(match[3]);
    push(match[0], new Date(Date.UTC(year, month - 1, day, 12)), match.index);
  }

  // De-duplicate identical iso+index hits from overlapping patterns.
  const seen = new Set();
  return dates.filter(item => {
    const key = `${item.iso}@${item.index}@${item.raw}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function findPhraseHits(text, phrases) {
  const hits = [];
  const lower = text.toLowerCase();
  for (const phrase of phrases) {
    const pattern = new RegExp(`\\b${escapeRegExp(phrase.toLowerCase()).replace(/\s+/g, "\\s+")}\\b`, "gi");
    for (const match of lower.matchAll(pattern)) {
      hits.push({ phrase, index: match.index });
    }
  }
  return hits.sort((a, b) => a.index - b.index);
}

function isNegatedAt(text, index, negations) {
  // Look a short way left of the match for "not / never / no longer".
  const start = Math.max(0, index - 48);
  const window = text.slice(start, index).toLowerCase();
  return negations.some(negation => new RegExp(`\\b${escapeRegExp(negation.toLowerCase())}\\s+(?:\\w+\\s+){0,3}$`).test(window));
}

function extractMoveStatements(text, movePhrases, dates) {
  const moves = [];
  const lower = text.toLowerCase();
  for (const phrase of movePhrases) {
    const pattern = new RegExp(`\\b${escapeRegExp(phrase.toLowerCase()).replace(/\s+/g, "\\s+")}\\b`, "gi");
    for (const match of lower.matchAll(pattern)) {
      const index = match.index;
      // Take dates that sit near this move phrase (same clause / next ~120 chars).
      const nearby = dates.filter(d => d.index >= index - 10 && d.index <= index + 140);
      let from = null;
      let to = null;
      if (nearby.length >= 2) {
        from = nearby[0].iso;
        to = nearby[1].iso;
      } else if (nearby.length === 1) {
        to = nearby[0].iso;
      }
      moves.push({ from, to, index });
    }
  }
  return moves;
}

function moveCoversCurrent(moves, currentFinish, dates = []) {
  const current = toIsoDay(currentFinish);
  if (!current) return false;
  if (moves.some(move => move.to === current)) return true;
  // Some GMPP lines say "increased to <date>" with only one nearby date.
  if (moves.length && dates.some(d => !d.fyQuarter && d.iso === current)) return true;
  return false;
}

function detectScheduleClaims(text, { currentFinish = null, previousFinish = null } = {}) {
  const rules = loadClaimRules();
  // Normalise before phrase / date matching so NBSP and double spaces do not hide hits.
  const source = normaliseNarrativeText(text);
  const dates = extractNarrativeDates(source);
  const stableHits = findPhraseHits(source, rules.stablePhrases)
    .filter(hit => !isNegatedAt(source, hit.index, rules.negations));
  const movesStated = extractMoveStatements(source, rules.movePhrases, dates);
  const finishMoved = (() => {
    const delta = dayDeltaIso(currentFinish, previousFinish);
    return delta !== null && delta !== 0;
  })();
  const covered = moveCoversCurrent(movesStated, currentFinish, dates);

  let contradiction = null;

  // Date mismatch: stability wording quotes a finish that disagrees with currentFinish
  // by more than 14 days. Skip FY-quarter labels. A stated move suppresses this.
  if (!covered && currentFinish && stableHits.length && movesStated.length === 0) {
    const current = toIsoDay(currentFinish);
    const mismatched = dates.find(d => {
      if (d.fyQuarter) return false;
      const delta = dayDeltaIso(d.iso, current);
      return delta !== null && Math.abs(delta) > 14;
    });
    if (mismatched) {
      contradiction = {
        kind: "date-mismatch",
        detail: `Narrative names ${mismatched.raw} (${mismatched.iso}) but the current finish is ${current}.`,
      };
    }
  }

  // Otherwise: stability wording while the finish moved, and no move is stated at all.
  if (!contradiction && finishMoved && stableHits.length && !covered && movesStated.length === 0) {
    contradiction = {
      kind: "says-unchanged",
      detail: `Narrative claims stability (${stableHits[0].phrase}) while the finish moved from ${toIsoDay(previousFinish)} to ${toIsoDay(currentFinish)}.`,
    };
  }

  // A stated move that lands on the current finish suppresses any contradiction.
  if (covered) contradiction = null;

  return {
    stable: stableHits,
    movesStated,
    dates,
    contradiction,
  };
}

function taskCodeSet(parsed) {
  // Works for both assurance and review parse shapes.
  if (parsed.taskByCode instanceof Map) return new Set(parsed.taskByCode.keys());
  return new Set((parsed.tasks || []).map(task => task.code).filter(Boolean));
}

function matchProjects(previousParsed, currentParsed) {
  const previousCodes = taskCodeSet(previousParsed);
  const currentCodes = taskCodeSet(currentParsed);
  let shared = 0;
  previousCodes.forEach(code => {
    if (currentCodes.has(code)) shared += 1;
  });
  const union = previousCodes.size + currentCodes.size - shared;
  const score = union === 0 ? 0 : shared / union;
  let decision = "different";
  if (score >= 0.8) decision = "same";
  else if (score >= 0.5) decision = "confirm";
  return { score, shared, decision };
}

function analyseReview(previous, current, narrative, options = {}) {
  const identityMismatch = previous.project.name !== current.project.name || previous.project.id !== current.project.id;
  if (identityMismatch) {
    const match = matchProjects(previous, current);
    if (match.decision === "different") {
      // Keep the familiar wording when codes do not overlap, so reviewers still see why.
      if (previous.project.name !== current.project.name) {
        throw new Error("The project names differ. Choose previous and current exports of the same project before comparing.");
      }
      throw new Error("The project IDs differ. Choose exports of the same source project; re-imported projects with new IDs are not supported by this comparison.");
    }
    if (!options.confirmedMatch) {
      const pct = Math.round(match.score * 100);
      const err = new Error(`These look like the same project (${pct}% of activity codes match). Compare them?`);
      err.code = "CONFIRM_PROJECT_MATCH";
      err.match = match;
      throw err;
    }
  }
  if (previous.sourceText === current.sourceText) throw new Error("These exports contain identical evidence. Choose a previous and a current schedule, then retry.");
  if (!previous.project.dataDate || !current.project.dataDate) throw new Error("Both exports need a supported project data date before comparison.");
  if (current.project.dataDate < previous.project.dataDate) throw new Error("The current data date precedes the previous export. Swap the files and retry.");
  const finishMovement = dayDifference(current.project.scheduledFinish, previous.project.scheduledFinish);
  const changes = taskChanges(previous, current);
  const claims = detectScheduleClaims(narrative, {
    currentFinish: current.project.scheduledFinish,
    previousFinish: previous.project.scheduledFinish,
  });
  const blockers = [];
  // Blank float on completed / WBS / LOE / finish-milestone rows is normal in P6 exports.
  const floatComparable = task => {
    if (/TK_Complete/i.test(task.status || "")) return false;
    if (/TT_WBS|TT_LOE|TT_FinMile/i.test(task.type || "")) return false;
    return true;
  };
  const coverage = [previous, current].map(schedule => {
    const comparable = schedule.tasks.filter(floatComparable);
    return {
      file: schedule.fileName,
      tasks: schedule.tasks.length,
      finishDates: schedule.tasks.filter(task => task.finish !== null).length,
      floatValues: comparable.filter(task => task.float !== null).length,
      floatExpected: comparable.length,
    };
  });
  if (coverage.some(item => item.finishDates !== item.tasks || item.floatValues !== item.floatExpected)) {
    blockers.push({
      type: "Evidence gap",
      title: "Some activity finish dates or float values could not be compared",
      detail: coverage.map(item => `${item.file}: ${item.finishDates}/${item.tasks} finishes and ${item.floatValues}/${item.floatExpected} comparable float values`).join("; "),
    });
  }

  if (!narrative.trim()) {
    blockers.push({
      type: "Evidence gap",
      title: "No change narrative was supplied",
      detail: "The schedules can show movement, but not the stated reason, mitigation or approval basis.",
    });
  }
  if (claims.contradiction && finishMovement !== null && Math.abs(finishMovement) > 0) {
    blockers.push({
      type: "Contradiction",
      title: `The finish moved ${finishMovement} days and the update needs clarifying`,
      detail: `${formatDate(previous.project.scheduledFinish)} → ${formatDate(current.project.scheduledFinish)}. ${claims.contradiction.detail}`,
    });
  } else if (finishMovement !== null && Math.abs(finishMovement) >= 7) {
    blockers.push({ type: "Schedule movement", title: `The scheduled finish moved ${finishMovement > 0 ? "+" : ""}${finishMovement} days`, detail: "Check the reason and supporting evidence. Mentioning the movement does not verify its cause." });
  }
  if (changes.constraintChanges) {
    blockers.push({
      type: "Unexplained change",
      title: `${changes.constraintChanges} constraint changes need review`,
      detail: "Constraint type or date changed. Inspect the source fields and the instruction supporting each change; narrative wording does not close this review question.",
    });
  }
  if (changes.relationChanges) {
    blockers.push({
      type: "Unexplained change",
      title: `${changes.relationChanges} links between tasks changed`,
      detail: "Added or removed links between tasks (which task waits for which) can change the order of work. Ask which instruction explains the differences; these exports do not establish causation.",
    });
  }
  if (changes.floatErosion) {
    blockers.push({
      type: "Float loss",
      title: `Less spare time before the finish moves (float) on ${changes.floatErosion} changed activities`,
      detail: "The current schedule shows less total float: these activities can slip less before they delay the finish. Check the underlying fields and planning assumptions; this does not prove a delay or a critical path.",
    });
  }
  const text = String(narrative || "").toLowerCase();
  const explicitMissingOwnership = /\b(?:no|without|missing|unassigned)\s+(?:named\s+)?(?:owner|due date)\b|\b(?:owner|due date)\s+(?:is\s+)?(?:missing|unassigned|unknown|not set)\b/.test(text);
  if (explicitMissingOwnership || (/\b(manage\w*|mitigat\w*|governance|action)\b/.test(text) && !/\b(owner|responsible|accountable|due|by \d{1,2})\b/.test(text))) {
    blockers.push({
      type: "Ownership gap",
      title: "The stated response has no named owner or due date",
      detail: "Name an owner and a date so someone can follow the mitigation up.",
    });
  }

  return {
    previous,
    current,
    finishMovement,
    changes,
    blockers,
    coverage,
    generatedAt: new Date().toISOString(),
  };
}


  return {
    parseXer: parseXerReview,
    parseXerReview,
    parseXerAssurance,
    taskChanges,
    analyseReview,
    compareSchedules,
    buildReview,
    parseEvidenceText,
    detectScheduleClaims,
    matchProjects,
    decodeXerBytes,
    listProjects,
    loadClaimRules,
    assessCompleteness,
    pathCandidates,
    reconcileDecisions,
    compareBaseline,
    numberValue,
    parseDate,
    formatDate,
    dayDifference,
    isoDate,
  };
}));
