const reviewState = {
  files: { previous: null, current: null, baseline: null, risk: null, basis: null, decision: null },
  narrative: "",
  analysis: null,
  changeFilter: "all",
  selectedChangeId: null,
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 3600);
}

function numberValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(String(value).replace(/,/g, ""));
  return Number.isFinite(number) ? number : null;
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

function parseXer(text, fileName = "schedule.xer") {
  if (!text || !text.trim()) throw new Error(`${fileName} is empty.`);
  const tables = {};
  let tableName = null;
  let fields = [];
  const lines = text.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n").split("\n");

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

  if (!tables.TASK?.length) throw new Error(`${fileName} does not contain a readable TASK table.`);
  if (!tables.PROJECT?.length) throw new Error(`${fileName} does not contain a readable PROJECT table.`);

  const wbsById = new Map((tables.PROJWBS || []).map(row => [row.wbs_id, row]));
  const projectRow = tables.PROJECT[0];
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

function selectedRules() {
  return {
    maxDurationHours: Math.max(8, Number($("#maxDurationDays").value || 20) * 8),
    materialSlipDays: Math.max(1, Number($("#materialSlipDays").value || 10)),
    floatThresholdHours: Number($("#floatThresholdHours").value || 0),
    nearCriticalHours: Math.max(0, Number($("#nearCriticalDays").value || 10) * 8),
    executiveChangeLimit: Math.max(5, Math.min(20, Number($("#executiveChangeLimit").value || 10))),
    allowPositiveLag: $("#allowPositiveLag").checked,
  };
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
  const mustSee = materialChanges.filter(change => change.isMilestone || change.businessCritical);
  for (const change of [...mustSee, ...materialChanges]) {
    if (selected.some(item => item.id === change.id)) continue;
    const count = perWbs.get(change.wbsName) || 0;
    if (count >= 2 && !change.isMilestone && !change.businessCritical) continue;
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
  const statedStable = /\b(unchanged|no material change|on track|as planned|remains stable|no impact)\b/.test(text);
  const finishSlip = finishMovement.days || 0;

  if (!narrative.trim()) {
    conflicts.push({ type: "Evidence gap", title: "No schedule narrative was supplied", detail: "The XER can show what changed, but not the planner's stated reason or approval basis." });
    return conflicts;
  }
  if (statedStable && finishSlip >= 7) conflicts.push({ type: "Direct contradiction", title: `The narrative presents stability while the scheduled finish moved ${finishSlip} days`, detail: `${finishMovement.label}: ${formatDate(finishMovement.before)} → ${formatDate(finishMovement.after)}.` });
  if (statedStable && floatErosion.some(change => change.score >= 30)) conflicts.push({ type: "Hidden pressure", title: "The narrative presents stability while material float was consumed", detail: `${floatErosion.length} changed activities show float erosion that may reduce schedule resilience.` });
  if (relationshipChanges.length && !/\b(logic|relationship|predecessor|successor|resequence|sequence)\b/.test(text)) conflicts.push({ type: "Unexplained change", title: `${relationshipChanges.length} logic changes are not addressed in the narrative`, detail: "Changed predecessor logic can alter the driving path even when headline dates appear stable." });
  if (constraintChanges.length && !/\bconstraint|mandatory|must finish|must start|fixed date\b/.test(text)) conflicts.push({ type: "Unexplained change", title: `${constraintChanges.length} constraint changes are not addressed in the narrative`, detail: "A dated constraint can influence float and the apparent critical path." });
  if (finishSlip >= 7 && !/\b(delay|slip|later|movement|forecast|completion|finish)\b/.test(text)) conflicts.push({ type: "Narrative omission", title: "Material finish movement is not described", detail: `${finishMovement.label} moved ${finishSlip} days later between submissions.` });
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

function updateFileCard(kind) {
  const file = reviewState.files[kind];
  const capital = kind[0].toUpperCase() + kind.slice(1);
  const card = $(`#${kind}Drop`);
  card.classList.toggle("loaded", Boolean(file));
  $(`#${kind}FileName`).textContent = file?.name || `Drop ${kind} XER`;
  $(`#${kind}FileMeta`).textContent = file ? `${humanFileSize(file.size)} · ready for local review` : `Required · ${kind === "previous" ? "comparison point" : "latest update"}`;
  card.setAttribute("aria-label", file ? `${capital} XER loaded: ${file.name}` : `${capital} XER upload`);
  const ready = Boolean(reviewState.files.previous && reviewState.files.current);
  $("#runReview").disabled = !ready;
  $("#intakeStatus").textContent = ready ? "Submission pair ready" : "Waiting for two XER files";
}

async function loadFile(kind, file) {
  if (!file) return;
  if (!/\.xer$/i.test(file.name)) {
    showToast("Choose a Primavera P6 .xer file.");
    return;
  }
  const text = await file.text();
  reviewState.files[kind] = { name: file.name, size: file.size, text };
  updateFileCard(kind);
}

function updateEvidenceCard(kind) {
  const file = reviewState.files[kind];
  const card = $(`#${kind}Drop`);
  card.classList.toggle("loaded", Boolean(file));
  $(`#${kind}FileName`).textContent = file?.name || "Not supplied";
  $(`#${kind}FileMeta`).textContent = file ? `${humanFileSize(file.size)} · included in local review` : card.querySelector("small").dataset.default || card.querySelector("small").textContent;
}

async function loadEvidenceFile(kind, file) {
  if (!file) return;
  const allowed = kind === "baseline" ? /\.xer$/i : kind === "risk" ? /\.(csv|txt)$/i : kind === "basis" ? /\.(txt|md)$/i : /\.(csv|txt|md|json)$/i;
  if (!allowed.test(file.name)) {
    showToast(`Choose a supported ${kind} evidence file.`);
    return;
  }
  reviewState.files[kind] = { name: file.name, size: file.size, text: await file.text() };
  updateEvidenceCard(kind);
}

function renderReview() {
  const analysis = reviewState.analysis;
  const { current, previous, materialChanges, priorityChanges, conflicts, integrity, finishMovement, completeness } = analysis;
  const highChanges = priorityChanges.filter(change => change.severity === "high");
  $("#reviewProjectName").textContent = current.project.name;
  $("#reviewFilePair").textContent = `${previous.fileName} → ${current.fileName}`;
  $("#reviewTimestamp").textContent = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(analysis.generatedAt));
  $("#materialChangeCount").textContent = materialChanges.length;
  $("#finishMovement").textContent = finishMovement.days === null ? "N/A" : signed(finishMovement.days, "d");
  $("#finishMilestone").textContent = finishMovement.label;
  $("#contradictionCount").textContent = conflicts.length;
  $("#integrityCount").textContent = integrity.reduce((sum, item) => sum + item.count, 0);
  $("#evidenceCoverage").textContent = `${completeness.presentCount}/${completeness.totalCount}`;
  $("#coverageLabel").textContent = completeness.coverageLabel;

  $("#completenessSummary").textContent = `${current.tasks.length.toLocaleString()} activities, ${current.relations.length.toLocaleString()} relationships and ${current.calendars.length.toLocaleString()} calendars were readable. ${completeness.missing.length ? `Missing evidence: ${completeness.missing.join(", ")}.` : "All defined evidence classes were supplied."}`;
  $("#completenessGrid").innerHTML = completeness.entries.map(item => `<article class="completeness-card ${item.present ? "present" : "missing"}"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.status)}</strong><small>${escapeHtml(item.detail)}</small></article>`).join("");
  $("#completenessBoundary").innerHTML = `<strong>${escapeHtml(completeness.coverageLabel)}</strong><span>${escapeHtml(completeness.boundary)}</span>`;

  const headline = finishMovement.days > 0
    ? `The finish moved ${finishMovement.days} days. The narrative needs evidence.`
    : highChanges.length
      ? `${highChanges.length} high-priority changes need an owner.`
      : "No high-priority movement was detected.";
  $("#executiveHeadline").textContent = headline;
  $("#executiveSummary").textContent = `${current.tasks.length.toLocaleString()} activities and ${current.relations.length.toLocaleString()} relationships were compared with the previous submission. ProjectLens reduced ${analysis.changes.length.toLocaleString()} raw changes to ${materialChanges.length} material changes and a ${priorityChanges.length}-item executive queue. It found ${conflicts.length} narrative conflicts and ${integrity.reduce((sum, item) => sum + item.count, 0)} rule findings. These signals prioritise assurance review; they do not establish cause or entitlement.`;
  $("#headlineEvidence").innerHTML = [
    ["Submission dates", `${formatDate(previous.project.dataDate)} → ${formatDate(current.project.dataDate)}`],
    ["Scheduled finish", `${formatDate(finishMovement.before)} → ${formatDate(finishMovement.after)}`],
    ["Highest change", priorityChanges[0] ? `${priorityChanges[0].code} · ${priorityChanges[0].score}/100 review score` : "No material change"],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");

  $("#contradictionListXer").innerHTML = conflicts.length ? conflicts.slice(0, 5).map(conflict => `
    <article class="contradiction-card"><span>${escapeHtml(conflict.type)}</span><strong>${escapeHtml(conflict.title)}</strong><p>${escapeHtml(conflict.detail)}</p></article>`).join("") : `<div class="empty-panel">No deterministic narrative conflict was detected. This does not prove the narrative is complete.</div>`;

  $("#pathCandidateList").innerHTML = analysis.paths.length ? analysis.paths.slice(0, 8).map(path => `<article><span>${escapeHtml(path.classification)}</span><strong>${escapeHtml(path.code)} · ${escapeHtml(path.name)}</strong><small>${path.float}h exported float · ${path.predecessors} predecessors · ${path.successors} successors${path.businessCritical ? " · business milestone signal" : ""}</small></article>`).join("") : `<div class="empty-panel">No active tasks fall within the selected exported-float threshold.</div>`;
  const unlinked = analysis.decisionReconciliation.filter(item => !item.linked);
  $("#decisionReconciliation").innerHTML = analysis.evidence.decision?.decisions.length ? analysis.decisionReconciliation.slice(0, 8).map(item => `<article class="${item.linked ? "linked" : "unlinked"}"><span>${escapeHtml(item.status)}</span><strong>${escapeHtml(item.code)} · ${escapeHtml(item.name)}</strong><small>${item.linked ? escapeHtml(item.decisions.map(decision => decision.text).join(" | ")) : "No matching activity code in the supplied decision log"}</small></article>`).join("") : `<div class="empty-panel">No decision log was supplied. ${materialChanges.length} material changes therefore have no approval trace in this review.</div>`;
  $("#baselineMovementList").innerHTML = analysis.baselineMovements.length ? analysis.baselineMovements.slice(0, 8).map(item => `<article class="${item.movementDays > 0 ? "unlinked" : "linked"}"><span>${signed(item.movementDays, "d")}</span><strong>${escapeHtml(item.code)} · ${escapeHtml(item.name)}</strong><small>${escapeHtml(formatDate(item.baseline))} → ${escapeHtml(formatDate(item.current))}</small></article>`).join("") : `<div class="empty-panel">No separate baseline was supplied, or no matching baseline milestones were found.</div>`;
  $("#materialFunnelCopy").textContent = `${analysis.changes.length.toLocaleString()} raw changes → ${materialChanges.length} material → ${priorityChanges.length} executive priorities. Analyst view retains the full material set. ${unlinked.length} material changes have no linked decision record.`;

  renderChanges();
  $("#fitnessGrid").innerHTML = analysis.fitness.map(item => `<div class="fitness-item"><span>${escapeHtml(item.label)}</span><strong class="status-${item.tone}">${escapeHtml(item.status)}</strong><small>${escapeHtml(item.detail)}</small></div>`).join("");
  $("#integrityList").innerHTML = integrity.length ? integrity.map(item => `<div class="integrity-item"><span>${escapeHtml(item.label)}</span><strong class="status-${item.severity === "high" ? "warn" : "info"}">${item.count}</strong><small>${escapeHtml(item.detail)} ${item.examples.length ? `Examples: ${escapeHtml(item.examples.join(", "))}.` : ""}</small></div>`).join("") : `<div class="empty-panel">No findings were triggered by the selected rules.</div>`;
  $("#questionList").innerHTML = analysis.questions.length ? analysis.questions.map((item, index) => `<article class="question-card"><span>${escapeHtml(item.type)} · Q${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(item.question)}</strong><small>${escapeHtml(item.evidence)}</small><button type="button" data-question-index="${index}">Add to review log +</button></article>`).join("") : `<div class="empty-panel">No assurance questions were generated from the selected rules.</div>`;
  $$('[data-question-index]').forEach(button => button.addEventListener("click", () => {
    const question = analysis.questions[Number(button.dataset.questionIndex)];
    const change = question.changeId ? analysis.materialChanges.find(item => item.id === question.changeId) : null;
    addAction(question.question, question.evidence, change);
  }));
  renderActions();
}

function changeMatches(change, filter) {
  if (filter === "all") return true;
  if (filter === "high") return change.severity === "high";
  if (filter === "milestone") return change.isMilestone;
  if (filter === "path") return change.pathClass !== "Outside threshold" && change.pathClass !== "Not supplied";
  return change.types.includes(filter);
}

function renderChanges() {
  const source = document.body.dataset.mode === "analyst" ? reviewState.analysis.materialChanges : reviewState.analysis.priorityChanges;
  const changes = source.filter(change => changeMatches(change, reviewState.changeFilter));
  $("#changeList").innerHTML = changes.length ? changes.map(change => `
    <button class="change-row ${change.severity}" type="button" data-change-id="${escapeHtml(change.id)}">
      <span class="change-score">${change.score}</span>
      <span class="change-title"><span>${escapeHtml(change.types.join(" · "))}</span><strong>${escapeHtml(change.code)} · ${escapeHtml(change.name)}</strong><small>${escapeHtml(change.wbsName)} · ${escapeHtml(change.pathClass)}${reviewState.analysis.evidence.risk ? ` · ${change.riskLinked ? "risk linked" : "no linked risk"}` : ""}</small></span>
      <span class="change-evidence-summary"><span>Primary evidence</span><strong>${escapeHtml(change.summary)}</strong></span>
      <i>↗</i>
    </button>`).join("") : `<div class="no-changes">No material changes match this filter.</div>`;
  $$('[data-change-id]').forEach(button => button.addEventListener("click", () => openChange(button.dataset.changeId)));
}

function openChange(changeId) {
  const change = reviewState.analysis.materialChanges.find(item => item.id === changeId);
  if (!change) return;
  reviewState.selectedChangeId = changeId;
  $("#changeDialogSeverity").textContent = `${change.severity} priority · ${change.score}/100`;
  $("#changeDialogCode").textContent = change.code;
  $("#changeDialogTitle").textContent = change.name;
  $("#changeDialogSummary").textContent = `This activity was ranked using ${change.types.join(", ")} evidence. The score is an assurance priority, not a probability or causal finding.`;
  $("#changeDialogEvidence").innerHTML = change.evidence.map(item => `<div><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join("");
  $("#changeDialog").showModal();
}

function actionKey() {
  return "projectlens:schedule-actions:v1";
}

function storedActions() {
  try { return JSON.parse(localStorage.getItem(actionKey()) || "[]"); } catch { return []; }
}

function saveActions(actions) {
  localStorage.setItem(actionKey(), JSON.stringify(actions));
}

function currentProjectKey() {
  return reviewState.analysis?.current.project.id || "unassigned";
}

function addAction(title, evidence = "", change = null) {
  const actions = storedActions();
  actions.unshift({
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    projectId: currentProjectKey(),
    projectName: reviewState.analysis?.current.project.name || "Unassigned schedule",
    title,
    evidence,
    owner: $("#actionOwner")?.value.trim() || "Unassigned",
    due: $("#actionDue")?.value || "",
    done: false,
    changeCode: change?.code || null,
    sourceScore: change?.score ?? null,
    sourceDataDate: isoDate(reviewState.analysis?.current.project.dataDate),
    outcomeOverride: "",
    createdAt: new Date().toISOString(),
  });
  saveActions(actions);
  renderActions();
  showToast("Action saved in this browser.");
}

function renderActions() {
  const actions = storedActions().filter(action => action.projectId === currentProjectKey());
  $("#actionList").innerHTML = actions.length ? actions.map(action => `
    <article class="action-row ${action.done ? "done" : ""}" data-action-id="${escapeHtml(action.id)}">
      <input type="checkbox" ${action.done ? "checked" : ""} aria-label="Mark action complete">
      <div><strong>${escapeHtml(action.title)}</strong><small>${escapeHtml(action.evidence || "Manual review action")}${action.changeCode ? ` · Linked to ${escapeHtml(action.changeCode)}` : ""}</small></div>
      <span>${escapeHtml(action.owner)}${action.due ? ` · ${escapeHtml(action.due)}` : ""}</span>
      <select aria-label="Set intervention outcome">
        <option value="" ${!action.outcomeOverride ? "selected" : ""}>Auto outcome</option>
        <option value="Effective" ${action.outcomeOverride === "Effective" ? "selected" : ""}>Effective</option>
        <option value="Partly effective" ${action.outcomeOverride === "Partly effective" ? "selected" : ""}>Partly effective</option>
        <option value="Not effective" ${action.outcomeOverride === "Not effective" ? "selected" : ""}>Not effective</option>
        <option value="Not assessed" ${action.outcomeOverride === "Not assessed" ? "selected" : ""}>Not assessed</option>
      </select>
      <button type="button" aria-label="Delete action">×</button>
    </article>`).join("") : `<div class="empty-panel">No actions recorded for this schedule yet.</div>`;
  $$(".action-row").forEach(row => {
    row.querySelector('input[type="checkbox"]').addEventListener("change", event => {
      const all = storedActions();
      const action = all.find(item => item.id === row.dataset.actionId);
      if (action) action.done = event.target.checked;
      saveActions(all); renderActions();
    });
    row.querySelector("select").addEventListener("change", event => {
      const all = storedActions();
      const action = all.find(item => item.id === row.dataset.actionId);
      if (action) action.outcomeOverride = event.target.value;
      saveActions(all); renderActions(); renderFollowThrough();
    });
    row.querySelector("button").addEventListener("click", () => {
      saveActions(storedActions().filter(item => item.id !== row.dataset.actionId)); renderActions();
    });
  });
  renderFollowThrough();
}

function interventionOutcomes() {
  if (!reviewState.analysis) return [];
  const currentDate = isoDate(reviewState.analysis.current.project.dataDate);
  return storedActions()
    .filter(action => action.projectId === currentProjectKey())
    .map(action => {
      const laterSubmission = action.sourceDataDate && currentDate && currentDate > action.sourceDataDate;
      const currentChange = action.changeCode ? reviewState.analysis.materialChanges.find(change => change.code === action.changeCode) : null;
      let outcome = action.outcomeOverride || "Not yet assessable";
      let detail = action.outcomeOverride ? "Outcome set by the reviewer." : "Run a later submission pair or set a human outcome.";
      if (!action.outcomeOverride && laterSubmission && action.changeCode) {
        if (!currentChange) {
          outcome = "Resolved in latest comparison";
          detail = `${action.changeCode} no longer appears in the material-change set.`;
        } else if (action.sourceScore !== null && currentChange.score < action.sourceScore) {
          outcome = "Improving but repeated";
          detail = `${action.changeCode} remains material, with review score ${action.sourceScore} → ${currentChange.score}.`;
        } else {
          outcome = "Repeated or worsened";
          detail = `${action.changeCode} remains material at ${currentChange.score}/100.`;
        }
      } else if (!action.outcomeOverride && laterSubmission && !action.changeCode) {
        outcome = action.done ? "Closed by reviewer" : "Needs human assessment";
        detail = "This action was not linked to an activity code, so schedule resolution cannot be inferred.";
      }
      return { ...action, outcome, detail, laterSubmission };
    });
}

function renderFollowThrough() {
  const target = $("#followThroughList");
  if (!target || !reviewState.analysis) return;
  const outcomes = interventionOutcomes();
  target.innerHTML = outcomes.length ? outcomes.map(item => `
    <article class="follow-through-row">
      <span class="outcome-${escapeHtml(item.outcome.toLowerCase().replace(/[^a-z]+/g, "-"))}">${escapeHtml(item.outcome)}</span>
      <div><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.detail)}</small></div>
      <i>${escapeHtml(item.changeCode || "Manual decision")}</i>
    </article>`).join("") : `<div class="empty-panel">No interventions have been recorded for this project yet. Add an evidence-linked action below, then compare the next submission.</div>`;
}

function reviewBrief() {
  const analysis = reviewState.analysis;
  return `${analysis.current.project.name} schedule evidence review\n\nSubmission: ${analysis.previous.fileName} to ${analysis.current.fileName}\nData dates: ${formatDate(analysis.previous.project.dataDate)} to ${formatDate(analysis.current.project.dataDate)}\nScheduled finish movement: ${signed(analysis.finishMovement.days, " days")}\nEvidence coverage: ${analysis.completeness.presentCount}/${analysis.completeness.totalCount} (${analysis.completeness.coverageLabel})\nRaw changes: ${analysis.changes.length}\nMaterial changes: ${analysis.materialChanges.length}\nExecutive priorities: ${analysis.priorityChanges.length}\nNarrative conflicts: ${analysis.conflicts.length}\nUnlinked material decisions: ${analysis.decisionReconciliation.filter(item => !item.linked).length}\nIntegrity findings: ${analysis.integrity.reduce((sum, item) => sum + item.count, 0)}\n\nPriority evidence:\n${analysis.priorityChanges.map((change, index) => `${index + 1}. ${change.code} ${change.name}: ${change.summary}`).join("\n") || "No material changes detected."}\n\nAssurance questions:\n${analysis.questions.map((item, index) => `${index + 1}. ${item.question}`).join("\n") || "No questions generated."}\n\nEvidence boundary: ${analysis.completeness.boundary}\nThis deterministic review does not establish contractual entitlement, causation or a probability of failure.`;
}

function publicReviewPack() {
  const analysis = reviewState.analysis;
  return {
    generatedAt: analysis.generatedAt,
    product: "ProjectLens Schedule Evidence Review",
    project: analysis.current.project,
    submissions: {
      previous: { fileName: analysis.previous.fileName, dataDate: isoDate(analysis.previous.project.dataDate), activities: analysis.previous.tasks.length, relationships: analysis.previous.relations.length },
      current: { fileName: analysis.current.fileName, dataDate: isoDate(analysis.current.project.dataDate), activities: analysis.current.tasks.length, relationships: analysis.current.relations.length },
    },
    rules: analysis.rules,
    completeness: analysis.completeness,
    finishMovement: { ...analysis.finishMovement, before: isoDate(analysis.finishMovement.before), after: isoDate(analysis.finishMovement.after) },
    rawChangeCount: analysis.changes.length,
    executivePriorities: analysis.priorityChanges.map(change => ({ id: change.id, code: change.code, name: change.name, wbsName: change.wbsName, score: change.score, severity: change.severity, pathClass: change.pathClass, businessCritical: change.businessCritical, types: change.types, evidence: change.evidence })),
    materialChanges: analysis.materialChanges.map(change => ({ id: change.id, code: change.code, name: change.name, wbsName: change.wbsName, score: change.score, severity: change.severity, pathClass: change.pathClass, businessCritical: change.businessCritical, riskLinked: change.riskLinked, types: change.types, evidence: change.evidence })),
    pathCandidates: analysis.paths,
    decisionReconciliation: analysis.decisionReconciliation,
    baselineMovements: analysis.baselineMovements,
    narrativeConflicts: analysis.conflicts,
    integrityFindings: analysis.integrity,
    fitness: analysis.fitness,
    assuranceQuestions: analysis.questions,
    actions: storedActions().filter(action => action.projectId === currentProjectKey()),
    interventionOutcomes: interventionOutcomes(),
    boundary: "Deterministic evidence review. Not contractual entitlement, causal proof or probability of failure.",
  };
}

function downloadReviewPack() {
  const content = JSON.stringify(publicReviewPack(), null, 2);
  const blob = new Blob([content], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${currentProjectKey().replace(/[^a-z0-9]+/gi, "-").toLowerCase()}-schedule-review.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function runReview() {
  if (!reviewState.files.previous || !reviewState.files.current) return;
  const loader = $("#reviewLoader");
  loader.hidden = false;
  const steps = [
    ["Parsing schedule network", "Reading activities, relationships, WBS and calendars"],
    ["Gating the evidence", "Checking baseline, risk, basis, codes, calendars and decisions"],
    ["Comparing submissions", "Ranking dates, float, logic, constraints and durations"],
    ["Testing claims and approvals", "Reconciling narrative, decisions and intervention memory"],
  ];
  try {
    for (const [title, detail] of steps) {
      $("#loaderTitle").textContent = title;
      $("#loaderStep").textContent = detail;
      await new Promise(resolve => setTimeout(resolve, 260));
    }
    const previous = parseXer(reviewState.files.previous.text, reviewState.files.previous.name);
    const current = parseXer(reviewState.files.current.text, reviewState.files.current.name);
    const baseline = reviewState.files.baseline ? parseXer(reviewState.files.baseline.text, reviewState.files.baseline.name) : null;
    const evidence = {
      risk: reviewState.files.risk ? parseEvidenceText(reviewState.files.risk.text, "risk") : null,
      basis: reviewState.files.basis ? parseEvidenceText(reviewState.files.basis.text, "basis") : null,
      decision: reviewState.files.decision ? parseEvidenceText(reviewState.files.decision.text, "decision") : null,
      files: Object.fromEntries(["baseline", "risk", "basis", "decision"].map(kind => [kind, reviewState.files[kind] ? { name: reviewState.files[kind].name, size: reviewState.files[kind].size } : null])),
    };
    reviewState.narrative = $("#narrativeText").value.trim();
    reviewState.analysis = buildReview(previous, current, reviewState.narrative, selectedRules(), evidence, baseline);
    renderReview();
    $("#reviewResults").hidden = false;
    loader.hidden = true;
    $("#reviewResults").scrollIntoView({ behavior: "smooth", block: "start" });
    showToast("Review complete. Files remained in this browser.");
  } catch (error) {
    loader.hidden = true;
    console.error(error);
    showToast(error.message || "The XER review could not be completed.");
  }
}

async function loadDemo() {
  try {
    const responses = await Promise.all([
      fetch("demo/northstar-previous.xer"),
      fetch("demo/northstar-current.xer"),
      fetch("demo/northstar-narrative.txt"),
      fetch("demo/northstar-baseline.xer"),
      fetch("demo/northstar-risks.csv"),
      fetch("demo/northstar-schedule-basis.md"),
      fetch("demo/northstar-decisions.csv"),
    ]);
    if (!responses.every(response => response.ok)) throw new Error("Demo evidence could not be loaded.");
    const [previousText, currentText, narrative, baselineText, riskText, basisText, decisionText] = await Promise.all(responses.map(response => response.text()));
    reviewState.files.previous = { name: "northstar-previous.xer", size: previousText.length, text: previousText };
    reviewState.files.current = { name: "northstar-current.xer", size: currentText.length, text: currentText };
    reviewState.files.baseline = { name: "northstar-baseline.xer", size: baselineText.length, text: baselineText };
    reviewState.files.risk = { name: "northstar-risks.csv", size: riskText.length, text: riskText };
    reviewState.files.basis = { name: "northstar-schedule-basis.md", size: basisText.length, text: basisText };
    reviewState.files.decision = { name: "northstar-decisions.csv", size: decisionText.length, text: decisionText };
    $("#narrativeText").value = narrative;
    updateFileCard("previous"); updateFileCard("current");
    ["baseline", "risk", "basis", "decision"].forEach(updateEvidenceCard);
    $("#intakeSection").scrollIntoView({ behavior: "smooth", block: "start" });
    setTimeout(runReview, 500);
  } catch (error) {
    console.error(error); showToast(error.message);
  }
}

function bindDropZone(kind) {
  const drop = $(`#${kind}Drop`);
  const input = $(`#${kind}File`);
  input.addEventListener("change", () => loadFile(kind, input.files[0]));
  ["dragenter", "dragover"].forEach(eventName => drop.addEventListener(eventName, event => { event.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(eventName => drop.addEventListener(eventName, event => { event.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", event => loadFile(kind, event.dataTransfer.files[0]));
}

function bindEvents() {
  bindDropZone("previous"); bindDropZone("current");
  ["baseline", "risk", "basis", "decision"].forEach(kind => {
    $(`#${kind}File`).addEventListener("change", event => loadEvidenceFile(kind, event.target.files[0]));
  });
  $("#jumpToFiles").addEventListener("click", () => $("#intakeSection").scrollIntoView({ behavior: "smooth" }));
  $("#loadDemo").addEventListener("click", loadDemo);
  $("#runReview").addEventListener("click", runReview);
  $("#narrativeFile").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (file) $("#narrativeText").value = await file.text();
  });
  $$('[data-mode-switch]').forEach(button => button.addEventListener("click", () => {
    document.body.dataset.mode = button.dataset.modeSwitch;
    $$('[data-mode-switch]').forEach(item => item.classList.toggle("active", item === button));
    if (reviewState.analysis) renderChanges();
  }));
  $$('[data-change-filter]').forEach(button => button.addEventListener("click", () => {
    reviewState.changeFilter = button.dataset.changeFilter;
    $$('[data-change-filter]').forEach(item => item.classList.toggle("active", item === button));
    renderChanges();
  }));
  $("#changeDialog .dialog-close").addEventListener("click", () => $("#changeDialog").close());
  $("#changeDialogAction").addEventListener("click", () => {
    const change = reviewState.analysis.materialChanges.find(item => item.id === reviewState.selectedChangeId);
    if (change) addAction(`Explain and agree the response to ${change.code} ${change.name}`, change.summary, change);
    $("#changeDialog").close();
  });
  $("#actionForm").addEventListener("submit", event => {
    event.preventDefault();
    const title = $("#actionTitle").value.trim();
    if (!title) return;
    addAction(title, "Manual review action");
    $("#actionTitle").value = "";
  });
  $("#copyBrief").addEventListener("click", async () => {
    await navigator.clipboard.writeText(reviewBrief());
    showToast("Executive brief copied.");
  });
  $("#downloadPack").addEventListener("click", downloadReviewPack);
}

bindEvents();
window.ProjectLensXer = { parseXer, parseEvidenceText, compareSchedules, assessCompleteness, pathCandidates, reconcileDecisions, compareBaseline, buildReview };
