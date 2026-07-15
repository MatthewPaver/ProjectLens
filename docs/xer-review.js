const reviewState = {
  files: { previous: null, current: null },
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
  const tasks = tables.TASK.map(row => {
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
  const relations = (tables.TASKPRED || []).map(row => ({
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
    project: {
      id: projectRow.proj_id || "unknown-project",
      name: projectRow.proj_short_name || projectRow.proj_name || "Unnamed P6 project",
      dataDate: parseDate(projectRow.last_recalc_date || projectRow.data_date),
      plannedStart: parseDate(projectRow.plan_start_date || projectRow.scd_start_date),
      scheduledFinish: parseDate(projectRow.scd_end_date || projectRow.plan_end_date),
    },
    tableCounts: Object.fromEntries(Object.entries(tables).map(([name, rows]) => [name, rows.length])),
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
    allowPositiveLag: $("#allowPositiveLag").checked,
  };
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

    if (!before || !after) {
      const added = Boolean(after);
      types.push(added ? "added" : "removed");
      evidence.push({ label: added ? "Current XER" : "Previous XER", value: `${added ? "Added" : "Removed"} activity ${code}` });
      score += 22;
      if ((after || before)?.isMilestone) score += 20;
    } else {
      const finishDelta = dayDifference(after.finish, before.finish);
      const startDelta = dayDifference(after.start, before.start);
      const durationDelta = after.duration !== null && before.duration !== null ? after.duration - before.duration : null;
      const floatDelta = after.float !== null && before.float !== null ? after.float - before.float : null;
      const constraintChanged = `${before.constraint}|${isoDate(before.constraintDate)}` !== `${after.constraint}|${isoDate(after.constraintDate)}`;
      const beforeLogic = (previousRelations.get(code) || []).join(";");
      const afterLogic = (currentRelations.get(code) || []).join(";");
      const logicChanged = beforeLogic !== afterLogic;

      if (finishDelta !== null && finishDelta !== 0) {
        types.push("dates");
        score += Math.min(36, 8 + Math.abs(finishDelta));
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
    changes.push({
      id: `${code}-${changes.length}`,
      code,
      name: task.name,
      wbsName: task.wbsName,
      isMilestone: task.isMilestone,
      types: [...new Set(types)],
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

function assessFitness(previous, current, narrative) {
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
    { label: "Baseline evidence", status: "Separate file", tone: "info", detail: "XER does not guarantee embedded baseline export. The previous submission is the explicit comparison point." },
    { label: "Narrative", status: narrative.trim() ? "Supplied" : "Missing", tone: narrative.trim() ? "good" : "warn", detail: narrative.trim() ? `${narrative.trim().length.toLocaleString()} characters available for claim testing` : "Contradiction testing is limited without the update narrative." },
  ];
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

function buildQuestions(changes, conflicts, integrity) {
  const questions = [];
  changes.filter(change => change.severity === "high").slice(0, 4).forEach(change => {
    questions.push({ type: "Material change", question: `What approved change or delivery event explains the movement in ${change.code} ${change.name}?`, evidence: change.summary, changeId: change.id });
  });
  conflicts.slice(0, 3).forEach(conflict => questions.push({ type: conflict.type, question: `How should the submission narrative be corrected or evidenced to address: ${conflict.title.toLowerCase()}?`, evidence: conflict.detail }));
  integrity.filter(item => item.severity === "high" || item.count >= 3).slice(0, 3).forEach(item => questions.push({ type: "Schedule integrity", question: `What is the acceptance or remediation decision for ${item.count} ${item.label.toLowerCase()}?`, evidence: `${item.detail} Examples: ${item.examples.join(", ") || "none"}.` }));
  return questions.slice(0, 8);
}

function buildReview(previous, current, narrative, rules) {
  const changes = compareSchedules(previous, current, rules);
  const integrity = assessIntegrity(current, rules);
  const fitness = assessFitness(previous, current, narrative);
  const finishMovement = projectFinishMovement(previous, current);
  const conflicts = narrativeAssessment(narrative, changes, finishMovement);
  const materialChanges = changes.filter(change => change.score >= 27 || change.isMilestone || change.types.includes("constraint"));
  const questions = buildQuestions(materialChanges, conflicts, integrity);
  return {
    generatedAt: new Date().toISOString(),
    previous,
    current,
    narrative,
    rules,
    changes,
    materialChanges,
    integrity,
    fitness,
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

function renderReview() {
  const analysis = reviewState.analysis;
  const { current, previous, materialChanges, conflicts, integrity, finishMovement } = analysis;
  const highChanges = materialChanges.filter(change => change.severity === "high");
  $("#reviewProjectName").textContent = current.project.name;
  $("#reviewFilePair").textContent = `${previous.fileName} → ${current.fileName}`;
  $("#reviewTimestamp").textContent = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(analysis.generatedAt));
  $("#materialChangeCount").textContent = materialChanges.length;
  $("#finishMovement").textContent = finishMovement.days === null ? "N/A" : signed(finishMovement.days, "d");
  $("#finishMilestone").textContent = finishMovement.label;
  $("#contradictionCount").textContent = conflicts.length;
  $("#integrityCount").textContent = integrity.reduce((sum, item) => sum + item.count, 0);

  const headline = finishMovement.days > 0
    ? `The finish moved ${finishMovement.days} days. The narrative needs evidence.`
    : highChanges.length
      ? `${highChanges.length} high-priority changes need an owner.`
      : "No high-priority movement was detected.";
  $("#executiveHeadline").textContent = headline;
  $("#executiveSummary").textContent = `${current.tasks.length.toLocaleString()} activities and ${current.relations.length.toLocaleString()} relationships were compared with the previous submission. ProjectLens found ${materialChanges.length} material changes, ${conflicts.length} narrative conflicts and ${integrity.reduce((sum, item) => sum + item.count, 0)} rule findings. These signals prioritise assurance review; they do not establish cause or entitlement.`;
  $("#headlineEvidence").innerHTML = [
    ["Submission dates", `${formatDate(previous.project.dataDate)} → ${formatDate(current.project.dataDate)}`],
    ["Scheduled finish", `${formatDate(finishMovement.before)} → ${formatDate(finishMovement.after)}`],
    ["Highest change", materialChanges[0] ? `${materialChanges[0].code} · ${materialChanges[0].score}/100 review score` : "No material change"],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");

  $("#contradictionListXer").innerHTML = conflicts.length ? conflicts.slice(0, 5).map(conflict => `
    <article class="contradiction-card"><span>${escapeHtml(conflict.type)}</span><strong>${escapeHtml(conflict.title)}</strong><p>${escapeHtml(conflict.detail)}</p></article>`).join("") : `<div class="empty-panel">No deterministic narrative conflict was detected. This does not prove the narrative is complete.</div>`;

  renderChanges();
  $("#fitnessGrid").innerHTML = analysis.fitness.map(item => `<div class="fitness-item"><span>${escapeHtml(item.label)}</span><strong class="status-${item.tone}">${escapeHtml(item.status)}</strong><small>${escapeHtml(item.detail)}</small></div>`).join("");
  $("#integrityList").innerHTML = integrity.length ? integrity.map(item => `<div class="integrity-item"><span>${escapeHtml(item.label)}</span><strong class="status-${item.severity === "high" ? "warn" : "info"}">${item.count}</strong><small>${escapeHtml(item.detail)} ${item.examples.length ? `Examples: ${escapeHtml(item.examples.join(", "))}.` : ""}</small></div>`).join("") : `<div class="empty-panel">No findings were triggered by the selected rules.</div>`;
  $("#questionList").innerHTML = analysis.questions.length ? analysis.questions.map((item, index) => `<article class="question-card"><span>${escapeHtml(item.type)} · Q${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(item.question)}</strong><small>${escapeHtml(item.evidence)}</small><button type="button" data-question-index="${index}">Add to review log +</button></article>`).join("") : `<div class="empty-panel">No assurance questions were generated from the selected rules.</div>`;
  $$('[data-question-index]').forEach(button => button.addEventListener("click", () => {
    const question = analysis.questions[Number(button.dataset.questionIndex)];
    addAction(question.question, question.evidence);
  }));
  renderActions();
}

function changeMatches(change, filter) {
  if (filter === "all") return true;
  if (filter === "high") return change.severity === "high";
  if (filter === "milestone") return change.isMilestone;
  return change.types.includes(filter);
}

function renderChanges() {
  const changes = reviewState.analysis.materialChanges.filter(change => changeMatches(change, reviewState.changeFilter));
  $("#changeList").innerHTML = changes.length ? changes.map(change => `
    <button class="change-row ${change.severity}" type="button" data-change-id="${escapeHtml(change.id)}">
      <span class="change-score">${change.score}</span>
      <span class="change-title"><span>${escapeHtml(change.types.join(" · "))}</span><strong>${escapeHtml(change.code)} · ${escapeHtml(change.name)}</strong><small>${escapeHtml(change.wbsName)}</small></span>
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

function addAction(title, evidence = "") {
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
      <div><strong>${escapeHtml(action.title)}</strong><small>${escapeHtml(action.evidence || "Manual review action")}</small></div>
      <span>${escapeHtml(action.owner)}${action.due ? ` · ${escapeHtml(action.due)}` : ""}</span>
      <button type="button" aria-label="Delete action">×</button>
    </article>`).join("") : `<div class="empty-panel">No actions recorded for this schedule yet.</div>`;
  $$(".action-row").forEach(row => {
    row.querySelector('input[type="checkbox"]').addEventListener("change", event => {
      const all = storedActions();
      const action = all.find(item => item.id === row.dataset.actionId);
      if (action) action.done = event.target.checked;
      saveActions(all); renderActions();
    });
    row.querySelector("button").addEventListener("click", () => {
      saveActions(storedActions().filter(item => item.id !== row.dataset.actionId)); renderActions();
    });
  });
}

function reviewBrief() {
  const analysis = reviewState.analysis;
  return `${analysis.current.project.name} schedule evidence review\n\nSubmission: ${analysis.previous.fileName} to ${analysis.current.fileName}\nData dates: ${formatDate(analysis.previous.project.dataDate)} to ${formatDate(analysis.current.project.dataDate)}\nScheduled finish movement: ${signed(analysis.finishMovement.days, " days")}\nMaterial changes: ${analysis.materialChanges.length}\nNarrative conflicts: ${analysis.conflicts.length}\nIntegrity findings: ${analysis.integrity.reduce((sum, item) => sum + item.count, 0)}\n\nPriority evidence:\n${analysis.materialChanges.slice(0, 5).map((change, index) => `${index + 1}. ${change.code} ${change.name}: ${change.summary}`).join("\n") || "No material changes detected."}\n\nAssurance questions:\n${analysis.questions.map((item, index) => `${index + 1}. ${item.question}`).join("\n") || "No questions generated."}\n\nBoundary: This deterministic review does not establish contractual entitlement, causation or a probability of failure.`;
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
    finishMovement: { ...analysis.finishMovement, before: isoDate(analysis.finishMovement.before), after: isoDate(analysis.finishMovement.after) },
    materialChanges: analysis.materialChanges.map(change => ({ id: change.id, code: change.code, name: change.name, wbsName: change.wbsName, score: change.score, severity: change.severity, types: change.types, evidence: change.evidence })),
    narrativeConflicts: analysis.conflicts,
    integrityFindings: analysis.integrity,
    fitness: analysis.fitness,
    assuranceQuestions: analysis.questions,
    actions: storedActions().filter(action => action.projectId === currentProjectKey()),
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
    ["Comparing submissions", "Ranking dates, float, logic, constraints and durations"],
    ["Testing the narrative", "Checking claims against deterministic schedule evidence"],
  ];
  try {
    for (const [title, detail] of steps) {
      $("#loaderTitle").textContent = title;
      $("#loaderStep").textContent = detail;
      await new Promise(resolve => setTimeout(resolve, 260));
    }
    const previous = parseXer(reviewState.files.previous.text, reviewState.files.previous.name);
    const current = parseXer(reviewState.files.current.text, reviewState.files.current.name);
    reviewState.narrative = $("#narrativeText").value.trim();
    reviewState.analysis = buildReview(previous, current, reviewState.narrative, selectedRules());
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
    const [previousResponse, currentResponse, narrativeResponse] = await Promise.all([
      fetch("demo/northstar-previous.xer"),
      fetch("demo/northstar-current.xer"),
      fetch("demo/northstar-narrative.txt"),
    ]);
    if (![previousResponse, currentResponse, narrativeResponse].every(response => response.ok)) throw new Error("Demo evidence could not be loaded.");
    const [previousText, currentText, narrative] = await Promise.all([previousResponse.text(), currentResponse.text(), narrativeResponse.text()]);
    reviewState.files.previous = { name: "northstar-previous.xer", size: previousText.length, text: previousText };
    reviewState.files.current = { name: "northstar-current.xer", size: currentText.length, text: currentText };
    $("#narrativeText").value = narrative;
    updateFileCard("previous"); updateFileCard("current");
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
  }));
  $$('[data-change-filter]').forEach(button => button.addEventListener("click", () => {
    reviewState.changeFilter = button.dataset.changeFilter;
    $$('[data-change-filter]').forEach(item => item.classList.toggle("active", item === button));
    renderChanges();
  }));
  $("#changeDialog .dialog-close").addEventListener("click", () => $("#changeDialog").close());
  $("#changeDialogAction").addEventListener("click", () => {
    const change = reviewState.analysis.materialChanges.find(item => item.id === reviewState.selectedChangeId);
    if (change) addAction(`Explain and agree the response to ${change.code} ${change.name}`, change.summary);
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
window.ProjectLensXer = { parseXer, compareSchedules, buildReview };
