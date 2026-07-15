const assuranceState = {
  files: { previous: null, current: null },
  review: null,
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

const comparableCases = [
  {
    id: "DG-024",
    title: "Late signalling interface change",
    decision: "Conditional approval with a 10-day design freeze, named interface owner and protected regression-test window.",
    outcome: "+8 days; no repeated safety retest",
  },
  {
    id: "DG-021",
    title: "Passenger information requirement added",
    decision: "Minimum compliant scope approved behind a feature flag; non-regulatory functions deferred.",
    outcome: "+3 days; deferred functions followed later",
  },
  {
    id: "DG-012",
    title: "Additional security screening route",
    decision: "Full change approved while integrated testing was compressed to protect cutover.",
    outcome: "+17 days after integration defects escaped",
  },
];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function numberValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(String(value).replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function parseDate(value) {
  if (!value) return null;
  const clean = String(value).trim();
  const isoMatch = clean.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (isoMatch) return new Date(`${isoMatch[1]}-${isoMatch[2]}-${isoMatch[3]}T12:00:00Z`);
  const timestamp = Date.parse(clean);
  return Number.isNaN(timestamp) ? null : new Date(timestamp);
}

function formatDate(value) {
  const date = value instanceof Date ? value : parseDate(value);
  return date ? new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(date) : "Not supplied";
}

function dayDifference(current, previous) {
  if (!current || !previous) return null;
  return Math.round((current.getTime() - previous.getTime()) / 86400000);
}

function parseXer(text, fileName) {
  if (!text?.trim()) throw new Error(`${fileName} is empty.`);
  const tables = {};
  let tableName = "";
  let fields = [];

  text.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n").split("\n").forEach(line => {
    if (!line) return;
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
  });

  if (!tables.PROJECT?.length || !tables.TASK?.length) throw new Error(`${fileName} is not a readable Primavera P6 XER export.`);

  const project = tables.PROJECT[0];
  const tasks = tables.TASK.map(row => ({
    code: row.task_code || row.task_id,
    name: row.task_name || row.task_code || row.task_id,
    finish: parseDate(row.early_end_date || row.target_end_date || row.act_end_date),
    float: numberValue(row.total_float_hr_cnt),
    constraint: row.constraint_type || row.cstr_type || "",
    constraintDate: parseDate(row.cstr_date || row.constraint_date),
  }));
  const relations = (tables.TASKPRED || []).map(row => `${row.pred_task_id}|${row.task_id}|${row.pred_type || "PR_FS"}|${row.lag_hr_cnt || 0}`);

  return {
    fileName,
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
  let constraintChanges = 0;
  let floatErosion = 0;

  codes.forEach(code => {
    const before = previous.taskByCode.get(code);
    const after = current.taskByCode.get(code);
    if (!before || !after) {
      changes.push({ code, name: (after || before).name, summary: after ? "Activity added" : "Activity removed" });
      return;
    }
    const finishDelta = dayDifference(after.finish, before.finish);
    const constraintChanged = `${before.constraint}|${formatDate(before.constraintDate)}` !== `${after.constraint}|${formatDate(after.constraintDate)}`;
    const floatDelta = before.float !== null && after.float !== null ? after.float - before.float : null;
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

  const relationChanges = [...new Set([...previous.relations, ...current.relations])].filter(signature => previous.relations.has(signature) !== current.relations.has(signature)).length;
  return { changes, constraintChanges, floatErosion, relationChanges };
}

function analyseReview(previous, current, narrative) {
  const finishMovement = dayDifference(current.project.scheduledFinish, previous.project.scheduledFinish);
  const changes = taskChanges(previous, current);
  const text = narrative.toLowerCase();
  const statedStable = /\b(unchanged|no material change|on track|as planned|remains stable|no impact)\b/.test(text);
  const blockers = [];

  if (!narrative.trim()) {
    blockers.push({
      type: "Evidence gap",
      title: "No change narrative was supplied",
      detail: "The schedules can show movement, but not the stated reason, mitigation or approval basis.",
    });
  }
  if (statedStable && finishMovement >= 7) {
    blockers.push({
      type: "Direct contradiction",
      title: `Finish moved ${finishMovement} days but the narrative presents stability`,
      detail: `${formatDate(previous.project.scheduledFinish)} → ${formatDate(current.project.scheduledFinish)}.`,
    });
  }
  if (changes.constraintChanges && !/\bconstraint|mandatory|fixed date|must finish|must start\b/.test(text)) {
    blockers.push({
      type: "Unexplained change",
      title: `${changes.constraintChanges} constraint changes are not addressed`,
      detail: "Dated constraints can influence float and the apparent driving path.",
    });
  }
  if (changes.relationChanges && !/\b(logic|relationship|predecessor|successor|resequence|sequence)\b/.test(text)) {
    blockers.push({
      type: "Unexplained change",
      title: `${changes.relationChanges} relationship changes are not addressed`,
      detail: "Changed logic can move the driving path even when headline dates appear stable.",
    });
  }
  if (changes.floatErosion && statedStable) {
    blockers.push({
      type: "Hidden pressure",
      title: `${changes.floatErosion} changed activities consumed float`,
      detail: "The narrative presents stability while schedule resilience reduced.",
    });
  }
  if (/\b(manage|mitigat|governance|action)\b/.test(text) && !/\b(owner|responsible|accountable|due|by \d{1,2})\b/.test(text)) {
    blockers.push({
      type: "Ownership gap",
      title: "The stated response has no named owner or due date",
      detail: "A mitigation cannot be followed through unless responsibility and timing are explicit.",
    });
  }

  return {
    previous,
    current,
    finishMovement,
    changes,
    blockers: blockers.slice(0, 3),
    generatedAt: new Date().toISOString(),
  };
}

function showToast(message) {
  const toast = $("#assuranceToast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 3300);
}

function storage(key) {
  try { return JSON.parse(localStorage.getItem(key) || "[]"); } catch { return []; }
}

function saveStorage(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

const storageKeys = {
  decisions: "projectlens:change-decisions:v1",
  conditions: "projectlens:change-conditions:v1",
  requests: "projectlens:evidence-requests:v1",
};

function evidenceVersion(review) {
  return `${review.previous.fileName} → ${review.current.fileName}`;
}

function renderReview(review) {
  assuranceState.review = review;
  const ready = review.blockers.length === 0;
  const topBlockers = review.blockers.slice(0, 3);
  $("#reviewName").textContent = review.current.project.name;
  $("#reviewEvidenceVersion").textContent = `${evidenceVersion(review)} · checked ${new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(review.generatedAt))}`;
  $("#readinessCard").classList.toggle("ready", ready);
  $("#readinessStatus").textContent = ready ? "Ready for human decision" : "Needs evidence before decision";
  $("#readinessHeadline").textContent = ready
    ? "No deterministic decision blocker was found in the supplied evidence."
    : topBlockers[0].title;
  $("#readinessExplanation").textContent = ready
    ? "This does not guarantee that the pack is complete. A named decision authority must still review it."
    : `Resolve ${review.blockers.length} blocker${review.blockers.length === 1 ? "" : "s"} or record why the decision authority accepts the remaining uncertainty.`;
  $("#blockerList").innerHTML = topBlockers.length ? topBlockers.map((blocker, index) => `
    <article class="blocker">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <div><strong>${escapeHtml(blocker.title)}</strong><small>${escapeHtml(blocker.detail)}</small></div>
      <i>${escapeHtml(blocker.type)}</i>
    </article>`).join("") : `
    <article class="blocker"><span>✓</span><div><strong>No deterministic blockers found</strong><small>Continue with professional and contractual review.</small></div><i>Review</i></article>`;
  $("#requestAnswers").hidden = ready;
  $("#decisionWarning").hidden = ready;

  $("#requestList").innerHTML = topBlockers.map((blocker, index) => `
    <article class="request-item"><span>Question ${String(index + 1).padStart(2, "0")} · ${escapeHtml(blocker.type)}</span><strong>${escapeHtml(questionFor(blocker))}</strong><small>${escapeHtml(blocker.detail)}</small></article>`).join("");

  const facts = [
    ["Scheduled finish", review.finishMovement === null ? "Not comparable" : `${review.finishMovement > 0 ? "+" : ""}${review.finishMovement} days`, `${formatDate(review.previous.project.scheduledFinish)} → ${formatDate(review.current.project.scheduledFinish)}`],
    ["Material activities", String(review.changes.changes.length), "Activities with material date, float, constraint or presence changes"],
    ["Constraint changes", String(review.changes.constraintChanges), "Compared by constraint type and date"],
    ["Float erosion", String(review.changes.floatErosion), "Changed activities with lower total float"],
    ["Relationship changes", String(review.changes.relationChanges), "Added or removed predecessor signatures"],
    ["Evidence supplied", "2 XER files", $("#assuranceNarrative").value.trim() ? "Schedule narrative included" : "No narrative supplied"],
  ];
  $("#evidenceSummary").innerHTML = facts.map(([label, value, detail]) => `
    <article class="evidence-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></article>`).join("");
  $("#comparableCases").innerHTML = comparableCases.map(item => `
    <article class="comparable-case"><span>${escapeHtml(item.id)} · Synthetic precedent</span><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.decision)}</p><small>${escapeHtml(item.outcome)}</small></article>`).join("");

  $("#packIntake").hidden = true;
  $("#readinessWorkspace").hidden = false;
  $("#requestPanel").hidden = true;
  $("#decisionPanel").hidden = true;
  $("#readinessWorkspace").scrollIntoView({ behavior: "smooth", block: "start" });
}

function questionFor(blocker) {
  if (blocker.type === "Direct contradiction") return "How should the narrative be corrected or evidenced against the finish movement?";
  if (blocker.type === "Ownership gap") return "Who owns the stated response, and by when must it be complete?";
  if (blocker.type === "Evidence gap") return "What narrative, reason and approval basis supports this submission?";
  return `What approved change or evidence explains: ${blocker.title.toLowerCase()}?`;
}

async function loadFile(kind, file) {
  if (!file) return;
  if (!/\.xer$/i.test(file.name)) {
    showToast("Choose a Primavera P6 .xer file.");
    return;
  }
  assuranceState.files[kind] = { name: file.name, text: await file.text() };
  const label = kind === "previous" ? $("#assurancePreviousName") : $("#assuranceCurrentName");
  label.textContent = file.name;
  label.closest(".assurance-file").classList.add("loaded");
  const ready = Boolean(assuranceState.files.previous && assuranceState.files.current);
  $("#runAssuranceReview").disabled = !ready;
  $("#intakeMessage").textContent = ready ? "Evidence pair ready for a browser-local check." : "Choose the second XER file to continue.";
}

async function runReview() {
  if (!assuranceState.files.previous || !assuranceState.files.current) return;
  const button = $("#runAssuranceReview");
  button.disabled = true;
  button.innerHTML = "Checking evidence…";
  try {
    const previous = parseXer(assuranceState.files.previous.text, assuranceState.files.previous.name);
    const current = parseXer(assuranceState.files.current.text, assuranceState.files.current.name);
    const review = analyseReview(previous, current, $("#assuranceNarrative").value.trim());
    renderReview(review);
    showToast("Readiness check complete. Files remained in this browser.");
  } catch (error) {
    console.error(error);
    showToast(error.message || "The evidence could not be checked.");
  } finally {
    button.disabled = false;
    button.innerHTML = "Check decision readiness <span>→</span>";
  }
}

async function loadDemo() {
  const button = $("#loadAssuranceDemo");
  button.disabled = true;
  button.innerHTML = "Loading example…";
  try {
    const [previousResponse, currentResponse, narrativeResponse] = await Promise.all([
      fetch("demo/northstar-previous.xer"),
      fetch("demo/northstar-current.xer"),
      fetch("demo/northstar-narrative.txt"),
    ]);
    if (![previousResponse, currentResponse, narrativeResponse].every(response => response.ok)) throw new Error("The Northstar evidence could not be loaded.");
    const [previousText, currentText, narrative] = await Promise.all([previousResponse.text(), currentResponse.text(), narrativeResponse.text()]);
    assuranceState.files.previous = { name: "northstar-previous.xer", text: previousText };
    assuranceState.files.current = { name: "northstar-current.xer", text: currentText };
    $("#assuranceNarrative").value = narrative;
    const previous = parseXer(previousText, assuranceState.files.previous.name);
    const current = parseXer(currentText, assuranceState.files.current.name);
    renderReview(analyseReview(previous, current, narrative));
  } catch (error) {
    console.error(error);
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = "Try the Northstar example <span>→</span>";
  }
}

function showView(name) {
  const safeName = ["reviews", "decisions", "follow-up"].includes(name) ? name : "reviews";
  $$("[data-view]").forEach(view => {
    const active = view.dataset.view === safeName;
    view.hidden = !active;
    view.classList.toggle("active", active);
  });
  $$("[data-view-link]").forEach(link => link.classList.toggle("active", link.dataset.viewLink === safeName));
  if (safeName === "decisions") renderDecisionRegister();
  if (safeName === "follow-up") renderConditionRegister();
}

function renderDecisionRegister() {
  const decisions = storage(storageKeys.decisions);
  $("#decisionRegister").innerHTML = decisions.length ? decisions.map(item => `
    <article class="register-row">
      <div><span>${escapeHtml(item.project)} · ${escapeHtml(item.evidenceVersion)}</span><strong>${escapeHtml(item.rationale)}</strong><small>Decision owner: ${escapeHtml(item.owner)} · ${escapeHtml(formatTimestamp(item.createdAt))}</small></div>
      <aside><b>${escapeHtml(item.decision)}</b>${item.unresolved ? `<small>${item.unresolved} unresolved blocker${item.unresolved === 1 ? "" : "s"} preserved</small>` : ""}</aside>
    </article>`).join("") : `<div class="register-empty">No decision has been recorded in this browser yet. Try the Northstar review to create one.</div>`;
}

function renderConditionRegister() {
  const conditions = storage(storageKeys.conditions);
  $("#conditionNavCount").textContent = conditions.filter(item => !item.closed).length;
  $("#conditionRegister").innerHTML = conditions.length ? conditions.map(item => `
    <article class="register-row ${item.closed ? "closed" : ""}" data-condition-id="${escapeHtml(item.id)}">
      <div><span>${escapeHtml(item.project)}</span><strong>${escapeHtml(item.condition)}</strong><small>Owner: ${escapeHtml(item.owner || "Unassigned")} · Due: ${escapeHtml(item.due ? formatDate(`${item.due}T12:00:00Z`) : "Not set")}</small></div>
      <aside><b>${item.closed ? "Closed" : "Open"}</b><button type="button">${item.closed ? "Reopen" : "Mark closed"}</button></aside>
    </article>`).join("") : `<div class="register-empty">No approval conditions are stored in this browser.</div>`;
  $$("[data-condition-id] button").forEach(button => button.addEventListener("click", () => {
    const row = button.closest("[data-condition-id]");
    const all = storage(storageKeys.conditions);
    const condition = all.find(item => item.id === row.dataset.conditionId);
    if (condition) condition.closed = !condition.closed;
    saveStorage(storageKeys.conditions, all);
    renderConditionRegister();
  }));
}

function formatTimestamp(value) {
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function saveEvidenceRequest() {
  const review = assuranceState.review;
  if (!review) return;
  const owner = $("#requestOwner").value.trim();
  const due = $("#requestDue").value;
  if (!owner || !due) {
    showToast("Add an owner and due date before saving the request.");
    return;
  }
  const requests = storage(storageKeys.requests);
  requests.unshift({
    id: `ER-${Date.now()}`,
    project: review.current.project.name,
    owner,
    due,
    questions: review.blockers.slice(0, 3).map(questionFor),
    createdAt: new Date().toISOString(),
  });
  saveStorage(storageKeys.requests, requests);
  $("#readinessStatus").textContent = "Waiting for evidence";
  showToast("Evidence request saved in this browser.");
}

function saveDecision(event) {
  event.preventDefault();
  const review = assuranceState.review;
  if (!review) return;
  const form = new FormData(event.currentTarget);
  const decision = form.get("decision");
  const condition = String(form.get("condition") || "").trim();
  if (decision === "Approve with conditions" && !condition) {
    showToast("Add at least one condition for a conditional approval.");
    $("#decisionCondition").focus();
    return;
  }
  const record = {
    id: `DR-${Date.now()}`,
    project: review.current.project.name,
    evidenceVersion: evidenceVersion(review),
    decision,
    owner: String(form.get("owner") || "").trim(),
    rationale: String(form.get("rationale") || "").trim(),
    unresolved: review.blockers.length,
    createdAt: new Date().toISOString(),
  };
  const decisions = storage(storageKeys.decisions);
  decisions.unshift(record);
  saveStorage(storageKeys.decisions, decisions);

  if (condition) {
    const conditions = storage(storageKeys.conditions);
    conditions.unshift({
      id: `DC-${Date.now()}`,
      decisionId: record.id,
      project: record.project,
      condition,
      owner: String(form.get("conditionOwner") || "").trim(),
      due: String(form.get("conditionDue") || ""),
      closed: false,
    });
    saveStorage(storageKeys.conditions, conditions);
  }
  renderConditionRegister();
  window.location.hash = "decisions";
  showToast("Decision record saved with its evidence version.");
  event.currentTarget.reset();
}

function resetReview() {
  assuranceState.files = { previous: null, current: null };
  assuranceState.review = null;
  $("#assurancePrevious").value = "";
  $("#assuranceCurrent").value = "";
  $("#assuranceNarrative").value = "";
  $("#assurancePreviousName").textContent = "Choose comparison point";
  $("#assuranceCurrentName").textContent = "Choose latest submission";
  $$(".assurance-file").forEach(card => card.classList.remove("loaded"));
  $("#runAssuranceReview").disabled = true;
  $("#intakeMessage").textContent = "Choose 2 XER files to begin.";
  $("#readinessWorkspace").hidden = true;
  $("#packIntake").hidden = false;
  $("#packIntake").scrollIntoView({ behavior: "smooth" });
}

function bindEvents() {
  $("#loadAssuranceDemo").addEventListener("click", loadDemo);
  $("#showOwnPack").addEventListener("click", () => {
    $("#packIntake").hidden = false;
    $("#packIntake").scrollIntoView({ behavior: "smooth" });
  });
  $("#assurancePrevious").addEventListener("change", event => loadFile("previous", event.target.files[0]));
  $("#assuranceCurrent").addEventListener("change", event => loadFile("current", event.target.files[0]));
  $("#runAssuranceReview").addEventListener("click", runReview);
  $("#startAnotherReview").addEventListener("click", resetReview);
  $("#requestAnswers").addEventListener("click", () => {
    $("#requestPanel").hidden = false;
    $("#requestPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("#showDecisionForm").addEventListener("click", () => {
    $("#decisionPanel").hidden = false;
    $("#decisionPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("#saveEvidenceRequest").addEventListener("click", saveEvidenceRequest);
  $("#decisionForm").addEventListener("submit", saveDecision);
  window.addEventListener("hashchange", () => showView(window.location.hash.slice(1)));
}

bindEvents();
renderConditionRegister();
showView(window.location.hash.slice(1) || "reviews");
window.ProjectLensChangeAssurance = { parseXer, analyseReview };
