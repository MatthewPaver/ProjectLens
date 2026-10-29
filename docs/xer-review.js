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


// Engine bindings: schedule parsing and review live in ProjectLensEngine.
const parseXer = (text, fileName, options) => ProjectLensEngine.parseXer(text, fileName, options);
const parseEvidenceText = (...args) => ProjectLensEngine.parseEvidenceText(...args);
const compareSchedules = (...args) => ProjectLensEngine.compareSchedules(...args);
const buildReview = (...args) => ProjectLensEngine.buildReview(...args);
const assessCompleteness = (...args) => ProjectLensEngine.assessCompleteness(...args);
const pathCandidates = (...args) => ProjectLensEngine.pathCandidates(...args);
const reconcileDecisions = (...args) => ProjectLensEngine.reconcileDecisions(...args);
const compareBaseline = (...args) => ProjectLensEngine.compareBaseline(...args);
const numberValue = (...args) => ProjectLensEngine.numberValue(...args);
const parseDate = (...args) => ProjectLensEngine.parseDate(...args);
const formatDate = (...args) => ProjectLensEngine.formatDate(...args);
const dayDifference = (...args) => ProjectLensEngine.dayDifference(...args);
const isoDate = (...args) => ProjectLensEngine.isoDate(...args);

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
  const decoded = ProjectLensEngine.decodeXerBytes(new Uint8Array(await file.arrayBuffer()));
  reviewState.files[kind] = { name: file.name, size: file.size, text: decoded.text, encoding: decoded.encoding, projectId: null };
  updateFileCard(kind);
}

function parseReviewFile(file) {
  const projects = ProjectLensEngine.listProjects(file.text);
  let projectId = file.projectId;
  if (projects.length > 1 && !projectId) {
    const listing = projects.map((p, i) => `${i + 1}. ${p.shortName} (${p.projId}, ${p.activityCount} activities)`).join("\n");
    const choice = window.prompt(`This file contains ${projects.length} projects. Enter the number to use:\n\n${listing}`, "1");
    if (!choice) throw new Error("Choose a project in the multi-project export before comparing.");
    const index = Number(choice) - 1;
    if (!projects[index]) throw new Error("That project number is not in the list.");
    projectId = projects[index].projId;
    file.projectId = projectId;
  }
  return parseXer(file.text, file.name, projectId ? { projectId } : {});
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
    const previous = parseReviewFile(reviewState.files.previous);
    const current = parseReviewFile(reviewState.files.current);
    const baseline = reviewState.files.baseline ? parseReviewFile(reviewState.files.baseline) : null;
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
