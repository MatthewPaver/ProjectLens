const PROJECTS = {
  northstar: {
    name: "Northstar Rail Upgrade",
    short: "Northstar",
    target: "2026-09-30",
    committedDelay: 6,
    contingency: 18,
    briefing: "Northstar remains outside the September commitment. Civil handover is the dominant driver, but resequencing commissioning can recover more time per unit of effort than adding another shift.",
    drivers: [
      { id: "handover", name: "Civil handover", category: "Delivery", probability: .82, min: 5, mode: 12, max: 22, criticality: 1, evidence: "6 critical activities moved in 3 consecutive updates" },
      { id: "approval", name: "Design approvals", category: "Governance", probability: .66, min: 2, mode: 8, max: 16, criticality: .85, evidence: "Median approval cycle increased from 8 to 13 days" },
      { id: "testing", name: "Testing capacity", category: "Resource", probability: .58, min: 3, mode: 7, max: 14, criticality: .9, evidence: "Two specialist teams cover three parallel workfronts" },
      { id: "supplier", name: "Supplier recovery", category: "Supply", probability: .42, min: 2, mode: 6, max: 12, criticality: .8, evidence: "Package NS-44 missed two promised delivery dates" }
    ]
  },
  thames: {
    name: "Thames Energy Transition",
    short: "Thames",
    target: "2026-11-14",
    committedDelay: 4,
    contingency: 15,
    briefing: "Grid consent now drives the November commitment. A focused approval sprint provides the strongest confidence uplift, while additional field capacity has limited value until access is released.",
    drivers: [
      { id: "handover", name: "Site access release", category: "Delivery", probability: .61, min: 3, mode: 8, max: 16, criticality: .9, evidence: "Access hold remains on 4 of 11 work areas" },
      { id: "approval", name: "Grid consent", category: "Governance", probability: .84, min: 5, mode: 11, max: 20, criticality: 1, evidence: "Two consent responses remain outside the agreed SLA" },
      { id: "testing", name: "Commissioning crew", category: "Resource", probability: .44, min: 2, mode: 6, max: 12, criticality: .75, evidence: "Crew availability is protected from 3 November" },
      { id: "supplier", name: "Switchgear delivery", category: "Supply", probability: .52, min: 3, mode: 7, max: 14, criticality: .85, evidence: "Factory acceptance test moved by 5 working days" }
    ]
  },
  meridian: {
    name: "Meridian Systems Integration",
    short: "Meridian",
    target: "2026-12-18",
    committedDelay: 8,
    contingency: 20,
    briefing: "Interface testing is now the principal completion risk. Resequencing enables earlier defect discovery, but the decision depends on confirming the supplier environment by Friday.",
    drivers: [
      { id: "handover", name: "Platform readiness", category: "Delivery", probability: .64, min: 4, mode: 9, max: 18, criticality: .92, evidence: "Readiness burn-down flattened across the last two sprints" },
      { id: "approval", name: "Security approval", category: "Governance", probability: .57, min: 3, mode: 7, max: 15, criticality: .88, evidence: "Evidence pack has 5 unresolved review comments" },
      { id: "testing", name: "Interface testing", category: "Resource", probability: .79, min: 5, mode: 12, max: 23, criticality: 1, evidence: "Defect arrival rate exceeds closure rate by 1.6×" },
      { id: "supplier", name: "Supplier environment", category: "Supply", probability: .49, min: 2, mode: 8, max: 17, criticality: .84, evidence: "Environment refresh remains uncommitted" }
    ]
  }
};

const INTERVENTIONS = [
  { id: "resequence", name: "Resequence commissioning", description: "Decouple testing preparation from final handover and expose defects earlier.", owner: "Planning lead", effort: 2, cost: "Low", reductions: { handover: .45, testing: .2 } },
  { id: "approval-sprint", name: "Run an approval sprint", description: "Create a five-day decision room with named approvers and a daily ageing review.", owner: "Design authority", effort: 2, cost: "Low", reductions: { approval: .55, supplier: .1 } },
  { id: "night-shift", name: "Add specialist capacity", description: "Fund a protected second shift for four weeks around the constrained workfront.", owner: "Delivery director", effort: 4, cost: "High", reductions: { testing: .65, handover: .15 } }
];

const EVIDENCE = [
  ["NS-142 Civil handover", "Delivery lead", "Repeated slippage", "+12 days", "High", "18 Jul"],
  ["NS-168 System proving", "Test manager", "Resource contention", "+7 days", "Medium", "17 Jul"],
  ["NS-091 Design release", "Design authority", "Cycle-time shift", "+8 days", "High", "16 Jul"],
  ["NS-204 Supplier package", "Commercial lead", "Promise missed", "+6 days", "Medium", "19 Jul"],
  ["NS-177 Entry into service", "Programme director", "Milestone pressure", "+18 days", "High", "22 Jul"],
  ["NS-119 Access readiness", "Construction lead", "Sparse history", "+4 days", "Low", "18 Jul"]
];

let activeProjectKey = "northstar";
let activeScenario = "resequence";
const DATASET_DEFINITIONS = [
  { id: "task_cleaned", file: "task_cleaned.csv", label: "Task history", required: ["task_id", "task_name", "update_phase", "slip_days"] },
  { id: "slippage_summary", file: "slippage_summary.csv", label: "Slippage history", required: ["task_id", "task_name", "update_phase", "slip_days", "severity_score"] },
  { id: "milestone_analysis", file: "milestone_analysis.csv", label: "Milestones", required: ["task_id", "task_name", "slip_days", "severity_score"] },
  { id: "forecast_results", file: "forecast_results.csv", label: "Forecasts", required: ["task_id", "task_name", "forecast_confidence", "predicted_end_date"] },
  { id: "changepoints", file: "changepoints.csv", label: "Change points", required: ["task_id", "task_name", "change_type", "severity_score"] },
  { id: "recommendations", file: "recommendations.csv", label: "Recommendations", required: ["task_id", "task_name", "recommendation", "severity", "confidence"] },
  { id: "model_evaluation", file: "model_evaluation.csv", label: "Model scorecard", required: ["model_type", "mae_days", "bias_days", "validation_predictions", "champion"] },
  { id: "action_briefs", file: "action_briefs.csv", label: "Action briefs", required: ["task_id", "priority", "decision_question", "recommended_action", "owner_role", "evidence"] }
];
const DEMO_DATA_BASE = window.location.pathname.includes("/docs/") ? "../Data/output/Alpha/" : "./demo-data/alpha/";
const PAGE_SIZE = 50;
let datasets = {};
let activeDataset = "task_cleaned";
let explorerPage = 1;
let dataSearch = "";
let currentWatchlist = [];
let guideStep = 0;
let decisions = [
  { id: "D-014", name: "Protect test window", owner: "Test manager", promised: 6, observed: 5, review: "Reviewed 08 Jul", status: "68% realised", note: "Second shift protected the defect closure rate." },
  { id: "D-015", name: "Escalate design release", owner: "Design authority", promised: 4, observed: 3, review: "Reviewed 11 Jul", status: "75% realised", note: "Two decisions cleared; one interface remains open." },
  { id: "D-016", name: "Resequence commissioning", owner: "Planning lead", promised: 8, observed: null, review: "Review due 21 Jul", status: "Committed", note: "Preparation starts before full civil handover." },
  { id: "D-017", name: "Supplier recovery room", owner: "Commercial lead", promised: 3, observed: null, review: "Overdue 12 Jul", status: "Evidence due", note: "Awaiting confirmed factory dispatch evidence." }
];

function mulberry32(seed) {
  return function random() {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function triangular(random, min, mode, max) {
  const u = random();
  const split = (mode - min) / (max - min);
  return u < split
    ? min + Math.sqrt(u * (max - min) * (mode - min))
    : max - Math.sqrt((1 - u) * (max - min) * (max - mode));
}

function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * p;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[Math.min(lower + 1, sorted.length - 1)] * fraction;
}

function simulate(reductions = {}, seed = 2026) {
  const project = PROJECTS[activeProjectKey];
  const random = mulberry32(seed);
  const outcomes = [];
  for (let i = 0; i < 10000; i += 1) {
    let delay = project.committedDelay;
    project.drivers.forEach(driver => {
      if (random() <= driver.probability) {
        delay += triangular(random, driver.min, driver.mode, driver.max)
          * driver.criticality * (1 - (reductions[driver.id] || 0));
      }
    });
    outcomes.push(Math.max(0, delay));
  }
  return {
    p10: percentile(outcomes, .1), p50: percentile(outcomes, .5),
    p80: percentile(outcomes, .8), p90: percentile(outcomes, .9),
    onTime: outcomes.filter(value => value <= project.contingency).length / outcomes.length * 100
  };
}

function scenarioResults() {
  const baseline = simulate();
  const scenarios = INTERVENTIONS.map(item => {
    const result = simulate(item.reductions);
    const recovery = baseline.p80 - result.p80;
    const uplift = result.onTime - baseline.onTime;
    return { ...item, result, recovery, uplift, value: (recovery + uplift / 4) / item.effort };
  }).sort((a, b) => b.value - a.value);
  return { baseline, scenarios };
}

function expectedDrivers() {
  return PROJECTS[activeProjectKey].drivers.map(driver => ({
    ...driver, expected: driver.probability * driver.mode * driver.criticality
  })).sort((a, b) => b.expected - a.expected);
}

function addDays(dateString, days) {
  const date = new Date(`${dateString}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + Math.round(days));
  return date;
}

function shortDate(date) {
  return new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", timeZone: "UTC" }).format(date).toUpperCase();
}

function renderExecutive() {
  const project = PROJECTS[activeProjectKey];
  const { baseline, scenarios } = scenarioResults();
  const selected = scenarios.find(item => item.id === activeScenario) || scenarios[0];
  activeScenario = selected.id;
  document.querySelector('[data-view="executive"] .eyebrow span').textContent = `Portfolio / ${project.short}`;
  document.getElementById("briefingText").textContent = project.briefing;
  document.getElementById("onTimeMetric").innerHTML = `${Math.round(baseline.onTime)}<small>%</small>`;
  document.getElementById("p80DateMetric").innerHTML = `${shortDate(addDays(project.target, baseline.p80)).replace(" ", "<small> ")} </small>`;
  document.getElementById("p80Caption").textContent = `${Math.round(baseline.p80)} days beyond commitment`;
  document.getElementById("recoveryMetric").innerHTML = `${Math.round(Math.max(...scenarios.map(item => item.recovery)))}<small> DAYS</small>`;
  document.getElementById("p50Days").textContent = `+${Math.round(baseline.p50)} days`;
  document.getElementById("p80Days").textContent = `+${Math.round(baseline.p80)} days`;
  document.getElementById("targetChance").textContent = `${Math.round(baseline.onTime)}%`;
  document.getElementById("decisionConfidence").textContent = baseline.onTime > 55 ? "HIGH" : baseline.onTime > 25 ? "MEDIUM" : "LOW";
  renderDrivers();
  renderScenarios(scenarios);
  updateDistribution(selected, baseline);
}

function renderDrivers() {
  const drivers = expectedDrivers();
  const max = Math.max(...drivers.map(item => item.expected));
  document.getElementById("driverList").innerHTML = drivers.map((driver, index) => `
    <div class="driver">
      <span class="driver-index">0${index + 1}</span>
      <div class="driver-main"><strong>${driver.name}</strong><small>${driver.evidence}</small><div class="driver-bar"><i style="--width:${driver.expected / max * 100}%"></i></div></div>
      <div class="driver-value"><strong>${driver.expected.toFixed(1)}d</strong><small>expected</small></div>
    </div>`).join("");
  document.getElementById("tornadoChart").innerHTML = drivers.map(driver => `
    <div class="tornado-row"><span>${driver.name}</span><div class="tornado-bar"><i style="--width:${driver.expected / max * 100}%"></i></div><b>${driver.expected.toFixed(1)}d</b></div>`).join("");
}

function renderScenarios(scenarios) {
  document.getElementById("scenarioCards").innerHTML = scenarios.map((item, index) => `
    <article class="scenario-card ${index === 0 ? "recommended" : ""}" data-rank="0${index + 1}">
      <div class="scenario-tag"><span>Option 0${index + 1}</span>${index === 0 ? "<b>Best value</b>" : `<span>${item.cost} cost</span>`}</div>
      <h3>${item.name}</h3><p>${item.description}</p>
      <div class="scenario-impact"><div><span>P80 recovery</span><b>${item.recovery.toFixed(1)} days</b></div><div><span>Confidence</span><b>+${item.uplift.toFixed(0)} pts</b></div></div>
      <div class="scenario-foot"><span>Owner: ${item.owner}</span><span>Effort ${item.effort}/5</span></div>
      <button type="button" data-scenario="${item.id}">Compare and decide</button>
    </article>`).join("");
  document.querySelectorAll("[data-scenario]").forEach(button => button.addEventListener("click", () => {
    activeScenario = button.dataset.scenario;
    const scenario = INTERVENTIONS.find(item => item.id === activeScenario);
    populateLab(scenario);
    openDialog(document.getElementById("scenarioDialog"));
  }));
}

function updateDistribution(selected, baseline) {
  document.getElementById("scenarioCurve").style.opacity = "0";
  requestAnimationFrame(() => { document.getElementById("scenarioCurve").style.opacity = "1"; });
  document.getElementById("targetChance").textContent = `${Math.round(baseline.onTime)}%`;
}

function renderEvidence() {
  const query = document.getElementById("evidenceFilter").value.toLowerCase();
  const rows = EVIDENCE.filter(row => row.join(" ").toLowerCase().includes(query));
  document.getElementById("evidenceTable").innerHTML = rows.map(row => `<tr><td><strong>${row[0]}</strong></td><td>${row[1]}</td><td><span class="signal-pill">${row[2]}</span></td><td>${row[3]}</td><td>${row[4]}</td><td>${row[5]}</td></tr>`).join("");
}

function renderDecisions() {
  document.getElementById("decisionLedger").innerHTML = decisions.map(item => `
    <article class="decision-row">
      <span class="decision-id">${item.id}</span>
      <div><strong>${item.name}</strong><p>${item.note}</p></div>
      <div><small>Owner</small><strong>${item.owner}</strong></div>
      <div class="decision-metric"><small>Promised</small><b>${item.promised} days</b></div>
      <div class="decision-metric"><small>Observed</small><b>${item.observed === null ? "Pending" : `${item.observed} days`}</b></div>
      <span class="decision-status ${item.observed === null ? "pending" : ""}">${item.status}</span>
    </article>`).join("");
  const open = decisions.filter(item => item.observed === null).length;
  document.getElementById("decisionCount").textContent = open;
  document.getElementById("openCommitments").textContent = String(open).padStart(2, "0");
}

function switchView(name) {
  document.querySelectorAll(".view").forEach(view => {
    const active = view.dataset.view === name;
    view.hidden = !active;
    view.classList.toggle("active", active);
  });
  document.querySelectorAll(".rail-item").forEach(item => item.classList.toggle("active", item.dataset.viewTarget === name));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function openDialog(dialog) {
  if (typeof dialog.showModal === "function") dialog.showModal();
}

function populateLab(scenario = INTERVENTIONS[0]) {
  const fields = { handover: "handoverRange", approval: "approvalRange", testing: "testingRange", supplier: "supplierRange" };
  Object.entries(fields).forEach(([key, id]) => {
    document.getElementById(id).value = Math.round((scenario.reductions[key] || 0) * 100);
  });
  document.getElementById("scenarioName").value = scenario.name;
  document.getElementById("scenarioOwner").value = scenario.owner;
  updateLab();
}

function updateLab() {
  const reductions = {};
  ["handover", "approval", "testing", "supplier"].forEach(key => {
    const value = Number(document.getElementById(`${key}Range`).value);
    reductions[key] = value / 100;
    document.getElementById(`${key}Output`).textContent = `${value}%`;
  });
  const baseline = simulate();
  const result = simulate(reductions);
  const recovery = Math.max(0, baseline.p80 - result.p80);
  document.getElementById("labProbability").textContent = `${Math.round(result.onTime)}%`;
  document.getElementById("labBaseP80").textContent = `+${Math.round(baseline.p80)} days`;
  document.getElementById("labScenarioP80").textContent = `+${Math.round(result.p80)} days`;
  document.getElementById("labRecovery").textContent = `${recovery.toFixed(1)} days`;
  const strongest = Object.entries(reductions).sort((a, b) => b[1] - a[1])[0][0];
  const driver = PROJECTS[activeProjectKey].drivers.find(item => item.id === strongest);
  document.getElementById("labNarrative").textContent = `This case is most sensitive to the assumed ${Math.round(reductions[strongest] * 100)}% reduction in ${driver.name.toLowerCase()}. Confirm the operational mechanism before committing.`;
}

function commitLabScenario() {
  const baseline = simulate();
  const reductions = Object.fromEntries(["handover", "approval", "testing", "supplier"].map(key => [key, Number(document.getElementById(`${key}Range`).value) / 100]));
  const result = simulate(reductions);
  const id = `D-${String(14 + decisions.length).padStart(3, "0")}`;
  decisions.push({
    id, name: document.getElementById("scenarioName").value || "Custom recovery scenario",
    owner: document.getElementById("scenarioOwner").value || "Unassigned",
    promised: Math.max(0, Math.round(baseline.p80 - result.p80)), observed: null,
    review: "Review due next update", status: "Committed", note: "Custom scenario committed from the ProjectLens lab. Assumptions preserved for review."
  });
  renderDecisions();
  document.getElementById("scenarioDialog").close();
  showToast(`${id} added to the decision ledger with its assumptions and expected recovery.`);
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 4200);
}

function exportBrief() {
  const project = PROJECTS[activeProjectKey];
  const { baseline, scenarios } = scenarioResults();
  const content = `# ${project.name} decision brief\n\nGenerated 14 July 2026\n\n## Position\n\n${project.briefing}\n\n- On-time confidence: ${baseline.onTime.toFixed(1)}%\n- P50 delay: ${baseline.p50.toFixed(1)} days\n- P80 delay: ${baseline.p80.toFixed(1)} days\n\n## Options\n\n${scenarios.map(item => `- ${item.name}: ${item.recovery.toFixed(1)} days P80 recovery, +${item.uplift.toFixed(1)} confidence points, effort ${item.effort}/5`).join("\n")}\n\nSynthetic demonstration data. Review assumptions before use.`;
  const url = URL.createObjectURL(new Blob([content], { type: "text/markdown" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: `${project.short.toLowerCase()}-decision-brief.md` });
  link.click();
  URL.revokeObjectURL(url);
  showToast("Decision brief exported with model boundaries and scenario assumptions.");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') { field += '"'; index += 1; }
      else if (character === '"') quoted = false;
      else field += character;
    } else if (character === '"') quoted = true;
    else if (character === ",") { row.push(field); field = ""; }
    else if (character === "\n") { row.push(field); if (row.some(cell => cell.trim() !== "")) rows.push(row); row = []; field = ""; }
    else if (character !== "\r") field += character;
  }
  row.push(field);
  if (row.some(cell => cell.trim() !== "")) rows.push(row);
  if (rows.length < 2) return [];
  const headers = rows.shift().map(header => header.trim().replace(/^\uFEFF/, ""));
  return rows.map(values => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function numeric(value) {
  const result = Number(value);
  return Number.isFinite(result) ? result : 0;
}

function datasetDefinition(id) {
  return DATASET_DEFINITIONS.find(item => item.id === id);
}

function validateDataset(definition, rows) {
  if (!rows?.length) return { ready: false, message: "Not loaded" };
  const columns = Object.keys(rows[0]);
  const missing = definition.required.filter(column => !columns.includes(column));
  return missing.length ? { ready: false, message: `Missing: ${missing.join(", ")}` } : { ready: true, message: `${columns.length} columns recognised` };
}

function setDatasets(nextDatasets, source, navigate = true) {
  datasets = nextDatasets;
  activeDataset = DATASET_DEFINITIONS.find(item => datasets[item.id]?.length)?.id || "task_cleaned";
  explorerPage = 1;
  dataSearch = "";
  document.getElementById("dataSearch").value = "";
  document.getElementById("dataSourceLabel").textContent = source;
  renderDataRoom();
  if (navigate) switchView("data");
}

async function loadDemoDatasets(navigate = true) {
  try {
    const entries = await Promise.all(DATASET_DEFINITIONS.map(async definition => {
      const response = await fetch(`${DEMO_DATA_BASE}${definition.file}`);
      if (!response.ok) throw new Error(`${definition.file} could not be loaded`);
      return [definition.id, parseCsv(await response.text())];
    }));
    const nextDatasets = Object.fromEntries(entries);
    const total = Object.values(nextDatasets).reduce((sum, rows) => sum + rows.length, 0);
    setDatasets(nextDatasets, `Public Alpha demo · ${total.toLocaleString("en-GB")} rows`, navigate);
    if (document.getElementById("importDialog").open) document.getElementById("importDialog").close();
    if (navigate) showToast("Complete Alpha evidence set loaded locally in your browser.");
  } catch (error) {
    showToast(`Demo data unavailable: ${error.message}`);
  }
}

async function importFiles(fileList) {
  const nextDatasets = {};
  const ignored = [];
  for (const file of Array.from(fileList)) {
    const id = file.name.toLowerCase().replace(/\.csv$/, "");
    if (!datasetDefinition(id)) { ignored.push(file.name); continue; }
    nextDatasets[id] = parseCsv(await file.text());
  }
  if (!Object.keys(nextDatasets).length) {
    showToast("No recognised ProjectLens output files were selected.");
    return;
  }
  const total = Object.values(nextDatasets).reduce((sum, rows) => sum + rows.length, 0);
  setDatasets(nextDatasets, `Your local files · ${total.toLocaleString("en-GB")} rows`);
  document.getElementById("importDialog").close();
  showToast(ignored.length ? `${total.toLocaleString("en-GB")} rows loaded. ${ignored.length} unrecognised file(s) ignored.` : `${total.toLocaleString("en-GB")} rows loaded. Nothing left your browser.`);
}

function renderDatasetCards() {
  document.getElementById("datasetCards").innerHTML = DATASET_DEFINITIONS.map(definition => {
    const rows = datasets[definition.id] || [];
    const validation = validateDataset(definition, rows);
    return `<article class="dataset-card ${validation.ready ? "ready" : rows.length ? "issue" : ""}"><span>${escapeHtml(definition.file)}</span><strong>${escapeHtml(definition.label)}</strong><small>${escapeHtml(validation.message)}</small><b>${rows.length.toLocaleString("en-GB")} rows</b></article>`;
  }).join("");
}

function renderQuality() {
  const loaded = DATASET_DEFINITIONS.filter(item => datasets[item.id]?.length);
  const valid = loaded.filter(item => validateDataset(item, datasets[item.id]).ready);
  const completeFields = loaded.reduce((sum, definition) => {
    const rows = datasets[definition.id];
    const checks = rows.length * definition.required.length;
    const present = rows.reduce((rowSum, row) => rowSum + definition.required.filter(column => String(row[column] ?? "").trim() !== "").length, 0);
    return sum + (checks ? present / checks : 0);
  }, 0);
  const completeness = loaded.length ? completeFields / loaded.length : 0;
  const score = Math.round((valid.length / DATASET_DEFINITIONS.length * .65 + completeness * .35) * 100);
  document.getElementById("qualityScore").textContent = `${score} / 100`;
  const checks = [
    ["Recognised datasets", `${loaded.length} of ${DATASET_DEFINITIONS.length}`, loaded.length >= 2],
    ["Required columns", `${valid.length} valid`, valid.length === loaded.length && loaded.length > 0],
    ["Required-field completeness", `${Math.round(completeness * 100)}%`, completeness >= .95],
    ["Period comparison available", datasets.slippage_summary?.length ? "Ready" : "Unavailable", Boolean(datasets.slippage_summary?.length)]
  ];
  document.getElementById("qualityChecks").innerHTML = checks.map(([label, value, pass]) => `<div class="quality-check"><span>${label}</span><b class="${pass ? "" : "fail"}">${value}</b></div>`).join("");
}

function buildWatchlist() {
  const signals = [];
  const add = (severity, title, detail, source, taskId = "") => signals.push({ severity, title, detail, source, taskId });
  (datasets.action_briefs || []).filter(row => row.priority === "P1").forEach(row => add("High", row.decision_question, `${row.what_changed} ${row.recommended_action}`, "action_briefs.csv", row.task_id));
  (datasets.recommendations || []).filter(row => String(row.severity).toLowerCase() === "high").forEach(row => add("High", row.task_name, row.recommendation, "recommendations.csv", row.task_id));
  (datasets.forecast_results || []).filter(row => numeric(row.forecast_confidence) < .7 || String(row.low_confidence_flag).toLowerCase() === "true").forEach(row => add("High", row.task_name, `Forecast confidence ${Math.round(numeric(row.forecast_confidence) * 100)}%. Predicted end ${row.predicted_end_date || "not supplied"}.`, "forecast_results.csv", row.task_id));
  (datasets.changepoints || []).forEach(row => add(numeric(row.severity_score) >= 8 ? "High" : "Medium", row.task_name, `${row.change_type || "Schedule change"} detected in ${row.update_phase || "the latest update"}.`, "changepoints.csv", row.task_id));
  (datasets.milestone_analysis || []).filter(row => numeric(row.slip_days) > 0 && numeric(row.severity_score) >= 7).forEach(row => add("High", row.task_name, `Milestone is ${numeric(row.slip_days).toFixed(0)} days late with severity ${numeric(row.severity_score).toFixed(1)}.`, "milestone_analysis.csv", row.task_id));
  const latestCritical = new Map();
  (datasets.task_cleaned || []).filter(row => String(row.is_critical).toLowerCase() === "true" && numeric(row.severity_score) >= 8).forEach(row => latestCritical.set(row.task_id, row));
  latestCritical.forEach(row => add("High", row.task_name, `Critical task is ${numeric(row.slip_days).toFixed(0)} days late in ${row.update_phase}.`, "task_cleaned.csv", row.task_id));
  const unique = new Map();
  signals.forEach(signal => unique.set(`${signal.source}:${signal.taskId}:${signal.detail}`, signal));
  return [...unique.values()].sort((a, b) => (a.severity === "High" ? -1 : 1) - (b.severity === "High" ? -1 : 1));
}

function renderWatchlist() {
  currentWatchlist = buildWatchlist();
  const container = document.getElementById("watchlist");
  if (!currentWatchlist.length) { container.className = "watchlist empty-state"; container.textContent = Object.keys(datasets).length ? "No configured thresholds were triggered by this data." : "Load data to generate an evidence-linked watchlist."; }
  else {
    container.className = "watchlist";
    container.innerHTML = currentWatchlist.slice(0, 40).map(signal => `<div class="signal-item"><span class="signal-severity ${signal.severity.toLowerCase()}">${signal.severity}</span><div><strong>${escapeHtml(signal.title)}</strong><p>${escapeHtml(signal.detail)}</p><small>${escapeHtml(signal.taskId)}</small></div><code>${escapeHtml(signal.source)}</code></div>`).join("");
  }
  renderWatchPreview();
}

function renderWatchPreview() {
  const fallback = [
    { priority: "P1", decision_question: "Which owned intervention protects civil handover?", what_changed: "Six critical activities moved across three updates.", recommended_action: "Bring one costed recovery option to the Friday control gate.", owner_role: "Construction lead", decision_by: "Before Friday" },
    { priority: "P1", decision_question: "Do we recover the design release or reforecast it?", what_changed: "Approval cycle time widened from 8 to 13 days.", recommended_action: "Run a five-day approval sprint with named approvers.", owner_role: "Design authority", decision_by: "Before Friday" },
    { priority: "P2", decision_question: "Is supplier dispatch evidence strong enough?", what_changed: "The promised evidence is overdue.", recommended_action: "Confirm dispatch proof before protecting the downstream date.", owner_role: "Commercial lead", decision_by: "Today" }
  ];
  const briefs = datasets.action_briefs?.length ? [...datasets.action_briefs].sort((a, b) => numeric(b.priority_score) - numeric(a.priority_score)).slice(0, 3) : fallback;
  document.getElementById("watchPreview").innerHTML = briefs.map(brief => `<article class="preview-signal ${brief.priority === "P1" ? "high" : "medium"}"><small>${escapeHtml(brief.priority)} · decision required</small><strong>${escapeHtml(brief.decision_question)}</strong><p><b>Changed:</b> ${escapeHtml(brief.what_changed)}</p><p><b>Do:</b> ${escapeHtml(brief.recommended_action)}</p><footer>${escapeHtml(brief.owner_role)} · ${escapeHtml(brief.decision_by)}</footer></article>`).join("");
}

function renderModelGovernance() {
  const rows = datasets.model_evaluation || [];
  const champion = rows.find(row => String(row.champion).toLowerCase() === "true") || rows[0];
  if (!champion) return;
  document.getElementById("modelChampion").textContent = champion.model_type;
  document.getElementById("modelMae").textContent = `${numeric(champion.mae_days).toFixed(2)} days MAE`;
  document.getElementById("modelValidation").textContent = `${numeric(champion.validation_predictions).toLocaleString("en-GB")} past predictions tested`;
  document.getElementById("modelScorecard").innerHTML = rows.map(row => `<div class="model-row ${String(row.champion).toLowerCase() === "true" ? "winner" : ""}"><span>${escapeHtml(row.model_type)}</span><b>${numeric(row.mae_days).toFixed(2)}d MAE</b><small>${numeric(row.interval_coverage_80).toFixed(1)}% interval coverage</small></div>`).join("");
}

function phaseNumber(value) {
  const match = String(value).match(/(\d+)/);
  return match ? Number(match[1]) : 0;
}

function preparePhaseControls() {
  const rows = datasets.slippage_summary || [];
  const phases = [...new Set(rows.map(row => row.update_phase).filter(Boolean))].sort((a, b) => phaseNumber(a) - phaseNumber(b));
  ["phaseFrom", "phaseTo"].forEach(id => {
    const select = document.getElementById(id);
    select.innerHTML = phases.map(phase => `<option value="${escapeHtml(phase)}">${escapeHtml(phase)}</option>`).join("");
    select.disabled = phases.length < 2;
  });
  if (phases.length >= 2) {
    document.getElementById("phaseFrom").value = phases[Math.max(0, phases.length - 2)];
    document.getElementById("phaseTo").value = phases[phases.length - 1];
  }
  renderUpdateComparison();
}

function renderUpdateComparison() {
  const rows = datasets.slippage_summary || [];
  const from = document.getElementById("phaseFrom").value;
  const to = document.getElementById("phaseTo").value;
  const container = document.getElementById("updateComparison");
  if (!rows.length || !from || !to || from === to) { container.className = "update-comparison empty-state"; container.textContent = "Load slippage history and choose two different reporting periods."; return; }
  const before = new Map(rows.filter(row => row.update_phase === from).map(row => [row.task_id, numeric(row.slip_days)]));
  const after = new Map(rows.filter(row => row.update_phase === to).map(row => [row.task_id, numeric(row.slip_days)]));
  const deltas = [...after].filter(([id]) => before.has(id)).map(([id, value]) => ({ id, delta: value - before.get(id) }));
  const worsened = deltas.filter(item => item.delta > .5).length;
  const improved = deltas.filter(item => item.delta < -.5).length;
  const unchanged = deltas.length - worsened - improved;
  const mean = deltas.length ? deltas.reduce((sum, item) => sum + item.delta, 0) / deltas.length : 0;
  container.className = "update-comparison";
  container.innerHTML = `<div class="comparison-stat negative"><span>Worsened</span><strong>${worsened}</strong><small>tasks gained slippage</small></div><div class="comparison-stat positive"><span>Improved</span><strong>${improved}</strong><small>tasks recovered time</small></div><div class="comparison-stat"><span>Unchanged</span><strong>${unchanged}</strong><small>within half a day</small></div><div class="comparison-stat ${mean > 0 ? "negative" : "positive"}"><span>Mean movement</span><strong>${mean > 0 ? "+" : ""}${mean.toFixed(1)}d</strong><small>${from} to ${to}</small></div>`;
}

function renderExplorer() {
  document.getElementById("datasetTabs").innerHTML = DATASET_DEFINITIONS.map(definition => `<button type="button" data-dataset="${definition.id}" class="${definition.id === activeDataset ? "active" : ""}" ${datasets[definition.id]?.length ? "" : "disabled"}>${escapeHtml(definition.label)} · ${(datasets[definition.id] || []).length.toLocaleString("en-GB")}</button>`).join("");
  document.querySelectorAll("[data-dataset]").forEach(button => button.addEventListener("click", () => { activeDataset = button.dataset.dataset; explorerPage = 1; renderExplorer(); }));
  const allRows = datasets[activeDataset] || [];
  const filtered = dataSearch ? allRows.filter(row => Object.values(row).some(value => String(value).toLowerCase().includes(dataSearch))) : allRows;
  const pageCount = Math.ceil(filtered.length / PAGE_SIZE);
  explorerPage = Math.min(Math.max(explorerPage, 1), Math.max(pageCount, 1));
  const rows = filtered.slice((explorerPage - 1) * PAGE_SIZE, explorerPage * PAGE_SIZE);
  document.getElementById("explorerMeta").textContent = `${datasetDefinition(activeDataset)?.label || "Dataset"} · ${filtered.length.toLocaleString("en-GB")} of ${allRows.length.toLocaleString("en-GB")} rows`;
  document.getElementById("explorerPage").textContent = `Page ${pageCount ? explorerPage : 0} of ${pageCount}`;
  document.getElementById("dataSearch").disabled = !allRows.length;
  document.getElementById("prevPage").disabled = explorerPage <= 1 || !pageCount;
  document.getElementById("nextPage").disabled = explorerPage >= pageCount || !pageCount;
  const table = document.getElementById("dataTable");
  if (!rows.length) { table.innerHTML = `<tbody><tr><td class="empty-state">${allRows.length ? "No rows match this search." : "Choose the full demo or import ProjectLens output CSV files."}</td></tr></tbody>`; return; }
  const columns = Object.keys(rows[0]);
  table.innerHTML = `<thead><tr>${columns.map(column => `<th>${escapeHtml(column.replaceAll("_", " "))}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${columns.map(column => `<td title="${escapeHtml(row[column])}">${escapeHtml(row[column])}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function renderDataRoom() {
  const total = Object.values(datasets).reduce((sum, rows) => sum + rows.length, 0);
  document.getElementById("dataCount").textContent = total > 999 ? `${(total / 1000).toFixed(1)}k` : total;
  renderDatasetCards();
  renderQuality();
  renderWatchlist();
  renderModelGovernance();
  preparePhaseControls();
  renderExplorer();
}

function exportWatchlist() {
  if (!currentWatchlist.length) { showToast("Load data before exporting the watchlist."); return; }
  const quote = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = ["severity,task_id,title,detail,source", ...currentWatchlist.map(item => [item.severity, item.taskId, item.title, item.detail, item.source].map(quote).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  Object.assign(document.createElement("a"), { href: url, download: "projectlens-proactive-watchlist.csv" }).click();
  URL.revokeObjectURL(url);
  showToast("Evidence-linked watchlist exported.");
}

const GUIDE_STEPS = [
  { view: "executive", title: "Start with the director briefing", copy: "ProjectLens translates the current schedule position into a short, evidence-linked decision brief.", next: "Next: inspect the evidence →" },
  { view: "analyst", title: "Challenge every conclusion", copy: "The analyst view exposes ranked drivers, confidence boundaries and the task-level evidence behind each signal.", next: "Next: test an intervention →" },
  { view: "executive", title: "Model a recovery case", copy: "The scenario lab compares the same 10,000 futures, then preserves the expected recovery and owner in the decision ledger.", next: "Next: open the data room →" },
  { view: "data", title: "Bring your own evidence", copy: "Inspect all 2,700+ demo rows or select your own ProjectLens CSV outputs. Your files stay in the browser.", next: "Load the complete demo →" }
];

function renderGuide() {
  const step = GUIDE_STEPS[guideStep];
  document.getElementById("guideTitle").textContent = step.title;
  document.getElementById("guideCopy").textContent = step.copy;
  document.getElementById("guideBack").disabled = guideStep === 0;
  document.getElementById("guideNext").textContent = step.next;
  document.querySelectorAll("#guideProgress i").forEach((item, index) => item.classList.toggle("active", index <= guideStep));
  switchView(step.view);
}

function openGuide() {
  guideStep = 0;
  renderGuide();
  openDialog(document.getElementById("guideDialog"));
}

document.querySelectorAll(".rail-item").forEach(item => item.addEventListener("click", () => switchView(item.dataset.viewTarget)));
document.querySelectorAll("[data-view-jump]").forEach(item => item.addEventListener("click", () => switchView(item.dataset.viewJump)));
document.getElementById("projectSelect").addEventListener("change", event => { activeProjectKey = event.target.value; renderExecutive(); showToast(`${PROJECTS[activeProjectKey].name} loaded with its own risk assumptions.`); });
document.getElementById("openLab").addEventListener("click", () => { populateLab(); openDialog(document.getElementById("scenarioDialog")); });
document.getElementById("openLabFromLedger").addEventListener("click", () => { populateLab(); openDialog(document.getElementById("scenarioDialog")); });
document.getElementById("methodButton").addEventListener("click", () => openDialog(document.getElementById("methodDialog")));
document.getElementById("exportBrief").addEventListener("click", exportBrief);
document.getElementById("evidenceFilter").addEventListener("input", renderEvidence);
document.getElementById("commitScenario").addEventListener("click", commitLabScenario);
["handoverRange", "approvalRange", "testingRange", "supplierRange"].forEach(id => document.getElementById(id).addEventListener("input", updateLab));
document.getElementById("guideButton").addEventListener("click", openGuide);
["openImportTop", "openImport"].forEach(id => document.getElementById(id).addEventListener("click", () => openDialog(document.getElementById("importDialog"))));
["loadDemoData", "loadDemoFromDialog"].forEach(id => document.getElementById(id).addEventListener("click", () => loadDemoDatasets(true)));
document.getElementById("dataFiles").addEventListener("change", event => importFiles(event.target.files));
document.getElementById("dataSearch").addEventListener("input", event => { dataSearch = event.target.value.trim().toLowerCase(); explorerPage = 1; renderExplorer(); });
document.getElementById("prevPage").addEventListener("click", () => { explorerPage -= 1; renderExplorer(); });
document.getElementById("nextPage").addEventListener("click", () => { explorerPage += 1; renderExplorer(); });
["phaseFrom", "phaseTo"].forEach(id => document.getElementById(id).addEventListener("change", renderUpdateComparison));
document.getElementById("exportWatchlist").addEventListener("click", exportWatchlist);
document.getElementById("guideBack").addEventListener("click", event => { event.preventDefault(); guideStep = Math.max(0, guideStep - 1); renderGuide(); });
document.getElementById("guideNext").addEventListener("click", async event => {
  event.preventDefault();
  if (guideStep === GUIDE_STEPS.length - 1) {
    document.getElementById("guideDialog").close();
    await loadDemoDatasets();
    return;
  }
  guideStep += 1;
  renderGuide();
});

renderExecutive();
renderEvidence();
renderDecisions();
renderDataRoom();
loadDemoDatasets(false);
