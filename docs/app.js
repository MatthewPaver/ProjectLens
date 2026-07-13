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

renderExecutive();
renderEvidence();
renderDecisions();
