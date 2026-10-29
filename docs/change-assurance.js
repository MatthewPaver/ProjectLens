// Walkthrough records are disposable and never share the real-work store.
const IS_WALKTHROUGH = new URLSearchParams(window.location.search).get("walkthrough") === "1";
const walkthroughRecords = new Map();
function recordStore() {
  return IS_WALKTHROUGH ? {
    getItem: key => walkthroughRecords.get(key) ?? null,
    setItem: (key, value) => walkthroughRecords.set(key, value),
    removeItem: key => walkthroughRecords.delete(key),
  } : localStorage;
}
if (IS_WALKTHROUGH) {
  document.body.classList.add("walkthrough-session");
  document.querySelector("#walkthroughSession").hidden = false;
}

const assuranceState = {
  files: { previous: null, current: null },
  review: null,
  originalNarrative: null,
  // Optional context stays human-controlled: use / ignore only, never auto-apply.
  precedents: { items: [], used: {}, ignored: {}, summary: null, mode: null },
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

// offline fallback if the local Gemini/LangSmith sidecar is not running
const comparableCasesFallback = [
  {
    id: "GMPP-DFT_0033_1819-Q1",
    title: "East Coast Mainline Programme",
    decision: "Published delivery confidence: Red (SRO Q4 assessment).",
    outcome: "Comparable public status evidence; the annual snapshot does not establish a causal outcome.",
    reasons: ["Infrastructure · schedule · scope · cost"],
    evidence: ["NISTA Major Projects Annual Report 2025-26", "Source record: DFT_0033_1819-Q1"],
  },
  {
    id: "GMPP-DCMS_0019_2021-Q4",
    title: "Project Gigabit",
    decision: "Published delivery confidence: Amber (SRO Q4 assessment).",
    outcome: "The source reports finance and supply-chain concerns; it does not prove which intervention worked.",
    reasons: ["Infrastructure · delivery · commercial supply chain"],
    evidence: ["NISTA Major Projects Annual Report 2025-26", "Source record: DCMS_0019_2021-Q4"],
  },
  {
    id: "GMPP-HO_0286_2324-Q2",
    title: "Asylum Support Accommodation Programme",
    decision: "Published delivery confidence: Red.",
    outcome: "A traced status comparison, not a decision precedent or recommendation.",
    reasons: ["Transformation · governance · schedule · commercial"],
    evidence: ["NISTA Major Projects Annual Report 2025-26", "Source record: HO_0286_2324-Q2"],
  },
];

// Static and local demos do not probe a port by default. Opt into the sidecar
// explicitly so GitHub Pages and browser tests do not produce a failed network
// request that looks like an application error.
const PRECEDENT_RAG_URL = window.PROJECTLENS_PRECEDENT_RAG_URL || null;
const IS_PUBLIC_PAGES = /github\.io$/i.test(window.location.hostname);

function precedentSidecarHint() {
  if (window.PROJECTLENS_PRECEDENT_RAG_URL) {
    return `Using configured RAG API · ${PRECEDENT_RAG_URL}`;
  }
  if (IS_PUBLIC_PAGES) {
    return "This GitHub Pages site is static, so it shows bundled example cards. Schedule checks still run here. For live Gemini retrieval, run `make precedent-rag` locally or set PROJECTLENS_PRECEDENT_RAG_URL.";
  }
  return "Static precedent mode. Start `make precedent-rag`, then set window.PROJECTLENS_PRECEDENT_RAG_URL to http://127.0.0.1:8787 to enable live retrieval.";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}


// Engine bindings: schedule parsing and review live in ProjectLensEngine.
const parseXer = (text, fileName) => ProjectLensEngine.parseXerAssurance(text, fileName);
const taskChanges = (...args) => ProjectLensEngine.taskChanges(...args);
const analyseReview = (...args) => ProjectLensEngine.analyseReview(...args);
const numberValue = (...args) => ProjectLensEngine.numberValue(...args);
const parseDate = (...args) => ProjectLensEngine.parseDate(...args);
const formatDate = (...args) => ProjectLensEngine.formatDate(...args);
const dayDifference = (...args) => ProjectLensEngine.dayDifference(...args);

function showToast(message) {
  const toast = $("#assuranceToast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 3300);
}

function storage(key) {
  try {
    const value = JSON.parse(recordStore().getItem(key) || "[]");
    return key === storageKeys.draft ? value : Array.isArray(value) ? value : [];
  } catch { return []; }
}

function saveStorage(key, value) {
  try {
    recordStore().setItem(key, JSON.stringify(value));
    return true;
  } catch {
    showError("This browser could not save the record. Free device storage or allow site storage, then try again. Keep this tab open.");
    return false;
  }
}

const storageKeys = {
  decisions: "projectlens:change-decisions:v1",
  conditions: "projectlens:change-conditions:v1",
  requests: "projectlens:evidence-requests:v1",
  draft: "projectlens:review-draft:v1",
};

function evidenceVersion(review) {
  return `${review.previous.fileName} → ${review.current.fileName}${review.inputIdentity ? ` · SHA-256 ${review.inputIdentity.slice(0, 12)}` : " · identity unavailable (legacy record)"}`;
}

async function inputIdentity() {
  const inputs = { schema: 2, previous: assuranceState.files.previous.text, current: assuranceState.files.current.text, narrative: $("#assuranceNarrative").value };
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(inputs)));
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("");
}

// The four change counts as one plain sentence, each with its meaning.
function numbersSentence(changes) {
  const count = (value, one, many) => `${value} ${value === 1 ? one : many}`;
  const activities = changes.changes.length;
  return `Between the two schedules, ${count(activities, "activity has", "activities have")} changed dates, spare time or fixed dates, or ${activities === 1 ? "was" : "were"} added or removed; `
    + `there ${changes.constraintChanges === 1 ? "is" : "are"} ${count(changes.constraintChanges, "constraint change", "constraint changes")} (imposed dates or rules); `
    + `${count(changes.floatErosion, "activity has", "activities have")} less spare time before delaying the finish (total float); `
    + `and ${count(changes.relationChanges, "activity link has", "activity links have")} changed (predecessor, successor, link type or lag).`;
}

function renderReview(review, restoredContext = null) {
  assuranceState.review = review;
  $("#revisedNarrative").value = $("#assuranceNarrative").value;
  $("#narrativeRevision").open = false;
  const ready = review.blockers.length === 0;
  const topBlockers = review.blockers.slice(0, 3);
  $("#reviewName").textContent = displayProjectName(review.current.project.name);
  $("#reviewEvidenceVersion").textContent = `${evidenceVersion(review)} · checked ${new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(review.generatedAt))}`;
  $("#readinessCard").classList.toggle("ready", ready);
  $("#readinessStatus").textContent = ready ? "Human review still required" : "Needs evidence before decision";
  document.body.classList.add("has-review");
  $("#readinessHeadline").textContent = ready ? "Review the evidence and record a decision" : `${topBlockers.length} findings need your review${review.blockers.length > 3 ? ` · ${review.blockers.length} total` : ""}`;
  $("#readinessExplanation").textContent = ready
    ? "These fixed checks raised no additional questions. Inspect all source facts; this is not an approval or an assurance verdict."
    : "Select a finding to check its source. Resolve it, or record why the person or board who approves changes accepts it.";
  $("#blockerList").innerHTML = topBlockers.length ? topBlockers.map((blocker, index) => `
    <article class="blocker">
      <button type="button" class="finding-select" data-finding="${index}" aria-pressed="${index === 0}">
        <span>${escapeHtml(blocker.type)}</span><strong>${escapeHtml(blocker.title)}</strong><small>${escapeHtml(blocker.detail)}</small>
      </button>
    </article>`).join("") : `<p>No blockers found. Inspect the finish dates and narrative beside this list.</p>`;
  if (review.blockers.length > 3) $("#blockerList").insertAdjacentHTML("beforeend", `<details><summary>Additional findings (${review.blockers.length - 3})</summary>${review.blockers.slice(3).map((blocker, index) => `<button class="finding-select" type="button" data-finding="${index + 3}" aria-pressed="false"><span>${escapeHtml(blocker.type)}</span><strong>${escapeHtml(blocker.title)}</strong></button>`).join("")}</details>`);
  assuranceState.selectedFinding = 0;
  renderSelectedFinding(0);
  setFindingViewed(false);
  $("#requestAnswers").hidden = ready;
  $("#decisionWarning").hidden = ready;

  $("#requestList").innerHTML = review.blockers.map((blocker, index) => `
    <article class="request-item"><span>Question ${index + 1} · ${escapeHtml(blocker.type)}</span><strong>${escapeHtml(questionFor(blocker))}</strong><small>${escapeHtml(blocker.detail)}</small></article>`).join("");

  const facts = [
    ["Scheduled finish", review.finishMovement === null ? "Not comparable" : `${review.finishMovement > 0 ? "+" : ""}${review.finishMovement} days`, `${formatDate(review.previous.project.scheduledFinish)} → ${formatDate(review.current.project.scheduledFinish)}`],
    ["Material activities", String(review.changes.changes.length), "Activities with changed dates, spare time, fixed dates, or added or removed activities"],
    ["Constraint changes", String(review.changes.constraintChanges), "Changes to imposed dates or rules (constraints)"],
    ["Less spare time (float)", String(review.changes.floatErosion), "Activities with less spare time before delaying the finish (total float)"],
    ["Task link changes", String(review.changes.relationChanges), "Changed links between tasks: which task comes first, link type or waiting time (lag)"],
    ["Evidence supplied", "2 schedule files (XER)", $("#assuranceNarrative").value.trim() ? "Schedule narrative included" : "No narrative supplied"],
  ];
  $("#evidenceSummary").innerHTML = facts.map(([label, value, detail]) => `
    <article class="evidence-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></article>`).join("");
  $("#reviewFacts").textContent = numbersSentence(review.changes);

  // kick off cited precedent retrieval — XER never leaves the browser; only narrative + blockers
  assuranceState.precedents = { items: [], used: {}, ignored: {}, summary: null, mode: null };
  renderPrecedentPanel({ loading: true });
  if (restoredContext) { assuranceState.precedents = restoredContext; renderPrecedentPanel({ offline: restoredContext.mode === "offline-fallback" }); }
  else void loadPrecedents(review);

  // Keep optional context available without displacing primary schedule evidence.
  $("#supportingEvidence").open = false;

  $("#packIntake").hidden = true;
  $("#readinessWorkspace").hidden = false;
  $("#requestPanel").hidden = true;
  $("#decisionPanel").hidden = true;
  moveTo($("#readinessWorkspace"));
}

function inferFilters(review) {
  // soft defaults from project name + narrative — sidecar still does semantic ranking
  const blob = `${review?.current?.project?.name || ""} ${$("#assuranceNarrative")?.value || ""}`.toLowerCase();
  const sector = /\brail\b|signall|metro|tram/.test(blob) ? "Rail"
    : /\bairport\b|aviation|baggage/.test(blob) ? "Aviation"
    : /\bhospital\b|building|civic/.test(blob) ? "Buildings"
    : /\bwater\b|pump|hydraulic/.test(blob) ? "Water"
    : /\benergy\b|power|substation|\bgrid\b|switchgear|protection testing/.test(blob) ? "Energy"
    : "Energy";
  const phase = /integrat|interface|systems|commission/.test(blob) ? "Integration"
    : /test|assurance|dynamic test/.test(blob) ? "Testing"
    : /construct|piling|civil|mobilisation/.test(blob) ? "Construction"
    : /procur|long-lead|factory/.test(blob) ? "Procurement"
    : "Integration";
  const type = /accelerat|second shift|night work|recover delay/.test(blob) ? "Acceleration"
    : /firmware|software|plc|technical|protection standard/.test(blob) ? "Technical change"
    : /design|foundation|redesign|layout/.test(blob) ? "Design change"
    : "Scope change";
  return { sector, phase, type };
}

function buildPrecedentQuery(review) {
  // shape the ask like a real change-authority brief, not a raw narrative paste
  const narrative = ($("#assuranceNarrative")?.value || "").trim();
  const project = review?.current?.project?.name || "the live programme";
  const finish = review?.finishMovement;
  const finishLine = finish == null
    ? "Finish movement could not be compared from the supplied pair."
    : `Control schedule finish moved ${finish > 0 ? "+" : ""}${finish} days versus the comparison file `
      + `(${formatDate(review.previous.project.scheduledFinish)} → ${formatDate(review.current.project.scheduledFinish)}).`;
  const blockerLines = (review.blockers || []).map(
    blocker => `${blocker.type}: ${blocker.title}. ${blocker.detail || ""}`
  );
  const material = (review.changes?.changes || []).slice(0, 5).map(
    change => `${change.code} — ${change.name}: ${change.summary}`
  );
  return [
    `Change-assurance review for ${project}.`,
    finishLine,
    narrative ? `Period narrative / covering note (claim under review): ${narrative}` : "No period narrative was supplied.",
    blockerLines.length ? `Deterministic blockers raised by the evidence check: ${blockerLines.join(" | ")}` : "",
    material.length ? `Material schedule movements in the pack: ${material.join("; ")}.` : "",
    "Need comparable past board/change-authority decisions where status narrative understated finish movement, or late interface and procurement changes during integration created similar assurance risk.",
  ].filter(Boolean).join("\n\n");
}

function openPrecedentSection({ scroll = true } = {}) {
  const evidence = $("#supportingEvidence") || document.querySelector(".supporting-evidence");
  if (evidence) evidence.open = true;
  if (scroll) {
    const target = $("#precedentSection") || evidence;
    target?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
  }
}

function renderPrecedentPanel({ loading = false, offline = false } = {}) {
  const host = $("#comparableCases");
  const summaryHost = $("#precedentSummary");
  const statusHost = $("#precedentStatus");
  const sectionGate = $("#precedentSectionGate");
  if (!host) return;

  if (loading) {
    // three skeletons so a 3–8s retrieve doesn't look like a broken one-card grid
    host.innerHTML = [1, 2, 3].map(n => `
      <article class="comparable-case is-skeleton" aria-hidden="true">
        <span>Retrieving ${n}</span>
        <strong>Loading comparable public evidence</strong>
        <p>Asking the RAG sidecar for similar published programme records…</p>
        <small>Schedule files stay in this browser</small>
      </article>`).join("");
    if (summaryHost) summaryHost.hidden = true;
    if (sectionGate) sectionGate.hidden = true;
    if (statusHost) statusHost.textContent = "Retrieving similar cases. Schedule files stay in this browser.";
    return;
  }

  const items = assuranceState.precedents.items.length
    ? assuranceState.precedents.items
    : comparableCasesFallback;
  assuranceState.precedents.items = items;

  host.innerHTML = items.map(item => {
    const gate = assuranceState.precedents.used[item.id]
      ? "used"
      : assuranceState.precedents.ignored[item.id]
        ? "ignored"
        : "pending";
    const semantic = Number(item.semantic);
    const reasons = (item.reasons || []).slice(0, 3).map(escapeHtml).join(" · ");
    const evidence = (item.evidence || []).slice(0, 3).map(escapeHtml).join(" · ");
    // Retrieval relevance is a ranking signal, not confidence or probability.
    const scoreLabel = item.score != null
      ? `relevance ${escapeHtml(String(item.score))}/100 · not confidence`
      : "n/a";
    const semanticLabel = Number.isFinite(semantic)
      ? ` · semantic ${semantic.toFixed(2)}`
      : "";
    const gateLabel = gate === "used" ? "Used as context" : gate === "ignored" ? "Ignored" : "Not reviewed";
    return `
    <article class="comparable-case gate-${gate}" data-precedent-id="${escapeHtml(item.id)}">
      <span>${escapeHtml(item.id)} · ${offline ? "Bundled public record" : "Retrieved public record"} · ${scoreLabel}${semanticLabel}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.decision)}</p>
      <small>${escapeHtml(item.outcome || "")}</small>
      <p class="precedent-reasons">${reasons || "No match reasons supplied"}</p>
      <p class="precedent-evidence">Sources: ${evidence || "None listed"}</p>
      <p class="precedent-gate-copy">Use adds this case to the decision record as context. Ignore records that you set it aside. Neither adds a recommendation.</p>
      <div class="precedent-gate">
        <button type="button" data-gate="use" data-id="${escapeHtml(item.id)}">Use</button>
        <button type="button" data-gate="ignore" data-id="${escapeHtml(item.id)}">Ignore</button>
        <i>${gateLabel}</i>
      </div>
    </article>`;
  }).join("");

  const brief = assuranceState.precedents.summary;
  if (summaryHost) {
    if (brief?.text) {
      summaryHost.hidden = false;
      summaryHost.innerHTML = `<span>Gemini evidence brief · citations only</span><p>${escapeHtml(brief.text)}</p><small>Comparable status evidence only. It is not advice and does not prove a precedent.</small>`;
    } else {
      summaryHost.hidden = true;
      summaryHost.innerHTML = "";
    }
  }
  if (statusHost) {
    const mode = assuranceState.precedents.mode || (offline ? "offline-fallback" : "unknown");
    statusHost.textContent = offline
      ? "Bundled public records, shown as illustrative context. They were not matched to your project and do not establish causation."
      : `Retrieved public records (${mode}). Check source and relevance before using a case as context.`;
  }
  updateDecisionGateState();
}

function pendingPrecedentIds() {
  const items = assuranceState.precedents.items || [];
  if (!items.length) return [];
  const gated = new Set([
    ...Object.keys(assuranceState.precedents.used || {}),
    ...Object.keys(assuranceState.precedents.ignored || {}),
  ]);
  return items.map(item => item.id).filter(id => !gated.has(id));
}

function updateDecisionGateState() {
  const pending = pendingPrecedentIds();
  $("#precedentGateBanner").hidden = false;
  $("#precedentGateBanner").textContent = pending.length
    ? "You can save without reviewing the optional public-record examples. They stay marked as not reviewed."
    : "You have marked every public-record example. Your choices are saved with the decision.";
  // One section-level status, not repeated instructions on every card.
  $("#precedentSectionGate").hidden = true;
  $("#decisionForm button[type=submit]").disabled = false;
}

async function loadPrecedents(review) {
  const filters = inferFilters(review);
  const problem = buildPrecedentQuery(review);

  if (!PRECEDENT_RAG_URL) {
    assuranceState.precedents.items = comparableCasesFallback;
    assuranceState.precedents.summary = null;
    assuranceState.precedents.mode = "offline-fallback";
    renderPrecedentPanel({ offline: true });
    return;
  }

  try {
    const response = await fetch(`${PRECEDENT_RAG_URL}/precedents/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        problem,
        narrative: ($("#assuranceNarrative")?.value || "").trim(),
        sector: filters.sector,
        phase: filters.phase,
        type: filters.type,
        blockers: (review.blockers || []).map(blocker => `${blocker.type}: ${blocker.title}`),
        limit: 5,
        summarise: true,
      }),
    });
    if (!response.ok) throw new Error(`Precedent API ${response.status}`);
    const payload = await response.json();
    if (assuranceState.review !== review) return;
    assuranceState.precedents.items = payload.cases || [];
    assuranceState.precedents.summary = payload.summary || null;
    assuranceState.precedents.mode = payload.mode || "gemini-hybrid";
    renderPrecedentPanel({ offline: false });
  } catch (error) {
    if (assuranceState.review !== review) return;
    console.warn("Precedent RAG sidecar unavailable", error);
    assuranceState.precedents.items = comparableCasesFallback;
    assuranceState.precedents.summary = null;
    assuranceState.precedents.mode = "offline-fallback";
    renderPrecedentPanel({ offline: true });
  }
}

function setPrecedentGate(id, gate) {
  if (gate === "use") {
    assuranceState.precedents.used[id] = true;
    delete assuranceState.precedents.ignored[id];
  } else {
    assuranceState.precedents.ignored[id] = true;
    delete assuranceState.precedents.used[id];
  }
  renderPrecedentPanel({
    offline: assuranceState.precedents.mode === "offline-fallback",
  });
}

function questionFor(blocker) {
  if (blocker.type === "Contradiction") return "How should the narrative be corrected or evidenced against the finish movement?";
  if (blocker.type === "Schedule movement") return "What source evidence explains the scheduled finish movement?";
  if (blocker.title.includes("constraint")) return "Which instruction or approval explains each constraint change?";
  if (blocker.title.includes("links between tasks")) return "What changed in the delivery sequence, and why?";
  if (blocker.type === "Ownership gap") return "Who owns the stated response, and by when must it be complete?";
  if (blocker.type === "Evidence gap") return "What narrative, reason and approval basis supports this submission?";
  if (blocker.type === "Float loss") return "Which approved change explains the lost spare time (float), and does the narrative need correcting?";
  return `Which approved change or evidence explains this? ${blocker.title}.`;
}

async function loadFile(kind, file) {
  if (!file) return;
  if (!/\.xer$/i.test(file.name)) {
    showError("Choose a Primavera P6 .xer file, then run the check again.");
    return;
  }
  assuranceState.files[kind] = { name: file.name, text: await file.text() };
  assuranceState.originalNarrative = null;
  const label = kind === "previous" ? $("#assurancePreviousName") : $("#assuranceCurrentName");
  label.textContent = file.name;
  label.closest(".assurance-file").classList.add("loaded");
  const ready = Boolean(assuranceState.files.previous && assuranceState.files.current);
  $("#runAssuranceReview").disabled = !ready;
  $("#intakeMessage").textContent = ready ? "Both files loaded. Ready to check." : "Choose the second schedule file to continue.";
}

async function runReview() {
  if (!assuranceState.files.previous || !assuranceState.files.current) return;
  $("#workspaceError").hidden = true;
  const button = $("#runAssuranceReview");
  button.disabled = true;
  button.innerHTML = "Checking evidence…";
  try {
    const previous = parseXer(assuranceState.files.previous.text, assuranceState.files.previous.name);
    const current = parseXer(assuranceState.files.current.text, assuranceState.files.current.name);
    const review = analyseReview(previous, current, $("#assuranceNarrative").value.trim());
    assuranceState.originalNarrative ??= $("#assuranceNarrative").value;
    review.inputIdentity = await inputIdentity();
    renderReview(review);
    showToast("Readiness check complete. Files remained in this browser.");
  } catch (error) {
    console.error(error);
    showError(error.message || "The evidence could not be checked. Choose two readable schedule files (XER), each holding one project, and try again.");
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
    if (![previousResponse, currentResponse, narrativeResponse].every(response => response.ok)) throw new Error("The example project files could not be loaded.");
    const [previousText, currentText, narrative] = await Promise.all([previousResponse.text(), currentResponse.text(), narrativeResponse.text()]);
    assuranceState.files.previous = { name: "northstar-previous.xer", text: previousText };
    assuranceState.files.current = { name: "northstar-current.xer", text: currentText };
    $("#assuranceNarrative").value = narrative;
    assuranceState.originalNarrative = narrative;
    $("#assurancePreviousName").textContent = "northstar-previous.xer";
    $("#assuranceCurrentName").textContent = "northstar-current.xer";
    $$(".assurance-file").forEach(card => card.classList.add("loaded"));
    $("#runAssuranceReview").disabled = false;
    $("#intakeMessage").textContent = "Example project loaded. Checking readiness…";
    // Go straight to the review. Scrolling to the intake first raced the review's own scroll
    // and left visitors past the summary and the decision button.
    const previous = parseXer(previousText, assuranceState.files.previous.name);
    const current = parseXer(currentText, assuranceState.files.current.name);
    const review = analyseReview(previous, current, narrative);
    review.inputIdentity = await inputIdentity();
    renderReview(review);
  } catch (error) {
    console.error(error);
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Try the example project";
  }
}

function showView(name) {
  const safeName = ["reviews", "decisions", "follow-up"].includes(name) ? name : "reviews";
  document.body.dataset.currentView = safeName;
  $$("[data-view]").forEach(view => {
    const active = view.dataset.view === safeName;
    view.hidden = !active;
    view.classList.toggle("active", active);
  });
  $$("[data-view-link]").forEach(link => {
    link.classList.toggle("active", link.dataset.viewLink === safeName);
    if (link.dataset.viewLink === safeName) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (safeName === "decisions") renderDecisionRegister();
  if (safeName === "follow-up") renderConditionRegister();
}

function renderDecisionRegister() {
  const decisions = storage(storageKeys.decisions);
  $("#decisionRegister").innerHTML = decisions.length ? decisions.map(item => `
    <article class="register-row">
      <div><span>${escapeHtml(item.project)} · ${escapeHtml(item.evidenceVersion)}</span><strong>${escapeHtml(item.rationale)}</strong><small>Decision owner: ${escapeHtml(item.owner)} · ${escapeHtml(formatTimestamp(item.createdAt))}</small></div>
      <aside><b>${escapeHtml(item.decision)}</b>${item.unresolved ? `<small>${item.unresolved} unresolved blocker${item.unresolved === 1 ? "" : "s"} preserved</small>` : ""}</aside>
    </article>`).join("") : `<div class="register-empty">No decisions saved in this browser yet. Try the example project to record one.</div>`;
}

function renderConditionRegister() {
  const conditions = storage(storageKeys.conditions);
  const openCount = conditions.filter(item => !item.closed).length;
  const badge = $("#conditionNavCount");
  badge.textContent = String(openCount);
  badge.hidden = openCount === 0;
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
    document.dispatchEvent(new Event("projectlens:condition-changed"));
  }));
}

function formatTimestamp(value) {
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function decisionExportText() {
  const decisions = storage(storageKeys.decisions);
  const conditions = storage(storageKeys.conditions);
  const lines = ["# ProjectLens decision record", ""];
  decisions.forEach(item => {
    lines.push(`## ${item.project}: ${item.decision}`);
    lines.push(`- Date: ${formatTimestamp(item.createdAt)}`);
    lines.push(`- Verdict: ${item.decision}`);
    lines.push(`- Decision owner: ${item.owner}`);
    lines.push(`- Rationale: ${item.rationale}`);
    lines.push(`- Evidence version: ${item.evidenceVersion}`);
    lines.push(`- Input SHA-256: ${item.inputIdentity || "Unavailable (legacy record)"}`);
    if (item.sourceEvidence) lines.push(`- Scheduled finish: ${item.sourceEvidence.finishPrevious} → ${item.sourceEvidence.finishCurrent}`);
    if (item.narrative) lines.push(`- Narrative under review: ${item.narrative}`);
    if (item.originalNarrative !== undefined) lines.push(`- Original narrative at first check: ${item.originalNarrative}`);
    (item.sourceEvidence?.activityEvidence || []).forEach(activity => lines.push(`- Activity ${activity.code} (${activity.name}): previous ${JSON.stringify(activity.before)}; current ${JSON.stringify(activity.after)}`));
    (item.sourceEvidence?.relationEvidence || []).forEach(relation => lines.push(`- ${relation.change} relationship: ${relation.predecessor} → ${relation.successor}; type ${relation.type}; lag ${relation.lagHours === null ? "Not supplied" : `${relation.lagHours} hours`}`));
    if (item.unresolved) lines.push(`- Unresolved blockers preserved: ${item.unresolved}`);
    (item.findings || []).forEach(finding => lines.push(`  - ${finding.type}: ${finding.title}. ${finding.detail}`));
    lines.push(`- Context status: ${item.contextStatus || "Unknown (legacy record)"}`);
    lines.push(`- Public-record context used: ${(item.precedentsUsed || []).join(", ") || "None"}`);
    lines.push(`- Public-record context ignored: ${(item.precedentsIgnored || []).join(", ") || "None"}`);
    lines.push(`- Public-record context not reviewed: ${item.schemaVersion === 2 ? (item.precedentsNotReviewed.join(", ") || "None") : "Unknown (legacy record)"}`);
    (item.contextSources || []).forEach(source => lines.push(`  - ${source.id}: ${source.title}; ${(source.evidence || []).join("; ")}`));
    const linked = conditions.filter(condition => condition.decisionId === item.id);
    if (linked.length) {
      lines.push("- Conditions:");
      linked.forEach(condition => lines.push(`  - ${condition.condition} (owner: ${condition.owner || "Unassigned"} · due: ${condition.due || "not set"} · ${condition.closed ? "closed" : "open"})`));
    }
    lines.push("");
  });
  return `${lines.join("\n").trim()}\n`;
}

async function exportDecisions() {
  if (!storage(storageKeys.decisions).length) {
    showToast("No decision has been recorded in this browser yet.");
    return;
  }
  const text = decisionExportText();
  try {
    await navigator.clipboard.writeText(text);
    $("#exportFallback").hidden = true;
    showToast("Decision record copied to your clipboard.");
  } catch {
    const fallback = $("#exportFallback");
    const area = $("#exportFallbackText");
    fallback.hidden = false;
    area.value = text;
    area.focus();
    area.select();
    showToast("Clipboard unavailable. Select and copy the record below.");
  }
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
    questions: review.blockers.map(questionFor),
    createdAt: new Date().toISOString(),
  });
  if (!saveStorage(storageKeys.requests, requests)) return;
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
  if ((decision === "Approve with conditions" || condition) && (!condition || !String(form.get("conditionOwner") || "").trim() || !form.get("conditionDue"))) {
    showError("Conditional approval needs a condition, its owner and a due date.");
    $("#decisionCondition").focus();
    return;
  }
  const usedIds = Object.keys(assuranceState.precedents.used);
  const ignoredIds = Object.keys(assuranceState.precedents.ignored);
  const pending = pendingPrecedentIds();
  if (!decision || !String(form.get("owner") || "").trim() || !String(form.get("rationale") || "").trim()) {
    showError("Choose a decision and add a named owner and rationale.");
    return;
  }
  const record = {
    id: `DR-${Date.now()}`,
    project: review.current.project.name,
    evidenceVersion: evidenceVersion(review),
    schemaVersion: 2,
    inputIdentity: review.inputIdentity,
    decision,
    owner: String(form.get("owner") || "").trim(),
    rationale: String(form.get("rationale") || "").trim(),
    unresolved: review.blockers.length,
    createdAt: new Date().toISOString(),
    precedentsUsed: usedIds,
    precedentsIgnored: ignoredIds,
    precedentsNotReviewed: pending,
    findings: review.blockers,
    narrative: $("#assuranceNarrative").value,
    originalNarrative: assuranceState.originalNarrative,
    sourceEvidence: { finishPrevious: formatDate(review.previous.project.scheduledFinish), finishCurrent: formatDate(review.current.project.scheduledFinish), materialChanges: review.changes.changes, activityEvidence: review.changes.activityEvidence, relationEvidence: review.changes.relationEvidence },
    contextSources: assuranceState.precedents.items.filter(item => usedIds.includes(item.id)),
    precedentSummary: usedIds.length ? (assuranceState.precedents.summary?.text || null) : null,
    precedentMode: assuranceState.precedents.mode,
    contextStatus: assuranceState.precedents.mode ? "Available context; dispositions recorded below" : "Retrieval incomplete; no context review is implied",
  };
  const decisions = storage(storageKeys.decisions);
  decisions.unshift(record);
  const updates = { [storageKeys.decisions]: decisions };

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
    updates[storageKeys.conditions] = conditions;
  }
  if (!saveRecordGroup(updates)) return;
  $("#workspaceError").hidden = true;
  if (storage(storageKeys.draft)?.inputIdentity === review.inputIdentity) {
    try { recordStore().removeItem(storageKeys.draft); $("#draftNotice").hidden = true; } catch { /* A saved decision remains valid if the old draft cannot be removed. */ }
  }
  renderConditionRegister();
  window.location.hash = "decisions";
  showToast("Decision record saved with its evidence version.");
  event.currentTarget.reset();
  document.dispatchEvent(new Event("projectlens:decision-saved"));
}

function resetReview(options = {}) {
  const landing = options.landing === true;
  if ((assuranceState.review || assuranceState.files.previous || assuranceState.files.current) && !window.confirm("Start another review? Unsaved work in this tab will be cleared. A saved device draft is kept until you delete or replace it.")) return;
  document.body.classList.remove("has-review");
  document.body.classList.toggle("has-intake", !landing);
  assuranceState.files = { previous: null, current: null };
  assuranceState.review = null;
  assuranceState.originalNarrative = null;
  $("#assurancePrevious").value = "";
  $("#assuranceCurrent").value = "";
  $("#assuranceNarrative").value = "";
  $("#assurancePreviousName").textContent = "Choose comparison point";
  $("#assuranceCurrentName").textContent = "Choose latest submission";
  $$(".assurance-file").forEach(card => card.classList.remove("loaded"));
  $("#runAssuranceReview").disabled = true;
  $("#intakeMessage").textContent = "Choose both schedule files (XER) to begin.";
  $("#readinessWorkspace").hidden = true;
  $("#packIntake").hidden = landing;
  moveTo(landing ? $("#productTitle") : $("#intakeTitle"));
}

function bindEvents() {
  $("#loadAssuranceDemo").addEventListener("click", loadDemo);
  $("#showOwnPack").addEventListener("click", () => {
    document.body.classList.add("has-intake");
    $("#packIntake").hidden = false;
    moveTo($("#intakeTitle"));
  });
  $("#backToProduct").addEventListener("click", () => resetReview({ landing: true }));
  $("#assurancePrevious").addEventListener("change", event => loadFile("previous", event.target.files[0]));
  $("#assuranceCurrent").addEventListener("change", event => loadFile("current", event.target.files[0]));
  $("#runAssuranceReview").addEventListener("click", runReview);
  $("#recheckNarrative").addEventListener("click", () => {
    $("#assuranceNarrative").value = $("#revisedNarrative").value;
    void runReview();
  });
  $("#startAnotherReview").addEventListener("click", resetReview);
  $("#reviewPrecedents")?.addEventListener("click", () => openPrecedentSection({ scroll: true }));
  $("#requestAnswers").addEventListener("click", () => {
    $("#requestPanel").hidden = false;
    moveTo($("#requestTitle"));
  });
  $("#showDecisionForm").addEventListener("click", () => {
    $("#decisionPanel").hidden = false;
    updateDecisionGateState();
    moveTo($("#decisionTitle"));
  });
  $("#reviewFirstFinding").addEventListener("click", () => openFinding(0, { focus: true }));
  $("#blockerList").addEventListener("click", event => {
    const button = event.target.closest("[data-finding]");
    if (!button) return;
    openFinding(Number(button.dataset.finding));
  });
  $("#findingEvidence").addEventListener("click", event => {
    if (!event.target.closest("#backToFindings")) return;
    $("#evidenceDesk").classList.remove("evidence-open");
    const finding = $(`[data-finding="${assuranceState.selectedFinding}"]`);
    finding.focus({ preventScroll: true });
    window.scrollTo({ top: assuranceState.findingScrollY || 0, behavior: "instant" });
    // Put the finding back where it was on screen, even if the actions above it changed height.
    if (assuranceState.findingTop !== null && assuranceState.findingTop !== undefined) {
      window.scrollBy({ top: finding.getBoundingClientRect().top - assuranceState.findingTop, behavior: "instant" });
    }
  });
  $("#saveDraft").addEventListener("click", saveDraft);
  $("#resumeDraft").addEventListener("click", resumeDraft);
  $("#discardDraft").addEventListener("click", () => {
    if (!window.confirm("Delete the saved draft from this device? The current open review stays in this tab.")) return;
    try { recordStore().removeItem(storageKeys.draft); $("#draftNotice").hidden = true; }
    catch { showError("The saved draft could not be deleted. Check browser storage permissions."); }
  });
  $("#saveEvidenceRequest").addEventListener("click", saveEvidenceRequest);
  $("#decisionForm").addEventListener("submit", saveDecision);
  $("#exportDecisions").addEventListener("click", exportDecisions);
  $("#comparableCases")?.addEventListener("click", event => {
    const button = event.target.closest("button[data-gate]");
    if (!button) return;
    setPrecedentGate(button.dataset.id, button.dataset.gate);
  });
  window.addEventListener("hashchange", () => showView(window.location.hash.slice(1)));
}


function showError(message) {
  const error = $("#workspaceError");
  error.textContent = message;
  error.hidden = false;
  moveTo(error);
}

// XER project names are often all capitals. Show those in normal case so the heading reads as a name.
function displayProjectName(name) {
  const text = String(name || "");
  if (/[a-z]/.test(text) || !/[A-Z]/.test(text)) return text;
  return text.toLowerCase().replace(/\b([a-z])/g, letter => letter.toUpperCase());
}

function moveTo(element) {
  if (!element) return;
  if (!element.hasAttribute("tabindex")) element.setAttribute("tabindex", "-1");
  element.focus({ preventScroll: true });
  element.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
}

// Until a finding has been opened, reviewing it is the main action. After that, recording the decision is.
function setFindingViewed(viewed) {
  assuranceState.findingViewed = viewed;
  const recordFirst = viewed || !assuranceState.review?.blockers.length;
  $("#reviewFirstFinding").hidden = recordFirst;
  $("#showDecisionForm").className = recordFirst ? "assurance-primary" : "product-text-button";
}

function openFinding(index, { focus = false } = {}) {
  const narrow = matchMedia("(max-width: 760px)").matches;
  // Remember where the finding sat on screen, before the main action swaps and the layout above it changes.
  if (narrow) assuranceState.findingTop = $(`[data-finding="${index}"]`)?.getBoundingClientRect().top ?? null;
  renderSelectedFinding(index);
  setFindingViewed(true);
  if (narrow) {
    assuranceState.findingScrollY = window.scrollY;
    $("#evidenceDesk").classList.add("evidence-open");
  }
  if (narrow || focus) moveTo($("#findingEvidence"));
}

function renderSelectedFinding(index) {
  assuranceState.selectedFinding = index;
  const review = assuranceState.review;
  const finding = review.blockers[index];
  $$("[data-finding]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.finding) === index)));
  const constraintFinding = finding?.title.includes("constraint");
  const floatFinding = finding?.type === "Float loss";
  const relationshipFinding = finding?.title.includes("links between tasks");
  const relevant = review.changes.activityEvidence.filter(change => {
    if (constraintFinding) return change.before && change.after && (change.before.constraintType !== change.after.constraintType || change.before.constraintDate !== change.after.constraintDate);
    if (floatFinding) return change.before?.totalFloatHours !== null && change.after?.totalFloatHours !== null && change.before?.totalFloatHours > change.after?.totalFloatHours;
    return true;
  });
  const value = item => !item ? "Activity absent" : constraintFinding ? `${item.constraintType} · ${item.constraintDate}` : floatFinding ? `${item.totalFloatHours ?? "Not supplied"} hours total float` : item.finish;
  const activityRows = relevant.map(change => `<li><code>${escapeHtml(change.code)}</code> ${escapeHtml(change.name)}<br><span>Previous: ${escapeHtml(value(change.before))}<br>Current: ${escapeHtml(value(change.after))}</span></li>`).join("") || "<li>No supporting activity changes found.</li>";
  const relationshipRows = review.changes.relationEvidence.map(relation => `<li>${escapeHtml(relation.change)}: <code>${escapeHtml(relation.predecessor)}</code> → <code>${escapeHtml(relation.successor)}</code><br><span>Type: ${escapeHtml(relation.type)} · lag: ${relation.lagHours === null ? "Not supplied" : `${escapeHtml(relation.lagHours)} hours`}</span></li>`).join("");
  $("#findingEvidence").innerHTML = `
    <button id="backToFindings" type="button">Back to findings</button>
    <h3>${escapeHtml(finding?.title || "Schedule evidence")}</h3>
    <p>${escapeHtml(finding?.detail || "No blockers found by these fixed checks.")}</p>
    <div class="review-question"><h4>Next review question</h4><p>${escapeHtml(finding ? questionFor(finding) : "What evidence supports the proposed update?")}</p></div>
    <dl class="source-dates"><div><dt>Previous finish</dt><dd>${escapeHtml(formatDate(review.previous.project.scheduledFinish))}</dd></div><div><dt>Current finish</dt><dd>${escapeHtml(formatDate(review.current.project.scheduledFinish))}</dd></div></dl>
    <details><summary>Inspect source files and fields</summary><p class="source-files">Sources: ${escapeHtml(evidenceVersion(review))} · PROJECT scheduled finish; TASK activity dates, constraints and total float.</p></details>
    <details ${constraintFinding || floatFinding || relationshipFinding ? "open" : ""}><summary>${relationshipFinding ? `Task link field evidence (${review.changes.relationEvidence.length})` : `Activity field evidence (${relevant.length})`}</summary><ul>${relationshipFinding ? relationshipRows : activityRows}</ul></details>
    <h4>Narrative under review</h4><blockquote>${escapeHtml(($("#assuranceNarrative").value.split(/\n\s*\n/).find(paragraph => /unchanged|on track|stable|no impact/i.test(paragraph)) || $("#assuranceNarrative").value || "No narrative supplied.").trim())}</blockquote><details><summary>Read full narrative</summary><p>${escapeHtml($("#assuranceNarrative").value || "No narrative supplied.")}</p></details>
    ${assuranceState.originalNarrative !== null && assuranceState.originalNarrative !== $("#assuranceNarrative").value ? `<details class="original-narrative"><summary>Read original narrative from the first check</summary><p>${escapeHtml(assuranceState.originalNarrative || "No narrative was supplied at the first check.")}</p></details>` : ""}
    <p class="source-boundary">Field coverage: ${review.coverage.map(item => `${escapeHtml(item.file)}: ${item.finishDates}/${item.tasks} finishes, ${item.floatValues}/${item.tasks} float values`).join("; ")}. Fixed checks flag conflicts. They do not establish causation, entitlement or contractual approval.</p>`;
}

function saveRecordGroup(updates) {
  const previous = {};
  try {
    for (const key of Object.keys(updates)) previous[key] = recordStore().getItem(key);
    for (const [key, value] of Object.entries(updates)) recordStore().setItem(key, JSON.stringify(value));
    return true;
  } catch {
    for (const [key, value] of Object.entries(previous)) {
      try { if (value === null) recordStore().removeItem(key); else recordStore().setItem(key, value); } catch { /* Keep the open form available for recovery. */ }
    }
    showError("The decision could not be saved. Keep this tab open, allow site storage or free device space, then retry.");
    return false;
  }
}

function saveDraft() {
  const fields = Object.fromEntries(new FormData($("#decisionForm")));
  const draft = { inputIdentity: assuranceState.review?.inputIdentity, files: assuranceState.files, narrative: $("#assuranceNarrative").value, originalNarrative: assuranceState.originalNarrative, pendingNarrative: $("#revisedNarrative").value, revisionOpen: $("#narrativeRevision").open, fields, precedents: assuranceState.precedents, selectedFinding: assuranceState.selectedFinding, decisionOpen: !$("#decisionPanel").hidden, savedAt: new Date().toISOString() };
  if (!saveStorage(storageKeys.draft, draft)) return;
  $("#draftNotice").hidden = false;
  $("#draftStatus").textContent = "Draft saved on this device, including the schedule file contents and decision fields. Save again after further edits. Delete it when finished.";
  showToast("Draft saved on this device. You can resume after reopening.");
}

async function resumeDraft() {
  try {
    const draft = storage(storageKeys.draft);
    if (!draft.files?.previous?.text || !draft.files?.current?.text) throw new Error("The saved draft is incomplete. Delete it and reload your two schedule files.");
    assuranceState.files = draft.files;
    assuranceState.originalNarrative = draft.originalNarrative ?? draft.narrative ?? "";
    $("#assuranceNarrative").value = draft.narrative || "";
    const review = analyseReview(parseXer(draft.files.previous.text, draft.files.previous.name), parseXer(draft.files.current.text, draft.files.current.name), draft.narrative || "");
    review.inputIdentity = await inputIdentity();
    renderReview(review, draft.precedents);
    $("#revisedNarrative").value = draft.pendingNarrative ?? draft.narrative ?? "";
    $("#narrativeRevision").open = Boolean(draft.revisionOpen);
    assuranceState.precedents = draft.precedents || assuranceState.precedents;
    renderPrecedentPanel({ offline: assuranceState.precedents.mode === "offline-fallback" });
    for (const [name, value] of Object.entries(draft.fields || {})) {
      const input = $("#decisionForm").elements.namedItem(name);
      if (input) input.value = value;
    }
    $("#decisionPanel").hidden = !draft.decisionOpen;
    renderSelectedFinding(draft.selectedFinding || 0);
    showToast("Saved draft restored. Recheck the evidence before deciding.");
  } catch (error) { showError(error.message || "The saved draft could not be restored. Reload your source files."); }
}

bindEvents();
$("#draftNotice").hidden = !storage(storageKeys.draft)?.files;
renderConditionRegister();
showView(window.location.hash.slice(1) || "reviews");
const bootStatus = $("#precedentStatus");
if (bootStatus) bootStatus.textContent = `Run a review to retrieve precedents. ${precedentSidecarHint()}`;
window.ProjectLensChangeAssurance = { parseXer, analyseReview };

// A narrow bridge lets the guide use the product's real actions and records.
if (IS_WALKTHROUGH) window.ProjectLensWalkthrough = {
  async load() {
    $("#workspaceError").hidden = true;
    await loadDemo();
    if (!assuranceState.review) throw new Error("The example could not load. Retry the walkthrough.");
  },
  async recheck(narrative) {
    $("#assuranceNarrative").value = narrative;
    await runReview();
    if (!$("#workspaceError").hidden) throw new Error($("#workspaceError").textContent);
  },
  snapshot() {
    return structuredClone({ review: assuranceState.review, narrative: $("#assuranceNarrative").value,
      decisions: storage(storageKeys.decisions), conditions: storage(storageKeys.conditions) });
  },
};
