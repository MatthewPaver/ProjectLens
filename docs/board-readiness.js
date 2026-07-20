const state = {
  files: {},
  example: false,
  reviewed: new Set(),
  approved: false,
};

const findings = [
  {
    id: "F-01",
    type: "Direct conflict",
    title: "The scheduled finish moved 73 days, but the board narrative says delivery is unchanged.",
    detail: "The current finish is 14 May 2027; the previous commitment was 2 March 2027.",
    source: "Current and previous XER",
    reference: "PROJECT.scd_end_date",
    action: "Correct the narrative or provide an approved recovery basis.",
  },
  {
    id: "F-02",
    type: "Unsupported status",
    title: "The green delivery status is not supported by the current schedule and risk evidence.",
    detail: "A critical milestone moved and the related high risk has no active owner.",
    source: "July board pack + RAID log",
    reference: "Status §2 / R-017",
    action: "Decide the supported status and the recovery threshold.",
  },
  {
    id: "F-03",
    type: "Open condition",
    title: "A condition from the June approval remains open and is absent from the current pack.",
    detail: "The protected regression-test window was due to be confirmed before design freeze.",
    source: "Decision log",
    reference: "DEC-024 / condition 2",
    action: "Confirm owner, due date and whether the approval remains valid.",
  },
  {
    id: "F-04",
    type: "Ownership gap",
    title: "High risk R-017 has no named accountable owner.",
    detail: "The risk was last reviewed 21 days ago and is linked to the moved integration milestone.",
    source: "RAID log",
    reference: "R-017",
    action: "Name the accountable owner before accepting the proposed response.",
  },
  {
    id: "F-05",
    type: "Overdue commitment",
    title: "The cost sensitivity action is overdue and has no completion evidence.",
    detail: "The action was due on 17 July 2026 and is still marked open.",
    source: "Action log",
    reference: "ACT-041",
    action: "Bring the completed sensitivity or move the funding decision.",
  },
];

const decisions = [
  {
    title: "Confirm the supportable delivery status",
    detail: "Choose the July status using the corrected schedule and risk evidence; do not simply roll forward green.",
  },
  {
    title: "Confirm whether the June approval remains valid",
    detail: "Resolve, replace or formally waive the open regression-test condition.",
  },
  {
    title: "Set the recovery evidence and owners",
    detail: "Name owners and dates for R-017, ACT-041 and the schedule recovery basis.",
  },
];

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
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 3200);
}

function formatDate(value) {
  if (!value) return "Not set";
  const date = new Date(`${value}T12:00:00Z`);
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric", month: "long", year: "numeric", timeZone: "UTC",
  }).format(date);
}

function evidenceCount() {
  return Object.keys(state.files).length;
}

function updateEvidenceState() {
  const count = evidenceCount();
  $("#evidenceStatus").textContent = `${count} of 5 evidence groups present`;
  $("#runReview").disabled = !(state.files.current && state.files.previous);
}

function registerFile(kind, name) {
  state.files[kind] = name;
  const status = document.querySelector(`[data-file-state="${kind}"]`);
  status.textContent = "Present";
  status.classList.add("present");
  document.querySelector(`[data-file-name="${kind}"]`).textContent = name;
  updateEvidenceState();
}

function clearEvidence() {
  state.files = {};
  state.example = false;
  $$("[data-file-kind]").forEach(input => { input.value = ""; });
  $$("[data-file-state]").forEach(status => {
    status.textContent = "Missing";
    status.classList.remove("present");
  });
  $$("[data-file-name]").forEach(name => { name.textContent = "Choose file"; });
  updateEvidenceState();
}

function loadExample() {
  state.example = true;
  $("#projectName").value = "Northstar Grid Connection";
  $("#boardDate").value = "2026-07-24";
  $("#reviewOwner").value = "Programme Controls Lead";
  [
    ["current", "northstar-july-board-pack.pdf"],
    ["previous", "northstar-june-board-pack.pdf"],
    ["raid", "northstar-raid-log.csv"],
    ["actions", "northstar-actions-decisions.csv"],
    ["schedule", "northstar-june-july-schedules.xer"],
  ].forEach(([kind, name]) => registerFile(kind, name));
  runReview();
}

function renderFindings() {
  $("#findingRows").innerHTML = findings.map(item => `
    <div class="finding-row" role="row" data-finding="${item.id}">
      <label role="cell"><input type="checkbox" data-review-finding="${item.id}"><span>Checked</span></label>
      <div class="finding-copy" role="cell"><span>${escapeHtml(item.type)}</span><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p></div>
      <div class="finding-source" role="cell"><b>${escapeHtml(item.source)}</b>${escapeHtml(item.reference)}</div>
      <div class="finding-action" role="cell">${escapeHtml(item.action)}</div>
    </div>`).join("");
  $$("[data-review-finding]").forEach(input => input.addEventListener("change", event => {
    if (event.currentTarget.checked) state.reviewed.add(event.currentTarget.dataset.reviewFinding);
    else state.reviewed.delete(event.currentTarget.dataset.reviewFinding);
    updateHumanGate();
  }));
}

function renderDecisions() {
  $("#decisionList").innerHTML = decisions.map((item, index) => `
    <li><span>${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p></div></li>`).join("");
}

function updateHumanGate() {
  const reviewed = state.reviewed.size;
  $("#auditReviewText").textContent = `${reviewed} of ${findings.length} findings checked against source.`;
  const ready = reviewed === findings.length && $("#finalReviewer").value.trim() && $("#finalCheck").checked;
  $("#approveBrief").disabled = !ready;
}

function runReview() {
  if (!(state.files.current && state.files.previous)) {
    showToast("Add the current and previous board packs first.");
    return;
  }
  const project = $("#projectName").value.trim() || "Untitled project";
  const date = $("#boardDate").value;
  state.reviewed.clear();
  state.approved = false;
  $("#resultProject").textContent = project;
  $("#resultDate").textContent = formatDate(date);
  $("#resultVersion").textContent = state.example ? "July pack / review 01" : "Current and previous pack / review 01";
  $("#finalReviewer").value = $("#reviewOwner").value.trim();
  $("#finalCheck").checked = false;
  $("#approveBrief").disabled = true;
  $("#downloadJson").disabled = true;
  $("#downloadBrief").disabled = true;
  $("#exportStatus").textContent = "Review all five findings and name the reviewer before export.";
  renderFindings();
  renderDecisions();
  $("#results").hidden = false;
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function approveBrief() {
  state.approved = true;
  $("#approveBrief").textContent = "Brief reviewed";
  $("#approveBrief").disabled = true;
  $("#downloadJson").disabled = false;
  $("#downloadBrief").disabled = false;
  $("#exportStatus").textContent = `Human review recorded by ${$("#finalReviewer").value.trim()}.`;
  showToast("Board brief marked as reviewed. The underlying project decision remains with the board.");
}

function reviewPayload() {
  return {
    schemaVersion: "project-evidence-desk/board-readiness/v1",
    project: $("#resultProject").textContent,
    boardDate: $("#boardDate").value || null,
    evidence: Object.entries(state.files).map(([kind, fileName]) => ({ kind, fileName })),
    findings,
    decisions,
    review: {
      reviewedBy: $("#finalReviewer").value.trim(),
      reviewedFindingIds: [...state.reviewed],
      humanReviewed: state.approved,
      generatedAt: new Date().toISOString(),
    },
    boundaries: [
      "The public demonstrator does not establish contractual entitlement or causation.",
      "Northstar is synthetic. Production use requires agreed source, access and retention controls.",
      "A board or named authority remains responsible for the project decision.",
    ],
  };
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadJson() {
  download("northstar-board-readiness.json", JSON.stringify(reviewPayload(), null, 2), "application/json");
}

function downloadBrief() {
  const payload = reviewPayload();
  const markdown = [
    `# ${payload.project} — board readiness`,
    "",
    `**Board date:** ${formatDate(payload.boardDate)}`,
    `**Human reviewer:** ${payload.review.reviewedBy}`,
    "**Readiness:** Not ready for the stated decision",
    "",
    "## Board decisions required",
    ...payload.decisions.map((item, index) => `${index + 1}. **${item.title}** — ${item.detail}`),
    "",
    "## Reviewed findings",
    ...payload.findings.map(item => `- **${item.id} · ${item.type}:** ${item.title} Source: ${item.source}, ${item.reference}.`),
    "",
    "## Evidence files",
    ...payload.evidence.map(item => `- ${item.kind}: ${item.fileName}`),
    "",
    "## Boundary",
    ...payload.boundaries.map(item => `- ${item}`),
    "",
  ].join("\n");
  download("northstar-board-brief.md", markdown, "text/markdown");
}

$$("[data-file-kind]").forEach(input => input.addEventListener("change", event => {
  const file = event.currentTarget.files[0];
  if (file) registerFile(event.currentTarget.dataset.fileKind, file.name);
}));
$("#loadExample").addEventListener("click", loadExample);
$("#clearEvidence").addEventListener("click", clearEvidence);
$("#runReview").addEventListener("click", runReview);
$("#startAgain").addEventListener("click", () => {
  $("#results").hidden = true;
  $("#intake").scrollIntoView({ behavior: "smooth", block: "start" });
});
$("#finalReviewer").addEventListener("input", updateHumanGate);
$("#finalCheck").addEventListener("change", updateHumanGate);
$("#approveBrief").addEventListener("click", approveBrief);
$("#downloadJson").addEventListener("click", downloadJson);
$("#downloadBrief").addEventListener("click", downloadBrief);

window.ProjectEvidenceDesk = { findings, decisions, reviewPayload };
