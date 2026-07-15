const state = {
  data: null,
  view: "briefing",
  page: 1,
  pageSize: 18,
  filters: { query: "", movement: "", rating: "", department: "", theme: "" },
  detailId: null,
  demoStep: 0
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[character]));
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString("en-GB");
}

function formatDate(value) {
  if (!value) return "Not published";
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })
    .format(new Date(`${value}T12:00:00Z`));
}

function formatDays(value) {
  if (value === null || value === undefined) return "Not comparable";
  if (value === 0) return "No change";
  return `${Math.abs(value).toLocaleString("en-GB")} days ${value > 0 ? "later" : "earlier"}`;
}

function ratingClass(rating) {
  return (rating || "unpublished").toLowerCase().replace(/\s+/g, "-");
}

function transitionClass(transition) {
  return transition.toLowerCase().replace(/\s+/g, "-");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 3600);
}

function setView(view, { scroll = true } = {}) {
  state.view = view;
  $$("[data-view-panel]").forEach(panel => {
    const active = panel.dataset.viewPanel === view;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  $$(".nav-button").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  if (view === "cases") renderCases();
  if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function renderSummary() {
  const summary = state.data.summary;
  setText("heroCount", summary.latestProjects);
  setText("matchedCount", summary.matchedLatest);
  setText("fourYearCount", summary.allFourYears);
  setText("worsenedCount", summary.transitions.Worsened || 0);
  setText("improvedCount", summary.transitions.Improved || 0);
  setText("redCount", summary.ratingCounts.Red || 0);
  setText("recordCount", summary.historyRecords);
}

function priorityProjects() {
  const worsened = state.data.projects.filter(project => project.transition === "Worsened");
  const remaining = state.data.projects.filter(project => project.transition !== "Worsened");
  return [...worsened, ...remaining].slice(0, 6);
}

function renderPriority() {
  $("#priorityGrid").innerHTML = priorityProjects().map((project, index) => `
    <button class="priority-card ${index === 0 ? "lead" : ""}" data-project-id="${escapeHtml(project.id)}" type="button">
      <div class="priority-top"><span>0${index + 1}</span><b class="rating ${ratingClass(project.ipaDca)}">${escapeHtml(project.ipaDca || "Not published")}</b></div>
      <div class="score-ring" style="--score:${project.attentionScore}"><strong>${project.attentionScore}</strong><small>attention</small></div>
      <h3>${escapeHtml(project.name)}</h3>
      <p>${escapeHtml(project.attentionReasons[1] || project.attentionReasons[0] || "Published evidence available for review")}</p>
      <footer><span>${escapeHtml(project.department)}</span><i>${escapeHtml(project.transition)} ↗</i></footer>
    </button>`).join("");
  $$("[data-project-id]").forEach(button => button.addEventListener("click", () => openProject(button.dataset.projectId)));
}

function renderMovement() {
  const counts = state.data.summary.transitions;
  const items = [
    ["Worsened", counts.Worsened || 0],
    ["Improved", counts.Improved || 0],
    ["Unchanged", counts.Unchanged || 0],
    ["Not comparable", counts["Not comparable"] || 0]
  ];
  const max = Math.max(...items.map(([, count]) => count));
  $("#transitionChart").innerHTML = `
    <div class="chart-head"><span>2024-25 → 2025-26</span><b>Published IPA rating movement</b></div>
    ${items.map(([label, count]) => `<button type="button" data-transition-preset="${label}" class="transition-row ${transitionClass(label)}">
      <span>${label}</span><div><i style="--width:${Math.max(3, count / max * 100)}%"></i></div><strong>${count}</strong>
    </button>`).join("")}
    <p>Only exact Green, Amber and Red assessments are compared. Missing and exempt ratings remain visible as not comparable.</p>`;
  $$('[data-transition-preset]').forEach(button => button.addEventListener("click", () => {
    state.filters.movement = button.dataset.transitionPreset;
    state.page = 1;
    syncFilterInputs();
    renderExplorer();
    setView("explorer");
  }));

  const contradictions = state.data.projects
    .filter(project => project.transition === "Improved" && project.endDateDeltaDays > 30)
    .sort((a, b) => b.endDateDeltaDays - a.endDateDeltaDays)
    .slice(0, 3);
  $("#contradictionList").innerHTML = contradictions.map(project => `
    <button data-project-id="${escapeHtml(project.id)}" type="button"><span>${escapeHtml(project.previous?.ipaDca)} → ${escapeHtml(project.ipaDca)}</span><strong>${escapeHtml(project.name)}</strong><small>${formatDays(project.endDateDeltaDays)}</small></button>`).join("");
  $$("#contradictionList [data-project-id]").forEach(button => button.addEventListener("click", () => openProject(button.dataset.projectId)));
}

function renderThemes() {
  const themes = state.data.summary.topThemes;
  const max = themes[0]?.count || 1;
  $("#themeCloud").innerHTML = themes.map((item, index) => `
    <button data-theme-preset="${escapeHtml(item.theme)}" type="button" style="--weight:${.45 + item.count / max * .55}">
      <span>0${index + 1}</span><strong>${escapeHtml(item.theme)}</strong><b>${item.count}</b><small>current narratives</small>
    </button>`).join("");
  $$('[data-theme-preset]').forEach(button => button.addEventListener("click", () => {
    state.filters.theme = button.dataset.themePreset;
    state.page = 1;
    syncFilterInputs();
    renderExplorer();
    setView("explorer");
  }));
}

function optionList(values, label) {
  return `<option value="">${label}</option>${[...new Set(values.filter(Boolean))].sort().map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
}

function initialiseFilters() {
  const projects = state.data.projects;
  $("#movementFilter").innerHTML = optionList(projects.map(project => project.transition), "All movements");
  $("#ratingFilter").innerHTML = optionList(projects.map(project => project.ipaDca || "Not published"), "All ratings");
  $("#departmentFilter").innerHTML = optionList(projects.map(project => project.department), "All departments");
  $("#themeFilter").innerHTML = optionList(Object.keys(state.data.taxonomy), "All themes");
  $("#caseProjectSelect").innerHTML = projects
    .filter(project => project.themes.length)
    .map(project => `<option value="${escapeHtml(project.id)}">${escapeHtml(project.name)}</option>`).join("");
}

function filteredProjects() {
  const { query, movement, rating, department, theme } = state.filters;
  const needle = query.trim().toLowerCase();
  return state.data.projects.filter(project => {
    const searchable = [project.name, project.department, project.category, project.description, project.evidenceExcerpt, ...project.themes].join(" ").toLowerCase();
    return (!needle || searchable.includes(needle))
      && (!movement || project.transition === movement)
      && (!rating || (project.ipaDca || "Not published") === rating)
      && (!department || project.department === department)
      && (!theme || project.themes.includes(theme));
  });
}

function renderExplorer() {
  const projects = filteredProjects();
  const pages = Math.max(1, Math.ceil(projects.length / state.pageSize));
  state.page = Math.min(state.page, pages);
  const start = (state.page - 1) * state.pageSize;
  const page = projects.slice(start, start + state.pageSize);
  setText("resultCount", `${projects.length.toLocaleString("en-GB")} project${projects.length === 1 ? "" : "s"}`);
  setText("pageStatus", `Page ${state.page} of ${pages}`);
  $("#prevPage").disabled = state.page <= 1;
  $("#nextPage").disabled = state.page >= pages;
  $("#projectTable").innerHTML = page.length ? page.map(project => `
    <tr data-project-id="${escapeHtml(project.id)}" tabindex="0">
      <td><span class="table-score">${project.attentionScore}</span></td>
      <td><strong>${escapeHtml(project.name)}</strong><small>${escapeHtml(project.department)} · ${escapeHtml(project.category || "Uncategorised")}</small></td>
      <td><b class="rating ${ratingClass(project.ipaDca)}">${escapeHtml(project.ipaDca || "Not published")}</b></td>
      <td><span class="movement ${transitionClass(project.transition)}">${escapeHtml(project.transition)}</span></td>
      <td>${escapeHtml(formatDays(project.endDateDeltaDays))}</td>
      <td><div class="mini-themes">${project.themes.slice(0, 2).map(theme => `<span>${escapeHtml(theme)}</span>`).join("") || "No signal"}</div></td>
      <td><button type="button" aria-label="Open ${escapeHtml(project.name)}">↗</button></td>
    </tr>`).join("") : `<tr><td colspan="7" class="empty-result">No projects match those filters.</td></tr>`;
  $$("#projectTable tr[data-project-id]").forEach(row => {
    const open = () => openProject(row.dataset.projectId);
    row.addEventListener("click", open);
    row.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") open(); });
  });
}

function syncFilterInputs() {
  $("#searchInput").value = state.filters.query;
  $("#movementFilter").value = state.filters.movement;
  $("#ratingFilter").value = state.filters.rating;
  $("#departmentFilter").value = state.filters.department;
  $("#themeFilter").value = state.filters.theme;
}

function openProject(projectId) {
  const project = state.data.projects.find(item => item.id === projectId);
  if (!project) return;
  state.detailId = projectId;
  setText("detailDepartment", project.department || "Department not published");
  setText("detailCategory", project.category || "Category not published");
  setText("detailName", project.name);
  $("#detailSummary").innerHTML = `
    <div><span>Attention</span><strong>${project.attentionScore}</strong></div>
    <div><span>IPA DCA</span><strong class="${ratingClass(project.ipaDca)}">${escapeHtml(project.ipaDca || "N/P")}</strong></div>
    <div><span>Movement</span><strong>${escapeHtml(project.transition)}</strong></div>
    <div><span>End date</span><strong>${escapeHtml(formatDate(project.endDate))}</strong></div>`;
  $("#detailReasons").innerHTML = project.attentionReasons.length
    ? project.attentionReasons.map(reason => `<li>${escapeHtml(reason)}</li>`).join("")
    : "<li>No elevated scoring signal. Open the evidence for context.</li>";
  $("#detailEvidence").textContent = project.evidenceExcerpt || "No public narrative was available for this record.";
  $("#detailSource").href = project.sourceUrl;
  $("#detailTimeline").innerHTML = project.history.map(record => `
    <article>
      <time>${record.year - 1}–${String(record.year).slice(-2)}</time>
      <i class="${ratingClass(record.ipaDca)}"></i>
      <div><strong>${escapeHtml(record.ipaDca || "Not published")}</strong><span>${escapeHtml(formatDate(record.endDate))}</span><p>${escapeHtml(record.evidenceExcerpt || "No narrative published.")}</p><a href="${escapeHtml(record.sourceUrl)}" target="_blank" rel="noreferrer">Source ↗</a></div>
    </article>`).join("");
  $("#detailThemes").innerHTML = project.themes.map(theme => `<span>${escapeHtml(theme)}</span>`).join("") || "<span>No theme signal</span>";
  const dialog = $("#projectDialog");
  if (typeof dialog.showModal === "function") dialog.showModal();
}

function historicalImprovements() {
  const cases = [];
  state.data.projects.forEach(project => {
    const history = project.history;
    for (let index = 1; index < history.length; index += 1) {
      const previous = history[index - 1];
      const current = history[index];
      const score = { Green: 0, Amber: 1, Red: 2 };
      if (score[current.ipaDca] !== undefined && score[previous.ipaDca] !== undefined && score[current.ipaDca] < score[previous.ipaDca]) {
        cases.push({ project, previous, current });
      }
    }
  });
  return cases;
}

function comparableCases(project) {
  const targetThemes = new Set(project.themes);
  return historicalImprovements().filter(item => item.project.id !== project.id).map(item => {
    const sharedThemes = item.current.themes.filter(theme => targetThemes.has(theme));
    let similarity = sharedThemes.length * 18;
    if (item.current.category && item.current.category === project.category) similarity += 24;
    if (item.current.department && item.current.department === project.department) similarity += 12;
    similarity = Math.min(94, similarity);
    return { ...item, sharedThemes, similarity };
  }).filter(item => item.similarity > 0).sort((a, b) => b.similarity - a.similarity || b.current.year - a.current.year).slice(0, 6);
}

function renderCases() {
  if (!state.data) return;
  const select = $("#caseProjectSelect");
  const projectId = state.detailId || select.value || state.data.projects.find(item => item.transition === "Worsened")?.id;
  select.value = projectId;
  const project = state.data.projects.find(item => item.id === projectId) || state.data.projects[0];
  const cases = comparableCases(project);
  $("#caseContext").innerHTML = `<span>Comparing against</span><strong>${escapeHtml(project.name)}</strong><div>${project.themes.map(theme => `<i>${escapeHtml(theme)}</i>`).join("")}</div>`;
  $("#caseGrid").innerHTML = cases.length ? cases.map((item, index) => `
    <article class="case-card">
      <div class="case-rank"><span>0${index + 1}</span><b>${item.similarity}% signal overlap</b></div>
      <h2>${escapeHtml(item.project.name)}</h2>
      <div class="case-shift"><span class="rating ${ratingClass(item.previous.ipaDca)}">${escapeHtml(item.previous.ipaDca)}</span><i>→</i><span class="rating ${ratingClass(item.current.ipaDca)}">${escapeHtml(item.current.ipaDca)}</span><small>${item.previous.year - 1}–${String(item.current.year).slice(-2)}</small></div>
      <p>${escapeHtml(item.current.evidenceExcerpt || "No improvement narrative published.")}</p>
      <div class="shared-themes">${item.sharedThemes.map(theme => `<span>${escapeHtml(theme)}</span>`).join("")}</div>
      <footer><button type="button" data-project-id="${escapeHtml(item.project.id)}">Inspect case evidence ↗</button></footer>
    </article>`).join("") : `<div class="empty-cases">No comparable improved cases met the transparent similarity rules.</div>`;
  $$("#caseGrid [data-project-id]").forEach(button => button.addEventListener("click", () => openProject(button.dataset.projectId)));
}

function renderMethod() {
  $("#limitationsList").innerHTML = state.data.meta.limitations.map(item => `<li>${escapeHtml(item)}</li>`).join("");
  $("#sourceList").innerHTML = state.data.sources.map(source => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer"><span>${source.year - 1}–${String(source.year).slice(-2)}</span><strong>${escapeHtml(source.label)}</strong><i>Official CSV ↗</i></a>`).join("");
}

const demoSteps = [
  { title: "Start with the portfolio pulse", copy: "ProjectLens begins with the six comparable projects whose published IPA rating worsened this year.", action() { setView("briefing"); $(".pulse-strip").scrollIntoView({ behavior: "smooth", block: "center" }); } },
  { title: "Open the evidence, not a black box", copy: "The Unity Programme moved from Amber to Red. The detail view shows the exact scoring reasons and the published narrative.", action() { const project = state.data.projects.find(item => item.name.includes("UNITY PROGRAMME")) || state.data.projects.find(item => item.transition === "Worsened"); openProject(project.id); } },
  { title: "Ask what happened elsewhere", copy: "Case match retrieves projects that improved in a later release and explains the theme, category and department overlap.", action() { $("#projectDialog").close(); setView("cases"); renderCases(); } },
  { title: "End with the boundary", copy: "The score is a transparent attention queue. It is not a probability of failure, and every recommendation remains a human decision.", action() { setView("method"); } }
];

function renderDemo() {
  const step = demoSteps[state.demoStep];
  setText("demoStep", `${state.demoStep + 1} / ${demoSteps.length}`);
  setText("demoTitle", step.title);
  setText("demoCopy", step.copy);
  $$(".demo-progress i").forEach((item, index) => item.classList.toggle("active", index <= state.demoStep));
  $("#demoBack").disabled = state.demoStep === 0;
  $("#demoNext").textContent = state.demoStep === demoSteps.length - 1 ? "Finish demo" : "Show me →";
}

function bindEvents() {
  $$(".nav-button").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  $$('[data-go]').forEach(button => button.addEventListener("click", () => {
    if (button.dataset.preset) state.filters.movement = button.dataset.preset;
    syncFilterInputs(); renderExplorer(); setView(button.dataset.go);
  }));
  $("#openPriority").addEventListener("click", () => openProject(priorityProjects()[0].id));
  $("#searchInput").addEventListener("input", event => { state.filters.query = event.target.value; state.page = 1; renderExplorer(); });
  [["movementFilter", "movement"], ["ratingFilter", "rating"], ["departmentFilter", "department"], ["themeFilter", "theme"]].forEach(([id, key]) => {
    $(`#${id}`).addEventListener("change", event => { state.filters[key] = event.target.value; state.page = 1; renderExplorer(); });
  });
  $("#clearFilters").addEventListener("click", () => { state.filters = { query: "", movement: "", rating: "", department: "", theme: "" }; state.page = 1; syncFilterInputs(); renderExplorer(); });
  $("#prevPage").addEventListener("click", () => { state.page -= 1; renderExplorer(); $(".filter-bar").scrollIntoView({ behavior: "smooth" }); });
  $("#nextPage").addEventListener("click", () => { state.page += 1; renderExplorer(); $(".filter-bar").scrollIntoView({ behavior: "smooth" }); });
  $("#findCases").addEventListener("click", () => { state.detailId = $("#caseProjectSelect").value; renderCases(); showToast("Comparable improvements ranked using visible similarity rules."); });
  $("#caseProjectSelect").addEventListener("change", event => { state.detailId = event.target.value; });
  $(".dialog-close").addEventListener("click", () => $("#projectDialog").close());
  $$(".dialog-close")[1].addEventListener("click", () => $("#demoDialog").close());
  $("#detailFindCases").addEventListener("click", () => { $("#projectDialog").close(); setView("cases"); });
  $("#demoButton").addEventListener("click", () => { state.demoStep = 0; renderDemo(); $("#demoDialog").showModal(); });
  $("#demoBack").addEventListener("click", () => { if (state.demoStep > 0) state.demoStep -= 1; renderDemo(); });
  $("#demoNext").addEventListener("click", () => {
    const step = demoSteps[state.demoStep];
    $("#demoDialog").close();
    step.action();
    if (state.demoStep < demoSteps.length - 1) {
      state.demoStep += 1;
      setTimeout(() => { renderDemo(); $("#demoDialog").showModal(); }, 900);
    } else {
      showToast("Demo complete. Every signal remains linked to official evidence.");
    }
  });
}

async function init() {
  try {
    const response = await fetch("data/gmpp.json");
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    state.data = await response.json();
    renderSummary();
    renderPriority();
    renderMovement();
    renderThemes();
    initialiseFilters();
    renderExplorer();
    renderMethod();
    bindEvents();
    $("#loading").classList.add("hidden");
    const requestedProject = new URLSearchParams(window.location.search).get("project");
    if (requestedProject && state.data.projects.some(project => project.id === requestedProject)) openProject(requestedProject);
  } catch (error) {
    $("#loading").innerHTML = `<strong>ProjectLens could not load the public dataset.</strong><span>${escapeHtml(error.message)}</span>`;
    console.error(error);
  }
}

init();
