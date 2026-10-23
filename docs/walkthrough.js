/* A guide around the existing product. All checks and records use its engine. */
(() => {
  const engine = window.ProjectLensWalkthrough;
  if (!engine) return;
  const byId = id => document.getElementById(id);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const steps = [
    ['Read the pack', 'The update says “unchanged”. What do the two schedules actually show?', 'These are the bundled Northstar XER files. The dates below come from the same parser used by the full application.'],
    ['Inspect the conflict', 'The finish moved. The narrative did not.', 'Select a finding in the real evidence desk. Inspect the original activity fields and the narrative beside them. All detected findings remain available.'],
    ['Change the claim', 'Same schedules. A different narrative.', 'Edit the text, then rerun the real checks. Acknowledging movement can remove a keyword conflict; it does not repair the schedule or prove the explanation.'],
    ['Make a decision', 'The next step belongs to a person.', 'Choose an outcome and complete the real form. “Approve with conditions” also needs a condition, owner and due date. Unresolved findings stay in the record.'],
    ['Export the record', 'The decision keeps its evidence.', 'Download the engine-generated record, including the exact narrative, findings, source identity and any condition. This walkthrough stores nothing in your saved-work register.'],
    ['Track the follow-up', 'An owned condition stays visible.', 'If you added a condition, mark it closed in the real register. Closure changes its status; it does not erase the decision or its original findings.'],
  ];
  let step = 0, playing = false, timer, busy = false, loaded = false, dirty = false, original = '', originalCount = 0;
  let downloaded = false;
  const guide = byId('walkthroughGuide');
  guide.innerHTML = `<h1>See a change review through.</h1><p class="tour-intro">A short guided example, with the working application. Play the introduction or explore at your pace.</p>
    <ol class="tour-steps">${steps.map(([label], i) => `<li><button type="button" data-tour-step="${i}" disabled><span>${i + 1}</span>${label}</button></li>`).join('')}</ol>
    <div class="tour-controls"><button type="button" id="tourPlay" disabled>Play walkthrough</button><button type="button" id="tourRestart">Restart</button></div>
    <p id="tourPlayback" class="tour-fine" role="status">Loading the original evidence…</p>
    <div class="tour-navigation"><button type="button" id="tourBack" disabled>Back</button><button type="button" id="tourNext" disabled>Next: inspect the conflict</button></div>
    <p class="tour-fine">About one minute plus your decision. No upload or AI call. Reloading discards the example records.</p>
    <a class="tour-exit" href="change-assurance.html">Try with your own change pack</a>`;
  guide.hidden = false;
  guide.insertBefore(guide.querySelector(".tour-navigation"), guide.querySelector(".tour-steps"));
  const stage = document.createElement('section');
  stage.className = 'tour-stage';
  stage.setAttribute('aria-labelledby', 'tourTitle');
  stage.innerHTML = `<div class="tour-stage-heading"><p id="tourPosition">Step 1 of 6</p><h2 id="tourTitle" tabindex="-1"></h2><p id="tourDescription"></p></div>
    <div id="tourError" class="tour-error" role="alert" hidden><p></p><button type="button" id="tourRetry">Retry loading the example</button></div>
    <section id="tourPack" aria-label="Parsed schedule comparison"><div class="tour-timeline">
      <div class="tour-schedule"><span>Previous scheduled finish</span><strong id="tourPreviousDate">Loading…</strong><div class="tour-track"><div class="tour-bar previous"></div></div><code>northstar-previous.xer</code></div>
      <div class="tour-schedule"><span>Current scheduled finish</span><strong id="tourCurrentDate">Loading…</strong><div class="tour-track"><div class="tour-bar current"></div></div><code>northstar-current.xer</code></div>
      <p id="tourMovement"></p><p class="tour-fine">Finish-date comparison, not a critical-path calculation. Both bars share a date scale starting at the previous data date.</p></div>
      <blockquote id="tourOriginalClaim"></blockquote><p id="tourInputDetails" class="tour-fine"></p></section>
    <section id="tourExperiment" hidden><label for="tourNarrative">Narrative to check against these same two schedules</label><textarea id="tourNarrative" rows="7" spellcheck="false"></textarea>
      <div class="tour-experiment-actions"><button type="button" id="tourRerun">Rerun checks</button><button type="button" id="tourAcknowledge">Try an acknowledged finish change</button><button type="button" id="tourOriginal">Restore original narrative</button></div>
      <p id="tourCheckStatus" role="status"></p><p class="tour-fine">These are keyword checks. A smaller finding count does not establish that the narrative is correct.</p></section>
    <section id="tourRecord" hidden><p id="tourRecordSummary"></p><button type="button" id="tourDownload">Download walkthrough record</button><p id="tourDownloadStatus" role="status"></p></section>
    <p id="tourDecisionHint" class="tour-form-hint" hidden>Example only. The owner and rationale are editable. Choose your own outcome; nothing is saved outside this tab.</p>
    <p id="tourComplete" class="tour-complete" role="status" hidden></p>`;
  byId('workspace').prepend(stage);
  const date = value => new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(new Date(value));
  const currentDecision = snapshot => snapshot.decisions.find(record => record.inputIdentity === snapshot.review?.inputIdentity);
  function stop(message) {
    playing = false; clearTimeout(timer);
    byId('tourPlay').textContent = 'Play walkthrough';
    byId('tourPlay').setAttribute('aria-pressed', 'false');
    if (message) byId('tourPlayback').textContent = message;
  }
  function schedule() {
    clearTimeout(timer);
    if (playing && step < 3 && !busy && !dirty) timer = setTimeout(() => go(step + 1, false), 10000);
  }
  function updateControls() {
    const snapshot = engine.snapshot(), decision = currentDecision(snapshot);
    const locked = busy || !loaded;
    byId('tourBack').disabled = locked || step === 0 || (step === 2 && dirty);
    byId('tourNext').disabled = locked || (step === 2 && dirty) || (step === 3 && !decision) || step === 5;
    const openConditions = snapshot.conditions.some(condition => condition.decisionId === decision?.id && !condition.closed);
    byId('tourNext').textContent = step === 5 ? (openConditions ? 'Follow-up still open' : 'Walkthrough complete') : `Next: ${steps[step + 1][0].toLowerCase()}`;
    byId('tourPlay').disabled = locked || step >= 3 || reduced.matches || dirty;
    guide.querySelectorAll('[data-tour-step]').forEach(button => {
      const index = Number(button.dataset.tourStep);
      button.disabled = locked || index > step || (step === 2 && dirty && index !== step);
      if (index === step) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current');
    });
    ['tourRerun', 'tourAcknowledge', 'tourOriginal'].forEach(id => { byId(id).disabled = locked; });
  }
  function recordSummary() {
    const snapshot = engine.snapshot(), decision = currentDecision(snapshot);
    if (!decision) return;
    const conditions = snapshot.conditions.filter(condition => condition.decisionId === decision.id);
    byId('tourRecordSummary').textContent = `${decision.decision}. ${decision.unresolved} unresolved findings preserved. Source identity: ${decision.inputIdentity.slice(0, 12)}. ${conditions.length} condition${conditions.length === 1 ? '' : 's'} attached.`;
    byId('tourDownloadStatus').textContent = downloaded ? 'Record downloaded. Continue to the follow-up register.' : 'Download is optional and starts only when you choose it.';
    byId('tourComplete').hidden = step !== 5;
    byId('tourComplete').textContent = conditions.length
      ? conditions.every(condition => condition.closed) ? 'Condition closed. The original decision and evidence remain in the record. You have completed the review journey.' : 'Your decision is recorded. Close the example condition below to complete the follow-up.'
      : 'You recorded a decision without conditions. There is no condition to close. You have completed the review journey.';
  }
  function go(index, focus = true) {
    if (busy || !loaded || index < 0 || index > 5) return;
    if (step === 2 && dirty && index !== 2) { stop('Rerun the edited narrative before continuing.'); return; }
    if (index > 3 && !currentDecision(engine.snapshot())) return;
    step = index;
    document.body.dataset.tourStage = String(step);
    byId('tourPosition').textContent = `Step ${step + 1} of 6 · ${steps[step][0]}`;
    byId('tourTitle').textContent = steps[step][1];
    byId('tourDescription').textContent = steps[step][2];
    byId('tourPack').hidden = step !== 0;
    byId('tourExperiment').hidden = step !== 2;
    byId('tourDecisionHint').hidden = step !== 3;
    byId('tourRecord').hidden = step < 4;
    byId('decisionPanel').hidden = step !== 3;
    byId('tourComplete').hidden = step !== 5;
    // These are the application's views, not replicas of their content.
    const view = step < 4 ? 'reviews' : step === 4 ? 'decisions' : 'follow-up';
    location.hash = view;
    document.querySelectorAll('[data-view]').forEach(element => { element.hidden = element.dataset.view !== view; });
    if (step === 3) {
      stop('Playback paused. Choose the human decision, then save the example record.');
      if (!byId('decisionOwner').value) byId('decisionOwner').value = 'Example review chair';
      if (!byId('decisionRationale').value) byId('decisionRationale').value = 'Example: preserve the unresolved findings and require the schedule basis to be confirmed.';
    } else if (!playing) byId('tourPlayback').textContent = reduced.matches ? 'Reduced motion is on. Use Next to explore at your own pace.' : 'Use Next, or play the introduction. You control every decision and download.';
    recordSummary(); updateControls(); schedule();
    if (focus) {
      byId('tourTitle').focus({ preventScroll: true });
      stage.scrollIntoView({ block: 'start', behavior: 'instant' });
    }
  }
  async function recheck(value) {
    stop('Rechecking the narrative with the existing engine…');
    busy = true; updateControls();
    try {
      await engine.recheck(value);
      byId('tourNarrative').value = engine.snapshot().narrative;
      dirty = false; downloaded = false;
      const review = engine.snapshot().review;
      byId('tourCheckStatus').textContent = `${originalCount} findings in the original pack; ${review.blockers.length} now. The scheduled finish still moved ${review.finishMovement} days. Check the findings below.`;
      byId('tourError').hidden = true;
      byId('tourPlayback').textContent = 'Checks rerun. Inspect the result or continue to the decision.';
    } catch (error) { showError(error.message); }
    finally { busy = false; updateControls(); }
  }
  function showError(message) {
    stop('The example is paused. Your saved-work register has not changed.');
    byId('tourError').hidden = false;
    byId('tourError').querySelector('p').textContent = message;
    byId('tourRetry').hidden = loaded;
  }
  async function load() {
    busy = true; loaded = false; updateControls();
    byId('tourError').hidden = true;
    try {
      await engine.load();
      const snapshot = engine.snapshot(), review = snapshot.review;
      original = snapshot.narrative; originalCount = review.blockers.length;
      byId('tourPreviousDate').textContent = date(review.previous.project.scheduledFinish);
      byId('tourCurrentDate').textContent = date(review.current.project.scheduledFinish);
      const start = new Date(review.previous.project.dataDate).getTime();
      const previous = new Date(review.previous.project.scheduledFinish).getTime() - start;
      const current = new Date(review.current.project.scheduledFinish).getTime() - start;
      document.querySelector('.tour-bar.previous').style.setProperty('--finish-scale', String(Math.max(0, previous) / Math.max(1, current, previous)));
      document.querySelector('.tour-bar.current').style.setProperty('--finish-scale', String(Math.max(0, current) / Math.max(1, current, previous)));
      byId('tourMovement').textContent = `${review.finishMovement} days later. ${review.blockers.length} findings from the current checks.`;
      byId('tourOriginalClaim').textContent = original.split(/\n\s*\n/).find(part => /unchanged|on track/i.test(part)) || original;
      byId('tourInputDetails').textContent = `Parsed ${review.previous.tasks.length} previous and ${review.current.tasks.length} current activities. Nothing was uploaded.`;
      byId('tourNarrative').value = original;
      loaded = true;
    } catch (error) { showError(error.message); }
    finally { busy = false; if (loaded) { go(0, false); window.scrollTo({ top: 0, behavior: 'instant' }); } else updateControls(); }
  }
  byId('tourPlay').addEventListener('click', () => {
    if (playing) { stop('Playback paused. Continue at your pace.'); return; }
    playing = true;
    byId('tourPlay').textContent = 'Pause walkthrough'; byId('tourPlay').setAttribute('aria-pressed', 'true');
    byId('tourPlayback').textContent = 'Playing the introduction. Each step stays visible for ten seconds; the decision always pauses playback.';
    schedule();
  });
  byId('tourNext').addEventListener('click', () => { stop(); go(step + 1); });
  byId('tourBack').addEventListener('click', () => { stop(); go(step - 1); });
  guide.querySelectorAll('[data-tour-step]').forEach(button => button.addEventListener('click', () => { stop(); go(Number(button.dataset.tourStep)); }));
  byId('tourRestart').addEventListener('click', () => location.reload());
  byId('tourRetry').addEventListener('click', load);
  byId('tourNarrative').addEventListener('input', () => { dirty = true; stop('Narrative changed. Rerun the checks before continuing.'); updateControls(); });
  byId('tourNarrative').addEventListener('focus', () => stop('Playback paused while you edit.'));
  byId('tourRerun').addEventListener('click', () => recheck(byId('tourNarrative').value));
  byId('tourOriginal').addEventListener('click', () => recheck(original));
  byId('tourAcknowledge').addEventListener('click', () => {
    const review = engine.snapshot().review;
    recheck(`The scheduled finish moves from ${date(review.previous.project.scheduledFinish)} to ${date(review.current.project.scheduledFinish)}, ${review.finishMovement} days later.\n\nGovernance action is under review.`);
  });
  document.addEventListener('projectlens:decision-saved', () => { stop(); downloaded = false; go(4); });
  document.addEventListener('projectlens:condition-changed', () => { recordSummary(); updateControls(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop('Playback paused while this tab is hidden.'); });
  reduced.addEventListener('change', () => { if (reduced.matches) stop('Reduced motion is on. Use Next to explore.'); updateControls(); });
  document.querySelectorAll('[data-view-link]').forEach(link => link.addEventListener('click', event => {
    event.preventDefault(); stop();
    go(link.dataset.viewLink === 'reviews' ? 1 : link.dataset.viewLink === 'decisions' ? 4 : 5);
  }));
  document.querySelectorAll('#blockerList, #decisionForm, #findingEvidence').forEach(element => ['pointerdown', 'focusin'].forEach(event => element.addEventListener(event, () => stop('Playback paused while you explore the application.'))));
  byId('tourDownload').addEventListener('click', () => {
    const snapshot = engine.snapshot(), decision = currentDecision(snapshot);
    if (!decision) return;
    const record = { schemaVersion: 1, demonstration: true, storage: 'memory-only', decisions: [decision], conditions: snapshot.conditions.filter(condition => condition.decisionId === decision.id) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'projectlens-walkthrough-record.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    downloaded = true; recordSummary();
  });
  document.title = 'ProjectLens | Guided change-review walkthrough';
  go(0, false);
  load();
})();
