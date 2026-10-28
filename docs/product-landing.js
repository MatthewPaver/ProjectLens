(() => {
  const byId = id => document.getElementById(id);
  if (new URLSearchParams(location.search).get('walkthrough') === '1') return;
  const tabs = [...document.querySelectorAll('[data-example]')];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 760px)');
  let review = null, narrative = '', selected = 'finish';
  const escape = text => String(text ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const date = value => new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(value);
  const sentence = pattern => narrative.split(/(?<=[.!?])\s+/).find(text => pattern.test(text))?.trim() || 'Read the supplied narrative alongside the field evidence.';
  function replay() {
    const line = document.querySelector('.example-line');
    if (!line || reduced.matches) return;
    line.getAnimations().forEach(animation => animation.cancel());
    line.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: 700, easing: 'cubic-bezier(.16,1,.3,1)' });
  }
  function sourceRows() {
    if (selected === 'finish') return `<p>PROJECT scheduled finish from each original export.</p><table><thead><tr><th>Source file</th><th>Field</th><th>Value</th></tr></thead><tbody><tr><td>${escape(review.previous.fileName)}</td><td><code>PROJECT.scd_end_date</code></td><td>${escape(date(review.previous.project.scheduledFinish))}</td></tr><tr><td>${escape(review.current.fileName)}</td><td><code>PROJECT.scd_end_date</code></td><td>${escape(date(review.current.project.scheduledFinish))}</td></tr></tbody></table>`;
    if (selected === 'relationships') return `<p>TASKPRED signatures from the previous and current exports. Added and removed signatures are counted separately.</p><table><thead><tr><th>Change</th><th>Predecessor</th><th>Successor</th><th>Type / lag</th></tr></thead><tbody>${review.changes.relationEvidence.map(row => `<tr><td>${escape(row.change)}</td><td><code>${escape(row.predecessor)}</code></td><td><code>${escape(row.successor)}</code></td><td>${escape(row.type)} / ${row.lagHours === null ? 'not supplied' : `${escape(row.lagHours)}h`}</td></tr>`).join('')}</tbody></table>`;
    const rows = review.changes.activityEvidence.filter(row => row.before && row.after && (row.before.constraintType !== row.after.constraintType || row.before.constraintDate !== row.after.constraintDate));
    return `<p>TASK constraint type and date, compared by activity code.</p><table><thead><tr><th>Activity</th><th>Previous constraint</th><th>Current constraint</th></tr></thead><tbody>${rows.map(row => `<tr><td><code>${escape(row.code)}</code> ${escape(row.name)}</td><td>${escape(row.before.constraintType)}<br>${escape(row.before.constraintDate)}</td><td>${escape(row.after.constraintType)}<br>${escape(row.after.constraintDate)}</td></tr>`).join('')}</tbody></table>`;
  }
  function select(name, animate = true) {
    if (!review) return;
    selected = name;
    tabs.forEach(tab => { const active = tab.dataset.example === name; tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1; });
    byId('examplePanel').setAttribute('aria-labelledby', `example-${name}`);
    const options = {
      finish: { title: `The finish moves by ${review.finishMovement} days. The update says unchanged.`, claim: (narrative.match(/the key completion date[^.!?]*[.!?]/i)?.[0] || sentence(/key completion date/)).replace(/^the/, 'The'), observation: `Scheduled finish moves by ${review.finishMovement} days.`, question: 'What explains the revised finish?' },
      constraints: { title: `${review.changes.constraintChanges} activities have changed constraints.`, claim: sentence(/programme continues to manage/), observation: 'Constraint types or dates changed between the two exports.', question: 'Which instruction or approval explains each constraint change?' },
      relationships: { title: `${review.changes.relationChanges} relationship signatures changed.`, claim: sentence(/No material change to the delivery sequence/), observation: 'Predecessor links, relationship types or lag values differ.', question: 'What changed in the delivery sequence, and why?' },
    };
    const content = options[name];
    byId('exampleTitle').textContent = content.title;
    byId('exampleClaim').textContent = `“${content.claim}”`;
    byId('exampleObservation').textContent = content.observation;
    byId('exampleQuestion').textContent = content.question;
    byId('exampleComparison').innerHTML = name === 'finish'
      ? `<div><span>Previous finish</span><strong>${escape(date(review.previous.project.scheduledFinish))}</strong></div><div class="example-connection"><span>+${escape(review.finishMovement)} days</span><div class="example-line" aria-hidden="true"></div></div><div><span>Current finish</span><strong>${escape(date(review.current.project.scheduledFinish))}</strong></div>`
      : `<div><span>Previous export</span><strong class="example-file">${escape(review.previous.fileName)}</strong></div><div class="example-connection"><span>Compared</span><div class="example-line" aria-hidden="true"></div></div><div><span>Current export</span><strong class="example-file">${escape(review.current.fileName)}</strong></div>`;
    byId('exampleSource').innerHTML = sourceRows();
    byId('exampleReplay').hidden = name !== 'finish' || reduced.matches;
    if (animate && !reduced.matches) byId('examplePanel').animate([{ opacity: .65 }, { opacity: 1 }], { duration: 180 });
  }
  async function initialise() {
    byId('exampleRetry').hidden = true;
    byId('exampleStatus').textContent = 'Loading the real comparison…';
    try {
      const paths = ['northstar-previous.xer', 'northstar-current.xer', 'northstar-narrative.txt'];
      const values = await Promise.all(paths.map(async path => { const response = await fetch(`demo/${path}`); if (!response.ok) throw new Error('Example unavailable'); return response.text(); }));
      const engine = window.ProjectLensChangeAssurance;
      narrative = values[2];
      review = engine.analyseReview(engine.parseXer(values[0], paths[0]), engine.parseXer(values[1], paths[1]), narrative);
      tabs.forEach(tab => { tab.disabled = false; });
      byId('exampleInspect').disabled = false;
      byId('exampleReplay').disabled = false;
      byId('exampleStatus').textContent = 'Schedule movement is a fact. Its cause still needs evidence.';
      select(selected, false);
    } catch {
      byId('exampleStatus').textContent = 'The interactive example could not load. Retry, or use your own XER files. The preview above is the bundled made-up example.';
      byId('exampleRetry').hidden = false;
    }
  }
  function orientation() { document.querySelector('.example-selector [role="tablist"]').setAttribute('aria-orientation', narrow.matches ? 'horizontal' : 'vertical'); }
  tabs.forEach(tab => {
    tab.addEventListener('click', () => select(tab.dataset.example));
    tab.addEventListener('keydown', event => {
      const nextKeys = narrow.matches ? ['ArrowRight'] : ['ArrowDown'];
      const previousKeys = narrow.matches ? ['ArrowLeft'] : ['ArrowUp'];
      if (![...nextKeys, ...previousKeys, 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      let index = tabs.indexOf(tab);
      if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = tabs.length - 1;
      else index = (index + (nextKeys.includes(event.key) ? 1 : -1) + tabs.length) % tabs.length;
      tabs[index].focus(); select(tabs[index].dataset.example);
    });
  });
  byId('exampleInspect').addEventListener('click', () => {
    const open = byId('exampleSource').hidden;
    byId('exampleSource').hidden = !open;
    byId('exampleInspect').setAttribute('aria-expanded', String(open));
    byId('exampleInspect').textContent = open ? 'Hide source evidence' : 'Inspect source evidence';
  });
  byId('exampleReplay').addEventListener('click', replay);
  byId('exampleRetry').addEventListener('click', initialise);
  document.querySelectorAll('[data-load-example]').forEach(button => button.addEventListener('click', () => byId('loadAssuranceDemo').click()));
  document.querySelectorAll('[data-own-files]').forEach(button => button.addEventListener('click', () => byId('showOwnPack').click()));
  narrow.addEventListener('change', orientation);
  reduced.addEventListener('change', () => { byId('exampleReplay').hidden = reduced.matches || selected !== 'finish'; if (reduced.matches) document.querySelectorAll('.example-panel,.example-line').forEach(element => element.getAnimations().forEach(animation => animation.cancel())); });
  orientation();
  void initialise();
})();
