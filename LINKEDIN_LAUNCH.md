# Project Evidence Desk: LinkedIn series

## The story

This is not a series about four AI applications. It follows one project-delivery problem:

> The board pack looks complete, but the commitments, schedule, risks and previous decisions do not agree.

The series moves through one recognisable week in the life of a project:

1. a meeting creates commitments;
2. the board pack rolls some forward and loses others;
3. the schedule and narrative conflict;
4. the board asks what happened on similar projects;
5. a person decides with conditions;
6. the next review checks whether those conditions worked.

The product is **Project Evidence Desk**. MeetingProof, ProjectLens and DecisionGraph are modules in that story, not unrelated products the reader must understand first.

## Series rules

- Lead with the delivery problem, not LangGraph, LangSmith, MCP or an agent framework.
- Use Northstar throughout so readers recognise the same project and facts.
- Show one source, one finding and one human action in every post.
- Name synthetic fixtures clearly.
- Explain where code, a language model and a person each have authority.
- End with a practical question a project professional can answer from experience.
- Put implementation detail in the first comment or a later tutorial.

## Post 1 — The board pack is not the evidence

### Draft

Most project-board meetings should not begin by reconciling the pack.

But that is often what happens.

The status page says Green. The latest schedule moved the finish by 73 days. A high risk has no owner. An approval condition from the previous meeting is still open, but it has disappeared from the current narrative.

Each record can look reasonable on its own. The problem appears when they are read together.

I built a browser-local **Project Evidence Desk** to test that moment before the meeting. It brings together:

- the current and previous board pack
- RAID
- actions and decisions
- schedule evidence
- prior approval conditions

The output is not an AI summary. It is a short, source-linked board brief:

- what materially changed
- what conflicts
- what is missing or unowned
- which decisions the board can make
- which decisions still lack evidence

In the Northstar example, the correct opening question is not “Are we still Green?”

It is “What evidence would make any status supportable?”

The example is synthetic and safe to share. The checks are deterministic and inspectable. A named person must review every finding before the brief can be exported.

What does your team still discover during the meeting that should have been found before it?

#ProjectDelivery #ProjectControls #Governance

### First comment

Try the board-readiness example:

https://matthewpaver.github.io/ProjectLens/board-readiness.html

The wider public-data view uses official UK Government Major Projects Portfolio releases. Northstar is a separate, clearly labelled synthetic governance pack used to test the end-to-end workflow.

## Post 2 — A commitment without a source becomes folklore

Use MeetingProof and the same Northstar meeting.

Hook:

> “We agreed that last week” is not a usable project control.

Show:

- the decision line in the notes
- an action with a missing owner and due date
- the human review gate
- how the reviewed item later appears in board readiness

Practical question:

> Which meeting commitment causes the most avoidable rework when its wording or owner is unclear?

## Post 3 — Green narrative, 73-day movement

Use the change-assurance and detailed XER modules.

Hook:

> The schedule did not contradict the narrative. The two records had simply never been compared.

Show:

- previous and current finish
- the exact narrative claim
- three bounded blockers
- evidence request with owner and due date

Keep forecasting and “AI insight” out of the hook. The value is deterministic reconciliation.

## Post 4 — What happened last time?

Use DecisionGraph.

Hook:

> A lessons-learned document is not organisational memory if nobody can retrieve it at the next decision.

Show:

- three comparable synthetic cases
- why each matched
- intervention
- measured schedule and cost outcome
- the evidence boundary

Do not call the result “strong evidence.” Say exactly how many comparable cases were used.

## Post 5 — Approval is the start of follow-up

Use the decision and condition registers.

Hook:

> “Approved with conditions” is often where accountability goes to disappear.

Show:

- named decision authority
- unresolved warnings preserved
- condition, owner and date
- open/closed state at the next review

Practical question:

> Where does your team keep approval conditions after the minutes are issued?

## Post 6 — Where an LLM belongs

This is the technical reveal after the audience understands the problem.

Hook:

> I do not want an LLM deciding whether a project is Green.

Explain the boundary:

- code owns dates, counts, status rules and source validation
- a bounded model can propose structure or plain-language wording
- LangGraph stops the workflow for human review
- LangSmith measures unsupported claims, missed items, edits, cost and latency
- the named authority owns the decision

Link to the MeetingProof repository and evaluations.

## Tutorial follow-ons

Publish tutorials as a separate “build the control” strand:

1. Connect a sample Obsidian vault to Claude through MCP without exposing the real vault.
2. Build a LangGraph interrupt that cannot silently skip approval.
3. Use LangSmith to test evidence links and approval bypasses rather than subjective summary quality.
4. Parse two Primavera P6 XER files in the browser and compare only observable fields.
5. Import an organisational decision history without uploading it.
6. Use multi-agent orchestration only where isolated research or review roles improve the evidence trail.

The tutorial should always begin with the operational control it implements. Ruflo, MCP, LangGraph and LangSmith are implementation choices, not the product proposition.
