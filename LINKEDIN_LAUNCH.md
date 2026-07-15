# ProjectLens LinkedIn launch

## Recommended post

An XER can show what changed.

It cannot tell you whether the submission is complete enough to support a decision.

The schedule analytics market already has strong tools for XER ingestion, quality checks, version comparison, critical-path analysis and forecasting. I did not want to build another thinner version of those products.

So I took ProjectLens in a different direction: evidence-linked schedule assurance.

The browser-local workflow now:

- reports what is missing before presenting conclusions
- reduces raw schedule differences to a small material-change queue
- compares current milestones with a separately supplied baseline
- reconciles activity codes with risks and approved or deferred decisions
- challenges the status narrative against the schedule evidence
- checks whether a previous intervention resolved, improved or repeated in the next submission

In the synthetic Northstar demo, it reduces 22 raw changes to 9 material changes and an 8-item executive queue, identifies a 73-day finish movement and reports 9 of 11 evidence classes present. The two missing classes remain visible rather than being silently ignored.

The calculations are deterministic and inspectable. The scores prioritise human review. They are not probabilities of failure, delay attribution or a replacement for validated CPM and quantitative risk analysis.

It runs on GitHub Pages, and imported files stay in the browser.

AI helped me research, build and test the product. The useful part was not generating another summary. It was turning a messy assurance question into a workflow with evidence, limits and a review gate.

What would you want this to challenge before your next schedule approval?

#ProjectDelivery #ProjectControls #DataAnalytics

## First comment

The interactive GitHub Pages demo is here:  
https://matthewpaver.github.io/ProjectLens/

The source code, methodology and official data links are available from the project repository.

## Video structure

The ready-to-post [silent MP4 product walkthrough](docs/assets/projectlens-evidence-demo.mp4) follows this sequence:

1. Open on the assurance question: find the change they did not explain
2. Show the six-part submission and evidence intake
3. Run the synthetic Northstar evidence kit
4. Reveal the 9 of 11 completeness result and 73-day finish movement
5. Show the narrative challenge, exported-float path view, decision trace and baseline comparison
6. Show 22 raw changes reduced to 9 material changes and 8 executive priorities
7. Open one evidence-linked change
8. Show intervention follow-through, the local action log and the model boundary

Use the opening line of the post as the video caption. Avoid adding music unless it is licensed for LinkedIn use.

## Suggested thumbnail text

**What did the XER not tell you?**

Small line: **Evidence-linked schedule assurance, in your browser**
