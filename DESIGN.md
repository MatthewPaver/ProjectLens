---
name: ProjectLens
description: Clear schedule evidence before a human decision
colors:
  primary: "#0857e8"
  primary-hover: "#0645bd"
  white: "#fff"
  ground: "#fafbfc"
  panel: "#f3f6fa"
  ink: "#0b1933"
  muted: "#4c5b70"
  selection: "#e8f0ff"
  line: "#d5dce6"
  attention: "#805300"
  attention-ground: "#fff6de"
typography:
  display:
    fontFamily: '"Encode Sans", sans-serif'
    fontSize: "61px"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-.035em"
  title:
    fontFamily: '"Encode Sans", sans-serif'
    fontSize: "29px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-.025em"
  body:
    fontFamily: '"Encode Sans", sans-serif'
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: '"Encode Sans", sans-serif'
    fontSize: "13px"
    fontWeight: 500
    lineHeight: 1.4
  code:
    fontFamily: '"JetBrains Mono", monospace'
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1.5
rounded:
  control: "7px"
  field: "6px"
  panel: "12px"
  example: "9px"
  status: "5px"
spacing:
  small: "12px"
  medium: "16px"
  gutter: "24px"
  panel: "28px"
  roomy: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.white}"
    rounded: "{rounded.control}"
    padding: "12px 28px"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
  evidence-panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.panel}"
    padding: "24px"
  status:
    textColor: "{colors.attention}"
    padding: "0"
---

## Overview

**Creative North Star: "Evidence in plain view"**

ProjectLens makes schedule evidence readable before a person records a decision. Cool white surfaces, blue-grey text and cobalt actions frame one continuous evidence workspace. The landing explains the value through a real-engine synthetic comparison; the review and walkthrough use the same visual language at task scale.

This contract covers docs/change-assurance.html, its product-landing.css/js layer, and the normal review, decision, follow-up and guided walkthrough surfaces reached there. GMPP and separate detailed schedule/board pages retain their existing identities. It replaces the former cream/rust guidance for this scoped experience only.

**Key Characteristics:**
- Precise sans typography and restrained cobalt emphasis.
- Continuous evidence workspace with calm tonal panels.
- Explicit human actions and bounded explanatory motion.

## Colors

Primary cobalt identifies actions, active navigation, selection and keyboard focus. The deeper primary is its hover state. Neutral ink carries claims and values; muted text carries supporting explanation. White and ground establish the canvas; panel and selection distinguish grouping and active evidence. Line separates content. Attention and its pale ground label unresolved review work, including the ready state: no green approval signal is implied.

**The Evidence Rule.** Colour communicates interface state, never proof or permission to approve.

## Typography

Encode Sans is self-hosted in docs/assets/fonts at weights 400, 500, 600 and 700; the bundled OFL governs distribution. Display type is reserved for the landing headline. Review headings use the title scale; evidence copy uses compact body type, with metadata generally 12–13px. The landing introduction and claim/observation comparison use larger reading type. Source codes retain the JetBrains Mono/monospace stack. Dates and quantitative comparisons use tabular numerals.

## Layout

The shared maximum content width is 1380px. The header uses max(5vw, 24px) gutters, changing to 18px on phones. The approved landing uses a centered headline and actions above a wide comparison workspace. Its left selector is 280px on large screens and 210px at tablet width. Review findings and source evidence form a two-column desk; phone layouts use a single pane and an explicit return to findings.

At 1101–1535px, selected landing dimensions scale through the existing --pl-u factor, with compact labels bounded at 12px. At 761–1100px, comparison type and selector width compress. At 760px and below, selectors become three horizontal tabs; dates sit side by side above their connection, and claim, observation and question stack. Review navigation occupies its own full-width header row. Forms, decision choices and register rows stack. Do not infer a universal spacing grid: use the recorded spacing tokens and existing component-specific measurements.

## Elevation & Depth

This layer uses flat white and blue-grey surfaces, thin separators and tonal selection rather than decorative gradients or added shadows. Preserve functional inherited overlays. The inactive toast must be hidden with visibility, opacity and pointer-event rules; only its visible status state may occupy visual attention.

## Shapes

Moderately rounded controls and fields sit within larger rounded evidence and decision panels. The comparison frame uses its own slightly tighter radius. Borders are thin and cool; structure remains rectangular rather than pill-shaped.

## Components

- **Actions:** cobalt primary buttons, white text, restrained background hover; secondary actions use white or transparent backgrounds and blue text. Keyboard focus uses a 3px cobalt outline with 4px offset. Disabled buttons retain their disabled behaviour and reduced opacity.
- **Navigation:** native header links preserve real destinations. Active review links use cobalt text and a bottom rule with aria-current. Landing navigation switches to Reviews, Decisions and Follow-up within the operating journey.
- **Evidence selectors:** selected state uses pale blue, cobalt text and a labelled selected/pressed state. Findings retain title, explanation and source context.
- **Evidence panels:** tonal backgrounds lead with the practical finding, dates, quoted narrative and a separately labelled review question. The finish example states the schedule/update contradiction in its heading. The review finding uses “The finish moved N days. The update needs clarification.” while retaining the exact narrative as evidence. Review provenance sits in “Source files and check time”; exact source identity and fields remain in “Inspect source files and fields”, alongside activity or relationship evidence disclosures. File names can wrap.
- **Inputs and decisions:** white fields on tonal panels, visible labels and thin borders. Owner and rationale remain required; conditional decisions retain condition, owner and due date. Persistent errors include recovery guidance.
- **Preview and walkthrough:** the clearly labelled synthetic preview reads the real comparison engine. Factual questions are fixed independently of narrative matching. Replay explicitly draws the finish relationship over 700ms; evidence changes use bounded 180ms transitions. No looping motion. Reduced motion removes animation and replay while retaining facts and relationships. Walkthrough records remain disposable and isolated from ordinary decisions.

**The Human Decision Rule.** Interface hierarchy leads with the finding and next review question, keeps source evidence easy to inspect, and leaves the decision to a person.

## Do's and Don'ts

- Do keep observations, narrative claims, factual questions and human decisions visually distinct.
- Do preserve source facts and both narrative versions when text changes.
- Do use the shared cool palette and self-hosted type across landing, review and walkthrough.
- Do retain meaningful static dates and state when reduced motion is requested.
- Don't use colour as evidence of causation, entitlement or approval.
- Don't let narrative wording remove independent factual review questions.
- Don't add fabricated metrics, customer proof or decorative feature-card marketing.
- Don't extend this identity to separate routes without an explicit redesign.


Implementation authority: docs/product-landing.css and docs/product-landing.js; product truth: PRODUCT.md; approved composition: .impeccable/surfaces/docs-change-assurance-html.md. Finished captures are ../../outputs/projectlens-{product,review,decision}-{desktop,phone}.png. The finish verdict at ../product-finish-verdict.txt resolves desktop clarity and the two phone fixes; it is not a full security or accessibility certification. Local full pytest collection lacked pandas; CI remains the stated verification boundary.
