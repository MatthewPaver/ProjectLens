# Status note — real-input hardening (2 Aug 2026)

## Friction found

Cold-starting the live change-assurance review as a stranger, "Review my
change pack" led to a dead-end: two required XER pickers with nothing that
said what an XER is, that the files are read locally, or where to get a
safe pair to try. The engine already handled user-supplied XER files; the
gap was guidance, not capability.

## What PR #10 fixed

- The intake panel names the format (Primavera P6 export, File → Export →
  Primavera XER) and states that both files are read in the browser and
  never uploaded.
- A one-line sample path: download the safe synthetic Northstar pair and
  choose the files below.
- README "Bring your own XER" section with the exact export steps and the
  caveat that the parser is exercised against the bundled fixtures.

## Still out of scope

No accounts, no sharing, no hosting, no multi-user SaaS. Robustness against
arbitrary real-world XER variants, P6 XML/MPP support and CSV register
guidance remain outside this pass — see Boundaries in the README.
