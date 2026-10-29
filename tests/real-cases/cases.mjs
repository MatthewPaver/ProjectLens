// Public / OGL tuning cases. Paths resolve under the sibling research_notes folder
// or under this repo. Missing optional real-xer files are skipped (see download script).
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const REPO = path.resolve(__dirname, "../..");
export const RESEARCH = path.resolve(REPO, "../research_notes/Real case tests/projectlens-inputs");

export const CASES = [
  {
    id: "northstar",
    title: "Northstar fixture",
    source: "Repo synthetic demo",
    kind: "real",
    previous: path.join(REPO, "docs/demo/northstar-previous.xer"),
    current: path.join(REPO, "docs/demo/northstar-current.xer"),
    narrative: path.join(REPO, "docs/demo/northstar-narrative.txt"),
    expect: { contradiction: true, daysMoved: 73 },
  },
  {
    id: "land-istar-2025-2026",
    title: "Land ISTAR 2025→2026",
    source: "GMPP / NISTA (OGL)",
    kind: "real",
    previous: path.join(RESEARCH, "gmpp-xer/MOD_0056_2122-Q3_2025-2026/previous.xer"),
    current: path.join(RESEARCH, "gmpp-xer/MOD_0056_2122-Q3_2025-2026/current.xer"),
    narrative: path.join(RESEARCH, "gmpp-xer/MOD_0056_2122-Q3_2025-2026/narrative.txt"),
    expect: { contradiction: true, daysMoved: 1949 },
  },
  {
    id: "e7-wedgetail-2025-2026",
    title: "E7 Wedgetail 2025→2026",
    source: "GMPP / NISTA (OGL)",
    kind: "real",
    previous: path.join(RESEARCH, "gmpp-xer/MOD_0120_2021-Q2_2025-2026/previous.xer"),
    current: path.join(RESEARCH, "gmpp-xer/MOD_0120_2021-Q2_2025-2026/current.xer"),
    narrative: path.join(RESEARCH, "gmpp-xer/MOD_0120_2021-Q2_2025-2026/narrative.txt"),
    expect: { contradiction: true, daysMoved: 913 },
  },
  {
    id: "e7-wedgetail-2021-2022",
    title: "E7 Wedgetail 2021→2022 stated correction",
    source: "GMPP / NISTA (OGL)",
    kind: "real",
    // Built on the fly from gmpp_pairs when the xer folder is absent for this year pair.
    pairFilter: { id: "MOD_0120_2021-Q2", prevYear: 2021, year: 2022 },
    expect: { contradiction: false, daysMoved: 336 },
  },
  {
    id: "r1-airport-stable",
    title: "R1 Terminal 2 synthetic update, stable narrative",
    source: "XER-Splitter airport.xer + synthetic edit",
    kind: "real",
    previous: path.join(RESEARCH, "variants/R1-airport-terminal2/previous.xer"),
    current: path.join(RESEARCH, "variants/R1-airport-terminal2/current.xer"),
    narrative: path.join(RESEARCH, "narratives/r1-stable.txt"),
    expect: { contradiction: true, daysMoved: 28 },
    optional: true,
  },
  {
    id: "crossrail-c1-tfl-jan2018",
    title: "Crossrail C1 TfL Board Jan 2018",
    source: "Reconstructed from NAO / London Assembly (public)",
    kind: "reconstructed",
    previous: path.join(RESEARCH, "crossrail/C1-feb2018-handover-vs-contractor-programmes/previous.xer"),
    current: path.join(RESEARCH, "crossrail/C1-feb2018-handover-vs-contractor-programmes/current.xer"),
    narrative: path.join(RESEARCH, "crossrail/narratives/tfl-board-2018-01-30.txt"),
    expect: { contradiction: true },
    optional: true,
  },
];
