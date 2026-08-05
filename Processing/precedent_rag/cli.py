"""CLI helpers: run a query, or score the eval set against Gemini hybrid retrieve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# make `python -m Processing.precedent_rag.cli` work from the repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Processing.precedent_rag.cases import load_eval_queries  # noqa: E402
from Processing.precedent_rag.graph import run_precedent_query  # noqa: E402


def cmd_query(args: argparse.Namespace) -> int:
    result = run_precedent_query(
        {
            "problem": args.problem,
            "sector": args.sector,
            "phase": args.phase,
            "type": args.type,
        },
        limit=args.limit,
        summarise=not args.no_summary,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Check that gold case ids land in the top-k — this is the measurement gate."""
    queries = load_eval_queries()
    hits = 0
    rows = []
    for item in queries:
        result = run_precedent_query(
            {
                "problem": item["problem"],
                "sector": item.get("sector"),
                "phase": item.get("phase"),
                "type": item.get("type"),
            },
            limit=args.limit,
            summarise=False,
        )
        got = [case["id"] for case in result["cases"]]
        expected = set(item.get("must_include_any") or [])
        ok = bool(expected.intersection(got))
        hits += int(ok)
        rows.append({"id": item["id"], "ok": ok, "got": got, "expected_any": sorted(expected)})
        mark = "PASS" if ok else "FAIL"
        print(f"{mark} {item['id']}: got={got} expected_any={sorted(expected)}")

    total = len(queries) or 1
    rate = hits / total
    print(f"\nHit rate: {hits}/{total} = {rate:.0%}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"hit_rate": rate, "rows": rows}, indent=2))
    # soft gate for a 24-case corpus — raise later once the eval set grows
    return 0 if rate >= 0.66 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ProjectLens cited precedent RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Run one precedent query")
    q.add_argument("--problem", required=True)
    q.add_argument("--sector", default=None)
    q.add_argument("--phase", default=None)
    q.add_argument("--type", default=None)
    q.add_argument("--limit", type=int, default=5)
    q.add_argument("--no-summary", action="store_true")
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval", help="Score eval_queries.json against hybrid retrieve")
    e.add_argument("--limit", type=int, default=5)
    e.add_argument("--json-out", default=None)
    e.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("GEMINI_API_KEY missing — set it in ProjectLens/.env", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
