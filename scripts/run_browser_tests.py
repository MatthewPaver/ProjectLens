#!/usr/bin/env python3
"""Run the browser QA suites against a temporary local docs server."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from threading import Thread


ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    "browser_smoke.py",
    "browser_board_readiness.py",
    "browser_change_assurance.py",
    "browser_xer_review.py",
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return


def main() -> None:
    handler = partial(QuietHandler, directory=ROOT / "docs")
    server = ThreadingHTTPServer(("127.0.0.1", 8765), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for suite in SUITES:
            subprocess.run(
                [sys.executable, str(ROOT / "Processing" / "tests" / suite)],
                cwd=ROOT,
                check=True,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
