#!/usr/bin/env bash
# Optional download of public sample XERs used by the real-case suite.
# Skipped tests stay skipped until these files exist under research_notes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REAL_XER_DIR:-$ROOT/../research_notes/Real case tests/projectlens-inputs/real-xer}"
echo "Download destinations are documented in the research notes real-xer folders."
echo "Set REAL_XER_DIR and place licensed public samples under: $DEST"
echo "This script does not fetch automatically (licence terms vary by upstream)."
exit 0
