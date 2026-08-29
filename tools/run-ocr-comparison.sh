#!/usr/bin/env bash
# Extraction through each cached OCR engine, on the same documents, same model, same
# prompts. Only the engine varies -- the only comparison that ranks engines.
#
# The cohort is NAMED, not re-derived. `--limit 15` stood in for it twice and picked a
# different 75 both times: it overlaps the cached set on 52 documents, and
# data/sample75.txt overlaps it on 1. Three different "75-document sets" are in
# circulation here, and a comparison across two of them would look entirely normal.
#
# docTR is re-run rather than read from reports/doctr75.jsonl, which predates the corpus
# rebuild to ten page designs per type. report.json was overstating field accuracy by
# three and a half points for exactly that reason.
#
# Sequential: three extractions against one endpoint would measure its queueing.
set -uo pipefail
cd /c/Users/joey/Documents/Codex/DocumentIntelligence || exit 1

set -a
. ./.env
set +a

export DI_NORMALIZED_DIR=data/normalized
export DI_CACHED_CORPUS=data/degraded
export DI_REPORTS_DIR=reports

# The cohort is tracked in git alongside this script. It is 75 filenames, not
# corpus data, and it is the one thing that must not drift between engines.
SET="$(dirname "$0")/ocr-cohort-75.txt"
test -s "$SET" || { echo "cohort file missing or empty: $SET" >&2; exit 1; }
echo "cohort: $(wc -l < "$SET") documents"

run_one () {
  local engine="$1" out="$2"
  echo "=== $engine -> $out ==="
  DI_CACHED_ENGINE="$engine" .venv/Scripts/python.exe -m extract.cli run \
      --corpus data/degraded --normalizer cached --files "$SET" \
      --out "$out" --concurrency 2
  echo "=== $engine exit $? ==="
}

run_one doctr            ocr-doctr75.jsonl
run_one paddle           ocr-paddle75.jsonl
run_one paddle-pp-ocrv6  ocr-paddlev6-75.jsonl
echo "ALL DONE"
