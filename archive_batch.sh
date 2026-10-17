#!/usr/bin/env bash
# archive_batch.sh -- archive a finished duel batch's logs, with provenance.
#
# Run from anywhere AFTER a batch has fully ended (not mid-batch).
# It snapshots run_results.log, renames output/logs to
# output/logs-<end-date>, and records the exact git commit the batch
# ran on in a COMMIT.txt inside the archive, so the dataset stays
# attributable to its harness version.
#
# Usage: archive_batch.sh [label]   (label defaults to today's date)

set -euo pipefail

cd "$(dirname "$0")"

LABEL="${1:-$(date +%F)}"
DEST="output/logs-$LABEL"

if pgrep -f "ollama_duel.py" >/dev/null 2>&1; then
    echo "A duel appears to be running -- archive after the batch ends." >&2
    exit 1
fi

if [ ! -d output/logs ]; then
    echo "No output/logs directory -- nothing to archive." >&2
    exit 1
fi

if [ -e "$DEST" ]; then
    echo "$DEST already exists -- refusing to overwrite an archive." >&2
    exit 1
fi

if [ -f run_results.log ]; then
    cp run_results.log "output/run_results-$LABEL.log"
    echo "Snapshotted run_results.log -> output/run_results-$LABEL.log"
fi

mv output/logs "$DEST"
git rev-parse HEAD > "$DEST/COMMIT.txt" 2>/dev/null || echo "unknown (not a git checkout)" > "$DEST/COMMIT.txt"

echo "Archived batch logs -> $DEST (commit: $(cat "$DEST/COMMIT.txt"))"
