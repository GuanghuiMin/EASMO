#!/usr/bin/env bash
# Background watcher that calls sync_and_push.sh every WATCH_INTERVAL_MIN
# minutes. Use to keep the EASMO repo continuously up to date with
# whatever the in-flight experiments are writing into outputs/ and
# whatever updates land in /workspace/guidances/.
#
# Start:   nohup bash motivation/scripts/auto_push_watcher.sh > /tmp/easmo_watcher.log 2>&1 &
# Stop:    pkill -f auto_push_watcher.sh

set -euo pipefail

WATCH_INTERVAL_MIN="${WATCH_INTERVAL_MIN:-20}"
REPO="/workspace/EASMO"

cd "$REPO"
echo "[$(date -u +%Y-%m-%dT%H:%MZ)] watcher started (interval=${WATCH_INTERVAL_MIN}min)"

while true; do
    sleep $((WATCH_INTERVAL_MIN * 60))
    bash motivation/scripts/sync_and_push.sh "auto-sync" || true
done
