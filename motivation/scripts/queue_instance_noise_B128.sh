#!/usr/bin/env bash
# Wait for the B=512 instance_noise rerun to finish, then start B=128.
# This is the second leg of the audit re-run (handoff calls B=128
# "T1's strongest claim").
#
# Started 2026-05-24 17:30 UTC; expected to fire ~3 h later.

set -euo pipefail

WAIT_PID="${WAIT_PID:-539832}"
REPO="/workspace/EASMO"
cd "$REPO/motivation"

echo "[queue_B128] $(date -u +%Y-%m-%dT%H:%MZ) waiting for PID $WAIT_PID (B=512 n=30 rerun) to exit"

# Poll every 60s; tail/wait won't work across sessions because we're not
# the parent of $WAIT_PID.
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done

echo "[queue_B128] $(date -u +%Y-%m-%dT%H:%MZ) PID $WAIT_PID exited; starting B=128 n=30 rerun"

LOG="$REPO/motivation/outputs/instance_noise_rerun_B128_n30.log"

# Detach so this wrapper can exit cleanly
nohup "$REPO/.venv/bin/python" -m scripts.instance_noise_test \
    --config configs/default_longmemeval.yaml \
    --budget 128 --n-contexts 30 --candidates-per-agent 3 \
    > "$LOG" 2>&1 &

NEW_PID=$!
echo "[queue_B128] $(date -u +%Y-%m-%dT%H:%MZ) started B=128 rerun as PID $NEW_PID; log: $LOG"
