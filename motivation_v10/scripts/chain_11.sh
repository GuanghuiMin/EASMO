#!/usr/bin/env bash
# Run stage 11a (select + segment + chunk-minus stress) → 11b (agent runs)
# → 11c (label + advantage + aggregate) → 12 (auto-write report) sequentially.
#
# All idempotent (resume on partial output).

set -euo pipefail
cd /workspace/EASMO/motivation_v10
mkdir -p outputs/logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
ACONPY="${ACONPY:-/workspace/acon/.venv/bin/python}"

log() { echo "[chain11] $(TZ=America/Los_Angeles date +%H:%M:%S) $*"; }

log "==== 11a select + segment + build chunk-minus contexts ===="
${PYBIN} -u scripts/11a_select_segment_build.py --workers 6 \
  > outputs/logs/stage11a.log 2>&1
log "11a done"

log "==== 11b AppWorld agent runs on chunk-minus contexts ===="
${ACONPY} -u scripts/11b_run_ablation.py --workers 6 \
  > outputs/logs/stage11b.log 2>&1
log "11b done"

log "==== 11c label + advantage + aggregate ===="
${PYBIN} -u scripts/11c_label_and_analyze.py --phase all --workers 6 \
  > outputs/logs/stage11c.log 2>&1
log "11c done"

log "==== 12 re-run auto report (now picks up Claim 4 chunk advantage) ===="
${PYBIN} -u scripts/12_write_report.py \
  > outputs/logs/stage12_report_v2.log 2>&1
log "12 done"

log "==== CHAIN 11 COMPLETE ===="
