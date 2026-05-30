#!/usr/bin/env bash
# Run stage 09b stress → 09 phase B agent → 10 GRPO readiness in sequence.
# All three are background-friendly. Idempotent (resume on partial output).

set -euo pipefail
cd /workspace/EASMO/motivation_v10
mkdir -p outputs/logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
ACONPY="${ACONPY:-/workspace/acon/.venv/bin/python}"

log() { echo "[chain] $(TZ=America/Los_Angeles date +%H:%M:%S) $*"; }

log "==== 09b stress students ===="
${PYBIN} -u scripts/09b_stress_students.py --rounds 2 --workers 6 \
  > outputs/logs/stage09b_stress.log 2>&1
log "09b done"

log "==== 09 phase B agent runs ===="
${ACONPY} -u scripts/09_evaluate_students.py \
  --phase B --splits "test_behavior,legacy_v9" --workers 6 \
  > outputs/logs/stage09_phaseB.log 2>&1
log "09 phase B done"

log "==== 10 GRPO readiness sampling (compress phase) ===="
${PYBIN} -u scripts/10_grpo_readiness_sampling.py --phase compress \
  --splits "test_behavior,legacy_v9" --n_samples 8 \
  > outputs/logs/stage10_compress.log 2>&1
log "10 compress done"

log "==== 10 stress ===="
${PYBIN} -u scripts/10_grpo_readiness_sampling.py --phase stress --workers 6 \
  > outputs/logs/stage10_stress.log 2>&1
log "10 stress done"

log "==== 10 score ===="
${PYBIN} -u scripts/10_grpo_readiness_sampling.py --phase score --workers 6 \
  > outputs/logs/stage10_score.log 2>&1
log "10 score done"

log "==== 10 summarize ===="
${PYBIN} -u scripts/10_grpo_readiness_sampling.py --phase summarize \
  > outputs/logs/stage10_summarize.log 2>&1
log "10 summarize done"

log "==== 12 write_report ===="
${PYBIN} -u scripts/12_write_report.py \
  > outputs/logs/stage12_report.log 2>&1
log "12 write_report done"

log "==== CHAIN COMPLETE ===="
