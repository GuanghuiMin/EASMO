#!/usr/bin/env bash
# Sequence two capped-budget cross-task transfer runs back-to-back.
# Setup A1: max_iter=15 (sweet-spot pick from existing iter distribution)
# Setup A2: max_iter=8  (aggressive cap; stress-tests the differential)
#
# Both reuse the existing 6-consumer × 4-condition × 3-budget xtask design
# (see run_cross_task_transfer.py). Outputs go to separate dirs so neither
# clobbers the original max_iter=50 run at outputs/mv2_xtask/.
set -euo pipefail

REPO="/workspace/EASMO"
SCRIPT="${REPO}/motivation_v2/scripts/run_cross_task_transfer.py"
PY="/workspace/acon/.venv/bin/python"

cd "${REPO}/motivation_v2"

mkdir -p outputs/mv2_xtask_cap15 outputs/mv2_xtask_cap8

echo "============================================="
echo "[$(date -u +%Y-%m-%dT%H:%MZ)] STARTING Setup A1 — max_iter=15"
echo "============================================="
"${PY}" "${SCRIPT}" \
    --workers 4 \
    --tag mv2_xtask_cap15 \
    --max_iter 15 \
    --output_dir outputs/mv2_xtask_cap15 \
    > outputs/mv2_xtask_cap15/driver.log 2>&1

echo "============================================="
echo "[$(date -u +%Y-%m-%dT%H:%MZ)] Setup A1 done. STARTING Setup A2 — max_iter=8"
echo "============================================="
"${PY}" "${SCRIPT}" \
    --workers 4 \
    --tag mv2_xtask_cap8 \
    --max_iter 8 \
    --output_dir outputs/mv2_xtask_cap8 \
    > outputs/mv2_xtask_cap8/driver.log 2>&1

echo "============================================="
echo "[$(date -u +%Y-%m-%dT%H:%MZ)] Both setups complete."
echo "============================================="
echo "Setup A1 (max_iter=15) results: outputs/mv2_xtask_cap15/transfer_results.jsonl"
echo "Setup A2 (max_iter=8)  results: outputs/mv2_xtask_cap8/transfer_results.jsonl"
