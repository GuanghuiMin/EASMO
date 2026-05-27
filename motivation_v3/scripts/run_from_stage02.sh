#!/usr/bin/env bash
# Restart from Stage 02 — Stage 01's outputs (motivation_full_trajectories.jsonl
# and the per-task acon trajectory dirs) are reusable; bug only affects
# stages 02-06. Skips 01 to save ~17 min.

set -e

cd /workspace/EASMO/motivation_v3
mkdir -p outputs/sprint_logs

ACONPY=/workspace/acon/.venv/bin/python
PYBIN=/workspace/EASMO/.venv/bin/python

LOG=outputs/sprint_logs/runall_main.log
echo "" >> $LOG
echo "==== motivation_v3 RESTART FROM STAGE 02 ====" | tee -a $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG

run_stage() {
    local name=$1
    local cmd=$2
    echo
    echo "==== ${name} ====" | tee -a $LOG
    echo "Cmd: ${cmd}" | tee -a $LOG
    echo "Started: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a $LOG
    bash -c "${cmd}" 2>&1 | tee -a "outputs/sprint_logs/runall_${name}.log"
    echo "Finished: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a $LOG
}

run_stage "02_build_compressions" "${PYBIN}  -u scripts/02_build_compressions.py --workers 6"
run_stage "03_label_evidence"     "${PYBIN}  -u scripts/03_label_evidence.py --workers 8 --max_units_per_task 40"
run_stage "04_audit_compressions" "${PYBIN}  -u scripts/04_audit_compressions.py --workers 6"
run_stage "05_run_downstream"     "${ACONPY} -u scripts/05_run_downstream.py --workers 6"
run_stage "06_label_recovery"     "${PYBIN}  -u scripts/06_label_recovery.py --workers 8 --max_calls_per_run 8"
run_stage "07_aggregate_tables"   "${PYBIN}  -u scripts/07_aggregate_tables.py"
run_stage "08_plot_figures"       "${PYBIN}  -u scripts/08_plot_figures.py"
run_stage "09_write_report"       "${PYBIN}  -u scripts/09_write_report.py"

echo
echo "==== motivation_v3 RESTART DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/ | tee -a $LOG
