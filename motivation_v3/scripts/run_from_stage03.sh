#!/usr/bin/env bash
# Restart from Stage 03 — Stages 01+02 outputs are now correct
# (Stage 02 raw responses re-parsed by 02b_reparse_symbolic). Skip 01+02.

set -e
cd /workspace/EASMO/motivation_v3
mkdir -p outputs/sprint_logs
ACONPY=/workspace/acon/.venv/bin/python
PYBIN=/workspace/EASMO/.venv/bin/python

LOG=outputs/sprint_logs/runall_main.log
echo "" >> $LOG
echo "==== motivation_v3 RESTART FROM STAGE 03 ====" | tee -a $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG

run_stage() {
    local name=$1
    local cmd=$2
    echo
    echo "==== ${name} ====" | tee -a $LOG
    echo "Cmd: ${cmd}" | tee -a $LOG
    echo "Started: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a $LOG
    bash -c "${cmd}" 2>&1 | tee "outputs/sprint_logs/runall_${name}.log"
    echo "Finished: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a $LOG
}

# Wipe stale stage-03+ outputs so the restart is clean
rm -f outputs/motivation_behavioral_evidence.jsonl \
      outputs/motivation_audits.jsonl \
      outputs/motivation_behavior_runs.jsonl \
      outputs/motivation_behavior_runs_with_recovery.jsonl \
      outputs/motivation_recovery_labels.jsonl

run_stage "03_label_evidence"     "${PYBIN}  -u scripts/03_label_evidence.py --workers 8 --max_units_per_task 40"
run_stage "04_audit_compressions" "${PYBIN}  -u scripts/04_audit_compressions.py --workers 6"
run_stage "05_run_downstream"     "${ACONPY} -u scripts/05_run_downstream.py --workers 6"
run_stage "06_label_recovery"     "${PYBIN}  -u scripts/06_label_recovery.py --workers 8 --max_calls_per_run 8"
run_stage "07_aggregate_tables"   "${PYBIN}  -u scripts/07_aggregate_tables.py"
run_stage "08_plot_figures"       "${PYBIN}  -u scripts/08_plot_figures.py"
run_stage "09_write_report"       "${PYBIN}  -u scripts/09_write_report.py"

echo
echo "==== motivation_v3 RESTART FROM STAGE 03 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/ | tee -a $LOG
