#!/usr/bin/env bash
# Orchestrator for motivation_v3 — runs all 9 stages sequentially.
#
# Stage 01 + 05 are agent runs (use acon venv).
# Stages 02, 03, 04, 06 are LLM-only (use EASMO venv).
# Stages 07, 08, 09 are post-processing.
#
# Logs go to outputs/sprint_logs/runall_<stage>.log

set -e

cd /workspace/EASMO/motivation_v3
mkdir -p outputs/sprint_logs

ACONPY=/workspace/acon/.venv/bin/python
PYBIN=/workspace/EASMO/.venv/bin/python

echo "==== motivation_v3 orchestrator ====" | tee outputs/sprint_logs/runall_main.log
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a outputs/sprint_logs/runall_main.log

run_stage() {
    local name=$1
    local cmd=$2
    echo
    echo "==== ${name} ====" | tee -a outputs/sprint_logs/runall_main.log
    echo "Cmd: ${cmd}" | tee -a outputs/sprint_logs/runall_main.log
    echo "Started: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a outputs/sprint_logs/runall_main.log
    bash -c "${cmd}" 2>&1 | tee -a "outputs/sprint_logs/runall_${name}.log"
    echo "Finished: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a outputs/sprint_logs/runall_main.log
}

run_stage "01_select_consumers"   "${ACONPY} -u scripts/01_select_consumers.py --workers 6 --max_iter 30 --n_target 30"
run_stage "02_build_compressions" "${PYBIN}  -u scripts/02_build_compressions.py --workers 6"
run_stage "03_label_evidence"     "${PYBIN}  -u scripts/03_label_evidence.py --workers 8 --max_units_per_task 40"
run_stage "04_audit_compressions" "${PYBIN}  -u scripts/04_audit_compressions.py --workers 6"
run_stage "05_run_downstream"     "${ACONPY} -u scripts/05_run_downstream.py --workers 6"
run_stage "06_label_recovery"     "${PYBIN}  -u scripts/06_label_recovery.py --workers 8 --max_calls_per_run 8"
run_stage "07_aggregate_tables"   "${PYBIN}  -u scripts/07_aggregate_tables.py"
run_stage "08_plot_figures"       "${PYBIN}  -u scripts/08_plot_figures.py"
run_stage "09_write_report"       "${PYBIN}  -u scripts/09_write_report.py"

echo
echo "==== motivation_v3 orchestrator DONE ====" | tee -a outputs/sprint_logs/runall_main.log
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a outputs/sprint_logs/runall_main.log
ls -la outputs/ | tee -a outputs/sprint_logs/runall_main.log
