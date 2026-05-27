#!/usr/bin/env bash
# motivation_v4 orchestrator — runs all 10 stages sequentially.

set -e

cd /workspace/EASMO/motivation_v4
mkdir -p outputs/sprint_logs

ACONPY=/workspace/acon/.venv/bin/python
PYBIN=/workspace/EASMO/.venv/bin/python

LOG=outputs/sprint_logs/runall_main.log
echo "==== motivation_v4 orchestrator ====" | tee $LOG
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

run_stage "01_make_spans"           "${PYBIN}  -u scripts/01_make_spans.py"
run_stage "02_reference_probes"     "${PYBIN}  -u scripts/02_reference_probes.py --workers 8"
run_stage "03_ablation_probes"      "${PYBIN}  -u scripts/03_ablation_probes.py --workers 8"
run_stage "04_judge_distance"       "${PYBIN}  -u scripts/04_judge_distance.py --workers 8"
run_stage "05_compute_sensitivity"  "${PYBIN}  -u scripts/05_compute_sensitivity.py"
run_stage "06_compose_contexts"     "${PYBIN}  -u scripts/06_compose_contexts.py"
run_stage "07_run_downstream"       "${ACONPY} -u scripts/07_run_downstream.py --workers 6"
run_stage "08_aggregate_tables"     "${PYBIN}  -u scripts/08_aggregate_tables.py"
run_stage "09_plot_figures"         "${PYBIN}  -u scripts/09_plot_figures.py"
run_stage "10_write_report"         "${PYBIN}  -u scripts/10_write_report.py"

echo
echo "==== motivation_v4 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports | tee -a $LOG
