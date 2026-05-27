#!/usr/bin/env bash
# motivation_v5 orchestrator — runs all 12 stages sequentially.
#
# Prerequisite: Qwen3-4B vLLM server at http://127.0.0.1:8000/v1
#   bash /workspace/qwen3-vllm/serve.sh  (or with nohup)
# Plus MiniMax at the shared endpoint http://10.183.22.68:8005/v1.

set -e

cd /workspace/EASMO/motivation_v5
mkdir -p outputs/sprint_logs

ACONPY=/workspace/acon/.venv/bin/python
PYBIN=/workspace/EASMO/.venv/bin/python

LOG=outputs/sprint_logs/runall_main.log
echo "==== motivation_v5 orchestrator ====" | tee $LOG
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

run_stage "01_build_raw_cases"        "${PYBIN}  -u scripts/01_build_raw_cases.py"
run_stage "02_sample_cases"           "${PYBIN}  -u scripts/02_sample_cases.py"
run_stage "03_build_audit_augmented"  "${PYBIN}  -u scripts/03_build_audit_augmented.py --workers 4"
run_stage "04_recompress"             "${PYBIN}  -u scripts/04_recompress.py --workers 6"
run_stage "05_rerun_downstream"       "${ACONPY} -u scripts/05_rerun_downstream.py --workers 4"
run_stage "06_run_audit"              "${PYBIN}  -u scripts/06_run_audit.py --workers 4"
run_stage "07_run_verify"             "${PYBIN}  -u scripts/07_run_verify.py --workers 4"
run_stage "08_merge_audits"           "${PYBIN}  -u scripts/08_merge_audits.py"
run_stage "09_aggregate"              "${PYBIN}  -u scripts/09_aggregate.py"
run_stage "10_plot_figures"           "${PYBIN}  -u scripts/10_plot_figures.py"
run_stage "11_write_per_case"         "${PYBIN}  -u scripts/11_write_per_case.py"
run_stage "12_write_motivation"       "${PYBIN}  -u scripts/12_write_motivation.py"

echo
echo "==== motivation_v5 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
echo
ls -la outputs/raw outputs/tables outputs/figures outputs/reports | tee -a $LOG
