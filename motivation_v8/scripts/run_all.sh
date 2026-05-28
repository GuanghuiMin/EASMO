#!/usr/bin/env bash
# motivation_v8 orchestrator — Plan B (full config, spec §22).
#
# Env knobs:
#   PYBIN              python with statsmodels (default EASMO/.venv)
#   MODELS             comma list (default qwen,minimax)
#   PROMPT_FAMILIES    comma list (default P1,P2)
#   BUDGET_CHARS       1500
#   ROUNDS             6
#   N_CASES            30 (omit → use all v7 cases)
#   MAX_FACTS_PER_CASE 6
#   N_ITER_CASES       20
#   N_BASIN_CASES      12
#   STAGES             comma list, default 00..09

set -euo pipefail

cd /workspace/EASMO/motivation_v8
mkdir -p outputs/logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
MODELS="${MODELS:-qwen,minimax}"
PROMPT_FAMILIES="${PROMPT_FAMILIES:-P1,P2}"
BUDGET_CHARS="${BUDGET_CHARS:-1500}"
ROUNDS="${ROUNDS:-6}"
MAX_FACTS_PER_CASE="${MAX_FACTS_PER_CASE:-6}"
N_ITER_CASES="${N_ITER_CASES:-20}"
N_BASIN_CASES="${N_BASIN_CASES:-12}"
WANTED="${STAGES:-00,01,02,03,04,05,06,07,08,09}"

N_CASES_FLAG=""; [[ -n "${N_CASES:-}" ]] && N_CASES_FLAG="--max_cases ${N_CASES}"

LOG=outputs/logs/runall_main.log
echo "==== motivation_v8 orchestrator ====" | tee $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG
echo "models=$MODELS  prompts=$PROMPT_FAMILIES  budget=$BUDGET_CHARS  rounds=$ROUNDS" | tee -a $LOG
echo "stages=$WANTED" | tee -a $LOG

run_stage() {
    local name=$1
    local cmd=$2
    if [[ ",${WANTED}," != *",${name%%_*},"* ]]; then
        echo "==== SKIP ${name} ====" | tee -a $LOG
        return
    fi
    echo
    echo "==== ${name} ====" | tee -a $LOG
    echo "Cmd: ${cmd}" | tee -a $LOG
    echo "Started: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a $LOG
    bash -c "${cmd}" 2>&1 | tee "outputs/logs/runall_${name}.log"
    echo "Finished: $(TZ=America/Los_Angeles date +%H:%M:%S)" | tee -a $LOG
}

run_stage "00_prepare_inputs" \
  "${PYBIN} -u scripts/00_prepare_inputs.py ${N_CASES_FLAG}"

run_stage "01_fact_bank" \
  "${PYBIN} -u scripts/01_build_or_reuse_fact_bank.py"

run_stage "02_need_conditions" \
  "${PYBIN} -u scripts/02_build_or_reuse_need_conditions.py"

run_stage "03_single_round" \
  "${PYBIN} -u scripts/03_run_single_round.py \
    --models ${MODELS} --prompt_families ${PROMPT_FAMILIES} \
    --budget_chars ${BUDGET_CHARS} --max_facts_per_case ${MAX_FACTS_PER_CASE} \
    --workers 6"

run_stage "04_iterative" \
  "${PYBIN} -u scripts/04_run_iterative_fixed_points.py \
    --models ${MODELS} --prompt_families ${PROMPT_FAMILIES} \
    --rounds ${ROUNDS} --budget_chars ${BUDGET_CHARS} \
    --n_iter_cases ${N_ITER_CASES} --workers 4"

run_stage "05_basin" \
  "${PYBIN} -u scripts/05_run_basin_experiment.py \
    --models ${MODELS} --rounds ${ROUNDS} --budget_chars ${BUDGET_CHARS} \
    --n_basin_cases ${N_BASIN_CASES} --workers 3"

run_stage "06_retention" \
  "${PYBIN} -u scripts/06_score_retention.py --workers 12"

run_stage "07_metrics" \
  "${PYBIN} -u scripts/07_compute_metrics.py --rounds_cap ${ROUNDS}"

run_stage "08_plots" \
  "${PYBIN} -u scripts/08_plot_figures.py"

run_stage "09_report" \
  "${PYBIN} -u scripts/09_write_report.py"

echo
echo "==== motivation_v8 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports 2>/dev/null | tee -a $LOG || true
