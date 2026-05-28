#!/usr/bin/env bash
# motivation_v7 orchestrator — Plan B (Qwen + MiniMax, UTCO only).
#
# Knobs (env vars):
#   PYBIN                python with torch + statsmodels (default EASMO/.venv)
#   N_CASES              cap on cases (omit for all v3 cases)
#   MAX_FACTS_PER_CASE   cap on facts per case for stage 04
#   MODELS               comma list (default qwen,minimax)
#   PROMPT_VARIANTS      comma list (default UTCO)
#   BUDGET_CHARS         compression budget (default 1500)
#   ROUNDS               iterative rounds (default 5)
#   STAGES               comma list of stages to run (default all)

set -euo pipefail

cd /workspace/EASMO/motivation_v7
mkdir -p outputs/logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
N_CASES_FLAG=""; [[ -n "${N_CASES:-}" ]] && N_CASES_FLAG="--max_cases ${N_CASES}"
MAX_FACTS_FLAG=""; [[ -n "${MAX_FACTS_PER_CASE:-}" ]] && \
  MAX_FACTS_FLAG="--max_facts_per_case ${MAX_FACTS_PER_CASE}"
MODELS="${MODELS:-qwen,minimax}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-UTCO}"
BUDGET_CHARS="${BUDGET_CHARS:-1500}"
ROUNDS="${ROUNDS:-5}"
WANTED="${STAGES:-00,01,02,03,04,05,06,07,08,09,10}"

LOG=outputs/logs/runall_main.log
echo "==== motivation_v7 orchestrator ====" | tee $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG
echo "models: $MODELS  variants: $PROMPT_VARIANTS  budget: $BUDGET_CHARS  rounds: $ROUNDS" | tee -a $LOG
echo "n_cases: ${N_CASES:-all}  max_facts_per_case: ${MAX_FACTS_PER_CASE:-all}" | tee -a $LOG
echo "stages: $WANTED" | tee -a $LOG

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

run_stage "00_sync_acon" \
  "${PYBIN} -u scripts/00_sync_acon_prompts.py"

run_stage "01_case_pool" \
  "${PYBIN} -u scripts/01_build_case_pool.py ${N_CASES_FLAG}"

run_stage "02_fact_bank" \
  "${PYBIN} -u scripts/02_extract_fact_bank.py --workers 4 ${N_CASES_FLAG}"

run_stage "03_need_conditions" \
  "${PYBIN} -u scripts/03_build_need_conditions.py --workers 6"

run_stage "04_compress" \
  "${PYBIN} -u scripts/04_run_need_conditioned_compression.py \
    --models ${MODELS} --prompt_variants ${PROMPT_VARIANTS} \
    --budget_chars ${BUDGET_CHARS} --workers 6 ${MAX_FACTS_FLAG}"

run_stage "05_score_single" \
  "${PYBIN} -u scripts/05_score_single_round_retention.py --workers 8"

run_stage "06_iter_compress" \
  "${PYBIN} -u scripts/06_run_iterative_compression.py \
    --models ${MODELS} --prompt_variants ${PROMPT_VARIANTS} \
    --rounds ${ROUNDS} --budget_chars ${BUDGET_CHARS} --workers 4"

run_stage "07_score_iter" \
  "${PYBIN} -u scripts/07_score_iterative_survival.py --workers 12"

run_stage "08_metrics" \
  "${PYBIN} -u scripts/08_compute_metrics.py --rounds_cap ${ROUNDS}"

run_stage "09_plots" \
  "${PYBIN} -u scripts/09_plot_figures.py"

run_stage "10_report" \
  "${PYBIN} -u scripts/10_write_report.py"

echo
echo "==== motivation_v7 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports 2>/dev/null | tee -a $LOG || true
