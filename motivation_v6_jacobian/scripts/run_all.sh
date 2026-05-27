#!/usr/bin/env bash
# motivation_v6_jacobian orchestrator — runs A/B/C stages with the
# white-box Qwen3-4B-Instruct-2507, then optional Experiment D.
#
# Driver knobs (override via environment):
#   PYBIN              python that has torch + transformers (default EASMO/.venv)
#   ACONPY             python that can import appworld (default acon/.venv)
#   QWEN_MODEL_PATH    HF model path or local snapshot path
#   MAX_CONTEXT_TOKENS truncation for the white-box probe forward
#   N_CASES            cap on number of tasks (omit to use all)
#   LAYER_INDEX        layer at which to capture H + grad (default N/2)
#   SOFT_KS            comma-separated soft-token counts
#   SOFT_STEPS         soft-token optimisation steps
#   RUN_EXP_D          set "1" to run gradient-ranked downstream
set -euo pipefail

cd /workspace/EASMO/motivation_v6_jacobian
mkdir -p outputs/sprint_logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
ACONPY="${ACONPY:-/workspace/acon/.venv/bin/python}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-12000}"
N_CASES_FLAG=""
if [[ -n "${N_CASES:-}" ]]; then
    N_CASES_FLAG="--max_cases ${N_CASES}"
fi
LAYER_INDEX_FLAG=""
if [[ -n "${LAYER_INDEX:-}" ]]; then
    LAYER_INDEX_FLAG="--layer_index ${LAYER_INDEX}"
fi
SOFT_KS="${SOFT_KS:-4,8,16,32,64}"
SOFT_STEPS="${SOFT_STEPS:-200}"
RUN_EXP_D="${RUN_EXP_D:-0}"

LOG=outputs/sprint_logs/runall_main.log
echo "==== motivation_v6_jacobian orchestrator ====" | tee $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG
echo "model: $QWEN_MODEL_PATH" | tee -a $LOG
echo "max_context_tokens: $MAX_CONTEXT_TOKENS" | tee -a $LOG
echo "soft_ks: $SOFT_KS  soft_steps: $SOFT_STEPS" | tee -a $LOG
echo "n_cases: ${N_CASES:-all}  layer_index: ${LAYER_INDEX:-default(N/2)}" | tee -a $LOG
echo "run_exp_d: $RUN_EXP_D" | tee -a $LOG

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

run_stage "01_build_cases" \
  "${PYBIN} -u scripts/01_build_cases.py ${N_CASES_FLAG}"

run_stage "02_jacobian_saliency" \
  "${PYBIN} -u scripts/02_compute_jacobian_saliency.py \
     --model_path '${QWEN_MODEL_PATH}' \
     --cases outputs/raw/cases.jsonl \
     --max_context_tokens ${MAX_CONTEXT_TOKENS} \
     --capture_active \
     ${LAYER_INDEX_FLAG} \
     ${N_CASES_FLAG} \
     --gradient_checkpointing"

run_stage "03_compare_to_v4" \
  "${PYBIN} -u scripts/03_compare_to_v4_sensitivity.py"

run_stage "04_active_subspace" \
  "${PYBIN} -u scripts/04_active_subspace_spectrum.py"

run_stage "05_soft_token_oracle" \
  "${PYBIN} -u scripts/05_soft_token_oracle.py \
     --model_path '${QWEN_MODEL_PATH}' \
     --cases outputs/raw/cases.jsonl \
     --ks ${SOFT_KS} --num_steps ${SOFT_STEPS} \
     --max_context_tokens ${MAX_CONTEXT_TOKENS} \
     ${N_CASES_FLAG}"

run_stage "06_aggregate" \
  "${PYBIN} -u scripts/06_aggregate.py"

run_stage "07_plot" \
  "${PYBIN} -u scripts/07_plot.py"

run_stage "08_write_report" \
  "${PYBIN} -u scripts/08_write_report.py \
     --model_path '${QWEN_MODEL_PATH}' \
     --max_context_tokens ${MAX_CONTEXT_TOKENS} \
     ${LAYER_INDEX_FLAG}"

if [[ "${RUN_EXP_D}" == "1" ]]; then
    run_stage "09_compose_jacobian_contexts" \
      "${PYBIN} -u scripts/09_compose_jacobian_contexts.py"
    run_stage "10_run_jacobian_downstream" \
      "${ACONPY} -u scripts/10_run_jacobian_downstream.py --workers 6"
    run_stage "11_summarise_jacobian_downstream" \
      "${PYBIN} -u scripts/11_summarise_jacobian_downstream.py"
fi

echo
echo "==== motivation_v6_jacobian DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports 2>/dev/null | tee -a $LOG || true
