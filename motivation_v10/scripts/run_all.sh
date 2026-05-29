#!/usr/bin/env bash
# motivation_v10 orchestrator (spec stage 00-12)
#
# Env knobs:
#   PYBIN              EASMO/.venv python (analysis + LLM clients + SFT)
#   ACONPY             acon/.venv python (AppWorld agent runner)
#   N_SAMPLES          stochastic samples per case (default 8)
#   STRESS_ROUNDS_K    recompression rounds (default 2)
#   CAP_STEPS          AppWorld step budget per agent run (default 15)
#   STAGES             comma list of stages to run (default 00..12)
#   MAX_TEST_N         cap on test_normal slice (default 30)

set -euo pipefail

cd /workspace/EASMO/motivation_v10
mkdir -p outputs/logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
ACONPY="${ACONPY:-/workspace/acon/.venv/bin/python}"
N_SAMPLES="${N_SAMPLES:-8}"
STRESS_ROUNDS_K="${STRESS_ROUNDS_K:-2}"
CAP_STEPS="${CAP_STEPS:-15}"
WANTED="${STAGES:-00,01,02,03,04,05,06,07,08,09,10,11,12}"
MAX_TEST_N="${MAX_TEST_N:-30}"

LOG=outputs/logs/runall_main.log
echo "==== motivation_v10 orchestrator ====" | tee $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG
echo "stages=$WANTED  N_SAMPLES=$N_SAMPLES  K=$STRESS_ROUNDS_K  cap=$CAP_STEPS  max_test_n=$MAX_TEST_N" | tee -a $LOG

run_stage() {
    local name=$1
    local cmd=$2
    local key=${name%%_*}
    if [[ ",${WANTED}," != *",${key},"* ]]; then
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

run_stage "00_prepare" \
  "${PYBIN} -u scripts/00_prepare.py --n_samples ${N_SAMPLES} --stress_rounds_k ${STRESS_ROUNDS_K} --cap_steps ${CAP_STEPS}"

run_stage "01_build_cases" \
  "${ACONPY} -u scripts/01_build_cases.py --max_test_n ${MAX_TEST_N} --max_steps ${CAP_STEPS} --workers 6"

# Stage 02 onward — scripts to be added in subsequent commits.
# Placeholders here keep the orchestrator runnable end-to-end as
# we wire each new stage. SKIP at the env level until then.

run_stage "02_generate_minimax_candidates" \
  "${PYBIN} -u scripts/02_generate_minimax_candidates.py --n_samples ${N_SAMPLES} --workers 6"

run_stage "03_stress_candidates" \
  "${PYBIN} -u scripts/03_stress_candidates.py --rounds ${STRESS_ROUNDS_K} --workers 6"

run_stage "04_behavior_evaluate_candidates" \
  "${ACONPY} -u scripts/04_behavior_evaluate_candidates.py --workers 6 --cap_steps ${CAP_STEPS}"

run_stage "05_proxy_score_candidates" \
  "${PYBIN} -u scripts/05_proxy_score_candidates.py --workers 6"

run_stage "06_proxy_selection_analysis" \
  "${PYBIN} -u scripts/06_proxy_selection_analysis.py"

run_stage "07_construct_teacher_targets" \
  "${PYBIN} -u scripts/07_construct_teacher_targets.py"

run_stage "08_train_qwen_sft" \
  "${PYBIN} -u scripts/08_train_qwen_sft.py --student both"

run_stage "09_evaluate_students" \
  "${ACONPY} -u scripts/09_evaluate_students.py --workers 6 --cap_steps ${CAP_STEPS}"

run_stage "10_grpo_readiness_sampling" \
  "${PYBIN} -u scripts/10_grpo_readiness_sampling.py --n_samples ${N_SAMPLES}"

run_stage "11_chunk_advantage_reanalysis" \
  "${PYBIN} -u scripts/11_chunk_advantage_reanalysis.py --workers 6"

run_stage "12_write_report" \
  "${PYBIN} -u scripts/12_write_report.py"

echo
echo "==== motivation_v10 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports outputs/models 2>/dev/null | tee -a $LOG || true
