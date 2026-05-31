#!/usr/bin/env bash
# motivation_v11 orchestrator — runs stages 00-13 in sequence.
#
# Env knobs:
#   PYBIN            EASMO/.venv python (LLM stages + analysis)
#   ACONPY           acon/.venv python (AppWorld agent runner: stages 01, 05)
#   N_SAMPLES        N stochastic samples per (case, family); default 8
#   STRESS_ROUNDS_K  recompression rounds; default 2
#   CAP_STEPS        AppWorld step budget per run; default 15
#   TASK_POOL        train+dev (default) | dev | train+dev+test_normal
#   ENTROPY_FAMILIES initial entropy selector families; default "ACON_UTCO"
#                    (extend later to all 4 with --families ACON_UT,...)
#   STAGES           comma list of stages to run (default 00..13)
#   WORKERS          parallel workers per stage; default 6
#
# All stages are RESUMABLE — interrupted runs skip already-done items.

set -euo pipefail
cd /workspace/EASMO/motivation_v11
mkdir -p outputs/logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
ACONPY="${ACONPY:-/workspace/acon/.venv/bin/python}"
N_SAMPLES="${N_SAMPLES:-8}"
STRESS_ROUNDS_K="${STRESS_ROUNDS_K:-2}"
CAP_STEPS="${CAP_STEPS:-15}"
TASK_POOL="${TASK_POOL:-train+dev}"
ENTROPY_FAMILIES="${ENTROPY_FAMILIES:-ACON_UTCO}"
WORKERS="${WORKERS:-6}"
WANTED="${STAGES:-00,01,02,03,04,05,06a,06b,06c,07,08,09,10,11,12,13}"

LOG=outputs/logs/runall_main.log
echo "==== motivation_v11 orchestrator ====" | tee $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG
echo "task_pool=$TASK_POOL  N_SAMPLES=$N_SAMPLES  K=$STRESS_ROUNDS_K  cap=$CAP_STEPS" | tee -a $LOG
echo "entropy_families=$ENTROPY_FAMILIES  workers=$WORKERS  stages=$WANTED" | tee -a $LOG

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
  "${PYBIN} -u scripts/00_prepare.py --n_samples ${N_SAMPLES} --stress_rounds_k ${STRESS_ROUNDS_K} --cap_steps ${CAP_STEPS} --task_pool ${TASK_POOL}"

run_stage "01_build_full_dev_cases" \
  "${ACONPY} -u scripts/01_build_full_dev_cases.py --task_pool ${TASK_POOL} --max_steps ${CAP_STEPS} --workers ${WORKERS}"

run_stage "02_render_prompts" \
  "${PYBIN} -u scripts/02_render_prompts.py"

run_stage "03_generate_candidates" \
  "${PYBIN} -u scripts/03_generate_candidates.py --n_samples ${N_SAMPLES} --workers ${WORKERS} --cases data/v11_secondary_all_cases.jsonl"

run_stage "04_serial_recompression_stress" \
  "${PYBIN} -u scripts/04_serial_recompression_stress.py --rounds ${STRESS_ROUNDS_K} --workers ${WORKERS}"

run_stage "05_run_behavior_c1_ck" \
  "${ACONPY} -u scripts/05_run_behavior_c1_ck.py --cap_steps ${CAP_STEPS} --workers ${WORKERS}"

run_stage "06a_pointwise_verifier" \
  "${PYBIN} -u scripts/06a_pointwise_verifier.py --workers ${WORKERS}"

run_stage "06b_pairwise_verifier" \
  "${PYBIN} -u scripts/06b_pairwise_verifier.py --workers ${WORKERS}"

run_stage "06c_continuation_entropy" \
  "${PYBIN} -u scripts/06c_continuation_entropy.py --families ${ENTROPY_FAMILIES} --workers ${WORKERS}"

run_stage "07_selection_analysis" \
  "${PYBIN} -u scripts/07_selection_analysis.py"

run_stage "08_distribution_quality_calibration" \
  "${PYBIN} -u scripts/08_distribution_quality_calibration.py"

run_stage "09_stress_invariance_analysis" \
  "${PYBIN} -u scripts/09_stress_invariance_analysis.py"

run_stage "10_pass_at_n_curve" \
  "${PYBIN} -u scripts/10_pass_at_n_curve.py"

run_stage "11_build_candidate_bank" \
  "${PYBIN} -u scripts/11_build_candidate_bank.py"

run_stage "12_plot_figures" \
  "${PYBIN} -u scripts/12_plot_figures.py"

run_stage "13_write_report" \
  "${PYBIN} -u scripts/13_write_report.py"

echo
echo "==== motivation_v11 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports outputs/data 2>/dev/null | tee -a $LOG || true
