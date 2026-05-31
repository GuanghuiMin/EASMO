#!/usr/bin/env bash
# motivation_v11 orchestrator — runs spec §19 stages 00-16 in sequence.
#
# Env knobs:
#   PYBIN              EASMO/.venv python (LLM stages + analysis)
#   ACONPY             acon/.venv python (AppWorld agent runner: stages 02, 07)
#   N_SAMPLES          N stochastic samples per (case, family); default 8
#   STRESS_ROUNDS_K    recompression rounds; default 2
#   CAP_STEPS          AppWorld step budget per run; default 15
#   TASK_POOL          train+dev (default) | dev
#   ENTROPY_FAMILIES   initial entropy selector families; default "ACON_UTCO"
#                      (extend later to all 4 with --families on stage 08c)
#   STAGES             comma list of stages to run; default 00..16 + 13b
#                      stages: 00 01 02 03 04 05 06 07 08a 08b 08c 09 10 11 12 13 13b 14 15 16
#                      13b = export full candidate bank per spec §12.2
#   WORKERS            parallel workers per stage; default 6
#
# All stages are RESUMABLE — interrupted runs skip already-done items.
# Stage numbering matches spec §19 exactly.

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
WANTED="${STAGES:-00,01,02,03,04,05,06,07,08a,08b,08c,09,10,11,12,13,13b,14,15,16}"

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

run_stage "01_build_task_inventory" \
  "${PYBIN} -u scripts/01_build_task_inventory.py --task_pool ${TASK_POOL}"

run_stage "02_run_full_context_baseline" \
  "${ACONPY} -u scripts/02_run_full_context_baseline.py --max_steps ${CAP_STEPS} --workers ${WORKERS}"

run_stage "03_build_compression_boundaries" \
  "${PYBIN} -u scripts/03_build_compression_boundaries.py"

run_stage "04_render_prompts_and_provenance" \
  "${PYBIN} -u scripts/04_render_prompts_and_provenance.py"

run_stage "05_generate_candidate_compressions" \
  "${PYBIN} -u scripts/05_generate_candidate_compressions.py --n_samples ${N_SAMPLES} --workers ${WORKERS}"

run_stage "06_run_serial_recompression_stress" \
  "${PYBIN} -u scripts/06_run_serial_recompression_stress.py --rounds ${STRESS_ROUNDS_K} --workers ${WORKERS}"

run_stage "07_run_behavior_c1_ck" \
  "${ACONPY} -u scripts/07_run_behavior_c1_ck.py --cap_steps ${CAP_STEPS} --workers ${WORKERS}"

run_stage "08a_pointwise_verifier" \
  "${PYBIN} -u scripts/08a_pointwise_verifier.py --workers ${WORKERS}"

run_stage "08b_pairwise_verifier" \
  "${PYBIN} -u scripts/08b_pairwise_verifier.py --workers ${WORKERS}"

run_stage "08c_continuation_entropy" \
  "${PYBIN} -u scripts/08c_continuation_entropy.py --families ${ENTROPY_FAMILIES} --workers ${WORKERS}"

run_stage "09_compute_selectors" \
  "${PYBIN} -u scripts/09_compute_selectors.py"

run_stage "10_compute_transition_metrics" \
  "${PYBIN} -u scripts/10_compute_transition_metrics.py"

run_stage "11_compute_distribution_quality_calibration" \
  "${PYBIN} -u scripts/11_compute_distribution_quality_calibration.py"

run_stage "12_compute_serial_recompression_metrics" \
  "${PYBIN} -u scripts/12_compute_serial_recompression_metrics.py"

run_stage "13_bootstrap_confidence_intervals" \
  "${PYBIN} -u scripts/13_bootstrap_confidence_intervals.py"

run_stage "13b_export_candidate_bank" \
  "${PYBIN} -u scripts/13b_export_candidate_bank.py"

run_stage "14_plot_figures" \
  "${PYBIN} -u scripts/14_plot_figures.py"

run_stage "15_write_case_studies" \
  "${PYBIN} -u scripts/15_write_case_studies.py"

run_stage "16_write_report" \
  "${PYBIN} -u scripts/16_write_report.py"

echo
echo "==== motivation_v11 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports outputs/data 2>/dev/null | tee -a $LOG || true
