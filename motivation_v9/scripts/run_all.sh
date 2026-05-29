#!/usr/bin/env bash
# motivation_v9 orchestrator — full config (spec §22), MiniMax-only track.
#
# Env knobs:
#   PYBIN              EASMO/.venv python (for LLM stages 00-03, 05-08, 09a, 10-14)
#   ACONPY             acon/.venv python (for AppWorld runner: stages 04, 09)
#   MODELS             comma list of compressors (default 'minimax')
#   N_SAMPLES          number of stochastic samples per (case, model) (default 8)
#   STRESS_ROUNDS_K    repeated-compression rounds (default 2)
#   N_CASES            cap on cases (omit to use all 30 from v3)
#   STAGES             comma list of stages to run (default 00..14)
#   SKIP_CHUNK         set to "1" to skip stages 07-13 (chunk analysis)
#   CHUNK_MAX_CASES    cap for stage 07 chunk-case selection (default 12)
#   CHUNK_MAX_CHUNKS   cap for stage 08 chunks-per-case (default 12)

set -euo pipefail

cd /workspace/EASMO/motivation_v9
mkdir -p outputs/logs

PYBIN="${PYBIN:-/workspace/EASMO/.venv/bin/python}"
ACONPY="${ACONPY:-/workspace/acon/.venv/bin/python}"
MODELS="${MODELS:-minimax}"
N_SAMPLES="${N_SAMPLES:-8}"
STRESS_ROUNDS_K="${STRESS_ROUNDS_K:-2}"
N_CASES="${N_CASES:-}"
WANTED="${STAGES:-00,01,02,03,04,05,06,07,08,09a,09,10,11,12,13,14}"
SKIP_CHUNK="${SKIP_CHUNK:-0}"
CHUNK_MAX_CASES="${CHUNK_MAX_CASES:-12}"
CHUNK_MAX_CHUNKS="${CHUNK_MAX_CHUNKS:-12}"

N_CASES_FLAG=""; [[ -n "$N_CASES" ]] && N_CASES_FLAG="--max_cases ${N_CASES}"

LOG=outputs/logs/runall_main.log
echo "==== motivation_v9 orchestrator ====" | tee $LOG
echo "Started: $(TZ=America/Los_Angeles date)" | tee -a $LOG
echo "models=$MODELS  N_SAMPLES=$N_SAMPLES  K=$STRESS_ROUNDS_K  stages=$WANTED" | tee -a $LOG
echo "skip_chunk=$SKIP_CHUNK" | tee -a $LOG

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
  "${PYBIN} -u scripts/00_prepare.py --n_samples ${N_SAMPLES} --stress_rounds_k ${STRESS_ROUNDS_K}"

run_stage "01_build_cases" \
  "${PYBIN} -u scripts/01_build_cases.py ${N_CASES_FLAG}"

run_stage "02_generate_candidates" \
  "${PYBIN} -u scripts/02_generate_candidates.py --models ${MODELS} --n_samples ${N_SAMPLES} --workers 6"

run_stage "03_stress_recompress" \
  "${PYBIN} -u scripts/03_stress_recompress.py --rounds ${STRESS_ROUNDS_K} --workers 6"

run_stage "04_behavior_c1_ck" \
  "${ACONPY} -u scripts/04_run_behavior_c1_ck.py --workers 6"

run_stage "05_best_of_n" \
  "${PYBIN} -u scripts/05_compute_best_of_n.py"

run_stage "06_c1_ck_fragility" \
  "${PYBIN} -u scripts/06_compute_c1_ck_fragility.py"

if [[ "${SKIP_CHUNK}" != "1" ]]; then
    run_stage "07_select_chunk_cases" \
      "${PYBIN} -u scripts/07_select_chunk_cases.py --max_cases ${CHUNK_MAX_CASES}"

    run_stage "08_segment_chunks" \
      "${PYBIN} -u scripts/08_segment_chunks.py --max_chunks ${CHUNK_MAX_CHUNKS}"

    run_stage "09a_build_chunk_contexts" \
      "${PYBIN} -u scripts/09a_build_chunk_contexts.py --rounds ${STRESS_ROUNDS_K} --workers 6"

    run_stage "09_chunk_ablation_behavior" \
      "${ACONPY} -u scripts/09_run_chunk_ablation.py --workers 6"

    run_stage "10_chunk_advantage" \
      "${PYBIN} -u scripts/10_compute_chunk_advantage.py"

    run_stage "11_chunk_labels" \
      "${PYBIN} -u scripts/11_label_chunks_minimax.py --workers 6"

    run_stage "12_chunk_advantage_by_type" \
      "${PYBIN} -u scripts/12_analyze_chunk_advantage_by_type.py"
fi

run_stage "13_plot_figures" \
  "${PYBIN} -u scripts/13_plot_figures.py"

run_stage "14_write_report" \
  "${PYBIN} -u scripts/14_write_report.py"

echo
echo "==== motivation_v9 DONE ====" | tee -a $LOG
echo "Finished: $(TZ=America/Los_Angeles date)" | tee -a $LOG
ls -la outputs/raw outputs/tables outputs/figures outputs/reports 2>/dev/null | tee -a $LOG || true
