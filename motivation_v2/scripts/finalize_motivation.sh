#!/usr/bin/env bash
# Sprint 4 finalisation — re-runs all canonicalize_* scripts with the
# full extended dataset (Phase 2 + Sprint 3 outputs) and regenerates
# every canonical CSV / JSONL / PDF under outputs/motivation/ and
# figures/motivation/.
#
# Idempotent: safe to re-run as new compute lands.
#
# Usage:
#   bash scripts/finalize_motivation.sh
#
# Expected env:
#   /workspace/EASMO/.venv/bin/python  with matplotlib/numpy/pandas
#
# Outputs are in spec layout (experiment_modification.md §14):
#   outputs/motivation/{hierarchy,multistage_role,behavior_cost,prompted_compression}_{raw.jsonl,summary.csv}
#   outputs/motivation/code_abstraction_{per_line.jsonl,summary.csv}
#   figures/motivation/{hierarchy_b512,hierarchy_by_executor,multistage_role_heatmap,
#                        behavior_success_cap15,behavior_cost_tokens_iters,
#                        prompted_vs_reference_heatmap,prompted_role_recall,
#                        code_abstraction_share}.pdf

set -e

cd /workspace/EASMO/motivation_v2
PYBIN=/workspace/EASMO/.venv/bin/python

echo "============================================================"
echo "Sprint 4 finalisation: rebuilding all canonical outputs."
echo "============================================================"

echo
echo "=== A. hierarchy ==="
$PYBIN scripts/canonicalize_hierarchy.py

echo
echo "=== B. multistage ==="
$PYBIN scripts/canonicalize_multistage.py

echo
echo "=== C. behavior cost (load all 5 transfer_results sources) ==="
EXTRA_ARGS=""
for f in \
    "cap=50:mv2_xtask_ext_existing6_cap50:/workspace/EASMO/motivation_v2/outputs/mv2_xtask_ext_existing6_cap50/transfer_results.jsonl" \
    "cap=15:mv2_xtask_ext_existing6_cap15:/workspace/EASMO/motivation_v2/outputs/mv2_xtask_ext_existing6_cap15/transfer_results.jsonl" \
    "cap=50:mv2_xtask_ext_extra12_cap50:/workspace/EASMO/motivation_v2/outputs/mv2_xtask_ext_extra12_cap50/transfer_results.jsonl" \
    "cap=15:mv2_xtask_ext_extra12_cap15:/workspace/EASMO/motivation_v2/outputs/mv2_xtask_ext_extra12_cap15/transfer_results.jsonl"
do
    path="${f##*:}"
    if [ -f "$path" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --extra_jsonl $f"
    else
        echo "  (skip missing: $path)"
    fi
done
$PYBIN scripts/canonicalize_behavior_cost.py $EXTRA_ARGS

echo
echo "=== D. prompted compression (load all 4 prompted variants) ==="
EXTRA_PROMPTED=""
for spec in \
    "prompted_generic:/workspace/EASMO/motivation_v2/outputs/mv2_pilot_variants/prompted_generic.jsonl" \
    "prompted_task:/workspace/EASMO/motivation_v2/outputs/mv2_pilot_variants/prompted_task.jsonl" \
    "prompted_role:/workspace/EASMO/motivation_v2/outputs/mv2_pilot_variants/prompted_role.jsonl"
do
    path="${spec##*:}"
    if [ -f "$path" ]; then
        EXTRA_PROMPTED="$EXTRA_PROMPTED --prompted_jsonl_extra $spec"
    else
        echo "  (skip missing: $path)"
    fi
done
$PYBIN scripts/canonicalize_prompted.py $EXTRA_PROMPTED

echo
echo "=== D'. code-role abstraction diagnostic ==="
$PYBIN scripts/code_role_abstraction.py $EXTRA_PROMPTED

echo
echo "============================================================"
echo "Outputs:"
ls -la outputs/motivation/
echo
echo "Figures:"
ls -la figures/motivation/
echo "============================================================"
