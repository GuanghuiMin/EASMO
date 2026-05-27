"""Stage 08 — aggregate the 4 spec tables.

Tables:
  table_span_sensitivity_stats.csv          per-task sensitivity stats
  table_behavior_by_method.csv               n_runs / success / steps / tokens per (method, budget)
  table_sensitivity_vs_static_metrics.csv    correlations of metrics with success
  table_top_span_case_studies.csv            top sensitive spans + their behavioral effect
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _safe_mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _entropy(xs):
    """Shannon entropy of a positive vector treated as a distribution."""
    s = sum(xs)
    if s <= 0:
        return 0.0
    p = [x / s for x in xs if x > 0]
    return -sum(pp * math.log(pp) for pp in p)


def _pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = _safe_mean(xs), _safe_mean(ys)
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 <= 0 or sy2 <= 0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sx2 ** 0.5 * sy2 ** 0.5)


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, table_path, read_jsonl,
        load_v3_behavior_runs,
    )

    ensure_outputs()
    spans = read_jsonl(raw_path("history_spans.jsonl"))
    sens = read_jsonl(raw_path("span_sensitivity_scores.jsonl"))
    contexts = read_jsonl(raw_path("compressed_contexts.jsonl"))
    runs_v4 = read_jsonl(raw_path("behavior_runs.jsonl"))
    runs_v3 = load_v3_behavior_runs()

    # ------------------------------------------------------------------
    # Table 1: per-task sensitivity stats
    # ------------------------------------------------------------------
    sens_by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in sens:
        sens_by_task[r["task_id"]].append(r)
    table1 = []
    for tid, rs in sorted(sens_by_task.items()):
        scores = [float(r["final_sensitivity"]) for r in rs]
        if not scores:
            continue
        # Top-3 spans by sensitivity
        rs_sorted = sorted(rs, key=lambda r: -float(r["final_sensitivity"]))[:3]
        top_ids = ";".join(r["span_id"] for r in rs_sorted)
        top_scores = ";".join(f"{float(r['final_sensitivity']):.3f}" for r in rs_sorted)
        table1.append({
            "task_id": tid,
            "num_spans": len(scores),
            "avg_sensitivity": round(_safe_mean(scores), 4),
            "max_sensitivity": round(max(scores), 4),
            "min_sensitivity": round(min(scores), 4),
            "sensitivity_entropy": round(_entropy(scores), 4),
            "top_span_step_ids": top_ids,
            "top_span_scores": top_scores,
        })
    p = table_path("table_span_sensitivity_stats.csv")
    with open(p, "w", newline="") as f:
        if table1:
            w = csv.DictWriter(f, fieldnames=list(table1[0].keys()))
            w.writeheader()
            for r in table1:
                w.writerow(r)
    print(f"[08] Table 1 -> {p}  ({len(table1)} tasks)")

    # ------------------------------------------------------------------
    # Table 2: behavior by method
    # ------------------------------------------------------------------
    # Merge v3 + v4 runs. v3 conditions used (we reuse): task_aware_summary,
    # acon_style_summary, full_context, no_context. v4 contributes the
    # 6 NEW span-based conditions.
    REUSE_FROM_V3 = {
        "task_aware_summary": "task_aware_summary",
        "acon_style_summary": "acon_style_summary",
        "full_context":       "truncated_full_context",
        "no_context":         "no_context",
    }
    by_mb: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in runs_v3:
        if r.get("error"):
            continue
        method_v3 = r["method"]
        if method_v3 not in REUSE_FROM_V3:
            continue
        method_v4 = REUSE_FROM_V3[method_v3]
        cap = int(r.get("budget_max_steps", 0))
        if cap not in (15, 8):
            continue
        by_mb[(method_v4, cap)].append(r)
    for r in runs_v4:
        if r.get("error"):
            continue
        cap = int(r.get("budget_max_steps", 0))
        by_mb[(r["method"], cap)].append(r)

    method_order = [
        "high_sensitivity_spans", "low_sensitivity_spans", "recent_spans",
        "random_spans_seed1", "random_spans_seed2", "random_spans_seed3",
        "task_aware_summary", "acon_style_summary",
        "truncated_full_context", "no_context",
    ]
    # Random spans mean row (computed across 3 seeds).
    table2 = []
    for method in method_order:
        for cap in (15, 8):
            xs = by_mb.get((method, cap), [])
            if not xs:
                continue
            n = len(xs)
            success = sum(1 for r in xs if r.get("success")) / n
            score   = _safe_mean([float(r.get("score", r.get("final_reward", 0))) for r in xs])
            steps   = _safe_mean([int(r.get("iterations", r.get("num_steps", 0))) for r in xs])
            peak    = _safe_mean([int(r.get("input_tokens", r.get("peak_input_tokens", 0))) for r in xs])
            total   = _safe_mean([int(r.get("input_tokens", r.get("total_input_tokens", 0))) for r in xs])
            apis    = _safe_mean([int(r.get("api_call_count", 0)) for r in xs])
            table2.append({
                "method": method,
                "budget": f"{'loose_15' if cap==15 else 'strict_8'}",
                "num_tasks": n,
                "success_rate": round(success, 4),
                "avg_score":    round(score, 4),
                "avg_steps":    round(steps, 2),
                "avg_peak_tokens": round(peak, 0),
                "avg_total_input_tokens": round(total, 0),
                "avg_api_calls": round(apis, 2),
            })

    # Add random_spans_mean row (mean over seed1/2/3 of success).
    for cap in (15, 8):
        rows = [r for r in table2
                if r["method"].startswith("random_spans_seed")
                and r["budget"] == f"{'loose_15' if cap==15 else 'strict_8'}"]
        if not rows:
            continue
        n = max(int(r["num_tasks"]) for r in rows)
        table2.append({
            "method": "random_spans_mean",
            "budget": rows[0]["budget"],
            "num_tasks": n,
            "success_rate": round(_safe_mean([float(r["success_rate"]) for r in rows]), 4),
            "avg_score":    round(_safe_mean([float(r["avg_score"]) for r in rows]), 4),
            "avg_steps":    round(_safe_mean([float(r["avg_steps"]) for r in rows]), 2),
            "avg_peak_tokens": round(_safe_mean([float(r["avg_peak_tokens"]) for r in rows]), 0),
            "avg_total_input_tokens": round(_safe_mean([float(r["avg_total_input_tokens"]) for r in rows]), 0),
            "avg_api_calls": round(_safe_mean([float(r["avg_api_calls"]) for r in rows]), 2),
        })

    p2 = table_path("table_behavior_by_method.csv")
    with open(p2, "w", newline="") as f:
        if table2:
            w = csv.DictWriter(f, fieldnames=list(table2[0].keys()))
            w.writeheader()
            for r in table2:
                w.writerow(r)
    print(f"[08] Table 2 -> {p2}  ({len(table2)} method×budget rows)")

    # ------------------------------------------------------------------
    # Table 3: sensitivity vs static metrics — correlations with task-level success
    # ------------------------------------------------------------------
    # Per-task success at cap=15 (using the same v3+v4 mapping).
    succ_by_task_method: Dict[Tuple[str, str], int] = {}
    for r in runs_v3:
        if r.get("error"): continue
        if r.get("budget_max_steps") != 15: continue
        method_v3 = r["method"]
        method_v4 = REUSE_FROM_V3.get(method_v3)
        if method_v4 is None: continue
        succ_by_task_method[(r["task_id"], method_v4)] = int(bool(r["success"]))
    for r in runs_v4:
        if r.get("error"): continue
        if r.get("budget_max_steps") != 15: continue
        succ_by_task_method[(r["task_id"], r["method"])] = int(bool(r["success"]))

    # Per-task aggregates of metrics
    sens_by_task_avg: Dict[str, float] = {tid: _safe_mean([float(r["final_sensitivity"]) for r in rs])
                                           for tid, rs in sens_by_task.items()}
    sens_by_task_max: Dict[str, float] = {tid: max(float(r["final_sensitivity"]) for r in rs)
                                           for tid, rs in sens_by_task.items() if rs}
    spans_by_task = defaultdict(list)
    for r in spans:
        spans_by_task[r["task_id"]].append(r)
    n_spans_by_task = {tid: len(rs) for tid, rs in spans_by_task.items()}
    tokens_by_task = {tid: sum(int(r.get("token_count", 0)) for r in rs)
                      for tid, rs in spans_by_task.items()}

    # Correlate each metric with high_sensitivity_spans success at cap=15.
    metrics: Dict[str, List[float]] = {
        "decision_state_sensitivity_avg": [],
        "decision_state_sensitivity_max": [],
        "trajectory_token_count": [],
        "num_spans_in_trajectory": [],
    }
    success_high: List[float] = []
    success_recent: List[float] = []
    success_random: List[float] = []
    for tid, _ in sorted(spans_by_task.items()):
        sh = succ_by_task_method.get((tid, "high_sensitivity_spans"))
        sr = succ_by_task_method.get((tid, "recent_spans"))
        sa_seeds = [succ_by_task_method.get((tid, f"random_spans_seed{i}"))
                    for i in (1, 2, 3)]
        sa_seeds = [x for x in sa_seeds if x is not None]
        if sh is None and sr is None and not sa_seeds:
            continue
        success_high.append(float(sh) if sh is not None else 0.0)
        success_recent.append(float(sr) if sr is not None else 0.0)
        success_random.append(_safe_mean([float(x) for x in sa_seeds]) if sa_seeds else 0.0)
        metrics["decision_state_sensitivity_avg"].append(sens_by_task_avg.get(tid, 0.0))
        metrics["decision_state_sensitivity_max"].append(sens_by_task_max.get(tid, 0.0))
        metrics["trajectory_token_count"].append(float(tokens_by_task.get(tid, 0)))
        metrics["num_spans_in_trajectory"].append(float(n_spans_by_task.get(tid, 0)))

    table3 = []
    for k, vals in metrics.items():
        table3.append({
            "metric": k,
            "corr_with_high_sensitivity_success": round(_pearson(vals, success_high), 4),
            "corr_with_recent_spans_success":     round(_pearson(vals, success_recent), 4),
            "corr_with_random_spans_success":     round(_pearson(vals, success_random), 4),
            "n_tasks": len(vals),
        })
    p3 = table_path("table_sensitivity_vs_static_metrics.csv")
    with open(p3, "w", newline="") as f:
        if table3:
            w = csv.DictWriter(f, fieldnames=list(table3[0].keys()))
            w.writeheader()
            for r in table3:
                w.writerow(r)
    print(f"[08] Table 3 -> {p3}  ({len(table3)} metric rows)")

    # ------------------------------------------------------------------
    # Table 4: top span case studies
    # ------------------------------------------------------------------
    span_text_by_key: Dict[Tuple[str, str], str] = {
        (r["task_id"], r["span_id"]): r.get("span_text", "")
        for r in spans
    }
    table4 = []
    for tid, rs in sens_by_task.items():
        if not rs:
            continue
        top = sorted(rs, key=lambda r: -float(r["final_sensitivity"]))[0]
        text = span_text_by_key.get((tid, top["span_id"]), "")[:240]
        table4.append({
            "task_id": tid,
            "top_span_id": top["span_id"],
            "top_span_text_short": text.replace("\n", " "),
            "sensitivity_score": float(top["final_sensitivity"]),
            "changed_decision_fields": ";".join(top.get("judge_changed_fields") or []),
            "judge_severity": top.get("judge_severity", "none"),
            "high_sensitivity_success_15":
                succ_by_task_method.get((tid, "high_sensitivity_spans"), -1),
            "recent_success_15":
                succ_by_task_method.get((tid, "recent_spans"), -1),
            "task_aware_success_15":
                succ_by_task_method.get((tid, "task_aware_summary"), -1),
        })
    p4 = table_path("table_top_span_case_studies.csv")
    with open(p4, "w", newline="") as f:
        if table4:
            w = csv.DictWriter(f, fieldnames=list(table4[0].keys()))
            w.writeheader()
            for r in table4:
                w.writerow(r)
    print(f"[08] Table 4 -> {p4}  ({len(table4)} case studies)")

    # Pretty-print Table 2 for quick eyeballing.
    print()
    print("=== Behavior by method (loose=15 / strict=8) ===")
    print(f"{'method':>26} | {'budget':>10} | {'n':>3} | {'succ':>5} | {'steps':>5} | {'tokens':>7}")
    print("-" * 80)
    for r in table2:
        print(f"{r['method']:>26} | {r['budget']:>10} | {r['num_tasks']:>3} | "
              f"{float(r['success_rate'])*100:>4.0f}% | {r['avg_steps']:>5.1f} | "
              f"{int(r['avg_total_input_tokens']):>7d}")


if __name__ == "__main__":
    main()
