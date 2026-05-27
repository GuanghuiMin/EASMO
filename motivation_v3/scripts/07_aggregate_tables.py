"""Stage 7 — aggregate the 3 spec tables.

Table 1 (Summary vs Symbolic Compactness):
  method, avg_tokens, avg_ids_preserved, avg_bindings_preserved,
  avg_constraints_preserved, avg_action_outcomes_preserved

Table 2 (Behavioral Evidence Coverage):
  method, behavioral_evidence_coverage, identifier_coverage,
  binding_coverage, constraint_coverage, action_outcome_coverage,
  top_missing_error_type

Table 3 (Behavioral Utility):
  method, budget, success_rate, avg_steps, avg_peak_tokens,
  avg_total_input_tokens, avg_api_calls, avg_recovery_calls

CSVs are written under outputs/tables/.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _safe_mean(xs):
    return statistics.mean(xs) if xs else 0.0


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.data import (
        TABLES, ensure_outputs, jsonl_path, read_jsonl,
    )

    ensure_outputs()
    compressed = read_jsonl(jsonl_path("motivation_compressed_contexts.jsonl"))
    audits = (read_jsonl(jsonl_path("motivation_audits.jsonl"))
              if jsonl_path("motivation_audits.jsonl").exists() else [])
    aug_runs_path = jsonl_path("motivation_behavior_runs_with_recovery.jsonl")
    runs_path = jsonl_path("motivation_behavior_runs.jsonl")
    runs = read_jsonl(aug_runs_path if aug_runs_path.exists() else runs_path)

    # ------------------------------------------------------------------
    # Table 1
    # ------------------------------------------------------------------
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in compressed:
        if r.get("error"):
            continue
        by_method[r["method"]].append(r)

    table1_rows: List[dict] = []
    for method in ("task_aware_summary", "acon_style_summary", "symbolic_evidence"):
        rows = by_method.get(method, [])
        if not rows:
            continue
        table1_rows.append({
            "method": method,
            "n_tasks": len(rows),
            "avg_tokens":                    round(_safe_mean([r["n_tokens"] for r in rows]), 1),
            "avg_ids_preserved":             round(_safe_mean([r.get("n_ids_preserved", 0) for r in rows]), 2),
            "avg_bindings_preserved":        round(_safe_mean([r.get("n_bindings_preserved", 0) for r in rows]), 2),
            "avg_constraints_preserved":     round(_safe_mean([r.get("n_constraints_preserved", 0) for r in rows]), 2),
            "avg_action_outcomes_preserved": round(_safe_mean([r.get("n_action_outcomes_preserved", 0) for r in rows]), 2),
            "avg_ids_total":                 round(_safe_mean([r.get("n_ids_total", 0) for r in rows]), 1),
            "avg_bindings_total":            round(_safe_mean([r.get("n_bindings_total", 0) for r in rows]), 1),
            "avg_constraints_total":         round(_safe_mean([r.get("n_constraints_total", 0) for r in rows]), 1),
            "avg_action_outcomes_total":     round(_safe_mean([r.get("n_action_outcomes_total", 0) for r in rows]), 1),
        })

    table1_path = TABLES / "table1_compactness.csv"
    with open(table1_path, "w", newline="") as f:
        if table1_rows:
            w = csv.DictWriter(f, fieldnames=list(table1_rows[0].keys()))
            w.writeheader()
            for r in table1_rows:
                w.writerow(r)
    print(f"[07] Table 1 -> {table1_path}")
    for r in table1_rows:
        print("    ", r)

    # ------------------------------------------------------------------
    # Table 2 — Behavioral Evidence Coverage
    # ------------------------------------------------------------------
    by_method_audit: Dict[str, List[dict]] = defaultdict(list)
    for a in audits:
        if a.get("error"):
            continue
        by_method_audit[a["method"]].append(a)

    table2_rows: List[dict] = []
    for method in ("task_aware_summary", "acon_style_summary", "symbolic_evidence"):
        audits_m = by_method_audit.get(method, [])
        if not audits_m:
            continue
        n_units_total = 0
        n_preserved = 0
        per_label = Counter()
        # Sub-category counts: identifier / binding / constraint / action_outcome
        # broken out by the spec's drop labels.
        n_ident_total = 0; n_ident_preserved = 0
        n_bind_total = 0; n_bind_preserved = 0
        n_constr_total = 0; n_constr_preserved = 0
        n_act_total = 0; n_act_preserved = 0
        for a in audits_m:
            for u in a.get("unit_results", []):
                n_units_total += 1
                lbl = u.get("label", "")
                per_label[lbl] += 1
                if lbl == "preserved":
                    n_preserved += 1
                # Sub-category buckets are inferred from the drop label.
                if lbl in {"preserved", "dropped_identifier"}:
                    n_ident_total += 1
                    if lbl == "preserved":
                        n_ident_preserved += 1
                if lbl in {"preserved", "dropped_binding"}:
                    n_bind_total += 1
                    if lbl == "preserved":
                        n_bind_preserved += 1
                if lbl in {"preserved", "dropped_constraint"}:
                    n_constr_total += 1
                    if lbl == "preserved":
                        n_constr_preserved += 1
                if lbl in {"preserved", "dropped_action_outcome"}:
                    n_act_total += 1
                    if lbl == "preserved":
                        n_act_preserved += 1

        top_err = ""
        if per_label:
            non_pres = [(k, v) for k, v in per_label.items() if k != "preserved"]
            if non_pres:
                top_err = max(non_pres, key=lambda kv: kv[1])[0]

        def _safe_div(num, den):
            return round(num / den, 4) if den else 0.0

        table2_rows.append({
            "method": method,
            "n_audits": len(audits_m),
            "n_units": n_units_total,
            "behavioral_evidence_coverage": _safe_div(n_preserved, n_units_total),
            "identifier_coverage":          _safe_div(n_ident_preserved, n_ident_total),
            "binding_coverage":             _safe_div(n_bind_preserved, n_bind_total),
            "constraint_coverage":          _safe_div(n_constr_preserved, n_constr_total),
            "action_outcome_coverage":      _safe_div(n_act_preserved, n_act_total),
            "top_missing_error_type":       top_err,
        })

    table2_path = TABLES / "table2_evidence_coverage.csv"
    with open(table2_path, "w", newline="") as f:
        if table2_rows:
            w = csv.DictWriter(f, fieldnames=list(table2_rows[0].keys()))
            w.writeheader()
            for r in table2_rows:
                w.writerow(r)
    print(f"[07] Table 2 -> {table2_path}")
    for r in table2_rows:
        print("    ", r)

    # ------------------------------------------------------------------
    # Table 3 — Behavioral Utility
    # ------------------------------------------------------------------
    by_mb: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in runs:
        if r.get("error"):
            continue
        by_mb[(r["method"], int(r["budget_max_steps"]))].append(r)

    cond_order = ["full_context", "task_aware_summary", "acon_style_summary",
                  "symbolic_evidence", "wrong_task_symbolic_same_app",
                  "wrong_task_symbolic_cross_app", "no_context"]

    table3_rows: List[dict] = []
    for method in cond_order:
        for cap in (15, 8):
            xs = by_mb.get((method, cap), [])
            if not xs:
                continue
            n = len(xs)
            success = sum(1 for r in xs if r["success"]) / n
            table3_rows.append({
                "method": method,
                "budget_max_steps": cap,
                "n_runs": n,
                "success_rate":          round(success, 4),
                "avg_steps":             round(_safe_mean([r["iterations"] for r in xs]), 2),
                "avg_total_input_tokens": round(_safe_mean([r["input_tokens"] for r in xs]), 0),
                # peak tokens — proxy: max input_tokens across the run; we
                # don't have step-by-step peak so we use total.
                "avg_peak_tokens":       round(_safe_mean([r["input_tokens"] for r in xs]), 0),
                "avg_api_calls":         round(_safe_mean([r.get("api_call_count", 0) for r in xs]), 2),
                "avg_recovery_calls":    round(_safe_mean([r.get("recovery_api_call_count", 0) for r in xs]), 2),
            })

    table3_path = TABLES / "table3_behavioral_utility.csv"
    with open(table3_path, "w", newline="") as f:
        if table3_rows:
            w = csv.DictWriter(f, fieldnames=list(table3_rows[0].keys()))
            w.writeheader()
            for r in table3_rows:
                w.writerow(r)
    print(f"[07] Table 3 -> {table3_path}")
    for r in table3_rows:
        print("    ", r)


if __name__ == "__main__":
    main()
