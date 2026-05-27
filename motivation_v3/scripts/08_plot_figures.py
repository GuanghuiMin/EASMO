"""Stage 8 — generate the 3 spec figures.

  fig_compactness_vs_evidence_coverage.pdf
  fig_budgeted_success.pdf
  fig_recovery_calls_by_method.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _load_csv(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.data import FIGURES, TABLES, ensure_outputs

    ensure_outputs()
    table1 = _load_csv(TABLES / "table1_compactness.csv")
    table2 = _load_csv(TABLES / "table2_evidence_coverage.csv")
    table3 = _load_csv(TABLES / "table3_behavioral_utility.csv")

    import matplotlib.pyplot as plt
    import numpy as np

    method_order = ["task_aware_summary", "acon_style_summary", "symbolic_evidence"]
    palette = {"task_aware_summary": "#1f77b4",
               "acon_style_summary": "#ff7f0e",
               "symbolic_evidence":  "#2ca02c",
               "full_context":       "#9467bd",
               "wrong_task_symbolic_same_app":  "#d62728",
               "wrong_task_symbolic_cross_app": "#8c564b",
               "no_context":         "#7f7f7f"}

    # ------------------------------------------------------------------
    # Figure 1: compactness vs evidence coverage (scatter)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_m1 = {r["method"]: r for r in table1}
    by_m2 = {r["method"]: r for r in table2}
    for m in method_order:
        r1 = by_m1.get(m); r2 = by_m2.get(m)
        if not r1 or not r2:
            continue
        x = float(r1.get("avg_tokens", 0))
        y = float(r2.get("behavioral_evidence_coverage", 0))
        ax.scatter([x], [y], s=180, color=palette.get(m, "gray"),
                   edgecolor="black", zorder=3, label=m)
        ax.annotate(f"  {m}", (x, y), fontsize=10)
    ax.set_xlabel("Avg tokens in compressed context")
    ax.set_ylabel("Behavioral evidence coverage (Exp 2.2 audit)")
    ax.set_title("Compactness vs evidence coverage (Pareto view)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_compactness_vs_evidence_coverage.pdf", format="pdf")
    plt.close(fig)
    print(f"[08] wrote {FIGURES/'fig_compactness_vs_evidence_coverage.pdf'}")

    # ------------------------------------------------------------------
    # Figure 2: budgeted success
    # ------------------------------------------------------------------
    cond_order = ["full_context", "task_aware_summary", "acon_style_summary",
                  "symbolic_evidence", "wrong_task_symbolic_same_app",
                  "wrong_task_symbolic_cross_app", "no_context"]
    by_mb = {(r["method"], int(r["budget_max_steps"])): r for r in table3}
    budgets = [15, 8]
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    width = 0.4
    x = np.arange(len(cond_order))
    for i, cap in enumerate(budgets):
        ys = []
        ns = []
        for m in cond_order:
            r = by_mb.get((m, cap))
            ys.append(float(r["success_rate"]) if r else 0.0)
            ns.append(int(r["n_runs"]) if r else 0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, ys, width=width,
                      label=f"max_steps={cap} (n={max(ns) if ns else 0})",
                      color=("#3776ab" if cap == 15 else "#cd5c5c"),
                      edgecolor="black")
        for j, v in enumerate(ys):
            ax.text(j + offset, v + 0.01, f"{v*100:.0f}%", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Behavioral utility: success under loose vs strict budget")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_budgeted_success.pdf", format="pdf")
    plt.close(fig)
    print(f"[08] wrote {FIGURES/'fig_budgeted_success.pdf'}")

    # ------------------------------------------------------------------
    # Figure 3: recovery calls by method (cap=15)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    cap = 15
    methods_with_data = []
    api_vals = []
    rec_vals = []
    ns = []
    for m in cond_order:
        r = by_mb.get((m, cap))
        if not r:
            continue
        methods_with_data.append(m)
        api_vals.append(float(r["avg_api_calls"]))
        rec_vals.append(float(r["avg_recovery_calls"]))
        ns.append(int(r["n_runs"]))
    x = np.arange(len(methods_with_data))
    ax.bar(x, api_vals, label="Avg API calls (sampled)", color="#bbbbbb",
           edgecolor="black")
    ax.bar(x, rec_vals, label="Avg recovery API calls", color="#d62728",
           edgecolor="black")
    for i, (a, r, n) in enumerate(zip(api_vals, rec_vals, ns)):
        ax.text(i, a + 0.05, f"{r:.1f}/{a:.1f}\nn={n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_with_data, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Avg API calls per run (sampled, capped 8/run)")
    ax.set_title(f"Recovery API calls vs total API calls (max_steps={cap})")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_recovery_calls_by_method.pdf", format="pdf")
    plt.close(fig)
    print(f"[08] wrote {FIGURES/'fig_recovery_calls_by_method.pdf'}")


if __name__ == "__main__":
    main()
