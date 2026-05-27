"""Stage 09 — generate the 4 spec figures (PDF + PNG).

  fig_sensitivity_distribution
  fig_budgeted_success_by_method
  fig_sensitivity_vs_behavior
  fig_sensitivity_vs_recency
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _load_csv(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _save_pdf_png(fig, base: Path):
    fig.savefig(str(base) + ".pdf", format="pdf")
    fig.savefig(str(base) + ".png", format="png", dpi=200)


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, table_path, figure_path, read_jsonl,
    )

    ensure_outputs()
    sens_rows = read_jsonl(raw_path("span_sensitivity_scores.jsonl"))
    spans_rows = read_jsonl(raw_path("history_spans.jsonl"))
    table2 = _load_csv(table_path("table_behavior_by_method.csv"))

    import matplotlib.pyplot as plt
    import numpy as np

    # ------------------------------------------------------------------
    # Figure 1: sensitivity distribution
    # ------------------------------------------------------------------
    sensitivities = [float(r["final_sensitivity"]) for r in sens_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(sensitivities, bins=20, color="#3776ab", edgecolor="black")
    ax.set_xlabel("Span decision-state sensitivity (0=no change, 1=max change)")
    ax.set_ylabel("# spans")
    ax.set_title(f"Distribution of span sensitivity scores (n={len(sensitivities)} spans)")
    ax.axvline(0.3, color="gray", linestyle="--", linewidth=1, label="τ=0.3")
    ax.axvline(0.6, color="red",  linestyle="--", linewidth=1, label="τ=0.6")
    ax.legend(loc="upper right")
    plt.tight_layout()
    _save_pdf_png(fig, figure_path("fig_sensitivity_distribution"))
    plt.close(fig)
    print(f"[09] wrote fig_sensitivity_distribution.{{pdf,png}}")

    # ------------------------------------------------------------------
    # Figure 2: budgeted success by method
    # ------------------------------------------------------------------
    method_order = [
        "high_sensitivity_spans", "low_sensitivity_spans", "recent_spans",
        "random_spans_mean",
        "task_aware_summary", "acon_style_summary",
        "truncated_full_context", "no_context",
    ]
    by_mb = {(r["method"], r["budget"]): r for r in table2}
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    width = 0.4
    x = np.arange(len(method_order))
    for i, b in enumerate(["loose_15", "strict_8"]):
        ys, ns = [], []
        for m in method_order:
            r = by_mb.get((m, b))
            ys.append(float(r["success_rate"]) if r else 0.0)
            ns.append(int(r["num_tasks"]) if r else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, ys, width=width,
               label=f"{b} (n={max(ns) if ns else 0})",
               color="#3776ab" if b == "loose_15" else "#cd5c5c",
               edgecolor="black")
        for j, v in enumerate(ys):
            ax.text(j + offset, v + 0.01, f"{v*100:.0f}%", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(method_order, rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Behavioral utility by compression method (decision-state sensitivity vs static)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    _save_pdf_png(fig, figure_path("fig_budgeted_success_by_method"))
    plt.close(fig)
    print(f"[09] wrote fig_budgeted_success_by_method.{{pdf,png}}")

    # ------------------------------------------------------------------
    # Figure 3: sensitivity vs behavior (per-task, per-method scatter)
    # ------------------------------------------------------------------
    runs_v4 = read_jsonl(raw_path("behavior_runs.jsonl"))
    sens_avg_per_task: Dict[str, float] = {}
    sens_by_task_dict: Dict[str, List[float]] = defaultdict(list)
    for r in sens_rows:
        sens_by_task_dict[r["task_id"]].append(float(r["final_sensitivity"]))
    for t, vs in sens_by_task_dict.items():
        sens_avg_per_task[t] = sum(vs) / len(vs)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, color in [("high_sensitivity_spans", "#2ca02c"),
                           ("low_sensitivity_spans", "#d62728"),
                           ("recent_spans", "#1f77b4"),
                           ("random_spans_seed1", "#7f7f7f")]:
        xs, ys = [], []
        for r in runs_v4:
            if r.get("method") != method:
                continue
            if r.get("budget_max_steps") != 15:
                continue
            xs.append(sens_avg_per_task.get(r["task_id"], 0.0))
            ys.append(float(r.get("score", r.get("final_reward", 0))))
        if xs:
            ax.scatter(xs, ys, label=method, alpha=0.6, color=color, s=40,
                       edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Per-task average decision-state sensitivity")
    ax.set_ylabel("Task score (0–1) at cap=15")
    ax.set_title("Decision-state sensitivity vs downstream behavior")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _save_pdf_png(fig, figure_path("fig_sensitivity_vs_behavior"))
    plt.close(fig)
    print(f"[09] wrote fig_sensitivity_vs_behavior.{{pdf,png}}")

    # ------------------------------------------------------------------
    # Figure 4: sensitivity vs recency
    # ------------------------------------------------------------------
    spans_by_task: Dict[str, List[dict]] = defaultdict(list)
    for s in spans_rows:
        spans_by_task[s["task_id"]].append(s)

    sens_by_key = {(r["task_id"], r["span_id"]): float(r["final_sensitivity"])
                   for r in sens_rows}
    rec_xs, sens_ys = [], []
    for tid, sps in spans_by_task.items():
        sps_sorted = sorted(sps, key=lambda x: -int(x["step_id"]))  # recent first
        n = len(sps_sorted)
        for rank, sp in enumerate(sps_sorted):
            sens = sens_by_key.get((tid, sp["span_id"]))
            if sens is None:
                continue
            rec_xs.append(rank / max(n - 1, 1))  # 0 = most recent, 1 = oldest
            sens_ys.append(sens)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(rec_xs, sens_ys, alpha=0.4, color="#3776ab", s=24,
               edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Recency rank (0 = most recent span, 1 = oldest)")
    ax.set_ylabel("Decision-state sensitivity")
    ax.set_title("Span sensitivity vs recency rank")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_pdf_png(fig, figure_path("fig_sensitivity_vs_recency"))
    plt.close(fig)
    print(f"[09] wrote fig_sensitivity_vs_recency.{{pdf,png}}")


if __name__ == "__main__":
    main()
