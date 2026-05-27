"""Stage 10 — generate the 3 spec figures.

  fig_failure_mode_bar.{pdf,png}
  fig_recovered_then_dropped_bar.{pdf,png}
  fig_information_flow_sankey.{pdf,png}
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _save(fig, base: Path):
    fig.savefig(str(base) + ".pdf", format="pdf")
    fig.savefig(str(base) + ".png", format="png", dpi=200)


def _load_csv(p: Path):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import FIGURES, RAW, TABLES, read_jsonl

    import matplotlib.pyplot as plt
    import numpy as np

    merged = read_jsonl(RAW / "merged_case_audits.jsonl")
    failure_rows = _load_csv(TABLES / "failure_mode_counts.csv")
    rtd_rows = _load_csv(TABLES / "recovered_then_dropped.csv")

    # ------------------------------------------------------------------
    # Figure 1: failure mode bar chart
    # ------------------------------------------------------------------
    if failure_rows:
        modes = [r["primary_failure_mode"] for r in failure_rows]
        n_easy = [int(r["n_easy"]) for r in failure_rows]
        n_med = [int(r["n_medium"]) for r in failure_rows]
        n_hard = [int(r["n_hard"]) for r in failure_rows]
        x = np.arange(len(modes))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x, n_easy, label="easy", color="#a6cee3", edgecolor="black")
        ax.bar(x, n_med, bottom=n_easy, label="medium",
               color="#1f78b4", edgecolor="black")
        ax.bar(x, n_hard, bottom=[a + b for a, b in zip(n_easy, n_med)],
               label="hard", color="#08306b", edgecolor="black")
        for i, r in enumerate(failure_rows):
            ax.text(i, int(r["n_cases"]) + 0.2, r["n_cases"],
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("# cases")
        ax.set_title("ACON failure-mode distribution (n_cases per primary mode)")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        _save(fig, FIGURES / "fig_failure_mode_bar")
        plt.close(fig)
        print(f"[10] wrote fig_failure_mode_bar.{{pdf,png}}")

    # ------------------------------------------------------------------
    # Figure 2: recovered-then-dropped categories
    # ------------------------------------------------------------------
    if rtd_rows:
        cat_cnt = Counter(r["category"] for r in rtd_rows)
        cat_cnt_high = Counter(
            r["category"] for r in rtd_rows
            if str(r.get("criticality", "")).lower() == "high"
        )
        cats = list(cat_cnt.keys())
        cats.sort(key=lambda c: -cat_cnt[c])
        x = np.arange(len(cats))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        all_n = [cat_cnt[c] for c in cats]
        hi_n = [cat_cnt_high[c] for c in cats]
        width = 0.4
        ax.bar(x - width/2, all_n, width=width, label="all criticalities",
               color="#fdae6b", edgecolor="black")
        ax.bar(x + width/2, hi_n, width=width, label="criticality=high",
               color="#d94701", edgecolor="black")
        for i, (a, h) in enumerate(zip(all_n, hi_n)):
            ax.text(i - width/2, a + 0.1, a, ha="center", va="bottom", fontsize=8)
            ax.text(i + width/2, h + 0.1, h, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("# recovered_then_dropped items")
        ax.set_title("Audit-recovered information categories that the recompressor drops again")
        ax.legend(loc="upper right")
        plt.tight_layout()
        _save(fig, FIGURES / "fig_recovered_then_dropped_bar")
        plt.close(fig)
        print(f"[10] wrote fig_recovered_then_dropped_bar.{{pdf,png}}")

    # ------------------------------------------------------------------
    # Figure 3: information flow (bar-chart proxy for sankey)
    # ------------------------------------------------------------------
    if merged:
        n_cases = len(merged)
        n_missing_total = sum(m["n_missing_items"] for m in merged)
        n_added_total = sum(m["n_audit_added_items"] for m in merged)
        n_added_grounded = sum(m["n_grounded_audit_added_items"] for m in merged)
        n_rec_dropped = sum(m["n_recovered_then_dropped_items"] for m in merged)
        n_critical_rec_dropped = sum(
            1 for m in merged if m["critical_recovered_then_dropped"]
        )
        n_final_fail = sum(
            1 for m in merged if m.get("final_after_recompression_success") is False
        )
        n_final_succ = sum(
            1 for m in merged if m.get("final_after_recompression_success") is True
        )

        stages = [
            "Cases\nanalysed",
            "ACON-dropped\nfacts (Q audit)",
            "Audit\nadded",
            "Audit added\n(grounded)",
            "Recompressor\ndropped",
            "Critical\nrec_dropped\ncases",
            "Recompressed\nfinal SUCC",
            "Recompressed\nfinal FAIL",
        ]
        counts = [
            n_cases, n_missing_total, n_added_total, n_added_grounded,
            n_rec_dropped, n_critical_rec_dropped, n_final_succ, n_final_fail,
        ]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        colors = ["#08306b", "#08519c", "#2171b5", "#4292c6",
                  "#fb6a4a", "#cb181d", "#238b45", "#a50f15"]
        bars = ax.bar(range(len(stages)), counts, color=colors, edgecolor="black")
        for i, v in enumerate(counts):
            ax.text(i, v + 0.2, v, ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, fontsize=9)
        ax.set_ylabel("count")
        ax.set_title("Information flow: full → ACON → audit → recompression → downstream")
        plt.tight_layout()
        _save(fig, FIGURES / "fig_information_flow_sankey")
        plt.close(fig)
        print(f"[10] wrote fig_information_flow_sankey.{{pdf,png}}")


if __name__ == "__main__":
    main()
