"""Stage 03 — compare Jacobian span scores against v4 finite-difference
sensitivity. Per spec §5.5.

Outputs:
  outputs/tables/jacobian_vs_v4_correlations.csv
  outputs/tables/jacobian_topk_overlap.csv
  outputs/figures/fig_jacobian_vs_v4_scatter.{png,pdf}
  outputs/figures/fig_jacobian_rank_overlap.{png,pdf}
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, raw_path, table_path, figure_path, read_jsonl,
)
from motivation_v6_jacobian.metrics import (  # noqa: E402
    spearman_pearson, topk_overlap, rank_of_top1,
)
from motivation_v6_jacobian.plotting import (  # noqa: E402
    scatter_jacobian_vs_v4, bar_topk_enrichment,
)


SCORE_VARIANTS = [
    "span_grad_sum", "span_grad_mean",
    "span_gxa_sum", "span_gxa_mean", "span_gxa_sqrtlen",
    "span_top10_mean", "span_g_dot_x_abs_sum",
]
PRIMARY = "span_gxa_sqrtlen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(raw_path("jacobian_span_scores.jsonl")))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    ensure_outputs()
    rows = read_jsonl(Path(args.scores))
    df = pd.DataFrame(rows)
    if df.empty:
        print("[03] no rows; aborting")
        return

    df = df.dropna(subset=["v4_final_sensitivity"]).copy()
    df["token_count"] = df["token_count"].fillna(0).astype(int)

    # ---------- global + per-task correlations ----------
    corr_rows = []
    for var in SCORE_VARIANTS:
        x = df[var].values
        y = df["v4_final_sensitivity"].values
        stats = spearman_pearson(x, y)
        corr_rows.append({"score": var, "scope": "global",
                          "spearman_r": stats["spearman_r"],
                          "pearson_r": stats["pearson_r"],
                          "n": stats["n"]})

    per_task = []
    for tid, grp in df.groupby("task_id"):
        for var in SCORE_VARIANTS:
            stats = spearman_pearson(grp[var].values,
                                     grp["v4_final_sensitivity"].values)
            per_task.append({"task_id": tid, "score": var,
                             "spearman_r": stats["spearman_r"],
                             "pearson_r": stats["pearson_r"],
                             "n": stats["n"]})
    pt = pd.DataFrame(per_task)
    if not pt.empty:
        for var in SCORE_VARIANTS:
            sub = pt[pt["score"] == var]
            med_s = float(sub["spearman_r"].median(skipna=True))
            med_p = float(sub["pearson_r"].median(skipna=True))
            corr_rows.append({"score": var, "scope": "median_per_task",
                              "spearman_r": med_s,
                              "pearson_r": med_p,
                              "n": int(sub["spearman_r"].count())})

    # length / recency confounder checks
    for var in ["token_count"]:
        x = df[var].values
        y = df["v4_final_sensitivity"].values
        stats = spearman_pearson(x, y)
        corr_rows.append({"score": f"v4_vs_{var}", "scope": "global",
                          "spearman_r": stats["spearman_r"],
                          "pearson_r": stats["pearson_r"],
                          "n": stats["n"]})
        x = df[PRIMARY].values
        stats = spearman_pearson(x, df[var].values)
        corr_rows.append({"score": f"{PRIMARY}_vs_{var}", "scope": "global",
                          "spearman_r": stats["spearman_r"],
                          "pearson_r": stats["pearson_r"],
                          "n": stats["n"]})
    # recency = step_id (higher = more recent)
    x = df[PRIMARY].values
    stats = spearman_pearson(x, df["step_id"].values)
    corr_rows.append({"score": f"{PRIMARY}_vs_step_id", "scope": "global",
                      "spearman_r": stats["spearman_r"],
                      "pearson_r": stats["pearson_r"],
                      "n": stats["n"]})

    corr_df = pd.DataFrame(corr_rows)
    corr_path = table_path("jacobian_vs_v4_correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"[03] wrote correlations -> {corr_path}")

    # ---------- top-k overlap per task ----------
    topk_rows = []
    for tid, grp in df.groupby("task_id"):
        n_spans = len(grp)
        for k in [1, 3, 5, 10]:
            stats = topk_overlap(grp[PRIMARY].values,
                                 grp["v4_final_sensitivity"].values,
                                 k=k)
            topk_rows.append({"task_id": tid, **stats})
    topk_df = pd.DataFrame(topk_rows)
    topk_path = table_path("jacobian_topk_overlap.csv")
    topk_df.to_csv(topk_path, index=False)
    print(f"[03] wrote top-k -> {topk_path}")

    # ---------- ranks of top-1 ----------
    rank_rows = []
    for tid, grp in df.groupby("task_id"):
        if len(grp) < 2:
            continue
        rank_v4_under_j = rank_of_top1(grp[PRIMARY].values,
                                        grp["v4_final_sensitivity"].values)
        rank_j_under_v4 = rank_of_top1(grp["v4_final_sensitivity"].values,
                                        grp[PRIMARY].values)
        rank_rows.append({"task_id": tid,
                          "rank_v4top1_under_jacobian": rank_v4_under_j,
                          "rank_jacobiantop1_under_v4": rank_j_under_v4,
                          "n_spans": len(grp)})
    rank_df = pd.DataFrame(rank_rows)
    rank_path = table_path("jacobian_top1_ranks.csv")
    rank_df.to_csv(rank_path, index=False)
    print(f"[03] wrote ranks -> {rank_path}")

    # ---------- figures ----------
    primary_stats = spearman_pearson(df[PRIMARY].values,
                                      df["v4_final_sensitivity"].values)
    scatter_jacobian_vs_v4(
        jacobian_scores=df[PRIMARY].values,
        v4_scores=df["v4_final_sensitivity"].values,
        spearman=primary_stats["spearman_r"] if math.isfinite(
            primary_stats["spearman_r"]) else 0.0,
        save_to=figure_path("fig_jacobian_vs_v4_scatter"),
    )

    # top-k bar chart (aggregate enrichment by k across all tasks)
    agg_enrich = []
    for k in [1, 3, 5, 10]:
        sub = topk_df[topk_df["k"] == k]
        if sub.empty:
            continue
        median_enr = float(sub["enrichment"].replace(
            [np.inf, -np.inf], np.nan).median(skipna=True))
        agg_enrich.append({"k": k, "enrichment": median_enr})
    bar_topk_enrichment(
        rows=agg_enrich,
        save_to=figure_path("fig_jacobian_rank_overlap"),
    )

    # ---------- console summary ----------
    print("\n[03] verdict summary (primary score = span_gxa_sqrtlen)")
    print(f"  global Spearman   = {primary_stats['spearman_r']:.3f}")
    print(f"  global Pearson    = {primary_stats['pearson_r']:.3f}")
    if not pt.empty:
        sub = pt[pt["score"] == PRIMARY]
        med_s = sub["spearman_r"].median(skipna=True)
        print(f"  median per-task Spearman = {med_s:.3f}")
    if agg_enrich:
        for r in agg_enrich:
            print(f"  top-{r['k']} median enrichment = {r['enrichment']:.2f}×")


if __name__ == "__main__":
    main()
