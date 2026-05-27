"""Stage 06 — aggregate per-task numbers into a single dashboard CSV
that the report writer can consume directly.

Outputs:
  outputs/tables/v6_dashboard.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, raw_path, table_path, read_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(table_path("v6_dashboard.csv")))
    args = ap.parse_args()

    ensure_outputs()
    rows = []

    # ---------- Experiment A summary ----------
    corr_path = table_path("jacobian_vs_v4_correlations.csv")
    topk_path = table_path("jacobian_topk_overlap.csv")
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        primary = corr[(corr["score"] == "span_gxa_sqrtlen") & (corr["scope"] == "global")]
        if not primary.empty:
            rows.append({"section": "A", "metric": "global_spearman_primary",
                         "value": float(primary["spearman_r"].iloc[0])})
            rows.append({"section": "A", "metric": "global_pearson_primary",
                         "value": float(primary["pearson_r"].iloc[0])})
        median = corr[(corr["score"] == "span_gxa_sqrtlen") &
                      (corr["scope"] == "median_per_task")]
        if not median.empty:
            rows.append({"section": "A", "metric": "median_per_task_spearman",
                         "value": float(median["spearman_r"].iloc[0])})
        for var in ("v4_vs_token_count",
                    "span_gxa_sqrtlen_vs_token_count",
                    "span_gxa_sqrtlen_vs_step_id"):
            sub = corr[corr["score"] == var]
            if not sub.empty:
                rows.append({"section": "A",
                             "metric": f"corr_{var}",
                             "value": float(sub["spearman_r"].iloc[0])})
    if topk_path.exists():
        tk = pd.read_csv(topk_path)
        for k in (1, 3, 5, 10):
            sub = tk[tk["k"] == k]
            if not sub.empty:
                med_enr = float(sub["enrichment"].replace([np.inf, -np.inf], np.nan)
                                                    .median(skipna=True))
                rows.append({"section": "A",
                             "metric": f"median_top{k}_enrichment",
                             "value": med_enr})

    # ---------- Experiment B summary ----------
    spec_path = table_path("active_subspace_spectrum.csv")
    if spec_path.exists():
        spec = pd.read_csv(spec_path)
        for matrix in ("example", "span", "high_v4", "low_v4"):
            sub = spec[spec["matrix"] == matrix]
            for k in (4, 8, 16, 32, 64):
                kr = sub[sub["component"] == k]
                if not kr.empty:
                    rows.append({"section": "B",
                                 "metric": f"{matrix}_cumvar_k{k}",
                                 "value": float(kr["cumulative_explained_variance"].iloc[0])})

    # ---------- Experiment C summary ----------
    csv_path = table_path("soft_token_oracle_losses.csv")
    if csv_path.exists():
        st = pd.read_csv(csv_path)
        for col in ("full_loss", "no_loss", "recent_loss", "acon_loss"):
            if col in st:
                rows.append({"section": "C",
                             "metric": f"median_{col}",
                             "value": float(pd.to_numeric(st[col], errors="coerce")
                                            .median(skipna=True))})
        for col in st.columns:
            if col.startswith("soft_loss_k") or col.startswith("soft_gap_recovery_"):
                rows.append({"section": "C",
                             "metric": f"median_{col}",
                             "value": float(pd.to_numeric(st[col], errors="coerce")
                                            .median(skipna=True))})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[06] wrote dashboard ({len(rows)} rows) -> {args.out}")
    for r in rows:
        v = r["value"]
        if isinstance(v, float) and math.isnan(v):
            v_str = "nan"
        elif isinstance(v, float):
            v_str = f"{v:.4f}"
        else:
            v_str = str(v)
        print(f"  {r['section']:>2s} | {r['metric']:<42s} | {v_str}")


if __name__ == "__main__":
    main()
