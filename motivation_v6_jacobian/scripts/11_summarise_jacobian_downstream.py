"""Stage 11 — Experiment D part 3: summary table for gradient-ranked
downstream runs.

Computes success/score by method and budget; compares against v4's
recent_spans baseline if behavior_runs.jsonl is reachable.

Outputs:
  outputs/tables/jacobian_span_downstream_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, table_path, raw_path, read_jsonl,
)

V4_BEHAVIOR = Path("/workspace/EASMO/motivation_v4/outputs/raw/behavior_runs.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs",
                    default=str(raw_path("jacobian_behavior_runs.jsonl")))
    ap.add_argument("--out",
                    default=str(table_path("jacobian_span_downstream_results.csv")))
    args = ap.parse_args()
    ensure_outputs()

    rows = read_jsonl(Path(args.runs))
    if not rows:
        print("[11] no jacobian_behavior_runs.jsonl rows; aborting")
        return

    df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(bool)
    summary = (df.groupby(["method", "budget_max_steps"])
                 .agg(n=("task_id", "count"),
                      success_rate=("success", "mean"),
                      mean_score=("score", "mean"),
                      mean_iters=("num_steps", "mean"),
                      mean_input_tokens=("input_tokens", "mean"))
                 .reset_index())
    if V4_BEHAVIOR.exists():
        v4_rows = read_jsonl(V4_BEHAVIOR)
        v4 = pd.DataFrame(v4_rows)
        if not v4.empty:
            v4["success"] = v4["success"].astype(bool)
            v4_sub = (v4[v4["method"].isin(["recent_spans", "high_sensitivity_spans"])]
                      .groupby(["method", "budget_max_steps"])
                      .agg(n=("task_id", "count"),
                           success_rate=("success", "mean"),
                           mean_score=("score", "mean"),
                           mean_iters=("num_steps", "mean"),
                           mean_input_tokens=("input_tokens", "mean"))
                      .reset_index())
            v4_sub["method"] = "v4_" + v4_sub["method"].astype(str)
            summary = pd.concat([summary, v4_sub], ignore_index=True)
    summary.to_csv(args.out, index=False)
    print(f"[11] wrote {args.out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
