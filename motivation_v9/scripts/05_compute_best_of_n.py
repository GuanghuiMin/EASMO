"""Stage 05 — best-of-N + reward spread metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, read_jsonl, raw_path, table_path  # noqa
from motivation_v9.metrics import (  # noqa
    best_of_n_by_case, best_of_n_summary, reward_spread_by_case,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(raw_path("behavior_runs_c1_ck.jsonl")))
    args = ap.parse_args()
    ensure_outputs()

    df = pd.DataFrame(read_jsonl(Path(args.runs)))
    if df.empty:
        print("[05] no behavior runs found"); return
    df["compressed_tokens_est"] = df["compressed_tokens_est"].astype(float)

    per_case = best_of_n_by_case(df)
    per_case.to_csv(table_path("best_of_n_by_case.csv"), index=False)
    summary = best_of_n_summary(per_case)
    summary.to_csv(table_path("best_of_n_summary.csv"), index=False)
    spread = reward_spread_by_case(df)
    spread.to_csv(table_path("reward_spread_by_case.csv"), index=False)
    print(f"[05] best_of_n_by_case rows: {len(per_case)}")
    print(f"[05] best_of_n_summary rows: {len(summary)}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
