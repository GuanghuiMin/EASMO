"""Stage 06 — C1 vs CK fragility metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, read_jsonl, raw_path, table_path  # noqa
from motivation_v9.metrics import c1_ck_transition, c1_ck_fragility_by_model  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(raw_path("behavior_runs_c1_ck.jsonl")))
    args = ap.parse_args()
    ensure_outputs()

    df = pd.DataFrame(read_jsonl(Path(args.runs)))
    if df.empty:
        print("[06] no behavior runs"); return
    trans = c1_ck_transition(df)
    trans.to_csv(table_path("c1_ck_transition.csv"), index=False)
    summary = c1_ck_fragility_by_model(trans)
    summary.to_csv(table_path("c1_ck_fragility_by_model.csv"), index=False)
    print(f"[06] transitions: {len(trans)} candidates")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
