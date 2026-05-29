"""Stage 10 — compute per-chunk score and pass advantage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, read_jsonl, raw_path, table_path  # noqa
from motivation_v9.metrics import chunk_advantage  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs",
                    default=str(raw_path("chunk_ablation_behavior_runs.jsonl")))
    ap.add_argument("--contexts",
                    default=str(raw_path("chunk_ablation_contexts.jsonl")))
    ap.add_argument("--chunks", default=str(raw_path("chunks.jsonl")))
    args = ap.parse_args()
    ensure_outputs()

    df_runs = pd.DataFrame(read_jsonl(Path(args.runs)))
    if df_runs.empty:
        print("[10] no ablation runs"); return
    # Merge chunk text into runs via ablation_id
    df_ctx = pd.DataFrame(read_jsonl(Path(args.contexts)))
    df_chunks = pd.DataFrame(read_jsonl(Path(args.chunks)))
    df_runs = df_runs.merge(
        df_ctx[["ablation_id", "chunk_id", "chunk_index", "chunk_text"]],
        on="ablation_id", how="left", suffixes=("", "_ctx"),
    )
    # Coalesce chunk_id from context if missing
    if "chunk_id" not in df_runs.columns or df_runs["chunk_id"].isna().any():
        if "chunk_id_ctx" in df_runs.columns:
            df_runs["chunk_id"] = df_runs["chunk_id"].fillna(df_runs.get("chunk_id_ctx"))
    ablation = df_runs[df_runs["ablation_type"] == "remove_chunk"].copy()
    full = df_runs[df_runs["ablation_type"] == "full_context_control"].copy()

    adv = chunk_advantage(ablation, full)
    adv.to_csv(table_path("chunk_information_advantage.csv"), index=False)
    print(f"[10] wrote {len(adv)} chunk-advantage rows")
    if not adv.empty:
        print(adv.head(10)[["case_id", "chunk_id", "chunk_score_advantage",
                             "chunk_pass_advantage", "chunk_adv_norm"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
