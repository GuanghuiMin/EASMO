"""Stage 07 — select up to CHUNK_ABLATION_MAX_CASES candidate pairs for chunk analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd

from motivation_v9.data import (  # noqa
    ensure_outputs, read_jsonl, write_jsonl, raw_path, table_path,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_case",
                    default=str(table_path("best_of_n_by_case.csv")))
    ap.add_argument("--candidates",
                    default=str(raw_path("candidate_compressions.jsonl")))
    ap.add_argument("--runs",
                    default=str(raw_path("behavior_runs_c1_ck.jsonl")))
    ap.add_argument("--transitions",
                    default=str(table_path("c1_ck_transition.csv")))
    ap.add_argument("--out", default=str(raw_path("chunk_case_selection.jsonl")))
    ap.add_argument("--max_cases", type=int, default=12)
    args = ap.parse_args()
    ensure_outputs()

    per_case = pd.read_csv(args.per_case) if Path(args.per_case).exists() else pd.DataFrame()
    trans = pd.read_csv(args.transitions) if Path(args.transitions).exists() else pd.DataFrame()
    candidates = read_jsonl(Path(args.candidates))
    cand_idx: Dict[str, dict] = {c["candidate_id"]: c for c in candidates}

    # Priority groups (spec §7)
    chosen: List[dict] = []
    selected_cases = set()

    # 1) best_sample_CK succeeds while greedy_CK fails
    ck = per_case[per_case["eval_context_round"] == "CK"] if not per_case.empty else pd.DataFrame()
    grp1 = ck[(~ck["greedy_success"]) & (ck["best_sample_success"])]
    for _, r in grp1.iterrows():
        if r["case_id"] in selected_cases:
            continue
        cmp_cand_id = f"{r['case_id']}__{r['compressor_model']}__greedy"
        chosen.append({
            "case_id": r["case_id"], "compressor_model": r["compressor_model"],
            "selected_candidate_id": f"{r['case_id']}__{r['compressor_model']}__{r['best_sample_id']}",
            "comparison_candidate_id": cmp_cand_id,
            "selection_reason": "best_sample_CK_succeeds_greedy_fails",
            "context_round_for_chunks": "C1",
            "stress_round_for_eval": "CK",
        })
        selected_cases.add(r["case_id"])

    # 2) fragile_pass candidates
    if not trans.empty:
        frag = trans[trans["class"] == "fragile_pass"]
        for _, r in frag.iterrows():
            cid = cand_idx.get(r["candidate_id"], {}).get("case_id")
            if not cid or cid in selected_cases or len(chosen) >= args.max_cases:
                continue
            chosen.append({
                "case_id": cid, "compressor_model": r["compressor_model"],
                "selected_candidate_id": r["candidate_id"],
                "comparison_candidate_id": None,
                "selection_reason": "fragile_pass",
                "context_round_for_chunks": "C1",
                "stress_round_for_eval": "CK",
            })
            selected_cases.add(cid)
            if len(chosen) >= args.max_cases:
                break

    # 3) both pass; sample shorter (length saving)
    grp3 = ck[
        (ck["greedy_success"]) & (ck["best_sample_success"]) &
        (ck["best_sample_length"] < ck["greedy_length"])
    ] if not ck.empty else pd.DataFrame()
    for _, r in grp3.iterrows():
        if r["case_id"] in selected_cases or len(chosen) >= args.max_cases:
            continue
        chosen.append({
            "case_id": r["case_id"], "compressor_model": r["compressor_model"],
            "selected_candidate_id": f"{r['case_id']}__{r['compressor_model']}__{r['best_sample_id']}",
            "comparison_candidate_id": f"{r['case_id']}__{r['compressor_model']}__greedy",
            "selection_reason": "both_pass_shorter_best",
            "context_round_for_chunks": "C1",
            "stress_round_for_eval": "CK",
        })
        selected_cases.add(r["case_id"])

    chosen = chosen[: args.max_cases]
    write_jsonl(Path(args.out), chosen)
    print(f"[07] selected {len(chosen)} candidate pairs -> {args.out}")
    if chosen:
        from collections import Counter
        print("[07] reason breakdown:", dict(Counter(c["selection_reason"] for c in chosen)))


if __name__ == "__main__":
    main()
