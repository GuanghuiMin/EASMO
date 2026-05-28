"""Stage 06 — retention scoring (spec §12).

Two passes:
  A) single-round: score each (compressed_context, target_fact) row
     in single_round_compressions.jsonl.
  B) iterative + basin: score each (compressed_context_round_r,
     ALL_facts_in_case) — so that fixed-point composition can be
     computed across fact types.

Cross-model rule: Qwen compressions scored by MiniMax, MiniMax by Qwen.
Deterministic substring match short-circuits the LLM call.

Output: outputs/raw/retention_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, read_jsonl, raw_path,
)
from motivation_v8.clients import make_client  # noqa: E402
from motivation_v8.retention import score_fact_against_text, scorer_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts",
                    default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--single",
                    default=str(raw_path("single_round_compressions.jsonl")))
    ap.add_argument("--iterative",
                    default=str(raw_path("iterative_chains.jsonl")))
    ap.add_argument("--out", default=str(raw_path("retention_scores.jsonl")))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score_single_round", action="store_true", default=True)
    ap.add_argument("--score_iterative", action="store_true", default=True)
    args = ap.parse_args()

    ensure_outputs()
    fact_by_id = {f["fact_id"]: f for f in read_jsonl(Path(args.facts))}
    facts_by_case: Dict[str, List[dict]] = defaultdict(list)
    for f in fact_by_id.values():
        facts_by_case[f["case_id"]].append(f)

    clients = {n: make_client(n) for n in ("qwen", "minimax")}

    work: List[Tuple] = []

    # Single-round: score the target fact only
    if args.score_single_round and Path(args.single).exists():
        for row in read_jsonl(Path(args.single)):
            fact = fact_by_id.get(row["fact_id"])
            if not fact:
                continue
            work.append(("single_round", row, fact))

    # Iterative + basin: score ALL case-level facts at every round
    if args.score_iterative and Path(args.iterative).exists():
        for row in read_jsonl(Path(args.iterative)):
            if int(row.get("round", 0)) == 0:
                # round 0 is the initial context; still score (gives baseline)
                pass
            case_facts = facts_by_case.get(row["case_id"], [])
            for fact in case_facts:
                work.append(("iterative", row, fact))

    print(f"[06] {len(work)} retention scoring calls")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    def _score_one(item):
        source, row, fact = item
        scorer = scorer_for(row["model"])
        text = (row.get("compressed_context")
                if source == "single_round"
                else row.get("context_text") or "")
        result = score_fact_against_text(
            fact=fact, compressed_text=text or "",
            scorer_name=scorer, client=clients[scorer],
        )
        out = {
            "context_source": source,
            "case_id": row["case_id"],
            "fact_id": fact["fact_id"],
            "fact_type": fact["fact_type"],
            "coarse_group": fact["coarse_group"],
            "model": row["model"],
            "prompt_family": row.get("prompt_family"),
            "budget_chars": row["budget_chars"],
            "scorer_model": scorer,
            **result,
        }
        if source == "single_round":
            out["need_label"] = int(row.get("need_label", -1))
            out["chain_id"] = None
            out["round"] = 1
            out["init_type"] = None
            out["condition_type"] = None
        else:
            out["need_label"] = (1 if row.get("condition_type") == "needed"
                                 else (0 if row.get("condition_type") == "unneeded" else None))
            out["chain_id"] = row.get("chain_id")
            out["round"] = int(row.get("round"))
            out["init_type"] = row.get("init_type")
            out["condition_type"] = row.get("condition_type")
        return out

    t0 = time.time()
    n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_score_one, w) for w in work]
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("scorer_error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 500 == 0 or n_done <= 5:
                    eta = (time.time() - t0) / n_done * (len(work) - n_done)
                    print(f"  [{n_done:>5d}/{len(work)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
                if n_done % 500 == 0:
                    f_out.flush()
    print(f"[06] wrote {n_done} scores ({n_err} errors) -> {out_path}")
    print(f"[06] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
