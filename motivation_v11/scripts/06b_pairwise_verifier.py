"""Stage 06b — MiniMax pairwise tournament selector (spec §10.8).

Tournament procedure: current = sample_0; for sample_i in 1..7:
  current = pairwise_winner(current, sample_i)

Output: outputs/raw/pairwise_verifier_matches.jsonl  (one row per pairwise match)
        outputs/tables/pairwise_selector_by_case.csv (winner per case-family-round)
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

import pandas as pd

from motivation_v11.data import ensure_outputs, read_jsonl, raw_path, table_path  # noqa
from motivation_v11.clients import make_client                                     # noqa
from motivation_v11.selectors import pairwise_match                                # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v11_primary_cases.jsonl"))
    ap.add_argument("--candidates", default=str(raw_path("compression_candidates_c1.jsonl")))
    ap.add_argument("--stress", default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--matches_out", default=str(raw_path("pairwise_verifier_matches.jsonl")))
    ap.add_argument("--selector_out", default=str(table_path("pairwise_selector_by_case.csv")))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=1024)
    args = ap.parse_args()
    ensure_outputs()

    cases = {c["task_id"]: c for c in read_jsonl(Path(args.cases))}
    cands = [c for c in _read_jsonl(args.candidates)
             if c.get("c1_text") and not c.get("generation_error")]
    stress = _read_jsonl(args.stress)
    ck_text: Dict[str, Tuple[int, str]] = {}
    final_round: Dict[str, int] = {}
    for r in stress:
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]; ck_text[cid] = (r["round"], r["context_text"])

    # Group samples per (task, family) → tournament
    by_tf: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for c in cands:
        if c["candidate_type"] == "sample":
            by_tf[(c["task_id"], c["prompt_family"])].append(c)
    # Sort by sample_id
    for k in by_tf:
        by_tf[k].sort(key=lambda c: c["sample_id"])

    client = make_client("minimax")
    out_path = Path(args.matches_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_matches = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done_matches.add((r["candidate_a_id"], r["candidate_b_id"], r["eval_round"]))

    # Build tournament list: per (task,family) × {C1, CK} run 7 matches sequentially
    # We can parallelize across (task, family, round) but each tournament is sequential.
    # For simplicity: launch all 7 matches per tournament at once? That breaks tournament logic.
    # Instead: run each tournament inline within one worker.

    work_units: List[Tuple] = []
    for (task_id, family), samples in by_tf.items():
        if len(samples) < 2: continue
        case = cases.get(task_id)
        if not case: continue
        for eval_round in ("C1", "CK"):
            # Build per-sample text for this round
            texts = {}
            for s in samples:
                if eval_round == "C1":
                    texts[s["candidate_id"]] = s["c1_text"]
                else:
                    ck = ck_text.get(s["candidate_id"])
                    if ck: texts[s["candidate_id"]] = ck[1]
            if len(texts) < 2: continue
            work_units.append((task_id, family, eval_round, samples,
                               texts, case["user_instruction"]))

    print(f"[06b] {len(work_units)} tournament units "
          f"({len(work_units)} × ~7 matches = ~{len(work_units)*7} matches)")

    def _tournament(item):
        task_id, family, eval_round, samples, texts, instr = item
        current = samples[0]
        rows = []
        for nxt in samples[1:]:
            key = (current["candidate_id"], nxt["candidate_id"], eval_round)
            if key in done_matches:
                # Skip to next; we'd need to load prior winner — but since matches
                # are sequential, prior winner unknown without state. Re-run for safety.
                pass
            ta = texts.get(current["candidate_id"], "")
            tb = texts.get(nxt["candidate_id"], "")
            if not ta or not tb:
                # Skip incomplete
                current = current  # no change
                continue
            try:
                m = pairwise_match(
                    case_id=task_id, prompt_family=family,
                    eval_round=eval_round,
                    candidate_a_id=current["candidate_id"],
                    candidate_b_id=nxt["candidate_id"],
                    user_instruction=instr, context_a=ta, context_b=tb,
                    client=client, max_tokens=args.max_tokens,
                )
                rows.append({
                    "task_id":        task_id,
                    "prompt_family":  family,
                    "eval_round":     eval_round,
                    "candidate_a_id": current["candidate_id"],
                    "candidate_b_id": nxt["candidate_id"],
                    "winner":         m.winner,
                    "confidence":     m.confidence,
                    "reason":         m.reason,
                    "error":          m.error,
                })
                if m.winner == "B":
                    current = nxt
            except Exception as e:
                rows.append({
                    "task_id": task_id, "prompt_family": family,
                    "eval_round": eval_round,
                    "candidate_a_id": current["candidate_id"],
                    "candidate_b_id": nxt["candidate_id"],
                    "winner": "tie", "error": str(e),
                })
        return rows, {
            "task_id":        task_id,
            "prompt_family":  family,
            "eval_round":     eval_round,
            "winner_candidate_id": current["candidate_id"],
        }

    t0 = time.time(); n_done = 0; n_err = 0; selector_rows = []
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_tournament, w): w for w in work_units}
            for fut in as_completed(futs):
                rows, sel = fut.result()
                for r in rows:
                    if r.get("error"): n_err += 1
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                selector_rows.append(sel)
                n_done += 1
                if n_done % 25 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(work_units)-n_done)
                    print(f"  [{n_done}/{len(work_units)} tourneys] err_matches={n_err} eta={eta:.0f}s",
                          flush=True)

    pd.DataFrame(selector_rows).to_csv(args.selector_out, index=False)
    print(f"[06b] done: {n_done} tournaments ({n_err} match errors); "
          f"elapsed {(time.time()-t0)/60:.1f} min")
    print(f"[06b] selector decisions -> {args.selector_out}")


if __name__ == "__main__":
    main()
