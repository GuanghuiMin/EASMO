"""Stage 06c — continuation-entropy selector (spec §10.9).

For each (candidate, eval_round), sample M=5 short diagnostic
continuations and compute disagreement features:
  next_action_type_entropy / argument_key_jaccard_distance /
  missing_info_count_variance / confidence_entropy

Output: outputs/raw/continuation_entropy_samples.jsonl
        outputs/tables/continuation_entropy_selector_by_case.csv

By default runs only on `--families ACON_UTCO` (v11 plan (β)).
Pass `--families general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO`
to upgrade to (α) — incremental, no rework needed.
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
from motivation_v11.selectors import (                                             # noqa
    entropy_sample, entropy_features, entropy_selector_score,
)


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
    ap.add_argument("--samples_out",
                    default=str(raw_path("continuation_entropy_samples.jsonl")))
    ap.add_argument("--selector_out",
                    default=str(table_path("continuation_entropy_selector_by_case.csv")))
    ap.add_argument("--families", default="ACON_UTCO",
                    help="Comma-separated. Default ACON_UTCO only (v11 plan β). "
                         "Pass full list to extend to plan α.")
    ap.add_argument("--M", type=int, default=5, help="samples per candidate")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=1024)
    args = ap.parse_args()
    ensure_outputs()

    wanted_families = {f.strip() for f in args.families.split(",") if f.strip()}
    cases = {c["task_id"]: c for c in read_jsonl(Path(args.cases))}
    cands = [c for c in _read_jsonl(args.candidates)
             if c.get("c1_text") and not c.get("generation_error")
             and c["prompt_family"] in wanted_families]
    stress = _read_jsonl(args.stress)
    ck_text: Dict[str, Tuple[int, str]] = {}
    final_round: Dict[str, int] = {}
    for r in stress:
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]; ck_text[cid] = (r["round"], r["context_text"])

    out_path = Path(args.samples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done_keys.add((r["candidate_id"], r["eval_round"], r["sample_index"]))

    client = make_client("minimax")

    # Build work: per (candidate, round) × M samples
    work: List[Tuple] = []
    for cand in cands:
        case = cases.get(cand["task_id"])
        if not case: continue
        for eval_round, text in (("C1", cand["c1_text"]),
                                  ("CK", ck_text.get(cand["candidate_id"], (None, None))[1])):
            if not text: continue
            for i in range(args.M):
                if (cand["candidate_id"], eval_round, i) in done_keys: continue
                work.append((cand, eval_round, text, case["user_instruction"], i))

    print(f"[06c] entropy_samples: {len(work)} pending "
          f"(families={sorted(wanted_families)}, M={args.M})")

    def _do(item):
        cand, eval_round, text, instr, idx = item
        try:
            s = entropy_sample(
                candidate_id=cand["candidate_id"], eval_round=eval_round,
                sample_index=idx, user_instruction=instr,
                compressed_context=text, client=client,
                seed=2000 + idx, max_tokens=args.max_tokens,
            )
            return {
                "candidate_id":  s.candidate_id,
                "task_id":       cand["task_id"],
                "prompt_family": cand["prompt_family"],
                "eval_round":    s.eval_round,
                "sample_index":  s.sample_index,
                "next_action_type": s.next_action_type,
                "required_arguments_keys": s.required_arguments_keys,
                "missing_information":     s.missing_information,
                "confidence":              s.confidence,
                "error":                   s.error,
            }
        except Exception as e:
            return {"candidate_id": cand["candidate_id"], "task_id": cand["task_id"],
                     "prompt_family": cand["prompt_family"], "eval_round": eval_round,
                     "sample_index": idx, "error": str(e)}

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 100 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(work)-n_done)
                    print(f"  [{n_done}/{len(work)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[06c] sample collection done: {n_done} new ({n_err} errors); "
          f"elapsed {(time.time()-t0)/60:.1f} min")

    # Aggregate features → selector decisions per (task, family, round)
    samples = _read_jsonl(out_path)
    from motivation_v11.selectors import EntropySample
    by_cand = defaultdict(list)
    for r in samples:
        if r.get("error"): continue
        by_cand[(r["candidate_id"], r["eval_round"])].append(
            EntropySample(
                candidate_id=r["candidate_id"], eval_round=r["eval_round"],
                sample_index=r["sample_index"],
                next_action_type=r["next_action_type"],
                required_arguments_keys=r["required_arguments_keys"],
                missing_information=r["missing_information"],
                confidence=r["confidence"],
            )
        )

    # Build length lookup
    text_lens = {}
    for cand in cands:
        text_lens[(cand["candidate_id"], "C1")] = cand["c1_chars"]
        ck = ck_text.get(cand["candidate_id"])
        if ck: text_lens[(cand["candidate_id"], "CK")] = len(ck[1])

    # Compute selector score per (candidate, round) → group by (task, family, round)
    cand_score: Dict[Tuple[str, str], float] = {}
    for key, ss in by_cand.items():
        feats = entropy_features(ss)
        length_chars = text_lens.get(key, 0)
        cand_score[key] = entropy_selector_score(feats, length_chars)

    by_tf: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    cand_idx = {c["candidate_id"]: c for c in cands}
    for (cid, rnd), score in cand_score.items():
        cand = cand_idx.get(cid)
        if not cand or cand["candidate_type"] != "sample": continue
        by_tf[(cand["task_id"], cand["prompt_family"], rnd)].append((cid, score))

    sel_rows = []
    for (task_id, family, rnd), pairs in by_tf.items():
        pairs.sort(key=lambda x: -x[1])  # highest score first
        winner = pairs[0]
        sel_rows.append({
            "task_id":             task_id,
            "prompt_family":       family,
            "eval_round":          rnd,
            "winner_candidate_id": winner[0],
            "winner_score":        winner[1],
            "n_candidates":        len(pairs),
        })
    pd.DataFrame(sel_rows).to_csv(args.selector_out, index=False)
    print(f"[06c] entropy selector decisions -> {args.selector_out} "
          f"({len(sel_rows)} rows)")


if __name__ == "__main__":
    main()
