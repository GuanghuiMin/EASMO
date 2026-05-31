"""Stage 06a — MiniMax pointwise verifier scores (spec §10.7)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, read_jsonl, raw_path  # noqa
from motivation_v11.clients import make_client                         # noqa
from motivation_v11.selectors import pointwise_score                   # noqa


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
    ap.add_argument("--out", default=str(raw_path("pointwise_verifier_scores.jsonl")))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=1536)
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

    client = make_client("minimax")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done.add((r["candidate_id"], r["eval_round"]))

    work = []
    for cand in cands:
        case = cases.get(cand["task_id"])
        if not case: continue
        key_c1 = (cand["candidate_id"], "C1")
        if key_c1 not in done:
            work.append((cand, "C1", cand["c1_text"], case["user_instruction"]))
        ck = ck_text.get(cand["candidate_id"])
        if ck:
            key_ck = (cand["candidate_id"], "CK")
            if key_ck not in done:
                work.append((cand, "CK", ck[1], case["user_instruction"]))

    print(f"[06a] pointwise verifier: {len(work)} pending of "
          f"{2*len(cands)} total")

    def _do(item):
        cand, eval_round, text, instr = item
        s = pointwise_score(
            candidate_id=cand["candidate_id"], eval_round=eval_round,
            user_instruction=instr, compressed_context=text,
            client=client, max_tokens=args.max_tokens,
        )
        return {
            "candidate_id":   s.candidate_id,
            "task_id":        cand["task_id"],
            "prompt_family":  cand["prompt_family"],
            "eval_round":     s.eval_round,
            "verifier_model": s.verifier_model,
            "sufficiency_score": s.sufficiency_score,
            "risk_score":        s.risk_score,
            "missing_critical_information": s.missing_critical_information,
            "likely_to_succeed": s.likely_to_succeed,
            "one_sentence_reason": s.one_sentence_reason,
            "length_chars": len(text),
            "selector_score":  s.selector_score(len(text)),
            "error":           s.error,
        }

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(work)-n_done)
                    print(f"  [{n_done}/{len(work)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[06a] done: {n_done} ({n_err} errors); elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
