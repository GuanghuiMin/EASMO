"""Stage 05 — proxy scoring of candidates (spec §11).

Two MiniMax proxies, both deterministic (temp=0.0, seed=42):

  §11.1 continuation_verifier   — pointwise 5-axis JSON rubric
                                  scored on (candidate × {C1, CK})
  §11.2 pairwise_preference     — sample vs greedy under CK only,
                                  one pairwise call per (case × sample)

§11.3 future_action_nll_proxy is an optional Qwen3-4B logprob run
implemented separately in `scripts/05b_nll_proxy.py` (not wired
into run_all.sh).

Outputs:
  outputs/raw/proxy_verifier_scores.jsonl
  outputs/raw/proxy_pairwise_scores.jsonl
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

from motivation_v10.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path,
)
from motivation_v10.clients import make_client                     # noqa
from motivation_v10.proxy import (                                 # noqa
    verifier_score, pairwise_preference,
)


def _read_jsonl(path):
    out = []
    if not Path(path).exists():
        return out
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _ck_text_per_candidate(stress_rows):
    """Return {candidate_id: (round, text)} for the final stress round."""
    final = {}
    for r in stress_rows:
        cid = r["candidate_id"]
        if r["round"] >= final.get(cid, (-1, ""))[0]:
            final[cid] = (r["round"], r["context_text"])
    return final


def _run_verifier_pass(*, cases, candidates, ck_text, client,
                       out_path, workers, max_steps, max_tokens):
    """Score every (candidate × {C1, CK}). Append to out_path."""
    work: List[Tuple] = []
    for cand in candidates:
        case = cases.get(cand["case_id"])
        if not case:
            continue
        work.append((cand, "C1", cand["compressed_text"], case["user_instruction"]))
        ck = ck_text.get(cand["candidate_id"])
        if ck is not None:
            work.append((cand, "CK", ck[1], case["user_instruction"]))

    done = set()
    if Path(out_path).exists():
        for r in _read_jsonl(out_path):
            done.add((r["candidate_id"], r["eval_round"]))
    pending = [w for w in work if (w[0]["candidate_id"], w[1]) not in done]
    print(f"[05.1] verifier: {len(work)} total work units, "
          f"{len(done)} done, {len(pending)} pending")

    def _do(item):
        cand, eval_round, text, instr = item
        s = verifier_score(
            candidate_id=cand["candidate_id"], eval_round=eval_round,
            user_instruction=instr, compressed_text=text,
            max_steps=max_steps, client=client, max_tokens=max_tokens,
        )
        return {
            "candidate_id":   s.candidate_id,
            "case_id":        cand["case_id"],
            "split":          cand.get("split", "unknown"),
            "eval_round":     s.eval_round,
            "verifier_model": s.verifier_model,
            "predicted_success_probability": s.predicted_success_probability,
            "missing_information_risk":      s.missing_information_risk,
            "execution_specificity":         s.execution_specificity,
            "risk_of_repeating_completed_actions": s.risk_of_repeating_completed_actions,
            "risk_of_wrong_api_arguments":   s.risk_of_wrong_api_arguments,
            "composite":      s.composite(),
            "short_reason":   s.short_reason,
            "error":          s.error,
        }

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_do, w): w for w in pending}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(pending) - n_done)
                    print(f"  [{n_done:>4d}/{len(pending)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[05.1] verifier done: {n_done} new ({n_err} errors), "
          f"elapsed {(time.time()-t0)/60:.1f} min")


def _run_pairwise_pass(*, cases, candidates, ck_text, client,
                       out_path, workers, max_steps, max_tokens):
    """Compare each sample vs greedy under CK. Append to out_path."""
    by_case_greedy: Dict[str, dict] = {}
    by_case_samples: Dict[str, List[dict]] = defaultdict(list)
    for c in candidates:
        if c["generation_type"] == "greedy":
            by_case_greedy[c["case_id"]] = c
        else:
            by_case_samples[c["case_id"]].append(c)

    work: List[Tuple] = []
    for case_id, greedy in by_case_greedy.items():
        case = cases.get(case_id)
        if not case:
            continue
        ck_g = ck_text.get(greedy["candidate_id"])
        if ck_g is None:
            continue
        for sample in by_case_samples.get(case_id, []):
            ck_s = ck_text.get(sample["candidate_id"])
            if ck_s is None:
                continue
            work.append((case_id, case.get("split", "unknown"),
                         greedy["candidate_id"], sample["candidate_id"],
                         case["user_instruction"], ck_g[1], ck_s[1]))

    done = set()
    if Path(out_path).exists():
        for r in _read_jsonl(out_path):
            done.add((r["candidate_a_id"], r["candidate_b_id"]))
    pending = [w for w in work if (w[2], w[3]) not in done]
    print(f"[05.2] pairwise: {len(work)} total, {len(done)} done, "
          f"{len(pending)} pending")

    def _do(item):
        case_id, split, a_id, b_id, instr, text_a, text_b = item
        p = pairwise_preference(
            case_id=case_id, eval_round="CK",
            candidate_a_id=a_id, candidate_b_id=b_id,
            user_instruction=instr, text_a=text_a, text_b=text_b,
            max_steps=max_steps, client=client, max_tokens=max_tokens,
        )
        return {
            "case_id":         p.case_id,
            "split":           split,
            "eval_round":      p.eval_round,
            "candidate_a_id":  p.candidate_a_id,
            "candidate_b_id":  p.candidate_b_id,
            "verifier_model":  p.verifier_model,
            "winner":          p.winner,
            "confidence":      p.confidence,
            "reason":          p.reason,
            "error":           p.error,
        }

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_do, w): w for w in pending}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(pending) - n_done)
                    print(f"  [{n_done:>4d}/{len(pending)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[05.2] pairwise done: {n_done} new ({n_err} errors), "
          f"elapsed {(time.time()-t0)/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v10_cases.jsonl"))
    ap.add_argument("--candidates",
                    default=str(raw_path("minimax_candidates.jsonl")))
    ap.add_argument("--stress",
                    default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--verifier_out",
                    default=str(raw_path("proxy_verifier_scores.jsonl")))
    ap.add_argument("--pairwise_out",
                    default=str(raw_path("proxy_pairwise_scores.jsonl")))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--max_tokens_verifier", type=int, default=2048)
    ap.add_argument("--max_tokens_pairwise", type=int, default=1536)
    ap.add_argument("--skip_pairwise", action="store_true", default=False)
    args = ap.parse_args()
    ensure_outputs()

    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    candidates = read_jsonl(Path(args.candidates))
    candidates = [c for c in candidates
                  if c.get("compressed_text") and not c.get("error")]
    stress_rows = _read_jsonl(args.stress)
    ck_text = _ck_text_per_candidate(stress_rows)
    client = make_client("minimax")

    print(f"[05] {len(candidates)} candidates, {len(ck_text)} CK texts resolved")

    _run_verifier_pass(
        cases=cases, candidates=candidates, ck_text=ck_text, client=client,
        out_path=args.verifier_out, workers=args.workers,
        max_steps=args.max_steps, max_tokens=args.max_tokens_verifier,
    )

    if not args.skip_pairwise:
        _run_pairwise_pass(
            cases=cases, candidates=candidates, ck_text=ck_text, client=client,
            out_path=args.pairwise_out, workers=args.workers,
            max_steps=args.max_steps, max_tokens=args.max_tokens_pairwise,
        )

    print("[05] all proxy passes done.")


if __name__ == "__main__":
    main()
