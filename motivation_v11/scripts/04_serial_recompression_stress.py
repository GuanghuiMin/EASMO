"""Stage 04 — serial recompression stress C1 → CK with **same prompt family**.

For every C1 candidate:
  x = c1_text
  for r in 1..K:
    x = compress(family, history=x, task=instruction, temp=0.0, seed=42)

Output: outputs/raw/stress_chains.jsonl
        (K+1 rows per candidate: round=0 is c1, round=K is ck)
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, read_jsonl, raw_path  # noqa
from motivation_v11.clients import make_client, chat                   # noqa
from motivation_v11 import prompt_families as pf                        # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def _sha256(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _stress_one_candidate(*, cand, bundle, client, K_STRESS, max_chars, max_tokens):
    rounds = [{
        "task_id":        cand["task_id"],
        "prompt_family":  cand["prompt_family"],
        "candidate_id":   cand["candidate_id"],
        "round":          0,
        "round_name":     "C1",
        "context_text":   cand["c1_text"],
        "chars":          cand["c1_chars"],
        "tokens_est":     cand.get("c1_tokens_est", max(1, cand["c1_chars"] // 4)),
        "text_sha256":    cand.get("text_sha256", _sha256(cand["c1_text"])),
        "text_similarity_to_prev": None,
        "length_drift_from_c1_pct": 0.0,
        "exact_same_as_prev": False,
        "generation_error": None,
    }]
    current = cand["c1_text"]
    for r in range(1, K_STRESS + 1):
        user = pf.render(bundle, task=cand["task_instruction"],
                          history=current, max_chars=max_chars)
        try:
            res = chat(name="minimax", user=user, system=bundle.system_text,
                        temperature=0.0, max_tokens=max_tokens, seed=42,
                        client=client, json_mode=False)
            text = res.text or current  # if collapse, keep prev
            err = res.error
        except Exception as e:
            text = current; err = str(e)
        sim = difflib.SequenceMatcher(None, text, current).ratio()
        rounds.append({
            "task_id":        cand["task_id"],
            "prompt_family":  cand["prompt_family"],
            "candidate_id":   cand["candidate_id"],
            "round":          r,
            "round_name":     f"stress_{r}" if r < K_STRESS else "CK",
            "context_text":   text,
            "chars":          len(text),
            "tokens_est":     max(1, len(text) // 4),
            "text_sha256":    _sha256(text),
            "text_similarity_to_prev": sim,
            "length_drift_from_c1_pct":
                (len(text) - cand["c1_chars"]) / max(cand["c1_chars"], 1),
            "exact_same_as_prev": text == current,
            "generation_error": err,
        })
        current = text
    return rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(raw_path("compression_candidates_c1.jsonl")))
    ap.add_argument("--out", default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    cands = _read_jsonl(args.candidates)
    cands = [c for c in cands if c.get("c1_text") and not c.get("generation_error")]
    print(f"[04] {len(cands)} candidates × K={args.rounds} stress rounds")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_cids = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done_cids.add(r["candidate_id"])
    pending = [c for c in cands if c["candidate_id"] not in done_cids]
    print(f"[04] {len(done_cids)} already stressed; {len(pending)} pending")

    client = make_client("minimax")
    bundles = {f: pf.get_bundle(f) for f in pf.all_families()}

    def _do(c):
        return _stress_one_candidate(
            cand=c, bundle=bundles[c["prompt_family"]],
            client=client, K_STRESS=args.rounds,
            max_chars=args.max_chars, max_tokens=args.max_tokens,
        )

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, c): c for c in pending}
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    if r.get("generation_error"): n_err += 1
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(pending)-n_done)
                    print(f"  [{n_done}/{len(pending)}] err_rows={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[04] stressed {n_done} candidates ({n_err} row errors); "
          f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
