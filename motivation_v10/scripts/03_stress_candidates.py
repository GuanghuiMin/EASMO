"""Stage 03 — repeated-compression stress chains (v10).

For each candidate, compute T^0..T^K via MiniMax-M2.5 ACON UTCO
recompression. Mirrors v9 stage 03 but reads/writes v10 paths.

Output: outputs/raw/stress_chains.jsonl  (K+1 rows per candidate)
        outputs/tables/stress_chain_convergence.csv  (one row / candidate)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd

from motivation_v10.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path, table_path,
)
from motivation_v10.clients import make_client  # noqa
from motivation_v10.acon_prompt_loader import load_utco_bundle  # noqa
from motivation_v10.stress import stress_chain, chain_convergence  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v10_cases.jsonl"))
    ap.add_argument("--candidates",
                    default=str(raw_path("minimax_candidates.jsonl")))
    ap.add_argument("--out", default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--conv_out",
                    default=str(table_path("stress_chain_convergence.csv")))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--target_max_chars", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    candidates = read_jsonl(Path(args.candidates))
    candidates = [c for c in candidates
                  if c.get("compressed_text") and not c.get("error")]
    bundle = load_utco_bundle()
    client = make_client("minimax")

    print(f"[03] {len(candidates)} candidates × {args.rounds} stress rounds (MiniMax)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumability — skip already-stressed candidate_ids
    done_cids = set()
    if out_path.exists():
        for line in open(out_path):
            line = line.strip()
            if not line:
                continue
            try:
                done_cids.add(json.loads(line).get("candidate_id"))
            except Exception:
                pass
    pending = [c for c in candidates if c["candidate_id"] not in done_cids]
    print(f"[03] {len(done_cids)} candidates already stressed; {len(pending)} pending")

    def _run_one(cand):
        case = cases.get(cand["case_id"])
        if not case:
            return [], None
        rows = stress_chain(
            candidate_id=cand["candidate_id"],
            case_id=cand["case_id"],
            client=client,
            model_name="minimax",
            bundle=bundle,
            user_instruction=case["user_instruction"],
            c0=cand["compressed_text"],
            rounds=args.rounds,
            max_chars=args.target_max_chars,
            max_tokens=args.max_tokens,
        )
        json_rows = [{
            "candidate_id":     r.candidate_id,
            "case_id":          r.case_id,
            "compressor_model": r.compressor_model,
            "round":            r.round,
            "context_text":     r.context_text,
            "chars":            r.chars,
            "tokens_est":       r.tokens_est,
            "text_hash":        r.text_hash,
            "elapsed_s":        r.elapsed_s,
            "error":            r.error,
        } for r in rows]
        conv = chain_convergence(rows)
        conv_row = {
            "candidate_id":    cand["candidate_id"],
            "case_id":         cand["case_id"],
            "model":           "minimax",
            "split":           cand.get("split", "unknown"),
            "generation_type": cand["generation_type"],
            **conv,
        }
        return json_rows, conv_row

    t0 = time.time()
    n_done = 0; n_err = 0; conv_rows: List[dict] = []
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, c): c for c in pending}
            for fut in as_completed(futs):
                rows, conv = fut.result()
                for r in rows:
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if r.get("error"):
                        n_err += 1
                f_out.flush()
                if conv:
                    conv_rows.append(conv)
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(pending) - n_done)
                    print(f"  [{n_done:>4d}/{len(pending)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)

    # If we had pre-existing rows, merge their convergence rows from disk.
    if Path(args.conv_out).exists() and conv_rows:
        try:
            prev = pd.read_csv(args.conv_out).to_dict(orient="records")
            seen = {r["candidate_id"] for r in conv_rows}
            for r in prev:
                if r["candidate_id"] not in seen:
                    conv_rows.append(r)
        except Exception:
            pass
    if conv_rows:
        pd.DataFrame(conv_rows).to_csv(args.conv_out, index=False)
    print(f"[03] wrote {n_done} new chains ({n_err} compression errors) -> {out_path}")
    print(f"[03] convergence -> {args.conv_out}")
    print(f"[03] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
