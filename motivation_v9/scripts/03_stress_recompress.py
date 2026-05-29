"""Stage 03 — repeated-compression stress chains.

For each candidate, compute c_0 = candidate; c_r = ACON(c_{r-1}; task).
Write all rounds to outputs/raw/stress_chains.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd

from motivation_v9.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path, table_path,
)
from motivation_v9.clients import make_client  # noqa
from motivation_v9.acon_prompt_loader import load_utco_bundle  # noqa
from motivation_v9.stress import stress_chain, chain_convergence  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v9_cases.jsonl"))
    ap.add_argument("--candidates",
                    default=str(raw_path("candidate_compressions.jsonl")))
    ap.add_argument("--out", default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--conv_out",
                    default=str(table_path("stress_chain_convergence.csv")))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--target_max_chars", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    candidates = read_jsonl(Path(args.candidates))
    candidates = [c for c in candidates if c.get("compressed_text") and not c.get("error")]
    bundle = load_utco_bundle()
    models = sorted({c["compressor_model"] for c in candidates})
    clients = {m: make_client(m) for m in models}

    print(f"[03] {len(candidates)} candidates × {args.rounds} stress rounds; "
          f"models={models}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    def _run_one(cand):
        case = cases.get(cand["case_id"])
        if not case:
            return [], {}
        rows = stress_chain(
            candidate_id=cand["candidate_id"],
            case_id=cand["case_id"],
            client=clients[cand["compressor_model"]],
            model_name=cand["compressor_model"],
            bundle=bundle,
            user_instruction=case["user_instruction"],
            c0=cand["compressed_text"],
            rounds=args.rounds,
            max_chars=args.target_max_chars,
        )
        json_rows = [{
            "candidate_id": r.candidate_id,
            "case_id":      r.case_id,
            "compressor_model": r.compressor_model,
            "round":        r.round,
            "context_text": r.context_text,
            "chars":        r.chars,
            "tokens_est":   r.tokens_est,
            "text_hash":    r.text_hash,
            "elapsed_s":    r.elapsed_s,
            "error":        r.error,
        } for r in rows]
        conv = chain_convergence(rows)
        conv_row = {
            "candidate_id": cand["candidate_id"],
            "case_id":      cand["case_id"],
            "model":        cand["compressor_model"],
            "generation_type": cand["generation_type"],
            **conv,
        }
        return json_rows, conv_row

    t0 = time.time()
    n_done = 0; n_err = 0; conv_rows: List[dict] = []
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, c): c for c in candidates}
            for fut in as_completed(futs):
                rows, conv = fut.result()
                for r in rows:
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if r.get("error"):
                        n_err += 1
                if conv:
                    conv_rows.append(conv)
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(candidates) - n_done)
                    print(f"  [{n_done:>4d}/{len(candidates)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)

    pd.DataFrame(conv_rows).to_csv(args.conv_out, index=False)
    print(f"[03] wrote {n_done} chains ({n_err} compression errors) -> {out_path}")
    print(f"[03] convergence -> {args.conv_out}")
    print(f"[03] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
