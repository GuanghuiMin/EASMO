"""Stage 09a (EASMO/.venv) — build ablation_contexts.jsonl.

For each selected candidate × chunk:
  remove chunk → re-stress (T^K) → store post-stress text
Plus the "full_context_control" row: T^K(candidate.compressed_text)
(may be reusable from stress_chains.jsonl).

Output: outputs/raw/chunk_ablation_contexts.jsonl
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

from motivation_v9.data import ensure_outputs, read_jsonl, write_jsonl, raw_path  # noqa
from motivation_v9.clients import make_client  # noqa
from motivation_v9.acon_prompt_loader import load_utco_bundle  # noqa
from motivation_v9.stress import stress_chain  # noqa
from motivation_v9.chunks import remove_chunk, Chunk  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v9_cases.jsonl"))
    ap.add_argument("--candidates",
                    default=str(raw_path("candidate_compressions.jsonl")))
    ap.add_argument("--chunks", default=str(raw_path("chunks.jsonl")))
    ap.add_argument("--selection", default=str(raw_path("chunk_case_selection.jsonl")))
    ap.add_argument("--stress_chains",
                    default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--out", default=str(raw_path("chunk_ablation_contexts.jsonl")))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--target_max_chars", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    candidates = {c["candidate_id"]: c for c in read_jsonl(Path(args.candidates))}
    chunks = read_jsonl(Path(args.chunks))
    selection = read_jsonl(Path(args.selection))

    # Pre-load existing T^K(c) for full-context control from stress_chains.jsonl
    chain_final: Dict[str, str] = {}
    chain_final_round: Dict[str, int] = {}
    for r in read_jsonl(Path(args.stress_chains)):
        cid = r["candidate_id"]; round_ = int(r["round"])
        if round_ >= chain_final_round.get(cid, -1):
            chain_final_round[cid] = round_
            chain_final[cid] = r["context_text"]

    bundle = load_utco_bundle()
    models = sorted({c["compressor_model"] for c in candidates.values()})
    clients = {m: make_client(m) for m in models}

    # Build work list: per (selected candidate × each chunk + 1 control)
    selected_cand_ids = {s["selected_candidate_id"] for s in selection}
    chunks_by_cand: Dict[str, List[dict]] = defaultdict(list)
    for ch in chunks:
        chunks_by_cand[ch["candidate_id"]].append(ch)

    work: List[Tuple] = []
    rows: List[dict] = []
    # full-context controls
    for cid in selected_cand_ids:
        cand = candidates.get(cid)
        case = cases.get(cand["case_id"]) if cand else None
        if not cand or not case:
            continue
        post_text = chain_final.get(cid, cand["compressed_text"])
        rows.append({
            "ablation_id":      f"{cid}__full_control",
            "candidate_id":     cid,
            "chunk_id":         None,
            "case_id":          cand["case_id"],
            "ablation_type":    "full_context_control",
            "pre_stress_text":  cand["compressed_text"],
            "post_stress_text": post_text,
            "pre_stress_chars": len(cand["compressed_text"]),
            "post_stress_chars": len(post_text),
        })
    # chunk-removed contexts (these we need to re-stress)
    for cid in selected_cand_ids:
        cand = candidates.get(cid)
        case = cases.get(cand["case_id"]) if cand else None
        if not cand or not case:
            continue
        for ch in chunks_by_cand[cid]:
            work.append((cid, cand, case, ch))

    print(f"[09a] {len(selected_cand_ids)} controls + "
          f"{len(work)} chunk-removed contexts to re-stress")

    def _stress_one(item):
        cid, cand, case, ch = item
        # Build a Chunk-like object to use remove_chunk
        chunk_obj = Chunk(
            chunk_id=ch["chunk_id"], candidate_id=cid, case_id=ch["case_id"],
            chunk_index=ch["chunk_index"], chunk_text=ch["chunk_text"],
            chunk_chars=ch["chunk_chars"], chunk_tokens_est=ch["chunk_tokens_est"],
            char_span_start=ch["char_span_start"], char_span_end=ch["char_span_end"],
        )
        c_minus_j = remove_chunk(cand["compressed_text"], chunk_obj)
        rounds = stress_chain(
            candidate_id=cid, case_id=ch["case_id"],
            client=clients[cand["compressor_model"]],
            model_name=cand["compressor_model"], bundle=bundle,
            user_instruction=case["user_instruction"],
            c0=c_minus_j, rounds=args.rounds,
            max_chars=args.target_max_chars,
        )
        post_text = rounds[-1].context_text
        return {
            "ablation_id":      f"{cid}__{ch['chunk_id']}__removed",
            "candidate_id":     cid,
            "chunk_id":         ch["chunk_id"],
            "chunk_index":      ch["chunk_index"],
            "chunk_text":       ch["chunk_text"],
            "case_id":          ch["case_id"],
            "ablation_type":    "remove_chunk",
            "pre_stress_text":  c_minus_j,
            "post_stress_text": post_text,
            "pre_stress_chars": len(c_minus_j),
            "post_stress_chars": len(post_text),
        }

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_stress_one, w): w for w in work}
        for fut in as_completed(futs):
            rows.append(fut.result())
            n_done += 1
            if n_done % 20 == 0 or n_done <= 3:
                eta = (time.time() - t0) / n_done * (len(work) - n_done)
                print(f"  [{n_done:>4d}/{len(work)}] eta={eta:.0f}s", flush=True)
    write_jsonl(Path(args.out), rows)
    print(f"[09a] wrote {len(rows)} ablation contexts -> {args.out}")
    print(f"[09a] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
