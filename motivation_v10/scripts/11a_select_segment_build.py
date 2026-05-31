"""Stage 11a — select chunk candidates per case + segment + build
chunk-minus contexts via MiniMax stress.

For each test_behavior case (held-out) and per spec §17.2, take 5
candidate variants (MiniMax greedy / MiniMax oracle-best / proxy-
selected / Qwen-SFT-C1 / Qwen-SFT-CK) if they passed full CK.
Segment each into ≤12 chunks. For each chunk, build a chunk-minus
context by removing the chunk text and re-stressing K=2 rounds
via MiniMax-ACON-UTCO.

Writes:
  outputs/raw/v11_chunk_selection.jsonl     (which candidate per variant per case)
  outputs/raw/v11_chunks.jsonl               (one row per chunk)
  outputs/raw/v11_chunk_ablation_contexts.jsonl (one row per (chunk, control), with post-stress text)

Uses MiniMax for stress (EASMO/.venv).
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
    ensure_outputs, read_jsonl, write_jsonl, raw_path,
)
from motivation_v10.clients import make_client                      # noqa
from motivation_v10.acon_prompt_loader import load_utco_bundle      # noqa
from motivation_v10.stress import stress_chain                      # noqa
from motivation_v10.chunks import segment_chunks, remove_chunk, Chunk  # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------
# Step 1: select candidates per (case, variant)
# ----------------------------------------------------------------------

def select_candidates(args) -> List[dict]:
    """Return one row per (case_id, variant, candidate_text). Only includes
    candidates whose full-context CK passes.
    """
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    eligible_cases = [c for c in cases.values()
                      if c.get("split") in {"test_behavior"}]
    case_ids = {c["case_id"] for c in eligible_cases}
    print(f"[11a] eligible cases (test_behavior only): {len(case_ids)}")

    # ----- MiniMax candidates + their CK pass -----
    cands = _read_jsonl(args.candidates)
    cand_by_id = {c["candidate_id"]: c for c in cands}
    behavior = _read_jsonl(args.behavior_candidates)
    pass_ck: Dict[str, bool] = {}
    chars_ck: Dict[str, int] = {}
    score_ck: Dict[str, float] = {}
    for r in behavior:
        if r["eval_round"] == "CK" and not r.get("error"):
            pass_ck[r["candidate_id"]] = bool(r.get("success"))
            chars_ck[r["candidate_id"]] = r.get("compressed_chars", 0)
            score_ck[r["candidate_id"]] = r.get("score", 0.0)

    # Verifier composite for proxy-selected
    verifier = _read_jsonl(args.verifier)
    verifier_composite: Dict[str, float] = {}
    for r in verifier:
        if r["eval_round"] == "CK":
            verifier_composite[r["candidate_id"]] = r.get("composite", 0.0)

    # Stress chains for CK text per minimax candidate
    stress = _read_jsonl(args.stress_chains)
    ck_text: Dict[str, str] = {}
    final_round: Dict[str, int] = {}
    for r in stress:
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            ck_text[cid] = r["context_text"]

    # ----- Group MiniMax candidates by case -----
    minimax_by_case: Dict[str, List[dict]] = defaultdict(list)
    for c in cands:
        if c["case_id"] in case_ids:
            minimax_by_case[c["case_id"]].append(c)

    selections = []
    for cid in sorted(case_ids):
        mm_cands = minimax_by_case.get(cid, [])
        if not mm_cands:
            continue

        # (a) MiniMax greedy
        greedy = next((c for c in mm_cands if c["generation_type"]=="greedy"), None)
        if greedy and pass_ck.get(greedy["candidate_id"]):
            selections.append({
                "case_id": cid, "variant": "MiniMax-greedy",
                "candidate_id": greedy["candidate_id"],
                "compressed_text": greedy["compressed_text"],
                "selection_reason": "minimax_greedy_ck_pass",
                "stress_ck_text": ck_text.get(greedy["candidate_id"], greedy["compressed_text"]),
                "score_ck": score_ck.get(greedy["candidate_id"], 0.0),
            })

        # (b) MiniMax oracle: max CK score among passing candidates
        passing = [c for c in mm_cands
                   if pass_ck.get(c["candidate_id"], False)]
        if passing:
            oracle = max(passing, key=lambda c: score_ck.get(c["candidate_id"], 0.0))
            selections.append({
                "case_id": cid, "variant": "MiniMax-oracle",
                "candidate_id": oracle["candidate_id"],
                "compressed_text": oracle["compressed_text"],
                "selection_reason": "minimax_oracle_max_ck_score",
                "stress_ck_text": ck_text.get(oracle["candidate_id"], oracle["compressed_text"]),
                "score_ck": score_ck.get(oracle["candidate_id"], 0.0),
            })

        # (c) MiniMax proxy-selected (max verifier composite among CK-passing)
        if passing:
            with_proxy = [(c, verifier_composite.get(c["candidate_id"], float("-inf")))
                          for c in passing]
            with_proxy = [(c, s) for c, s in with_proxy if s != float("-inf")]
            if with_proxy:
                proxy_pick = max(with_proxy, key=lambda x: x[1])[0]
                selections.append({
                    "case_id": cid, "variant": "MiniMax-proxy-selected",
                    "candidate_id": proxy_pick["candidate_id"],
                    "compressed_text": proxy_pick["compressed_text"],
                    "selection_reason": "minimax_proxy_selected_ck_pass",
                    "stress_ck_text": ck_text.get(proxy_pick["candidate_id"], proxy_pick["compressed_text"]),
                    "score_ck": score_ck.get(proxy_pick["candidate_id"], 0.0),
                })

    # ----- (d) (e) Qwen-SFT-C1 and Qwen-SFT-CK greedy compressions -----
    # student_compressions.jsonl holds greedy outputs for the 3 Qwen variants.
    # student_behavior_runs.jsonl tells us CK pass per (variant, case_id).
    student_comps = _read_jsonl(args.student_compressions)
    student_behavior = _read_jsonl(args.student_behavior)
    s_pass_ck: Dict[Tuple[str, str], bool] = {}
    s_score_ck: Dict[Tuple[str, str], float] = {}
    for r in student_behavior:
        if r.get("eval_round") == "CK" and not r.get("error"):
            s_pass_ck[(r["variant"], r["case_id"])] = bool(r.get("success"))
            s_score_ck[(r["variant"], r["case_id"])] = r.get("score", 0.0)

    student_stress = _read_jsonl(args.student_stress)
    s_ck_text: Dict[Tuple[str, str], str] = {}
    s_final_round: Dict[Tuple[str, str], int] = {}
    for r in student_stress:
        k = (r["variant"], r["case_id"])
        if r["round"] >= s_final_round.get(k, -1):
            s_final_round[k] = r["round"]
            s_ck_text[k] = r["context_text"]

    for variant in ("Qwen-SFT-C1", "Qwen-SFT-CK"):
        for r in student_comps:
            if r["variant"] != variant or r["case_id"] not in case_ids:
                continue
            cid_full = f"{variant}__{r['case_id']}"
            if not s_pass_ck.get((variant, r["case_id"]), False):
                continue
            selections.append({
                "case_id": r["case_id"], "variant": variant,
                "candidate_id": cid_full,
                "compressed_text": r["compressed_text"],
                "selection_reason": f"{variant}_greedy_ck_pass",
                "stress_ck_text": s_ck_text.get((variant, r["case_id"]),
                                                r["compressed_text"]),
                "score_ck": s_score_ck.get((variant, r["case_id"]), 0.0),
            })

    return selections


# ----------------------------------------------------------------------
# Step 2: segment + Step 3: build chunk-minus contexts
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v10_cases.jsonl"))
    ap.add_argument("--candidates",
                    default=str(raw_path("minimax_candidates.jsonl")))
    ap.add_argument("--behavior_candidates",
                    default=str(raw_path("behavior_runs_candidates.jsonl")))
    ap.add_argument("--verifier",
                    default=str(raw_path("proxy_verifier_scores.jsonl")))
    ap.add_argument("--stress_chains",
                    default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--student_compressions",
                    default=str(raw_path("student_compressions.jsonl")))
    ap.add_argument("--student_behavior",
                    default=str(raw_path("student_behavior_runs.jsonl")))
    ap.add_argument("--student_stress",
                    default=str(raw_path("student_stress_chains.jsonl")))
    ap.add_argument("--selection_out", default=str(raw_path("v11_chunk_selection.jsonl")))
    ap.add_argument("--chunks_out", default=str(raw_path("v11_chunks.jsonl")))
    ap.add_argument("--ablation_out",
                    default=str(raw_path("v11_chunk_ablation_contexts.jsonl")))
    ap.add_argument("--max_chunks", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    # Step 1: select
    selections = select_candidates(args)
    write_jsonl(Path(args.selection_out), selections)
    print(f"[11a] wrote {len(selections)} candidate selections -> {args.selection_out}")
    from collections import Counter
    var_counter = Counter(s["variant"] for s in selections)
    print(f"     by variant: {dict(var_counter)}")

    # Step 2: segment
    chunk_rows: List[dict] = []
    for s in selections:
        chunks = segment_chunks(
            candidate_id=s["candidate_id"],
            case_id=s["case_id"],
            text=s["compressed_text"],
            max_chunks=args.max_chunks,
        )
        for ch in chunks:
            chunk_rows.append({
                "chunk_id":         ch.chunk_id,
                "candidate_id":     ch.candidate_id,
                "case_id":          ch.case_id,
                "variant":          s["variant"],
                "chunk_index":      ch.chunk_index,
                "chunk_text":       ch.chunk_text,
                "chunk_chars":      ch.chunk_chars,
                "chunk_tokens_est": ch.chunk_tokens_est,
                "char_span_start":  ch.char_span_start,
                "char_span_end":    ch.char_span_end,
            })
    write_jsonl(Path(args.chunks_out), chunk_rows)
    print(f"[11a] wrote {len(chunk_rows)} chunks -> {args.chunks_out}")
    if chunk_rows:
        per_var = Counter((c["variant"]) for c in chunk_rows)
        print(f"     chunks by variant: {dict(per_var)}")

    # Step 3: build chunk-minus contexts via MiniMax stress
    cases_by_id = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    bundle = load_utco_bundle()
    client = make_client("minimax")

    # Resume: skip already-written ablation_ids
    out_path = Path(args.ablation_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done_ids.add(r["ablation_id"])

    # full-context controls (reuse cached stress_ck_text from selection)
    controls = []
    for s in selections:
        ablation_id = f"{s['candidate_id']}__full_control"
        if ablation_id in done_ids:
            continue
        controls.append({
            "ablation_id":      ablation_id,
            "candidate_id":     s["candidate_id"],
            "chunk_id":         None,
            "case_id":          s["case_id"],
            "variant":          s["variant"],
            "ablation_type":    "full_context_control",
            "pre_stress_text":  s["compressed_text"],
            "post_stress_text": s["stress_ck_text"],
            "pre_stress_chars": len(s["compressed_text"]),
            "post_stress_chars": len(s["stress_ck_text"]),
        })

    # Write controls immediately (they need no API)
    with open(out_path, "a", encoding="utf-8") as f_out:
        for r in controls:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Chunk-removed contexts need MiniMax stress
    sel_by_cand = {s["candidate_id"]: s for s in selections}
    work: List[Tuple] = []
    for ch in chunk_rows:
        ablation_id = f"{ch['candidate_id']}__{ch['chunk_id']}__removed"
        if ablation_id in done_ids:
            continue
        s = sel_by_cand.get(ch["candidate_id"])
        case = cases_by_id.get(ch["case_id"])
        if not s or not case:
            continue
        work.append((s, ch, case))

    print(f"[11a] {len(controls)} controls written; "
          f"{len(work)} chunk-removed contexts to re-stress")

    def _stress_one(item):
        s, ch, case = item
        chunk_obj = Chunk(
            chunk_id=ch["chunk_id"], candidate_id=ch["candidate_id"],
            case_id=ch["case_id"], chunk_index=ch["chunk_index"],
            chunk_text=ch["chunk_text"], chunk_chars=ch["chunk_chars"],
            chunk_tokens_est=ch["chunk_tokens_est"],
            char_span_start=ch["char_span_start"],
            char_span_end=ch["char_span_end"],
        )
        c_minus_j = remove_chunk(s["compressed_text"], chunk_obj)
        if not c_minus_j.strip():
            return None
        rows_ = stress_chain(
            candidate_id=ch["candidate_id"], case_id=ch["case_id"],
            client=client, model_name="minimax", bundle=bundle,
            user_instruction=case["user_instruction"],
            c0=c_minus_j, rounds=args.rounds,
            max_chars=1500, max_tokens=2048,
        )
        post_text = rows_[-1].context_text
        return {
            "ablation_id":      f"{ch['candidate_id']}__{ch['chunk_id']}__removed",
            "candidate_id":     ch["candidate_id"],
            "chunk_id":         ch["chunk_id"],
            "chunk_index":      ch["chunk_index"],
            "chunk_text":       ch["chunk_text"],
            "case_id":          ch["case_id"],
            "variant":          ch["variant"],
            "ablation_type":    "remove_chunk",
            "pre_stress_text":  c_minus_j,
            "post_stress_text": post_text,
            "pre_stress_chars": len(c_minus_j),
            "post_stress_chars": len(post_text),
        }

    t0 = time.time(); n_done = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_stress_one, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is None: continue
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(work)-n_done)
                    print(f"  [{n_done}/{len(work)}] eta={eta:.0f}s", flush=True)
    print(f"[11a] stressed {n_done} chunk-minus contexts in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
