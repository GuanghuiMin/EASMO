"""Stage 04 — run AppWorld downstream MiniMax agent on C1 and CK per candidate.

MUST be invoked with /workspace/acon/.venv/bin/python (AppWorld /
productive_agents pydantic-v1 venv). Reuses v3's runner via
motivation_v4.runner.run_with_compressed_context.

For each candidate:
  C1 = candidate's own compressed_text
  CK = final stress-chain round text (from stress_chains.jsonl)

Output: outputs/raw/behavior_runs_candidates.jsonl
        (one row per candidate × {C1, CK})
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _run_one(args_tuple):
    (cand_id, case_id, split, model, gen_type, sample_id,
     eval_round, stress_round, text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context  # noqa

    res = run_with_compressed_context(
        task_id=case_id,
        method=f"v10_{model}_{gen_type}_{eval_round}",
        compressed_context=text,
        max_steps=max_steps,
        tag=tag,
    )
    return {
        "run_id":               f"{cand_id}__{eval_round}__cap{max_steps}",
        "candidate_id":         cand_id,
        "case_id":              case_id,
        "split":                split,
        "compressor_model":     model,
        "generation_type":      gen_type,
        "sample_id":            sample_id,
        "eval_round":           eval_round,
        "stress_round":         stress_round,
        "max_steps":            max_steps,
        "success":              res.success,
        "score":                res.final_reward,
        "iterations":           res.iterations,
        "termination_reason":   res.termination_reason,
        "input_tokens":         res.input_tokens,
        "output_tokens":        res.output_tokens,
        "compressed_chars":     len(text),
        "compressed_tokens_est": max(1, len(text) // 4),
        "elapsed_s":            res.elapsed_s,
        "output_dir":           res.output_dir,
        "error":                res.error,
    }


def _read_jsonl_plain(path):
    out = []
    if not Path(path).exists():
        return out
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default=str(_REPO / "outputs" / "raw" / "minimax_candidates.jsonl"))
    ap.add_argument("--stress",
                    default=str(_REPO / "outputs" / "raw" / "stress_chains.jsonl"))
    ap.add_argument("--out",
                    default=str(_REPO / "outputs" / "raw" / "behavior_runs_candidates.jsonl"))
    ap.add_argument("--cap_steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv10_behavior")
    ap.add_argument("--skip_ck", action="store_true", default=False)
    args = ap.parse_args()

    candidates = _read_jsonl_plain(args.candidates)
    candidates = [c for c in candidates
                  if c.get("compressed_text") and not c.get("error")]
    cand_index = {c["candidate_id"]: c for c in candidates}

    # Resolve CK text per candidate from stress_chains.jsonl
    stress_rows = _read_jsonl_plain(args.stress)
    final_round: Dict[str, int] = {}
    final_round_text: Dict[str, Tuple[int, str]] = {}
    for r in stress_rows:
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            final_round_text[cid] = (r["round"], r["context_text"])

    # Build work units
    work: List[Tuple] = []
    for cand in candidates:
        work.append((cand["candidate_id"], cand["case_id"],
                     cand.get("split", "unknown"),
                     cand["compressor_model"], cand["generation_type"],
                     cand["sample_id"], "C1", 0,
                     cand["compressed_text"], args.cap_steps, args.tag))
        if not args.skip_ck:
            fr = final_round_text.get(cand["candidate_id"])
            if fr is None:
                continue
            stress_r, text = fr
            work.append((cand["candidate_id"], cand["case_id"],
                         cand.get("split", "unknown"),
                         cand["compressor_model"], cand["generation_type"],
                         cand["sample_id"], "CK", stress_r,
                         text, args.cap_steps, args.tag))

    print(f"[04] {len(work)} agent runs "
          f"(candidates={len(candidates)}, cap_steps={args.cap_steps}, "
          f"include_ck={not args.skip_ck}, workers={args.workers})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_ids = set()
    if out_path.exists():
        for r in _read_jsonl_plain(out_path):
            done_ids.add(r.get("run_id"))
    print(f"[04] {len(done_ids)} already done")
    pending = [w for w in work
               if f"{w[0]}__{w[6]}__cap{w[9]}" not in done_ids]
    print(f"[04] pending: {len(pending)}")

    t0 = time.time()
    n_done = 0; n_err = 0; n_pass = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w for w in pending}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"error": str(e)}
                if rec.get("error"):
                    n_err += 1
                if rec.get("success"):
                    n_pass += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done <= 5:
                    eta = (time.time() - t0) / n_done * (len(pending) - n_done)
                    print(f"  [{n_done:>4d}/{len(pending)}] "
                          f"pass={n_pass:<3d} err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[04] done: {n_done} new runs ({n_pass} success, {n_err} errors)")
    print(f"[04] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
