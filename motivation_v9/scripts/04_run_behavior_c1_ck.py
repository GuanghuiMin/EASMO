"""Stage 04 — run AppWorld downstream agent on C1 and CK per candidate.

MUST be invoked with /workspace/acon/.venv/bin/python (the appworld /
productive_agents pydantic-v1 venv). Reuses v3 runner via
motivation_v4.runner.run_with_compressed_context.

Output: outputs/raw/behavior_runs_c1_ck.jsonl  (one row per
        (candidate × eval_context_round × budget)).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _run_one(args_tuple):
    (cand_id, case_id, model, gen_type, sample_id,
     eval_round, stress_round, text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context  # noqa

    res = run_with_compressed_context(
        task_id=case_id,
        method=f"v9_{model}_{gen_type}_{eval_round}",
        compressed_context=text,
        max_steps=max_steps,
        tag=tag,
    )
    return {
        "run_id":             f"{cand_id}__{eval_round}__cap{max_steps}",
        "candidate_id":       cand_id,
        "case_id":            case_id,
        "compressor_model":   model,
        "generation_type":    gen_type,
        "sample_id":          sample_id,
        "eval_context_round": eval_round,
        "stress_round":       stress_round,
        "budget_name":        "loose_15" if max_steps == 15 else f"max_{max_steps}",
        "max_steps":          max_steps,
        "success":            res.success,
        "score":              res.final_reward,
        "iterations":         res.iterations,
        "termination_reason": res.termination_reason,
        "total_input_tokens": res.input_tokens,
        "peak_input_tokens":  res.input_tokens,
        "output_tokens":      res.output_tokens,
        "compressed_chars":   len(text),
        "compressed_tokens_est": max(1, len(text) // 4),
        "elapsed_s":          res.elapsed_s,
        "output_dir":         res.output_dir,
        "error":              res.error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default=str(_REPO / "outputs" / "raw" / "candidate_compressions.jsonl"))
    ap.add_argument("--stress",
                    default=str(_REPO / "outputs" / "raw" / "stress_chains.jsonl"))
    ap.add_argument("--out",
                    default=str(_REPO / "outputs" / "raw" / "behavior_runs_c1_ck.jsonl"))
    ap.add_argument("--budgets", nargs="+", type=int, default=[15])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv9_run")
    ap.add_argument("--skip_ck", action="store_true", default=False)
    args = ap.parse_args()

    # Load files manually (no v9 package imports in acon/.venv).
    def _read(p):
        out = []
        with open(p) as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

    candidates = _read(args.candidates)
    candidates = [c for c in candidates if c.get("compressed_text") and not c.get("error")]
    # Build candidate index by candidate_id
    cand_index = {c["candidate_id"]: c for c in candidates}

    stress_rows = _read(args.stress)
    final_round_text: Dict[str, Tuple[int, str]] = {}
    final_round: Dict[str, int] = {}
    for r in stress_rows:
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            final_round_text[cid] = (r["round"], r["context_text"])

    work: List[Tuple] = []
    for cand in candidates:
        # C1 = candidate's own compressed_text
        for cap in args.budgets:
            work.append((cand["candidate_id"], cand["case_id"],
                         cand["compressor_model"], cand["generation_type"],
                         cand["sample_id"], "C1", 0,
                         cand["compressed_text"], cap, args.tag))
        # CK = final stress-chain text
        if not args.skip_ck:
            fr = final_round_text.get(cand["candidate_id"])
            if fr:
                stress_r, text = fr
                for cap in args.budgets:
                    work.append((cand["candidate_id"], cand["case_id"],
                                 cand["compressor_model"], cand["generation_type"],
                                 cand["sample_id"], "CK", stress_r,
                                 text, cap, args.tag))

    print(f"[04] {len(work)} agent runs "
          f"(candidates={len(candidates)}, "
          f"budgets={args.budgets}, "
          f"include_ck={not args.skip_ck}, workers={args.workers})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append mode for resumability; check existing run_ids
    done_ids = set()
    if out_path.exists():
        for r in _read(out_path):
            done_ids.add(r.get("run_id"))
    print(f"[04] {len(done_ids)} already done; running remaining")

    pending = [w for w in work
               if f"{w[0]}__{w[6]}__cap{w[8]}" not in done_ids]
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
