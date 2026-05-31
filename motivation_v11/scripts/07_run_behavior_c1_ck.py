"""Stage 05 — AppWorld downstream agent runs on C1 and CK per candidate.

MUST run with /workspace/acon/.venv/bin/python.

Output: outputs/raw/behavior_runs.jsonl
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
    (cid, task_id, family, candidate_type, sample_id,
     eval_round, text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context  # noqa

    res = run_with_compressed_context(
        task_id=task_id,
        method=f"v11_{family}_{candidate_type}_{eval_round}",
        compressed_context=text,
        max_steps=max_steps,
        tag=tag,
    )
    return {
        "run_id":           f"{cid}__{eval_round}__cap{max_steps}",
        "task_id":          task_id,
        "prompt_family":    family,
        "candidate_id":     cid,
        "candidate_type":   candidate_type,
        "sample_id":        sample_id,
        "eval_round":       eval_round,
        "compressed_context_chars": len(text),
        "compressed_context_tokens_est": max(1, len(text) // 4),
        "max_steps":        max_steps,
        "success":          res.success,
        "score":            res.final_reward,
        "iterations":       res.iterations,
        "termination_reason": res.termination_reason,
        "total_input_tokens": res.input_tokens,
        "output_tokens":    res.output_tokens,
        "elapsed_s":        res.elapsed_s,
        "output_dir":       res.output_dir,
        "error":            res.error,
    }


def _read_jsonl_plain(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default=str(_REPO / "outputs" / "raw" / "candidate_compressions_c1.jsonl"))
    ap.add_argument("--stress",
                    default=str(_REPO / "outputs" / "raw" / "stress_chains.jsonl"))
    ap.add_argument("--out",
                    default=str(_REPO / "outputs" / "raw" / "behavior_runs.jsonl"))
    ap.add_argument("--cap_steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv11_behavior")
    args = ap.parse_args()

    cands = _read_jsonl_plain(args.candidates)
    cands = [c for c in cands if c.get("c1_text") and not c.get("generation_error")]
    cand_idx = {c["candidate_id"]: c for c in cands}

    # Resolve CK text per candidate
    stress = _read_jsonl_plain(args.stress)
    ck_text: Dict[str, Tuple[int, str]] = {}
    final_round: Dict[str, int] = {}
    for r in stress:
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            ck_text[cid] = (r["round"], r["context_text"])

    work: List[Tuple] = []
    for cand in cands:
        cid = cand["candidate_id"]
        family = cand["prompt_family"]
        ctype = cand["candidate_type"]
        sid = cand.get("sample_id", -1)
        # C1
        work.append((cid, cand["task_id"], family, ctype, sid,
                     "C1", cand["c1_text"], args.cap_steps, args.tag))
        # CK
        ck = ck_text.get(cid)
        if ck is not None:
            work.append((cid, cand["task_id"], family, ctype, sid,
                         "CK", ck[1], args.cap_steps, args.tag))

    print(f"[05] {len(work)} agent runs "
          f"(candidates={len(cands)}, both C1+CK)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for r in _read_jsonl_plain(out_path):
            done_ids.add(r.get("run_id"))
    pending = [w for w in work
               if f"{w[0]}__{w[5]}__cap{w[7]}" not in done_ids]
    print(f"[05] {len(done_ids)} done; {len(pending)} pending")

    t0 = time.time(); n_done = 0; n_pass = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w for w in pending}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"error": str(e)}
                if rec.get("error"): n_err += 1
                if rec.get("success"): n_pass += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(pending)-n_done)
                    print(f"  [{n_done}/{len(pending)}] "
                          f"pass={n_pass} err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[05] done: {n_done} runs ({n_pass} success, {n_err} errors); "
          f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
