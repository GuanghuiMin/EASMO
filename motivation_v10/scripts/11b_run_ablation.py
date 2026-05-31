"""Stage 11b — run AppWorld agent on each chunk-minus context.

Reads `outputs/raw/v11_chunk_ablation_contexts.jsonl` and runs the
MiniMax downstream agent on every `post_stress_text` (full_context
controls + remove_chunk ablations). Each run gives us the per-chunk
behavioral advantage when paired with the full-control row.

Must run in /workspace/acon/.venv (AppWorld / productive_agents
pydantic-v1 venv).

Output: outputs/raw/v11_chunk_ablation_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _run_one(args_tuple):
    (ablation_id, case_id, variant, ablation_type, chunk_id,
     text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context  # noqa

    res = run_with_compressed_context(
        task_id=case_id,
        method=f"v11_{variant}_{ablation_type}",
        compressed_context=text,
        max_steps=max_steps,
        tag=tag,
    )
    return {
        "run_id":           f"{ablation_id}__cap{max_steps}",
        "ablation_id":      ablation_id,
        "case_id":          case_id,
        "variant":          variant,
        "ablation_type":    ablation_type,
        "chunk_id":         chunk_id,
        "max_steps":        max_steps,
        "success":          res.success,
        "score":            res.final_reward,
        "iterations":       res.iterations,
        "context_chars":    len(text),
        "termination_reason": res.termination_reason,
        "elapsed_s":        res.elapsed_s,
        "output_dir":       res.output_dir,
        "error":            res.error,
    }


def _read_jsonl_plain(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_contexts",
                    default=str(_REPO / "outputs" / "raw" / "v11_chunk_ablation_contexts.jsonl"))
    ap.add_argument("--out",
                    default=str(_REPO / "outputs" / "raw" / "v11_chunk_ablation_runs.jsonl"))
    ap.add_argument("--cap_steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv11_chunk")
    args = ap.parse_args()

    abl = _read_jsonl_plain(args.ablation_contexts)
    print(f"[11b] {len(abl)} ablation contexts (controls + chunk-minus)")

    out_path = Path(args.out)
    done_ids = set()
    if out_path.exists():
        for r in _read_jsonl_plain(out_path):
            done_ids.add(r.get("run_id"))

    work: List[Tuple] = []
    for r in abl:
        if not r.get("post_stress_text", "").strip():
            continue
        wid = f"{r['ablation_id']}__cap{args.cap_steps}"
        if wid in done_ids:
            continue
        work.append((r["ablation_id"], r["case_id"], r["variant"],
                     r["ablation_type"], r.get("chunk_id"),
                     r["post_stress_text"], args.cap_steps, args.tag))

    print(f"[11b] {len(done_ids)} already done; {len(work)} pending")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); n_done = 0; n_err = 0; n_pass = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w for w in work}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"error": str(e)}
                if rec.get("error"): n_err += 1
                if rec.get("success"): n_pass += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done <= 5:
                    eta = (time.time()-t0)/n_done * (len(work)-n_done)
                    print(f"  [{n_done}/{len(work)}] pass={n_pass} err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[11b] done: {n_done} new runs ({n_pass} success, {n_err} errors); "
          f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
