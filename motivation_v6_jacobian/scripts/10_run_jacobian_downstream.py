"""Stage 10 — Experiment D part 2: run MiniMax downstream agent on the
gradient-ranked compressed contexts produced by stage 09.

MUST be invoked with /workspace/acon/.venv/bin/python because the
runner imports `appworld` and `productive_agents`. Same wrapper as
v4's stage 07.

Outputs:
  outputs/raw/jacobian_behavior_runs.jsonl
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
    (task_id, method, compressed_text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context
    res = run_with_compressed_context(
        task_id=task_id,
        method=method,
        compressed_context=compressed_text,
        max_steps=max_steps,
        tag=tag,
    )
    return {
        "task_id": task_id,
        "method": method,
        "budget_max_steps": max_steps,
        "success": res.success,
        "score": res.final_reward,
        "num_steps": res.iterations,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "elapsed_s": res.elapsed_s,
        "memory_text_len": len(compressed_text),
        "output_dir": res.output_dir,
        "termination_reason": res.termination_reason,
        "error": res.error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts",
                    default=str(_REPO / "outputs/raw/jacobian_compressed_contexts.jsonl"))
    ap.add_argument("--out",
                    default=str(_REPO / "outputs/raw/jacobian_behavior_runs.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budgets", nargs="+", type=int, default=[15])
    ap.add_argument("--tag", default="mv6_run")
    ap.add_argument("--methods", nargs="+", default=[
        "jacobian_high_spans",
        "jacobian_high_spans_raw",
        "jacobian_low_spans",
        "jacobian_recent_hybrid",
    ])
    args = ap.parse_args()

    contexts = []
    with open(args.contexts) as f:
        for line in f:
            if line.strip():
                contexts.append(json.loads(line))

    cells: List[Tuple] = []
    for r in contexts:
        if r.get("method") not in args.methods:
            continue
        for cap in args.budgets:
            cells.append((r["task_id"], r["method"], r["compressed_text"], cap, args.tag))

    print(f"[10] {len(cells)} agent cells "
          f"(tasks={len({c[0] for c in cells})}, "
          f"methods={len(args.methods)}, budgets={len(args.budgets)}, "
          f"workers={args.workers})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0
    n_err = 0
    n_ok = 0
    with open(args.out, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one, c): c for c in cells}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"error": str(e)}
                if rec.get("error"):
                    n_err += 1
                if rec.get("success"):
                    n_ok += 1
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                n_done += 1
                print(f"  [{n_done:>3d}/{len(cells)}] "
                      f"{rec.get('task_id','?'):>11s} "
                      f"({rec.get('method','?'):>26s}@cap{rec.get('budget_max_steps','?')})  "
                      f"success={rec.get('success')}  "
                      f"iter={rec.get('num_steps','?')}  "
                      f"elapsed={rec.get('elapsed_s', 0):.0f}s",
                      flush=True)
    print(f"\n[10] {n_done} runs ({n_ok} success, {n_err} err) -> {args.out}")
    print(f"[10] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
