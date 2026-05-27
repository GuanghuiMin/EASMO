"""Stage 1 — generate full-context successful trajectories on dev split.

Runs MiniMax-M2.5 against all 56 AppWorld dev tasks with the standard
``direct`` strategy, then selects the top n_target tasks by trajectory
length (preferring multi-step / tool-rich tasks per spec).

Outputs:
  outputs/motivation_full_trajectories.jsonl   one row per selected task
  outputs/sprint_logs/01_select_consumers.log

The full agent trajectories themselves live under
  /workspace/acon/experiments/appworld/outputs/MiniMaxAI_MiniMax-M2.5_mv3_fullctx/dev/task_*

This stage is the one that requires the acon venv (productive_agents).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ----------------------------------------------------------------------
# Worker — run one dev task with the direct strategy.
# ----------------------------------------------------------------------


def _run_one_dev_task(task_id: str, max_iter: int, tag: str):
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v2.runner import run_with_compressed_memory

    res = run_with_compressed_memory(
        task_id=task_id,
        strategy="direct",
        memory_text="",  # no compression — full agent context
        compressor="fullctx_dev",
        budget=0,
        tag=tag,
        max_iter=max_iter,
        split="dev",
    )
    return {
        "task_id": task_id,
        "success": res.success,
        "iterations": res.iterations,
        "final_reward": res.final_reward,
        "termination_reason": res.termination_reason,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "elapsed_s": res.elapsed_s,
        "output_dir": res.output_dir,
        "error": res.error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_target", type=int, default=30,
                        help="Target number of successful tasks to keep.")
    parser.add_argument("--max_iter", type=int, default=30,
                        help="Per-task step cap for the full-context run.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="mv3_fullctx")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on tasks attempted (for smoke-test).")
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v2.data import load_split_task_ids
    from motivation_v3.data import (
        OUTPUTS, ensure_outputs, jsonl_path,
    )

    ensure_outputs()
    out_path = jsonl_path("motivation_full_trajectories.jsonl")
    log_path = OUTPUTS / "sprint_logs" / "01_select_consumers.log"

    task_ids = load_split_task_ids(args.split)
    if args.limit:
        task_ids = task_ids[: args.limit]
    print(f"[01] running {len(task_ids)} {args.split} tasks "
          f"with strategy=direct, max_iter={args.max_iter}, workers={args.workers}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_run_one_dev_task, tid, args.max_iter, args.tag): tid
            for tid in task_ids
        }
        n_done = 0
        with open(log_path, "w") as logf:
            for fut in as_completed(futures):
                rec = fut.result()
                results.append(rec)
                n_done += 1
                line = (f"  [{n_done:>2d}/{len(task_ids)}] {rec['task_id']:>11s}  "
                        f"success={rec['success']!s:<5}  iter={rec['iterations']:>2d}  "
                        f"reward={rec['final_reward']:.2f}  elapsed={rec['elapsed_s']:.0f}s")
                print(line, flush=True)
                logf.write(line + "\n")
                logf.flush()

    successful = [r for r in results if r["success"]]
    successful.sort(key=lambda r: -r["iterations"])  # prefer longer (richer) trajectories
    selected = successful[: args.n_target]

    print(f"\n[01] {len(successful)}/{len(results)} successful; "
          f"selected top-{len(selected)} by iterations")

    rows = []
    for r in selected:
        rows.append({
            "task_id": r["task_id"],
            "split": args.split,
            "tag": args.tag,
            "iterations": r["iterations"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "elapsed_s": r["elapsed_s"],
            "output_dir": r["output_dir"],
            "selected": True,
        })

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[01] wrote {len(rows)} rows -> {out_path}")
    print(f"[01] elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
