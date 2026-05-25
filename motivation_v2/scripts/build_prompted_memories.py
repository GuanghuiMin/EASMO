"""Build prompted-LLM-compressor memories at scale (T2 baseline).

For each successful trajectory in the pilot, calls MiniMax with the
four role-specific prompts (tool / code / plan / verify) at multiple
budgets, parses the response into MemoryUnits, and persists.

Concurrency: uses a thread pool because OpenAI client calls release
the GIL on the network wait. Default 8 workers; tune via --workers.

Output: one JSONL row per (task, role, B):
    {"task_id", "policy_role", "compressor": "prompted_<role>",
     "budget_tokens", "memory_text", "n_units", "n_tokens",
     "elapsed_s", "raw_response_chars"}

Scale: 83 successful direct trajectories × 4 roles × 4 budgets =
1328 calls. ~10 s/call ⇒ ~28 min wall-clock with 8 workers.
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

from motivation_v2.data import successful_trajectories
from motivation_v2.prompted_memory import (
    PROMPTED_ROLES,
    build_prompted_memory,
)


def _one_cell(args_tuple):
    traj, role, budget = args_tuple
    em = build_prompted_memory(traj, role, budget)
    rec = {
        "task_id": traj.task_id,
        "policy_role": role,
        "compressor": f"prompted_{role}",
        "budget_tokens": budget,
        "memory_text": em.text,
        "n_units": em.n_units,
        "n_tokens": em.n_tokens,
        "model": em.executor,
        "elapsed_s": getattr(em, "extra_elapsed_s", None),
        "raw_response_chars": len(getattr(em, "extra_raw_response", "") or ""),
    }
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="direct",
                        help="which strategy's trajectories to use as upstream context")
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument("--roles", nargs="+", default=PROMPTED_ROLES)
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Cap on tasks (for debugging); default = all successful")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompted_memories.jsonl"

    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{args.tag}_{args.strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    if args.max_tasks:
        trajs = trajs[: args.max_tasks]

    cells = []
    for t in trajs:
        for role in args.roles:
            for B in args.budgets:
                cells.append((t, role, B))

    print(f"Loaded {len(trajs)} trajectories")
    print(f"Designed {len(cells)} cells "
          f"({len(trajs)} tasks × {len(args.roles)} roles × {len(args.budgets)} budgets)")
    print(f"Workers: {args.workers}")
    print(f"Output: {out_path}")
    print()

    t0 = time.time()
    n_done = 0
    with open(out_path, "w") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_one_cell, c): c for c in cells}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as exc:
                    cell = futures[fut]
                    rec = {
                        "task_id": cell[0].task_id,
                        "policy_role": cell[1],
                        "budget_tokens": cell[2],
                        "error": str(exc),
                    }
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == len(cells):
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1)
                    eta = (len(cells) - n_done) / max(rate, 0.01)
                    print(f"  [{n_done:>4d}/{len(cells)}] "
                          f"elapsed={elapsed/60:.1f}min  "
                          f"rate={rate*60:.1f}/min  "
                          f"ETA={eta/60:.1f}min")

    print(f"\nDone in {(time.time() - t0)/60:.1f} min. Wrote {out_path}")


if __name__ == "__main__":
    main()
