"""Stage 07 — downstream agent runs for the 6 NEW v4 conditions.

For each (task, method, budget) cell in:
  high_sensitivity_spans, low_sensitivity_spans, recent_spans,
  random_spans_seed{1,2,3}

run the same MiniMax-M2.5 downstream agent with the v3-compatible
prompt and the per-task compressed context produced by stage 06.

The 4 reused conditions (task_aware_summary, acon_style_summary,
truncated_full_context, no_context) are NOT run here — Stage 08
merges in v3's existing behavior_runs.jsonl.

Outputs:
  outputs/raw/behavior_runs.jsonl
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
    (task_id, method, compressed_text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
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
        "budget": f"{'loose_15' if max_steps==15 else 'strict_8'}",
        "budget_max_steps": max_steps,
        "success": res.success,
        "score": res.final_reward,
        "num_steps": res.iterations,
        "iterations": res.iterations,  # alias
        "total_input_tokens": res.input_tokens,
        "peak_input_tokens": res.input_tokens,  # we don't track step-by-step peak
        "input_tokens": res.input_tokens,  # alias
        "output_tokens": res.output_tokens,
        "elapsed_s": res.elapsed_s,
        "memory_text_len": len(compressed_text),
        "output_dir": res.output_dir,
        "termination_reason": res.termination_reason,
        "error": res.error,
        "failure_reason": res.error if res.error else (res.termination_reason if not res.success else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budgets", nargs="+", type=int, default=[15, 8])
    parser.add_argument("--methods", nargs="+", default=[
        "high_sensitivity_spans", "low_sensitivity_spans", "recent_spans",
        "random_spans_seed1", "random_spans_seed2", "random_spans_seed3",
    ])
    parser.add_argument("--tag", default="mv4_run")
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, read_jsonl, write_jsonl,
    )

    ensure_outputs()
    contexts = read_jsonl(raw_path("compressed_contexts.jsonl"))

    by_tm: Dict[Tuple[str, str], str] = {}
    for r in contexts:
        by_tm[(r["task_id"], r["method"])] = r.get("compressed_text", "")

    cells: List[Tuple] = []
    for (tid, method), ctx in by_tm.items():
        if method not in args.methods:
            continue
        for cap in args.budgets:
            cells.append((tid, method, ctx, cap, args.tag))

    print(f"[07] {len(cells)} cells "
          f"(tasks={len({c[0] for c in cells})}, "
          f"methods={len(args.methods)}, budgets={len(args.budgets)}, "
          f"workers={args.workers})")
    print()

    out_path = raw_path("behavior_runs.jsonl")
    t0 = time.time()
    n_done = 0
    n_err = 0
    with open(out_path, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one, c): c for c in cells}
            for fut in as_completed(futures):
                rec = fut.result()
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                n_done += 1
                tag_str = (f"{rec['task_id']:>11s} "
                           f"({rec['method']:>26s}@cap{rec['budget_max_steps']:<2d})")
                print(f"  [{n_done:>3d}/{len(cells)}] {tag_str}  "
                      f"success={rec['success']!s:<5}  "
                      f"iter={rec['num_steps']:>2d}  "
                      f"elapsed={rec['elapsed_s']:.0f}s",
                      flush=True)

    print(f"\n[07] wrote {len(cells)} runs -> {out_path}")
    print(f"[07] elapsed: {(time.time()-t0)/60:.1f} min  err={n_err}")


if __name__ == "__main__":
    main()
