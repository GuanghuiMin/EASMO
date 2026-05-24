"""Cross-task memory transfer experiment.

For each consumer task B, run B's executor with memory derived from
DIFFERENT source tasks A, plus the self-baseline. This tests the
user's hypothesis (1:55 PM PT 2026-05-24):

    "At tight B, single-task RLVR could yield generic memory because
     all tasks converge on shared plumbing. At loose B, single-task
     compression overfits and fails to transfer."

The transfer drop pattern is the headline finding for the paper:
flat across budgets → fully-generic memory works → no need for
task-conditional. Sharp drop above the plumbing floor → task-
conditional necessary.

Conditions tested per consumer task B:
  * self          : feed B with m*_exec_minimal(B, budget)
  * within_gen    : feed B with m*_exec_minimal(A_sibling, budget)
                    where A is from the same generator as B
  * within_app    : feed B with m*_exec_minimal(A_other_gen, budget)
                    where A is in same app, different generator
  * cross_app     : feed B with m*_exec_minimal(A_other_app, budget)

Output: outputs/<tag>/transfer_results.jsonl with one record per cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ----------------------------------------------------------------------
# Helper: pull pre-built m*_exec_minimal text from compressed_memories.jsonl.
# ----------------------------------------------------------------------


def _load_minimal_memories(path: Path):
    """Returns dict[(task_id, budget)] -> memory_text (m*_exec_minimal only)."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["compressor"] != "m_exec_minimal":
                continue
            # m_exec_minimal is strategy-/executor-independent, so the
            # text is the same regardless of which strategy's row it
            # came from. Take the first occurrence.
            key = (r["task_id"], r["budget_tokens"])
            if key not in out:
                out[key] = r["memory_text"]
    return out


# ----------------------------------------------------------------------
# Worker: run one cell.
# ----------------------------------------------------------------------


def _run_one_cell(args_tuple):
    (consumer_task_id, condition, source_task_id, budget,
     memory_text, tag) = args_tuple

    # Defer the heavy import until we're inside the worker, so the
    # parent doesn't load productive_agents in the main process.
    sys.path.insert(0, str(_REPO))
    from motivation_v2.runner import run_with_compressed_memory

    res = run_with_compressed_memory(
        task_id=consumer_task_id,
        strategy="direct",
        memory_text=memory_text,
        compressor=f"xtask_{condition}_from_{source_task_id}",
        budget=budget,
        tag=tag,
    )
    return {
        "consumer_task_id": consumer_task_id,
        "source_task_id": source_task_id,
        "condition": condition,
        "budget": budget,
        "success": res.success,
        "iterations": res.iterations,
        "final_reward": res.final_reward,
        "termination_reason": res.termination_reason,
        "input_tokens": res.input_tokens,
        "elapsed_s": res.elapsed_s,
        "memory_text_len": len(memory_text),
        "error": res.error,
    }


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories_jsonl",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot/compressed_memories.jsonl")
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--tag", default="mv2_xtask")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_dir",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_xtask")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer_results.jsonl"

    minimal = _load_minimal_memories(Path(args.memories_jsonl))
    print(f"Loaded {len(minimal)} (task, budget) memory cells from {args.memories_jsonl}")

    # ----- Hand-picked design -----
    # 6 spotify consumers from 6 different generators.
    consumers = [
        "82e2fac_3", "ccb4494_1", "e7a10f8_1",
        "692c77d_1", "ce359b5_1", "07b42fd_1",
    ]
    # within-gen sibling for each (must be a successful direct task different
    # from consumer; using the consumer's gen siblings).
    within_gen_sources = {
        "82e2fac_3": "82e2fac_2",
        "ccb4494_1": "ccb4494_2",
        "e7a10f8_1": "e7a10f8_2",
        "692c77d_1": "692c77d_2",
        "ce359b5_1": "ce359b5_2",
        "07b42fd_1": "07b42fd_2",
    }
    # within-app cross-gen: pick a different spotify generator's task.
    # Rotate so each consumer gets a different cross-gen source.
    within_app_sources = {
        "82e2fac_3": "ccb4494_1",
        "ccb4494_1": "e7a10f8_1",
        "e7a10f8_1": "692c77d_1",
        "692c77d_1": "ce359b5_1",
        "ce359b5_1": "07b42fd_1",
        "07b42fd_1": "82e2fac_1",
    }
    # cross-app: feed each consumer a memory from a non-spotify task.
    # Mix file_system / phone / simple_note for diversity.
    cross_app_sources = {
        "82e2fac_3": "34d9492_1",      # file_system
        "ccb4494_1": "302c169_1",      # phone
        "e7a10f8_1": "cf6abd2_1",      # simple_note
        "692c77d_1": "76f2c72_1",      # file_system
        "ce359b5_1": "771d8fc_1",      # phone
        "07b42fd_1": "7d7fbf6_1",      # file_system
    }

    cells = []
    for consumer in consumers:
        for B in args.budgets:
            # self
            mem = minimal.get((consumer, B))
            if mem is not None:
                cells.append((consumer, "self", consumer, B, mem, args.tag))
            # within-gen
            src = within_gen_sources[consumer]
            mem = minimal.get((src, B))
            if mem is not None:
                cells.append((consumer, "within_gen", src, B, mem, args.tag))
            # within-app cross-gen
            src = within_app_sources[consumer]
            mem = minimal.get((src, B))
            if mem is not None:
                cells.append((consumer, "within_app", src, B, mem, args.tag))
            # cross-app
            src = cross_app_sources[consumer]
            mem = minimal.get((src, B))
            if mem is not None:
                cells.append((consumer, "cross_app", src, B, mem, args.tag))

    print(f"Designed {len(cells)} cells "
          f"({len(consumers)} consumers × ~4 conditions × {len(args.budgets)} budgets)")
    print(f"Output: {out_path}")
    print(f"Workers: {args.workers}")
    print()

    # Stream results to disk as they come in so partial progress is preserved.
    with open(out_path, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one_cell, c): c for c in cells}
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                done += 1
                tag_str = (f"{rec['consumer_task_id']:>11s} "
                           f"({rec['condition']:>10s}@B{rec['budget']:<4d}"
                           f"  src={rec['source_task_id']:>11s})")
                print(f"  [{done:>3d}/{len(cells)}] {tag_str}  "
                      f"success={rec['success']!s:<5}  iter={rec['iterations']:>2d}  "
                      f"elapsed={rec['elapsed_s']:.0f}s")

    print(f"\nWrote {len(cells)} results to {out_path}")


if __name__ == "__main__":
    main()
