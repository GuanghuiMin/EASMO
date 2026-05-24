"""Build the compressed-memory matrix from successful pilot trajectories.

Input:  acon trajectory outputs (one per task × strategy)
Output: motivation_v2/outputs/<run-name>/compressed_memories.jsonl

Each line in the output is one compressed memory variant:
    {
      "task_id": ...,
      "policy_strategy": "direct" | "verify" | "explore",
      "topic": "spotify" | "phone" | ... | "multi_app",
      "compressor": "m_recent" | "m_freq" | "m_bm25" | "m_exec_minimal" | "m_exec_trajectory",
      "exec_memory_executor": "MiniMaxAI/MiniMax-M2.5" | null,
      "budget_tokens": 128 | 256 | 512 | 1024 | 2048,
      "memory_text": <str>,
      "n_units": int,
      "n_tokens": int,
      "n_units_dropped": int,
    }

This is the deterministic post-hoc step: no LLM calls, no
randomness. Compressors operate on the trajectory's own observation
pool. The runner (next script) will read this file and re-run the
agent with each memory_text injected as starting context.

Usage:
    /workspace/EASMO/.venv/bin/python motivation_v2/scripts/build_compressed_memories.py
        --tag mv2_pilot
        --strategies direct verify explore
        --budgets 128 256 512 1024 2048
        --output_dir motivation_v2/outputs/mv2_pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.compressors import (
    m_bm25, m_freq, m_recent, m_embedding_topk,
)
from motivation_v2.data import (
    iter_trajectories,
    load_ground_truth,
)
from motivation_v2.exec_memory import (
    m_exec_minimal,
    m_exec_trajectory,
)
from motivation_v2.policy_family import assign_policy_family
from motivation_v2.units import trajectory_unit_pool


def _compressors_for(strategy: str, traj, gt, B: int, query: str) -> list:
    """Return list of (compressor_name, ExecMemory) tuples for one (task, strat, B)."""
    pool = trajectory_unit_pool(traj)
    out = []

    # Generic / non-policy baselines (no use of strategy or task instruction).
    out.append(("m_recent", m_recent(pool, B, task_id=traj.task_id)))
    out.append(("m_freq",   m_freq(pool, B, task_id=traj.task_id)))

    # Task-instruction-conditioned baselines.
    out.append(("m_bm25",   m_bm25(pool, query, B, task_id=traj.task_id)))

    # Embedding-top-k is gated on sentence-transformers availability;
    # the helper falls back to BM25 if missing, so it's safe to call.
    out.append(("m_embedding_topk", m_embedding_topk(pool, query, B, task_id=traj.task_id)))

    # Execution-derived oracles (the headline T1 evidence).
    out.append(("m_exec_minimal", m_exec_minimal(gt, B)))
    out.append((
        "m_exec_trajectory",
        m_exec_trajectory(gt, traj, B),
    ))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True,
                        help="acon run tag (e.g. mv2_pilot); reads outputs from "
                             "/workspace/acon/.../outputs/MiniMaxAI_MiniMax-M2.5_<tag>_<strategy>/<split>/")
    parser.add_argument("--strategies", nargs="+",
                        default=["direct", "verify", "explore"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--require_success", action="store_true", default=True,
                        help="Only build memories from successful trajectories")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compressed_memories.jsonl"
    summary_path = out_dir / "compressed_memories_summary.json"

    # Iterate per strategy and per task.
    n_lines = 0
    n_tasks_seen = 0
    n_tasks_used = 0
    per_strategy_count = {s: {"seen": 0, "used": 0, "success": 0} for s in args.strategies}

    with open(out_path, "w", encoding="utf-8") as f_out:
        for strategy in args.strategies:
            glob = (
                f"/workspace/acon/experiments/appworld/outputs/"
                f"MiniMaxAI_MiniMax-M2.5_{args.tag}_{strategy}/"
                f"{args.split}/task_*"
            )
            for traj in iter_trajectories(experiments_glob=glob):
                n_tasks_seen += 1
                per_strategy_count[strategy]["seen"] += 1
                if traj.success:
                    per_strategy_count[strategy]["success"] += 1

                if args.require_success and not traj.success:
                    continue

                try:
                    gt = load_ground_truth(traj.task_id)
                except FileNotFoundError:
                    continue
                pa = assign_policy_family(gt)

                n_tasks_used += 1
                per_strategy_count[strategy]["used"] += 1

                for B in args.budgets:
                    for cname, em in _compressors_for(strategy, traj, gt, B, gt.instruction):
                        rec = {
                            "task_id": traj.task_id,
                            "policy_strategy": strategy,
                            "topic": pa.family,
                            "is_single_app": pa.is_single_app,
                            "primary_app": pa.primary_app,
                            "compressor": cname,
                            "exec_memory_executor": em.executor,
                            "budget_tokens": B,
                            "memory_text": em.text,
                            "n_units": em.n_units,
                            "n_tokens": em.n_tokens,
                            "n_units_dropped": em.n_units_dropped,
                            "trajectory_iterations": traj.iterations,
                            "trajectory_success": traj.success,
                        }
                        f_out.write(json.dumps(rec) + "\n")
                        n_lines += 1

    summary = {
        "tag": args.tag,
        "strategies": list(args.strategies),
        "split": args.split,
        "budgets": list(args.budgets),
        "total_jsonl_lines": n_lines,
        "tasks_seen": n_tasks_seen,
        "tasks_used": n_tasks_used,
        "per_strategy": per_strategy_count,
        "output_jsonl": str(out_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {n_lines} compressed-memory rows to {out_path}")
    print(f"Per-strategy task counts:")
    for s, c in per_strategy_count.items():
        success_rate = c["success"] / c["seen"] if c["seen"] else 0.0
        print(f"  {s:<10s}  seen={c['seen']:>3d}  success={c['success']:>3d} ({100*success_rate:.0f}%)  used={c['used']}")


if __name__ == "__main__":
    main()
