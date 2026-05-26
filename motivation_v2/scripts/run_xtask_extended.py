"""Extended cross-task transfer experiment (Sprint 2 / Exp C).

Adds the spec-required ``generic_recent`` and ``no_memory`` baseline
conditions, and (optionally) scales the consumer set from 6 -> 18.

Conditions implemented (keys are spec §7.2 names; raw column in
transfer_results.jsonl uses the legacy short names so existing
canonicalize_behavior_cost.py picks them up):

    matched               (raw: self)              — m_exec_minimal of self
    wrong_task_same_gen   (raw: within_gen)        — m_exec_minimal of sibling
    wrong_task_diff_gen   (raw: within_app)        — m_exec_minimal of cross-gen same-app
    cross_domain          (raw: cross_app)         — m_exec_minimal of cross-app
    generic_recent        (raw: generic_recent)    — m_recent of consumer's OWN trajectory
    no_memory             (raw: no_memory)         — empty memory_text

Output: ``outputs/<tag>/transfer_results.jsonl`` with one record per
cell; canonicalize_behavior_cost.py joins this with API metrics
extracted post-hoc from env_history.json.

Usage examples
--------------
# Phase 2a — no_memory + generic_recent for existing 6 spotify consumers, cap=15:
    python scripts/run_xtask_extended.py \\
        --consumer_set existing6 \\
        --conditions generic_recent no_memory \\
        --max_iter 15 --tag mv2_xtask_ext_cap15 --workers 4

# Phase 2b — full 18-consumer grid, cap=50 baseline:
    python scripts/run_xtask_extended.py \\
        --consumer_set extended18 \\
        --conditions matched wrong_task_same_gen wrong_task_diff_gen \\
                     cross_domain generic_recent no_memory \\
        --max_iter 50 --tag mv2_xtask_ext_cap50 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ----------------------------------------------------------------------
# Consumer-set definitions
# ----------------------------------------------------------------------

_EXISTING_6 = [
    "82e2fac_3", "ccb4494_1", "e7a10f8_1",
    "692c77d_1", "ce359b5_1", "07b42fd_1",
]

_EXTRA_12 = [
    # 6 new spotify consumers from generators not in existing 6
    "229360a_1", "287e338_1", "aa8502b_1",
    "c901732_1", "e3d6c94_1", "e85d92a_1",
    # 3 file_system
    "34d9492_2", "76f2c72_2", "7d7fbf6_2",
    # 2 phone
    "302c169_2", "771d8fc_2",
    # 1 simple_note (only 1 generator available)
    "cf6abd2_2",
]


# Source-task mappings keyed by consumer task_id.
_WITHIN_GEN: Dict[str, str] = {
    "82e2fac_3": "82e2fac_2",
    "ccb4494_1": "ccb4494_2",
    "e7a10f8_1": "e7a10f8_2",
    "692c77d_1": "692c77d_2",
    "ce359b5_1": "ce359b5_2",
    "07b42fd_1": "07b42fd_2",
    "229360a_1": "229360a_2",
    "287e338_1": "287e338_2",
    "aa8502b_1": "aa8502b_2",
    "c901732_1": "c901732_2",
    "e3d6c94_1": "e3d6c94_2",
    "e85d92a_1": "e85d92a_2",
    "34d9492_2": "34d9492_1",
    "76f2c72_2": "76f2c72_1",
    "7d7fbf6_2": "7d7fbf6_1",
    "302c169_2": "302c169_1",
    "771d8fc_2": "771d8fc_1",
    "cf6abd2_2": "cf6abd2_1",
}

# Within-app cross-generator (consumer's app -> some other generator's task)
_WITHIN_APP: Dict[str, str] = {
    "82e2fac_3": "ccb4494_1",
    "ccb4494_1": "e7a10f8_1",
    "e7a10f8_1": "692c77d_1",
    "692c77d_1": "ce359b5_1",
    "ce359b5_1": "07b42fd_1",
    "07b42fd_1": "82e2fac_1",
    "229360a_1": "287e338_1",
    "287e338_1": "aa8502b_1",
    "aa8502b_1": "c901732_1",
    "c901732_1": "e3d6c94_1",
    "e3d6c94_1": "e85d92a_1",
    "e85d92a_1": "229360a_1",
    "34d9492_2": "76f2c72_1",
    "76f2c72_2": "7d7fbf6_1",
    "7d7fbf6_2": "34d9492_1",
    "302c169_2": "771d8fc_1",
    "771d8fc_2": "302c169_1",
    # simple_note has only 1 generator -> no within-app cross-gen source.
    # cf6abd2_2 will skip this condition.
}

# Cross-app: pick a successful task from a different app.
_CROSS_APP: Dict[str, str] = {
    "82e2fac_3": "34d9492_1",
    "ccb4494_1": "302c169_1",
    "e7a10f8_1": "cf6abd2_1",
    "692c77d_1": "76f2c72_1",
    "ce359b5_1": "771d8fc_1",
    "07b42fd_1": "7d7fbf6_1",
    "229360a_1": "34d9492_3",
    "287e338_1": "76f2c72_3",
    "aa8502b_1": "7d7fbf6_3",
    "c901732_1": "302c169_3",
    "e3d6c94_1": "771d8fc_3",
    "e85d92a_1": "cf6abd2_3",
    # File_system / phone / simple_note → spotify
    "34d9492_2": "82e2fac_2",
    "76f2c72_2": "ccb4494_2",
    "7d7fbf6_2": "e7a10f8_2",
    "302c169_2": "692c77d_2",
    "771d8fc_2": "ce359b5_2",
    "cf6abd2_2": "07b42fd_2",
}


# ----------------------------------------------------------------------
# Memory-text builders (run inside the worker)
# ----------------------------------------------------------------------


def _build_memory_text_in_worker(
    *,
    consumer_task: str,
    condition: str,
    source_task: str,
    budget: int,
    minimal_lookup: Dict[Tuple[str, int], str],
) -> str:
    """Returns memory_text for a given cell. Empty string for no_memory.

    For ``generic_recent`` we build m_recent on the consumer's own gold
    trajectory pool. For everything else we look up
    ``m_exec_minimal(source_task, B)`` in the precomputed table.
    """
    if condition == "no_memory":
        return ""
    if condition == "generic_recent":
        sys.path.insert(0, str(_REPO))
        from motivation_v2.compressors import m_recent
        from motivation_v2.data import load_trajectory
        from motivation_v2.units import trajectory_unit_pool

        traj_dir = Path(
            f"/workspace/acon/experiments/appworld/outputs/"
            f"MiniMaxAI_MiniMax-M2.5_mv2_pilot_direct/train/task_{consumer_task}"
        )
        if not traj_dir.exists():
            return ""
        try:
            traj = load_trajectory(traj_dir)
        except Exception:
            return ""
        pool = trajectory_unit_pool(traj)
        em = m_recent(pool, budget, task_id=consumer_task)
        return em.text
    # matched / wrong_task_* / cross_domain — look up precomputed minimal text
    return minimal_lookup.get((source_task, budget), "")


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------


def _run_one_cell(args_tuple):
    (consumer_task, condition_canonical, condition_raw, source_task, budget,
     memory_text, tag, max_iter) = args_tuple

    sys.path.insert(0, str(_REPO))
    from motivation_v2.runner import run_with_compressed_memory

    res = run_with_compressed_memory(
        task_id=consumer_task,
        strategy="direct",
        memory_text=memory_text,
        compressor=f"xtask_{condition_raw}_from_{source_task or 'none'}",
        budget=budget,
        tag=tag,
        max_iter=max_iter,
    )
    return {
        "consumer_task_id": consumer_task,
        "source_task_id": source_task or "",
        "condition": condition_raw,
        "condition_canonical": condition_canonical,
        "budget": budget,
        "max_iter": max_iter,
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
# Main
# ----------------------------------------------------------------------


def _build_minimal_lookup(
    memories_jsonl: Path,
) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    with open(memories_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("compressor") != "m_exec_minimal":
                continue
            key = (r["task_id"], r["budget_tokens"])
            if key not in out:
                out[key] = r["memory_text"]
    return out


def _resolve_consumer_set(name: str) -> List[str]:
    if name == "existing6":
        return list(_EXISTING_6)
    if name == "extra12":
        return list(_EXTRA_12)
    if name == "extended18":
        return list(_EXISTING_6) + list(_EXTRA_12)
    raise SystemExit(f"unknown --consumer_set {name!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumer_set",
                        choices=["existing6", "extra12", "extended18"],
                        required=True)
    parser.add_argument("--conditions", nargs="+",
                        default=["matched", "wrong_task_same_gen",
                                 "wrong_task_diff_gen", "cross_domain",
                                 "generic_recent", "no_memory"])
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512])
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tag", required=True,
                        help="Output tag, e.g. mv2_xtask_ext_cap15")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memories_jsonl",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot/compressed_memories.jsonl")
    parser.add_argument("--output_dir",
                        default=None,
                        help="Defaults to outputs/<tag>")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/workspace/EASMO/motivation_v2/outputs/{args.tag}"

    consumers = _resolve_consumer_set(args.consumer_set)
    print(f"[ext] consumer_set={args.consumer_set!r} -> {len(consumers)} consumers")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer_results.jsonl"

    minimal = _build_minimal_lookup(Path(args.memories_jsonl))
    print(f"[ext] loaded {len(minimal)} m_exec_minimal cells from {args.memories_jsonl}")

    # ----- Build cell list -----
    raw_map = {
        "matched":             "self",
        "wrong_task_same_gen": "within_gen",
        "wrong_task_diff_gen": "within_app",
        "cross_domain":        "cross_app",
        "generic_recent":      "generic_recent",
        "no_memory":           "no_memory",
    }
    cells: List[Tuple] = []
    skipped_reasons: List[str] = []
    for consumer in consumers:
        for cond_canon in args.conditions:
            cond_raw = raw_map[cond_canon]
            # Resolve source
            if cond_canon == "matched":
                source = consumer
            elif cond_canon == "wrong_task_same_gen":
                source = _WITHIN_GEN.get(consumer)
            elif cond_canon == "wrong_task_diff_gen":
                source = _WITHIN_APP.get(consumer)
            elif cond_canon == "cross_domain":
                source = _CROSS_APP.get(consumer)
            else:
                # generic_recent / no_memory — no source task per se
                source = None

            if cond_canon in {"matched", "wrong_task_same_gen",
                              "wrong_task_diff_gen", "cross_domain"}:
                if source is None:
                    skipped_reasons.append(
                        f"{consumer}/{cond_canon}: no source mapping")
                    continue
            if cond_canon == "no_memory":
                # one cell per consumer, no budget axis (memory is empty)
                cells.append((consumer, cond_canon, cond_raw, "", 0, "",
                              args.tag, args.max_iter))
                continue

            for B in args.budgets:
                # Build memory text up-front for matched/wrong/cross,
                # defer for generic_recent (built inside worker since it
                # needs the trajectory).
                if cond_canon == "generic_recent":
                    mem_text = "__GENERIC_RECENT__"  # sentinel; resolved in worker
                else:
                    mem_text = minimal.get((source, B), None)
                    if mem_text is None:
                        skipped_reasons.append(
                            f"{consumer}/{cond_canon}/B={B}: missing m_exec_minimal({source}, {B})")
                        continue
                cells.append((consumer, cond_canon, cond_raw, source or "",
                              B, mem_text, args.tag, args.max_iter))

    print(f"[ext] designed {len(cells)} cells "
          f"({len(consumers)} consumers, {len(args.conditions)} conditions, "
          f"{len(args.budgets)} budgets, max_iter={args.max_iter})")
    if skipped_reasons:
        print(f"[ext] skipped {len(skipped_reasons)} cells:")
        for r in skipped_reasons[:20]:
            print(f"        {r}")
        if len(skipped_reasons) > 20:
            print(f"        ... +{len(skipped_reasons) - 20} more")

    # ---- Resolve generic_recent memory text in main process so workers
    # don't need to re-load trajectories. NOTE: budget-dependent.
    minimal_for_workers = dict(minimal)  # passed by value to subprocesses
    final_cells: List[Tuple] = []
    for c in cells:
        consumer_task, cond_canon, cond_raw, source, budget, mem_text, tag, mi = c
        if cond_canon == "generic_recent":
            # Resolve in main process to keep workers light.
            mem_text = _build_memory_text_in_worker(
                consumer_task=consumer_task,
                condition=cond_canon,
                source_task=source,
                budget=budget,
                minimal_lookup=minimal_for_workers,
            )
        final_cells.append((consumer_task, cond_canon, cond_raw, source,
                            budget, mem_text, tag, mi))

    print(f"[ext] running {len(final_cells)} cells with {args.workers} workers")
    print(f"[ext] output -> {out_path}")
    print()

    # Stream to disk
    with open(out_path, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one_cell, c): c for c in final_cells}
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                done += 1
                tag_str = (f"{rec['consumer_task_id']:>11s} "
                           f"({rec['condition']:>14s}@B{rec['budget']:<4d})")
                print(f"  [{done:>3d}/{len(final_cells)}] {tag_str}  "
                      f"success={rec['success']!s:<5}  iter={rec['iterations']:>2d}  "
                      f"elapsed={rec['elapsed_s']:.0f}s")

    print(f"\n[ext] Wrote {len(final_cells)} results to {out_path}")


if __name__ == "__main__":
    main()
