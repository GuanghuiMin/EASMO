"""Build prompted-LLM-compressor ablation variants (Sprint 3 / Exp D §8.3).

For each successful direct trajectory and each prompted condition in
{prompted_generic, prompted_task, prompted_role, prompted_task_role},
calls MiniMax with the appropriate prompt and persists the result.

Note: ``prompted_task_role`` already exists from the original
``build_prompted_memories.py`` run; this script can either skip it
(default) or rebuild it (--include_existing).

Output: one JSONL per condition in
``outputs/mv2_pilot_variants/prompted_<condition>.jsonl``.

Cost estimate at 8 workers, 83 tasks, 4 roles, 4 budgets, 3 NEW conditions:
  ~ 3,984 calls × ~10 s/call ÷ 8 = ~83 min wall-clock
But prompted_generic and prompted_task don't depend on role, so we
can dedupe: each (task, budget) triggers ONE call for those two
conditions, and we copy the same memory for all 4 roles in the
output. That cuts the call count to:
  prompted_generic:  83 × 4 budgets = 332 calls
  prompted_task:     83 × 4 budgets = 332 calls
  prompted_role:     83 × 4 budgets × 4 roles = 1,328 calls
  Total NEW: 1,992 calls, ~42 min at 8 workers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.data import successful_trajectories
from motivation_v2.prompted_memory_variants import build_prompted_variant


_ROLES = ["tool", "code", "plan", "verify"]


def _one_cell(args_tuple):
    traj, role, budget, condition = args_tuple
    em = build_prompted_variant(traj, role, budget, condition=condition)
    return {
        "task_id": traj.task_id,
        "policy_role": role,
        "compressor": condition,
        "budget_tokens": budget,
        "memory_text": em.text,
        "n_units": em.n_units,
        "n_tokens": em.n_tokens,
        "model": em.executor,
        "elapsed_s": getattr(em, "extra_elapsed_s", None),
        "raw_response_chars": len(getattr(em, "extra_raw_response", "") or ""),
    }


def _design_cells(
    trajs, conditions: List[str], budgets: List[int],
) -> List[Tuple]:
    """Build the cell list. For role-independent conditions
    (prompted_generic, prompted_task) we still iterate roles in the
    OUTPUT (so cross-role comparisons line up) but only call the LLM
    ONCE per (traj, budget) and reuse the memory for all 4 roles."""
    cells: List[Tuple] = []
    for cond in conditions:
        if cond in {"prompted_generic", "prompted_task"}:
            for t in trajs:
                for B in budgets:
                    # role param is "tool" for the actual LLM call —
                    # _build_prompt ignores role for these conditions.
                    cells.append((t, "tool", B, cond))
        else:
            for t in trajs:
                for role in _ROLES:
                    for B in budgets:
                        cells.append((t, role, B, cond))
    return cells


def _expand_role_independent_records(
    records: List[dict], conditions: List[str],
) -> List[dict]:
    """For prompted_generic / prompted_task, copy each (task, budget)
    record into 4 role-tagged records so the analyzer can match
    cross-role pairs the same way it does for prompted_role /
    prompted_task_role."""
    out: List[dict] = []
    for r in records:
        cond = r["compressor"]
        if cond in {"prompted_generic", "prompted_task"}:
            for role in _ROLES:
                clone = dict(r)
                clone["policy_role"] = role
                clone["original_policy_role"] = "tool"  # the LLM-call key
                out.append(clone)
        else:
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="direct")
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument("--conditions", nargs="+",
                        default=["prompted_generic", "prompted_task", "prompted_role"])
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot_variants")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{args.tag}_{args.strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    if args.max_tasks:
        trajs = trajs[: args.max_tasks]
    print(f"[D] {len(trajs)} successful direct trajectories")

    cells = _design_cells(trajs, args.conditions, args.budgets)
    print(f"[D] designed {len(cells)} LLM calls "
          f"(conditions={args.conditions})")
    print(f"[D] workers: {args.workers}")
    print()

    by_cond_records: dict = {c: [] for c in args.conditions}

    t0 = time.time()
    n_done = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_one_cell, c): c for c in cells}
        for fut in as_completed(futures):
            cell = futures[fut]
            try:
                rec = fut.result()
                by_cond_records[rec["compressor"]].append(rec)
            except Exception as exc:
                n_err += 1
                t, role, B, cond = cell
                rec = {
                    "task_id": t.task_id, "policy_role": role,
                    "compressor": cond, "budget_tokens": B,
                    "error": str(exc),
                }
                by_cond_records[cond].append(rec)
            n_done += 1
            if n_done % 25 == 0 or n_done == len(cells):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(cells) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>4d}/{len(cells)}] "
                      f"elapsed={elapsed/60:.1f}min  "
                      f"rate={rate*60:.1f}/min  "
                      f"ETA={eta/60:.1f}min  "
                      f"err={n_err}")

    # Expand role-independent conditions before saving so each output
    # JSONL has the same shape (task, role, B → memory_text).
    for cond in args.conditions:
        records = by_cond_records[cond]
        records = _expand_role_independent_records(records, [cond])
        path = out_dir / f"{cond}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[D] wrote {len(records)} rows -> {path}")

    print(f"\n[D] Done in {(time.time() - t0)/60:.1f} min "
          f"(n_done={n_done}, n_err={n_err})")


if __name__ == "__main__":
    main()
