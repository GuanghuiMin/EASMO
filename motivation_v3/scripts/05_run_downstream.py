"""Stage 5 — Exp 3: behavioral utility runs.

For each (task, condition, budget) cell, runs MiniMax-M2.5 on the
AppWorld task with the spec's downstream-agent prompt and the
condition-specific compressed context spliced in.

Conditions (7):
  1. full_context                        — render(traj) up to 12K chars
  2. task_aware_summary                  — Stage 2 output for this task
  3. acon_style_summary                  — Stage 2 output for this task
  4. symbolic_evidence                   — Stage 2 output for this task
  5. wrong_task_symbolic_same_app        — symbolic_evidence from another
                                            task in the same primary app
  6. wrong_task_symbolic_cross_app       — symbolic_evidence from a task in
                                            a different app
  7. no_context                          — empty

Budgets (max_steps): 15 (loose), 8 (strict)
Total cells: 7 × 2 × n_tasks = 420 cells for n=30.

Output: outputs/motivation_behavior_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _run_one_cell(args_tuple):
    (task_id, method, compressed_text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.runner import run_with_compressed_context

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
        "iterations": res.iterations,
        "final_reward": res.final_reward,
        "termination_reason": res.termination_reason,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "elapsed_s": res.elapsed_s,
        "memory_text_len": len(compressed_text),
        "output_dir": res.output_dir,
        "error": res.error,
    }


def _build_full_context_text(traj, max_chars: int = 12000) -> str:
    sys.path.insert(0, str(_REPO))
    from motivation_v3.data import render_trajectory
    return render_trajectory(traj, max_total_chars=max_chars)


def _build_wrong_task_pairs(
    sel_tasks: List[str], app_of: Dict[str, str], rng: random.Random,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """For each consumer task, pick:
      same_app: another consumer task with the same primary_app
      cross_app: another consumer task with a different primary_app

    Returns (same_app_map, cross_app_map). If a partner is unavailable
    for some consumer (e.g. only one task in its app), that key is
    omitted; the runner treats missing as no_context for that cell.
    """
    by_app: Dict[str, List[str]] = defaultdict(list)
    for tid in sel_tasks:
        by_app[app_of.get(tid, "unknown")].append(tid)

    same_map: Dict[str, str] = {}
    cross_map: Dict[str, str] = {}
    for tid in sel_tasks:
        app = app_of.get(tid, "unknown")
        siblings = [x for x in by_app[app] if x != tid]
        if siblings:
            same_map[tid] = rng.choice(siblings)
        others = [x for x in sel_tasks if app_of.get(x, "unknown") != app]
        if others:
            cross_map[tid] = rng.choice(others)
    return same_map, cross_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budgets", nargs="+", type=int, default=[15, 8])
    parser.add_argument("--conditions", nargs="+",
                        default=["full_context", "task_aware_summary",
                                 "acon_style_summary", "symbolic_evidence",
                                 "wrong_task_symbolic_same_app",
                                 "wrong_task_symbolic_cross_app",
                                 "no_context"])
    parser.add_argument("--tag", default="mv3_run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.data import (
        OUTPUTS, assign_policy_family, ensure_outputs, jsonl_path,
        iter_tasks_with_ground_truth, load_trajectory, read_jsonl,
    )
    rng = random.Random(args.seed)
    ensure_outputs()

    sel = read_jsonl(jsonl_path("motivation_full_trajectories.jsonl"))
    compressed = read_jsonl(jsonl_path("motivation_compressed_contexts.jsonl"))
    sel_tids = [r["task_id"] for r in sel]

    # Map task -> primary app via ground truth.
    app_of: Dict[str, str] = {}
    for gt in iter_tasks_with_ground_truth("dev"):
        if gt.task_id in sel_tids:
            pa = assign_policy_family(gt)
            app_of[gt.task_id] = pa.primary_app or "unknown"

    # compressed text per (task, method)
    by_tm: Dict[Tuple[str, str], str] = {}
    for r in compressed:
        if r.get("error"):
            continue
        by_tm[(r["task_id"], r["method"])] = r.get("text", "")

    # full-context lookup
    trajs: Dict[str, "Trajectory"] = {}  # type: ignore  # noqa
    for r in sel:
        td = Path(r["output_dir"])
        if td.exists():
            try:
                trajs[r["task_id"]] = load_trajectory(td)
            except Exception:
                pass

    same_map, cross_map = _build_wrong_task_pairs(sel_tids, app_of, rng)

    # Build cells.
    cells: List[Tuple] = []
    skipped: List[str] = []
    for tid in sel_tids:
        for cond in args.conditions:
            if cond == "no_context":
                text = ""
            elif cond == "full_context":
                if tid not in trajs:
                    skipped.append(f"{tid}/{cond}: no trajectory loaded")
                    continue
                text = _build_full_context_text(trajs[tid])
            elif cond in ("task_aware_summary", "acon_style_summary",
                          "symbolic_evidence"):
                text = by_tm.get((tid, cond), None)
                if text is None:
                    skipped.append(f"{tid}/{cond}: no compressed context cell")
                    continue
            elif cond == "wrong_task_symbolic_same_app":
                src = same_map.get(tid)
                if src is None:
                    skipped.append(f"{tid}/{cond}: no same-app partner")
                    continue
                text = by_tm.get((src, "symbolic_evidence"))
                if text is None:
                    skipped.append(
                        f"{tid}/{cond}: partner {src} has no symbolic_evidence"
                    )
                    continue
            elif cond == "wrong_task_symbolic_cross_app":
                src = cross_map.get(tid)
                if src is None:
                    skipped.append(f"{tid}/{cond}: no cross-app partner")
                    continue
                text = by_tm.get((src, "symbolic_evidence"))
                if text is None:
                    skipped.append(
                        f"{tid}/{cond}: partner {src} has no symbolic_evidence"
                    )
                    continue
            else:
                raise SystemExit(f"unknown condition {cond!r}")

            for cap in args.budgets:
                cells.append((tid, cond, text, cap, args.tag))

    print(f"[05] designed {len(cells)} cells "
          f"({len(sel_tids)} tasks × {len(args.conditions)} conditions × "
          f"{len(args.budgets)} budgets)")
    if skipped:
        print(f"[05] skipped {len(skipped)} cells (sample):")
        for s in skipped[:5]:
            print(f"        {s}")

    out_path = jsonl_path("motivation_behavior_runs.jsonl")
    t0 = time.time()
    n_done = 0
    n_err = 0
    with open(out_path, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one_cell, c): c for c in cells}
            for fut in as_completed(futures):
                rec = fut.result()
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                n_done += 1
                tag_str = (f"{rec['task_id']:>11s} "
                           f"({rec['method']:>30s}@cap{rec['budget_max_steps']:<2d})")
                print(f"  [{n_done:>3d}/{len(cells)}] {tag_str}  "
                      f"success={rec['success']!s:<5}  "
                      f"iter={rec['iterations']:>2d}  elapsed={rec['elapsed_s']:.0f}s",
                      flush=True)

    print(f"\n[05] wrote {len(cells)} rows -> {out_path}")
    print(f"[05] elapsed: {(time.time()-t0)/60:.1f} min  err={n_err}")


if __name__ == "__main__":
    main()
