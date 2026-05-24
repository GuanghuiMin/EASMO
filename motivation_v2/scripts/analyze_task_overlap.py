"""Strong-T1 (corrected): how much do m*_exec memories differ ACROSS TASKS?

The user's actual thesis: *different downstream tasks need different
optimal compressions of a long context*. This is the natural reading
of "policy-conditional memory" once we recognise that "policy" is
the task-instantiated agent objective, not a strategy variant.

For each pair of successful tasks (A, B), compute
``Jaccard(m*_exec_minimal(A), m*_exec_minimal(B))`` at multiple
budgets. We use ``minimal`` (ground-truth API call list) rather
than ``trajectory`` so the comparison is task-objective-driven, not
executor-driven.

Three pair categories:

1. **Within-generator** (same 7-char task-id prefix; e.g.
   ``82e2fac_1`` vs ``82e2fac_2``). Same task generator, different
   parametric instance. Expected Jaccard HIGH — these are
   parametric variants of essentially the same task.
2. **Within-app, cross-generator** (e.g. one spotify generator vs
   another spotify generator). Different goals, same app. Expected
   Jaccard MEDIUM — they all use spotify endpoints, but different
   sub-functionality.
3. **Cross-app** (e.g. spotify vs venmo). Expected Jaccard LOW —
   different apps entirely.

The thesis hinges on category 2: if cross-generator within-app
Jaccard is low (≤ 0.5), tasks demand different memory even when
operating in the same domain → task-conditional memory matters.
If category 2 Jaccard is high (≥ 0.8), the within-app variation is
trivial and we have to look at category 3 to make the point.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.data import (
    iter_tasks_with_ground_truth,
    load_ground_truth,
)
from motivation_v2.exec_memory import (
    GroundTruth,
    m_exec_minimal,
)
from motivation_v2.policy_family import assign_policy_family


def _unit_key(unit) -> str:
    return f"{unit.app}|{unit.text[:120]}"


def _endpoint_key(unit) -> str:
    """Coarser key: only (app, METHOD, /path/template) — drops query
    arguments. Useful because two tasks with different specific
    parameters (page_index=0 vs 1) shouldn't be counted as different
    'endpoints'.
    """
    # unit.text looks like '[spotify] GET /spotify/library/songs(page_index=0, ...)'
    text = unit.text
    if "(" in text:
        text = text.split("(", 1)[0]
    return f"{unit.app}|{text}"


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument("--key_mode", choices=["unit", "endpoint"], default="endpoint",
                        help="'endpoint' compares (app, method, /path) "
                             "ignoring args (recommended); 'unit' is exact.")
    args = parser.parse_args()

    keyfn = _endpoint_key if args.key_mode == "endpoint" else _unit_key

    # Load all single-app tasks with ground truth.
    gts: List[GroundTruth] = []
    for gt in iter_tasks_with_ground_truth(args.split):
        pa = assign_policy_family(gt)
        if pa.is_single_app and pa.family != "supervisor_only":
            gts.append(gt)

    print(f"[task_overlap] {len(gts)} single-app tasks with GT in {args.split}")

    # Index by (app, generator-prefix).
    by_app_gen: Dict[Tuple[str, str], List[GroundTruth]] = defaultdict(list)
    for gt in gts:
        pa = assign_policy_family(gt)
        gen = gt.task_id.split("_", 1)[0]
        by_app_gen[(pa.primary_app, gen)].append(gt)

    apps = sorted({a for (a, _) in by_app_gen})
    print(f"  per-app counts:")
    for a in apps:
        n_tasks = sum(len(v) for (app, gen), v in by_app_gen.items() if app == a)
        n_gens = sum(1 for (app, gen) in by_app_gen if app == a)
        print(f"    {a}: {n_tasks} tasks across {n_gens} generators")
    print()

    for B in args.budgets:
        # Build m_exec_minimal sets per task.
        sets_by_task: Dict[str, Set[str]] = {}
        for gt in gts:
            mem = m_exec_minimal(gt, B)
            sets_by_task[gt.task_id] = {keyfn(u) for u in mem.units}

        within_gen: List[float] = []
        within_app_cross_gen: List[float] = []
        cross_app: List[float] = []

        for ga, sa in sets_by_task.items():
            gt_a = next(g for g in gts if g.task_id == ga)
            pa_a = assign_policy_family(gt_a)
            gen_a = ga.split("_", 1)[0]
            for gb, sb in sets_by_task.items():
                if gb <= ga:
                    continue
                gt_b = next(g for g in gts if g.task_id == gb)
                pa_b = assign_policy_family(gt_b)
                gen_b = gb.split("_", 1)[0]
                j = _jaccard(sa, sb)
                if pa_a.primary_app != pa_b.primary_app:
                    cross_app.append(j)
                elif gen_a == gen_b:
                    within_gen.append(j)
                else:
                    within_app_cross_gen.append(j)

        def _stat(xs):
            if not xs:
                return "  n=0"
            return (f"  n={len(xs):>4d}  mean={statistics.mean(xs):.3f}  "
                    f"median={statistics.median(xs):.3f}  "
                    f"min={min(xs):.3f}  max={max(xs):.3f}")

        print(f"=== B={B} (key_mode={args.key_mode}) ===")
        print(f"  within-generator (sibling tasks):           {_stat(within_gen)}")
        print(f"  within-app, cross-generator:                 {_stat(within_app_cross_gen)}")
        print(f"  cross-app (different apps):                  {_stat(cross_app)}")
        print()

    print("Verdict (rough):")
    print("  If within-app cross-generator Jaccard is < 0.5, ")
    print("  tasks within the same domain demand different memory ⇒ T1 supported")
    print("  in the task-conditional framing.")


if __name__ == "__main__":
    main()
