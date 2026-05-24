"""Cross-role m*_exec overlap on AppWorld trajectories.

Tests the strong T1 claim: for the SAME upstream context (one
successful trajectory), four agent roles (tool / code / plan /
verify) build measurably different m*_exec memories.

If Jaccard across roles is low (≤ 0.3 across all budgets), the
strong claim "different roles need different memory" is supported,
even when the roles are projecting from the same underlying
trajectory.

For comparison we also report:
* Same-role cross-task Jaccard (task variation effect, role fixed)
  — this should be HIGHER than cross-role within-task, supporting
  the "memory policy is role-conditional, not task-conditional"
  message.

Reads successful trajectories directly from acon's outputs. No LLM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.data import successful_trajectories
from motivation_v2.role_memory import ROLE_BUILDERS


def _unit_key(u) -> str:
    """Role-agnostic key: strip the role-specific bracket prefix (e.g.
    `[code step 5]` or `[plan step 5 intent]`) and use the underlying
    text content, normalised to first 80 chars.

    Including ``u.kind`` would make cross-role Jaccard exactly 0 by
    construction since each role builder emits unique kind strings.
    """
    txt = u.text
    if "]" in txt:
        txt = txt.split("]", 1)[1].strip()
    # Normalise whitespace.
    txt = " ".join(txt.split())
    return txt[:80]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="direct",
                        help="which strategy's trajectories to use as the upstream-context source")
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{args.tag}_{args.strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    print(f"[role_overlap] {len(trajs)} successful '{args.strategy}' trajectories")
    if not trajs:
        sys.exit("no trajectories found")

    # Build role memories per task.
    # role_mem[(task_id, role, B)] = set of unit keys
    role_mem: Dict[tuple, Set[str]] = {}
    for t in trajs:
        for role, builder in ROLE_BUILDERS.items():
            for B in args.budgets:
                em = builder(t, B)
                role_mem[(t.task_id, role, B)] = {_unit_key(u) for u in em.units}

    # ---- Cross-role within-task Jaccard ----
    print(f"\n=== Cross-role Jaccard (same task, different role) ===")
    print(f"  hypothesis: roles want orthogonal memory ⇒ Jaccard low across all B")
    print()
    print(f"{'B':>6} | " + "  ".join(f"{r1}-{r2:<6s}" for r1, r2 in combinations(ROLE_BUILDERS, 2)))
    print("-" * 110)

    per_budget_xrole: Dict[int, Dict[str, list]] = {B: defaultdict(list) for B in args.budgets}
    for t in trajs:
        for B in args.budgets:
            for r1, r2 in combinations(ROLE_BUILDERS, 2):
                a = role_mem[(t.task_id, r1, B)]
                b = role_mem[(t.task_id, r2, B)]
                per_budget_xrole[B][f"{r1}_{r2}"].append(_jaccard(a, b))

    summary = {"cross_role": {}, "cross_task_per_role": {}}
    for B in args.budgets:
        cells = []
        line_data = {}
        for r1, r2 in combinations(ROLE_BUILDERS, 2):
            xs = per_budget_xrole[B][f"{r1}_{r2}"]
            mean = statistics.mean(xs) if xs else 0.0
            cells.append(f"{mean:>9.3f}")
            line_data[f"{r1}_{r2}_mean"] = mean
            line_data[f"{r1}_{r2}_n"] = len(xs)
        print(f"{B:>6} | " + "  ".join(cells))
        summary["cross_role"][B] = line_data

    # ---- Cross-task within-role Jaccard ----
    print(f"\n=== Cross-task Jaccard (same role, different task) ===")
    print(f"  hypothesis: within-role memory transfers across tasks ⇒ Jaccard moderate-to-high")
    print()
    print(f"{'B':>6} | " + "  ".join(f"{role:>10s}" for role in ROLE_BUILDERS))
    print("-" * 80)

    for B in args.budgets:
        cells = []
        line_data = {}
        for role in ROLE_BUILDERS:
            jacs: list = []
            tids = [t.task_id for t in trajs]
            for ta, tb in combinations(tids, 2):
                a = role_mem[(ta, role, B)]
                b = role_mem[(tb, role, B)]
                jacs.append(_jaccard(a, b))
            mean = statistics.mean(jacs) if jacs else 0.0
            cells.append(f"{mean:>10.3f}")
            line_data[f"{role}_mean"] = mean
            line_data[f"{role}_n_pairs"] = len(jacs)
        print(f"{B:>6} | " + "  ".join(cells))
        summary["cross_task_per_role"][B] = line_data

    # ---- Verdict ----
    print(f"\n=== Verdict (rough) ===")
    # cross-role at B=512 should be low if strong T1 holds
    if 512 in args.budgets:
        xrole_512_mean = statistics.mean([
            v for k, v in summary["cross_role"][512].items() if k.endswith("_mean")
        ])
        xtask_512_mean = statistics.mean([
            v for k, v in summary["cross_task_per_role"][512].items() if k.endswith("_mean")
        ])
        print(f"  B=512:  cross-role mean Jaccard = {xrole_512_mean:.3f}")
        print(f"  B=512:  cross-task within-role mean Jaccard = {xtask_512_mean:.3f}")
        if xrole_512_mean < xtask_512_mean:
            print(f"  ✓ cross-role < cross-task → memory variation is role-driven, not task-driven")
        else:
            print(f"  ⚠ cross-role >= cross-task → roles converge as much as tasks; strong-T1 weak")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
