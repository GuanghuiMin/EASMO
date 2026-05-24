"""Smoke test: read AppWorld ground truth + acon trajectories, classify
policy families, build m*_exec_minimal / m*_exec_trajectory at several
budgets. Prints a small report.

Doesn't make any LLM calls. Run from this directory with the EASMO venv
(motivation_v2 doesn't import AppWorld so the broken pydantic isn't an
issue here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.data import (
    iter_tasks_with_ground_truth,
    successful_trajectories,
    load_ground_truth,
)
from motivation_v2.policy_family import (
    assign_policy_family,
    assign_all,
    family_distribution,
    shared_state_pairs,
)
from motivation_v2.exec_memory import m_exec_minimal, m_exec_trajectory


def main():
    print("=" * 70)
    print("AppWorld ground-truth audit")
    print("=" * 70)
    for split in ("train", "dev", "train_tiny"):
        gts = list(iter_tasks_with_ground_truth(split))
        if not gts:
            print(f"  [{split}] no tasks with ground truth on disk.")
            continue
        assignments = assign_all(gts)
        dist = family_distribution(assignments)
        single = sum(1 for a in assignments if a.is_single_app)
        print(f"\n  [{split}] {len(gts)} tasks with ground truth")
        print(f"     single-app: {single}/{len(gts)} ({100*single/len(gts):.0f}%)")
        print(f"     family distribution:")
        for fam, cnt in sorted(dist.items(), key=lambda kv: -kv[1]):
            tag = "" if fam != "multi_app" else "  ← excluded from M3 heatmap"
            print(f"       {fam:25s} {cnt}{tag}")
        # Difficulty stratification (R-4)
        from collections import Counter
        diffs = Counter(g.difficulty for g in gts)
        print(f"     difficulty distribution: {dict(sorted(diffs.items()))}")

        if split == "train":
            ssp = shared_state_pairs(assignments)
            if ssp:
                print(f"     shared-state cross-policy pairs (R-3):")
                for (p, q), cnt in sorted(ssp.items(), key=lambda kv: -kv[1]):
                    print(f"       {p} ↔ {q}: {cnt}")
            else:
                print("     shared-state cross-policy pairs: 0 (M3 will need fallback matching)")

    print()
    print("=" * 70)
    print("Successful trajectories on disk (acon outputs)")
    print("=" * 70)
    trajs = successful_trajectories()
    print(f"\n  {len(trajs)} successful trajectories found:")
    for t in trajs:
        print(f"    {t.experiment_name:55s} {t.task_id:>15s}  iters={t.iterations:>3d} reward={t.final_reward}")

    if not trajs:
        print("\n(no successful trajectories — cannot exercise m_exec_trajectory)")
        return

    # Pick the first successful trajectory and build both m_exec variants
    # at multiple budgets.
    print()
    print("=" * 70)
    print("m*_exec smoke test on first successful trajectory")
    print("=" * 70)
    traj = trajs[0]
    gt = load_ground_truth(traj.task_id)
    pa = assign_policy_family(gt)

    print(f"\n  task: {traj.task_id}")
    print(f"  policy family: {pa.family}  (single_app={pa.is_single_app})")
    print(f"  difficulty: {gt.difficulty}, num_apps={gt.num_apps}, "
          f"gt_api_calls={len(gt.api_calls)}, traj_steps={len(traj.steps)}")
    print(f"  instruction: {traj.instruction}")

    print(f"\n  m*_exec_minimal (ground-truth oracle):")
    for B in (128, 256, 512, 1024, 2048):
        em = m_exec_minimal(gt, budget_tokens=B)
        print(f"    B={B:>5d}: {em.n_units:>3d} units / "
              f"{em.n_tokens:>5d} tokens  (dropped {em.n_units_dropped})")

    print(f"\n  m*_exec_trajectory (executor-conditioned, executor={traj.model_name}):")
    for B in (128, 256, 512, 1024, 2048):
        em = m_exec_trajectory(gt, traj, budget_tokens=B)
        print(f"    B={B:>5d}: {em.n_units:>3d} units / "
              f"{em.n_tokens:>5d} tokens  (dropped {em.n_units_dropped})")

    print(f"\n  Sample: m*_exec_minimal(B=512) text:")
    em = m_exec_minimal(gt, 512)
    print("    " + em.text.replace("\n", "\n    "))


if __name__ == "__main__":
    main()
