"""Strong-T1 preview: how much do m*_exec memories overlap across strategies?

For each task that succeeded under ALL THREE strategies (direct, verify,
explore), build:

    m_d = m*_exec_trajectory(task, executor=MiniMax, traj=direct, B)
    m_v = m*_exec_trajectory(task, executor=MiniMax, traj=verify, B)
    m_e = m*_exec_trajectory(task, executor=MiniMax, traj=explore, B)

Then report, per task and aggregated:

* Jaccard(m_d.units, m_v.units), Jaccard(m_d.units, m_e.units),
  Jaccard(m_v.units, m_e.units), and the 3-way Jaccard.
* The same on the *minimal* variant (which is executor-independent
  and so should be identical across strategies — sanity check).

A high Jaccard (≥ 0.8) means the three strategies want roughly the
same information; the strong-T1 narrative ("policy-conditional
memory matters") would be empty.

Reads from the pilot output directory directly. No LLM calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.data import (
    iter_trajectories,
    load_ground_truth,
    successful_trajectories,
)
from motivation_v2.exec_memory import (
    m_exec_minimal,
    m_exec_trajectory,
)


def _unit_key(unit) -> str:
    """Key by app + first-100-chars-of-text. Matches "same row" without
    being defeated by minor formatting differences."""
    return f"{unit.app}|{unit.text[:100]}"


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    # Index successful trajectories by (task_id, strategy).
    by_task: Dict[str, Dict[str, object]] = {}
    for strategy in ("direct", "verify", "explore"):
        glob = (
            f"/workspace/acon/experiments/appworld/outputs/"
            f"MiniMaxAI_MiniMax-M2.5_{args.tag}_{strategy}/train/task_*"
        )
        for traj in iter_trajectories(experiments_glob=glob):
            if not traj.success:
                continue
            by_task.setdefault(traj.task_id, {})[strategy] = traj

    # Filter to tasks present in all three strategies.
    triples = {tid: ts for tid, ts in by_task.items()
               if {"direct", "verify", "explore"}.issubset(ts.keys())}
    print(f"[analyze_exec_overlap] {len(by_task)} tasks succeeded somewhere; "
          f"{len(triples)} succeeded under ALL THREE strategies")
    if not triples:
        print("  (no triple-success tasks yet — pilot still in flight)")
        sys.exit(0)

    # ---- Aggregate per budget ----
    per_budget: Dict[int, Dict[str, list]] = {
        B: {
            "j_dv": [],     # direct vs verify
            "j_de": [],     # direct vs explore
            "j_ve": [],     # verify vs explore
            "j_3way": [],   # 3-way intersection / union
            "minimal_dv": [],   # ground-truth-only: should be 1.0
            "n_units_d": [],
            "n_units_v": [],
            "n_units_e": [],
        }
        for B in args.budgets
    }

    for tid, ts in triples.items():
        try:
            gt = load_ground_truth(tid)
        except FileNotFoundError:
            continue
        for B in args.budgets:
            md = m_exec_trajectory(gt, ts["direct"], B)
            mv = m_exec_trajectory(gt, ts["verify"], B)
            me = m_exec_trajectory(gt, ts["explore"], B)
            sd = {_unit_key(u) for u in md.units}
            sv = {_unit_key(u) for u in mv.units}
            se = {_unit_key(u) for u in me.units}
            j_dv = _jaccard(sd, sv)
            j_de = _jaccard(sd, se)
            j_ve = _jaccard(sv, se)
            inter = sd & sv & se
            uni = sd | sv | se
            j_3 = len(inter) / max(len(uni), 1)
            per_budget[B]["j_dv"].append(j_dv)
            per_budget[B]["j_de"].append(j_de)
            per_budget[B]["j_ve"].append(j_ve)
            per_budget[B]["j_3way"].append(j_3)
            per_budget[B]["n_units_d"].append(md.n_units)
            per_budget[B]["n_units_v"].append(mv.n_units)
            per_budget[B]["n_units_e"].append(me.n_units)

            # Sanity check: m_exec_minimal is executor-/strategy-independent
            # so its Jaccard between any two strategies should be 1.0.
            mim = m_exec_minimal(gt, B)
            sm = {_unit_key(u) for u in mim.units}
            per_budget[B]["minimal_dv"].append(_jaccard(sm, sm))

    # ---- Print summary ----
    print()
    print(f"{'B':>6} | {'n':>3} | {'J(d,v)':>8} {'J(d,e)':>8} {'J(v,e)':>8} {'J(3way)':>8} | "
          f"{'|d|':>5} {'|v|':>5} {'|e|':>5} | minimal_dv")
    print("-" * 100)
    summary = {"per_budget": {}}
    for B, vals in per_budget.items():
        n = len(vals["j_dv"])
        if n == 0:
            continue
        j_dv = statistics.mean(vals["j_dv"])
        j_de = statistics.mean(vals["j_de"])
        j_ve = statistics.mean(vals["j_ve"])
        j_3 = statistics.mean(vals["j_3way"])
        nd = statistics.mean(vals["n_units_d"])
        nv = statistics.mean(vals["n_units_v"])
        ne = statistics.mean(vals["n_units_e"])
        mim_dv = statistics.mean(vals["minimal_dv"])
        print(f"{B:>6} | {n:>3} | {j_dv:>8.3f} {j_de:>8.3f} {j_ve:>8.3f} {j_3:>8.3f} | "
              f"{nd:>5.1f} {nv:>5.1f} {ne:>5.1f} | {mim_dv:.2f}")
        summary["per_budget"][B] = {
            "n_tasks": n,
            "jaccard_direct_verify": j_dv,
            "jaccard_direct_explore": j_de,
            "jaccard_verify_explore": j_ve,
            "jaccard_3way_intersection_over_union": j_3,
            "mean_units_direct": nd,
            "mean_units_verify": nv,
            "mean_units_explore": ne,
            "minimal_dv_sanity_should_be_1.0": mim_dv,
        }

    print()
    print("Verdict (rough):")
    j_dv_mid = summary["per_budget"].get(512, {}).get("jaccard_direct_verify")
    if j_dv_mid is not None:
        if j_dv_mid >= 0.8:
            print(f"  ⚠ Jaccard(direct,verify) at B=512 = {j_dv_mid:.2f} ≥ 0.80")
            print(f"    → strategies want nearly identical memory; STRONG T1 in trouble.")
            print(f"    → fallback narrative: 'compression for tool-use is necessary at tight B'")
        elif j_dv_mid >= 0.5:
            print(f"  → Jaccard(direct,verify) at B=512 = {j_dv_mid:.2f} ∈ [0.50, 0.80)")
            print(f"    → moderate divergence; weak-T1 publishable, not spotlight.")
        else:
            print(f"  ✓ Jaccard(direct,verify) at B=512 = {j_dv_mid:.2f} < 0.50")
            print(f"    → strategies want clearly different memory; STRONG T1 defensible.")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
