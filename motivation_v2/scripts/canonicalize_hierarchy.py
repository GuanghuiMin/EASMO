"""Experiment A — Three-level memory divergence hierarchy (canonical output).

Reads existing successful trajectories and re-computes per-pair Jaccard
cells for three axes (strategy / task / role), then writes:

    outputs/motivation/hierarchy_raw.jsonl      one row per pair
    outputs/motivation/hierarchy_summary.csv    aggregated stats
    figures/motivation/hierarchy_b512.pdf       bar plot at B=512
    figures/motivation/hierarchy_by_executor.pdf one panel per executor

Spec reference: experiment_modification.md §5 (Experiment A).

Each row in hierarchy_raw.jsonl looks like:

    {"axis": "strategy", "executor": "MiniMaxAI/MiniMax-M2.5",
     "budget": 512, "task_id": "82e2fac_3", "pair_label": "direct_verify",
     "jaccard": 0.91, "n_a": 25, "n_b": 25}

Aggregate stats per (axis, budget, executor) include mean / std / median
/ min / max / n_pairs.

All entity-token Jaccards use the canonical_io.entity_tokens definition
so cross-experiment comparisons are like-for-like.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.canonical_io import (
    DistribStats,
    OUTPUTS_DIR,
    FIGURES_DIR,
    ensure_dirs,
    entity_tokens,
    jaccard,
    log_run_meta,
    unit_text_normalized,
    write_csv,
    write_jsonl,
)
from motivation_v2.data import successful_trajectories
from motivation_v2.role_memory import ROLE_BUILDERS


_EXEC_DEFAULT = "MiniMaxAI/MiniMax-M2.5"
_BUDGETS_DEFAULT = (128, 256, 512, 1024)


# ----------------------------------------------------------------------
# Memory builders (entity-token sets per (task, axis-value, budget)).
# ----------------------------------------------------------------------


def _strategy_token_sets(
    strategies: List[str],
    tag: str,
    budgets: Tuple[int, ...],
) -> Dict[Tuple[str, str, int], Set[str]]:
    """Returns {(task_id, strategy, budget) -> token_set}.

    For the strategy axis we use m_tool memory (default role) so that
    only strategy varies. Trajectories are loaded per-strategy and the
    intersection of task_ids across all strategies is kept so we
    compare the SAME tasks across strategies.
    """
    by_strategy: Dict[str, List] = {}
    for s in strategies:
        glob = (
            f"/workspace/acon/experiments/appworld/outputs/"
            f"MiniMaxAI_MiniMax-M2.5_{tag}_{s}/train/task_*"
        )
        trajs = successful_trajectories(experiments_glob=glob)
        by_strategy[s] = trajs

    common = set(t.task_id for t in by_strategy[strategies[0]])
    for s in strategies[1:]:
        common &= set(t.task_id for t in by_strategy[s])
    print(f"[A/strategy] common task_ids across {strategies}: n={len(common)}")

    out: Dict[Tuple[str, str, int], Set[str]] = {}
    builder = ROLE_BUILDERS["tool"]
    for s in strategies:
        for t in by_strategy[s]:
            if t.task_id not in common:
                continue
            for B in budgets:
                em = builder(t, B)
                joined = "\n".join(u.text for u in em.units)
                out[(t.task_id, s, B)] = entity_tokens(joined)
    return out, sorted(common)


def _role_token_sets(
    strategy: str,
    tag: str,
    budgets: Tuple[int, ...],
) -> Dict[Tuple[str, str, int], Set[str]]:
    """Returns {(task_id, role, budget) -> token_set}."""
    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{tag}_{strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    print(f"[A/role] {strategy} successful trajectories: n={len(trajs)}")
    out: Dict[Tuple[str, str, int], Set[str]] = {}
    for t in trajs:
        for role, builder in ROLE_BUILDERS.items():
            for B in budgets:
                em = builder(t, B)
                joined = "\n".join(u.text for u in em.units)
                out[(t.task_id, role, B)] = entity_tokens(joined)
    return out, [t.task_id for t in trajs]


def _task_token_sets_per_role(
    strategy: str,
    tag: str,
    budgets: Tuple[int, ...],
) -> Dict[Tuple[str, str, int], Set[str]]:
    """Same as _role_token_sets but used for the cross-task axis.

    For the task axis we report cross-task Jaccard *within each role*
    separately (so the spec's headline 'Task Jaccard' becomes the mean
    across roles to match the role-balanced framing). Returns the same
    structure as _role_token_sets.
    """
    return _role_token_sets(strategy, tag, budgets)


# ----------------------------------------------------------------------
# Per-axis pair iterator
# ----------------------------------------------------------------------


def _strategy_pairs(
    sets_: Dict[Tuple[str, str, int], Set[str]],
    common_tids: List[str],
    strategies: List[str],
    budget: int,
):
    for tid in common_tids:
        for s1, s2 in combinations(strategies, 2):
            a = sets_[(tid, s1, budget)]
            b = sets_[(tid, s2, budget)]
            yield {
                "axis": "strategy",
                "budget": budget,
                "task_id": tid,
                "pair_label": f"{s1}_{s2}",
                "jaccard": jaccard(a, b),
                "n_a": len(a),
                "n_b": len(b),
            }


def _role_pairs(
    sets_: Dict[Tuple[str, str, int], Set[str]],
    tids: List[str],
    budget: int,
):
    role_names = list(ROLE_BUILDERS.keys())
    for tid in tids:
        for r1, r2 in combinations(role_names, 2):
            a = sets_[(tid, r1, budget)]
            b = sets_[(tid, r2, budget)]
            yield {
                "axis": "role",
                "budget": budget,
                "task_id": tid,
                "pair_label": f"{r1}_{r2}",
                "jaccard": jaccard(a, b),
                "n_a": len(a),
                "n_b": len(b),
            }


def _task_pairs_per_role(
    sets_: Dict[Tuple[str, str, int], Set[str]],
    tids: List[str],
    budget: int,
):
    """Cross-task pairs *within the same role*."""
    role_names = list(ROLE_BUILDERS.keys())
    for role in role_names:
        for ta, tb in combinations(tids, 2):
            a = sets_[(ta, role, budget)]
            b = sets_[(tb, role, budget)]
            yield {
                "axis": "task",
                "budget": budget,
                "task_id": f"{ta}|{tb}",
                "pair_label": f"role={role}",
                "role": role,
                "jaccard": jaccard(a, b),
                "n_a": len(a),
                "n_b": len(b),
            }


# ----------------------------------------------------------------------
# Plotting (matplotlib)
# ----------------------------------------------------------------------


def _plot_b512(rows: List[dict], out_path: Path, executor: str):
    import matplotlib.pyplot as plt
    import numpy as np

    by_axis = {"strategy": [], "task": [], "role": []}
    for r in rows:
        if r["budget"] != 512:
            continue
        if r["axis"] not in by_axis:
            continue
        by_axis[r["axis"]].append(r["jaccard"])

    axis_labels = ["strategy", "task", "role"]
    means = [
        sum(by_axis[a]) / max(len(by_axis[a]), 1) for a in axis_labels
    ]
    n_pairs = [len(by_axis[a]) for a in axis_labels]
    stds = []
    for a in axis_labels:
        xs = by_axis[a]
        if len(xs) >= 2:
            mu = sum(xs) / len(xs)
            v = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
            stds.append(v ** 0.5)
        else:
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(axis_labels))
    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=["#69b3a2", "#f3b562", "#c44a4a"],
        edgecolor="black",
    )
    for i, (m, n) in enumerate(zip(means, n_pairs)):
        ax.text(i, m + 0.02, f"{m:.3f}\nn={n}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(varied)" for a in axis_labels])
    ax.set_ylim(0, max(1.0, max(means) * 1.25))
    ax.set_ylabel("Pair-wise Jaccard at B=512")
    ax.set_title(f"Three-level memory divergence hierarchy — {executor}")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def _aggregate(rows: List[dict]) -> List[dict]:
    """Return one row per (axis, budget, executor) with full stats."""
    bucket: Dict[Tuple[str, int, str], List[float]] = {}
    for r in rows:
        key = (r["axis"], r["budget"], r["executor"])
        bucket.setdefault(key, []).append(r["jaccard"])
    out: List[dict] = []
    for (axis, B, exe), xs in sorted(bucket.items()):
        s = DistribStats.from_values(xs)
        out.append({
            "experiment": "A_hierarchy",
            "executor": exe,
            "axis": axis,
            "budget": B,
            "n_pairs": s.n,
            "jaccard_mean": round(s.mean, 4),
            "jaccard_std": round(s.std, 4),
            "jaccard_median": round(s.median, 4),
            "jaccard_min": round(s.min_, 4),
            "jaccard_max": round(s.max_, 4),
        })
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["direct", "verify", "explore"])
    parser.add_argument("--main_strategy", default="direct",
                        help="Strategy used for the role and task axes "
                             "(strategy axis uses all three).")
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int, default=list(_BUDGETS_DEFAULT))
    parser.add_argument("--executor", default=_EXEC_DEFAULT)
    args = parser.parse_args()

    ensure_dirs()
    log_run_meta(
        "A_hierarchy",
        executor=args.executor,
        seed=42,
    )

    # ----- Build per-axis token sets -----
    print("[A] Building strategy-axis token sets...")
    strat_sets, common_tids_strategy = _strategy_token_sets(
        args.strategies, args.tag, tuple(args.budgets),
    )
    print(f"[A] strategy axis: {len(common_tids_strategy)} common tasks")

    print("[A] Building role/task-axis token sets (strategy=%s)..." % args.main_strategy)
    role_sets, all_tids = _role_token_sets(
        args.main_strategy, args.tag, tuple(args.budgets),
    )
    print(f"[A] role/task axes: {len(all_tids)} successful tasks at strategy='{args.main_strategy}'")

    # ----- Emit raw cells -----
    rows: List[dict] = []
    for B in args.budgets:
        for cell in _strategy_pairs(strat_sets, common_tids_strategy, args.strategies, B):
            cell["executor"] = args.executor
            rows.append(cell)
        for cell in _role_pairs(role_sets, all_tids, B):
            cell["executor"] = args.executor
            rows.append(cell)
        for cell in _task_pairs_per_role(role_sets, all_tids, B):
            cell["executor"] = args.executor
            rows.append(cell)

    raw_path = OUTPUTS_DIR / "hierarchy_raw.jsonl"
    n_raw = write_jsonl(raw_path, rows)
    print(f"[A] wrote {n_raw} raw pair rows -> {raw_path}")

    summary = _aggregate(rows)
    sum_path = OUTPUTS_DIR / "hierarchy_summary.csv"
    n_sum = write_csv(sum_path, summary)
    print(f"[A] wrote {n_sum} summary rows -> {sum_path}")

    # ----- Print human-readable table -----
    print()
    print("=== Three-level hierarchy (mean Jaccard, executor=%s) ===" % args.executor)
    print(f"{'budget':>6} | {'strategy':>10}  | {'task':>10}  | {'role':>10}")
    print("-" * 60)
    by_axis_budget = {(r["axis"], r["budget"]): r for r in summary}
    for B in args.budgets:
        s = by_axis_budget.get(("strategy", B), {})
        t = by_axis_budget.get(("task", B), {})
        r = by_axis_budget.get(("role", B), {})
        def _f(d):
            if not d: return "n/a"
            return (f"{d['jaccard_mean']:.3f}±{d['jaccard_std']:.2f}"
                    f" (n={d['n_pairs']})")
        print(f"{B:>6} | {_f(s):>14}  | {_f(t):>14}  | {_f(r):>14}")

    # ----- Plots -----
    _plot_b512([{**r, "executor": args.executor} for r in rows],
               FIGURES_DIR / "hierarchy_b512.pdf",
               args.executor)
    # by_executor: same plot for the only executor; placeholder for cross-executor.
    _plot_b512([{**r, "executor": args.executor} for r in rows],
               FIGURES_DIR / "hierarchy_by_executor.pdf",
               args.executor)

    print()
    print("[A] Done.")


if __name__ == "__main__":
    main()
