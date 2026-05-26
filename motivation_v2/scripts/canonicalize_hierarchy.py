"""Experiment A — Three-level memory divergence hierarchy (canonical output).

Reads existing successful trajectories and re-computes per-pair Jaccard
cells for three axes (strategy / task / role), then writes:

    outputs/motivation/hierarchy_raw.jsonl      one row per pair
    outputs/motivation/hierarchy_summary.csv    aggregated stats
    figures/motivation/hierarchy_b512.pdf       bar plot at B=512
    figures/motivation/hierarchy_by_executor.pdf one panel per executor

Spec reference: experiment_modification.md §5 (Experiment A).

Each row in hierarchy_raw.jsonl looks like:

    {"axis": "role", "executor": "MiniMaxAI/MiniMax-M2.5",
     "budget": 512, "task_id": "82e2fac_3", "pair_label": "tool_code",
     "jaccard_token": 0.04, "jaccard_unit": 0.00,
     "n_a_tokens": 38, "n_b_tokens": 41}

We report two Jaccard variants (spec §4.1):
  * jaccard_token: bracket-stripped entity-token bag Jaccard (primary).
  * jaccard_unit:  unit-text-normalized hashable-key set Jaccard.

Aggregate stats per (axis, budget, executor) include mean / std /
median / min / max / n_pairs for both variants.
"""

from __future__ import annotations

import argparse
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
# Memory builders (entity-token + unit sets per (task, axis-value, budget))
# ----------------------------------------------------------------------


def _strategy_memory_sets(
    strategies: List[str],
    tag: str,
    budgets: Tuple[int, ...],
):
    """Build (token_sets, unit_sets, common_tids) keyed by
    (task_id, strategy, budget).

    For the strategy axis we use m_tool memory (default role) so that
    only strategy varies. We keep the intersection of task_ids
    successful under all three strategies so we compare matched cells.
    """
    by_strategy: Dict[str, list] = {}
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

    tok: Dict[Tuple[str, str, int], Set[str]] = {}
    unit: Dict[Tuple[str, str, int], Set[str]] = {}
    builder = ROLE_BUILDERS["tool"]
    for s in strategies:
        for t in by_strategy[s]:
            if t.task_id not in common:
                continue
            for B in budgets:
                em = builder(t, B)
                joined = "\n".join(u.text for u in em.units)
                tok[(t.task_id, s, B)] = entity_tokens(joined)
                unit[(t.task_id, s, B)] = {
                    unit_text_normalized(u.text) for u in em.units
                }
    return tok, unit, sorted(common)


def _role_memory_sets(
    strategy: str,
    tag: str,
    budgets: Tuple[int, ...],
):
    """Build (token_sets, unit_sets, all_tids) keyed by
    (task_id, role, budget)."""
    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{tag}_{strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    print(f"[A/role] {strategy} successful trajectories: n={len(trajs)}")
    tok: Dict[Tuple[str, str, int], Set[str]] = {}
    unit: Dict[Tuple[str, str, int], Set[str]] = {}
    for t in trajs:
        for role, builder in ROLE_BUILDERS.items():
            for B in budgets:
                em = builder(t, B)
                joined = "\n".join(u.text for u in em.units)
                tok[(t.task_id, role, B)] = entity_tokens(joined)
                unit[(t.task_id, role, B)] = {
                    unit_text_normalized(u.text) for u in em.units
                }
    return tok, unit, [t.task_id for t in trajs]


# ----------------------------------------------------------------------
# Per-axis pair iterator
# ----------------------------------------------------------------------


def _strategy_pairs(
    tok_sets: Dict[Tuple[str, str, int], Set[str]],
    unit_sets: Dict[Tuple[str, str, int], Set[str]],
    common_tids: List[str],
    strategies: List[str],
    budget: int,
):
    for tid in common_tids:
        for s1, s2 in combinations(strategies, 2):
            a_t, b_t = tok_sets[(tid, s1, budget)], tok_sets[(tid, s2, budget)]
            a_u, b_u = unit_sets[(tid, s1, budget)], unit_sets[(tid, s2, budget)]
            yield {
                "axis": "strategy",
                "budget": budget,
                "task_id": tid,
                "pair_label": f"{s1}_{s2}",
                "jaccard_token": jaccard(a_t, b_t),
                "jaccard_unit": jaccard(a_u, b_u),
                "n_a_tokens": len(a_t),
                "n_b_tokens": len(b_t),
                "n_a_units": len(a_u),
                "n_b_units": len(b_u),
            }


def _role_pairs(
    tok_sets: Dict[Tuple[str, str, int], Set[str]],
    unit_sets: Dict[Tuple[str, str, int], Set[str]],
    tids: List[str],
    budget: int,
):
    role_names = list(ROLE_BUILDERS.keys())
    for tid in tids:
        for r1, r2 in combinations(role_names, 2):
            a_t, b_t = tok_sets[(tid, r1, budget)], tok_sets[(tid, r2, budget)]
            a_u, b_u = unit_sets[(tid, r1, budget)], unit_sets[(tid, r2, budget)]
            yield {
                "axis": "role",
                "budget": budget,
                "task_id": tid,
                "pair_label": f"{r1}_{r2}",
                "jaccard_token": jaccard(a_t, b_t),
                "jaccard_unit": jaccard(a_u, b_u),
                "n_a_tokens": len(a_t),
                "n_b_tokens": len(b_t),
                "n_a_units": len(a_u),
                "n_b_units": len(b_u),
            }


def _task_pairs_per_role(
    tok_sets: Dict[Tuple[str, str, int], Set[str]],
    unit_sets: Dict[Tuple[str, str, int], Set[str]],
    tids: List[str],
    budget: int,
):
    """Cross-task pairs *within the same role*."""
    role_names = list(ROLE_BUILDERS.keys())
    for role in role_names:
        for ta, tb in combinations(tids, 2):
            a_t, b_t = tok_sets[(ta, role, budget)], tok_sets[(tb, role, budget)]
            a_u, b_u = unit_sets[(ta, role, budget)], unit_sets[(tb, role, budget)]
            yield {
                "axis": "task",
                "budget": budget,
                "task_id": f"{ta}|{tb}",
                "pair_label": f"role={role}",
                "role": role,
                "jaccard_token": jaccard(a_t, b_t),
                "jaccard_unit": jaccard(a_u, b_u),
                "n_a_tokens": len(a_t),
                "n_b_tokens": len(b_t),
                "n_a_units": len(a_u),
                "n_b_units": len(b_u),
            }


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def _aggregate(rows: List[dict]) -> List[dict]:
    """Return one row per (axis, budget, executor) with full stats for
    both Jaccard variants."""
    bucket: Dict[Tuple[str, int, str], Dict[str, List[float]]] = {}
    for r in rows:
        key = (r["axis"], r["budget"], r["executor"])
        b = bucket.setdefault(key, {"token": [], "unit": []})
        b["token"].append(r["jaccard_token"])
        b["unit"].append(r["jaccard_unit"])
    out: List[dict] = []
    for (axis, B, exe), b in sorted(bucket.items()):
        s_t = DistribStats.from_values(b["token"])
        s_u = DistribStats.from_values(b["unit"])
        out.append({
            "experiment": "A_hierarchy",
            "executor": exe,
            "axis": axis,
            "budget": B,
            "n_pairs": s_t.n,
            "jaccard_token_mean":   round(s_t.mean,   4),
            "jaccard_token_std":    round(s_t.std,    4),
            "jaccard_token_median": round(s_t.median, 4),
            "jaccard_token_min":    round(s_t.min_,   4),
            "jaccard_token_max":    round(s_t.max_,   4),
            "jaccard_unit_mean":    round(s_u.mean,   4),
            "jaccard_unit_std":     round(s_u.std,    4),
            "jaccard_unit_median":  round(s_u.median, 4),
            "jaccard_unit_min":     round(s_u.min_,   4),
            "jaccard_unit_max":     round(s_u.max_,   4),
        })
    return out


# ----------------------------------------------------------------------
# Plotting (matplotlib) — bar plot of mean Jaccard per axis at B=512
# ----------------------------------------------------------------------


def _plot_b512(rows: List[dict], out_path: Path, executor: str):
    import matplotlib.pyplot as plt
    import numpy as np

    by_axis_token: Dict[str, List[float]] = {"strategy": [], "task": [], "role": []}
    by_axis_unit:  Dict[str, List[float]] = {"strategy": [], "task": [], "role": []}
    for r in rows:
        if r["budget"] != 512 or r["axis"] not in by_axis_token:
            continue
        by_axis_token[r["axis"]].append(r["jaccard_token"])
        by_axis_unit[r["axis"]].append(r["jaccard_unit"])

    axis_labels = ["strategy", "task", "role"]
    means_t = [
        sum(by_axis_token[a]) / max(len(by_axis_token[a]), 1) for a in axis_labels
    ]
    means_u = [
        sum(by_axis_unit[a]) / max(len(by_axis_unit[a]), 1) for a in axis_labels
    ]
    n_pairs = [len(by_axis_token[a]) for a in axis_labels]

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    stds_t = [_std(by_axis_token[a]) for a in axis_labels]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(axis_labels))
    width = 0.36
    bars_t = ax.bar(
        x - width / 2, means_t, width=width, yerr=stds_t, capsize=3,
        color=["#1f77b4"] * 3, edgecolor="black", label="Entity-token Jaccard",
    )
    bars_u = ax.bar(
        x + width / 2, means_u, width=width,
        color=["#ff7f0e"] * 3, edgecolor="black", label="Unit-text Jaccard",
    )
    for i, (mt, mu, n) in enumerate(zip(means_t, means_u, n_pairs)):
        ax.text(i - width / 2, mt + 0.015, f"{mt:.3f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, mu + 0.015, f"{mu:.3f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i, -0.07, f"n={n}", ha="center", va="top", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(varied)" for a in axis_labels])
    ax.set_ylim(-0.10, max(1.0, max(means_t + means_u) * 1.30))
    ax.set_ylabel("Pair-wise Jaccard at B=512")
    ax.set_title(f"Three-level memory divergence hierarchy — {executor}")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["direct", "verify", "explore"])
    parser.add_argument("--main_strategy", default="direct",
                        help="Strategy used for the role and task axes "
                             "(strategy axis uses all strategies).")
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

    print("[A] Building strategy-axis token/unit sets...")
    strat_tok, strat_unit, common_tids_strategy = _strategy_memory_sets(
        args.strategies, args.tag, tuple(args.budgets),
    )
    print(f"[A] strategy axis: {len(common_tids_strategy)} common tasks")

    print("[A] Building role/task-axis token/unit sets (strategy=%s)..." % args.main_strategy)
    role_tok, role_unit, all_tids = _role_memory_sets(
        args.main_strategy, args.tag, tuple(args.budgets),
    )
    print(f"[A] role/task axes: {len(all_tids)} successful tasks at strategy='{args.main_strategy}'")

    rows: List[dict] = []
    for B in args.budgets:
        for cell in _strategy_pairs(strat_tok, strat_unit, common_tids_strategy, args.strategies, B):
            cell["executor"] = args.executor
            rows.append(cell)
        for cell in _role_pairs(role_tok, role_unit, all_tids, B):
            cell["executor"] = args.executor
            rows.append(cell)
        for cell in _task_pairs_per_role(role_tok, role_unit, all_tids, B):
            cell["executor"] = args.executor
            rows.append(cell)

    raw_path = OUTPUTS_DIR / "hierarchy_raw.jsonl"
    n_raw = write_jsonl(raw_path, rows)
    print(f"[A] wrote {n_raw} raw pair rows -> {raw_path}")

    summary = _aggregate(rows)
    sum_path = OUTPUTS_DIR / "hierarchy_summary.csv"
    n_sum = write_csv(sum_path, summary)
    print(f"[A] wrote {n_sum} summary rows -> {sum_path}")

    print()
    print("=== Three-level hierarchy (mean entity-token Jaccard, executor=%s) ===" % args.executor)
    print(f"{'budget':>6} | {'strategy':>20} | {'task':>20} | {'role':>20}")
    print("-" * 80)
    by_axis_budget = {(r["axis"], r["budget"]): r for r in summary}
    for B in args.budgets:
        s = by_axis_budget.get(("strategy", B), {})
        t = by_axis_budget.get(("task", B), {})
        r = by_axis_budget.get(("role", B), {})
        def _f(d):
            if not d: return "n/a"
            return (f"{d['jaccard_token_mean']:.3f}±{d['jaccard_token_std']:.2f} (n={d['n_pairs']})")
        print(f"{B:>6} | {_f(s):>20} | {_f(t):>20} | {_f(r):>20}")

    print()
    print("=== Same table — unit-text Jaccard ===")
    print(f"{'budget':>6} | {'strategy':>20} | {'task':>20} | {'role':>20}")
    print("-" * 80)
    for B in args.budgets:
        s = by_axis_budget.get(("strategy", B), {})
        t = by_axis_budget.get(("task", B), {})
        r = by_axis_budget.get(("role", B), {})
        def _f(d):
            if not d: return "n/a"
            return (f"{d['jaccard_unit_mean']:.3f}±{d['jaccard_unit_std']:.2f} (n={d['n_pairs']})")
        print(f"{B:>6} | {_f(s):>20} | {_f(t):>20} | {_f(r):>20}")

    _plot_b512([{**r, "executor": args.executor} for r in rows],
               FIGURES_DIR / "hierarchy_b512.pdf",
               args.executor)
    _plot_b512([{**r, "executor": args.executor} for r in rows],
               FIGURES_DIR / "hierarchy_by_executor.pdf",
               args.executor)

    print()
    print("[A] Done.")


if __name__ == "__main__":
    main()
