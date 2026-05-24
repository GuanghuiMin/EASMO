"""Budget-regime test — Path-C decision support.

Reads conditional drop per budget from one or more transfer_results.csv,
filters out memories contaminated by the truncated-<think> bug, computes
bootstrap CIs per budget, runs a quadratic fit in log(B), and prints a
verdict on whether the "B=256 valley" generalises beyond LongMemEval.

Usage:
    python -m scripts.budget_regime_test \\
        --runs outputs/default_longmemeval outputs/wide_longmemeval \\
               outputs/default_locomo outputs/wide_locomo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

random.seed(42)


def bootstrap_ci(values, n_iter=5000):
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    means = []
    for _ in range(n_iter):
        means.append(sum(values[random.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return statistics.mean(values), means[int(0.025 * n_iter)], means[int(0.975 * n_iter)]


def contaminated_keys(run_dir):
    """(ctx_id, agent_id, budget) tuples whose oracle memory_text starts with <think>."""
    path = Path(run_dir) / "oracle_memories.jsonl"
    if not path.exists():
        return set()
    bad = set()
    for line in open(path):
        r = json.loads(line)
        if r.get("memory_text", "").lstrip().lower().startswith("<think>"):
            bad.add((r["context_id"], r["agent_id"], int(r["budget"])))
    return bad


def load_clean_per_budget(run_dirs):
    """Merge per-budget clean conditional-drop values across runs."""
    by_b = defaultdict(list)
    for d in run_dirs:
        d = Path(d)
        csv_path = d / "transfer_results.csv"
        if not csv_path.exists():
            print(f"[warn] no transfer_results.csv in {d}", file=sys.stderr)
            continue
        bad = contaminated_keys(d)
        for r in csv.DictReader(open(csv_path)):
            ctx, b = r["context_id"], int(r["budget"])
            src, tgt = r["source_agent"], r["target_agent"]
            if (ctx, src, b) in bad or (ctx, tgt, b) in bad:
                continue
            if float(r["action_match_self"]) >= 0.999:
                by_b[b].append(1.0 - float(r["action_match_cross"]))
    return by_b


def quadratic_fit_log(budgets, means):
    """Fit y = a + b·log(B) + c·log(B)² and return (a, b, c, R²,
    argmin_B). argmin_B is the budget at which the quadratic minimum
    sits (None if no real minimum)."""
    if len(budgets) < 3:
        return None
    xs = [math.log2(b) for b in budgets]
    ys = list(means)
    n = len(xs)
    sum_x = sum(xs); sum_x2 = sum(x ** 2 for x in xs); sum_x3 = sum(x ** 3 for x in xs)
    sum_x4 = sum(x ** 4 for x in xs)
    sum_y = sum(ys); sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2y = sum((x ** 2) * y for x, y in zip(xs, ys))

    # Solve normal equations for y = a + b*x + c*x^2.
    import numpy as np
    X = np.array([[1, x, x ** 2] for x in xs])
    y = np.array(ys)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b_, c = coeffs.tolist()
    ypred = X @ coeffs
    ss_res = float(np.sum((y - ypred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    argmin = None
    if c > 0:                          # parabola opens up; has min
        x_min = -b_ / (2 * c)
        argmin = 2 ** x_min
    return {"a": a, "b": b_, "c": c, "r2": r2, "argmin_B": argmin}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="one or more output dirs to merge per dataset")
    parser.add_argument("--name", default="merged",
                        help="display name for the merged dataset")
    args = parser.parse_args()

    by_b = load_clean_per_budget(args.runs)
    if not by_b:
        print("No data found.", file=sys.stderr)
        sys.exit(2)

    print(f"\n===== Budget-regime test on '{args.name}' =====")
    print(f"Source runs: {args.runs}")
    print(f"\n{'B':>6}  {'n':>4}  {'mean':>5}  {'95% CI':>15}")
    budgets, means, los, his = [], [], [], []
    for b in sorted(by_b):
        vs = by_b[b]
        m, lo, hi = bootstrap_ci(vs)
        budgets.append(b); means.append(m); los.append(lo); his.append(hi)
        print(f"{b:>6d}  {len(vs):>4d}  {m:>5.2f}  [{lo:.2f}, {hi:.2f}]")

    fit = quadratic_fit_log(budgets, means)
    if fit is None:
        print("\nNot enough budgets for quadratic fit (need ≥3).")
        sys.exit(0)
    print(f"\nQuadratic fit in log2(B):  y = {fit['a']:+.3f} + {fit['b']:+.3f}·log2(B) + {fit['c']:+.3f}·log2(B)²")
    print(f"  R²:        {fit['r2']:.3f}")
    print(f"  c sign:    {'positive (bowl-shaped)' if fit['c'] > 0 else 'negative (n-shaped)'}")
    if fit["argmin_B"] is not None:
        print(f"  argmin B:  {fit['argmin_B']:.0f}")
    else:
        print(f"  argmin B:  N/A (no minimum)")

    # Path-C decision
    print("\n--- Path-C decision criteria ---")
    has_min = fit["c"] > 0 and fit["argmin_B"] is not None and 200 < fit["argmin_B"] < 2000
    r2_ok = fit["r2"] >= 0.70
    print(f"  Bowl-shaped (c > 0):                 {fit['c'] > 0}")
    print(f"  Valley in plausible range (200-2000): {has_min}")
    print(f"  R² ≥ 0.7:                            {r2_ok}")
    if has_min and r2_ok:
        print("\n  → Path-C SUPPORTED. Reframe paper around budget-regime emergence.")
    elif has_min:
        print("\n  → Path-C partially supported. Quadratic fits a valley but explains <70% variance; treat as a finding, not the main framing.")
    elif fit["c"] < 0:
        print("\n  → Path-C REJECTED — pattern is n-shaped, not bowl-shaped (monotonic-like).")
    else:
        print("\n  → Path-C inconclusive. Saturation rather than valley; use Path D.")


if __name__ == "__main__":
    main()
