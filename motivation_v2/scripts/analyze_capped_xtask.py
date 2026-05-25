"""Analyze capped-budget cross-task transfer results.

Compares two capped runs (e.g. max_iter=15 and max_iter=8) against
the unbounded baseline (max_iter=50, in `outputs/mv2_xtask/`). The
question: does tightening max_iter convert wrong-memory's efficiency
cost (visible in the unbounded run as +40% iter inflation but 100%
success) into a measurable capability cost (success-rate drop)?

Usage:
    python analyze_capped_xtask.py
        [--cap15_path ... --cap8_path ... --baseline_path ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def _success_rate_table(rows, tag):
    """Build {(condition, B): success_rate} dict for one set of rows."""
    by_cb = defaultdict(list)
    for r in rows:
        by_cb[(r["condition"], r["budget"])].append(r["success"])
    print(f"\n=== {tag} (n={len(rows)} cells) ===")
    print(f"  {'condition':>12} | {'B':>4} | {'n':>3} | {'success%':>9} | {'mean iters':>11}")
    print("-" * 70)
    table = {}
    iter_table = {}
    for cond in ["self", "within_gen", "within_app", "cross_app"]:
        for B in [128, 256, 512]:
            xs = by_cb.get((cond, B), [])
            if not xs:
                continue
            n = len(xs)
            ok = sum(1 for x in xs if x)
            iters = [r["iterations"] for r in rows
                     if r["condition"] == cond and r["budget"] == B]
            mean_it = statistics.mean(iters) if iters else 0
            table[(cond, B)] = ok / n
            iter_table[(cond, B)] = mean_it
            print(f"  {cond:>12} | {B:>4} | {n:>3} | "
                  f"{100*ok/n:>8.0f}% | {mean_it:>11.1f}")
        print()
    return table, iter_table


def _diff_table(self_rate, wrong_rate, tag):
    """Print task-success drop from self → wrong-memory conditions."""
    print(f"\n=== Capability drop (success% self − wrong) — {tag} ===")
    print(f"  {'B':>5} | {'within_gen':>11} | {'within_app':>11} | {'cross_app':>11}")
    print("-" * 50)
    for B in [128, 256, 512]:
        s = self_rate.get(("self", B), 0)
        wg = self_rate.get(("within_gen", B), 0)
        wa = self_rate.get(("within_app", B), 0)
        ca = self_rate.get(("cross_app", B), 0)
        print(f"  {B:>5} | "
              f"{100*(s-wg):>+9.0f}pp | "
              f"{100*(s-wa):>+9.0f}pp | "
              f"{100*(s-ca):>+9.0f}pp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_xtask/transfer_results.jsonl")
    parser.add_argument("--cap15_path",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_xtask_cap15/transfer_results.jsonl")
    parser.add_argument("--cap8_path",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_xtask_cap8/transfer_results.jsonl")
    args = parser.parse_args()

    base_rows = _load(args.baseline_path)
    cap15_rows = _load(args.cap15_path)
    cap8_rows = _load(args.cap8_path)

    print("=" * 75)
    print("Capped-budget cross-task transfer — capability cost analysis")
    print("=" * 75)

    baseline_rates, baseline_iters = _success_rate_table(base_rows, "Baseline (max_iter=50)")
    if cap15_rows:
        cap15_rates, _ = _success_rate_table(cap15_rows, "Setup A1 (max_iter=15)")
        _diff_table(cap15_rates, None, "max_iter=15")
    else:
        print("\n(cap15 data not yet available)")

    if cap8_rows:
        cap8_rates, _ = _success_rate_table(cap8_rows, "Setup A2 (max_iter=8)")
        _diff_table(cap8_rates, None, "max_iter=8")
    else:
        print("\n(cap8 data not yet available)")

    # Headline cross-cap comparison
    if cap15_rows and cap8_rows:
        print("\n" + "=" * 75)
        print("=== Cross-cap success rate comparison (B=512) ===")
        print("=" * 75)
        print(f"  {'condition':>12} | {'cap=50 (base)':>13} | {'cap=15':>8} | {'cap=8':>8}")
        print("-" * 60)
        for cond in ["self", "within_gen", "within_app", "cross_app"]:
            b = baseline_rates.get((cond, 512), None)
            c15 = cap15_rates.get((cond, 512), None)
            c8 = cap8_rates.get((cond, 512), None)
            cells = []
            for r in (b, c15, c8):
                cells.append(f"{100*r:>10.0f}%" if r is not None else "    n/a")
            print(f"  {cond:>12} | {cells[0]:>13} | {cells[1]:>8} | {cells[2]:>8}")
        print()
        # Headline metric
        print("Headline differential at B=512 (self vs cross_app):")
        for label, table in [("cap=50", baseline_rates), ("cap=15", cap15_rates), ("cap=8", cap8_rates)]:
            s = table.get(("self", 512))
            ca = table.get(("cross_app", 512))
            if s is None or ca is None:
                continue
            print(f"  {label}: self={100*s:.0f}%, cross_app={100*ca:.0f}%, "
                  f"drop={100*(s-ca):+.0f}pp")


if __name__ == "__main__":
    main()
