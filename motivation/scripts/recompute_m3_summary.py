"""Re-derive m3_summary.json from an existing transfer_results.csv.

Use when you've just upgraded the summary schema (e.g. added conditional
drop) and don't want to re-run M3 with LLM calls. Reads
``outputs/<exp>/transfer_results.csv`` and writes a fresh
``m3_summary.json`` next to it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.metrics import linear_fit_r2, spearman
from motivation.utils import write_json, setup_logging

_logger = setup_logging("scripts.recompute_m3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to transfer_results.csv")
    parser.add_argument("--out", default=None,
                        help="output m3_summary.json (default: same dir as csv)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "m3_summary.json"

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "context_id":         r["context_id"],
                "budget":             int(r["budget"]),
                "source_agent":       r["source_agent"],
                "target_agent":       r["target_agent"],
                "action_match_self":  float(r["action_match_self"]),
                "action_match_cross": float(r["action_match_cross"]),
                "task_drop":          float(r["task_drop"]),
                "policy_divergence":  float(r["policy_divergence"]),
            })
    _logger.info("Loaded %d transfer rows from %s", len(rows), csv_path)

    drops = [r["task_drop"] for r in rows]
    divs  = [r["policy_divergence"] for r in rows]

    signal_rows = [r for r in rows if r["action_match_self"] >= 0.999]
    cond_drops = [1.0 - r["action_match_cross"] for r in signal_rows]
    cond_divs  = [r["policy_divergence"]       for r in signal_rows]
    mean_cond_drop = sum(cond_drops) / len(cond_drops) if cond_drops else 0.0

    summary = {
        "m3_total_runs": len(rows),

        "m3_mean_task_drop": sum(drops) / len(drops) if drops else 0.0,

        "m3_signal_rows": len(signal_rows),
        "m3_signal_fraction": len(signal_rows) / max(len(rows), 1),
        "m3_mean_conditional_drop": mean_cond_drop,
        "m3_conditional_spearman_drop_vs_div": spearman(cond_divs, cond_drops),
        "m3_conditional_linear_fit_r2": linear_fit_r2(cond_divs, cond_drops),

        "m3_spearman_drop_vs_div": spearman(divs, drops),
        "m3_linear_fit_r2": linear_fit_r2(divs, drops),
    }
    summary["m3_pass_unconditional_drop_15"] = summary["m3_mean_task_drop"] > 0.15
    summary["m3_pass_conditional_drop_15"]   = mean_cond_drop > 0.15
    summary["m3_pass_signal_fraction_10"]    = summary["m3_signal_fraction"] >= 0.10

    by_pair: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_pair[(r["source_agent"], r["target_agent"])].append(r)
    pair_breakdown = []
    for (src, tgt), rs in sorted(by_pair.items()):
        d = [r["task_drop"] for r in rs]
        v = [r["policy_divergence"] for r in rs]
        signal = [r for r in rs if r["action_match_self"] >= 0.999]
        cd = [1.0 - r["action_match_cross"] for r in signal]
        pair_breakdown.append({
            "source_agent": src, "target_agent": tgt,
            "n": len(rs),
            "n_signal": len(signal),
            "mean_task_drop_unconditional": sum(d) / len(d) if d else 0.0,
            "mean_conditional_drop": sum(cd) / len(cd) if cd else 0.0,
            "mean_policy_div": sum(v) / len(v) if v else 0.0,
        })
    summary["m3_pair_breakdown"] = pair_breakdown

    write_json(out_path, summary)
    _logger.info("Wrote %s", out_path)
    # Quick stdout report.
    print(f"\n=== {csv_path} ===")
    print(f"  total rows: {summary['m3_total_runs']}")
    print(f"  signal rows: {summary['m3_signal_rows']} "
          f"({100 * summary['m3_signal_fraction']:.1f}%)")
    print(f"  CONDITIONAL drop: {summary['m3_mean_conditional_drop']:.3f}  "
          f"(pass>15%: {summary['m3_pass_conditional_drop_15']})")
    print(f"  unconditional drop: {summary['m3_mean_task_drop']:.3f}")
    print(f"  conditional R²: {summary['m3_conditional_linear_fit_r2']:.3f}")


if __name__ == "__main__":
    main()
