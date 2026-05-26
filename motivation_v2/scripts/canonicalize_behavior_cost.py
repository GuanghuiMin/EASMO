"""Experiment C — Behavioral cost of wrong memory (canonical output).

Joins the existing transfer_results.jsonl files (cap=50 baseline and
the capped variants) with post-hoc API-call metrics extracted from
the corresponding env_history.json trajectories.

Spec reference: experiment_modification.md §7 (Experiment C).

Outputs:

    outputs/motivation/behavior_cost_raw.jsonl
    outputs/motivation/behavior_cost_summary.csv
    figures/motivation/behavior_success_cap15.pdf  (Sprint 2 finalises)
    figures/motivation/behavior_cost_tokens_iters.pdf (Sprint 2 finalises)

Each raw row carries the spec's RunResult schema (§3.3) plus the
canonical condition labels:

    matched              (was: self)
    wrong_task_same_gen  (was: within_gen)
    wrong_task_diff_gen  (was: within_app)
    cross_domain         (was: cross_app)
    generic_recent       (added in Sprint 2)
    no_memory            (added in Sprint 2)

Heuristics for API-call metrics (extracted from env_history.json
output strings; see ``_extract_api_metrics``):

  * api_calls_total       : count of `apis.<app>.<fn>(` matches in actions
  * api_calls_unique      : unique (app, fn) tuples
  * show_api_doc_calls    : count of api_docs.{show_api_doc,show_api_descriptions}
  * wrong_endpoint_calls  : count of actions where output contains a Python
                            traceback / explicit Error / failure-message JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.canonical_io import (
    DistribStats,
    OUTPUTS_DIR,
    FIGURES_DIR,
    ensure_dirs,
    log_failure,
    log_run_meta,
    write_csv,
    write_jsonl,
)


_EXEC_DEFAULT = "MiniMaxAI/MiniMax-M2.5"
_API_RE = re.compile(r"apis\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\(", re.DOTALL)
_TRACEBACK_RE = re.compile(r"\bTraceback\b|\bException\b|^\s*\{[\s\S]*?\"message\"\s*:")

# Map raw condition (used in transfer_results.jsonl) → canonical spec name.
_CONDITION_REMAP = {
    "self":       "matched",
    "within_gen": "wrong_task_same_gen",
    "within_app": "wrong_task_diff_gen",
    "cross_app":  "cross_domain",
    "generic_recent": "generic_recent",
    "no_memory":  "no_memory",
}

# Cap → tag mapping for trajectory directory lookup.
_TAG_BY_CAP = {
    50: "mv2_xtask",
    15: "mv2_xtask_cap15",
    8:  "mv2_xtask_cap8",
}


# ----------------------------------------------------------------------
# Heuristic API-call metric extractor
# ----------------------------------------------------------------------

def _extract_api_metrics(env_history_path: Path) -> Dict[str, int]:
    """Returns {api_calls_total, api_calls_unique, wrong_endpoint_calls,
    show_api_doc_calls}. All zeros if the file can't be loaded."""
    out = {
        "api_calls_total": 0,
        "api_calls_unique": 0,
        "wrong_endpoint_calls": 0,
        "show_api_doc_calls": 0,
    }
    if not env_history_path.exists():
        return out
    try:
        steps = json.loads(env_history_path.read_text())
    except Exception:
        return out

    seen = set()
    for s in steps:
        action = s.get("action") or ""
        output = s.get("output") or ""
        # Accumulate API calls
        for m in _API_RE.finditer(action):
            app, fn = m.group(1), m.group(2)
            out["api_calls_total"] += 1
            seen.add((app, fn))
            if app == "api_docs" and fn in ("show_api_doc", "show_api_descriptions"):
                out["show_api_doc_calls"] += 1
        # Heuristic wrong-endpoint detection: action made an api call AND
        # output looks like a Python error/traceback or a failure-message JSON.
        if _API_RE.search(action) and _TRACEBACK_RE.search(output):
            out["wrong_endpoint_calls"] += 1
    out["api_calls_unique"] = len(seen)
    return out


def _trajectory_dir(
    *,
    consumer_task: str,
    condition: str,
    source_task: str,
    budget: int,
    tag: str,
    strategy: str = "direct",
) -> Path:
    """Resolve the appworld trajectory directory for a given cell.

    Convention: <model>_<tag>_<strategy>_xtask_<condition>_from_<source>_B<budget>/train/task_<consumer>
    For the new conditions (generic_recent, no_memory), source_task is
    a placeholder (e.g. "none"), but the directory still follows the
    same convention (we'll wire that up in Sprint 2).
    """
    name = (
        f"MiniMaxAI_MiniMax-M2.5_{tag}_{strategy}_xtask_"
        f"{condition}_from_{source_task}_B{budget}"
    )
    return Path(
        f"/workspace/acon/experiments/appworld/outputs/{name}/train/task_{consumer_task}"
    )


# ----------------------------------------------------------------------
# Convert one transfer_results.jsonl row → canonical RunResult
# ----------------------------------------------------------------------

def _to_run_result(row: dict, max_iter: int, tag: str) -> dict:
    consumer = row["consumer_task_id"]
    raw_cond = row["condition"]
    source = row.get("source_task_id") or ""
    budget = row["budget"]
    cond_canonical = _CONDITION_REMAP.get(raw_cond, raw_cond)

    api_metrics = {
        "api_calls_total": 0,
        "api_calls_unique": 0,
        "wrong_endpoint_calls": 0,
        "show_api_doc_calls": 0,
    }
    if source:
        traj_dir = _trajectory_dir(
            consumer_task=consumer,
            condition=raw_cond,
            source_task=source,
            budget=budget,
            tag=tag,
        )
        env_path = traj_dir / "env_history.json"
        api_metrics = _extract_api_metrics(env_path)

    run_id = (
        f"{tag}_{cond_canonical}_from_{source or 'none'}"
        f"_B{budget}_cap{max_iter}_t{consumer}"
    )

    return {
        "run_id": run_id,
        "task_id": consumer,
        "executor": _EXEC_DEFAULT,
        "benchmark": "appworld",
        "memory_condition": cond_canonical,
        "memory_condition_raw": raw_cond,
        "source_task_id": source,
        "memory_id": f"m_exec_minimal:{source or 'none'}:B{budget}",
        "budget": budget,
        "max_iter": max_iter,
        "success": bool(row.get("success", False)),
        "final_reward": float(row.get("final_reward", 0.0)),
        "iterations": int(row.get("iterations", 0)),
        "input_tokens": int(row.get("input_tokens", 0)),
        "elapsed_s": float(row.get("elapsed_s", 0.0)),
        "memory_text_len": int(row.get("memory_text_len", 0)),
        "termination_reason": row.get("termination_reason", "?"),
        "error": row.get("error"),
        **api_metrics,
        "notes": "",
    }


# ----------------------------------------------------------------------
# Aggregator: spec-required summary metrics
# ----------------------------------------------------------------------

def _aggregate(rows: List[dict]) -> List[dict]:
    """Aggregate by (memory_condition, budget, max_iter), and compute
    efficiency_tax_iters / efficiency_tax_tokens / capability_drop
    against matched within the same (budget, max_iter) cell.

    Special case: ``no_memory`` lives at budget=0 (memory is empty,
    budget axis is meaningless). We compare it against matched at
    B=512 (the main budget) for the cap_drop / eff_tax columns so
    the row isn't a NaN island.
    """
    by_cbm: Dict[Tuple[str, int, int], List[dict]] = defaultdict(list)
    for r in rows:
        by_cbm[(r["memory_condition"], r["budget"], r["max_iter"])].append(r)

    matched_means: Dict[Tuple[int, int], dict] = {}
    for (cond, B, cap), xs in by_cbm.items():
        if cond != "matched":
            continue
        if not xs:
            continue
        n = len(xs)
        matched_means[(B, cap)] = {
            "iters": sum(r["iterations"] for r in xs) / n,
            "tokens": sum(r["input_tokens"] for r in xs) / n,
            "success": sum(1 for r in xs if r["success"]) / n,
        }

    out: List[dict] = []
    for (cond, B, cap), xs in sorted(by_cbm.items()):
        n = len(xs)
        success = sum(1 for r in xs if r["success"]) / n if n else 0.0
        iters_s = DistribStats.from_values([r["iterations"] for r in xs])
        toks_s = DistribStats.from_values([r["input_tokens"] for r in xs])
        api_total_s = DistribStats.from_values([r["api_calls_total"] for r in xs])
        api_unique_s = DistribStats.from_values([r["api_calls_unique"] for r in xs])
        wrong_endpoint_s = DistribStats.from_values([r["wrong_endpoint_calls"] for r in xs])
        show_doc_s = DistribStats.from_values([r["show_api_doc_calls"] for r in xs])

        # Compare against matched at the same (B, cap) when possible;
        # for no_memory (B=0) compare against matched at B=512.
        matched_ref = matched_means.get((B, cap))
        if matched_ref is None and cond == "no_memory":
            matched_ref = matched_means.get((512, cap))
        eff_iters = (iters_s.mean - matched_ref["iters"]) if matched_ref else None
        eff_tokens = (toks_s.mean - matched_ref["tokens"]) if matched_ref else None
        capability_drop = (matched_ref["success"] - success) if matched_ref else None

        out.append({
            "experiment": "C_behavior_cost",
            "executor": _EXEC_DEFAULT,
            "memory_condition": cond,
            "budget": B,
            "max_iter": cap,
            "n_runs": n,
            "success_rate": round(success, 4),
            "n_success": int(success * n),
            "iters_mean":   round(iters_s.mean, 2),
            "iters_std":    round(iters_s.std, 2),
            "iters_median": round(iters_s.median, 2),
            "iters_min":    round(iters_s.min_, 2),
            "iters_max":    round(iters_s.max_, 2),
            "tokens_mean":   round(toks_s.mean, 0),
            "tokens_std":    round(toks_s.std, 0),
            "tokens_median": round(toks_s.median, 0),
            "api_calls_total_mean":   round(api_total_s.mean, 2),
            "api_calls_unique_mean":  round(api_unique_s.mean, 2),
            "wrong_endpoint_calls_mean": round(wrong_endpoint_s.mean, 3),
            "show_api_doc_calls_mean":   round(show_doc_s.mean, 2),
            "efficiency_tax_iters":   None if eff_iters is None else round(eff_iters, 2),
            "efficiency_tax_tokens":  None if eff_tokens is None else round(eff_tokens, 0),
            "capability_drop":        None if capability_drop is None else round(capability_drop, 4),
        })
    return out


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_success_at_cap15(summary: List[dict], out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    # Budget-axis conditions (B=128, 256, 512 columns)
    budgeted = ["matched", "wrong_task_same_gen", "wrong_task_diff_gen",
                "cross_domain", "generic_recent"]
    budgets = [128, 256, 512]
    by_cb = {(r["memory_condition"], r["budget"]): r for r in summary
             if r["max_iter"] == 15}
    no_mem = next((r for r in summary
                   if r["max_iter"] == 15 and r["memory_condition"] == "no_memory"),
                  None)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    palette = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    width = 0.16
    n_groups = len(budgets) + (1 if no_mem else 0)
    x_budgets = np.arange(len(budgets))
    plotted = 0
    for i, cond in enumerate(budgeted):
        ys, ns = [], []
        for B in budgets:
            r = by_cb.get((cond, B))
            ys.append(r["success_rate"] if r else 0.0)
            ns.append(r["n_runs"] if r else 0)
        if all(n == 0 for n in ns):
            continue
        ax.bar(x_budgets + (plotted - len(budgeted) / 2) * width, ys,
               width=width, label=f"{cond} (n={max(ns)})",
               color=palette[i % len(palette)], edgecolor="black")
        plotted += 1
    if no_mem:
        x_no = np.array([len(budgets) + 0.5])
        ax.bar(x_no, [no_mem["success_rate"]], width=width * len(budgeted) * 0.7,
               label=f"no_memory (B=0, n={no_mem['n_runs']})",
               color="#7f7f7f", edgecolor="black")

    xticks = list(x_budgets) + ([float(len(budgets) + 0.5)] if no_mem else [])
    xtick_labels = [f"B={B}" for B in budgets] + (["B=0\n(no memory)"] if no_mem else [])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Behavioral cost under bounded inference (max_iter=15)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _plot_iters_tokens_at_cap50(summary: List[dict], out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    cond_order = ["matched", "wrong_task_same_gen", "wrong_task_diff_gen",
                  "cross_domain", "generic_recent", "no_memory"]
    by_cb = {(r["memory_condition"], r["budget"]): r for r in summary
             if r["max_iter"] == 50}
    budgets = sorted({r["budget"] for r in summary if r["max_iter"] == 50})
    if not budgets:
        print(f"[plot] no max_iter=50 data; skipping {out_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    palette = [
        "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#7f7f7f",
    ]
    for ax, metric, ylabel in [
        (axes[0], "iters_mean", "Mean iterations"),
        (axes[1], "tokens_mean", "Mean input tokens"),
    ]:
        plotted = 0
        for i, cond in enumerate(cond_order):
            ys = []
            ns = []
            for B in budgets:
                r = by_cb.get((cond, B))
                ys.append(r[metric] if r else 0.0)
                ns.append(r["n_runs"] if r else 0)
            if all(n == 0 for n in ns):
                continue
            ax.plot(budgets, ys, marker="o", color=palette[i % len(palette)],
                    label=f"{cond} (n={max(ns)})")
            plotted += 1
        ax.set_xlabel("Budget B (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(B) for B in budgets])
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("Behavioral cost — iterations & input tokens (max_iter=50)")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _load_one_cap(path: Path, max_iter: int, tag: str) -> List[dict]:
    if not path.exists():
        print(f"[C] missing: {path}")
        return []
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(_to_run_result(r, max_iter=max_iter, tag=tag))
    print(f"[C] loaded {len(rows):>3d} rows from {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_xtask/transfer_results.jsonl")
    parser.add_argument("--cap15_path",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_xtask_cap15/transfer_results.jsonl")
    parser.add_argument("--cap8_path",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_xtask_cap8/transfer_results.jsonl")
    parser.add_argument("--extra_jsonl", action="append", default=[],
                        help="Additional transfer_results.jsonl files "
                             "(used to fold in Sprint-2 generic_recent / no_memory cells). "
                             "Format: cap=<int>:<tag>:<path>")
    args = parser.parse_args()

    ensure_dirs()
    log_run_meta("C_behavior_cost", executor=_EXEC_DEFAULT, seed=42)

    rows: List[dict] = []
    rows += _load_one_cap(Path(args.baseline_path), max_iter=50, tag="mv2_xtask")
    rows += _load_one_cap(Path(args.cap15_path), max_iter=15, tag="mv2_xtask_cap15")
    rows += _load_one_cap(Path(args.cap8_path),  max_iter=8,  tag="mv2_xtask_cap8")

    for spec in args.extra_jsonl:
        cap_str, tag, path_str = spec.split(":", 2)
        cap = int(cap_str.split("=", 1)[1])
        rows += _load_one_cap(Path(path_str), max_iter=cap, tag=tag)

    raw_path = OUTPUTS_DIR / "behavior_cost_raw.jsonl"
    n_raw = write_jsonl(raw_path, rows)
    print(f"[C] wrote {n_raw} raw RunResult rows -> {raw_path}")

    summary = _aggregate(rows)
    sum_path = OUTPUTS_DIR / "behavior_cost_summary.csv"
    n_sum = write_csv(sum_path, summary)
    print(f"[C] wrote {n_sum} summary rows -> {sum_path}")

    print()
    print("=== Behavioral cost summary (B=512 main; no_memory at B=0) ===")
    print(f"{'condition':>22} | {'cap':>4} | {'n':>3} | "
          f"{'success%':>8} | {'iters':>6} | {'tokens':>7} | "
          f"{'wrong_ep':>8} | {'doc_calls':>9} | {'eff_tax_it':>10} | "
          f"{'cap_drop':>8}")
    print("-" * 110)
    # Show B=512 rows for normal conditions and B=0 for no_memory.
    def _keep(r):
        if r["memory_condition"] == "no_memory":
            return r["budget"] == 0
        return r["budget"] == 512
    for r in summary:
        if not _keep(r):
            continue
        eff = "" if r["efficiency_tax_iters"] is None else f"{r['efficiency_tax_iters']:+.1f}"
        cap_drop = "" if r["capability_drop"] is None else f"{r['capability_drop']*100:+.0f}pp"
        print(f"{r['memory_condition']:>22} | {r['max_iter']:>4} | {r['n_runs']:>3} | "
              f"{r['success_rate']*100:>7.0f}% | {r['iters_mean']:>6.1f} | "
              f"{r['tokens_mean']:>7.0f} | {r['wrong_endpoint_calls_mean']:>8.2f} | "
              f"{r['show_api_doc_calls_mean']:>9.1f} | "
              f"{eff:>10} | {cap_drop:>8}")

    _plot_success_at_cap15(summary, FIGURES_DIR / "behavior_success_cap15.pdf")
    _plot_iters_tokens_at_cap50(summary, FIGURES_DIR / "behavior_cost_tokens_iters.pdf")

    print()
    print("[C] Done.")


if __name__ == "__main__":
    main()
