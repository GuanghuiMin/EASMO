"""Did the strategy directives actually change agent behaviour?

Compares trajectories across strategies on the same task set and reports
the behavioural metrics defined in ``prompts/STRATEGY_DESIGN.md``:

    | metric                                    | direct   | verify   | explore   |
    | num_interactions median                   | low      | high     | medium    |
    | first-3-step show_app_descriptions calls  | rare     | rare     | very common |
    | unique GT API endpoints touched           | small    | medium   | large     |
    | duplicate-fact retrieval (same fact via 2 endpoints) | none | always | sometimes |

This script is a *manipulation check*: if these patterns don't appear,
the strategy injection didn't take and the downstream M1 / M2 / M3
results aren't measuring what we think they are.

Usage:
    /workspace/acon/.venv/bin/python motivation_v2/scripts/manipulation_check.py
        --tag motivation_v2_pilot
        [--strategies direct verify explore]
        [--split train]

It scans
    /workspace/acon/experiments/appworld/outputs/MiniMaxAI_MiniMax-M2.5_<tag>_<strategy>/<split>/task_*
for each requested strategy and reports the comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


_API_RE = re.compile(r"apis\.(\w+)\.(\w+)")
_OUTPUTS_ROOT = Path("/workspace/acon/experiments/appworld/outputs")


def _api_calls(traj: dict) -> List[str]:
    out: List[str] = []
    for s in traj.get("trajectory") or []:
        for m in _API_RE.finditer(s.get("action") or ""):
            out.append(f"{m.group(1)}.{m.group(2)}")
    return out


def _first_n_apis(traj: dict, n: int) -> List[str]:
    out: List[str] = []
    for s in (traj.get("trajectory") or [])[:n]:
        for m in _API_RE.finditer(s.get("action") or ""):
            out.append(f"{m.group(1)}.{m.group(2)}")
    return out


def _per_task_stats(task_dir: Path) -> Optional[dict]:
    res_path = task_dir / "results.json"
    traj_path = task_dir / "appworld_trajectory.json"
    if not (res_path.exists() and traj_path.exists()):
        return None
    with open(res_path) as f:
        res = json.load(f)
    with open(traj_path) as f:
        traj = json.load(f)

    apis = _api_calls(traj)
    unique_apis = set(apis)
    first3 = _first_n_apis(traj, 3)

    # "duplicate-fact retrieval" proxy: count instances where the same
    # spotify song / record id appears in BOTH a `show_song` and another
    # endpoint (`show_album`, `show_playlist`, …) with the same id.
    # Cheap heuristic that's nonzero on verify-style trajectories.
    show_song_calls = sum(1 for a in apis if a == "spotify.show_song")
    cross_endpoint_calls = sum(
        1 for a in apis
        if a in {"spotify.show_album", "spotify.show_playlist"}
    )
    duplicate_proxy = min(show_song_calls, cross_endpoint_calls)

    return {
        "task_id": str(task_dir.name).replace("task_", ""),
        "success": bool(res.get("success", False)),
        "iters": int(res.get("iterations", 0)),
        "n_api_calls": len(apis),
        "n_unique_apis": len(unique_apis),
        "show_app_descriptions_in_first3": int(
            "api_docs.show_app_descriptions" in first3
        ),
        "show_song_calls": show_song_calls,
        "cross_endpoint_calls": cross_endpoint_calls,
        "duplicate_fact_proxy": duplicate_proxy,
        "input_tokens": int(
            (res.get("token_usage") or {}).get("total_input_tokens", 0)
        ),
    }


def _aggregate_strategy(
    tag: str, strategy: str, split: str,
) -> Dict[str, object]:
    exp_dir = _OUTPUTS_ROOT / f"MiniMaxAI_MiniMax-M2.5_{tag}_{strategy}" / split
    if not exp_dir.is_dir():
        return {
            "strategy": strategy,
            "exp_dir": str(exp_dir),
            "n_tasks": 0,
            "missing": True,
        }
    rows: List[dict] = []
    for task_dir in sorted(exp_dir.glob("task_*")):
        st = _per_task_stats(task_dir)
        if st is None:
            continue
        rows.append(st)
    if not rows:
        return {
            "strategy": strategy,
            "exp_dir": str(exp_dir),
            "n_tasks": 0,
            "missing": True,
        }

    def _med(key: str) -> float:
        return float(statistics.median([r[key] for r in rows]))

    def _mean(key: str) -> float:
        return float(sum(r[key] for r in rows) / len(rows))

    return {
        "strategy": strategy,
        "exp_dir": str(exp_dir),
        "n_tasks": len(rows),
        "n_success": sum(1 for r in rows if r["success"]),
        "success_rate": sum(1 for r in rows if r["success"]) / len(rows),
        "iters_median": _med("iters"),
        "iters_mean": _mean("iters"),
        "n_api_calls_median": _med("n_api_calls"),
        "n_unique_apis_mean": _mean("n_unique_apis"),
        "show_app_descriptions_first3_rate": _mean(
            "show_app_descriptions_in_first3"
        ),
        "show_song_calls_mean": _mean("show_song_calls"),
        "duplicate_fact_proxy_mean": _mean("duplicate_fact_proxy"),
        "input_tokens_mean": _mean("input_tokens"),
        "missing": False,
        "_rows": rows,  # kept for debug; not printed
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--strategies", nargs="+",
                        default=["direct", "verify", "explore"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--also_baseline_tag", default=None,
                        help="optional tag of a no-strategy baseline run for comparison")
    args = parser.parse_args()

    aggregates: List[Dict[str, object]] = []
    for s in args.strategies:
        aggregates.append(_aggregate_strategy(args.tag, s, args.split))

    if args.also_baseline_tag:
        # Baseline runs have no strategy suffix; we use the empty string.
        # The launcher's experiment_name is f"...{tag}_{strategy}", so
        # baseline (run via acon's run_all.py with bare tag) has the
        # form MiniMaxAI_MiniMax-M2.5_<tag>/<split>. Approximate this by
        # constructing the expected dir directly.
        bdir = _OUTPUTS_ROOT / f"MiniMaxAI_MiniMax-M2.5_{args.also_baseline_tag}" / args.split
        if bdir.is_dir():
            rows: List[dict] = []
            for td in sorted(bdir.glob("task_*")):
                st = _per_task_stats(td)
                if st is not None:
                    rows.append(st)
            if rows:
                aggregates.insert(0, {
                    "strategy": "(baseline)",
                    "exp_dir": str(bdir),
                    "n_tasks": len(rows),
                    "n_success": sum(1 for r in rows if r["success"]),
                    "success_rate": sum(1 for r in rows if r["success"]) / len(rows),
                    "iters_median": float(statistics.median([r["iters"] for r in rows])),
                    "iters_mean": sum(r["iters"] for r in rows) / len(rows),
                    "n_api_calls_median": float(statistics.median([r["n_api_calls"] for r in rows])),
                    "n_unique_apis_mean": sum(r["n_unique_apis"] for r in rows) / len(rows),
                    "show_app_descriptions_first3_rate":
                        sum(r["show_app_descriptions_in_first3"] for r in rows) / len(rows),
                    "show_song_calls_mean": sum(r["show_song_calls"] for r in rows) / len(rows),
                    "duplicate_fact_proxy_mean": sum(r["duplicate_fact_proxy"] for r in rows) / len(rows),
                    "input_tokens_mean": sum(r["input_tokens"] for r in rows) / len(rows),
                    "missing": False,
                    "_rows": rows,
                })

    cols = [
        "strategy", "n_tasks", "n_success", "success_rate",
        "iters_median", "iters_mean",
        "n_api_calls_median", "n_unique_apis_mean",
        "show_app_descriptions_first3_rate",
        "show_song_calls_mean", "duplicate_fact_proxy_mean",
        "input_tokens_mean",
    ]
    headers = [
        "strat", "n", "ok", "ok%",
        "iters_med", "iters_mean",
        "calls_med", "uniq_apis",
        "first3_explore",
        "song_calls", "dup_proxy",
        "in_tokens",
    ]

    fmt_row = "  {:<10s}  " + "  ".join(["{:>10}" for _ in headers[1:]])
    print(fmt_row.format(*headers))
    print("  " + "-" * 130)
    for a in aggregates:
        if a["missing"]:
            print(f"  {a['strategy']:<10s}  (no tasks found at {a['exp_dir']})")
            continue
        cells = [a["strategy"]]
        for c in cols[1:]:
            v = a.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        print(fmt_row.format(*cells))

    print()
    print("Manipulation-check verdict (rough):")
    if len(aggregates) >= 2:
        d = next((a for a in aggregates if a["strategy"] == "direct"), None)
        v = next((a for a in aggregates if a["strategy"] == "verify"), None)
        e = next((a for a in aggregates if a["strategy"] == "explore"), None)
        if d and v and not d["missing"] and not v["missing"]:
            iter_ratio = float(v["iters_median"]) / max(float(d["iters_median"]), 1.0)
            if iter_ratio >= 1.5:
                print(f"  ✓ verify median iters / direct median iters = {iter_ratio:.2f}× — strategies differ")
            else:
                print(f"  ⚠ verify / direct iter ratio = {iter_ratio:.2f}× — too small; strategies barely differ")
        if e and not e["missing"]:
            rate = float(e["show_app_descriptions_first3_rate"])
            if rate >= 0.5:
                print(f"  ✓ explore: show_app_descriptions in first 3 steps {100*rate:.0f}% — exploring is real")
            else:
                print(f"  ⚠ explore: show_app_descriptions in first 3 steps {100*rate:.0f}% — strategy not adhered")


if __name__ == "__main__":
    main()
