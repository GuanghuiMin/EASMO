"""Stage 2 — Exp 1: build 3 compressed contexts per selected task.

For each successful trajectory selected in Stage 1, calls MiniMax to
produce three compressed contexts (task-aware NL summary, ACON-style
structured summary, symbolic-evidence JSON unit list).

Outputs:
  outputs/motivation_symbolic_units.jsonl       one row per (task, unit)
  outputs/motivation_compressed_contexts.jsonl  one row per (task, method)

Plus per-method, per-task gold-vs-preserved counters for Table 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _build_one_method(traj, method: str):
    """Worker: build one Compressed for (traj, method)."""
    sys.path.insert(0, str(_REPO))
    from motivation_v3.compressors import (
        COMPRESSOR_REGISTRY, make_client,
    )
    fn = COMPRESSOR_REGISTRY[method]
    client = make_client()
    em = fn(traj, client=client)
    return em.to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--methods", nargs="+",
                        default=["task_aware_summary", "acon_style_summary",
                                 "symbolic_evidence"])
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.data import (
        OUTPUTS, ensure_outputs, jsonl_path, load_trajectory, read_jsonl,
    )
    from motivation_v3.metrics import (
        count_preserved, gold_sizes, mine_gold_items,
    )

    ensure_outputs()
    sel_path = jsonl_path("motivation_full_trajectories.jsonl")
    if not sel_path.exists():
        sys.exit(f"missing {sel_path}; run scripts/01_select_consumers.py first")
    selected = read_jsonl(sel_path)
    print(f"[02] {len(selected)} selected tasks; methods={args.methods}")

    # Pre-load trajectories.
    trajs: Dict[str, "Trajectory"] = {}  # type: ignore  # noqa
    for r in selected:
        td = Path(r["output_dir"])
        if td.exists():
            try:
                trajs[r["task_id"]] = load_trajectory(td)
            except Exception as exc:
                print(f"  skip {r['task_id']} (load failed: {exc})")

    print(f"[02] loaded {len(trajs)}/{len(selected)} trajectories")

    cells: List[tuple] = []
    for tid, traj in trajs.items():
        for m in args.methods:
            cells.append((traj, m))

    print(f"[02] {len(cells)} compression cells; workers={args.workers}")
    print()

    out_compressed = jsonl_path("motivation_compressed_contexts.jsonl")
    out_units = jsonl_path("motivation_symbolic_units.jsonl")

    t0 = time.time()
    n_done = 0
    n_err = 0
    compressed_records: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_build_one_method, t, m): (t, m) for t, m in cells}
        for fut in as_completed(futures):
            traj, method = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                n_err += 1
                rec = {
                    "method": method,
                    "task_id": traj.task_id,
                    "error": str(exc),
                    "text": "",
                    "n_tokens": 0,
                    "n_units": 0,
                    "units": None,
                }
            compressed_records.append(rec)
            n_done += 1
            if n_done % 5 == 0 or n_done == len(cells):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(cells) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>3d}/{len(cells)}] elapsed={elapsed/60:.1f}min  "
                      f"rate={rate*60:.1f}/min  ETA={eta/60:.1f}min  err={n_err}")

    # ------------------------------------------------------------------
    # Compute Exp 1 metrics: gold-vs-preserved per method
    # ------------------------------------------------------------------
    # Mine gold items once per task.
    gold_per_task = {tid: mine_gold_items(t) for tid, t in trajs.items()}

    # Augment each compressed record with metric counts.
    for rec in compressed_records:
        tid = rec.get("task_id")
        text = rec.get("text", "") or ""
        gold = gold_per_task.get(tid)
        if gold is not None and not rec.get("error"):
            rec.update(count_preserved(text, gold))
            rec.update(gold_sizes(gold))

    with open(out_compressed, "w") as f:
        for r in compressed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[02] wrote {len(compressed_records)} rows -> {out_compressed}")

    # Symbolic units side-table (one row per unit, only for symbolic_evidence).
    n_units_total = 0
    with open(out_units, "w") as f:
        for rec in compressed_records:
            if rec.get("method") != "symbolic_evidence" or rec.get("error"):
                continue
            for u in (rec.get("units") or []):
                f.write(json.dumps({
                    "task_id": rec["task_id"],
                    **u,
                }, ensure_ascii=False) + "\n")
                n_units_total += 1
    print(f"[02] wrote {n_units_total} symbolic units -> {out_units}")
    print(f"[02] elapsed: {(time.time()-t0)/60:.1f} min  err={n_err}")


if __name__ == "__main__":
    main()
