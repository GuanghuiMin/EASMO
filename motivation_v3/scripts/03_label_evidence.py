"""Stage 3 — Exp 2.1: label each symbolic unit's behavioral usefulness.

For each (task, symbolic_unit) emitted by Stage 2, asks the LLM whether
removing this unit would force the agent to re-query, increase steps,
cause wrong API arguments, etc. Keeps units with useful=True at
medium/high confidence as 'behavioral evidence'.

Outputs:
  outputs/motivation_behavioral_evidence.jsonl
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


def _label_one(traj, unit: dict):
    sys.path.insert(0, str(_REPO))
    from motivation_v3.evidence import label_unit_usefulness
    lbl = label_unit_usefulness(traj, unit)
    return {"task_id": traj.task_id, "unit": unit, "label": lbl.to_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_units_per_task", type=int, default=40,
                        help="Cap per-task symbolic units to keep cost bounded.")
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.data import (
        OUTPUTS, ensure_outputs, jsonl_path, load_trajectory, read_jsonl,
    )

    ensure_outputs()
    sel = read_jsonl(jsonl_path("motivation_full_trajectories.jsonl"))
    units_path = jsonl_path("motivation_symbolic_units.jsonl")
    if not units_path.exists():
        sys.exit(f"missing {units_path}; run 02_build_compressions.py first")
    units = read_jsonl(units_path)

    by_task: Dict[str, List[dict]] = {}
    for u in units:
        by_task.setdefault(u["task_id"], []).append(u)

    trajs: Dict[str, "Trajectory"] = {}  # type: ignore  # noqa
    for r in sel:
        td = Path(r["output_dir"])
        if td.exists():
            try:
                trajs[r["task_id"]] = load_trajectory(td)
            except Exception:
                pass

    cells = []
    for tid, ulist in by_task.items():
        if tid not in trajs:
            continue
        ulist = ulist[: args.max_units_per_task]
        for u in ulist:
            cells.append((trajs[tid], u))
    print(f"[03] {len(cells)} (task, unit) cells across {len(by_task)} tasks "
          f"(cap {args.max_units_per_task}/task)")

    out_path = jsonl_path("motivation_behavioral_evidence.jsonl")
    t0 = time.time()
    n_done = 0
    n_useful = 0
    n_err = 0
    out_records: List[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_label_one, t, u): (t, u) for t, u in cells}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                out_records.append(rec)
                lbl = rec["label"]
                if lbl["useful"] and lbl["confidence"] in ("high", "medium"):
                    n_useful += 1
            except Exception as exc:
                n_err += 1
                t, u = futures[fut]
                out_records.append({
                    "task_id": t.task_id, "unit": u,
                    "label": {"useful": False, "confidence": "low",
                              "used_as": "not_used", "reason": str(exc)},
                    "error": str(exc),
                })
            n_done += 1
            if n_done % 25 == 0 or n_done == len(cells):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(cells) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>4d}/{len(cells)}] elapsed={elapsed/60:.1f}min  "
                      f"rate={rate*60:.1f}/min  ETA={eta/60:.1f}min  "
                      f"useful_so_far={n_useful}  err={n_err}")

    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[03] wrote {len(out_records)} rows -> {out_path}")
    print(f"[03] {n_useful} units kept as behavioral evidence "
          f"(useful=True with high/medium confidence)")
    print(f"[03] elapsed: {(time.time()-t0)/60:.1f} min  err={n_err}")


if __name__ == "__main__":
    main()
