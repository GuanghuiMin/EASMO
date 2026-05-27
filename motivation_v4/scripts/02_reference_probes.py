"""Stage 02 — compute the full-context reference decision state per task.

For each v3 trajectory, render the full trajectory as the probe context
and ask the decision-state probe. Save the structured state as the
reference for Stage 03 distance comparisons.

Outputs:
  outputs/raw/reference_decision_states.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _probe_one(task_id: str, instr: str, ctx_text: str):
    sys.path.insert(0, str(_REPO))
    from motivation_v4.probe import probe_decision_state
    ds = probe_decision_state(task_instruction=instr, context_text=ctx_text)
    return {
        "task_id": task_id,
        "context_type": "full_context",
        "decision_state": ds.state,
        "elapsed_s": ds.elapsed_s,
        "parse_ok": ds.parse_ok,
        "raw": ds.raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_chars", type=int, default=18000,
                        help="Cap full-trajectory rendering to this many chars.")
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, write_jsonl,
        load_v3_trajectories, load_trajectory,
    )

    # Use v3's render_trajectory helper (already cap-aware).
    from motivation_v3.data import render_trajectory

    ensure_outputs()
    sel = load_v3_trajectories()

    cells = []
    for r in sel:
        td = Path(r["output_dir"])
        if not td.exists():
            continue
        try:
            traj = load_trajectory(td)
        except Exception:
            continue
        ctx = render_trajectory(traj, max_total_chars=args.max_chars)
        cells.append((traj.task_id, traj.instruction or "", ctx))
    print(f"[02] {len(cells)} reference probes; workers={args.workers}")
    print()

    out_records: List[dict] = []
    n_ok = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_probe_one, tid, instr, ctx): tid
                   for (tid, instr, ctx) in cells}
        n_done = 0
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                out_records.append(rec)
                if rec["parse_ok"]:
                    n_ok += 1
            except Exception as exc:
                tid = futures[fut]
                out_records.append({
                    "task_id": tid, "context_type": "full_context",
                    "decision_state": {}, "elapsed_s": 0,
                    "parse_ok": False, "error": str(exc),
                })
            n_done += 1
            elapsed = time.time() - t0
            eta = (len(cells) - n_done) / max(n_done / max(elapsed, 1), 0.01)
            print(f"  [{n_done:>2d}/{len(cells)}] elapsed={elapsed:.0f}s  "
                  f"ETA={eta:.0f}s  parse_ok_so_far={n_ok}")

    out_path = raw_path("reference_decision_states.jsonl")
    write_jsonl(out_path, out_records)
    print(f"[02] wrote {len(out_records)} rows ({n_ok} parse_ok) -> {out_path}")
    print(f"[02] elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
