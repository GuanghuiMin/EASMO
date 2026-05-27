"""Stage 03 — leave-one-span-out ablation probes.

For each (task, span), remove that span from the chronological history,
re-render, and run the decision-state probe. Persist the ablated state
for Stage 04 distance scoring.

Outputs:
  outputs/raw/span_ablation_probes.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _probe_ablation(task_id: str, span_id: str, instr: str, ctx_minus: str):
    sys.path.insert(0, str(_REPO))
    from motivation_v4.probe import probe_decision_state
    ds = probe_decision_state(task_instruction=instr, context_text=ctx_minus)
    return {
        "task_id": task_id,
        "span_id": span_id,
        "context_type": "ablated",
        "decision_state": ds.state,
        "elapsed_s": ds.elapsed_s,
        "parse_ok": ds.parse_ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, read_jsonl, write_jsonl,
        load_v3_trajectories, load_trajectory,
    )
    from motivation_v4.spans import (
        Span, render_history, render_history_minus,
    )

    ensure_outputs()
    sel = load_v3_trajectories()
    spans_rows = read_jsonl(raw_path("history_spans.jsonl"))

    # Group spans by task.
    spans_by_task: Dict[str, List[Span]] = defaultdict(list)
    for r in spans_rows:
        spans_by_task[r["task_id"]].append(Span(**r))
    for tid in spans_by_task:
        spans_by_task[tid].sort(key=lambda s: s.step_id)

    # Look up task instruction per task.
    instr_by_task: Dict[str, str] = {}
    for r in sel:
        td = Path(r["output_dir"])
        if td.exists():
            try:
                traj = load_trajectory(td)
                instr_by_task[traj.task_id] = traj.instruction or ""
            except Exception:
                instr_by_task[r["task_id"]] = ""

    cells = []
    for tid, spans in spans_by_task.items():
        instr = instr_by_task.get(tid, "")
        for s in spans:
            ctx = render_history_minus(spans, s.span_id)
            cells.append((tid, s.span_id, instr, ctx))
    print(f"[03] {len(cells)} ablation probes across {len(spans_by_task)} tasks "
          f"(workers={args.workers})")
    print()

    out_records: List[dict] = []
    n_ok = 0
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_probe_ablation, *c): c for c in cells}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                out_records.append(rec)
                if rec["parse_ok"]:
                    n_ok += 1
            except Exception as exc:
                tid, span_id, *_ = futures[fut]
                out_records.append({
                    "task_id": tid, "span_id": span_id,
                    "context_type": "ablated", "decision_state": {},
                    "elapsed_s": 0, "parse_ok": False, "error": str(exc),
                })
            n_done += 1
            if n_done % 25 == 0 or n_done == len(cells):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(cells) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>4d}/{len(cells)}] "
                      f"elapsed={elapsed/60:.1f}min  "
                      f"rate={rate*60:.1f}/min  ETA={eta/60:.1f}min  "
                      f"parse_ok={n_ok}")

    out_path = raw_path("span_ablation_probes.jsonl")
    write_jsonl(out_path, out_records)
    print(f"[03] wrote {len(out_records)} rows -> {out_path}")
    print(f"[03] elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
