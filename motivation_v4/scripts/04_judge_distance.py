"""Stage 04 — LLM-judge distance between reference and ablated states.

For each ablated probe (task, span), call the LLM-judge with the spec's
prompt to verdict whether removing the span changed the decision state
in a behaviorally meaningful way.

Outputs:
  outputs/raw/span_judge_distances.jsonl
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


def _judge_one(task_id: str, span_id: str, instr: str, ref: dict, abl: dict):
    sys.path.insert(0, str(_REPO))
    from motivation_v4.probe import judge_distance
    v = judge_distance(
        task_instruction=instr,
        reference_state=ref,
        ablated_state=abl,
    )
    return {
        "task_id": task_id,
        "span_id": span_id,
        "judge": v.to_dict(),
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

    ensure_outputs()
    sel = load_v3_trajectories()
    refs = read_jsonl(raw_path("reference_decision_states.jsonl"))
    abls = read_jsonl(raw_path("span_ablation_probes.jsonl"))

    ref_by_task = {r["task_id"]: r["decision_state"] for r in refs}

    instr_by_task: Dict[str, str] = {}
    for r in sel:
        td = Path(r["output_dir"])
        if td.exists():
            try:
                t = load_trajectory(td)
                instr_by_task[t.task_id] = t.instruction or ""
            except Exception:
                pass

    cells = []
    for r in abls:
        if not r.get("parse_ok"):
            continue  # skip failed parses; judge would be meaningless
        tid = r["task_id"]
        ref = ref_by_task.get(tid)
        if ref is None:
            continue
        cells.append((tid, r["span_id"], instr_by_task.get(tid, ""),
                      ref, r["decision_state"]))
    print(f"[04] {len(cells)} judge cells; workers={args.workers}")
    print()

    out_records: List[dict] = []
    n_done = 0
    n_meaningful = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_judge_one, *c): c for c in cells}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                out_records.append(rec)
                if rec["judge"]["meaningful_change"]:
                    n_meaningful += 1
            except Exception as exc:
                tid, span_id, *_ = futures[fut]
                out_records.append({
                    "task_id": tid, "span_id": span_id,
                    "judge": {"meaningful_change": False, "severity": "none",
                              "score": 0.0, "changed_fields": [],
                              "reason": f"error: {exc}"},
                })
            n_done += 1
            if n_done % 25 == 0 or n_done == len(cells):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(cells) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>4d}/{len(cells)}] "
                      f"elapsed={elapsed/60:.1f}min  rate={rate*60:.1f}/min  "
                      f"ETA={eta/60:.1f}min  meaningful={n_meaningful}")

    out_path = raw_path("span_judge_distances.jsonl")
    write_jsonl(out_path, out_records)
    print(f"[04] wrote {len(out_records)} rows -> {out_path}")
    print(f"[04] elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
