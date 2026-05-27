"""Stage 4 — Exp 2.2: audit each compressed context against behavioral evidence.

For each (task, method) compressed context produced in Stage 2, asks
the LLM to label per behavioral-evidence unit whether the compressed
context preserves it (per spec's preservation labels).

Outputs:
  outputs/motivation_audits.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _audit_one(task_id: str, method: str, task_instr: str,
               compressed_text: str, evidence_units: List[dict]):
    sys.path.insert(0, str(_REPO))
    from motivation_v3.evidence import audit_compressed
    a = audit_compressed(
        task_id=task_id, method=method,
        task_instruction=task_instr,
        compressed_context=compressed_text,
        behavioral_evidence_units=evidence_units,
    )
    return a.to_dict()


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

    ensure_outputs()
    sel = read_jsonl(jsonl_path("motivation_full_trajectories.jsonl"))
    compressed = read_jsonl(jsonl_path("motivation_compressed_contexts.jsonl"))
    evidence = read_jsonl(jsonl_path("motivation_behavioral_evidence.jsonl"))

    # Build behavioral evidence per task — keep only useful=True with high/medium.
    evidence_by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in evidence:
        lbl = r.get("label") or {}
        if lbl.get("useful") and lbl.get("confidence") in ("high", "medium"):
            evidence_by_task[r["task_id"]].append(r["unit"])
    print(f"[04] tasks with >= 1 behavioral evidence unit: {len(evidence_by_task)}")

    # Build (compressed_text, task_instruction) lookup.
    instr_by_task: Dict[str, str] = {}
    trajs: Dict[str, "Trajectory"] = {}  # type: ignore  # noqa
    for r in sel:
        td = Path(r["output_dir"])
        if td.exists():
            try:
                t = load_trajectory(td)
                trajs[r["task_id"]] = t
                instr_by_task[r["task_id"]] = t.instruction or ""
            except Exception:
                pass

    cells = []
    for c in compressed:
        if c.get("error") or c.get("method") not in args.methods:
            continue
        tid = c["task_id"]
        if tid not in evidence_by_task:
            continue
        cells.append((tid, c["method"], instr_by_task.get(tid, ""),
                      c["text"], evidence_by_task[tid]))

    print(f"[04] {len(cells)} audit cells; workers={args.workers}")

    out_path = jsonl_path("motivation_audits.jsonl")
    t0 = time.time()
    n_done = 0
    n_err = 0
    out_records: List[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_audit_one, *c): c for c in cells}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                out_records.append(rec)
            except Exception as exc:
                n_err += 1
                tid, method, *_ = futures[fut]
                out_records.append({
                    "task_id": tid, "method": method,
                    "unit_results": [], "summary": {},
                    "error": str(exc),
                })
            n_done += 1
            if n_done % 5 == 0 or n_done == len(cells):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(cells) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>3d}/{len(cells)}] elapsed={elapsed/60:.1f}min  "
                      f"rate={rate*60:.1f}/min  ETA={eta/60:.1f}min  err={n_err}")

    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[04] wrote {len(out_records)} rows -> {out_path}")
    print(f"[04] elapsed: {(time.time()-t0)/60:.1f} min  err={n_err}")


if __name__ == "__main__":
    main()
