"""Stage 6 — Exp 3 post-hoc: label each downstream-run API call as
recovery-or-not.

For each successful or progressing run from Stage 5, we re-parse the
agent's env_history.json, extract every apis.X.Y(...) call + its
output, and ask the LLM to label whether that call re-fetched
information already present in the original full-context trajectory
(per spec's recovery-call definition).

To keep cost bounded, we sample at most ``--max_calls_per_run`` calls
per cell (default 8 — covers most short cells fully, samples the long
ones).

Output: per-run aggregate counts written to a side-table JSONL:
  outputs/motivation_recovery_labels.jsonl   one row per (cell, api_call)
  outputs/motivation_behavior_runs_with_recovery.jsonl
                                              same as runs but with
                                              api_call_count and
                                              recovery_api_call_count fields
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


_API_RE = re.compile(r"apis\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\((.*?)\)", re.DOTALL)


def _api_calls_from_env_history(path: Path, max_calls: int) -> List[Tuple[str, str]]:
    """Returns up to ``max_calls`` (api_call_text, output) pairs."""
    if not path.exists():
        return []
    try:
        steps = json.loads(path.read_text())
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    for s in steps:
        action = s.get("action") or ""
        output = s.get("output") or ""
        for m in _API_RE.finditer(action):
            call_text = f"apis.{m.group(1)}.{m.group(2)}({m.group(3)[:200].strip()})"
            out.append((call_text, output[:600]))
            if len(out) >= max_calls:
                return out
    return out


def _label_one(args_tuple):
    (run, evidence_units, api_call, api_response) = args_tuple
    sys.path.insert(0, str(_REPO))
    from motivation_v3.evidence import label_recovery_call
    lbl = label_recovery_call(
        compressed_context=run.get("compressed_context", "")[:6000],
        behavioral_evidence_units=evidence_units,
        api_call=api_call, api_response=api_response,
    )
    return {
        "task_id": run["task_id"],
        "method": run["method"],
        "budget_max_steps": run["budget_max_steps"],
        "api_call": api_call,
        "label": lbl.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_calls_per_run", type=int, default=8)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.data import (
        OUTPUTS, ensure_outputs, jsonl_path, read_jsonl,
    )

    ensure_outputs()
    runs = read_jsonl(jsonl_path("motivation_behavior_runs.jsonl"))
    compressed = read_jsonl(jsonl_path("motivation_compressed_contexts.jsonl"))
    evidence = read_jsonl(jsonl_path("motivation_behavioral_evidence.jsonl"))

    by_tm: Dict[Tuple[str, str], str] = {}
    for r in compressed:
        if r.get("error"):
            continue
        by_tm[(r["task_id"], r["method"])] = r.get("text", "")
    evidence_by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in evidence:
        lbl = r.get("label") or {}
        if lbl.get("useful") and lbl.get("confidence") in ("high", "medium"):
            evidence_by_task[r["task_id"]].append(r["unit"])

    cells = []
    skipped = 0
    for run in runs:
        if run.get("error"):
            skipped += 1
            continue
        ev = evidence_by_task.get(run["task_id"], [])
        # Compressed context for this cell (recover from Stage 2).
        cond = run["method"]
        if cond in ("task_aware_summary", "acon_style_summary", "symbolic_evidence"):
            ctx = by_tm.get((run["task_id"], cond), "")
        else:
            ctx = ""
        run = dict(run)
        run["compressed_context"] = ctx
        env_path = Path(run["output_dir"]) / "env_history.json"
        api_pairs = _api_calls_from_env_history(env_path, args.max_calls_per_run)
        for call, resp in api_pairs:
            cells.append((run, ev, call, resp))

    print(f"[06] {len(cells)} (run, api_call) cells; workers={args.workers}; "
          f"skipped {skipped} runs with errors")

    out_labels = jsonl_path("motivation_recovery_labels.jsonl")
    t0 = time.time()
    n_done = 0
    label_records: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_label_one, c): c for c in cells}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                label_records.append(rec)
            except Exception as exc:
                pass
            n_done += 1
            if n_done % 50 == 0 or n_done == len(cells):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(cells) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>4d}/{len(cells)}] elapsed={elapsed/60:.1f}min  "
                      f"rate={rate*60:.1f}/min  ETA={eta/60:.1f}min")

    with open(out_labels, "w") as f:
        for r in label_records:
            f.write(json.dumps(r) + "\n")
    print(f"[06] wrote {len(label_records)} label rows -> {out_labels}")

    # Aggregate per-cell recovery counts and write augmented runs.
    counts: Dict[Tuple[str, str, int], Dict[str, int]] = defaultdict(
        lambda: {"api_call_count": 0, "recovery_api_call_count": 0}
    )
    for r in label_records:
        key = (r["task_id"], r["method"], r["budget_max_steps"])
        counts[key]["api_call_count"] += 1
        if r["label"].get("recovery_call") and r["label"].get("confidence") in ("high", "medium"):
            counts[key]["recovery_api_call_count"] += 1

    aug_path = jsonl_path("motivation_behavior_runs_with_recovery.jsonl")
    with open(aug_path, "w") as f:
        for run in runs:
            key = (run["task_id"], run["method"], run["budget_max_steps"])
            c = counts.get(key, {"api_call_count": 0, "recovery_api_call_count": 0})
            run = {**run, **c}
            f.write(json.dumps(run) + "\n")
    print(f"[06] wrote {len(runs)} augmented runs -> {aug_path}")
    print(f"[06] elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
