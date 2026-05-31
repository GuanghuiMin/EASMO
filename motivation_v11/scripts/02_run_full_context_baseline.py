"""Stage 02 — full-context baseline on every task in inventory (spec §3.2 + §12.1).

Runs the MiniMax downstream agent with `compressed_context=""`
at cap_steps=15 on all 145 task_ids from the inventory. Records
success/fail/score regardless of outcome (do NOT filter — primary
analysis includes baseline-fail cases per spec §4.2).

MUST run with /workspace/acon/.venv/bin/python.

Output: outputs/raw/full_context_runs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _baseline_run(args_tuple):
    task_id, split, max_steps, tag = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context  # noqa

    res = run_with_compressed_context(
        task_id=task_id,
        method=f"v11_baseline_{split}",
        compressed_context="",
        max_steps=max_steps,
        split=split,
        tag=tag,
    )
    return {
        "task_id":              task_id,
        "split":                split,
        "full_success":         res.success,
        "full_score":           res.final_reward,
        "full_steps":           res.iterations,
        "full_peak_tokens":     res.input_tokens,
        "full_total_tokens":    res.input_tokens + res.output_tokens,
        "termination_reason":   res.termination_reason,
        "elapsed_s":            res.elapsed_s,
        "output_dir":           res.output_dir,
        "error":                res.error,
    }


def _read_jsonl_plain(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory",
                    default=str(_REPO / "outputs" / "provenance" / "appworld_task_inventory.csv"))
    ap.add_argument("--out",
                    default=str(_REPO / "outputs" / "raw" / "full_context_runs.jsonl"))
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv11_baseline")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inventory)))
    # Exclude (a) anything not marked included=true and (b) meta-note
    # rows that document expected-vs-actual count drift (META__/ACON_).
    rows = [r for r in rows if r["included"].lower() == "true"
            and not r["task_id"].startswith(("ACON_", "META__"))]
    print(f"[02] {len(rows)} tasks in inventory")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["split"]): r for r in _read_jsonl_plain(out_path)}
    pending = [(r["task_id"], r["split"], args.max_steps, args.tag)
               for r in rows if (r["task_id"], r["split"]) not in done]
    print(f"[02] {len(done)} already done; {len(pending)} pending")

    t0 = time.time(); n_done = 0; n_pass = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_baseline_run, w): w for w in pending}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"task_id": "?", "split": "?",
                            "full_success": False, "error": str(e)}
                if rec.get("full_success"): n_pass += 1
                if rec.get("error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 5 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(pending)-n_done)
                    print(f"  [{n_done}/{len(pending)}] "
                          f"pass={n_pass} err={n_err} eta={eta:.0f}s",
                          flush=True)
    all_rows = _read_jsonl_plain(out_path)
    total_pass = sum(1 for r in all_rows if r.get("full_success"))
    print(f"[02] baseline done: {n_done} new, {n_pass} pass, {n_err} err")
    print(f"     overall: {total_pass}/{len(all_rows)} = "
          f"{100*total_pass/max(len(all_rows),1):.1f}% pass")


if __name__ == "__main__":
    main()
