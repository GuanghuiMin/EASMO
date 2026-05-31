"""Stage 01 — full AppWorld dev + train baseline + primary case build (spec §3).

Runs the MiniMax downstream agent (no compression) on the full train+dev
split (145 tasks), records success/fail per cap_steps=15, then writes:

  data/v11_baseline_runs.jsonl                — one row per attempted task
  data/v11_primary_cases.jsonl                — only baseline_success=True
  data/v11_secondary_all_cases.jsonl          — all 145 cases (incl. fails)
  outputs/tables/case_pool_summary.csv

Resumable. Must run with /workspace/acon/.venv/bin/python.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

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
        "task_id":           task_id,
        "split":             split,
        "baseline_success":  res.success,
        "baseline_iterations": res.iterations,
        "baseline_score":    res.final_reward,
        "termination_reason": res.termination_reason,
        "input_tokens":      res.input_tokens,
        "output_tokens":     res.output_tokens,
        "elapsed_s":         res.elapsed_s,
        "output_dir":        res.output_dir,
        "error":             res.error,
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


def _appworld_split(name):
    p = Path("/workspace/acon/experiments/appworld/data/datasets") / f"{name}.txt"
    return sorted({l.strip() for l in open(p).read().splitlines() if l.strip()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_pool", choices=("dev", "train", "train+dev",
                                              "train+dev+test_normal"),
                    default="train+dev")
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv11_baseline")
    ap.add_argument("--baseline_out",
                    default=str(_REPO / "data" / "v11_baseline_runs.jsonl"))
    ap.add_argument("--primary_out",
                    default=str(_REPO / "data" / "v11_primary_cases.jsonl"))
    ap.add_argument("--secondary_out",
                    default=str(_REPO / "data" / "v11_secondary_all_cases.jsonl"))
    ap.add_argument("--summary_out",
                    default=str(_REPO / "outputs" / "tables" / "case_pool_summary.csv"))
    args = ap.parse_args()

    (_REPO / "data").mkdir(parents=True, exist_ok=True)

    # Build task plan
    plan: List[tuple] = []
    if args.task_pool == "dev":
        for tid in _appworld_split("dev"):
            plan.append((tid, "dev"))
    elif args.task_pool == "train":
        for tid in _appworld_split("train"):
            plan.append((tid, "train"))
    elif args.task_pool == "train+dev":
        for tid in _appworld_split("train"):
            plan.append((tid, "train"))
        for tid in _appworld_split("dev"):
            plan.append((tid, "dev"))
    elif args.task_pool == "train+dev+test_normal":
        for tid in _appworld_split("train"):
            plan.append((tid, "train"))
        for tid in _appworld_split("dev"):
            plan.append((tid, "dev"))
        for tid in _appworld_split("test_normal"):
            plan.append((tid, "test_normal"))

    print(f"[01] task_pool={args.task_pool}, total={len(plan)} tasks")

    out_path = Path(args.baseline_out)
    done = {(r["task_id"], r["split"]): r for r in _read_jsonl_plain(out_path)}
    pending = [(tid, sp, args.max_steps, args.tag)
               for (tid, sp) in plan if (tid, sp) not in done]
    print(f"[01] {len(done)} already done; {len(pending)} pending")

    t0 = time.time(); n_done = 0; n_pass = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_baseline_run, w): w for w in pending}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"task_id":"?", "split":"?", "baseline_success":False,
                           "error": str(e)}
                if rec.get("baseline_success"): n_pass += 1
                if rec.get("error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 5 == 0 or n_done <= 3:
                    eta = (time.time()-t0) / n_done * (len(pending)-n_done)
                    print(f"  [{n_done}/{len(pending)}] "
                          f"pass={n_pass} err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[01] baseline done: {n_done} new, {n_pass} pass, {n_err} err; "
          f"elapsed {(time.time()-t0)/60:.1f} min")

    # Build primary + secondary case files
    all_runs = _read_jsonl_plain(out_path)
    print(f"[01] aggregating {len(all_runs)} baseline rows")

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v2.data import load_trajectory  # noqa
    from motivation_v3.data import render_trajectory  # noqa

    def _length_bucket(steps: int) -> str:
        if steps < 15: return "short"
        if steps < 25: return "medium"
        return "long"

    def _build_case(r) -> dict:
        tid = r["task_id"]
        out_dir = Path(r["output_dir"])
        traj_text = ""; steps = []
        apps = []
        instruction = ""
        try:
            if out_dir.exists():
                traj = load_trajectory(out_dir)
                instruction = traj.instruction or ""
                steps = [{
                    "step_id": int(s.step),
                    "thought": None,
                    "action":  s.action or "",
                    "observation": s.output or "",
                } for s in traj.steps]
                apps = sorted({m.group(1) for s in traj.steps
                                for m in re.finditer(
                                    r"\bapis\.([a-zA-Z0-9_]+)\.", s.action or "")})
                traj_text = render_trajectory(traj, max_total_chars=18000)
        except Exception:
            pass
        return {
            "case_id":              tid,
            "task_id":              tid,
            "split":                r["split"],
            "tier":                 "primary" if r.get("baseline_success") else "secondary_only",
            "user_instruction":     instruction,
            "full_trajectory_text": traj_text,
            "trajectory_steps":     steps,
            "baseline_success":     bool(r.get("baseline_success")),
            "baseline_iterations":  int(r.get("baseline_iterations", 0)),
            "baseline_score":       float(r.get("baseline_score", 0.0)),
            "compression_boundary": "full",
            "max_steps_for_continuation": args.max_steps,
            "apps_used":            apps,
            "n_apps":               len(apps),
            "length_bucket":        _length_bucket(int(r.get("baseline_iterations", 0))),
            "case_priority":        "high_value" if r.get("baseline_success") else "secondary",
            "notes":                "",
        }

    cases = [_build_case(r) for r in all_runs]
    primary = [c for c in cases if c["baseline_success"]]
    primary.sort(key=lambda c: (c["split"], c["case_id"]))
    secondary = sorted(cases, key=lambda c: (c["split"], c["case_id"]))

    with open(args.primary_out, "w") as f:
        for c in primary:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(args.secondary_out, "w") as f:
        for c in secondary:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[01] primary cases: {len(primary)}/{len(cases)} = "
          f"{100*len(primary)/max(len(cases),1):.1f}%")

    # case_pool_summary.csv
    import csv, collections
    sp_n = collections.Counter(c["split"] for c in cases)
    sp_pass = collections.Counter(c["split"] for c in primary)
    bucket = collections.Counter(c["length_bucket"] for c in primary)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "key", "value"])
        w.writerow(["totals", "n_total_tasks", len(cases)])
        w.writerow(["totals", "n_primary_cases", len(primary)])
        w.writerow(["totals", "n_secondary_cases", len(cases)])
        w.writerow(["totals", "primary_pass_rate", f"{len(primary)/max(len(cases),1):.4f}"])
        for sp in sorted(sp_n):
            w.writerow(["split", sp, sp_n[sp]])
            w.writerow(["split_pass", sp, sp_pass.get(sp, 0)])
        for b in ("short", "medium", "long"):
            w.writerow(["length_bucket_primary", b, bucket.get(b, 0)])
    print(f"[01] wrote case_pool_summary -> {args.summary_out}")
    print(f"     split counts:  {dict(sp_n)}")
    print(f"     split passing: {dict(sp_pass)}")
    print(f"     primary by length bucket: {dict(bucket)}")


if __name__ == "__main__":
    main()
