"""Stage 01 — build v10 case pool by running baseline MiniMax agent on
the new tasks (89 train + 26 dev_proxy + 30 test_behavior) and
filtering to baseline_success=true.

Runs inside acon/.venv. Uses motivation_v4.runner (which delegates to
motivation_v3.runner) with `compressed_context=""` to get a clean
no-compression baseline.

Writes:
  data/v10_baseline_runs.jsonl        (one row per attempted task)
  data/v10_cases.jsonl                (only baseline_success=true cases,
                                       enriched with trajectory text)

Skips tasks that are already in v9's legacy set (those count as
legacy_v9 split and are loaded separately).

Resumable: re-running picks up where the previous run left off.
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


# ---------------------------------------------------------------------
# Worker — must be top-level for ProcessPoolExecutor.
# ---------------------------------------------------------------------

def _baseline_run(args_tuple):
    task_id, split, max_steps, tag = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context  # noqa

    res = run_with_compressed_context(
        task_id=task_id,
        method=f"v10_baseline_{split}",
        compressed_context="",          # baseline = no prior compression
        max_steps=max_steps,
        split=split,
        tag=tag,
    )
    return {
        "task_id":           task_id,
        "split":             split,
        "baseline_success":  res.success,
        "baseline_iterations": res.iterations,
        "final_reward":      res.final_reward,
        "termination_reason": res.termination_reason,
        "input_tokens":      res.input_tokens,
        "output_tokens":     res.output_tokens,
        "elapsed_s":         res.elapsed_s,
        "output_dir":        res.output_dir,
        "error":             res.error,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _read_jsonl_plain(p: Path):
    out = []
    if not p.exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _appworld_split(name: str) -> List[str]:
    p = Path("/workspace/acon/experiments/appworld/data/datasets") / f"{name}.txt"
    return sorted({l.strip() for l in open(p).read().splitlines() if l.strip()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_test_n", type=int, default=30,
                    help="Limit test_normal to first N task_ids (default 30).")
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv10_baseline")
    ap.add_argument("--baseline_out",
                    default=str(_REPO / "data" / "v10_baseline_runs.jsonl"))
    ap.add_argument("--cases_out",
                    default=str(_REPO / "data" / "v10_cases.jsonl"))
    args = ap.parse_args()

    (_REPO / "data").mkdir(parents=True, exist_ok=True)

    # 1) Build the universe of task_ids to baseline -------------------
    legacy_v9 = set()
    v9_cases_path = Path("/workspace/EASMO/motivation_v9/data/v9_cases.jsonl")
    if v9_cases_path.exists():
        legacy_v9 = {json.loads(l)["task_id"] for l in open(v9_cases_path) if l.strip()}
    print(f"[01] legacy_v9 task_ids cached: {len(legacy_v9)}")

    train_ids = _appworld_split("train")
    dev_ids   = _appworld_split("dev")
    test_ids  = _appworld_split("test_normal")[:args.max_test_n]

    # Plan: train.txt entirely; dev minus legacy_v9; test_normal first N
    plan = []
    for tid in train_ids:
        if tid in legacy_v9: continue
        plan.append((tid, "train", "teacher_train"))
    for tid in dev_ids:
        if tid in legacy_v9: continue
        plan.append((tid, "dev", "dev_proxy"))
    for tid in test_ids:
        if tid in legacy_v9: continue
        plan.append((tid, "test_normal", "test_behavior"))

    print(f"[01] plan: {len(plan)} tasks "
          f"({sum(1 for x in plan if x[2]=='teacher_train')} teacher_train, "
          f"{sum(1 for x in plan if x[2]=='dev_proxy')} dev_proxy, "
          f"{sum(1 for x in plan if x[2]=='test_behavior')} test_behavior)")

    # 2) Resume — skip already-done -----------------------------------
    out_path = Path(args.baseline_out)
    done = {(r["task_id"], r["split"]): r
            for r in _read_jsonl_plain(out_path)}
    pending = [(tid, ap_split) for (tid, ap_split, _v10s) in plan
               if (tid, ap_split) not in done]
    print(f"[01] {len(done)} already done; {len(pending)} pending")

    # 3) Run baselines ------------------------------------------------
    t0 = time.time()
    n_done = 0; n_pass = 0; n_err = 0
    work_args = [(tid, ap_split, args.max_steps, args.tag)
                 for (tid, ap_split) in pending]
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_baseline_run, w): w for w in work_args}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"task_id": "?", "split": "?", "error": str(e),
                           "baseline_success": False}
                if rec.get("baseline_success"):
                    n_pass += 1
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 5 == 0 or n_done <= 3:
                    eta = (time.time()-t0) / n_done * (len(pending)-n_done)
                    print(f"  [{n_done:>3d}/{len(pending)}] "
                          f"pass={n_pass:<3d} err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[01] baseline done: {n_done} new runs ({n_pass} success, {n_err} errors)")
    print(f"[01] total elapsed {(time.time()-t0)/60:.1f} min")

    # 4) Build v10_cases.jsonl: passing cases + legacy_v9 -------------
    plan_split_by_task = {tid: v10s for (tid, _ap, v10s) in plan}
    all_runs = _read_jsonl_plain(out_path)
    pass_runs = [r for r in all_runs if r.get("baseline_success")]
    print(f"[01] passing baseline runs: {len(pass_runs)}/{len(all_runs)} = "
          f"{100*len(pass_runs)/max(len(all_runs),1):.1f}%")

    # Enrich each passing run with trajectory text + steps from acon output dir
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v2.data import load_trajectory  # noqa
    from motivation_v3.data import render_trajectory  # noqa

    cases = []
    skipped = 0
    for r in pass_runs:
        tid = r["task_id"]
        out_dir = Path(r["output_dir"])
        if not out_dir.exists():
            skipped += 1; continue
        try:
            traj = load_trajectory(out_dir)
        except Exception:
            skipped += 1; continue
        if not traj.steps:
            skipped += 1; continue
        steps = [{
            "step_id": int(s.step),
            "thought": None,
            "action":  s.action or "",
            "observation": s.output or "",
        } for s in traj.steps]
        apps = sorted({
            m.group(1)
            for s in traj.steps
            for m in re.finditer(r"\bapis\.([a-zA-Z0-9_]+)\.", s.action or "")
        })
        text = render_trajectory(traj, max_total_chars=18000)
        cases.append({
            "case_id":              traj.task_id,
            "task_id":              traj.task_id,
            "split":                plan_split_by_task.get(tid, "unknown"),
            "user_instruction":     traj.instruction or "",
            "full_trajectory_text": text,
            "trajectory_steps":     steps,
            "baseline_success":     True,
            "baseline_iterations":  len(steps),
            "compression_boundary": "full",
            "max_steps_for_continuation": args.max_steps,
            "apps_used":            apps,
            "n_apps":               len(apps),
            "case_priority":        "long" if len(steps) >= 25
                                    else ("medium" if len(steps) >= 15 else "short"),
            "notes":                "",
        })

    # 5) Add legacy_v9 cases verbatim ---------------------------------
    legacy_rows = _read_jsonl_plain(v9_cases_path)
    for r in legacy_rows:
        r["split"] = "legacy_v9"
        r["compression_boundary"] = "full"
        r.setdefault("max_steps_for_continuation", args.max_steps)
        cases.append(r)

    cases.sort(key=lambda c: (c["split"], c["case_id"]))
    out_cases = Path(args.cases_out)
    with open(out_cases, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    sp = Counter(c["split"] for c in cases)
    print(f"[01] wrote {len(cases)} cases to {out_cases} (skipped {skipped} traj-load failures)")
    print(f"     split distribution: {dict(sp)}")
    print(f"     legacy_v9 cases included: {len(legacy_rows)}")


if __name__ == "__main__":
    main()
