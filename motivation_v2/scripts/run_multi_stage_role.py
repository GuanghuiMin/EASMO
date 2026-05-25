"""Run the 3-agent role-specialised AppWorld pipeline on a task list.

For each task: planner → executor → verifier. The executor is acon's
standard AppWorldAgent with the planner's plan injected as a
"PRE-LOADED PLAN" turn (reusing the runner.py prompt-splice
machinery). Each agent's output is persisted so we can extract
real per-role memories afterward.

Outputs (per task):
    outputs/<tag>/task_<id>/
        plan.json
        executor_trajectory.json    (acon's standard format)
        verifier.json
        results.json                (acon's, reflecting executor's outcome)

Usage:
    /workspace/acon/.venv/bin/python motivation_v2/scripts/run_multi_stage_role.py \
        --tasks 82e2fac_3 ccb4494_1 ... \
        --tag mv2_multi_stage \
        --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

_REPO = Path(__file__).resolve().parent.parent
_ACON_APPWORLD = Path("/workspace/acon/experiments/appworld")
_PLAN_INJECT_ROOT = _ACON_APPWORLD / "prompts" / "_motivation_v2" / "_plan_cells"


# ----------------------------------------------------------------------
# Per-task plan-injected prompt jinja
# ----------------------------------------------------------------------


def _materialise_plan_prompt(plan_text: str) -> Path:
    """Build a per-plan jinja that splices a PRE-LOADED PLAN block in
    front of the task instruction. Reuses the canonical AppWorld jinja.
    """
    src_jinja = _ACON_APPWORLD / "prompts" / "prompt_v1.jinja"
    if not src_jinja.exists():
        raise FileNotFoundError(f"Missing canonical jinja: {src_jinja}")
    src_text = src_jinja.read_text(encoding="utf-8")

    block = (
        "USER:\n"
        "**PRE-LOADED PLAN** from upstream PLANNER agent. Follow this plan when "
        "executing. You may deviate if you see clear evidence the plan is wrong, "
        "but document why.\n"
        "\n"
        f"{plan_text.strip()}\n"
        "\n"
        "USER:\n"
    )

    splice_marker = "Using these APIs, now generate code to solve the actual task:"
    head_marker = f"USER:\n{splice_marker}"
    if head_marker not in src_text:
        # Defensive fallback
        head, tail = src_text.split(splice_marker, 1)
        new_text = head + block.replace("USER:\n", "", 1) + splice_marker + tail
    else:
        head, tail = src_text.split(head_marker, 1)
        new_text = head + block + head_marker[len("USER:\n"):] + tail

    h = hashlib.sha1(plan_text.encode()).hexdigest()[:16]
    cell_dir = _PLAN_INJECT_ROOT / h
    cell_dir.mkdir(parents=True, exist_ok=True)
    out_jinja = cell_dir / "prompt.jinja"
    out_json = cell_dir / "prompts.json"
    out_jinja.write_text(new_text, encoding="utf-8")

    canonical = json.loads(
        (_ACON_APPWORLD / "prompts" / "prompts_v1.json").read_text()
    )
    canonical["main_prompt_template"] = "./" + str(
        out_jinja.relative_to(_ACON_APPWORLD)
    )
    out_json.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
    return out_json


# ----------------------------------------------------------------------
# Worker: run the full 3-agent pipeline on one task
# ----------------------------------------------------------------------


def _one_task(args_tuple):
    task_id, tag, max_iter, model_name, co_config_path = args_tuple

    sys.path.insert(0, str(_REPO))
    from motivation_v2.multi_stage_agents import (
        run_planner, run_verifier,
    )
    from motivation_v2.data import load_trajectory
    from motivation_v2.data import load_ground_truth

    # Need acon's run.main.
    import os as _os
    _os.chdir(_ACON_APPWORLD)
    sys.path.insert(0, str(_ACON_APPWORLD))
    import run as acon_run  # type: ignore

    out_dir = (
        _ACON_APPWORLD / "outputs" /
        f"MiniMaxAI_MiniMax-M2.5_{tag}" / "train" / f"task_{task_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_t0 = time.time()
    rec: dict = {"task_id": task_id, "tag": tag}

    # ---- 1. Planner ------------------------------------------------------
    try:
        gt = load_ground_truth(task_id)
        instruction = gt.instruction
    except FileNotFoundError:
        rec["error"] = f"no ground truth for {task_id}"
        return rec

    planner_out = run_planner(task_id, instruction, model=model_name)
    (out_dir / "plan.json").write_text(json.dumps(planner_out.to_dict(), indent=2))
    rec["planner"] = planner_out.to_dict()

    # ---- 2. Executor (acon's AppWorldAgent with plan injected) ---------
    plan_prompt_json = _materialise_plan_prompt(planner_out.plan_text)
    rel_prompt_file = "./" + os.path.relpath(plan_prompt_json, _ACON_APPWORLD)

    base_yaml = _ACON_APPWORLD / "configs" / "base_config.yaml"
    if base_yaml.exists():
        import yaml
        with open(base_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    exp_id = f"{tag}_executor"
    cfg.update({
        "exp_id": exp_id,
        "model_name": model_name,
        "tag": tag,
        "max_iter": max_iter,
        "use_workflow_memory": False,
        "use_thinking_tokens": True,
        "prompt_file": rel_prompt_file,
        "co_config_path": co_config_path,
        "experiment_name": f"experiment_{exp_id}",
        "seed": 42,
        "debug_mode": False,
    })
    experiment_name = f"MiniMaxAI_MiniMax-M2.5_{tag}"

    try:
        exec_t0 = time.time()
        exec_res = acon_run.main(
            task_id=task_id,
            split="train",
            output_dir=str(out_dir),
            exp_config=cfg,
            model_name=model_name,
            debug_mode=False,
            experiment_name=experiment_name,
            max_iter=max_iter,
        )
        exec_elapsed = time.time() - exec_t0
        rec["executor"] = {
            "success": bool(exec_res.get("success")),
            "iterations": int(exec_res.get("iterations", 0)),
            "final_reward": float(exec_res.get("final_reward", 0.0)),
            "termination_reason": exec_res.get("termination_reason"),
            "elapsed_s": exec_elapsed,
            "input_tokens": (exec_res.get("token_usage") or {}).get("total_input_tokens", 0),
            "output_tokens": (exec_res.get("token_usage") or {}).get("total_output_tokens", 0),
        }
    except Exception as exc:
        rec["executor"] = {"error": str(exc)}
        rec["pipeline_elapsed_s"] = time.time() - pipeline_t0
        return rec

    # ---- 3. Verifier ----------------------------------------------------
    try:
        traj = load_trajectory(out_dir)
        ver_out = run_verifier(traj, model=model_name)
        (out_dir / "verifier.json").write_text(json.dumps(ver_out.to_dict(), indent=2))
        rec["verifier"] = ver_out.to_dict()
    except Exception as exc:
        rec["verifier"] = {"error": str(exc)}

    rec["pipeline_elapsed_s"] = time.time() - pipeline_t0
    return rec


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=False,
                        help="Task IDs; if omitted, uses --task_file.")
    parser.add_argument("--task_file", type=str, default=None,
                        help="Optional newline-delimited task-id list.")
    parser.add_argument("--tag", default="mv2_multi_stage")
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--model_name", default="MiniMaxAI/MiniMax-M2.5")
    parser.add_argument("--co_config_path",
                        default="configs/context_opt/minimax-m25_history.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_dir",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_multi_stage")
    args = parser.parse_args()

    if args.tasks:
        tasks = list(args.tasks)
    elif args.task_file:
        tasks = [
            line.strip() for line in Path(args.task_file).read_text().splitlines()
            if line.strip()
        ]
    else:
        # Default: 6 spotify consumers from the xtask design.
        tasks = [
            "82e2fac_3", "ccb4494_1", "e7a10f8_1",
            "692c77d_1", "ce359b5_1", "07b42fd_1",
        ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pipeline_summary.jsonl"

    print(f"Running multi-stage pipeline on {len(tasks)} tasks ({args.tag})")
    print(f"Workers: {args.workers}, max_iter: {args.max_iter}")

    cells = [(tid, args.tag, args.max_iter, args.model_name, args.co_config_path)
             for tid in tasks]

    t0 = time.time()
    n_done = 0
    with open(summary_path, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_one_task, c): c for c in cells}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as exc:
                    cell = futures[fut]
                    rec = {"task_id": cell[0], "error": str(exc)}
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                n_done += 1
                ok = (rec.get("executor") or {}).get("success", False)
                ver = (rec.get("verifier") or {}).get("verdict", "?")
                print(f"  [{n_done:>3d}/{len(tasks)}] {rec.get('task_id'):>12s}  "
                      f"executor={ok!s:<5}  verifier={ver:<10}  "
                      f"elapsed={rec.get('pipeline_elapsed_s', 0):.0f}s")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Summary at {summary_path}")


if __name__ == "__main__":
    main()
