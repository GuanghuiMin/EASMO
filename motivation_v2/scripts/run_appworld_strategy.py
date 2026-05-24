"""Run AppWorld tasks with a chosen Option-X strategy.

This is a thin wrapper around acon's ``run.main`` (single task) that:

1. Ensures the strategy-specific prompt files exist under
   ``acon/experiments/appworld/prompts/_motivation_v2/<strategy>/``
   (re-materialises them if missing or stale).
2. Builds the same exp_config that acon's ``run_all.py`` uses, but
   overrides ``prompt_file`` to point at the strategy's JSON.
3. Invokes ``run.main`` for one task, or iterates a list of tasks.

Outputs land at:

    acon/experiments/appworld/outputs/MiniMaxAI_MiniMax-M2.5_<tag>_<strategy>/<split>/task_<task_id>/

mirroring acon's directory layout, so the existing
``motivation_v2.data.iter_trajectories`` reader picks them up
unchanged.

CLI:
    cd /workspace/acon/experiments/appworld
    /workspace/acon/.venv/bin/python /workspace/EASMO/motivation_v2/scripts/run_appworld_strategy.py \
        --strategy direct \
        --split train \
        --task_id 82e2fac_3 \
        --tag motivation_v2_pilot

The cwd MUST be acon's ``experiments/appworld`` directory because
acon's run.py uses several relative paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List


# Paths
_ACON_APPWORLD = Path("/workspace/acon/experiments/appworld")
_STRATEGIES_ROOT = _ACON_APPWORLD / "prompts" / "_motivation_v2"
_BUILD_SCRIPT = Path("/workspace/EASMO/motivation_v2/prompts/build_strategy_prompts.py")

VALID_STRATEGIES = {"direct", "verify", "explore"}


def _ensure_strategy_files(strategy: str) -> Path:
    """Return path to the strategy's prompts_<strategy>.json, building it if needed."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected one of {sorted(VALID_STRATEGIES)}"
        )
    strat_dir = _STRATEGIES_ROOT / strategy
    json_path = strat_dir / f"prompts_{strategy}.json"
    jinja_path = strat_dir / f"prompt_{strategy}.jinja"
    if not (json_path.exists() and jinja_path.exists()):
        print(f"[strategy] Materialising strategy files for {strategy} …")
        # Re-run the builder.
        os.system(f"/workspace/acon/.venv/bin/python {_BUILD_SCRIPT}")
        if not (json_path.exists() and jinja_path.exists()):
            raise RuntimeError(
                f"Strategy build did not produce expected files: {json_path}, {jinja_path}"
            )
    return json_path


def _import_acon():
    """Add acon's experiments/appworld directory to sys.path and import run.main."""
    sys.path.insert(0, str(_ACON_APPWORLD))
    # Required for relative imports inside acon's run.py.
    os.chdir(_ACON_APPWORLD)
    import run as acon_run  # type: ignore  # noqa
    import run_all as acon_run_all  # type: ignore  # noqa
    return acon_run, acon_run_all


def _build_exp_config(
    strategy: str,
    tag: str,
    model_name: str,
    co_config_path: str,
    max_iter: int,
    seed: int,
) -> dict:
    """Replicate acon run_all.py's exp_config construction with a strategy override."""
    base_yaml = _ACON_APPWORLD / "configs" / "base_config.yaml"
    if base_yaml.exists():
        import yaml
        with open(base_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    prompt_file = str(_STRATEGIES_ROOT / strategy / f"prompts_{strategy}.json")
    rel_prompt_file = "./" + os.path.relpath(prompt_file, _ACON_APPWORLD)

    cfg.update({
        "exp_id": f"{tag}_{strategy}_{model_name}",
        "model_name": model_name,
        "tag": f"{tag}_{strategy}",
        "max_iter": max_iter,
        "use_workflow_memory": False,
        "use_thinking_tokens": True,
        # NOTE: this is the line that swaps the agent's main prompt for
        # the strategy-specific variant. Everything else mirrors
        # run_all.py's defaults.
        "prompt_file": rel_prompt_file,
        "co_config_path": co_config_path,
        "experiment_name": f"experiment_{tag}_{strategy}",
        "seed": seed,
        "debug_mode": False,
    })
    return cfg


def _experiment_name(tag: str, strategy: str, model_name: str) -> str:
    return f"{model_name.replace('/', '_')}_{tag}_{strategy}"


def _output_root(tag: str, strategy: str, model_name: str, split: str) -> Path:
    return (
        _ACON_APPWORLD / "outputs" /
        _experiment_name(tag, strategy, model_name) / split
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=sorted(VALID_STRATEGIES))
    parser.add_argument("--split", default="train")
    parser.add_argument("--task_id", help="single task id; if omitted, runs --task_ids or whole split")
    parser.add_argument("--task_ids", nargs="+", help="list of task ids")
    parser.add_argument("--tag", default="motivation_v2")
    parser.add_argument("--model_name", default="MiniMaxAI/MiniMax-M2.5")
    parser.add_argument("--co_config_path",
                        default="configs/context_opt/minimax-m25_history.yaml")
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue_existing", action="store_true",
                        help="Skip tasks whose output dir already exists")
    parser.add_argument("--first_n", type=int, default=None,
                        help="Run only the first N tasks of the split (for pilots)")
    args = parser.parse_args()

    _ensure_strategy_files(args.strategy)
    acon_run, _ = _import_acon()

    # Build task list.
    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_ids:
        task_ids = list(args.task_ids)
    else:
        # Read split file directly so we don't import the AppWorld
        # package's load_task_ids (which has its own root-resolution).
        split_file = _ACON_APPWORLD / "data" / "datasets" / f"{args.split}.txt"
        task_ids = [
            line.strip() for line in split_file.read_text().splitlines()
            if line.strip()
        ]
    if args.first_n:
        task_ids = task_ids[: args.first_n]

    exp_config = _build_exp_config(
        strategy=args.strategy, tag=args.tag, model_name=args.model_name,
        co_config_path=args.co_config_path, max_iter=args.max_iter, seed=args.seed,
    )
    experiment_name = _experiment_name(args.tag, args.strategy, args.model_name)
    output_root = _output_root(args.tag, args.strategy, args.model_name, args.split)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[run_appworld_strategy] strategy={args.strategy}  split={args.split}  "
          f"tasks={len(task_ids)}  tag={args.tag}")
    print(f"[run_appworld_strategy] prompt_file={exp_config['prompt_file']}")
    print(f"[run_appworld_strategy] output_root={output_root}")

    n_done = 0
    n_success = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.time()
    for i, task_id in enumerate(task_ids):
        task_output_dir = output_root / f"task_{task_id}"
        if args.continue_existing and (task_output_dir / "results.json").exists():
            print(f"[{i+1}/{len(task_ids)}] {task_id}: skipping (already done)")
            n_skipped += 1
            continue

        print(f"\n[{i+1}/{len(task_ids)}] {task_id} (strategy={args.strategy}) …")
        try:
            res = acon_run.main(
                task_id=task_id,
                split=args.split,
                output_dir=str(task_output_dir),
                exp_config=exp_config,
                model_name=args.model_name,
                debug_mode=False,
                experiment_name=experiment_name,
                max_iter=args.max_iter,
            )
            ok = bool(res.get("success", False))
            n_done += 1
            if ok:
                n_success += 1
            else:
                n_failed += 1
            elapsed = time.time() - t0
            print(f"  → success={ok}  iter={res.get('iterations')}  elapsed_total={elapsed/60:.1f}min")
        except Exception as exc:
            n_failed += 1
            print(f"  ! exception on {task_id}: {exc}", file=sys.stderr)
            continue

    print(f"\n[run_appworld_strategy] done in {(time.time()-t0)/60:.1f} min")
    print(f"  ran:    {n_done}")
    print(f"  ok:     {n_success}")
    print(f"  failed: {n_failed}")
    print(f"  skipped:{n_skipped}")

    # Drop a small summary so manipulation-check scripts can find this run.
    summary = {
        "strategy": args.strategy,
        "split": args.split,
        "tag": args.tag,
        "model_name": args.model_name,
        "n_attempted": len(task_ids),
        "n_success": n_success,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    with open(output_root / "_motivation_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
