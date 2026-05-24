"""Run an AppWorld task with a pre-loaded compressed memory string.

The compressed-memory mode injects a new USER turn into the agent's
main prompt template, between the strategy block and the task
instruction:

    USER: **STRATEGY: <X>** ...
    USER: **PRE-LOADED MEMORY** (relevant context cached from the
          supervisor; you may use these facts to skip API queries you
          can answer from them — but you can still call APIs as needed):
          <memory_text>
    USER: Using these APIs, now generate code to solve the actual task:
          ...

Implementation: we materialise a per-cell jinja by splicing a literal
memory turn into the strategy-specific jinja, then call acon's
``run.main`` with ``prompt_file`` pointing at the cell-specific JSON.
The acon agent renders the rest of the template (supervisor info,
task instruction) at runtime as it normally would.

Per-cell jinjas live under
``acon/experiments/appworld/prompts/_motivation_v2/_cells/<cell-hash>/``
and are reused across runs (cell-hash is deterministic in
strategy + memory_text + budget).

Requires ``/workspace/acon/.venv`` because we import ``run`` and the
``productive_agents`` agent stack.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

_ACON_APPWORLD = Path("/workspace/acon/experiments/appworld")
_STRATEGIES_ROOT = _ACON_APPWORLD / "prompts" / "_motivation_v2"
_CELLS_ROOT = _ACON_APPWORLD / "prompts" / "_motivation_v2" / "_cells"


# Memory injection block — inserted RIGHT AFTER the strategy turn,
# RIGHT BEFORE the "Using these APIs, now generate code" turn.
_MEM_BLOCK_TEMPLATE = (
    "USER:\n"
    "**PRE-LOADED MEMORY** "
    "(relevant context the supervisor pre-cached for you; "
    "you may use these facts to skip API queries you can answer from "
    "them, but you can still call APIs whenever you need more info):\n"
    "\n"
    "{memory_text}\n"
    "\n"
    "USER:\n"
)


_SPLICE_MARKER = "Using these APIs, now generate code to solve the actual task:"


# ----------------------------------------------------------------------
# Cell jinja materialisation
# ----------------------------------------------------------------------


def _strategy_jinja_path(strategy: str) -> Path:
    return _STRATEGIES_ROOT / strategy / f"prompt_{strategy}.jinja"


def _cell_hash(strategy: str, memory_text: str, budget: int) -> str:
    h = hashlib.sha1()
    h.update(strategy.encode())
    h.update(b"\0")
    h.update(str(budget).encode())
    h.update(b"\0")
    h.update(memory_text.encode())
    return h.hexdigest()[:16]


def materialise_cell_prompt(
    strategy: str,
    memory_text: str,
    budget: int,
) -> Path:
    """Build (or look up) a per-cell jinja+JSON pair. Returns path to
    the JSON, suitable for use as ``exp_config['prompt_file']``.
    """
    if strategy not in {"direct", "verify", "explore"}:
        raise ValueError(f"Unknown strategy {strategy!r}")
    src_jinja = _strategy_jinja_path(strategy)
    if not src_jinja.exists():
        raise FileNotFoundError(
            f"Strategy jinja missing: {src_jinja}. "
            "Run prompts/build_strategy_prompts.py first."
        )

    h = _cell_hash(strategy, memory_text, budget)
    cell_dir = _CELLS_ROOT / h
    cell_dir.mkdir(parents=True, exist_ok=True)

    out_jinja = cell_dir / "prompt.jinja"
    out_json = cell_dir / "prompts.json"

    if out_jinja.exists() and out_json.exists():
        return out_json  # reuse

    src_text = src_jinja.read_text(encoding="utf-8")

    # Splice the memory block before the task-instruction USER turn.
    # The strategy jinja's structure has a USER turn with the splice
    # marker on the line just after; we insert the memory turn before
    # that marker.
    head_marker = f"USER:\n{_SPLICE_MARKER}"
    if head_marker not in src_text:
        # Defensive fallback: just insert before the marker line.
        if _SPLICE_MARKER not in src_text:
            raise RuntimeError(
                f"Splice marker not found in strategy jinja {src_jinja}"
            )
        head, tail = src_text.split(_SPLICE_MARKER, 1)
        new_text = head + _MEM_BLOCK_TEMPLATE.format(
            memory_text=memory_text
        ) + _SPLICE_MARKER + tail
    else:
        head, tail = src_text.split(head_marker, 1)
        new_text = head + _MEM_BLOCK_TEMPLATE.format(
            memory_text=memory_text
        ) + head_marker[len("USER:\n"):] + tail
    out_jinja.write_text(new_text, encoding="utf-8")

    # Build the prompts JSON pointing at the new jinja. Read the
    # canonical strategy JSON to inherit `system_message`.
    strategy_json = _STRATEGIES_ROOT / strategy / f"prompts_{strategy}.json"
    canonical = json.loads(strategy_json.read_text(encoding="utf-8"))
    out_dict = dict(canonical)
    rel = out_jinja.relative_to(_ACON_APPWORLD)
    out_dict["main_prompt_template"] = f"./{rel}"
    out_json.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    return out_json


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


@dataclass
class RunnerResult:
    task_id: str
    strategy: str
    compressor: str
    budget: int
    success: bool
    iterations: int
    final_reward: float
    termination_reason: str
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    output_dir: str
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {**self.__dict__}


def _import_acon():
    sys.path.insert(0, str(_ACON_APPWORLD))
    os.chdir(_ACON_APPWORLD)
    import run as acon_run  # type: ignore
    return acon_run


def run_with_compressed_memory(
    task_id: str,
    *,
    strategy: str,
    memory_text: str,
    compressor: str,
    budget: int,
    split: str = "train",
    model_name: str = "MiniMaxAI/MiniMax-M2.5",
    co_config_path: str = "configs/context_opt/minimax-m25_history.yaml",
    max_iter: int = 50,
    seed: int = 42,
    tag: str = "mv2_m1",
) -> RunnerResult:
    """Run one task with one compressed-memory variant; return outcome.

    The output directory is namespaced by
    ``MiniMaxAI_MiniMax-M2.5_<tag>_<strategy>_<compressor>_B<budget>/<split>/task_<id>``
    so different cells don't clobber each other.
    """
    prompt_json = materialise_cell_prompt(strategy, memory_text, budget)
    rel_prompt_file = "./" + os.path.relpath(prompt_json, _ACON_APPWORLD)

    # Build acon-style exp_config (same schema run_all.py uses).
    base_yaml = _ACON_APPWORLD / "configs" / "base_config.yaml"
    if base_yaml.exists():
        import yaml
        with open(base_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    exp_id = f"{tag}_{strategy}_{compressor}_B{budget}"
    cfg.update({
        "exp_id": exp_id,
        "model_name": model_name,
        "tag": exp_id,
        "max_iter": max_iter,
        "use_workflow_memory": False,
        "use_thinking_tokens": True,
        "prompt_file": rel_prompt_file,
        "co_config_path": co_config_path,
        "experiment_name": f"experiment_{exp_id}",
        "seed": seed,
        "debug_mode": False,
    })

    experiment_name = (
        f"{model_name.replace('/', '_')}_{tag}_{strategy}_"
        f"{compressor}_B{budget}"
    )
    output_dir = _ACON_APPWORLD / "outputs" / experiment_name / split / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    acon_run = _import_acon()
    t0 = time.time()
    try:
        res = acon_run.main(
            task_id=task_id,
            split=split,
            output_dir=str(output_dir),
            exp_config=cfg,
            model_name=model_name,
            debug_mode=False,
            experiment_name=experiment_name,
            max_iter=max_iter,
        )
        elapsed = time.time() - t0
        token_usage = (res.get("token_usage") or {})
        return RunnerResult(
            task_id=task_id,
            strategy=strategy,
            compressor=compressor,
            budget=budget,
            success=bool(res.get("success", False)),
            iterations=int(res.get("iterations", 0)),
            final_reward=float(res.get("final_reward", 0.0)),
            termination_reason=str(res.get("termination_reason", "?")),
            input_tokens=int(token_usage.get("total_input_tokens", 0)),
            output_tokens=int(token_usage.get("total_output_tokens", 0)),
            elapsed_s=elapsed,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        elapsed = time.time() - t0
        return RunnerResult(
            task_id=task_id,
            strategy=strategy,
            compressor=compressor,
            budget=budget,
            success=False,
            iterations=0,
            final_reward=0.0,
            termination_reason="exception",
            input_tokens=0,
            output_tokens=0,
            elapsed_s=elapsed,
            output_dir=str(output_dir),
            error=str(exc),
        )


# ----------------------------------------------------------------------
# CLI smoke
# ----------------------------------------------------------------------


def _cli():
    """Smoke: pick the first compressed-memory row for a known-good task,
    run it, print outcome.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", default="82e2fac_3")
    parser.add_argument("--strategy", default="direct")
    parser.add_argument("--compressor", default="m_exec_minimal")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument(
        "--memories_jsonl",
        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot/compressed_memories.jsonl",
        help="path to compressed_memories.jsonl produced by build_compressed_memories.py",
    )
    parser.add_argument("--tag", default="mv2_runner_smoke")
    args = parser.parse_args()

    # Fetch the matching memory row.
    memory_text = None
    with open(args.memories_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r["task_id"] == args.task_id
                    and r["policy_strategy"] == args.strategy
                    and r["compressor"] == args.compressor
                    and r["budget_tokens"] == args.budget):
                memory_text = r["memory_text"]
                break
    if memory_text is None:
        sys.exit(
            f"No row in {args.memories_jsonl} for "
            f"({args.task_id},{args.strategy},{args.compressor},B={args.budget})"
        )

    print(f"Running task={args.task_id} strategy={args.strategy} "
          f"compressor={args.compressor} B={args.budget}")
    print(f"  memory_text ({len(memory_text)} chars):")
    for line in memory_text.splitlines()[:5]:
        print(f"    {line}")
    if len(memory_text.splitlines()) > 5:
        print(f"    ...({len(memory_text.splitlines())-5} more lines)")

    res = run_with_compressed_memory(
        task_id=args.task_id,
        strategy=args.strategy,
        memory_text=memory_text,
        compressor=args.compressor,
        budget=args.budget,
        tag=args.tag,
    )
    print(f"\nResult:")
    for k, v in res.to_dict().items():
        if k == "extra" or v is None:
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
