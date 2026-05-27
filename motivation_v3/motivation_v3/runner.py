"""Run an AppWorld task with a compressed-context block injected per
the spec's DOWNSTREAM_AGENT_INSTRUCTION (see prompts.py).

Mirrors motivation_v2/runner.py but uses the v3 prompt template:

    USER:
    You are given compressed context from previous interaction:

    {compressed_context}

    Continue solving the task.

    Rules:
    1. Use exact IDs, values, and bindings from the compressed context when reliable.
    2. If critical information is missing or ambiguous, call tools to verify it.
    3. Avoid modifying unrelated objects or causing collateral damage.
    4. Do not repeat completed state-changing actions unless necessary and safe.
    5. Prefer fewer tool calls, but correctness is more important.
    6. Stop only when the task is complete.

    You have at most {max_steps} action steps.

    USER:

The block is spliced right before acon's "Using these APIs, now
generate code to solve the actual task:" turn.

The strategy used as the base is ``direct`` (matched to motivation_v2's
direct strategy) — we already have its compiled jinja under
``acon/experiments/appworld/prompts/_motivation_v2/direct/``.
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

from .prompts import DOWNSTREAM_AGENT_INSTRUCTION


_ACON_APPWORLD = Path("/workspace/acon/experiments/appworld")
_STRATEGIES_ROOT = _ACON_APPWORLD / "prompts" / "_motivation_v2"
_CELLS_ROOT = _ACON_APPWORLD / "prompts" / "_motivation_v3" / "_cells"

_SPLICE_MARKER = "Using these APIs, now generate code to solve the actual task:"


def _strategy_jinja_path(strategy: str) -> Path:
    return _STRATEGIES_ROOT / strategy / f"prompt_{strategy}.jinja"


def _strategy_json_path(strategy: str) -> Path:
    return _STRATEGIES_ROOT / strategy / f"prompts_{strategy}.json"


def _cell_hash(strategy: str, block: str) -> str:
    h = hashlib.sha1()
    h.update(strategy.encode())
    h.update(b"\0")
    h.update(block.encode())
    return h.hexdigest()[:16]


def _materialise_cell_prompt(strategy: str, downstream_block: str) -> Path:
    """Materialise (or look up) a per-cell jinja+json that splices
    ``downstream_block`` (a fully-formatted USER turn) right before
    the canonical splice marker."""
    if strategy not in {"direct", "verify", "explore"}:
        raise ValueError(f"Unknown strategy {strategy!r}")
    src_jinja = _strategy_jinja_path(strategy)
    if not src_jinja.exists():
        raise FileNotFoundError(
            f"Strategy jinja missing: {src_jinja}. "
            "Run motivation_v2/prompts/build_strategy_prompts.py first."
        )

    h = _cell_hash(strategy, downstream_block)
    cell_dir = _CELLS_ROOT / h
    cell_dir.mkdir(parents=True, exist_ok=True)
    out_jinja = cell_dir / "prompt.jinja"
    out_json = cell_dir / "prompts.json"

    if out_jinja.exists() and out_json.exists():
        return out_json

    src_text = src_jinja.read_text(encoding="utf-8")
    head_marker = f"USER:\n{_SPLICE_MARKER}"
    if head_marker in src_text:
        head, tail = src_text.split(head_marker, 1)
        new_text = (
            head + downstream_block + head_marker[len("USER:\n"):] + tail
        )
    else:
        head, tail = src_text.split(_SPLICE_MARKER, 1)
        new_text = head + downstream_block + _SPLICE_MARKER + tail
    out_jinja.write_text(new_text, encoding="utf-8")

    canonical = json.loads(_strategy_json_path(strategy).read_text(encoding="utf-8"))
    new_json = dict(canonical)
    rel = out_jinja.relative_to(_ACON_APPWORLD)
    new_json["main_prompt_template"] = f"./{rel}"
    out_json.write_text(json.dumps(new_json, indent=2), encoding="utf-8")
    return out_json


def build_downstream_block(
    *, compressed_context: str, max_steps: int,
) -> str:
    """Build the spec-compliant USER turn injected before the task
    instruction. Wraps DOWNSTREAM_AGENT_INSTRUCTION in USER: turns so
    acon's prompt-rendering picks it up."""
    body = DOWNSTREAM_AGENT_INSTRUCTION.format(
        compressed_context=compressed_context or "(no compressed context provided)",
        max_steps=max_steps,
    )
    return f"USER:\n{body.rstrip()}\n\nUSER:\n"


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


@dataclass
class RunResult:
    task_id: str
    method: str
    budget_max_steps: int
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


def run_with_compressed_context(
    task_id: str,
    *,
    method: str,
    compressed_context: str,
    max_steps: int,
    split: str = "dev",
    strategy: str = "direct",
    model_name: str = "MiniMaxAI/MiniMax-M2.5",
    co_config_path: str = "configs/context_opt/minimax-m25_history.yaml",
    seed: int = 42,
    tag: str = "mv3",
) -> RunResult:
    """Run one AppWorld task with the spec's downstream-agent prompt and
    a given compressed-context block."""
    block = build_downstream_block(
        compressed_context=compressed_context, max_steps=max_steps,
    )
    prompt_json = _materialise_cell_prompt(strategy, block)
    rel_prompt_file = "./" + os.path.relpath(prompt_json, _ACON_APPWORLD)

    base_yaml = _ACON_APPWORLD / "configs" / "base_config.yaml"
    if base_yaml.exists():
        import yaml
        with open(base_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    exp_id = f"{tag}_{method}_cap{max_steps}"
    cfg.update({
        "exp_id": exp_id,
        "model_name": model_name,
        "tag": exp_id,
        "max_iter": max_steps,
        "use_workflow_memory": False,
        "use_thinking_tokens": True,
        "prompt_file": rel_prompt_file,
        "co_config_path": co_config_path,
        "experiment_name": f"experiment_{exp_id}",
        "seed": seed,
        "debug_mode": False,
    })

    experiment_name = (
        f"{model_name.replace('/', '_')}_{tag}_{method}_cap{max_steps}"
    )
    output_dir = (
        _ACON_APPWORLD / "outputs" / experiment_name / split / f"task_{task_id}"
    )
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
            max_iter=max_steps,
        )
        elapsed = time.time() - t0
        token_usage = (res.get("token_usage") or {})
        return RunResult(
            task_id=task_id,
            method=method,
            budget_max_steps=max_steps,
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
        return RunResult(
            task_id=task_id,
            method=method,
            budget_max_steps=max_steps,
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
