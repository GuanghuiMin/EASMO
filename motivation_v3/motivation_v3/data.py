"""Data helpers for motivation_v3.

Mostly thin wrappers over motivation_v2's data layer (which already
parses AppWorld trajectories and ground truth) with additions needed
for this round:

  * canonical paths under outputs/  * filter to "successful full-context" trajectories given a task list
  * trajectory_text rendering with budget control
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

# Reuse motivation_v2's data loader (still installed at this path)
_MV2 = Path("/workspace/EASMO/motivation_v2")
sys.path.insert(0, str(_MV2))
from motivation_v2.data import (  # noqa: E402
    Trajectory,
    TrajectoryStep,
    iter_tasks_with_ground_truth,
    load_split_task_ids,
    load_trajectory,
    successful_trajectories,
)
from motivation_v2.policy_family import assign_policy_family  # noqa: E402


REPO = Path(__file__).resolve().parent.parent  # .../motivation_v3/
OUTPUTS = REPO / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "sprint_logs"


def ensure_outputs() -> None:
    for d in (OUTPUTS, TABLES, FIGURES, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def jsonl_path(name: str) -> Path:
    return OUTPUTS / name


# ----------------------------------------------------------------------
# Trajectory rendering
# ----------------------------------------------------------------------


def render_trajectory(
    traj: Trajectory,
    *,
    max_chars_per_step: int = 800,
    max_total_chars: int = 16000,
) -> str:
    """Plain-text rendering of (action, output) pairs for the LLM.

    Larger char budget than motivation_v2 because the spec's compression
    prompts ask for richer evidence (IDs, bindings, etc.) and need more
    of the trajectory in context.
    """
    parts: List[str] = []
    used = 0
    for s in traj.steps:
        action = (s.action or "").strip()
        output = (s.output or "").strip()
        if len(action) > max_chars_per_step:
            action = action[:max_chars_per_step] + "…[truncated]"
        if len(output) > max_chars_per_step:
            output = output[:max_chars_per_step] + "…[truncated]"
        block = f"### step {s.step}\naction:\n{action}\n\noutput:\n{output}\n"
        if used + len(block) > max_total_chars:
            parts.append("…[trajectory truncated to fit budget]")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def trajectory_step_window(
    traj: Trajectory,
    *,
    after_step: int,
    n: int = 8,
    max_chars_per_step: int = 600,
) -> str:
    """Render the first ``n`` steps after ``after_step`` (used for
    behavioral-usefulness labelling: 'future trajectory steps after
    this unit')."""
    steps = [s for s in traj.steps if s.step > after_step][: n]
    parts = []
    for s in steps:
        a = (s.action or "").strip()
        o = (s.output or "").strip()
        if len(a) > max_chars_per_step:
            a = a[:max_chars_per_step] + "…"
        if len(o) > max_chars_per_step:
            o = o[:max_chars_per_step] + "…"
        parts.append(f"### step {s.step}\naction: {a}\noutput: {o}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Successful-trajectory iterator for a given experiment tag
# ----------------------------------------------------------------------


def successful_dev_trajectories(
    *,
    tag: str,
    split: str = "dev",
    model_name: str = "MiniMaxAI/MiniMax-M2.5",
) -> List[Trajectory]:
    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"{model_name.replace('/', '_')}_{tag}/{split}/task_*"
    )
    return successful_trajectories(experiments_glob=glob)


# ----------------------------------------------------------------------
# JSONL streaming helpers
# ----------------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = [
    "Trajectory", "TrajectoryStep", "iter_tasks_with_ground_truth",
    "load_split_task_ids", "load_trajectory", "successful_trajectories",
    "assign_policy_family",
    "REPO", "OUTPUTS", "TABLES", "FIGURES", "LOGS",
    "ensure_outputs", "jsonl_path",
    "render_trajectory", "trajectory_step_window",
    "successful_dev_trajectories",
    "write_jsonl", "read_jsonl", "append_jsonl",
]
