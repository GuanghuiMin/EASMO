"""Data helpers for motivation_v5.

Builds the spec §4.1 case schema from motivation_v3's existing data
(30 dev tasks × 2 budgets, with full_context = baseline and
acon_style_summary = ACON-style compressed history).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Reuse v3 data layer (which itself reuses v2).
_MV3 = Path("/workspace/EASMO/motivation_v3")
_MV2 = Path("/workspace/EASMO/motivation_v2")
sys.path.insert(0, str(_MV3))
sys.path.insert(0, str(_MV2))

from motivation_v3.data import (  # noqa: E402
    Trajectory,
    TrajectoryStep,
    load_trajectory,
    read_jsonl,
    write_jsonl,
    append_jsonl,
    render_trajectory,
)


REPO = Path(__file__).resolve().parent.parent  # .../motivation_v5/
OUTPUTS = REPO / "outputs"
RAW = OUTPUTS / "raw"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
PER_CASE = REPORTS / "per_case_markdown"
DATA = REPO / "data"
PROMPTS = REPO / "prompts"
LOGS = OUTPUTS / "sprint_logs"


def ensure_dirs() -> None:
    for d in (OUTPUTS, RAW, TABLES, FIGURES, REPORTS, PER_CASE, DATA, LOGS):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Build raw cases from v3 data
# ----------------------------------------------------------------------


def _difficulty_bucket(iters: int) -> str:
    if iters <= 7:
        return "easy"
    if iters <= 15:
        return "medium"
    return "hard"


def build_raw_cases(
    *,
    baseline_method: str = "full_context",
    acon_method: str = "acon_style_summary",
    budgets: tuple = (15, 8),
    max_chars: int = 12000,
) -> List[dict]:
    """Returns one case-record per (task, budget) per spec §4.1 schema.

    Pulls baseline_history from v3 stage-01 trajectories (full_context),
    acon_compressed_history from v3 compressed_contexts (acon_style),
    acon_full_trajectory from v3 stage-05 behavior_runs.
    """
    sel_path = _MV3 / "outputs" / "motivation_full_trajectories.jsonl"
    compressed_path = _MV3 / "outputs" / "motivation_compressed_contexts.jsonl"
    runs_path = _MV3 / "outputs" / "motivation_behavior_runs.jsonl"

    if not sel_path.exists() or not compressed_path.exists() or not runs_path.exists():
        raise FileNotFoundError(
            f"Missing v3 outputs at {sel_path.parent}; run v3 stages 01-05 first."
        )

    sel = read_jsonl(sel_path)
    compressed = read_jsonl(compressed_path)
    runs = read_jsonl(runs_path)

    # Index compressed by (task, method)
    by_tm: Dict[tuple, str] = {}
    for c in compressed:
        if c.get("error"):
            continue
        by_tm[(c["task_id"], c["method"])] = c.get("text", "")

    # Index baseline trajectories (full_context behavior runs at cap=15 and cap=8
    # are the "baseline" for spec; we use the v3-stage-01 successful direct trajectory
    # text as baseline_history (rendered), since the v3 full_context runs at cap=15/8
    # are subsequent re-runs)
    baseline_text_by_task: Dict[str, str] = {}
    baseline_steps_by_task: Dict[str, int] = {}
    baseline_success_by_task: Dict[str, bool] = {}
    for r in sel:
        td = Path(r["output_dir"])
        if not td.exists():
            continue
        try:
            traj = load_trajectory(td)
        except Exception:
            continue
        baseline_text_by_task[traj.task_id] = render_trajectory(
            traj, max_total_chars=max_chars,
        )
        baseline_steps_by_task[traj.task_id] = traj.iterations
        baseline_success_by_task[traj.task_id] = True  # v3 only kept successes

    # Index ACON runs by (task, budget)
    acon_by_tb: Dict[tuple, dict] = {}
    for r in runs:
        if r.get("error"):
            continue
        if r.get("method") != acon_method:
            continue
        cap = int(r.get("budget_max_steps", 0))
        acon_by_tb[(r["task_id"], cap)] = r

    # Render ACON trajectory text from the output_dir's env_history
    def _render_acon_trajectory(run: dict) -> str:
        env_path = Path(run.get("output_dir", "")) / "env_history.json"
        if not env_path.exists():
            return ""
        try:
            steps = json.loads(env_path.read_text())
        except Exception:
            return ""
        parts: List[str] = []
        used = 0
        for s in steps:
            action = (s.get("action") or "").strip()
            output = (s.get("output") or "").strip()
            if len(action) > 600:
                action = action[:600] + "…[truncated]"
            if len(output) > 600:
                output = output[:600] + "…[truncated]"
            block = (f"### step {s.get('step', '?')}\n"
                     f"action:\n{action}\n\noutput:\n{output}\n")
            if used + len(block) > max_chars:
                parts.append("…[ACON trajectory truncated to fit budget]")
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts)

    cases: List[dict] = []
    for r in sel:
        tid = r["task_id"]
        if tid not in baseline_text_by_task:
            continue
        acon_compressed_text = by_tm.get((tid, acon_method), "")
        baseline_history = baseline_text_by_task[tid]
        baseline_env_steps = baseline_steps_by_task[tid]

        for cap in budgets:
            acon_run = acon_by_tb.get((tid, cap))
            if acon_run is None:
                continue
            acon_full_trajectory = _render_acon_trajectory(acon_run)
            acon_success = bool(acon_run.get("success", False))
            acon_env_steps = int(acon_run.get("iterations", 0))
            step_ratio = (acon_env_steps / baseline_env_steps
                          if baseline_env_steps > 0 else 0.0)
            cases.append({
                "task_id": tid,
                "case_id": f"{tid}_cap{cap}",
                "task_name": "appworld_dev_" + tid,
                "difficulty": _difficulty_bucket(baseline_env_steps),
                "user_instruction": _load_task_instruction(tid),
                "baseline_success": True,
                "acon_success": acon_success,
                "baseline_env_steps": baseline_env_steps,
                "acon_env_steps": acon_env_steps,
                "step_ratio": round(step_ratio, 3),
                "baseline_history": baseline_history,
                "acon_compressed_history": acon_compressed_text,
                "acon_full_trajectory": acon_full_trajectory,
                "audit_augmented_context": None,        # filled by stage 03
                "recompressed_context": None,           # filled by stage 04
                "final_after_recompression_success": None,  # filled by stage 05
                "failure_report": (acon_run.get("termination_reason", "")
                                   if not acon_success else ""),
                "compression_type": "history",
                "acon_variant": "prompting",  # we use v3's acon_style prompt
                "agent_model": "MiniMaxAI/MiniMax-M2.5",
                "compressor_model": "MiniMaxAI/MiniMax-M2.5",
                "audit_model": "qwen3-4b",
                "verifier_model": "MiniMaxAI/MiniMax-M2.5",
                "budget_max_steps": cap,
                "acon_output_dir": acon_run.get("output_dir", ""),
            })
    return cases


def _load_task_instruction(task_id: str) -> str:
    """AppWorld task instructions live under
    /workspace/acon/experiments/appworld/data/tasks/<task_id>/ground_truth/."""
    p = Path(f"/workspace/acon/experiments/appworld/data/tasks/{task_id}/ground_truth/metadata.json")
    if not p.exists():
        # Fallback: pull from baseline trajectory if available
        return ""
    try:
        d = json.loads(p.read_text())
        return str(d.get("instruction") or "").strip()
    except Exception:
        return ""


# ----------------------------------------------------------------------
# Tier filtering (spec §4.2)
# ----------------------------------------------------------------------


def filter_tiers(cases: List[dict], *, step_ratio_threshold: float = 1.5) -> dict:
    """Returns a dict with keys 'tier1', 'tier2', 'all', and 'sampled'
    (= union of Tier 1 + Tier 2, dedup by case_id)."""
    tier1 = [c for c in cases
             if c["baseline_success"] and not c["acon_success"]]
    tier2 = [c for c in cases
             if c["baseline_success"] and c["acon_success"]
             and c["step_ratio"] >= step_ratio_threshold]
    sampled_ids = set()
    sampled: List[dict] = []
    for c in tier1 + tier2:
        if c["case_id"] in sampled_ids:
            continue
        sampled_ids.add(c["case_id"])
        sampled.append(c)
    return {
        "tier1": tier1,
        "tier2": tier2,
        "all": cases,
        "sampled": sampled,
    }


__all__ = [
    "Trajectory", "TrajectoryStep", "load_trajectory",
    "read_jsonl", "write_jsonl", "append_jsonl",
    "render_trajectory",
    "REPO", "OUTPUTS", "RAW", "TABLES", "FIGURES", "REPORTS",
    "PER_CASE", "DATA", "PROMPTS", "LOGS",
    "ensure_dirs",
    "build_raw_cases", "filter_tiers",
]
