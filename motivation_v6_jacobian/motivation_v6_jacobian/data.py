"""Data layer for motivation_v6.

Loads existing v4 outputs (history_spans, reference_decision_states,
span_sensitivity_scores, compressed_contexts, behavior_runs) and the
underlying v3-selected trajectories so we can recover task instructions
that v4 does not persist directly.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Reuse v3 / v2 trajectory loaders for task_instruction (v4 spans don't
# carry the instruction string itself).
_MV3 = Path("/workspace/EASMO/motivation_v3")
_MV2 = Path("/workspace/EASMO/motivation_v2")
sys.path.insert(0, str(_MV3))
sys.path.insert(0, str(_MV2))

REPO = Path(__file__).resolve().parent.parent  # .../motivation_v6_jacobian/
OUTPUTS = REPO / "outputs"
RAW = OUTPUTS / "raw"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "sprint_logs"

V4_RAW = Path("/workspace/EASMO/motivation_v4/outputs/raw")


def ensure_outputs() -> None:
    for d in (OUTPUTS, RAW, TABLES, FIGURES, REPORTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    return RAW / name


def table_path(name: str) -> Path:
    return TABLES / name


def figure_path(name: str) -> Path:
    return FIGURES / name


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


# ----------------------------------------------------------------------
# v4 reuse loaders
# ----------------------------------------------------------------------


def load_v4_spans_by_task() -> Dict[str, List[dict]]:
    """Returns {task_id: [span_dict_sorted_by_step_id, ...]}.

    v4's history_spans.jsonl is flat (one span per line); we group and
    sort by step_id so the spans line up chronologically with the
    rendered trajectory.
    """
    p = V4_RAW / "history_spans.jsonl"
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in read_jsonl(p):
        buckets[r["task_id"]].append(r)
    for tid in buckets:
        buckets[tid].sort(key=lambda s: s["step_id"])
    return dict(buckets)


def load_v4_reference_states() -> Dict[str, dict]:
    """Returns {task_id: decision_state_dict}."""
    p = V4_RAW / "reference_decision_states.jsonl"
    out: Dict[str, dict] = {}
    for r in read_jsonl(p):
        ds = r.get("decision_state") or {}
        if isinstance(ds, dict) and ds:
            out[r["task_id"]] = ds
    return out


def load_v4_span_sensitivities() -> Dict[str, Dict[str, dict]]:
    """Returns {task_id: {span_id: sensitivity_row, ...}}."""
    p = V4_RAW / "span_sensitivity_scores.jsonl"
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(p):
        out[r["task_id"]][r["span_id"]] = r
    return dict(out)


def load_v4_compressed_contexts() -> Dict[str, Dict[str, dict]]:
    """v4 stores per-(task, method, budget) compressed text rows.

    Returns {task_id: {method: row}}. method values include
    'recent', 'acon_baseline_uncapped', etc. We keep whatever is
    there; downstream code chooses by method-name prefix.
    """
    p = V4_RAW / "compressed_contexts.jsonl"
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(p):
        out[r["task_id"]][r.get("method", "?")] = r
    return dict(out)


# ----------------------------------------------------------------------
# v3 trajectory access (for task_instruction strings)
# ----------------------------------------------------------------------


def load_task_instructions() -> Dict[str, str]:
    """Returns {task_id: instruction_text} for the v3-selected tasks.

    Reuses the same v2 loader path that v4 stage 02 uses, so the
    instructions are guaranteed to match.
    """
    from motivation_v3.data import load_trajectory  # noqa: WPS433
    sel_path = _MV3 / "outputs" / "motivation_full_trajectories.jsonl"
    instructions: Dict[str, str] = {}
    for r in read_jsonl(sel_path):
        td = Path(r["output_dir"])
        if not td.exists():
            continue
        try:
            traj = load_trajectory(td)
        except Exception:
            continue
        instructions[traj.task_id] = traj.instruction or ""
    return instructions


# ----------------------------------------------------------------------
# Context rendering from v4 spans (chronological, with [STEP N] sentinels)
# ----------------------------------------------------------------------


def render_full_context(spans: List[dict]) -> str:
    """Concatenate v4 span_text in chronological order with a blank
    line between spans. v4 spans already begin with '[STEP N]\\n…' so
    natural step boundaries are preserved.
    """
    return "\n\n".join(s["span_text"] for s in spans)


def render_recent_context(spans: List[dict], n_tail: int = 5) -> str:
    """Last-N spans concatenation (used as the 'recent' baseline in
    soft-token experiment §7.3.3)."""
    return "\n\n".join(s["span_text"] for s in spans[-n_tail:])


__all__ = [
    "REPO", "OUTPUTS", "RAW", "TABLES", "FIGURES", "REPORTS", "LOGS",
    "V4_RAW",
    "ensure_outputs", "raw_path", "table_path", "figure_path",
    "write_jsonl", "read_jsonl", "append_jsonl",
    "load_v4_spans_by_task", "load_v4_reference_states",
    "load_v4_span_sensitivities", "load_v4_compressed_contexts",
    "load_task_instructions",
    "render_full_context", "render_recent_context",
]
