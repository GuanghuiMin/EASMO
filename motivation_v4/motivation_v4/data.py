"""Data helpers for motivation_v4.

Mostly thin wrappers over motivation_v3 (which already loads dev
trajectories and provides JSONL helpers). Adds v4 paths and the
span / probe-record dataclasses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

# Reuse v3 / v2 data layer.
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
)


REPO = Path(__file__).resolve().parent.parent  # .../motivation_v4/
OUTPUTS = REPO / "outputs"
RAW = OUTPUTS / "raw"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "sprint_logs"


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
# v3 reuse helpers
# ----------------------------------------------------------------------


def load_v3_trajectories() -> list:
    """Returns the list of v3-selected dev trajectories (rows from
    motivation_full_trajectories.jsonl)."""
    p = _MV3 / "outputs" / "motivation_full_trajectories.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"v3 trajectories not found at {p}; run v3 stage 01 first."
        )
    return read_jsonl(p)


def load_v3_compressed_contexts() -> list:
    p = _MV3 / "outputs" / "motivation_compressed_contexts.jsonl"
    return read_jsonl(p) if p.exists() else []


def load_v3_behavior_runs() -> list:
    p = _MV3 / "outputs" / "motivation_behavior_runs.jsonl"
    return read_jsonl(p) if p.exists() else []


__all__ = [
    "Trajectory", "TrajectoryStep", "load_trajectory",
    "read_jsonl", "write_jsonl", "append_jsonl",
    "REPO", "OUTPUTS", "RAW", "TABLES", "FIGURES", "REPORTS", "LOGS",
    "ensure_outputs", "raw_path", "table_path", "figure_path",
    "load_v3_trajectories",
    "load_v3_compressed_contexts",
    "load_v3_behavior_runs",
]
