"""Data layer for motivation_v9.

Reuses motivation_v3 successful AppWorld dev trajectories as the
case pool. v9 does **not** depend on v7/v8 fact banks or need
conditions — it is behavior-first, not retention-first.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Reuse v3/v2 trajectory loaders.
_MV3 = Path("/workspace/EASMO/motivation_v3")
_MV2 = Path("/workspace/EASMO/motivation_v2")
sys.path.insert(0, str(_MV3))
sys.path.insert(0, str(_MV2))

REPO = Path(__file__).resolve().parent.parent  # .../motivation_v9/
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
PROVENANCE = OUTPUTS / "provenance"
RAW = OUTPUTS / "raw"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "logs"


def ensure_outputs() -> None:
    for d in (DATA, OUTPUTS, PROVENANCE, RAW, TABLES, FIGURES, REPORTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    return RAW / name


def table_path(name: str) -> Path:
    return TABLES / name


def figure_path(name: str) -> Path:
    return FIGURES / name


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_v3_trajectories() -> list:
    """Return the v3-selected successful dev trajectories."""
    p = _MV3 / "outputs" / "motivation_full_trajectories.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"v3 trajectories missing at {p}")
    return read_jsonl(p)


def render_full_trajectory_text(traj, max_total_chars: int = 18000) -> str:
    from motivation_v3.data import render_trajectory  # noqa
    return render_trajectory(traj, max_total_chars=max_total_chars)


__all__ = [
    "REPO", "DATA", "OUTPUTS", "PROVENANCE", "RAW", "TABLES",
    "FIGURES", "REPORTS", "LOGS",
    "ensure_outputs", "raw_path", "table_path", "figure_path",
    "write_jsonl", "read_jsonl", "append_jsonl", "sha256_text",
    "load_v3_trajectories", "render_full_trajectory_text",
]
