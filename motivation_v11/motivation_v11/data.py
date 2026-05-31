"""Data layer for motivation_v11.

v11 case pool = full AppWorld dev split with baseline filter (spec §3.1, §3.3).

Per-case row schema (data/v11_primary_cases.jsonl):
{
  "case_id":               "0d8a4ee_2",
  "task_id":               "0d8a4ee_2",
  "split":                 "dev",
  "tier":                  "primary",      # full-context baseline passes
  "user_instruction":      "...",
  "full_trajectory_text":  "...",
  "trajectory_steps":      [...],
  "baseline_success":      true,
  "baseline_iterations":   12,
  "baseline_score":        1.0,
  "compression_boundary":  "full",
  "max_steps_for_continuation": 15,
  "apps_used":             ["..."],
  "n_apps":                3,
  "length_bucket":         "short|medium|long",  # spec §3.5
  "case_priority":         "...",
}

Secondary cases (all dev, including baseline-fail) live in
`data/v11_secondary_all_dev_cases.jsonl`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Set

_MV3 = Path("/workspace/EASMO/motivation_v3")
_MV2 = Path("/workspace/EASMO/motivation_v2")
sys.path.insert(0, str(_MV3))
sys.path.insert(0, str(_MV2))

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
PROVENANCE = OUTPUTS / "provenance"
RAW = OUTPUTS / "raw"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "logs"
DATA_OUT = OUTPUTS / "data"

APPWORLD_SPLITS = Path("/workspace/acon/experiments/appworld/data/datasets")


def ensure_outputs() -> None:
    for d in (DATA, OUTPUTS, PROVENANCE, RAW, TABLES, FIGURES, REPORTS,
              LOGS, DATA_OUT):
        d.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    return RAW / name


def table_path(name: str) -> Path:
    return TABLES / name


def figure_path(name: str) -> Path:
    return FIGURES / name


def data_out_path(name: str) -> Path:
    return DATA_OUT / name


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


def load_appworld_dev() -> List[str]:
    """Return sorted list of all dev task_ids."""
    p = APPWORLD_SPLITS / "dev.txt"
    if not p.exists():
        raise FileNotFoundError(f"AppWorld dev.txt missing: {p}")
    return sorted({l.strip() for l in open(p).read().splitlines() if l.strip()})


def length_bucket(baseline_iterations: int) -> str:
    if baseline_iterations < 15:
        return "short"
    if baseline_iterations < 25:
        return "medium"
    return "long"


def apps_used_from_steps(steps: list) -> list:
    apps = set()
    for s in steps:
        for m in re.finditer(r"\bapis\.([a-zA-Z0-9_]+)\.", s.get("action") or ""):
            apps.add(m.group(1))
    return sorted(apps)


__all__ = [
    "REPO", "DATA", "OUTPUTS", "PROVENANCE", "RAW", "TABLES",
    "FIGURES", "REPORTS", "LOGS", "DATA_OUT", "APPWORLD_SPLITS",
    "ensure_outputs", "raw_path", "table_path", "figure_path",
    "data_out_path", "write_jsonl", "read_jsonl", "append_jsonl",
    "sha256_text",
    "load_appworld_dev", "length_bucket", "apps_used_from_steps",
]
