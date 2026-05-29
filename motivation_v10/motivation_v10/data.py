"""Data layer for motivation_v10.

v10 extends v9's case pool to include AppWorld train.txt (89 tasks)
as the main `teacher_train` source, plus a `dev_proxy` (the v9-
non-overlap subset of dev) and a `test_behavior` slice from
`test_normal.txt`. v9's 30 dev cases stay as a warm-start cache
under `data/legacy_v9_cases.jsonl` (symlink) for SFT smoke tests.

A v10 case row:
{
  "case_id": "...",
  "task_id": "...",
  "split":   "teacher_train|dev_proxy|test_behavior|legacy_v9",
  "user_instruction": "...",
  "full_trajectory_text": "...",      # populated AFTER baseline run
  "trajectory_steps":     [...],
  "baseline_success":     true,        # from stage 01 baseline run
  "baseline_iterations":  20,
  "compression_boundary": "full",      # v10 reuses v9 full-prefix style
  "max_steps_for_continuation": 15,
  "apps_used":            [...],
  "n_apps":               3,
  "case_priority":        "...",
  "notes":                "..."
}
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

# Add motivation_v2/v3 to path for trajectory loaders (legacy helpers).
_MV3 = Path("/workspace/EASMO/motivation_v3")
_MV2 = Path("/workspace/EASMO/motivation_v2")
_MV9 = Path("/workspace/EASMO/motivation_v9")
sys.path.insert(0, str(_MV3))
sys.path.insert(0, str(_MV2))

REPO = Path(__file__).resolve().parent.parent  # .../motivation_v10/
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
PROVENANCE = OUTPUTS / "provenance"
RAW = OUTPUTS / "raw"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "logs"
SFT_DATA = OUTPUTS / "data"
MODELS = OUTPUTS / "models"

APPWORLD_SPLITS = Path("/workspace/acon/experiments/appworld/data/datasets")


def ensure_outputs() -> None:
    for d in (DATA, OUTPUTS, PROVENANCE, RAW, TABLES, FIGURES, REPORTS,
              LOGS, SFT_DATA, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    return RAW / name


def table_path(name: str) -> Path:
    return TABLES / name


def figure_path(name: str) -> Path:
    return FIGURES / name


def sft_data_path(name: str) -> Path:
    return SFT_DATA / name


def model_path(name: str) -> Path:
    return MODELS / name


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


# ---------------------------------------------------------------------
# AppWorld split + task_id helpers
# ---------------------------------------------------------------------

def load_appworld_split(name: str) -> List[str]:
    """Return sorted list of task_ids for an AppWorld split.

    name in {"train", "dev", "train_tiny", "test_normal", "test_challenge"}.
    """
    p = APPWORLD_SPLITS / f"{name}.txt"
    if not p.exists():
        raise FileNotFoundError(f"AppWorld split file missing: {p}")
    ids = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return sorted(set(ids))


def load_legacy_v9_task_ids() -> Set[str]:
    """Return the set of task_ids already covered by v9 (30 v3-dev)."""
    p = _MV9 / "data" / "v9_cases.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["task_id"] for l in open(p) if l.strip()}


def load_legacy_v9_cases() -> List[dict]:
    """Return the v9 case rows verbatim (for reuse as legacy_v9 cases)."""
    p = _MV9 / "data" / "v9_cases.jsonl"
    if not p.exists():
        return []
    rows = read_jsonl(p)
    for r in rows:
        r["split"] = "legacy_v9"
        r["compression_boundary"] = "full"
        r["max_steps_for_continuation"] = 15
    return rows


# ---------------------------------------------------------------------
# Per-task baseline trajectory loaders (from acon AppWorld output dirs)
# ---------------------------------------------------------------------

def _apps_used_from_steps(steps: list) -> list:
    apps = set()
    for s in steps:
        for m in re.finditer(r"\bapis\.([a-zA-Z0-9_]+)\.", s.get("action") or ""):
            apps.add(m.group(1))
    return sorted(apps)


def case_priority(n_steps: int) -> str:
    if n_steps >= 25:
        return "long"
    if n_steps >= 15:
        return "medium"
    return "short"


def render_full_trajectory_text(traj, max_total_chars: int = 18000) -> str:
    """Render a v3-style trajectory object to plain text."""
    from motivation_v3.data import render_trajectory  # noqa
    return render_trajectory(traj, max_total_chars=max_total_chars)


__all__ = [
    "REPO", "DATA", "OUTPUTS", "PROVENANCE", "RAW", "TABLES",
    "FIGURES", "REPORTS", "LOGS", "SFT_DATA", "MODELS",
    "APPWORLD_SPLITS",
    "ensure_outputs", "raw_path", "table_path", "figure_path",
    "sft_data_path", "model_path",
    "write_jsonl", "read_jsonl", "append_jsonl", "sha256_text",
    "load_appworld_split", "load_legacy_v9_task_ids",
    "load_legacy_v9_cases",
    "_apps_used_from_steps", "case_priority",
    "render_full_trajectory_text",
]
