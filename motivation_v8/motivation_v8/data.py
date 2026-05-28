"""Data layer for motivation_v8.

By default, v8 reuses v7's case pool, fact bank, and need conditions —
that lets us directly compare the v7 (ACON) results against v8
(general prompts) without re-extracting facts or re-generating
counterfactual conditions.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent  # .../motivation_v8/
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
PROVENANCE = OUTPUTS / "provenance"
RAW = OUTPUTS / "raw"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "logs"

V7_DATA = Path("/workspace/EASMO/motivation_v7/data")
V7_TABLES = Path("/workspace/EASMO/motivation_v7/outputs/tables")


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


# ----------------------------------------------------------------------
# v7 reuse helpers
# ----------------------------------------------------------------------


def v7_case_pool_path() -> Optional[Path]:
    p = V7_DATA / "case_pool.jsonl"
    return p if p.exists() else None


def v7_fact_bank_path() -> Optional[Path]:
    p = V7_DATA / "fact_bank_filtered.jsonl"
    return p if p.exists() else None


def v7_need_conditions_path() -> Optional[Path]:
    p = V7_DATA / "need_conditions.jsonl"
    return p if p.exists() else None


def load_v7_cases() -> List[dict]:
    p = v7_case_pool_path()
    if p is None:
        raise FileNotFoundError(
            f"v7 case pool not found at {V7_DATA / 'case_pool.jsonl'}"
        )
    return read_jsonl(p)


def load_v7_facts() -> List[dict]:
    p = v7_fact_bank_path()
    if p is None:
        raise FileNotFoundError(
            f"v7 fact bank not found at {V7_DATA / 'fact_bank_filtered.jsonl'}"
        )
    return read_jsonl(p)


def load_v7_need_conditions(*, only_passed_quality: bool = True) -> List[dict]:
    p = v7_need_conditions_path()
    if p is None:
        raise FileNotFoundError(
            f"v7 need conditions not found at {V7_DATA / 'need_conditions.jsonl'}"
        )
    rows = read_jsonl(p)
    if only_passed_quality:
        rows = [r for r in rows if r.get("quality_passed")]
    return rows


__all__ = [
    "REPO", "DATA", "OUTPUTS", "PROVENANCE", "RAW", "TABLES", "FIGURES",
    "REPORTS", "LOGS",
    "V7_DATA", "V7_TABLES",
    "ensure_outputs",
    "raw_path", "table_path", "figure_path",
    "write_jsonl", "read_jsonl", "append_jsonl",
    "sha256_text",
    "v7_case_pool_path", "v7_fact_bank_path", "v7_need_conditions_path",
    "load_v7_cases", "load_v7_facts", "load_v7_need_conditions",
]
