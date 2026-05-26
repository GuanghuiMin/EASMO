"""Helpers for the canonical motivation-experiment outputs.

All experiments under outputs/motivation/ and figures/motivation/
follow the schemas defined in docs/user_feedback/experiment_modification.md.
This module centralises:

* the entity-token Jaccard implementation (single source of truth);
* the distributional summary helper (mean/std/median/min/max/n);
* CSV / JSONL writers;
* the canonical output paths.

Keeping it in motivation_v2/ (the package) means analyzers can import
from either venv as long as they don't pull in matplotlib (which lives
only in EASMO/.venv).
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


# ----------------------------------------------------------------------
# Canonical paths
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent  # .../motivation_v2/
OUTPUTS_DIR = REPO_ROOT / "outputs" / "motivation"
FIGURES_DIR = REPO_ROOT / "figures" / "motivation"


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Entity-token + Jaccard (single source of truth)
# ----------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{4,}")
_BRACKET_PREFIX_RE = re.compile(r"^\s*\[[^\]]*\]\s*", re.MULTILINE)

# Stopwords + role-projection scaffolding markers. The latter (intent,
# milestone, etc.) appear in role-projected oracle bracket prefixes; if
# we don't filter them, cross-role Jaccard inflates artificially.
_STOPWORDS = frozenset({
    # generic English / domain-meta stopwords
    "this", "that", "with", "from", "have", "will", "been",
    "should", "would", "could", "their", "there", "which",
    "where", "when", "what", "while", "more", "most", "some",
    "into", "about", "after", "before", "needs", "need", "memory",
    "agent", "agents", "context", "trace", "step", "steps",
    "useful", "below", "based", "your", "ours", "task",
    "tasks", "selecting", "selected", "select", "compressed",
    "compress", "calls", "call", "data", "results", "result",
    "returns", "return", "each", "give", "given", "either",
    "first", "second", "third", "true", "false", "these",
    "those", "lines", "line", "items", "item", "the", "and",
    "for", "are", "was", "were", "you", "but", "not", "all",
    "any",
    # role-projection scaffolding (only relevant if bracket-stripping
    # missed a marker word). Conservative: keep everything that's a
    # pure role-marker word, drop nothing that could be content.
    "intent", "milestone", "final",
})


def entity_tokens(
    text: str,
    *,
    strip_bracket_prefix: bool = True,
) -> Set[str]:
    """Lowercased alphanumeric tokens of length >= 4, with stopwords filtered.

    By default, strips a leading ``[anything]`` prefix from each line
    before tokenising; this removes role-projection scaffolding (e.g.
    ``[plan milestone spotify step 5]``) so cross-role comparisons
    measure content overlap rather than projection-marker overlap.
    Set ``strip_bracket_prefix=False`` for raw-text comparisons (e.g.
    when comparing prompted memory whose lines don't start with
    brackets).
    """
    s = text or ""
    if strip_bracket_prefix:
        s = _BRACKET_PREFIX_RE.sub("", s)
    return {
        t.lower() for t in _TOKEN_RE.findall(s)
        if t.lower() not in _STOPWORDS
    }


def unit_text_normalized(text: str, max_chars: int = 80) -> str:
    """Stable hashable key for unit-ID Jaccard.

    Strips bracket prefix, collapses whitespace, truncates to
    ``max_chars``. Used by the unit-ID Jaccard metric (spec §4.1
    `unit-ID Jaccard when applicable`).
    """
    s = _BRACKET_PREFIX_RE.sub("", text or "")
    s = " ".join(s.split())
    return s[:max_chars]


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------
# Distributional summary
# ----------------------------------------------------------------------

@dataclass
class DistribStats:
    n: int
    mean: float
    std: float
    median: float
    min_: float
    max_: float

    @classmethod
    def from_values(cls, xs: Sequence[float]) -> "DistribStats":
        xs = list(xs)
        if not xs:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) >= 2 else 0.0
        return cls(
            n=len(xs),
            mean=m,
            std=s,
            median=statistics.median(xs),
            min_=min(xs),
            max_=max(xs),
        )

    def to_row(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min_,
            "max": self.max_,
        }


# ----------------------------------------------------------------------
# JSONL / CSV writers
# ----------------------------------------------------------------------

def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still create an empty file with header if fieldnames given.
        if fieldnames:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return 0
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------
# Logging metadata (spec §12)
# ----------------------------------------------------------------------

@dataclass
class RunMetadata:
    script: str
    experiment: str
    executor: str = "MiniMaxAI/MiniMax-M2.5"
    benchmark: str = "appworld"
    task_split: str = "train"
    model_endpoint: str = "http://10.183.22.68:8005/v1"
    seed: Optional[int] = 42
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    git_commit: Optional[str] = None

    def __post_init__(self):
        if self.git_commit is None:
            try:
                self.git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(REPO_ROOT),
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
            except Exception:
                self.git_commit = "unknown"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def log_run_meta(experiment: str, **kwargs) -> RunMetadata:
    """Convenience: build + write run metadata to outputs/motivation/run_log.jsonl."""
    meta = RunMetadata(
        script=Path(sys.argv[0]).name if sys.argv else "?",
        experiment=experiment,
        **kwargs,
    )
    ensure_dirs()
    append_jsonl(OUTPUTS_DIR / "run_log.jsonl", meta.to_dict())
    return meta


# ----------------------------------------------------------------------
# Failure tracking (spec §12)
# ----------------------------------------------------------------------

def log_failure(
    experiment: str,
    *,
    task_id: str,
    failure_type: str,
    error_message: str,
    executor: str = "MiniMaxAI/MiniMax-M2.5",
) -> None:
    ensure_dirs()
    append_jsonl(OUTPUTS_DIR / "failures.jsonl", {
        "task_id": task_id,
        "executor": executor,
        "experiment": experiment,
        "failure_type": failure_type,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
