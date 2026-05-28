"""Iterative-chain initialization builders for the basin experiment (spec §11).

Four initializations:
  * RAW_FULL          : the original trajectory text (≤18K chars)
  * DETAIL_HEAVY      : fact table prepended to the raw trajectory
  * NARRATIVE_HEAVY   : generic narrative summary (P2 one-shot) +
                        last 30 % of the trajectory
  * FACT_TABLE_ONLY   : only the fact table; no raw trajectory

No ACON-style headings. The fact-table format is plain bullet text.
"""

from __future__ import annotations

from typing import Dict, List, Optional


HISTORY_CHAR_CAP = 18_000


def _fact_table(facts: List[dict]) -> str:
    lines = ["Known facts extracted from the trajectory:"]
    for f in facts:
        fid = f.get("fact_id", "?")
        ftype = f.get("fact_type", "?")
        canonical = (f.get("canonical_fact") or "").strip()
        lines.append(f"- [FACT_ID={fid}][TYPE={ftype}] {canonical}")
    return "\n".join(lines)


def _truncate_full(text: str, cap: int = HISTORY_CHAR_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap]


def build_raw_full(*, full_trajectory_text: str) -> str:
    return _truncate_full(full_trajectory_text)


def build_detail_heavy(
    *,
    full_trajectory_text: str,
    facts: List[dict],
) -> str:
    table = _fact_table(facts)
    body = _truncate_full(full_trajectory_text)
    # Leave 600 chars of headroom so the combined block still fits a
    # comparable budget envelope to RAW_FULL.
    text = f"{table}\n\nOriginal trajectory:\n{body}"
    # If combined too long, trim the trajectory.
    if len(text) > HISTORY_CHAR_CAP + len(table) + 50:
        remaining = HISTORY_CHAR_CAP - len(table) - 50
        body = full_trajectory_text[:max(0, remaining)]
        text = f"{table}\n\nOriginal trajectory:\n{body}"
    return text


def build_narrative_heavy(
    *,
    full_trajectory_text: str,
    generic_summary: str,
) -> str:
    """Caller is responsible for having already produced ``generic_summary``
    via a single P2 (general_task_agnostic) call on the raw trajectory."""
    if len(full_trajectory_text) <= HISTORY_CHAR_CAP:
        body = full_trajectory_text
    else:
        body = full_trajectory_text
    tail_n = max(1, len(body) // 10 * 3)  # last 30 %
    tail = body[-tail_n:]
    text = f"Narrative overview:\n{generic_summary}\n\nRecent trajectory tail:\n{tail}"
    return _truncate_full(text)


def build_fact_table_only(*, facts: List[dict]) -> str:
    return _fact_table(facts)


__all__ = [
    "HISTORY_CHAR_CAP",
    "build_raw_full",
    "build_detail_heavy",
    "build_narrative_heavy",
    "build_fact_table_only",
]
