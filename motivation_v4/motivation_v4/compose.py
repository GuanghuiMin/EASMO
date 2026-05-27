"""Compression-condition composers (§7).

Given a list of Spans for a task, builds the rendered context block
under a fixed token budget for each method:

  high_sensitivity_spans : argmax sensitivity-per-token
  low_sensitivity_spans  : argmin sensitivity-per-token
  recent_spans           : argmax step_id (most recent first)
  random_spans           : uniform random with given seed

All methods produce contexts in original chronological order, wrapped
in:

  [SELECTED_HISTORY_SPANS]
  [STEP <a>]...[/STEP <a>]
  ...
  [/SELECTED_HISTORY_SPANS]
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .spans import Span


_HEADER = "[SELECTED_HISTORY_SPANS]"
_FOOTER = "[/SELECTED_HISTORY_SPANS]"


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _wrap(spans_chrono: List[Span]) -> str:
    if not spans_chrono:
        return _HEADER + "\n" + _FOOTER
    body = "\n".join(s.span_text for s in spans_chrono)
    return f"{_HEADER}\n{body}\n{_FOOTER}"


def _greedy_fill_by_score(
    candidates: List[Tuple[Span, float]],
    budget_tokens: int,
    *,
    descending: bool = True,
) -> List[Span]:
    """Take spans by score order until budget is reached.

    Score is sensitivity-per-token (or its negative for low-sensitivity).
    We fill greedily by score; ties broken by original step_id (smaller first).
    """
    sorted_cands = sorted(
        candidates,
        key=lambda sx: (sx[1], -sx[0].step_id),
        reverse=descending,
    )
    chosen: List[Span] = []
    used = 0
    overhead = _approx_tokens(_HEADER) + _approx_tokens(_FOOTER) + 4
    for span, _score in sorted_cands:
        if used + span.token_count + overhead > budget_tokens:
            continue
        chosen.append(span)
        used += span.token_count
        if used + overhead >= budget_tokens:
            break
    chosen.sort(key=lambda x: x.step_id)
    return chosen


def compose_high_sensitivity(
    spans: List[Span], sensitivity: Dict[str, float], budget: int,
) -> str:
    cands = [(s, sensitivity.get(s.span_id, 0.0) / max(s.token_count, 1))
             for s in spans]
    return _wrap(_greedy_fill_by_score(cands, budget, descending=True))


def compose_low_sensitivity(
    spans: List[Span], sensitivity: Dict[str, float], budget: int,
) -> str:
    cands = [(s, sensitivity.get(s.span_id, 0.0) / max(s.token_count, 1))
             for s in spans]
    return _wrap(_greedy_fill_by_score(cands, budget, descending=False))


def compose_recent(spans: List[Span], budget: int) -> str:
    cands = [(s, float(s.step_id)) for s in spans]
    return _wrap(_greedy_fill_by_score(cands, budget, descending=True))


def compose_random(spans: List[Span], budget: int, *, seed: int = 0) -> str:
    rng = random.Random(seed)
    shuffled = list(spans)
    rng.shuffle(shuffled)
    cands = [(s, 0.0) for s in shuffled]
    # Fill in shuffled order, taking spans until budget — descending=True
    # is ignored since all scores are 0 (ties → original list order).
    chosen: List[Span] = []
    used = 0
    overhead = _approx_tokens(_HEADER) + _approx_tokens(_FOOTER) + 4
    for s in shuffled:
        if used + s.token_count + overhead > budget:
            continue
        chosen.append(s)
        used += s.token_count
        if used + overhead >= budget:
            break
    chosen.sort(key=lambda x: x.step_id)
    return _wrap(chosen)
