"""Need-counterfactual condition generation and validation (spec §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jinja2


_NEED_CONDITION_TEMPLATE = """\
We are testing whether a compressor retains a fact because it is needed for the downstream task.

Given a fact from a trajectory, create two matched task instructions:
1. a NEEDED condition where the fact is necessary for future continuation;
2. an UNNEEDED condition where the same context can be summarized without needing that fact.

Important:
- Do not change the trajectory/history.
- Do not invent new APIs, IDs, or entities.
- Do not explicitly say "remember this fact".
- Do not quote the target fact in the task unless unavoidable.
- Keep the two task instructions similar in length and style.
- The NEEDED condition should imply the fact is needed through the next objective.
- The UNNEEDED condition should make the fact irrelevant or unnecessary.

Return JSON only:
{
  "case_id": "{{ case_id }}",
  "fact_id": "{{ fact_id }}",
  "needed_condition": {
    "condition_task": "...",
    "why_fact_is_needed": "..."
  },
  "unneeded_condition": {
    "condition_task": "...",
    "why_fact_is_not_needed": "..."
  },
  "quality_notes": {
    "mentions_fact_directly": false,
    "uses_same_context": true,
    "length_match_ok": true
  }
}

Original task:
{{ user_instruction }}

Target fact:
{{ canonical_fact }}

Fact type:
{{ fact_type }}

Source quote:
{{ source_quote }}

Short trajectory excerpt around source:
{{ local_context }}
"""

_VALIDATOR_TEMPLATE = """\
Check whether the following needed/unneeded condition pair is valid.

A valid pair:
- uses the same underlying history;
- differs mainly in downstream need;
- makes the target fact necessary in the needed condition;
- makes the target fact unnecessary in the unneeded condition;
- does not trivially instruct the compressor to copy the fact;
- has comparable task-instruction length.

Return JSON only:
{
  "valid": true,
  "needed_fact_actually_needed": true,
  "unneeded_fact_actually_unneeded": true,
  "trivially_mentions_fact": false,
  "length_match_ok": true,
  "problems": [],
  "confidence": 0.0
}

Target fact:
{{ canonical_fact }}

Needed condition:
{{ needed_condition_task }}

Unneeded condition:
{{ unneeded_condition_task }}
"""

_env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False,
                          keep_trailing_newline=True)


def render_need_prompt(
    *,
    case_id: str,
    fact_id: str,
    user_instruction: str,
    canonical_fact: str,
    fact_type: str,
    source_quote: str,
    local_context: str,
) -> str:
    return _env.from_string(_NEED_CONDITION_TEMPLATE).render(
        case_id=case_id,
        fact_id=fact_id,
        user_instruction=user_instruction,
        canonical_fact=canonical_fact,
        fact_type=fact_type,
        source_quote=source_quote,
        local_context=local_context,
    )


def render_validator_prompt(
    *,
    canonical_fact: str,
    needed_condition_task: str,
    unneeded_condition_task: str,
) -> str:
    return _env.from_string(_VALIDATOR_TEMPLATE).render(
        canonical_fact=canonical_fact,
        needed_condition_task=needed_condition_task,
        unneeded_condition_task=unneeded_condition_task,
    )


def quality_check_pair(
    *,
    canonical_fact: str,
    needed_task: str,
    unneeded_task: str,
    literal_values: List[str],
) -> Dict[str, object]:
    """Spec §8.3 quality checks (rule-based, no LLM)."""
    # Lengths in tokens — char/4 surrogate
    needed_len = len(needed_task) // 4
    unneeded_len = len(unneeded_task) // 4
    if max(needed_len, unneeded_len) == 0:
        length_match = False
    else:
        delta = abs(needed_len - unneeded_len) / max(needed_len, unneeded_len)
        length_match = delta <= 0.35

    needed_mentions = any(
        canonical_fact.lower() in needed_task.lower()
        or any(v and v.lower() in needed_task.lower() for v in literal_values if v)
        for _ in [None]
    )
    unneeded_mentions = (
        canonical_fact.lower() in unneeded_task.lower()
        or any(v and v.lower() in unneeded_task.lower() for v in literal_values if v)
    )
    return {
        "needed_condition_mentions_fact": bool(needed_mentions),
        "unneeded_condition_mentions_fact": bool(unneeded_mentions),
        "length_match_ok": bool(length_match),
        "delta_len_pct": round(
            abs(needed_len - unneeded_len) / max(needed_len, unneeded_len, 1), 3,
        ),
        "needed_chars": len(needed_task),
        "unneeded_chars": len(unneeded_task),
    }


__all__ = [
    "render_need_prompt",
    "render_validator_prompt",
    "quality_check_pair",
]
