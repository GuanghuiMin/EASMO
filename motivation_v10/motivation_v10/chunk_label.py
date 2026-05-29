"""Chunk type labeling for v10 (spec §17.5) — MiniMax-M2.5 only.

v10 enriches v9's schema to expose both surface form and possible
function (functional_role_guess). The v9 takeaway is that surface
labels (e.g. ENTITY_LIST_ONLY) confuse form with function — many
"entity lists" are actually exact runtime bindings the next API
call needs. The v10 schema asks the labeler to declare both.

WARN_THINKING_MIN_MAX_TOKENS guard in clients.chat() enforces
max_tokens >= 1024 for MiniMax; we default to 2048 here to leave
plenty of room for thinking + structured JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .clients import chat, parse_json_object, MINIMAX_MODEL


CHUNK_LABELER_SYSTEM = (
    "You are a strict analyst of compressed tool-use agent context. "
    "Return only JSON. Distinguish surface form (how the chunk reads) "
    "from possible function (what role it would play if a tool-use "
    "agent had to act on it next)."
)


CHUNK_LABELER_USER_TEMPLATE = """\
Classify the following compressed-context chunk for a tool-use agent.

Surface labels (pick ONE primary chunk_type):
- CAUSAL_PRECONDITION: explicitly explains why a future action requires a condition, parameter, credential, or prior result.
- CONTROL_NEGATIVE_EVIDENCE: records a failed attempt, error, invalid path, or thing to avoid.
- ACTION_OUTCOME: records whether a previous action succeeded, failed, returned empty, returned objects, or changed state.
- RUNTIME_BINDING: binds an exact value (token, ID, path, email, amount, date, object set) to its role.
- ENTITY_LIST_ONLY: mostly lists entities/IDs/values without explaining use or relation.
- NARRATIVE_PROGRESS: high-level summary of progress, intent, or reasoning.
- TASK_GOAL_OR_TODO: states the goal or remaining subtask.
- OTHER: none of the above.

Boolean form/content flags (independent, may co-occur):
- contains_exact_literals: contains literal tokens, IDs, paths, dates, or amounts.
- contains_entity_list_form: visually formatted as a list/bullet of entities.
- contains_causal_relation: contains a "because" / "requires" / "in order to" style relation.
- contains_negative_evidence: explicitly records a failure, miss, or thing to avoid.
- contains_action_outcome: records the result of a previously-taken action.
- contains_runtime_binding: contains a specific value that will be re-used in a future tool call.

Functional role guess (what behavior this chunk would drive if preserved):
- api_argument_binding: this is the actual argument a future call needs.
- object_set_binding: enumerates the set of objects to iterate / operate on.
- failure_prevention: warns against an action that previously failed.
- progress_summary: tells the agent what is already done.
- task_restatement: restates the goal.
- unknown: cannot tell.

User task:
{user_instruction}

Chunk:
{chunk_text}

Return JSON ONLY:
{{
  "chunk_type": "...",
  "contains_exact_literals": true,
  "contains_entity_list_form": false,
  "contains_causal_relation": false,
  "contains_negative_evidence": false,
  "contains_action_outcome": false,
  "contains_runtime_binding": false,
  "functional_role_guess": "...",
  "confidence": 0.0,
  "one_sentence_rationale": "..."
}}
"""


_VALID_TYPES = (
    "CAUSAL_PRECONDITION",
    "CONTROL_NEGATIVE_EVIDENCE",
    "ACTION_OUTCOME",
    "RUNTIME_BINDING",
    "ENTITY_LIST_ONLY",
    "NARRATIVE_PROGRESS",
    "TASK_GOAL_OR_TODO",
    "OTHER",
)

_VALID_FUNCTIONAL_ROLES = (
    "api_argument_binding",
    "object_set_binding",
    "failure_prevention",
    "progress_summary",
    "task_restatement",
    "unknown",
)


@dataclass
class ChunkLabel:
    chunk_id: str
    labeler_model: str
    chunk_type: str
    contains_exact_literals: bool
    contains_entity_list_form: bool
    contains_causal_relation: bool
    contains_negative_evidence: bool
    contains_action_outcome: bool
    contains_runtime_binding: bool
    functional_role_guess: str
    confidence: float
    one_sentence_rationale: str
    raw_response: str = ""
    error: Optional[str] = None


def label_chunk(
    *,
    chunk_id: str,
    chunk_text: str,
    user_instruction: str,
    client=None,
    max_tokens: int = 2048,
) -> ChunkLabel:
    user = CHUNK_LABELER_USER_TEMPLATE.format(
        user_instruction=user_instruction,
        chunk_text=chunk_text,
    )
    res = chat(
        name="minimax", user=user, system=CHUNK_LABELER_SYSTEM,
        temperature=0.0, max_tokens=max_tokens, seed=42,
        client=client, json_mode=True,
    )
    obj = parse_json_object(res.text) or {}
    ctype = str(obj.get("chunk_type", "OTHER")).strip().upper()
    if ctype not in _VALID_TYPES:
        ctype = "OTHER"
    role = str(obj.get("functional_role_guess", "unknown")).strip().lower()
    if role not in _VALID_FUNCTIONAL_ROLES:
        role = "unknown"
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return ChunkLabel(
        chunk_id=chunk_id,
        labeler_model=MINIMAX_MODEL,
        chunk_type=ctype,
        contains_exact_literals=bool(obj.get("contains_exact_literals", False)),
        contains_entity_list_form=bool(obj.get("contains_entity_list_form", False)),
        contains_causal_relation=bool(obj.get("contains_causal_relation", False)),
        contains_negative_evidence=bool(obj.get("contains_negative_evidence", False)),
        contains_action_outcome=bool(obj.get("contains_action_outcome", False)),
        contains_runtime_binding=bool(obj.get("contains_runtime_binding", False)),
        functional_role_guess=role,
        confidence=confidence,
        one_sentence_rationale=str(obj.get("one_sentence_rationale", ""))[:300],
        raw_response=res.raw,
        error=res.error,
    )


__all__ = [
    "CHUNK_LABELER_SYSTEM",
    "CHUNK_LABELER_USER_TEMPLATE",
    "ChunkLabel",
    "label_chunk",
]
