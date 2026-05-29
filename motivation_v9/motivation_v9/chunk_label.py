"""Chunk type labeling (spec §11) — MiniMax-M2.5 only.

This is purely interpretive analysis. Qwen3-4B is **explicitly
forbidden** as labeler per spec §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .clients import chat, parse_json_object, MINIMAX_MODEL


CHUNK_LABELER_SYSTEM = (
    "You are a strict analyst of compressed tool-use agent context. "
    "Return only JSON."
)


CHUNK_LABELER_USER_TEMPLATE = """\
Classify the following compressed-context chunk.

The goal is to understand what kind of information this natural-language chunk carries.
Do not judge whether it is correct. Do not invent context.

Labels:
- CAUSAL_PRECONDITION: explains why a future action requires a condition, parameter, credential, or prior result.
- CONTROL_NEGATIVE_EVIDENCE: records a failed attempt, error, invalid path, or action to avoid.
- ACTION_OUTCOME: records whether a previous action succeeded, failed, returned empty, returned objects, or changed state.
- RUNTIME_BINDING: binds an exact value such as token, ID, path, email, amount, date, or object set to its role.
- ENTITY_LIST_ONLY: mostly lists entities/IDs/values without explaining use or relation.
- NARRATIVE_PROGRESS: high-level summary of progress, intent, or reasoning.
- TASK_GOAL_OR_TODO: states the goal or remaining subtask.
- OTHER: none of the above.

Return JSON:
{{
  "chunk_type": "...",
  "contains_exact_literals": true,
  "contains_causal_relation": true,
  "contains_negative_evidence": false,
  "one_sentence_rationale": "..."
}}

Task:
{user_instruction}

Chunk:
{chunk_text}
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


@dataclass
class ChunkLabel:
    chunk_id: str
    labeler_model: str
    chunk_type: str
    contains_exact_literals: bool
    contains_causal_relation: bool
    contains_negative_evidence: bool
    one_sentence_rationale: str
    raw_response: str = ""
    error: Optional[str] = None


def label_chunk(
    *,
    chunk_id: str,
    chunk_text: str,
    user_instruction: str,
    client=None,
    max_tokens: int = 256,
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
    return ChunkLabel(
        chunk_id=chunk_id,
        labeler_model=MINIMAX_MODEL,
        chunk_type=ctype,
        contains_exact_literals=bool(obj.get("contains_exact_literals", False)),
        contains_causal_relation=bool(obj.get("contains_causal_relation", False)),
        contains_negative_evidence=bool(obj.get("contains_negative_evidence", False)),
        one_sentence_rationale=str(obj.get("one_sentence_rationale", ""))[:280],
        raw_response=res.raw,
        error=res.error,
    )


__all__ = [
    "CHUNK_LABELER_SYSTEM",
    "CHUNK_LABELER_USER_TEMPLATE",
    "ChunkLabel",
    "label_chunk",
]
