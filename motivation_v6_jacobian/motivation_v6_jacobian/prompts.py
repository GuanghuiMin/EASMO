"""Prompts for motivation_v6.

We reuse v4's DECISION_STATE_PROBE verbatim (spec §5.2 step 5) so the
white-box gradient signal targets the same probe distribution that
v4's MiniMax probe + judge measured. The target string for
teacher-forcing is the v4 reference decision state, canonicalised to
compact JSON.
"""

from __future__ import annotations

import json
from typing import Optional


# Verbatim from motivation_v4/motivation_v4/prompts.py
DECISION_STATE_PROBE = """\
You are analyzing the current decision state of a tool-use agent in AppWorld.

You will be given:
1. the original task,
2. a previous interaction history or compressed history.

Your job is NOT to solve the task.
Your job is to infer the current decision state needed for the next action.

Return JSON only.

Definitions:
- active_subgoal: what the agent should focus on next.
- completed_actions: actions that have already been completed and should not be repeated.
- active_constraints: constraints that still matter for future actions.
- candidate_objects: objects/entities/files/requests/items/messages/etc. that may be acted on next.
- avoid_objects: objects/entities/files/requests/items/messages/etc. that look related but should not be acted on.
- missing_information: information still needed before a safe next action.
- next_action_type: the type of next action likely needed.
- next_action_arguments: concrete arguments likely needed for the next action, if known.
- confidence: confidence in the inferred decision state.

Hard rules:
1. Do not invent IDs or facts not present in the context.
2. If an ID or argument is not available, use null.
3. If multiple candidates are possible, list all plausible candidates.
4. Mark completed actions clearly so that they are not repeated.
5. Mark avoid objects clearly when the context indicates they should not be touched.
6. Return valid JSON only.

Output schema:
{schema}

Original task:
{task_instruction}

Context:
{context_text}
"""

_DECISION_STATE_SCHEMA = """{
  "active_subgoal": "...",
  "completed_actions": [
    {"action": "...", "object": "...", "evidence": "..."}
  ],
  "active_constraints": [
    {"constraint": "...", "evidence": "..."}
  ],
  "candidate_objects": [
    {"object_id": "...", "object_type": "...", "reason": "...", "required_action": "..."}
  ],
  "avoid_objects": [
    {"object_id": "...", "object_type": "...", "reason": "..."}
  ],
  "missing_information": ["..."],
  "next_action_type": "...",
  "next_action_arguments": {"arg_name": "arg_value"},
  "confidence": "high | medium | low"
}"""


def build_probe_prompt(
    task_instruction: str,
    context_text: str,
) -> str:
    """Render the user-turn prompt the white-box probe sees."""
    return DECISION_STATE_PROBE.format(
        schema=_DECISION_STATE_SCHEMA,
        task_instruction=task_instruction or "(no task instruction)",
        context_text=context_text or "(empty context)",
    )


def canonicalise_target(decision_state: dict) -> str:
    """Compact JSON form used as the teacher-forcing target.

    Sort keys so that the byte-sequence is determined by the dict
    contents, not Python iteration order. ensure_ascii=False keeps
    unicode characters one-token-per-glyph for the Qwen tokenizer.
    """
    return json.dumps(decision_state, ensure_ascii=False, sort_keys=True)


__all__ = [
    "DECISION_STATE_PROBE",
    "build_probe_prompt",
    "canonicalise_target",
]
