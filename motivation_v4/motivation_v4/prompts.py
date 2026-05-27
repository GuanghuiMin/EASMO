"""All LLM prompts for motivation_v4 (verbatim from spec §4 and §6.2).

  * DECISION_STATE_PROBE      — §4.2  (used in stages 02, 03)
  * LLM_JUDGE_DISTANCE        — §6.2  (used in stage 04)
  * DOWNSTREAM_AGENT_PROMPT   — reused from motivation_v3 (verbatim)
"""

from __future__ import annotations


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
{{
  "active_subgoal": "...",
  "completed_actions": [
    {{
      "action": "...",
      "object": "...",
      "evidence": "..."
    }}
  ],
  "active_constraints": [
    {{
      "constraint": "...",
      "evidence": "..."
    }}
  ],
  "candidate_objects": [
    {{
      "object_id": "...",
      "object_type": "...",
      "reason": "...",
      "required_action": "..."
    }}
  ],
  "avoid_objects": [
    {{
      "object_id": "...",
      "object_type": "...",
      "reason": "..."
    }}
  ],
  "missing_information": [
    "..."
  ],
  "next_action_type": "...",
  "next_action_arguments": {{
    "arg_name": "arg_value"
  }},
  "confidence": "high | medium | low"
}}

Original task:
{task_instruction}

Context:
{context_text}
"""


LLM_JUDGE_DISTANCE = """\
You are comparing two decision-state descriptions for the same AppWorld task.

The reference decision state was inferred from the full history.
The ablated decision state was inferred after removing one history span.

Your job:
Decide whether removing the span changed the agent's decision state in a way that could affect future behavior.

Important changes include:
- different next action type;
- missing or changed next action arguments;
- missing candidate object;
- added wrong candidate object;
- missing active constraint;
- missing completed action, causing risk of repetition;
- missing avoid object, causing risk of collateral damage;
- increased uncertainty or missing information.

Return JSON only.

Output schema:
{{
  "meaningful_change": true,
  "severity": "none | low | medium | high",
  "changed_fields": [
    "next_action_type",
    "next_action_arguments",
    "candidate_objects",
    "avoid_objects",
    "active_constraints",
    "completed_actions",
    "missing_information",
    "confidence"
  ],
  "reason": "one concise explanation"
}}

Original task:
{task_instruction}

Reference decision state:
{reference_decision_state_json}

Ablated decision state:
{ablated_decision_state_json}
"""


# Reused verbatim from motivation_v3/motivation_v3/prompts.py — same
# downstream agent rules so all conditions are comparable across v3
# and v4. Spliced into the runner as a USER turn before the task.
DOWNSTREAM_AGENT_INSTRUCTION = """\
You are given compressed context from previous interaction:

{compressed_context}

Continue solving the task.

Rules:
1. Use exact IDs, values, and bindings from the compressed context when reliable.
2. If critical information is missing or ambiguous, call tools to verify it.
3. Avoid modifying unrelated objects or causing collateral damage.
4. Do not repeat completed state-changing actions unless necessary and safe.
5. Prefer fewer tool calls, but correctness is more important.
6. Stop only when the task is complete.

You have at most {max_steps} action steps.
"""
