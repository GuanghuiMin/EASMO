"""All LLM prompts used in the compression experiments (spec-verbatim).

Five prompt families:

  * COMPRESS_TASK_AWARE_SUMMARY    — Exp 1.A  (NL summary)
  * COMPRESS_ACON_STYLE            — Exp 1.B  (structured sections)
  * COMPRESS_SYMBOLIC_EVIDENCE     — Exp 1.C  (JSON unit list)
  * LABEL_BEHAVIORAL_USEFULNESS    — Exp 2.1  (per-unit usefulness)
  * AUDIT_COVERAGE                 — Exp 2.2  (per-unit preservation in compressed)
  * LABEL_RECOVERY_CALL            — Exp 3    (per-API-call recovery labelling)
  * DOWNSTREAM_AGENT_INSTRUCTION   — Exp 3    (system text spliced into runner)

Verbatim from compression_experiments.md to avoid drift.
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# Exp 1.A — Task-aware natural-language summary
# ----------------------------------------------------------------------

COMPRESS_TASK_AWARE_SUMMARY = """\
You are compressing a tool-use agent trajectory for a downstream agent.

Task:
{task_instruction}

Write a concise natural-language summary that helps the downstream agent continue the task.

Preserve:
- exact object IDs, file names, request IDs, order IDs, playlist IDs, thread IDs;
- entity bindings, such as which ID belongs to which person or object;
- dates, amounts, quantities, and statuses;
- completed actions and failed actions if they matter;
- task constraints and unresolved subgoals.

Do not invent facts.
Do not include irrelevant details.
Return plain text only.

Trajectory:
{trajectory_text}
"""


# ----------------------------------------------------------------------
# Exp 1.B — ACON-style structured summary
# ----------------------------------------------------------------------

COMPRESS_ACON_STYLE = """\
You are maintaining a compact state summary for a long-horizon tool-use agent.

Task:
{task_instruction}

Compress the trajectory into the following sections:

TASK STATE:
- Current goal:
- Completed subgoals:
- Remaining subgoals:

IMPORTANT IDENTIFIERS AND VALUES:
- Object IDs:
- File names or thread IDs:
- Runtime variables:
- Dates, amounts, quantities:

ENTITY AND RELATION BINDINGS:
- People/entities:
- Relations:
- Object-to-entity bindings:

ACTION OUTCOMES:
- Successful actions:
- Failed actions and fixes:

CONSTRAINTS AND RISKS:
- User constraints:
- Conditions to avoid collateral damage:

Rules:
- Preserve exact identifiers and values.
- Preserve bindings between entities and values.
- Do not invent facts.
- Be concise.
- Return plain text only.

Trajectory:
{trajectory_text}
"""


# ----------------------------------------------------------------------
# Exp 1.C — Symbolic evidence units (JSON output)
# ----------------------------------------------------------------------

COMPRESS_SYMBOLIC_EVIDENCE = """\
You are converting a tool-use agent trajectory into compact symbolic evidence units.

The downstream agent does not need a fluent summary. It needs executable evidence for future actions.

Task:
{task_instruction}

Extract atomic units that may help the agent continue or complete the task.

Keep:
- object IDs, request IDs, order IDs, playlist IDs, file names, thread IDs;
- API names and important API arguments;
- entity bindings, such as payer=Alice or relation(Alice, roommate);
- dates, amounts, quantities, and statuses;
- action outcomes;
- failed actions and fixes;
- unresolved subgoals;
- task constraints.

Rules:
1. Do not summarize in prose.
2. Use short symbolic forms such as key=value or relation triples.
3. Preserve exact values.
4. Do not add facts.
5. Split multi-fact sentences into separate units.
6. Return JSON only.

Output schema:
{{
  "units": [
    {{
      "unit_type": "object_id | entity_binding | variable_value | action_outcome | constraint | unresolved_subgoal | api_argument | date_time | amount_quantity | status | error_fix | other",
      "text": "short symbolic unit",
      "source_step": 0,
      "supporting_quote": "minimal quote or code span"
    }}
  ]
}}

Trajectory:
{trajectory_text}
"""


# ----------------------------------------------------------------------
# Exp 2.1 — Per-unit behavioral usefulness label
# ----------------------------------------------------------------------

LABEL_BEHAVIORAL_USEFULNESS = """\
You are deciding whether a symbolic context unit was behaviorally useful in a successful tool-use trajectory.

Task:
{task_instruction}

A unit is useful if removing it would likely:
- force the agent to re-query a tool;
- increase the number of steps;
- cause a wrong API argument;
- cause selection of the wrong entity/object/file/request;
- violate a task constraint;
- or cause task failure under a limited step budget.

Return JSON only.

Output schema:
{{
  "useful": true,
  "confidence": "high | medium | low",
  "used_as": "api_argument | filter_condition | entity_disambiguation | constraint_check | state_change | final_answer | planning_state | not_used",
  "reason": "one concise sentence"
}}

Candidate unit:
{unit_json}

Future trajectory steps after this unit:
{future_steps_text}
"""


# ----------------------------------------------------------------------
# Exp 2.2 — Audit a compressed context against the behavioral evidence
# ----------------------------------------------------------------------

AUDIT_COVERAGE = """\
You are checking whether a compressed context preserves behaviorally useful evidence.

Task:
{task_instruction}

For each behavioral evidence unit, decide whether it is preserved in the compressed context.

A unit is preserved only if:
- the exact value or identifier is present;
- the entity/value binding is clear;
- the constraint or action outcome is not vague;
- the fact is not distorted.

Labels:
- preserved
- dropped_identifier
- dropped_binding
- dropped_constraint
- dropped_action_outcome
- vague_or_wrong_abstraction
- distorted_or_hallucinated

Return JSON only.

Output schema:
{{
  "unit_results": [
    {{
      "unit_text": "...",
      "label": "...",
      "matched_span": "...",
      "reason": "short reason"
    }}
  ],
  "summary": {{
    "num_preserved": 0,
    "num_missing_or_bad": 0,
    "main_error_types": ["..."]
  }}
}}

Compressed context:
{compressed_context}

Behavioral evidence units:
{behavioral_evidence_units}
"""


# ----------------------------------------------------------------------
# Exp 3 — Recovery-call labelling (post-hoc per API call)
# ----------------------------------------------------------------------

LABEL_RECOVERY_CALL = """\
You are labeling whether an API call is a recovery call.

A recovery call re-fetches information that was already present in the successful full-context trajectory but was missing, vague, or unusable in the compressed context.

Return JSON only.

Output schema:
{{
  "recovery_call": true,
  "confidence": "high | medium | low",
  "reason": "one concise sentence"
}}

Compressed context:
{compressed_context}

Behavioral evidence from full trajectory:
{behavioral_evidence_units}

API call:
{api_call}

API response:
{api_response}
"""


# ----------------------------------------------------------------------
# Exp 3 — Downstream agent's spliced-in user turn (shown to AppWorld
# agent as a USER message right after the strategy block).
# ----------------------------------------------------------------------

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
