# motivation_v4 — Probe and Judge Prompts (paper-appendix-ready)

> Verbatim from spec §4.2 and §6.2, plus the v3-shared downstream
> agent USER turn. All three prompts are reproduced here exactly as
> sent to the MiniMax-M2.5 vLLM endpoint, so the appendix can cite
> this file as a single source of truth.

## 1. Decision-state probe (spec §4.2)

Used in Stage 02 (reference probes, full context per task) and
Stage 03 (leave-one-span-out ablation probes). The same prompt text;
only the `context_text` placeholder changes between full-history and
ablated-history renderings.

### System message

```
You are a careful analyst that follows instructions exactly.
Respond ONLY in the requested output format.
Do not include any internal reasoning, analysis, preface, or explanation.
Emit the structured JSON answer directly.
```

### User prompt

```
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
{
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
}

Original task:
{task_instruction}

Context:
{context_text}
```

### Generation hyperparameters

| Parameter | Value |
|---|---|
| temperature | 0.0 |
| max_tokens | 4096 |
| timeout | 240s |

The 4096 max_tokens is required because MiniMax-M2.5 emits a `<think>...</think>` reasoning block before the JSON; we strip the think block post-hoc.

## 2. LLM-judge distance prompt (spec §6.2)

Used in Stage 04. For each (task, span) the judge compares the
reference and ablated decision states from Stages 02/03.

### System message

Same as the probe.

### User prompt

```
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
{
  "meaningful_change": true,
  "severity": "none | low | medium | high",
  "changed_fields": [
    "next_action_type", "next_action_arguments", "candidate_objects",
    "avoid_objects", "active_constraints", "completed_actions",
    "missing_information", "confidence"
  ],
  "reason": "one concise explanation"
}

Original task:
{task_instruction}

Reference decision state:
{reference_decision_state_json}

Ablated decision state:
{ablated_decision_state_json}
```

### Severity-to-score mapping (per spec §6.2)

| severity | score |
|---|---|
| none | 0.00 |
| low | 0.25 |
| medium | 0.60 |
| high | 1.00 |

### Generation hyperparameters

| Parameter | Value |
|---|---|
| temperature | 0.0 |
| max_tokens | 1024 |
| timeout | 240s |

## 3. Downstream-agent USER turn (shared with v3)

Spliced into the AppWorld agent's main jinja prompt as a new USER
turn between the canonical strategy block and the "Using these APIs,
now generate code..." turn. Reproduced verbatim from
`motivation_v3/motivation_v3/prompts.py::DOWNSTREAM_AGENT_INSTRUCTION`
so v3 and v4 conditions are apples-to-apples comparable in the merged
Table 2 of [`05_results_summary.md`](05_results_summary.md).

```
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
```

`{max_steps}` is `15` (loose budget) or `8` (strict budget).
`{compressed_context}` is the rendered text from the condition under
test (e.g. `[SELECTED_HISTORY_SPANS]\n[STEP 4]...\n[/SELECTED_HISTORY_SPANS]`
for the span-based methods).

## 4. JSON parsing notes

Both probe and judge responses are parsed with a defensive routine
(`motivation_v4/probe.py::_parse_json_object`) that:

1. Strips `<think>...</think>` reasoning blocks (and any unclosed
   `<think>...` trailing chunk) — MiniMax-M2.5 always emits one.
2. Strips ``` ` ``json code fences.
3. Finds the largest balanced `{...}` block.
4. Tries `json.loads`. If that fails, tries successively shorter
   prefixes until parsing succeeds (recovers from cells where the
   model was truncated mid-JSON).

The probe normalises the parsed dict into a fixed schema (always 9
top-level keys; missing keys → empty list/string/dict) so downstream
distance computation has a stable structure to compare against.
