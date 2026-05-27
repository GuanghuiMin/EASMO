# Motivation Experiments: Instructions for Coding Agent

## Goal

Run motivation-only experiments to support the following claims:

1. **Natural-language summaries are not an efficient interface for tool-use agents.**  
   Agents often need exact symbolic evidence: IDs, variable values, entity bindings, constraints, and action outcomes.

2. **Prompted compression can miss task-useful evidence.**  
   A summary may look reasonable but still drop executable details needed by the downstream agent.

3. **Compression utility should be measured behaviorally.**  
   A compressed context is useful if it helps the same fixed agent complete the task with fewer steps, fewer tokens, and fewer API calls.

---

## Scope

Use a manageable subset of AppWorld tasks for motivation.

```text
Split: dev or test-normal
Number of tasks: 20–30 successful full-context trajectories
Task types: prefer medium/hard tasks with multi-step tool use
```

Only use tasks where the full-context agent succeeds. We are not benchmarking final model performance here; we are diagnosing compression behavior.

---

# Experiment 1: Summary vs Symbolic Evidence

## Purpose

Show that symbolic evidence units preserve executable information more compactly than natural-language summaries.

## For Each Successful Trajectory, Create Three Compressed Contexts

### A. Task-aware Natural-Language Summary

**Prompt:**

```text
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
```

---

### B. ACON-style Structured Summary

**Prompt:**

```text
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
```

---

### C. Symbolic Evidence Units

Extract compact symbolic units from the trajectory.

Use this prompt after simple rule-based candidate extraction.

**Prompt:**

```text
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
{
  "units": [
    {
      "unit_type": "object_id | entity_binding | variable_value | action_outcome | constraint | unresolved_subgoal | api_argument | date_time | amount_quantity | status | error_fix | other",
      "text": "short symbolic unit",
      "source_step": 0,
      "supporting_quote": "minimal quote or code span"
    }
  ]
}

Trajectory:
{trajectory_text}
```

Then format units as:

```text
[SYMBOLIC_CONTEXT]
- ...
- ...
[/SYMBOLIC_CONTEXT]
```

---

## Metrics

For each compressed context, compute:

```text
token_count
number_of_exact_ids_preserved
number_of_entity_bindings_preserved
number_of_constraints_preserved
number_of_action_outcomes_preserved
```

## Expected Motivation Result

```text
Symbolic evidence preserves more executable details per token than summaries.
```

---

# Experiment 2: Prompted Compression Misses Behavioral Evidence

## Purpose

Show that task-aware or structured summaries still drop evidence later needed by the agent.

---

## Step 1: Derive Behavioral Evidence from the Successful Trajectory

A unit is behavioral evidence if it is reused later in:

```text
API arguments
code variables
filtering conditions
entity disambiguation
constraint checks
final answer
state-changing actions
```

Use this prompt to label candidate symbolic units.

**Prompt:**

```text
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
{
  "useful": true,
  "confidence": "high | medium | low",
  "used_as": "api_argument | filter_condition | entity_disambiguation | constraint_check | state_change | final_answer | planning_state | not_used",
  "reason": "one concise sentence"
}

Candidate unit:
{unit_json}

Future trajectory steps after this unit:
{future_steps_text}
```

Keep units labeled useful with medium/high confidence.

---

## Step 2: Audit Summaries Against Behavioral Evidence

Use this prompt.

**Prompt:**

```text
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
{
  "unit_results": [
    {
      "unit_text": "...",
      "label": "...",
      "matched_span": "...",
      "reason": "short reason"
    }
  ],
  "summary": {
    "num_preserved": 0,
    "num_missing_or_bad": 0,
    "main_error_types": ["..."]
  }
}

Compressed context:
{compressed_context}

Behavioral evidence units:
{behavioral_evidence_units}
```

---

## Metrics

For each method, compute:

```text
behavioral_evidence_coverage
identifier_coverage
binding_coverage
constraint_coverage
action_outcome_coverage
missing_evidence_error_counts
```

## Expected Motivation Result

```text
Task-aware and ACON-style summaries improve over generic summaries, but still drop IDs, bindings, constraints, or action outcomes needed for behavior.
```

---

# Experiment 3: Behavioral Utility of Compressed Context

## Purpose

Show that missing or mismatched evidence changes downstream agent behavior.

---

## Conditions

For each task, run the same downstream agent with:

```text
1. full_context
2. task_aware_summary
3. acon_style_summary
4. symbolic_evidence
5. wrong_task_symbolic_evidence
6. no_context
```

Wrong-task symbolic evidence should come from another task using the same main app/domain when possible.

---

## Downstream Agent Prompt

```text
You are a tool-use coding agent operating in AppWorld.

Original task:
{task_instruction}

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

Now produce the next reasoning step and executable Python code.
```

---

## Budgets

Run two budgets:

```text
loose: max_steps = 15
strict: max_steps = 8
```

---

## Metrics

For each condition, compute:

```text
task_success
num_steps
total_input_tokens
peak_input_tokens
api_call_count
recovery_api_call_count
failure_reason
```

Recovery API call approximation:

```text
An API call is a recovery call if it re-fetches information already present in the full successful trajectory.
```

Use this labeling prompt if needed.

**Prompt:**

```text
You are labeling whether an API call is a recovery call.

A recovery call re-fetches information that was already present in the successful full-context trajectory but was missing, vague, or unusable in the compressed context.

Return JSON only.

Output schema:
{
  "recovery_call": true,
  "confidence": "high | medium | low",
  "reason": "one concise sentence"
}

Compressed context:
{compressed_context}

Behavioral evidence from full trajectory:
{behavioral_evidence_units}

API call:
{api_call}

API response:
{api_response}
```

---

## Expected Motivation Result

```text
Under loose budget, bad compression mainly increases recovery calls, steps, and tokens.
Under strict budget, bad compression causes success drops.
Symbolic evidence should reduce recovery cost compared with summaries.
Wrong-task evidence should look superficially relevant but hurt behavior.
```

---

# Required Outputs

## JSONL Files

Save one JSONL file per stage:

```text
motivation_full_trajectories.jsonl
motivation_symbolic_units.jsonl
motivation_behavioral_evidence.jsonl
motivation_compressed_contexts.jsonl
motivation_behavior_runs.jsonl
motivation_audits.jsonl
```

---

## Tables

Generate three tables.

### Table 1: Summary vs Symbolic Compactness

Columns:

```text
method
avg_tokens
avg_ids_preserved
avg_bindings_preserved
avg_constraints_preserved
avg_action_outcomes_preserved
```

---

### Table 2: Behavioral Evidence Coverage

Columns:

```text
method
behavioral_evidence_coverage
identifier_coverage
binding_coverage
constraint_coverage
action_outcome_coverage
top_missing_error_type
```

---

### Table 3: Behavioral Utility

Columns:

```text
method
budget
success_rate
avg_steps
avg_peak_tokens
avg_total_input_tokens
avg_api_calls
avg_recovery_calls
```

---

## Figures

Generate three figures:

```text
fig_compactness_vs_evidence_coverage.pdf
fig_budgeted_success.pdf
fig_recovery_calls_by_method.pdf
```

---

# Required Final Report

Generate a markdown report:

```text
motivation_results.md
```

The report should include the following sections.

```markdown
# Motivation Experiment Results

## Setup

- AppWorld split:
- Number of tasks attempted:
- Number of successful full-context trajectories used:
- Downstream agent model:
- Compressor model:
- Budgets:
- Compression methods:

## Claim 1: Natural-Language Summaries Are Not an Efficient Interface

Summarize Table 1 and `fig_compactness_vs_evidence_coverage.pdf`.

Required content:
- Compare average token counts.
- Compare executable evidence preserved per method.
- State whether symbolic evidence preserves more IDs, bindings, constraints, or action outcomes per token.

## Claim 2: Prompted Compression Misses Behavioral Evidence

Summarize Table 2.

Required content:
- Compare behavioral evidence coverage across task-aware summary, ACON-style summary, and symbolic evidence.
- Report the most common missing evidence types.
- Include 1–2 qualitative examples if available.

## Claim 3: Compression Utility Is Behavioral

Summarize Table 3, `fig_budgeted_success.pdf`, and `fig_recovery_calls_by_method.pdf`.

Required content:
- Compare success under loose and strict budgets.
- Compare recovery API calls.
- State whether bad compression first appears as efficiency loss and then as success loss under strict budget.

## Key Aggregate Numbers

Include a compact markdown table:

| Method | Avg Tokens | Evidence Coverage | Success@15 | Success@8 | Recovery Calls@15 |
|---|---:|---:|---:|---:|---:|
| Task-Aware Summary | | | | | |
| ACON-Style Summary | | | | | |
| Symbolic Evidence | | | | | |

## Representative Examples

Include 2–3 examples where symbolic evidence helps.

For each example:
- task_id
- task instruction
- evidence needed
- what summary missed
- downstream behavior difference

## Failure or Inconclusive Cases

Include 1–2 cases where symbolic evidence does not help or where all methods fail.

For each case:
- task_id
- what went wrong
- likely reason

## Interpretation for Paper

Write one concise paragraph explaining how the results support the Motivation section.

## Caveats

Mention:
- full-context failures are excluded;
- behavioral evidence attribution is proxy-based;
- recovery calls may require heuristic or LLM labeling;
- this is a motivation experiment, not the final method benchmark.
```

---

# Important Notes

This is **not** the full method training experiment.

Do not implement:

```text
RL training
selector training
full benchmark comparison
large-scale ablations
```

The purpose is only to produce motivation evidence showing:

```text
natural-language summaries are inefficient,
prompted compression drops behavioral evidence,
and compression utility should be evaluated by downstream behavior.
```