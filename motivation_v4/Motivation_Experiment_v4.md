# Motivation Experiment v4: Decision-State Sensitivity for Agent Context Compression
## Goal
Run a motivation experiment to test the following claim:
> For LLM agents, context importance should be measured by whether a history span changes the downstream agent's decision state, not by textual salience, evidence coverage, or summary completeness.
> This experiment is **not** about designing a better summary prompt or symbolic context format.  
> It is about testing whether we can identify decision-relevant history spans by probing the downstream agent itself.
---
## Key Hypothesis
A history span is important if removing it changes the downstream agent's inferred decision state.
Formally, for a full history \(h\), a candidate span \(u_i\), and a fixed downstream agent \(\pi\), define:
\[
I(u_i; h, \pi) = d(q_\pi(h), q_\pi(h \setminus u_i)),
\]
where \(q_\pi(\cdot)\) is a structured decision-state probe and \(d\) measures disagreement.
We expect high-sensitivity spans to be more useful for downstream continuation than low-sensitivity spans, recent spans, random spans, or structurally high-coverage symbolic units.
---
# 1. Scope
Use AppWorld dev split.
Use the same 30 successful full-context trajectories from the previous motivation run if available.
Input files from previous run may be reused:
```text
outputs/motivation/raw/motivation_full_trajectories.jsonl
outputs/motivation/raw/motivation_compressed_contexts.jsonl
outputs/motivation/raw/motivation_behavior_runs.jsonl

If these files are unavailable, rerun full-context agent on AppWorld dev and keep 20–30 successful trajectories.

Do not train a model.
Do not implement RL.
Do not tune prompt formats.

This is a diagnostic motivation experiment.

⸻

2. Fixed Models and Agent

Use the same downstream model as the previous run:

Model: MiniMaxAI/MiniMax-M2.5
Temperature: 0
Environment: AppWorld

Use the same fixed downstream agent prompt for all continuation runs.

Do not change the downstream agent across methods.

⸻

3. Candidate History Spans

For each successful full-context trajectory, split the history into coarse spans.

Preferred span granularity:

one span = one action-observation step

Each span should contain:

step_id
agent thought if available
action code
API calls
observation
error if any

Example span format:

[STEP 4]
Thought:
...
Action:
...
Observation:
...
[/STEP 4]

Save spans to:

outputs/motivation_v4/raw/history_spans.jsonl

Each record:

{
  "task_id": "...",
  "span_id": "step_04",
  "step_id": 4,
  "span_text": "...",
  "token_count": 123
}

⸻

4. Decision-State Probe

4.1 Purpose

The probe asks the same downstream model to infer its current decision state from a given context.

This probe is only a measurement tool.
It is not the compressed context format used by the final method.

⸻

4.2 Decision-State Probe Prompt

Use this exact prompt.

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
    {
      "action": "...",
      "object": "...",
      "evidence": "..."
    }
  ],
  "active_constraints": [
    {
      "constraint": "...",
      "evidence": "..."
    }
  ],
  "candidate_objects": [
    {
      "object_id": "...",
      "object_type": "...",
      "reason": "...",
      "required_action": "..."
    }
  ],
  "avoid_objects": [
    {
      "object_id": "...",
      "object_type": "...",
      "reason": "..."
    }
  ],
  "missing_information": [
    "..."
  ],
  "next_action_type": "...",
  "next_action_arguments": {
    "arg_name": "arg_value"
  },
  "confidence": "high | medium | low"
}
Original task:
{task_instruction}
Context:
{context_text}

⸻

5. Full-Context Reference Decision State

For each task, run the decision-state probe on the full rendered trajectory context.

Save output to:

outputs/motivation_v4/raw/reference_decision_states.jsonl

Each record:

{
  "task_id": "...",
  "context_type": "full_context",
  "decision_state": {...}
}

This reference decision state is the target used to measure sensitivity.

Important caveat:
The reference is not a gold oracle. It is the downstream model’s own decision-state interpretation under full context.

⸻

6. Leave-One-Span-Out Sensitivity

For each task and each span:

1. Remove that span from the history.
2. Run the same decision-state probe.
3. Compare the ablated decision state against the full-context reference.
4. Assign a sensitivity score to the removed span.

Save outputs to:

outputs/motivation_v4/raw/span_ablation_probes.jsonl
outputs/motivation_v4/tables/span_sensitivity_scores.csv

⸻

6.1 Decision-State Distance

Compute a distance score between reference state and ablated state.

Use both rule-based and LLM-judge comparison.

Rule-based Components

For each ablation, compute:

next_action_type_changed: 0/1
next_action_arguments_f1_loss: 0 to 1
candidate_objects_f1_loss: 0 to 1
avoid_objects_f1_loss: 0 to 1
active_constraints_f1_loss: 0 to 1
completed_actions_f1_loss: 0 to 1
missing_information_increase: 0/1
confidence_drop: 0/1

Suggested weighted sensitivity score:

sensitivity =
  2.0 * next_action_type_changed
+ 2.0 * next_action_arguments_f1_loss
+ 1.5 * candidate_objects_f1_loss
+ 1.5 * active_constraints_f1_loss
+ 1.0 * avoid_objects_f1_loss
+ 1.0 * completed_actions_f1_loss
+ 1.0 * missing_information_increase
+ 0.5 * confidence_drop

Normalize to 0–1 per task if needed.

⸻

6.2 LLM-Judge Decision-State Comparison Prompt

Use this prompt to judge whether the ablation changed the decision state in a behaviorally meaningful way.

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
}
Original task:
{task_instruction}
Reference decision state:
{reference_decision_state_json}
Ablated decision state:
{ablated_decision_state_json}

Map severity to score:

none = 0.0
low = 0.25
medium = 0.6
high = 1.0

Final span sensitivity can be:

final_sensitivity = 0.5 * rule_based_score + 0.5 * llm_judge_score

⸻

7. Construct Compression Conditions

For each task, construct the following compressed contexts under a matched token budget.

Use the same token budget for span-based methods:

budget = average token count of task_aware_summary for that task

If task-aware summary is unavailable, use:

budget = 400 tokens

⸻

7.1 High-Sensitivity Span Context

Select spans with highest sensitivity-per-token until budget is reached.

method = high_sensitivity_spans
score = final_sensitivity / token_count

Render selected spans in original chronological order.

Format:

[SELECTED_HISTORY_SPANS]
[STEP 2]
...
[/STEP 2]
[STEP 5]
...
[/STEP 5]
[/SELECTED_HISTORY_SPANS]

⸻

7.2 Low-Sensitivity Span Context

Select spans with lowest sensitivity-per-token until the same budget is reached.

method = low_sensitivity_spans

Render selected spans in original chronological order.

Purpose:
Negative control. If sensitivity is meaningful, low-sensitivity spans should perform worse.

⸻

7.3 Recent Span Context

Select the most recent spans until budget is reached.

method = recent_spans

Purpose:
Baseline for recency heuristic.

⸻

7.4 Random Span Context

Randomly select spans until budget is reached.

method = random_spans

Use 3 random seeds if budget allows.

random_spans_seed1
random_spans_seed2
random_spans_seed3

Report mean and standard deviation.

⸻

7.5 Task-Aware Summary

Reuse the previous task-aware summary condition.

method = task_aware_summary

⸻

7.6 ACON-Style Summary

Reuse the previous ACON-style summary condition.

method = acon_style_summary

⸻

7.7 Truncated Full Context

Use the previous 12K-char rendered full trajectory condition.

method = truncated_full_context

⸻

8. Downstream Behavior Evaluation

Run the same downstream agent continuation for each condition.

Conditions:

high_sensitivity_spans
low_sensitivity_spans
recent_spans
random_spans_seed1
random_spans_seed2
random_spans_seed3
task_aware_summary
acon_style_summary
truncated_full_context
no_context

Budgets:

loose: max_steps = 15
strict: max_steps = 8

Use the same downstream agent prompt as previous experiments.

Save runs to:

outputs/motivation_v4/raw/behavior_runs.jsonl

Each record:

{
  "task_id": "...",
  "method": "...",
  "budget": "loose_15 | strict_8",
  "success": true,
  "score": 1.0,
  "num_steps": 0,
  "peak_input_tokens": 0,
  "total_input_tokens": 0,
  "api_call_count": 0,
  "failure_reason": null,
  "trajectory": []
}

⸻

9. Main Questions to Answer

The final analysis should answer these questions.

Q1. Does decision-state sensitivity predict downstream behavior better than structural coverage?

Compare:

span_sensitivity
behavioral success
steps
API calls

against previous static metrics:

evidence coverage
identifier coverage
token count
recency

Expected result:
High-sensitivity spans should outperform low-sensitivity and random spans.

⸻

Q2. Are high-sensitivity spans more useful than recent spans?

Compare:

high_sensitivity_spans
recent_spans
random_spans

Expected result:
High-sensitivity spans should outperform recent spans if decision-state sensitivity captures non-recency information.

⸻

Q3. Can extractive span selection compete with summaries without designing a new prompt format?

Compare:

high_sensitivity_spans
task_aware_summary
acon_style_summary

Expected result:
Even if summaries remain strong, high-sensitivity spans should close the gap or outperform under some budgets/tasks.

Important:
The main claim does not require high-sensitivity spans to beat all summaries.
The key claim is that decision-state sensitivity provides a better selection signal than static coverage or recency.

⸻

Q4. Does low-sensitivity context behave like no context?

Compare:

low_sensitivity_spans
random_spans
no_context

Expected result:
Low-sensitivity spans should be close to random or no-context conditions.

⸻

10. Required Tables

Save tables under:

outputs/motivation_v4/tables/

⸻

Table 1: Span Sensitivity Statistics

File:

table_span_sensitivity_stats.csv

Columns:

task_id
num_spans
avg_sensitivity
max_sensitivity
min_sensitivity
sensitivity_entropy
top_span_step_ids
top_span_scores

Purpose:
Show that decision-state sensitivity is non-uniform across history spans.

⸻

Table 2: Behavior by Compression Method

File:

table_behavior_by_method.csv

Columns:

method
budget
num_tasks
success_rate
avg_score
avg_steps
avg_peak_tokens
avg_total_input_tokens
avg_api_calls

Rows:

high_sensitivity_spans
low_sensitivity_spans
recent_spans
random_spans_mean
task_aware_summary
acon_style_summary
truncated_full_context
no_context

⸻

Table 3: Sensitivity vs Static Metrics

File:

table_sensitivity_vs_static_metrics.csv

Columns:

metric
correlation_with_success
correlation_with_score
correlation_with_steps_negative
correlation_with_api_calls_negative

Metrics to include:

decision_state_sensitivity
recency_score
token_count
previous_evidence_coverage_if_available
identifier_coverage_if_available

Purpose:
Show whether decision-state sensitivity is more predictive of downstream behavior than static metrics.

⸻

Table 4: Top Span Case Studies

File:

table_top_span_case_studies.csv

Columns:

task_id
top_span_id
top_span_text_short
sensitivity_score
changed_decision_fields
why_it_matters
high_sensitivity_success
recent_success
summary_success

Purpose:
Provide qualitative examples for the paper.

⸻

11. Required Figures

Save figures under:

outputs/motivation_v4/figures/

Save both .pdf and .png.

⸻

Figure 1: Sensitivity Distribution

Files:

fig_sensitivity_distribution.pdf
fig_sensitivity_distribution.png

Plot:
Histogram or violin plot of span sensitivity scores.

Main message:
Decision-state sensitivity is sparse/non-uniform; not all history spans matter equally.

⸻

Figure 2: Budgeted Success by Method

Files:

fig_budgeted_success_by_method.pdf
fig_budgeted_success_by_method.png

Plot:
Grouped bar chart.

x-axis:
method

y-axis:
success rate

groups:
loose_15 and strict_8

Methods:

high_sensitivity_spans
low_sensitivity_spans
recent_spans
random_spans_mean
task_aware_summary
acon_style_summary
truncated_full_context
no_context

Main message:
High-sensitivity spans should outperform low/random/recent spans.

⸻

Figure 3: Sensitivity vs Behavior

Files:

fig_sensitivity_vs_behavior.pdf
fig_sensitivity_vs_behavior.png

Plot:
Scatter plot.

x-axis:
average selected span sensitivity

y-axis:
task score or success probability

One point per task-method pair for span-selection methods.

Main message:
Decision-state sensitivity should correlate with downstream behavior.

⸻

Figure 4: Sensitivity vs Recency

Files:

fig_sensitivity_vs_recency.pdf
fig_sensitivity_vs_recency.png

Plot:
Scatter plot.

x-axis:
span recency rank

y-axis:
span sensitivity score

Main message:
Important spans are not always the most recent spans.

⸻

12. Required Markdown Report

Generate:

outputs/motivation_v4/reports/decision_probe_results.md

Required structure:

# Decision-State Sensitivity Motivation Results
## Setup
- AppWorld split:
- Number of tasks:
- Downstream model:
- Number of spans:
- Probe model:
- Budgets:
- Methods:
## Finding 1: Structural Coverage Is Not Enough
Briefly summarize previous v3 result:
- symbolic evidence had highest structural coverage but lowest behavioral success.
- This motivates decision-state sensitivity.
## Finding 2: Decision-State Sensitivity Is Non-Uniform
Summarize Table 1 and Figure 1.
State whether only a small subset of spans strongly changes the decision state.
## Finding 3: High-Sensitivity Spans Improve Downstream Behavior
Summarize Table 2 and Figure 2.
Compare high-sensitivity spans against low-sensitivity, random, recent, no-context, and summaries.
## Finding 4: Sensitivity Is Not Just Recency
Summarize Figure 4.
State whether high-sensitivity spans are often not the most recent spans.
## Finding 5: Sensitivity Predicts Behavior Better Than Static Metrics
Summarize Table 3 and Figure 3.
Compare decision-state sensitivity against token count, recency, and previous evidence coverage.
## Representative Examples
Include 3 examples.
For each:
- task_id
- task instruction
- top sensitive span
- decision-state fields changed when removed
- why this span matters
- downstream behavior comparison
## Failure Cases
Include 2 examples where high-sensitivity spans did not help.
Explain whether failure was due to:
- bad probe output
- incomplete span granularity
- downstream agent error
- budget too strict
- task requires information spread across many spans
## Interpretation for Paper
Write one paragraph:
The key message is not that a particular context format is best. The key message is that context importance should be measured by its effect on the downstream agent's decision state.
## Caveats
Mention:
- decision-state probe is a proxy, not ground truth;
- probe uses the same or similar model as downstream agent;
- span-level ablation is more expensive than static compression;
- full policy sensitivity is approximated by decision-state sensitivity;
- experiment is motivation/diagnostic, not final benchmark.

⸻

13. Success Criteria

This experiment is useful if at least two of the following hold:

1. High-sensitivity spans outperform low-sensitivity spans.
2. High-sensitivity spans outperform random spans.
3. High-sensitivity spans outperform recent spans.
4. Decision-state sensitivity correlates with downstream success or score better than recency/token count/coverage.
5. Removing high-sensitivity spans changes meaningful decision fields such as next_action_arguments, active_constraints, candidate_objects, or completed_actions.
6. Case studies show high-sensitivity spans are not simply the longest or most recent spans.

It is acceptable if high-sensitivity spans do not beat ACON-style or task-aware summaries in all settings.
The primary motivation claim is that decision-state sensitivity is a better signal for context importance than static coverage or recency.

⸻

14. Important Notes

Do not optimize prompt format.
Do not train a selector.
Do not implement RL.
Do not hand-tune compression schemas.

This experiment tests a specific claim:

A history span matters if removing it changes the downstream agent's decision state.