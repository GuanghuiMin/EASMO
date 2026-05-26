# Execution Spec: Strengthening Motivation Experiments for Role-Conditioned Agent Memory
## 0. Objective
We are strengthening the **motivation experiments** before method design.
The goal is not to implement a learned memory compressor yet.  
The goal is to empirically justify the core claim:
> Compact agent memory should not be treated as a universal summary of past experience.  
> Its useful content depends on downstream use, especially the agent role.
> This round should strengthen the motivation along four axes:
1. structural divergence: role is a stronger driver of memory variation than strategy;
2. artifact control: role divergence is not only caused by deterministic slicing rules;
3. behavioral validation: wrong memory causes measurable efficiency or capability cost;
4. robustness: the phenomenon is not specific to one executor model.
Do **not** implement any learned compressor in this round.  
Do **not** run baseline competition against memory methods in this round.  
Do **not** add RL training in this round.
---
## 1. Critical Definition: What Counts as Oracle Memory?
### 1.1 T1 must not rely on prompted LLM summaries
T1 is the claim that role-conditioned memory structure exists and matters.
For T1, valid reference memories are:
- executable traces;
- gold API call sequences;
- deterministic diagnostic role projections;
- independent role-agent outputs;
- final-state / trajectory evidence.
Do **not** use prompted LLM summaries as the ground-truth memory for T1.
### 1.2 T2 is where prompted LLM compression is tested
Prompted LLM compression should only be used for T2:
> Can an LLM, when explicitly prompted with task and role information, recover the role-specific memory structure revealed by trace-derived or role-agent-derived memories?
> This distinction is essential.
> Correct logic:
```text
T1:
  Trace-derived / role-agent-derived memory shows role-specific structure.
  Wrong memory causes behavioral cost.
T2:
  Prompted LLM compression fails to recover this structure.

Incorrect logic:

T1:
  LLM summary says roles need different memory.
T2:
  LLM summary is bad.

Avoid the incorrect logic.

⸻

2. Scope for This Round

Implement and run only the following experiments:

Required

1. Experiment A: Three-level memory divergence hierarchy
    * strategy vs task vs role overlap;
    * reproduce for existing executor;
    * add at least one new executor if available.
2. Experiment B: Real role-agent artifact control
    * planner → executor → verifier multi-stage pipeline;
    * measure cross-role overlap from independent role-agent outputs.
3. Experiment C: Behavioral cost of wrong memory
    * matched vs mismatched memory;
    * report efficiency cost under default budget;
    * report capability cost under bounded inference.
4. Experiment D: Prompted compression gap
    * role-aware prompted compression;
    * compare against trace-derived / role-diagnostic reference memory;
    * report role recall and cross-role uniformity.
5. Experiment E: Cross-executor robustness
    * rerun core diagnostics with one additional executor;
    * do not need full matrix at first.

Optional

6. Experiment F: One non-AppWorld benchmark diagnostic
    * LongMemEval or LoCoMo;
    * only if setup cost is reasonable;
    * focus on structural role-conditioned evidence selection, not full agent execution.

Explicitly out of scope

* learned compressor;
* RL / GRPO;
* method training;
* comparison against full memory-system baselines;
* paper-quality hyperparameter tuning;
* large-scale benchmark sweep.

⸻

3. Common Data Model

All experiments should write outputs in machine-readable JSONL or CSV.

3.1 MemoryUnit format

Each memory unit should have:

{
  "unit_id": "string",
  "task_id": "string",
  "executor": "string",
  "source": "trajectory | gold_api | role_agent | prompted",
  "role": "planner | tool | code | verifier | generic",
  "strategy": "direct | verify | explore | none",
  "kind": "action | observation | api_call | profile | plan | evidence | code_pattern | other",
  "text": "string",
  "source_step": "int_or_null",
  "token_count": "int"
}

3.2 MemoryRecord format

Each compressed memory should have:

{
  "memory_id": "string",
  "task_id": "string",
  "executor": "string",
  "benchmark": "appworld | longmemeval | locomo",
  "condition_axis": "strategy | task | role | executor | prompted",
  "condition_value": "string",
  "role": "planner | tool | code | verifier | generic",
  "strategy": "direct | verify | explore | none",
  "budget": 128,
  "source": "diagnostic_projection | real_role_agent | gold_trace | prompted_llm | bm25 | recent | none",
  "unit_ids": ["..."],
  "memory_text": "string",
  "token_count": 123
}

3.3 RunResult format

Each downstream run should write:

{
  "run_id": "string",
  "task_id": "string",
  "executor": "string",
  "benchmark": "appworld",
  "memory_condition": "matched | wrong_task | wrong_role | cross_domain | generic | no_memory",
  "memory_id": "string_or_null",
  "budget": 512,
  "max_iter": 15,
  "success": true,
  "final_reward": 1.0,
  "iterations": 12,
  "input_tokens": 79000,
  "api_calls_total": 15,
  "api_calls_unique": 8,
  "wrong_endpoint_calls": 2,
  "show_api_doc_calls": 1,
  "elapsed_s": 0.0,
  "notes": "string"
}

⸻

4. Metrics

4.1 Structural overlap

Use Jaccard overlap as the main structural metric.

Report at least:

* entity-token Jaccard;
* unit-ID Jaccard when applicable;
* token-normalized overlap if easy.

For each comparison, report:

mean, std, median, min, max, n_pairs

4.2 Prompted compression recall

For prompted memory (m_p) and reference role memory (m_r):

recall = Jaccard(m_p, m_r)

Report per role:

* tool;
* code;
* planner;
* verifier.

Special attention:

Code-role recall is important because the desired abstraction is control-flow pattern, not API fact list.

4.3 Behavioral metrics

For AppWorld behavior-level validation, report:

* success;
* final reward;
* iterations;
* input tokens;
* total API calls;
* unique API calls;
* wrong endpoint calls;
* API-doc exploration calls;
* capped-budget failure rate.

Final task success alone is not enough because the agent may recover by re-querying APIs.

⸻

5. Experiment A: Three-Level Memory Divergence Hierarchy

5.1 Purpose

Show that role is a stronger driver of memory variation than superficial strategy.

Expected hierarchy:

Jaccard(strategy)  >>  Jaccard(task)  >  Jaccard(role)

5.2 Inputs

Use existing successful AppWorld trajectories.

Required axes:

1. Strategy axis:
    * same task;
    * same executor;
    * same role;
    * different strategies: direct / verify / explore.
2. Task axis:
    * same executor;
    * same role;
    * different tasks.
3. Role axis:
    * same task;
    * same executor;
    * different roles: planner / tool / code / verifier.

Budgets:

B = 128, 256, 512, 1024

Main budget for figures:

B = 512

5.3 Procedure

For each executor:

1. Load successful trajectories.
2. Build memory records for:
    * strategy-conditioned memories;
    * task-conditioned memories;
    * role-conditioned diagnostic memories.
3. Compute pairwise overlap for each axis.
4. Aggregate by budget and executor.

5.4 Outputs

Write:

outputs/motivation/hierarchy_raw.jsonl
outputs/motivation/hierarchy_summary.csv
figures/motivation/hierarchy_by_executor.pdf
figures/motivation/hierarchy_b512.pdf

5.5 Required table

Produce a table:

Executor	Budget	Strategy Jaccard	Task Jaccard	Role Jaccard	n_tasks

5.6 Required figure

Bar plot at B=512:

Strategy overlap
Task overlap
Role overlap

One panel per executor if multiple executors are available.

5.7 Acceptance criteria

This experiment is complete only if the output includes:

* number of tasks used;
* number of successful trajectories;
* failure/missing counts;
* mean and std for each axis;
* plots for B=512;
* CSV summary for all budgets.

⸻

6. Experiment B: Real Role-Agent Artifact Control

6.1 Purpose

Rule out the critique:

Cross-role memory overlap is low only because deterministic extractors were hand-designed to emit disjoint slices.

6.2 Design

Run a multi-stage role-agent pipeline:

Task
  ↓
Planner agent → plan
  ↓
Executor agent → trajectory
  ↓
Verifier agent → verdict + evidence

Each role should produce its own natural output.

6.3 Role memory sources

Role	Memory source
Planner	planner’s generated plan
Tool-user	executor’s API trajectory
Coder	executor’s code/control-flow trace
Verifier	verifier’s evidence list

Important:

* planner memory must come from planner output;
* verifier memory must come from verifier evidence output;
* these should not be sliced from the same executor trajectory.

6.4 Inputs

Use AppWorld tasks.

Minimum pilot:

n_tasks >= 18

If possible:

n_tasks >= 30

Use the same executor model for planner, executor, and verifier in each run.

6.5 Procedure

For each task:

1. Planner receives task instruction and outputs 2–5 subgoals.
2. Executor receives task instruction + planner output and runs AppWorld.
3. Verifier receives:
    * task instruction;
    * claimed final answer;
    * last 5–10 trajectory steps;
    * outputs verdict/evidence/confidence.
4. Convert planner plan and verifier evidence into MemoryUnits.
5. Compute cross-role overlap.

6.6 Outputs

Write:

outputs/motivation/multistage_role_raw.jsonl
outputs/motivation/multistage_role_summary.csv
figures/motivation/multistage_role_heatmap.pdf

6.7 Required metrics

Report pairwise Jaccard:

* planner-tool;
* planner-code;
* planner-verifier;
* tool-code;
* tool-verifier;
* code-verifier.

Special emphasis:

planner-verifier overlap

because both are independent LLM role-agent outputs.

6.8 Acceptance criteria

The experiment is complete only if:

* at least 18 successful multi-stage tasks are collected;
* planner and verifier outputs are saved;
* overlap heatmap is generated;
* summary includes mean, median, min, max, and n.

⸻

7. Experiment C: Behavioral Cost of Wrong Memory

7.1 Purpose

Show that memory mismatch matters behaviorally.

Low overlap alone is not enough. We need to show:

Wrong memory increases inference cost and can reduce success under bounded inference.

7.2 Memory conditions

For each consumer task, evaluate:

1. matched_memory
    * memory from the same task / same role / same domain.
2. wrong_task_memory
    * memory from a different task in the same app/domain.
3. cross_domain_memory
    * memory from a task in a different app/domain.
4. generic_recent_memory
    * recent trajectory units.
5. no_memory
    * no pre-loaded memory.

Optional:

6. wrong_role_memory
    * memory from same task but wrong role.

7.3 Inference settings

Run each condition under:

max_iter = 50
max_iter = 15

Optional stress test:

max_iter = 8

Main bounded setting:

max_iter = 15

Do not use cap=8 as the main claim if matched memory also collapses.

7.4 Inputs

Minimum:

n_consumer_tasks >= 18

Preferred:

n_consumer_tasks >= 24

Budgets:

B = 128, 256, 512

Main budget:

B = 512

7.5 Procedure

For each consumer task:

1. Build or load memory for each condition.
2. Inject memory into the AppWorld prompt.
3. Run the agent.
4. Record full RunResult.
5. Aggregate by memory condition, budget, and max_iter.

7.6 Outputs

Write:

outputs/motivation/behavior_cost_raw.jsonl
outputs/motivation/behavior_cost_summary.csv
figures/motivation/behavior_success_cap15.pdf
figures/motivation/behavior_cost_tokens_iters.pdf

7.7 Required plots

Figure 1:

Success rate under max_iter=15
matched vs wrong_task vs cross_domain vs generic vs no_memory

Figure 2:

Iterations and input tokens under max_iter=50
matched vs wrong_task vs cross_domain

7.8 Required interpretation fields

In the summary CSV, include:

efficiency_tax_iters = mean_iters(condition) - mean_iters(matched)
efficiency_tax_tokens = mean_tokens(condition) - mean_tokens(matched)
capability_drop = success(matched) - success(condition)

7.9 Acceptance criteria

This experiment is complete only if:

* results include at least 18 consumer tasks;
* both max_iter=50 and max_iter=15 are run;
* missing/failed cells are reported;
* summary includes success, iterations, tokens, API calls;
* plots are generated.

⸻

8. Experiment D: Prompted Compression Gap

8.1 Purpose

Test whether role-aware prompting alone can recover role-specific memory.

This supports T2:

Prompted LLM compression is too uniform across roles and misses role-specific abstractions.

8.2 Important constraint

Prompted memory is not used as T1 oracle.
Prompted memory is the object being evaluated.

8.3 Prompted conditions

For each trajectory and role, generate:

1. prompted_generic
    * compress the trajectory generally.
2. prompted_task
    * compress for the downstream task.
3. prompted_role
    * compress for the target role.
4. prompted_task_role
    * compress for both task and role.
5. prompted_extractive
    * if possible, ask the LLM to select unit IDs rather than write a free-form summary.

8.4 Inputs

Use the same unit pool as the diagnostic role memory.

Budgets:

B = 128, 256, 512, 1024

Main budget:

B = 512

Roles:

planner, tool, code, verifier

8.5 Procedure

For each task, role, budget, and prompted condition:

1. Provide the memory unit pool.
2. Ask the LLM to compress/select memory under the budget.
3. Save raw LLM output.
4. Convert output into MemoryRecord.
5. Compute:
    * cross-role overlap among prompted memories;
    * cross-role overlap among diagnostic/reference memories;
    * recall against same-role diagnostic memory.

8.6 Outputs

Write:

outputs/motivation/prompted_compression_raw.jsonl
outputs/motivation/prompted_compression_summary.csv
figures/motivation/prompted_vs_reference_heatmap.pdf
figures/motivation/prompted_role_recall.pdf

8.7 Required tables

Table 1: Cross-role Jaccard

Pair	Prompted Jaccard	Reference Jaccard	Ratio

Table 2: Same-role recall

Role	Generic	Task	Role	Task+Role	Extractive

8.8 Special diagnostic for code role

For code role, additionally report:

% prompted memory units that contain API facts
% prompted memory units that contain control-flow patterns
recall against code-pattern diagnostic memory

The key expected failure mode is:

The LLM keeps API facts even when asked to compress for a coding agent, instead of selecting reusable control-flow abstractions.

8.9 Acceptance criteria

This experiment is complete only if:

* raw prompted outputs are saved;
* prompted memory is compared against reference diagnostic memory;
* cross-role uniformity is reported;
* same-role recall is reported;
* code-role abstraction diagnostic is reported.

⸻

9. Experiment E: Cross-Executor Robustness

9.1 Purpose

Show that the phenomenon is not specific to one executor model.

If the same hierarchy appears across models, the claim strengthens:

Role-conditioned memory structure is stable across executors.

If not, the conclusion is still useful:

Memory should condition jointly on role and executor capability.

9.2 Executors

Current:

MiniMax-M2.5

Add at least one:

Qwen
GPT-style executor
Claude-style executor

Use whichever endpoint is available.

9.3 Minimal required reruns

For each new executor, run:

1. Three-level hierarchy at B=512.
2. Prompted compression gap at B=512.
3. Behavioral cost at B=512, max_iter=15.

Do not rerun the full matrix at first.

9.4 Outputs

Write:

outputs/motivation/cross_executor_summary.csv
figures/motivation/cross_executor_hierarchy.pdf
figures/motivation/cross_executor_prompted_gap.pdf
figures/motivation/cross_executor_behavior_cost.pdf

9.5 Required summary table

Executor	Strategy Jaccard	Task Jaccard	Role Jaccard	Prompted/Reference Ratio	Matched Success	Cross-domain Success

9.6 Acceptance criteria

This experiment is complete only if:

* new executor identity is logged;
* number of successful tasks is reported;
* missing/failure cases are reported;
* core metrics are comparable to MiniMax;
* results are saved in the common format.

⸻

10. Optional Experiment F: Non-AppWorld Benchmark Diagnostic

10.1 Purpose

AppWorld is strong for tool-use behavior but does not cover all memory settings.

If feasible, add one non-tool benchmark to show that role-conditioned memory is not only an AppWorld artifact.

Choose one:

LongMemEval
LoCoMo

Do not attempt a full method comparison.

10.2 LongMemEval role views

Possible roles:

Role	Memory target
Answerer	facts needed to answer the query
Verifier	evidence supporting or refuting the answer
Preference tracker	stable and updated user preferences
Temporal reasoner	time order and changed facts

Diagnostics:

* overlap between role views;
* prompted compression uniformity;
* role-specific evidence recall.

10.3 LoCoMo role views

Possible roles:

Role	Memory target
Summarizer	high-level events
Answerer	query-relevant facts
Verifier	supporting evidence
Relationship tracker	user/person state and relationship changes

Diagnostics:

* overlap between role views;
* prompted compression uniformity;
* query-answer evidence recall.

10.4 Outputs

Write:

outputs/motivation/non_appworld_raw.jsonl
outputs/motivation/non_appworld_summary.csv
figures/motivation/non_appworld_role_overlap.pdf

10.5 Acceptance criteria

This optional experiment is complete only if:

* role definitions are documented;
* reference memory construction is not prompted-summary-based;
* overlap and recall are reported;
* limitations are clearly logged.

⸻

11. Figure Plan for Motivation Section

The final motivation section should support four figures.

Figure 1: Conditioning hierarchy

Purpose:

Role is the strongest driver of memory divergence.

Plot:

Bar plot at B=512:
strategy overlap, task overlap, role overlap

Include one panel per executor if available.

⸻

Figure 2: Real role-agent validation

Purpose:

Role divergence persists beyond deterministic projections.

Plot:

Heatmap of cross-role overlap from multi-stage role-agent setup.

Highlight:

planner-verifier overlap

because both are independent role-agent outputs.

⸻

Figure 3: Behavioral cost of wrong memory

Purpose:

Wrong memory is an efficiency tax under loose inference and a capability loss under bounded inference.

Plot A:

Success rate under max_iter=15.

Plot B:

Iterations / input tokens under max_iter=50.

⸻

Figure 4: Prompted compression gap

Purpose:

Role-aware prompting fails to recover role-specific memory abstractions.

Plot A:

Reference role-memory cross-role heatmap.

Plot B:

Prompted memory cross-role heatmap.

Plot C:

Same-role recall bar chart, especially code role.

⸻

12. Logging Requirements

Every script should log:

timestamp
git commit hash if available
executor name
model endpoint/config
benchmark
task split
number of tasks attempted
number of successful tasks
number of failed tasks
budgets
max_iter values
random seed
output file paths

For every failed cell, save:

{
  "task_id": "string",
  "executor": "string",
  "experiment": "string",
  "failure_type": "trajectory_failed | parse_failed | timeout | missing_output | other",
  "error_message": "string"
}

Do not silently drop failed cells.

⸻

13. Claim-Safety Rules

When generating summaries, tables, and figure captions, follow these rules.

13.1 Do not overclaim

Allowed:

In AppWorld-style tool-use agents, role-conditioned memory views are substantially more divergent than strategy-conditioned views.

Not allowed:

Agent memory is universally role-orthogonal.

Allowed:

Prompted role-aware compression fails to recover our trace-derived role-specific diagnostic memories in this setup.

Not allowed:

LLMs cannot compress memory.

Allowed:

Under bounded inference, mismatched memory can reduce success.

Not allowed:

Wrong memory always causes failure.

13.2 Always separate three things

1. Diagnostic/reference memory:
    * trace-derived;
    * gold-derived;
    * role-agent-derived.
2. Prompted memory:
    * evaluated as a candidate compressor;
    * not used as T1 ground truth.
3. Downstream behavior:
    * efficiency and success under actual agent execution.

⸻

14. Final Deliverables

At the end of this round, produce:

outputs/motivation/README_RESULTS.md
outputs/motivation/hierarchy_summary.csv
outputs/motivation/multistage_role_summary.csv
outputs/motivation/behavior_cost_summary.csv
outputs/motivation/prompted_compression_summary.csv
outputs/motivation/cross_executor_summary.csv

And figures:

figures/motivation/hierarchy_b512.pdf
figures/motivation/multistage_role_heatmap.pdf
figures/motivation/behavior_success_cap15.pdf
figures/motivation/behavior_cost_tokens_iters.pdf
figures/motivation/prompted_vs_reference_heatmap.pdf
figures/motivation/prompted_role_recall.pdf
figures/motivation/cross_executor_hierarchy.pdf

The results README should answer:

1. Does role produce stronger memory divergence than strategy and task?
2. Does role divergence persist in real role-agent outputs?
3. Does mismatched memory create behavioral cost?
4. Does prompted compression recover role-specific memory?
5. Are the findings stable across executors?
6. What are the remaining limitations?

⸻

15. Suggested Execution Order

Run in this order:

1. Clean up common data schema and logging.
2. Reproduce current hierarchy result.
3. Add real role-agent artifact-control summary.
4. Scale behavioral-cost experiment.
5. Run prompted-compression gap with strict oracle separation.
6. Add second executor for the three key diagnostics.
7. Optionally add LongMemEval or LoCoMo diagnostic.

Do not start optional benchmark work until AppWorld motivation experiments are clean and reproducible.

我建议你直接把这个保存成 `motivation_experiment_execution_spec.md`。给 coding agent 的时候再加一句：**“Follow this spec strictly. Do not add new experiments unless explicitly requested.”**