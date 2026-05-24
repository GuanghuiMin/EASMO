下面是一份新的 Motivation Experiments Checklist，目标是验证你的两个核心 thesis：

1. T1: Compression pressure induces policy-dependent memory.
2. T2: Prompted LLM selectors fail to realize policy-conditional compression.

重点：全部基于公开 benchmark，主战场是 AppWorld，LongMemEval / LoCoMo 作为辅助验证。

⸻

> ## 2026-05-24 Review notes (in-line below)
>
> A design review on 2026-05-24 raised 11 specific concerns. They are
> in-lined into the relevant sections below as `Review note (R-N):`
> blocks and surfaced in the checklist sections too. Summary of fixes
> baked into this revision:
>
> | # | Concern | Fix in this doc |
> |---|---------|-----------------|
> | R-1 | `m*_exec` selection algorithm undefined when trajectory > B | §3.2: spec **two** variants `m*_exec_minimal` / `m*_exec_trajectory` |
> | R-2 | Multi-app tasks' policy family ambiguous | §6.4 + §2.4: heatmap restricted to single-app tasks |
> | R-3 | M3 same-context cross-policy pairs may not exist in AppWorld | §10.2 pilot's first deliverable is the count |
> | R-4 | "Full-context success" filter is selection-biased | §4.3: pilot reports filter survival rate + difficulty stratification |
> | R-5 | Budget pattern may be bowl-shaped, not monotonic | §4.6: explicitly test for valley vs monotonic |
> | R-6 | M4 ablation conflates capacity and content | §7.3: switch to swap-with-padding rather than remove |
> | R-7 | T2 thresholds are rigid | §5.11: ratio-based verdict |
> | R-8 | `m*_exec` is per-(task, executor); cross-executor is transfer | §2.3 + §3.2 explicit |
> | R-9 | AppWorld infra setup is real engineering | new §10.0 "Infrastructure prep" |
> | R-10 | Sample size after filtering is unknown | §10.1 reports survival counts before main run |
> | R-11 | `m_generic` is a basket, not a baseline | §3.3: pick `m_recent` and `m_freq` as the two declared variants |
>
> Engineering update: the AppWorld + agent-runner infrastructure for
> all of M1–M4 is **already available** in `/workspace/acon/`
> (Microsoft's ACON repo, arXiv 2510.00615). It has a working
> `acon/.venv` with pydantic v1 / sqlmodel 0.0.10 (the EASMO venv has
> a v2 incompat that breaks AppWorld imports), the AppWorld task
> data downloaded under `acon/experiments/appworld/data/`, and a
> functional executor at `acon/experiments/appworld/run.py`. We
> consume acon's trajectory outputs as inputs to the M1–M4
> compression analysis rather than re-implement the runner.

Motivation Experiments for Policy-Conditional Agent Memory

0. Goal

The goal of these motivation experiments is to empirically validate the central premise of policy-conditional memory compression:

Under tight memory budgets, the optimal compressed memory is not a generic summary of the past, but a policy-conditioned state that preserves information needed for downstream agent behavior.

We aim to validate two theses before introducing the method.

Thesis 1: Compression pressure induces policy dependence

When memory budgets are loose, generic compression may retain enough information for many downstream agents. Under tight budgets, however, each retained memory unit competes with alternatives, and its value depends on the consuming policy.

Thesis 2: Prompting does not produce policy-conditional compression

Even when conditioned on task descriptions or agent/policy descriptions, off-the-shelf LLM selectors tend to produce generic, surface-salient compressions that fail to preserve behavior-critical information under tight budgets.

⸻

1. Benchmark Scope

1.1 Main benchmark: AppWorld

AppWorld is the primary benchmark because it provides executable agent tasks, structured app states, API calls, and state-based task evaluation. This makes it suitable for evaluating whether compressed memory preserves downstream agent behavior.

Why AppWorld is central

AppWorld supports:

* tool-use trajectories;
* structured app states;
* API calls and arguments;
* final state-based task success;
* multi-app dependency;
* execution traces from successful full-context agents.

This allows us to construct execution-derived policy memory without relying on an LLM oracle.

⸻

1.2 Secondary benchmarks: LongMemEval / LoCoMo

LongMemEval and LoCoMo are used as supporting evidence for long-conversation memory compression.

They are less suitable than AppWorld for action-level memory evaluation, but useful for showing that the same pattern appears in long-memory QA:

* different query types require different compressed evidence;
* tight budgets amplify query/policy dependence;
* prompted selectors often retain generic conversational facts rather than behavior-relevant evidence.

⸻

2. Shared Experimental Setup

2.1 Memory unit construction

Instead of compressing raw tokens directly, we convert each context into discrete memory units.

For AppWorld, memory units are derived from app states:

u_i = {
  source_app: calendar / email / contacts / shopping / travel / notes / ...
  entity_id: ...
  timestamp: ...
  text: natural-language rendering of the row / record / message / event
}

Examples:

[calendar] Event: Team sync with Alice on Friday 3–4 PM.
[email] Thread: Bob asked to reschedule the meeting to next Tuesday.
[contacts] Alice Chen: alice@example.com, colleague, works at Databricks.
[shopping] Order #1241: noise-canceling headphones, delivered to Sunnyvale address.

For LongMemEval / LoCoMo, memory units are conversation turns, event summaries, or sentence-level facts:

u_i = {
  session_id: ...
  turn_id: ...
  speaker: user / assistant
  timestamp: ...
  text: ...
}

⸻

2.2 Memory budget

Run a budget sweep:

B ∈ {128, 256, 512, 1024, 2048}

For AppWorld, if 128 tokens is too small for executable memory, use:

B ∈ {256, 512, 1024, 2048}

The key expected pattern is not absolute performance, but the interaction between budget and policy dependence:

policy-conditioned advantage should be largest at small B
policy-conditioned advantage should shrink as B increases

⸻

2.3 Executor agent

Use the same executor agent across all memory conditions.

Recommended first-stage executor:

Minimax-240B (or whatever endpoint is available — currently MiniMax-M2.5 at the in-house endpoint).

Later robustness executors:

Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
GPT-4.1 / Claude / Gemini, if budget allows

Important control:

Same executor, same action interface, same decoding setting, same task instruction. Only the compressed memory changes.

This avoids confounding memory effects with model capability or scaffold differences.

> **Review note (R-8):** `m*_exec` is constructed from the executor's
> own successful trajectory (it captures *which APIs / DB rows / entities
> THIS executor needed*). Using one executor's `m*_exec` to evaluate a
> different executor measures *transfer* (does executor B want what
> executor A needed?), not *robustness* (does the policy-conditioned
> memory generalise across executors?).
>
> For the headline robustness section, generate an
> **executor-matched** `m*_exec` per (task, executor) — i.e. run
> Qwen-7B end-to-end on the same task list, extract its own trajectory,
> derive its own `m*_exec`, then compress and re-run. Cross-executor
> transfer of `m*_exec` is a separate (interesting) experiment that
> belongs in an appendix.

⸻

2.4 Policy axis (revised 2026-05-24 — Option X primary)

> **Conceptual fix (after critique):** the original "policy family =
> task topic" framing conflates two different sources of variation —
> *which information the task needs* (topic) and *how an agent
> chooses to use it* (policy). Cross-policy memory transfer with
> task-family-as-policy degenerates into "is irrelevant memory
> bad?" which is trivially yes.
>
> The corrected design treats **"policy" as a behavioural strategy**
> chosen *independently* of task topic, and reports task topic as a
> **secondary axis** for stratification. We define policy via three
> strategy variants of the same agent loop running on the same
> executor model on the same task — each producing a distinct
> trajectory and therefore a distinct execution-derived memory.

### 2.4a Primary policy axis (Option X — strategy variants)

Three strategies are injected into the AppWorld agent's task prompt
right before the task instruction. They are short, behavioural
mandates that change the agent's API-call pattern, not the model or
the task. Canonical specs in
[`motivation_v2/prompts/STRATEGY_DESIGN.md`](../../motivation_v2/prompts/STRATEGY_DESIGN.md);
implementation in `motivation_v2/prompts/build_strategy_prompts.py`
+ `motivation_v2/scripts/run_appworld_strategy.py`.

| Strategy   | Search breadth | Verification | Action style       |
|------------|----------------|--------------|--------------------|
| `P_direct` | minimal        | none         | answer-first       |
| `P_verify` | task-driven    | mandatory cross-check | answer-then-confirm |
| `P_explore`| exhaustive     | optional     | survey-then-answer |

**Validation on `82e2fac_3`** (smoke run, MiniMax-M2.5 executor):

| | iters | input tokens | unique APIs | total API calls | distinctive behaviour |
|---|---|---|---|---|---|
| baseline (no strategy) | 11 | 46K | 8 | 11 | normal |
| `P_direct` | 11 | 57K | 7 | 11 | small task: little to compress; matches baseline |
| `P_verify` | **26** | **204K** | **11** | **36** | 17× `show_song` cross-checks + `show_album` + `show_profile` |
| `P_explore`| 14 | 82K | 8 | 14 | 5× `show_api_doc`, 100% `show_app_descriptions` in first 3 steps |

Manipulation-check rule (must pass before relying on M1/M2/M3
results): `median(iters_verify) / median(iters_direct) ≥ 1.5×`. On
the smoke run this is 2.36×. The `direct` vs `baseline` near-equality
on a 1-iter-tail task is expected; the strategy effect should be
strongest on difficulty-2/3 tasks where there's room to shrink.

### 2.4b Secondary axis (task topic, single-app)

Same definition as before:
* `policy_topic = primary_app` (spotify, file_system, phone,
  simple_note, venmo) for single-app tasks (`required_apps` minus
  `supervisor` plumbing has length 1).
* Multi-app tasks tagged `multi_app` and excluded from the M3
  cross-policy heatmap (R-2 stands).

This axis is now used **only for stratification** of the M1/M2
results, not as the policy axis itself: we want to show "the
strategy effect holds across task topics", which is much weaker
than "strategy IS the policy".

### 2.4c Backup policy axis (Option Y — executor variants)

If Option X's manipulation check fails or the strategy effect is
too small to be visible at any budget, fall back to Option Y
(different executor models — MiniMax / Qwen-7B / GPT-4o-mini —
treated as different policies on the same task). Y is more
conceptually robust ("real model differences are unfaked") but
requires extra LLM endpoints and 2-3× compute. Documented here so
the design has a fallback; not the primary plan.

⸻

3. Memory Conditions

Each experiment should compare the following memory variants.

3.1 Full context / full app state

Upper bound.

Agent receives full task context and full relevant app state.

This establishes whether the executor can solve the task before compression.

Only tasks solved under full context should be included in the main compression analysis.

⸻

3.2 Execution-derived policy memory

This is the main “gold” policy-conditioned memory.

It should not be generated by an LLM. It is derived from successful execution traces.

For each successful AppWorld task, collect memory units associated with:

* APIs called;
* API arguments;
* database rows retrieved;
* database rows modified;
* entities mentioned in successful trajectory;
* records referenced by final state tests, if accessible;
* app-specific rows necessary for successful execution.

Denote:

m^*_{\text{exec}}(x, B)

This is the strongest evidence-based compressed memory for task x under budget B.

> **Review note (R-1, R-8):** "execution-derived" is underspecified
> until you say *which* of those bullet points contributes and *how*
> they are budget-trimmed. We commit to **two declared variants** so
> the headline figure can show them side by side and any gap between
> them is itself a useful diagnostic:
>
> * **`m*_exec_minimal(x, B, executor)`** — start from the union of
>   DB rows and entities **directly referenced by the final-state
>   evaluator**. If under budget, pad with the next-most-touched rows
>   (by trajectory frequency). This is the tightest possible "if
>   the agent had nothing else, would it still pass?" memory.
>
> * **`m*_exec_trajectory(x, B, executor)`** — start from the union
>   of all rows touched in the successful trajectory; rank by
>   `2 × final_state_referenced + trajectory_touch_count`; greedily
>   include in rank order until budget is exhausted. This represents
>   "everything the agent demonstrably touched, sorted by demonstrated
>   importance".
>
> The (executor) suffix is mandatory: a Minimax trajectory and a
> Qwen-7B trajectory may touch different rows even on the same task,
> so `m*_exec` is per-(task, executor). The pilot reports both
> variants and the gap between their downstream task success.
>
> Both algorithms are deterministic post-hoc functions of the
> trajectory + final-state-evaluator log + budget — no LLM in the
> loop. Their definitions live in `motivation_v2/exec_memory.py`.

⸻

3.3 Generic memory

A policy-agnostic memory baseline.

It does not use task or policy information.

Generic selection rules may include:

* most recent units;
* user profile units;
* globally salient records;
* high-frequency entities;
* long-term preference records;
* global summary of the app state.

Denote:

m_{\text{generic}}(x, B)

This baseline tests whether generic compression is sufficient.

> **Review note (R-11):** "generic" as listed above is a basket of
> rules, not a single comparable baseline. We pin **two declared
> variants** for the headline plot:
>
> * **`m_recent(x, B)`** — the N most recent memory units that fit
>   within B tokens (recency-only, no semantic filter).
> * **`m_freq(x, B)`** — the N most frequent entities across the
>   user's app state up to B tokens (no task / policy info).
>
> Other variants (user profile, global summary, ...) live in the
> appendix or in §11's checklist as optional. The Figure 1 main
> plot uses `m_recent`; `m_freq` shows up as a secondary line to
> reassure reviewers that the gap to `m*_exec` isn't an artifact of
> "recent ≠ relevant".

⸻

3.4 Retrieval memory

Strong non-learned baselines.

Use task instruction as query.

Recommended retrieval baselines:

BM25
embedding top-k
hybrid BM25 + embedding
recency + BM25

Denote:

m_{\text{retrieval}}(x, B)

This is important because reviewers will compare your method against retrieval.

⸻

3.5 Prompted generic memory

Use Minimax-240B as a selector without policy condition.

Prompt:

Given the following memory units, select the most important information under the token budget.
Do not exceed B tokens.
Return only the selected memory units.

Denote:

m_{\text{prompt-generic}}(x, B)

⸻

3.6 Prompted task-conditioned memory

Use Minimax-240B as a selector with task instruction.

Prompt:

Given the task instruction and the following memory units, select the information most useful for completing the task.
Do not exceed B tokens.
Return only the selected memory units.
Task:
{task_instruction}
Memory units:
{memory_units}

Denote:

m_{\text{prompt-task}}(x, B)

⸻

3.7 Prompted policy-conditioned memory

Use Minimax-240B as a selector with policy family.

Prompt:

You are selecting compressed memory for a {policy_family} agent.
Given the following memory units, select the information most useful for this agent's next action.
Do not exceed B tokens.
Return only the selected memory units.
Policy family:
{policy_family}
Memory units:
{memory_units}

Denote:

m_{\text{prompt-policy}}(x, B)

⸻

3.8 Prompted task + policy-conditioned memory

Use both task instruction and policy family.

Prompt:

You are selecting compressed memory for a {policy_family} agent.
The agent must complete the following task.
Task:
{task_instruction}
Select memory units that are most useful for the agent's downstream behavior.
Do not exceed B tokens.
Return only the selected memory units.
Memory units:
{memory_units}

Denote:

m_{\text{prompt-task-policy}}(x, B)

This is the strongest prompting baseline.

⸻

4. Experiment M1: Compression-Pressure Sweep

4.1 Purpose

Validate T1:

Policy-conditioned memory becomes more valuable as compression pressure increases.

⸻

4.2 Question

Does execution-derived policy memory outperform generic memory under tight budgets, and does this advantage shrink as budget increases?

⸻

4.3 Dataset

Use AppWorld.

Recommended pilot:

100 successful full-context tasks
4 policy families
B ∈ {256, 512, 1024, 2048}

Full run:

300–500 successful full-context tasks
5–6 policy families
B ∈ {128, 256, 512, 1024, 2048}

Only include tasks where the executor succeeds with full context.

> **Review note (R-4, R-10):** the "successful full-context" filter
> introduces selection bias and is also a sample-size bottleneck.
> The pilot must report:
>
> * `n_total_tasks` attempted at full context.
> * `n_full_context_success` (the filter survival rate).
> * Per-policy-family survival and post-filter counts.
> * **Hardness-stratified results**: split surviving tasks into
>   quartiles by full-context num-interactions (proxy for difficulty)
>   and report Δ_policy-generic in each quartile separately. The
>   T1 claim is much stronger if the gap appears across the
>   spectrum, not only on the easiest quartile.
>
> If `n_full_context_success < 60` for any policy family, that family
> is demoted to a robustness side panel rather than appearing in the
> heatmap or main figure.

⸻

4.4 Compared memory conditions

Compare:

Full context
Execution-derived policy memory
Generic memory
BM25 / embedding retrieval
Recency memory
Prompted generic memory
Prompted task+policy memory

⸻

4.5 Metrics

Primary:

Task success rate

Secondary:

First-action match against full-context trajectory
API-call F1
Argument exact match / F1
Invalid API call rate
Trajectory edit distance
Touched-entity recall

⸻

4.6 Expected pattern

Main desired result:

At tight budgets:
Execution-derived policy memory >> generic memory
At loose budgets:
gap becomes smaller

More formally:

\Delta_{\text{policy-generic}}(B)
=
\text{Success}(m^*_{\text{exec}}, B)
- \text{Success}(m_{\text{generic}}, B)

Expected:

\Delta_{\text{policy-generic}}(256)
>
>\Delta_{\text{policy-generic}}(512)
>
>\Delta_{\text{policy-generic}}(1024)

> **Review note (R-5):** Δ may not be monotonically decreasing in B.
> Three plausible shapes; the analysis must report which one occurred:
>
> * **Monotonic decreasing** (the assumed shape) — strongest T1.
> * **Bowl-shaped** with a peak at intermediate B (e.g. B=256). At
>   very tight B both methods fail to fit anything useful → Δ → 0;
>   at very loose B both methods include everything → Δ → 0.
>   T1 still holds; wording becomes "policy-dependence emerges in
>   the *useful-budget regime*".
> * **Saturating / increasing** — T1 dies on this benchmark.
>
> Pilot adds B=128 (and possibly B=4096) to the grid so the valley
> shape, if any, is observable.

⸻

4.7 Main figure

Line plot:

x-axis: memory budget B
y-axis: AppWorld task success
Curves:
- Full context
- Execution-derived policy memory
- Prompted task+policy memory
- BM25 / embedding retrieval
- Prompted generic memory
- Generic memory
- Recency memory

⸻

4.8 Pass criteria

The motivation claim is supported if:

Execution-derived policy memory consistently outperforms generic memory at tight budgets.
The policy-generic gap decreases as budget increases.
The result holds across at least 3 policy families.

Suggested threshold:

At B = 256 or 512:
Execution-derived policy memory improves task success over generic memory by ≥ 10–15%.

⸻

4.9 Failure interpretation

If execution-derived memory does not outperform generic memory:

* the task may not require policy-specific memory;
* the memory budget may be too loose;
* the memory units may be too coarse;
* the executor may be insensitive to memory;
* AppWorld task subset may be too simple.

Next action:

Filter for multi-app / multi-hop / preference-dependent tasks.
Lower the budget.
Use more granular memory units.
Evaluate first-action accuracy in addition to full task success.

⸻

5. Experiment M2: Prompted Selector Gap

5.1 Purpose

Validate T2:

Prompted LLM selectors do not reliably produce policy-conditional compression.

⸻

5.2 Question

Does adding task and policy descriptions to a strong LLM selector close the gap to execution-derived memory?

⸻

5.3 Dataset

Use the same AppWorld task set as M1.

Recommended pilot:

100 full-context successful tasks
B ∈ {256, 512, 1024}

Full run:

300–500 tasks
B ∈ {128, 256, 512, 1024, 2048}

⸻

5.4 Compared memory conditions

Compare:

Execution-derived policy memory
Prompted task+policy memory
Prompted task-only memory
Prompted policy-only memory
Prompted generic memory
BM25 / embedding retrieval
Generic memory

⸻

5.5 Metrics

Primary:

Task success rate under selected memory

Secondary:

Evidence recall against execution-derived memory
First-action match
API-call F1
Argument F1
Selected-unit overlap with execution-derived memory
Cross-policy selected-unit overlap

⸻

5.6 Evidence recall metric

Let E_{\text{exec}}(x) be the execution-derived memory units for task x, and m_{\text{prompt}}(x, B) be the prompted selection.

\text{EvidenceRecall}(x, B)
=
\frac{
|m_{\text{prompt}}(x, B) \cap E_{\text{exec}}(x)|
}{
|E_{\text{exec}}(x)|
}

This measures whether the prompted selector actually selects behavior-relevant memory units.

⸻

5.7 Specialization gain

Measure how much prompt conditioning helps over generic prompting:

\Delta_{\text{specialization}}(B)
=
\text{Success}(m_{\text{prompt-task-policy}}, B)
- \text{Success}(m_{\text{prompt-generic}}, B)

Expected if prompting fails:

Δ_specialization is small, especially under tight budgets.

⸻

5.8 Gold-prompt gap

Measure the remaining gap to execution-derived memory:

\Delta_{\text{gold-prompt}}(B)
=
\text{Success}(m^*_{\text{exec}}, B)
- \text{Success}(m_{\text{prompt-task-policy}}, B)

Expected:

Gold-prompt gap is large under tight budgets.

This directly supports:

Policy-conditioned compression is necessary, but prompting does not achieve it.

⸻

5.9 Expected pattern

Desired result:

Execution-derived policy memory
>
>Prompted task+policy memory
>≈ Prompted task-only memory
>≈ BM25 / embedding retrieval
>
>Prompted generic memory
>≈ Generic memory

A stronger result:

Prompted task+policy memory only marginally improves over prompted generic memory,
while execution-derived memory is substantially better.

⸻

5.10 Main figure

Bar plot at tight budget B = 256 or B = 512:

y-axis: AppWorld task success
Bars:
- Execution-derived policy memory
- Prompted task+policy memory
- Prompted task-only memory
- Prompted policy-only memory
- Prompted generic memory
- BM25 / embedding retrieval
- Generic memory

Optional second panel:

y-axis: evidence recall
same memory conditions

⸻

5.11 Pass criteria

The thesis is supported if:

Prompted task+policy memory does not close the gap to execution-derived policy memory.
Prompted task+policy memory only modestly improves over prompted generic memory.
This pattern is strongest at tight budgets.

> **Review note (R-7):** the rigid `Δ_gold-prompt ≥ 10% AND Δ_spec ≤ 5%`
> conjunction is brittle (e.g. observed `Δ_gold-prompt = 15%`,
> `Δ_spec = 8%` would fail the AND but is qualitatively
> "prompting helps a bit, doesn't close gap"). Use a ratio instead.
>
> Define the **prompting closure ratio**:
>
> $$ r_{\text{T2}}(B) = \frac{\Delta_{\text{specialization}}(B)}{\text{Success}(m^*_{\text{exec}}, B) - \text{Success}(m_{\text{generic}}, B)} $$
>
> i.e. "what fraction of the achievable-by-policy-conditioning gap
> does prompting close?". Verdicts:
>
> * `r_T2 ≤ 0.3`: STRONG T2 — prompting closes ≤ 30% of the gap.
> * `0.3 < r_T2 ≤ 0.7`: WEAK T2 — prompting partially works.
> * `r_T2 > 0.7`: T2 fails — prompting essentially achieves
>   policy-conditional compression and the paper's contribution
>   becomes "we make it cheaper / smaller" rather than "we enable
>   what prompting can't".
>
> Old fixed thresholds retained as legacy gates only.

⸻

6. Experiment M3: Cross-Policy Memory Transfer

6.1 Purpose

Show that memory is not interchangeable across policies.

This strengthens T1 by providing a behavioral counterfactual:

A memory selected for one policy should not transfer cleanly to another policy under tight budgets.

⸻

6.2 Question

If a task-policy p receives memory derived for another policy q, does performance drop?

⸻

6.3 Dataset

Use AppWorld tasks grouped by policy family.

Example policy families:

calendar
email
contacts
shopping
travel
finance

For each policy family, collect tasks where full-context execution succeeds.

⸻

6.4 Memory construction (revised — strategy as policy)

Under the §2.4a strategy-as-policy framing, M3 has a much cleaner
shape than the original task-family-as-policy version. Same task
`x`, three trajectories `traj_X(x)` for `X ∈ {direct, verify,
explore}`, three execution-derived memories
`m*_exec_trajectory(x, X)`. The M3 cross-policy heatmap is now:

| | consumer P_direct | consumer P_verify | consumer P_explore |
|---|---|---|---|
| **memory from P_direct**  | self (diagonal) | cross | cross |
| **memory from P_verify**  | cross           | self | cross |
| **memory from P_explore** | cross           | cross | self |

Each cell reports task success rate when a consumer agent running
strategy `Y` is given the memory derived from a producer running
strategy `X` on the same task. **Same task, same context, same
executor, only the producer/consumer policy differs.** This is a
genuine cross-policy test, not a topic-mismatch test.

> **Review note (R-3 — resolved):** the "shared-state pair" problem
> from the original draft disappears under strategy-as-policy: every
> task automatically yields three same-context same-task-instance
> data points (one per strategy). We no longer need to find pairs
> of tasks with shared user-state snapshots; the snapshot is held
> constant by construction.
>
> Task-topic axis (R-2): the heatmap is reported per task family
> (spotify-only, file_system-only, …) AND aggregated across
> single-app families. Multi-app tasks join the aggregated panel
> only.

⸻

6.5 Evaluation

For a consuming task from policy p, compare:

Correct-policy memory: m*_p
Cross-policy memory: m*_q
Generic memory
BM25 / embedding retrieval
Prompted task+policy memory

⸻

6.6 Transfer drop

Define:

\text{TransferDrop}_{p \leftarrow q}(B)
=
\text{Success}(p \mid m^*_p, B)
- \text{Success}(p \mid m^*_q, B)

Expected:

TransferDrop is large under tight budgets.
TransferDrop shrinks as budget increases.

⸻

6.7 Main figure

Heatmap at B = 256 or B = 512:

Rows: memory source policy
Columns: consuming task policy
Cell value: task success

Expected pattern:

Diagonal cells are high.
Off-diagonal cells are lower.
The diagonal-off-diagonal gap is largest at tight budgets.

⸻

6.8 Pass criteria

The thesis is supported if:

Same-policy memory outperforms cross-policy memory.
The transfer drop is largest at tight budgets.
The result holds for multiple policy families.

Suggested threshold:

At B = 256 or 512:
mean diagonal success - mean off-diagonal success ≥ 10%

⸻

7. Experiment M4: Memory-Unit Ablation Diagnostic

7.1 Purpose

Provide direct evidence that execution-derived units are behaviorally important.

This replaces expensive token-level saliency.

⸻

7.2 Question

Do removing execution-derived memory units hurt behavior more than removing generic or random units?

⸻

7.3 Method

For each successful task and budget condition:

1. Start from execution-derived memory at budget B.
2. **Swap** one memory unit (or one group of units) — replace it with
   a unit drawn from a fixed neutral pool of fillers, padded so the
   total token count remains B.
3. Re-run the executor.
4. Measure behaviour drop.

Compare swapping out:

execution-derived units
generic selected units
random units
recent units

> **Review note (R-6):** the original wording ("remove a unit") changes
> two things at once — the *content* (lost info) and the *capacity*
> (fewer tokens). Removing 30 tokens of `m*_exec` and 30 tokens of
> generic both shrink the memory; if behaviour drops, you can't tell
> whether it was the lost info or the lost tokens. The swap-with-
> filler protocol holds the budget constant so the only varying
> quantity is the content of the displaced unit.
>
> "Filler" can be (a) a same-token-length string of neutral
> connectives ("the user owns various items. there are records in the
> system."), (b) a same-token-length unit drawn from a UNRELATED user
> state, or (c) zero-padding via `<pad>` tokens. The headline ablation
> uses (a) since it preserves the prose-like character of the memory.

⸻

7.4 Metrics

Task success drop
First-action match drop
API-call F1 drop
Argument F1 drop

⸻

7.5 Expected pattern

Removing execution-derived units causes the largest behavior drop.
Removing random or generic units causes smaller drops.

This supports:

execution-derived policy memory is not merely shorter; it contains behavior-critical information.

⸻

7.6 Status in paper

This should probably be an appendix or supplementary diagnostic, unless the result is very visually strong.

⸻

8. LongMemEval / LoCoMo Supporting Experiments

These are not the main motivation experiments, but useful for showing generality beyond AppWorld.

⸻

8.1 Policy view definition

In LongMemEval / LoCoMo, define policy by question type or memory-use type.

Example policy views:

Policy view	Description
Extraction policy	retrieve a specific fact
Temporal policy	compare or order events over time
Update policy	resolve changed or overwritten information
Abstention policy	determine that information is unavailable
Multi-session reasoning policy	combine evidence across sessions
Preference policy	infer stable user preferences

⸻

8.2 Memory conditions

Compare:

Full conversation
Evidence-derived memory, if available
BM25 / embedding retrieval
Generic summary
Prompted generic memory
Prompted question-conditioned memory
Prompted question-type-conditioned memory
Prompted question + policy memory

⸻

8.3 Metrics

QA accuracy
Exact match / F1
LLM-judge correctness
Evidence recall
Abstention accuracy
Temporal reasoning accuracy

⸻

8.4 Key hypothesis

At tight budgets:

question/policy-conditioned evidence memory > generic memory
prompted selectors do not fully close the gap

⸻

8.5 Main use in paper

Use these results to support a general statement:

The same compression-pressure pattern appears beyond executable tool-use tasks: long-memory QA also requires query/policy-conditioned evidence under tight budgets.

This can go into the appendix or a secondary results table.

⸻

9. Recommended Paper Figure 1

Figure 1: Policy dependence emerges under compression pressure

Panel A: Compression-pressure sweep on AppWorld

Line plot:

x-axis: budget B
y-axis: AppWorld task success
Curves:
- Full context
- Execution-derived policy memory
- Prompted task+policy memory
- BM25 / embedding retrieval
- Generic memory
- Prompted generic memory
- Recency

Main message:

Policy-conditioned memory provides the largest advantage at tight budgets.

⸻

Panel B: Prompted selector gap

Bar plot at B = 256 or B = 512:

Execution-derived policy memory
Prompted task+policy memory
Prompted task-only memory
Prompted policy-only memory
Prompted generic memory
BM25 / embedding retrieval
Generic memory

Main message:

Prompting with task and policy descriptions does not close the gap to behavior-derived memory.

⸻

Panel C: Cross-policy transfer heatmap

Heatmap:

Rows: memory source policy
Columns: consuming task policy
Cell: task success

Main message:

Memory selected for one policy transfers poorly to another policy under tight budgets.

⸻

10. Minimal Pilot Plan

10.0 Infrastructure prep (must complete before pilot)

> **Review note (R-9):** the new design's experiments require a real
> agentic execution loop, not just sample-action-distribution. The
> setup is non-trivial. Before §10.2 is meaningful, the following
> must be working end-to-end on at least 1 AppWorld task.
>
> Components and current state (as of 2026-05-24):
>
> | Component | State | Source |
> |---|---|---|
> | AppWorld package + `appworld install` data | ✅ done | `/workspace/acon/experiments/appworld/data/` |
> | Working venv (pydantic v1 / sqlmodel 0.0.10) | ✅ done | `/workspace/acon/.venv` (EASMO `.venv` is broken on AppWorld imports) |
> | Agent runner end-to-end | ✅ exists | `/workspace/acon/experiments/appworld/run.py` |
> | Per-task trajectory schema | ✅ verified | `appworld_trajectory.json` (task_id, instruction, num_interactions, completed, final_reward, trajectory[]) |
> | Per-task `env_history.json` (state changes) | ✅ exists | needed for `m*_exec_minimal` final-state derivation |
> | MiniMax endpoint config | ✅ done | `/workspace/acon/configs/private_config.yaml` → `http://10.183.22.68:8005/v1` |
> | Trajectories generated so far | ⚠️ **only 7** | need ~89 (full train) for the pilot |
> | Memory-unit converter (raw env state → units) | ❌ not built | `motivation_v2/units.py` (TODO) |
> | `m*_exec` extractor | ❌ not built | `motivation_v2/exec_memory.py` (TODO) |
> | Compressed-memory executor wrapper | ❌ not built | needs to insert `compressed_memory` into the AppWorld agent's prompt instead of full state |
> | Per-policy-family classifier | ❌ not built | `motivation_v2/policy_family.py` — derive from final-state evaluator's app set |
>
> Order of work:
>
> 1. Smoke-test acon's `run.py` on a single task in the current
>    environment to verify it actually runs end-to-end and produces a
>    trajectory.
> 2. Generate full-context trajectories on the AppWorld `train` split
>    (89 tasks, parallel ≤ 8). Estimated 4–8 h on the in-house
>    MiniMax endpoint.
> 3. Build `motivation_v2/units.py` + `exec_memory.py` +
>    `policy_family.py` against acon's trajectory schema.
> 4. Wire a "compressed-memory executor" — same as acon's
>    `AppWorldAgent` but with the env state (or context) replaced by
>    a memory string of length ≤ B.
> 5. End-to-end smoke: pick 1 successfully solved task, derive its
>    `m*_exec_trajectory(B=512)`, re-run executor with this memory,
>    confirm the run completes (success or failure).
>
> The pilot in §10.2 only starts after all 5 above are green.

10.1 Pilot goal

Before running full experiments, test whether the core pattern exists.

> **Empirical AppWorld audit (2026-05-24, run via
> `motivation_v2/scripts/smoke_data_pipeline.py`):**
>
> ```
> [train] 90 tasks with ground truth
>   single-app: 60/90 (67%)
>   per-family: spotify=42, file_system=9, phone=6, simple_note=3
>   multi_app:  30 (excluded from M3 heatmap)
>   difficulty: 1=36, 2=36, 3=18
>   shared-state cross-policy pairs: 0  ← R-3: must use matched-pair fallback
>
> [dev]   57 tasks with ground truth
>   single-app: 36/57 (63%)
>   per-family: spotify=30, venmo=3, file_system=3
>   multi_app:  21
>   difficulty: 1=30, 2=24, 3=3
> ```
>
> Implications for the pilot, in order of impact:
>
> 1. **spotify is the only family with enough tasks** for a serious
>    per-family compression-pressure curve (42 single-app on train,
>    +30 on dev = 72 total). The headline M1 figure is built on
>    spotify only; other families appear as robustness side panels.
> 2. **train + dev must be combined** to populate all 5 family slots
>    (venmo only exists in dev). This requires regenerating
>    full-context trajectories on dev too, so total trajectory cost
>    is ~147 tasks, not 90.
> 3. **The interesting B regime is [128, 512]** — `m*_exec_minimal`
>    on spotify task `82e2fac_2` saturates at 33 / 33 GT API calls
>    by B=512 (~480 tokens). Beyond B=512 there is no compression
>    pressure on this oracle, so Δ collapses by construction. Add
>    B=64 to the grid as the lowest interesting point; drop B=2048
>    from the headline (keep as "saturation control" only).
> 4. **R-3 fallback is mandatory**: AppWorld's task-id prefix groups
>    encode a single generator, so same-prefix tasks are in the
>    SAME policy family by construction. There are zero same-prefix
>    cross-policy pairs in train. M3 will use matched-pair pairing
>    (closest memory-unit count + entity universe) and the heatmap
>    must disclose this in the figure caption.

⸻

10.2 Pilot setting (revised — strategy-as-policy primary)

Benchmark: AppWorld (split = `train` — 90 tasks; `dev` — 57 tasks
joinable later for additional families)
Executor: MiniMax-M2.5 (in-house endpoint at 10.183.22.68:8005)
Strategies (§2.4a): `direct`, `verify`, `explore`
Tasks: all `train` tasks where **all three strategies succeed at
full context** (the M1/M3 analysis only includes tasks where every
strategy can solve the task baseline; otherwise diagonals would be
artificially deflated for some strategies)
Budgets: 128, 256, 512, 1024, 2048 (full grid so we can see the
valley shape; B=4096 as saturation control)
Memory conditions (Pilot):
  - full context (upper bound)
  - `m*_exec_minimal`             (ground-truth oracle, executor-independent)
  - `m*_exec_trajectory(direct)`  (P_direct's own derived memory)
  - `m*_exec_trajectory(verify)`  (P_verify's own derived memory)
  - `m*_exec_trajectory(explore)` (P_explore's own derived memory)
  - `m_recent`                    (declared generic baseline)
  - BM25                          (retrieval baseline)
  - prompted task+strategy memory (M2 condition)

Pilot deliverables (gate to full run):

* **Manipulation check**: `motivation_v2/scripts/manipulation_check.py`
  produces the per-strategy iter / API call distributions.
  Verdict required: `median_iters(verify) / median_iters(direct)
  ≥ 1.5` AND `show_app_descriptions_first3(explore) ≥ 0.5`. Without
  these, X has not taken and we fall back to Option Y.
* `n_full_context_success_all3 / n_total_train` (R-4, R-10): tasks
  where ALL three strategies succeed at full context.
* `tasks_per_topic_family` table (single-app stratification, R-2)
* M1 success curves vs B for each (strategy, memory) condition,
  stratified by topic (with the bowl-shape probe of R-5)
* M3 3×3 heatmap at the tightest B with overlap signal
* Closure ratio `r_T2(B)` at B=256 / 512 (R-7)

> **Compute estimate**: with the validated 71s/task throughput on
> short tasks (smoke ran in 71s), and verify's 2.4× iter cost:
>
> * `direct` over 90 tasks: ~70s × 90 = **~1.8 h**
> * `verify` over 90 tasks: ~120s × 90 = **~3.0 h**
> * `explore` over 90 tasks: ~90s × 90 = **~2.3 h**
>
> Sequential total: ~7 h. Three parallel jobs (one per strategy)
> bring wall clock to ~3 h with shared MiniMax endpoint contention.
> Pilot is doable in one half-day. Full train + dev (147 tasks)
> raises wall clock to ~5 h.

⸻

10.3 Pilot success criteria

Proceed if the following pattern appears:

Execution-derived policy memory > prompted task+policy memory ≈ BM25 > prompted generic ≈ generic

and:

gap at 256 > gap at 512 > gap at 1024

⸻

10.4 Pilot failure diagnosis

If no gap appears:

Case 1: All methods perform similarly well

Likely reason:

Budget is too loose or tasks are too easy.

Fix:

Lower budget.
Filter for multi-app, multi-hop, preference-dependent tasks.

Case 2: All compressed methods fail

Likely reason:

Memory units are too coarse or executor is too sensitive to missing details.

Fix:

Use smaller memory units.
Add API schema hints.
Evaluate first-action match instead of full task success.

Case 3: Prompted task+policy memory matches execution-derived memory

Likely reason:

Prompting may be sufficient on this benchmark subset.

Fix:

Use harder tasks.
Use tighter budgets.
Analyze whether selector is simply retrieving explicit entity mentions.
Try less direct task descriptions.
Move T2 claim from "prompting fails" to "prompting is brittle and not consistently behavior-optimal."

Case 4: BM25 beats everything

Likely reason:

Tasks are mostly lexical retrieval rather than policy-conditioned memory.

Fix:

Filter for tasks requiring implicit constraints, cross-app dependencies, overwritten preferences, or temporal reasoning.

⸻

11. Full Experimental Checklist

Data preparation

* Download / set up AppWorld.
* Run full-context executor on candidate tasks.
* Keep only full-context successful tasks.
* Group tasks into policy families.
* Convert app states into memory units.
* Store memory units with source app, entity ID, timestamp, and text.
* Extract successful trajectories.
* Record APIs called, arguments, read rows, written rows, and final-state-relevant entities.
* Build execution-derived memory for each task.

⸻

Baseline construction

* Implement generic memory selector.
* Implement recency selector.
* Implement BM25 selector.
* Implement embedding retrieval selector.
* Implement hybrid BM25 + embedding selector.
* Implement prompted generic selector using Minimax-240B.
* Implement prompted task-conditioned selector.
* Implement prompted policy-conditioned selector.
* Implement prompted task+policy-conditioned selector.

⸻

M1: Compression-pressure sweep

* Run each memory condition at each budget.
* Evaluate task success.
* Evaluate first-action match.
* Evaluate API-call F1.
* Evaluate argument F1.
* Plot success vs budget.
* Compute policy-generic gap at each budget.
* Check whether gap decreases with budget.

⸻

M2: Prompted selector gap

* Compute task success for prompted selectors.
* Compute evidence recall against execution-derived memory.
* Compute specialization gain.
* Compute gold-prompt gap.
* Plot bar chart at tight budget.
* Analyze failure cases where prompting misses execution-critical units.

⸻

M3: Cross-policy transfer

* Construct cross-policy memory assignments.
* Run consuming policy tasks with same-policy memory.
* Run consuming policy tasks with cross-policy memory.
* Compute transfer drop.
* Plot policy-source vs policy-consumer heatmap.
* Repeat at multiple budgets.
* Check whether transfer drop shrinks as budget increases.

⸻

M4: Memory-unit ablation

* Remove execution-derived units.
* Remove generic units.
* Remove random units.
* Re-run executor.
* Measure behavior drop.
* Plot ablation impact by memory-unit type.

⸻

LongMemEval / LoCoMo auxiliary

* Select subset with long contexts and diverse question types.
* Define question-type policy views.
* Convert conversations into memory units.
* Run generic / retrieval / prompted selectors.
* Evaluate QA accuracy and evidence recall.
* Compare budget-dependent gaps.
* Add secondary table or appendix figure.

⸻

12. Main Claims Supported by Each Experiment

Claim	Experiment	Required result
Compression pressure induces policy dependence	M1	execution-derived memory beats generic memory most at tight budgets
Prompting does not realize policy-conditioned compression	M2	prompted task+policy memory fails to close gap to execution-derived memory
Memory is not interchangeable across policies	M3	same-policy memory beats cross-policy memory
Behavior-critical memory units are identifiable	M4	removing execution-derived units causes largest behavior drop
Pattern generalizes beyond AppWorld	LongMemEval / LoCoMo	query/policy-conditioned memory helps under tight budgets

⸻

13. What Not to Do

Avoid the following as main motivation evidence:

Do not use LLM-generated oracle memory as the main gold memory

Reason:

It creates a circular argument: the oracle may simply reflect the prompt.

Prefer:

execution-derived memory from successful trajectories

⸻

Do not use token-level Jaccard as a killer metric

Reason:

Low overlap does not necessarily imply behavioral difference.
High overlap does not necessarily imply behavioral equivalence.

Use overlap only as auxiliary analysis.

⸻

Do not rely on token-level leave-one-out saliency

Reason:

Expensive, noisy, hard to interpret.

Prefer:

memory-unit ablation

⸻

Do not use ReAct / Plan / CoT as the main policy distinction

Reason:

Scaffold differences introduce confounds.

Prefer:

same executor, same action space, same decoding, different task/policy family

⸻

14. Final Recommended Experiment Names

Use these names in the paper or internal docs:

M1. Compression-Pressure Sweep
M2. Prompted Selector Gap
M3. Cross-Policy Memory Transfer
M4. Memory-Unit Ablation
M5. Long-Memory QA Generalization

⸻

15. One-Sentence Summary

The motivation experiments should show that, on public agent-memory benchmarks, execution-derived policy memory preserves downstream behavior under tight budgets better than generic or prompted compression, and this advantage disappears as the budget becomes loose.