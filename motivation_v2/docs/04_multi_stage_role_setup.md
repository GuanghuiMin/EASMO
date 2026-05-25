# Multi-stage role-specialised AppWorld setup (real role agents, not projections)

> **Why this exists**: the headline result in `01_experimental_design.md`
> §8 reports cross-role Jaccard 0.04 between four role projections
> (`m_tool` / `m_code` / `m_plan` / `m_verify`) of the same trajectory.
> A skeptical reviewer will (rightly) ask: *the orthogonality is by
> construction — your selection rules are different per role*. This
> doc designs the upgrade that converts the projection-based finding
> into a real-multi-agent finding.

## 1. The critique we are answering

Four role memories sliced from one trajectory:

* `m_tool(t)` keeps API calls, drops Python control flow.
* `m_code(t)` keeps Python patterns, drops API arguments.
* `m_plan(t)` keeps task / milestones, drops both.
* `m_verify(t)` keeps tail observations, drops everything else.

**Reviewer steel-man**: "Of course Jaccard is 0.04 — your slicing
rules emit disjoint kinds of units. Try a setup where each role
*runs* and produces its own trajectory; then compare m*_exec
extracted from those *independent* trajectories. If the orthogonality
holds in real multi-agent execution, the result is meaningful."

We agree. The projection result is a structural lower bound; the
multi-agent result is the load-bearing one.

## 2. Multi-stage AppWorld pipeline

We turn a single AppWorld task into a 3-agent flow on the same
underlying environment. The same model (MiniMax-M2.5) drives all
three agents but each has a distinct system prompt and a distinct
information surface.

```
                 ┌──────────────────┐
   task ─────────│   PLANNER agent  │── plan ──┐
                 │ (LLM call, no env)           │
                 └──────────────────┘           │
                                                ▼
                 ┌──────────────────────────────────────┐
                 │           EXECUTOR agent             │
                 │ (full AppWorldAgent loop with        │
                 │  plan in starting context, env       │
                 │  access via apis.<app>.<fn>)         │
                 └────────────┬─────────────────────────┘
                              │
                              ▼ trajectory tail + agent's claimed answer
                 ┌──────────────────┐
                 │  VERIFIER agent  │── pass / fail
                 │ (LLM call, no env)
                 └──────────────────┘
```

Pipeline outputs per task:

| Artefact | Source | What it contains |
|---|---|---|
| `plan.txt` | planner's LLM response | a numbered plan of sub-goals (text) |
| `executor_trajectory.json` | acon's AppWorldAgent run | (action, output) pairs — same schema as our pilot |
| `verifier_judgment.json` | verifier's LLM response | `{verdict, evidence, confidence}` |

Each artefact represents the *real working memory* of one role. The
m*_exec for each role is now extracted from its own trajectory, not
from a projection of someone else's:

* `m_plan*(task) = the plan agent's output` (the natural plan
  representation it emitted, not a slice of executor's comments)
* `m_tool*(task) = standard m_exec_trajectory of the executor's
  AppWorld run` (reuse our existing builder, drop the slicing)
* `m_code*(task) = control-flow patterns from the executor's
  Python actions` (still a projection, but of the executor's own
  trajectory)
* `m_verify*(task) = the verifier agent's evidence list` (the
  facts the verifier cited as proof)

The strongest cross-role comparison is `m_plan*` vs `m_verify*`,
since both are *outputs of independent LLM agents* (no slicing by
design). Their Jaccard tells us whether the planner and verifier,
working on the same task with their own information needs, end up
referencing the same upstream content.

## 3. Agent prompts (provisional — refine after smoke)

### Planner

```
SYSTEM: You are a PLANNER agent. Your sole job is to read a task
description and emit a concise, executable plan as a numbered list
of sub-goals. You do NOT execute anything. Output only the plan,
nothing else.

USER: Task: {task_instruction}

Produce a plan with 2–5 numbered sub-goals. Each sub-goal should be
a concrete action describable in one sentence. Do not call any APIs.
```

### Executor

The executor is acon's standard `AppWorldAgent` with two extra
constraints in its system prompt:

* It receives the planner's output as a `**PRE-LOADED PLAN**` block
  (analogous to how we inject memory in the runner).
* Its strategy is implicit "follow the plan" (we do NOT layer the
  direct/verify/explore strategies on top in this experiment).

```
SYSTEM: <acon's standard system prompt>

USER: <17 numbered AppWorld instructions>

USER: **PRE-LOADED PLAN** from upstream planner agent. Follow this
plan when executing. You may deviate if you see clear evidence the
plan is wrong, but document why.

{planner_output}

USER: Using these APIs, now generate code to solve the actual task:
Task: {task_instruction}
```

### Verifier

```
SYSTEM: You are a VERIFIER agent. Your job is to decide whether a
downstream agent's claimed answer to a task is correct. You see the
task, the agent's final answer, and the last 5–10 trajectory steps.
You do NOT execute anything. Output a JSON object:
{
  "verdict": "pass" | "fail" | "uncertain",
  "evidence": ["fact 1", "fact 2", ...],
  "confidence": 0.0–1.0
}

USER: Task: {task_instruction}
Claimed answer: {claimed_answer}
Trajectory tail (last 5 steps):
{trajectory_tail}
```

## 4. Memory extraction per role (independent of slicing rules)

### `m_plan*(task)` — planner's actual output
The planner agent's output is already a compressed plan (text).
Each numbered sub-goal becomes a `MemoryUnit`. No slicing of someone
else's trajectory.

### `m_tool*(task)` — executor's own trajectory
Reuse the existing `m_exec_trajectory` builder from
`exec_memory.py` but applied to the executor's trajectory. This is
literally "what API calls the agent made and what it observed",
selected by greedy budget fill.

### `m_code*(task)` — executor's code patterns
Reuse `m_code` from `role_memory.py`. This IS a projection of the
executor's trajectory, but it's a fair one because the executor
genuinely emits Python control flow as part of its work.

### `m_verify*(task)` — verifier's actual evidence list
The verifier agent's output's `evidence` field is already a
compressed evidence list. Each evidence entry becomes a `MemoryUnit`.
No slicing.

So **two of the four** role memories (plan, verify) are *outputs of
independent agents*, not projections. The other two (tool, code)
are projections of the executor's trajectory but the executor's
trajectory itself is *its own independent run*. This is the
strongest version of the cross-role test we can do without a true
multi-agent benchmark.

## 5. Experimental design

### Pilot: 6 tasks across 3 generators

* 6 spotify tasks (same consumers as the cross-task transfer pilot).
* For each, run the 3-agent pipeline.
* Extract 4 role memories.
* Compute pairwise Jaccard at B ∈ {128, 256, 512, 1024}.

Compute: ~30 tasks × 3 agents (planner short + executor full +
verifier short) ≈ 30 × (1 + 13 + 1) = 450 LLM calls × 5 s ≈ 38 min.

### Full: 30 tasks × all 5 single-app generators

Once the pilot validates the pipeline, scale to ~30 tasks. ~150 min
compute on 4 parallel workers (~40 min wall).

### Headline metric

Cross-role Jaccard at B=512 from real-agent trajectories. Target:
**≤ 0.10** (within 2.5× of the projection result of 0.04). If
achieved, the projection-vs-agent critique is fully addressed.

A more aggressive target: cross-role Jaccard between `m_plan*` and
`m_verify*` specifically. Both are independent agent outputs;
their Jaccard is the cleanest possible test. Target ≤ 0.15.

## 6. Implementation outline

| Component | File | Status |
|---|---|---|
| Planner agent (single LLM call) | `motivation_v2/multi_stage_agents.py` | TODO |
| Verifier agent (single LLM call) | `motivation_v2/multi_stage_agents.py` | TODO |
| Multi-stage runner (orchestrator) | `motivation_v2/scripts/run_multi_stage_role.py` | TODO |
| Per-role m*_exec extractor | `motivation_v2/multi_stage_memory.py` | TODO |
| Cross-role Jaccard analysis | `motivation_v2/scripts/analyze_multi_stage_overlap.py` | TODO |
| Smoke test on 1–3 tasks | `motivation_v2/scripts/smoke_multi_stage.py` | TODO |

Estimated code budget: 1 day. Estimated compute budget: ~40 min for
the full 30-task pilot.

## 7. Anti-patterns to avoid

* **Don't have the planner do its planning by calling APIs.** The
  whole point is that the planner sees only the task — it's a
  one-shot LLM call producing a plan, not a multi-step
  agent loop.
* **Don't reuse the strategy variants (direct/verify/explore) here.**
  Those test agent *style*, not agent *role*. The 3-stage pipeline
  is orthogonal.
* **Don't let the verifier execute code.** The verifier inspects
  evidence, doesn't act.
* **Don't measure success on the verifier's pass/fail.** Use
  AppWorld's native ground-truth evaluator (already in ACON). The
  verifier's judgment is data, not the metric.
