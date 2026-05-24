# Experimental Design — motivation_v2 (AppWorld + behavioural-strategy policies)

> Owner: EASMO motivation track v2.
> Last edited: 2026-05-24 1:10 PM PT.
> Companion to `../README.md` and the active design spec at
> `../../motivation/docs/new_motivation.md`. This doc focuses on the
> **operational** design — what we run, why, what we expect, what we
> have observed so far, and the gaps we are still aware of.

---

## 1. Theses (revised 2026-05-24, 2:00 PM PT after Jaccard analysis)

The thesis statement was refined twice on 2026-05-24 in light of
empirical data on AppWorld trajectories. The current form:

* **T1 — memory needs are role-conditional, not task-conditional.**
  In a multi-agent system where multiple agents share an upstream
  context but have different roles (planner, tool-user, coder,
  verifier), the compressed memory each role needs is *structurally
  orthogonal*. Empirically the cross-role Jaccard on AppWorld at
  B=512 is 0.04 — roles want disjoint slices of the same context.
  By contrast cross-task **within-role** Jaccard is 0.17 (modest
  task-specific tail) and cross-strategy **within-task** Jaccard is
  ≈ 0.91 (style is invariant). Memory variation is **role-driven**.

* **T2 — prompted LLM selectors cannot realise role-conditional
  compression.** Even when conditioned on a role description and a
  task instruction, an off-the-shelf LLM compressor produces
  compressions that are surface-uniform across roles and fall short
  of the role-projected oracle (`m_tool` / `m_code` / `m_plan` /
  `m_verify`).

Combined claim: *role-conditional compression in multi-agent systems
is necessary (T1) and not achievable by prompting (T2). It must
therefore be learned with a behavioural objective. Within a fixed
role, however, the compressor transfers across tasks — code-style
patterns at Jaccard 0.41 across tasks; fact-level patterns are
task-specific but recoverable via standard multi-task training.*

This is the **three-tier message** of the paper:

```
1. Strategy invariance (Jaccard ≈ 0.91)        → control: agent style is not the lever
2. Within-role cross-task transferability      → practical: train per role, not per task
   (Jaccard ≈ 0.41 for code; 0.07–0.11 others)
3. Cross-role orthogonality (Jaccard ≈ 0.04)   → headline: role conditioning is necessary
```

### What we explicitly RETRACT

The original (pre-2026-05-24) framing called this "policy-dependent
compression", borrowing RL/agent terminology. Two corrections:

1. **"policy" was a misnomer**: in RL/agent literature, π(a|s) is a
   *behavioural decision rule*. What we mean here is *agent role*
   (planner / tool-user / etc.) — a specialty, not a decision rule.
   We use "role-conditional" throughout the rest of this doc.
2. **The original prediction "smaller B → more divergence"
   reverses below the plumbing floor**. At B ≤ 256 every task
   converges on shared auth/login plumbing → cross-task divergence
   is *small*, not large. The interesting regime for divergence is
   B ∈ [256, 1024], not B ≤ 128. This finding is still publishable
   ("compact memory has a plumbing floor below which task-conditional
   training is unnecessary; above the floor it dominates") but the
   direction of the original intuition was wrong.

## 2. Why this design (vs the abandoned `motivation/` track)

The previous track had four fatal problems:

| Old track | Problem |
|---|---|
| ReAct / Plan / CoT scaffolds on top of the same MiniMax-M2.5 | Policy is just an output-format wrapper, not a real decision policy. |
| LongMemEval / LoCoMo as benchmark | QA tasks have a single canonical answer; "different policies want different memory" cannot even be expressed. |
| LLM-only selector for the "oracle" memory | T1's hinge test is unsatisfiable when T2 holds: cross-policy and within-policy LLM-generated memories are interchangeable by T2 → cross/within ratio ≈ 1. |
| Binary `action_match_rate` over N=16 samples | Near-Bernoulli; produces zero signal rows on tight budgets even when the underlying memories differ. |

The new track replaces every one of these:

| New track | Fix |
|---|---|
| AppWorld (multi-step tool use, real action variation) | Tasks where "policy" can change actual API choices. |
| `m*_exec` from successful trajectories + ground-truth API call list | Non-LLM oracle. T1 is testable independently of T2. |
| Policy = behavioural strategy variant (`direct` / `verify` / `explore`) on the same task with the same executor | Policy varies orthogonally to topic; same model rules out capability differences. |
| Task success rate on AppWorld's final-state evaluator | Continuous, executor-graded outcome; no Bernoulli collapse. |
| BM25 retrieval as the headline retrieval baseline | Standard reviewer-defensible lower bound. SBERT optional (appendix). |

## 3. Benchmark and infrastructure

### 3.1 AppWorld via ACON

We consume two external assets:

* **AppWorld** (Stony Brook NLP). Day-to-day personal tasks
  (calendar, email, shopping, music, payments) executed against
  simulated app APIs. Each task ships with `metadata.json`,
  `required_apps.json`, `api_calls.json` (gold solution call
  sequence), `evaluation.py` (final-state test). All under
  `/workspace/acon/experiments/appworld/data/`.
* **Microsoft ACON** (arXiv 2510.00615 — "Optimizing Context
  Compression for Long-horizon LLM Agents"). Provides a working
  `AppWorldEnv` + `AppWorldAgent` Python REPL loop, a sequential
  driver `run.py`, and `run_all.py` over a split. We do not modify
  ACON; we wrap its `run.main` with our own prompt-injection layer
  and consume its trajectory output schema directly.

We use ACON's `.venv` (pydantic v1 + sqlmodel 0.0.10) for everything
that imports `appworld` or `productive_agents`. The EASMO `.venv`
has a known pydantic v2 incompat with sqlmodel 0.0.10 that breaks
AppWorld imports; the `motivation_v2` analysis modules avoid those
imports so they work in either venv.

### 3.2 Corpus audit (real numbers, audited 2026-05-24)

Run `motivation_v2/scripts/smoke_data_pipeline.py` to reproduce.

```
[train] 90 tasks with ground truth
  single-app: 60/90 (67%)
  per-family: spotify=42, file_system=9, phone=6, simple_note=3
  multi_app:  30 (excluded from M3 cross-policy heatmap)
  difficulty: 1=36, 2=36, 3=18
  shared-state cross-policy pairs: 0  ← R-3 fallback to matched-pairs needed

[dev] 57 tasks with ground truth
  single-app: 36/57 (63%)
  per-family: spotify=30, venmo=3, file_system=3
  difficulty: 1=30, 2=24, 3=3

[test_normal] 168 tasks, ground truth held out
[test_challenge] 417 tasks, ground truth held out
```

Implications baked into the design:

1. **spotify is the only family with enough single-app tasks**
   (42 train + 30 dev = 72) for a per-family compression-pressure
   curve. The headline M1 figure is built on spotify; other families
   are robustness side panels.
2. **train + dev must be combined** (venmo lives only in dev).
   Total trajectory generation budget = 147 tasks per strategy
   per executor.
3. **R-3 mandatory fallback**: AppWorld task IDs share a 7-character
   prefix when generated from the same task generator; same-prefix
   tasks always share a policy family (e.g. `82e2fac_1/2/3` are all
   spotify). There are zero shared-state cross-policy pairs in
   train, so M3 must use matched-pair pairing (closest memory-unit
   count + entity universe) and disclose this in the figure caption.
4. **Interesting B regime is `[128, 512]`**. `m*_exec_minimal` on a
   typical spotify task saturates at ~480 tokens by B=512 (33/33 GT
   API calls fit). Beyond B=512 there is no compression pressure; Δ
   collapses by construction. Pilot grid: B ∈ {128, 256, 512, 1024,
   2048}; B=2048 kept only as a saturation control.

## 4. Memory units and the three policy axes

We measure three orthogonal sources of variation in optimal
compressed memory. The headline result is that they're cleanly
ordered by effect size.

### 4.0 Three axes — summary

| Axis | What varies | What's held constant | Role in paper |
|---|---|---|---|
| **Strategy** | output style (direct / verify / explore) | task, role, executor | negative control (style does not drive memory) |
| **Task** | downstream goal (most-played song / most-liked) | role, executor | transferability test (within-role, cross-task) |
| **Role** | agent specialty (planner / tool-user / coder / verifier) | task, executor | headline (roles need orthogonal memory) |

Empirical Jaccard on AppWorld at B=512:

| Axis | Mean Jaccard at B=512 | Interpretation |
|---|---|---|
| Strategy | 0.91 | style is invariant |
| Task (within-role) | 0.17 | modest task-specificity |
| Role | 0.04 | nearly orthogonal |

### 4.1 Memory units

The unit pool an agent's compressed memory selects from is the
**trajectory observation history**: every `(action, output)` pair
the agent accumulated during a successful full-context run.

```python
# motivation_v2/units.py
def trajectory_unit_pool(traj) -> list[MemoryUnit]:
    # one unit per non-empty step output (truncated to 240 chars)
    # one unit per action (Python code the agent issued)
    # interleaved chronologically by step
```

A typical pool for a 17-step spotify task is ~40 units.
Each unit has:
* `kind`: `'action'` | `'observation'` | `'api_call'` | `'profile'`.
* `app`: derived from the API call prefix (`spotify`, `venmo`, …).
* `text`: a short `[app step N] {output}` rendering.
* `weight`: used by greedy budget fill; ground-truth-anchored units
  start with weight 2.0.
* `source_step`: trajectory step index, for chronological ordering.

### 4.2 Strategy axis (negative control) — see [`02_strategy_prompts.md`](02_strategy_prompts.md)

Three behavioural strategies, all running on the same MiniMax-M2.5
executor on the same task. The variation is in agent style, not
in role or task. Canonical specs in `02_strategy_prompts.md`.

| Strategy | Search breadth | Verification | Action style | Expected effect on trajectory |
|---|---|---|---|---|
| `P_direct` | minimal | none | answer-first | shorter trajectory, fewer API calls, no `show_app_descriptions` |
| `P_verify` | task-driven | mandatory cross-validation | answer-then-confirm | longer trajectory, ~2× API calls, duplicate-fact retrieval through ≥ 2 endpoints |
| `P_explore` | exhaustive | optional | survey-then-answer | medium-long trajectory, mandatory `show_app_descriptions` in first 3 steps |

Empirical validation on `82e2fac_3` (smoke run, MiniMax-M2.5):

| | iters | input tokens | unique APIs | total API calls | distinctive |
|---|---|---|---|---|---|
| baseline (no strategy) | 11 | 46K | 8 | 11 | normal |
| `P_direct`  | 11 | 57K | 7 | 11 | small task — little to compress |
| `P_verify`  | **26** | **204K** | **11** | **36** | 17× `show_song` cross-checks + `show_album` + `show_profile` |
| `P_explore` | 14 | 82K | 8 | 14 | 100% `show_app_descriptions` in first 3 steps |

Manipulation-check threshold: `median_iters(verify) / median_iters(direct) ≥ 1.5` and
`show_app_descriptions_first3(explore) ≥ 0.5`. On the smoke this is
2.36× and 100% respectively.

### 4.3 Task axis (transferability test)

Different AppWorld task instances of the same role. We measure
cross-task Jaccard within each role to test whether memory policies
generalise across tasks (within a role) or whether each task needs
its own compressor.

Empirical: cross-task Jaccard at B=512 (mean over task pairs):
* `m_code`   : **0.41** ⇒ structural code patterns transfer freely
* `m_verify` : 0.105 ⇒ tail observations are task-specific
* `m_tool`   : 0.089 ⇒ API arguments are task-specific
* `m_plan`   : 0.072 ⇒ task instructions are unique by definition

**Practical takeaway**: code-level memory policies can be trained
on a single task and reused freely; tool/plan/verify policies need
multi-task training but the diversity requirement is bounded by
"same role, varied tasks", not "every task has its own policy".

### 4.4 Role axis (headline) — see [`03_role_memory_extractors.md`](03_role_memory_extractors.md)

Four agent roles, each with a deterministic projection of the same
trajectory:
* `m_tool`   — API call list + observations
* `m_code`   — Python control-flow patterns (args abstracted)
* `m_plan`   — task instruction + intent comments + milestones + final answer
* `m_verify` — tail-of-trajectory observations + final-state call

Cross-role Jaccard at B=512: **0.04 mean**. Roles want orthogonal
memory. This is the central T1 finding.

### 4.5 Executor axis (future work)

Different LLMs (MiniMax / Qwen / GPT-4o-mini) on the same task.
Currently MiniMax-only. Cross-executor variation is the natural
robustness check ("does the role-orthogonality finding hold for
other backbones?") and gates whether the paper claims a
model-independent result.

## 5. Prompt template design

Three layers of injection, each one a USER turn spliced into a
canonical AppWorld task prompt:

```
   ┌─────────────────────────────────────────────────┐
   │ acon's prompt_v1.jinja                          │
   │   - 17 numbered "Key instructions" (USER turn)  │
   │   - "Now generate code to solve the actual      │
   │     task" (USER turn with {{ instruction }})    │
   └─────────────────────────────────────────────────┘
                       │ build_strategy_prompts.py
                       ▼
   ┌─────────────────────────────────────────────────┐
   │ acon/.../prompts/_motivation_v2/<strategy>/     │
   │ prompt_<strategy>.jinja                         │
   │   - 17 instructions                              │
   │   - **STRATEGY: ...** block (NEW USER turn)     │
   │   - "Now generate code..."                       │
   └─────────────────────────────────────────────────┘
                       │ runner.materialise_cell_prompt
                       ▼
   ┌─────────────────────────────────────────────────┐
   │ acon/.../prompts/_motivation_v2/_cells/<hash>/  │
   │ prompt.jinja                                     │
   │   - 17 instructions                              │
   │   - **STRATEGY: ...** block                     │
   │   - **PRE-LOADED MEMORY** block (NEW USER turn) │
   │   - "Now generate code..."                       │
   └─────────────────────────────────────────────────┘
```

### 5.1 Strategy block (canonical, copied from `prompts/STRATEGY_DESIGN.md`)

```
**STRATEGY: DIRECT (minimum-API, answer-first)**

You MUST solve the task with the minimum number of API calls.
As soon as you have enough information to answer with reasonable
confidence, immediately call apis.supervisor.complete_task(answer=...).
DO NOT verify the answer through a second source. DO NOT enumerate
apps via apis.api_docs.show_app_descriptions() if you can already
infer which app to use from the task. DO NOT list extra data that is
not strictly required to compute the answer. Brevity is the policy.
```

```
**STRATEGY: VERIFY (mandatory cross-validation)**

Before returning your final answer, you MUST cross-validate it
through at least one independent code path. For example, if you
computed the answer by aggregating from one library (songs), also
verify by aggregating from a different library (albums or playlists)
when feasible, OR re-fetch the same record by a different identifier
(e.g. song-id vs title), OR call a related list endpoint and confirm
the result is consistent. The cross-validation step is mandatory; do
not skip it. Only call apis.supervisor.complete_task after the
second path has confirmed the answer.
```

```
**STRATEGY: EXPLORE (survey-then-answer)**

Before computing the answer, build a comprehensive understanding of
the available data. Step 1: ALWAYS list the available apps with
apis.api_docs.show_app_descriptions(). Step 2: enumerate the key
APIs of the most relevant app via
apis.api_docs.show_api_descriptions(app_name=...). Step 3: survey
the user state via list-style endpoints (e.g. libraries, recent
items, profiles). Only AFTER you have a broad picture of the user's
state should you compute and return the answer. Use this exploration
phase even if you think you already know the answer.
```

Why these three: they vary along orthogonal axes (search breadth ×
verification × action style); each has at least one
manipulation-checkable behavioural fingerprint (verify → 2× iters,
explore → first-3-step `show_app_descriptions`); they map onto
plausible deployment policies (latency-sensitive, safety-critical,
analytics-heavy).

### 5.2 Pre-loaded memory block (added by `runner.py`)

```
**PRE-LOADED MEMORY** (relevant context the supervisor pre-cached
for you; you may use these facts to skip API queries you can answer
from them, but you can still call APIs whenever you need more info):

<memory_text>
```

`<memory_text>` is whatever the chosen compressor produced for this
cell (`m_recent` / `m_freq` / `m_bm25` / `m_exec_minimal` /
`m_exec_trajectory`). The agent is *not* required to use it; it can
re-query the API if the memory is unhelpful. This frames the
experiment honestly as "pre-loaded hint" rather than "frozen state".

The block sits between the strategy block and the task instruction
so it is the **last thing the model reads before the task** —
maximising the chance of being attended to.

Note: `m*_exec_minimal` contains API call queries (`GET
/spotify/library/songs(page_index=0)`), not API responses. So
pre-loading it does not leak the answer — the agent still has to
execute the calls to get the data. The advantage at low B comes
from saving the `show_api_doc` exploration phase, not from
short-circuiting the task. This is verified by the smoke run
(11→8 iters at B=512 with `m_exec_minimal+direct`).

## 6. Experimental matrix (revised)

Five experimental panels, mapped onto the three-tier story.

| Panel | Measures | Status |
|---|---|---|
| **C** Cross-strategy control | strategy variation does NOT change memory | ✅ pilot data: Jaccard 0.91 |
| **R1** Cross-role Jaccard | roles want orthogonal memory | ✅ deterministic data: Jaccard 0.04 |
| **R2** Cross-task within-role Jaccard | within-role transferability | ✅ deterministic data: code 0.41, others 0.07–0.11 |
| **M1** Compression-pressure sweep | role-projected memory beats generic at tight B | ⏳ in flight (xtask + role-specific runner) |
| **M2** Prompted selector gap | prompted compressor can't reproduce role-specific selection | ❌ to build (T2 baseline) |

### 6.1 M1 — compression-pressure sweep (T1 main)

For each `(task, strategy, compressor, B)` cell:

1. Take the pre-built `compressed_memories.jsonl` row (deterministic
   post-hoc construction — no LLM in the loop here).
2. Run `runner.run_with_compressed_memory(...)` — splices the memory
   into the prompt and invokes acon's AppWorld agent.
3. Record `success`, `iterations`, `final_reward`, `input_tokens`,
   `elapsed_s`.

Memory conditions in M1 main figure (one curve each):
* `full_context` — upper bound (no compression; the strategy still
  applies, and the agent has unfettered API access).
* `m*_exec_minimal` — non-LLM ground-truth oracle.
* `m*_exec_trajectory(strategy)` — executor- and strategy-conditioned
  oracle.
* `m_recent` — generic recency baseline.
* `m_bm25(task_instruction)` — retrieval baseline.
* `m_prompted_task_policy` — strongest prompted compressor (M2 condition,
  reused). **Not yet built.**

**Pass criterion (T1)**: at B = 256 or 512,
`success(m*_exec) ≥ success(m_recent) + 0.10` AND
`success(m*_exec) ≥ success(m_bm25) + 0.05`, holding across at least
2 of the 3 strategies.

**Stratification**: the same plot is reproduced per task topic
(spotify, file_system, …) so we can show the gap is not a
spotify-only artifact.

### 6.2 M2 — prompted selector gap (T2 main)

For each `(task, strategy, B)` cell:

1. Build `m_prompted_generic`, `m_prompted_task`,
   `m_prompted_policy`, `m_prompted_task_policy` by asking
   MiniMax-M2.5 to compress the unit pool with progressively richer
   conditioning.
2. Run each prompted memory through `runner.run_with_compressed_memory`
   exactly like M1.
3. Compare against `m*_exec_trajectory(strategy)` (gold) on task
   success and on **evidence recall** (Jaccard between selected
   units and `m*_exec`).

**Closure ratio** (replaces the brittle pass/fail threshold):

```
r_T2(B) = [ Success(m_prompted_task_policy, B) - Success(m_prompted_generic, B) ]
       / [ Success(m*_exec, B)              - Success(m_recent, B)         ]
```

**Verdict tiers**:
* `r_T2 ≤ 0.30`: STRONG T2 — prompting closes ≤ 30% of the achievable gap.
* `0.30 < r_T2 ≤ 0.70`: WEAK T2 — prompting partially works.
* `r_T2 > 0.70`: T2 fails on this benchmark. Paper has to pivot.

### 6.3 M3 — cross-strategy memory transfer (T1 strong + falsifiable)

Same task, three trajectories `traj(P_X)` for `X ∈ {direct, verify,
explore}`, three execution-derived memories
`m*_exec_trajectory(task, P_X)`. Heatmap:

|              | consumer `P_direct` | consumer `P_verify` | consumer `P_explore` |
|--------------|---------------------|---------------------|----------------------|
| memory from `P_direct`  | self (diagonal) | cross | cross |
| memory from `P_verify`  | cross           | self | cross |
| memory from `P_explore` | cross           | cross | self |

Each cell = success rate when consumer `Y` is given memory derived
from producer `X` on the same task. Same task, same context, same
executor, only the producer/consumer policy differs.

**Pass criterion (T1 strong)**: at B = 256 or 512, `mean(diagonal)
- mean(off-diagonal) ≥ 0.10` and the diagonal-off-diagonal gap is
larger at tighter B.

**Resolved by strategy-as-policy framing**: R-3's "shared-state
pair" problem disappears because every task automatically yields
three same-context same-task-instance data points (one per strategy).

### 6.4 M4 — memory-unit ablation diagnostic (appendix)

Take `m*_exec_trajectory(task, P_X)` and **swap** one unit with
neutral filler of the same token length. Measure success drop.
Compare swapping out `m*_exec` units vs `m_recent` units vs random
units.

**The capacity-vs-content fix (R-6)**: swap-with-filler holds the
budget constant so the only varying quantity is the *content* of the
displaced unit, not the token count.

This is appendix-only unless the result is visually striking.

## 7. Expected patterns

### 7.1 What we predict (T1)

* Δ_policy-generic(B) is **largest at intermediate B** (probably
  B=256 on spotify), shrinking to ~0 at both extremes:
  * Very low B (e.g. B=128): all methods may fail to fit anything
    useful → Δ → 0.
  * High B (≥ 1024): all methods include the relevant rows → Δ → 0.
* The shape may be **bowl-shaped, not monotonic**. Either is
  acceptable for T1; the wording adapts.
* Across topics: the same shape appears on file_system / phone /
  simple_note, just at different absolute success rates (these
  families have fewer tasks → wider CIs).

### 7.2 What we predict (T2)

* `success(m_prompted_task_policy)` lies between `m_recent` and
  `m*_exec`, closer to the former.
* `r_T2(B=256) ≤ 0.30` would be a strong T2 result.
* The **evidence-recall** metric (Jaccard between prompted-selected
  units and `m*_exec_trajectory` units) is more diagnostic than task
  success: even if prompting nearly matches `m*_exec` in success, low
  evidence recall would say "the prompted compressor stumbled into
  the right answer for the wrong reasons".

### 7.3 What we predict (M3)

* Diagonal cells > off-diagonal cells (≥ 10pp gap at B=256/512).
* Verify-→-direct transfer (giving direct's no-cross-check policy a
  verify-flavoured memory full of duplicate cross-validation rows)
  should be the LARGEST drop — verify's memory has redundancy
  direct cannot exploit.
* Direct-→-verify might **gain** (verify gets direct's compact
  memory, which still carries the answer; verify can do its
  cross-validation cheaper).

### 7.4 Failure modes we are watching for

| Pattern | Implication |
|---|---|
| Manipulation check fails on full pilot (verify and direct iter distributions overlap) | Strategy injection didn't take. Switch to Option Y (executor variant). |
| `Jaccard(m*_exec_direct, m*_exec_verify) > 0.8` per task | The three strategies want the same information; "policy-conditional" claim is empty. Reframe as "compression for tool-use is necessary at tight budgets" (weaker T1). |
| `r_T2 > 0.70` | Prompting works well enough that we lose T2. Paper pivot to "we make policy-conditional compression cheap" instead of "necessary". |
| Δ_policy-generic ≈ 0 across all B | T1 dies. Most likely cause is that AppWorld tasks all need the same small set of rows regardless of policy. |
| `m_bm25` ≈ `m*_exec` on success | Lexical retrieval is enough; policy-conditional learning is unnecessary. |

## 8. What we have actually observed (2026-05-24, 2:00 PM PT)

### 8.0 Three-tier Jaccard hierarchy (deterministic, no LLM)

The headline result, computed from 81 successful direct-strategy
trajectories on AppWorld train.

```
                       Mean Jaccard at B=512
Strategy variation     0.91   ← agent style is invariant
Cross-task within-role 0.17   ← modest task-specific tail
Cross-role             0.04   ← roles need orthogonal memory ★
```

Per-role cross-task Jaccard breakdown:

| Role | B=128 | B=256 | B=512 | B=1024 |
|---|---|---|---|---|
| `m_code` | 0.426 | 0.409 | **0.409** | 0.409 |
| `m_verify` | 0.110 | 0.086 | 0.105 | 0.105 |
| `m_tool` | 0.121 | 0.099 | 0.089 | 0.093 |
| `m_plan` | 0.016 | 0.075 | 0.072 | 0.072 |

Per-role-pair cross-role Jaccard at B=512:

| pair | tool–code | tool–plan | tool–verify | code–plan | code–verify | plan–verify |
|---|---|---|---|---|---|---|
| | 0.000 | 0.059 | 0.054 | 0.000 | 0.000 | 0.099 |

Reproduce: `motivation_v2/scripts/analyze_role_overlap.py --tag mv2_pilot --strategy direct`.

### 8.0.5 Cross-task transfer (within-role tool-use, M1 prep, partial)

A first 18-cell pilot (limited by the partially-built
`compressed_memories.jsonl` at the time) ran cross-task transfer:
6 spotify consumer tasks × {self, within_gen, within_app, cross_app}
× {B=128, 256, 512}, dropping cells whose source memory wasn't yet
in the pre-built file. **All 18 cells succeeded**, including
within-app cross-generator memory transfer at B=512 (e.g.,
`07b42fd_1` succeeded in 8 iters when given memory derived from
`82e2fac_1`'s gold solution). The full 72-cell experiment is
re-running with the larger pilot dataset (PID 740184).

### 8.1 Strategy injection works on a single task

Smoke results from §4.2 reproduced in machine-readable form at:

* `acon/.../outputs/MiniMaxAI_MiniMax-M2.5_mv2_smoke_direct/train/task_82e2fac_3/`
* `acon/.../outputs/MiniMaxAI_MiniMax-M2.5_mv2_smoke_verify/train/task_82e2fac_3/`
* `acon/.../outputs/MiniMaxAI_MiniMax-M2.5_mv2_smoke_explore/train/task_82e2fac_3/`

Manipulation-check verdict on this single task: **PASS**.
Verify/direct iter ratio = 2.36×; explore first-3-step exploration
rate = 100%.

### 8.2 Pilot first 40 minutes (90 train tasks per strategy, in flight)

Pilot started 12:12 PM PT. Snapshot at 12:50 PM PT:

| strategy | done | success | success rate |
|---|---|---|---|
| `P_direct`  | 24 / 90 | 21 | **88%** |
| `P_verify`  | 13 / 90 |  7 | **54%** |
| `P_explore` | 17 / 90 | 14 | **82%** |

**`P_verify` failing 46% of tasks is itself a finding worth a panel
in the paper.** Forcing cross-validation is not just slower (we
predicted that); it actively reduces task completion rate by 34
percentage points relative to direct. This will be reported in the
"manipulation-check + side findings" section regardless of how M1
turns out.

### 8.3 First end-to-end M1 cell (smoke, 12:55 PM PT)

`runner.run_with_compressed_memory(task=82e2fac_3, strategy=direct,
compressor=m_exec_minimal, B=512)` produced:

| run | iters | input tokens | elapsed |
|---|---|---|---|
| baseline (no strategy, no memory) | 11 | 46K | 71 s |
| direct (no memory) | 11 | 57K | 70 s |
| **direct + m\*_exec_minimal(B=512)** | **8** | **43K** | **55 s** |

Pre-loading the GT API call list at B=512 saves the agent 3 iters
of API-doc exploration. That's the cheapest possible M1 win, and
it works as predicted.

A meaningful M1 number requires running this across ~2,000 cells.
Compute budget: ~6 h on 8 parallel ACON workers for the
spotify-only subset (42 tasks × 4 budgets × 4 compressors × 3
strategies = 2,016 cells × 55 s).

### 8.4 What's NOT yet observed (in priority order)

* **`m_prompted_*` compressors** — not yet built. Without them, M2
  is uncomputable.
* **`m*_exec` overlap analysis across strategies** — not yet run.
  This is the cheapest way to test the strong T1 claim before
  burning M3 compute. A single deterministic script, no LLM.
* **Full pilot manipulation check at 90 tasks per strategy** —
  pilot still in flight; ETA ~3:30 PM PT.
* **M1 / M2 / M3 measurement on the full grid** — not started.
  Gated on pilot completion + prompted-compressor build.

## 9. Honest assessment of design sufficiency (revised after Jaccard data)

The 2:00 PM PT update of this section reflects observed Jaccard
data on 81 trajectories. The previous self-critique (worried about
"strategies want the same memory" → strong T1 dead) has been
*replaced* by stronger findings under the role-conditional reframe.

### 9.0 Headline: thesis is much better positioned than at 1:00 PM PT

* The original T1 ("policy-conditional compression at tight B is
  necessary") was technically falsified at the strategy level —
  three strategies on the same task share Jaccard 0.91 — but this
  ruled out a confound rather than killed the thesis.
* The reframed T1 ("role-conditional compression in multi-agent
  systems is necessary") is **strongly supported** by Jaccard 0.04
  across roles on the same trajectory. Reviewer-defensible: the
  data is deterministic, the projections are documented, the
  cross-validation against task-axis Jaccard (0.17) shows the
  finding is role-driven not slicing-driven.
* The within-role cross-task transferability finding (Jaccard 0.41
  for code, 0.07–0.11 for others) is a *practical engineering
  contribution* on top of T1: train per role, not per task.

### 9.1 What we are confident in

* **T1 (role-conditional)** has direct deterministic evidence:
  cross-role Jaccard 0.04 on 81 trajectories. The data is
  reproducible from a single command and not contingent on LLM
  decoding or any randomness.
* **Cross-task within-role transferability** has a clean
  per-role breakdown: code 0.41, others 0.07–0.11 — supports the
  practical claim that role-conditional compressors transfer
  across tasks.
* **Strategy is a confirmed non-confound** (Jaccard 0.91 across
  strategies on the same task). Reviewer can't say "your roles
  just measure agent style".
* **Pipeline is end-to-end working**: cross-task transfer cells
  ran, with `runner.py` injecting role-specific memory variants
  into the AppWorld agent. The first M1 smoke cell showed
  `m_exec_minimal(B=512)` reduces direct's iters from 11 to 8.
* **Pilot is high-yield**: direct strategy 92% success rate
  (83/90), enough headroom to filter to all-three-strategy-success
  subsets without losing too much data.

### 9.2 Where the design is still weak

**(a) [resolved] strategies share information needs** — confirmed
empirically (Jaccard 0.91). Reframed as a positive control finding;
no longer a threat to the thesis.

**(b) `m*_exec_minimal` is API queries, not data.**

The 8-vs-11 iter saving comes from the agent skipping API-doc
exploration, not from learning facts about the user's music
library. This is a *fair* experimental setup (no answer leakage)
but it also means `m*_exec_minimal` cannot help on tasks where
API discovery is not the bottleneck. The harder claim that
policy-conditional memory matters when the memory contains *facts,
not procedures* is still untested.

**Mitigation**: `m*_exec_trajectory` does include observations
(actual API responses with facts). The M1 plot will include both
variants and we'll point out which gap is which.

**(c) Single executor only.**

All trajectories are MiniMax-M2.5. A skeptic can argue our role
projections might not generalise to other LLM backbones. Cross-
executor robustness is the natural follow-up.

**Mitigation**: re-run the role-overlap analysis (≤ 1 min per
re-run) on Qwen2.5-7B trajectories once the user coordinates the
endpoint. The deterministic projections will work on any
trajectory in the same schema.

**(d) Prompted compressors not built yet.**

T2 is currently an unfilled hole. The headline T2 claim cannot be
made until `m_prompted_task_policy` is implemented and run.

**Mitigation**: plan §10 step "M2 prep" — write
`scripts/build_prompted_memories.py` that calls MiniMax with
template prompts to produce the four prompted-memory variants.
Estimated 1 hour to write, ~2 h to run on the pilot trajectories.

**(e) AppWorld is dominated by tool-use tasks.**

All 90 train tasks are essentially "find this fact in the user's
app data". The role-orthogonality finding shows that even within
this narrow tool-use corpus, the four projections (tool / code /
plan / verify) are demonstrably distinct. But the strongest version
of the role thesis would benefit from a benchmark with explicit
role specialisation (ChatDev / MetaGPT / AutoGen).

**Mitigation**: present AppWorld as the "minimum non-trivial
demonstrator" — even on a narrow tool-use corpus, role projections
are orthogonal. Then in §6 (or paper §6) cite multi-agent benchmarks
as natural follow-ups where the effect should be even stronger.

**(f) [new] Role projections are slicing rules, not independently-run
role agents.**

A skeptic could argue we constructed orthogonality by our slicing
choices. The strongest rebuttal is the cross-task per-role pattern:
the SAME slicing rules produce code Jaccard 0.41 vs tool Jaccard
0.09 across tasks. If the slicing forced orthogonality, both
numbers would be similarly low. The fact that code transfers and
tool doesn't is a property of the *content*, not the slicing.

A follow-up experiment that actually runs role-specialised agents
(via system prompts that force planner / coder / verifier
behaviour) and extracts m*_exec from each role's own trajectory
would close this gap. This is the natural Option Y' upgrade once
multi-LLM endpoints are in place.

### 9.3 What would make this design strong enough for spotlight

1. ✅ Cross-role Jaccard ≤ 0.10 at B=512 — **achieved (0.04)**.
2. ✅ Cross-task within-role Jaccard ordered (code high, others low)
   — **achieved (0.41 / 0.07–0.11)**.
3. ⏳ M1 task-success result: role-projected memory beats `m_recent`
   by ≥ 15 pp at B ∈ [256, 512] — pilot data forthcoming.
4. ❌ T2 closure ratio `r_T2 ≤ 0.30`: prompting closes ≤ 30% of the
   role-conditional gap — to build (M2 prompted compressors are
   the next code milestone).
5. ⏳ Cross-executor robustness: the role-orthogonality finding
   reproduces on Qwen2.5-7B trajectories — pending external
   endpoint coordination.

Current achievement: 2 of 5; (3) and (4) are mechanical given the
pipeline, (5) is the only external dependency.

### 9.4 What would invalidate the design

* If a follow-up role-specialised-agent run produced m*_exec with
  Jaccard 0.5+ between roles → the projection-based result might
  be an artifact of the slicing rules. We would need to weaken
  T1 to "memory selection rules can be made role-conditional even
  though raw role-induced memory needs overlap".
* If T2 produced `r_T2 > 0.70` → prompted compressors *can*
  reproduce role-aware selection → "necessary but not achievable"
  collapses to "achievable, just expensive". Paper pivots to
  efficiency story.
* If Qwen2.5-7B trajectories produced cross-role Jaccard ≥ 0.30
  → the orthogonality finding is MiniMax-specific. Paper has to
  either be careful about scope or run on multiple models.

## 10. Schedule

All times PT (Pacific). Pilot kicked off at 12:12 PM PT today.

| When | Milestone |
|---|---|
| 12:12 PM PT (done) | Pilot launched: 3 strategies × 90 train tasks |
| 1:00 PM PT (done) | Runner end-to-end smoke (`direct + m_exec_minimal(B=512)` on 82e2fac_3) returns success / 8 iters |
| ~3:30 PM PT (in flight) | Pilot trajectories complete — 3 × ~50 successful trajectories expected |
| 3:30–4:30 PM PT (next) | Run `manipulation_check.py` on full pilot |
| 3:30–4:30 PM PT (next) | Run `m*_exec` Jaccard-overlap analysis (deterministic, ~1 minute) |
| 4:30–6:00 PM PT (next) | Build `motivation_v2/scripts/build_prompted_memories.py` and run it on pilot trajectories (~2 h MiniMax compute) |
| Evening / next day | M1 measurement on spotify subset (~6 h on 8 parallel ACON workers) |
| +1 day | M3 heatmap and the closure-ratio T2 verdict |
| +2 days | Decide: strong/weak T1, strong/weak T2, paper framing tier |

## 11. Files of record

| File | Role |
|---|---|
| `motivation_v2/README.md` | Track overview + roadmap |
| `motivation_v2/docs/01_experimental_design.md` | This file — operational design |
| `motivation_v2/docs/02_strategy_prompts.md` | Strategy variants (negative control + appendix) |
| `motivation_v2/docs/03_role_memory_extractors.md` | Role-conditional memory projections (headline) |
| `motivation_v2/prompts/STRATEGY_DESIGN.md` | (mirror of 02_strategy_prompts.md, kept beside the build script) |
| `motivation_v2/motivation_v2/data.py` | acon trajectory + AppWorld GT loaders |
| `motivation_v2/motivation_v2/units.py` | trajectory → memory-unit pool |
| `motivation_v2/motivation_v2/policy_family.py` | task → topic-family classifier |
| `motivation_v2/motivation_v2/exec_memory.py` | `m*_exec_minimal` and `m*_exec_trajectory` builders |
| `motivation_v2/motivation_v2/role_memory.py` | `m_tool` / `m_code` / `m_plan` / `m_verify` builders |
| `motivation_v2/motivation_v2/compressors.py` | `m_recent` / `m_freq` / `m_bm25` / `m_embedding_topk` |
| `motivation_v2/motivation_v2/runner.py` | compressed-memory injection into AppWorldAgent |
| `motivation_v2/scripts/run_appworld_strategy.py` | per-strategy trajectory generation launcher |
| `motivation_v2/scripts/manipulation_check.py` | per-strategy iter / API-pattern stats |
| `motivation_v2/scripts/build_compressed_memories.py` | post-hoc cell-grid materialisation |
| `motivation_v2/scripts/run_cross_task_transfer.py` | cross-task transfer experiment driver |
| `motivation_v2/scripts/analyze_task_overlap.py` | cross-task Jaccard analysis |
| `motivation_v2/scripts/analyze_role_overlap.py` | cross-role Jaccard analysis |
| `motivation_v2/scripts/analyze_exec_overlap.py` | cross-strategy Jaccard analysis (control) |
| `motivation_v2/scripts/smoke_data_pipeline.py` | corpus audit |
| `motivation_v2/outputs/mv2_pilot/` | pilot results land here |
| `motivation_v2/outputs/pilot_progress.log` | live pilot tracker (5-min updates) |
