# Experimental Design — motivation_v2 (AppWorld + behavioural-strategy policies)

> Owner: EASMO motivation track v2.
> Last edited: 2026-05-24 1:10 PM PT.
> Companion to `../README.md` and the active design spec at
> `../../motivation/docs/new_motivation.md`. This doc focuses on the
> **operational** design — what we run, why, what we expect, what we
> have observed so far, and the gaps we are still aware of.

---

## 1. Two theses we are trying to validate

* **T1 — compression pressure induces policy-dependent memory.**
  Under tight memory budgets, the optimal compressed memory for a
  long-horizon agent is policy-conditional. Different downstream
  policies prefer measurably different compressions of the same
  context, and this difference grows as the budget tightens.
* **T2 — prompted LLM selectors cannot realise policy-conditional
  compression.** Even when conditioned on a task description and a
  policy/strategy description, an off-the-shelf LLM compressor
  produces compressions that are surface-similar across policies and
  fall short of the execution-derived oracle.

Combined claim: *policy-conditional compression at tight budgets is
necessary (T1) and not achievable by prompting (T2). It must therefore
be learned with a behavioural objective.*

`policy-dependency` is the central conceptual lever. The design must
make policy variation **measurable, controllable, and orthogonal to
task topic** — that is the bar this design either clears or fails.

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

## 4. Memory units and policy axis

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

### 4.2 Policy axis — Option X (primary)

Three behavioural strategies, all running on the same MiniMax-M2.5
executor on the same task. The variation is in the agent's
*decision policy*, not the model. Canonical specs in
`prompts/STRATEGY_DESIGN.md`.

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

### 4.3 Secondary axis — task topic

`policy_topic = primary_app` (spotify, file_system, phone,
simple_note, venmo) for single-app tasks. Used **only for
stratification** of M1 / M2 results, never as the policy axis
itself. The premise is that the same strategy effect holds across
topics, not that topic *is* policy.

### 4.4 Backup policy axis — Option Y (executor variants)

If the manipulation check on the full pilot fails (i.e., strategies
collapse on the large dataset despite passing on a single task),
the fallback is to use different executor models (MiniMax / Qwen /
GPT-4o-mini) on the same task. Documented but not the primary plan;
requires multi-LLM endpoint coordination that the user is still
sorting out externally.

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

## 6. Experimental matrix

Four experiments, all on the same dataset / same executor / same
budget grid.

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

## 8. What we have actually observed (2026-05-24, 1:00 PM PT)

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

## 9. Honest assessment of design sufficiency

This is a self-critique done **before** the data lands so we are not
fitting the assessment to results.

### 9.1 What we are confident in

* **T1 minimal version** ("policy-conditional ≥ generic at tight B")
  is testable and the pipeline is end-to-end working. The single
  M1 cell at 8 vs 11 iters is the first empirical confirmation that
  the experimental setup *can* detect a compression benefit.
* **Strategy injection genuinely changes agent behaviour** — verify
  fetched 17× show_song cross-validating, explore did
  `show_app_descriptions` first 100% of the time, direct skipped
  exploration. The policy axis is real.
* **`m*_exec` is non-LLM** (deterministic function of trajectory
  and ground-truth API call list), so T1 has an independent oracle
  and is not circular.

### 9.2 Where the design is still weak

**(a) Same task, same answer, same data → strategies may converge
on similar information needs.**

`P_direct`, `P_verify`, `P_explore` differ in *how aggressively* they
query the data, but for "what's the most-played song", all three
ultimately need to look at `play_count` rows. Verify just checks
them more times. **`m*_exec_trajectory(verify)` may be a superset
of `m*_exec_trajectory(direct)` rather than a different set.** If
that's the case, the M3 cross-policy heatmap measures
"is the verbose memory wasteful?" rather than "is policy-conditional
necessary?".

**Mitigation**: a Jaccard-overlap analysis on `m*_exec` across the
three strategies, run as soon as the pilot finishes. If
`Jaccard(direct, verify) > 0.8` per-task on average, the strong T1
claim should be retracted in favour of "compression for tool-use
agents is necessary at tight budgets" (weaker but still publishable).

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

All three policies run on MiniMax-M2.5. A skeptic can argue we are
measuring "MiniMax with three different system-prompt nudges", and
the result might not transfer to other LLMs. Option Y (executor
variants) is the proper robustness check.

**Mitigation**: write a follow-up experiment that runs at least
the M1 main figure on Qwen2.5-7B for the spotify subset. This is
the user's pending external coordination item.

**(d) Prompted compressors not built yet.**

T2 is currently an unfilled hole. The headline T2 claim cannot be
made until `m_prompted_task_policy` is implemented and run.

**Mitigation**: plan §10 step "M2 prep" — write
`scripts/build_prompted_memories.py` that calls MiniMax with
template prompts to produce the four prompted-memory variants.
Estimated 1 hour to write, ~2 h to run on the pilot trajectories.

**(e) AppWorld task structure is narrow.**

Most train tasks are spotify variants ("most-played song",
"least-liked song", etc.). They have one canonical answer and one
canonical data path. "Different policies want different memory"
gets the most traction on tasks where the solution path can vary
(e.g., "send a message to my closest friend" — what counts as
"closest"?). These tasks are rarer in the corpus.

**Mitigation**: in M3, report the heatmap on the spotify subset
*and* on the multi-app subset separately. Multi-app tasks are
where genuine policy variation should bite.

### 9.3 What would make this design strong enough for spotlight

1. M1 result with diagonal-vs-off-diagonal gap ≥ 15 pp at B=256/512.
2. M3 cross-strategy heatmap with the off-diagonal cells visibly
   below the diagonal AND a story for *why* each strategy's memory
   misses for the others.
3. T2's `r_T2 ≤ 0.30`: prompting closes ≤ 30% of the
   policy-conditional gap.
4. Robustness: the same M1 effect on at least one other executor
   (Qwen2.5-7B) for the spotify subset.
5. Multi-app heatmap: the policy-conditional effect visible on
   tasks that genuinely admit multiple solution paths.

We currently have 0 of these. The pilot will resolve (1) and the
diagonal-vs-off-diagonal direction of (2). The other three are
explicit future work.

### 9.4 What would invalidate the design

* `Jaccard(m*_exec_direct, m*_exec_verify) > 0.8` averaged across
  pilot tasks → strategies share information needs → policy-
  conditional thesis is empty for AppWorld.
* Manipulation check fails on full pilot → strategies didn't take
  on most tasks → switch to Option Y.
* `r_T2 > 0.70` → prompting works well enough that "necessary but
  not achievable" collapses to "achievable, just expensive".

In each of these cases the doc spells out a fallback paper framing
so we don't end up scrambling.

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
| `motivation_v2/prompts/STRATEGY_DESIGN.md` | Canonical strategy texts + manipulation-check spec |
| `motivation_v2/motivation_v2/data.py` | acon trajectory + AppWorld GT loaders |
| `motivation_v2/motivation_v2/units.py` | trajectory → memory-unit pool |
| `motivation_v2/motivation_v2/policy_family.py` | task → topic-family classifier |
| `motivation_v2/motivation_v2/exec_memory.py` | `m*_exec_minimal` and `m*_exec_trajectory` builders |
| `motivation_v2/motivation_v2/compressors.py` | `m_recent` / `m_freq` / `m_bm25` / `m_embedding_topk` |
| `motivation_v2/motivation_v2/runner.py` | compressed-memory injection into AppWorldAgent |
| `motivation_v2/scripts/run_appworld_strategy.py` | per-strategy trajectory generation launcher |
| `motivation_v2/scripts/manipulation_check.py` | per-strategy iter / API-pattern stats |
| `motivation_v2/scripts/build_compressed_memories.py` | post-hoc cell-grid materialisation |
| `motivation_v2/scripts/smoke_data_pipeline.py` | corpus audit |
| `motivation_v2/outputs/mv2_pilot/` | pilot results land here |
| `motivation_v2/outputs/pilot_progress.log` | live pilot tracker (5-min updates) |
