# Strategy variants — design for Option X (policy = behavioural strategy)

This file documents the three policy strategies used by the M1 / M2 / M3
motivation experiments and how they're injected into the AppWorld
agent. Renders the `policy ≠ task topic` distinction operational.

## Why three strategies, not just two

We need at least **two** strategies that produce **demonstrably
different trajectories** on the same task; that's the bare minimum for
"different policies want different memory" to be falsifiable. We pick
**three** so the M3 cross-policy heatmap has three distinct rows and
three columns (3×3 = 9 cells, of which 6 are off-diagonal).

The three were chosen so they vary along orthogonal axes:

| Strategy | Search breadth | Verification | Action style |
|---|---|---|---|
| `P_direct`   | minimal | none | answer-first |
| `P_verify`   | task-driven | mandatory cross-check | answer-then-confirm |
| `P_explore`  | exhaustive | optional | survey-then-answer |

## Manipulation check (must run before relying on the results)

A strategy that doesn't actually change the trajectory is useless. After
generating trajectories for all three strategies on the same task set,
we report:

| Metric | `P_direct` (expected) | `P_verify` (expected) | `P_explore` (expected) |
|---|---|---|---|
| Median `num_interactions`   | low (≤ 12) | medium-high (≥ 18) | high (≥ 22) |
| `show_app_descriptions` calls in first 3 steps | rare | rare | very common |
| Cross-source duplicate fact retrieval | none | always | sometimes |
| Avg unique GT API endpoints touched | small | medium | large |

If `direct` and `verify` produce statistically indistinguishable
trajectories on `≥ 60%` of tasks, the strategy injection didn't take
and we need to either (a) make the strategy directives more aggressive,
or (b) switch to Option Y (executor-variant) per the design doc.

## How injection works

The acon agent loads its main prompt template from
`./prompts/prompt_v1.jinja`. The prompt JSON
(`./prompts/prompts_v1.json`) just specifies `system_message` (very
short) and `main_prompt_template` (path to the jinja).

Strategy injection is done by **swapping the prompt JSON + jinja pair
for the strategy-specific variant** at agent construction time. The
strategy-specific jinja is identical to `prompt_v1.jinja` except that
a `**STRATEGY: ...**` block is inserted between the 17 numbered
disclaimers and the final "Now generate code to solve the actual
task" line. Position chosen so the strategy is the LAST thing the LLM
reads before the task instruction — maximising adherence.

The launcher (`scripts/run_appworld_strategy.py`) materialises the
strategy-specific prompt files into acon's
`experiments/appworld/prompts/_motivation_v2/<strategy>/` directory at
run time, then calls acon's `run.main` with `prompt_file` pointing
into that directory.

## The three strategy texts (canonical)

Each is rendered into a 4–6 sentence block. They are deliberately
phrased as *behavioural mandates* with explicit MUST / DO NOT verbs
and at least one concrete instantiation, since instruction-following
is fragile with vague directives.

### `P_direct`

```
**STRATEGY: DIRECT (minimum-API, answer-first)**

You MUST solve the task with the minimum number of API calls.
As soon as you have enough information to answer with reasonable
confidence, immediately call
`apis.supervisor.complete_task(answer=<answer>)`. DO NOT verify the
answer through a second source. DO NOT enumerate apps via
`apis.api_docs.show_app_descriptions()` if you can already infer
which app to use from the task. DO NOT list extra data that is not
strictly required to compute the answer. Brevity is the policy.
```

### `P_verify`

```
**STRATEGY: VERIFY (mandatory cross-validation)**

Before returning your final answer, you MUST cross-validate it
through at least one independent code path. For example, if you
computed the answer by aggregating from one library (songs), also
verify by aggregating from a different library (albums or playlists)
when feasible, OR re-fetch the same record by a different identifier
(e.g. song-id vs title), OR call a related list endpoint and confirm
the result is consistent. The cross-validation step is mandatory; do
not skip it. Only call `apis.supervisor.complete_task` after the
second path has confirmed the answer.
```

### `P_explore`

```
**STRATEGY: EXPLORE (survey-then-answer)**

Before computing the answer, build a comprehensive understanding of
the available data. Step 1: ALWAYS list the available apps with
`apis.api_docs.show_app_descriptions()`. Step 2: enumerate the key
APIs of the most relevant app via
`apis.api_docs.show_api_descriptions(app_name=<app>)`. Step 3:
survey the user state via list-style endpoints (e.g. libraries,
recent items, profiles). Only AFTER you have a broad picture of the
user's state should you compute and return the answer. Use this
exploration phase even if you think you already know the answer.
```

## Why this matches the new_motivation.md framing

The M1 / M2 / M3 numbering and pass criteria in `new_motivation.md`
all generalise cleanly when we interpret "policy" as a strategy
choice rather than a task-family choice:

* **M1** (compression-pressure sweep): for each (task, B), compare
  `m*_exec_trajectory(task, P_direct)` vs `m*_exec_trajectory(task,
  P_verify)` vs `m*_exec_trajectory(task, P_explore)`. T1 = "the
  three differ more under tight B than under loose B".
* **M2** (prompted selector gap): can a prompted compressor that's
  told "compress for an agent following P_verify" reproduce
  `m*_exec_trajectory(P_verify)`? T2 = "no, prompted compression
  collapses across strategies even when conditioned on the strategy
  description".
* **M3** (cross-policy transfer heatmap): feed `m*_exec(P_X)` to a
  consumer agent running `P_Y`. The diagonal (X=Y) should beat
  off-diagonal under tight B.

The "task family" axis (R-2 in `new_motivation.md`) is preserved as
a *secondary* dimension: heatmaps and curves are stratified by
single-app task family (spotify, file_system, phone, simple_note,
venmo) so we can show the strategy effect is consistent across
topics, not a topic-specific quirk.
