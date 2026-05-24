# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-24 19:55 UTC
>
> The old `motivation/` track was abandoned on 2026-05-24 — see
> §Abandoned-track at the very bottom for context. The active line is
> `motivation_v2/` (AppWorld + behavioural-strategy policy variants).
> Read top-down; everything is current unless the section is tagged
> "abandoned".

## What this project is

EASMO — **policy-conditional context compression for long-horizon LLM
agents**. Two empirical theses:

* **T1**: under tight memory budgets, the optimal compressed memory is
  policy-dependent (different policies want different things from the
  same context).
* **T2**: prompted off-the-shelf LLM selectors cannot reproduce
  policy-conditional compression, even when given task + policy
  descriptions.

Combined claim: policy-conditional compression at tight budgets is
**necessary** (T1) and **not achievable by prompting** (T2). It must
be learned with a behavioural objective.

## Where things stand (2026-05-24 19:55 UTC)

* **Old track abandoned.** `motivation/` (LongMemEval/LoCoMo + binary
  action-match + ReAct/Plan/CoT scaffolds + LLM-only selector) had a
  fatal logical issue (T2 holding makes the T1 hinge test
  unsatisfiable) AND a topic-vs-policy framing problem AND used QA
  benchmarks where there's no real policy variation. All processes
  killed; numbers retained only as git history.
* **New track** (`motivation_v2/`) uses AppWorld + execution-derived
  ground-truth memory + behavioural strategy as the policy axis.
* **Pilot is running** — 3 strategies × 90 train tasks, ~3 h ETA from
  19:12 UTC (so ~22:00 UTC target).
* **Pipeline is fully built** for everything except the
  compressed-memory executor wrapper (next big TODO).

## Active background processes

```
636982  pilot_direct       AppWorld train, P_direct   (25/90 @ 19:50)
637116  pilot_verify       AppWorld train, P_verify   (15/90 @ 19:50, slowest — verify uses 2.4× iters)
637280  pilot_explore      AppWorld train, P_explore  (18/90 @ 19:50)
641062  pilot_watch.sh     progress logger → outputs/pilot_progress.log every 5 min
3916707 auto_push_watcher  pushes motivation/ + motivation_v2/ changes every 20 min
```

Per-strategy progress is in `motivation_v2/outputs/pilot_progress.log`.

## Layout

```
EASMO/
├── motivation/              [ABANDONED — see §Abandoned-track]
│   ├── docs/
│   │   ├── 02_results_and_interpretation.md   (banner: not even appendix)
│   │   └── new_motivation.md                  ← active design doc, 11 review-fix notes inline
│   └── scripts/{auto_push_watcher,sync_and_push}.sh   (still useful plumbing — kept)
└── motivation_v2/                              ← active
    ├── README.md
    ├── motivation_v2/                          ← python package
    │   ├── data.py             (acon trajectory + AppWorld ground-truth loaders)
    │   ├── units.py            (trajectory → memory-unit pool)
    │   ├── policy_family.py    (task → topic family classifier)
    │   ├── exec_memory.py      (m*_exec_minimal + m*_exec_trajectory; deterministic)
    │   └── compressors.py      (m_recent, m_freq, m_bm25, m_embedding_topk[opt])
    ├── prompts/
    │   ├── STRATEGY_DESIGN.md  (canonical strategy texts + manipulation-check spec)
    │   └── build_strategy_prompts.py   (splices strategy block into acon's jinja)
    ├── scripts/
    │   ├── smoke_data_pipeline.py        (corpus audit; runs in 0.5 s)
    │   ├── run_appworld_strategy.py      (launcher: acon run.main + custom prompt)
    │   ├── manipulation_check.py         (per-strategy iter / API-pattern stats)
    │   ├── build_compressed_memories.py  (post-hoc: trajectories → compressed memory matrix)
    │   └── queue_instance_noise_B128.sh  [orphaned, can delete]
    └── outputs/<run-tag>/   results, summaries, progress logs
```

External infrastructure (consumed read-only):

```
/workspace/acon/                  # Microsoft ACON repo, arXiv 2510.00615
├── .venv/                        # working pydantic v1 / sqlmodel 0.0.10
├── experiments/appworld/
│   ├── run.py / run_all.py       # agent runner end-to-end
│   ├── data/{tasks,api_docs,base_dbs,datasets}/  # AppWorld data
│   ├── prompts/_motivation_v2/<strategy>/  # spliced jinja files
│   └── outputs/<exp>/<split>/task_<id>/   # per-task trajectory outputs
└── src/productive_agents/        # AppWorldEnv, AppWorldAgent
```

EASMO `.venv` is broken on AppWorld imports (pydantic v2 incompat with
sqlmodel 0.0.10). Always use `/workspace/acon/.venv` for anything that
touches the `appworld` or `productive_agents` packages. The `motivation_v2`
analysis modules (data, exec_memory, compressors, …) work in either
venv since they don't import AppWorld directly.

## Empirical state so far

### AppWorld corpus (from `motivation_v2/scripts/smoke_data_pipeline.py`)

```
[train] 90 tasks ground truth
  single-app: 60/90 (67%)
  per-family: spotify=42, file_system=9, phone=6, simple_note=3, multi_app=30
  difficulty: easy=36 / medium=36 / hard=18
  shared-state cross-policy pairs: 0 (M3 must use matched-pair fallback)

[dev]   57 tasks ground truth
  single-app: 36/57 (63%)
  per-family: spotify=30, venmo=3, file_system=3, multi_app=21

[test_normal/test_challenge]  168 / 417 tasks, ground truth held out
```

Headline implication: spotify is the only family with enough single-app
tasks for a per-family compression-pressure curve (42 train + 30 dev =
72 single-app spotify total). Other families are robustness side
panels; venmo only exists in dev so dev must be added to the run.

### Strategy injection works (smoke validation on `82e2fac_3`)

| | iters | input tokens | unique APIs | total API calls | distinctive |
|---|---|---|---|---|---|
| baseline (no strategy) | 11 | 46K | 8 | 11 | normal |
| `P_direct`  | 11 | 57K | 7 | 11 | small task: little to compress |
| `P_verify`  | **26** | **204K** | **11** | **36** | 17× show_song cross-checks |
| `P_explore` | 14 | 82K | 8 | 14 | 100% show_app_descriptions in first 3 steps |

Manipulation-check verdict: ✓ `verify_iters / direct_iters = 2.36×`
(threshold ≥ 1.5); ✓ explore first-3-step exploration = 100%
(threshold ≥ 50%).

### Pilot early data (40 min into 3-h run)

| strategy | tasks done (~40 min) | success | success rate |
|---|---|---|---|
| `P_direct` | 24 | 21 | 88% |
| `P_verify` | 13 | 7 | 54% |
| `P_explore` | 17 | 14 | 82% |

**`P_verify`'s 54% success rate is itself a paper-worthy finding** —
strategy specifications interact with task success, not just memory
content. Forcing cross-validation makes the agent run out of
iterations or confuse itself. Keep this in the M1 narrative.

## What to do when the pilot finishes (~22:00 UTC)

Pilot completion is when **all three strategy jobs (PIDs 636982 /
637116 / 637280) have exited**. Check via
`tail outputs/pilot_progress.log`.

### Step 1 — Manipulation check on the full pilot dataset

```bash
/workspace/acon/.venv/bin/python \
    /workspace/EASMO/motivation_v2/scripts/manipulation_check.py \
    --tag mv2_pilot
```

Required verdicts:
* `median_iters(verify) / median_iters(direct) ≥ 1.5`
* `show_app_descriptions_first3(explore) ≥ 0.5`

If both pass, strategies are functionally distinct in the large.
If either fails, the strategy directives didn't take consistently — fall
back to Option Y (executor-variant policy axis), which needs cross-LLM
endpoint coordination (see `new_motivation.md` §2.4c).

### Step 2 — Build compressed-memory matrix

```bash
/workspace/EASMO/.venv/bin/python \
    /workspace/EASMO/motivation_v2/scripts/build_compressed_memories.py \
    --tag mv2_pilot \
    --strategies direct verify explore \
    --budgets 128 256 512 1024 2048 \
    --output_dir /workspace/EASMO/motivation_v2/outputs/mv2_pilot
```

This runs in ~1.3 s on a partial dataset and writes
`compressed_memories.jsonl` with one row per (task, strategy,
compressor, B, memory_text). No LLM calls.

### Step 3 — Build the compressed-memory executor wrapper (NOT YET DONE)

The remaining missing piece is `motivation_v2/motivation_v2/runner.py`
— wraps acon's `AppWorldAgent` so that an arbitrary "preloaded
memory string ≤ B" gets injected into the agent's first user turn,
then the agent runs as normal (it can still call APIs to fill gaps).
This requires acon's `.venv` since it imports `productive_agents`.

Sketch:
```python
# motivation_v2/motivation_v2/runner.py
from productive_agents.env.appworld import AppWorldEnv, AppWorldEnvConfig
from productive_agents.agents.appworld import AppWorldAgent, AppWorldAgentConfig

def run_with_compressed_memory(task_id, split, memory_text, model_name, strategy):
    # 1. construct env + reset
    # 2. construct agent with a prompt_file that has an extra "Pre-loaded memory" turn before the task instruction
    # 3. agent.run(env, max_iter=...)
    # 4. return success / iters / final_reward
```

The trickiest part is splicing `memory_text` into the agent's
first turn. Reuse the `build_strategy_prompts.py` machinery —
add a new turn between the strategy block and the task instruction
that says "Here is a pre-loaded compressed memory you may use:
\n\n{{ memory_text }}".

### Step 4 — Run M1: every (task, strategy, compressor, B) cell

Once the runner exists, the M1 driver loops over
`compressed_memories.jsonl` rows and runs `runner.run_with_compressed_memory`
for each. Each cell is one AppWorld task run (~70 s). Pilot has
~840 rows from this morning's partial dataset; full pilot will have
~3500 (90 task × 3 strategy × ~5 budgets × ~5 compressors). At 70s
each that's 68 hours sequential, way too long.

**Plan to make M1 tractable**:
* Don't measure every cell. Pick: 4 budgets × 6 compressors = 24 cells
  per task. Run each cell ONCE (no sample-multiplication), success-rate
  is the per-task binary outcome.
* 90 tasks × 24 cells × 70 s = ~42 h sequential, **~5 h parallel × 8**.
* Or, drop to the spotify-only subset (42 tasks × 24 = 1008 cells; ~20 h
  sequential / ~3 h parallel × 8).

Decision: start with spotify-only at 4 budgets × 4 compressors (m_recent,
m_bm25, m_exec_minimal, m_exec_trajectory) × 3 strategies = 48 cells/task
× 42 tasks = 2016 cells. ~6 h on 8 parallel workers.

## Key design notes (don't re-derive)

* **policy = behavioural strategy** (direct / verify / explore), NOT
  task family. Task family (spotify / phone / venmo / file_system /
  simple_note) is a *secondary* stratification axis only. See
  `new_motivation.md` §2.4 for the full rationale.
* **m*_exec is deterministic, no LLM in the loop**. Two declared
  variants: `m_exec_minimal` (ground-truth API calls only,
  executor-independent) and `m_exec_trajectory(executor)`
  (everything the executor's successful trajectory touched, ranked
  by GT-anchor + frequency).
* **BM25 is the headline retrieval baseline**; SBERT is appendix-only
  (gated by `--include_embedding`). AppWorld memory units are
  structured enough that BM25 captures the lexical signal and SBERT
  costs ~6 s startup × tasks for negligible gain.
* **Manipulation check is mandatory**. Before reporting any
  M1/M2/M3 number, verify the strategy distinction held in the data
  (script: `manipulation_check.py`). If verify and direct produce
  statistically indistinguishable trajectories, the policy axis
  collapsed and the experiment doesn't measure what we claim.
* **Action-match metric**: deprecated. Use overlap = 1 − TV when
  measuring action distributions. (Relevant only if we ever do an
  action-distribution-based version of M1 again.)

## Anti-patterns to avoid

From `new_motivation.md` §13:

* Do not use LLM-generated oracle memory as gold (circular).
* Do not use binary action-match as the headline metric.
* Do not use ReAct/Plan/CoT as the policy distinction.
* Do not use task-family as the policy distinction (it's topic, not policy).
* Do not use token-level Jaccard or leave-one-out as killer metrics.

## Files of record (single sources of truth)

| File | Role |
|---|---|
| `motivation/docs/new_motivation.md` | Active design doc with 11 review-fix notes (R-1..R-11) |
| `motivation_v2/README.md` | Track overview + roadmap |
| `motivation_v2/prompts/STRATEGY_DESIGN.md` | Canonical strategy texts + manipulation-check spec |
| `motivation/docs/SESSION_HANDOFF.md` | This file |
| `motivation_v2/outputs/mv2_pilot/` | Pilot results land here |
| `motivation_v2/outputs/pilot_progress.log` | Live pilot tracker |

## Trust no in-doc framings until verified

`02_results_and_interpretation.md` is abandoned but kept; do not cite
its numbers. `new_motivation.md` is being actively edited as findings
land — re-read it whenever you pick up.

---

## §Abandoned-track (2026-05-24, kept for context only)

The old motivation track had:

* Three "agents" `A_react / A_plan / A_cot` = same MiniMax-M2.5 model
  with different system prompts → not really different policies
* QA benchmarks (LongMemEval / LoCoMo) where one canonical answer
  exists per task → no real policy variation possible
* Binary action-match metric → near-Bernoulli on N=16 samples
* LLM-only selector for "oracle" memory → circular T1+T2 logic
* Task-family-as-policy framing in the original new_motivation draft →
  topic mismatch, not policy mismatch

The runs killed at 19:55 UTC (final state):

* `wide_longmemeval`: M1 done (22.6% pass, flat across budgets), M2
  done, M3 in progress (transfer_results.csv at 733 bytes when killed).
* `wide_locomo`: M1 85% (pass rates 0–12% across budgets, junk by any
  standard).
* `instance_noise_v2_B512_n30`: 20/30 contexts done, would have taken
  ~50 min more to produce the first non-degenerate verdict. Verdict
  no longer informs the paper.

The `motivation/` directory is left in place because
`auto_push_watcher.sh` and `sync_and_push.sh` live there as plumbing
the new track still uses. Do not edit anything else under
`motivation/` going forward.
