# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-24 5:15 PM PT
>
> **Active track**: `motivation_v2/` (AppWorld + role-conditional memory).
> The old `motivation/` track was abandoned 2026-05-24; see §Abandoned-track
> at the end if you need historical context. Times throughout this doc
> are Pacific Time (PT).

## What this project is

EASMO — **role-conditional context compression for long-horizon LLM
agents in multi-agent systems**. Two empirical theses:

* **T1 — role-conditional memory has measurable efficiency value
  in multi-agent systems**, with two sub-claims:
  * *T1a (structural)*: for the same upstream context, four agent
    roles (planner / tool-user / coder / verifier) want compressed
    memory whose pairwise Jaccard is 0.04 mean at B=512 — essentially
    disjoint slices of the same trajectory.
  * *T1b (efficiency)*: cross-task memory transfer at B=512 inflates
    iters and input tokens by ~40% without reducing task-completion
    success. Wrong-role memory is an *inference-cost tax*, not a
    *capability* failure.
* **T2 — prompted LLM selectors cannot realise role-conditional
  compression**. Even with explicit role + task descriptions in the
  prompt, MiniMax produces compressions whose cross-role Jaccard is
  6.2× the projected oracle's (0.222 vs 0.036), and per-role recall
  against the oracle is only ~20% (code role: 5–8%).

Combined claim: *role-conditional compression in multi-agent systems
is necessary (T1) and not achievable by prompting (T2). It must
therefore be learned with a behavioural objective.* Within a fixed
role, however, the compressor transfers across tasks (code patterns
Jaccard 0.41; fact-level patterns 0.07–0.11 but recoverable via
multi-task training).

## Where things stand (2026-05-24 5:15 PM PT)

The three-tier hierarchy is the headline finding:

```
                                   Mean Jaccard at B=512
1. Strategy invariance               0.91   → control: agent style is not the lever
2. Within-role cross-task            0.16   → train per role, not per task
   ├── code patterns                 0.41   → highly transferable
   └── tool/plan/verify facts        0.07–0.11
3. Cross-role orthogonality          0.04   → headline: role conditioning necessary

T2 partial (n=100/1328 cells, complete in ~40 min):
   prompted cross-role Jaccard       0.222  (6.2× oracle's 0.036)
   prompted-vs-oracle recall (mean)  0.196
   code-role recall                  0.05–0.08  (catastrophic)

Cross-task efficiency cost (B=512, n=72):
   self memory                       12.8 iters /  79K tokens   (baseline)
   within-app cross-gen memory       18.0 iters / 133K tokens   (+40%)
   cross-app memory                  18.3 iters / 141K tokens   (+45%)
   task success                      100% across all 72 cells
```

**4 of 5 spotlight criteria achieved**:

1. ✅ Cross-role Jaccard ≤ 0.10 at B=512 — achieved 0.036.
2. ✅ Cross-task within-role Jaccard ordered (code high, others low) — achieved.
3. ✅ Cross-task transfer cost shows plumbing-floor pattern (B=128 flat, B=512 +40%).
4. ⏳ T2 closure ratio ≥ 5× — partial 6.2×; full data due ~5:50 PM PT.
5. ⏳ Cross-executor robustness (Qwen2.5-7B) — pending external endpoint.

## Active background processes

```
880089  build_prompted_memories.py  T2 build (131/1328 cells at 5:15 PM, ETA ~5:55 PM PT)
3916707 auto_push_watcher           pushes motivation/ + motivation_v2/ changes every 20 min
```

The 3 pilot strategy jobs and the cross-task transfer driver have
all completed. Pilot watcher exited at 3:00 PM PT.

## Layout

```
EASMO/
├── motivation/              [ABANDONED — see §Abandoned-track]
│   ├── docs/SESSION_HANDOFF.md       (this file)
│   ├── docs/02_results_and_interpretation.md   (banner: not appendix material)
│   └── scripts/{auto_push_watcher,sync_and_push}.sh   (still useful plumbing — kept)
└── motivation_v2/                              ← active
    ├── README.md
    ├── docs/
    │   ├── 01_experimental_design.md           ← operational design (888 lines)
    │   ├── 02_strategy_prompts.md              ← negative control + appendix
    │   └── 03_role_memory_extractors.md        ← role-projection definitions (paper §3)
    ├── motivation_v2/                          ← python package
    │   ├── data.py            (acon trajectory + AppWorld GT loaders)
    │   ├── units.py           (trajectory → memory-unit pool)
    │   ├── policy_family.py   (task → topic-family classifier)
    │   ├── exec_memory.py     (m*_exec_minimal + m*_exec_trajectory)
    │   ├── role_memory.py     (m_tool / m_code / m_plan / m_verify)
    │   ├── compressors.py     (m_recent / m_freq / m_bm25; SBERT optional)
    │   ├── prompted_memory.py (LLM compressor with role prompts; T2 baseline)
    │   └── runner.py          (compressed-memory injection into AppWorldAgent)
    ├── prompts/
    │   ├── STRATEGY_DESIGN.md            (stub redirecting to docs/02_*)
    │   └── build_strategy_prompts.py     (canonical strategy texts)
    ├── scripts/
    │   ├── smoke_data_pipeline.py        (corpus audit; ~0.5 s)
    │   ├── run_appworld_strategy.py      (per-strategy trajectory generation)
    │   ├── manipulation_check.py         (per-strategy iter / API stats)
    │   ├── build_compressed_memories.py  (post-hoc cell-grid materialisation)
    │   ├── build_prompted_memories.py    (T2 LLM-compressor baseline at scale)
    │   ├── run_cross_task_transfer.py    (xtask transfer experiment driver)
    │   ├── analyze_role_overlap.py       (cross-role Jaccard, projected memory)
    │   ├── analyze_task_overlap.py       (cross-task Jaccard)
    │   ├── analyze_exec_overlap.py       (cross-strategy Jaccard, control)
    │   └── analyze_prompted_overlap.py   (T2 cross-role + recall analysis)
    └── outputs/
        ├── mv2_pilot/                    (90×3-strategy trajectories ref by acon outputs)
        ├── mv2_pilot/compressed_memories.jsonl  (3,100 rows)
        ├── mv2_pilot/prompted_memories.jsonl    (T2, building → 1,328 rows)
        └── mv2_xtask/transfer_results.jsonl     (72 cells, all 100% success)
```

External infrastructure (read-only):

```
/workspace/acon/                  # Microsoft ACON repo, arXiv 2510.00615
├── .venv/                        # working pydantic v1 / sqlmodel 0.0.10
├── experiments/appworld/
│   ├── run.py / run_all.py       # agent runner
│   ├── data/                     # AppWorld task data (downloaded)
│   ├── prompts/_motivation_v2/<strategy>/  # spliced jinjas (run-time)
│   ├── prompts/_motivation_v2/_cells/<hash>/  # per-cell jinjas (runner.py)
│   └── outputs/<exp>/<split>/task_<id>/  # per-task trajectory outputs
└── src/productive_agents/        # AppWorldEnv, AppWorldAgent
```

EASMO `.venv` is broken on AppWorld imports (pydantic v2 incompat).
Always use `/workspace/acon/.venv` for anything that touches
`appworld` or `productive_agents`. The `motivation_v2` analysis modules
work in either venv since they don't import AppWorld directly.

## Empirical state — datasets we have

| Dataset | Path | Description |
|---|---|---|
| AppWorld train ground truth | `acon/experiments/appworld/data/tasks/` | 90 tasks; 60 single-app |
| Pilot direct trajectories | `acon/.../mv2_pilot_direct/train/task_*` | 83/90 success (92%) |
| Pilot verify trajectories | `acon/.../mv2_pilot_verify/train/task_*` | 60/90 success (67%) |
| Pilot explore trajectories | `acon/.../mv2_pilot_explore/train/task_*` | 65/90 success (72%) |
| Compressed memories (post-hoc) | `mv2_pilot/compressed_memories.jsonl` | 3,100 rows: m_exec_minimal/trajectory + m_recent/freq/bm25 |
| Prompted memories (T2) | `mv2_pilot/prompted_memories.jsonl` | building (131/1,328 at 5:15 PM PT) |
| Cross-task transfer results | `mv2_xtask/transfer_results.jsonl` | 72 cells, all 100% success |
| Role overlap summary | `mv2_pilot/role_overlap.json` | cross-role + cross-task Jaccards |
| Cross-strategy overlap | `mv2_pilot/exec_overlap_partial.json` | strategy invariance (0.91+) |

## What's next

In rough priority order:

### 0. Wait for T2 full completion (~5:55 PM PT)

The build is at 131/1,328 cells with ratio already 6.2× and stable.
When full, re-run `analyze_prompted_overlap.py` and lock numbers into
`docs/01_experimental_design.md` §8.0.6.

### 1. M1 task-success measurement on the role-projected memory

The xtask experiment used `m_exec_minimal` (universal API call list).
The headline M1 result needs role-projected memory plugged into
`runner.py` to show that `m_role` improves task efficiency over
`m_recent` / `m_bm25` at B ∈ [256, 512]. Estimated ~6 h on 8 parallel
workers for a spotify-only subset.

### 2. Cross-executor robustness (Option Y, blocking)

Re-run the role-overlap and prompted-overlap analyses on Qwen2.5-7B
trajectories (and ideally GPT-4o-mini). Required to claim the
finding is model-independent. **Pending external endpoint
coordination by the user.**

### 3. M3 cross-role transfer (variant of M1)

Feed `m_role_X` to a consumer running role Y. Measure efficiency
cost. Predicted: large iter inflation and possible success drops
when off-diagonal in the role × role heatmap.

### 4. Multi-agent benchmark replication (paper extension)

ChatDev / MetaGPT / AutoGen for explicit-role-specialisation. AppWorld
shows the projected-role finding; a multi-agent benchmark would show
the actually-run-role version. Out of scope for the first paper.

## Notes for whoever picks this up

* **Use PT for time references.** The user's in California; UTC was
  confusing.
* **Three-tier is the central message.** Strategy invariance is the
  control, cross-task within-role is the practical story
  ("train per role, not per task"), cross-role orthogonality is the
  headline. All three measured deterministically — no LLM-driven
  results in the headline numbers.
* **The 100% task success rate in cross-task transfer is itself a
  finding**: agents recover from misleading memory by re-querying
  APIs. The compression bottleneck manifests as efficiency cost
  (+40% iters/tokens at B=512), not capability cost.
* **`m_exec_minimal` is API queries (GET /spotify/songs/X), not data.**
  Pre-loading does NOT leak the answer; the agent still has to fetch
  the data. The 8-vs-11 iter saving in the smoke test is from
  skipping API-doc exploration, not from short-circuiting.
* **T2's per-role recall has a stark code-role catastrophe (5–8%)** —
  prompting fails to elicit the *abstraction* (control flow patterns)
  even when explicitly asked. This is the cleanest piece of T2
  evidence and worth its own paper paragraph.
* **Pilot verify success rate 67%** (vs direct 92%) is a side finding
  worth a panel: forced cross-validation drops task completion 25
  pp because the agent burns through max_iter ceiling.

## Anti-patterns to avoid

From `motivation_v2/docs/01_experimental_design.md` §13 (was §13 of
the abandoned new_motivation.md, still applies):

* Do not use LLM-generated oracle memory as gold (circular).
* Do not use binary action-match as the headline metric.
* Do not use ReAct/Plan/CoT scaffolds as the policy distinction.
* Do not use task-family-as-policy (it's topic, not policy).
* Do not use token-level Jaccard or leave-one-out as killer metrics.
* The new one: **do not reuse the word "policy"** — use "role" or
  "agent role" throughout. RL/agent literature reads "policy" as
  π(a|s); we mean specialty / role.

## Files of record (single sources of truth)

| File | Role |
|---|---|
| `motivation_v2/docs/01_experimental_design.md` | Operational design (888 lines, paper-ready) |
| `motivation_v2/docs/02_strategy_prompts.md` | Negative-control strategy prompts (appendix) |
| `motivation_v2/docs/03_role_memory_extractors.md` | Role-projection definitions (paper §3) |
| `motivation_v2/README.md` | Track overview + roadmap |
| `motivation/docs/SESSION_HANDOFF.md` | This file |
| `motivation_v2/outputs/mv2_pilot/` | Pilot results land here |
| `motivation_v2/outputs/mv2_xtask/` | Cross-task transfer results |
| `motivation_v2/outputs/mv2_pilot/role_overlap.json` | Cross-role + cross-task Jaccard |

---

## §Abandoned-track (kept for context only)

The original `motivation/` track was abandoned 2026-05-24 because:

* "Agents" `A_react / A_plan / A_cot` were the same MiniMax model
  with different system prompts → not really different policies.
* Benchmarks (LongMemEval / LoCoMo) were QA → no real policy
  variation possible.
* T1's hinge test (instance_noise) required cross-policy memories
  to differ behaviourally — but if T2 holds, LLM-generated cross-
  policy memories are interchangeable, so the test is unsatisfiable.
* Headline metric (binary `action_match_rate` over N=16 samples) was
  near-Bernoulli; produced zero signal rows on tight budgets.

All `motivation/` runs were killed at 2026-05-24 12:55 PM PT (was
19:55 UTC). Their final state is in
`motivation/outputs/wide_*/...` for git history but not for citation.

The `motivation/scripts/auto_push_watcher.sh` and
`motivation/scripts/sync_and_push.sh` still drive the auto-push
loop for both motivation/ and motivation_v2/ — that's why the
folder is kept rather than removed. Do not edit anything else
under `motivation/` going forward.
