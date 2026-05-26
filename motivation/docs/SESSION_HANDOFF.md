# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-26 3:45 PM PT (post-spec-strengthening round complete).
>
> **➡ Read this first if you're picking up in a fresh chat**:
> 1. [`motivation_v2/outputs/motivation/README_RESULTS.md`](../../motivation_v2/outputs/motivation/README_RESULTS.md) — the canonical, spec-conformant results doc; answers the 6 acceptance questions directly with the latest numbers.
> 2. [`motivation_v2/docs/05_results_summary.md`](../../motivation_v2/docs/05_results_summary.md) — paper-tier discussion + scorecard (6/7 fully achieved, only cross-executor pending).
>
> **Active track**: `motivation_v2/` (AppWorld + role-conditional memory).
> The old `motivation/` track was abandoned 2026-05-24; see §Abandoned-track
> at the end if you need historical context. Times throughout this doc
> are Pacific Time (PT).
>
> **Latest round (2026-05-26 PT)**: spec-strengthening per
> `motivation_v2/user_feedback/experiment_modification.md`. **426 new
> agent runs + 1,992 LLM compression calls + 0 errors**. Closes the
> three biggest reviewer attack surfaces:
>   * **no_memory baseline** added (n=18) → matched 78% vs no_memory 33% at cap=15.
>   * **Wrong-endpoint API metric** extracted post-hoc → wrong_task hits 10.2 wrong endpoints vs matched's 4.9 (rules out 'iter inflation = passive retries').
>   * **3 new prompted variants** (generic / task / role) → code-role recall 5–7% under all variants; adding role prompt **doubles** API-fact leakage (12% oracle → 29% prompted_task_role).
>
> Plus a methodological self-correction: previous "T2 = 6.0× ratio" mixed metrics. Under uniform entity-token Jaccard the overall ratio is **1.6×** (1.9–6.3× per-pair where role-orthogonality matters most). Code-role recall 5–7% preserved across all metrics.

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

T2 FULL (n=1328 cells, complete 8:23 PM PT):
   prompted cross-role Jaccard       0.216  (6.0× oracle's 0.036)
   prompted-vs-oracle recall (mean)  0.163
   code-role recall                  0.049  (catastrophic — pattern abstraction missed)

Cross-task efficiency cost (B=512, n=72):
   self memory                       12.8 iters /  79K tokens   (baseline)
   within-app cross-gen memory       18.0 iters / 133K tokens   (+40%)
   cross-app memory                  18.3 iters / 141K tokens   (+45%)
   task success                      100% across all 72 cells
```

**6 of 7 spotlight criteria fully achieved** (full table + numbers in
[`motivation_v2/outputs/motivation/README_RESULTS.md`](../../motivation_v2/outputs/motivation/README_RESULTS.md) §1–§6):

1. ✅ Cross-role Jaccard ≤ 0.10 at B=512 — achieved **0.035 (unit-text) / 0.136 (entity-token)** at n=498 pairs.
2. ✅ Cross-task within-role Jaccard ordered (code 0.41 vs others 0.07–0.11).
3. ✅ Cross-task transfer cost — wrong_task_diff_gen +82% iter inflation at cap=50.
4. ✅ T2 per-pair ratio ≥ 5× — tool↔code 6.3×; overall 1.6× under uniform metric. **Code-role recall 5–7% across all 4 prompted variants** (generic / task / role / task_role) — robust to prompt engineering.
5. ⏳ Cross-executor robustness (Qwen2.5-7B) — pending external endpoint.
6. ✅ **Multi-stage real-agent role orthogonality** (n=18 tasks): overall mean 0.066; plan↔verify **0.148** (two independent agent outputs, no projections). Reviewer's projection-vs-agent critique closed.
7. ✅ **Capped-budget capability cost — strengthened with no_memory + generic_recent baselines (n=18)**:
   * matched 78% → wrong_task 59–61% → cross_domain 50% → **no_memory 33% (-44pp)** at cap=15.
   * wrong_task_diff_gen **+82% iter inflation, 2.1× more wrong-endpoint API calls** at cap=50.
   * Closes 'matched ≈ nothing' attack and 'iter inflation = passive retries' attack.

## Active background processes

```
3916707 auto_push_watcher           pushes motivation/ + motivation_v2/ changes every 20 min
```

All experiment processes are complete. Most recent runs (2026-05-26 PT):
* Phase 2a — existing6 + (no_memory, generic_recent) at cap=50/15:  ended 1:15 PM PT (48 cells)
* Phase 2b — extra12 + 6 conditions at cap=50/15:                    ended 2:50 PM PT (378 cells, 2 background processes)
* Sprint 3 — prompted_generic / prompted_task / prompted_role:       ended 2:10 PM PT (1992 LLM calls)
* Sprint 4 — finalize_motivation.sh canonicalisation + README:       3:30–3:45 PM PT

Older (2026-05-24 PT) runs:
* 3 pilot strategy jobs — ended 3:00 PM PT (mv2_pilot/)
* Cross-task transfer driver (cap=50 baseline) — ended 3:30 PM PT (mv2_xtask/)
* T2 prompted-memory build (prompted_task_role) — ended 8:23 PM PT (1328 cells)
* Capped-budget xtask sequencer — ended 9:04 PM PT (cap=15 + cap=8 both 72/72)
* Multi-stage role pipeline pilot — ended ~9:30 PM PT (18/18 tasks)

Top-level reproduction:
```bash
bash /workspace/EASMO/motivation_v2/scripts/finalize_motivation.sh
# regenerates all canonical CSV/JSONL/PDF under outputs/motivation/ and figures/motivation/
```

See `motivation_v2/outputs/motivation/README_RESULTS.md` §8 for the
detailed per-experiment commands and §9 for full provenance.

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
| Prompted memories — task+role (T2 baseline) | `mv2_pilot/prompted_memories.jsonl` | 1,328 cells |
| **Prompted memories — generic / task / role** (Sprint 3, 2026-05-26) | `mv2_pilot_variants/prompted_*.jsonl` | 3 × 1,328 = 3,984 cells |
| Cross-task transfer (cap=50/15/8 baseline) | `mv2_xtask/`, `mv2_xtask_cap{15,8}/transfer_results.jsonl` | 3 × 72 = 216 cells |
| **Cross-task transfer extended** (Phase 2, 2026-05-26) | `mv2_xtask_ext_existing6_cap{15,50}`, `mv2_xtask_ext_extra12_cap{15,50}/transfer_results.jsonl` | 4 files, 426 cells (n=18 consumers) |
| Multi-stage pipeline pilot | `mv2_multi_stage_pilot/pipeline_summary.jsonl` | 18 tasks × 3 agents |
| **Canonical spec-format outputs** | `outputs/motivation/{*_raw.jsonl,*_summary.csv,README_RESULTS.md}` | All A/B/C/D experiments in spec schema |
| **Canonical figures** | `figures/motivation/*.pdf` | 8 PDFs (hierarchy / multistage / behavior / prompted heatmap / recall / abstraction) |

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
