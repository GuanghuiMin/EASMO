# motivation_v11 conditions matrix

> Stage-by-stage workload at the (β) scope: train+dev=145 tasks,
> 4 prompt families, N=8 samples, K=2 stress, entropy_selector
> initially only on ACON_UTCO.

## 1. Task pool

| split | n_tasks | role |
|---|---:|---|
| `train` (AppWorld train.txt) | 89 | source of primary + secondary |
| `dev` (AppWorld dev.txt) | 56 | source of primary + secondary |
| **total** | **145** | |

## 2. Stage workload table

| stage | unit | n units | rate (with 6 workers) | wall-clock |
|---|---|---:|---:|---:|
| 00 prepare | (nothing) | – | – | < 1 min |
| 01 build_cases | AppWorld baseline run | 145 | ~10/min | ~24 min |
| 02 render_prompts | rendered example | 3 × 4 = 12 | instant | < 1 min |
| 03 generate_candidates | MiniMax compress | 145 × 4 × 9 = **5,220** | ~14/min | ~6.2 h |
| 04 serial_recompression_stress | MiniMax stress | 5,220 × 2 = **10,440** | ~14/min | ~12.4 h |
| 05 run_behavior_c1_ck | AppWorld agent run | 5,220 × 2 = **10,440** | ~10/min | ~17.4 h |
| 06a pointwise_verifier | MiniMax verifier | 5,220 × 2 = **10,440** | ~14/min | ~12.4 h |
| 06b pairwise_verifier | MiniMax pairwise match | 145 × 4 × 2 × 7 = **8,120** | ~14/min | ~9.7 h |
| 06c continuation_entropy | MiniMax entropy sample | 145 × 1 × 2 × 5 = **1,450** (ACON_UTCO only) | ~14/min | ~1.7 h |
| 07-13 analysis + plots + report | pandas / matplotlib | – | – | ~30 min |
| **TOTAL** | | | | **~63 h ≈ 2.6 days** |

(My earlier ~68 h estimate was slightly conservative.)

## 3. Forward-compat to plan (α)

To upgrade entropy_selector to all 4 families:

```bash
PYBIN=/workspace/EASMO/.venv/bin/python ${PYBIN} -u scripts/06c_continuation_entropy.py \
    --families general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO
```

This **adds** 1,450 × 3 = 4,350 entropy calls (~5.2 h) for the 3
remaining families, with no rework on stages 00-05 or 06a/06b. Then
re-run stages 07 + 12 + 13 (~30 min) to refresh selector recovery
tables and figures.

## 4. Per-family conditions cross-table

For every (task, family, eval_round) we evaluate:

| selector | source | needs ground truth? |
|---|---|---|
| `greedy` | the single greedy candidate | no |
| `random_sample` | random pick over 8 samples (fixed seed 20260531) | no |
| `shortest_sample` | shortest sample at this round | no |
| `oracle_best_of_n` | sample with max CK reward | **yes** |
| `best_c1` | sample with max C1 reward, evaluated at this round | **yes** |
| `best_ck` | sample with max CK reward, evaluated at this round | **yes** |
| `pointwise_verifier` | sample with max selector_score from §06a | no |
| `pairwise_verifier` | winner of §06b tournament | no |
| `continuation_entropy` | sample with max entropy-selector-score from §06c | no |

So there are **9 selectors × 2 eval rounds × 4 prompt families × ~145 cases ≈ 10,440 selector decisions**.

## 5. Behavior runs not in the matrix

Stage 05 runs the AppWorld agent on EVERY candidate × {C1, CK}.
The matrix above lists what the selector PICKS; the universe of
agent runs is fixed (10,440). Selector evaluation reads from the
same stage-05 jsonl, so adding/removing selectors costs nothing on
the agent-runs side.

## 6. What conditions are NOT in v11

Explicitly out of scope (spec §2):
* No entity / fact retention as main evidence
* No recovered-then-dropped audit loop
* No Qwen SFT or GRPO
* No policy training
* No runtime retrieval / fact-table / projection
* No claim that any surface chunk type is universally important
* No rewritten ACON prompts

Deferred (could be added with `STAGES="..."` env if needed):
* Budget sweep (`max_chars ∈ {1000, 1500, 2000}`) — spec §15 says optional
* `EVAL_INTERMEDIATE_STRESS=true` — spec §9.1, off by default
