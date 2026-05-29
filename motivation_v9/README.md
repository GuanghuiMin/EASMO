# motivation_v9 — Behavioral Validation, Fixed-Point Stress, Chunk Information Advantage

Diagnostic-only behavior-first follow-up to v7/v8. Where v7/v8 measured
**fact retention** under ACON-style and general LLM compression, **v9
measures whether the surface-type abstraction prior actually matters
for AppWorld task pass rate**. v9 is the motivation step for a later
RL method (fixed-point-stressed compressor RL with chunk-level
advantage shaping).

## Three claims (spec §1)

| | Claim | Method |
|---|---|---|
| **1** | ACON greedy compression is not behavior-optimal under its own output distribution | Sample N=8 stochastic ACON outputs per case; compare best-of-N pass rate to greedy. |
| **2** | One-step compression is behaviorally fragile under repeated-compression stress | Compute `T^K(c)`; compare `Pass(C1)` vs `Pass(CK)`. |
| **3** | Useful compression encodes natural-language causal/control relations, not just entities | Segment compressed contexts into NL chunks; ablate per chunk; measure behavior advantage; label chunk types via MiniMax. |

## Models (spec §3)

| role | model | endpoint |
|---|---|---|
| Compressor (primary) | `MiniMaxAI/MiniMax-M2.5` | `http://10.183.22.68:8005/v1` |
| Compressor (optional small) | `qwen3-4b-instruct-2507` | `http://127.0.0.1:8000/v1` |
| Downstream agent | `MiniMaxAI/MiniMax-M2.5` | same as compressor endpoint |
| All auditors / chunk-labelers | **MiniMax-M2.5 only** (Qwen forbidden as auditor) | |

## Run config (Full, spec §22)

| | value |
|---|---|
| N_CASES | 30 (reused from v3) |
| N_SAMPLES | 8 |
| STRESS_ROUNDS_K | 2 |
| TARGET_MAX_CHARS | 1500 |
| BUDGET_MAX_STEPS_PRIMARY | 15 |
| CHUNK_ABLATION_MAX_CASES | 12 |
| Compressor track | **MiniMax only** (user decision; Qwen deferred) |

## Cost estimate

| stage | calls | time |
|---|---:|---:|
| 02 candidates | 270 LLM compressions | ~10 min |
| 03 stress chains | 540 LLM compressions | ~10 min |
| **04 behavior C1+CK** | **540 AppWorld agent runs** (workers=6) | **~5 h** |
| 09a chunk re-stress | ~120 LLM compressions | ~5 min |
| **09 chunk ablation behavior** | **~120 AppWorld runs** | **~1 h** |
| 11 chunk labels | ~120 MiniMax calls | ~5 min |
| total | **~6-7 h** | |

## Run

```bash
nohup bash scripts/run_all.sh > outputs/logs/runall_full.log 2>&1 &
```

Stage 04 + 09 spawn AppWorld agent subprocesses via
`/workspace/acon/.venv` (pydantic-v1) using
`motivation_v4.runner.run_with_compressed_context`.

Env knobs documented at top of `scripts/run_all.sh`.

## Layout

```
motivation_v9/
├── README.md
├── docs/01_experimental_design.md, 02_prompts.md, 03_metrics.md, 04_results_summary.md
├── prompts/acon_utco_official.md, acon_system_prompt.md
├── motivation_v9/    # python package
├── scripts/00..14, 09a + run_all.sh
├── data/v9_cases.jsonl
└── outputs/
    ├── raw/          candidate_compressions, stress_chains, behavior_runs_c1_ck,
    │                 chunks, chunk_ablation_*, chunk_type_labels
    ├── tables/       best_of_n_*, c1_ck_*, chunk_*
    ├── figures/      9 figures (PNG + PDF)
    ├── reports/      motivation_v9_results_summary.md
    └── provenance/   acon prompt sha256, run_config, model_endpoints
```

## Comparison to v7/v8

| | v7 / v8 | **v9** |
|---|---|---|
| Output metric | retention probability per fact | **AppWorld task pass rate** |
| Per-case cost | ~10 LLM scoring calls | **~18 AppWorld agent runs (C1+CK × 9 candidates)** |
| Prompt | ACON UTCO (v7) / general (v8) | **ACON UTCO only** (same as v7) |
| Compressor distribution | greedy only | **greedy + N stochastic samples** |
| Stress chain | informal in v7 | **explicit T^K** |
| Chunk advantage | n/a | **per-chunk ablation behavior** |

## Status

See `docs/04_results_summary.md` after the pipeline completes.
