# motivation_v8 — Fixed-Point Analysis of General LLM Context Compression

Diagnostic-only follow-up to motivation_v7. v7 used ACON's structured
prompts and found **SDI ≈ 1** (need-label explains essentially zero
retention variance) + a stable cross-model fact-type hierarchy.
**v8 removes ACON entirely** and tests whether the same fixed-point
abstraction-prior phenomenon holds under general LLM compression
prompts.

## Key changes vs v7

| | v7 (ACON) | v8 (general) |
|---|---|---|
| Compressor prompt | `prompt_history_v2` + UTCO `samples_4` (from microsoft/acon) | P1 `general_task_aware`, P2 `general_task_agnostic` |
| Initial context | RAW_FULL only | **RAW_FULL + DETAIL_HEAVY + NARRATIVE_HEAVY + FACT_TABLE_ONLY** (Exp 3) |
| Fixed-point analysis | implicit via convergence rate | **explicit fixed-point composition + need-shift + basin contraction** |
| Cases / facts / conditions | 30 / 233 / 150 (quality-passed) | **same** (reused from v7) |

## Run config (full, spec §22)

| | value |
|---|---|
| N_CASES | 30 |
| MAX_FACTS_PER_CASE_SINGLE | 6 |
| N_ITER_CASES | 20 |
| N_BASIN_CASES | 12 |
| ROUNDS | 6 |
| Budget | 1500 chars |
| Prompt families | P1 (task-aware) + P2 (task-agnostic) |
| Compressors | Qwen3-4B-Instruct-2507 + MiniMax-M2.5 |

## Run

```bash
nohup bash scripts/run_all.sh > outputs/logs/runall_full.log 2>&1 &
```

Env knobs documented at the top of `scripts/run_all.sh`.

## Layout

```
motivation_v8/
├── README.md
├── docs/                   01-experimental_design, 02-prompts, 03-metrics, 04-results_summary
├── prompts/                P1/P2/P3 prompt md files (for reference)
├── motivation_v8/          python package
├── scripts/                10 stage scripts + run_all.sh
├── data/                   cases (from v7), fact bank (from v7), need_conditions (from v7)
└── outputs/
    ├── raw/                JSONL
    ├── tables/             CSV (14+ metric tables)
    ├── figures/            PNG + PDF (9 figures)
    ├── reports/            results_summary.md
    └── provenance/         prompt SHA256, run config, source artifacts
```

## Comparison to v7

Both rounds share:
- the same 30 AppWorld dev cases
- the same 233 substring-grounded facts
- the same 150 quality-passed need/unneeded condition pairs
- the same cross-model retention scorer (Qwen ↔ MiniMax)

This makes v7 vs v8 a direct A/B test of **prompt family** while
holding everything else fixed. Final analysis lives in
`docs/04_results_summary.md`.
