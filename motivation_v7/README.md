# motivation_v7 — Abstraction Prior & Iterative Compression Dynamics

This track tests whether LLM history compressors have an
**unconditioned abstraction preference** (Claim A) and whether
repeated compression has a **stable information-loss hierarchy**
across fact types (Claim B). Diagnostic only — no DPO/SFT/RL, no
prompt optimisation.

## Compressors (per spec §4)

| role | model id | endpoint |
|---|---|---|
| Compressor A | `qwen3-4b-instruct-2507` | local vLLM `http://127.0.0.1:8000/v1` |
| Compressor B | `MiniMaxAI/MiniMax-M2.5` | shared `http://10.183.22.68:8005/v1` |
| Cross-evaluator scorer | each model scores the *other* model's outputs |

ACON prompts loaded verbatim from the official microsoft/acon repo at
`/workspace/acon` (commit `d63f9ae`).

- UT: `experiments/appworld/prompts/context_opt/prompt_history_v2.jinja`
- UTCO: `experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja`

Neither template references `max_chars` directly — length control is
implicit via the "concise" instruction. We pass the value but it is a
no-op for the template renderer.

## Plan B scope (per user buy-in)

| dimension | value |
|---|---|
| N_CASES | 30 (all v3 successful AppWorld dev) |
| max facts/case | 8 (per spec caps) |
| compressors | Qwen3-4B-Instruct-2507 + MiniMax-M2.5 |
| prompt variants | UTCO (samples_4) only |
| budget | 1500 chars (primary) |
| rounds (iterative) | 5 |
| chains per case | **2** (needed + unneeded for one representative EXECUTABLE fact) |
| fact extractor / condition generator | MiniMax-M2.5 |

This is a deliberate scope reduction vs spec; the deviation is
documented in `docs/04_results_summary.md` once the run finishes.

## Run

```bash
nohup bash scripts/run_all.sh > outputs/logs/runall_full.log 2>&1 &
```

Env knobs (override defaults):

| var | default |
|---|---|
| `MODELS` | `qwen,minimax` |
| `PROMPT_VARIANTS` | `UTCO` |
| `BUDGET_CHARS` | `1500` |
| `ROUNDS` | `5` |
| `N_CASES` | unset (all 30) |
| `STAGES` | `00,01,02,03,04,05,06,07,08,09,10` |

## Layout

```
motivation_v7/
├── README.md
├── docs/                         — 01-04 paper-tier docs
├── prompts/                      — verbatim ACON copies (UT + UTCO + system)
├── motivation_v7/                — python package
├── scripts/                      — 11 stage scripts + run_all.sh
├── data/                         — case_pool, fact bank, need conditions
└── outputs/
    ├── provenance/               — ACON commit + sha256
    ├── raw/                      — JSONL: compressions, retention scores
    ├── tables/                   — CSV: metrics
    ├── figures/                  — PNG + PDF
    ├── reports/                  — markdown summary
    └── logs/                     — per-stage stdout
```

## Status

See `outputs/reports/motivation_v7_results_summary.md` after the
pipeline completes (stage 10 writes it automatically).
