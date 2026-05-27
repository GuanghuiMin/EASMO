# motivation_v5 — ACON Failure-Mode Audit on AppWorld

> Track: `EASMO/motivation_v5/`
> Spec: [`docs/00_spec.md`](docs/00_spec.md) (symlinked to
> `user_feedback/acon_appworld_failure_audit_motivation_experiment.md`)
>
> **NOT** a new compression method. A diagnostic motivation study
> that audits ACON-style failures to identify the real compression
> bottleneck before designing a method.

## Information flow we are auditing

```
full context trajectory
        ↓ ACON compressor          (= motivation_v3's acon_style_summary)
ACON compressed trajectory / context
        ↓ audit model supplementation   (Qwen3-4B reads baseline + compressed,
audit-augmented context                  adds back missing actionable facts)
        ↓ compressor again         (re-run acon_style_summary on augmented)
recompressed context
        ↓ downstream agent
final_after_recompression_success
```

The headline question is the **recovered-then-dropped** pattern: how
often does the audit model recover useful state from the full
trajectory only to have the recompressor drop it again?

## Models

| Role | Model | Endpoint |
|---|---|---|
| Primary auditor | **Qwen3-4B** (`qwen3-4b`) | `http://127.0.0.1:8000/v1` (local vLLM, started via `/workspace/qwen3-vllm/serve.sh`) |
| Verifier auditor | **MiniMax-M2.5** | `http://10.183.22.68:8005/v1` (shared vLLM) |
| Recompressor + downstream agent | **MiniMax-M2.5** | same |
| Rule-based grounding verifier | n/a (deterministic substring check) | local |

Temperature 0 for all audit calls; temperature 0.2 for recompressor;
temperature 0.0 (or default seed=42) for downstream agent.

## Case pool

Reuse motivation_v3's 30 dev tasks. Per spec §4.2 we prioritise:

* **Tier 1**: `baseline_success=True AND acon_success=False`
  (~8 cases at cap=15, ~5 at cap=8 in v3 data)
* **Tier 2**: `baseline_success=True AND acon_success=True BUT step_ratio >= 1.5`
  (~1 case at cap=15)

After dedup ≈ 12 unique Tier 1+2 cases. We additionally run the full
audit pipeline on all 60 (30 tasks × 2 budgets) cases so the
aggregate denominator is well-defined, but the headline numbers focus
on Tier 1+2.

## Pipeline (12 stages)

| Stage | Script | Output |
|---|---|---|
| 01 | `01_build_raw_cases.py` | `data/raw_cases.jsonl` (60 cells from v3) |
| 02 | `02_sample_cases.py` | `data/sampled_cases.jsonl` (Tier 1+2 filtered) |
| 03 | `03_build_audit_augmented.py` | adds `audit_augmented_context` to sampled cases (Qwen call) |
| 04 | `04_recompress.py` | adds `recompressed_context` (MiniMax recompressor) |
| 05 | `05_rerun_downstream.py` | adds `final_after_recompression_success` (downstream agent on recompressed) |
| 06 | `06_run_audit.py` | `outputs/raw/qwen_case_audits.jsonl` + `qwen_addition_audits.jsonl` + `qwen_recompression_audits.jsonl` |
| 07 | `07_run_verify.py` | `outputs/raw/minimax_verifications.jsonl` + `rule_based_grounding.jsonl` |
| 08 | `08_merge_audits.py` | `outputs/raw/merged_case_audits.jsonl` |
| 09 | `09_aggregate.py` | 5 spec tables in `outputs/tables/` |
| 10 | `10_plot_figures.py` | 3 spec figures in `outputs/figures/` |
| 11 | `11_write_per_case.py` | per-case markdown reports |
| 12 | `12_write_motivation.py` | `outputs/reports/motivation_summary.md` |

Top-level: `bash scripts/run_all.sh`. Estimated wall-clock 1.5-2h.

## File layout (per spec §5)

```
motivation_v5/
├── docs/                              v2-style explanatory docs (written post-run)
│   └── 00_spec.md                     → user_feedback spec
├── prompts/                           5 prompt templates verbatim from spec §8-12
│   ├── 01_case_failure_audit.md
│   ├── 02_audit_addition_audit.md
│   ├── 03_recompression_loss_audit.md
│   ├── 04_verifier_resolution.md
│   └── 05_aggregate_summary.md
├── motivation_v5/                     python package
│   ├── data.py                        case I/O + v3 reuse + schema
│   ├── clients.py                     Qwen + MiniMax OpenAI clients
│   ├── audit.py                       case / addition / recompression / verifier prompts
│   ├── compressor.py                  ACON-style recompressor (reuses v3 prompt)
│   ├── augmenter.py                   audit-augmented context generator
│   ├── rule_verify.py                 deterministic grounding checks
│   └── runner.py                      downstream agent for final_after_recompression_success
├── scripts/                           12 stage scripts + run_all.sh
├── data/
│   ├── raw_cases.jsonl
│   └── sampled_cases.jsonl
└── outputs/
    ├── raw/                           all JSONL audits + verifications + merged
    ├── tables/                        5 spec CSVs
    ├── figures/                       3 spec PNG/PDFs
    ├── reports/                       motivation_summary.md + per_case_markdown/
    └── sprint_logs/
```

## Reproduction

```bash
QWENPY=/workspace/qwen3-vllm/.venv/bin/python   # has openai client + httpx
PYBIN=/workspace/EASMO/.venv/bin/python          # general analysis
ACONPY=/workspace/acon/.venv/bin/python          # downstream agent runner

cd /workspace/EASMO/motivation_v5
bash scripts/run_all.sh
```

Each stage is idempotent and re-readable; if a single stage fails,
re-run only that script with its inputs from `data/` or
`outputs/raw/`.
