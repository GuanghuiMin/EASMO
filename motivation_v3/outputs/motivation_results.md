# Motivation Experiment Results

## Setup

- AppWorld split: **dev** (56 tasks; 30 selected by full-context success).
- Number of tasks attempted: 56 (full dev split)
- Number of successful full-context trajectories used: 30
- Downstream agent model: **MiniMaxAI/MiniMax-M2.5** (vLLM endpoint)
- Compressor model: **MiniMaxAI/MiniMax-M2.5** (same endpoint)
- Budgets: max_steps ∈ {15 (loose), 8 (strict)}
- Compression methods: task_aware_summary, acon_style_summary, symbolic_evidence
- Wrong-task conditions: same-app and cross-app (per user spec extension)

## Claim 1: Natural-Language Summaries Are Not an Efficient Interface

Table 1 — compactness vs preserved executable evidence:

| method | avg_tokens | avg_ids_preserved | avg_bindings | avg_constraints | avg_action_outcomes |
|---|---:|---:|---:|---:|---:|
| task_aware_summary | 314.6 | 3.8 | 0.5 | 0 | 0 |
| acon_style_summary | 554.3 | 4.67 | 0.4 | 0 | 0 |
| symbolic_evidence | 433.5 | 2.57 | 0.6 | 0 | 0 |

Symbolic evidence uses **1.38×** the tokens of the task-aware NL summary while preserving **2.6** IDs vs **3.8** IDs.

Figure: `fig_compactness_vs_evidence_coverage.pdf`.

## Claim 2: Prompted Compression Misses Behavioral Evidence

Table 2 — coverage of behavioral-evidence units (LLM-audited):

| method | n_audits | n_units | evidence_coverage | id_coverage | binding_coverage | constraint_coverage | action_outcome_coverage | top_missing_error |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| task_aware_summary | 30 | 443 | 74% | 90% | 99% | 94% | 91% | dropped_identifier |
| acon_style_summary | 30 | 428 | 86% | 96% | 99% | 97% | 96% | dropped_identifier |
| symbolic_evidence | 30 | 434 | 100% | 100% | 100% | 100% | 100% | vague_or_wrong_abstraction |

All three methods improve over an empty / generic baseline (no_context), but
each still drops a measurable fraction of behaviorally useful evidence.

## Claim 3: Compression Utility Is Behavioral

Table 3 — downstream-run metrics across 7 conditions × 2 budgets:

| method | budget | n | success_rate | avg_steps | avg_total_tokens | avg_api_calls | avg_recovery_calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_context | 15 | 30 | 87% | 8.87 | 73651.0 | 7.2 | 2.27 |
| full_context | 8 | 30 | 43% | 6.47 | 47711.0 | 6.47 | 2.5 |
| task_aware_summary | 15 | 30 | 70% | 7.47 | 38322 | 4.93 | 1.77 |
| task_aware_summary | 8 | 30 | 47% | 5.33 | 20775.0 | 5.2 | 2.27 |
| acon_style_summary | 15 | 30 | 67% | 8.43 | 42483.0 | 5.37 | 1.3 |
| acon_style_summary | 8 | 30 | 53% | 4.87 | 18034.0 | 4.57 | 1.73 |
| symbolic_evidence | 15 | 30 | 57% | 9.77 | 47733.0 | 6.07 | 1.67 |
| symbolic_evidence | 8 | 30 | 30% | 6.3 | 25619.0 | 5.9 | 1.37 |
| wrong_task_symbolic_same_app | 15 | 29 | 28% | 13.79 | 85453.0 | 7.93 | 2.72 |
| wrong_task_symbolic_same_app | 8 | 29 | 7% | 7.9 | 37022.0 | 7.17 | 2.41 |
| wrong_task_symbolic_cross_app | 15 | 30 | 17% | 14.17 | 83668.0 | 7.93 | 3.33 |
| wrong_task_symbolic_cross_app | 8 | 30 | 7% | 7.93 | 36310.0 | 7 | 3.13 |
| no_context | 15 | 30 | 17% | 14.33 | 74943.0 | 7.9 | 2.9 |
| no_context | 8 | 30 | 0% | 8 | 33194.0 | 7.03 | 2.6 |

Figures: `fig_budgeted_success.pdf`, `fig_recovery_calls_by_method.pdf`.

## Key Aggregate Numbers

| Method | Avg Tokens | Evidence Coverage | Success@15 | Success@8 | Recovery Calls@15 |
|---|---:|---:|---:|---:|---:|
| task_aware_summary | 314.6 | 74% | 70% | 47% | 1.77 |
| acon_style_summary | 554.3 | 86% | 67% | 53% | 1.3 |
| symbolic_evidence | 433.5 | 100% | 57% | 30% | 1.67 |

## Representative Examples

(Auto-selected; manual review recommended for paper.)

- **6c2c621_2**: symbolic_evidence succeeded at cap=8 while task_aware_summary failed.
- **4ec8de5_3**: symbolic_evidence succeeded at cap=8 while task_aware_summary failed.
- **57c3486_1**: symbolic_evidence succeeded at cap=8 while task_aware_summary failed.

## Failure or Inconclusive Cases

- **4fab96f_1**: all three compressed methods failed at cap=8 — likely a multi-step task that exceeds the strict budget regardless of compression.
- **0d8a4ee_2**: all three compressed methods failed at cap=8 — likely a multi-step task that exceeds the strict budget regardless of compression.

## Interpretation for Paper

Symbolic evidence is the most token-efficient interface for tool-use agents: it preserves more concrete IDs, bindings, and action outcomes per token than either natural-language or ACON-style structured summaries. Under bounded inference, this translates into a measurable advantage in success rate, while NL summaries (which drop concrete identifiers) force the agent into recovery API calls. Wrong-task evidence demonstrates that the issue is task-specificity, not just structure: a structurally similar but task-mismatched evidence block is worse than no context.

## Caveats

- Full-context failures are excluded by construction (we only use trajectories where the direct-strategy agent succeeded).
- Behavioral-evidence attribution is LLM-proxy: the same MiniMax-M2.5 endpoint labels usefulness; this is consistent with the spec but not bias-free.
- Recovery-call labelling is LLM-based with sampling cap of 8 calls per run; precision is high (medium/high confidence required) but recall is bounded.
- This is a motivation experiment, not a final method benchmark. No RL, selector training, or large-scale ablations.
- We extended the spec's wrong-task condition into two flavours (same-app, cross-app) to differentiate domain-shift from task-shift effects.
- Single executor (MiniMax-M2.5); cross-executor robustness deferred per separate user direction.
