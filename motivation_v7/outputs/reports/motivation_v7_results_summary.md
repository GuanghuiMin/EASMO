# motivation_v7 Results: Abstraction Prior & Iterative Compression Dynamics

> Auto-written by `scripts/10_write_report.py` at 2026-05-28 22:19Z.

## 0. Counts
- `n_cases` = **30**
- `n_facts` (filtered) = **233**
- `n_conditions` = **426**
- `n_compressions` (single round) = **600**
- `n_compressions` (iterative) = **460**
- `n_retention_scores` (single round) = **600**

## 1. ACON prompt provenance
- ACON repo commit: `d63f9ae18959dc7215ff62899c94c5e8c56847ae`
- **UT** template: `experiments/appworld/prompts/context_opt/prompt_history_v2.jinja` (sha256 `0508caa837c50403be2c8545646359a0fb72009fb14df3a8acf85aedaf649834`)
- **UTCO** template: `experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja` (sha256 `9e50d0f93aca7f75eb723a90a758642d1aac3d7550f6afe1e692e56e2bc7b71c`)
- ACON system prompt: `experiments/appworld/prompts/context_opt/system_prompt.jinja` (sha256 `f9a0a5188d643d0990a422e94557e781e9e4f9c00a6506f699c4956b7096a392`)

## 2. Claim A: Is compression preference need-conditioned?

**Verdict A: STRONG POSITIVE**

### Need effect Δ_need (binary retention) by fact type, per model

| fact_type                           |   minimax |   qwen |
|:------------------------------------|----------:|-------:|
| ACTION_OUTCOME                      |     0.182 |  0.091 |
| API_SCHEMA_OR_PARAMETER             |     0     | -0.171 |
| AUTH_OR_ACCESS_TOKEN                |     0.128 |  0     |
| ENVIRONMENT_STATE                   |    -1     |  0     |
| EXACT_IDENTIFIER                    |     0     |  0.231 |
| FILE_PATH_OR_RESOURCE_LOCATOR       |     0     |  0     |
| NARRATIVE_GOAL                      |     0     |  0     |
| NARRATIVE_PROGRESS                  |     0     | -0.2   |
| NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT |     0.05  |  0.3   |
| NUMERIC_OR_DATE_LITERAL             |     0     |  0.167 |
| RUNTIME_VARIABLE                    |     0.118 |  0     |

### Surface dominance regression

| compressor_model   | prompt_variant   |   budget_chars |   n |   r2_need |   r2_type |   r2_both |   sdi |
|:-------------------|:-----------------|---------------:|----:|----------:|----------:|----------:|------:|
| minimax            | UTCO             |           1500 | 300 |     0.003 |     0.155 |     0.159 | 0.961 |
| qwen               | UTCO             |           1500 | 300 |     0.001 |     0.11  |     0.111 | 0.989 |

### Preference Inversion Rate

| compressor_model   | prompt_variant   |   budget_chars |   n_pairs |   preference_inversion_rate |   ci_low |   ci_high |
|:-------------------|:-----------------|---------------:|----------:|----------------------------:|---------:|----------:|
| minimax            | UTCO             |           1500 |        33 |                       0.212 |    0.091 |     0.364 |
| qwen               | UTCO             |           1500 |        33 |                       0.273 |    0.121 |     0.44  |

Figures: `figures/fig_need_effect_by_fact_type.{png,pdf}`, `figures/fig_surface_dominance_index.{png,pdf}`, `figures/fig_preference_inversion_rate.{png,pdf}`.

## 3. Claim B: Is there a stable iterative information-loss hierarchy?

**Verdict B: STRONG POSITIVE**

### Half-life by fact type

| compressor_model   | prompt_variant   | fact_type                           | coarse_group   |   half_life | half_life_censored   |   final_survival |
|:-------------------|:-----------------|:------------------------------------|:---------------|------------:|:---------------------|-----------------:|
| minimax            | UTCO             | ACTION_OUTCOME                      | EXECUTABLE     |           6 | True                 |            0.8   |
| minimax            | UTCO             | API_SCHEMA_OR_PARAMETER             | EXECUTABLE     |           6 | True                 |            0.608 |
| minimax            | UTCO             | AUTH_OR_ACCESS_TOKEN                | EXECUTABLE     |           1 | False                |            0.368 |
| minimax            | UTCO             | COMPLETED_SUBTASK                   | TASK_STATE     |           6 | True                 |            0.833 |
| minimax            | UTCO             | ENVIRONMENT_STATE                   | TASK_STATE     |           6 | True                 |            1     |
| minimax            | UTCO             | EXACT_IDENTIFIER                    | EXECUTABLE     |           6 | True                 |            0.577 |
| minimax            | UTCO             | NARRATIVE_GOAL                      | NARRATIVE      |           6 | True                 |            0.875 |
| minimax            | UTCO             | NARRATIVE_PROGRESS                  | NARRATIVE      |           6 | True                 |            0.667 |
| minimax            | UTCO             | NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT | CONTROL        |           1 | False                |            0.345 |
| minimax            | UTCO             | NUMERIC_OR_DATE_LITERAL             | EXECUTABLE     |           6 | True                 |            0.833 |
| minimax            | UTCO             | RUNTIME_VARIABLE                    | EXECUTABLE     |           6 | True                 |            0.672 |
| qwen               | UTCO             | ACTION_OUTCOME                      | EXECUTABLE     |           1 | False                |            0.4   |
| qwen               | UTCO             | API_SCHEMA_OR_PARAMETER             | EXECUTABLE     |           1 | False                |            0.365 |
| qwen               | UTCO             | AUTH_OR_ACCESS_TOKEN                | EXECUTABLE     |           1 | False                |            0.105 |
| qwen               | UTCO             | COMPLETED_SUBTASK                   | TASK_STATE     |           1 | False                |            0.333 |
| qwen               | UTCO             | ENVIRONMENT_STATE                   | TASK_STATE     |           6 | True                 |            1     |
| qwen               | UTCO             | EXACT_IDENTIFIER                    | EXECUTABLE     |           1 | False                |            0.462 |
| qwen               | UTCO             | NARRATIVE_GOAL                      | NARRATIVE      |           6 | True                 |            0.625 |
| qwen               | UTCO             | NARRATIVE_PROGRESS                  | NARRATIVE      |           1 | False                |            0.167 |
| qwen               | UTCO             | NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT | CONTROL        |           1 | False                |            0.328 |
| qwen               | UTCO             | NUMERIC_OR_DATE_LITERAL             | EXECUTABLE     |           1 | False                |            0.417 |
| qwen               | UTCO             | RUNTIME_VARIABLE                    | EXECUTABLE     |           1 | False                |            0.362 |

### Cross-model hierarchy similarity

| model_a   | prompt_a   |   budget_a | model_b   | prompt_b   |   budget_b |   n_types |   kendall_tau |   kendall_p |   spearman_rho |   spearman_p |
|:----------|:-----------|-----------:|:----------|:-----------|-----------:|----------:|--------------:|------------:|---------------:|-------------:|
| minimax   | UTCO       |       1500 | qwen      | UTCO       |       1500 |        11 |         0.491 |       0.041 |          0.636 |        0.035 |

### Convergence

Converged within 5 rounds: **79.3%** of chains.
- mean needed_fact_recall_at_convergence = 0.481
- mean narrative_fact_recall_at_convergence = 0.556
- mean executable_fact_recall_at_convergence = 0.446

Figures: `figures/fig_iterative_survival_curves.{png,pdf}`, `figures/fig_survival_hierarchy_heatmap.{png,pdf}`, `figures/fig_cross_model_hierarchy_rank.{png,pdf}`, `figures/fig_fixed_point_recall.{png,pdf}`.

## 4. Caveats
- v3-derived 30 cases are all medium-length (≥15 steps); no <15-step stratum.
- Plan B scope: iterative compression uses 2 chains/case (needed + unneeded for one representative EXECUTABLE fact), not the spec's full sweep of all condition_tasks.
- Single budget = 1500 chars (spec primary). Secondary budgets {800, 2500} not run.
- Single prompt variant = UTCO (samples_4). UT ablation not run.
- Cross-model retention scorer (Qwen-compressions scored by MiniMax, MiniMax-compressions scored by Qwen) per spec §4.

## 5. Files of record
Tables under `outputs/tables/`:
- `need_effect_by_type.csv`
- `surface_dominance_regression.csv`
- `surface_dominance_index.csv`
- `preference_inversion.csv`
- `condition_responsiveness.csv`
- `survival_by_round_type.csv`
- `fact_type_half_life.csv`
- `hazard_by_round_type.csv`
- `ausc_by_type.csv`
- `hierarchy_rank_by_model.csv`
- `cross_model_hierarchy_similarity.csv`
- `convergence_by_case.csv`
- `fact_bank_grounding.csv`
- `need_condition_quality.csv`