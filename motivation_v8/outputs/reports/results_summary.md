# motivation_v8 Results — Fixed Points of General LLM Compression

> Auto-written by `scripts/09_write_report.py` at 2026-05-29 01:15Z.

## 1. Setup
- `n_cases` = **30** (reused from motivation_v7)
- `n_facts` (filtered, substring-grounded) = **233**
- `n_conditions` (quality-passed pairs) = **300**
- `n_compressions` (single round) = **1160**
- `n_compressions` (iterative + basin) = **1470**
- `n_retention_scores` = **12640**
- Single-round budget violation rate (mean across (model, prompt)) = 0.005
- Iterative budget violation rate (mean) = 0.000
- Compressors: `qwen3-4b-instruct-2507`, `MiniMaxAI/MiniMax-M2.5`
- Prompt families: `general_task_aware` (P1), `general_task_agnostic` (P2)
- Budget: 1500 chars (primary)
- ACON prompts: NOT USED in v8 (spec §1).

## 2. Claim A: Single-Round Need Conditioning

**Verdict A: STRONG POSITIVE**

Surface-dominance regression:

| model   | prompt_family         |   budget_chars |   n |   r2_need |   r2_type |   r2_both |   sdi |
|:--------|:----------------------|---------------:|----:|----------:|----------:|----------:|------:|
| minimax | general_task_agnostic |           1500 | 290 |     0     |     0.292 |     0.292 | 1     |
| minimax | general_task_aware    |           1500 | 290 |     0.026 |     0.279 |     0.319 | 0.83  |
| qwen    | general_task_agnostic |           1500 | 290 |     0     |     0.14  |     0.14  | 0.998 |
| qwen    | general_task_aware    |           1500 | 290 |     0.012 |     0.082 |     0.096 | 0.737 |

Preference Inversion Rate:

| model   | prompt_family         |   budget_chars |   n_pairs |   preference_inversion_rate |   ci_low |   ci_high |
|:--------|:----------------------|---------------:|----------:|----------------------------:|---------:|----------:|
| minimax | general_task_agnostic |           1500 |        15 |                       0.133 |    0     |     0.333 |
| minimax | general_task_aware    |           1500 |        15 |                       0     |    0     |     0     |
| qwen    | general_task_agnostic |           1500 |        15 |                       0.267 |    0.067 |     0.533 |
| qwen    | general_task_aware    |           1500 |        15 |                       0     |    0     |     0     |

Figures: `fig_need_effect_by_fact_type`, `fig_surface_dominance_index`, `fig_preference_inversion_rate`.

## 3. Claim B: Fixed-Point Convergence

Convergence rate (text sim ≥ 0.95 ∧ fact Jaccard ≥ 0.95 ∧ |Δlen|/len ≤ 0.02): **84.9%** of 186 chains.

Convergence rate by (model, prompt):

|                                      |   converged_rate |
|:-------------------------------------|-----------------:|
| ('minimax', 'general_task_agnostic') |            0.737 |
| ('minimax', 'general_task_aware')    |            0.757 |
| ('qwen', 'general_task_agnostic')    |            1     |
| ('qwen', 'general_task_aware')       |            0.932 |


## 4. Claim C: Fixed-Point Composition

Mean retention at fixed point by coarse group:

| model   | prompt_family         | coarse_group   |   survival_rate_fixed |
|:--------|:----------------------|:---------------|----------------------:|
| minimax | general_task_agnostic | CONTROL        |                 0.083 |
| minimax | general_task_agnostic | EXECUTABLE     |                 0.549 |
| minimax | general_task_agnostic | NARRATIVE      |                 0.875 |
| minimax | general_task_agnostic | TASK_STATE     |                 1     |
| minimax | general_task_aware    | CONTROL        |                 0.392 |
| minimax | general_task_aware    | EXECUTABLE     |                 0.642 |
| minimax | general_task_aware    | NARRATIVE      |                 0.464 |
| minimax | general_task_aware    | TASK_STATE     |                 0.667 |
| qwen    | general_task_agnostic | CONTROL        |                 0.125 |
| qwen    | general_task_agnostic | EXECUTABLE     |                 0.453 |
| qwen    | general_task_agnostic | NARRATIVE      |                 0.25  |
| qwen    | general_task_agnostic | TASK_STATE     |                 0.833 |
| qwen    | general_task_aware    | CONTROL        |                 0.396 |
| qwen    | general_task_aware    | EXECUTABLE     |                 0.629 |
| qwen    | general_task_aware    | NARRATIVE      |                 0.286 |
| qwen    | general_task_aware    | TASK_STATE     |                 0.625 |

Bottom-3 fact types by AUSC per (model, prompt_family):

- minimax / general_task_agnostic: NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT (1.58), API_SCHEMA_OR_PARAMETER (2.74), AUTH_OR_ACCESS_TOKEN (3.43)
- minimax / general_task_aware: NARRATIVE_PROGRESS (1.50), NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT (1.58), NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT (1.69)
- qwen / general_task_agnostic: NARRATIVE_PROGRESS (0.75), NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT (1.92), RUNTIME_VARIABLE (2.62)
- qwen / general_task_aware: NARRATIVE_PROGRESS (0.50), NARRATIVE_PROGRESS (0.75), ENVIRONMENT_STATE (1.00)


## 5. Claim D: Need-Conditioned Fixed-Point Shift

**Verdict D: WEAK / NEGATIVE**

Mean Δ_need^∞ by (model, prompt, coarse_group):

| model   | prompt_family      | coarse_group   |   delta_need_infty |
|:--------|:-------------------|:---------------|-------------------:|
| minimax | general_task_aware | CONTROL        |              0.435 |
| minimax | general_task_aware | EXECUTABLE     |              0.264 |
| minimax | general_task_aware | NARRATIVE      |              0.312 |
| minimax | general_task_aware | TASK_STATE     |              0.444 |
| qwen    | general_task_aware | CONTROL        |              0.476 |
| qwen    | general_task_aware | EXECUTABLE     |              0.297 |
| qwen    | general_task_aware | NARRATIVE      |              0.125 |
| qwen    | general_task_aware | TASK_STATE     |              0.611 |


## 6. Claim E: Basin of Attraction

**Verdict E: WEAK / NEGATIVE**

Mean basin metrics (init → final pairwise distance, contraction = final/init):

| model   | prompt_family      |   init_fact_jaccard_distance |   fin_fact_jaccard_distance |   contraction_fact_jaccard |   init_type_l1_distance |   fin_type_l1_distance |   contraction_type_l1 |
|:--------|:-------------------|-----------------------------:|----------------------------:|---------------------------:|------------------------:|-----------------------:|----------------------:|
| minimax | general_task_aware |                        0.051 |                       0.728 |                4.59965e+07 |                   0.045 |                  0.439 |           2.58537e+07 |
| qwen    | general_task_aware |                        0.158 |                       0.694 |                3.60312e+07 |                   0.107 |                  0.394 |           1.91369e+07 |


## 7. Cross-Model and Cross-Prompt Stability

Top pairwise Kendall τ (rank correlation across (model, prompt, init, cond)):

| model_a   | prompt_a              | init_a          | cond_a        | model_b   | prompt_b           | init_b          | cond_b   |   n_types |   kendall_tau |   kendall_p |   spearman_rho |   spearman_p |
|:----------|:----------------------|:----------------|:--------------|:----------|:-------------------|:----------------|:---------|----------:|--------------:|------------:|---------------:|-------------:|
| minimax   | general_task_aware    | RAW_FULL        | needed        | qwen      | general_task_aware | NARRATIVE_HEAVY | needed   |        10 |         0.778 |       0.001 |          0.927 |        0     |
| qwen      | general_task_agnostic | RAW_FULL        | task_agnostic | qwen      | general_task_aware | NARRATIVE_HEAVY | needed   |        10 |         0.778 |       0.001 |          0.915 |        0     |
| minimax   | general_task_agnostic | RAW_FULL        | task_agnostic | minimax   | general_task_aware | RAW_FULL        | needed   |        11 |         0.745 |       0.001 |          0.873 |        0     |
| minimax   | general_task_aware    | NARRATIVE_HEAVY | needed        | qwen      | general_task_aware | RAW_FULL        | needed   |        10 |         0.689 |       0.005 |          0.855 |        0.002 |
| qwen      | general_task_aware    | RAW_FULL        | needed        | qwen      | general_task_aware | RAW_FULL        | unneeded |        11 |         0.673 |       0.003 |          0.827 |        0.002 |
| minimax   | general_task_aware    | RAW_FULL        | unneeded      | qwen      | general_task_aware | NARRATIVE_HEAVY | needed   |        10 |         0.644 |       0.009 |          0.758 |        0.011 |
| minimax   | general_task_aware    | RAW_FULL        | needed        | minimax   | general_task_aware | RAW_FULL        | unneeded |        11 |         0.636 |       0.006 |          0.845 |        0.001 |
| qwen      | general_task_aware    | DETAIL_HEAVY    | needed        | qwen      | general_task_aware | FACT_TABLE_ONLY | needed   |        10 |         0.6   |       0.017 |          0.758 |        0.011 |

**Verdict B/C: PARTIAL POSITIVE**

## 8. Comparison to v7
v7 used ACON UTCO prompts and reported SDI ≈ 0.96 (MiniMax) / 0.99 (Qwen), cross-model Kendall τ = 0.49, AUTH_OR_ACCESS_TOKEN as universal repellor.
v8 uses general (non-ACON) prompts. See §2 SDI and §7 τ above to compare.

## 9. Negative Findings and Caveats
Budget violations (top-3 worst):

| model   | prompt_family         |   budget_chars |   n |   violation_rate |   median_length |   p90_length |   p99_length |
|:--------|:----------------------|---------------:|----:|-----------------:|----------------:|-------------:|-------------:|
| minimax | general_task_aware    |           1500 | 290 |            0.014 |             973 |       1307.2 |      1716.59 |
| qwen    | general_task_aware    |           1500 | 290 |            0.007 |             822 |       1044.3 |      1267.15 |
| minimax | general_task_agnostic |           1500 | 290 |            0     |             791 |       1224   |      1554    |

- v7-derived case pool: no <15-step trajectories.
- Plan B → no UT/P3 ablations; only P1 (task-aware) + P2 (task-agnostic).
- Cross-model retention scoring (Qwen↔MiniMax) may inherit prompt-family-shaped biases.
- Length tolerance for budget = 10 %.

## 10. Paper-Level Interpretation
(Filled in manually in `docs/04_results_summary.md` after the auto-report is reviewed.)

## 11. Files of Record
Tables (`outputs/tables/`):
- `need_effect_by_type.csv`
- `surface_dominance_regression.csv`
- `surface_dominance_index.csv`
- `preference_inversion.csv`
- `condition_responsiveness.csv`
- `survival_by_round_type.csv`
- `fact_type_half_life.csv`
- `ausc_by_type.csv`
- `hazard_by_round_type.csv`
- `hierarchy_rank_by_model_prompt.csv`
- `cross_model_prompt_hierarchy_similarity.csv`
- `convergence_by_chain.csv`
- `fixed_point_composition_by_type.csv`
- `fixed_point_need_shift.csv`
- `basin_similarity.csv`
- `basin_contraction.csv`
- `budget_compliance_single_round.csv`
- `budget_compliance_iterative.csv`

Figures (`outputs/figures/`):
- `fig_need_effect_by_fact_type.{png,pdf}`
- `fig_surface_dominance_index.{png,pdf}`
- `fig_preference_inversion_rate.{png,pdf}`
- `fig_iterative_survival_curves.{png,pdf}`
- `fig_fixed_point_composition_by_type.{png,pdf}`
- `fig_fixed_point_need_shift.{png,pdf}`
- `fig_basin_contraction.{png,pdf}`
- `fig_fixed_point_recall_groups.{png,pdf}`
- `fig_cross_model_prompt_hierarchy_rank.{png,pdf}`