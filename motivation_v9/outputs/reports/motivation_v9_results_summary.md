# motivation_v9 Results — Behavioral Compression Stress and Chunk Information Advantage

> Auto-written by `scripts/14_write_report.py` at 2026-05-29 18:50Z.

## TL;DR
- Behavioral validation of v7/v8 surface-type abstraction prior findings.
- Tests if ACON's greedy compression is optimal under its own distribution.
- Tests if one-step compression survives repeated-compression stress.
- Tests if natural-language chunks (causal/control) carry behavioral information beyond entity recall.

## Setup
- n_cases = **30**
- n_candidate_compressions = **270**
- n_behavior_runs (C1+CK) = **540**
- n_chunks = **144**
- n_chunk_labels = **144**
- n_chunk_ablation_runs = **156**
- ACON UTCO commit = `d63f9ae18959dc7215ff62899c94c5e8c56847ae`
- ACON history prompt sha256 = `9e50d0f93aca7f75eb723a90a758642d1aac3d7550f6afe1e692e56e2bc7b71c`

## Claim 1: ACON Best-of-N Behavioral Gap

Per (compressor_model, eval_context_round):

| compressor_model   | eval_context_round   |   n_cases |   greedy_pass_rate |   best_of_n_pass_rate |   pass_gain_pp |   greedy_mean_score |   best_of_n_mean_score |   score_gain |   greedy_mean_length |   best_of_n_mean_length |   oracle_win_rate |
|:-------------------|:---------------------|----------:|-------------------:|----------------------:|---------------:|--------------------:|-----------------------:|-------------:|---------------------:|------------------------:|------------------:|
| minimax            | C1                   |        30 |                0.7 |                 0.967 |         26.667 |                 0.7 |                  0.967 |        0.267 |              486.8   |                 451.567 |             0.9   |
| minimax            | CK                   |        30 |                0.6 |                 0.967 |         36.667 |                 0.6 |                  0.967 |        0.367 |              464.567 |                 419.567 |             0.833 |

**Verdict Claim 1:** STRONG POSITIVE (oracle_win_rate_CK=0.83, best gain=36.7 pp)

## Claim 2: C1 vs CK Pass Fragility

| compressor_model   | generation_type   |   n_candidates |   count_robust_pass |   count_fragile_pass |   count_stress_improved |   count_robust_fail |   pass_rate_C1 |   pass_rate_CK |   stress_drop_pp |   fragility_rate |
|:-------------------|:------------------|---------------:|--------------------:|---------------------:|------------------------:|--------------------:|---------------:|---------------:|-----------------:|-----------------:|
| minimax            | greedy            |             30 |                  15 |                    6 |                       3 |                   6 |           0.7  |          0.6   |           10     |            0.286 |
| minimax            | sample            |            240 |                 122 |                   34 |                      29 |                  55 |           0.65 |          0.629 |            2.083 |            0.218 |

**Verdict Claim 2:** STRONG POSITIVE (max fragility_rate=0.29, max stress drop=10.0 pp)

## Claim 3: Chunk Information Advantage

| chunk_type   |   n_chunks |   mean_score_advantage |   mean_pass_advantage |   frac_positive_advantage |   frac_top_advantage |   contains_causal_relation_rate |   contains_exact_literals_rate |
|:-------------|-----------:|-----------------------:|----------------------:|--------------------------:|---------------------:|--------------------------------:|-------------------------------:|
| OTHER        |        144 |                  0.028 |                 0.028 |                     0.201 |                0.069 |                               0 |                              0 |


## What This Motivates for RL
- If Claim 1 + 2 pass: train compressor with reward = behavior_after_stress(T^K) - λ·length.
- If Claim 3 passes: per-chunk information advantage supports IAPO-style natural-language credit assignment.

## Negative / Null Results
(Filled in after manual review.)

## Files of record
- `tables/best_of_n_by_case.csv`
- `tables/best_of_n_summary.csv`
- `tables/reward_spread_by_case.csv`
- `tables/c1_ck_transition.csv`
- `tables/c1_ck_fragility_by_model.csv`
- `tables/stress_chain_convergence.csv`
- `tables/chunk_information_advantage.csv`
- `tables/chunk_advantage_by_type.csv`