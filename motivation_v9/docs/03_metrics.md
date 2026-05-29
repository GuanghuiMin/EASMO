# motivation_v9 metrics

All metric implementations live in `motivation_v9/metrics.py`. Tables
are saved to `outputs/tables/`.

## 1. Reward (spec §5.1)

```python
def compute_reward(score, compressed_tokens_est, lambda_len=0.02):
    return score - lambda_len * (compressed_tokens_est / 1000)
```

`score` is the AppWorld final reward (binary 0 or 1 for most tasks).
`compressed_tokens_est = max(1, len(compressed_text) // 4)`.

## 2. Best-of-N (spec §5.2)

For each `(case_id, compressor_model, eval_context_round)`:

```
greedy_reward      = R(greedy)
best_sample_reward = max_i R(sample_i)
best_of_n_reward   = max(greedy_reward, best_sample_reward)
best_of_n_gain     = best_of_n_reward - greedy_reward
oracle_win         = best_sample_reward > greedy_reward
```

Tables:

* `best_of_n_by_case.csv`
* `best_of_n_summary.csv` (aggregated by model × eval_round)
* `reward_spread_by_case.csv` (mean/std/min/max across N samples)

## 3. C1 vs CK fragility (spec §6)

Per candidate, classify (C1 pass, CK pass):

| class | C1 pass | CK pass |
|---|:---:|:---:|
| `robust_pass` | 1 | 1 |
| `fragile_pass` | 1 | 0 |
| `stress_improved` | 0 | 1 |
| `robust_fail` | 0 | 0 |

Aggregate per (model, generation_type):

```
fragility_rate = count(fragile_pass) / max(count(C1 pass), 1)
stress_drop_pp = (pass_rate_C1 - pass_rate_CK) * 100
```

Tables:

* `c1_ck_transition.csv` (per candidate)
* `c1_ck_fragility_by_model.csv` (per model, gen_type)

## 4. Chunk information advantage (spec §10)

Per (candidate, chunk):

```
chunk_score_advantage = score_full - score_minus_chunk
chunk_pass_advantage  = int(success_full) - int(success_minus_chunk)
```

Normalize within candidate:

```
positive_adv = max(chunk_score_advantage, 0)
chunk_adv_norm = positive_adv / sum_positive_adv_for_candidate
```

If a candidate's full-context CK fails (`not_interpretable_due_to_full_fail=True`),
the row is excluded from "top-advantage" aggregation.

Tables:

* `chunk_information_advantage.csv` (per chunk)

## 5. Chunk advantage by type (spec §12)

Per chunk_type:

```
mean_score_advantage
mean_pass_advantage
frac_positive_advantage
frac_top_advantage   # chunks with chunk_adv_norm >= 0.25 OR top-1 per candidate
contains_causal_relation_rate
contains_exact_literals_rate
```

Tables:

* `chunk_advantage_by_type.csv`

## 6. Convergence (stress chain)

Per (candidate, round transition):

```
converged_binary = (text_sim >= 0.95) AND (|delta_len| / len_prev <= 0.02)
convergence_round = first round satisfying both
```

Tables:

* `stress_chain_convergence.csv`

## 7. Budget compliance (light)

Reported in the auto-report header:

* median / p90 of `compressed_chars` per (model, generation_type)
* fraction exceeding `TARGET_MAX_CHARS × 1.10` = 1650 chars

## 8. Success thresholds (spec §14)

| claim | strong-positive condition |
|---|---|
| 1 | `best_of_N_CK_pass_rate - greedy_CK_pass_rate ≥ 10 pp` OR `oracle_win_rate_CK ≥ 25%` |
| 2 | `fragility_rate ≥ 20%` OR `pass_rate_C1 - pass_rate_CK ≥ 10 pp` |
| 3 | causal/control/action chunks have higher mean advantage than entity-only chunks, OR ≥ 40 % of top-advantage chunks contain causal relations |

Verdicts are computed automatically in stage 14 and re-checked
manually in `docs/04_results_summary.md` after the pipeline finishes.
