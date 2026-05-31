# motivation_v10 results (auto-written)

> Auto-written by `scripts/12_write_report.py` at 2026-05-31 20:37Z.

> Honest hand-written counterpart: `docs/04_results_summary.md` (TBD).


## TL;DR (per spec §19 Go / No-go)

* **Claim 1 — proxy recovers best-of-N gain**: FAIL (proxy gain=-2.7 pp on CK, recovered=-0.11)
* **Claim 2 — stress-selected SFT targets better than one-step**: see §3 student eval table below for raw Qwen vs SFT-C1 vs SFT-CK comparison.
* **Claim 3 — Qwen-SFT-CK has GRPO-trainable reward spread**: PASS (oracle_win=0.81, std=0.467, all_low=0.00)
* **Claim 4 — chunk surface labels insufficient**: requires stage 11 (chunk reanalysis).

## §1 Proxy selection summary (stage 06)

| eval_round   |   n_cases |   greedy_pass |   random_sample_pass |   proxy_pass |   pairwise_pass |   oracle_pass |   proxy_gain_pp |   pairwise_gain_pp |   oracle_gain_pp |   recovered_gain_proxy |   recovered_gain_pairwise |   verifier_AUROC |   verifier_spearman_score |   mean_chars_greedy |   mean_chars_oracle |   mean_chars_proxy |   mean_chars_pairwise |
|:-------------|----------:|--------------:|---------------------:|-------------:|----------------:|--------------:|----------------:|-------------------:|-----------------:|-----------------------:|--------------------------:|-----------------:|--------------------------:|--------------------:|--------------------:|-------------------:|----------------------:|
| C1           |        75 |      0.586667 |             0.653333 |     0.64     |        0.706667 |      0.986667 |         5.33333 |                 12 |          40      |               0.133333 |                  0.3      |         0.557523 |                 0.122457  |             1743.33 |             1551.96 |            1728.28 |               1798.63 |
| CK           |        75 |      0.733333 |             0.72     |     0.706667 |        0.773333 |      0.986667 |        -2.66667 |                  4 |          25.3333 |              -0.105263 |                  0.157895 |         0.532593 |                 0.0397438 |             1679.59 |             1468.63 |            1682.12 |               1743.93 |

## §2 Per-case proxy detail (head)

| case_id   | split     | eval_round   |   n_candidates |   greedy_pass |   oracle_pass |   proxy_pass |   pairwise_pass |   random_pass |   greedy_score |   oracle_score |   proxy_score |   pairwise_score |   greedy_chars |   oracle_chars |   proxy_chars |   pairwise_chars | proxy_cid                     | oracle_cid                    | pairwise_cid                  |
|:----------|:----------|:-------------|---------------:|--------------:|--------------:|-------------:|----------------:|--------------:|---------------:|---------------:|--------------:|-----------------:|---------------:|---------------:|--------------:|-----------------:|:------------------------------|:------------------------------|:------------------------------|
| 23cf851_1 | dev_proxy | C1           |              9 |             1 |             1 |            0 |               1 |             1 |       0.960975 |       0.965575 |     -0.031725 |         0.958925 |           1561 |           1377 |          1269 |             1643 | 23cf851_1__minimax__sample_01 | 23cf851_1__minimax__sample_03 | 23cf851_1__minimax__sample_04 |
| 23cf851_1 | dev_proxy | CK           |              9 |             1 |             1 |            1 |               1 |             1 |       0.963825 |       0.97025  |      0.967675 |         0.95895  |           1447 |           1190 |          1293 |             1642 | 23cf851_1__minimax__sample_01 | 23cf851_1__minimax__sample_03 | 23cf851_1__minimax__sample_04 |
| 23cf851_2 | dev_proxy | C1           |              9 |             1 |             1 |            1 |               1 |             1 |       0.948525 |       0.970725 |      0.948525 |         0.970625 |           2059 |           1171 |          2059 |             1175 | 23cf851_2__minimax__greedy    | 23cf851_2__minimax__sample_05 | 23cf851_2__minimax__sample_02 |
| 23cf851_2 | dev_proxy | CK           |              9 |             1 |             1 |            1 |               1 |             1 |       0.9495   |       0.9647   |      0.96435  |         0.9647   |           2020 |           1412 |          1426 |             1412 | 23cf851_2__minimax__sample_00 | 23cf851_2__minimax__sample_02 | 23cf851_2__minimax__sample_02 |
| 23cf851_3 | dev_proxy | C1           |              9 |             1 |             1 |            1 |               1 |             1 |       0.961275 |       0.966575 |      0.966575 |         0.96295  |           1549 |           1337 |          1337 |             1482 | 23cf851_3__minimax__sample_00 | 23cf851_3__minimax__sample_00 | 23cf851_3__minimax__sample_01 |
| 23cf851_3 | dev_proxy | CK           |              9 |             1 |             1 |            1 |               1 |             1 |       0.95565  |       0.96145  |      0.96145  |         0.96145  |           1774 |           1542 |          1542 |             1542 | 23cf851_3__minimax__sample_01 | 23cf851_3__minimax__sample_01 | 23cf851_3__minimax__sample_01 |
| 396c5a2_3 | dev_proxy | C1           |              9 |             1 |             1 |            1 |               1 |             1 |       0.959225 |       0.96595  |      0.9654   |         0.9654   |           1631 |           1362 |          1384 |             1384 | 396c5a2_3__minimax__sample_00 | 396c5a2_3__minimax__sample_07 | 396c5a2_3__minimax__sample_00 |
| 396c5a2_3 | dev_proxy | CK           |              9 |             1 |             1 |            1 |               1 |             1 |       0.958175 |       0.9674   |      0.958175 |         0.963275 |           1673 |           1304 |          1673 |             1469 | 396c5a2_3__minimax__greedy    | 396c5a2_3__minimax__sample_07 | 396c5a2_3__minimax__sample_00 |
| 3ab5b8b_3 | dev_proxy | C1           |              9 |             0 |             1 |            0 |               0 |             0 |      -0.04275  |       0.95655  |     -0.037425 |        -0.04275  |           1710 |           1738 |          1497 |             1710 | 3ab5b8b_3__minimax__sample_05 | 3ab5b8b_3__minimax__sample_03 | 3ab5b8b_3__minimax__greedy    |
| 3ab5b8b_3 | dev_proxy | CK           |              9 |             1 |             1 |            0 |               1 |             0 |       0.950925 |       0.953725 |     -0.044775 |         0.950925 |           1963 |           1851 |          1791 |             1963 | 3ab5b8b_3__minimax__sample_01 | 3ab5b8b_3__minimax__sample_06 | 3ab5b8b_3__minimax__greedy    |
| 6bdbc26_1 | dev_proxy | C1           |              9 |             0 |             1 |            1 |               1 |             1 |      -0.02945  |       0.972675 |      0.96475  |         0.958825 |           1178 |           1093 |          1410 |             1647 | 6bdbc26_1__minimax__sample_00 | 6bdbc26_1__minimax__sample_05 | 6bdbc26_1__minimax__sample_07 |
| 6bdbc26_1 | dev_proxy | CK           |              9 |             1 |             1 |            1 |               1 |             1 |       0.967225 |       0.973325 |      0.967225 |         0.96395  |           1311 |           1067 |          1311 |             1442 | 6bdbc26_1__minimax__greedy    | 6bdbc26_1__minimax__sample_00 | 6bdbc26_1__minimax__sample_07 |
| 6bdbc26_2 | dev_proxy | C1           |              9 |             1 |             1 |            1 |               0 |             1 |       0.962325 |       0.967675 |      0.96665  |        -0.038675 |           1507 |           1293 |          1334 |             1547 | 6bdbc26_2__minimax__sample_04 | 6bdbc26_2__minimax__sample_06 | 6bdbc26_2__minimax__sample_02 |
| 6bdbc26_2 | dev_proxy | CK           |              9 |             1 |             1 |            1 |               1 |             1 |       0.962225 |       0.9679   |      0.96685  |         0.96135  |           1511 |           1284 |          1326 |             1546 | 6bdbc26_2__minimax__sample_01 | 6bdbc26_2__minimax__sample_06 | 6bdbc26_2__minimax__sample_02 |
| b119b1f_3 | dev_proxy | C1           |              9 |             1 |             1 |            1 |               1 |             0 |       0.961875 |       0.96455  |      0.957025 |         0.957025 |           1525 |           1418 |          1719 |             1719 | b119b1f_3__minimax__sample_02 | b119b1f_3__minimax__sample_05 | b119b1f_3__minimax__sample_02 |

## §3 GRPO readiness summary (stage 10)

| variant     |   n_cases |   mean_within_case_std |   oracle_win_rate_over_greedy |   all_low_rate |   all_high_rate |   mean_best_of_n_gain |   mean_greedy_score |
|:------------|----------:|-----------------------:|------------------------------:|---------------:|----------------:|----------------------:|--------------------:|
| Qwen-SFT-C1 |        42 |               0.464684 |                      0.809524 |              0 |        0.119048 |              0.565119 |            0.659167 |
| Qwen-SFT-CK |        42 |               0.466545 |                      0.809524 |              0 |        0.142857 |              0.600119 |            0.634524 |
| Raw-Qwen    |        42 |               0.422446 |                      0.833333 |              0 |        0.190476 |              0.459524 |            0.818929 |

## §4 C1-vs-CK fragility (stage 04 post-hoc)

*(no data — table missing or empty)*

## §5 Chunk advantage by type (stage 11c)

| chunk_type                |   n_chunks |   mean_score_advantage |   median_score_advantage |   mean_pass_advantage |   frac_positive_advantage |   contains_causal_relation_rate |   contains_runtime_binding_rate |   contains_negative_evidence_rate |
|:--------------------------|-----------:|-----------------------:|-------------------------:|----------------------:|--------------------------:|--------------------------------:|--------------------------------:|----------------------------------:|
| CAUSAL_PRECONDITION       |          6 |              0         |                        0 |             0         |                 0.166667  |                       0.833333  |                       0.333333  |                         0.333333  |
| CONTROL_NEGATIVE_EVIDENCE |          6 |              0         |                        0 |             0         |                 0.166667  |                       0.333333  |                       0.166667  |                         0.833333  |
| TASK_GOAL_OR_TODO         |          1 |              0         |                        0 |             0         |                 0         |                       0         |                       0         |                         0         |
| ACTION_OUTCOME            |        177 |             -0.0169492 |                        0 |            -0.0169492 |                 0.175141  |                       0.0112994 |                       0.101695  |                         0.0112994 |
| RUNTIME_BINDING           |         44 |             -0.0681818 |                        0 |            -0.0681818 |                 0.136364  |                       0         |                       1         |                         0         |
| ENTITY_LIST_ONLY          |         14 |             -0.0714286 |                        0 |            -0.0714286 |                 0.0714286 |                       0         |                       0.0714286 |                         0         |
| NARRATIVE_PROGRESS        |        139 |             -0.0863309 |                        0 |            -0.0863309 |                 0.129496  |                       0.0791367 |                       0.028777  |                         0.028777  |
| OTHER                     |         49 |             -0.163265  |                        0 |            -0.163265  |                 0.0204082 |                       0.0204082 |                       0         |                         0         |


## §5b Chunk advantage by functional_role_guess (stage 11c)

| functional_role_guess   |   n_chunks |   mean_score_advantage |   median_score_advantage |   mean_pass_advantage |   frac_positive_advantage |
|:------------------------|-----------:|-----------------------:|-------------------------:|----------------------:|--------------------------:|
| api_argument_binding    |         62 |             -0.0483871 |                      0   |            -0.0483871 |                 0.129032  |
| progress_summary        |        299 |             -0.0501672 |                      0   |            -0.0501672 |                 0.160535  |
| object_set_binding      |         10 |             -0.1       |                      0   |            -0.1       |                 0         |
| unknown                 |         56 |             -0.107143  |                      0   |            -0.107143  |                 0.0357143 |
| failure_prevention      |          7 |             -0.142857  |                      0   |            -0.142857  |                 0.142857  |
| task_restatement        |          2 |             -0.5       |                     -0.5 |            -0.5       |                 0         |


## §5c Claim 4 regression (label-only R² vs full R²)

| feature                      | kind       |   univariate_pearson |
|:-----------------------------|:-----------|---------------------:|
| chunk_chars                  | numeric    |           -0.0244203 |
| chunk_index                  | numeric    |           -0.0259392 |
| contains_exact_literals      | bool       |            0.0284808 |
| contains_entity_list_form    | bool       |            0.0457371 |
| contains_causal_relation     | bool       |            0.0616621 |
| contains_negative_evidence   | bool       |           -0.0281052 |
| contains_action_outcome      | bool       |            0.0395779 |
| contains_runtime_binding     | bool       |            0.0363384 |
| MULTIVARIATE_R2_FULL         | regression |            0.0369157 |
| MULTIVARIATE_R2_LABELS_ONLY  | regression |            0.0190046 |
| MULTIVARIATE_R2_NUMERIC_ONLY | regression |            0.0171235 |


## Files of record

* `outputs/raw/v10_baseline_runs.jsonl`
* `outputs/raw/minimax_candidates.jsonl`
* `outputs/raw/stress_chains.jsonl`
* `outputs/raw/behavior_runs_candidates.jsonl`
* `outputs/raw/proxy_verifier_scores.jsonl`
* `outputs/raw/proxy_pairwise_scores.jsonl`
* `outputs/raw/student_compressions.jsonl`
* `outputs/raw/student_behavior_runs.jsonl`
* `outputs/raw/grpo_readiness_*.jsonl`
* `outputs/raw/chunks.jsonl`
* `outputs/raw/chunk_type_labels.jsonl`
* `outputs/raw/chunk_information_advantage.csv`
* `outputs/data/sft_targets_c1.jsonl`
* `outputs/data/sft_targets_ck.jsonl`
* `outputs/models/qwen_sft_c1/`
* `outputs/models/qwen_sft_ck/`
