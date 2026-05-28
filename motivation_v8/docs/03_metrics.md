# motivation_v8 metric definitions

All metric implementations live in `motivation_v8/metrics.py`.
Bootstrap CIs use 500 resamples by default. Tables land in
`outputs/tables/`.

## A. Single-round metrics (Stage 07)

### A.1 `delta_need` per fact type (`need_effect_by_type.csv`)

For each `(model, prompt_family, budget_chars, fact_type)`:

```
delta_need        = mean(retained_binary | need_label=1)
                  - mean(retained_binary | need_label=0)
delta_need_score  = mean(retention_score | need_label=1)
                  - mean(retention_score | need_label=0)
ci_low, ci_high   = difference-bootstrap percentile CI
```

### A.2 Surface dominance regression
(`surface_dominance_regression.csv`, `surface_dominance_index.csv`)

Three logistic models per `(model, prompt_family, budget_chars)`:

```
M_need   : retained_binary ~ need_label
M_type   : retained_binary ~ C(fact_type)
M_both   : retained_binary ~ need_label + C(fact_type)
```

Reported:

```
n             # observations
r2_need       # McFadden pseudo R²
r2_type
r2_both
sdi = (r2_type - r2_need) / (r2_type + r2_need + 1e-8)
coef_need_in_both
```

### A.3 Preference Inversion Rate (`preference_inversion.csv`)

Within case, all `(needed-concrete, unneeded-narrative)` pairs:

```
PIR = mean(retained(unneeded_narrative)==1 AND retained(needed_concrete)==0)
```

### A.4 Condition Responsiveness Score (`condition_responsiveness.csv`)

```
CRS_f = retention_score(needed) - retention_score(unneeded)
mean_crs, median_crs, frac_positive
```

## B. Iterative metrics (Stage 07)

### B.1 Survival curve (`survival_by_round_type.csv`)

For each `(model, prompt_family, init_type, condition_type, round, fact_type)`:

```
survival_rate        = mean(retained_binary)
survival_score_mean  = mean(retention_score)
```

### B.2 Half-life (`fact_type_half_life.csv`)

```
half_life = min{r : S(r) ≤ 0.5}
```

Censored at `ROUNDS + 1` when never crossed.

### B.3 Hazard (`hazard_by_round_type.csv`)

```
hazard[1] = 1 - S[1]
hazard[r] = 1 - S[r] / max(S[r-1], 1e-8)
```

### B.4 AUSC (`ausc_by_type.csv`)

```
AUSC = Σ_{r=1..R} S[r]
```

### B.5 Hierarchy rank (`hierarchy_rank_by_model_prompt.csv`)

Rank fact types by `half_life desc`, tie-break by `final_survival desc`.

### B.6 Cross-(model, prompt) similarity
(`cross_model_prompt_hierarchy_similarity.csv`)

Pairwise Kendall τ and Spearman ρ over rank vectors of fact types.

## C. Fixed-point metrics (Stage 07)

### C.1 Convergence (`convergence_by_chain.csv`)

For each chain, declare convergence at round r if all of:

```
text_similarity(x_r, x_{r-1})   ≥ 0.95
fact_jaccard(retained_r, retained_{r-1}) ≥ 0.95
|len(x_r) - len(x_{r-1})| / len(x_{r-1}) ≤ 0.02
```

Text similarity uses `difflib.SequenceMatcher` ratio. Fact Jaccard
uses retained-fact-id sets (case-level, not just target fact).

Per-chain output also records group recall at the fixed/final
round:

```
needed_fact_recall_fixed
narrative_fact_recall_fixed
executable_fact_recall_fixed
control_fact_recall_fixed
```

### C.2 Fixed-point composition (`fixed_point_composition_by_type.csv`)

For each `(model, prompt_family, init_type, condition_type, fact_type)`:

```
survival_rate_fixed   = mean(retained_binary at fixed_round)
survival_score_fixed  = mean(retention_score at fixed_round)
```

### C.3 Fixed-point need shift (`fixed_point_need_shift.csv`)

For each `(model, prompt_family, fact_type)`:

```
delta_need_infty = mean(retained_binary_fixed | need_label=1)
                 - mean(retained_binary_fixed | need_label=0)
```

## D. Basin-of-attraction metrics (Stage 07)

### D.1 Pairwise distances (`basin_similarity.csv`)

For each `(case_id, model, prompt_family, condition_type)` with
multiple init_types:

```
init_fact_jaccard_distance  = 1 - J(retained_0_a, retained_0_b)
init_type_l1_distance       = L1(hist_a, hist_b)         # at round 0
init_text_distance          = 1 - token_jaccard(text_0_a, text_0_b)

fin_fact_jaccard_distance   = 1 - J(retained_fixed_a, retained_fixed_b)
fin_type_l1_distance        = ...
fin_text_distance           = ...

contraction_fact_jaccard    = fin_fact_jaccard_distance / max(init_fact_jaccard_distance, 1e-8)
contraction_type_l1         = fin_type_l1_distance / max(init_type_l1_distance, 1e-8)
contraction_text            = ...
```

### D.2 Summary (`basin_contraction.csv`)

Mean contraction per (model, prompt_family, init_pair).

## E. Budget compliance (Stage 07)

```
budget_violation = compressed_chars > budget_chars * 1.10
violation_rate
median_length, p90_length, p99_length
```

Tables: `budget_compliance_single_round.csv`, `budget_compliance_iterative.csv`.
