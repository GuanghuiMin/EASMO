# Metrics for motivation_v7

This document precisely defines each metric the pipeline produces.
Tables are in `outputs/tables/`; computation lives in
`motivation_v7/metrics.py`. All bootstrap intervals are computed with
1000 resamples by default (500 for the lighter need-effect grouping)
and 95 % percentile CIs.

## A — Single-round metrics (Claim A)

### A.1 Need-effect Δ_need

For each `(model, prompt_variant, budget_chars, fact_type)`:

```
delta_need        = mean(retained_binary | need_label=1)
                  - mean(retained_binary | need_label=0)
delta_need_score  = mean(retention_score | need_label=1)
                  - mean(retention_score | need_label=0)
ci_low, ci_high   = difference-bootstrap percentile CI
```

Output: `tables/need_effect_by_type.csv`.

### A.2 Surface-dominance regression

Three logistic models fit per `(model, prompt_variant, budget_chars)`:

```
M_need:   retained_binary ~ need_label
M_type:   retained_binary ~ C(fact_type)         (one-hot, drop first)
M_both:   retained_binary ~ need_label + C(fact_type)
```

Reported per row:

```
n             # observations
r2_need       # McFadden pseudo R²
r2_type
r2_both
sdi           = (r2_type - r2_need) / (r2_type + r2_need + 1e-8)
coef_need_in_both
```

Output: `tables/surface_dominance_regression.csv`,
`tables/surface_dominance_index.csv`.

SDI sign:
- `> 0`  → fact type explains retention better than need;
- `≈ 0`  → comparable;
- `< 0`  → need explains retention better than fact type.

### A.3 Preference inversion rate (PIR)

Within each case, all `(needed-concrete, unneeded-narrative)` pairs
across all matched fact-condition rows:

```
inversion = retained(unneeded_narrative)==1 AND retained(needed_concrete)==0
PIR       = mean(inversion across pairs in (model, variant, budget))
```

`needed-concrete = {EXECUTABLE, CONTROL}` group, need_label=1.
`unneeded-narrative = NARRATIVE` group, need_label=0.

Output: `tables/preference_inversion.csv`.

### A.4 Condition Responsiveness Score (CRS)

Per matched fact pair:

```
CRS_f = retention_score(needed) - retention_score(unneeded)
```

Aggregated by `(model, prompt_variant, budget_chars, fact_type)`:

```
mean_crs        # signed CRS (positive ⇒ compressor responds to need)
median_crs
frac_positive   # share of facts where CRS > 0
```

Output: `tables/condition_responsiveness.csv`.

## B — Iterative metrics (Claim B)

### B.1 Survival curve

For each `(model, prompt_variant, budget_chars, round, fact_type)`:

```
survival_rate        = mean(retained_binary)
survival_score_mean  = mean(retention_score)
ci_low, ci_high      = bootstrap CI on retained_binary
```

Output: `tables/survival_by_round_type.csv`.

### B.2 Half-life

```
half_life = min{r : survival_rate(r) ≤ 0.5}
```

If no round drops below 0.5 within `ROUNDS=5`, we set
`half_life = ROUNDS+1` and flag `half_life_censored=True`.

Output: `tables/fact_type_half_life.csv`.

### B.3 Hazard

Round-to-round forgetting rate:

```
hazard(r) = 1 - S(r) / S(r-1)
hazard(1) = 1 - S(1)
```

Output: `tables/hazard_by_round_type.csv`.

### B.4 Area under survival curve

```
AUSC = Σ_{r=1..R} S(r)
```

Output: `tables/ausc_by_type.csv`.

### B.5 Hierarchy rank + stability

Rank fact types by half-life (descending) within each
`(model, prompt_variant, budget_chars)`. Tie-break by final survival
(descending).

Cross-model similarity:

```
Kendall τ, Spearman ρ  over fact-type rank vectors of model pairs
```

Outputs:
- `tables/hierarchy_rank_by_model.csv`
- `tables/cross_model_hierarchy_similarity.csv`

### B.6 Compression convergence (fixed-point)

Per chain (case, condition, model, prompt, budget), declare
convergence at round `r` if:

```
text_similarity(x_r, x_{r-1})       ≥ 0.95
fact_jaccard(retained_r, retained_{r-1}) ≥ 0.95
|len(x_r) - len(x_{r-1})| / len(x_{r-1}) ≤ 0.02
```

If never satisfied, `converged = False`, `convergence_round = -1`.

At the **final** round we also record per-group recall:

```
needed_fact_recall_at_convergence    = mean(retained_binary | need_label=1)
narrative_fact_recall_at_convergence = mean(retained_binary | coarse=NARRATIVE)
executable_fact_recall_at_convergence = mean(retained_binary | coarse=EXECUTABLE)
```

Output: `tables/convergence_by_case.csv`.

## C — Joint metrics & dashboard

`scripts/08_compute_metrics.py` writes all the above CSVs in a single
pass. `scripts/10_write_report.py` then composes
`outputs/reports/motivation_v7_results_summary.md` with the headline
verdict for each claim using the spec §18 thresholds.
