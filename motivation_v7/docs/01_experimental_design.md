# motivation_v7 — Experimental design

## 0. Framing (spec §0)

Two claims under test:

* **Claim A — Unconditioned compression preference.** For some fact
  types (especially concrete execution facts), the probability that
  a compressor retains the fact does not depend strongly on whether
  the downstream task needs it:
  \[
  \mathbb{E}[R_m(f) \mid \text{needed}=1] \approx
  \mathbb{E}[R_m(f) \mid \text{needed}=0]
  \]
  while retention is instead well predicted by the **surface type** of
  the fact.

* **Claim B — Stable iterative information-loss hierarchy.** Repeated
  application of the same compressor induces survival curves
  $S_{m,c}(r)$ that decay at type-specific rates, with a stable
  ranking across cases and partial stability across models.

The experiment must support either supporting or rejecting these.

## 1. Models

| role | model | endpoint |
|---|---|---|
| Compressor A | `qwen3-4b-instruct-2507` | local vLLM `http://127.0.0.1:8000/v1` |
| Compressor B | `MiniMaxAI/MiniMax-M2.5` | `http://10.183.22.68:8005/v1` |
| Fact-bank extractor | MiniMax-M2.5 (one call per case) |
| Need-condition generator | MiniMax-M2.5 (one call per fact) |
| Retention scorer | **cross-model**: Qwen-compressions scored by MiniMax, MiniMax-compressions scored by Qwen |

Temperature = 0.0, seed = 42 wherever supported.

## 2. Data (spec §5)

* `N_CASES = 30`: all v3-selected successful AppWorld dev trajectories
  (reused via `motivation_v3/outputs/motivation_full_trajectories.jsonl`).
* Length distribution: median 20 steps; **0 short / 24 medium / 6 long**.
  v3 produced no `<15`-step successes; spec's 1/3 stratification not
  achievable from v3 alone. Documented in the results summary.
* Each case carries `full_trajectory_text` (capped at 18,000 chars),
  structured `trajectory_steps`, and the v3 `user_instruction`.

## 3. Fact taxonomy (spec §6)

Fixed surface-type taxonomy of 16 labels grouped into five coarse
groups:

```
NARRATIVE   = NARRATIVE_GOAL, NARRATIVE_PROGRESS, HIGH_LEVEL_REASONING
TASK_STATE  = PENDING_SUBTASK, COMPLETED_SUBTASK, ENVIRONMENT_STATE,
              STALE_OR_OVERWRITTEN_STATE
EXECUTABLE  = RUNTIME_VARIABLE, AUTH_OR_ACCESS_TOKEN, EXACT_IDENTIFIER,
              FILE_PATH_OR_RESOURCE_LOCATOR, API_SCHEMA_OR_PARAMETER,
              ACTION_OUTCOME, NUMERIC_OR_DATE_LITERAL
CONTROL     = NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT
OTHER       = OTHER_CONCRETE_DETAIL
```

Per-case caps (after substring grounding + length ≤ 80 tokens):

```
NARRATIVE 3, TASK_STATE 3, EXECUTABLE 6, CONTROL 2, OTHER 1
```

Total ≤ 15 facts/case but typically 7–8 because cases lack control /
narrative facts of all types.

## 4. Fact-bank construction (spec §7)

* **Stage 1 (deterministic).** Regex over the rendered trajectory for
  API calls (`apis\.app\.api(...)`), keyword args, paths, IDs, tokens,
  dates, amounts, exception/error strings, state-changing verbs.
  Used as a diagnostic file (`fact_candidates_deterministic.jsonl`).
* **Stage 2 (LLM inventory).** One MiniMax call per case, JSON-only,
  with the verbatim `FACT_INVENTORY_PROMPT` (Appendix A of spec).
* **Grounding.** Each LLM fact must have a `source_quote` that
  substring-matches the trajectory (case-insensitive,
  whitespace-collapsed). Ungrounded facts are kept in
  `fact_bank_raw.jsonl` but excluded from `fact_bank_filtered.jsonl`.
* **Caps.** Within each coarse group, prefer `is_exact_literal=True`,
  then shorter facts, then earlier source steps.

## 5. Need-counterfactual conditions (spec §8)

Per fact, MiniMax-M2.5 generates a matched pair using
`NEED_CONDITION_PROMPT` (Appendix B). The two task instructions
share the same history but differ in downstream need.

**Quality checks (rule-based, no LLM):**

1. neither needed nor unneeded condition mentions the fact verbatim
   or any of its literal values;
2. token-length match within 35 %;
3. both conditions non-empty.

Primary analysis uses only condition pairs that pass all checks.

## 6. Compression input (spec §9)

```
[USER INSTRUCTION]    {condition_task}
[PREVIOUS SUMMARY]    ""                # primary mode, spec §10.4
[HISTORY OF INTERACTIONS]  {full_trajectory_text}     # capped to 18 K chars
```

Budget: `TARGET_MAX_CHARS = 1500` (only used by templates that
reference `{{ max_chars }}`; UT and UTCO do not, so the variable is a
no-op here — recorded in provenance).

## 7. Single-round experiment (spec §12)

Per `(case, fact, condition, model)` × UTCO:
- render ACON prompt;
- compress;
- store output.

Total = 30 × 8 × 2 × 2 = **960 compressions** (Plan B scope: UTCO only,
≤8 facts/case).

## 8. Iterative experiment (spec §13, Plan B reduced)

Per case, pick one representative EXECUTABLE fact (preferring
`is_exact_literal=True`) and run two iterative chains:

```
x_0 = full_trajectory_text
x_{r+1} = ACON(task=condition_task, history=x_r, prev_summary="", max_chars=1500)
for r in 1..5
```

`needed` chain + `unneeded` chain per case × 2 models = 4 chains/case
× 30 cases = 120 chains × 5 rounds = **600 iterative compressions**.

## 9. Retention scoring (spec §14)

Deterministic exact substring match (case-insensitive, whitespace-
collapsed) + LLM semantic scorer with `RETENTION_SCORER_PROMPT`
(Appendix C). When the deterministic match is exact, we skip the LLM
call for cost (the assigned label is `exact`).

Cross-model scorer: Qwen-compressions scored by MiniMax;
MiniMax-compressions scored by Qwen.

Primary binary retention:

```
retained_binary = exact_retained or retention_label in {exact, semantic}
```

Primary continuous retention:

```
retention_score = max(deterministic_exact_score, llm_retention_score)
```

## 10. Metrics

### Claim A
- $\Delta_{\text{need}}$ per fact type (spec §15.1) — bootstrap 95 % CIs.
- Surface dominance regression: McFadden $R^2$ for `M_need`, `M_type`,
  `M_both` (spec §15.2).
- Surface Dominance Index $\text{SDI} = (R^2_{type} - R^2_{need}) /
  (R^2_{type} + R^2_{need})$ (spec §15.3).
- Preference inversion rate `PIR` (spec §15.4) — unneeded-narrative
  retained while needed-concrete dropped, within case.
- Condition responsiveness score `CRS` (spec §15.5).

### Claim B
- Survival curve $S_c(r)$ (spec §16.1).
- Half-life $h_c$ (spec §16.2).
- Hazard $1 - S_c(r) / S_c(r-1)$ (spec §16.3).
- Area under survival curve (spec §16.4).
- Hierarchy rank stability — within-model bootstrap +
  cross-model Kendall $\tau$ and Spearman $\rho$ (spec §16.5).
- Compression fixed-point / convergence (spec §16.6) — text
  similarity ≥ 0.95 + fact Jaccard ≥ 0.95 + length change ≤ 2 %.

## 11. Figures (spec §17)

Seven figures, PNG + PDF:

1. `fig_need_effect_by_fact_type` — Δ_need per type, hue=model.
2. `fig_surface_dominance_index` — SDI per (model, prompt, budget).
3. `fig_preference_inversion_rate` — PIR per (model, prompt, budget).
4. `fig_iterative_survival_curves` — survival vs round, hue=type, facet=model.
5. `fig_survival_hierarchy_heatmap` — type × round heatmap per model.
6. `fig_cross_model_hierarchy_rank` — slope plot of rank across models.
7. `fig_fixed_point_recall` — needed / narrative / executable recall at convergence.

## 12. Verdicts (spec §18)

**Claim A is STRONG positive** if ≥3 of:

1. Mean Δ_need for EXECUTABLE < 0.15 while NARRATIVE has higher baseline retention.
2. SDI > 0.3 for at least one compressor.
3. PIR > 0.25.
4. Logistic regression — `fact_type` beats `need_label`.
5. ≥2 concrete fact types with similar retention across needed/unneeded.

**Claim B is STRONG positive** if ≥3 of:

1. Survival curves differ clearly by fact type.
2. Narrative half-life > executable / control half-life.
3. Cross-model Kendall $\tau$ > 0.4.
4. Convergence within 3–5 rounds for most cases.
5. Needed executable recall at convergence << narrative recall.

## 13. Scope deviations from spec

* Iterative compression uses **2 chains/case**, not the spec's
  per-condition_task sweep. This is documented as a Plan B reduction
  to fit ~4 h wallclock vs ~30 h.
* Only the **UTCO** prompt variant is run for primary; the **UT**
  ablation is deferred.
* Only the **1500-char** primary budget is run; secondary budgets
  {800, 2500} are deferred.
