# motivation_v7 — Abstraction Prior and Iterative Compression Dynamics

> Track: `EASMO/motivation_v7/`  
> Goal: diagnostic motivation only. No DPO, no SFT, no RL, no prompt optimization.  
> Primary benchmark: AppWorld.  
> Primary compressor prompts: original ACON history-compression prompts copied from the official `microsoft/acon` repository, not re-written locally.

---

## 0. One-paragraph framing

This experiment tests whether LLM-based history compressors have an **unconditioned abstraction preference**: when compressing agent trajectories, do they retain facts according to their surface type — narrative/progress facts versus concrete execution details — rather than according to whether those facts are needed for the future task?

We also test whether repeated compression behaves like an **iterative abstraction process** with a stable information-loss hierarchy. In other words, if we repeatedly apply the same ACON-style compressor,

\[
x_{r+1}=C_m(x_r; q, B),
\]

do different fact types decay at stable rates, and are these survival hierarchies similar across compressor models?

The desired output is a set of motivation figures and tables showing either:

1. **Positive evidence:** fact retention is weakly conditioned on downstream need and strongly predicted by fact surface type; repeated compression produces a stable hierarchy of information loss.
2. **Negative evidence:** compressors do respond strongly to downstream need, or the hierarchy is unstable/noisy; in that case the abstraction-prior story should be abandoned or revised.

---

## 1. Claims under test

### Claim A — Unconditioned compression preference

For a fact \(f\), condition \(q\), compressor model \(m\), and compression budget \(B\), define:

\[
R_m(f \mid x, q, B)=\mathbb{1}[f\ \text{survives in}\ C_m(x;q,B)].
\]

A need-conditioned compressor should satisfy:

\[
\mathbb{E}[R_m(f) \mid \mathrm{needed}(f,q)=1]
\gg
\mathbb{E}[R_m(f) \mid \mathrm{needed}(f,q)=0].
\]

The unconditioned-abstraction-prior hypothesis predicts a weaker need effect:

\[
\mathbb{E}[R_m(f) \mid \mathrm{needed}=1]
\approx
\mathbb{E}[R_m(f) \mid \mathrm{needed}=0]
\]

for certain fact types, especially concrete execution details, while retention is instead dominated by fact surface type:

\[
P(\mathrm{retain}\ f) \approx P(\mathrm{retain}\ f \mid \mathrm{fact\_type}(f)).
\]

### Claim B — Stable iterative information-loss hierarchy

Repeated compression should induce survival curves:

\[
S_{m,c}(r)=\Pr[f\ \text{survives after round}\ r \mid \mathrm{type}(f)=c, m],
\]

where \(c\) is a fact type such as `NARRATIVE_PROGRESS`, `RUNTIME_VARIABLE`, `API_SCHEMA`, or `NEGATIVE_EVIDENCE`.

The hierarchy hypothesis predicts:

1. different fact types have different survival half-lives;
2. the ranking by half-life is stable across cases;
3. the ranking is at least partially stable across models;
4. needed concrete facts may still decay faster than unneeded narrative facts unless the compressor is strongly conditioned.

---

## 2. What this experiment is NOT

Do **not** implement any method improvement here.

Do **not** train a compressor.

Do **not** do DPO/SFT/RL.

Do **not** optimize prompts.

Do **not** use previous diagnostic outputs, prior audit labels, or earlier version-specific prompts as ground truth.

Do **not** write a new ACON-style prompt by hand. The ACON compressor prompt must be loaded from the official ACON repository or, if unavailable, the official paper appendix prompt text. Fail loudly if neither source is available.

---

## 3. Main experimental questions

### Q1. Need conditioning

For the same input context and the same fact, does retention increase when the task condition explicitly makes that fact needed?

Primary metric:

\[
\Delta_{need}(c,m)=
\mathbb{E}[R_m(f)\mid \mathrm{type}=c,\mathrm{needed}=1]
-
\mathbb{E}[R_m(f)\mid \mathrm{type}=c,\mathrm{needed}=0].
\]

A small \(\Delta_{need}\) indicates weak conditioning.

### Q2. Surface-type dominance

Is retention better predicted by fact type than by need label?

Primary metrics:

- logistic-regression coefficient for `needed` after controlling for `fact_type`;
- McFadden \(R^2\) of three models:
  - `retain ~ needed`
  - `retain ~ fact_type`
  - `retain ~ needed + fact_type + needed:fact_type`
- bootstrap confidence intervals for each.

### Q3. Preference inversion

When budget forces a choice between an unneeded narrative fact and a needed concrete execution fact, how often does the compressor keep the narrative fact and drop the needed concrete fact?

\[
\mathrm{PIR}=\Pr[\mathrm{keep}(f_{narrative,unneeded})=1 \land \mathrm{keep}(f_{concrete,needed})=0].
\]

### Q4. Iterative hierarchy

Under repeated compression, which fact types disappear first?

Primary metrics:

- survival curve \(S_c(r)\);
- half-life round \(h_c = \min\{r: S_c(r) \leq 0.5\}\);
- area under survival curve `AUSC`;
- hazard rate between rounds;
- rank ordering of fact types by half-life.

### Q5. Cross-model stability

Are survival hierarchies similar across compressor models?

Primary metric:

- Kendall \(\tau\) and Spearman \(\rho\) between fact-type rankings across models;
- bootstrap confidence intervals over cases.

---

## 4. Models

### Required compressor models

| Role | Model | Purpose |
|---|---|---|
| Compressor A | `Qwen3-4B-Instruct` or local `qwen3-4b` | open/white-box-ish small compressor |
| Compressor B | `MiniMax-M2.5` local/shared endpoint | stronger compressor and cross-model comparison |

### Optional compressors

Add only if endpoints are already available and budget allows:

| Role | Model | Purpose |
|---|---|---|
| Compressor C | `gpt-4.1-mini` or equivalent | compare against ACON paper setup |
| Compressor D | stronger closed model | robustness check |

### Evaluator model

Use a different model from the compressor being evaluated whenever possible. If scoring Qwen compressions, use MiniMax as evaluator. If scoring MiniMax compressions, use Qwen + rule-based exact matching, and optionally MiniMax self-check only as auxiliary.

All evaluator calls must be temperature `0.0` and JSON-only.

---

## 5. Data source and case pool

### Source trajectories

Use successful full-context AppWorld trajectories. Accept either:

1. existing successful AppWorld dev trajectories already stored in the workspace, or
2. freshly generated full-context successful trajectories from AppWorld dev.

Each case must contain:

```json
{
  "case_id": "string",
  "task_id": "string",
  "task_name": "string | null",
  "user_instruction": "string",
  "full_trajectory_text": "string",
  "trajectory_steps": [
    {
      "step_id": 0,
      "thought": "string | null",
      "action": "string",
      "observation": "string"
    }
  ],
  "success": true,
  "num_steps": 0,
  "apps_used": ["spotify", "file_system", "..."]
}
```

### Sample size

Default:

- `N_CASES = 60` successful AppWorld dev trajectories.
- Minimum acceptable for initial run: `N_CASES = 30`.
- Prefer length-stratified sampling:
  - 1/3 short: 8–14 steps
  - 1/3 medium: 15–24 steps
  - 1/3 long: 25+ steps

Do not filter to only failed compression cases. This experiment is about the compressor's general retention dynamics, not just known failures.

---

## 6. Fact taxonomy

Use the following fixed fact types. The taxonomy is intentionally surface-type based because the hypothesis is that compressors prefer certain surface types independent of downstream need.

```text
NARRATIVE_GOAL
NARRATIVE_PROGRESS
HIGH_LEVEL_REASONING
PENDING_SUBTASK
COMPLETED_SUBTASK
RUNTIME_VARIABLE
AUTH_OR_ACCESS_TOKEN
EXACT_IDENTIFIER
FILE_PATH_OR_RESOURCE_LOCATOR
API_SCHEMA_OR_PARAMETER
ACTION_OUTCOME
ENVIRONMENT_STATE
NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT
STALE_OR_OVERWRITTEN_STATE
NUMERIC_OR_DATE_LITERAL
OTHER_CONCRETE_DETAIL
```

### Type groups

For aggregate analysis, also map each type to a coarse group:

```text
NARRATIVE = {NARRATIVE_GOAL, NARRATIVE_PROGRESS, HIGH_LEVEL_REASONING}
TASK_STATE = {PENDING_SUBTASK, COMPLETED_SUBTASK, ENVIRONMENT_STATE, STALE_OR_OVERWRITTEN_STATE}
EXECUTABLE = {RUNTIME_VARIABLE, AUTH_OR_ACCESS_TOKEN, EXACT_IDENTIFIER, FILE_PATH_OR_RESOURCE_LOCATOR, API_SCHEMA_OR_PARAMETER, ACTION_OUTCOME, NUMERIC_OR_DATE_LITERAL}
CONTROL = {NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT}
OTHER = {OTHER_CONCRETE_DETAIL}
```

---

## 7. Fact-bank construction

### Stage 1 — deterministic candidates

Before using any LLM extraction, run deterministic extractors over the trajectory text and step objects.

Extract:

- API calls: regex `apis\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\([^\n]*\)`
- API names: `app_name.api_name`
- keyword args in calls: `key=value`
- paths: strings starting with `/`, `~/`, or containing common file extensions
- IDs: JSON fields ending in `_id`, `id`, `message_id`, `playlist_id`, `song_id`, etc.
- tokens: long alphanumeric strings, JWT-like strings, values named `access_token`
- dates and amounts
- exception/error messages
- state-changing verbs: delete, update, create, send, accept, like, unlike, move, copy, login, logout

Write candidates to:

```text
data/fact_candidates_deterministic.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "fact_id": "caseid.fact0001",
  "candidate_text": "...",
  "source_step_ids": [3, 4],
  "source_span": "short verbatim quote",
  "deterministic_type_hint": "API_SCHEMA_OR_PARAMETER | ...",
  "literal_keys": ["access_token", "file_path"],
  "literal_values": ["..."],
  "extraction_method": "regex_api_call | json_field | error_message | ..."
}
```

### Stage 2 — LLM fact inventory

Use the `FACT_INVENTORY_PROMPT` in Appendix A to extract and normalize facts. The LLM may merge duplicates, assign types, and add narrative facts that deterministic extractors miss.

Important constraints:

- Every extracted fact must cite a verbatim source quote from the full trajectory.
- The quote must be checked by deterministic substring match.
- Facts without source quote grounding must be excluded from the primary analysis and stored only in a diagnostic file.

Output:

```text
data/fact_bank_raw.jsonl
outputs/tables/fact_bank_grounding.csv
```

Primary fact schema:

```json
{
  "case_id": "...",
  "fact_id": "caseid.fact0001",
  "fact_type": "API_SCHEMA_OR_PARAMETER",
  "coarse_group": "EXECUTABLE",
  "canonical_fact": "delete_file requires file_path and access_token",
  "verbatim_surface": "access_token obtained from file_system app login",
  "source_step_ids": [12],
  "source_quote": "...",
  "is_exact_literal": true,
  "literal_values": ["access_token", "file_path"],
  "grounded_by_substring": true,
  "length_tokens": 12,
  "notes": "..."
}
```

### Stage 3 — fact filtering

Primary analysis uses only facts satisfying:

```python
grounded_by_substring == True
fact_type in TAXONOMY
length_tokens <= 80
```

Per case, keep at most:

- 3 narrative facts
- 3 task-state facts
- 6 executable facts
- 2 control facts

If a case has too few facts in a group, keep what exists and record missingness.

---

## 8. Need-condition construction

The experiment requires **matched counterfactual conditions**. For the same context and same fact, create at least two task conditions:

1. `NEEDED`: future continuation requires the fact.
2. `UNNEEDED`: future continuation does not require the fact.

The only thing that changes between conditions is the `task` field passed to the ACON history-compression prompt. The history/context input must be identical.

### Need labels

For each fact \(f\) in the fact bank, create:

```json
{
  "case_id": "...",
  "fact_id": "...",
  "condition_id": "caseid.fact0001.needed",
  "need_label": 1,
  "condition_task": "...",
  "condition_rationale": "why this task makes the fact needed",
  "matched_condition_id": "caseid.fact0001.unneeded"
}
```

and a matched unneeded condition:

```json
{
  "case_id": "...",
  "fact_id": "...",
  "condition_id": "caseid.fact0001.unneeded",
  "need_label": 0,
  "condition_task": "...",
  "condition_rationale": "why this task does not need the fact",
  "matched_condition_id": "caseid.fact0001.needed"
}
```

### How to generate conditions

Use `NEED_CONDITION_PROMPT` in Appendix B.

The condition generator must obey:

- Do not change the history.
- Do not invent new APIs or entities.
- For `NEEDED`, make the target fact necessary for future continuation.
- For `UNNEEDED`, make the task solvable without the target fact while keeping the context realistic.
- Keep both condition instructions similar in length.
- Avoid wording like “remember this exact fact” because that would trivially force retention. The condition should imply the need through the downstream task.

### Need-condition quality checks

For each generated pair, run a validator prompt and rule checks:

1. `needed_condition_mentions_fact = False` unless the fact is naturally part of the task instruction. We do not want trivial copying.
2. `unneeded_condition_mentions_fact = False`.
3. The needed and unneeded task instructions differ by no more than 35% in token length.
4. The condition generator provides a structured rationale.

Store:

```text
data/need_conditions_raw.jsonl
outputs/tables/need_condition_quality.csv
```

Primary analysis uses only condition pairs passing quality checks.

---

## 9. Compression inputs

For each case, construct a compression input packet:

```text
[USER INSTRUCTION]
{condition_task}

[PREVIOUS SUMMARY]
{empty string}

[HISTORY OF INTERACTIONS]
{full_trajectory_text or selected prefix}
```

### History length control

Use the same history for all matched conditions for a given case.

Default use `full_trajectory_text` capped at 18k characters. If the trajectory exceeds the model context limit, use the first 70% + last 30% of trajectory steps, preserving chronological order. Record any truncation.

### Budget control

Use exactly the same compression budget across matched conditions.

Primary budget:

```text
TARGET_MAX_CHARS = 1500
```

If the original ACON prompt includes a `max_chars` variable, pass `max_chars=1500`. If the prompt has no budget variable, add a wrapper instruction outside the ACON prompt only if the original repo runner does so. Otherwise do not modify the prompt. Record which mode was used.

Secondary budgets for robustness:

```text
TARGET_MAX_CHARS in {800, 1500, 2500}
```

Run secondary budgets only after primary pipeline works.

---

## 10. ACON prompt provenance and strict loading

### Required

Create script:

```text
scripts/00_sync_acon_prompts.py
```

This script must:

1. Clone or locate the official ACON repository:

```bash
git clone https://github.com/microsoft/acon.git external/acon_official
```

2. Record commit hash:

```bash
git -C external/acon_official rev-parse HEAD > outputs/provenance/acon_commit.txt
```

3. Locate official AppWorld history-compression prompt templates by grepping for:

```text
HISTORY_SUMMARY
REASONING
VARS
TODO
COMPLETED
GUARDRAILS
```

4. Copy the exact prompt template files into:

```text
prompts/acon_history_ut_original.md
prompts/acon_history_utco_original.md
```

5. Compute SHA256:

```text
outputs/provenance/acon_prompt_sha256.json
```

6. Fail loudly if the prompts cannot be found.

### Fallback

Only if the repository does not contain the prompt text, extract the official AppWorld history-compression prompts from the ACON paper appendix prompts corresponding to:

- AppWorld history compression after utility optimization (`UT`)
- AppWorld history compression after utility + compression optimization (`UTCO`)

The fallback extraction must also write provenance:

```json
{
  "source": "ACON paper appendix",
  "prompt_ids": ["E.6", "E.7"],
  "extraction_method": "manual_or_pdf_parse",
  "sha256": "..."
}
```

### Primary prompt variant

Use `acon_history_utco_original.md` as the primary compressor because it is the aggressive compression setting and should expose iterative information-loss hierarchy most clearly.

Use `acon_history_ut_original.md` as an ablation.

### Rendering convention

The compressor must be called with exactly these variables:

```python
rendered_prompt = template.render(
    task=condition_task,
    prev_summary="",
    history=current_context,
    max_chars=TARGET_MAX_CHARS,
)
```

For iterative compression round `r`:

```python
x_0 = initial_context
x_{r+1} = ACON_COMPRESS(task=condition_task, prev_summary="", history=x_r, max_chars=TARGET_MAX_CHARS)
```

Do not pass previous compressed output through `prev_summary` in the primary run. We are studying the compressor as a map from an arbitrary current context to a compressed context, so the current context should always be in `history`.

Optional ablation:

```python
x_{r+1} = ACON_COMPRESS(task=condition_task, prev_summary=x_r, history="", max_chars=TARGET_MAX_CHARS)
```

but do not mix this with the primary results.

---

## 11. Generation settings

Use the official ACON repo config if available. If config is absent, use:

```yaml
temperature: 0.0
top_p: 1.0
seed: 42
max_tokens: 2048
```

Rationale: we want deterministic compression dynamics. If MiniMax endpoint does not support seed, record this.

Run three repeated seeds only for robustness after the main run:

```text
seed in {1, 2, 3}
```

Primary results should use the deterministic run.

---

## 12. Single-round need-conditioning experiment

### Stage

```text
04_run_need_conditioned_compression.py
```

For every `(case, fact, condition, model, prompt_variant, budget)`:

1. render original ACON prompt;
2. call compressor;
3. save raw output.

Output:

```text
outputs/raw/single_round_compressions.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "fact_id": "...",
  "condition_id": "...",
  "need_label": 1,
  "fact_type": "API_SCHEMA_OR_PARAMETER",
  "coarse_group": "EXECUTABLE",
  "compressor_model": "qwen3-4b",
  "prompt_variant": "UTCO",
  "budget_chars": 1500,
  "round": 1,
  "input_chars": 13210,
  "output_chars": 1422,
  "output_tokens_est": 356,
  "compressed_text": "...",
  "raw_response": "...",
  "error": null
}
```

---

## 13. Iterative compression experiment

### Stage

```text
06_run_iterative_compression.py
```

For every `(case, condition_task, model, prompt_variant, budget)`:

Run:

```python
x = initial_context
for r in range(1, ROUNDS + 1):
    x = compressor(task=condition_task, history=x, prev_summary="", max_chars=budget)
    save(x, r)
```

Default:

```text
ROUNDS = 5
```

If output becomes empty or invalid, continue with the empty/invalid text but mark `invalid_output=True`.

Output:

```text
outputs/raw/iterative_compressions.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "condition_id": "...",
  "need_label_map_available": true,
  "compressor_model": "minimax-m2.5",
  "prompt_variant": "UTCO",
  "budget_chars": 1500,
  "round": 3,
  "input_chars": 1510,
  "output_chars": 1280,
  "compressed_text": "...",
  "text_sha256": "...",
  "invalid_output": false,
  "error": null
}
```

---

## 14. Retention scoring

Retention scoring must combine deterministic checks and LLM checks.

### Deterministic exact retention

For each fact and compressed text:

- exact substring match for `source_quote`, `verbatim_surface`, and `literal_values`;
- normalized match for punctuation/case;
- ID/path/token match;
- API name and parameter match.

Produce:

```python
exact_retained = any(exact_match_conditions)
```

### LLM semantic retention

Use `RETENTION_SCORER_PROMPT` in Appendix C.

The scorer must output:

```json
{
  "fact_id": "...",
  "retention_label": "exact | semantic | partial | absent | contradicted",
  "retention_score": 1.0,
  "evidence_in_compressed_text": "verbatim quote or empty",
  "is_distorted": false,
  "confidence": 0.0
}
```

Map labels to score:

| label | score |
|---|---:|
| exact | 1.0 |
| semantic | 0.75 |
| partial | 0.4 |
| absent | 0.0 |
| contradicted | -0.5 |

Primary binary retention:

```python
retained_binary = exact_retained or retention_label in {"exact", "semantic"}
```

Primary continuous retention:

```python
retention_score = max(deterministic_exact_score, llm_retention_score)
```

where deterministic exact score is 1.0 if exact retained else 0.0.

### Output

```text
outputs/raw/fact_retention_scores.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "fact_id": "...",
  "condition_id": "...",
  "need_label": 1,
  "fact_type": "ACTION_OUTCOME",
  "coarse_group": "EXECUTABLE",
  "compressor_model": "qwen3-4b",
  "prompt_variant": "UTCO",
  "budget_chars": 1500,
  "round": 1,
  "exact_retained": false,
  "retention_label": "absent",
  "retention_score": 0.0,
  "retained_binary": false,
  "evidence_in_compressed_text": "",
  "is_distorted": false,
  "confidence": 0.91,
  "scorer_model": "minimax-m2.5",
  "scorer_error": null
}
```

---

## 15. Metrics — unconditioned preference

### 15.1 Need effect by fact type

For each `(model, prompt_variant, budget, fact_type)`:

```python
delta_need = mean(retained_binary | need_label=1) - mean(retained_binary | need_label=0)
delta_need_score = mean(retention_score | need_label=1) - mean(retention_score | need_label=0)
```

Bootstrap 95% CI over `(case_id, fact_id)` pairs.

Output:

```text
outputs/tables/need_effect_by_type.csv
```

Columns:

```text
compressor_model,prompt_variant,budget_chars,fact_type,coarse_group,
n_needed,n_unneeded,retain_needed,retain_unneeded,delta_need,
score_needed,score_unneeded,delta_need_score,ci_low,ci_high
```

### 15.2 Surface dominance regression

Fit logistic regressions for each `(model, prompt_variant, budget)`:

```text
M0: retained_binary ~ 1
M_need: retained_binary ~ need_label
M_type: retained_binary ~ C(fact_type)
M_both: retained_binary ~ need_label + C(fact_type)
M_interact: retained_binary ~ need_label * C(fact_type)
```

Also fit controls:

```text
M_controls: retained_binary ~ need_label + C(fact_type) + log(length_tokens+1) + C(case_length_bucket)
```

Report:

- coefficient for `need_label`;
- odds ratio for `need_label`;
- McFadden \(R^2\);
- likelihood-ratio test improvements where available;
- bootstrap CI over cases.

Output:

```text
outputs/tables/surface_dominance_regression.csv
```

### 15.3 Surface Dominance Index

Define:

\[
\mathrm{SDI} = \frac{R^2_{type} - R^2_{need}}{R^2_{type}+R^2_{need}+10^{-8}}.
\]

Interpretation:

- `SDI > 0`: fact type explains retention better than need.
- `SDI ≈ 0`: comparable.
- `SDI < 0`: need explains retention better than fact type.

Output:

```text
outputs/tables/surface_dominance_index.csv
```

### 15.4 Preference inversion rate

Build matched pairs within each case:

- one `needed concrete` fact: `need_label=1`, `coarse_group in {EXECUTABLE, CONTROL}`;
- one `unneeded narrative` fact: `need_label=0`, `coarse_group=NARRATIVE`;
- same compressed output condition if possible, or same `(case, model, prompt_variant, budget)`.

Define:

```python
inversion = retained(unneeded_narrative) == 1 and retained(needed_concrete) == 0
```

Output:

```text
outputs/tables/preference_inversion.csv
```

Columns:

```text
compressor_model,prompt_variant,budget_chars,n_pairs,
preference_inversion_rate,ci_low,ci_high
```

### 15.5 Condition Responsiveness Score

For each fact with a needed/unneeded pair:

```python
CRS_f = retention_score_needed_condition - retention_score_unneeded_condition
```

Aggregate by type and model.

Output:

```text
outputs/tables/condition_responsiveness.csv
```

---

## 16. Metrics — iterative hierarchy

### 16.1 Survival curve

For each `(model, prompt_variant, budget, fact_type, round)`:

```python
S = mean(retained_binary)
S_score = mean(retention_score)
```

Output:

```text
outputs/tables/survival_by_round_type.csv
```

Columns:

```text
compressor_model,prompt_variant,budget_chars,round,fact_type,coarse_group,
n_facts,survival_rate,survival_score_mean,ci_low,ci_high
```

### 16.2 Half-life

For each type:

```python
half_life = min(round where survival_rate <= 0.5)
```

If survival never falls below 0.5 by `ROUNDS`, set:

```python
half_life = ROUNDS + 1
half_life_censored = True
```

Output:

```text
outputs/tables/fact_type_half_life.csv
```

### 16.3 Hazard rate

Between rounds:

\[
h_c(r)=1-\frac{S_c(r)}{S_c(r-1)+10^{-8}}.
\]

Output:

```text
outputs/tables/hazard_by_round_type.csv
```

### 16.4 Area under survival curve

\[
\mathrm{AUSC}_c=\sum_{r=1}^{R} S_c(r).
\]

Output:

```text
outputs/tables/ausc_by_type.csv
```

### 16.5 Hierarchy rank and stability

Rank fact types by:

1. half-life descending;
2. AUSC descending;
3. survival at final round descending.

Compute stability:

- within-model bootstrap rank stability;
- cross-model Kendall \(\tau\);
- cross-model Spearman \(\rho\).

Output:

```text
outputs/tables/hierarchy_rank_by_model.csv
outputs/tables/cross_model_hierarchy_similarity.csv
```

### 16.6 Compression fixed point / convergence

For each iterative chain:

```python
text_similarity_round_r = normalized_text_similarity(x_r, x_{r-1})
fact_jaccard_round_r = jaccard(retained_fact_set_r, retained_fact_set_{r-1})
length_change = abs(len(x_r)-len(x_{r-1})) / max(len(x_{r-1}),1)
```

Declare convergence at round `r` if:

```python
text_similarity >= 0.95 and fact_jaccard >= 0.95 and length_change <= 0.02
```

or if fact set unchanged for two consecutive rounds.

Output:

```text
outputs/tables/convergence_by_case.csv
```

Columns:

```text
case_id,condition_id,compressor_model,prompt_variant,budget_chars,
converged,convergence_round,final_output_chars,final_fact_count,
needed_fact_recall_at_convergence,narrative_fact_recall_at_convergence,
executable_fact_recall_at_convergence
```

---

## 17. Main figures

Generate both `.png` and `.pdf`.

### Figure 1 — Need effect by fact type

Bar plot:

- x-axis: fact type
- y-axis: \(\Delta_{need}\)
- hue: model
- error bars: bootstrap 95% CI

Path:

```text
outputs/figures/fig_need_effect_by_fact_type.{png,pdf}
```

### Figure 2 — Surface dominance

Bar plot of `SDI` per model/prompt.

Path:

```text
outputs/figures/fig_surface_dominance_index.{png,pdf}
```

### Figure 3 — Preference inversion

Bar plot of preference inversion rate per model/prompt/budget.

Path:

```text
outputs/figures/fig_preference_inversion_rate.{png,pdf}
```

### Figure 4 — Iterative survival curves

Line plot:

- x-axis: compression round
- y-axis: survival rate
- line: fact type
- facet: model

Path:

```text
outputs/figures/fig_iterative_survival_curves.{png,pdf}
```

### Figure 5 — Information-loss hierarchy heatmap

Heatmap:

- rows: fact types
- columns: rounds
- values: survival rate
- separate panels for models

Path:

```text
outputs/figures/fig_survival_hierarchy_heatmap.{png,pdf}
```

### Figure 6 — Cross-model hierarchy rank

Slope/rank plot comparing fact-type half-life ranks across models.

Path:

```text
outputs/figures/fig_cross_model_hierarchy_rank.{png,pdf}
```

### Figure 7 — Fixed-point recall

Grouped bar plot:

- needed fact recall at convergence
- narrative recall at convergence
- executable recall at convergence

Path:

```text
outputs/figures/fig_fixed_point_recall.{png,pdf}
```

---

## 18. Success criteria

### Strong positive for Claim A

At least three of the following hold:

1. Mean \(\Delta_{need}\) for concrete/executable facts is small, e.g. `< 0.15`, while narrative facts have higher baseline retention.
2. `SDI > 0.3` for at least one primary compressor model.
3. Preference inversion rate is high, e.g. `> 0.25`, under the primary budget.
4. Logistic regression shows `fact_type` explains more retention variance than `need_label`.
5. Need-conditioned and unneeded conditions produce similar retention for at least two concrete fact types.

### Strong positive for Claim B

At least three of the following hold:

1. Survival curves differ clearly by fact type.
2. Narrative facts have longer half-life than executable/control facts.
3. Same hierarchy appears under both Qwen3-4B and MiniMax-M2.5 with Kendall \(\tau > 0.4\).
4. Repeated compression converges within 3–5 rounds for most cases.
5. Needed executable fact recall at convergence is substantially below narrative recall.

### Negative result

Declare the abstraction-prior hypothesis weak or unsupported if:

- \(\Delta_{need}\) is large for most fact types, e.g. `> 0.3`;
- preference inversion rate is near zero;
- fact survival hierarchy is not stable across cases or models;
- retention is mostly explained by length or position, not type or need.

Do not force a positive story if these criteria fail.

---

## 19. Directory layout

```text
motivation_v7/
├── README.md
├── docs/
│   ├── 01_experimental_design.md
│   ├── 02_prompt_templates.md
│   ├── 03_metrics.md
│   ├── 04_results_summary.md
│   └── 05_paper_claims.md
├── prompts/
│   ├── acon_history_ut_original.md
│   ├── acon_history_utco_original.md
│   ├── fact_inventory_prompt.md
│   ├── need_condition_prompt.md
│   ├── condition_validator_prompt.md
│   ├── retention_scorer_prompt.md
│   └── aggregate_report_prompt.md
├── motivation_v8/
│   ├── data.py
│   ├── clients.py
│   ├── acon_prompt_loader.py
│   ├── fact_extract.py
│   ├── need_conditions.py
│   ├── compress.py
│   ├── retention.py
│   ├── metrics.py
│   └── plots.py
├── scripts/
│   ├── 00_sync_acon_prompts.py
│   ├── 01_build_case_pool.py
│   ├── 02_extract_fact_bank.py
│   ├── 03_build_need_conditions.py
│   ├── 04_run_need_conditioned_compression.py
│   ├── 05_score_single_round_retention.py
│   ├── 06_run_iterative_compression.py
│   ├── 07_score_iterative_survival.py
│   ├── 08_compute_metrics.py
│   ├── 09_plot_figures.py
│   ├── 10_write_report.py
│   └── run_all.sh
├── data/
│   ├── case_pool.jsonl
│   ├── fact_candidates_deterministic.jsonl
│   ├── fact_bank_raw.jsonl
│   ├── fact_bank_filtered.jsonl
│   └── need_conditions.jsonl
└── outputs/
    ├── provenance/
    ├── raw/
    ├── tables/
    ├── figures/
    ├── reports/
    └── logs/
```

---

## 20. Pipeline stages

### Stage 00 — Sync original ACON prompts

Command:

```bash
python scripts/00_sync_acon_prompts.py \
  --repo_url https://github.com/microsoft/acon.git \
  --out_dir prompts \
  --provenance_dir outputs/provenance
```

Required outputs:

```text
prompts/acon_history_ut_original.md
prompts/acon_history_utco_original.md
outputs/provenance/acon_commit.txt
outputs/provenance/acon_prompt_sha256.json
```

### Stage 01 — Build case pool

```bash
python scripts/01_build_case_pool.py \
  --appworld_split dev \
  --n_cases 60 \
  --out data/case_pool.jsonl
```

### Stage 02 — Extract fact bank

```bash
python scripts/02_extract_fact_bank.py \
  --cases data/case_pool.jsonl \
  --out_raw data/fact_bank_raw.jsonl \
  --out_filtered data/fact_bank_filtered.jsonl \
  --grounding_table outputs/tables/fact_bank_grounding.csv
```

### Stage 03 — Build need-counterfactual conditions

```bash
python scripts/03_build_need_conditions.py \
  --cases data/case_pool.jsonl \
  --facts data/fact_bank_filtered.jsonl \
  --out data/need_conditions.jsonl \
  --quality_out outputs/tables/need_condition_quality.csv
```

### Stage 04 — Run single-round need-conditioned compression

```bash
python scripts/04_run_need_conditioned_compression.py \
  --cases data/case_pool.jsonl \
  --facts data/fact_bank_filtered.jsonl \
  --conditions data/need_conditions.jsonl \
  --models qwen3-4b,minimax-m2.5 \
  --prompt_variants UTCO,UT \
  --budget_chars 1500 \
  --out outputs/raw/single_round_compressions.jsonl
```

### Stage 05 — Score single-round retention

```bash
python scripts/05_score_single_round_retention.py \
  --facts data/fact_bank_filtered.jsonl \
  --compressions outputs/raw/single_round_compressions.jsonl \
  --out outputs/raw/fact_retention_scores_single_round.jsonl
```

### Stage 06 — Run iterative compression

```bash
python scripts/06_run_iterative_compression.py \
  --cases data/case_pool.jsonl \
  --conditions data/need_conditions.jsonl \
  --models qwen3-4b,minimax-m2.5 \
  --prompt_variants UTCO,UT \
  --rounds 5 \
  --budget_chars 1500 \
  --out outputs/raw/iterative_compressions.jsonl
```

### Stage 07 — Score iterative survival

```bash
python scripts/07_score_iterative_survival.py \
  --facts data/fact_bank_filtered.jsonl \
  --iterative outputs/raw/iterative_compressions.jsonl \
  --out outputs/raw/fact_retention_scores_iterative.jsonl
```

### Stage 08 — Compute metrics

```bash
python scripts/08_compute_metrics.py \
  --single outputs/raw/fact_retention_scores_single_round.jsonl \
  --iterative outputs/raw/fact_retention_scores_iterative.jsonl \
  --out_dir outputs/tables
```

### Stage 09 — Plot figures

```bash
python scripts/09_plot_figures.py \
  --tables outputs/tables \
  --out_dir outputs/figures
```

### Stage 10 — Write report

```bash
python scripts/10_write_report.py \
  --tables outputs/tables \
  --figures outputs/figures \
  --out outputs/reports/motivation_v8_results_summary.md
```

---

## 21. Prompt templates

### Appendix A — `FACT_INVENTORY_PROMPT`

System:

```text
You are a careful AppWorld trajectory information auditor.
Return only valid JSON. Do not include prose outside JSON.
Do not invent facts. Every fact must have a short verbatim quote from the trajectory.
```

User:

```text
You will be given an AppWorld task and a successful full trajectory.
Extract atomic facts that could plausibly affect future compression or future tool-use.

You must classify each fact into exactly one fact_type from this list:
NARRATIVE_GOAL, NARRATIVE_PROGRESS, HIGH_LEVEL_REASONING,
PENDING_SUBTASK, COMPLETED_SUBTASK, RUNTIME_VARIABLE,
AUTH_OR_ACCESS_TOKEN, EXACT_IDENTIFIER, FILE_PATH_OR_RESOURCE_LOCATOR,
API_SCHEMA_OR_PARAMETER, ACTION_OUTCOME, ENVIRONMENT_STATE,
NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT, STALE_OR_OVERWRITTEN_STATE,
NUMERIC_OR_DATE_LITERAL, OTHER_CONCRETE_DETAIL.

Rules:
1. Extract atomic facts, not long paragraphs.
2. Every fact must include a source_quote copied verbatim from the trajectory.
3. Do not invent IDs, tokens, paths, API names, dates, amounts, or action outcomes.
4. Prefer concrete execution facts when present: API parameters, IDs, tokens, paths, returned objects, failures, state changes.
5. Also extract a small number of narrative/progress facts for comparison.
6. Mark is_exact_literal=true for facts that must be copied exactly to remain useful.
7. If unsure, omit the fact.

Return JSON:
{
  "case_id": "{{ case_id }}",
  "facts": [
    {
      "canonical_fact": "short normalized fact",
      "fact_type": "one label",
      "source_step_ids": [0],
      "source_quote": "verbatim quote from trajectory",
      "verbatim_surface": "short surface form likely to appear in summaries",
      "is_exact_literal": true,
      "literal_values": ["optional exact literals"],
      "why_it_might_matter": "one sentence"
    }
  ]
}

Task:
{{ user_instruction }}

Trajectory:
{{ full_trajectory_text }}
```

### Appendix B — `NEED_CONDITION_PROMPT`

System:

```text
You create controlled counterfactual task conditions for an AppWorld compression experiment.
Return only valid JSON. Do not include prose outside JSON.
```

User:

```text
We are testing whether a compressor retains a fact because it is needed for the downstream task.

Given a fact from a trajectory, create two matched task instructions:
1. a NEEDED condition where the fact is necessary for future continuation;
2. an UNNEEDED condition where the same context can be summarized without needing that fact.

Important:
- Do not change the trajectory/history.
- Do not invent new APIs, IDs, or entities.
- Do not explicitly say "remember this fact".
- Do not quote the target fact in the task unless unavoidable.
- Keep the two task instructions similar in length and style.
- The NEEDED condition should imply the fact is needed through the next objective.
- The UNNEEDED condition should make the fact irrelevant or unnecessary.

Return JSON:
{
  "case_id": "{{ case_id }}",
  "fact_id": "{{ fact_id }}",
  "needed_condition": {
    "condition_task": "...",
    "why_fact_is_needed": "..."
  },
  "unneeded_condition": {
    "condition_task": "...",
    "why_fact_is_not_needed": "..."
  },
  "quality_notes": {
    "mentions_fact_directly": false,
    "uses_same_context": true,
    "length_match_ok": true
  }
}

Original task:
{{ user_instruction }}

Target fact:
{{ canonical_fact }}

Fact type:
{{ fact_type }}

Source quote:
{{ source_quote }}

Short trajectory excerpt around source:
{{ local_context }}
```

### Appendix C — `RETENTION_SCORER_PROMPT`

System:

```text
You are a strict retention evaluator.
Return only valid JSON. Do not include prose outside JSON.
```

User:

```text
Decide whether a compressed context preserves a target fact.

Retention labels:
- exact: the compressed context preserves the exact literal or exact API/ID/path/value.
- semantic: the compressed context preserves the fact accurately, but not verbatim; use only if exact literal is not required.
- partial: part of the fact is present but important details are missing.
- absent: the fact is not present.
- contradicted: the compressed context says something inconsistent with the fact.

Rules:
1. If the fact contains an exact ID, token, file path, API name, parameter name, amount, or date, semantic paraphrase is not enough; mark partial or absent unless the required literal is preserved.
2. Quote evidence from the compressed context if present.
3. Do not infer facts that are not explicitly in the compressed context.
4. Return JSON only.

Return JSON:
{
  "fact_id": "{{ fact_id }}",
  "retention_label": "exact | semantic | partial | absent | contradicted",
  "retention_score": 0.0,
  "evidence_in_compressed_text": "verbatim quote or empty string",
  "is_distorted": false,
  "confidence": 0.0,
  "short_reason": "one sentence"
}

Target fact:
{{ canonical_fact }}

Fact type:
{{ fact_type }}

Source quote from original trajectory:
{{ source_quote }}

Literal values that must be preserved if relevant:
{{ literal_values }}

Compressed context:
{{ compressed_text }}
```

### Appendix D — `CONDITION_VALIDATOR_PROMPT`

System:

```text
You validate controlled counterfactual conditions.
Return only valid JSON.
```

User:

```text
Check whether the following needed/unneeded condition pair is valid.

A valid pair:
- uses the same underlying history;
- differs mainly in downstream need;
- makes the target fact necessary in the needed condition;
- makes the target fact unnecessary in the unneeded condition;
- does not trivially instruct the compressor to copy the fact;
- has comparable task-instruction length.

Return JSON:
{
  "valid": true,
  "needed_fact_actually_needed": true,
  "unneeded_fact_actually_unneeded": true,
  "trivially_mentions_fact": false,
  "length_match_ok": true,
  "problems": [],
  "confidence": 0.0
}

Target fact:
{{ canonical_fact }}

Needed condition:
{{ needed_condition_task }}

Unneeded condition:
{{ unneeded_condition_task }}
```

### Appendix E — `AGGREGATE_REPORT_PROMPT`

System:

```text
You are a research analyst writing a concise motivation-experiment report.
Use only the provided tables and numbers. Do not invent results.
```

User:

```text
Write a Markdown report with these exact sections:

# motivation_v8 Results
## Claim A: Is compression preference need-conditioned?
## Claim B: Is there a stable iterative information-loss hierarchy?
## Cross-model stability
## Negative results and caveats
## Recommended next experiment

Inputs:
{{ aggregate_tables_json }}
```

---

## 22. Report template

The final report must include:

1. `n_cases`, `n_facts`, `n_conditions`, `n_compressions`, `n_retention_scores`.
2. A table for need effect by type.
3. A table for surface dominance regression.
4. A figure summary for preference inversion.
5. Survival curves and half-life hierarchy.
6. Cross-model Kendall tau.
7. A clear verdict:
   - supports unconditioned abstraction prior;
   - partially supports;
   - does not support.
8. Caveats.
9. Exact ACON prompt provenance.

Output:

```text
outputs/reports/motivation_v8_results_summary.md
```

---

## 23. Minimum viable run

For fast debugging:

```text
N_CASES=10
MAX_FACTS_PER_CASE=8
MODELS=qwen3-4b
PROMPT_VARIANTS=UTCO
BUDGET_CHARS=1500
ROUNDS=3
```

For full run:

```text
N_CASES=60
MAX_FACTS_PER_CASE=14
MODELS=qwen3-4b,minimax-m2.5
PROMPT_VARIANTS=UTCO,UT
BUDGET_CHARS=1500
ROUNDS=5
```

---

## 24. Expected interpretations

### If Claim A is supported

Use wording like:

> LLM history compressors appear to retain facts according to a surface-level abstraction prior: narrative progress and high-level state survive more often than concrete execution facts, and retention changes only weakly when those concrete facts are made downstream-needed.

### If Claim B is supported

Use wording like:

> Iterated compression induces a stable information-loss hierarchy. The compressor converges to a fixed-point summary that preserves narrative/task-progress information while progressively erasing concrete execution details.

### If both are supported

Use wording like:

> The bottleneck is not only that one compression call may omit details; rather, the compressor defines an abstraction dynamics whose attractor may exclude tool-use-critical facts unless the retention preference itself is changed.

### If not supported

Use wording like:

> We do not find evidence for an unconditioned abstraction prior. Retention is strongly task-conditioned or unstable across models, so future work should focus on other compression bottlenecks.

---

## 25. Implementation checklist

- [ ] `00_sync_acon_prompts.py` successfully copies official ACON prompts and records provenance.
- [ ] No local hand-written ACON compressor prompt is used.
- [ ] Fact bank contains both narrative and concrete execution facts.
- [ ] All primary facts have substring-grounded source quotes.
- [ ] Need/unneeded conditions pass validation.
- [ ] Matched conditions differ only in downstream need, not history.
- [ ] Retention scoring uses deterministic exact matching plus LLM semantic scoring.
- [ ] Single-round retention metrics are computed.
- [ ] Iterative survival metrics are computed for rounds 1–5.
- [ ] Cross-model hierarchy similarity is computed.
- [ ] Figures are generated in both PNG and PDF.
- [ ] Final report states positive and negative evidence honestly.
