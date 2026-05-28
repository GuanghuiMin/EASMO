# motivation_v8 — Fixed-Point Analysis of General LLM Context Compression

> Track: `EASMO/motivation_v8/`  
> Goal: motivation-only diagnostics. No ACON prompt templates, no prompt optimization, no DPO, no SFT, no RL.  
> Primary benchmark: AppWorld successful long-horizon trajectories.  
> Primary question: Do general LLM compression prompts induce fixed points whose fact composition is dominated by surface-type preference rather than downstream need?

---

## 0. High-level framing

In v7, ACON-style structured state compression showed a strong surface-type retention bias and a stable iterative information-loss hierarchy. However, ACON prompts are already highly structured and state-preserving, so v8 removes ACON-style templates entirely.

This experiment treats an LLM compressor as an **iterative compression operator**:

\[
    x_{r+1}=C_{m,p,B}(x_r; q),
\]

where:

- `m` is the compressor model;
- `p` is a **general compression prompt family**, not ACON;
- `B` is a hard character budget;
- `q` is an optional downstream task condition;
- `x_r` is the context after round `r`.

The central object is the **compression fixed point**:

\[
    x_\infty \approx C_{m,p,B}(x_\infty; q).
\]

A fixed point is the context that remains stable after repeated compression. It reveals what the compressor considers stable, summary-worthy, or worth preserving.

The main hypothesis is:

> General LLM context compressors have a surface-type abstraction prior. Repeated compression converges to a fixed point whose fact composition is only weakly conditioned on downstream need. Concrete tool-use facts such as exact IDs, access tokens, API parameters, runtime variables, file paths, and negative evidence decay faster than narrative or high-level state facts, even when the downstream task needs them.

This experiment is **not** a method. It should either support, falsify, or refine this fixed-point abstraction-prior hypothesis.

---

## 1. What v8 changes relative to v7

### Keep from v7

Reuse the following if available:

- AppWorld successful full-context trajectories.
- The 16-class fact taxonomy.
- Substring-grounded fact bank.
- Matched needed/unneeded counterfactual task conditions.
- Cross-model retention scoring protocol.
- Single-round retention metrics: `delta_need`, SDI, PIR, CRS.
- Iterative metrics: survival curves, half-life, hazard, AUSC, hierarchy rank, convergence.

### Change from v7

Do **not** use ACON prompts.

Do **not** load:

```text
prompts/acon_history_ut_original.md
prompts/acon_history_utco_original.md
external/acon_official/...
```

Do **not** use headings such as:

```text
HISTORY_SUMMARY
REASONING
VARS
TODO
COMPLETED
GUARDRAILS
STATE RETAINED
```

Do **not** use any prompt copied from the ACON paper appendix or the `microsoft/acon` repository.

Instead, use the general compression prompts defined in §8 of this document.

### New v8 emphasis

v7 mainly measured survival curves. v8 explicitly studies:

1. **General-prompt fixed points** — does repeated generic compression converge?
2. **Fixed-point fact composition** — what fact types survive at convergence?
3. **Need-conditioned fixed-point shift** — does making a fact downstream-needed move it into the fixed point?
4. **Basin of attraction** — do different initial representations converge to similar fixed points?
5. **Cross-model / cross-prompt hierarchy stability** — is the survival hierarchy stable under model and prompt-family changes?

---

## 2. Claims under test

### Claim A — General compression retains facts by surface type more than need

For a fact `f`, condition `q`, model `m`, prompt family `p`, and budget `B`, define:

\[
    R_{m,p,B}(f \mid x,q) = \mathbf{1}[f \text{ survives in } C_{m,p,B}(x;q)].
\]

A genuinely need-conditioned compressor should show:

\[
    \mathbb{E}[R(f) \mid \mathrm{needed}=1]
    \gg
    \mathbb{E}[R(f) \mid \mathrm{needed}=0].
\]

The abstraction-prior hypothesis predicts:

\[
    \mathbb{E}[R(f) \mid \mathrm{needed}=1]
    \approx
    \mathbb{E}[R(f) \mid \mathrm{needed}=0]
\]

for many concrete execution facts, while retention is better explained by `fact_type`.

### Claim B — Repeated general compression converges to a fixed point

For each chain:

\[
    x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_R,
\]

there exists a small `r <= R` such that:

\[
    x_r \approx x_{r-1}
\]

under text similarity, fact-retention Jaccard, and length-change thresholds.

### Claim C — Fixed-point composition is surface-type biased

At convergence, narrative/high-level facts survive more than concrete tool-use facts:

\[
    \mathrm{Recall}_\infty(\mathrm{NARRATIVE})
    >
    \mathrm{Recall}_\infty(\mathrm{EXECUTABLE})
\]

and concrete categories such as `AUTH_OR_ACCESS_TOKEN`, `API_SCHEMA_OR_PARAMETER`, `RUNTIME_VARIABLE`, `FILE_PATH_OR_RESOURCE_LOCATOR`, and `NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT` have short half-lives or low fixed-point survival.

### Claim D — Fixed-point survival is weakly shifted by downstream need

The key fixed-point need-effect metric is:

\[
    \Delta^\infty_{need}(c)
    =
    \mathbb{E}[R_\infty(f) \mid type(f)=c, needed=1]
    -
    \mathbb{E}[R_\infty(f) \mid type(f)=c, needed=0].
\]

The hypothesis predicts that for concrete execution categories, `Delta_need_infty` is small, even when single-round retention sometimes changes.

### Claim E — Fixed-point basin is stable across initializations

If the compressor has a strong abstraction attractor, then different initial contexts containing the same facts should converge to similar fixed-point fact compositions:

\[
    d_{fact}(x_\infty^{(a)}, x_\infty^{(b)})
    \ll
    d_{fact}(x_0^{(a)}, x_0^{(b)}).
\]

This is tested with `RAW_FULL`, `DETAIL_HEAVY`, `NARRATIVE_HEAVY`, and `FACT_TABLE` initializations.

---

## 3. Models

Use the same compressor models as v7 unless unavailable.

| Role | Model | Endpoint |
|---|---|---|
| Compressor A | `qwen3-4b-instruct-2507` | local vLLM `http://127.0.0.1:8000/v1` |
| Compressor B | `MiniMaxAI/MiniMax-M2.5` | `http://10.183.22.68:8005/v1` |
| Fact-bank extractor | MiniMax-M2.5 | same endpoint |
| Need-condition generator | MiniMax-M2.5 | same endpoint |
| Retention scorer | cross-model scorer | Qwen scores MiniMax outputs; MiniMax scores Qwen outputs |

Generation settings:

```yaml
temperature: 0.0
seed: 42
max_tokens_compression: 2048
max_tokens_json_scoring: 512
response_format_json: true for JSON calls only
```

Do not use stochastic sampling in the primary run. Optional seed-sweep `{1,2,3}` is deferred.

---

## 4. Data

### Primary case pool

Reuse v7's successful AppWorld dev trajectories if present:

```text
motivation_v7/data/cases.jsonl
motivation_v7/data/fact_bank_filtered.jsonl
motivation_v7/data/need_conditions_validated.jsonl
```

Fallback sources:

```text
motivation_v3/outputs/motivation_full_trajectories.jsonl
motivation_v7/data/fact_bank_filtered.jsonl
motivation_v7/data/need_conditions_validated.jsonl
```

Primary setting:

```yaml
N_CASES: 30
history_char_cap: 18000
```

If fewer than 30 cases are available, run all available cases and document the count.

### Fact sampling

For single-round experiments, use at most `MAX_FACTS_PER_CASE_SINGLE = 6` valid matched needed/unneeded pairs per case.

Priority order:

1. `AUTH_OR_ACCESS_TOKEN`
2. `API_SCHEMA_OR_PARAMETER`
3. `RUNTIME_VARIABLE`
4. `FILE_PATH_OR_RESOURCE_LOCATOR`
5. `EXACT_IDENTIFIER`
6. `ACTION_OUTCOME`
7. `NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT`
8. `ENVIRONMENT_STATE`
9. `NARRATIVE_GOAL`
10. `NARRATIVE_PROGRESS`

For iterative fixed-point experiments, choose one representative concrete/executable target fact per case.

Priority order for the target fact:

1. exact-literal executable fact;
2. `AUTH_OR_ACCESS_TOKEN`;
3. `API_SCHEMA_OR_PARAMETER`;
4. `RUNTIME_VARIABLE`;
5. `FILE_PATH_OR_RESOURCE_LOCATOR`;
6. `EXACT_IDENTIFIER`;
7. any executable fact.

For fixed-point composition analysis, score **all filtered facts in that case**, not just the target fact.

---

## 5. Fact taxonomy

Reuse v7's 16-class taxonomy.

```text
NARRATIVE:
  NARRATIVE_GOAL
  NARRATIVE_PROGRESS
  HIGH_LEVEL_REASONING

TASK_STATE:
  PENDING_SUBTASK
  COMPLETED_SUBTASK
  ENVIRONMENT_STATE
  STALE_OR_OVERWRITTEN_STATE

EXECUTABLE:
  RUNTIME_VARIABLE
  AUTH_OR_ACCESS_TOKEN
  EXACT_IDENTIFIER
  FILE_PATH_OR_RESOURCE_LOCATOR
  API_SCHEMA_OR_PARAMETER
  ACTION_OUTCOME
  NUMERIC_OR_DATE_LITERAL

CONTROL:
  NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT

OTHER:
  OTHER_CONCRETE_DETAIL
```

For v8, additionally define:

```python
CONCRETE_TOOL_USE_FACTS = {
    "RUNTIME_VARIABLE",
    "AUTH_OR_ACCESS_TOKEN",
    "EXACT_IDENTIFIER",
    "FILE_PATH_OR_RESOURCE_LOCATOR",
    "API_SCHEMA_OR_PARAMETER",
    "ACTION_OUTCOME",
    "NUMERIC_OR_DATE_LITERAL",
    "NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT",
}
```

---

## 6. Need/unneeded conditions

Reuse v7's validated needed/unneeded counterfactual conditions where possible.

If regenerating, use the same rule-based quality constraints:

1. Neither condition may quote the target fact.
2. Neither condition may include any literal value from the target fact.
3. Length mismatch must be <= 35%.
4. The needed condition must make the fact plausibly required for future execution.
5. The unneeded condition must make the same fact irrelevant while keeping the task length and style similar.

Save:

```text
data/need_conditions_raw.jsonl
data/need_conditions_validated.jsonl
outputs/tables/need_condition_quality.csv
```

---

## 7. Prompt families

Do not use ACON. Use only the following general prompt families.

### Prompt family P1: `general_task_aware`

Purpose: generic task-aware compression without fixed sections.

System message:

```text
You are a careful context compression module for a tool-use agent.
Return only the compressed context. Do not include explanations about your compression process.
```

User message:

```text
Compress the previous interaction history into a shorter context for a downstream tool-use agent.

The downstream agent will continue the following task:
{condition_task}

Hard budget:
- The compressed context must be no more than {max_chars} characters.

Compression goals:
- Preserve information that may help the downstream agent continue the task correctly.
- Preserve exact identifiers, API names, parameter names, file paths, dates, amounts, access/auth values, object IDs, and state-changing action outcomes when they may matter.
- Preserve failed attempts or negative evidence only if they may prevent repeated mistakes.
- Remove redundant, obsolete, or irrelevant details.
- Do not invent facts.
- Do not solve the task.
- Do not output the original input.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Previous interaction history:
{context}

Compressed context:
```

### Prompt family P2: `general_task_agnostic`

Purpose: generic compression without downstream task conditioning.

System message:

```text
You are a careful context compression module.
Return only the compressed context. Do not include explanations about your compression process.
```

User message:

```text
Compress the following interaction history into a shorter version.

Hard budget:
- The compressed context must be no more than {max_chars} characters.

Compression goals:
- Preserve important information.
- Remove redundant, obsolete, or irrelevant details.
- Keep exact values only if they appear important in the history.
- Do not invent facts.
- Do not output the original input.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Interaction history:
{context}

Compressed context:
```

### Prompt family P3: `general_strict_extract_then_compress` optional ablation

Run only after P1/P2 complete.

Purpose: test whether stronger generic instruction to preserve exact facts changes the fixed point without using ACON-style schema.

System message:

```text
You are a loss-aware compression module for a tool-use agent.
Return only the compressed context.
```

User message:

```text
Compress the interaction history under {max_chars} characters.

Task condition:
{condition_task}

Rules:
1. First identify exact facts that could be needed later: IDs, tokens, file paths, API names, API parameters, dates, amounts, action outcomes, and failed attempts.
2. Preserve only the exact facts that are likely to matter for continuing the task.
3. Compress everything else aggressively.
4. Do not use a fixed output schema.
5. Do not invent or alter exact values.
6. Return only the compressed context.

Interaction history:
{context}

Compressed context:
```

Primary v8 should run P1 and P2. P3 is optional.

---

## 8. Budgets

Primary budget:

```yaml
TARGET_MAX_CHARS: 1500
```

Secondary budgets, run only if primary pipeline is stable:

```yaml
TARGET_MAX_CHARS: [800, 2500]
```

Unlike v7's ACON prompt, v8 general prompts explicitly include `{max_chars}`. Record actual output length and budget violation rate.

Budget violation:

```python
budget_violation = len(compressed_context) > TARGET_MAX_CHARS * 1.10
```

The 10% tolerance accounts for tokenizer and model formatting variability.

---

## 9. Experiment 1 — Single-round need sensitivity under general prompts

### Purpose

Test whether general compression is need-conditioned or surface-type-conditioned.

### Inputs

For each valid `(case, fact, needed_condition, unneeded_condition)` pair:

```text
context = full_trajectory_text
condition_task = needed_condition or unneeded_condition
prompt_family = P1 or P2
model = Qwen or MiniMax
budget = 1500
```

For P2, the prompt ignores `condition_task`, but still store the condition label so the output can serve as a task-agnostic control.

### Run matrix

Primary:

```text
N_cases <= 30
MAX_FACTS_PER_CASE_SINGLE = 6
conditions = {needed, unneeded}
prompt_families = {general_task_aware, general_task_agnostic}
models = {Qwen3-4B, MiniMax-M2.5}
```

Approximate compression calls:

```text
30 cases × 6 facts × 2 conditions × 2 prompts × 2 models = 1440 max
```

If cost is too high, set `MAX_FACTS_PER_CASE_SINGLE=4`.

### Output

```text
outputs/raw/single_round_compressions.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "fact_id": "...",
  "fact_type": "API_SCHEMA_OR_PARAMETER",
  "coarse_group": "EXECUTABLE",
  "need_label": 1,
  "condition_task": "...",
  "prompt_family": "general_task_aware",
  "model": "qwen3-4b-instruct-2507",
  "budget_chars": 1500,
  "input_context_chars": 15832,
  "compressed_context": "...",
  "compressed_chars": 1412,
  "budget_violation": false,
  "error": null
}
```

---

## 10. Experiment 2 — Iterative fixed-point compression

### Purpose

Test whether repeated general compression converges and whether the fixed point has a stable fact-type composition.

### Chain definition

For each selected case and target fact:

\[
    x_0 = \text{initial context},
\]

\[
    x_{r+1} = C_{m,p,B}(x_r; q),\quad r=0,\ldots,R-1.
\]

Primary setting:

```yaml
ROUNDS: 6
N_ITER_CASES: 20
prompt_families: [general_task_aware, general_task_agnostic]
models: [Qwen3-4B, MiniMax-M2.5]
budget_chars: 1500
```

For `general_task_aware`, run both target-fact conditions:

```text
needed
unneeded
```

For `general_task_agnostic`, run a single chain per case using no task condition. Store `condition_type=task_agnostic`.

### Initial context

Primary:

```text
init_type = RAW_FULL
x_0 = full_trajectory_text capped at 18k chars
```

### Output

```text
outputs/raw/iterative_chains.jsonl
```

One row per round:

```json
{
  "chain_id": "case123__fact456__needed__general_task_aware__qwen__raw_full",
  "case_id": "case123",
  "target_fact_id": "fact456",
  "target_fact_type": "AUTH_OR_ACCESS_TOKEN",
  "condition_type": "needed",
  "condition_task": "...",
  "prompt_family": "general_task_aware",
  "model": "qwen3-4b-instruct-2507",
  "budget_chars": 1500,
  "init_type": "RAW_FULL",
  "round": 0,
  "context_text": "...",
  "context_chars": 16000,
  "budget_violation": false,
  "error": null
}
```

Round 0 is the initial context. Rounds 1..6 are compressor outputs.

---

## 11. Experiment 3 — Basin-of-attraction analysis

### Purpose

Test whether different initial representations of the same underlying trajectory converge to similar fixed points.

### Scope

Run on a smaller subset:

```yaml
N_BASIN_CASES: 12
prompt_family: general_task_aware
condition_type: needed
models: [Qwen3-4B, MiniMax-M2.5]
ROUNDS: 6
budget_chars: 1500
```

### Initializations

Use four initial contexts for each case:

#### 1. `RAW_FULL`

```text
x_0 = full_trajectory_text capped at 18k chars
```

#### 2. `DETAIL_HEAVY`

A deterministic context that prepends a fact table to the raw trajectory:

```text
Known facts extracted from the trajectory:
- [FACT_ID=f1][TYPE=...] canonical_fact
- [FACT_ID=f2][TYPE=...] canonical_fact
...

Original trajectory:
{full_trajectory_text}
```

#### 3. `NARRATIVE_HEAVY`

Generate one generic narrative summary from P2 first, then append the raw trajectory tail.

```text
Narrative overview:
{generic_summary}

Recent trajectory tail:
{last 30% of trajectory}
```

#### 4. `FACT_TABLE_ONLY`

Only the fact table, no raw trajectory:

```text
Known facts extracted from the trajectory:
- [FACT_ID=f1][TYPE=...] canonical_fact
- [FACT_ID=f2][TYPE=...] canonical_fact
...
```

Do not use ACON or ACON-like headings.

### Output

Reuse `outputs/raw/iterative_chains.jsonl` with `init_type` set accordingly.

---

## 12. Retention scoring

### Scoring rule

For each `(fact, compressed_context)`:

1. Run deterministic exact substring match first.
2. If exact match succeeds, set:

```json
{
  "retention_label": "exact",
  "retention_score": 1.0,
  "retained_binary": true,
  "match_type": "substring_exact"
}
```

3. Otherwise, call the cross-model retention scorer.

Cross-model rule:

| Compressor | Scorer |
|---|---|
| Qwen3-4B | MiniMax-M2.5 |
| MiniMax-M2.5 | Qwen3-4B |

### Retention scorer prompt

System message:

```text
You are a strict fact-retention judge.
Return only JSON. Do not explain outside JSON.
```

User message:

```text
You must decide whether a compressed context retains a target fact from an original tool-use trajectory.

Target fact:
{canonical_fact}

Fact type:
{fact_type}

Literal values, if any:
{literal_values}

Compressed context:
{compressed_context}

Labels:
- exact: the fact is preserved exactly or all literal values needed for the fact are present.
- semantic: the fact is clearly preserved with equivalent meaning, even if phrased differently.
- partial: only part of the fact is preserved; some exact values, bindings, or conditions are missing.
- absent: the fact is not present.
- contradicted: the compressed context conflicts with the fact.

Be strict for exact identifiers, file paths, tokens, API names, parameter names, dates, amounts, and object IDs. If an exact literal is needed but paraphrased or omitted, do not label it semantic.

Return JSON:
{
  "retention_label": "exact | semantic | partial | absent | contradicted",
  "retention_score": 1.0 | 0.75 | 0.4 | 0.0 | -0.5,
  "evidence_in_compressed_text": "short quote or empty string",
  "is_distorted": true | false,
  "confidence": "high | medium | low",
  "short_reason": "one sentence"
}
```

### Output

```text
outputs/raw/retention_scores.jsonl
```

Schema:

```json
{
  "context_source": "single_round | iterative | basin",
  "case_id": "...",
  "chain_id": "...",
  "round": 3,
  "model": "MiniMaxAI/MiniMax-M2.5",
  "prompt_family": "general_task_aware",
  "init_type": "RAW_FULL",
  "condition_type": "needed",
  "fact_id": "...",
  "fact_type": "AUTH_OR_ACCESS_TOKEN",
  "coarse_group": "EXECUTABLE",
  "need_label": 1,
  "retention_label": "absent",
  "retention_score": 0.0,
  "retained_binary": false,
  "match_type": "llm_semantic",
  "evidence_in_compressed_text": "",
  "confidence": "high"
}
```

---

## 13. Metrics

Implement all metrics in:

```text
motivation_v8/metrics.py
```

### 13.1 Single-round need effect

For each `(model, prompt_family, budget_chars, fact_type)`:

```python
delta_need = mean(retained_binary | need_label=1) - mean(retained_binary | need_label=0)
delta_need_score = mean(retention_score | need_label=1) - mean(retention_score | need_label=0)
```

Bootstrap 95% CI with 1000 resamples.

Output:

```text
outputs/tables/need_effect_by_type.csv
```

### 13.2 Surface dominance regression

Fit three logistic models per `(model, prompt_family, budget_chars)`:

```text
M_need: retained_binary ~ need_label
M_type: retained_binary ~ C(fact_type)
M_both: retained_binary ~ need_label + C(fact_type)
```

Report McFadden pseudo-R²:

```python
sdi = (r2_type - r2_need) / (r2_type + r2_need + 1e-8)
```

Output:

```text
outputs/tables/surface_dominance_regression.csv
outputs/tables/surface_dominance_index.csv
```

### 13.3 Preference inversion rate

Within case and prompt/model, pair:

- needed concrete fact: `coarse_group in {EXECUTABLE, CONTROL}`, `need_label=1`
- unneeded narrative fact: `coarse_group=NARRATIVE`, `need_label=0`

Inversion:

```python
inversion = retained(unneeded_narrative)==1 and retained(needed_concrete)==0
PIR = mean(inversion)
```

Output:

```text
outputs/tables/preference_inversion.csv
```

### 13.4 Condition responsiveness score

For matched fact pair:

```python
CRS_f = retention_score(needed) - retention_score(unneeded)
```

Aggregate by fact type and coarse group.

Output:

```text
outputs/tables/condition_responsiveness.csv
```

### 13.5 Iterative survival curve

For each `(model, prompt_family, init_type, condition_type, round, fact_type)`:

```python
survival_rate = mean(retained_binary)
survival_score_mean = mean(retention_score)
```

Output:

```text
outputs/tables/survival_by_round_type.csv
```

### 13.6 Half-life

```python
half_life = min(r for r in rounds if survival_rate[r] <= 0.5)
```

If no round crosses 0.5 by `ROUNDS`, set:

```python
half_life = ROUNDS + 1
half_life_censored = True
```

Output:

```text
outputs/tables/fact_type_half_life.csv
```

### 13.7 Hazard

```python
hazard[1] = 1 - S[1]
hazard[r] = 1 - S[r] / max(S[r-1], 1e-8)
```

Output:

```text
outputs/tables/hazard_by_round_type.csv
```

### 13.8 Area under survival curve

```python
AUSC = sum(S[r] for r in 1..ROUNDS)
```

Output:

```text
outputs/tables/ausc_by_type.csv
```

### 13.9 Fixed-point convergence

Declare a chain converged at round `r` if all three hold:

```python
text_similarity(x_r, x_{r-1}) >= 0.95
fact_jaccard(retained_facts_r, retained_facts_{r-1}) >= 0.95
abs(len(x_r) - len(x_{r-1})) / max(len(x_{r-1}), 1) <= 0.02
```

Text similarity can be cosine similarity of sentence-transformer embeddings if available; fallback to token-level Jaccard.

Fact Jaccard:

```python
J = |retained_fact_ids_r ∩ retained_fact_ids_{r-1}| / |retained_fact_ids_r ∪ retained_fact_ids_{r-1}|
```

Output:

```text
outputs/tables/convergence_by_chain.csv
```

### 13.10 Fixed-point composition

For each chain, take:

```python
fixed_round = convergence_round if converged else ROUNDS
x_fixed = x_fixed_round
```

Compute:

```python
fixed_point_survival_by_type = mean(retained_binary at fixed_round | fact_type)
fixed_point_share_by_type = (# retained facts of type c at fixed_round) / (# all retained facts at fixed_round)
needed_fact_recall_fixed = mean(retained_binary at fixed_round | need_label=1)
executable_recall_fixed = mean(retained_binary at fixed_round | coarse_group=EXECUTABLE)
narrative_recall_fixed = mean(retained_binary at fixed_round | coarse_group=NARRATIVE)
control_recall_fixed = mean(retained_binary at fixed_round | coarse_group=CONTROL)
```

Output:

```text
outputs/tables/fixed_point_composition_by_type.csv
outputs/tables/fixed_point_recall_by_chain.csv
```

### 13.11 Fixed-point need shift

For each fact type:

```python
delta_need_infty = mean(retained_binary_fixed | need_label=1) - mean(retained_binary_fixed | need_label=0)
```

Compute separately for:

- `general_task_aware`
- `general_task_agnostic`
- each model
- each budget

Output:

```text
outputs/tables/fixed_point_need_shift.csv
```

### 13.12 Hierarchy rank stability

Rank fact types by:

1. half-life descending;
2. final survival descending;
3. AUSC descending.

Compute Kendall tau and Spearman rho across:

- Qwen vs MiniMax under the same prompt family;
- P1 vs P2 under the same model;
- v7 ACON results vs v8 general results if v7 table is available, but do not require this for primary results.

Output:

```text
outputs/tables/hierarchy_rank_by_model_prompt.csv
outputs/tables/cross_model_prompt_hierarchy_similarity.csv
```

### 13.13 Basin-of-attraction metrics

For each `(case_id, model, prompt_family, condition_type)` with multiple initializations:

Compute initial pairwise distances between all `x_0` variants and final pairwise distances between fixed points.

Distances:

```python
fact_jaccard_distance = 1 - Jaccard(retained_fact_ids_a, retained_fact_ids_b)
fact_type_l1_distance = L1(normalized_fact_type_distribution_a, normalized_fact_type_distribution_b)
text_embedding_distance = 1 - cosine(embedding_a, embedding_b)
```

Basin contraction:

```python
contraction_ratio = final_distance / max(initial_distance, 1e-8)
```

Attractor if:

```python
contraction_ratio < 0.5
```

Output:

```text
outputs/tables/basin_similarity.csv
outputs/tables/basin_contraction.csv
```

### 13.14 Budget compliance

For every compressed context:

```python
budget_violation = compressed_chars > budget_chars * 1.10
```

Aggregate:

```python
violation_rate = mean(budget_violation)
median_length
p90_length
```

Output:

```text
outputs/tables/budget_compliance.csv
```

---

## 14. Figures

Generate both `.png` and `.pdf`.

### Figure 1 — Single-round need effect by fact type

```text
outputs/figures/fig_need_effect_by_fact_type.{png,pdf}
```

Bar plot:

- x-axis: fact type;
- y-axis: `delta_need`;
- hue: model;
- facet: prompt family;
- error bars: bootstrap 95% CI.

### Figure 2 — Surface dominance index

```text
outputs/figures/fig_surface_dominance_index.{png,pdf}
```

Grouped bar:

- x-axis: prompt family;
- y-axis: SDI;
- hue: model.

### Figure 3 — Preference inversion rate

```text
outputs/figures/fig_preference_inversion_rate.{png,pdf}
```

Bar plot by model and prompt family.

### Figure 4 — Iterative survival curves

```text
outputs/figures/fig_iterative_survival_curves.{png,pdf}
```

Line plot:

- x-axis: round;
- y-axis: survival rate;
- line: fact type or coarse group;
- facet: model × prompt family.

### Figure 5 — Fixed-point composition

```text
outputs/figures/fig_fixed_point_composition_by_type.{png,pdf}
```

Stacked bar or heatmap showing retained fact-type share at fixed point.

### Figure 6 — Fixed-point need shift

```text
outputs/figures/fig_fixed_point_need_shift.{png,pdf}
```

Bar plot:

- x-axis: fact type;
- y-axis: `delta_need_infty`;
- hue: model;
- facet: prompt family.

### Figure 7 — Cross-model / cross-prompt hierarchy rank

```text
outputs/figures/fig_cross_model_prompt_hierarchy_rank.{png,pdf}
```

Slope plot of fact-type ranks across model/prompt pairs.

### Figure 8 — Basin contraction

```text
outputs/figures/fig_basin_contraction.{png,pdf}
```

Boxplot of `contraction_ratio` by model and init pair.

### Figure 9 — Fixed-point recall groups

```text
outputs/figures/fig_fixed_point_recall_groups.{png,pdf}
```

Grouped bar:

- needed fact recall at fixed point;
- narrative recall;
- executable recall;
- control recall.

---

## 15. Directory layout

```text
motivation_v8/
├── README.md
├── docs/
│   ├── 01_experimental_design.md
│   ├── 02_prompt_templates.md
│   ├── 03_metrics.md
│   └── 04_results_summary.md
├── prompts/
│   ├── general_task_aware.md
│   ├── general_task_agnostic.md
│   ├── general_strict_extract_then_compress.md
│   ├── fact_inventory.md
│   ├── need_condition_generator.md
│   └── retention_scorer.md
├── motivation_v8/
│   ├── clients.py
│   ├── data.py
│   ├── prompts.py
│   ├── fact_bank.py
│   ├── need_conditions.py
│   ├── compress.py
│   ├── iterate.py
│   ├── retention.py
│   ├── metrics.py
│   └── plots.py
├── scripts/
│   ├── 00_prepare_inputs.py
│   ├── 01_build_or_reuse_fact_bank.py
│   ├── 02_build_or_reuse_need_conditions.py
│   ├── 03_run_single_round.py
│   ├── 04_run_iterative_fixed_points.py
│   ├── 05_run_basin_experiment.py
│   ├── 06_score_retention.py
│   ├── 07_compute_metrics.py
│   ├── 08_plot_figures.py
│   ├── 09_write_report.py
│   └── run_all.sh
├── data/
│   ├── cases.jsonl
│   ├── fact_bank_filtered.jsonl
│   ├── need_conditions_validated.jsonl
│   └── selected_iterative_targets.jsonl
└── outputs/
    ├── raw/
    │   ├── single_round_compressions.jsonl
    │   ├── iterative_chains.jsonl
    │   ├── retention_scores.jsonl
    │   └── errors.jsonl
    ├── tables/
    ├── figures/
    ├── reports/
    │   └── results_summary.md
    └── provenance/
        ├── prompt_sha256.json
        ├── run_config.yaml
        └── source_artifacts.json
```

---

## 16. Pipeline stages

### Stage 00 — Prepare inputs

Script:

```bash
python scripts/00_prepare_inputs.py
```

Tasks:

- Load successful AppWorld trajectories.
- Cap context length at 18k chars.
- Copy v7 fact bank and need conditions if available.
- Write `data/cases.jsonl`.
- Write provenance.

### Stage 01 — Build or reuse fact bank

Script:

```bash
python scripts/01_build_or_reuse_fact_bank.py
```

Tasks:

- If v7 `fact_bank_filtered.jsonl` exists, reuse it.
- Else run fact-inventory extraction.
- Enforce substring grounding.
- Cap facts per case.

### Stage 02 — Build or reuse need conditions

Script:

```bash
python scripts/02_build_or_reuse_need_conditions.py
```

Tasks:

- If v7 validated conditions exist, reuse them.
- Else generate needed/unneeded pairs.
- Apply rule-based quality filters.

### Stage 03 — Single-round compression

Script:

```bash
python scripts/03_run_single_round.py
```

Tasks:

- Run P1/P2 prompts on matched needed/unneeded conditions.
- Store outputs.
- Record budget violations.

### Stage 04 — Iterative fixed-point compression

Script:

```bash
python scripts/04_run_iterative_fixed_points.py
```

Tasks:

- Select representative target fact per case.
- Run iterative chains for P1/P2.
- Store all rounds including round 0.

### Stage 05 — Basin experiment

Script:

```bash
python scripts/05_run_basin_experiment.py
```

Tasks:

- Build initializations `RAW_FULL`, `DETAIL_HEAVY`, `NARRATIVE_HEAVY`, `FACT_TABLE_ONLY`.
- Run iterative compression for each initialization.
- Store all chains.

### Stage 06 — Retention scoring

Script:

```bash
python scripts/06_score_retention.py
```

Tasks:

- Score fact retention for single-round outputs.
- Score fact retention for each iterative round.
- Score all facts for fixed-point composition.
- Use deterministic exact match before LLM scoring.
- Use cross-model scoring.

### Stage 07 — Compute metrics

Script:

```bash
python scripts/07_compute_metrics.py
```

Tasks:

- Compute all tables in §13.

### Stage 08 — Plot figures

Script:

```bash
python scripts/08_plot_figures.py
```

Tasks:

- Generate all figures in §14.

### Stage 09 — Write report

Script:

```bash
python scripts/09_write_report.py
```

Tasks:

- Write deterministic Markdown report.
- Do not use an LLM to interpret results.

### Run all

```bash
bash scripts/run_all.sh
```

---

## 17. Success criteria

### Strong positive for Claim A

At least three of:

1. For `general_task_aware`, `R2_need < 0.02` and `R2_type >= 5 * R2_need`.
2. SDI > 0.5 for at least one model under general prompts.
3. Mean `delta_need` for executable facts is < 0.15.
4. At least two concrete fact types have |delta_need| <= 0.05.
5. Preference inversion rate > 0.20.

### Strong positive for Claim B / C

At least three of:

1. >= 60% of chains converge within 6 rounds.
2. Narrative/high-level facts have longer half-life or higher AUSC than executable/control facts.
3. Fixed-point executable recall is at least 0.10 below narrative recall.
4. `AUTH_OR_ACCESS_TOKEN` or `API_SCHEMA_OR_PARAMETER` is bottom-3 by AUSC in both models.
5. Cross-model hierarchy Kendall tau > 0.35.

### Strong positive for Claim D

At least two of:

1. For executable facts, `abs(delta_need_infty) < 0.15` under `general_task_aware`.
2. `delta_need_infty` is smaller than the gap between narrative and executable fixed-point recall.
3. The fixed-point fact-type regression has higher explanatory power than need-label regression.

### Strong positive for Claim E

At least two of:

1. Mean basin contraction ratio < 0.5 for fact-type distribution distance.
2. Mean final pairwise fact Jaccard similarity > initial pairwise fact Jaccard similarity by >= 0.2.
3. Different initializations converge to the same top-3 retained fact types in >= 70% of cases.

### Negative result

Declare the hypothesis weak or unsupported if:

- retention is strongly need-conditioned under general prompts;
- no stable fixed points appear;
- survival hierarchy is not stable across cases or models;
- budget violations dominate outputs;
- retention is mostly explained by output length rather than fact type or need.

Do not force a positive story if these criteria fail.

---

## 18. Report template

Write:

```text
outputs/reports/results_summary.md
```

with exactly these sections:

```markdown
# motivation_v8 Results — Fixed Points of General LLM Compression

## 1. Setup
- cases
- models
- prompt families
- budgets
- number of facts
- number of compression calls
- number of retention scoring calls
- budget violation rate

## 2. Claim A: Single-Round Need Conditioning
Report R2_need, R2_type, SDI, delta_need by type, PIR, CRS.
State whether general compression is need-conditioned.

## 3. Claim B: Fixed-Point Convergence
Report convergence rate, convergence rounds, length dynamics, and examples.
State whether repeated general compression converges.

## 4. Claim C: Fixed-Point Composition
Report final/fixed-point survival by fact type and coarse group.
Identify attractors and repellors.

## 5. Claim D: Need-Conditioned Fixed-Point Shift
Report delta_need_infty by fact type.
State whether downstream need changes the fixed point.

## 6. Claim E: Basin of Attraction
Report contraction ratios and fixed-point similarity across initializations.
State whether the fixed point is an attractor rather than an artifact of one initial context.

## 7. Cross-Model and Cross-Prompt Stability
Report hierarchy Kendall tau and Spearman rho across models and prompt families.

## 8. Comparison to v7
Do not re-run ACON. If v7 tables are available, compare at a high level:
- general prompt vs ACON prompt SDI
- general prompt vs ACON prompt fixed-point hierarchy
- whether the phenomenon is generic or ACON-specific

## 9. Negative Findings and Caveats
Include budget violations, scorer uncertainty, missing short trajectories, low pair quality, or model-specific quirks.

## 10. Paper-Level Interpretation
Write one concise paragraph stating what this experiment supports or rejects.

## 11. Files of Record
List all raw JSONL, tables, figures, and provenance files.
```

---

## 19. Expected interpretation patterns

### Pattern 1 — strongest positive

If general prompts reproduce v7-like findings:

- SDI near 1;
- small `delta_need`;
- strong fixed-point hierarchy;
- stable cross-model rank;
- executable facts low at fixed point;
- convergence under multiple initializations;

then claim:

> The abstraction-prior phenomenon is not an artifact of ACON's structured prompt. General LLM compression itself induces a fixed point dominated by surface-type preference. ACON may partially reshape this prior, but it does not create it.

### Pattern 2 — ACON stronger than general

If general prompts have weaker fixed points or noisier hierarchy:

> ACON-style structured prompts impose a stronger projection operator than generic compression. The fixed-point framework still applies, but the attractor is prompt-family dependent.

### Pattern 3 — general is need-conditioned

If general task-aware prompts strongly retain needed facts:

> The unconditioned prior is not universal. v7 may be a property of ACON-style structured compression or of its UTCO prompt. Future method should use simpler task-aware prompts or avoid overly rigid structured state projections.

### Pattern 4 — no convergence

If chains do not converge:

> Fixed-point analysis is not appropriate for general prompts under these budgets. The iterative hierarchy observed in v7 may require structured prompts or shorter output constraints.

---

## 20. Implementation notes

1. Use deterministic caching for all model calls. Key cache by SHA256 of `(model, system, user, temperature, seed)`.
2. Write outputs incrementally to JSONL to allow resume after failures.
3. Log every prompt and response to `outputs/provenance/model_call_logs/` if storage allows.
4. Strip model preambles such as `Here is...` from compressed outputs only if they violate the instruction; preserve the exact text for auditing.
5. Do not post-process compressed contexts to improve retention.
6. Do not add ACON-style headings during post-processing.
7. Use the same retention scorer across P1/P2 so prompt comparison is fair.
8. Always report budget violations. A prompt that retains facts only by exceeding budget should not be counted as successful compression.

---

## 21. Minimal run configuration

If cost is constrained, use:

```yaml
N_CASES: 20
MAX_FACTS_PER_CASE_SINGLE: 4
N_ITER_CASES: 15
N_BASIN_CASES: 8
ROUNDS: 5
prompt_families: [general_task_aware, general_task_agnostic]
models: [qwen3-4b-instruct-2507, MiniMaxAI/MiniMax-M2.5]
budget_chars: 1500
```

If this minimal run shows strong positive trends, run the full configuration.

---

## 22. Full run configuration

```yaml
N_CASES: 30
MAX_FACTS_PER_CASE_SINGLE: 6
N_ITER_CASES: 20
N_BASIN_CASES: 12
ROUNDS: 6
prompt_families:
  - general_task_aware
  - general_task_agnostic
models:
  - qwen3-4b-instruct-2507
  - MiniMaxAI/MiniMax-M2.5
budget_chars: 1500
secondary_budgets: []
```

Do not enable secondary budgets until primary report is written.

---

## 23. Final instruction to the coding agent

The purpose of v8 is to verify whether **general** LLM compression induces fixed points with surface-type-biased fact survival.

Do not optimize prompts. Do not train. Do not use ACON templates. Do not improve compression manually.

The primary deliverable is a reproducible diagnostic package:

- raw compression chains;
- fact retention scores;
- fixed-point composition tables;
- need-shift tables;
- basin-of-attraction metrics;
- figures;
- deterministic results report.

If the hypothesis fails, write that clearly. The negative result is useful.
