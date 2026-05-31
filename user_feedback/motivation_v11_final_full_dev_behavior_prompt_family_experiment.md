# motivation_v11 — Full-Dev Behavior Study of Prompt Families and Compression Policy Headroom

> **Purpose:** final motivation-section experiment for the paper.
>
> **Core framing:** agent histories are operational state, not documents. Structured compression prompts can create useful compression distributions, but they do not guarantee that default decoding selects the behavior-optimal compression. We therefore study the gap between prompt-induced distribution quality and decoding / selector calibration under **serial recompression robustness**.
>
> **Primary output:** paper-ready behavior tables and figures supporting the motivation section:
>
> 1. **Agent Histories Are Operational State, Not Documents**
> 2. **Structured Prompts Improve the Distribution, Not the Selection Policy**
> 3. **High-Utility Compressions Hide in the Sampling Distribution**

---

## 0. Executive summary for the coding agent

This experiment should be treated as a **claim-driven, behavior-first motivation consolidation**, not a new exploratory taxonomy study.

The experiment compares four prompt families on the **full AppWorld dev split**:

1. `general_task_agnostic`
2. `general_task_aware`
3. `ACON_UT`
4. `ACON_UTCO`

For each full-context-successful AppWorld dev case, each prompt family generates:

- one greedy compressed context;
- `N=8` stochastic compressed contexts;
- a serial recompression chain of depth `K=2` for each candidate;
- AppWorld continuation behavior for both the one-step compressed context `C1` and the stress-compressed context `CK`.

The main analysis decomposes prompt-family performance into:

\[
Q_{\mathrm{dist}}(p) = \mathrm{BestOfNPass@CK}(p)
\]

and

\[
G_{\mathrm{calib}}(p) = \mathrm{BestOfNPass@CK}(p) - \mathrm{GreedyPass@CK}(p).
\]

Interpretation:

- `Q_dist` measures whether the prompt family induces a compression distribution containing good behavior-preserving samples.
- `G_calib` measures whether default greedy decoding actually selects those samples.

The experiment must **not** use entity/fact retention as main evidence. Store such information only if it already exists from prior pipelines or is needed for qualitative appendix examples. The main evidence is behavior, length, stress robustness, and selector recovery.

---

## 1. Claims under test

### Claim 1 — Structured guidelines improve the compression distribution

Generic compression prompts tend to behave like ordinary summarizers. Agent histories, however, are operational state: they contain runtime bindings, API preconditions, action outcomes, failed attempts, environment state, and decision cues. Structured guideline prompts such as ACON UT / UTCO should induce a better compression distribution than generic summaries.

**Primary test:** compare `Best-of-N Pass@CK` and `pass-per-token@CK` across prompt families.

Expected pattern:

```text
ACON_UT / ACON_UTCO distribution quality >= general_task_agnostic distribution quality.
```

`general_task_aware` may be competitive; if it is, this is useful because it shows task framing can steer the distribution, but it still does not solve policy selection unless its calibration gap is small.

---

### Claim 2 — Structured prompts improve the distribution, not necessarily the selection policy

A structured prompt specifies what the compressed context should look like, but greedy/default decoding selects high-likelihood outputs under the model, not necessarily high-utility outputs for downstream tool use.

**Primary test:** for each prompt family, compute:

```text
Distribution quality Q_dist = Best-of-N Pass@CK
Calibration gap G_calib = Best-of-N Pass@CK - Greedy Pass@CK
```

Expected pattern:

```text
ACON_UTCO has high Q_dist and non-trivial G_calib.
```

This supports the paper framing:

> Structured prompts define a useful compression distribution; behavior reward is needed to optimize the compressor policy inside that distribution.

---

### Claim 3 — Serial recompression robustness matters for long-horizon agents

In real long-horizon compression, compressed histories are not terminal artifacts. An early summary can later be merged with new interactions and compressed again. Therefore a good compressed context should remain behaviorally useful under serial recompression.

Define:

- `C1`: the initial compressed context generated from the raw history.
- `CK`: the context after `K=2` additional recompression rounds.

**Primary test:** compare pass rates, fragility, and length drift between `C1` and `CK`.

A compression is **fragile** if:

```text
Pass(C1) = True and Pass(CK) = False.
```

Expected pattern:

```text
Greedy has non-trivial fragility.
Best-of-N / Best-CK should be more robust.
```

This motivates stress-aware training targets and rewards:

\[
R_K(c) = U(T^K(c)) - \lambda |c|.
\]

---

### Claim 4 — Verbal selectors are not behavior reward

If another LLM judge could reliably select the best compression from samples, then policy optimization might be unnecessary. Prior v10 evidence suggests pointwise and pairwise MiniMax verbal selectors recover little oracle headroom. In v11, verbal selectors should be evaluated as **negative baselines**, not as proposed solutions.

**Primary test:** compare greedy, random, shortest, pointwise verifier, pairwise selector, continuation-entropy selector, and oracle best-of-N.

Expected pattern:

```text
verbal selector recovery < 50% of oracle gain, especially at CK.
```

Interpretation:

> More judging/prompting is not enough; behavior is the reward signal.

---

## 2. Non-goals

Do **not** make these part of the main experiment:

1. No entity/fact-retention main metrics.
2. No recovered-then-dropped audit loop.
3. No Qwen SFT or GRPO in v11.
4. No policy training.
5. No runtime retrieval, fact table, or projection pipeline.
6. No claim that any surface chunk type such as `entity`, `causal`, or `narrative` is universally important.
7. No rewritten ACON prompts. ACON prompt families must load official ACON templates.

v11 is a **motivation experiment**, not a method experiment. Its purpose is to make the paper’s motivation section presentation-ready.

---

## 3. Dataset and case pool

### 3.1 Benchmark

Use the **full AppWorld dev split**.

Do not use the previous 30 length-biased selected cases as the primary set. They may only appear as an appendix comparison if already available.

### 3.2 Baseline full-context run

For every dev task:

1. Run the fixed downstream agent with no compression / full context.
2. Store the full trajectory, success flag, score, number of steps, rendered trajectory text, and output directory.
3. If a full-context run already exists and is compatible with the current downstream agent version and prompt, reuse it only if provenance is exact. Otherwise rerun.

### 3.3 Primary analysis set

Primary analysis should include:

```text
all dev tasks where the full-context baseline succeeds.
```

Reason: if full context itself fails, compressed-context failure is not a clean signal of compression quality.

### 3.4 Secondary analysis set

Also report an all-dev secondary table:

```text
all dev tasks, including full-context failures.
```

But do not mix secondary all-dev results into main figures.

### 3.5 Stratification

For the primary set, report case distribution by:

- trajectory length bucket: `short`, `medium`, `long`;
- baseline step count bucket;
- task difficulty if available from AppWorld metadata;
- app/domain if available.

Suggested length buckets:

```python
short:  baseline_steps < 15
medium: 15 <= baseline_steps < 25
long:   baseline_steps >= 25
```

Save:

```text
outputs/tables/case_pool_summary.csv
outputs/raw/full_context_baseline_runs.jsonl
outputs/data/primary_dev_cases.jsonl
outputs/data/secondary_all_dev_cases.jsonl
```

---

## 4. Models and roles

### 4.1 Main model roles

| Role | Model | Notes |
|---|---|---|
| Compressor | `MiniMaxAI/MiniMax-M2.5` | Main compressor for all prompt families. |
| Downstream agent | `MiniMaxAI/MiniMax-M2.5` | Fixed AppWorld agent used for behavior evaluation. |
| Verbal selector / verifier | `MiniMaxAI/MiniMax-M2.5` | Negative baseline selector, not ground truth. |

### 4.2 Why not Qwen in v11?

Do not use Qwen3-4B as a verifier, auditor, or selector in v11. Qwen may be a future student compressor in method experiments, but v11 is a prompt-family behavior study using MiniMax for consistent compression and evaluation.

### 4.3 Generation settings

#### Greedy compression

```yaml
temperature: 0.0
seed: 42
max_tokens: 2048
top_p: 1.0
```

#### Stochastic compression samples

```yaml
temperature: 0.7
seeds: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
N_SAMPLES: 8
max_tokens: 2048
top_p: 1.0
```

#### Serial recompression stress

```yaml
temperature: 0.0
seed: 42
K_STRESS: 2
max_tokens: 2048
```

#### Behavior evaluation

Use the same downstream agent runner as prior v9/v10 runs.

```yaml
cap_steps: 15
seed: 42
```

If v9/v10 used a different behavior step cap for a critical baseline, preserve the same cap and document it.

---

## 5. Prompt families

### 5.1 `general_task_agnostic`

Purpose: generic summarization/compression control.

System:

```text
You are a careful context compression module.
Return only the compressed context. Do not include explanations about your compression process.
```

User template:

```text
Compress the following interaction history into a shorter version.

Hard budget:
- The compressed context should be no more than {max_chars} characters.

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

Set:

```yaml
max_chars: 2000
```

Rationale: prior ACON-style compressed outputs were around this scale. Exact length will still be measured and controlled by pass-per-token / length tables.

---

### 5.2 `general_task_aware`

Purpose: generic compression with non-oracle task framing.

Important: this prompt may use only the original user task / continuation setting. It must not include future suffix, needed facts, counterfactual labels, or oracle failure information.

System:

```text
You are a careful context compression module for a tool-use agent.
Return only the compressed context. Do not include explanations about your compression process.
```

User template:

```text
Compress the previous interaction history into a shorter context for a downstream tool-use agent.

The downstream agent will continue the following task:
{task_instruction}

Hard budget:
- The compressed context should be no more than {max_chars} characters.

Compression goals:
- Preserve information that may help the downstream agent continue the task correctly.
- Preserve exact identifiers, API names, parameter names, file paths, dates, amounts, auth values, object IDs, and state-changing action outcomes when they may matter.
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

Set:

```yaml
max_chars: 2000
```

---

### 5.3 `ACON_UT`

Purpose: official ACON utility-optimized structured history-compression prompt.

Load from official `microsoft/acon` repository:

```text
experiments/appworld/prompts/context_opt/prompt_history_v2.jinja
```

Also load official ACON system prompt:

```text
experiments/appworld/prompts/context_opt/system_prompt.jinja
```

Rendering convention:

```python
rendered = jinja2.from_string(template_text).render(
    task=task_instruction,
    prev_summary="",
    history=context,
    max_chars=2000,  # likely no-op for official template, still pass for provenance
)
```

Do not edit the prompt text. If the template does not use `max_chars`, record this in provenance.

---

### 5.4 `ACON_UTCO`

Purpose: official ACON utility + compression optimized structured prompt.

Load from official `microsoft/acon` repository:

```text
experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja
```

Use the same system prompt as above.

Rendering convention:

```python
rendered = jinja2.from_string(template_text).render(
    task=task_instruction,
    prev_summary="",
    history=context,
    max_chars=2000,  # likely no-op for official template, still pass for provenance
)
```

Do not edit the prompt text.

---

## 6. Prompt provenance requirements

The coding agent must record:

```text
outputs/provenance/acon_repo_commit.txt
outputs/provenance/acon_ut_prompt.txt
outputs/provenance/acon_utco_prompt.txt
outputs/provenance/acon_system_prompt.txt
outputs/provenance/prompt_sha256.json
outputs/provenance/rendered_prompt_examples/{prompt_family}/{task_id}.txt
outputs/provenance/general_prompt_templates.md
```

For every prompt family, compute SHA256 of:

1. canonical system prompt;
2. canonical user template;
3. at least three rendered prompt examples.

If official ACON repo path is missing, stop and report failure. Do not silently substitute an ACON-like prompt.

---

## 7. Candidate generation

For every primary dev case and every prompt family:

### 7.1 Greedy candidate

Generate exactly one greedy output:

```text
candidate_type = greedy
temperature = 0.0
seed = 42
sample_id = -1
```

### 7.2 Stochastic samples

Generate `N=8` independent samples:

```text
candidate_type = sample
temperature = 0.7
seed = 1000 + sample_id
sample_id in [0, 7]
```

### 7.3 Candidate schema

Write one JSONL row per C1 candidate:

```json
{
  "task_id": "...",
  "prompt_family": "ACON_UTCO",
  "candidate_id": "taskid__ACON_UTCO__sample_03",
  "candidate_type": "greedy | sample",
  "sample_id": 3,
  "temperature": 0.7,
  "seed": 1003,
  "task_instruction": "...",
  "source_context_chars": 18000,
  "c1_text": "...",
  "c1_chars": 1234,
  "c1_tokens_est": 309,
  "generation_error": null,
  "prompt_sha256": "...",
  "model": "MiniMaxAI/MiniMax-M2.5"
}
```

Output:

```text
outputs/raw/compression_candidates_c1.jsonl
```

---

## 8. Serial recompression stress: C1 → CK

### 8.1 Motivation

Serial recompression simulates a real long-horizon agent setting where an old compressed history is later re-entered into the context and compressed again together with new interactions. A good compression should be robust under this repeated maintenance process.

### 8.2 Procedure

For every C1 candidate:

```python
x = c1_text
stress_chain = [x]
for r in range(1, K_STRESS + 1):
    x = compress(prompt_family, context=x, task_instruction=task_instruction,
                 temperature=0.0, seed=42)
    stress_chain.append(x)
ck_text = stress_chain[-1]
```

Use the **same prompt family** for recompression. For ACON prompts, feed the previous compressed text into the `history` field with `prev_summary=""`, matching the prior primary-mode convention.

### 8.3 Stress chain schema

Write one row per candidate per stress round:

```json
{
  "task_id": "...",
  "prompt_family": "ACON_UTCO",
  "candidate_id": "...",
  "round": 0,
  "round_name": "C1 | stress_1 | CK",
  "context_text": "...",
  "chars": 1234,
  "tokens_est": 309,
  "text_similarity_to_prev": null,
  "length_drift_from_c1_pct": 0.0,
  "exact_same_as_prev": false,
  "generation_error": null
}
```

Output:

```text
outputs/raw/stress_chains.jsonl
```

---

## 9. Behavior evaluation

### 9.1 Contexts to evaluate

Evaluate downstream AppWorld continuation with:

1. `C1` context for every candidate;
2. `CK` context for every candidate.

Do not evaluate intermediate stress rounds unless explicitly enabled by `EVAL_INTERMEDIATE_STRESS=true`. For primary v11, C1 and CK are enough.

### 9.2 Downstream prompt

Use the same downstream-agent compressed-context instruction as prior v9/v10 runs:

```text
You are given compressed context from previous interaction:

{compressed_context}

Continue solving the task.

Rules:
1. Use exact IDs, values, and bindings from the compressed context when reliable.
2. If critical information is missing or ambiguous, call tools to verify it.
3. Avoid modifying unrelated objects or causing collateral damage.
4. Do not repeat completed state-changing actions unless necessary and safe.
5. Prefer fewer tool calls, but correctness is more important.
6. Stop only when the task is complete.

You have at most {max_steps} action steps.
```

Set:

```yaml
max_steps: 15
```

### 9.3 Behavior run schema

Write one row per candidate × eval round:

```json
{
  "task_id": "...",
  "prompt_family": "ACON_UTCO",
  "candidate_id": "...",
  "candidate_type": "greedy | sample",
  "sample_id": 3,
  "eval_round": "C1 | CK",
  "compressed_context_chars": 1234,
  "compressed_context_tokens_est": 309,
  "success": true,
  "score": 1.0,
  "iterations": 10,
  "termination_reason": "task_completed",
  "error": null,
  "output_dir": "...",
  "total_input_tokens": 12345,
  "output_tokens": 678,
  "elapsed_s": 42.0
}
```

Output:

```text
outputs/raw/behavior_runs_c1_ck.jsonl
```

---

## 10. Selectors

Selectors choose one candidate per `(task_id, prompt_family, eval_round)` from the candidate set.

### 10.1 `greedy`

Select the greedy candidate.

### 10.2 `random_sample`

Select one of the 8 stochastic samples using fixed selector seed:

```yaml
selector_seed: 20260531
```

### 10.3 `shortest_sample`

Select the shortest stochastic sample at the relevant eval round. Ties broken by lower sample_id.

### 10.4 `oracle_best_of_n`

Analysis-only upper bound. Select among the 8 stochastic samples by:

1. highest behavior score at the relevant eval round;
2. success over failure if score unavailable;
3. shorter context length;
4. lower sample_id.

This selector uses true behavior and must never be described as deployable.

### 10.5 `best_c1`

Select the sample with best C1 behavior, then evaluate it both at C1 and CK.

Purpose: tests what happens if training targets are selected by one-step success.

### 10.6 `best_ck`

Select the sample with best CK behavior, then evaluate it both at C1 and CK.

Purpose: tests stress-aware target selection.

### 10.7 `pointwise_verifier`

A negative baseline. Ask MiniMax to score each candidate without knowing true behavior.

System:

```text
You are a strict evaluator of compressed context for a tool-use agent.
Return only JSON. Do not include prose outside JSON.
```

User template:

```text
You will judge whether a compressed context is sufficient for a downstream AppWorld tool-use agent to continue the task.

Original task:
{task_instruction}

Compressed context:
{compressed_context}

Evaluate the compressed context for downstream task success.
Do not assume facts not present in the compressed context.
Do not reward verbosity. A shorter context is better if it preserves the necessary information.

Return JSON with:
{
  "sufficiency_score": 0.0 to 1.0,
  "risk_score": 0.0 to 1.0,
  "missing_critical_information": ["..."],
  "likely_to_succeed": true or false,
  "one_sentence_reason": "..."
}
```

Selection score:

```python
selector_score = sufficiency_score - 0.25 * risk_score - 0.02 * length_kchars
```

Do not tune this on test data.

Output:

```text
outputs/raw/pointwise_verifier_scores.jsonl
```

### 10.8 `pairwise_verifier`

A stronger negative baseline. Pairwise tournament among the 8 samples.

System:

```text
You are a strict pairwise evaluator of compressed contexts for a tool-use agent.
Return only JSON.
```

User template:

```text
A downstream AppWorld tool-use agent will continue the task using one of two compressed contexts.

Original task:
{task_instruction}

Compressed context A:
{context_a}

Compressed context B:
{context_b}

Choose the context that is more likely to let the downstream agent complete the task.
Prefer shorter context only when both seem equally sufficient.
Do not assume facts not present in the context.

Return JSON:
{
  "winner": "A" or "B" or "tie",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence"
}
```

Tournament procedure:

```python
current = sample_0
for sample_i in sample_1..sample_7:
    current = pairwise_winner(current, sample_i)
return current
```

Output:

```text
outputs/raw/pairwise_verifier_matches.jsonl
outputs/tables/pairwise_selector_by_case.csv
```

### 10.9 `continuation_entropy_selector`

Optional but recommended as a proxy baseline inspired by memory/belief-uncertainty work. This is **not** our method.

For each candidate, sample `M=5` short diagnostic continuations from MiniMax using the compressed context.

System:

```text
You are diagnosing whether a compressed context gives a tool-use agent a clear next-step state.
Return only JSON.
```

User template:

```text
Given the original task and compressed context, infer what the downstream agent should do next.

Original task:
{task_instruction}

Compressed context:
{compressed_context}

Return JSON:
{
  "next_action_type": "...",
  "required_arguments": {"arg": "value or null"},
  "missing_information": ["..."],
  "confidence": "high | medium | low"
}
```

Compute disagreement features across the `M=5` samples:

```text
next_action_type_entropy
argument_key_jaccard_distance
missing_info_count_variance
confidence_entropy
```

Selector score:

```python
entropy_score = - next_action_type_entropy \
                - argument_key_jaccard_distance \
                - 0.25 * missing_info_count_variance \
                - confidence_entropy \
                - 0.02 * length_kchars
```

Select the candidate with highest entropy_score.

Output:

```text
outputs/raw/continuation_entropy_samples.jsonl
outputs/tables/continuation_entropy_selector_by_case.csv
```

---

## 11. Main metrics

### 11.1 Distribution quality

For each prompt family:

```python
Q_dist_C1 = oracle_best_of_n_pass_rate_C1
Q_dist_CK = oracle_best_of_n_pass_rate_CK
```

### 11.2 Decoding calibration gap

```python
G_calib_C1 = oracle_best_of_n_pass_rate_C1 - greedy_pass_rate_C1
G_calib_CK = oracle_best_of_n_pass_rate_CK - greedy_pass_rate_CK
```

### 11.3 Calibration ratio

```python
calibration_ratio_CK = greedy_pass_rate_CK / max(oracle_best_of_n_pass_rate_CK, 1e-8)
```

### 11.4 Pass per token / length-normalized utility

```python
pass_per_1k_chars = pass_rate / (mean_chars / 1000.0)
```

Also report mean and median compressed chars for every selector.

### 11.5 Serial recompression fragility

For every selector or generation type:

```python
fragile = (pass_C1 == 1 and pass_CK == 0)
robust_pass = (pass_C1 == 1 and pass_CK == 1)
stress_improved = (pass_C1 == 0 and pass_CK == 1)
robust_fail = (pass_C1 == 0 and pass_CK == 0)
fragility_rate = count(fragile) / max(count(pass_C1 == 1), 1)
stress_delta_pp = 100 * (pass_rate_CK - pass_rate_C1)
```

### 11.6 Stress invariance / fixed-point landing

For every candidate:

```python
length_drift_pct = (chars_CK - chars_C1) / max(chars_C1, 1)
text_similarity_C1_CK = difflib.SequenceMatcher(None, c1_text, ck_text).ratio()
exact_fixed_point = (normalize(c1_text) == normalize(ck_text))
```

Aggregate by prompt family and selector.

### 11.7 Selector recovery of oracle gain

For selector `s`:

```python
recovery_CK = (pass_rate_s_CK - pass_rate_greedy_CK) / max(pass_rate_oracle_CK - pass_rate_greedy_CK, 1e-8)
```

A selector can have negative recovery if it performs worse than greedy.

### 11.8 Pass@N curve

Using the 8 stochastic samples in fixed seed order, compute:

```python
Pass@N = fraction of cases where at least one of first N samples passes
N in {1, 2, 4, 8}
```

Do this separately for C1 and CK.

### 11.9 Better-than-greedy mass estimate

Let `W_N` be oracle win rate: probability that at least one of N samples beats greedy.

Estimate:

\[
\hat p = 1 - (1-W_N)^{1/N}.
\]

This estimates the per-sample probability mass of better-than-greedy outputs, assuming independent samples.

---

## 12. Tables to write

### 12.1 Case pool

```text
outputs/tables/case_pool_summary.csv
```

Columns:

```text
n_dev_tasks
n_full_context_success
n_primary_cases
n_secondary_cases
success_rate_full_context
short_count
medium_count
long_count
```

---

### 12.2 Prompt-family behavior summary

```text
outputs/tables/prompt_family_behavior_summary.csv
```

Columns:

```text
prompt_family
selector                 # greedy, random, shortest, oracle_best_n, best_c1, best_ck, etc.
eval_round               # C1 or CK
n_cases
pass_rate
score_mean
score_std
mean_chars
median_chars
pass_per_1k_chars
mean_iterations
error_rate
```

---

### 12.3 Distribution quality and calibration gap

```text
outputs/tables/distribution_quality_calibration_gap.csv
```

Columns:

```text
prompt_family
Q_dist_C1
Q_dist_CK
greedy_pass_C1
greedy_pass_CK
G_calib_C1
G_calib_CK
calibration_ratio_CK
oracle_len_CK
greedy_len_CK
length_ratio_oracle_over_greedy_CK
```

---

### 12.4 UT vs UTCO headroom

```text
outputs/tables/ut_vs_utco_headroom.csv
```

Columns:

```text
prompt_family          # ACON_UT or ACON_UTCO
greedy_C1
oracle_C1
headroom_C1
greedy_CK
oracle_CK
headroom_CK
greedy_len_CK
oracle_len_CK
```

---

### 12.5 Pass@N curve

```text
outputs/tables/pass_at_n_curve.csv
```

Columns:

```text
prompt_family
eval_round
N
pass_at_N
oracle_win_rate_at_N
mean_selected_chars
better_than_greedy_mass_estimate
```

---

### 12.6 Stress invariance by prompt and selector

```text
outputs/tables/stress_invariance_by_prompt_selector.csv
```

Columns:

```text
prompt_family
selector
n_cases
pass_C1
pass_CK
delta_pass_C1_to_CK_pp
fragility_rate
stress_improved_rate
mean_chars_C1
mean_chars_CK
length_drift_pct
text_similarity_C1_CK
exact_fixed_point_rate
pass_per_1k_chars_CK
```

---

### 12.7 Best-C1 vs Best-CK cross evaluation

```text
outputs/tables/best_c1_vs_best_ck_cross_eval.csv
```

Columns:

```text
prompt_family
selector          # greedy, random, best_c1, best_ck, oracle_best_n
selected_by_round # C1 or CK or none
eval_C1_pass
eval_CK_pass
eval_C1_chars
eval_CK_chars
stress_selection_gain_pp
```

`stress_selection_gain_pp` should be:

```python
Pass@CK(best_ck) - Pass@CK(best_c1)
```

---

### 12.8 Selector recovery summary

```text
outputs/tables/selector_recovery_summary.csv
```

Columns:

```text
prompt_family
selector
n_cases
eval_round
selected_pass_rate
selected_mean_chars
greedy_pass_rate
oracle_pass_rate
oracle_gain_pp
selector_gain_pp
oracle_recovery
```

---

### 12.9 Statistical tests

```text
outputs/tables/statistical_tests.csv
```

Include:

- bootstrap 95% CI over tasks for every pass rate;
- bootstrap CI for headroom;
- McNemar or paired bootstrap comparison for greedy vs oracle;
- paired bootstrap comparison for `best_ck` vs `best_c1` at CK;
- bootstrap comparison for ACON prompt families vs general prompts.

---

## 13. Figures to write

All figures must be saved as both PDF and PNG.

### Figure 1 — Prompt-family behavior comparison

```text
outputs/figures/fig_prompt_family_pass_c1_ck.{pdf,png}
```

Bar plot:

- x-axis: prompt family;
- grouped bars: greedy C1, greedy CK, oracle C1, oracle CK;
- annotate mean chars.

---

### Figure 2 — Distribution quality vs calibration gap

```text
outputs/figures/fig_distribution_quality_vs_calibration_gap.{pdf,png}
```

Scatter:

- x-axis: `Q_dist_CK`;
- y-axis: `G_calib_CK`;
- each point: prompt family;
- label all points.

This is the main paper-facing v11 figure.

---

### Figure 3 — Pass@N curve

```text
outputs/figures/fig_pass_at_n_curve.{pdf,png}
```

Line plot:

- x-axis: N = 1,2,4,8;
- y-axis: Pass@N;
- separate lines for prompt families;
- separate panels for C1 and CK.

---

### Figure 4 — Serial recompression robustness

```text
outputs/figures/fig_serial_recompression_fragility.{pdf,png}
```

Show:

- C1→CK pass delta;
- fragility rate;
- length drift.

---

### Figure 5 — Selector recovery

```text
outputs/figures/fig_selector_recovery.{pdf,png}
```

Bar plot:

- selectors on x-axis;
- oracle recovery on y-axis;
- separate panels for C1 and CK;
- draw a horizontal line at 50% recovery.

---

### Optional Figure 6 — Rate–utility frontier

If budget sweep is enabled:

```text
outputs/figures/fig_rate_utility_frontier.{pdf,png}
```

- x-axis: mean compressed chars;
- y-axis: pass rate;
- lines: greedy vs best-of-N, by prompt family.

---

## 14. Candidate bank

v11 must save a complete reusable candidate bank.

```text
outputs/data/full_dev_compression_candidate_bank.jsonl
```

Each row should contain:

```json
{
  "task_id": "...",
  "prompt_family": "ACON_UTCO",
  "candidate_id": "...",
  "candidate_type": "greedy | sample",
  "sample_id": 3,
  "temperature": 0.7,
  "seed": 1003,
  "task_instruction": "...",
  "source_context_hash": "...",
  "c1_text": "...",
  "stress_chain": ["c1", "stress_1", "ck"],
  "ck_text": "...",
  "pass_c1": true,
  "pass_ck": true,
  "score_c1": 1.0,
  "score_ck": 1.0,
  "chars_c1": 1234,
  "chars_ck": 1180,
  "selector_tags": ["oracle_best_ck", "shortest_passing_ck"],
  "output_dir_c1": "...",
  "output_dir_ck": "..."
}
```

This candidate bank is for analysis, reproducibility, and future train-split target construction. Do not train final models on dev-set candidates.

---

## 15. Optional rate–utility budget sweep

If compute allows, run budget sweep for:

```text
general_task_aware
ACON_UTCO
```

Budgets:

```yaml
max_chars: [1000, 1500, 2000]
```

For ACON prompts that do not use `max_chars`, enforce budget only through generation `max_tokens` variants if feasible. If not feasible, skip ACON budget sweep and only report observed length.

The budget sweep is optional and should not block main v11.

---

## 16. Stage-by-stage implementation plan

### Stage 00 — Prepare

- Create output directories.
- Resolve model endpoints.
- Resolve official ACON prompt paths.
- Write provenance files.
- Write config snapshot.

Outputs:

```text
outputs/provenance/*
outputs/config_v11.json
```

---

### Stage 01 — Build full-dev case pool

- Enumerate full AppWorld dev tasks.
- Run or reuse full-context baseline.
- Render successful trajectories.
- Build primary and secondary case files.

Outputs:

```text
outputs/raw/full_context_baseline_runs.jsonl
outputs/data/primary_dev_cases.jsonl
outputs/data/secondary_all_dev_cases.jsonl
outputs/tables/case_pool_summary.csv
```

---

### Stage 02 — Render prompts

- Render one example per prompt family per selected case.
- Save examples and hashes.

Outputs:

```text
outputs/provenance/rendered_prompt_examples/
outputs/provenance/prompt_sha256.json
```

---

### Stage 03 — Generate C1 compression candidates

- Generate greedy + 8 samples per case × prompt family.

Output:

```text
outputs/raw/compression_candidates_c1.jsonl
```

---

### Stage 04 — Serial recompression stress

- Recompress each candidate for K=2 rounds.
- Save chain rows.

Output:

```text
outputs/raw/stress_chains.jsonl
```

---

### Stage 05 — Behavior evaluation

- Run downstream agent for C1 and CK for every candidate.
- Store all behavior runs.

Output:

```text
outputs/raw/behavior_runs_c1_ck.jsonl
```

---

### Stage 06 — Verbal and entropy selectors

- Run pointwise verifier.
- Run pairwise verifier tournament.
- Run continuation entropy selector if enabled.

Outputs:

```text
outputs/raw/pointwise_verifier_scores.jsonl
outputs/raw/pairwise_verifier_matches.jsonl
outputs/raw/continuation_entropy_samples.jsonl
```

---

### Stage 07 — Selection analysis

- Build selector decisions.
- Compute selected pass rates and oracle recovery.

Outputs:

```text
outputs/tables/selector_recovery_summary.csv
outputs/tables/pairwise_selector_by_case.csv
outputs/tables/continuation_entropy_selector_by_case.csv
```

---

### Stage 08 — Distribution quality and calibration gap

- Compute `Q_dist`, `G_calib`, calibration ratio.

Output:

```text
outputs/tables/distribution_quality_calibration_gap.csv
```

---

### Stage 09 — Serial recompression robustness analysis

- Compute fragility, C1→CK delta, length drift, text similarity, exact fixed-point rate.

Output:

```text
outputs/tables/stress_invariance_by_prompt_selector.csv
outputs/tables/best_c1_vs_best_ck_cross_eval.csv
```

---

### Stage 10 — Pass@N and headroom mass

- Compute Pass@1/2/4/8.
- Estimate better-than-greedy mass.

Output:

```text
outputs/tables/pass_at_n_curve.csv
```

---

### Stage 11 — Candidate bank assembly

- Merge candidate text, stress chains, behavior results, selector tags.

Output:

```text
outputs/data/full_dev_compression_candidate_bank.jsonl
```

---

### Stage 12 — Plot figures

- Generate all required figures.

Output:

```text
outputs/figures/*.pdf
outputs/figures/*.png
```

---

### Stage 13 — Write report

Write a paper-facing Markdown report with exactly these sections:

```text
# motivation_v11 Results
## 1. Case Pool
## 2. Structured vs Generic Compression
## 3. Distribution Quality vs Decoding Calibration
## 4. UT vs UTCO: Prompt Optimization vs Policy Headroom
## 5. Serial Recompression Robustness
## 6. Verbal Selectors Are Not Behavior Reward
## 7. Main Figures for the Paper
## 8. What This Motivates for TRACE / Policy Optimization
## 9. Negative Results and Caveats
## 10. Files of Record
```

Output:

```text
outputs/reports/motivation_v11_results_summary.md
```

---

## 17. Success criteria

### Criterion 1 — Structured distribution quality

At least one structured ACON prompt (`ACON_UT` or `ACON_UTCO`) should outperform `general_task_agnostic` by:

```text
Best-of-N Pass@CK >= +10 pp
```

or show better pass-per-token at CK.

If not, weaken the claim to: simple task-aware prompting is sufficient to induce a good distribution on this setting.

---

### Criterion 2 — Policy calibration gap

For `ACON_UTCO` or `ACON_UT`:

```text
Best-of-N Pass@CK - Greedy Pass@CK >= 15 pp
```

and:

```text
oracle_bestN_mean_length_CK <= 1.10 * greedy_mean_length_CK
```

If this fails, the policy-headroom story is weak for full dev.

---

### Criterion 3 — Serial recompression matters

At least one of:

```text
fragility_rate_greedy >= 0.20
```

or:

```text
Pass@CK(Best-CK) - Pass@CK(Best-C1) >= 5 pp
```

If neither holds, CK should be downgraded to a diagnostic rather than a core training principle.

---

### Criterion 4 — Verbal selectors are insufficient

For pointwise / pairwise / entropy selectors:

```text
oracle_recovery_CK < 0.50
```

This is a **positive motivation result**, not a failure: it supports behavior reward over verifier prompting.

---

### Criterion 5 — No length-mediated win

For any main best-of-N result, selected contexts should not be substantially longer than greedy:

```text
selected_mean_chars <= 1.10 * greedy_mean_chars
```

If best-of-N wins only by being longer, the claim must be reframed as budget insufficiency rather than policy miscalibration.

---

## 18. Falsification criteria

The current paper framing should be reconsidered if any of the following happens:

1. Generic prompts and ACON prompts have similar distribution quality and similar calibration gaps.
2. ACON greedy is within 5 pp of best-of-N on CK on the primary full-dev set.
3. Best-of-N wins only by using much longer outputs.
4. Verbal selectors recover most oracle gain, making behavior reward less necessary.
5. C1 and CK outcomes are nearly identical across all prompt families and selectors.

---

## 19. Main paper interpretation guide

### If expected results hold

Write:

> Agent histories are operational state, so structured guidelines help create a useful compression distribution. However, structured prompts do not solve the selection problem: high-utility, serially robust compressions already exist in the distribution, but greedy decoding and verbal selectors fail to choose them. This motivates training the compressor policy with behavior reward under serial recompression stress.

### If structured prompts do not beat general task-aware prompts

Write:

> Task framing alone can induce a strong distribution on AppWorld, but greedy decoding still leaves policy headroom. The method motivation shifts from structured prompt superiority to policy optimization inside any capable compression distribution.

### If UTCO beats UT and has low headroom

Write:

> ACON UTCO solves much of the distribution and selection problem on this subset. Our method should focus on harder cases or student distillation rather than prompt-family headroom.

### If UTCO best-of-N is high but greedy is low

Write:

> UTCO improves distribution quality but leaves policy-level selection miscalibrated. This is the strongest support for TRACE.

---

## 20. Minimal `run_all.sh` outline

```bash
#!/usr/bin/env bash
set -euo pipefail

python scripts/00_prepare.py
python scripts/01_build_full_dev_cases.py
python scripts/02_render_prompts.py
python scripts/03_generate_candidates.py
python scripts/04_serial_recompression_stress.py
python scripts/05_run_behavior_c1_ck.py
python scripts/06_run_verbal_selectors.py
python scripts/07_selection_analysis.py
python scripts/08_distribution_quality_calibration.py
python scripts/09_stress_invariance_analysis.py
python scripts/10_pass_at_n_curve.py
python scripts/11_build_candidate_bank.py
python scripts/12_plot_figures.py
python scripts/13_write_report.py
```

All scripts must be restartable. Each stage should skip completed rows unless `--overwrite` is passed.

---

## 21. Final checklist before declaring v11 complete

- [ ] Full AppWorld dev baseline completed or exactly reused with provenance.
- [ ] Primary case pool is full-context-successful dev tasks.
- [ ] Official ACON UT / UTCO prompts loaded from repo, not rewritten.
- [ ] General prompt templates saved and hashed.
- [ ] Greedy + N=8 samples generated for all prompt families.
- [ ] K=2 stress chains saved for every candidate.
- [ ] C1 and CK behavior runs completed with zero or documented errors.
- [ ] Candidate bank saved with all texts and selector tags.
- [ ] Distribution quality vs calibration gap table and figure produced.
- [ ] UT vs UTCO headroom table produced.
- [ ] Stress invariance table produced.
- [ ] Verbal selector recovery table produced.
- [ ] Main report written with no entity/fact-retention claims in the headline.

---

## 22. Expected paper-facing outputs

The v11 run should produce the following core artifacts for the paper:

1. `fig_distribution_quality_vs_calibration_gap` — main motivation figure.
2. `fig_prompt_family_pass_c1_ck` — structured vs generic behavior figure.
3. `fig_pass_at_n_curve` — evidence that high-utility outputs occupy non-trivial sampling mass.
4. `fig_serial_recompression_fragility` — evidence that CK matters for long-horizon robustness.
5. `fig_selector_recovery` — evidence that verbal selectors cannot replace behavior reward.
6. `full_dev_compression_candidate_bank.jsonl` — reusable evidence and case-study source.

---

## 23. One-sentence summary

v11 should establish, on full AppWorld dev, that structured prompts create capable compression distributions, but long-horizon performance depends on selecting short, serially robust, high-utility compressions from that distribution — a selection problem that greedy decoding and verbal proxies do not solve.
