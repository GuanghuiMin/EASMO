# Motivation v11: Full Train+Dev Behavioral Evidence for Compression Policy Optimization

**Experiment name:** `motivation_v11_final_train_dev_transition_experiment`  
**Scope:** AppWorld train + dev, history compression only  
**Splits:** `train` = 89 tasks available locally, `dev` = 56 tasks, `combined` = 145 tasks  
**Primary model:** MiniMax-M2.5 for compressor, downstream agent, verifier, and selector baselines  
**Main purpose:** produce the final paper-facing motivation figures/tables supporting the transition from structured prompt compression to behavior-optimized compressor policies.

---

## 0. Executive framing

This experiment should be the final motivation consolidation run. It must not introduce a new entity taxonomy or a new heuristic retention metric. The core claim is behavioral:

> Structured prompts can induce useful compression distributions, but they do not solve the selection problem. High-utility, serially robust compressions often exist in the sampling distribution, while greedy decoding and verbal selectors may fail to choose them. Therefore, compressor training should optimize behavior reward rather than summary-likeness, surface labels, or verifier prose.

The new update relative to previous v11 drafts is critical:

> **Do not restrict analysis to full-context-success trajectories.**

ACON reports that compression can sometimes outperform full context, so compression is not only a preservation mechanism. It is a behavior-transforming policy: it can preserve, harm, or rescue tasks. The primary analysis must therefore compare every full-context run against every compressed run using a **full-vs-compressed transition matrix**.

The experiment should answer four paper-level questions:

1. **Do structured prompts produce better compression distributions than generic prompts?**
2. **Does greedy decoding select the best compression from that distribution?**
3. **Do compressed contexts remain useful under serial recompression, as required in long-horizon agents?**
4. **Can verbal selector prompts replace behavior reward, or do they leave most oracle headroom unrecovered?**

---

## 1. Paper motivation sections supported by v11

### 1.1 Agent histories are operational state, not documents

Generic summarization is a poor model of long-horizon tool-use history. AppWorld histories contain runtime bindings, API preconditions, action outcomes, failed attempts, environment state, and decision cues. v11 must compare generic compression prompts with official ACON structured prompts on actual AppWorld behavior.

### 1.2 Structured prompts improve the distribution, not necessarily the selection policy

A prompt family induces a distribution over compressed contexts:

\[
    c \sim \pi_p(c \mid h)
\]

A structured prompt may improve the distribution quality, but greedy decoding is only:

\[
    c_{\text{greedy}} = \arg\max_c \pi_p(c \mid h)
\]

The behavior-optimal compression is instead:

\[
    c^\star = \arg\max_c \left[ U(T^K(c)) - \lambda |c| \right]
\]

v11 decomposes prompt performance into:

\[
    Q_{\text{dist}}(p)=\text{Best-of-N Pass@CK}
\]

and

\[
    G_{\text{calib}}(p)=\text{Best-of-N Pass@CK} - \text{Greedy Pass@CK}.
\]

This separates **distribution quality** from **decoding calibration**.

### 1.3 High-utility compressions hide in the sampling distribution

v9/v10 showed that MiniMax + ACON-UTCO stochastic samples contain much better compressions than greedy. v11 must test this at train+dev scale, report pass@N, harm/rescue decomposition, length, and CK robustness.

---

## 2. Key conceptual definitions

### 2.1 Full-vs-compressed transition matrix

For each task \(i\):

\[
F_i \in \{0,1\}
\]

is the full-context baseline result. For each compressed condition:

\[
C_i^{p,s,r} \in \{0,1\}
\]

where:

- \(p\): prompt family
- \(s\): selector or generation condition
- \(r\): evaluation round, either C1 or CK

Report the 2×2 transition matrix:

| Full context | Compressed context | Name | Meaning |
|---|---|---|---|
| pass | pass | preserved success | Compression preserved a task full context solved. |
| pass | fail | harm | Compression broke a task full context solved. |
| fail | pass | rescue | Compression solved a task full context failed. |
| fail | fail | both fail | Neither context solved the task. |

The main identity:

\[
\Delta_{\text{overall}}
= P(C=1)-P(F=1)
= P(F=0,C=1)-P(F=1,C=0)
\]

That is:

> **overall gain = rescue rate − harm rate**.

This identity should appear in the final paper because it explains how compression can outperform full context.

### 2.2 Serial recompression robustness: C1 vs CK

C1 is the first compressed output:

\[
    c_1 = C(h)
\]

CK is the same compressed output after K deterministic recompression rounds:

\[
    T^K(c_1) = C(C(\cdots C(c_1)))
\]

Use **K = 2** by default, matching v9/v10.

Do not describe CK primarily as an abstract fixed point. In the paper and report, describe it as **serial recompression robustness**:

> In long-horizon agents, a compressed history is not a terminal artifact. It re-enters later contexts and may be compressed again as the interaction grows. A good compression should therefore remain useful after serial recompression.

C1 answers:

> Does the compression work immediately?

CK answers:

> Does the compression remain useful if it is maintained and recompressed later?

### 2.3 Distribution quality and calibration gap

For each prompt family \(p\), selector set, and split:

\[
Q^{all}_{dist}(p)=\text{Best-of-N Pass@CK on all tasks}
\]

\[
G^{all}_{calib}(p)=\text{Best-of-N Pass@CK} - \text{Greedy Pass@CK}
\]

Also decompose by full-context outcome:

\[
Q^{preserve}_{dist}(p)=P(\text{BestN CK}=1 \mid F=1)
\]

\[
Q^{rescue}_{dist}(p)=P(\text{BestN CK}=1 \mid F=0)
\]

and similarly for calibration gaps.

---

## 3. Non-goals and strict constraints

1. **No entity/fact-retention main claim.** Fact retention may be saved as optional appendix diagnostics only. Main evidence is behavior.
2. **No heuristic chunk-type reward claim.** Previous chunk analyses showed surface labels are unstable proxies. Do not claim entity, causal, narrative, or action-outcome chunks are universally important.
3. **No Qwen verifier or auditor.** MiniMax-M2.5 is the only verifier / pairwise judge / continuation entropy judge in this experiment.
4. **No observation compression.** v11 is about history compression only.
5. **No student SFT / GRPO in v11.** v10 handles student warm-up and GRPO readiness. v11 is final motivation evidence.
6. **No dev oracle target leakage into method training.** Train split candidate bank may later be used for teacher target construction. Dev split is analysis/evaluation only.
7. **Use official ACON prompts.** No ACON-like proxy prompt is allowed for ACON_UT or ACON_UTCO.

---

## 4. Data scope

### 4.1 Splits

Use all locally available AppWorld train and dev tasks:

```text
train: 89 tasks available locally
dev:   56 tasks
combined: 145 tasks
```

The official ACON paper refers to 90 AppWorld train tasks. Since the local run has 89, record which task is missing and why.

Output:

```text
outputs/provenance/appworld_task_inventory.csv
```

Schema:

```text
task_id, split, included, exclusion_reason, load_error, notes
```

### 4.2 Primary analysis set

Primary analysis is **all tasks with valid full-context baseline and valid compressed behavior runs**, regardless of whether full context succeeds.

Do not filter to full-context-success tasks.

### 4.3 Split reporting

Every table must report:

```text
train
dev
combined
```

If train and dev trends diverge, the report must say so explicitly.

---

## 5. Models and roles

| Role | Model | Notes |
|---|---|---|
| Compressor | MiniMaxAI/MiniMax-M2.5 | Used for all prompt families. |
| Downstream agent | MiniMaxAI/MiniMax-M2.5 | Fixed downstream AppWorld agent. |
| Verbal selector / verifier | MiniMaxAI/MiniMax-M2.5 | Negative selector baseline, not ground truth. |
| Continuation entropy probe | MiniMaxAI/MiniMax-M2.5 | MMPO-inspired selector baseline. |

Generation settings:

```yaml
greedy_temperature: 0.0
sample_temperature: 0.7
sample_seeds: [1000,1001,1002,1003,1004,1005,1006,1007]
N_SAMPLES: 8
stress_temperature: 0.0
stress_seed: 42
stress_K: 2
compression_max_tokens: 2048
behavior_cap_steps: 15
```

If the MiniMax server has nondeterminism even at temperature 0.0, record it in the caveats and rely on repeated split-level statistics rather than exact reproduction of a single run.

---

## 6. Prompt families

Run four prompt families.

### 6.1 `general_task_agnostic`

Purpose: generic summary prior control.

System:

```text
You are a careful context compression module.
Return only the compressed context. Do not include explanations about your compression process.
```

User template:

```text
Compress the following interaction history into a shorter version.

Hard constraints:
- Do not invent facts.
- Do not output the original input verbatim.
- Preserve important information.
- Remove redundant, obsolete, or irrelevant details.
- Keep exact values only if they appear important in the history.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Interaction history:
{history}

Compressed context:
```

### 6.2 `general_task_aware`

Purpose: general prompt with current task conditioning, but no ACON headings.

This prompt must not include future suffix, known needed facts, oracle action labels, or anything derived from the successful future trajectory.

System:

```text
You are a careful context compression module for a long-horizon tool-use agent.
Return only the compressed context. Do not include explanations about your compression process.
```

User template:

```text
Compress the previous interaction history into a shorter context for a downstream tool-use agent.

The agent is continuing this task:
{task_instruction}

Compression goals:
- Preserve information that may help the agent continue the task correctly.
- Preserve exact identifiers, API names, parameter names, file paths, dates, amounts, access/auth values, object IDs, and state-changing action outcomes when they may matter.
- Preserve failed attempts or negative evidence when it may prevent repeated mistakes.
- Remove redundant, obsolete, or irrelevant details.
- Do not invent facts.
- Do not solve the task.
- Do not output the original input verbatim.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Previous interaction history:
{history}

Compressed context:
```

### 6.3 `ACON_UT`

Purpose: official ACON structured utility-optimized history compression prompt.

Load from the official `microsoft/acon` repo. Do not rewrite.

Expected source path from prior runs:

```text
experiments/appworld/prompts/context_opt/prompt_history_v2.jinja
```

Render convention:

```python
rendered = jinja2.from_string(template_text).render(
    task=task_instruction,
    prev_summary="",
    history=history,
    max_chars=max_chars,   # no-op if template does not reference it
)
```

### 6.4 `ACON_UTCO`

Purpose: official ACON structured utility + compression optimized history compression prompt.

Load from the official `microsoft/acon` repo. Do not rewrite.

Expected source path from prior runs:

```text
experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja
```

Render convention is identical to `ACON_UT`.

### 6.5 ACON prompt provenance outputs

Save:

```text
outputs/provenance/acon_repo_commit.txt
outputs/provenance/acon_ut_prompt.txt
outputs/provenance/acon_utco_prompt.txt
outputs/provenance/acon_system_prompt.txt
outputs/provenance/acon_prompt_sha256.json
outputs/provenance/rendered_prompt_examples/{prompt_family}_{task_id}.txt
```

The report must include prompt SHA256 values and the repo commit.

---

## 7. Evaluation boundary and compression input

This experiment is about **history compression**.

For each task, build a history compression input from the raw agent interaction history. Use the existing AppWorld/ACON runner if it already supports online history-compression triggers. Otherwise use the following standardized boundary protocol.

### 7.1 Preferred online-style boundary protocol

1. Run the full-context baseline on each task and save every intermediate state/history.
2. Identify the first step \(t\) where rendered history length exceeds the AppWorld history compression threshold, preferably `T_hist = 4096` tokens/chars as implemented in the local ACON runner.
3. If the task never exceeds the threshold, mark `compression_triggered=False` and still include the task in all-task overall metrics, but exclude it from prompt-family compression-comparison tables unless a fallback boundary is explicitly used.
4. Compression input is the rendered history up to step \(t\), optionally keeping the latest action/observation pair exactly if the local runner follows ACON's tail-preservation convention.
5. Evaluate compressed candidates by restoring the environment to step \(t\) and continuing with the compressed history.

### 7.2 Fallback trajectory-derived protocol

If environment checkpoint continuation is not available:

1. Use the rendered full baseline trajectory as `history` for compression, regardless of whether the full baseline succeeds or fails.
2. Run the same downstream evaluation protocol used in v9/v10 with the compressed context.
3. Mark `evaluation_protocol="trajectory_derived"` in every row.
4. Interpret rescue cases carefully: they indicate that a compressed representation of the observed trajectory can guide a rerun better than the original full context, but they are not identical to online checkpoint continuation.

The report must state which protocol was used.

Output:

```text
outputs/raw/compression_boundaries.jsonl
```

Schema:

```json
{
  "task_id": "...",
  "split": "train|dev",
  "task_instruction": "...",
  "evaluation_protocol": "checkpoint_continuation|trajectory_derived",
  "full_success": true,
  "full_score": 1.0,
  "full_steps": 12,
  "compression_triggered": true,
  "boundary_step": 8,
  "history_chars": 9123,
  "history_tokens": 2280,
  "history_text": "...",
  "latest_observation_text": "... optional ...",
  "notes": ""
}
```

---

## 8. Candidate generation

For each task boundary and prompt family:

1. Generate one greedy compression:

```text
temperature=0.0, seed=42
```

2. Generate N=8 stochastic samples:

```text
temperature=0.7, seeds=1000..1007
```

Each output is C1.

Output:

```text
outputs/raw/candidate_compressions_c1.jsonl
```

Schema:

```json
{
  "task_id": "...",
  "split": "train|dev",
  "prompt_family": "ACON_UTCO",
  "candidate_type": "greedy|sample",
  "sample_id": 0,
  "temperature": 0.7,
  "seed": 1000,
  "input_history_chars": 9123,
  "c1_text": "...",
  "c1_chars": 1200,
  "c1_tokens_est": 300,
  "compression_error": null,
  "prompt_sha256": "..."
}
```

---

## 9. Serial recompression stress

For every C1 candidate, run K=2 deterministic recompression rounds with the same prompt family and the same task instruction.

```text
c0 = C1 output
c1 = recompress(c0)
c2 = recompress(c1) = CK output
```

Use:

```text
temperature=0.0, seed=42
```

Output:

```text
outputs/raw/stress_chains.jsonl
```

Schema:

```json
{
  "task_id": "...",
  "split": "train|dev",
  "prompt_family": "ACON_UTCO",
  "candidate_type": "sample",
  "sample_id": 3,
  "round": 0,
  "text": "...",
  "chars": 1234,
  "tokens_est": 309,
  "text_hash": "..."
}
```

Round 0 is C1. Round K is CK.

Also compute text-level stress metrics:

```text
length_drift_pct = (chars_CK - chars_C1) / max(chars_C1, 1)
text_similarity_C1_CK
exact_text_fixed_point = hash(CK) == hash(previous_round)
```

---

## 10. Behavior evaluation

Evaluate every candidate at both C1 and CK.

Output:

```text
outputs/raw/behavior_runs.jsonl
```

Schema:

```json
{
  "task_id": "...",
  "split": "train|dev",
  "prompt_family": "ACON_UTCO",
  "candidate_type": "sample",
  "sample_id": 3,
  "eval_round": "C1|CK",
  "compressed_context_text_hash": "...",
  "success": true,
  "score": 1.0,
  "num_steps": 8,
  "termination_reason": "task_completed",
  "error": null,
  "compressed_chars": 1180,
  "compressed_tokens_est": 295,
  "full_success": false,
  "full_score": 0.0,
  "evaluation_protocol": "checkpoint_continuation|trajectory_derived"
}
```

The downstream agent prompt must be identical across all compressed candidates. Only the compressed context changes.

---

## 11. Selectors

Selectors are applied per `(task_id, prompt_family, eval_round)` over the greedy and sampled candidates.

### 11.1 `greedy`

Select the deterministic temperature 0 candidate.

### 11.2 `random_sample_mean`

Not a single selected context. Report the mean pass/score/length over all 8 stochastic samples.

### 11.3 `random_sample_fixed`

Optional single-sample baseline, select `sample_id=0`.

### 11.4 `shortest_sample`

Select the shortest stochastic sample, independent of behavior.

### 11.5 `oracle_best_of_N_C1`

For eval round C1, select the sample with highest true C1 score. Tie-break by shortest C1 length, then smaller sample_id.

### 11.6 `oracle_best_of_N_CK`

For eval round CK, select the sample with highest true CK score. Tie-break by shortest CK length, then smaller sample_id.

This is the key oracle selector for distribution quality.

### 11.7 `pointwise_minimax_verifier`

MiniMax verbal scoring of each candidate. This is a negative baseline, not ground truth.

System:

```text
You are a strict evaluator of compressed context for a tool-use agent.
Return only valid JSON.
```

User:

```text
You are given a task and a compressed context that will be used by an AppWorld tool-use agent.

Judge whether the compressed context is sufficient for the agent to continue the task correctly.
Do not judge writing style. Do not reward verbosity. Focus on whether the context contains the operational information required for correct future tool use.

Task:
{task_instruction}

Compressed context:
{compressed_context}

Return JSON:
{
  "sufficiency_score": 0.0,
  "missing_critical_information": ["..."],
  "risk_of_repeating_failed_steps": 0.0,
  "risk_of_wrong_api_call": 0.0,
  "risk_of_wrong_object_or_id": 0.0,
  "one_sentence_reason": "..."
}
```

Select highest `sufficiency_score`; tie-break shortest length.

### 11.8 `pairwise_minimax_selector`

MiniMax pairwise tournament over 8 samples. This is also a negative baseline.

Pairwise prompt:

```text
You are comparing two compressed contexts for the same AppWorld tool-use task.

Choose the context that is more likely to let the downstream agent continue correctly.
Ignore writing style unless it affects execution. Prefer shorter context only if both are equally sufficient.

Task:
{task_instruction}

Context A:
{context_a}

Context B:
{context_b}

Return JSON only:
{
  "winner": "A|B|tie",
  "confidence": 0.0,
  "reason": "one short sentence"
}
```

Tournament procedure:

1. Randomized but deterministic bracket with seed 42.
2. If tie, choose shorter context.
3. Save all pairwise comparisons.

### 11.9 `continuation_entropy_selector`

MMPO-inspired selector baseline. It is not a method claim.

For each candidate, ask MiniMax `M=5` times at temperature 0.7:

```text
Given the task and compressed context, infer what the agent should do next.
Do not solve the task. Return JSON only.

Task:
{task_instruction}

Compressed context:
{compressed_context}

Return JSON:
{
  "next_action_type": "...",
  "likely_required_arguments": {"...": "..."},
  "missing_information_before_safe_action": ["..."],
  "confidence": "high|medium|low"
}
```

Compute:

```text
next_action_type_entropy
argument_key_jaccard_disagreement
missing_info_count_variance
confidence_entropy
composite_entropy
```

Select lowest composite entropy. Tie-break shortest length.

---

## 12. Core outputs

### 12.1 Full context baselines

```text
outputs/raw/full_context_runs.jsonl
```

Schema:

```json
{
  "task_id": "...",
  "split": "train|dev",
  "full_success": true,
  "full_score": 1.0,
  "full_steps": 14,
  "full_peak_tokens": 9000,
  "full_total_tokens": 50000,
  "termination_reason": "task_completed",
  "error": null
}
```

### 12.2 Full candidate bank

```text
outputs/data/full_train_dev_compression_candidate_bank.jsonl
```

One row per candidate with C1, CK, behavior, selector tags, and stress chain metadata.

Schema:

```json
{
  "task_id": "...",
  "split": "train|dev",
  "prompt_family": "ACON_UTCO",
  "candidate_type": "greedy|sample",
  "sample_id": 3,
  "temperature": 0.7,
  "seed": 1003,
  "task_instruction": "...",
  "evaluation_protocol": "checkpoint_continuation|trajectory_derived",
  "full_success": false,
  "full_score": 0.0,
  "c1_text": "...",
  "ck_text": "...",
  "stress_chain_hashes": ["...", "...", "..."],
  "pass_c1": true,
  "score_c1": 1.0,
  "pass_ck": true,
  "score_ck": 1.0,
  "chars_c1": 1220,
  "chars_ck": 1185,
  "length_drift_pct": -0.029,
  "selector_tags": ["oracle_best_ck", "shortest_passing_ck"]
}
```

Do not use dev rows for method training.

---

## 13. Main tables

### 13.1 `transition_matrix_by_prompt_selector_round.csv`

Primary paper table.

Rows:

```text
split,
prompt_family,
selector,
eval_round,
n_tasks,
full_pass_rate,
compressed_pass_rate,
overall_delta_pp,
preserved_success_count,
harm_count,
rescue_count,
both_fail_count,
preserve_success_rate,
harm_rate,
rescue_rate,
net_gain_pp,
mean_chars,
median_chars,
pass_per_1k_chars
```

Definitions:

```text
preserve_success_rate = P(C=1 | F=1)
harm_rate             = P(F=1, C=0) over all tasks
rescue_rate           = P(F=0, C=1) over all tasks
net_gain_pp           = 100 * (rescue_rate - harm_rate)
overall_delta_pp      = 100 * (compressed_pass_rate - full_pass_rate)
```

Check that `overall_delta_pp ≈ net_gain_pp`.

### 13.2 `distribution_quality_calibration_gap.csv`

Rows:

```text
split,
prompt_family,
eval_round,
greedy_pass_all,
bestn_pass_all,
q_dist_all,
gap_all_pp,
greedy_pass_given_full_success,
bestn_pass_given_full_success,
q_dist_preserve,
gap_preserve_pp,
greedy_pass_given_full_fail,
bestn_pass_given_full_fail,
q_dist_rescue,
gap_rescue_pp,
greedy_harm_rate,
bestn_harm_rate,
greedy_rescue_rate,
bestn_rescue_rate,
greedy_mean_chars,
bestn_mean_chars
```

### 13.3 `serial_recompression_transition.csv`

Rows:

```text
split,
prompt_family,
selector,
n_tasks,
pass_C1,
pass_CK,
delta_pass_C1_to_CK_pp,
fragility_rate,
stress_improvement_rate,
fragile_preserve_count,
stable_preserve_count,
fragile_rescue_count,
stable_rescue_count,
mean_chars_C1,
mean_chars_CK,
length_drift_pct,
text_similarity_C1_CK,
exact_fixed_point_rate
```

Definitions:

```text
fragility_rate = P(CK=0 | C1=1)
stress_improvement_rate = P(CK=1 | C1=0)
fragile_rescue = F=0, C1=1, CK=0
stable_rescue  = F=0, C1=1, CK=1
fragile_preserve = F=1, C1=1, CK=0
stable_preserve  = F=1, C1=1, CK=1
```

### 13.4 `ut_vs_utco_headroom.csv`

Rows:

```text
split,
eval_round,
metric,
ACON_UT_greedy,
ACON_UT_bestN,
ACON_UT_headroom_pp,
ACON_UTCO_greedy,
ACON_UTCO_bestN,
ACON_UTCO_headroom_pp,
UTCO_minus_UT_greedy_pp,
UTCO_minus_UT_bestN_pp
```

This table answers:

> Does UTCO improve the distribution, the greedy output, or both?

### 13.5 `pass_at_n_curve.csv`

For N in `{1,2,4,8}` where N refers to stochastic samples, not greedy.

Rows:

```text
split,
prompt_family,
eval_round,
N,
pass_at_N,
oracle_win_rate_vs_greedy,
estimated_better_than_greedy_mass,
mean_selected_chars,
median_selected_chars
```

Estimated better-than-greedy mass:

\[
\hat p = 1 - (1-W_N)^{1/N}
\]

where \(W_N\) is oracle win rate with N samples.

### 13.6 `selector_transition_summary.csv`

Rows:

```text
split,
prompt_family,
selector,
eval_round,
overall_pass,
harm_rate,
rescue_rate,
net_gain_pp,
mean_chars,
oracle_recovery_overall,
oracle_recovery_net_gain
```

Oracle recovery:

\[
\frac{Pass(selector)-Pass(greedy)}{Pass(oracle)-Pass(greedy)}
\]

and similarly for net gain.

### 13.7 `bootstrap_confidence_intervals.csv`

Use paired bootstrap over task IDs. Use 2,000 resamples.

Rows:

```text
split,
comparison_name,
metric,
mean_diff,
ci_low,
ci_high,
p_bootstrap_two_sided
```

Required comparisons:

```text
ACON_UTCO bestN CK - ACON_UTCO greedy CK
ACON_UT bestN CK - ACON_UT greedy CK
ACON_UTCO bestN CK - general_task_aware bestN CK
ACON_UTCO greedy CK - general_task_aware greedy CK
ACON_UTCO bestN net_gain - ACON_UTCO greedy net_gain
pairwise selector CK - greedy CK
pairwise selector CK - oracle bestN CK
Best-CK evaluated CK - Best-C1 evaluated CK
```

---

## 14. Main figures

### Figure 1: Prompt family behavior comparison

Bar chart grouped by prompt family:

```text
greedy C1
greedy CK
best-of-N C1
best-of-N CK
```

Include length markers or separate companion plot.

### Figure 2: Distribution quality vs calibration gap

Scatter:

```text
x-axis: Q_dist = Best-of-N Pass@CK
y-axis: G_calib = Best-of-N Pass@CK - Greedy Pass@CK
point: prompt_family
facets: train/dev/combined
```

### Figure 3: Full-vs-compressed transition matrix

For ACON_UTCO greedy and ACON_UTCO best-of-N CK, show 2×2 heatmaps:

```text
F pass/fail × C pass/fail
```

This is the best figure for the harm/rescue framing.

### Figure 4: Serial recompression robustness

C1→CK pass changes for each prompt_family × selector.

Also show fragility rate:

```text
P(CK=0 | C1=1)
```

### Figure 5: Pass@N curve

For each prompt family:

```text
x-axis: N = 1,2,4,8
y-axis: pass@N CK
```

Greedy should be shown as a separate horizontal marker, not as N=1.

### Figure 6: Selector recovery

For each selector:

```text
overall pass
net gain
oracle recovery %
```

This figure should make clear that verbal selectors are negative baselines if they fail.

### Optional Figure 7: Rate–utility frontier

If budget variants are run:

```text
x-axis: mean compressed tokens
y-axis: pass rate
curves: greedy vs best-of-N, C1 vs CK
```

---

## 15. Success criteria

### Criterion 1: Structured prompts improve distribution quality

At least one of ACON_UT or ACON_UTCO should have higher `Best-of-N Pass@CK`, `net_gain`, or `pass_per_1k_chars` than `general_task_agnostic` and preferably `general_task_aware`.

### Criterion 2: Structured prompts can improve all-task behavior

At least one structured condition should satisfy:

\[
\text{rescue rate} > \text{harm rate}
\]

or have overall compressed pass rate at or above full-context baseline.

This criterion is important because ACON reports that compression can outperform no compression in some settings.

### Criterion 3: Calibration headroom remains under structured prompts

For ACON_UTCO or ACON_UT:

\[
\text{Best-of-N CK} - \text{Greedy CK} \geq 15\text{ pp}
\]

or:

\[
\text{Best-of-N net gain} - \text{Greedy net gain} \geq 10\text{ pp}.
\]

### Criterion 4: Best-of-N is not length-mediated

Best-of-N selected outputs should have mean length no more than 110% of greedy, and ideally shorter.

### Criterion 5: Serial recompression matters

At least one of:

```text
greedy fragility_rate >= 20%
Best-CK evaluated at CK > Best-C1 evaluated at CK
nontrivial fragile_rescue_count
```

### Criterion 6: Verbal selectors do not close the gap

Pointwise verifier / pairwise selector / continuation entropy selector recover less than 50% of oracle CK gain or oracle net-gain improvement.

This is a positive motivation result, not a failure.

---

## 16. Falsification criteria

The motivation is weakened if all of the following happen:

1. General prompts and ACON prompts have similar distribution quality.
2. Greedy decoding is within 5 pp of best-of-N across all prompt families.
3. Best-of-N gains are explained by longer contexts.
4. CK does not change behavior relative to C1.
5. Verbal selectors recover most oracle gain.
6. Transition matrices show no meaningful harm/rescue dynamics.

If this occurs, the paper should not claim that behavior reward is necessary for compression policy selection.

---

## 17. Required report structure

The final report should be saved to:

```text
outputs/reports/motivation_v11_results_summary.md
```

It must have these headings:

```markdown
# Motivation v11 Results

## 1. Setup and Scope
## 2. Full-Context Baseline and Task Inventory
## 3. Prompt Families: Generic vs Structured Compression
## 4. Full-vs-Compressed Transition Analysis
## 5. Distribution Quality vs Decoding Calibration Gap
## 6. UT vs UTCO: Does Compression Optimization Improve the Distribution or Selection?
## 7. Serial Recompression Robustness: C1 vs CK
## 8. Selector Analysis: Greedy, Verbal Proxies, and Oracle Best-of-N
## 9. Length and Pass-per-Token Analysis
## 10. Train vs Dev Split Consistency
## 11. Representative Case Studies
## 12. Paper-Facing Takeaways
## 13. Limitations and Failure Cases
```

The report must explicitly state whether the experiment used checkpoint continuation or trajectory-derived evaluation.

---

## 18. Representative case studies

Select at least six cases:

1. `full pass → greedy compressed fail → bestN compressed pass`  
   Shows best-of-N reduces harm.
2. `full fail → greedy compressed pass`  
   Shows compression rescue.
3. `full fail → greedy compressed fail → bestN compressed pass`  
   Shows hidden rescue mode in distribution.
4. `C1 pass → CK fail`  
   Shows serial recompression fragility.
5. `C1 fail → CK pass`  
   Shows recompression can clean up a compression.
6. `verifier selects fail but oracle selects pass`  
   Shows verbal proxy failure.

For each case, save:

```text
outputs/reports/case_studies/{task_id}.md
```

Include:

- task instruction
- full-context outcome
- compressed outcomes
- greedy compression text
- best-of-N compression text
- CK text
- concise explanation of harm/rescue/stress behavior

---

## 19. Suggested script stages

```text
00_prepare.py
01_build_task_inventory.py
02_run_full_context_baseline.py
03_build_compression_boundaries.py
04_render_prompts_and_provenance.py
05_generate_candidate_compressions.py
06_run_serial_recompression_stress.py
07_run_behavior_c1_ck.py
08_run_verbal_selectors.py
09_compute_selectors.py
10_compute_transition_metrics.py
11_compute_distribution_quality_calibration.py
12_compute_serial_recompression_metrics.py
13_bootstrap_confidence_intervals.py
14_plot_figures.py
15_write_case_studies.py
16_write_report.py
run_all.sh
```

Each stage should be resumable and should skip completed rows unless `--force` is passed.

---

## 20. Minimal run if compute is constrained

If full four-prompt-family N=8 is too expensive, use this priority order:

1. `ACON_UTCO`, greedy + N=8, C1 + CK, all train+dev tasks.
2. `ACON_UT`, greedy + N=8, C1 + CK.
3. `general_task_aware`, greedy + N=8, C1 + CK.
4. `general_task_agnostic`, greedy + N=4, C1 + CK.

Do not drop all-task transition analysis. If compute must be reduced, reduce N or prompt families, not the harm/rescue transition matrix.

---

## 21. Final paper-facing conclusion v11 should enable

If v11 succeeds, the motivation section can make the following claim:

> Agent history compression is not merely about preserving full-context successes. Compression can harm by deleting useful state, but it can also rescue failures by removing distracting history and clarifying operational dependencies. Structured prompts such as ACON improve the compression distribution, yet greedy decoding and verbal selectors leave substantial behavior headroom. Since high-utility serially robust compressions already exist in the sampling distribution, the natural next step is to train the compressor policy with behavior reward.

This is the exact bridge from motivation to method.
