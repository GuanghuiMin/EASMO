# Motivation v9 — Behavioral Validation of ACON Compression Bias, Fixed-Point Stress, and Chunk Information Advantage

**Status:** implementation spec for coding agent  
**Language:** English  
**Primary benchmark:** AppWorld dev trajectories from prior motivation runs  
**Primary baseline:** ACON-style history compressor using the official ACON UTCO history prompt  
**Primary goal:** move from retention-only motivation to behavior-first evidence that naturally motivates fixed-point-stressed compressor RL.
**Revision note:** all verifier / audit / judge / chunk-labeling LLM calls must use **MiniMax-M2.5**. Qwen3-4B is used only as a candidate compressor / future student-compressor observation, not as an auditor.

---

## 0. Executive Summary

Previous motivation stages showed that LLM compressors have a surface-type retention prior: they often preserve information that looks summary-worthy and discard concrete execution details, even when those details are relevant to downstream tool-use. v9 must answer a harder behavioral question:

> Does this retention bias actually matter for AppWorld pass rate, and does repeated-compression stress reveal which compressed contexts are robust enough to remain useful?

v9 should **not** train a new compressor. It is a diagnostic motivation experiment that prepares the ground for a later RL method.

The three required modules are:

1. **ACON Best-of-N / Qwen3-4B observation**  
   Under the same ACON prompt, sample multiple compressions for the same history. Test whether some sampled compressions achieve higher behavior reward than greedy compression. This validates whether there is a learnable policy preference inside the compressor distribution.

2. **C1 vs CK Pass Fragility Test**  
   Compare behavior with the first compressed output `C1` versus the output after repeated-compression stress `CK = T^K(C1)`. This validates whether fixed-point stress is behaviorally necessary.

3. **Chunk Information Advantage Diagnostic**  
   Segment compressed contexts into natural-language chunks and test which chunks causally affect downstream behavior. This is the motivation for IAPO-style information-aware advantage shaping. Crucially, this test must show that useful compression is not merely entity extraction: high-value chunks may encode causal, precondition, failure-avoidance, or control relations.

The intended method direction after v9 is:

```text
train compressor policy with reward = behavior_after_stress - length_penalty
and optionally assign token/chunk advantages using chunk information contribution.
```

No retrieval pipeline, no runtime fact-table projection, no online multi-round compression, and no downstream agent policy changes should be introduced in v9.

---

## 1. Core Claims to Validate

### Claim 1 — ACON greedy is not necessarily behavior-optimal under its own output distribution

For a fixed history `h`, fixed ACON prompt `P_ACON`, and fixed compressor model `m`, sample multiple compressed contexts:

```math
c_i \sim C_m(\cdot \mid h, P_{ACON}), \quad i=1,\dots,N.
```

If `best_of_N` has higher AppWorld continuation pass rate than greedy output at similar length, then the compressor already has higher-reward modes in its distribution. The failure is partly a **selection/preference problem**, not a pure capacity problem.

This directly motivates RL: increase the probability of high-reward compressed contexts.

### Claim 2 — One-step compression can be behaviorally fragile under repeated compression

Define repeated-compression stress:

```math
T^0(c)=c, \quad T^{r+1}(c)=C_m(T^r(c); P_{ACON}).
```

Let:

```text
C1 = c
CK = T^K(c)
```

If many compressed contexts pass at `C1` but fail at `CK`, then one-step pass is not robust. This motivates training against:

```math
R(T^K(C_\phi(h)))
```

rather than only:

```math
R(C_\phi(h)).
```

### Claim 3 — High-reward compressed contexts rely on causal natural-language chunks, not just entity preservation

Compressed context should not be treated as a bag of entities. A chunk such as:

```text
The previous file_system call failed because access_token was omitted; future file_system calls must pass the returned token explicitly.
```

may be more behaviorally important than an isolated entity list:

```text
access_token = ...; file_path = ...
```

v9 should estimate chunk-level behavioral contribution. This supports an IAPO-style training idea: assign higher advantage to output chunks that reduce future action uncertainty or improve task pass, rather than rewarding all tokens equally.

---

## 2. Prior Results to Reuse, Not Reprove

Do not rerun v7/v8 retention-only experiments unless needed for debugging.

Use prior artifacts as motivation background only:

- v7 showed that under ACON UTCO prompt, `fact_type` explains retention far more strongly than `need_label`, and repeated compression induces a stable fact-type loss hierarchy.
- v8 showed that the same default abstraction tendency appears under general compression prompts, but task-aware prompting can partially reshape fixed-point composition.
- v5 showed a recovered-then-dropped diagnostic pattern under ACON-style recompression.
- v4 showed that span/entity/extractive signals alone do not compete with natural-language summaries on AppWorld behavior.

v9 must add **behavioral evidence**, not more retention-only evidence.

---

## 3. Models and Roles

### 3.1 Compressor models

Run two compressor tracks.

| Track | Compressor | Purpose |
|---|---|---|
| `minimax_compressor` | `MiniMaxAI/MiniMax-M2.5` | Main behavior proof. Stronger compressor; higher chance of producing pass-preserving candidates. |
| `qwen4b_compressor` | `qwen3-4b-instruct-2507` | Small-compressor observation. Tests whether the future student policy already has reward spread or is too weak without warmup. |

### 3.2 Downstream agent

Use the same downstream agent setup as previous AppWorld behavior runs:

```text
Agent model: MiniMaxAI/MiniMax-M2.5
Prompt / runner: reuse motivation_v3/v4/v5 run_with_compressed_context path
Strategy: direct
Budgets: loose_15 primary, strict_8 optional
```

Do **not** modify the downstream agent policy in v9.

### 3.3 Verifier / auditor / analysis judge policy

Use **MiniMax-M2.5 for every LLM-based verification, audit, judging, and chunk-labeling call**. Qwen3-4B is too small for reliable auditing in this project and must not be used to adjudicate its own or MiniMax's outputs.

This applies to:

| Role | Required model | Notes |
|---|---|---|
| Chunk type labeler | **MiniMax-M2.5** | Stage 11 only; interpretive analysis, not reward. |
| Chunk contribution sanity checker, if needed | **MiniMax-M2.5** | Only for human-readable qualitative audit; quantitative contribution comes from behavior deltas. |
| Optional fact-retention scorer | **MiniMax-M2.5** | Use deterministic substring matching first; call MiniMax only for non-exact cases. |
| Output quality / hallucination auditor | **MiniMax-M2.5** | Use only for diagnostics and filtering; never use Qwen as verifier. |
| Report synthesis / representative case selection | **MiniMax-M2.5** or deterministic script | Prefer deterministic tables; MiniMax may summarize. |

Allowed Qwen3-4B roles:

| Role | Allowed? |
|---|---:|
| Candidate compressor in the `qwen4b_compressor` track | yes |
| Future student compressor for SFT/GRPO | yes |
| Proxy logprob scorer if MiniMax endpoint does not expose logprobs | optional, auxiliary only |
| Verifier / auditor / judge / chunk labeler | **no** |

All output rows that involve LLM-based auditing must include:

```text
auditor_model=MiniMaxAI/MiniMax-M2.5
auditor_role=<chunk_labeler|retention_scorer|quality_auditor|report_summarizer>
qwen_used_as_auditor=false
```

---

## 4. ACON Prompt Requirement

The primary baseline must be ACON. Use the official ACON UTCO history compression prompt exactly.

### 4.1 Source

Reuse the v7 prompt artifacts when available:

```text
prompts/official_acon/system_prompt.jinja
prompts/official_acon/improved_history_prompt_samples_4.jinja
```

or the official repo path:

```text
microsoft/acon
experiments/appworld/prompts/context_opt/system_prompt.jinja
experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja
```

Record:

```json
{
  "acon_repo_commit": "...",
  "system_prompt_sha256": "...",
  "history_prompt_sha256": "...",
  "prompt_variant": "ACON_UTCO_official"
}
```

### 4.2 Do not edit the ACON prompt

Do not add new instructions such as “preserve causal chunks” or “optimize for stress robustness”. v9 is diagnostic: we need to evaluate the existing ACON compressor distribution.

### 4.3 Rendering convention

Use the same rendering convention as v7:

```python
rendered = jinja2.from_string(template_text).render(
    task=user_instruction,
    prev_summary="",
    history=current_context,
    max_chars=TARGET_MAX_CHARS,
)
```

Note: if the official prompt ignores `max_chars`, still log `TARGET_MAX_CHARS` as the experimental budget but do not edit the template to force it.

---

## 5. Data and Case Selection

### 5.1 Source cases

Reuse existing successful full-context AppWorld dev trajectories:

```text
motivation_v3/outputs/motivation_full_trajectories.jsonl
```

Each case must provide:

```json
{
  "case_id": "...",
  "task_id": "...",
  "user_instruction": "...",
  "full_trajectory_text": "...",
  "trajectory_steps": [...],
  "baseline_success": true,
  "baseline_iterations": int
}
```

### 5.2 Primary case pool

Use 30 cases by default, same as v7/v8, prioritizing medium/long trajectories.

If cost is an issue, use a pilot pool:

```text
PILOT_N_CASES = 20
```

Selection priority:

1. cases where previous ACON-style compression failed;
2. cases where ACON succeeded but used many steps;
3. cases with multiple tool apps / auth / IDs / file paths / API schemas;
4. avoid cases whose task is almost solved entirely in the final one or two trajectory steps.

### 5.3 Behavior evaluation mode

Primary mode: reuse the existing behavior evaluation protocol from v3/v4/v5:

```text
Start the AppWorld task using the standard downstream agent runner.
Inject compressed_context into the downstream USER turn.
Run with max_steps = 15 for loose budget.
Optionally run max_steps = 8 for strict budget.
```

Do not implement environment checkpoint restoration for v9 unless it already exists and is easy to call. The primary goal is to remain comparable with prior ACON behavior baselines.

---

## 6. Experimental Overview

For each case and compressor model:

1. Generate greedy ACON compression `c_greedy`.
2. Sample `N` stochastic ACON compressions `c_1 ... c_N` using the same prompt.
3. For each candidate, compute:
   - one-step behavior: `Pass(C1)`;
   - stressed behavior: `Pass(CK)` after `K` recompression rounds.
4. Compare greedy vs best-of-N.
5. Analyze C1-vs-CK fragility.
6. Select representative candidates for chunk information advantage analysis.

---

## 7. Stage-by-Stage Implementation Plan

## Stage 00 — Prepare directories and provenance

Create:

```text
motivation_v9/
  docs/
  prompts/
  data/
  outputs/raw/
  outputs/tables/
  outputs/figures/
  outputs/reports/
  scripts/
  motivation_v9/
```

Write:

```text
outputs/provenance/run_config.json
outputs/provenance/prompt_sha256.json
outputs/provenance/model_endpoints.json
```

Minimum config:

```yaml
N_CASES: 30
PILOT_N_CASES: 20
N_SAMPLES: 8
STRESS_ROUNDS_K: 2
TEMPERATURE_GREEDY: 0.0
TEMPERATURE_SAMPLE: 0.7
BUDGET_MAX_STEPS_PRIMARY: 15
BUDGET_MAX_STEPS_SECONDARY: 8
MAX_TRAJECTORY_CHARS: 18000
MAX_COMPRESS_TOKENS: 2048
CHUNK_ABLATION_MAX_CASES: 12
CHUNK_ABLATION_MAX_CHUNKS_PER_CONTEXT: 12
```

---

## Stage 01 — Build v9 case pool

Script:

```text
scripts/01_build_cases.py
```

Input:

```text
motivation_v3/outputs/motivation_full_trajectories.jsonl
motivation_v5/data/sampled_cases.jsonl              # optional priority labels
motivation_v7/data/fact_bank_filtered.jsonl          # optional analysis only
```

Output:

```text
data/v9_cases.jsonl
```

Schema:

```json
{
  "case_id": "task_...",
  "task_id": "...",
  "user_instruction": "...",
  "full_trajectory_text": "...",
  "trajectory_steps": [...],
  "baseline_iterations": 20,
  "case_priority": "acon_failed | hard_success | long_multitool",
  "known_previous_acon_success": false,
  "known_previous_acon_steps": 15,
  "notes": "..."
}
```

---

## Stage 02 — Generate greedy and sampled ACON compressions

Script:

```text
scripts/02_generate_candidate_compressions.py
```

For each `(case, compressor_model)`:

### Greedy

```text
generation_type = greedy
temperature = 0.0
sample_id = greedy
```

### Samples

```text
generation_type = sample
temperature = 0.7
sample_id = sample_00 ... sample_{N-1}
```

Use the same official ACON UTCO prompt for all candidates.

Output:

```text
outputs/raw/candidate_compressions.jsonl
```

Schema:

```json
{
  "candidate_id": "case__model__sample",
  "case_id": "...",
  "compressor_model": "minimax | qwen4b",
  "prompt_variant": "ACON_UTCO_official",
  "generation_type": "greedy | sample",
  "sample_id": "greedy | sample_00",
  "temperature": 0.7,
  "seed": 42,
  "input_context_hash": "...",
  "compressed_text": "...",
  "compressed_chars": 1234,
  "compressed_tokens_est": 321,
  "error": null
}
```

Sanity checks:

- no empty outputs;
- no output that reproduces the entire input;
- log budget violations but do not discard;
- if Qwen fails frequently, keep rows with errors and continue MiniMax track.

---

## Stage 03 — Repeated-compression stress chains

Script:

```text
scripts/03_stress_recompress_candidates.py
```

For each candidate `c`:

```text
round 0: c0 = candidate.compressed_text
round 1: c1 = C_ACON(c0)
round 2: c2 = C_ACON(c1)
...
round K: cK = C_ACON(c_{K-1})
```

Important:

- Use the **same compressor model** as the candidate generator for stress.
  - MiniMax candidate → MiniMax stress compressor.
  - Qwen candidate → Qwen stress compressor.
- Use the same official ACON UTCO prompt.
- `task=user_instruction`, `history=current_round_text`, `prev_summary=""`.

Output:

```text
outputs/raw/stress_chains.jsonl
```

Schema:

```json
{
  "candidate_id": "...",
  "case_id": "...",
  "compressor_model": "minimax | qwen4b",
  "round": 0,
  "context_text": "...",
  "chars": 1234,
  "tokens_est": 321,
  "text_hash": "...",
  "error": null
}
```

Also write convergence summary:

```text
outputs/tables/stress_chain_convergence.csv
```

Fields:

```text
candidate_id, case_id, model, converged_binary, convergence_round,
text_similarity_last, length_change_last, token_jaccard_last
```

Convergence declaration:

```text
text_similarity(x_r, x_{r-1}) >= 0.95
AND |len_r - len_{r-1}| / max(len_{r-1}, 1) <= 0.02
```

No fact-retention Jaccard required in v9 primary. Keep this behavior-first.

---

## Stage 04 — Run downstream behavior on C1 and CK

Script:

```text
scripts/04_run_behavior_c1_ck.py
```

For each candidate, evaluate two contexts:

```text
C1 = round 0 context from candidate_compressions.jsonl
CK = round K context from stress_chains.jsonl
```

Use the same downstream runner as previous motivation experiments.

Output:

```text
outputs/raw/behavior_runs_c1_ck.jsonl
```

Schema:

```json
{
  "run_id": "candidate_id__C1__loose15",
  "candidate_id": "...",
  "case_id": "...",
  "compressor_model": "minimax | qwen4b",
  "generation_type": "greedy | sample",
  "sample_id": "...",
  "eval_context_round": "C1 | CK",
  "stress_round": 0,
  "budget_name": "loose_15",
  "max_steps": 15,
  "success": true,
  "score": 1.0,
  "iterations": 8,
  "termination_reason": "task_completed",
  "total_input_tokens": 48211,
  "peak_input_tokens": 48211,
  "compressed_chars": 1234,
  "compressed_tokens_est": 321,
  "output_dir": "...",
  "error": null
}
```

Primary budget:

```text
loose_15
```

Optional budget:

```text
strict_8
```

Do not skip greedy. Greedy is the ACON baseline.

Cost control:

- Always run all greedy candidates for C1 and CK.
- Run all MiniMax samples for C1 and CK if feasible.
- For Qwen samples, if cost is high, run proxy first, then run behavior on the top-2 proxy samples plus greedy.

---

## Stage 05 — Best-of-N and reward separability metrics

Script:

```text
scripts/05_compute_best_of_n_metrics.py
```

### 5.1 Candidate reward

Define behavior reward:

```text
R_C1 = score_C1 - lambda_len * normalized_length_C1
R_CK = score_CK - lambda_len * normalized_length_CK
```

Use:

```text
lambda_len = 0.02
normalized_length = compressed_tokens_est / 1000
```

Also report pass-only metrics without length penalty.

### 5.2 Best-of-N

For each `(case_id, compressor_model, eval_context_round)`:

```text
greedy_reward = R(greedy)
best_sample_reward = max_i R(sample_i)
best_of_n_reward = max(greedy_reward, best_sample_reward)
best_of_n_gain = best_of_n_reward - greedy_reward
oracle_win = best_sample_reward > greedy_reward
```

Use true AppWorld behavior reward when available.

### 5.3 Outputs

```text
outputs/tables/best_of_n_by_case.csv
outputs/tables/reward_spread_by_case.csv
outputs/tables/best_of_n_summary.csv
```

`best_of_n_by_case.csv` fields:

```text
case_id, compressor_model, eval_context_round,
greedy_success, greedy_score, greedy_length,
best_sample_id, best_sample_success, best_sample_score, best_sample_length,
best_of_n_gain_score, best_of_n_gain_reward, oracle_win
```

`reward_spread_by_case.csv` fields:

```text
case_id, model, eval_context_round,
num_candidates, mean_score, std_score, min_score, max_score,
mean_reward, std_reward, min_reward, max_reward,
pass_rate_among_samples
```

### 5.4 Success criteria

Strong signal for RL motivation if:

```text
oracle_win_rate_CK >= 0.25
OR best_of_N_CK_pass_rate - greedy_CK_pass_rate >= 10 percentage points
```

Moderate signal if:

```text
best_of_N_CK improves score or reward but not binary success.
```

Failure mode to report honestly:

```text
Qwen3-4B may show reward spread in proxy/score but not binary pass.
This should be interpreted as small-compressor weakness, not immediate falsification of the RL idea.
```

---

## Stage 06 — C1 vs CK Pass Fragility Test

Script:

```text
scripts/06_compute_c1_ck_fragility.py
```

For each candidate with both C1 and CK behavior runs, classify:

| class | definition |
|---|---|
| `robust_pass` | C1 success = 1 and CK success = 1 |
| `fragile_pass` | C1 success = 1 and CK success = 0 |
| `stress_improved` | C1 success = 0 and CK success = 1 |
| `robust_fail` | C1 success = 0 and CK success = 0 |

### Metrics

```text
fragility_rate = count(fragile_pass) / count(C1 success = 1)
robustness_rate = count(robust_pass) / count(C1 success = 1)
stress_drop_pp = pass_rate_C1 - pass_rate_CK
stress_auc = mean(success_C1 + success_CK) / 2
```

If scores are continuous:

```text
score_drop = score_C1 - score_CK
```

### Outputs

```text
outputs/tables/c1_ck_transition_matrix.csv
outputs/tables/c1_ck_fragility_by_model.csv
outputs/figures/fig_c1_ck_transition_matrix.pdf
outputs/figures/fig_c1_ck_pass_drop_by_model.pdf
```

Transition matrix fields:

```text
compressor_model, generation_type, count_robust_pass, count_fragile_pass,
count_stress_improved, count_robust_fail, fragility_rate, pass_rate_C1, pass_rate_CK
```

### Success criteria

Fixed-point stress is behaviorally necessary if:

```text
fragility_rate >= 0.20
OR pass_rate_C1 - pass_rate_CK >= 10 percentage points
OR many best-of-N C1 winners are not best-of-N CK winners.
```

If fragility is low but CK best-of-N still differs from greedy, report:

```text
Repeated compression is less about detecting fragility and more about defining a robustness frontier.
```

---

## Stage 07 — Select candidates for chunk information analysis

Script:

```text
scripts/07_select_chunk_cases.py
```

Select up to:

```text
CHUNK_ABLATION_MAX_CASES = 12
```

Prioritize cases in this order:

1. `best_sample_CK` succeeds while `greedy_CK` fails.
2. Candidate is `fragile_pass` (C1 pass, CK fail).
3. Both greedy and best pass, but best is shorter or uses fewer steps.
4. Qwen case with high proxy reward spread if real pass is unavailable.

For each selected case, include two contexts when possible:

- `high_reward_context`: best-of-N CK winner or robust pass candidate.
- `low_reward_context`: greedy or sample that fails under CK.

Output:

```text
outputs/raw/chunk_case_selection.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "compressor_model": "...",
  "selected_candidate_id": "...",
  "comparison_candidate_id": "...",
  "selection_reason": "best_sample_CK_succeeds_greedy_fails",
  "context_round_for_chunks": "C1",
  "stress_round_for_eval": "CK"
}
```

Use `C1` text for chunk segmentation, then stress after chunk deletion. This tests whether chunks remain useful under repeated compression.

---

## Stage 08 — Segment compressed outputs into natural-language chunks

Script:

```text
scripts/08_segment_chunks.py
```

### 8.1 Chunking rule

Do not use entity extraction. Segment only by natural-language structure:

1. split on bullet lines / numbered lines;
2. if no bullets, split into sentences;
3. merge very short chunks `< 20 chars` into neighbors;
4. cap to `CHUNK_ABLATION_MAX_CHUNKS_PER_CONTEXT = 12` chunks by merging adjacent low-length chunks.

Each chunk should be a coherent statement, not a token/entity.

Output:

```text
outputs/raw/chunks.jsonl
```

Schema:

```json
{
  "chunk_id": "candidate_id__chunk_03",
  "candidate_id": "...",
  "case_id": "...",
  "chunk_index": 3,
  "chunk_text": "...",
  "chunk_chars": 173,
  "chunk_tokens_est": 43
}
```

---

## Stage 09 — Chunk ablation behavior runs

Script:

```text
scripts/09_run_chunk_ablation_behavior.py
```

For each selected context `c` and each chunk `z_j`:

```text
c_minus_j = c with chunk j removed
stress_minus_j = T^K(c_minus_j)
```

Also run the original unablated context:

```text
stress_full = T^K(c)
```

Then evaluate downstream behavior using the same downstream agent.

Outputs:

```text
outputs/raw/chunk_ablation_contexts.jsonl
outputs/raw/chunk_ablation_behavior_runs.jsonl
```

`chunk_ablation_contexts.jsonl` schema:

```json
{
  "ablation_id": "chunk_id__removed",
  "candidate_id": "...",
  "chunk_id": "...",
  "case_id": "...",
  "ablation_type": "remove_chunk | full_context_control",
  "pre_stress_text": "...",
  "post_stress_text": "...",
  "pre_stress_chars": 1234,
  "post_stress_chars": 1000
}
```

`chunk_ablation_behavior_runs.jsonl` schema follows Stage 04 behavior schema plus:

```json
{
  "ablation_id": "...",
  "chunk_id": "...",
  "ablation_type": "remove_chunk | full_context_control"
}
```

Cost control:

- Run chunk ablation only on selected cases.
- If a context has more than 12 chunks, keep 12 by merging small chunks or prioritizing chunks in the first 1500 chars.
- If full context control fails, still keep the row but mark all chunk advantages as `not_interpretable_due_to_full_fail=true`.

---

## Stage 10 — Compute chunk information advantage

Script:

```text
scripts/10_compute_chunk_advantage.py
```

For each selected candidate and chunk:

Let:

```text
score_full = behavior score of T^K(c)
score_minus_j = behavior score of T^K(c without chunk j)
success_full = success of T^K(c)
success_minus_j = success of T^K(c without chunk j)
```

Define:

```text
chunk_score_advantage_j = score_full - score_minus_j
chunk_pass_advantage_j = int(success_full) - int(success_minus_j)
```

Interpretation:

- `chunk_pass_advantage = +1`: removing the chunk flips pass to fail.
- `chunk_score_advantage > 0`: chunk contributes positively to behavior score.
- `chunk_score_advantage <= 0`: chunk is neutral or possibly harmful.

Normalize within candidate:

```text
positive_adv = max(chunk_score_advantage, 0)
chunk_adv_norm = positive_adv / sum_positive_adv_for_candidate
```

If all advantages are zero, set `chunk_adv_norm=0` for all chunks.

Output:

```text
outputs/tables/chunk_information_advantage.csv
```

Fields:

```text
case_id,candidate_id,chunk_id,chunk_index,chunk_text,
score_full,score_minus_chunk,success_full,success_minus_chunk,
chunk_score_advantage,chunk_pass_advantage,chunk_adv_norm,
not_interpretable_due_to_full_fail
```

This is the key diagnostic for IAPO-style shaping:

```text
sequence reward says whether the compression passes;
chunk information advantage says which natural-language chunks deserve credit.
```

---

## Stage 11 — Chunk type labeling for analysis only

Script:

```text
scripts/11_label_chunks_for_analysis.py
```

Use a strict JSON prompt with **MiniMax-M2.5 only**. This is **not reward** and not part of the method; it is only for interpreting what high-advantage chunks are. Do not use Qwen3-4B for this labeling step.

### System prompt

```text
You are a strict analyst of compressed tool-use agent context. Return only JSON.
```

### User prompt

```text
Classify the following compressed-context chunk.

The goal is to understand what kind of information this natural-language chunk carries.
Do not judge whether it is correct. Do not invent context.

Labels:
- CAUSAL_PRECONDITION: explains why a future action requires a condition, parameter, credential, or prior result.
- CONTROL_NEGATIVE_EVIDENCE: records a failed attempt, error, invalid path, or action to avoid.
- ACTION_OUTCOME: records whether a previous action succeeded, failed, returned empty, returned objects, or changed state.
- RUNTIME_BINDING: binds an exact value such as token, ID, path, email, amount, date, or object set to its role.
- ENTITY_LIST_ONLY: mostly lists entities/IDs/values without explaining use or relation.
- NARRATIVE_PROGRESS: high-level summary of progress, intent, or reasoning.
- TASK_GOAL_OR_TODO: states the goal or remaining subtask.
- OTHER: none of the above.

Return JSON:
{
  "chunk_type": "...",
  "contains_exact_literals": true,
  "contains_causal_relation": true,
  "contains_negative_evidence": false,
  "one_sentence_rationale": "..."
}

Task:
{user_instruction}

Chunk:
{chunk_text}
```

Output:

```text
outputs/raw/chunk_type_labels.jsonl
```

Schema:

```json
{
  "chunk_id": "...",
  "labeler_model": "MiniMaxAI/MiniMax-M2.5",
  "chunk_type": "CAUSAL_PRECONDITION",
  "contains_exact_literals": true,
  "contains_causal_relation": true,
  "contains_negative_evidence": false,
  "one_sentence_rationale": "..."
}
```

---

## Stage 12 — Analyze chunk advantage by type

Script:

```text
scripts/12_analyze_chunk_advantage_by_type.py
```

Join:

```text
chunk_information_advantage.csv
chunk_type_labels.jsonl
```

Compute:

```text
mean_chunk_score_advantage_by_type
mean_chunk_pass_advantage_by_type
fraction_of_top_advantage_chunks_by_type
fraction_of_top_advantage_chunks_that_are_ENTITY_LIST_ONLY
fraction_of_top_advantage_chunks_that_contain_causal_relation
```

Define top-advantage chunks:

```text
top chunks = chunks with chunk_adv_norm >= 0.25 within a candidate
or top-1 chunk per candidate if all advantages are nonzero but below 0.25.
```

Outputs:

```text
outputs/tables/chunk_advantage_by_type.csv
outputs/tables/top_chunk_examples.csv
outputs/figures/fig_chunk_advantage_by_type.pdf
outputs/figures/fig_top_chunk_type_distribution.pdf
```

### Success criteria

This supports IAPO-style natural-language chunk shaping if:

```text
fraction_top_chunks_with_causal_relation >= 0.40
OR CAUSAL_PRECONDITION / CONTROL_NEGATIVE_EVIDENCE / ACTION_OUTCOME chunks have higher mean advantage than ENTITY_LIST_ONLY chunks.
```

This argues against entity-only reward if:

```text
ENTITY_LIST_ONLY chunks are not the dominant top-advantage category.
```

---

## Stage 13 — Optional future-action proxy calibration

This stage is optional but useful if full AppWorld pass runs are too expensive.

Script:

```text
scripts/13_compute_proxy_reward_calibration.py
```

If the MiniMax endpoint exposes logprobs or teacher-forced scoring, use MiniMax for the proxy. If it does not, the local Qwen model may be used only as an **auxiliary proxy scorer**, never as a verifier/auditor. In either case, calibrate the proxy against MiniMax downstream behavior on a held-out subset. Compute a proxy on successful future action suffixes:

```text
FutureActionNLL(context) = -sum_j log p(action_j^* | context, previous_gold_actions)
```

Then compare:

```text
proxy_reward = -FutureActionNLL(T^K(c)) - lambda_len * length
```

against behavior score/pass.

Outputs:

```text
outputs/tables/proxy_pass_calibration.csv
outputs/figures/fig_proxy_vs_pass.pdf
```

Metrics:

```text
Spearman(proxy_reward, behavior_score)
AUROC(proxy_reward predicting success)
Kendall rank agreement between proxy-best and pass-best
```

Use this only as a training-cost justification. Do not replace behavior metrics in the main motivation. If Qwen is used for proxy scoring, mark every row with `proxy_model=qwen3-4b` and `proxy_is_auxiliary=true`; do not cite those rows as verification evidence.

---

## Stage 14 — Write final v9 report

Script:

```text
scripts/14_write_report.py
```

Output:

```text
outputs/reports/motivation_v9_results_summary.md
```

Required headings:

```markdown
# motivation_v9 Results — Behavioral Compression Stress and Chunk Information Advantage

## TL;DR
## Setup
## Claim 1: ACON Best-of-N Behavioral Gap
## Claim 2: C1 vs CK Pass Fragility
## Claim 3: Chunk Information Advantage
## Qwen3-4B Compressor Observations
## What This Motivates for RL
## Negative / Null Results
## Representative Cases
## Files and Reproducibility
```

---

## 8. Main Tables to Produce

### 8.1 `best_of_n_summary.csv`

```text
compressor_model, eval_context_round, n_cases,
greedy_pass_rate, best_of_n_pass_rate, pass_gain_pp,
greedy_mean_score, best_of_n_mean_score, score_gain,
greedy_mean_length, best_of_n_mean_length, length_ratio,
oracle_win_rate
```

### 8.2 `c1_ck_transition_matrix.csv`

```text
compressor_model, generation_type, n_candidates,
robust_pass, fragile_pass, stress_improved, robust_fail,
pass_rate_C1, pass_rate_CK, stress_drop_pp, fragility_rate
```

### 8.3 `chunk_advantage_by_type.csv`

```text
chunk_type, n_chunks,
mean_score_advantage, mean_pass_advantage,
frac_positive_advantage, frac_top_advantage,
contains_causal_relation_rate, contains_exact_literals_rate
```

### 8.4 `qwen4b_observation_summary.csv`

```text
n_cases, n_candidates,
greedy_C1_pass, greedy_CK_pass,
bestN_C1_pass, bestN_CK_pass,
reward_spread_mean, oracle_win_rate,
interpretation
```

---

## 9. Main Figures to Produce

1. `fig_best_of_n_pass_gain.pdf`  
   Bar chart: greedy vs best-of-N pass rate at C1 and CK, split by compressor model.

2. `fig_c1_ck_transition_matrix.pdf`  
   2×2 transition heatmap: C1 pass/fail vs CK pass/fail.

3. `fig_stress_pass_curve_by_round.pdf`  
   Pass rate over stress round r = 0..K.

4. `fig_chunk_advantage_by_type.pdf`  
   Violin/bar plot of chunk advantage by chunk type.

5. `fig_top_chunk_type_distribution.pdf`  
   Distribution of chunk types among top-advantage chunks.

---

## 10. Interpretation Rules

### 10.1 If Best-of-N CK beats greedy CK

Interpretation:

```text
ACON's prompt can sometimes produce behaviorally better compression under the same prompt and similar length, but the default decoding preference does not reliably select it. This supports compressor policy optimization.
```

Method implication:

```text
Use RL to increase probability of high-reward compressed contexts.
```

### 10.2 If C1 pass often becomes CK fail

Interpretation:

```text
One-step compression quality is not robust. A compression can look sufficient once, but lose behavior-critical information under repeated compression. Fixed-point stress is a meaningful training-time robustness test.
```

Method implication:

```text
Reward should be applied to T^K(C(h)), not only C(h).
```

### 10.3 If high-advantage chunks are causal/control natural language

Interpretation:

```text
Useful compression is not entity extraction. The behavior-critical unit is often a natural-language causal or control relation that tells the agent how facts constrain future tool use.
```

Method implication:

```text
IAPO-style information-aware advantage should operate at chunk/token level using behavior contribution, not entity-recall reward.
```

### 10.4 If Qwen3-4B does not show pass gain

Do not immediately conclude the idea fails. Interpret according to proxy and MiniMax:

| Qwen result | MiniMax result | Interpretation |
|---|---|---|
| no pass gain, proxy spread exists | MiniMax pass gain exists | Qwen is weak as a compressor but has trainable signal; use SFT/RL warmup. |
| no pass gain, no proxy spread | MiniMax pass gain exists | Qwen distribution may be too poor initially; student needs supervised warmup before RL. |
| no pass gain | no MiniMax gain | RL motivation is weak; reconsider method. |

---

## 11. Non-Goals and Constraints

Do not implement:

- retrieval before compression;
- fact-table projection at runtime;
- JSON entity-only compressor as a primary method;
- downstream agent policy changes;
- online repeated compression at deployment;
- new ACON prompt engineering.

Allowed:

- repeated compression only as offline diagnostic / training stress proxy;
- chunk labeling only for interpretation, using **MiniMax-M2.5 only**;
- future-action likelihood proxy only for cost-saving analysis;
- MiniMax behavior proof and Qwen student-risk observation;
- Qwen proxy scoring only when logprobs are unavailable from MiniMax, and only as an auxiliary signal calibrated against MiniMax behavior.

---

## 12. Minimal Pilot Plan

Before full v9, run a pilot:

```text
N_CASES = 10
N_SAMPLES = 4
K = 2
compressor_models = [MiniMax, Qwen3-4B]
budget = loose_15 only
chunk_analysis = top 4 cases only
```

Pilot decision:

- If MiniMax best-of-N CK gain ≥ 10 pp or oracle_win_rate ≥ 25%, run full v9.
- If MiniMax has no gain but Qwen proxy has spread, add proxy calibration before full behavior.
- If neither has gain, stop and inspect outputs manually before scaling.

---

## 13. Expected Final Message If Results Are Positive

The final motivation should read like this:

```text
ACON-style compression already improves over naive summaries, but it is still a biased compressor. Under the same ACON prompt, greedy decoding often selects summary-like compressions that are not behavior-optimal. Best-of-N sampling reveals that higher-reward compressed contexts exist in the compressor's own output distribution. However, some one-step successful compressions fail after repeated-compression stress, showing that robust targeted compression must preserve behavior-critical information at the compression fixed point. Chunk ablation further shows that the behavior-critical units are natural-language causal/control chunks rather than entity lists. These findings motivate fixed-point-stressed, information-aware RL for a student compressor: reward compressed contexts by downstream behavior after stress, and assign credit to chunks that actually support agent continuation.
```

---

## 14. Success / Failure Criteria Summary

| Claim | Strong positive if |
|---|---|
| Best-of-N behavioral gap | `best_of_N_CK_pass - greedy_CK_pass >= 10 pp` OR `oracle_win_rate_CK >= 25%` |
| C1 vs CK fragility | `fragility_rate >= 20%` OR `C1_pass - CK_pass >= 10 pp` |
| Chunk information advantage | causal/control/action-outcome chunks have higher mean advantage than entity-only chunks, or ≥40% of top-advantage chunks contain causal relations |
| Qwen3-4B observation | Qwen shows reward/proxy spread; pass gain is a bonus, not required |

If all three primary claims fail, do not proceed to RL. If Claim 1 and Claim 2 pass, the RL method is strongly motivated. If Claim 3 passes, IAPO-style advantage shaping is motivated.

