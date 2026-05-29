# Motivation v10 (Revised) — From ACON Best-of-N Headroom to Trainable Compressor Policy

> Revision note after the final `motivation_v9` hand-written report: the widened chunk study **falsified** the earlier hypothesis that causal natural-language chunks generally dominate entity-list chunks. Therefore v10 must **not** build the method or reward around causal-vs-entity labels. Chunk labels are diagnostics only. The core method signal remains behavior: pass after repeated-compression stress minus length.

---

## 0. Purpose

`motivation_v9` showed two robust behavioral facts:

1. **ACON greedy is not optimal under its own compressor distribution.** With the same ACON UTCO prompt and MiniMax-M2.5 compressor, oracle best-of-N over stochastic samples strongly improves AppWorld continuation pass rate while also producing shorter compressed contexts.
2. **One-step compression can be behaviorally fragile under repeated-compression stress.** Some compressions pass immediately but fail after a small number of recompressions.

The earlier chunk-level claim must be revised:

- The widened n=239 chunk analysis shows `ENTITY_LIST_ONLY` chunks have higher mean advantage than causal-flagged chunks.
- This does **not** mean entity extraction is the method. It means surface labels like `entity-only` and `causal` are poor proxies for behavioral value. Some entity-looking chunks are exact runtime bindings that the next API call needs.
- The only stable positive semantic subtype in v9 is `CONTROL_NEGATIVE_EVIDENCE`, but even this should be treated as analysis, not as a hand-crafted reward term.

v10 should answer:

> Can the behavioral headroom discovered by ACON best-of-N be converted into trainable supervision for a smaller student compressor, without adding retrieval, runtime fact-table modules, post-hoc projection, or modifying the downstream agent?

This is still a motivation / feasibility experiment, not the final full-scale training run.

---

## 1. Updated core claims

### Claim 1 — Best-of-N headroom can be approximated by cheaper proxy selection

The v9 oracle best-of-N uses true AppWorld pass to choose the best compression, which is not deployable and too expensive for RL at scale.

v10 tests whether cheaper proxy scores can select candidates that recover a meaningful fraction of the oracle best-of-N gain.

**Primary comparison:**

```text
greedy ACON
proxy-selected best sample
oracle best-of-N sample
```

under both one-step (`C1`) and stress (`CK`) evaluation.

---

### Claim 2 — Fixed-point-stress-selected teacher targets are better for student training than one-step-selected targets

If a compression is only good at `C1` but fails after stress `T^K`, it is not a robust compression target.

v10 trains two Qwen3-4B SFT students:

```text
Qwen-SFT-C1 : trained on teacher targets selected by C1 reward
Qwen-SFT-CK : trained on teacher targets selected by CK reward
```

**Primary comparison:** `Qwen-SFT-CK` should improve `CK` pass rate and stress robustness over `Qwen-SFT-C1`.

---

### Claim 3 — Qwen3-4B needs SFT warmup before GRPO

Raw Qwen3-4B may be too weak as a compressor. v10 tests whether SFT on MiniMax stress-selected targets moves Qwen into a region where group-relative RL has non-degenerate reward variance.

**Primary comparison:**

```text
Raw-Qwen reward spread
Qwen-SFT-C1 reward spread
Qwen-SFT-CK reward spread
```

If SFT creates reward spread and best-of-N headroom in Qwen, GRPO becomes plausible.

---

### Claim 4 — Chunk labels are not reliable reward proxies; chunk credit must be behavior-based

The widened v9 chunk analysis invalidated the simple hypothesis `causal NL > entity list`.

v10 should not try to prove that one surface chunk label is always better. Instead, it should test:

1. Are chunk surface labels weak predictors of behavioral contribution?
2. Are the highest-advantage chunks often exact runtime bindings, action outcomes, or negative evidence, depending on the case?
3. Does leave-one-chunk-out behavioral advantage provide a better credit signal than entity count, literal count, chunk length, or categorical labels?

This motivates IAPO-style **information-aware credit assignment**, not hand-crafted entity or causal rewards.

---

## 2. Non-goals and hard constraints

Do **not** introduce a heavier runtime pipeline.

Forbidden in v10:

- retrieval before compression;
- runtime fact table / inventory module;
- post-hoc projection or reinsertion of facts;
- online repeated compression during deployment;
- downstream agent policy modification;
- entity recall as the main reward;
- causal-label reward as the main reward.

Allowed offline:

- MiniMax best-of-N teacher target construction;
- repeated-compression stress for training/evaluation;
- proxy scoring and verifier calls;
- Qwen3-4B LoRA SFT;
- optional small GRPO readiness smoke test;
- chunk leave-one-out analysis for credit-assignment diagnostics.

Deployment target remains:

```text
history/context -> one student compressor call -> compressed context -> unchanged downstream agent
```

---

## 3. Definitions

### 3.1 Candidate compression

For a compression boundary / case `x`, a compressor outputs a natural-language compressed context:

```text
c ~ C(. | x, prompt=ACON_UTCO)
```

### 3.2 Greedy decoding

`greedy` means a single deterministic compression output generated with `temperature=0.0` and fixed seed if supported.

```text
c_greedy = greedy_decode(C | x, ACON_UTCO)
```

This is the default deployed decoding mode and the v9 baseline.

### 3.3 Best-of-N

`best-of-N` means sampling `N` stochastic compressed contexts from the same compressor and same ACON UTCO prompt, then selecting the one with the highest reward.

```text
c_1, ..., c_N ~ C(. | x, temperature=0.7)
c_best = argmax_i R(c_i)
```

In v9, true AppWorld pass was used for oracle best-of-N. In v10, oracle best-of-N remains a diagnostic upper bound; proxy selection attempts to approximate it.

### 3.4 Recompression stress

Repeated-compression stress applies a fixed recompressor to a candidate:

```text
T^0(c) = c
T^{r+1}(c) = C_stress(T^r(c))
```

Use `K=2` by default to match v9.

### 3.5 Rewards

One-step reward:

```text
R_C1(c) = Pass(c) - lambda_length * length_norm(c)
```

Stress reward:

```text
R_CK(c) = Pass(T^K(c)) - lambda_length * length_norm(T^K(c))
```

Recommended default:

```text
lambda_length = 0.05
```

Pass dominates length. Length is a tie-breaker and compression-pressure term, not the main objective.

### 3.6 Proxy rewards

Proxy rewards estimate behavior without running full AppWorld pass for every candidate.

Possible proxies:

1. `future_action_nll_proxy` — teacher-forced likelihood of successful future actions, if logprobs are available.
2. `minimax_continuation_verifier_score` — MiniMax predicts whether the compressed context is sufficient for continuation under a strict rubric.
3. `minimax_pairwise_preference` — MiniMax compares two candidate compressions and chooses which is more likely to let the downstream agent succeed.
4. `stress_stability_length_proxy` — length-normalized stability under `T^K`; weak baseline only.
5. `chunk_behavior_proxy` — aggregate leave-one-chunk-out information advantage, used only on subsets because it is expensive.

Important: proxies must be evaluated by their ability to select candidates that pass. Do not treat proxy score itself as ground truth.

---

## 4. Model roles

| Role | Model | Notes |
|---|---|---|
| Teacher compressor | MiniMax-M2.5 | ACON UTCO prompt, same as v9 |
| Downstream agent | MiniMax-M2.5 | unchanged AppWorld productive agent runner |
| Verifier / judge / chunk labeler | MiniMax-M2.5 | Qwen must not judge or verify |
| Student compressor | Qwen3-4B-Instruct | SFT / optional GRPO target |
| Optional auxiliary scorer | Qwen3-4B | only for logprob-style proxy if MiniMax lacks logprobs; never final verifier |

All judge / verifier outputs must include `judge_model = MiniMaxAI/MiniMax-M2.5`.

---

## 5. Data splits

### 5.1 Preferred split

Use AppWorld successful full-context trajectories from a training split if available.

| Split | Size | Purpose |
|---|---:|---|
| `teacher_train` | 80-120 cases | generate MiniMax candidates and SFT targets |
| `dev_proxy` | 30 cases | proxy calibration and model selection |
| `test_behavior` | 30 cases | held-out behavior evaluation |

### 5.2 If only v9 cases are immediately available

Run a pilot:

| Split | Size | Purpose |
|---|---:|---|
| `pilot_train` | 20 cases | SFT smoke test |
| `pilot_eval` | 10 cases | held-out behavior sanity check |

Mark pilot results as preliminary, not generalization evidence.

### 5.3 Required fields

Each case row must contain:

```json
{
  "case_id": "...",
  "task_id": "...",
  "user_instruction": "...",
  "full_trajectory_text": "...",
  "trajectory_steps": [...],
  "baseline_success": true,
  "baseline_steps": 20,
  "compression_boundary": "mid_or_full",
  "max_steps_for_continuation": 15
}
```

Prefer mid-trajectory boundaries if environment restoration is available. If not, use v9 full-prefix continuation and document the limitation.

---

## 6. ACON prompt policy

Use the official ACON UTCO prompt exactly as in v9.

Requirements:

- record repo commit hash;
- record prompt SHA256;
- do not rewrite prompt text;
- use the same renderer as v9;
- use the same downstream agent prompt as v9.

Outputs:

```text
outputs/provenance/acon_commit.txt
outputs/provenance/acon_utco_prompt_sha256.json
outputs/provenance/rendered_prompt_examples/
```

---

## 7. Stage overview

```text
00_prepare
01_build_cases
02_generate_minimax_candidates
03_stress_candidates
04_behavior_evaluate_candidates
05_proxy_score_candidates
06_proxy_selection_analysis
07_construct_teacher_targets
08_train_qwen_sft
09_evaluate_students
10_grpo_readiness_sampling
11_chunk_advantage_reanalysis
12_write_report
```

---

## 8. Stage 02 — Generate MiniMax candidate compressions

For each case:

1. Generate one greedy ACON UTCO compression:

```text
temperature = 0.0
seed = 42
```

2. Generate `N=8` stochastic samples:

```text
temperature = 0.7
seeds = 1000 ... 1007
```

Optional if budget allows: also generate `N=16` for pass@N curves.

Output:

```text
outputs/raw/minimax_candidates.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "candidate_id": "sample_03",
  "generation_type": "greedy|sample",
  "sample_seed": 1003,
  "temperature": 0.7,
  "compressed_text": "...",
  "compressed_chars": 1234,
  "compressed_tokens_est": 308,
  "prompt_sha256": "..."
}
```

---

## 9. Stage 03 — Recompression stress

For every candidate, run:

```text
T^0(c) = c
T^1(c) = ACON_UTCO(c)
T^2(c) = ACON_UTCO(T^1(c))
```

Use MiniMax-M2.5 as the stress recompressor. Use the same ACON UTCO prompt and renderer as v9.

Output:

```text
outputs/raw/stress_chains.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "candidate_id": "sample_03",
  "round": 0,
  "context_text": "...",
  "chars": 1234,
  "tokens_est": 308,
  "text_sha256": "..."
}
```

---

## 10. Stage 04 — Behavior evaluate C1 and CK

For every candidate, run the downstream MiniMax agent with:

- `C1 = T^0(c)`
- `CK = T^K(c)`

Use identical downstream AppWorld runner settings as v9.

Output:

```text
outputs/raw/behavior_runs_candidates.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "candidate_id": "sample_03",
  "eval_round": "C1|CK",
  "success": true,
  "score": 1.0,
  "num_steps": 9,
  "compressed_chars": 1234,
  "output_dir": "...",
  "termination_reason": "task_completed"
}
```

Primary tables:

```text
outputs/tables/best_of_n_summary.csv
outputs/tables/best_of_n_by_case.csv
outputs/tables/c1_ck_fragility_by_generation.csv
outputs/tables/reward_spread_by_case.csv
```

---

## 11. Stage 05 — Proxy score candidates

Compute proxy scores for all candidates at `C1` and `CK`.

### 11.1 MiniMax continuation verifier score

Prompt MiniMax with:

- user task;
- compressed context;
- latest / relevant AppWorld setup;
- strict rubric.

Ask it to output JSON:

```json
{
  "predicted_success_probability": 0.0,
  "missing_information_risk": 0.0,
  "execution_specificity": 0.0,
  "risk_of_repeating_completed_actions": 0.0,
  "risk_of_wrong_api_arguments": 0.0,
  "short_reason": "..."
}
```

Do not use Qwen for this verifier.

### 11.2 Pairwise MiniMax preference

For each case, compare greedy against each sample under `CK`:

```text
Which compressed context is more likely to let the unchanged AppWorld agent finish the task?
```

Output:

```json
{
  "winner": "A|B|tie",
  "confidence": 0.0,
  "reason": "..."
}
```

### 11.3 Future-action NLL proxy

If logprobs are available for a local model, compute teacher-forced likelihood of future successful actions. If only Qwen supports this, mark it as auxiliary:

```text
proxy_model = qwen3-4b
proxy_role = auxiliary_only
```

Do not let this proxy be the sole final evidence.

Output:

```text
outputs/raw/proxy_scores.jsonl
```

---

## 12. Stage 06 — Proxy selection analysis

For each case and eval round:

1. Select candidate by each proxy.
2. Compare pass rate of proxy-selected candidate against:
   - greedy;
   - oracle best-of-N;
   - random sample.
3. Compute oracle recovery fraction:

```text
recovered_gain = (pass(proxy_selected) - pass(greedy)) / (pass(oracle_best) - pass(greedy) + eps)
```

Metrics:

- proxy-selected pass rate;
- top-1 and top-2 pass rate;
- AUROC for pass prediction;
- Spearman correlation with true reward;
- oracle regret;
- length of selected candidate;
- pass per token.

Outputs:

```text
outputs/tables/proxy_selection_summary.csv
outputs/tables/proxy_by_case.csv
outputs/figures/fig_proxy_vs_oracle_bestofn.pdf
outputs/figures/fig_proxy_roc.pdf
```

Success threshold:

- A useful proxy recovers at least 40% of oracle best-of-N gain over greedy, or improves pass over greedy by at least 10 percentage points on CK.

---

## 13. Stage 07 — Construct teacher targets

Construct two teacher target sets from MiniMax candidate pool.

### 13.1 C1-selected target

```text
teacher_C1 = argmax_i [Pass(C1_i), -length(C1_i), -num_steps(C1_i)]
```

### 13.2 CK-selected target

```text
teacher_CK = argmax_i [Pass(CK_i), -length(CK_i), -num_steps(CK_i)]
```

If multiple samples pass, choose the shortest. If no sample passes, either:

- exclude case from SFT target set; or
- use best proxy score with `target_quality = weak`.

Primary SFT should use only `target_quality = strong` cases.

Output:

```text
outputs/data/sft_targets_c1.jsonl
outputs/data/sft_targets_ck.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "input_text": "...",
  "target_text": "...",
  "target_type": "C1|CK",
  "teacher_candidate_id": "sample_05",
  "teacher_pass_C1": true,
  "teacher_pass_CK": true,
  "target_chars": 987,
  "target_quality": "strong|weak",
  "selection_reason": "stress_pass_shortest"
}
```

---

## 14. Stage 08 — Train Qwen3-4B SFT students

Train two LoRA SFT students:

```text
Qwen-SFT-C1: train on sft_targets_c1.jsonl
Qwen-SFT-CK: train on sft_targets_ck.jsonl
```

Recommended minimal hyperparameters:

```yaml
base_model: qwen3-4b-instruct-2507
lora_rank: 16
lora_alpha: 32
learning_rate: 1e-4
epochs: 2
batch_size: 2-4
gradient_accumulation: set to fit GPU
max_seq_length: 12000 or largest feasible
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
```

Training prompt should use the same ACON UTCO prompt format as v9, with the raw history as input and teacher target as completion.

Outputs:

```text
outputs/models/qwen_sft_c1/
outputs/models/qwen_sft_ck/
outputs/logs/sft_c1_train.log
outputs/logs/sft_ck_train.log
```

---

## 15. Stage 09 — Evaluate student compressors

Evaluate:

| Student | Description |
|---|---|
| `Raw-Qwen` | base Qwen3-4B with ACON UTCO prompt |
| `Qwen-SFT-C1` | trained on one-step targets |
| `Qwen-SFT-CK` | trained on stress-selected targets |
| `MiniMax-greedy` | teacher baseline |
| `MiniMax-oracle-bestofN` | upper bound diagnostic |

For each student, generate greedy compression at temperature 0.0.

Evaluate both:

- `C1` downstream pass;
- `CK` downstream pass after MiniMax or frozen-student stressor.

Recommended stressor for student evaluation:

1. primary: frozen `Qwen-SFT-CK` stressor;
2. secondary: MiniMax ACON stressor for cross-model stress.

Outputs:

```text
outputs/raw/student_behavior_runs.jsonl
outputs/tables/student_eval_summary.csv
outputs/figures/fig_student_c1_ck_pass.pdf
outputs/figures/fig_student_stress_fragility.pdf
```

Success criterion:

- `Qwen-SFT-CK` should beat `Raw-Qwen` on CK pass rate.
- `Qwen-SFT-CK` should beat or match `Qwen-SFT-C1` on CK pass rate and stress fragility.
- Output length should not inflate by more than 20% relative to teacher targets.

---

## 16. Stage 10 — GRPO readiness sampling

This stage does not require full GRPO training. It tests whether the SFT student distribution is suitable for GRPO.

For each held-out case and each Qwen model (`Raw-Qwen`, `SFT-C1`, `SFT-CK`):

1. Generate one greedy output.
2. Generate `N=8` stochastic outputs at temperature 0.7.
3. Apply stress `T^K`.
4. Evaluate proxy reward and, on a subset, true pass reward.

Metrics:

```text
within_case_reward_std
oracle_win_rate_over_greedy
best_of_n_gain
all_fail_rate
all_pass_rate
reward_entropy
length_reward_correlation
```

Outputs:

```text
outputs/tables/grpo_readiness_summary.csv
outputs/tables/grpo_readiness_by_case.csv
outputs/figures/fig_grpo_reward_spread.pdf
```

GRPO readiness criterion:

- `SFT-CK` should have lower all-fail rate and non-trivial reward spread compared with raw Qwen.
- At least 50% of cases should have at least one sampled output better than greedy under proxy or true stress reward.

---

## 17. Stage 11 — Chunk advantage reanalysis (revised after v9)

### 17.1 Goal

Do **not** test `causal chunks > entity chunks` as a primary hypothesis.

The new goal is:

> Determine whether surface chunk labels are insufficient as reward proxies, and whether behavior-based chunk advantage is a more reliable credit signal.

### 17.2 Chunk selection

Use candidates from:

- MiniMax greedy;
- MiniMax oracle best-of-N;
- proxy-selected candidates;
- Qwen-SFT-C1;
- Qwen-SFT-CK.

Prioritize candidates where full compressed context passes under CK, so leave-one-chunk-out deltas are interpretable.

### 17.3 Segment chunks

Split compressed text into natural chunks:

- bullet line;
- sentence;
- short paragraph;
- code/API line;
- table row if present.

Cap at 12 chunks per candidate. Preserve original order.

### 17.4 Leave-one-chunk-out behavior advantage

For each chunk `z_j`:

```text
c_minus_j = compressed context with z_j removed
adv_j = score(full c) - score(c_minus_j)
```

Compute both:

```text
pass_advantage
future_action_nll_advantage   # if available
verifier_score_advantage
```

If full `c` fails, mark chunk advantages as low-confidence.

### 17.5 MiniMax chunk labeling

Use MiniMax only.

Labels should include both surface form and possible function:

```json
{
  "chunk_type": "ACTION_OUTCOME|RUNTIME_BINDING|ENTITY_LIST_ONLY|CONTROL_NEGATIVE_EVIDENCE|CAUSAL_PRECONDITION|NARRATIVE_PROGRESS|TASK_GOAL_OR_TODO|OTHER",
  "contains_exact_literals": true,
  "contains_entity_list_form": true,
  "contains_causal_relation": false,
  "contains_negative_evidence": false,
  "contains_action_outcome": true,
  "contains_runtime_binding": true,
  "functional_role_guess": "api_argument_binding|object_set_binding|failure_prevention|progress_summary|unknown",
  "confidence": 0.0
}
```

### 17.6 Analysis

Compute:

1. Mean advantage by chunk type.
2. Mean advantage by functional role guess.
3. Correlation of advantage with:
   - entity count;
   - literal count;
   - chunk length;
   - contains causal relation;
   - contains negative evidence;
   - contains runtime binding;
   - contains action outcome.
4. Regression:

```text
advantage ~ entity_count + literal_count + chunk_length + chunk_type + functional_role_guess
```

5. Top-advantage chunk inspection: top 20 chunks by behavior advantage.

### 17.7 Expected interpretation

Accept any of the following as useful:

- If label-based predictors are weak, conclude that surface labels are insufficient and behavior-based advantage is needed.
- If `ENTITY_LIST_ONLY` remains high, interpret it as exact runtime binding value, not as proof that entity extraction is enough.
- If `CONTROL_NEGATIVE_EVIDENCE` remains high, use it as an example of a behavior-critical chunk type, not a hand-crafted reward.

Do not claim that causal chunks generally dominate entity chunks unless the widened analysis supports it.

Outputs:

```text
outputs/tables/chunk_advantage_revised.csv
outputs/tables/chunk_advantage_by_functional_role.csv
outputs/tables/chunk_advantage_regression.csv
outputs/reports/top_behavior_chunks.md
outputs/figures/fig_chunk_advantage_revised.pdf
```

---

## 18. Stage 12 — Write report

The report should answer these questions:

1. How much oracle best-of-N gain can proxy selection recover?
2. Are stress-selected targets better SFT targets than one-step targets?
3. Does SFT move Qwen3-4B into a GRPO-trainable region?
4. Are chunk surface labels reliable reward proxies?
5. Does the evidence support fixed-point-stressed GRPO as the next method?

Output:

```text
outputs/reports/motivation_v10_results_summary.md
```

---

## 19. Go / no-go criteria

### Go for GRPO if all hold:

1. Proxy-selected CK pass improves over greedy by at least 10 percentage points, or recovers at least 40% of oracle best-of-N gain.
2. `Qwen-SFT-CK` beats raw Qwen and `Qwen-SFT-C1` on CK pass or CK proxy reward.
3. `Qwen-SFT-CK` sampled outputs show non-degenerate reward spread: at least 50% of cases have one sample better than greedy.
4. Chunk reanalysis shows surface labels alone are insufficient to explain behavior advantage, supporting behavior-based credit assignment.

### No-go or revise if:

- proxy selection cannot recover any best-of-N gain;
- Qwen SFT does not improve over raw Qwen;
- stress-selected targets are not better than one-step targets;
- all sampled student outputs are all-fail or all-pass, making GRPO advantage degenerate;
- behavior advantage is fully explained by entity/literal count, making IAPO-style chunk shaping unnecessary.

---

## 20. Final expected paper framing if v10 succeeds

The intended conclusion is:

> ACON prompt optimization creates a capable compression distribution, but greedy decoding from that distribution is behaviorally suboptimal. High-reward, stress-robust compressions already exist. We can use them to warm-start a small compressor and then optimize the compressor policy with fixed-point-stressed RL. Chunk-level analysis shows that behavior credit should be assigned by downstream information contribution, not by surface labels such as entity vs causal prose.

This keeps the system lightweight:

```text
training time: best-of-N + stress + proxy / pass reward + SFT / GRPO
inference time: one Qwen student compressor call
```

No retrieval. No runtime fact table. No projection. No agent policy change.
