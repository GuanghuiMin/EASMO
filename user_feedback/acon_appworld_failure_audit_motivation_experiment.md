# Motivation Experiment: ACON Failure-Mode Audit on AppWorld

## 0. Goal

This experiment is **not** a new compression method yet. The goal is to build a rigorous motivation study by auditing ACON-style AppWorld failures.

We want to answer:

> When ACON fails on AppWorld, what information was missing, what did an audit model recover or add back, and what did the compressor drop again after that?

This should produce concrete evidence for what the real compression bottleneck is before we design any new method.

---

## 1. Background and Framing

ACON optimizes natural-language compression guidelines by contrasting successful full-context trajectories against failed compressed-context trajectories. Their analysis prompt asks an auditor model to identify missing facts, distorted summaries, lost variables/states, API/action errors, inefficiency patterns, and timeline divergence. We follow that construction style, but our experiment adds one more decomposition:

```text
full context trajectory
        ↓ ACON compressor
ACON compressed trajectory / context
        ↓ audit model supplementation
AUDIT-augmented context
        ↓ compressor again
RECOMPRESSED context
```

The key object of study is not just `full vs compressed`. It is the information flow:

```text
what full context had
what ACON dropped
what audit model recovered
what compressor dropped again
whether final failure is explained by the recovered-then-dropped information
```

This is a motivation experiment for a future compression paper. Do not train a new compressor yet. Do not optimize prompts yet. First, produce a clean failure taxonomy and evidence tables.

---

## 2. Experimental Questions

### Q1. What are the dominant ACON failure modes on AppWorld?

For each failure case, categorize whether the failure is caused by:

- missing runtime variable
- lost access token / credential / auth flow
- lost API schema / API parameter
- lost environment state
- lost previous action outcome
- lost pending subtask
- lost negative evidence or failed attempt
- stale or contradictory state
- over-compressed ambiguity
- hallucinated or distorted summary
- unnecessary rediscovery / inefficient loop
- agent reasoning failure not attributable to compression

### Q2. What does the audit model add back?

Given `ACON compressed context` and the available full trajectory, the audit model may supplement missing information. We need to identify:

- exact added facts
- whether each added fact is grounded in the full trajectory
- which original step/session contains the evidence
- why the added fact matters for future action
- whether the added fact is concrete state, API schema, plan, negative evidence, or generic prose

### Q3. What does the compressor drop again?

After the audit model supplements the context, we run the compressor again. We then compare:

```text
AUDIT-augmented context vs RECOMPRESSED context
```

We want to identify:

- which audit-added facts disappear
- whether the dropped facts were grounded and critical
- whether the compressor drops them due to verbosity pressure, schema mismatch, over-abstraction, or failure to recognize future utility

### Q4. Can this produce a concrete motivation claim?

Possible outcomes:

1. **Recovered-then-dropped bottleneck**: audit model can recover useful missing state, but the compressor repeatedly discards it.
2. **State/schema bottleneck**: most failures come from lost variables, API parameters, auth tokens, file paths, IDs, or runtime values.
3. **Negative-evidence bottleneck**: most failures come from losing failed attempts, constraints, or “do not repeat this” information.
4. **Reasoning bottleneck**: compression is not the real cause; the agent fails even with enough information.
5. **Audit hallucination bottleneck**: audit model adds ungrounded or misleading information; then the issue is audit reliability, not compression.

---

## 3. Models

Use two local/OpenAI-compatible model services.

### 3.1 Primary auditor

```yaml
model_alias: qwen3-4b
role: primary_audit_model
usage:
  - case-level ACON failure audit
  - audit-addition extraction
  - recompression-loss extraction
  - preliminary taxonomy assignment
settings:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 4096 or 8192 if available
  response_format: json_object if supported
```

### 3.2 Verification auditor

```yaml
model_alias: minimax-2.5m
role: verifier_audit_model
usage:
  - verify 20% random cases
  - verify all low-confidence Qwen cases
  - resolve contradictory or suspicious audits
  - check grounding of audit-added facts against full trajectory
settings:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 4096 or 8192 if available
  response_format: json_object if supported
```

### 3.3 Important rule

Do not ask models for free-form reasoning. Always request strict JSON with quoted evidence snippets and confidence scores.

---

## 4. Data Requirements

Each audited AppWorld case should include the following fields.

### 4.1 Required input schema

Store as JSONL. One line per task case.

```json
{
  "task_id": "string",
  "task_name": "string",
  "difficulty": "easy|medium|hard|null",
  "user_instruction": "string",
  "baseline_success": true,
  "acon_success": false,
  "baseline_env_steps": 16,
  "acon_env_steps": 24,
  "step_ratio": 1.5,
  "baseline_history": "full successful trajectory without compression",
  "acon_compressed_history": "compressed history injected by ACON at the relevant session boundary",
  "acon_full_trajectory": "full trajectory produced by agent using ACON compression",
  "audit_augmented_context": "context after audit model supplementation, if available",
  "recompressed_context": "context after compressor processes audit_augmented_context, if available",
  "final_after_recompression_success": false,
  "failure_report": "optional raw benchmark failure report or evaluator output",
  "compression_type": "history|observation|history+observation",
  "acon_variant": "prompting|UT|UTCO|other",
  "agent_model": "qwen3-4b or other",
  "compressor_model": "qwen3-4b or other",
  "audit_model": "qwen3-4b",
  "verifier_model": "minimax-2.5m"
}
```

### 4.2 Case selection

Prioritize these cases:

```text
Tier 1: baseline_success = true AND acon_success = false
Tier 2: baseline_success = true AND acon_success = true BUT step_ratio >= 1.5
Tier 3: acon_success = false AND audit_augmented_context exists
Tier 4: audit_augmented_context improves trajectory BUT recompressed_context fails again
```

Suggested first batch:

```text
n_cases = 50
- 20 hard cases
- 20 medium cases
- 10 easy cases
```

If we have limited cases, start with all Tier 1 cases.

---

## 5. File Organization

Expected project structure:

```text
appworld_acon_audit/
  data/
    raw_cases.jsonl
    sampled_cases.jsonl
  prompts/
    01_case_failure_audit.md
    02_audit_addition_audit.md
    03_recompression_loss_audit.md
    04_verifier_resolution.md
    05_aggregate_summary.md
  outputs/
    qwen_case_audits.jsonl
    qwen_addition_audits.jsonl
    qwen_recompression_audits.jsonl
    minimax_verifications.jsonl
    merged_case_audits.jsonl
    tables/
      failure_mode_counts.csv
      critical_info_loss.csv
      audit_added_facts.csv
      recovered_then_dropped.csv
      model_agreement.csv
    reports/
      per_case_markdown/
      motivation_summary.md
    figures/
      failure_mode_bar.png
      recovered_then_dropped_bar.png
      information_flow_sankey.png
  scripts/
    sample_cases.py
    run_audit.py
    run_verify.py
    merge_audits.py
    aggregate.py
```

---

## 6. Failure Taxonomy

Use this fixed taxonomy. The auditor may assign multiple labels, but must choose one `primary_failure_mode`.

```json
[
  "MISSING_RUNTIME_VARIABLE",
  "LOST_AUTH_OR_ACCESS_TOKEN",
  "LOST_API_SCHEMA_OR_PARAMETER",
  "LOST_ENVIRONMENT_STATE",
  "LOST_ACTION_OUTCOME",
  "LOST_PENDING_SUBTASK",
  "LOST_NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT",
  "STALE_OR_CONFLICTING_STATE",
  "OVER_COMPRESSED_AMBIGUITY",
  "SUMMARY_DISTORTION_OR_HALLUCINATION",
  "UNNECESSARY_REDISCOVERY_OR_LOOPING",
  "PREMATURE_COMPLETION",
  "TOOL_OR_API_MISUSE_NOT_CAUSED_BY_COMPRESSION",
  "AGENT_REASONING_FAILURE_NOT_COMPRESSION",
  "INSUFFICIENT_EVIDENCE",
  "OTHER"
]
```

### Definitions

#### MISSING_RUNTIME_VARIABLE
A variable needed for later API calls was omitted or replaced by vague prose.

Examples:

```text
access_token, file_path, directory_path, page_index, page_limit, message_id, transaction_id, playlist_id, album_id, user_id, email, amount, date range
```

#### LOST_AUTH_OR_ACCESS_TOKEN
The context lost whether login was required, which credentials were used, or how to pass the returned token.

#### LOST_API_SCHEMA_OR_PARAMETER
The context lost an API name, required parameter, default value, response field, or action format needed for future calls.

#### LOST_ENVIRONMENT_STATE
The context lost the current state of the simulated environment: files already modified, messages already deleted, queue state, payments sent, notes updated, etc.

#### LOST_ACTION_OUTCOME
The context retained that an action was attempted but dropped whether it succeeded, failed, returned empty, returned paginated results, or produced a specific object.

#### LOST_PENDING_SUBTASK
The context loses what remains to be done.

#### LOST_NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT
The context loses information like “this path failed,” “no proxy API exists,” “do not call this again,” or “search query X returned irrelevant results.”

#### STALE_OR_CONFLICTING_STATE
The summary preserves an old value but omits a later overwrite or correction.

#### OVER_COMPRESSED_AMBIGUITY
The summary is true but too generic to execute from.

#### SUMMARY_DISTORTION_OR_HALLUCINATION
The compressed summary invents, alters, or misstates a fact.

#### UNNECESSARY_REDISCOVERY_OR_LOOPING
The compressed agent repeats API discovery, login, search, pagination, or debugging that the baseline already resolved.

---

## 7. Core Pipeline

### Step 1: Sample cases

Run:

```bash
python scripts/sample_cases.py \
  --input data/raw_cases.jsonl \
  --output data/sampled_cases.jsonl \
  --n 50 \
  --prefer_tier1 \
  --balance_difficulty
```

### Step 2: Case-level ACON failure audit

For each case, compare:

```text
baseline_history vs acon_compressed_history + acon_full_trajectory
```

Output:

```text
outputs/qwen_case_audits.jsonl
```

### Step 3: Audit-addition audit

If `audit_augmented_context` exists, compare:

```text
acon_compressed_history vs audit_augmented_context, grounded by baseline_history
```

Output:

```text
outputs/qwen_addition_audits.jsonl
```

### Step 4: Recompression-loss audit

If `recompressed_context` exists, compare:

```text
audit_augmented_context vs recompressed_context, grounded by baseline_history
```

Output:

```text
outputs/qwen_recompression_audits.jsonl
```

### Step 5: Verification with Minimax-2.5m

Verify:

```text
- all cases where Qwen reliability_score < 0.7
- all cases with critical recovered-then-dropped facts
- 20% random sample of remaining cases
```

Output:

```text
outputs/minimax_verifications.jsonl
```

### Step 6: Merge and aggregate

Run:

```bash
python scripts/merge_audits.py \
  --case outputs/qwen_case_audits.jsonl \
  --addition outputs/qwen_addition_audits.jsonl \
  --recompression outputs/qwen_recompression_audits.jsonl \
  --verification outputs/minimax_verifications.jsonl \
  --output outputs/merged_case_audits.jsonl

python scripts/aggregate.py \
  --input outputs/merged_case_audits.jsonl \
  --out_dir outputs
```

---

## 8. Prompt 1: Case-Level ACON Failure Audit

Save as:

```text
prompts/01_case_failure_audit.md
```

Prompt:

```text
You are an expert AppWorld agent trajectory auditor.

Analyze why the ACON-compressed agent failed or became significantly less efficient while the full-context baseline succeeded.

You are given:
- task_id
- task_name
- user_instruction
- baseline full successful trajectory without compression
- ACON compressed history/context
- ACON trajectory produced under compressed context
- success and step metadata

Your goals:
1. Identify the first meaningful divergence between the baseline and ACON trajectory.
2. Determine whether the divergence is caused by compression or by agent reasoning unrelated to compression.
3. Identify exactly what information was missing, distorted, over-compressed, or misleading in the ACON compressed context.
4. Classify the root cause using the fixed taxonomy.
5. Quote exact evidence snippets from the baseline and ACON contexts.
6. Identify what the compressed context would have needed to preserve for the agent to continue correctly.
7. Output STRICT JSON only.

Fixed failure taxonomy:
- MISSING_RUNTIME_VARIABLE
- LOST_AUTH_OR_ACCESS_TOKEN
- LOST_API_SCHEMA_OR_PARAMETER
- LOST_ENVIRONMENT_STATE
- LOST_ACTION_OUTCOME
- LOST_PENDING_SUBTASK
- LOST_NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT
- STALE_OR_CONFLICTING_STATE
- OVER_COMPRESSED_AMBIGUITY
- SUMMARY_DISTORTION_OR_HALLUCINATION
- UNNECESSARY_REDISCOVERY_OR_LOOPING
- PREMATURE_COMPLETION
- TOOL_OR_API_MISUSE_NOT_CAUSED_BY_COMPRESSION
- AGENT_REASONING_FAILURE_NOT_COMPRESSION
- INSUFFICIENT_EVIDENCE
- OTHER

Return a STRICTLY valid JSON object with this schema:

{
  "task_id": "string",
  "task_name": "string",
  "primary_failure_mode": "one taxonomy label",
  "secondary_failure_modes": ["taxonomy label"],
  "is_compression_caused": true,
  "first_divergence": {
    "baseline_step_or_phase": "string or null",
    "acon_step_or_phase": "string or null",
    "description": "string",
    "baseline_evidence": "exact quote or null",
    "acon_evidence": "exact quote or null"
  },
  "missing_information": [
    {
      "info_type": "runtime_variable|api_schema|auth|environment_state|action_outcome|pending_task|negative_evidence|other",
      "missing_item": "string",
      "baseline_evidence": "exact quote",
      "acon_absent_or_distorted_evidence": "exact quote or explanation",
      "why_it_matters": "string",
      "criticality": "low|medium|high"
    }
  ],
  "distorted_or_hallucinated_information": [
    {
      "compressed_excerpt": "exact quote",
      "correct_baseline_reference": "exact quote",
      "issue": "string",
      "impact": "string"
    }
  ],
  "unnecessary_reexploration_or_looping": [
    {
      "acon_excerpt": "exact quote",
      "baseline_contrast": "exact quote or null",
      "cause": "string",
      "excess_steps_estimate": "integer or null"
    }
  ],
  "what_should_have_been_preserved": [
    {
      "preserved_item": "string",
      "preferred_format": "VARS|TODO|COMPLETED|GUARDRAIL|API_SCHEMA|RAW_LITERAL|OTHER",
      "reason": "string"
    }
  ],
  "compression_vs_reasoning_judgment": {
    "compression_fault_probability": 0.0,
    "agent_reasoning_fault_probability": 0.0,
    "explanation": "string"
  },
  "reliability_score": 0.0,
  "concise_failure_mechanism_summary": "one sentence"
}

If a field has no evidence, use an empty list or null. Do not include commentary outside JSON.

---
TASK_ID:
{{ task_id }}

TASK_NAME:
{{ task_name }}

USER_INSTRUCTION:
{{ user_instruction }}

METADATA:
baseline_success={{ baseline_success }}
acon_success={{ acon_success }}
baseline_env_steps={{ baseline_env_steps }}
acon_env_steps={{ acon_env_steps }}
step_ratio={{ step_ratio }}
compression_type={{ compression_type }}
acon_variant={{ acon_variant }}

BASELINE_HISTORY_START
{{ baseline_history }}
BASELINE_HISTORY_END

ACON_COMPRESSED_CONTEXT_START
{{ acon_compressed_history }}
ACON_COMPRESSED_CONTEXT_END

ACON_TRAJECTORY_START
{{ acon_full_trajectory }}
ACON_TRAJECTORY_END

FAILURE_REPORT_START
{{ failure_report }}
FAILURE_REPORT_END

Proceed with rigorous comparison.
```

---

## 9. Prompt 2: Audit-Addition Audit

This prompt answers: **what did the audit model add back?**

Save as:

```text
prompts/02_audit_addition_audit.md
```

Prompt:

```text
You are an expert auditor of AppWorld context compression.

Compare the ACON compressed context with the AUDIT-augmented context.
Your job is to identify what the audit model added back, whether each addition is grounded in the full baseline trajectory, and whether the addition is useful for future task completion.

Important:
- Do not praise the audit model.
- Do not assume additions are correct.
- Every claimed addition must be grounded by exact evidence from the baseline history, or marked as ungrounded.
- Output STRICT JSON only.

Return a STRICTLY valid JSON object:

{
  "task_id": "string",
  "audit_added_items": [
    {
      "added_item": "string",
      "category": "runtime_variable|api_schema|auth|environment_state|action_outcome|pending_task|negative_evidence|guardrail|other",
      "audit_augmented_excerpt": "exact quote",
      "already_present_in_acon": true,
      "acon_excerpt_if_present": "exact quote or null",
      "grounded_in_baseline": true,
      "baseline_evidence": "exact quote or null",
      "is_actionable": true,
      "why_it_matters": "string",
      "criticality": "low|medium|high",
      "risk_if_absent": "string"
    }
  ],
  "audit_added_hallucinations_or_unverified_items": [
    {
      "added_item": "string",
      "audit_augmented_excerpt": "exact quote",
      "why_unverified_or_wrong": "string"
    }
  ],
  "net_effect_of_audit": {
    "adds_grounded_critical_info": true,
    "adds_noise_or_hallucination": true,
    "summary": "string"
  },
  "reliability_score": 0.0
}

---
TASK_ID:
{{ task_id }}

USER_INSTRUCTION:
{{ user_instruction }}

BASELINE_HISTORY_START
{{ baseline_history }}
BASELINE_HISTORY_END

ACON_COMPRESSED_CONTEXT_START
{{ acon_compressed_history }}
ACON_COMPRESSED_CONTEXT_END

AUDIT_AUGMENTED_CONTEXT_START
{{ audit_augmented_context }}
AUDIT_AUGMENTED_CONTEXT_END

Proceed.
```

---

## 10. Prompt 3: Recompression-Loss Audit

This prompt answers: **what did the compressor drop again after the audit model added it back?**

Save as:

```text
prompts/03_recompression_loss_audit.md
```

Prompt:

```text
You are an expert auditor of AppWorld context compression.

Compare the AUDIT-augmented context with the RECOMPRESSED context.
Your job is to identify which audit-added, grounded, actionable information was dropped or distorted by the compressor.

Focus especially on recovered-then-dropped information:
- facts absent from ACON compressed context
- added back by the audit model
- grounded in the baseline trajectory
- then dropped again by recompression

Output STRICT JSON only.

Return a STRICTLY valid JSON object:

{
  "task_id": "string",
  "recovered_then_dropped_items": [
    {
      "item": "string",
      "category": "runtime_variable|api_schema|auth|environment_state|action_outcome|pending_task|negative_evidence|guardrail|other",
      "audit_augmented_excerpt": "exact quote",
      "recompressed_absent_or_changed_evidence": "exact quote or explanation",
      "baseline_evidence": "exact quote or null",
      "was_grounded_in_baseline": true,
      "criticality": "low|medium|high",
      "likely_reason_compressor_dropped_it": "verbosity_pressure|schema_not_supported|over_abstraction|looked_like_past_log|credential_truncation|unknown",
      "expected_effect_on_agent": "string"
    }
  ],
  "items_preserved_correctly": [
    {
      "item": "string",
      "category": "string",
      "recompressed_excerpt": "exact quote",
      "why_it_is_sufficient": "string"
    }
  ],
  "items_distorted_by_recompression": [
    {
      "item": "string",
      "audit_augmented_excerpt": "exact quote",
      "recompressed_excerpt": "exact quote",
      "distortion": "string",
      "impact": "string"
    }
  ],
  "recompression_judgment": {
    "drops_critical_audit_recovered_info": true,
    "mostly_safe_compression": false,
    "summary": "string"
  },
  "reliability_score": 0.0
}

---
TASK_ID:
{{ task_id }}

USER_INSTRUCTION:
{{ user_instruction }}

BASELINE_HISTORY_START
{{ baseline_history }}
BASELINE_HISTORY_END

ACON_COMPRESSED_CONTEXT_START
{{ acon_compressed_history }}
ACON_COMPRESSED_CONTEXT_END

AUDIT_AUGMENTED_CONTEXT_START
{{ audit_augmented_context }}
AUDIT_AUGMENTED_CONTEXT_END

RECOMPRESSED_CONTEXT_START
{{ recompressed_context }}
RECOMPRESSED_CONTEXT_END

Proceed.
```

---

## 11. Prompt 4: Minimax Verification / Disagreement Resolver

Save as:

```text
prompts/04_verifier_resolution.md
```

Prompt:

```text
You are a strict verification auditor.

You are given:
- the original AppWorld task
- full baseline trajectory
- compressed contexts
- Qwen's audit JSON

Your job is to verify whether Qwen's causal claims are supported by exact evidence.
Do not produce a new broad analysis unless Qwen is wrong or incomplete.

Return STRICT JSON only:

{
  "task_id": "string",
  "qwen_audit_supported": true,
  "unsupported_claims": [
    {
      "claim": "string",
      "reason": "string",
      "corrected_judgment": "string"
    }
  ],
  "missed_critical_items": [
    {
      "item": "string",
      "baseline_evidence": "exact quote",
      "why_critical": "string"
    }
  ],
  "verified_primary_failure_mode": "taxonomy label",
  "verified_is_compression_caused": true,
  "verified_recovered_then_dropped": true,
  "confidence": 0.0,
  "one_sentence_verdict": "string"
}

---
TASK_ID:
{{ task_id }}

USER_INSTRUCTION:
{{ user_instruction }}

BASELINE_HISTORY_START
{{ baseline_history }}
BASELINE_HISTORY_END

ACON_COMPRESSED_CONTEXT_START
{{ acon_compressed_history }}
ACON_COMPRESSED_CONTEXT_END

AUDIT_AUGMENTED_CONTEXT_START
{{ audit_augmented_context }}
AUDIT_AUGMENTED_CONTEXT_END

RECOMPRESSED_CONTEXT_START
{{ recompressed_context }}
RECOMPRESSED_CONTEXT_END

QWEN_AUDIT_JSON_START
{{ qwen_audit_json }}
QWEN_AUDIT_JSON_END

Proceed.
```

---

## 12. Prompt 5: Aggregate Motivation Summary

Save as:

```text
prompts/05_aggregate_summary.md
```

Prompt:

```text
You are helping write the motivation section for a research paper on agent context compression.

You are given aggregated JSON statistics and representative cases from AppWorld ACON failure audits.

Write a concise motivation analysis that answers:
1. What failure modes dominate ACON failures?
2. What information does the audit model most often add back?
3. What information does the compressor most often drop again?
4. What does this imply about the real bottleneck in agent context compression?
5. Which future method directions are supported by the evidence, and which are not?

Do not invent numbers. Use only the provided aggregate statistics.
Distinguish observations from hypotheses.

Return Markdown with these headings:

# Motivation Findings
## Observation 1: ...
## Observation 2: ...
## Observation 3: ...
## Implications for Method Design
## Negative Results / What Not to Pursue
## Representative Cases

---
AGGREGATE_STATS_JSON_START
{{ aggregate_stats_json }}
AGGREGATE_STATS_JSON_END

REPRESENTATIVE_CASES_JSON_START
{{ representative_cases_json }}
REPRESENTATIVE_CASES_JSON_END
```

---

## 13. Output Schemas for Aggregation

### 13.1 merged_case_audits.jsonl

Each line should contain:

```json
{
  "task_id": "string",
  "difficulty": "easy|medium|hard|null",
  "compression_type": "history|observation|history+observation",
  "acon_variant": "prompting|UT|UTCO|other",
  "baseline_success": true,
  "acon_success": false,
  "audit_augmented_exists": true,
  "recompressed_exists": true,
  "final_after_recompression_success": false,
  "primary_failure_mode": "string",
  "secondary_failure_modes": ["string"],
  "is_compression_caused": true,
  "compression_fault_probability": 0.85,
  "agent_reasoning_fault_probability": 0.15,
  "n_missing_items": 3,
  "n_audit_added_items": 4,
  "n_grounded_audit_added_items": 3,
  "n_critical_audit_added_items": 2,
  "n_recovered_then_dropped_items": 1,
  "critical_recovered_then_dropped": true,
  "qwen_reliability": 0.82,
  "minimax_verified": true,
  "minimax_confidence": 0.78,
  "final_failure_summary": "string"
}
```

### 13.2 audit_added_facts.csv

Columns:

```text
task_id,difficulty,category,added_item,grounded_in_baseline,criticality,is_actionable,already_present_in_acon,baseline_evidence,audit_augmented_excerpt
```

### 13.3 recovered_then_dropped.csv

Columns:

```text
task_id,difficulty,category,item,criticality,likely_reason_compressor_dropped_it,baseline_evidence,audit_augmented_excerpt,recompressed_absent_or_changed_evidence,expected_effect_on_agent
```

---

## 14. Metrics

### 14.1 Failure-mode distribution

```text
count(primary_failure_mode)
count(secondary_failure_modes)
count(primary_failure_mode | difficulty)
count(primary_failure_mode | compression_type)
```

### 14.2 Audit addition grounding

```text
grounded_addition_rate = grounded_audit_added_items / audit_added_items
critical_grounded_addition_rate = grounded_and_high_criticality_items / audit_added_items
actionable_addition_rate = actionable_added_items / audit_added_items
```

### 14.3 Recovered-then-dropped rate

```text
recovered_then_dropped_rate = recovered_then_dropped_items / grounded_audit_added_items
critical_recovered_then_dropped_rate = high_criticality_recovered_then_dropped_items / high_criticality_grounded_audit_added_items
```

### 14.4 Compression-causality rate

```text
compression_caused_rate = cases_with_is_compression_caused_true / total_audited_cases
reasoning_failure_rate = cases_with_primary_failure_mode == AGENT_REASONING_FAILURE_NOT_COMPRESSION / total_audited_cases
```

### 14.5 Model agreement

```text
qwen_minimax_primary_mode_agreement = exact_match(primary_failure_mode)
qwen_minimax_compression_causality_agreement = exact_match(is_compression_caused)
qwen_minimax_recovered_then_dropped_agreement = exact_match(critical_recovered_then_dropped)
```

---

## 15. Figures to Generate

### Figure 1: Failure mode bar chart

```text
x-axis: primary_failure_mode
y-axis: number of cases
color: difficulty
```

### Figure 2: Information flow Sankey

Nodes:

```text
Full trajectory info
ACON dropped
Audit added back
Compressor preserved
Compressor dropped again
Final failure
```

### Figure 3: Audit-added information categories

```text
runtime variables vs API schema vs auth vs pending subtasks vs negative evidence
```

### Figure 4: Recovered-then-dropped categories

```text
which types of audit-recovered info are most often dropped by compressor
```

### Figure 5: Compression vs reasoning fault split

```text
compression-caused vs agent-reasoning-caused vs insufficient evidence
```

---

## 16. Acceptance Criteria

This experiment is successful if it can produce at least one of the following evidence-backed claims.

### Strong positive signal

```text
A large fraction of ACON failures are caused by concrete state/schema information that exists in the full trajectory, is recoverable by an audit model, but is dropped again by the compressor.
```

This would motivate a method focused on preserving audit-recovered actionable state.

### Medium positive signal

```text
ACON failures cluster around a small number of information types, such as auth/token handling, API parameter preservation, or pending subtask tracking.
```

This would motivate a targeted compressor design.

### Negative result

```text
Most failures are agent reasoning failures, not compression failures.
```

Then do not build a new compressor around these cases.

### Audit unreliability result

```text
Audit-added information is often ungrounded or hallucinated.
```

Then the immediate problem is not compression, but grounded audit/recovery.

---

## 17. Implementation Notes

### 17.1 Strict JSON handling

All model calls should enforce JSON. If the output is invalid:

1. retry once with the same prompt plus:

```text
Your previous output was invalid JSON. Return only a valid JSON object matching the required schema.
```

2. If retry fails, save raw output and mark:

```json
{"parse_failed": true}
```

### 17.2 Quote evidence

Auditors must provide exact snippets from contexts. If no exact quote is available, mark the item as `grounded_in_baseline=false` or `INSUFFICIENT_EVIDENCE`.

### 17.3 Long trajectories

If contexts exceed model limit:

- preserve task instruction
- preserve session boundaries
- preserve compressed summaries
- preserve last 5 action-observation pairs before failure
- preserve any API docs/actions mentioned in failure report
- truncate middle raw logs only after inserting markers:

```text
[TRUNCATED_MIDDLE_STEPS: original steps 8-19 omitted]
```

Do not truncate the compressed context, audit-augmented context, or recompressed context.

### 17.4 Model usage policy

- Qwen3-4B runs all cases.
- Minimax-2.5m verifies only selected cases to save cost/time.
- Temperature must be 0.
- Do not run chain-of-thought prompting.
- Ask for evidence snippets and judgments, not hidden reasoning.

---

## 18. Minimal Pseudocode

```python
import json
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def call_model(client, model, prompt, temperature=0.0):
    # Use OpenAI-compatible local endpoint.
    # Return parsed JSON or raw text with parse_failed=True.
    pass


def render(template, case):
    # Use jinja2 or simple replacement.
    pass


def run_case_audit(case, qwen_client):
    prompt = render(Path("prompts/01_case_failure_audit.md").read_text(), case)
    return call_model(qwen_client, "qwen3-4b", prompt)


def run_addition_audit(case, qwen_client):
    if not case.get("audit_augmented_context"):
        return None
    prompt = render(Path("prompts/02_audit_addition_audit.md").read_text(), case)
    return call_model(qwen_client, "qwen3-4b", prompt)


def run_recompression_audit(case, qwen_client):
    if not case.get("recompressed_context"):
        return None
    prompt = render(Path("prompts/03_recompression_loss_audit.md").read_text(), case)
    return call_model(qwen_client, "qwen3-4b", prompt)


def should_verify(case_audit, addition_audit, recompression_audit):
    if case_audit.get("reliability_score", 1.0) < 0.7:
        return True
    if recompression_audit and recompression_audit.get("recompression_judgment", {}).get("drops_critical_audit_recovered_info"):
        return True
    return False
```

---

## 19. Suggested CLI

```bash
python scripts/run_audit.py \
  --input data/sampled_cases.jsonl \
  --out_case outputs/qwen_case_audits.jsonl \
  --out_addition outputs/qwen_addition_audits.jsonl \
  --out_recompression outputs/qwen_recompression_audits.jsonl \
  --qwen_base_url http://localhost:8000/v1 \
  --qwen_model qwen3-4b \
  --temperature 0.0

python scripts/run_verify.py \
  --input data/sampled_cases.jsonl \
  --qwen_audits outputs/qwen_case_audits.jsonl \
  --out outputs/minimax_verifications.jsonl \
  --minimax_base_url http://localhost:9000/v1 \
  --minimax_model minimax-2.5m \
  --verify_low_confidence \
  --verify_recovered_then_dropped \
  --random_verify_ratio 0.2

python scripts/aggregate.py \
  --input outputs/merged_case_audits.jsonl \
  --out_dir outputs
```

---

## 20. What to Look For Manually

After the first 20 cases, manually inspect:

1. Are Qwen's missing-info judgments actually grounded?
2. Is the audit model adding concrete state or just generic advice?
3. Does recompression drop exact literals like IDs, tokens, paths, parameter names, or pending actions?
4. Are failures concentrated in history compression, observation compression, or both?
5. Is the failure really caused by compression, or does the agent misuse APIs even when the info is present?

The most promising motivation pattern is:

```text
The full trajectory contains a small but concrete executable state.
ACON summary abstracts it away.
The audit model can recover it from the full trajectory.
The compressor drops it again because it looks like low-level log detail.
The agent then fails by rediscovering, looping, or using the wrong API call.
```

If this pattern appears often, it gives a much clearer motivation than generic “memory compression” or “context selection.”

---

## 21. Final Deliverables

The coding agent should produce:

```text
1. outputs/merged_case_audits.jsonl
2. outputs/tables/failure_mode_counts.csv
3. outputs/tables/audit_added_facts.csv
4. outputs/tables/recovered_then_dropped.csv
5. outputs/tables/model_agreement.csv
6. outputs/figures/failure_mode_bar.png
7. outputs/figures/recovered_then_dropped_bar.png
8. outputs/figures/information_flow_sankey.png
9. outputs/reports/motivation_summary.md
10. outputs/reports/per_case_markdown/*.md
```

The final `motivation_summary.md` should make no method claims. It should only state observed bottlenecks and representative evidence.

