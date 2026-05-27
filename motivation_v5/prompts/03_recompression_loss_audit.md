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
{{task_id}}

USER_INSTRUCTION:
{{user_instruction}}

BASELINE_HISTORY_START
{{baseline_history}}
BASELINE_HISTORY_END

ACON_COMPRESSED_CONTEXT_START
{{acon_compressed_history}}
ACON_COMPRESSED_CONTEXT_END

AUDIT_AUGMENTED_CONTEXT_START
{{audit_augmented_context}}
AUDIT_AUGMENTED_CONTEXT_END

RECOMPRESSED_CONTEXT_START
{{recompressed_context}}
RECOMPRESSED_CONTEXT_END

Proceed.
