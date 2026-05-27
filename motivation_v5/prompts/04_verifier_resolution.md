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

QWEN_AUDIT_JSON_START
{{qwen_audit_json}}
QWEN_AUDIT_JSON_END

Proceed.
