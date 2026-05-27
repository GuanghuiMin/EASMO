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

Proceed.
