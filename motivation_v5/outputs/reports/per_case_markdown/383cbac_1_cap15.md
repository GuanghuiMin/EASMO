# Case audit: 383cbac_1_cap15

- Task: `383cbac_1`  (budget=`max_steps=15`, difficulty=`hard`)
- baseline_success: **True**, baseline_env_steps: 22
- acon_success: **False**, acon_env_steps: 15, step_ratio: 0.682
- final_after_recompression_success: **False**

## User instruction

> 

## Qwen case-level audit (primary)

- primary_failure_mode: **LOST_API_SCHEMA_OR_PARAMETER**
- secondary_failure_modes: ['STALE_OR_CONFLICTING_STATE']
- is_compression_caused: **True**
- reliability_score: 0.15
- mechanism: _Compression caused loss of API schema details leading to incorrect API calls and authorization errors_

### Missing information (2)
  - **api_schema** `show_social_feed API parameters` (high): Missing parameters like access_token cause authorization errors
  - **api_schema** `show_social_feed API description` (high): Incorrect API name leads to failed calls

### Distorted/hallucinated (1)
  - Incorrectly assumes social feed contains transactions to manager without proper API parameters; impact: Failed to access correct API leading to authorization errors

## Addition audit

_Parse failed or not applicable_

## Recompression-loss audit

_Parse failed or not applicable_

## MiniMax verification

_Parse failed._

## Rule-based grounding check

- overall_grounding_score: 0.0
- case_audit_grounding: {'n_items': 1, 'n_grounded': 0, 'n_baseline_quote_ok': 0, 'n_acon_quote_present': 0}
- addition_audit_grounding: {}
- recompression_audit_grounding: {}
