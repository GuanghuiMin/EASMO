# Case audit: 383cbac_1_cap8

- Task: `383cbac_1`  (budget=`max_steps=8`, difficulty=`hard`)
- baseline_success: **True**, baseline_env_steps: 22
- acon_success: **False**, acon_env_steps: 8, step_ratio: 0.364
- final_after_recompression_success: **True**

## User instruction

> 

## Qwen case-level audit (primary)

- primary_failure_mode: **LOST_AUTH_OR_ACCESS_TOKEN**
- secondary_failure_modes: []
- is_compression_caused: **True**
- reliability_score: 0.0
- mechanism: _Compression caused loss of access token leading to authentication failures_

### Missing information (1)
  - **environment_state** `access_token` (high): Access token is required to authenticate API requests

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
