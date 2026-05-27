# Motivation Findings

## Observation 1: API Schema and Parameter Loss Dominates ACON Failures

The failure mode distribution shows that **LOST_API_SCHEMA_OR_PARAMETER** accounts for 12 out of 24 cases (50% of all failures). This is followed by LOST_ACTION_OUTCOME (4 cases), LOST_ACCESS_TOKEN (3 cases), and LOST_ENVIRONMENT_STATE (2 cases). The dominance of schema/parameter loss indicates that the compressor most frequently discards the structural information needed for agents to make correct API calls.

## Observation 2: Audit Model Primarily Recovers Action Outcomes and Environment State

The audit_added_categories distribution shows the model most frequently adds back:
- action_outcome: 8 items
- environment_state: 7 items
- negative_evidence: 4 items
- pending_task: 4 items
- auth: 3 items
- api_schema: 3 items

This suggests the audit model identifies execution feedback (action outcomes) and contextual state as frequently missing from compressed contexts.

## Observation 3: Compressor Repeatedly Drops What Audit Recovers

The recovered_then_dropped_rate is **92.86%** (26 out of 28 items). The compressor most frequently drops:
- action_outcome: 10 items
- environment_state: 5 items
- negative_evidence: 4 items
- auth: 3 items
- pending_task: 3 items

This high drop rate, particularly for action outcomes and environment state, indicates the compressor systematically removes information the audit model identifies as critical.

## Implications for Method Design

The data suggests the bottleneck is not **reasoning** (0% reasoning_failure_rate) but rather **compression aggressiveness**. The audit model can identify what matters (28 items recovered), but the compressor consistently discards it (26 items dropped again). This implies:

1. **Importance weighting mechanisms** that prevent critical items from being dropped during compression are needed
2. **Schema preservation** methods should be prioritized given the 50% failure rate from schema loss
3. **Execution trace retention** (action outcomes) is essential for maintaining agent functionality

## Negative Results / What Not to Pursue

- **Rule-based grounding alone is insufficient**: The rule_based_mean_grounding_score is only 0.4444, indicating simple heuristics fail to reliably identify critical information
- **Model agreement is low**: Qwen-Minimax agreement on primary mode (0.42), causality (0.50), and recovered items (0.38) suggests current audit approaches lack consistency
- **Re-compression does not solve the problem**: After re-compression, only 5 of 24 cases succeeded, while 19 still failed—indicating iterative compression is not a viable fix

## Representative Cases

| Case ID | Primary Failure | Recovered-Dropped Items | Outcome |
|---------|-----------------|------------------------|---------|
| 530b157_3_cap8 | LOST_API_SCHEMA_OR_PARAMETER | 2 | Missing API schema caused auth and message sending failures |
| 37a8675_2_cap8 | LOST_API_SCHEMA_OR_PARAMETER | 2 | Missing schema and action outcome caused agent failure |
| 57c3486_3_cap8 | LOST_API_SCHEMA_OR_PARAMETER | 1 | Missing API parameters caused incorrect API calls |

These cases illustrate that API schema loss leads to cascading failures in downstream tool use, confirming the aggregate finding that schema preservation is the most critical compression design constraint.