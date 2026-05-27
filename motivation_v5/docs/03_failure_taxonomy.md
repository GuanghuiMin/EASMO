# motivation_v5 — Failure Taxonomy

> Spec reference: §6 (taxonomy + definitions). All audit prompts
> require the auditor to choose one **`primary_failure_mode`** and
> any number of **`secondary_failure_modes`** from this fixed set.

## 1. The 16 labels

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

The first 12 are *compression-caused* labels (`is_compression_caused = True`
should be coupled with one of these). The 13th–14th
(`TOOL_OR_API_MISUSE_NOT_CAUSED_BY_COMPRESSION` and
`AGENT_REASONING_FAILURE_NOT_COMPRESSION`) are *reasoning-caused*; the
last two (`INSUFFICIENT_EVIDENCE`, `OTHER`) are escape hatches.

## 2. Definitions (spec §6)

### MISSING_RUNTIME_VARIABLE
A variable needed for later API calls was omitted or replaced by
vague prose. Examples: `access_token`, `file_path`, `directory_path`,
`page_index`, `page_limit`, `message_id`, `transaction_id`,
`playlist_id`, `album_id`, `user_id`, `email`, `amount`, `date range`.

### LOST_AUTH_OR_ACCESS_TOKEN
The context lost whether login was required, which credentials were
used, or how to pass the returned token.

### LOST_API_SCHEMA_OR_PARAMETER
The context lost an API name, required parameter, default value,
response field, or action format needed for future calls.

### LOST_ENVIRONMENT_STATE
The context lost the current state of the simulated environment:
files already modified, messages already deleted, queue state,
payments sent, notes updated, etc.

### LOST_ACTION_OUTCOME
The context retained that an action was attempted but dropped whether
it succeeded, failed, returned empty, returned paginated results, or
produced a specific object.

### LOST_PENDING_SUBTASK
The context loses what remains to be done.

### LOST_NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT
The context loses information like "this path failed", "no proxy API
exists", "do not call this again", or "search query X returned
irrelevant results".

### STALE_OR_CONFLICTING_STATE
The summary preserves an old value but omits a later overwrite or
correction.

### OVER_COMPRESSED_AMBIGUITY
The summary is true but too generic to execute from.

### SUMMARY_DISTORTION_OR_HALLUCINATION
The compressed summary invents, alters, or misstates a fact.

### UNNECESSARY_REDISCOVERY_OR_LOOPING
The compressed agent repeats API discovery, login, search,
pagination, or debugging that the baseline already resolved.

### PREMATURE_COMPLETION
The agent calls `complete_task` before the task is actually done.

### TOOL_OR_API_MISUSE_NOT_CAUSED_BY_COMPRESSION
The agent makes an API mistake that does not stem from compression
(e.g. a logical error in argument construction the baseline also
made but recovered from).

### AGENT_REASONING_FAILURE_NOT_COMPRESSION
The agent's reasoning is the failure cause; even with the full
context the same agent would have made the same error.

### INSUFFICIENT_EVIDENCE
Not enough signal to call any of the above.

### OTHER
Catch-all.

## 3. Observed distribution on n=24 (Stage 09 / Table 1)

| primary_failure_mode | n_cases | % | by_budget |
|---|---:|---:|---|
| **LOST_API_SCHEMA_OR_PARAMETER** | **12** | **50%** | 5 cap=15, 7 cap=8 |
| LOST_ACTION_OUTCOME | 4 | 17% | 1 cap=15, 3 cap=8 |
| LOST_ACCESS_TOKEN ★ | 3 | 13% | 1 cap=15, 2 cap=8 |
| LOST_ENVIRONMENT_STATE | 2 | 8% | 2 cap=15, 0 cap=8 |
| STALE_OR_CONFLICTING_STATE | 2 | 8% | 1 cap=15, 1 cap=8 |
| LOST_AUTH_OR_ACCESS_TOKEN | 1 | 4% | 0 cap=15, 1 cap=8 |
| AGENT_REASONING_FAILURE_NOT_COMPRESSION | 0 | 0% | — |

★ `LOST_ACCESS_TOKEN` is **NOT in the spec taxonomy** — Qwen invented
it. The spec equivalent is `LOST_AUTH_OR_ACCESS_TOKEN`. We treat them
as semantically merged for analysis (combined: 4/24 = 17%) but
preserve the raw label for fidelity. **This is a known issue with
the auditor and a documented caveat in
[`05_results_summary.md`](05_results_summary.md) §Caveats.**

## 4. Notable observations on the distribution

* **API-schema loss is dominant** (50% of failures). When ACON
  collapses concrete API names, parameters, or response-field
  references into prose, the downstream agent can't reconstruct the
  exact call. This is the single strongest signal in the round and
  the cleanest motivation for an "API-schema-preserving" compressor.
* **0% AGENT_REASONING_FAILURE.** This is suspicious — see §5 below
  for auditor bias discussion.
* **Cap=8 cases skew toward action-outcome / token loss**; cap=15
  cases skew toward environment-state / API-schema. Tighter budgets
  drop concrete past-action info first; looser budgets drop
  background environment state.

## 5. Auditor-bias caveat

The Qwen auditor was given the ACON summary explicitly framed as the
suspect ("Analyze why the ACON-compressed agent failed"). It is
therefore *primed* to attribute failures to compression rather than
to agent reasoning. Two pieces of evidence:

1. **All 24 cases are tagged `is_compression_caused=True`**. Real
   failure causality is rarely 100% one-sided.
2. **MiniMax verifier disagrees on `is_compression_caused` for 50%
   of cases** (12/24 — see Table 5 `model_agreement.csv`). On those
   12 cases MiniMax thinks the failure was at least partially
   reasoning-related.

For the paper, treat the **mode distribution** as illustrative of
*what kind of compression-caused issues exist when compression is
indeed the cause*, but use the **MiniMax-verified** rate (50%) as the
right denominator when claiming "X% of failures are caused by
compression."

## 6. Recovered-then-dropped categories (Stage 10 / Figure 2)

The spec asks (§Q3) which categories of audit-recovered info the
recompressor drops again. Audit-tagged categories are coarser than
the failure taxonomy (they tag the *content* of the recovered fact,
not the failure mode). Counts on n=26 recovered_then_dropped items:

| category | count | high-criticality count |
|---|---:|---:|
| `action_outcome` | 10 | 4 |
| `environment_state` | 5 | 4 |
| `negative_evidence` | 4 | 1 |
| `auth` | 3 | 2 |
| `pending_task` | 3 | 1 |
| `runtime_variable` | 1 | 0 |

**Top "why dropped" reasons** (from the recompression-loss audit,
multi-select):

| reason | count |
|---|---:|
| `over_abstraction` | 11 |
| `looked_like_past_log` | 10 |
| `verbosity_pressure` | 4 |
| `schema_not_supported` | 1 |

The two largest reasons (`over_abstraction` + `looked_like_past_log`)
together account for 21/26 = **81% of all recovered-then-dropped
items**. This is the most paper-quotable single number: the ACON-style
compressor's failure mode is not capacity-pressure (only 4 cases
attributed to verbosity_pressure); it is **format-aware abstraction**
that re-collapses concrete state into prose because that prose
"looks summary-shaped".

## 7. How auditors map taxonomy → categories

For convenience the audit prompts allow `category` to be one of:

```text
runtime_variable | api_schema | auth | environment_state |
action_outcome | pending_task | negative_evidence | guardrail | other
```

Mapping table (informal — used by the analyst when collapsing
counts):

| failure-taxonomy label | typical category tags |
|---|---|
| MISSING_RUNTIME_VARIABLE | runtime_variable, action_outcome |
| LOST_AUTH_OR_ACCESS_TOKEN / LOST_ACCESS_TOKEN | auth, runtime_variable |
| LOST_API_SCHEMA_OR_PARAMETER | api_schema |
| LOST_ENVIRONMENT_STATE | environment_state |
| LOST_ACTION_OUTCOME | action_outcome |
| LOST_PENDING_SUBTASK | pending_task |
| LOST_NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT | negative_evidence |
| STALE_OR_CONFLICTING_STATE | environment_state |
| (others) | other |
