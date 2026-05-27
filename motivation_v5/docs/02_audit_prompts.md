# motivation_v5 — Audit and Verification Prompts (paper-appendix-ready)

> All five prompts are reproduced verbatim from the spec
> (`user_feedback/acon_appworld_failure_audit_motivation_experiment.md`)
> with placeholder syntax converted to jinja-style `{{ name }}` so
> JSON schemas embedded in the prompt body don't conflict with our
> Python template renderer (`motivation_v5.clients.render_template`).
>
> The actual prompt files are under `prompts/0?_*.md`. This doc is the
> single source of truth for the appendix; sync any prompt-text
> revisions here AND in the prompt file together.

## 1. Decision and runtime hyperparameters

| Knob | Value (all prompts) |
|---|---|
| Auditor model (prompts 01-03) | Qwen3-4B (`qwen3-4b` at `http://127.0.0.1:8000/v1`) |
| Verifier model (prompt 04) | MiniMax-M2.5 |
| Aggregator model (prompt 05) | MiniMax-M2.5 |
| temperature | 0.0 |
| top_p | 1.0 |
| `chat_template_kwargs.enable_thinking` | `false` (Qwen-only — disable Qwen's reasoning block since we want strict JSON) |
| max_tokens | 2048 (case / addition / recompression audits); 1024 (verifier, augmenter); 4096 (aggregator) |
| `response_format` | `json_object` for prompts 01-04 when supported, else post-hoc parse |

Output JSON is parsed by `motivation_v5.clients.parse_json` which
strips `<think>...</think>` blocks, code fences, and recovers from
truncated trailing characters.

## 2. Prompt 01 — Case-Level ACON Failure Audit (spec §8)

Used in Stage 06a. One call per audited case. Auditor: Qwen3-4B.

System message:

```
You are a careful auditor. Respond ONLY with the requested JSON. No prose, no preface, no explanation outside the JSON object.
```

User prompt body (verbatim from spec §8; placeholders are jinja-style
`{{ name }}` and rendered by `clients.render_template`; the literal
`{` / `}` of the embedded JSON schema pass through unchanged):

```
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

Fixed failure taxonomy: (16 labels — see docs/03_failure_taxonomy.md)

Return STRICTLY valid JSON with the following fields (full schema in prompts/01_case_failure_audit.md):
  task_id, task_name,
  primary_failure_mode, secondary_failure_modes, is_compression_caused,
  first_divergence (4 sub-fields),
  missing_information [ array of 6-field items ],
  distorted_or_hallucinated_information,
  unnecessary_reexploration_or_looping,
  what_should_have_been_preserved,
  compression_vs_reasoning_judgment (3 sub-fields),
  reliability_score,
  concise_failure_mechanism_summary

[ TASK_ID / METADATA / BASELINE / ACON_COMPRESSED / ACON_TRAJECTORY / FAILURE_REPORT blocks ]
```

The full prompt with the JSON schema and metadata block is in
`prompts/01_case_failure_audit.md` (215 lines).

## 3. Prompt 02 — Audit-Addition Audit (spec §9)

Used in Stage 06b. One call per case where `audit_augmented_context`
exists. Auditor: Qwen3-4B.

System message: same as prompt 01.

Goal: identify what the audit model added back, whether each addition
is grounded in the baseline, and whether the addition is actionable.

Output JSON top-level fields:
* `task_id`
* `audit_added_items[]` — each item has 11 fields (added_item, category,
  audit_augmented_excerpt, already_present_in_acon, acon_excerpt_if_present,
  grounded_in_baseline, baseline_evidence, is_actionable, why_it_matters,
  criticality, risk_if_absent)
* `audit_added_hallucinations_or_unverified_items[]`
* `net_effect_of_audit { adds_grounded_critical_info, adds_noise_or_hallucination, summary }`
* `reliability_score`

Full prompt: `prompts/02_audit_addition_audit.md` (62 lines).

## 4. Prompt 03 — Recompression-Loss Audit (spec §10)

Used in Stage 06c. One call per case where `recompressed_context`
exists. Auditor: Qwen3-4B. **This is the prompt that produces the
headline `recovered_then_dropped` evidence**.

Goal: identify which audit-added, grounded, actionable information was
dropped or distorted by recompression.

Output JSON top-level fields:
* `recovered_then_dropped_items[]` — 9 fields per item (item, category,
  audit_augmented_excerpt, recompressed_absent_or_changed_evidence,
  baseline_evidence, was_grounded_in_baseline, criticality,
  likely_reason_compressor_dropped_it, expected_effect_on_agent)
* `items_preserved_correctly[]`
* `items_distorted_by_recompression[]`
* `recompression_judgment { drops_critical_audit_recovered_info, mostly_safe_compression, summary }`
* `reliability_score`

Full prompt: `prompts/03_recompression_loss_audit.md` (78 lines).

## 5. Prompt 04 — MiniMax Verifier / Disagreement Resolver (spec §11)

Used in Stage 07b. Called only on cases where:
* Qwen `reliability_score < 0.7`, OR
* Qwen flags `recompression_judgment.drops_critical_audit_recovered_info`, OR
* a 20% random sample of remaining cases

Verifier: MiniMax-M2.5 (different model family from Qwen for cross-validation).

Output JSON:
```
{
  "task_id", "qwen_audit_supported",
  "unsupported_claims[]" (3 fields each),
  "missed_critical_items[]" (3 fields each),
  "verified_primary_failure_mode", "verified_is_compression_caused",
  "verified_recovered_then_dropped",
  "confidence", "one_sentence_verdict"
}
```

Full prompt: `prompts/04_verifier_resolution.md` (75 lines).

In practice on this 24-case run **all 24 cases triggered the
`reliability_score < 0.7` rule** (Qwen averages 0.15), so MiniMax
verified all 24.

## 6. Prompt 05 — Aggregate Motivation Summary (spec §12)

Used in Stage 12. One call. Generator: MiniMax-M2.5.

Goal: write a concise motivation analysis (Markdown) given:
* the aggregate stats JSON (failure-mode distribution, RTD rate, etc.)
* up to 3 representative cases

Output: a markdown report under `outputs/reports/motivation_summary.md`.
The prompt asks for these headings exactly:
```
# Motivation Findings
## Observation 1: ...
## Observation 2: ...
## Observation 3: ...
## Implications for Method Design
## Negative Results / What Not to Pursue
## Representative Cases
```

If the LLM call fails or returns empty, `12_write_motivation.py`
falls back to a deterministic template that fills the same headings
from the merged JSON directly.

Full prompt: `prompts/05_aggregate_summary.md` (40 lines).

## 7. Inline (non-spec) prompt — Audit augmenter

Used in Stage 03 to build `audit_augmented_context`. **This prompt is
not in the spec** — the spec assumes `audit_augmented_context` is
provided as input, but in our setup we have to construct it ourselves
because no upstream produced it. The augmenter is intentionally a
*minimal* recoverer: it appends a bracketed block of grounded
actionable items to the ACON summary, without rewriting the summary.

System message:

```
You are a careful audit assistant. Add back only grounded, actionable
facts. Output the [AUDIT_AUGMENTATION] block exactly, no prose.
```

User prompt: see `motivation_v5/augmenter.py::_AUGMENT_PROMPT`.

The augmenter emits exactly one bracketed block:

```
[AUDIT_AUGMENTATION]
- (CATEGORY) item_text  // brief reason it matters
- (CATEGORY) item_text  // brief reason it matters
...
[/AUDIT_AUGMENTATION]
```

with up to 12 items, each tagged by one of:
`RUNTIME_VARIABLE`, `AUTH_CREDENTIAL`, `API_SCHEMA`,
`ENVIRONMENT_STATE`, `ACTION_OUTCOME`, `PENDING_SUBTASK`,
`NEGATIVE_EVIDENCE`, `GUARDRAIL`, `OTHER`.

The augmented context fed downstream is `ACON summary + augmentation
block`, so the recompressor sees BOTH and the comparison
ACON-vs-augmented and augmented-vs-recompressed are clean.
