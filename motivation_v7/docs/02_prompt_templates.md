# Prompt templates used in motivation_v7

This file documents every prompt the pipeline issues. The ACON
compressor prompts are **not rewritten** locally — they live under
`prompts/` as verbatim copies of the official microsoft/acon repo
(commit `d63f9ae18959dc7215ff62899c94c5e8c56847ae`). The other four
prompts (fact inventory, need-condition generator, retention scorer,
condition validator) are spec-mandated and are reproduced here for
clarity. SHA256 of each prompt is in
`outputs/provenance/acon_prompt_sha256.json`.

## 1. ACON UT (utility-only baseline)

Source: `experiments/appworld/prompts/context_opt/prompt_history_v2.jinja`
SHA256: `0508caa837c50403be2c8545646359a0fb72009fb14df3a8acf85aedaf649834`
`max_chars` variable: **absent** — no token-budget control inside the template.

Structure:
- system_prompt.jinja → "You are an agent tasked with extracting and refining a concise and optimized version of the context based on the user instruction and other provided information."
- user template asks for two sections — `### REASONING` and `### COMPLETED` — followed by an information block with `USER INSTRUCTION`, `PREVIOUS SUMMARY`, and `HISTORY OF INTERACTIONS`.

## 2. ACON UTCO (utility + compression optimised)

Source: `experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja`
SHA256: `9e50d0f93aca7f75eb723a90a758642d1aac3d7550f6afe1e692e56e2bc7b71c`
`max_chars` variable: **absent**.

This is the prompt-optimizer-tuned variant that adds:
- `### STATE RETAINED` section enumerating persistent state variables
  (`already_processed_song_ids`, `current_highest_liked_song`,
  `processing_progress`, `cached_api_results`, `early_exit_threshold`).
- 7 explicit COMPRESSION RULES emphasising preserving actual data
  (not just references), tracking processing state explicitly,
  never repeating discovery steps, batching operations, retaining
  error context.

Primary v7 results use UTCO.

## 3. ACON renderer convention

```python
rendered = jinja2.from_string(template_text).render(
    task=condition_task,
    prev_summary="",        # primary mode (spec §10.4)
    history=current_context,
    max_chars=1500,         # no-op if template doesn't reference it
)
```

Iterative compression at round `r+1` feeds the round-`r` compressed
text back through the `history` slot (not through `prev_summary`),
matching spec §10.4 primary mode.

## 4. Fact-inventory prompt (Appendix A)

Used once per case to extract atomic facts with a verbatim
`source_quote`. The prompt enumerates the 16 fact types and demands
JSON output. Lives in `scripts/02_extract_fact_bank.py` as the module-
level `_FACT_INVENTORY_TEMPLATE`.

Key constraints:
- ≥12 and ≤25 facts per case.
- Every fact must include `source_quote`, `fact_type`, `is_exact_literal`,
  optional `literal_values`, and a one-sentence `why_it_might_matter`.

## 5. Need-condition generator (Appendix B)

Used once per fact. Produces matched `needed` + `unneeded`
`condition_task` strings without quoting the target fact. Lives in
`motivation_v7/need_conditions.py:_NEED_CONDITION_TEMPLATE`.

## 6. Need-condition rule-based quality check

After generation we run:

1. `needed.lower() ∩ canonical_fact.lower()` — must be empty.
2. `unneeded.lower() ∩ canonical_fact.lower()` — must be empty.
3. neither condition contains any `literal_value` of the fact.
4. `abs(len(needed) - len(unneeded)) / max(len(needed), len(unneeded)) ≤ 0.35`.

Failing pairs are stored but **excluded** from primary analysis.

## 7. Retention scorer prompt (Appendix C)

Used per `(fact, compressed_text)` pair when the deterministic
substring match is **not** exact. Returns
`{retention_label, retention_score, evidence_in_compressed_text,
is_distorted, confidence, short_reason}`. Lives in
`motivation_v7/retention.py:_RETENTION_SCORER_TEMPLATE`.

Label-to-score mapping (spec §14):
```
exact 1.0 · semantic 0.75 · partial 0.4 · absent 0.0 · contradicted -0.5
```

## 8. Cross-model scorer rule

Per spec §4: scorer ≠ compressor. We map:

| compressor | scorer |
|---|---|
| Qwen3-4B-Instruct-2507 | MiniMax-M2.5 |
| MiniMax-M2.5 | Qwen3-4B-Instruct-2507 |

The deterministic substring score is computed first; if exact, the LLM
call is skipped and the result is labelled `exact`. This saves ~30 %
of LLM scoring calls in expectation because many concrete facts are
verbatim-preserved in compressed outputs (especially API names, paths).

## 9. Generation settings (spec §11)

```yaml
temperature: 0.0
seed: 42                   # passed if supported
max_tokens: 2048            # compression; 384 for retention scoring
response_format: {"type": "json_object"}   # only on JSON calls
```

We do not run the spec's secondary seed sweep (1, 2, 3) — primary
results are deterministic single-seed.
