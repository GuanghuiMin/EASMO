# Prompts used in motivation_v9

## 1. ACON UTCO history compressor (spec §4)

Loaded verbatim from `microsoft/acon` commit
`d63f9ae18959dc7215ff62899c94c5e8c56847ae`:

* User template: `experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja`
  (SHA256 `9e50d0f93aca7f75eb723a90a758642d1aac3d7550f6afe1e692e56e2bc7b71c`)
* System: `experiments/appworld/prompts/context_opt/system_prompt.jinja`
  (SHA256 `f9a0a5188d643d0990a422e94557e781e9e4f9c00a6506f699c4956b7096a392`)

Rendered with the canonical variables:

```python
rendered = jinja2.from_string(template).render(
    task=user_instruction,
    prev_summary="",
    history=current_context,
    max_chars=1500,
)
```

The official template does not actually reference `max_chars` (we
discovered this in v7 — `has_max_chars_variable=False`). v9 records
`TARGET_MAX_CHARS=1500` as the *intended* budget but does not edit
the prompt to enforce it. Reported `compressed_chars` vs 1500 gives
the observed budget violation rate.

The same prompt is used for stage 02 candidate generation, stage 03
stress chains, and stage 09a chunk-context re-stressing.

## 2. Chunk type labeler (MiniMax-M2.5 only) — spec §11

System:

```
You are a strict analyst of compressed tool-use agent context. Return only JSON.
```

User template:

```
Classify the following compressed-context chunk.

The goal is to understand what kind of information this natural-language chunk carries.
Do not judge whether it is correct. Do not invent context.

Labels:
- CAUSAL_PRECONDITION: explains why a future action requires a condition, parameter, credential, or prior result.
- CONTROL_NEGATIVE_EVIDENCE: records a failed attempt, error, invalid path, or action to avoid.
- ACTION_OUTCOME: records whether a previous action succeeded, failed, returned empty, returned objects, or changed state.
- RUNTIME_BINDING: binds an exact value such as token, ID, path, email, amount, date, or object set to its role.
- ENTITY_LIST_ONLY: mostly lists entities/IDs/values without explaining use or relation.
- NARRATIVE_PROGRESS: high-level summary of progress, intent, or reasoning.
- TASK_GOAL_OR_TODO: states the goal or remaining subtask.
- OTHER: none of the above.

Return JSON:
{
  "chunk_type": "...",
  "contains_exact_literals": true,
  "contains_causal_relation": true,
  "contains_negative_evidence": false,
  "one_sentence_rationale": "..."
}

Task:
{user_instruction}

Chunk:
{chunk_text}
```

Each labeled chunk row records:

```
labeler_model = "MiniMaxAI/MiniMax-M2.5"
labeler_role  = "chunk_labeler"
qwen_used_as_auditor = false
```

Per spec §3.3 Qwen3-4B is **forbidden** as auditor; the
chunk-labeling step is the only LLM-based interpretive call in v9.

## 3. Downstream AppWorld agent prompt

v9 does not modify the downstream agent prompt — it reuses v3/v4's
`run_with_compressed_context` runner under `acon/.venv`. The
compressed context is injected as a USER turn before the AppWorld
task. Strategy = `direct`, model = `MiniMaxAI/MiniMax-M2.5`,
`max_steps = 15` (loose budget primary).

## 4. Why no rewriting

Spec §4.2 forbids edits like "preserve causal chunks". v9 is
diagnostic: we evaluate the existing ACON compressor distribution,
not a tuned one.
