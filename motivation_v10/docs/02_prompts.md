# motivation_v10 prompt inventory

> Verbatim ACON UTCO prompts are loaded from /workspace/acon and
> recorded in outputs/provenance/. Non-ACON prompts (verifier,
> pairwise, chunk labeler) are versioned in this repo's python
> package and reproduced below.

## 1. ACON UTCO history-compression prompt (compressor + stress, stages 02 + 03)

Loaded from:

```
/workspace/acon/experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja
```

ACON commit: `d63f9ae18959dc7215ff62899c94c5e8c56847ae` (frozen, same
as v7/v8/v9). System prompt: `acon/experiments/appworld/prompts/context_opt/system_prompt.jinja`.
SHA256 of the history prompt: `9e50d0f93aca7f75eb723a90a758642d1aac3d7550f6afe1e692e56e2bc7b71c`.

Render via `motivation_v10.acon_prompt_loader.render_prompt(bundle,
task=..., history=..., max_chars=1500)`.

## 2. Downstream-agent prompt (stage 04)

Reused verbatim from v3 `motivation_v3.prompts.DOWNSTREAM_AGENT_INSTRUCTION`
via `motivation_v4.runner.run_with_compressed_context`. Spliced
into acon's `direct` strategy at the canonical `Using these APIs,
now generate code to solve the actual task:` marker. Same as v9
stage 04 — keeps cross-track comparability.

## 3. MiniMax continuation verifier (stage 05.1)

Source: `motivation_v10/motivation_v10/proxy.py` constants
`VERIFIER_SYSTEM` and `VERIFIER_USER_TEMPLATE`. Returns 5-axis JSON.
`max_tokens = 2048` (above `WARN_THINKING_MIN_MAX_TOKENS = 1024`).
Composite ranking score:

```
composite = psp − 0.5 × missing_risk + 0.3 × specificity
                  − 0.1 × repeat_risk − 0.1 × wrong_arg_risk
```

## 4. MiniMax pairwise preference (stage 05.2)

Source: `proxy.py` constants `PAIRWISE_SYSTEM` and
`PAIRWISE_USER_TEMPLATE`. For each case, every sample is compared
against greedy under CK; majority winner becomes `pairwise_selected`.
`max_tokens = 1536`.

## 5. Chunk labeler (stage 11.5, revised from v9)

Source: `motivation_v10/motivation_v10/chunk_label.py` constants
`CHUNK_LABELER_SYSTEM` and `CHUNK_LABELER_USER_TEMPLATE`. v10 enriches
the v9 schema with three new fields:

* `contains_entity_list_form` (form, separate from contains_runtime_binding which is function)
* `contains_action_outcome` (function-level)
* `contains_runtime_binding` (function-level)
* `functional_role_guess` ∈ {api_argument_binding, object_set_binding,
  failure_prevention, progress_summary, task_restatement, unknown}
* `confidence` ∈ [0,1]

The schema separates *form* (what the chunk looks like) from
*function* (what role it would play), addressing v9 §10's finding
that ENTITY_LIST_ONLY mislabels exact runtime bindings.

## 6. SFT chat template (stage 08)

```
system   = ACON UTCO system prompt (loaded verbatim)
user     = ACON UTCO history-compression prompt, rendered with
            (task=user_instruction, history=raw_full_trajectory,
             max_chars=1500)
assistant= teacher_target.compressed_text   (stripped of any <think>)
```

The chat template is applied by `Qwen3-4B-Instruct`'s tokenizer
(`apply_chat_template`). This means the deployed student takes the
exact same prompt as the MiniMax teacher would in v9 stage 02 and
emits a compression in the same format.

## 7. GRPO readiness sampling (stage 10)

Re-uses ACON UTCO prompt (no new prompt). Just changes decoding:

* greedy: temperature=0.0, seed=42
* samples: temperature=0.7, seeds=1000..1007

Same prompt rendering, same `max_tokens=2048`, same `max_chars=1500`.
