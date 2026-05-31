# motivation_v11 prompt inventory

> Frozen 2026-05-31 PT. ACON commit:
> `d63f9ae18959dc7215ff62899c94c5e8c56847ae`. All sha256 + raw
> templates dumped by `scripts/00_prepare.py` to
> `outputs/provenance/`.

## 1. The 4 compression prompt families

| family | path / source | template_kind | uses {task_instruction} ? |
|---|---|---|:---:|
| `general_task_agnostic` | in-repo `motivation_v11/prompt_families.py` | python `.format()` | ✗ |
| `general_task_aware` | in-repo `motivation_v11/prompt_families.py` | python `.format()` | ✓ |
| `ACON_UT` | `acon/experiments/appworld/prompts/context_opt/prompt_history_v2.jinja` | jinja2 | ✓ |
| `ACON_UTCO` | `acon/experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja` | jinja2 | ✓ |

ACON system prompt is loaded from
`acon/experiments/appworld/prompts/context_opt/system_prompt.jinja` and
is shared by `ACON_UT` and `ACON_UTCO`.

### 1.1 general_task_agnostic — generic control

> Purpose (spec §5.1): a generic summarization baseline that does
> NOT know what the agent is going to do next.

```text
System: You are a careful context compression module.
        Return only the compressed context. Do not include
        explanations about your compression process.

User:   Compress the following interaction history into a shorter
        version. Hard budget: ≤{max_chars} chars. Preserve important
        information. Remove redundant/obsolete details. ... Return
        plain text only. ... Interaction history: {context}
        Compressed context:
```

### 1.2 general_task_aware — generic with task framing

> Purpose (spec §5.2): a generic prompt that ALSO sees the user task,
> so it can preserve task-relevant info — but it is still a
> non-structured compressor.

```text
User:   Compress the previous interaction history into a shorter context
        for a downstream tool-use agent.

        The downstream agent will continue the following task:
        {task_instruction}

        Hard budget: ≤{max_chars} chars.
        Preserve information that may help the downstream agent
        continue the task correctly. Preserve exact identifiers,
        API names, parameter names, file paths, dates, amounts,
        auth values, object IDs, and state-changing action outcomes
        when they may matter. ...
        Interaction history: {context}
        Compressed context:
```

### 1.3 + 1.4 ACON_UT and ACON_UTCO

Loaded verbatim from the microsoft/acon repository. See
`outputs/provenance/acon_ut_prompt.txt` and
`acon_utco_prompt.txt` for full text. The two share the same system
prompt (`acon_system_prompt.txt`).

Rendering convention (matches v7/v8/v9/v10 ACON_UTCO usage):

```python
import jinja2
jinja2.Environment(...).from_string(template_text).render(
    task=task_instruction,
    prev_summary="",
    history=context,
    max_chars=2000,  # likely no-op for official templates
)
```

## 2. Downstream agent prompt (stage 05)

Reused **verbatim** from v3 `motivation_v3.prompts.DOWNSTREAM_AGENT_INSTRUCTION`
via `motivation_v4.runner.run_with_compressed_context`. The agent
prompt is spliced into ACON's `direct` strategy at the canonical
`Using these APIs, now generate code to solve the actual task:`
marker. Identical to v9/v10 — keeps cross-track comparability.

## 3. Verbal selectors

### 3.1 Pointwise verifier (`scripts/06a_pointwise_verifier.py`, spec §10.7)

Source: `motivation_v11/selectors.py` (`POINTWISE_SYSTEM`,
`POINTWISE_USER_TEMPLATE`). Returns 5-axis JSON. Selector score:

```python
selector_score = sufficiency_score
                 − 0.25 * risk_score
                 − 0.02 * (length_chars / 1000)
```

`max_tokens=1536` (above `WARN_THINKING_MIN_MAX_TOKENS=1024` from
v10's lesson — MiniMax thinking ≈ 543 tokens median).

### 3.2 Pairwise tournament (`scripts/06b_pairwise_verifier.py`, spec §10.8)

`PAIRWISE_SYSTEM` + `PAIRWISE_USER_TEMPLATE`. Tournament:
`current = sample_0; for i in 1..7: current = pairwise_winner(current, sample_i)`.

`max_tokens=1024`. Per (task, family, eval_round) one winner.

### 3.3 Continuation entropy (`scripts/06c_continuation_entropy.py`, spec §10.9)

`ENTROPY_SYSTEM` + `ENTROPY_USER_TEMPLATE`. Sample `M=5` short
diagnostic continuations per (candidate, eval_round). Aggregate:

* `next_action_type_entropy` — Shannon entropy over predicted action types
* `argument_key_jaccard_distance` — pairwise mean 1 − Jaccard over `required_arguments_keys`
* `missing_info_count_variance` — variance of `len(missing_information)`
* `confidence_entropy` — Shannon entropy over `confidence` ∈ {high, medium, low}

Selector score:

```python
score = -(H_action + J_dist + 0.25 * V_miss + H_conf + 0.02 * len_kchars)
```

Lower disagreement (higher score) is preferred. `max_tokens=1024`, `temperature=0.7`, seeds `2000..2004`.

**Initial scope (plan β)**: only `ACON_UTCO`. To upgrade to plan α
(all 4 families), re-run with
`--families general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO`
(incremental, no rework).

## 4. SHA256 manifest

Auto-written to `outputs/provenance/prompt_sha256.json` by stage 00,
along with the rendered example prompts at
`outputs/provenance/rendered_prompt_examples/{family}/{task_id}.txt`.
