# Prompt templates used in motivation_v8

All prompt text is canonical in `motivation_v8/prompts.py`. The
SHA256 of each rendered template lives in
`outputs/provenance/prompt_sha256.json`. Markdown copies under
`prompts/` are for human reference only.

**v8 deliberately does not load any ACON prompt.** The prompts below
are written from scratch following the v8 spec §7.

## 1. P1 — `general_task_aware`

System:

```text
You are a careful context compression module for a tool-use agent.
Return only the compressed context. Do not include explanations about your compression process.
```

User template (Jinja-style, with Python `str.format` substitution):

```text
Compress the previous interaction history into a shorter context for a downstream tool-use agent.

The downstream agent will continue the following task:
{condition_task}

Hard budget:
- The compressed context must be no more than {max_chars} characters.

Compression goals:
- Preserve information that may help the downstream agent continue the task correctly.
- Preserve exact identifiers, API names, parameter names, file paths, dates, amounts, access/auth values, object IDs, and state-changing action outcomes when they may matter.
- Preserve failed attempts or negative evidence only if they may prevent repeated mistakes.
- Remove redundant, obsolete, or irrelevant details.
- Do not invent facts.
- Do not solve the task.
- Do not output the original input.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Previous interaction history:
{context}

Compressed context:
```

## 2. P2 — `general_task_agnostic`

System:

```text
You are a careful context compression module.
Return only the compressed context. Do not include explanations about your compression process.
```

User template:

```text
Compress the following interaction history into a shorter version.

Hard budget:
- The compressed context must be no more than {max_chars} characters.

Compression goals:
- Preserve important information.
- Remove redundant, obsolete, or irrelevant details.
- Keep exact values only if they appear important in the history.
- Do not invent facts.
- Do not output the original input.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Interaction history:
{context}

Compressed context:
```

## 3. P3 — `general_strict_extract_then_compress` (defined, NOT run in v8 primary)

System:

```text
You are a loss-aware compression module for a tool-use agent.
Return only the compressed context.
```

User template:

```text
Compress the interaction history under {max_chars} characters.

Task condition:
{condition_task}

Rules:
1. First identify exact facts that could be needed later: IDs, tokens, file paths, API names, API parameters, dates, amounts, action outcomes, and failed attempts.
2. Preserve only the exact facts that are likely to matter for continuing the task.
3. Compress everything else aggressively.
4. Do not use a fixed output schema.
5. Do not invent or alter exact values.
6. Return only the compressed context.

Interaction history:
{context}

Compressed context:
```

## 4. Retention scorer prompt (cross-model)

System:

```text
You are a strict fact-retention judge.
Return only JSON. Do not explain outside JSON.
```

User template — see `motivation_v8/prompts.py::RETENTION_SCORER_TEMPLATE`.

| label | score |
|---|---:|
| exact | 1.0 |
| semantic | 0.75 |
| partial | 0.4 |
| absent | 0.0 |
| contradicted | -0.5 |

## 5. Rendering convention

```python
from motivation_v8.prompts import get_bundle
bundle = get_bundle("P1")
user_prompt = bundle.render(
    context=trajectory_text,
    max_chars=1500,
    condition_task="...",   # ignored by P2
)
```

The `{max_chars}` placeholder is filled by `compress.compress_once`
or `iterate_compression`; the v8 budget tolerance is 10 %
(`compressed_chars > 1500 * 1.10` is flagged as a budget violation
in `outputs/tables/budget_compliance_*.csv`).

## 6. Why NO ACON headings

v8 does not insert any of the following section names anywhere:

```
HISTORY_SUMMARY  REASONING  VARS  TODO  COMPLETED  GUARDRAILS  STATE RETAINED
```

The fact table used in the `DETAIL_HEAVY` and `FACT_TABLE_ONLY`
initialisations uses plain bullet text:

```
Known facts extracted from the trajectory:
- [FACT_ID=fact0001][TYPE=AUTH_OR_ACCESS_TOKEN] access_token from spotify login
- [FACT_ID=fact0002][TYPE=API_SCHEMA_OR_PARAMETER] apis.spotify.show_song(song_id)
- ...
```

The compressor sees this as a normal narrative bullet list and is
free to compress it the same way as the rest of the trajectory.
