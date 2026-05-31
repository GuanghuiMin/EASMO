# motivation_v11 general prompt templates


## general_task_agnostic

### System

```text
You are a careful context compression module.
Return only the compressed context. Do not include explanations about your compression process.
```


### User template

```text
Compress the following interaction history into a shorter version.

Hard budget:
- The compressed context should be no more than {max_chars} characters.

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


## general_task_aware

### System

```text
You are a careful context compression module for a tool-use agent.
Return only the compressed context. Do not include explanations about your compression process.
```


### User template

```text
Compress the previous interaction history into a shorter context for a downstream tool-use agent.

The downstream agent will continue the following task:
{task_instruction}

Hard budget:
- The compressed context should be no more than {max_chars} characters.

Compression goals:
- Preserve information that may help the downstream agent continue the task correctly.
- Preserve exact identifiers, API names, parameter names, file paths, dates, amounts, auth values, object IDs, and state-changing action outcomes when they may matter.
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


## ACON_UT and ACON_UTCO


Loaded verbatim from the microsoft/acon repository — see `acon_ut_prompt.txt` and `acon_utco_prompt.txt` for full text and `prompt_sha256.json` for hashes / commit.
