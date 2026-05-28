"""General LLM compression prompt families for v8 (spec §7).

These are **not** ACON. They are deliberately plain LLM compression
prompts. The system message tells the model it is a compression
module; the user message specifies the task condition (P1) or no
condition (P2), passes the raw history, and asks for plain-text
output under a hard character budget.

Prompt families:
  P1 = general_task_aware       (uses ``condition_task``)
  P2 = general_task_agnostic    (no task condition)
  P3 = general_strict_extract   (optional ablation — not run by default)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional


P1_SYSTEM = (
    "You are a careful context compression module for a tool-use agent.\n"
    "Return only the compressed context. Do not include explanations about "
    "your compression process."
)


P1_USER_TEMPLATE = """\
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
"""


P2_SYSTEM = (
    "You are a careful context compression module.\n"
    "Return only the compressed context. Do not include explanations about "
    "your compression process."
)


P2_USER_TEMPLATE = """\
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
"""


P3_SYSTEM = (
    "You are a loss-aware compression module for a tool-use agent.\n"
    "Return only the compressed context."
)


P3_USER_TEMPLATE = """\
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
"""


@dataclass
class PromptBundle:
    family: str
    system: str
    template: str
    uses_condition_task: bool

    def render(self, *, context: str, max_chars: int,
               condition_task: Optional[str] = None) -> str:
        if self.uses_condition_task and not condition_task:
            condition_task = "(no specific task condition was provided)"
        # str.format with placeholder values not in template is fine; missing ones raise.
        return self.template.format(
            condition_task=condition_task or "",
            max_chars=max_chars,
            context=context,
        )

    @property
    def sha256(self) -> str:
        text = self.system + "\n---\n" + self.template
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


P1 = PromptBundle(
    family="general_task_aware",
    system=P1_SYSTEM,
    template=P1_USER_TEMPLATE,
    uses_condition_task=True,
)
P2 = PromptBundle(
    family="general_task_agnostic",
    system=P2_SYSTEM,
    template=P2_USER_TEMPLATE,
    uses_condition_task=False,
)
P3 = PromptBundle(
    family="general_strict_extract_then_compress",
    system=P3_SYSTEM,
    template=P3_USER_TEMPLATE,
    uses_condition_task=True,
)


_BUNDLES = {
    "P1": P1, "general_task_aware": P1,
    "P2": P2, "general_task_agnostic": P2,
    "P3": P3, "general_strict_extract_then_compress": P3,
}


def get_bundle(name: str) -> PromptBundle:
    if name not in _BUNDLES:
        raise ValueError(
            f"unknown prompt family: {name} "
            f"(choose from {sorted(set(_BUNDLES.values()), key=lambda b: b.family)})"
        )
    return _BUNDLES[name]


# ----------------------------------------------------------------------
# Retention scorer prompt (spec §12)
# ----------------------------------------------------------------------


RETENTION_SCORER_SYSTEM = (
    "You are a strict fact-retention judge.\n"
    "Return only JSON. Do not explain outside JSON."
)


RETENTION_SCORER_TEMPLATE = """\
You must decide whether a compressed context retains a target fact from an original tool-use trajectory.

Target fact:
{canonical_fact}

Fact type:
{fact_type}

Literal values, if any:
{literal_values}

Compressed context:
{compressed_context}

Labels:
- exact: the fact is preserved exactly or all literal values needed for the fact are present.
- semantic: the fact is clearly preserved with equivalent meaning, even if phrased differently.
- partial: only part of the fact is preserved; some exact values, bindings, or conditions are missing.
- absent: the fact is not present.
- contradicted: the compressed context conflicts with the fact.

Be strict for exact identifiers, file paths, tokens, API names, parameter names, dates, amounts, and object IDs. If an exact literal is needed but paraphrased or omitted, do not label it semantic.

Return JSON:
{{
  "retention_label": "exact",
  "retention_score": 1.0,
  "evidence_in_compressed_text": "short quote or empty string",
  "is_distorted": false,
  "confidence": "high",
  "short_reason": "one sentence"
}}
"""


def render_retention_prompt(
    *,
    canonical_fact: str,
    fact_type: str,
    literal_values,
    compressed_context: str,
) -> str:
    if isinstance(literal_values, (list, tuple)):
        lit = ", ".join(str(v) for v in literal_values) if literal_values else "(none)"
    else:
        lit = str(literal_values) if literal_values else "(none)"
    return RETENTION_SCORER_TEMPLATE.format(
        canonical_fact=canonical_fact,
        fact_type=fact_type,
        literal_values=lit,
        compressed_context=compressed_context,
    )


__all__ = [
    "P1_SYSTEM", "P1_USER_TEMPLATE",
    "P2_SYSTEM", "P2_USER_TEMPLATE",
    "P3_SYSTEM", "P3_USER_TEMPLATE",
    "P1", "P2", "P3", "PromptBundle",
    "get_bundle",
    "RETENTION_SCORER_SYSTEM",
    "RETENTION_SCORER_TEMPLATE",
    "render_retention_prompt",
]
