"""Retention scoring (spec §14).

Combine deterministic exact matching with an LLM semantic scorer that
implements the ``RETENTION_SCORER_PROMPT`` in Appendix C.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import jinja2


_RETENTION_LABEL_TO_SCORE = {
    "exact": 1.0,
    "semantic": 0.75,
    "partial": 0.4,
    "absent": 0.0,
    "contradicted": -0.5,
}


_RETENTION_SCORER_TEMPLATE = """\
Decide whether a compressed context preserves a target fact.

Retention labels:
- exact: the compressed context preserves the exact literal or exact API/ID/path/value.
- semantic: the compressed context preserves the fact accurately, but not verbatim; use only if exact literal is not required.
- partial: part of the fact is present but important details are missing.
- absent: the fact is not present.
- contradicted: the compressed context says something inconsistent with the fact.

Rules:
1. If the fact contains an exact ID, token, file path, API name, parameter name, amount, or date, semantic paraphrase is not enough; mark partial or absent unless the required literal is preserved.
2. Quote evidence from the compressed context if present.
3. Do not infer facts that are not explicitly in the compressed context.
4. Return JSON only.

Return JSON:
{
  "fact_id": "{{ fact_id }}",
  "retention_label": "exact",
  "retention_score": 0.0,
  "evidence_in_compressed_text": "verbatim quote or empty string",
  "is_distorted": false,
  "confidence": 0.0,
  "short_reason": "one sentence"
}

Target fact:
{{ canonical_fact }}

Fact type:
{{ fact_type }}

Source quote from original trajectory:
{{ source_quote }}

Literal values that must be preserved if relevant:
{{ literal_values }}

Compressed context:
{{ compressed_text }}
"""

_env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False,
                          keep_trailing_newline=True)


def render_retention_prompt(
    *,
    fact_id: str,
    canonical_fact: str,
    fact_type: str,
    source_quote: str,
    literal_values: List[str],
    compressed_text: str,
) -> str:
    return _env.from_string(_RETENTION_SCORER_TEMPLATE).render(
        fact_id=fact_id,
        canonical_fact=canonical_fact,
        fact_type=fact_type,
        source_quote=source_quote,
        literal_values=", ".join(literal_values) if literal_values else "(none)",
        compressed_text=compressed_text,
    )


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    s = re.sub(r"\s+", " ", text.lower())
    s = s.strip()
    return s


def deterministic_exact_retained(
    *,
    compressed_text: str,
    source_quote: str = "",
    verbatim_surface: str = "",
    literal_values: Optional[List[str]] = None,
) -> bool:
    """True if any of the literal cues is a substring of the
    compressed context after permissive normalisation."""
    c_norm = _normalise(compressed_text)
    candidates: List[str] = []
    if source_quote:
        candidates.append(source_quote)
    if verbatim_surface:
        candidates.append(verbatim_surface)
    for v in (literal_values or []):
        if v and len(v) >= 3:
            candidates.append(v)
    for cand in candidates:
        if cand and _normalise(cand) and _normalise(cand) in c_norm:
            return True
    return False


@dataclass
class RetentionScore:
    fact_id: str
    retention_label: str          # "exact" | "semantic" | "partial" | "absent" | "contradicted"
    retention_score: float        # in [-0.5, 1.0]
    evidence: str
    is_distorted: bool
    confidence: float
    short_reason: str
    scorer_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "retention_label": self.retention_label,
            "retention_score": self.retention_score,
            "evidence_in_compressed_text": self.evidence,
            "is_distorted": self.is_distorted,
            "confidence": self.confidence,
            "short_reason": self.short_reason,
            "scorer_error": self.scorer_error,
        }


def parse_retention_response(fact_id: str, obj: Optional[dict]) -> RetentionScore:
    if not isinstance(obj, dict):
        return RetentionScore(
            fact_id=fact_id, retention_label="absent",
            retention_score=0.0, evidence="", is_distorted=False,
            confidence=0.0, short_reason="(scorer parse failure)",
            scorer_error="parse_failure",
        )
    label = (obj.get("retention_label") or "absent").strip().lower()
    if label not in _RETENTION_LABEL_TO_SCORE:
        label = "absent"
    return RetentionScore(
        fact_id=fact_id,
        retention_label=label,
        retention_score=_RETENTION_LABEL_TO_SCORE[label],
        evidence=str(obj.get("evidence_in_compressed_text", ""))[:240],
        is_distorted=bool(obj.get("is_distorted", False)),
        confidence=float(obj.get("confidence") or 0.0),
        short_reason=str(obj.get("short_reason", ""))[:240],
    )


def combine_scores(*, exact: bool, llm: Optional[RetentionScore]) -> dict:
    """Spec §14: primary binary retention = exact_retained or label
    in {exact, semantic}; primary continuous retention =
    max(deterministic, llm).
    """
    llm_label = llm.retention_label if llm else "absent"
    llm_score = llm.retention_score if llm else 0.0
    retained_binary = bool(exact) or llm_label in {"exact", "semantic"}
    retention_score = max(1.0 if exact else 0.0, llm_score)
    return {
        "exact_retained": bool(exact),
        "retention_label": llm_label,
        "retention_score": float(retention_score),
        "retained_binary": bool(retained_binary),
    }


__all__ = [
    "render_retention_prompt",
    "deterministic_exact_retained",
    "parse_retention_response",
    "combine_scores",
    "RetentionScore",
]
