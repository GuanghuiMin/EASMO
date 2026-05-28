"""Retention scoring (spec §12).

Identical protocol to v7: deterministic substring match first; if not
exact, call the LLM scorer. Cross-model rule:

  compressor = qwen     → scorer = minimax
  compressor = minimax  → scorer = qwen
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .clients import chat, ChatResult, parse_json_object, make_client
from .prompts import RETENTION_SCORER_SYSTEM, render_retention_prompt


_LABEL_TO_SCORE = {
    "exact": 1.0,
    "semantic": 0.75,
    "partial": 0.4,
    "absent": 0.0,
    "contradicted": -0.5,
}


CROSS_EVAL = {
    "qwen": "minimax",
    "minimax": "qwen",
}


@dataclass
class RetentionScore:
    fact_id: str
    retention_label: str
    retention_score: float
    evidence: str
    is_distorted: bool
    confidence: str
    short_reason: str
    scorer_error: Optional[str] = None
    match_type: str = "llm_semantic"  # or "substring_exact"

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "retention_label": self.retention_label,
            "retention_score": self.retention_score,
            "evidence_in_compressed_text": self.evidence,
            "is_distorted": self.is_distorted,
            "confidence": self.confidence,
            "short_reason": self.short_reason,
            "match_type": self.match_type,
            "scorer_error": self.scorer_error,
        }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def deterministic_exact_retained(
    *,
    compressed_text: str,
    source_quote: str = "",
    verbatim_surface: str = "",
    literal_values: Optional[List[str]] = None,
) -> bool:
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


def parse_retention_response(fact_id: str, obj: Optional[dict]) -> RetentionScore:
    if not isinstance(obj, dict):
        return RetentionScore(
            fact_id=fact_id, retention_label="absent", retention_score=0.0,
            evidence="", is_distorted=False, confidence="low",
            short_reason="(scorer parse failure)", scorer_error="parse_failure",
            match_type="llm_semantic",
        )
    label = (obj.get("retention_label") or "absent").strip().lower()
    if label not in _LABEL_TO_SCORE:
        label = "absent"
    return RetentionScore(
        fact_id=fact_id,
        retention_label=label,
        retention_score=_LABEL_TO_SCORE[label],
        evidence=str(obj.get("evidence_in_compressed_text", ""))[:240],
        is_distorted=bool(obj.get("is_distorted", False)),
        confidence=str(obj.get("confidence", "")),
        short_reason=str(obj.get("short_reason", ""))[:240],
        match_type="llm_semantic",
    )


def score_fact_against_text(
    *,
    fact: dict,
    compressed_text: str,
    scorer_name: str,
    client,
    max_tokens: int = 384,
) -> dict:
    """Return the retention dict for one (fact, compressed_text) pair.

    Deterministic exact match first; if exact, skip the LLM call.
    """
    exact = deterministic_exact_retained(
        compressed_text=compressed_text or "",
        source_quote=fact.get("source_quote") or "",
        verbatim_surface=fact.get("verbatim_surface") or "",
        literal_values=fact.get("literal_values") or [],
    )
    if exact:
        score = RetentionScore(
            fact_id=fact["fact_id"],
            retention_label="exact",
            retention_score=1.0,
            evidence=(fact.get("source_quote") or "")[:200],
            is_distorted=False, confidence="high",
            short_reason="deterministic exact substring match",
            match_type="substring_exact",
        )
    else:
        prompt = render_retention_prompt(
            canonical_fact=fact["canonical_fact"],
            fact_type=fact["fact_type"],
            literal_values=fact.get("literal_values") or [],
            compressed_context=compressed_text or "",
        )
        try:
            res = chat(
                name=scorer_name, user=prompt,
                system=RETENTION_SCORER_SYSTEM,
                client=client,
                temperature=0.0, max_tokens=max_tokens,
                json_mode=True, seed=42,
            )
            obj = parse_json_object(res.text)
            score = parse_retention_response(fact["fact_id"], obj)
            if res.error:
                score.scorer_error = res.error
        except Exception as e:
            score = parse_retention_response(fact["fact_id"], None)
            score.scorer_error = str(e)
    return {
        "exact_retained": (score.match_type == "substring_exact"),
        "retention_label": score.retention_label,
        "retention_score": score.retention_score,
        "retained_binary": score.retention_label in {"exact", "semantic"},
        "evidence_in_compressed_text": score.evidence,
        "is_distorted": score.is_distorted,
        "confidence": score.confidence,
        "short_reason": score.short_reason,
        "match_type": score.match_type,
        "scorer_error": score.scorer_error,
    }


def scorer_for(model_name: str) -> str:
    """Return the cross-model scorer for the given compressor model."""
    short = model_name.split("-")[0].lower()
    return CROSS_EVAL.get(short, "minimax")


__all__ = [
    "RetentionScore",
    "deterministic_exact_retained",
    "parse_retention_response",
    "score_fact_against_text",
    "scorer_for",
    "CROSS_EVAL",
]
