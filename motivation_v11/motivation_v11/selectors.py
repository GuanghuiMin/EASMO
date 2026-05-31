"""Verbal selectors for v11 (spec §10): pointwise verifier, pairwise
verifier tournament, continuation-entropy selector.

All three are NEGATIVE BASELINES — they exist to motivate "verbal
selectors are not behavior reward" (§19 Criterion 4 / Claim 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .clients import chat, parse_json_object, MINIMAX_MODEL


# ----------------------------------------------------------------------
# §10.7 Pointwise verifier
# ----------------------------------------------------------------------

POINTWISE_SYSTEM = (
    "You are a strict evaluator of compressed context for a tool-use agent.\n"
    "Return only JSON. Do not include prose outside JSON."
)


POINTWISE_USER_TEMPLATE = """\
You will judge whether a compressed context is sufficient for a downstream AppWorld tool-use agent to continue the task.

Original task:
{task_instruction}

Compressed context:
{compressed_context}

Evaluate the compressed context for downstream task success.
Do not assume facts not present in the compressed context.
Do not reward verbosity. A shorter context is better if it preserves the necessary information.

Return JSON with:
{{
  "sufficiency_score": 0.0 to 1.0,
  "risk_score": 0.0 to 1.0,
  "missing_critical_information": ["..."],
  "likely_to_succeed": true or false,
  "one_sentence_reason": "..."
}}
"""


@dataclass
class PointwiseScore:
    candidate_id: str
    eval_round: str
    verifier_model: str
    sufficiency_score: float
    risk_score: float
    missing_critical_information: List[str]
    likely_to_succeed: bool
    one_sentence_reason: str
    raw_response: str = ""
    error: Optional[str] = None

    def selector_score(self, length_chars: int,
                        risk_weight: float = 0.25,
                        length_weight: float = 0.02) -> float:
        """Spec §10.7 ranking:
            selector_score = sufficiency_score
                             - risk_weight * risk_score
                             - length_weight * length_kchars
        """
        return (self.sufficiency_score
                - risk_weight * self.risk_score
                - length_weight * (length_chars / 1000.0))


def pointwise_score(*, candidate_id: str, eval_round: str,
                     user_instruction: str, compressed_context: str,
                     client=None, max_tokens: int = 1536) -> PointwiseScore:
    user = POINTWISE_USER_TEMPLATE.format(
        task_instruction=user_instruction,
        compressed_context=compressed_context,
    )
    res = chat(name="minimax", user=user, system=POINTWISE_SYSTEM,
                temperature=0.0, max_tokens=max_tokens, seed=42,
                client=client, json_mode=True)
    obj = parse_json_object(res.text) or {}

    def _f(k, default=0.0):
        try: return float(obj.get(k, default))
        except Exception: return float(default)

    return PointwiseScore(
        candidate_id=candidate_id, eval_round=eval_round,
        verifier_model=MINIMAX_MODEL,
        sufficiency_score=_f("sufficiency_score"),
        risk_score=_f("risk_score"),
        missing_critical_information=list(obj.get("missing_critical_information", []))[:8],
        likely_to_succeed=bool(obj.get("likely_to_succeed", False)),
        one_sentence_reason=str(obj.get("one_sentence_reason", ""))[:400],
        raw_response=res.raw,
        error=res.error,
    )


# ----------------------------------------------------------------------
# §10.8 Pairwise verifier tournament
# ----------------------------------------------------------------------

PAIRWISE_SYSTEM = (
    "You are a strict pairwise evaluator of compressed contexts for a tool-use agent.\n"
    "Return only JSON."
)

PAIRWISE_USER_TEMPLATE = """\
A downstream AppWorld tool-use agent will continue the task using one of two compressed contexts.

Original task:
{task_instruction}

Compressed context A:
{context_a}

Compressed context B:
{context_b}

Choose the context that is more likely to let the downstream agent complete the task.
Prefer shorter context only when both seem equally sufficient.
Do not assume facts not present in the context.

Return JSON:
{{
  "winner": "A" or "B" or "tie",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence"
}}
"""


@dataclass
class PairwiseMatch:
    case_id: str
    prompt_family: str
    eval_round: str
    candidate_a_id: str
    candidate_b_id: str
    verifier_model: str
    winner: str
    confidence: float
    reason: str
    raw_response: str = ""
    error: Optional[str] = None


def pairwise_match(*, case_id: str, prompt_family: str, eval_round: str,
                    candidate_a_id: str, candidate_b_id: str,
                    user_instruction: str, context_a: str, context_b: str,
                    client=None, max_tokens: int = 1024) -> PairwiseMatch:
    user = PAIRWISE_USER_TEMPLATE.format(
        task_instruction=user_instruction,
        context_a=context_a, context_b=context_b,
    )
    res = chat(name="minimax", user=user, system=PAIRWISE_SYSTEM,
                temperature=0.0, max_tokens=max_tokens, seed=42,
                client=client, json_mode=True)
    obj = parse_json_object(res.text) or {}
    raw_winner = str(obj.get("winner", "")).strip().upper()
    if raw_winner not in ("A", "B", "TIE"):
        raw_winner = "TIE"
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return PairwiseMatch(
        case_id=case_id, prompt_family=prompt_family, eval_round=eval_round,
        candidate_a_id=candidate_a_id, candidate_b_id=candidate_b_id,
        verifier_model=MINIMAX_MODEL,
        winner=raw_winner if raw_winner != "TIE" else "tie",
        confidence=conf,
        reason=str(obj.get("reason", ""))[:400],
        raw_response=res.raw, error=res.error,
    )


# ----------------------------------------------------------------------
# §10.9 Continuation-entropy selector
# ----------------------------------------------------------------------

ENTROPY_SYSTEM = (
    "You are diagnosing whether a compressed context gives a tool-use agent a clear next-step state.\n"
    "Return only JSON."
)

ENTROPY_USER_TEMPLATE = """\
Given the original task and compressed context, infer what the downstream agent should do next.

Original task:
{task_instruction}

Compressed context:
{compressed_context}

Return JSON:
{{
  "next_action_type": "...",
  "required_arguments": {{"arg": "value or null"}},
  "missing_information": ["..."],
  "confidence": "high | medium | low"
}}
"""


@dataclass
class EntropySample:
    candidate_id: str
    eval_round: str
    sample_index: int          # 0..M-1
    next_action_type: str
    required_arguments_keys: List[str]
    missing_information: List[str]
    confidence: str
    raw_response: str = ""
    error: Optional[str] = None


def entropy_sample(*, candidate_id: str, eval_round: str, sample_index: int,
                    user_instruction: str, compressed_context: str,
                    client=None, seed: int = 1000, max_tokens: int = 1024
                    ) -> EntropySample:
    user = ENTROPY_USER_TEMPLATE.format(
        task_instruction=user_instruction,
        compressed_context=compressed_context,
    )
    res = chat(name="minimax", user=user, system=ENTROPY_SYSTEM,
                temperature=0.7, max_tokens=max_tokens, seed=seed,
                client=client, json_mode=True)
    obj = parse_json_object(res.text) or {}
    args = obj.get("required_arguments", {}) or {}
    return EntropySample(
        candidate_id=candidate_id, eval_round=eval_round,
        sample_index=sample_index,
        next_action_type=str(obj.get("next_action_type", "UNKNOWN"))[:80],
        required_arguments_keys=sorted([str(k) for k in args.keys()])[:16],
        missing_information=list(obj.get("missing_information", []))[:8],
        confidence=str(obj.get("confidence", "low")).strip().lower(),
        raw_response=res.raw, error=res.error,
    )


def entropy_features(samples: List[EntropySample]) -> dict:
    """Aggregate M entropy samples into selector features (spec §10.9)."""
    import math, collections
    if not samples:
        return {"next_action_type_entropy": 0.0,
                "argument_key_jaccard_distance": 0.0,
                "missing_info_count_variance": 0.0,
                "confidence_entropy": 0.0}

    # next_action_type entropy
    counts = collections.Counter(s.next_action_type for s in samples)
    total = sum(counts.values())
    H_action = -sum((c/total) * math.log(c/total) for c in counts.values() if c > 0)

    # argument_key jaccard distance (pairwise average; 1 - mean jaccard)
    key_sets = [set(s.required_arguments_keys) for s in samples]
    if len(key_sets) > 1:
        pairs = 0; total_j = 0.0
        for i in range(len(key_sets)):
            for j in range(i+1, len(key_sets)):
                a, b = key_sets[i], key_sets[j]
                if a or b:
                    j_sim = len(a & b) / len(a | b)
                else:
                    j_sim = 1.0
                total_j += (1.0 - j_sim); pairs += 1
        J_dist = total_j / pairs if pairs > 0 else 0.0
    else:
        J_dist = 0.0

    # missing info count variance
    miss_counts = [len(s.missing_information) for s in samples]
    mean_mc = sum(miss_counts) / max(len(miss_counts), 1)
    V_miss = sum((x - mean_mc) ** 2 for x in miss_counts) / max(len(miss_counts), 1)

    # confidence entropy
    conf_counts = collections.Counter(s.confidence for s in samples)
    H_conf = -sum((c/total) * math.log(c/total) for c in conf_counts.values() if c > 0)

    return {
        "next_action_type_entropy":      H_action,
        "argument_key_jaccard_distance": J_dist,
        "missing_info_count_variance":   V_miss,
        "confidence_entropy":            H_conf,
    }


def entropy_selector_score(features: dict, length_chars: int,
                            jaccard_weight: float = 1.0,
                            missvar_weight: float = 0.25,
                            length_weight: float = 0.02) -> float:
    """Spec §10.9 ranking:
        score = -H_action - jaccard_dist - 0.25*missvar - H_conf - 0.02*length_kchars
    """
    return -(
        features["next_action_type_entropy"]
        + jaccard_weight * features["argument_key_jaccard_distance"]
        + missvar_weight * features["missing_info_count_variance"]
        + features["confidence_entropy"]
        + length_weight * (length_chars / 1000.0)
    )


__all__ = [
    "POINTWISE_SYSTEM", "POINTWISE_USER_TEMPLATE",
    "PointwiseScore", "pointwise_score",
    "PAIRWISE_SYSTEM", "PAIRWISE_USER_TEMPLATE",
    "PairwiseMatch", "pairwise_match",
    "ENTROPY_SYSTEM", "ENTROPY_USER_TEMPLATE",
    "EntropySample", "entropy_sample", "entropy_features",
    "entropy_selector_score",
]
