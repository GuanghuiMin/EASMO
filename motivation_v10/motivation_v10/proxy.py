"""Proxy reward scorers (spec §11).

v10 implements two MiniMax-based proxy scorers:

  * continuation_verifier — pointwise JSON rubric (§11.1)
  * pairwise_preference   — A vs B preference under CK (§11.2)

A third proxy (`future_action_nll_proxy`, §11.3) is implemented in
`nll_proxy.py` and uses Qwen3-4B logprobs as an *auxiliary* signal
only (spec §4 forbids Qwen as final verifier).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .clients import chat, parse_json_object, MINIMAX_MODEL


# ---------------------------------------------------------------------
# §11.1 Continuation verifier (pointwise)
# ---------------------------------------------------------------------

VERIFIER_SYSTEM = (
    "You are a strict pre-flight verifier for an autonomous tool-use "
    "agent. Given a compressed context summary and the original user "
    "task, predict whether the unchanged downstream agent will be able "
    "to complete the task using only that compressed context plus its "
    "own tool calls. Be conservative: any missing exact identifier, "
    "credential, schema parameter, or action outcome is grounds to "
    "lower the predicted success probability. Return only JSON."
)


VERIFIER_USER_TEMPLATE = """\
Original user task:
{user_instruction}

Compressed context the downstream agent will see:
{compressed_text}

Rubric (independent 0-1 ratings; 1 = perfect, 0 = unusable):

* predicted_success_probability: probability the agent finishes the task
  with at most {max_steps} additional tool-use steps using only this
  compressed context and its standard APIs.
* missing_information_risk: how likely a piece of *necessary* information
  (token, ID, prior action outcome, app schema) is missing or vague.
  HIGH means risk is high.
* execution_specificity: how specific and ready-to-execute the
  compressed text is for API calls. HIGH means specific.
* risk_of_repeating_completed_actions: how likely the agent will redo a
  state-changing action that already succeeded (e.g. re-sending a
  payment) because the compression lost the completion signal. HIGH
  means risk is high.
* risk_of_wrong_api_arguments: how likely the agent will pass wrong or
  hallucinated arguments because the compressed text gives partial /
  ambiguous bindings. HIGH means risk is high.

Return JSON ONLY:
{{
  "predicted_success_probability": 0.0,
  "missing_information_risk": 0.0,
  "execution_specificity": 0.0,
  "risk_of_repeating_completed_actions": 0.0,
  "risk_of_wrong_api_arguments": 0.0,
  "short_reason": "..."
}}
"""


@dataclass
class VerifierScore:
    candidate_id: str
    eval_round: str        # "C1" | "CK"
    verifier_model: str
    predicted_success_probability: float
    missing_information_risk: float
    execution_specificity: float
    risk_of_repeating_completed_actions: float
    risk_of_wrong_api_arguments: float
    short_reason: str
    raw_response: str = ""
    error: Optional[str] = None

    def composite(self, w_miss: float = 0.5, w_spec: float = 0.3,
                  w_repeat: float = 0.1, w_wrong: float = 0.1) -> float:
        """Return a single ranking score for this candidate.

        Default weights emphasise the per-rubric predicted probability,
        then specificity and missing-info penalties.
        """
        return (
            self.predicted_success_probability
            - w_miss   * self.missing_information_risk
            + w_spec   * self.execution_specificity
            - w_repeat * self.risk_of_repeating_completed_actions
            - w_wrong  * self.risk_of_wrong_api_arguments
        )


def verifier_score(
    *,
    candidate_id: str,
    eval_round: str,
    user_instruction: str,
    compressed_text: str,
    max_steps: int = 15,
    client=None,
    max_tokens: int = 2048,
) -> VerifierScore:
    user = VERIFIER_USER_TEMPLATE.format(
        user_instruction=user_instruction,
        compressed_text=compressed_text,
        max_steps=max_steps,
    )
    res = chat(
        name="minimax", user=user, system=VERIFIER_SYSTEM,
        temperature=0.0, max_tokens=max_tokens, seed=42,
        client=client, json_mode=True,
    )
    obj = parse_json_object(res.text) or {}

    def _f(k, default=0.0):
        v = obj.get(k, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    return VerifierScore(
        candidate_id=candidate_id,
        eval_round=eval_round,
        verifier_model=MINIMAX_MODEL,
        predicted_success_probability=_f("predicted_success_probability"),
        missing_information_risk=_f("missing_information_risk"),
        execution_specificity=_f("execution_specificity"),
        risk_of_repeating_completed_actions=_f("risk_of_repeating_completed_actions"),
        risk_of_wrong_api_arguments=_f("risk_of_wrong_api_arguments"),
        short_reason=str(obj.get("short_reason", ""))[:400],
        raw_response=res.raw,
        error=res.error,
    )


# ---------------------------------------------------------------------
# §11.2 Pairwise preference
# ---------------------------------------------------------------------

PAIRWISE_SYSTEM = (
    "You are a strict pre-flight verifier for an autonomous tool-use "
    "agent. Choose which compressed context (A or B) is more likely to "
    "let the unchanged downstream agent finish the user's task within "
    "the step budget. Tie is allowed only if A and B are functionally "
    "indistinguishable. Return only JSON."
)


PAIRWISE_USER_TEMPLATE = """\
Original user task:
{user_instruction}

Step budget: {max_steps} additional actions.

Compressed context A:
\"\"\"
{text_a}
\"\"\"

Compressed context B:
\"\"\"
{text_b}
\"\"\"

Which is more likely to let the unchanged agent complete the task?

Return JSON ONLY:
{{
  "winner": "A",
  "confidence": 0.0,
  "reason": "..."
}}
"""


@dataclass
class PairwisePreference:
    case_id: str
    eval_round: str
    candidate_a_id: str
    candidate_b_id: str
    verifier_model: str
    winner: str      # "A" | "B" | "tie"
    confidence: float
    reason: str
    raw_response: str = ""
    error: Optional[str] = None


def pairwise_preference(
    *,
    case_id: str,
    eval_round: str,
    candidate_a_id: str,
    candidate_b_id: str,
    user_instruction: str,
    text_a: str,
    text_b: str,
    max_steps: int = 15,
    client=None,
    max_tokens: int = 1536,
) -> PairwisePreference:
    user = PAIRWISE_USER_TEMPLATE.format(
        user_instruction=user_instruction,
        max_steps=max_steps,
        text_a=text_a,
        text_b=text_b,
    )
    res = chat(
        name="minimax", user=user, system=PAIRWISE_SYSTEM,
        temperature=0.0, max_tokens=max_tokens, seed=42,
        client=client, json_mode=True,
    )
    obj = parse_json_object(res.text) or {}
    raw_winner = str(obj.get("winner", "")).strip().upper()
    if raw_winner not in ("A", "B", "TIE"):
        raw_winner = "TIE"
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return PairwisePreference(
        case_id=case_id,
        eval_round=eval_round,
        candidate_a_id=candidate_a_id,
        candidate_b_id=candidate_b_id,
        verifier_model=MINIMAX_MODEL,
        winner=raw_winner if raw_winner != "TIE" else "tie",
        confidence=conf,
        reason=str(obj.get("reason", ""))[:400],
        raw_response=res.raw,
        error=res.error,
    )


__all__ = [
    "VERIFIER_SYSTEM", "VERIFIER_USER_TEMPLATE",
    "PAIRWISE_SYSTEM", "PAIRWISE_USER_TEMPLATE",
    "VerifierScore", "verifier_score",
    "PairwisePreference", "pairwise_preference",
]
