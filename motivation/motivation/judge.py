"""LLM-as-judge for QA task success.

Given a downstream question Q, a gold answer A*, and a candidate answer
A, ask MiniMax to judge whether A is semantically equivalent to A*. We
keep the prompt strict (one of three responses, low temperature) so the
judge is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .llm import MinimaxClient
from .utils import setup_logging

_logger = setup_logging("motivation.judge")


JUDGE_SYSTEM = (
    "You are a strict factual-equivalence judge. Decide whether the "
    "candidate answer is correct given the gold answer to the question. "
    "Respond with exactly one of: YES / NO / PARTIAL.\n\n"
    "Rules:\n"
    "- YES: the candidate clearly conveys the same factual content as gold.\n"
    "- NO: the candidate is wrong, contradicts gold, or refuses.\n"
    "- PARTIAL: the candidate is partially correct or omits something material.\n"
    "Be strict; paraphrasing is fine, hallucinations are not."
)


@dataclass
class JudgeResult:
    candidate: str
    label: str       # YES / NO / PARTIAL / UNKNOWN
    raw: str


def _judge_one(client: MinimaxClient, question: str, gold: str, candidate: str) -> JudgeResult:
    user = (
        f"Question: {question}\n\n"
        f"Gold answer:\n\"\"\"\n{gold}\n\"\"\"\n\n"
        f"Candidate answer:\n\"\"\"\n{candidate}\n\"\"\"\n\n"
        "Your verdict (YES / NO / PARTIAL):"
    )
    res = client.chat(system=JUDGE_SYSTEM, user=user, temperature=0.0, max_tokens=8)
    raw = (res.text or "").strip().upper()
    if "YES" in raw[:8]:
        label = "YES"
    elif "PARTIAL" in raw[:12]:
        label = "PARTIAL"
    elif "NO" in raw[:8]:
        label = "NO"
    else:
        label = "UNKNOWN"
    return JudgeResult(candidate=candidate, label=label, raw=raw)


def judge_batch(
    client: MinimaxClient,
    question: str,
    gold: str,
    candidates: List[str],
    *,
    partial_credit: float = 0.5,
) -> tuple[float, List[JudgeResult]]:
    """Score each candidate via the judge, return (mean_correctness, per-candidate)."""
    if not candidates:
        return 0.0, []
    # Build batch of judge calls
    batches = []
    for cand in candidates:
        user = (
            f"Question: {question}\n\n"
            f"Gold answer:\n\"\"\"\n{gold}\n\"\"\"\n\n"
            f"Candidate answer:\n\"\"\"\n{cand}\n\"\"\"\n\n"
            "Your verdict (YES / NO / PARTIAL):"
        )
        batches.append({
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        })
    results = client.generate_batch(batches)
    out: list[JudgeResult] = []
    score_sum = 0.0
    for cand, res in zip(candidates, results):
        raw = (res.text or "").strip().upper()
        if "YES" in raw[:8]:
            label, s = "YES", 1.0
        elif "PARTIAL" in raw[:12]:
            label, s = "PARTIAL", partial_credit
        elif "NO" in raw[:8]:
            label, s = "NO", 0.0
        else:
            label, s = "UNKNOWN", 0.0
        out.append(JudgeResult(candidate=cand, label=label, raw=res.text or ""))
        score_sum += s
    return score_sum / len(candidates), out
