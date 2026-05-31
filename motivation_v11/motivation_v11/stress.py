"""Repeated-compression stress chains (spec §2 Claim 2, §7 stage 03).

T^0(c) = c
T^{r+1}(c) = ACON_compress(history=T^r(c), task=user_instruction)

Convergence:
  text_similarity(x_r, x_{r-1}) >= 0.95
  AND |len_r - len_{r-1}| / max(len_{r-1}, 1) <= 0.02
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from typing import List, Optional

from .clients import chat, ChatResult
from .acon_prompt_loader import AconPromptBundle, render_prompt


def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


@dataclass
class StressRound:
    candidate_id: str
    case_id: str
    compressor_model: str
    round: int
    context_text: str
    chars: int
    tokens_est: int
    text_hash: str
    elapsed_s: float = 0.0
    error: Optional[str] = None


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def token_jaccard(a: str, b: str) -> float:
    aw = set(a.lower().split()); bw = set(b.lower().split())
    if not aw and not bw:
        return 1.0
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def stress_chain(
    *,
    candidate_id: str,
    case_id: str,
    client,
    model_name: str,
    bundle: AconPromptBundle,
    user_instruction: str,
    c0: str,
    rounds: int,
    max_chars: int = 1500,
    max_tokens: int = 2048,
    seed: Optional[int] = 42,
) -> List[StressRound]:
    """Return list of StressRound for r = 0..rounds.

    Round 0 is c0 (candidate output); subsequent rounds are recompressions.
    If a compression errors or is empty, we keep iterating with the previous
    non-empty context to avoid silent terminations.
    """
    rows: List[StressRound] = []
    rows.append(StressRound(
        candidate_id=candidate_id,
        case_id=case_id,
        compressor_model=model_name,
        round=0,
        context_text=c0,
        chars=len(c0),
        tokens_est=max(1, len(c0) // 4),
        text_hash=_sha256(c0),
    ))
    current = c0
    for r in range(1, rounds + 1):
        user = render_prompt(bundle, task=user_instruction, history=current,
                             prev_summary="", max_chars=max_chars)
        res: ChatResult = chat(
            name=model_name, user=user, system=bundle.system_text,
            temperature=0.0, max_tokens=max_tokens, seed=seed,
            client=client, json_mode=False,
        )
        text = res.text or ""
        if not text.strip():
            # If compression collapses, keep iterating with the previous
            # text instead of empty (matches spec: don't silently stop).
            text = current
        rows.append(StressRound(
            candidate_id=candidate_id,
            case_id=case_id,
            compressor_model=model_name,
            round=r,
            context_text=text,
            chars=len(text),
            tokens_est=max(1, len(text) // 4),
            text_hash=_sha256(text),
            elapsed_s=res.elapsed_s,
            error=res.error,
        ))
        current = text
    return rows


def chain_convergence(rounds: List[StressRound]) -> dict:
    """Return convergence summary {converged_binary, convergence_round, …}."""
    converged_round: Optional[int] = None
    last_sim = 0.0; last_dlen = 1.0; last_tj = 0.0
    for i in range(1, len(rounds)):
        a, b = rounds[i].context_text, rounds[i-1].context_text
        sim = text_similarity(a, b)
        dlen = abs(len(a) - len(b)) / max(len(b), 1)
        tj = token_jaccard(a, b)
        last_sim, last_dlen, last_tj = sim, dlen, tj
        if sim >= 0.95 and dlen <= 0.02:
            converged_round = rounds[i].round
            break
    return {
        "converged_binary": converged_round is not None,
        "convergence_round": converged_round if converged_round is not None else -1,
        "text_similarity_last": last_sim,
        "length_change_last": last_dlen,
        "token_jaccard_last": last_tj,
    }


__all__ = [
    "StressRound",
    "text_similarity",
    "token_jaccard",
    "stress_chain",
    "chain_convergence",
]
