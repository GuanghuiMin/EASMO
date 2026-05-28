"""General LLM compressor wrapper (spec §9).

Single-round and iterative variants. The actual prompt content lives
in ``prompts.py``. This module just wires (prompt × context × task)
into a chat call and packages the result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

from .clients import chat, ChatResult
from .prompts import PromptBundle


def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


@dataclass
class CompressionResult:
    model: str
    prompt_family: str
    budget_chars: int
    round: int
    input_chars: int
    output_chars: int
    compressed_context: str
    raw_response: str
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    invalid_output: bool = False
    text_sha256: str = ""
    budget_violation: bool = False


def compress_once(
    *,
    client,
    model_name: str,
    bundle: PromptBundle,
    context: str,
    condition_task: Optional[str] = None,
    max_chars: int = 1500,
    round_idx: int = 1,
    max_tokens: int = 2048,
    seed: Optional[int] = 42,
    budget_tolerance: float = 0.10,
) -> CompressionResult:
    user = bundle.render(
        context=context,
        max_chars=max_chars,
        condition_task=condition_task,
    )
    r: ChatResult = chat(
        name=model_name, user=user, system=bundle.system,
        temperature=0.0, max_tokens=max_tokens, seed=seed,
        client=client, json_mode=False,
    )
    text = r.text or ""
    invalid = bool(r.error) or len(text.strip()) == 0
    violation = len(text) > int(max_chars * (1.0 + budget_tolerance))
    return CompressionResult(
        model=model_name,
        prompt_family=bundle.family,
        budget_chars=max_chars,
        round=round_idx,
        input_chars=len(context),
        output_chars=len(text),
        compressed_context=text,
        raw_response=r.raw,
        elapsed_s=r.elapsed_s,
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
        error=r.error,
        invalid_output=invalid,
        text_sha256=_sha256(text),
        budget_violation=violation,
    )


def iterate_compression(
    *,
    client,
    model_name: str,
    bundle: PromptBundle,
    x0: str,
    condition_task: Optional[str] = None,
    rounds: int = 6,
    max_chars: int = 1500,
    max_tokens: int = 2048,
    seed: Optional[int] = 42,
    budget_tolerance: float = 0.10,
) -> List[CompressionResult]:
    """Apply C_{m,p,B} repeatedly; feed previous compressed text as the
    new ``context``. Empty/invalid outputs propagate as the new context
    so the chain still completes ``rounds`` rounds (marked invalid)."""
    results: List[CompressionResult] = []
    current = x0
    for r in range(1, rounds + 1):
        res = compress_once(
            client=client, model_name=model_name, bundle=bundle,
            context=current, condition_task=condition_task,
            max_chars=max_chars, round_idx=r,
            max_tokens=max_tokens, seed=seed,
            budget_tolerance=budget_tolerance,
        )
        results.append(res)
        current = res.compressed_context if not res.invalid_output else ""
    return results


__all__ = [
    "CompressionResult",
    "compress_once",
    "iterate_compression",
]
