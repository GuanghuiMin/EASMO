"""Compressor wrappers around ACON original prompts (spec §9, §10, §13).

Single-round compression:

    compress_once(client, model_name, bundle, task, history, max_chars)

Iterative compression:

    iterate_compression(client, model_name, bundle, task, x0, rounds, max_chars)

The compressor is the ACON original `prompt_history_v2` (UT) or
`improved_history_prompt_samples_4` (UTCO) rendered with:

    task          = condition_task
    prev_summary  = "" (spec §10.4 primary mode)
    history       = current context
    max_chars     = TARGET_MAX_CHARS

The compressor system prompt is ACON's official one-liner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .clients import chat, ChatResult
from .acon_prompt_loader import AconPromptBundle, render_prompt


@dataclass
class CompressionResult:
    compressor_model: str
    prompt_variant: str            # 'UT' or 'UTCO'
    budget_chars: int
    round: int
    input_chars: int
    output_chars: int
    compressed_text: str
    raw_response: str
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    invalid_output: bool = False
    text_sha256: str = ""


def _sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def compress_once(
    *,
    client,
    model_name: str,
    bundle: AconPromptBundle,
    task: str,
    history: str,
    max_chars: int = 1500,
    round_idx: int = 1,
    max_tokens: int = 2048,
    seed: Optional[int] = 42,
) -> CompressionResult:
    """Run one ACON history-compression call. Always uses temperature 0."""
    user_prompt = render_prompt(
        bundle,
        task=task,
        history=history,
        prev_summary="",
        max_chars=max_chars,
    )
    r: ChatResult = chat(
        name=model_name,
        user=user_prompt,
        system=bundle.system_text,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
        client=client,
        json_mode=False,
    )
    text = r.text or ""
    invalid = bool(r.error) or len(text.strip()) == 0
    return CompressionResult(
        compressor_model=model_name,
        prompt_variant=bundle.variant,
        budget_chars=max_chars,
        round=round_idx,
        input_chars=len(history),
        output_chars=len(text),
        compressed_text=text,
        raw_response=r.raw,
        elapsed_s=r.elapsed_s,
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
        error=r.error,
        invalid_output=invalid,
        text_sha256=_sha256(text),
    )


def iterate_compression(
    *,
    client,
    model_name: str,
    bundle: AconPromptBundle,
    task: str,
    x0: str,
    rounds: int = 5,
    max_chars: int = 1500,
    max_tokens: int = 2048,
    seed: Optional[int] = 42,
) -> List[CompressionResult]:
    """Apply C_m repeatedly. Per spec §10.4 the previous compressed
    output is fed back through the ``history`` argument, **not** through
    ``prev_summary``.

    If the model emits an empty or error output, we keep iterating with
    the empty string and mark ``invalid_output=True``.
    """
    results: List[CompressionResult] = []
    current = x0
    for r in range(1, rounds + 1):
        res = compress_once(
            client=client,
            model_name=model_name,
            bundle=bundle,
            task=task,
            history=current,
            max_chars=max_chars,
            round_idx=r,
            max_tokens=max_tokens,
            seed=seed,
        )
        results.append(res)
        # Feed compressed text back as the new history.
        current = res.compressed_text if not res.invalid_output else ""
    return results


__all__ = [
    "CompressionResult",
    "compress_once",
    "iterate_compression",
]
