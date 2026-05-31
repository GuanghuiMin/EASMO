"""ACON compression wrapper for v9.

Returns ``CompressionResult`` with text, length, model metadata.
Single-round compression (greedy / sample) and helpers for
log-probable temperature use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from .clients import chat, ChatResult
from .acon_prompt_loader import AconPromptBundle, render_prompt


def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


@dataclass
class CompressionResult:
    compressor_model: str
    prompt_variant: str
    temperature: float
    seed: Optional[int]
    sample_id: str
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


def compress_once(
    *,
    client,
    model_name: str,
    bundle: AconPromptBundle,
    user_instruction: str,
    history: str,
    max_chars: int = 1500,
    temperature: float = 0.0,
    seed: Optional[int] = 42,
    sample_id: str = "greedy",
    max_tokens: int = 2048,
) -> CompressionResult:
    user = render_prompt(
        bundle,
        task=user_instruction,
        history=history,
        prev_summary="",
        max_chars=max_chars,
    )
    r: ChatResult = chat(
        name=model_name, user=user, system=bundle.system_text,
        temperature=temperature, max_tokens=max_tokens, seed=seed,
        client=client, json_mode=False,
    )
    text = r.text or ""
    invalid = bool(r.error) or len(text.strip()) == 0
    return CompressionResult(
        compressor_model=model_name,
        prompt_variant=bundle.variant,
        temperature=temperature,
        seed=seed,
        sample_id=sample_id,
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


__all__ = [
    "CompressionResult",
    "compress_once",
]
