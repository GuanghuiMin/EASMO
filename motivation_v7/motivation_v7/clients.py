"""LLM client helpers for motivation_v7.

Two clients:

  * Qwen3-4B-Instruct-2507 local vLLM at ``http://127.0.0.1:8000/v1``
    (model id ``qwen3-4b-instruct-2507``), used as compressor A.
  * MiniMax-M2.5 remote vLLM at ``http://10.183.22.68:8005/v1``
    (model id ``MiniMaxAI/MiniMax-M2.5``), used as compressor B,
    fact extractor, and (cross-model) retention scorer.

Defaults: temperature 0.0, deterministic seed where supported,
``response_format="json_object"`` for structured calls. The MiniMax
chat completions occasionally emit ``<think>…</think>`` blocks even
in JSON mode — those are stripped after the response is read.

Concurrency: ``parallel_chat`` runs many prompts in parallel through
a ``ThreadPoolExecutor``. Per-call timeouts default to 240 s.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI


QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen3-4b-instruct-2507"
MINIMAX_BASE_URL = "http://10.183.22.68:8005/v1"
MINIMAX_MODEL = "MiniMaxAI/MiniMax-M2.5"

_DEFAULT_SYSTEM = (
    "You are a careful analyst that follows instructions exactly. "
    "Respond ONLY in the requested output format. "
    "Do not include any internal reasoning, analysis, preface, or explanation. "
    "Emit the structured answer directly."
)


def make_client(name: str) -> OpenAI:
    """Return an OpenAI-compatible client for ``qwen`` or ``minimax``."""
    if name == "qwen" or name == "qwen3-4b" or name == QWEN_MODEL:
        return OpenAI(base_url=QWEN_BASE_URL, api_key="EMPTY", timeout=240)
    if name == "minimax" or name.startswith("MiniMax") or name == "minimax-m2.5":
        return OpenAI(base_url=MINIMAX_BASE_URL, api_key="EMPTY", timeout=240)
    raise ValueError(f"unknown client name: {name}")


def model_id(name: str) -> str:
    if name in ("qwen", "qwen3-4b", QWEN_MODEL):
        return QWEN_MODEL
    if name in ("minimax", "minimax-m2.5") or name.startswith("MiniMax"):
        return MINIMAX_MODEL
    return name


def _strip_think(s: str) -> str:
    s = re.sub(r"<think>[\s\S]*?</think>", "", s).strip()
    s = re.sub(r"<think>[\s\S]*$", "", s).strip()
    return s


def parse_json_object(raw: str) -> Optional[dict]:
    s = _strip_think(raw or "")
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    body = m.group(0)
    try:
        return json.loads(body)
    except Exception:
        for end in range(len(body), 0, -1):
            try:
                return json.loads(body[:end])
            except Exception:
                continue
        return None


@dataclass
class ChatResult:
    text: str
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    raw: str = ""

    @property
    def is_ok(self) -> bool:
        return self.error is None


def chat(
    *,
    name: str,
    user: str,
    system: str = _DEFAULT_SYSTEM,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = 42,
    json_mode: bool = False,
    client: Optional[OpenAI] = None,
) -> ChatResult:
    """Single chat completion with timing and defensive parsing.

    json_mode adds ``response_format={"type": "json_object"}`` — most
    OpenAI-compatible servers honour it; vLLM v1 supports it on Qwen
    and MiniMax. We additionally strip ``<think>...</think>`` blocks.
    """
    if client is None:
        client = make_client(name)
    extra: Dict[str, Any] = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    if seed is not None:
        extra["seed"] = seed
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_id(name),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        elapsed = time.time() - t0
        raw = resp.choices[0].message.content or ""
        return ChatResult(
            text=_strip_think(raw),
            elapsed_s=elapsed,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
            raw=raw,
        )
    except Exception as e:
        return ChatResult(
            text="", elapsed_s=time.time() - t0, error=str(e),
        )


def parallel_chat(
    *,
    name: str,
    prompts: List[Tuple[Any, str]],   # (key, user_prompt)
    system: str = _DEFAULT_SYSTEM,
    max_workers: int = 8,
    on_result: Optional[Callable[[Any, ChatResult], None]] = None,
    **chat_kwargs,
) -> Dict[Any, ChatResult]:
    """Run many chat() calls in a thread pool. Returns {key: ChatResult}.

    Each call uses a fresh client to avoid cross-thread contention on
    the OpenAI client's internal HTTP session.
    """
    out: Dict[Any, ChatResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                chat,
                name=name, user=user, system=system,
                client=make_client(name),
                **chat_kwargs,
            ): key
            for key, user in prompts
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = ChatResult(text="", elapsed_s=0.0, error=str(e))
            out[key] = res
            if on_result is not None:
                on_result(key, res)
    return out


__all__ = [
    "QWEN_BASE_URL", "QWEN_MODEL",
    "MINIMAX_BASE_URL", "MINIMAX_MODEL",
    "make_client", "model_id",
    "chat", "parallel_chat",
    "ChatResult",
    "parse_json_object",
]
