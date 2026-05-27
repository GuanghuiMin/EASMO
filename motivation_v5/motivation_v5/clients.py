"""OpenAI-compatible clients for Qwen3-4B (local) and MiniMax-M2.5 (shared).

Both wrap the openai SDK with the right base_url + model id + default
generation params. Used by every audit / verifier / recompressor /
augmenter call in motivation_v5.

Notes:
* Qwen3-4B's vLLM server has `max_model_len=8192`, so prompts must be
  pre-truncated by the caller. We add a helper `pack_prompt` that
  budgets sections (baseline/acon/augmented/recompressed) under a
  given total token target.
* MiniMax-M2.5 uses ``<think>...</think>`` reasoning blocks; we strip
  them post-hoc the same way v3 / v4 do.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen3-4b"
QWEN_MAX_MODEL_LEN = 8192

MINIMAX_BASE_URL = "http://10.183.22.68:8005/v1"
MINIMAX_MODEL = "MiniMaxAI/MiniMax-M2.5"


def qwen_client() -> OpenAI:
    return OpenAI(base_url=QWEN_BASE_URL, api_key="EMPTY", timeout=240)


def minimax_client() -> OpenAI:
    return OpenAI(base_url=MINIMAX_BASE_URL, api_key="EMPTY", timeout=240)


def _strip_think(text: str) -> str:
    s = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    s = re.sub(r"<think>[\s\S]*$", "", s).strip()
    return s


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class ChatResult:
    text: str            # _strip_think'd
    raw: str             # raw model output
    elapsed_s: float
    model: str
    finish_reason: str = ""


# ----------------------------------------------------------------------
# Public chat fns
# ----------------------------------------------------------------------


def chat_qwen(
    user_prompt: str,
    *,
    system: str = "You are a careful auditor. Respond ONLY with the requested JSON. No prose, no preface, no explanation outside the JSON object.",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    json_mode: bool = True,
    client: Optional[OpenAI] = None,
) -> ChatResult:
    if client is None:
        client = qwen_client()
    t0 = time.time()
    kwargs = dict(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        # vLLM honours `extra_body.chat_template_kwargs` to disable Qwen's
        # default thinking mode (much cleaner JSON).
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    if json_mode:
        # Many vLLM builds accept this for JSON structured output; if the
        # backend rejects it, we fall back silently.
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs,
            )
        except Exception:
            resp = client.chat.completions.create(**kwargs)
    else:
        resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    return ChatResult(
        text=_strip_think(text),
        raw=text,
        elapsed_s=elapsed,
        model=QWEN_MODEL,
        finish_reason=str(resp.choices[0].finish_reason or ""),
    )


def chat_minimax(
    user_prompt: str,
    *,
    system: str = "You are a careful auditor. Respond ONLY with the requested JSON. No prose, no preface, no explanation outside the JSON object.",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    json_mode: bool = False,
    client: Optional[OpenAI] = None,
) -> ChatResult:
    if client is None:
        client = minimax_client()
    t0 = time.time()
    kwargs = dict(
        model=MINIMAX_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    return ChatResult(
        text=_strip_think(text),
        raw=text,
        elapsed_s=elapsed,
        model=MINIMAX_MODEL,
        finish_reason=str(resp.choices[0].finish_reason or ""),
    )


# ----------------------------------------------------------------------
# Token-aware prompt packing
# ----------------------------------------------------------------------


def truncate_to_tokens(text: str, max_tokens: int, *, suffix: str = "\n…[truncated for context budget]") -> str:
    """Greedy char-level truncation to fit `max_tokens`. Adds suffix to mark."""
    if max_tokens <= 0 or not text:
        return ""
    if _approx_tokens(text) <= max_tokens:
        return text
    # rough binary search on chars
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _approx_tokens(text[:mid]) + _approx_tokens(suffix) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_]\w*)\s*\}\}")


def render_template(template: str, **fields) -> str:
    """Jinja-style ``{{ name }}`` substitution that ignores literal
    ``{`` / ``}`` characters in the template (so JSON schemas inside
    the template don't break parsing)."""
    def _sub(m):
        key = m.group(1)
        return str(fields.get(key, m.group(0)))
    return _TEMPLATE_PLACEHOLDER_RE.sub(_sub, template)


def pack_prompt_for_qwen(
    template: str,
    *,
    fields: dict,
    reserve_output_tokens: int = 2048,
    overhead_tokens: int = 400,
) -> str:
    """Render `template` with `fields`, but pre-truncate long fields so
    the full prompt fits in Qwen's `QWEN_MAX_MODEL_LEN - reserve_output`.

    Placeholders are jinja-style ``{{ name }}``; literal `{` and `}` in
    the template (e.g. JSON schemas) pass through unchanged.

    Truncation policy: long fields share the remaining budget
    proportionally to their original size; short fields (< 200 tokens)
    are left intact.
    """
    fixed_budget = QWEN_MAX_MODEL_LEN - reserve_output_tokens - overhead_tokens
    # Render template with empty placeholders first to estimate fixed text size
    skeleton = render_template(template, **{k: "" for k in fields})
    skeleton_tokens = _approx_tokens(skeleton)
    body_budget = max(fixed_budget - skeleton_tokens, 512)

    long_keys = [k for k, v in fields.items()
                 if _approx_tokens(str(v)) > 200]
    short_keys = [k for k in fields if k not in long_keys]
    short_total = sum(_approx_tokens(str(fields[k])) for k in short_keys)
    long_budget = max(body_budget - short_total, 256)

    long_sizes = {k: _approx_tokens(str(fields[k])) for k in long_keys}
    long_total = sum(long_sizes.values()) or 1
    packed = {}
    for k, v in fields.items():
        if k in long_keys:
            share = max(int(long_budget * long_sizes[k] / long_total), 200)
            packed[k] = truncate_to_tokens(str(v), share)
        else:
            packed[k] = str(v)
    return render_template(template, **packed)


# ----------------------------------------------------------------------
# Robust JSON extraction (shared)
# ----------------------------------------------------------------------


def parse_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of an LLM response; tolerate <think> blocks,
    code fences, prose preface/suffix, and truncated trailing chars."""
    s = _strip_think(text or "")
    s = re.sub(r"```(?:json)?\s*", "", s).strip()
    s = re.sub(r"\s*```\s*$", "", s).strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    body = m.group(0)
    try:
        return json.loads(body)
    except Exception:
        # try shorter prefixes (truncation recovery)
        for end in range(len(body), 0, -1):
            try:
                return json.loads(body[:end])
            except Exception:
                continue
        return None


__all__ = [
    "qwen_client", "minimax_client",
    "chat_qwen", "chat_minimax",
    "ChatResult",
    "truncate_to_tokens", "pack_prompt_for_qwen",
    "parse_json",
    "QWEN_BASE_URL", "QWEN_MODEL", "QWEN_MAX_MODEL_LEN",
    "MINIMAX_BASE_URL", "MINIMAX_MODEL",
]
