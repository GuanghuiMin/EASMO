"""Thin MiniMax-M2.5 client wrapper.

Wraps the OpenAI-compatible vLLM endpoint described in
``guidances/minimax-m25-api-guide.md``. Two design goals:

* **Thread-safe parallel generation** via a bounded thread pool so that
  M1 (which makes N samples × K candidates × P probe states calls per
  context) finishes in reasonable wall time.
* **Reasoning-quirk hygiene** — MiniMax returns chain-of-thought inside
  ``<think>...</think>`` blocks in ``content``; we strip them on the
  way out so downstream parsing (action extraction, etc.) sees only
  the user-facing text. The full content is also exposed via
  ``GenerateResult.raw_content`` if needed.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

import httpx

from .utils import setup_logging

_logger = setup_logging("motivation.llm")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_think(content: str) -> tuple[str, str]:
    """Return ``(visible, thinking)``."""
    if not content:
        return "", ""
    thinks = "\n\n".join(m.group(1).strip() for m in _THINK_RE.finditer(content))
    visible = _THINK_RE.sub("", content).strip()
    return visible, thinks


@dataclass
class GenerateResult:
    text: str
    raw_content: str
    thinking: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_s: float
    finish_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MinimaxClient:
    """OpenAI-compatible vLLM client for MiniMax-M2.5."""

    base_url: str = "http://10.183.22.68:8005/v1"
    model_id: str = "MiniMaxAI/MiniMax-M2.5"
    api_key: str = "EMPTY"
    request_timeout_s: float = 180.0
    max_concurrent: int = 8
    retry_attempts: int = 3
    retry_backoff_s: float = 5.0

    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self):
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=self.request_timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------
    # Single completion
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: Sequence[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        seed: int | None = None,
        stop: Optional[Sequence[str]] = None,
        extra_body: Optional[dict] = None,
    ) -> GenerateResult:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if stop:
            payload["stop"] = list(stop)
        if extra_body:
            payload.update(extra_body)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.retry_attempts + 1):
            t0 = time.time()
            try:
                r = self._client.post("/chat/completions", json=payload)
                r.raise_for_status()
                body = r.json()
                elapsed = time.time() - t0
                msg = body["choices"][0]["message"]
                raw = msg.get("content") or ""
                visible, thinking = split_think(raw)
                usage = body.get("usage") or {}
                return GenerateResult(
                    text=visible,
                    raw_content=raw,
                    thinking=thinking,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                    elapsed_s=elapsed,
                    finish_reason=body["choices"][0].get("finish_reason"),
                )
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_err = exc
                _logger.warning(
                    "MiniMax call failed (attempt %d/%d): %s",
                    attempt, self.retry_attempts, exc,
                )
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff_s * attempt)
        # All retries exhausted.
        return GenerateResult(
            text="",
            raw_content="",
            thinking="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            elapsed_s=0.0,
            error=str(last_err) if last_err else "unknown",
        )

    # ------------------------------------------------------------------
    # Convenience: prompt-only, returns just the visible text
    # ------------------------------------------------------------------

    def chat(
        self,
        system: str | None,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> GenerateResult:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.generate(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed,
        )

    # ------------------------------------------------------------------
    # Parallel batch
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batches: Sequence[dict],
    ) -> List[GenerateResult]:
        """Run a list of ``generate`` calls concurrently.

        ``batches`` is a list of dicts; each dict is forwarded as kwargs
        to ``generate``. Results are returned in input order.
        """
        results: list[Optional[GenerateResult]] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(self.generate, **b): i for i, b in enumerate(batches)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True iff the /models endpoint answers with 200."""
        try:
            r = self._client.get("/models", timeout=8.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
