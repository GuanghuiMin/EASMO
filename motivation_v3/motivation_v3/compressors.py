"""Compression methods (Exp 1).

All three call MiniMax-M2.5 with the corresponding prompt from
``prompts.py``. Outputs are dataclasses that carry both the rendered
text (used in Exp 3 as the downstream context) and the raw LLM
response (saved for reproducibility).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .data import Trajectory, render_trajectory
from .prompts import (
    COMPRESS_ACON_STYLE,
    COMPRESS_SYMBOLIC_EVIDENCE,
    COMPRESS_TASK_AWARE_SUMMARY,
)


_DEFAULT_BASE_URL = "http://10.183.22.68:8005/v1"
_DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.5"


def make_client(base_url: str = _DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=180)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _count_tokens(text: str) -> int:
    """Approximate token count using tiktoken cl100k_base when
    available; falls back to chars/4."""
    if not text:
        return 0
    try:  # pragma: no cover
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ----------------------------------------------------------------------
# Compressed-context dataclass
# ----------------------------------------------------------------------


@dataclass
class Compressed:
    method: str                 # 'task_aware_summary' | 'acon_style_summary' | 'symbolic_evidence'
    task_id: str
    text: str                   # rendered text the downstream agent sees
    n_tokens: int               # tiktoken count of ``text``
    n_units: int                # 1 for summaries; len(units) for symbolic
    units: Optional[List[Dict[str, Any]]] = None  # only for symbolic
    raw_response: str = ""
    elapsed_s: float = 0.0
    model: str = _DEFAULT_MODEL

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "task_id": self.task_id,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "n_units": self.n_units,
            "units": self.units,
            "raw_response": self.raw_response,
            "elapsed_s": self.elapsed_s,
            "model": self.model,
        }


# ----------------------------------------------------------------------
# Internal: one chat completion
# ----------------------------------------------------------------------


_DEFAULT_SYSTEM = (
    "You are a careful compressor that follows instructions exactly. "
    "Respond ONLY in the requested output format. "
    "Do not include any internal reasoning, analysis, preface, or explanation. "
    "Emit the structured answer directly. "
    "Brevity matters: do not pad."
)


def _chat(
    client: OpenAI,
    prompt_user: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    system: str = _DEFAULT_SYSTEM,
) -> tuple[str, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    return _strip_think(text), elapsed


# ----------------------------------------------------------------------
# Public: build one compressed context per method
# ----------------------------------------------------------------------


def build_task_aware_summary(
    traj: Trajectory,
    *,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
) -> Compressed:
    if client is None:
        client = make_client()
    prompt = COMPRESS_TASK_AWARE_SUMMARY.format(
        task_instruction=traj.instruction or "(no task instruction)",
        trajectory_text=render_trajectory(traj),
    )
    text, elapsed = _chat(client, prompt, model=model)
    return Compressed(
        method="task_aware_summary",
        task_id=traj.task_id,
        text=text.strip(),
        n_tokens=_count_tokens(text),
        n_units=1,
        units=None,
        raw_response=text,
        elapsed_s=elapsed,
        model=model,
    )


def build_acon_style_summary(
    traj: Trajectory,
    *,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
) -> Compressed:
    if client is None:
        client = make_client()
    prompt = COMPRESS_ACON_STYLE.format(
        task_instruction=traj.instruction or "(no task instruction)",
        trajectory_text=render_trajectory(traj),
    )
    text, elapsed = _chat(client, prompt, model=model)
    return Compressed(
        method="acon_style_summary",
        task_id=traj.task_id,
        text=text.strip(),
        n_tokens=_count_tokens(text),
        n_units=1,
        units=None,
        raw_response=text,
        elapsed_s=elapsed,
        model=model,
    )


# ----------------------------------------------------------------------
# Symbolic evidence
# ----------------------------------------------------------------------


_VALID_UNIT_TYPES = {
    "object_id", "entity_binding", "variable_value", "action_outcome",
    "constraint", "unresolved_subgoal", "api_argument", "date_time",
    "amount_quantity", "status", "error_fix", "other",
}


_UNIT_BLOCK_RE = re.compile(
    r'\{\s*"unit_type"\s*:[\s\S]*?(?<=})(?=\s*[,\]\}])',
)
_UNIT_TYPE_RE = re.compile(r'"unit_type"\s*:\s*"([^"\n]+)"')
# Match either "text": "..." (well-formed) or text="..." (Python-style).
_UNIT_TEXT_RE = re.compile(r'(?:"text"\s*:|text\s*=)\s*"([^"\n]{1,400})"')
_UNIT_STEP_RE = re.compile(r'"source_step"\s*:\s*(\d+)')
_UNIT_QUOTE_RE = re.compile(r'"supporting_quote"\s*:\s*"([^"\n]{0,300})"')


def _try_parse_units_array(s: str) -> Optional[List[Dict[str, Any]]]:
    """Try strict JSON parse first."""
    try:
        d = json.loads(s)
    except Exception:
        return None
    if isinstance(d, dict) and isinstance(d.get("units"), list):
        return d["units"]
    if isinstance(d, list):
        return d
    return None


def _parse_symbolic_units(raw: str) -> List[Dict[str, Any]]:
    """Robust extraction of symbolic-evidence units from an LLM response.

    Handles common LLM JSON-emission mistakes:
      * <think>...</think> reasoning blocks (already stripped before we
        get here, but be defensive).
      * leading prose, code fences, trailing commas, truncated arrays.
      * Python-style ``text="..."`` instead of ``"text":"..."``.
      * Unescaped quotes inside string fields (e.g. supporting_quote with
        embedded JSON-like fragments).

    Strategy: try strict JSON parse first; if that fails, fall back to
    per-unit regex extraction so we recover as many well-formed units
    as possible from a partially-broken response.
    """
    s = (raw or "").strip()
    s = re.sub(r"<think>[\s\S]*?</think>", "", s).strip()
    s = re.sub(r"<think>[\s\S]*$", "", s).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    m = re.search(r"\{[\s\S]*\}", s)
    body = m.group(0) if m else s

    # Strict parse path
    arr = _try_parse_units_array(body)
    if arr is None:
        # Try shorter prefixes to recover from truncated JSON.
        for end in range(len(body), 0, -1):
            arr = _try_parse_units_array(body[:end])
            if arr is not None:
                break

    out: List[Dict[str, Any]] = []
    if arr is not None:
        for i, u in enumerate(arr):
            if not isinstance(u, dict):
                continue
            unit_type = (u.get("unit_type") or "other").lower().strip()
            if unit_type not in _VALID_UNIT_TYPES:
                unit_type = "other"
            text = (u.get("text") or "").strip()
            if not text:
                continue
            out.append({
                "unit_id": f"u{i:04d}",
                "unit_type": unit_type,
                "text": text,
                "source_step": u.get("source_step"),
                "supporting_quote": (u.get("supporting_quote") or "").strip(),
            })
        if out:
            return out

    # Regex-per-unit fallback: extract each {... unit ...} block and pull
    # required fields with permissive regexes that accept malformed JSON.
    unit_starts = [m.start() for m in re.finditer(r'\{\s*"unit_type"', body)]
    if not unit_starts:
        return out
    unit_starts.append(len(body))
    for i, start in enumerate(unit_starts[:-1]):
        end = unit_starts[i + 1]
        block = body[start:end]
        ut = _UNIT_TYPE_RE.search(block)
        tx = _UNIT_TEXT_RE.search(block)
        if not ut or not tx:
            continue
        unit_type = (ut.group(1) or "other").lower().strip()
        if unit_type not in _VALID_UNIT_TYPES:
            unit_type = "other"
        text = (tx.group(1) or "").strip()
        if not text:
            continue
        step_m = _UNIT_STEP_RE.search(block)
        quote_m = _UNIT_QUOTE_RE.search(block)
        out.append({
            "unit_id": f"u{i:04d}",
            "unit_type": unit_type,
            "text": text,
            "source_step": int(step_m.group(1)) if step_m else None,
            "supporting_quote": (quote_m.group(1) if quote_m else "").strip(),
        })
    return out


def render_symbolic_block(units: List[Dict[str, Any]]) -> str:
    """Render the unit list as the spec's [SYMBOLIC_CONTEXT] block."""
    lines = ["[SYMBOLIC_CONTEXT]"]
    for u in units:
        lines.append(f"- ({u['unit_type']}) {u['text']}")
    lines.append("[/SYMBOLIC_CONTEXT]")
    return "\n".join(lines)


def build_symbolic_evidence(
    traj: Trajectory,
    *,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
) -> Compressed:
    if client is None:
        client = make_client()
    prompt = COMPRESS_SYMBOLIC_EVIDENCE.format(
        task_instruction=traj.instruction or "(no task instruction)",
        trajectory_text=render_trajectory(traj),
    )
    raw, elapsed = _chat(client, prompt, model=model)
    units = _parse_symbolic_units(raw)
    rendered = render_symbolic_block(units)
    return Compressed(
        method="symbolic_evidence",
        task_id=traj.task_id,
        text=rendered,
        n_tokens=_count_tokens(rendered),
        n_units=len(units),
        units=units,
        raw_response=raw,
        elapsed_s=elapsed,
        model=model,
    )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


COMPRESSOR_REGISTRY = {
    "task_aware_summary":  build_task_aware_summary,
    "acon_style_summary":  build_acon_style_summary,
    "symbolic_evidence":   build_symbolic_evidence,
}
