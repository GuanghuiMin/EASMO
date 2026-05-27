"""LLM probe for decision-state inference (§4) and judge for distance (§6.2).

Both wrap MiniMax-M2.5 with the spec's exact prompts. Output JSON is
parsed defensively (handles <think> blocks, code fences, partial JSON).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .prompts import DECISION_STATE_PROBE, LLM_JUDGE_DISTANCE


_DEFAULT_BASE_URL = "http://10.183.22.68:8005/v1"
_DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.5"
_DEFAULT_SYSTEM = (
    "You are a careful analyst that follows instructions exactly. "
    "Respond ONLY in the requested output format. "
    "Do not include any internal reasoning, analysis, preface, or explanation. "
    "Emit the structured JSON answer directly."
)


def make_client(base_url: str = _DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=240)


# ----------------------------------------------------------------------
# Robust JSON extractor (mirrors v3 evidence._parse_json_object behaviour
# but kept local for v4 self-containment).
# ----------------------------------------------------------------------


def _strip_think(s: str) -> str:
    s = re.sub(r"<think>[\s\S]*?</think>", "", s).strip()
    s = re.sub(r"<think>[\s\S]*$", "", s).strip()
    return s


def _parse_json_object(raw: str) -> Optional[dict]:
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
        # Recover from truncated JSON by trying shorter prefixes.
        for end in range(len(body), 0, -1):
            try:
                return json.loads(body[:end])
            except Exception:
                continue
        return None


def _chat(
    client: OpenAI,
    user_prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system: str = _DEFAULT_SYSTEM,
) -> tuple[str, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    return _strip_think(text), elapsed


# ----------------------------------------------------------------------
# Decision-state probe (§4)
# ----------------------------------------------------------------------


_VALID_FIELDS = (
    "active_subgoal", "completed_actions", "active_constraints",
    "candidate_objects", "avoid_objects", "missing_information",
    "next_action_type", "next_action_arguments", "confidence",
)


def _normalise_decision_state(d: Optional[dict]) -> dict:
    """Coerce LLM output into a stable schema so downstream comparators
    can rely on field types."""
    if not isinstance(d, dict):
        d = {}
    return {
        "active_subgoal": str(d.get("active_subgoal") or "").strip(),
        "completed_actions": [
            {"action": str(x.get("action") or ""),
             "object":  str(x.get("object") or ""),
             "evidence": str(x.get("evidence") or "")}
            for x in (d.get("completed_actions") or []) if isinstance(x, dict)
        ],
        "active_constraints": [
            {"constraint": str(x.get("constraint") or ""),
             "evidence":   str(x.get("evidence") or "")}
            for x in (d.get("active_constraints") or []) if isinstance(x, dict)
        ],
        "candidate_objects": [
            {"object_id": str(x.get("object_id") or ""),
             "object_type": str(x.get("object_type") or ""),
             "reason": str(x.get("reason") or ""),
             "required_action": str(x.get("required_action") or "")}
            for x in (d.get("candidate_objects") or []) if isinstance(x, dict)
        ],
        "avoid_objects": [
            {"object_id": str(x.get("object_id") or ""),
             "object_type": str(x.get("object_type") or ""),
             "reason": str(x.get("reason") or "")}
            for x in (d.get("avoid_objects") or []) if isinstance(x, dict)
        ],
        "missing_information": [
            str(x).strip() for x in (d.get("missing_information") or [])
            if isinstance(x, (str, int, float))
        ],
        "next_action_type": str(d.get("next_action_type") or "").strip(),
        "next_action_arguments": d.get("next_action_arguments") or {},
        "confidence": str(d.get("confidence") or "low").lower().strip(),
    }


@dataclass
class DecisionState:
    state: dict
    raw: str = ""
    elapsed_s: float = 0.0
    parse_ok: bool = True

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "elapsed_s": self.elapsed_s,
            "parse_ok": self.parse_ok,
        }


def probe_decision_state(
    *,
    task_instruction: str,
    context_text: str,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> DecisionState:
    if client is None:
        client = make_client()
    prompt = DECISION_STATE_PROBE.format(
        task_instruction=task_instruction or "(no task instruction)",
        context_text=context_text or "(empty context)",
    )
    raw, elapsed = _chat(client, prompt, model=model, max_tokens=max_tokens)
    obj = _parse_json_object(raw)
    state = _normalise_decision_state(obj)
    return DecisionState(
        state=state,
        raw=raw,
        elapsed_s=elapsed,
        parse_ok=obj is not None,
    )


# ----------------------------------------------------------------------
# LLM-judge distance (§6.2)
# ----------------------------------------------------------------------


_VALID_FIELDS_JUDGE = {
    "next_action_type", "next_action_arguments", "candidate_objects",
    "avoid_objects", "active_constraints", "completed_actions",
    "missing_information", "confidence",
}
_SEVERITY_TO_SCORE = {"none": 0.0, "low": 0.25, "medium": 0.6, "high": 1.0}


@dataclass
class JudgeVerdict:
    meaningful_change: bool
    severity: str        # 'none' | 'low' | 'medium' | 'high'
    score: float         # mapped from severity per spec
    changed_fields: List[str]
    reason: str
    raw: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "meaningful_change": self.meaningful_change,
            "severity": self.severity,
            "score": self.score,
            "changed_fields": self.changed_fields,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
        }


def judge_distance(
    *,
    task_instruction: str,
    reference_state: dict,
    ablated_state: dict,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> JudgeVerdict:
    if client is None:
        client = make_client()
    prompt = LLM_JUDGE_DISTANCE.format(
        task_instruction=task_instruction or "(no task instruction)",
        reference_decision_state_json=json.dumps(reference_state, ensure_ascii=False),
        ablated_decision_state_json=json.dumps(ablated_state, ensure_ascii=False),
    )
    raw, elapsed = _chat(client, prompt, model=model, max_tokens=max_tokens)
    obj = _parse_json_object(raw) or {}
    severity = str(obj.get("severity", "none")).lower().strip()
    if severity not in _SEVERITY_TO_SCORE:
        severity = "none"
    fields = obj.get("changed_fields") or []
    changed_fields = [
        f for f in fields if isinstance(f, str) and f in _VALID_FIELDS_JUDGE
    ]
    return JudgeVerdict(
        meaningful_change=bool(obj.get("meaningful_change", False)),
        severity=severity,
        score=_SEVERITY_TO_SCORE[severity],
        changed_fields=changed_fields,
        reason=str(obj.get("reason", ""))[:300],
        raw=raw,
        elapsed_s=elapsed,
    )
