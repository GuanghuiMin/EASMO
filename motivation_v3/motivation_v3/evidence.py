"""Behavioral evidence labelling (Exp 2.1) and audit (Exp 2.2).

Both are LLM-driven post-hoc labelling steps over the symbolic units
extracted in Exp 1.C. Labels are returned as JSON objects per the spec
schemas; we parse defensively (LLM JSON output is sometimes wrapped in
prose / code fences).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .compressors import _chat, make_client, _DEFAULT_MODEL
from .data import Trajectory, trajectory_step_window
from .prompts import (
    AUDIT_COVERAGE,
    LABEL_BEHAVIORAL_USEFULNESS,
    LABEL_RECOVERY_CALL,
)


_USEFUL_USED_AS = {
    "api_argument", "filter_condition", "entity_disambiguation",
    "constraint_check", "state_change", "final_answer",
    "planning_state", "not_used",
}
_USEFUL_CONFIDENCE = {"high", "medium", "low"}
_AUDIT_LABELS = {
    "preserved",
    "dropped_identifier",
    "dropped_binding",
    "dropped_constraint",
    "dropped_action_outcome",
    "vague_or_wrong_abstraction",
    "distorted_or_hallucinated",
}


# ----------------------------------------------------------------------
# Robust JSON parsing
# ----------------------------------------------------------------------


def _parse_json_object(raw: str) -> Optional[dict]:
    s = raw.strip()
    # Strip <think>...</think> blocks (MiniMax-M2.5 reasoning mode).
    s = re.sub(r"<think>[\s\S]*?</think>", "", s).strip()
    # Drop any orphaned <think> tag that didn't close.
    s = re.sub(r"<think>[\s\S]*$", "", s).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # Find the LAST balanced {...} object — for completions where the
    # model wrote prose then the JSON, this picks the JSON.
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # Try to recover by finding the largest valid JSON suffix.
        chunk = m.group(0)
        for end in range(len(chunk), 0, -1):
            try:
                return json.loads(chunk[:end])
            except Exception:
                continue
        return None


# ----------------------------------------------------------------------
# Exp 2.1 — per-unit behavioral usefulness label
# ----------------------------------------------------------------------


@dataclass
class UsefulnessLabel:
    unit_id: str
    useful: bool
    confidence: str
    used_as: str
    reason: str
    raw: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def label_unit_usefulness(
    traj: Trajectory,
    unit: Dict[str, Any],
    *,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
    future_window: int = 8,
) -> UsefulnessLabel:
    if client is None:
        client = make_client()
    after_step = unit.get("source_step") or 0
    future = trajectory_step_window(traj, after_step=after_step, n=future_window)
    prompt = LABEL_BEHAVIORAL_USEFULNESS.format(
        task_instruction=traj.instruction or "(no task instruction)",
        unit_json=json.dumps({
            "unit_type": unit.get("unit_type", "other"),
            "text": unit.get("text", ""),
            "source_step": unit.get("source_step"),
        }),
        future_steps_text=future or "(no future steps)",
    )
    # Bumped from 300 → 2048: MiniMax-M2.5 uses <think>...</think> blocks
    # that we strip post-hoc, so we need headroom for thinking + JSON.
    raw, _ = _chat(client, prompt, model=model, max_tokens=2048)
    obj = _parse_json_object(raw) or {}
    useful = bool(obj.get("useful", False))
    confidence = str(obj.get("confidence", "low")).lower().strip()
    if confidence not in _USEFUL_CONFIDENCE:
        confidence = "low"
    used_as = str(obj.get("used_as", "not_used")).lower().strip()
    if used_as not in _USEFUL_USED_AS:
        used_as = "not_used"
    reason = str(obj.get("reason", ""))[:240]
    return UsefulnessLabel(
        unit_id=unit.get("unit_id", ""),
        useful=useful,
        confidence=confidence,
        used_as=used_as,
        reason=reason,
        raw=raw,
    )


# ----------------------------------------------------------------------
# Exp 2.2 — audit a compressed context against behavioral evidence
# ----------------------------------------------------------------------


@dataclass
class AuditResult:
    task_id: str
    method: str
    unit_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    raw: str = ""

    def num_preserved(self) -> int:
        return sum(1 for r in self.unit_results if r.get("label") == "preserved")

    def num_total(self) -> int:
        return len(self.unit_results)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "method": self.method,
            "unit_results": self.unit_results,
            "summary": self.summary,
        }


def audit_compressed(
    task_id: str,
    method: str,
    *,
    task_instruction: str,
    compressed_context: str,
    behavioral_evidence_units: List[Dict[str, Any]],
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
) -> AuditResult:
    if client is None:
        client = make_client()
    units_repr = json.dumps(
        [{"unit_type": u.get("unit_type", "other"),
          "text": u.get("text", "")} for u in behavioral_evidence_units],
        ensure_ascii=False,
    )
    prompt = AUDIT_COVERAGE.format(
        task_instruction=task_instruction or "(no task instruction)",
        compressed_context=(compressed_context or "(empty)")[:8000],
        behavioral_evidence_units=units_repr[:8000],
    )
    raw, _ = _chat(client, prompt, model=model, max_tokens=4096)
    obj = _parse_json_object(raw) or {}
    raw_units = obj.get("unit_results", [])
    unit_results: List[Dict[str, Any]] = []
    for u in raw_units if isinstance(raw_units, list) else []:
        if not isinstance(u, dict):
            continue
        label = str(u.get("label", "vague_or_wrong_abstraction")).lower().strip()
        if label not in _AUDIT_LABELS:
            label = "vague_or_wrong_abstraction"
        unit_results.append({
            "unit_text": str(u.get("unit_text", ""))[:240],
            "label": label,
            "matched_span": str(u.get("matched_span", ""))[:240],
            "reason": str(u.get("reason", ""))[:240],
        })
    summary = obj.get("summary", {}) if isinstance(obj.get("summary"), dict) else {}
    return AuditResult(
        task_id=task_id,
        method=method,
        unit_results=unit_results,
        summary=summary,
        raw=raw,
    )


# ----------------------------------------------------------------------
# Exp 3 — recovery API call labelling
# ----------------------------------------------------------------------


@dataclass
class RecoveryLabel:
    api_call: str
    recovery_call: bool
    confidence: str
    reason: str
    raw: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def label_recovery_call(
    *,
    compressed_context: str,
    behavioral_evidence_units: List[Dict[str, Any]],
    api_call: str,
    api_response: str,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
) -> RecoveryLabel:
    if client is None:
        client = make_client()
    units_repr = json.dumps(
        [{"unit_type": u.get("unit_type", "other"),
          "text": u.get("text", "")} for u in behavioral_evidence_units],
        ensure_ascii=False,
    )[:6000]
    prompt = LABEL_RECOVERY_CALL.format(
        compressed_context=(compressed_context or "(empty)")[:6000],
        behavioral_evidence_units=units_repr,
        api_call=(api_call or "")[:600],
        api_response=(api_response or "")[:1200],
    )
    # Bumped 200 → 1024 for the same <think>-budget reason as the
    # usefulness labelling above.
    raw, _ = _chat(client, prompt, model=model, max_tokens=1024)
    obj = _parse_json_object(raw) or {}
    return RecoveryLabel(
        api_call=api_call[:240],
        recovery_call=bool(obj.get("recovery_call", False)),
        confidence=str(obj.get("confidence", "low")).lower().strip(),
        reason=str(obj.get("reason", ""))[:240],
        raw=raw,
    )
