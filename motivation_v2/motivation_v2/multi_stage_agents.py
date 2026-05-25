"""Planner and Verifier agents for the multi-stage role-specialised
AppWorld pipeline.

Each is a single-LLM-call agent (no env interaction): planner takes
a task instruction and emits a numbered plan; verifier takes a task
instruction + trajectory tail + claimed answer and emits a JSON
verdict. The Executor agent is acon's standard ``AppWorldAgent``
with the planner's plan injected into its prompt — see
``runner.py::run_with_compressed_memory`` (we reuse the same prompt-
splice machinery, calling the plan a "PRE-LOADED PLAN" turn).

See ``../docs/04_multi_stage_role_setup.md`` for the full design.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

from .data import Trajectory


_DEFAULT_BASE_URL = "http://10.183.22.68:8005/v1"
_DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.5"


def _client(base_url: str = _DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=120)


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------


PLANNER_SYSTEM = (
    "You are a PLANNER agent. Your sole job is to read a task description "
    "and emit a concise, executable plan as a numbered list of sub-goals. "
    "You do NOT execute anything. You do NOT call any APIs. Output only the "
    "plan itself, nothing else — no preface, no explanations beyond what is "
    "needed to specify each sub-goal."
)


PLANNER_USER_TEMPLATE = """Task: {task}

Produce a plan with 2–5 numbered sub-goals. Each sub-goal must be a concrete
action that can be described in one sentence. Format:

1. <sub-goal one>
2. <sub-goal two>
...

Do not call any APIs. Do not include any text outside the numbered list."""


@dataclass
class PlannerOutput:
    task_id: str
    instruction: str
    plan_text: str            # the raw response (numbered list)
    sub_goals: List[str]      # parsed list of sub-goal strings
    raw_response: str         # full LLM response
    elapsed_s: float
    model: str

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "plan_text": self.plan_text,
            "sub_goals": list(self.sub_goals),
            "n_sub_goals": len(self.sub_goals),
            "elapsed_s": self.elapsed_s,
            "model": self.model,
        }


def _parse_plan(response: str) -> tuple[str, List[str]]:
    """Return (plan_text_kept, [sub_goals])."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    sub_goals: List[str] = []
    kept_lines: List[str] = []
    pattern = re.compile(r"^\s*(\d+)[\.)]\s*(.+?)\s*$")
    for line in response.splitlines():
        m = pattern.match(line)
        if m:
            sub_goals.append(m.group(2).strip())
            kept_lines.append(line.strip())
    if not sub_goals:
        # Fallback: each non-empty line is a goal.
        for line in response.splitlines():
            line = line.strip()
            if line and len(line) > 8:
                sub_goals.append(line)
                kept_lines.append(line)
    return "\n".join(kept_lines), sub_goals


def run_planner(
    task_id: str,
    instruction: str,
    *,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> PlannerOutput:
    if client is None:
        client = _client()
    user_prompt = PLANNER_USER_TEMPLATE.format(task=instruction)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    raw = resp.choices[0].message.content or ""
    plan_text, sub_goals = _parse_plan(raw)
    return PlannerOutput(
        task_id=task_id,
        instruction=instruction,
        plan_text=plan_text,
        sub_goals=sub_goals,
        raw_response=raw,
        elapsed_s=elapsed,
        model=model,
    )


# ----------------------------------------------------------------------
# Verifier
# ----------------------------------------------------------------------


VERIFIER_SYSTEM = (
    "You are a VERIFIER agent. Your job is to inspect another agent's claimed "
    "answer to a task and judge whether it is correct, given the task statement "
    "and the trajectory tail. You do NOT execute anything. You do NOT call APIs. "
    "Output ONLY a single JSON object with keys: \"verdict\" (\"pass\" | \"fail\" | "
    "\"uncertain\"), \"evidence\" (a list of short factual statements that "
    "support your verdict), and \"confidence\" (a float in [0, 1]). Do not "
    "include any text outside the JSON object."
)


VERIFIER_USER_TEMPLATE = """Task: {task}
Claimed answer: {answer}

Trajectory tail (last {n_steps} steps):
{trajectory_tail}

Issue your JSON verdict now."""


@dataclass
class VerifierOutput:
    task_id: str
    instruction: str
    claimed_answer: str
    verdict: str               # 'pass' | 'fail' | 'uncertain' | 'parse_error'
    evidence: List[str]        # the verifier's evidence list
    confidence: float
    raw_response: str
    elapsed_s: float
    model: str

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "claimed_answer": self.claimed_answer,
            "verdict": self.verdict,
            "evidence": list(self.evidence),
            "n_evidence": len(self.evidence),
            "confidence": self.confidence,
            "elapsed_s": self.elapsed_s,
            "model": self.model,
        }


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_verifier(response: str) -> dict:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = _JSON_BLOCK_RE.search(response)
    if not m:
        return {"verdict": "parse_error", "evidence": [], "confidence": 0.0,
                "_raw": response}
    block = m.group(0)
    try:
        d = json.loads(block)
    except json.JSONDecodeError:
        # try truncating trailing junk
        try:
            d = json.loads(block.rstrip("}").rstrip(",") + "}")
        except json.JSONDecodeError:
            return {"verdict": "parse_error", "evidence": [], "confidence": 0.0,
                    "_raw": response}
    return {
        "verdict": str(d.get("verdict", "uncertain")).lower(),
        "evidence": list(d.get("evidence", [])),
        "confidence": float(d.get("confidence", 0.5)),
    }


def _render_trajectory_tail(traj: Trajectory, n_steps: int = 8,
                             max_chars_per_step: int = 400) -> str:
    parts = []
    for s in traj.steps[-n_steps:]:
        action = (s.action or "").strip()[:max_chars_per_step]
        output = (s.output or "").strip()[:max_chars_per_step]
        parts.append(
            f"### step {s.step}\naction: {action}\noutput: {output}"
        )
    return "\n\n".join(parts)


def _extract_claimed_answer(traj: Trajectory) -> str:
    """Fish out the apis.supervisor.complete_task(answer=...) value from
    the agent's last few actions."""
    pattern = re.compile(r"complete_task\s*\(.*?answer\s*=\s*(.*?)\)", re.DOTALL)
    for s in reversed(traj.steps):
        m = pattern.search(s.action or "")
        if m:
            return m.group(1).strip()
    return "(no answer found)"


def run_verifier(
    traj: Trajectory,
    *,
    n_tail_steps: int = 8,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> VerifierOutput:
    if client is None:
        client = _client()
    claimed = _extract_claimed_answer(traj)
    tail = _render_trajectory_tail(traj, n_steps=n_tail_steps)
    user_prompt = VERIFIER_USER_TEMPLATE.format(
        task=traj.instruction or "(no task instruction)",
        answer=claimed,
        n_steps=n_tail_steps,
        trajectory_tail=tail,
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    raw = resp.choices[0].message.content or ""
    parsed = _parse_verifier(raw)
    return VerifierOutput(
        task_id=traj.task_id,
        instruction=traj.instruction,
        claimed_answer=claimed,
        verdict=parsed["verdict"],
        evidence=parsed["evidence"],
        confidence=float(parsed["confidence"]),
        raw_response=raw,
        elapsed_s=elapsed,
        model=model,
    )
