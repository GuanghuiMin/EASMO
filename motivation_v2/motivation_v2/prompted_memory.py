"""Prompted-LLM-compressor baselines for T2 evaluation.

For each role in {tool, code, plan, verify} we have a prompt that
asks an off-the-shelf LLM (MiniMax-M2.5 by default) to produce a
budget-bounded compressed memory **conditioned on the role
description and the task instruction**. The prompt explicitly
describes what kind of agent the memory is for and what kind of
information it should keep.

T2's empirical claim: the LLM, despite the role conditioning,
produces memories that are **surface-uniform across roles**
(cross-role Jaccard between `m_prompted_tool` and `m_prompted_plan`
should be HIGH, much higher than the projected oracle's
0.04 baseline) AND **fail to match the role-projected oracle**
(per-role Jaccard between `m_prompted_role_X` and `m_role_X`
should be LOW).

Implementation note: this module performs LLM calls. Use the EASMO
.venv (has openai 2.37.0). The MiniMax-M2.5 endpoint is at
http://10.183.22.68:8005/v1 (read from
/workspace/acon/configs/private_config.yaml).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from .data import Trajectory
from .exec_memory import ExecMemory, MemoryUnit, _compose, _count_tokens


# ----------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://10.183.22.68:8005/v1"
_DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.5"


def _client(base_url: str = _DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=120)


# ----------------------------------------------------------------------
# Role prompts
# ----------------------------------------------------------------------

_PROMPT_TEMPLATES: Dict[str, str] = {
    "tool": """You are selecting compressed memory for a TOOL-USE agent.

Tool-use agents specialise in calling APIs, observing the results,
and chaining API calls together. They need to remember:
- which API endpoints have been called and with what arguments,
- the data those endpoints returned,
- the IDs and tokens that link calls together.

The agent is working on the following task:
{task}

The upstream agent's full execution trace is below. Select the
most useful compressed memory under approximately {budget} tokens.
Output ONLY the selected memory facts, one per line, no prose, no
numbering, no preface. Each line is a self-contained fact about an
API call or its result.

Trace:
{context}
""",
    "code": """You are selecting compressed memory for a CODING agent.

Coding agents specialise in writing and refactoring Python code.
They need to remember:
- common code patterns (loops, list comprehensions, error handling),
- API/library calling conventions,
- reusable code snippets and idioms — NOT specific data values.

The agent is working on the following task:
{task}

The upstream agent's full execution trace is below. Select the
most useful compressed memory under approximately {budget} tokens.
Output ONLY the selected memory items, one per line, no prose, no
numbering, no preface. Focus on code patterns and conventions, not
specific argument values.

Trace:
{context}
""",
    "plan": """You are selecting compressed memory for a PLANNING agent.

Planning agents specialise in setting goals, decomposing tasks
into sub-goals, sequencing actions, and tracking milestones. They
need to remember:
- the high-level task goal,
- sub-goal decompositions,
- major milestones achieved (e.g. discovered a relevant app),
- the structure of the final answer.

The agent is working on the following task:
{task}

The upstream agent's full execution trace is below. Select the
most useful compressed memory under approximately {budget} tokens.
Output ONLY the selected memory items, one per line, no prose, no
numbering, no preface. Focus on goals, plans, and milestones.

Trace:
{context}
""",
    "verify": """You are selecting compressed memory for a VERIFICATION agent.

Verification agents specialise in checking that a final answer is
correct. They need to remember:
- final-state observations,
- assertions and post-conditions,
- the specific values that justify the answer being correct.

The agent is working on the following task:
{task}

The upstream agent's full execution trace is below. Select the
most useful compressed memory under approximately {budget} tokens.
Output ONLY the selected memory items, one per line, no prose, no
numbering, no preface. Focus on facts that confirm correctness.

Trace:
{context}
""",
}

PROMPTED_ROLES: List[str] = list(_PROMPT_TEMPLATES.keys())


# ----------------------------------------------------------------------
# Trace rendering
# ----------------------------------------------------------------------


def render_trajectory_for_compressor(
    traj: Trajectory,
    *,
    max_chars_per_step: int = 600,
    max_total_chars: int = 12000,
) -> str:
    """Render the trajectory as plain text the LLM compressor reads.

    We cap each step's action+output to keep individual steps from
    dominating, and cap the whole rendering at ``max_total_chars``
    so we don't blow past MiniMax's context limit on long
    trajectories. Long observations get truncated with an explicit
    marker so the LLM knows the cut happened.
    """
    parts: List[str] = []
    used = 0
    for s in traj.steps:
        action = (s.action or "").strip()
        output = (s.output or "").strip()
        if len(output) > max_chars_per_step:
            output = output[:max_chars_per_step] + "…[truncated]"
        if len(action) > max_chars_per_step:
            action = action[:max_chars_per_step] + "…[truncated]"
        block = (
            f"### step {s.step}\n"
            f"action:\n{action}\n\n"
            f"output:\n{output}\n"
        )
        if used + len(block) > max_total_chars:
            parts.append("…[trajectory truncated]")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Build a prompted memory for one (trajectory, role, budget)
# ----------------------------------------------------------------------


def _parse_response(text: str, budget_tokens: int) -> List[MemoryUnit]:
    """Split the LLM response into one MemoryUnit per non-empty line,
    skipping obvious preface/numbering. Truncate to budget."""
    units: List[MemoryUnit] = []
    used = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip common LLM preface lines.
        low = line.lower()
        if low.startswith(("here ", "below ", "the most", "the following",
                            "selected memory", "compressed memory", "answer:",
                            "memory:", "facts:", "output:")):
            continue
        # Drop list-numbering and bullet markers.
        line = re.sub(r"^\s*(?:\d+[\.)]\s*|[-*+]\s*)", "", line).strip()
        if not line:
            continue
        n = _count_tokens(line)
        if used + n > budget_tokens:
            continue
        units.append(MemoryUnit(
            kind="prompted_unit",
            app="unknown",
            text=line,
            weight=1.0,
        ))
        used += n
    return units


def build_prompted_memory(
    traj: Trajectory,
    role: str,
    budget_tokens: int,
    *,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    max_response_tokens: int = 4096,
) -> ExecMemory:
    """Call the LLM with the role-specific prompt and parse the response."""
    if role not in _PROMPT_TEMPLATES:
        raise ValueError(f"Unknown role {role!r}; expected one of {PROMPTED_ROLES}")
    if client is None:
        client = _client()

    prompt_user = _PROMPT_TEMPLATES[role].format(
        task=traj.instruction or "(no task instruction provided)",
        budget=budget_tokens,
        context=render_trajectory_for_compressor(traj),
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a careful compressor that follows instructions exactly."},
            {"role": "user", "content": prompt_user},
        ],
        temperature=temperature,
        max_tokens=max_response_tokens,
    )
    elapsed = time.time() - t0
    response_text = resp.choices[0].message.content or ""
    # Strip <think> blocks if model emits them
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    units = _parse_response(response_text, budget_tokens)
    text, n = _compose(units)
    em = ExecMemory(
        variant=f"prompted:{role}",
        task_id=traj.task_id,
        budget_tokens=budget_tokens,
        units=units,
        text=text,
        n_tokens=n,
        n_units=len(units),
        n_units_dropped=0,  # we don't track full pool here; LLM is the selector
        executor=model,
    )
    em.extra_elapsed_s = elapsed  # type: ignore[attr-defined]
    em.extra_raw_response = response_text  # type: ignore[attr-defined]
    return em
