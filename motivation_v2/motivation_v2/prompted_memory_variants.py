"""Prompted-LLM-compressor ablation variants (T2, Exp D §8.3 of spec).

Spec requires 4-5 prompted conditions to ablate which axis matters:

  1. prompted_generic      no role, no task → "compress generally"
  2. prompted_task         with task,    no role
  3. prompted_role         no task,      with role
  4. prompted_task_role    with both                ← what existing
                                                     prompted_memory.py
                                                     produces
  5. prompted_extractive   ask LLM to select unit IDs (skipped this round
                           per spec "if possible")

The 4-condition factorial design isolates:
  * (task on/off)  effect when role is held fixed
  * (role on/off)  effect when task is held fixed

Hypothesis: under T2, none of the 4 prompted variants will recover the
role-projected oracle's content — adding role info should NOT produce
the orthogonality the projection achieves. If a prompted variant does
recover oracle structure, T2 weakens.

Implementation parallels prompted_memory.py; we re-use ``ExecMemory``,
``MemoryUnit``, and the response parser.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from openai import OpenAI

from .data import Trajectory
from .exec_memory import ExecMemory, MemoryUnit, _compose
from .prompted_memory import (
    _DEFAULT_BASE_URL,
    _DEFAULT_MODEL,
    _client,
    _parse_response,
    render_trajectory_for_compressor,
)


# ----------------------------------------------------------------------
# Prompt templates per (condition, role)
# ----------------------------------------------------------------------

_ROLE_DESCRIPTIONS: Dict[str, str] = {
    "tool": (
        "Tool-use agents specialise in calling APIs, observing the results, "
        "and chaining API calls together. They need to remember:\n"
        "- which API endpoints have been called and with what arguments,\n"
        "- the data those endpoints returned,\n"
        "- the IDs and tokens that link calls together."
    ),
    "code": (
        "Coding agents specialise in writing and refactoring Python code. "
        "They need to remember:\n"
        "- common code patterns (loops, list comprehensions, error handling),\n"
        "- API/library calling conventions,\n"
        "- reusable code snippets and idioms — NOT specific data values."
    ),
    "plan": (
        "Planning agents specialise in setting goals, decomposing tasks "
        "into sub-goals, sequencing actions, and tracking milestones. "
        "They need to remember:\n"
        "- the high-level task goal,\n"
        "- sub-goal decompositions,\n"
        "- major milestones achieved,\n"
        "- the structure of the final answer."
    ),
    "verify": (
        "Verification agents specialise in checking that a final answer "
        "is correct. They need to remember:\n"
        "- final-state observations,\n"
        "- assertions and post-conditions,\n"
        "- the specific values that justify the answer being correct."
    ),
}


# Generic "downstream agent" descriptor used by prompted_generic /
# prompted_task (no role).
_GENERIC_DESCRIPTION = (
    "You are selecting compressed memory for a downstream LLM agent.\n"
    "The agent will read this memory before working on a tool-use task. "
    "Pick the most useful facts and observations from the trace below; "
    "keep the bag of memory items small and informative."
)


def _build_prompt(condition: str, role: str, task: str, budget: int, context: str) -> str:
    """Returns the full user-message prompt text for the given condition.

    ``condition`` is one of:
      - prompted_generic
      - prompted_task
      - prompted_role
      - prompted_task_role  (matches existing prompted_memory.py templates)

    ``role`` may be None for prompted_generic / prompted_task; we still
    accept it for parameter-symmetric callers but ignore it.
    """
    suffix = (
        f"\nThe upstream agent's full execution trace is below. "
        f"Select the most useful compressed memory under approximately "
        f"{budget} tokens.\n"
        "Output ONLY the selected memory items, one per line, "
        "no prose, no numbering, no preface.\n\n"
        f"Trace:\n{context}\n"
    )

    if condition == "prompted_generic":
        return _GENERIC_DESCRIPTION + suffix

    if condition == "prompted_task":
        return (
            _GENERIC_DESCRIPTION
            + f"\n\nThe agent is working on the following task:\n{task}\n"
            + suffix
        )

    if condition == "prompted_role":
        rd = _ROLE_DESCRIPTIONS[role]
        return (
            f"You are selecting compressed memory for a {role.upper()} agent.\n\n"
            f"{rd}\n"
            + suffix
        )

    if condition == "prompted_task_role":
        rd = _ROLE_DESCRIPTIONS[role]
        return (
            f"You are selecting compressed memory for a {role.upper()} agent.\n\n"
            f"{rd}\n\n"
            f"The agent is working on the following task:\n{task}\n"
            + suffix
        )

    raise ValueError(f"unknown condition {condition!r}")


# ----------------------------------------------------------------------
# Build a single prompted memory cell
# ----------------------------------------------------------------------


def build_prompted_variant(
    traj: Trajectory,
    role: str,
    budget_tokens: int,
    *,
    condition: str,
    client: Optional[OpenAI] = None,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    max_response_tokens: int = 4096,
) -> ExecMemory:
    """Run one (traj, role, budget) under the chosen prompted condition.

    Note: prompted_generic and prompted_task do not depend on ``role``.
    We still vary role in the loop so that downstream analysis can
    compare cross-role Jaccard for these conditions; semantically the
    LLM sees an identical prompt for each role under prompted_generic /
    prompted_task, so the resulting memory should be the same up to
    sampling noise.
    """
    if client is None:
        client = _client()

    prompt_user = _build_prompt(
        condition=condition,
        role=role,
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
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    units = _parse_response(response_text, budget_tokens)
    text, n = _compose(units)
    em = ExecMemory(
        variant=f"{condition}:{role}",
        task_id=traj.task_id,
        budget_tokens=budget_tokens,
        units=units,
        text=text,
        n_tokens=n,
        n_units=len(units),
        n_units_dropped=0,
        executor=model,
    )
    em.extra_elapsed_s = elapsed  # type: ignore[attr-defined]
    em.extra_raw_response = response_text  # type: ignore[attr-defined]
    return em
