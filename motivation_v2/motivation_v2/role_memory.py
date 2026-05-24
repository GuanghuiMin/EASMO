"""Role-conditional m*_exec extractors.

Given a successful trajectory, build four role-specific compressed
memories — one per agent role (tool-use / code / plan / verify) —
each as a deterministic projection of the trajectory's content.

The projections aim at *what that role would have wanted to know* if
it were processing the trajectory's upstream context, not at what it
literally produced. So:

* `m_tool(traj, B)`: rendered API call list + key observations
  (canonical pattern: list endpoint → fetch detail → answer)
* `m_code(traj, B)`: Python action snippets — control flow, list
  comprehensions, error handling — stripping the API call args
* `m_plan(traj, B)`: task instruction + first action's intent comment
  + key milestone observations + final answer
* `m_verify(traj, B)`: tail-of-trajectory observations and the final
  ``apis.supervisor.complete_task(answer=...)`` call (the
  ground-truth-checker view)

These are projections of the same trajectory, not independent
oracles. The Jaccard between projections tells us how separable
role-conditioned memory needs to be — if low, role-conditional
compression is non-trivially useful even when downstream agents
share upstream context.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .data import Trajectory
from .exec_memory import (
    MemoryUnit,
    ExecMemory,
    _compose,
    _greedy_fill,
)


# ---------------------------------------------------------------------------
# Tool-use view
# ---------------------------------------------------------------------------


_API_RE = re.compile(r"apis\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\((.*?)\)", re.DOTALL)


def _api_calls_in_action(action: str):
    if not action:
        return
    for m in _API_RE.finditer(action):
        app, fn, args = m.group(1), m.group(2), m.group(3).strip()
        # Strip whitespace + access_token noise + JWTs.
        args = re.sub(r"access_token\s*=\s*['\"][^'\"]+['\"]", "access_token=<TOK>", args)
        args = re.sub(r"\s+", " ", args)
        yield app, fn, args


def m_tool(traj: Trajectory, budget_tokens: int) -> ExecMemory:
    """API call sequence + the observations that immediately followed."""
    units: List[MemoryUnit] = []
    seen = set()
    for s in traj.steps:
        for app, fn, args in _api_calls_in_action(s.action or ""):
            key = (app, fn, args[:80])
            if key in seen:
                continue
            seen.add(key)
            text = f"[{app} step {s.step} call] {app}.{fn}({args})"
            if len(text) > 240:
                text = text[:240] + "…[truncated]"
            units.append(MemoryUnit(
                kind="api_call", app=app, text=text, weight=2.0,
                source_step=s.step,
            ))
        # Pair each call with a short observation summary.
        out = (s.output or "").strip()
        if out:
            obs = out if len(out) <= 200 else out[:200] + "…"
            units.append(MemoryUnit(
                kind="observation",
                app="?",
                text=f"[step {s.step} obs] {obs}",
                weight=1.0,
                source_step=s.step,
            ))
    selected = _greedy_fill(units, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="role:tool", task_id=traj.task_id, budget_tokens=budget_tokens,
        units=selected, text=text, n_tokens=n, n_units=len(selected),
        n_units_dropped=len(units) - len(selected), executor=traj.model_name,
    )


# ---------------------------------------------------------------------------
# Code view
# ---------------------------------------------------------------------------


_CODE_PATTERN_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(for\s+\w+\s+in\s+|"
    r"while\s+|"
    r"if\s+|"
    r"try:|"
    r"except\s+\w+|"
    r"def\s+\w+|"
    r"\[\s*\w.*?for\s+\w+|"   # list comprehension
    r"max\(|min\(|sorted\(|filter\(|map\()",
)


def _strip_api_args(action: str) -> str:
    """Replace concrete api call args with placeholder so the code
    structure is preserved without leaking specific IDs/values."""
    if not action:
        return ""
    return _API_RE.sub(
        lambda m: f"apis.{m.group(1)}.{m.group(2)}(<args>)",
        action,
    )


def m_code(traj: Trajectory, budget_tokens: int) -> ExecMemory:
    """Python control-flow patterns from action code, with API args
    abstracted away.

    The premise: a *coding* role wants to learn 'how to combine API
    calls in code', not the specific arguments. So we keep the
    surrounding control flow and replace api args with `<args>`.
    """
    units: List[MemoryUnit] = []
    for s in traj.steps:
        action = (s.action or "").strip()
        if not action:
            continue
        if not _CODE_PATTERN_RE.search(action):
            # No structural pattern in this step — skip; no value to a
            # coding role.
            continue
        clean = _strip_api_args(action)
        # Keep first 240 chars as the structural signature.
        if len(clean) > 240:
            clean = clean[:240] + "…[truncated]"
        units.append(MemoryUnit(
            kind="code_pattern", app="python",
            text=f"[code step {s.step}] {clean}",
            weight=1.0, source_step=s.step,
        ))
    selected = _greedy_fill(units, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="role:code", task_id=traj.task_id, budget_tokens=budget_tokens,
        units=selected, text=text, n_tokens=n, n_units=len(selected),
        n_units_dropped=len(units) - len(selected), executor=traj.model_name,
    )


# ---------------------------------------------------------------------------
# Plan view
# ---------------------------------------------------------------------------


_INTENT_COMMENT_RE = re.compile(r"^\s*#\s*(.+)$", re.MULTILINE)
_COMPLETE_TASK_RE = re.compile(
    r"apis\.supervisor\.complete_task\s*\((.*?)\)", re.DOTALL,
)


def m_plan(traj: Trajectory, budget_tokens: int) -> ExecMemory:
    """Task instruction + intent comments at start of trajectory + final
    answer. The view a planning agent would want.
    """
    units: List[MemoryUnit] = []
    if traj.instruction:
        units.append(MemoryUnit(
            kind="task_instruction", app="meta",
            text=f"[plan task] {traj.instruction}",
            weight=3.0, source_step=0,
        ))
    # First-step intent comments (the agent's plan, encoded as comments
    # before any code).
    for s in traj.steps[:3]:
        for m in _INTENT_COMMENT_RE.finditer(s.action or ""):
            comment = m.group(1).strip()
            if not comment or len(comment) < 10:
                continue
            units.append(MemoryUnit(
                kind="intent_comment", app="meta",
                text=f"[plan step {s.step} intent] {comment[:200]}",
                weight=2.0, source_step=s.step,
            ))
    # Mid-trajectory milestone observations: pick the FIRST observation
    # from each unique app touched (signals "we discovered this app's
    # data").
    apps_seen = set()
    for s in traj.steps:
        for m in _API_RE.finditer(s.action or ""):
            app = m.group(1)
            if app in apps_seen:
                continue
            apps_seen.add(app)
            obs = (s.output or "").strip()
            if not obs:
                continue
            units.append(MemoryUnit(
                kind="milestone_obs", app=app,
                text=f"[plan milestone {app} step {s.step}] {obs[:150]}",
                weight=1.0, source_step=s.step,
            ))
            break
    # Final-answer call.
    for s in reversed(traj.steps):
        m = _COMPLETE_TASK_RE.search(s.action or "")
        if m:
            args = m.group(1).strip()
            units.append(MemoryUnit(
                kind="final_answer", app="meta",
                text=f"[plan final] complete_task({args[:200]})",
                weight=3.0, source_step=s.step,
            ))
            break
    selected = _greedy_fill(units, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="role:plan", task_id=traj.task_id, budget_tokens=budget_tokens,
        units=selected, text=text, n_tokens=n, n_units=len(selected),
        n_units_dropped=len(units) - len(selected), executor=traj.model_name,
    )


# ---------------------------------------------------------------------------
# Verify view
# ---------------------------------------------------------------------------


def m_verify(traj: Trajectory, budget_tokens: int) -> ExecMemory:
    """Final-state-relevant observations + complete_task call.

    The view a verification role would want: 'what facts justify the
    answer being correct?'.
    """
    units: List[MemoryUnit] = []
    # Tail-of-trajectory: last 5 non-empty observations.
    tail_obs: List[MemoryUnit] = []
    for s in reversed(traj.steps):
        out = (s.output or "").strip()
        if not out:
            continue
        # Try to grab the app name from the corresponding action.
        app = "unknown"
        m = _API_RE.search(s.action or "")
        if m:
            app = m.group(1)
        tail_obs.append(MemoryUnit(
            kind="verify_obs", app=app,
            text=f"[verify step {s.step} obs] {out[:200]}",
            weight=2.0, source_step=s.step,
        ))
        if len(tail_obs) >= 5:
            break
    # The final complete_task call (if any).
    final_call: Optional[MemoryUnit] = None
    for s in reversed(traj.steps):
        m = _COMPLETE_TASK_RE.search(s.action or "")
        if m:
            args = m.group(1).strip()
            final_call = MemoryUnit(
                kind="final_call", app="meta",
                text=f"[verify final call] complete_task({args[:200]})",
                weight=3.0, source_step=s.step,
            )
            break
    if final_call is not None:
        units.append(final_call)
    units.extend(reversed(tail_obs))  # restore chronological order
    selected = _greedy_fill(units, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="role:verify", task_id=traj.task_id, budget_tokens=budget_tokens,
        units=selected, text=text, n_tokens=n, n_units=len(selected),
        n_units_dropped=len(units) - len(selected), executor=traj.model_name,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


ROLE_BUILDERS = {
    "tool":   m_tool,
    "code":   m_code,
    "plan":   m_plan,
    "verify": m_verify,
}
