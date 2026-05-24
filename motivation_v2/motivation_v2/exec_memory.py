"""Build execution-derived "gold" memory ``m*_exec`` for an AppWorld task.

Two declared variants per Review note R-1 in ``new_motivation.md``:

* ``m_exec_minimal(task, B)`` — start from the union of API calls /
  DB rows the **ground-truth solution** invokes (`api_calls.json`).
  Pad with the next-most-touched rows by trajectory frequency until B.
* ``m_exec_trajectory(task, B, executor)`` — start from everything the
  **successful executor trajectory** touched; rank by
  ``2 * referenced_in_ground_truth + trajectory_touch_count``; greedy
  fill until B.

Both variants are deterministic functions of (task, budget [, executor])
with no LLM in the loop. ``m_exec_minimal`` is a pure ground-truth
oracle; ``m_exec_trajectory`` is an executor-conditioned oracle.

Memory units here are token-budget-aware text strings. The token
counter uses ``tiktoken`` if available, else falls back to a
4-chars-per-token heuristic (good enough for budget gating; the
executor itself will tokenise differently).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .data import GroundTruth, Trajectory, GroundTruthApiCall


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Approximate token count — tiktoken cl100k_base when available."""
    if not text:
        return 0
    try:  # pragma: no cover
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, Exception):
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Memory unit
# ---------------------------------------------------------------------------


@dataclass
class MemoryUnit:
    """One discrete piece of memory the executor can be given."""

    kind: str          # 'api_call' | 'observation' | 'auth' | 'fact' | …
    app: str           # e.g. 'spotify'
    text: str          # natural-language rendering ready to be concatenated
    weight: float = 1.0  # importance score used for budget greedy fill
    source_step: Optional[int] = None  # trajectory step index, if any
    extra: dict = field(default_factory=dict)

    def n_tokens(self) -> int:
        return _count_tokens(self.text)


# ---------------------------------------------------------------------------
# Helpers — convert ground-truth API calls and trajectory observations into
# MemoryUnit lists.
# ---------------------------------------------------------------------------


_API_CALL_RE = re.compile(
    r"\bapis\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\((.*?)\)",
    re.DOTALL,
)


def _render_gt_call(call: GroundTruthApiCall) -> str:
    """Render one ground-truth API call as a readable memory line."""
    args = ", ".join(f"{k}={v!r}" for k, v in (call.data or {}).items() if k != "access_token")
    if call.data and "access_token" in call.data:
        args = (args + (", " if args else "") + "access_token=<TOK>")
    return f"[{call.app}] {call.method.upper()} {call.url}({args})"


def gt_calls_to_units(gt: GroundTruth) -> List[MemoryUnit]:
    """One MemoryUnit per ground-truth API call (deduplicated)."""
    seen = set()
    units: List[MemoryUnit] = []
    for c in gt.api_calls:
        key = (c.method, c.url, tuple(sorted((c.data or {}).items())))
        if key in seen:
            continue
        seen.add(key)
        units.append(MemoryUnit(
            kind="api_call",
            app=c.app or "supervisor",
            text=_render_gt_call(c),
            weight=2.0,  # ground-truth API calls are the strongest signal
            extra={"endpoint": c.endpoint},
        ))
    return units


def trajectory_apis_called(traj: Trajectory) -> Counter:
    """Count of (app, function_name) pairs invoked across the trajectory."""
    c: Counter = Counter()
    for step in traj.steps:
        for m in _API_CALL_RE.finditer(step.action or ""):
            app, fn = m.group(1), m.group(2)
            c[(app, fn)] += 1
    return c


def trajectory_observations_to_units(traj: Trajectory, max_chars: int = 240) -> List[MemoryUnit]:
    """One MemoryUnit per non-empty trajectory step output (truncated)."""
    units: List[MemoryUnit] = []
    for step in traj.steps:
        out = (step.output or "").strip()
        if not out:
            continue
        # Find which app(s) the step's action touched.
        apps = list({m.group(1) for m in _API_CALL_RE.finditer(step.action or "")})
        app = apps[0] if apps else "unknown"
        # Truncate verbose JSON dumps to keep units below the budget granularity.
        if len(out) > max_chars:
            out = out[:max_chars] + "…[truncated]"
        units.append(MemoryUnit(
            kind="observation",
            app=app,
            text=f"[{app} step {step.step}] {out}",
            weight=1.0,
            source_step=step.step,
        ))
    return units


# ---------------------------------------------------------------------------
# Greedy budget fill
# ---------------------------------------------------------------------------


def _greedy_fill(
    candidates: List[MemoryUnit],
    budget_tokens: int,
) -> List[MemoryUnit]:
    """Greedy descending-weight fill. Stable order on ties (input order)."""
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda iu: (-iu[1].weight, iu[0]))

    selected: List[MemoryUnit] = []
    used = 0
    for _, u in indexed:
        n = u.n_tokens()
        if used + n > budget_tokens:
            continue
        selected.append(u)
        used += n
    # Return selected items in their original (input) order so the
    # composed memory string has stable ordering.
    selected.sort(key=lambda u: candidates.index(u))
    return selected


# ---------------------------------------------------------------------------
# Public API: the two declared m*_exec variants
# ---------------------------------------------------------------------------


@dataclass
class ExecMemory:
    """The result of an m*_exec build for one (task, budget [, executor])."""

    variant: str               # 'minimal' or 'trajectory'
    task_id: str
    budget_tokens: int
    units: List[MemoryUnit]
    text: str                  # concatenation of units.text with newlines
    n_tokens: int
    n_units: int
    n_units_dropped: int       # how many candidates didn't fit in budget
    executor: Optional[str] = None  # None for ground-truth-only variants


def _compose(units: List[MemoryUnit]) -> Tuple[str, int]:
    text = "\n".join(u.text for u in units)
    return text, _count_tokens(text)


def m_exec_minimal(gt: GroundTruth, budget_tokens: int) -> ExecMemory:
    """Strictly ground-truth oracle. No executor trajectory needed.

    This is the "if the agent saw exactly what the gold solution
    invokes, would it succeed?" upper bound. Independent of any
    particular executor.
    """
    candidates = gt_calls_to_units(gt)
    selected = _greedy_fill(candidates, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="minimal",
        task_id=gt.task_id,
        budget_tokens=budget_tokens,
        units=selected,
        text=text,
        n_tokens=n,
        n_units=len(selected),
        n_units_dropped=len(candidates) - len(selected),
        executor=None,
    )


def m_exec_trajectory(
    gt: GroundTruth,
    traj: Trajectory,
    budget_tokens: int,
) -> ExecMemory:
    """Executor-conditioned oracle. Uses everything the executor touched
    and ranks by ``2 × referenced_in_gold + trajectory_touch_count``.
    """
    if not traj.success:
        raise ValueError(
            f"m_exec_trajectory requires a successful trajectory; "
            f"task {traj.task_id} success={traj.success}"
        )

    # 1. Ground-truth-anchored units (high weight).
    gt_units = gt_calls_to_units(gt)
    gt_endpoints = {(u.app, u.extra.get("endpoint", "")) for u in gt_units}

    # 2. Trajectory-anchored units (one per non-empty step output).
    obs_units = trajectory_observations_to_units(traj)
    api_freq = trajectory_apis_called(traj)

    # 3. Re-weight observation units: bonus if they call a GT-listed endpoint.
    for u in obs_units:
        # crude endpoint match: any (app, fn) in this step's action that
        # corresponds to a GT call.
        bonus = 0
        for (app, fn), cnt in api_freq.items():
            if u.app == app and any(fn in (gt_e or "") for (a, gt_e) in gt_endpoints if a == app):
                bonus = 1
                break
        u.weight = 2.0 * bonus + 1.0  # 1 for plain step, 3 if step touched GT endpoint

    # 4. Combine candidate pool (GT calls first, then observations).
    pool: List[MemoryUnit] = list(gt_units) + list(obs_units)
    selected = _greedy_fill(pool, budget_tokens)
    text, n = _compose(selected)

    return ExecMemory(
        variant="trajectory",
        task_id=gt.task_id,
        budget_tokens=budget_tokens,
        units=selected,
        text=text,
        n_tokens=n,
        n_units=len(selected),
        n_units_dropped=len(pool) - len(selected),
        executor=traj.model_name,
    )
