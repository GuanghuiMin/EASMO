"""Build the candidate memory-unit pool for a task.

The "raw context" an AppWorld agent's compressed memory replaces is
the **trajectory observation history** — every (action, output) pair
the agent has accumulated up to the current step. We treat each step's
output as one memory unit, augmented with metadata so the compressors
can rank them.

This module is the single source for "what units does compressor X
choose between?" so M1 / M2 / M3 all see the same pool.
"""

from __future__ import annotations

from typing import List, Sequence

from .data import Trajectory
from .exec_memory import (
    MemoryUnit,
    trajectory_observations_to_units,
)


def trajectory_unit_pool(
    traj: Trajectory,
    *,
    include_actions: bool = True,
    obs_max_chars: int = 240,
) -> List[MemoryUnit]:
    """Build the candidate-unit pool for one task's trajectory.

    Parameters
    ----------
    traj
        A loaded acon trajectory (success or failure — the unit pool is
        the same; downstream compressors decide what to keep).
    include_actions
        Also emit one unit per action (the Python code the agent ran),
        not just outputs. Useful because some compressors care about
        *what was queried*, not only the response.
    obs_max_chars
        Per-unit char cap for observation strings. AppWorld outputs can
        be 10k+ chars (e.g. listing all songs); long-tailed JSON dumps
        would dominate the budget if uncapped. The cap keeps unit
        granularity comparable across tasks.

    Returns
    -------
    list of MemoryUnit. The pool order is the trajectory step order;
    compressors that care about recency rely on this convention.
    """
    units: List[MemoryUnit] = []

    # Observation units (from exec_memory's parser, which already
    # truncates long outputs and tags the originating app).
    obs = trajectory_observations_to_units(traj, max_chars=obs_max_chars)

    # Action units: one per non-empty action, capped to the same length.
    if include_actions:
        for s in traj.steps:
            act = (s.action or "").strip()
            if not act:
                continue
            text = act
            if len(text) > obs_max_chars:
                text = text[:obs_max_chars] + "…[truncated]"
            # Pull the first apis.<app>... mention to tag the action's app.
            app = "unknown"
            for prefix in ("apis.", "api_docs.", "supervisor.", "spotify.",
                           "venmo.", "phone.", "file_system.", "simple_note."):
                if prefix in act:
                    rest = act.split(prefix, 1)[1]
                    app = rest.split(".", 1)[0] if "." in rest else app
                    break
            units.append(MemoryUnit(
                kind="action",
                app=app,
                text=f"[{app} step {s.step} action] {text}",
                weight=0.5,  # actions weigh less than outputs by default
                source_step=s.step,
            ))

    # Interleave actions and observations by step so the pool is in
    # natural chronological order (action_i, obs_i, action_{i+1}, ...).
    by_step: dict[int, List[MemoryUnit]] = {}
    for u in units + obs:
        by_step.setdefault(u.source_step or -1, []).append(u)

    chronological: List[MemoryUnit] = []
    for step in sorted(by_step):
        for u in by_step[step]:
            chronological.append(u)
    return chronological


def supervisor_profile_units(traj: Trajectory) -> List[MemoryUnit]:
    """Convenience: extract the supervisor.* outputs (profile, passwords,
    addresses) that nearly every task touches in its first ~3 steps.

    Useful as a "minimum scaffolding" baseline — what would the agent
    have if it only had its own profile and credentials?
    """
    out: List[MemoryUnit] = []
    for s in traj.steps:
        if "apis.supervisor" not in (s.action or ""):
            continue
        if not (s.output or "").strip():
            continue
        text = s.output.strip()
        if len(text) > 240:
            text = text[:240] + "…[truncated]"
        out.append(MemoryUnit(
            kind="profile",
            app="supervisor",
            text=f"[supervisor step {s.step}] {text}",
            weight=2.0,
            source_step=s.step,
        ))
    return out
