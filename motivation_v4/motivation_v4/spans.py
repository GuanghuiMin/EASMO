"""Split a Trajectory into spans (one (action, observation) step per span).

Spec §3 format:

    [STEP <n>]
    Thought:
    ...
    Action:
    ...
    Observation:
    ...
    [/STEP <n>]

Notes:
- AppWorld trajectories don't separate "thought" from "action" — the
  agent's Python code carries any reasoning as comments. We extract
  leading `# ...` comments from the action as the Thought block; the
  rest is Action.
- We cap each field at a sensible char length to keep span text size
  bounded (so leave-one-out doesn't unbalance the prompt budget).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .data import Trajectory, TrajectoryStep


@dataclass
class Span:
    task_id: str
    span_id: str           # e.g. "step_04"
    step_id: int
    span_text: str
    token_count: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_LEADING_COMMENT_RE = re.compile(r"^(\s*#[^\n]*\n)+", re.MULTILINE)


def _split_thought_action(action_text: str, *, max_thought_chars: int = 300):
    """Return (thought_lines, code_lines).

    Heuristic: contiguous `# ...` comments at the very start of the
    action are treated as 'thought', everything else as 'action code'.
    """
    if not action_text:
        return "", ""
    s = action_text
    m = re.match(r"^(\s*#[^\n]*(?:\n|$))+", s)
    if m:
        thought_block = m.group(0).strip()
        rest = s[m.end():]
    else:
        thought_block, rest = "", s
    if len(thought_block) > max_thought_chars:
        thought_block = thought_block[:max_thought_chars] + "…[truncated]"
    return thought_block.strip(), rest.strip()


def _api_calls_in(action: str, max_calls: int = 8) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"apis\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+\s*\([^)]{0,200}\)", action):
        out.append(m.group(0)[:240])
        if len(out) >= max_calls:
            break
    return out


def _approx_tokens(text: str) -> int:
    """Tiktoken cl100k when available; falls back to chars/4."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def render_step_span(
    step: TrajectoryStep,
    *,
    max_action_chars: int = 600,
    max_obs_chars: int = 600,
) -> str:
    thought, action_code = _split_thought_action(step.action or "")
    if len(action_code) > max_action_chars:
        action_code = action_code[:max_action_chars] + "…[truncated]"
    obs = (step.output or "").strip()
    if len(obs) > max_obs_chars:
        obs = obs[:max_obs_chars] + "…[truncated]"

    api_calls = _api_calls_in(step.action or "")

    lines: List[str] = [f"[STEP {step.step}]"]
    if thought:
        lines.append("Thought:")
        lines.append(thought)
    if action_code:
        lines.append("Action:")
        lines.append(action_code)
    if api_calls:
        lines.append("API calls:")
        for c in api_calls:
            lines.append(f"  - {c}")
    if obs:
        lines.append("Observation:")
        lines.append(obs)
    lines.append(f"[/STEP {step.step}]")
    return "\n".join(lines)


def trajectory_to_spans(traj: Trajectory) -> List[Span]:
    """Split a trajectory into one Span per step. Steps with no
    (action, output) content are skipped."""
    out: List[Span] = []
    for s in traj.steps:
        action = (s.action or "").strip()
        output = (s.output or "").strip()
        if not action and not output:
            continue
        text = render_step_span(s)
        out.append(Span(
            task_id=traj.task_id,
            span_id=f"step_{s.step:03d}",
            step_id=int(s.step),
            span_text=text,
            token_count=_approx_tokens(text),
        ))
    return out


def render_history(spans: List[Span]) -> str:
    """Concatenate spans in chronological order with a blank line."""
    return "\n\n".join(s.span_text for s in sorted(spans, key=lambda x: x.step_id))


def render_history_minus(spans: List[Span], removed_span_id: str) -> str:
    """All spans except ``removed_span_id``, in chronological order."""
    kept = [s for s in spans if s.span_id != removed_span_id]
    return render_history(kept)
