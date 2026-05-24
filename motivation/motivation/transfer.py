"""M3 — cross-agent transfer degradation.

Feeds z*_{A_i}(C, B) to A_j and measures:
* next-action top-1 match rate (vs A_j's full-context distribution),
* a coarse downstream-task proxy = mean action-match averaged across probe states,
* TV-divergence between agents on reference probe states.

The final scatter (task-drop vs δ) is built from many (context, A_i, A_j, B)
points; correlation + linear fit are produced by the run_m3 driver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from .agents import AgentSpec, sample_action_distribution
from .data import Context
from .llm import MinimaxClient
from .metrics import action_match_rate, action_overlap_rate, total_variation
from .utils import setup_logging

_logger = setup_logging("motivation.transfer")


@dataclass
class TransferRow:
    context_id: str
    budget: int
    source_agent: str
    target_agent: str

    # ---- Binary top-1 match (legacy headline) -----------------------
    action_match_self: float       # A_j(z*_{A_j}) vs A_j(full)
    action_match_cross: float      # A_j(z*_{A_i}) vs A_j(full)
    task_drop: float               # match_self - match_cross

    # ---- Continuous behavioural overlap = 1 - TV (primary) ----------
    overlap_self: float            # mean (1-TV) on probe states
    overlap_cross: float
    overlap_drop: float            # overlap_self - overlap_cross

    policy_divergence: float       # mean TV(A_i, A_j) on reference states

    def to_dict(self) -> dict:
        return asdict(self)


def _policy_divergence(
    client: MinimaxClient,
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    context: Context,
    samples_per_state: int,
    base_seed: int,
) -> float:
    """Mean TV(π_{A}(·|s; full), π_{B}(·|s; full)) over probe states."""
    tvs: list[float] = []
    for ps in context.probe_states:
        da, _ = sample_action_distribution(
            client, agent_a, context.context_text, ps.state_text,
            n=samples_per_state, question=context.question, base_seed=base_seed,
            task_type=context.task_type,
        )
        db, _ = sample_action_distribution(
            client, agent_b, context.context_text, ps.state_text,
            n=samples_per_state, question=context.question, base_seed=base_seed + 1,
            task_type=context.task_type,
        )
        tvs.append(total_variation(da, db))
    return sum(tvs) / max(len(tvs), 1)


def cross_agent_eval(
    client: MinimaxClient,
    target_agent: AgentSpec,
    memory_text: str,
    context: Context,
    *,
    samples_per_state: int,
    base_seed: int,
) -> Dict[str, Dict[str, float]]:
    """π_{target}(·|state ; memory) for each probe state."""
    per_state: Dict[str, Dict[str, float]] = {}
    for ps in context.probe_states:
        dist, _ = sample_action_distribution(
            client, target_agent, memory_text, ps.state_text,
            n=samples_per_state, question=context.question, base_seed=base_seed,
            task_type=context.task_type,
        )
        per_state[ps.state_id] = dist
    return per_state


def mean_match_rate(
    target_full: Dict[str, Dict[str, float]],
    cross_dists: Dict[str, Dict[str, float]],
) -> float:
    """Binary top-1 action match averaged across probe states (legacy)."""
    keys = set(target_full) & set(cross_dists)
    if not keys:
        return 0.0
    return sum(action_match_rate(target_full[k], cross_dists[k]) for k in keys) / len(keys)


def mean_overlap_rate(
    target_full: Dict[str, Dict[str, float]],
    cross_dists: Dict[str, Dict[str, float]],
) -> float:
    """Continuous behavioural overlap (1 - TV) averaged across probe states.

    This is the primary M3 signal post-2026-05-24-audit. The binary
    top-1 match collapses too much information when N=16 samples spread
    across multiple high-probability actions, especially on QA tasks
    where one canonical answer dominates and a single sample flip
    rolls match between 0 and 1.
    """
    keys = set(target_full) & set(cross_dists)
    if not keys:
        return 0.0
    return sum(action_overlap_rate(target_full[k], cross_dists[k]) for k in keys) / len(keys)
