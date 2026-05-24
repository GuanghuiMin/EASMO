"""Instance-noise ablation — the missing test for Path D.

Question we're answering:
    Is the M3 cross-agent transfer drop driven by *policy* (A_react vs
    A_plan vs A_cot) or by *seed-level instance noise* of the selector?

Method (per context C, per budget B):
    1. Use the same selector regime as M1 (T=0.6, variation hint, K=3)
       to generate 3 candidate memories *per agent*.
    2. For each target agent T:
       a. Sample baseline π_T(·|s; full_C).
       b. Sample π_T on T's candidate 0  → "self"
       c. Sample π_T on T's candidates 1 and 2 → "within-agent variants"
       d. Sample π_T on each *other* agent's candidate 0 → "cross-agent"
    3. Compute action-match rate of each π vs baseline.
    4. Compare within-agent drop vs cross-agent drop on the same signal
       rows (rows where T's self-memory preserves behaviour).

Decision rule (paper-level):
    ratio = mean(cross-agent drop) / mean(within-agent drop)
    * ratio >= 3x: M3 is policy-driven → Path D survives, Spotlight defensible
    * 1.5x <= ratio < 3x: weak signal, Path D borderline
    * ratio < 1.5x: M3 drop is mostly instance noise → Path D dies
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence

from .agents import AgentSpec, get_agent, sample_action_distribution
from .data import Context
from .llm import MinimaxClient
from .metrics import action_match_rate
from .oracle import SELECTOR_SYSTEM, _build_selector_prompt, _extract_candidate
from .utils import count_tokens, setup_logging, truncate_to_tokens

_logger = setup_logging("motivation.instance_noise")


@dataclass
class InstanceNoiseRow:
    context_id: str
    budget: int
    target_agent: str
    self_match: float
    within_match_mean: float
    cross_match_mean: float
    within_drop: float          # self - within_mean
    cross_drop: float           # self - cross_mean
    n_within: int
    n_cross: int
    n_target_candidates: int    # K used in this run

    def to_dict(self):
        return asdict(self)


def _generate_candidates(
    client: MinimaxClient,
    agent: AgentSpec,
    context: Context,
    budget: int,
    k: int,
    seed: int,
) -> List[str]:
    """Reuse M1's exact selector regime (T=0.6, variation hint, K candidates)."""
    out = []
    for i in range(k):
        prompt = _build_selector_prompt(
            context.context_text, agent, context.question, budget, candidate_idx=i,
        )
        res = client.chat(
            system=SELECTOR_SYSTEM, user=prompt,
            temperature=0.6, max_tokens=4096,
            seed=seed + i,
        )
        cand = _extract_candidate(res.text)
        cand = truncate_to_tokens(cand, budget)
        if cand:
            out.append(cand)
    return out


def run_instance_noise_test(
    client: MinimaxClient,
    contexts: List[Context],
    scaffolds: List[str],
    budget: int,
    *,
    candidates_per_agent: int = 3,
    samples_per_state: int = 16,
    seed: int = 42,
) -> List[InstanceNoiseRow]:
    """Drive the ablation. Returns one row per (context, target_agent)."""
    agents = [get_agent(s) for s in scaffolds]
    rows: List[InstanceNoiseRow] = []

    for ci, ctx in enumerate(contexts):
        _logger.info(
            "[%d/%d] %s — generating %d candidates per agent (T=0.6, M1 protocol)",
            ci + 1, len(contexts), ctx.context_id, candidates_per_agent,
        )
        per_agent: dict[str, List[str]] = {}
        for ai, agent in enumerate(agents):
            cands = _generate_candidates(
                client, agent, ctx, budget,
                k=candidates_per_agent,
                seed=seed + 1000 * ci + 100 * ai,
            )
            per_agent[agent.id] = cands
            _logger.info("  %s: got %d valid candidates", agent.id, len(cands))

        for target in agents:
            target_cands = per_agent.get(target.id, [])
            if len(target_cands) < 2:
                _logger.warning(
                    "Skipping target=%s on ctx=%s: only %d valid target candidates",
                    target.id, ctx.context_id, len(target_cands),
                )
                continue

            # Baseline: target on full ctx
            baseline_dist, _ = sample_action_distribution(
                client, target, ctx.context_text, ctx.probe_states[0].state_text,
                n=samples_per_state, question=ctx.question,
                base_seed=seed + 2000 * ci, task_type=ctx.task_type,
            )

            # Self: candidate 0 of target
            self_dist, _ = sample_action_distribution(
                client, target, target_cands[0], ctx.probe_states[0].state_text,
                n=samples_per_state, question=ctx.question,
                base_seed=seed + 3000 * ci, task_type=ctx.task_type,
            )
            self_match = action_match_rate(baseline_dist, self_dist)

            # Within-agent variants: candidates 1, 2, … of the SAME target
            within_matches = []
            for k, cand in enumerate(target_cands[1:], start=1):
                dist, _ = sample_action_distribution(
                    client, target, cand, ctx.probe_states[0].state_text,
                    n=samples_per_state, question=ctx.question,
                    base_seed=seed + 4000 * ci + 10 * k, task_type=ctx.task_type,
                )
                within_matches.append(action_match_rate(baseline_dist, dist))

            # Cross-agent: candidate 0 of each *other* agent
            cross_matches = []
            for other in agents:
                if other.id == target.id:
                    continue
                other_cands = per_agent.get(other.id, [])
                if not other_cands:
                    continue
                dist, _ = sample_action_distribution(
                    client, target, other_cands[0], ctx.probe_states[0].state_text,
                    n=samples_per_state, question=ctx.question,
                    base_seed=seed + 5000 * ci + 10 * agents.index(other),
                    task_type=ctx.task_type,
                )
                cross_matches.append(action_match_rate(baseline_dist, dist))

            within_mean = sum(within_matches) / max(len(within_matches), 1)
            cross_mean = sum(cross_matches) / max(len(cross_matches), 1)
            row = InstanceNoiseRow(
                context_id=ctx.context_id,
                budget=budget,
                target_agent=target.id,
                self_match=self_match,
                within_match_mean=within_mean,
                cross_match_mean=cross_mean,
                within_drop=self_match - within_mean,
                cross_drop=self_match - cross_mean,
                n_within=len(within_matches),
                n_cross=len(cross_matches),
                n_target_candidates=len(target_cands),
            )
            rows.append(row)
            _logger.info(
                "  target=%s self=%.2f within=%.2f cross=%.2f (drop within=%.2f cross=%.2f)",
                target.id, self_match, within_mean, cross_mean,
                row.within_drop, row.cross_drop,
            )
    return rows
