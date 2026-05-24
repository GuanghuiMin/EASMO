"""M1 — oracle memory discovery.

For each (context C, agent A, budget B):

1. Ask MiniMax-M2.5 (the "behavior-aware selector") to propose ``K``
   candidate compressed memories, each ≤ B tokens, with the selector
   prompt described in the spec.
2. Score each candidate by:
     a. **action-match rate** over N probe states (top-1 next action
        with the candidate vs. with the full context);
     b. **L_BI** = mean KL(π_A(·|s; full) || π_A(·|s; candidate)).
3. Pick winner = argmax action-match (ties → min L_BI).

The result is a ``OracleResult`` per (context, agent, budget); we
persist all of them to a JSONL artifact for M2 / M3 to consume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .agents import AgentSpec, sample_action_distribution, top_action
from .data import Context, ProbeState
from .llm import MinimaxClient
from .metrics import action_match_rate, kl_divergence
from .utils import count_tokens, setup_logging, truncate_to_tokens

_logger = setup_logging("motivation.oracle")


SELECTOR_SYSTEM = (
    "You are a careful editor that compresses long contexts for downstream "
    "agents. Given a context C and a description of the consuming agent A, "
    "select the smallest subset of sentences/lines from C such that A's "
    "next-action distribution stays the same as when A is given C in full. "
    "You may rephrase or paraphrase only when it strictly shortens; "
    "otherwise copy spans verbatim. Do NOT introduce new facts."
)


def _build_selector_prompt(
    context: str,
    agent: AgentSpec,
    question: str,
    budget_tokens: int,
    candidate_idx: int,
) -> str:
    return (
        f"Agent description (id={agent.id}, scaffold={agent.scaffold}):\n"
        f"{agent.description}\n\n"
        f"Downstream task question: {question}\n\n"
        f"Token budget for the compressed memory: ≤ {budget_tokens} tokens (GPT-4 tokenizer).\n\n"
        "Return EXACTLY this format (no commentary, no markdown fence):\n"
        "---\n<compressed memory candidate>\n---\n"
        "Variation hint: produce candidate # "
        f"{candidate_idx + 1} — make it stylistically distinct from earlier "
        "candidates (different sentence selection or different "
        "paraphrasing density), but still respecting the budget.\n\n"
        f"Context C:\n\"\"\"\n{context}\n\"\"\""
    )


_CAND_DELIM = "---"


def _extract_candidate(text: str) -> str:
    """Pull the bit between the first two '---' delimiters (if any).

    Defensive against MiniMax-M2.5 reasoning-model failure mode: if the
    response runs out of ``max_tokens`` mid-``<think>...</think>`` block
    (no closing tag) and never emits the ``---<answer>---`` delimiter,
    we used to return the truncated reasoning blob as if it were the
    compressed memory. This contaminated up to 68% of records at the
    tightest budgets. We now reject such responses (return empty) so the
    selector loop produces another candidate.
    """
    if not text:
        return ""
    stripped = text.strip()
    # Reject obviously broken responses (unclosed think, no delimiter)
    if stripped.lower().startswith("<think>") and "</think>" not in stripped.lower():
        return ""
    if stripped.lower().startswith("<think>") and _CAND_DELIM not in stripped:
        return ""
    parts = text.split(_CAND_DELIM)
    if len(parts) >= 3:
        return parts[1].strip()
    # Final defensive line: if a stripped-think-but-no-delimiter response
    # leaked through (rare), and the text looks like a half-thought, reject.
    if "<think>" in stripped.lower() and "</think>" not in stripped.lower():
        return ""
    return stripped


@dataclass
class CandidateMemory:
    text: str
    n_tokens: int
    over_budget: bool


@dataclass
class OracleResult:
    context_id: str
    agent_id: str
    scaffold: str
    budget: int
    memory_text: str
    memory_tokens: int
    action_match_rate: float          # in [0,1]
    mean_kl_to_full: float            # L_BI
    n_probe_states: int
    n_candidates_considered: int
    all_candidate_match_rates: List[float] = field(default_factory=list)
    all_candidate_mean_kls: List[float] = field(default_factory=list)
    pass_action_match_85: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _baseline_distributions(
    client: MinimaxClient,
    agent: AgentSpec,
    context: Context,
    samples_per_state: int,
    base_seed: int,
) -> Dict[str, Dict[str, float]]:
    """π_A(·|s; full context) for every probe state."""
    out: Dict[str, Dict[str, float]] = {}
    for ps in context.probe_states:
        dist, _ = sample_action_distribution(
            client, agent, context.context_text, ps.state_text,
            n=samples_per_state, question=context.question,
            base_seed=base_seed,
            task_type=context.task_type,
        )
        out[ps.state_id] = dist
    return out


def _score_candidate(
    client: MinimaxClient,
    agent: AgentSpec,
    candidate_text: str,
    context: Context,
    baseline_dists: Dict[str, Dict[str, float]],
    samples_per_state: int,
    base_seed: int,
) -> tuple[float, float, Dict[str, Dict[str, float]]]:
    """Return ``(action_match_rate, mean_kl, per_state_dists)`` for one candidate."""
    matches: list[float] = []
    kls: list[float] = []
    per_state: Dict[str, Dict[str, float]] = {}
    for ps in context.probe_states:
        dist, _ = sample_action_distribution(
            client, agent, candidate_text, ps.state_text,
            n=samples_per_state, question=context.question,
            base_seed=base_seed + 1000,
            task_type=context.task_type,
        )
        per_state[ps.state_id] = dist
        baseline = baseline_dists.get(ps.state_id, {})
        matches.append(action_match_rate(baseline, dist))
        kls.append(kl_divergence(baseline, dist))
    return (
        sum(matches) / max(len(matches), 1),
        sum(kls) / max(len(kls), 1),
        per_state,
    )


# ---------------------------------------------------------------------------
# Top-level: oracle search for one (C, A, B)
# ---------------------------------------------------------------------------


def find_oracle_memory(
    client: MinimaxClient,
    agent: AgentSpec,
    context: Context,
    budget: int,
    *,
    samples_per_state: int,
    candidate_top_k: int,
    seed: int,
    action_match_pass: float = 0.85,
    baseline_dists: Optional[Dict[str, Dict[str, float]]] = None,
) -> OracleResult:
    """Find z*_A(C, B) for one (context, agent, budget) triple."""

    # 1. Baseline: π_A on the *full* context for every probe state.
    if baseline_dists is None:
        _logger.info(
            "[%s | %s | B=%d] Sampling baseline distributions (N=%d, %d probes).",
            context.context_id, agent.id, budget, samples_per_state, len(context.probe_states),
        )
        baseline_dists = _baseline_distributions(
            client, agent, context, samples_per_state, base_seed=seed,
        )

    # 2. Get K candidate compressed memories from MiniMax with the selector prompt.
    candidates: list[CandidateMemory] = []
    for i in range(candidate_top_k):
        prompt = _build_selector_prompt(
            context.context_text, agent, context.question, budget, candidate_idx=i,
        )
        # max_tokens needs headroom for MiniMax's <think> reasoning chain
        # (typically 1k-2k tokens) PLUS the actual budget for the memory.
        # Always give 4096 since reasoning is the dominant cost.
        res = client.chat(
            system=SELECTOR_SYSTEM, user=prompt,
            temperature=0.6, max_tokens=4096,
            seed=seed + i,
        )
        cand_text = _extract_candidate(res.text)
        cand_text = truncate_to_tokens(cand_text, budget)
        candidates.append(CandidateMemory(
            text=cand_text,
            n_tokens=count_tokens(cand_text),
            over_budget=count_tokens(cand_text) > budget,
        ))
        _logger.debug(
            "[%s | %s | B=%d] candidate %d: %d tokens",
            context.context_id, agent.id, budget, i, candidates[-1].n_tokens,
        )

    # 3. Score each candidate against the baseline distributions.
    cand_match_rates: list[float] = []
    cand_mean_kls: list[float] = []
    for i, cm in enumerate(candidates):
        if not cm.text:
            cand_match_rates.append(0.0)
            cand_mean_kls.append(float("inf"))
            continue
        rate, mean_kl, _ = _score_candidate(
            client, agent, cm.text, context, baseline_dists,
            samples_per_state, base_seed=seed + 50 + i,
        )
        cand_match_rates.append(rate)
        cand_mean_kls.append(mean_kl)
        _logger.info(
            "[%s | %s | B=%d] candidate %d → match=%.2f mean_kl=%.3f tokens=%d",
            context.context_id, agent.id, budget, i, rate, mean_kl, cm.n_tokens,
        )

    # 4. Pick winner: argmax(match), ties → min(kl).
    winner_idx = max(
        range(len(candidates)),
        key=lambda i: (cand_match_rates[i], -cand_mean_kls[i]),
    )
    winner = candidates[winner_idx]
    return OracleResult(
        context_id=context.context_id,
        agent_id=agent.id,
        scaffold=agent.scaffold,
        budget=budget,
        memory_text=winner.text,
        memory_tokens=winner.n_tokens,
        action_match_rate=cand_match_rates[winner_idx],
        mean_kl_to_full=cand_mean_kls[winner_idx],
        n_probe_states=len(context.probe_states),
        n_candidates_considered=len(candidates),
        all_candidate_match_rates=cand_match_rates,
        all_candidate_mean_kls=cand_mean_kls,
        pass_action_match_85=cand_match_rates[winner_idx] >= action_match_pass,
    )
